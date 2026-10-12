from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from app.api.dashboard import (
    create_dashboard_app,
    set_dashboard_session,
)
from app.backtest.models import BacktestConfig
from app.market.constants import EXCHANGE
from app.market.models import Candle, CandleOpenObservation
from app.strategy.breakout import BreakoutSignal
from app.trading.paper import PaperExecutor
from app.trading.session import PaperSession

UTC = timezone.utc
START = datetime(2024, 1, 2, tzinfo=UTC)
D = Decimal


def make_config(*, minimum_quantity=D("0.000001"), quantity_step=D("0.000001")):
    return BacktestConfig(minimum_quantity=minimum_quantity, quantity_step=quantity_step)


def make_candle(
    symbol="BTCUSDT",
    opened=START,
    interval="1h",
    *,
    open=D("10"),
    high=D("10.2"),
    low=D("9.9"),
    close=D("10"),
    volume=D("1"),
):
    return Candle(
        exchange=EXCHANGE,
        symbol=symbol,
        interval=interval,
        open_time=opened,
        open=open,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def make_observation(
    symbol="BTCUSDT",
    opened=START + timedelta(hours=1),
    *,
    price=D("10"),
    observed=None,
    interval="1h",
):
    return CandleOpenObservation(
        exchange=EXCHANGE,
        symbol=symbol,
        interval=interval,
        open_time=opened,
        observed_at=observed or opened + timedelta(seconds=1),
        open=price,
    )


def make_signal(symbol="BTCUSDT", opened=START, *, atr=1.0):
    return BreakoutSignal(
        exchange=EXCHANGE,
        symbol=symbol,
        side="long",
        decision_time=opened + timedelta(hours=1),
        signal_open_time=opened,
        atr_14=atr,
    )


def test_dashboard_uninitialized_session_returns_503():
    app = create_dashboard_app(None)
    set_dashboard_session(None)
    client = TestClient(app)

    response = client.get("/portfolio")
    assert response.status_code == 503
    assert "not initialized" in response.json()["detail"]


def test_get_portfolio_initial_state():
    executor = PaperExecutor(make_config())
    app = create_dashboard_app(executor)
    client = TestClient(app)

    response = client.get("/portfolio")
    assert response.status_code == 200
    data = response.json()
    assert Decimal(str(data["cash"])) == D("100")
    assert Decimal(str(data["equity"])) == D("100")
    assert Decimal(str(data["open_risk"])) == D("0")
    assert data["daily_loss_breached"] is False


def test_get_positions_empty_and_populated():
    executor = PaperExecutor(make_config())
    app = create_dashboard_app(executor)
    client = TestClient(app)

    # 1. Empty positions
    resp_empty = client.get("/positions")
    assert resp_empty.status_code == 200
    assert resp_empty.json() == []

    # 2. Enter a position
    c1 = make_candle(opened=START)
    executor.on_candle_closed(c1)
    sig = make_signal(opened=START)
    assert executor.submit_signal(sig)
    obs = make_observation(opened=START + timedelta(hours=1), price=D("10"))
    pos = executor.on_open_observation(obs)
    assert pos is not None

    # 3. Populated positions
    resp_pos = client.get("/positions")
    assert resp_pos.status_code == 200
    positions = resp_pos.json()
    assert len(positions) == 1
    p = positions[0]
    assert p["symbol"] == "BTCUSDT"
    assert p["exchange"] == "binance_spot"
    assert Decimal(str(p["quantity"])) == pos.quantity
    assert Decimal(str(p["stop_price"])) == pos.stop_price
    assert Decimal(str(p["target_price"])) == pos.target_price
    assert Decimal(str(p["initial_risk"])) == pos.initial_risk
    assert p["entry_fill"]["side"] == "buy"
    assert Decimal(str(p["entry_fill"]["price"])) == pos.entry_fill.price
    assert p["signal"]["symbol"] == "BTCUSDT"


def test_get_trades_empty_and_populated():
    executor = PaperExecutor(make_config())
    app = create_dashboard_app(executor)
    client = TestClient(app)

    # 1. Empty trades
    resp_empty = client.get("/trades")
    assert resp_empty.status_code == 200
    assert resp_empty.json() == []

    # 2. Open and exit position
    c1 = make_candle(opened=START)
    executor.on_candle_closed(c1)
    sig = make_signal(opened=START)
    executor.submit_signal(sig)
    obs = make_observation(opened=START + timedelta(hours=1), price=D("10"))
    executor.on_open_observation(obs)

    # Exit on take profit
    c2 = make_candle(
        opened=START + timedelta(hours=1),
        open=D("10"),
        high=D("25"),
        low=D("9.5"),
        close=D("20"),
    )
    trade = executor.on_candle_closed(c2)
    assert trade is not None

    # 3. Populated trades
    resp_trades = client.get("/trades")
    assert resp_trades.status_code == 200
    trades = resp_trades.json()
    assert len(trades) == 1
    t = trades[0]
    assert t["symbol"] == "BTCUSDT"
    assert t["exit_reason"] == "take_profit"
    assert Decimal(str(t["net_pnl"])) == trade.net_pnl
    assert Decimal(str(t["gross_pnl"])) == trade.gross_pnl
    assert t["entry_fill"]["side"] == "buy"
    assert t["exit_fill"]["side"] == "sell"


def test_get_signals_empty_and_populated():
    executor = PaperExecutor(make_config())
    app = create_dashboard_app(executor)
    client = TestClient(app)

    # 1. Empty signals
    resp_empty = client.get("/signals")
    assert resp_empty.status_code == 200
    assert resp_empty.json() == []

    # 2. Submit signal (pending)
    c1 = make_candle(opened=START)
    executor.on_candle_closed(c1)
    sig = make_signal(opened=START)
    assert executor.submit_signal(sig)

    resp_pending = client.get("/signals")
    assert resp_pending.status_code == 200
    signals = resp_pending.json()
    assert len(signals) == 1
    s = signals[0]
    assert s["symbol"] == "BTCUSDT"
    assert s["exchange"] == "binance_spot"
    assert s["side"] == "long"
    assert s["atr_14"] == 1.0

    # 3. Fill signal -> signals list becomes empty again
    obs = make_observation(opened=START + timedelta(hours=1))
    executor.on_open_observation(obs)

    resp_after = client.get("/signals")
    assert resp_after.status_code == 200
    assert resp_after.json() == []


@pytest.mark.asyncio
async def test_dashboard_with_paper_session():
    class DummyStream:
        async def events(self):
            if False:
                yield None

    executor = PaperExecutor(make_config())
    session = PaperSession(stream=DummyStream(), executor=executor)
    app = create_dashboard_app(session)
    client = TestClient(app)

    # Process events through session
    c1 = make_candle(opened=START)
    sig = make_signal(opened=START)
    session.strategy = lambda c: [sig]
    await session.process_event(c1)

    # Signal is now pending
    resp_signals = client.get("/signals")
    assert resp_signals.status_code == 200
    assert len(resp_signals.json()) == 1

    # Fill entry
    obs = make_observation(opened=START + timedelta(hours=1))
    await session.process_event(obs)

    resp_portfolio = client.get("/portfolio")
    assert resp_portfolio.status_code == 200
    data = resp_portfolio.json()
    assert Decimal(str(data["open_risk"])) > D("0")
    assert len(client.get("/positions").json()) == 1


def test_dashboard_daily_loss_breached_flag():
    executor = PaperExecutor(make_config(quantity_step=D("0.1"), minimum_quantity=D("0.1")))
    app = create_dashboard_app(executor)
    client = TestClient(app)

    # Open position
    c1 = make_candle(opened=START)
    executor.on_candle_closed(c1)
    sig = make_signal(opened=START)
    executor.submit_signal(sig)
    obs = make_observation(opened=START + timedelta(hours=1), price=D("10"))
    executor.on_open_observation(obs)

    # Severe loss bar
    c2 = make_candle(
        opened=START + timedelta(hours=1),
        open=D("1"),
        high=D("2"),
        low=D("0.5"),
        close=D("1"),
    )
    executor.on_candle_closed(c2)
    assert executor.daily_loss_breached

    resp = client.get("/portfolio")
    assert resp.status_code == 200
    assert resp.json()["daily_loss_breached"] is True


def test_dashboard_read_only_disallows_modifying_methods():
    executor = PaperExecutor(make_config())
    app = create_dashboard_app(executor)
    client = TestClient(app)

    for path in ("/portfolio", "/positions", "/trades", "/signals"):
        assert client.post(path).status_code == 405
        assert client.put(path).status_code == 405
        assert client.delete(path).status_code == 405


def test_dashboard_callable_session_provider():
    executor = PaperExecutor(make_config())
    current_target = executor

    app = create_dashboard_app(lambda: current_target)
    client = TestClient(app)

    assert client.get("/portfolio").status_code == 200

    current_target = None
    assert client.get("/portfolio").status_code == 503


def test_dashboard_ui_serves_html():
    app = create_dashboard_app(None)
    client = TestClient(app)

    # Root route /
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert "text/html" in res_root.headers["content-type"]
    assert "Crypto AI Trader" in res_root.text
    assert "tailwindcss" in res_root.text
    assert "Active Positions" in res_root.text
    assert "Pending Breakout Signals" in res_root.text
    assert (
        "Trade History &amp; AI Explanations" in res_root.text
        or "Trade History & AI Explanations" in res_root.text
    )

    # Dedicated route /ui
    res_ui = client.get("/ui")
    assert res_ui.status_code == 200
    assert "text/html" in res_ui.headers["content-type"]
    assert "Crypto AI Trader" in res_ui.text


def test_dashboard_serializes_explanations():
    executor = PaperExecutor(make_config())
    app = create_dashboard_app(executor)
    client = TestClient(app)

    # Signal with explanation
    c1 = make_candle(opened=START)
    executor.on_candle_closed(c1)
    sig = make_signal(opened=START)
    object.__setattr__(sig, "explanation", "Breakout Long Signal commentary text")
    assert executor.submit_signal(sig)

    signals_resp = client.get("/signals")
    assert signals_resp.status_code == 200
    signals_data = signals_resp.json()
    assert len(signals_data) == 1
    assert signals_data[0]["explanation"] == "Breakout Long Signal commentary text"

    # Trade with explanation
    obs = make_observation(opened=START + timedelta(hours=1), price=D("10"))
    executor.on_open_observation(obs)
    c_exit = make_candle(
        opened=START + timedelta(hours=1),
        open=D("10"),
        high=D("25"),
        low=D("9.5"),
        close=D("20"),
    )
    trade = executor.on_candle_closed(c_exit)
    assert trade is not None
    object.__setattr__(trade, "explanation", "Analytical trade exit retrospective text")

    trades_resp = client.get("/trades")
    assert trades_resp.status_code == 200
    trades_data = trades_resp.json()
    assert len(trades_data) == 1
    assert trades_data[0]["explanation"] == "Analytical trade exit retrospective text"

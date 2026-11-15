import sqlite3
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from app.api.dashboard import (
    create_dashboard_app,
    open_readonly_sqlite,
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


class MockKlineStream:
    def __init__(self, items=None):
        self.items = list(items or [])

    async def events(self):
        for item in self.items:
            yield item


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


def test_variants_equity_day_zero_defaults():
    """Verify that Day 0 uninitialized experiment returns safe default baseline for all 5 variants."""
    app = create_dashboard_app(None)
    set_dashboard_session(None)
    client = TestClient(app)

    res = client.get("/variants/equity")
    assert res.status_code == 200
    data = res.json()
    assert "variants" in data
    assert len(data["variants"]) == 5

    expected_variants = {
        "variant_a": "Variant A (Gate + Regime)",
        "variant_b": "Variant B (Regime Only)",
        "variant_c": "Variant C (Gate Only)",
        "variant_d": "Variant D (Baseline)",
        "variant_e": "Variant E (Gate + Regime on Core)",
    }

    for v_key, expected_name in expected_variants.items():
        assert v_key in data["variants"]
        var = data["variants"][v_key]
        assert var["name"] == expected_name
        assert var["starting_cash"] == 100.0
        assert var["current_equity"] == 100.0
        assert var["cumulative_pnl"] == 0.0
        assert var["open_positions_count"] == 0
        assert var["closed_trades_count"] == 0
        assert len(var["points"]) >= 1

    # Alias endpoint
    res_alias = client.get("/portfolio/variants")
    assert res_alias.status_code == 200
    assert len(res_alias.json()["variants"]) == 5


def test_system_status_day_zero_defaults():
    """Verify system status Day 0 defaults when no database or session is present."""
    app = create_dashboard_app(None)
    set_dashboard_session(None)
    client = TestClient(app)

    res = client.get("/system/status")
    assert res.status_code == 200
    data = res.json()
    assert data["btc_regime_state"] in ("UNKNOWN", "UPTREND", "DOWNTREND")
    assert data["fallback_rate_24h"] == 0.0
    assert data["fallback_count_24h"] == 0
    assert data["total_evals_24h"] == 0
    assert data["variant_positions"]["total"] == 0
    assert data["variant_positions"]["variant_a"] == 0
    assert data["variant_positions"]["variant_b"] == 0
    assert data["variant_positions"]["variant_c"] == 0
    assert data["variant_positions"]["variant_d"] == 0
    assert data["variant_positions"]["variant_e"] == 0


def test_gate_decisions_day_zero_empty():
    """Verify gate decisions endpoint handles missing fundamental_eval.db gracefully without error."""
    app = create_dashboard_app(None)
    client = TestClient(app)

    res = client.get("/gate/decisions")
    assert res.status_code == 200
    assert isinstance(res.json(), list)


def test_readonly_database_connection_blocks_writes(tmp_path):
    """Verify that open_readonly_sqlite enforces strictly read-only mode and rejects writes."""
    db_file = tmp_path / "test_ro.db"
    conn = sqlite3.connect(str(db_file))
    conn.execute("CREATE TABLE sample (id INT, val TEXT)")
    conn.execute("INSERT INTO sample VALUES (1, 'initial')")
    conn.commit()
    conn.close()

    # Open with read-only helper
    ro_conn = open_readonly_sqlite(db_file)
    assert ro_conn is not None

    cur = ro_conn.cursor()
    cur.execute("SELECT * FROM sample")
    rows = cur.fetchall()
    assert len(rows) == 1
    assert rows[0]["val"] == "initial"

    # Attempt write operations - must raise sqlite3.OperationalError
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        ro_conn.execute("INSERT INTO sample VALUES (2, 'blocked')")

    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        ro_conn.execute("UPDATE sample SET val = 'mutated' WHERE id = 1")

    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        ro_conn.execute("DROP TABLE sample")

    ro_conn.close()


def test_readonly_endpoints_disallow_mutations():
    """Verify that all new endpoints reject POST, PUT, DELETE, PATCH with 405 Method Not Allowed."""
    app = create_dashboard_app(None)
    client = TestClient(app)

    for path in ("/variants/equity", "/portfolio/variants", "/system/status", "/gate/decisions"):
        assert client.post(path).status_code == 405
        assert client.put(path).status_code == 405
        assert client.delete(path).status_code == 405
        assert client.patch(path).status_code == 405


def test_system_status_with_fundamental_eval_db_uptrend(tmp_path, monkeypatch):
    """Verify live BTC Regime Status card displays UPTREND from fundamental_eval.db."""
    eval_db = tmp_path / "test_uptrend.db"
    conn = sqlite3.connect(str(eval_db))
    with conn:
        conn.execute(
            """
            CREATE TABLE fundamental_evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                current_price REAL DEFAULT 100.0,
                verdict TEXT NOT NULL,
                risk_level TEXT DEFAULT 'LOW',
                confidence REAL DEFAULT 0.95,
                reason TEXT DEFAULT 'Bullish macro breakout',
                regime_state TEXT DEFAULT 'UPTREND',
                regime_blocked INTEGER DEFAULT 0,
                is_fallback INTEGER DEFAULT 0
            );
            """
        )
        conn.execute(
            """
            INSERT INTO fundamental_evaluations (timestamp, symbol, verdict, regime_state, regime_blocked)
            VALUES (?, 'BTCUSDT', 'APPROVE', 'UPTREND', 0)
            """,
            (datetime.now(timezone.utc).isoformat(),),
        )
    conn.close()

    monkeypatch.setenv("FUNDAMENTAL_EVAL_DB", str(eval_db))

    app = create_dashboard_app(None)
    client = TestClient(app)

    res = client.get("/system/status")
    assert res.status_code == 200
    data = res.json()
    assert data["btc_regime_state"] == "UPTREND"
    assert data["regime_blocked_recent"] is False


def test_system_status_with_fundamental_eval_db_downtrend(tmp_path, monkeypatch):
    """Verify live BTC Regime Status card displays DOWNTREND when regime filter is active."""
    eval_db = tmp_path / "test_downtrend.db"
    conn = sqlite3.connect(str(eval_db))
    with conn:
        conn.execute(
            """
            CREATE TABLE fundamental_evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                current_price REAL DEFAULT 50.0,
                verdict TEXT NOT NULL,
                risk_level TEXT DEFAULT 'HIGH',
                confidence REAL DEFAULT 0.88,
                reason TEXT DEFAULT 'BTC below 50-day SMA',
                regime_state TEXT DEFAULT 'DOWNTREND',
                regime_blocked INTEGER DEFAULT 1,
                is_fallback INTEGER DEFAULT 0
            );
            """
        )
        conn.execute(
            """
            INSERT INTO fundamental_evaluations (timestamp, symbol, verdict, regime_state, regime_blocked)
            VALUES (?, 'TAOUSDT', 'APPROVE', 'DOWNTREND', 1)
            """,
            (datetime.now(timezone.utc).isoformat(),),
        )
    conn.close()

    monkeypatch.setenv("FUNDAMENTAL_EVAL_DB", str(eval_db))

    app = create_dashboard_app(None)
    client = TestClient(app)

    res = client.get("/system/status")
    assert res.status_code == 200
    data = res.json()
    assert data["btc_regime_state"] == "DOWNTREND"
    assert data["regime_blocked_recent"] is True


def test_system_status_fallback_rate_calculation(tmp_path, monkeypatch):
    """Verify Fallback Rate (24h) KPI card accurately calculates percentage of FALLBACK_REJECT verdicts."""
    eval_db = tmp_path / "test_fallback_calc.db"
    now = datetime.now(timezone.utc)
    ts_recent = now.isoformat()
    ts_old = (now - timedelta(hours=48)).isoformat()

    conn = sqlite3.connect(str(eval_db))
    with conn:
        conn.execute(
            """
            CREATE TABLE fundamental_evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                current_price REAL DEFAULT 10.0,
                verdict TEXT NOT NULL,
                risk_level TEXT DEFAULT 'LOW',
                confidence REAL DEFAULT 0.9,
                reason TEXT DEFAULT 'Test',
                regime_state TEXT DEFAULT 'UPTREND',
                regime_blocked INTEGER DEFAULT 0,
                is_fallback INTEGER DEFAULT 0
            );
            """
        )
        # 3 recent evaluations: 1 FALLBACK_REJECT, 2 APPROVE -> 33.33% fallback rate
        conn.execute(
            "INSERT INTO fundamental_evaluations (timestamp, symbol, verdict, is_fallback) VALUES (?, 'SOLUSDT', 'FALLBACK_REJECT', 1)",
            (ts_recent,),
        )
        conn.execute(
            "INSERT INTO fundamental_evaluations (timestamp, symbol, verdict, is_fallback) VALUES (?, 'BTCUSDT', 'APPROVE', 0)",
            (ts_recent,),
        )
        conn.execute(
            "INSERT INTO fundamental_evaluations (timestamp, symbol, verdict, is_fallback) VALUES (?, 'ETHUSDT', 'APPROVE', 0)",
            (ts_recent,),
        )
        # 1 old evaluation outside 24h window (should not be counted)
        conn.execute(
            "INSERT INTO fundamental_evaluations (timestamp, symbol, verdict, is_fallback) VALUES (?, 'NEARUSDT', 'FALLBACK_REJECT', 1)",
            (ts_old,),
        )
    conn.close()

    monkeypatch.setenv("FUNDAMENTAL_EVAL_DB", str(eval_db))

    app = create_dashboard_app(None)
    client = TestClient(app)

    res = client.get("/system/status")
    assert res.status_code == 200
    data = res.json()
    assert data["total_evals_24h"] == 3
    assert data["fallback_count_24h"] == 1
    assert data["fallback_rate_24h"] == pytest.approx(33.33, rel=1e-2)


def test_system_status_variant_positions_count():
    """Verify active positions KPI card counts open positions across each of the 5 variants."""
    executor = PaperExecutor(make_config())
    session = PaperSession(stream=MockKlineStream(), executor=executor)

    # Put 1 open position in variant_a and variant_b
    c1 = make_candle(opened=START)
    sig = make_signal(opened=START)
    session.executors["variant_a"].on_candle_closed(c1)
    session.executors["variant_a"].submit_signal(sig)
    obs = make_observation(opened=START + timedelta(hours=1), price=D("10"))
    session.executors["variant_a"].on_open_observation(obs)

    app = create_dashboard_app(session)
    client = TestClient(app)

    res = client.get("/system/status")
    assert res.status_code == 200
    pos = res.json()["variant_positions"]
    assert pos["variant_a"] == 1
    assert pos["variant_b"] == 0
    assert pos["total"] == 1


def test_gate_decisions_feed_columns_and_filtering(tmp_path, monkeypatch):
    """Verify fundamental gate decision feed returns all required columns and supports verdict filtering."""
    eval_db = tmp_path / "test_decisions.db"
    now_iso = datetime.now(timezone.utc).isoformat()

    conn = sqlite3.connect(str(eval_db))
    with conn:
        conn.execute(
            """
            CREATE TABLE fundamental_evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                mode TEXT DEFAULT 'ENTRY',
                verdict TEXT NOT NULL,
                risk_level TEXT DEFAULT 'LOW',
                confidence REAL DEFAULT 0.92,
                reason TEXT NOT NULL,
                regime_state TEXT DEFAULT 'UPTREND',
                regime_blocked INTEGER DEFAULT 0,
                current_price REAL DEFAULT 150.0,
                is_fallback INTEGER DEFAULT 0
            );
            """
        )
        conn.execute(
            """
            INSERT INTO fundamental_evaluations (timestamp, symbol, mode, verdict, regime_state, regime_blocked, confidence, reason)
            VALUES (?, 'SOLUSDT', 'ENTRY', 'APPROVE', 'UPTREND', 0, 0.92, 'Strong developer momentum and TVL inflow')
            """,
            (now_iso,),
        )
        conn.execute(
            """
            INSERT INTO fundamental_evaluations (timestamp, symbol, mode, verdict, regime_state, regime_blocked, confidence, reason)
            VALUES (?, 'TAOUSDT', 'ENTRY', 'BLOCKED_BY_REGIME', 'DOWNTREND', 1, 1.0, 'Filtered by BTC downtrend 50 SMA')
            """,
            (now_iso,),
        )
        conn.execute(
            """
            INSERT INTO fundamental_evaluations (timestamp, symbol, mode, verdict, regime_state, regime_blocked, confidence, reason, is_fallback)
            VALUES (?, 'FETUSDT', 'REVIEW', 'FALLBACK_REJECT', 'UPTREND', 0, 0.5, 'LLM timeout fallback reject', 1)
            """,
            (now_iso,),
        )
    conn.close()

    monkeypatch.setenv("FUNDAMENTAL_EVAL_DB", str(eval_db))

    app = create_dashboard_app(None)
    client = TestClient(app)

    # 1. Fetch all
    res = client.get("/gate/decisions")
    assert res.status_code == 200
    decisions = res.json()
    assert len(decisions) == 3

    sol = next(d for d in decisions if d["symbol"] == "SOLUSDT")
    assert sol["mode"] == "ENTRY"
    assert sol["verdict"] == "APPROVE"
    assert sol["regime_state"] == "UPTREND"
    assert sol["regime_blocked"] is False
    assert sol["confidence"] == pytest.approx(0.92)
    assert "TVL inflow" in sol["reason"]

    tao = next(d for d in decisions if d["symbol"] == "TAOUSDT")
    assert tao["verdict"] == "BLOCKED_BY_REGIME"
    assert tao["regime_state"] == "DOWNTREND"
    assert tao["regime_blocked"] is True

    # 2. Filter by verdict
    res_app = client.get("/gate/decisions?verdict=APPROVE")
    assert len(res_app.json()) == 1
    assert res_app.json()[0]["symbol"] == "SOLUSDT"

    res_fb = client.get("/gate/decisions?verdict=FALLBACK_REJECT")
    assert len(res_fb.json()) == 1
    assert res_fb.json()[0]["symbol"] == "FETUSDT"


def test_variants_equity_read_from_sqlite_databases(tmp_path, monkeypatch):
    """Verify portfolio charting reads from variant databases on disk (e.g. portfolio_variant_a.db through e.db)."""
    # Create variant A and Variant D databases
    db_dir = tmp_path / "variant_dbs"
    db_dir.mkdir()

    db_a = db_dir / "portfolio_variant_a.db"
    conn_a = sqlite3.connect(str(db_a))
    with conn_a:
        conn_a.execute(
            """
            CREATE TABLE paper_trades (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                exit_time TEXT,
                pnl REAL
            );
            """
        )
        conn_a.execute(
            "INSERT INTO paper_trades VALUES (1, 'BTCUSDT', '2026-10-12T05:00:00Z', 5.50)"
        )
        conn_a.execute(
            "INSERT INTO paper_trades VALUES (2, 'ETHUSDT', '2026-10-12T06:00:00Z', 2.25)"
        )
    conn_a.close()

    db_d = db_dir / "portfolio_variant_d.db"
    conn_d = sqlite3.connect(str(db_d))
    with conn_d:
        conn_d.execute(
            """
            CREATE TABLE paper_trades (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                exit_time TEXT,
                pnl REAL
            );
            """
        )
        conn_d.execute(
            "INSERT INTO paper_trades VALUES (1, 'TAOUSDT', '2026-10-12T05:30:00Z', -4.00)"
        )
    conn_d.close()

    monkeypatch.setenv("PORTFOLIO_DB_DIR", str(db_dir))

    app = create_dashboard_app(None)
    client = TestClient(app)

    res = client.get("/variants/equity")
    assert res.status_code == 200
    data = res.json()["variants"]

    # Variant A has 2 closed trades totaling +7.75 PnL
    var_a = data["variant_a"]
    assert var_a["cumulative_pnl"] == pytest.approx(7.75)
    assert var_a["current_equity"] == pytest.approx(107.75)
    assert var_a["closed_trades_count"] == 2
    assert len(var_a["points"]) == 3  # Initial point + 2 trade points

    # Variant D has 1 closed trade totaling -4.00 PnL
    var_d = data["variant_d"]
    assert var_d["cumulative_pnl"] == pytest.approx(-4.00)
    assert var_d["current_equity"] == pytest.approx(96.00)
    assert var_d["closed_trades_count"] == 1

    # Missing variants (B, C, E) gracefully return Day 0 defaults
    assert data["variant_b"]["cumulative_pnl"] == 0.0
    assert data["variant_b"]["current_equity"] == 100.0


def test_ui_renders_experiment_components():
    """Verify that dashboard HTML UI contains multi-line chart, BTC regime card, and gate decision feed."""
    app = create_dashboard_app(None)
    client = TestClient(app)

    res = client.get("/ui")
    assert res.status_code == 200
    text = res.text

    # Experiment header & 5 variants
    assert "5-VARIANT EXPERIMENT" in text
    assert "Variant A" in text
    assert "Variant B" in text
    assert "Variant C" in text
    assert "Variant D" in text
    assert "Variant E" in text

    # Section 1 KPI cards
    assert "BTC Regime Status" in text
    assert "Fallback Rate (24h)" in text
    assert "Open Positions (5 Variants)" in text

    # Section 2 Multi-line chart
    assert "equityChart" in text
    assert "chart.js" in text.lower()
    assert "Cumulative PnL ($)" in text
    assert "Total Equity ($)" in text

    # Section 3 Fundamental gate decision feed
    assert "Fundamental Gate Decision Feed" in text
    assert "Timestamp" in text
    assert "Regime State" in text
    assert "Regime Blocked" in text
    assert "Confidence" in text
    assert "Reason / News Commentary" in text

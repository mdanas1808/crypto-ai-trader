import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.analysis.explainer import TradeExplainer
from app.backtest.models import BacktestConfig, Fill, Trade
from app.database.database import Base
from app.database.paper_models import PaperSignalRecord, PaperTradeRecord
from app.market.constants import EXCHANGE
from app.market.models import Candle, CandleOpenObservation
from app.strategy.breakout import BreakoutSignal
from app.trading.alerts import AlertHandler, CompositeAlertHandler
from app.trading.paper import PaperExecutor
from app.trading.persistence import DatabaseAlertHandler
from app.trading.session import PaperSession

UTC = timezone.utc
START = datetime(2024, 1, 2, tzinfo=UTC)
D = Decimal


@pytest.fixture
def sqlite_session_factory():
    """Create an in-memory SQLite database and return a sessionmaker factory."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    yield factory
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


def make_signal(symbol="BTCUSDT", opened=START, *, atr=2.5) -> BreakoutSignal:
    return BreakoutSignal(
        exchange=EXCHANGE,
        symbol=symbol,
        side="long",
        decision_time=opened + timedelta(hours=1),
        signal_open_time=opened,
        atr_14=atr,
    )


def make_trade(
    symbol="BTCUSDT",
    opened=START,
    *,
    exit_reason="take_profit",
    entry_price=D("10.0050"),
    exit_price=D("13.0000"),
    quantity=D("0.333333"),
    net_pnl=D("0.9900"),
) -> Trade:
    sig = make_signal(symbol, opened)
    entry_fill = Fill(
        symbol=symbol,
        exchange=EXCHANGE,
        side="buy",
        timestamp=opened + timedelta(hours=1),
        reference_price=D("10.0000"),
        price=entry_price,
        quantity=quantity,
        fee=D("0.0033"),
        reason="entry",
    )
    exit_fill = Fill(
        symbol=symbol,
        exchange=EXCHANGE,
        side="sell",
        timestamp=opened + timedelta(hours=2),
        reference_price=D("13.0050"),
        price=exit_price,
        quantity=quantity,
        fee=D("0.0043"),
        reason=exit_reason,
    )
    return Trade(
        signal=sig,
        quantity=quantity,
        entry_fill=entry_fill,
        exit_fill=exit_fill,
        stop_price=D("8.5050"),
        target_price=D("13.0050"),
        exit_reason=exit_reason,
        gross_pnl=D("0.9983"),
        net_pnl=net_pnl,
    )


def test_database_alert_handler_implements_protocol():
    handler = DatabaseAlertHandler(lambda: None)  # type: ignore
    assert isinstance(handler, AlertHandler)


def test_database_alert_handler_persists_signal(sqlite_session_factory):
    handler = DatabaseAlertHandler(sqlite_session_factory)
    sig = make_signal()
    explanation = "Breakout signal fired due to 4H EMA alignment and 2x volume surge."

    handler.on_signal_generated(sig, explanation)

    session = sqlite_session_factory()
    try:
        records = session.query(PaperSignalRecord).all()
        assert len(records) == 1
        r = records[0]
        assert r.symbol == "BTCUSDT"
        assert r.exchange == "binance_spot"
        assert r.side == "long"
        # SQLite returns naive datetime for timestamps; normalize to UTC for comparison
        dt = r.decision_time if r.decision_time.tzinfo else r.decision_time.replace(tzinfo=UTC)
        st = (
            r.signal_open_time
            if r.signal_open_time.tzinfo
            else r.signal_open_time.replace(tzinfo=UTC)
        )
        assert dt == sig.decision_time
        assert st == sig.signal_open_time
        assert round(Decimal(str(r.atr_14)), 2) == round(D("2.5"), 2)
        assert r.explanation == explanation
    finally:
        session.close()


def test_database_alert_handler_persists_trade(sqlite_session_factory):
    handler = DatabaseAlertHandler(sqlite_session_factory)
    trade = make_trade()

    handler.on_exit_filled(trade, "Trade exit explanation")

    session = sqlite_session_factory()
    try:
        records = session.query(PaperTradeRecord).all()
        assert len(records) == 1
        r = records[0]
        assert r.symbol == "BTCUSDT"
        assert r.exchange == "binance_spot"
        assert r.side == "long"
        assert round(Decimal(str(r.entry_price)), 4) == round(trade.entry_fill.price, 4)
        assert round(Decimal(str(r.exit_price)), 4) == round(trade.exit_fill.price, 4)
        assert round(Decimal(str(r.quantity)), 6) == round(trade.quantity, 6)
        assert round(Decimal(str(r.pnl)), 4) == round(trade.net_pnl, 4)
        assert r.exit_reason == "take_profit"
        et = r.entry_time if r.entry_time.tzinfo else r.entry_time.replace(tzinfo=UTC)
        xt = r.exit_time if r.exit_time.tzinfo else r.exit_time.replace(tzinfo=UTC)
        assert et == trade.entry_fill.timestamp
        assert xt == trade.exit_fill.timestamp
    finally:
        session.close()


def test_database_alert_handler_resilient_to_failures(caplog):
    def failing_session_factory():
        raise ConnectionError("Database unreachable")

    handler = DatabaseAlertHandler(failing_session_factory)
    sig = make_signal()
    trade = make_trade()

    with caplog.at_level(logging.ERROR):
        # Should not raise exception
        handler.on_signal_generated(sig, "test")
        handler.on_exit_filled(trade, "test")

    assert "Failed to persist paper signal" in caplog.text
    assert "Failed to persist paper trade" in caplog.text


def test_composite_alert_handler_multiplexing():
    class MockHandler:
        def __init__(self):
            self.signal_called = False
            self.entry_called = False
            self.exit_called = False
            self.breach_called = False

        def on_signal_generated(self, s, e):
            self.signal_called = True

        def on_entry_filled(self, p):
            self.entry_called = True

        def on_exit_filled(self, t, e):
            self.exit_called = True

        def on_daily_loss_breached(self, ex):
            self.breach_called = True

    h1 = MockHandler()
    h2 = MockHandler()
    composite = CompositeAlertHandler([h1, h2])
    assert isinstance(composite, AlertHandler)

    sig = make_signal()
    trade = make_trade()
    composite.on_signal_generated(sig, "expl")
    composite.on_entry_filled(object())  # type: ignore
    composite.on_exit_filled(trade, "expl")
    composite.on_daily_loss_breached(object())  # type: ignore

    assert h1.signal_called and h2.signal_called
    assert h1.entry_called and h2.entry_called
    assert h1.exit_called and h2.exit_called
    assert h1.breach_called and h2.breach_called


def test_composite_alert_handler_resilience_to_child_failure(caplog):
    class CrashingHandler:
        def on_signal_generated(self, s, e):
            raise ValueError("Child error")

        def on_entry_filled(self, p):
            raise ValueError("Child error")

        def on_exit_filled(self, t, e):
            raise ValueError("Child error")

        def on_daily_loss_breached(self, ex):
            raise ValueError("Child error")

    class WorkingHandler:
        def __init__(self):
            self.called = False

        def on_signal_generated(self, s, e):
            self.called = True

        def on_entry_filled(self, p):
            pass

        def on_exit_filled(self, t, e):
            pass

        def on_daily_loss_breached(self, ex):
            pass

    crashing = CrashingHandler()
    working = WorkingHandler()
    composite = CompositeAlertHandler([crashing, working])

    with caplog.at_level(logging.WARNING):
        composite.on_signal_generated(make_signal(), "expl")

    assert working.called
    assert "Error in composite alert handler CrashingHandler" in caplog.text


@pytest.mark.asyncio
async def test_paper_session_with_multiple_handlers_list(sqlite_session_factory):
    class DummyStream:
        async def events(self):
            if False:
                yield None

    class MockHandler:
        def __init__(self):
            self.signals = []
            self.exits = []

        def on_signal_generated(self, s, e):
            self.signals.append(s)

        def on_entry_filled(self, p):
            pass

        def on_exit_filled(self, t, e):
            self.exits.append(t)

        def on_daily_loss_breached(self, ex):
            pass

    mock_handler = MockHandler()
    db_handler = DatabaseAlertHandler(sqlite_session_factory)
    executor = PaperExecutor(BacktestConfig(minimum_quantity=D("0.000001")))

    # Pass multiple alert handlers as a list into PaperSession
    session = PaperSession(
        stream=DummyStream(),
        executor=executor,
        alert_handler=[mock_handler, db_handler],
        explainer=TradeExplainer(),
    )

    # 1. Closed candle produces signal
    c1 = Candle(
        exchange=EXCHANGE,
        symbol="BTCUSDT",
        interval="1h",
        open_time=START,
        open=D("10"),
        high=D("10.2"),
        low=D("9.9"),
        close=D("10"),
        volume=D("1"),
    )
    sig = make_signal(opened=START)
    session.strategy = lambda c: [sig]
    await session.process_event(c1)

    assert len(mock_handler.signals) == 1

    # Verify DB recorded the signal
    db_session = sqlite_session_factory()
    try:
        signals = db_session.query(PaperSignalRecord).all()
        assert len(signals) == 1
        assert signals[0].symbol == "BTCUSDT"
    finally:
        db_session.close()

    # 2. Open observation fills entry
    obs = CandleOpenObservation(
        exchange=EXCHANGE,
        symbol="BTCUSDT",
        interval="1h",
        open_time=START + timedelta(hours=1),
        observed_at=START + timedelta(hours=1, seconds=1),
        open=D("10"),
    )
    await session.process_event(obs)

    # 3. Next candle closes breaching profit target
    c2 = Candle(
        exchange=EXCHANGE,
        symbol="BTCUSDT",
        interval="1h",
        open_time=START + timedelta(hours=1),
        open=D("10"),
        high=D("25"),
        low=D("9.5"),
        close=D("20"),
        volume=D("1"),
    )
    await session.process_event(c2)

    assert len(mock_handler.exits) == 1

    # Verify DB recorded the trade
    db_session = sqlite_session_factory()
    try:
        trades = db_session.query(PaperTradeRecord).all()
        assert len(trades) == 1
        assert trades[0].symbol == "BTCUSDT"
        assert trades[0].exit_reason == "take_profit"
    finally:
        db_session.close()

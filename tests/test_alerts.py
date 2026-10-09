from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import logging
import pytest

from app.analysis.explainer import TradeExplainer
from app.backtest.data import MultiTimeframeBacktestData
from app.backtest.models import BacktestConfig, Position, Trade
from app.market.constants import EXCHANGE
from app.market.models import Candle, CandleOpenObservation
from app.strategy.breakout import BreakoutSignal
from app.trading.alerts import AlertHandler, LoggingAlertHandler
from app.trading.paper import PaperExecutor
from app.trading.session import PaperSession

import pandas as pd

UTC = timezone.utc
START = datetime(2024, 1, 2, tzinfo=UTC)
D = Decimal


class MockKlineStream:
    def __init__(self, events: list[Candle | CandleOpenObservation] | None = None) -> None:
        self._events = list(events or [])

    async def events(self) -> AsyncIterator[Candle | CandleOpenObservation]:
        for event in self._events:
            yield event


class RecordingAlertHandler:
    """Mock alert handler recording all callback invocations."""

    def __init__(self) -> None:
        self.signals: list[tuple[BreakoutSignal, str]] = []
        self.entries: list[Position] = []
        self.exits: list[tuple[Trade, str]] = []
        self.breaches: list[PaperExecutor] = []

    def on_signal_generated(self, signal: BreakoutSignal, explanation: str) -> None:
        self.signals.append((signal, explanation))

    def on_entry_filled(self, position: Position) -> None:
        self.entries.append(position)

    def on_exit_filled(self, trade: Trade, explanation: str) -> None:
        self.exits.append((trade, explanation))

    def on_daily_loss_breached(self, executor: PaperExecutor) -> None:
        self.breaches.append(executor)


class FailingAlertHandler:
    """Mock alert handler that raises exceptions in all callbacks."""

    def on_signal_generated(self, signal: BreakoutSignal, explanation: str) -> None:
        raise RuntimeError("Signal alert failed")

    def on_entry_filled(self, position: Position) -> None:
        raise RuntimeError("Entry alert failed")

    def on_exit_filled(self, trade: Trade, explanation: str) -> None:
        raise RuntimeError("Exit alert failed")

    def on_daily_loss_breached(self, executor: PaperExecutor) -> None:
        raise RuntimeError("Breach alert failed")


class AsyncRecordingAlertHandler:
    """Mock async alert handler whose callbacks are coroutines."""

    def __init__(self) -> None:
        self.signals: list[tuple[BreakoutSignal, str]] = []
        self.entries: list[Position] = []
        self.exits: list[tuple[Trade, str]] = []
        self.breaches: list[PaperExecutor] = []

    async def on_signal_generated(self, signal: BreakoutSignal, explanation: str) -> None:
        self.signals.append((signal, explanation))

    async def on_entry_filled(self, position: Position) -> None:
        self.entries.append(position)

    async def on_exit_filled(self, trade: Trade, explanation: str) -> None:
        self.exits.append((trade, explanation))

    async def on_daily_loss_breached(self, executor: PaperExecutor) -> None:
        self.breaches.append(executor)


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


def make_test_context_data(opened=START) -> MultiTimeframeBacktestData:
    context_rows = [
        {
            "exchange": "binance_spot",
            "symbol": "BTCUSDT",
            "open_time": opened + timedelta(hours=i),
            "high": D("100.00"),
        }
        for i in range(-20, 0)
    ]
    context = pd.DataFrame(
        context_rows,
        index=pd.DatetimeIndex(
            [opened + timedelta(hours=i + 1) for i in range(-20, 0)],
            name="decision_time",
            tz="UTC",
        ),
    )
    trading_rows = [
        {
            "exchange": "binance_spot",
            "symbol": "BTCUSDT",
            "interval": "1h",
            "open_time": opened,
            "open": D("100.00"),
            "high": D("102.50"),
            "low": D("99.50"),
            "close": D("101.50"),
            "volume": D("30.00"),
            "volume_sma_20": 15.00,
            "ema_200": 95.00,
            "rsi_14": 62.50,
            "atr_14": 2.50,
            "4h_close": 110.00,
            "4h_ema_20": 105.00,
            "4h_ema_50": 100.00,
            "4h_ema_200": 98.00,
        }
    ]
    trading = pd.DataFrame(
        trading_rows,
        index=pd.DatetimeIndex([opened + timedelta(hours=1)], name="decision_time", tz="UTC"),
    )
    return MultiTimeframeBacktestData(context_frame=context, trading_frame=trading)


def test_logging_alert_handler_implements_protocol():
    handler = LoggingAlertHandler()
    assert isinstance(handler, AlertHandler)


def test_logging_alert_handler_output(caplog):
    handler = LoggingAlertHandler()
    sig = make_signal()

    with caplog.at_level(logging.INFO):
        handler.on_signal_generated(sig, "Detailed signal explanation text")
    assert "SIGNAL GENERATED" in caplog.text
    assert "BTCUSDT" in caplog.text
    assert "Detailed signal explanation text" in caplog.text

    caplog.clear()
    pos = Position(
        signal=sig,
        quantity=D("0.5"),
        entry_fill=make_observation(),  # dummy
        stop_price=D("8.5"),
        target_price=D("13.0"),
        initial_risk=D("0.75"),
    )
    # mock entry fill for logging test
    pos = Position(
        signal=sig,
        quantity=D("0.5"),
        entry_fill=type(
            "MockFill",
            (),
            {"price": D("10.005"), "fee": D("0.005"), "notional": D("5.0025")},
        )(),  # type: ignore
        stop_price=D("8.5"),
        target_price=D("13.0"),
        initial_risk=D("0.75"),
    )
    with caplog.at_level(logging.INFO):
        handler.on_entry_filled(pos)
    assert "ENTRY FILLED" in caplog.text
    assert "Price=10.005" in caplog.text
    assert "Stop=8.5" in caplog.text

    caplog.clear()
    trade = type(
        "MockTrade",
        (),
        {
            "signal": sig,
            "exit_reason": "take_profit",
            "exit_fill": type(
                "MockFill",
                (),
                {"timestamp": START + timedelta(hours=2)},
            )(),
        },
    )()
    with caplog.at_level(logging.INFO):
        handler.on_exit_filled(trade, "Exit explanation text")  # type: ignore
    assert "EXIT FILLED" in caplog.text
    assert "TAKE_PROFIT" in caplog.text
    assert "Exit explanation text" in caplog.text

    caplog.clear()
    executor = PaperExecutor(make_config())
    with caplog.at_level(logging.WARNING):
        handler.on_daily_loss_breached(executor)
    assert "DAILY LOSS LIMIT BREACHED" in caplog.text


@pytest.mark.asyncio
async def test_session_alerts_on_signal_generated():
    executor = PaperExecutor(make_config())
    handler = RecordingAlertHandler()
    explainer = TradeExplainer()
    context_data = make_test_context_data()
    sig = make_signal(opened=START)

    session = PaperSession(
        stream=MockKlineStream(),
        executor=executor,
        strategy=lambda d: [sig],
        data_provider=lambda c: context_data,
        alert_handler=handler,
        explainer=explainer,
    )

    c1 = make_candle(opened=START)
    await session.process_event(c1)

    assert len(handler.signals) == 1
    signal_received, explanation = handler.signals[0]
    assert signal_received == sig
    assert "BTCUSDT" in explanation
    assert "Regime (4H)" in explanation
    assert "Trend & Breakout (1H)" in explanation


@pytest.mark.asyncio
async def test_session_alerts_on_entry_filled():
    executor = PaperExecutor(make_config())
    handler = RecordingAlertHandler()
    session = PaperSession(
        stream=MockKlineStream(),
        executor=executor,
        alert_handler=handler,
    )

    # 1. Prepare closed candle and signal
    c1 = make_candle(opened=START)
    executor.on_candle_closed(c1)
    sig = make_signal(opened=START)
    executor.submit_signal(sig)

    # 2. Open observation fills entry
    obs = make_observation(opened=START + timedelta(hours=1))
    pos = await session.process_event(obs)

    assert pos is not None
    assert len(handler.entries) == 1
    assert handler.entries[0] == pos
    assert handler.entries[0].symbol == "BTCUSDT"


@pytest.mark.asyncio
async def test_session_alerts_on_exit_filled():
    executor = PaperExecutor(make_config())
    handler = RecordingAlertHandler()
    explainer = TradeExplainer()
    session = PaperSession(
        stream=MockKlineStream(),
        executor=executor,
        alert_handler=handler,
        explainer=explainer,
    )

    # Enter position
    c1 = make_candle(opened=START)
    executor.on_candle_closed(c1)
    sig = make_signal(opened=START)
    executor.submit_signal(sig)
    obs = make_observation(opened=START + timedelta(hours=1), price=D("10"))
    assert session.executor.on_open_observation(obs) is not None

    # Candle 2 breaches target -> trade exit
    c2 = make_candle(
        opened=START + timedelta(hours=1),
        open=D("10"),
        high=D("25"),
        low=D("9.5"),
        close=D("20"),
    )
    trade = await session.process_event(c2)

    assert trade is not None
    assert len(handler.exits) == 1
    trade_received, explanation = handler.exits[0]
    assert trade_received == trade
    assert "Take Profit Hit" in explanation
    assert "Net PnL" in explanation


@pytest.mark.asyncio
async def test_session_alerts_on_daily_loss_breached_transition():
    # Setup executor with a step that will breach daily loss (> 2 USDT loss from 100 baseline)
    executor = PaperExecutor(make_config(quantity_step=D("0.1"), minimum_quantity=D("0.1")))
    handler = RecordingAlertHandler()
    session = PaperSession(
        stream=MockKlineStream(),
        executor=executor,
        alert_handler=handler,
    )

    # Open position on BTCUSDT
    c1 = make_candle(opened=START)
    executor.on_candle_closed(c1)
    sig = make_signal(opened=START)
    executor.submit_signal(sig)
    obs = make_observation(opened=START + timedelta(hours=1), price=D("10"))
    await session.process_event(obs)
    assert not executor.daily_loss_breached
    assert len(handler.breaches) == 0

    # Candle 2 causes severe loss: opens at 1, drops
    c2 = make_candle(
        opened=START + timedelta(hours=1),
        open=D("1"),
        high=D("2"),
        low=D("0.5"),
        close=D("1"),
    )
    trade = await session.process_event(c2)
    assert trade is not None
    assert executor.daily_loss_breached
    assert len(handler.breaches) == 1

    # Candle 3 closes: daily loss is STILL breached, but transition already fired
    c3 = make_candle(
        opened=START + timedelta(hours=2),
        open=D("1"),
        high=D("1.5"),
        low=D("0.8"),
        close=D("1"),
    )
    await session.process_event(c3)
    # Must NOT call on_daily_loss_breached again
    assert len(handler.breaches) == 1


@pytest.mark.asyncio
async def test_session_resilient_to_alert_handler_failures(caplog):
    executor = PaperExecutor(make_config())
    handler = FailingAlertHandler()
    explainer = TradeExplainer()
    session = PaperSession(
        stream=MockKlineStream(),
        executor=executor,
        alert_handler=handler,
        explainer=explainer,
    )

    # 1. Signal & candle closed
    c1 = make_candle(opened=START)
    sig = make_signal(opened=START)
    session.strategy = lambda c: [sig]

    with caplog.at_level(logging.WARNING):
        # Should NOT raise RuntimeError despite handler failing
        trade = await session.process_event(c1)
    assert trade is None
    assert len(executor.pending_signals) == 1
    assert "Error in alert handler callback on_signal_generated" in caplog.text

    # 2. Open observation fills entry
    caplog.clear()
    obs = make_observation(opened=START + timedelta(hours=1))
    with caplog.at_level(logging.WARNING):
        pos = await session.process_event(obs)
    assert pos is not None
    assert "BTCUSDT" in executor.positions
    assert "Error in alert handler callback on_entry_filled" in caplog.text

    # 3. Exit filled
    caplog.clear()
    c2 = make_candle(
        opened=START + timedelta(hours=1),
        open=D("10"),
        high=D("25"),
        low=D("9.5"),
        close=D("20"),
    )
    with caplog.at_level(logging.WARNING):
        exit_trade = await session.process_event(c2)
    assert exit_trade is not None
    assert "Error in alert handler callback on_exit_filled" in caplog.text


@pytest.mark.asyncio
async def test_session_supports_async_alert_handler():
    executor = PaperExecutor(make_config())
    handler = AsyncRecordingAlertHandler()
    session = PaperSession(
        stream=MockKlineStream(),
        executor=executor,
        alert_handler=handler,
    )

    c1 = make_candle(opened=START)
    sig = make_signal(opened=START)
    session.strategy = lambda c: [sig]

    await session.process_event(c1)
    assert len(handler.signals) == 1

    obs = make_observation(opened=START + timedelta(hours=1))
    pos = await session.process_event(obs)
    assert pos is not None
    assert len(handler.entries) == 1


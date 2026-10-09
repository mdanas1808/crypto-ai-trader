from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import asyncio
import os
import signal
import pytest

from app.backtest.models import BacktestConfig
from app.market.constants import EXCHANGE
from app.market.models import Candle, CandleOpenObservation
from app.strategy.breakout import BreakoutSignal
from app.trading.paper import PaperExecutor
from app.trading.session import PaperSession

UTC = timezone.utc
START = datetime(2024, 1, 2, tzinfo=UTC)
D = Decimal


class MockKlineStream:
    """Mock stream that yields predefined events asynchronously."""

    def __init__(self, events: list[Candle | CandleOpenObservation] | None = None) -> None:
        self._events = list(events or [])

    def add_event(self, event: Candle | CandleOpenObservation) -> None:
        self._events.append(event)

    async def events(self) -> AsyncIterator[Candle | CandleOpenObservation]:
        for event in self._events:
            yield event


class InfiniteMockStream:
    """Mock stream that yields events and sleeps until cancelled."""

    def __init__(self, initial_events: list[Candle | CandleOpenObservation] | None = None) -> None:
        self._initial = list(initial_events or [])

    async def events(self) -> AsyncIterator[Candle | CandleOpenObservation]:
        for event in self._initial:
            yield event
        while True:
            await asyncio.sleep(0.01)


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


def test_session_init_validates_types():
    executor = PaperExecutor(make_config())
    stream = MockKlineStream()

    with pytest.raises(TypeError, match="executor must be PaperExecutor"):
        PaperSession(stream=stream, executor="not_an_executor")  # type: ignore

    with pytest.raises(TypeError, match="stream must provide a callable events"):
        PaperSession(stream=object(), executor=executor)

    session = PaperSession(stream=stream, executor=executor)
    assert session.stream is stream
    assert session.executor is executor
    assert session.events_processed == 0
    assert session.signals_submitted == 0
    assert not session.is_running
    assert session.positions == {}
    assert session.trades == ()
    assert session.pending_signals == ()


@pytest.mark.asyncio
async def test_session_routes_open_observation_to_executor():
    executor = PaperExecutor(make_config())
    # Prepare closed candle and pending signal
    c1 = make_candle(opened=START)
    executor.on_candle_closed(c1)
    sig = make_signal(opened=START)
    assert executor.submit_signal(sig)
    assert len(executor.pending_signals) == 1

    stream = MockKlineStream()
    session = PaperSession(stream=stream, executor=executor)

    # Process next open observation
    obs = make_observation(opened=START + timedelta(hours=1), price=D("10"))
    position = await session.process_event(obs)

    assert position is not None
    assert session.events_processed == 1
    assert "BTCUSDT" in session.positions
    assert len(session.pending_signals) == 0


@pytest.mark.asyncio
async def test_session_routes_completed_1h_candle_and_submits_signals():
    executor = PaperExecutor(make_config())
    stream = MockKlineStream()

    sig = make_signal(opened=START)
    strategy_called = []

    def mock_strategy(candle):
        strategy_called.append(candle)
        return [sig]

    session = PaperSession(stream=stream, executor=executor, strategy=mock_strategy)

    c1 = make_candle(opened=START)
    trade = await session.process_event(c1)

    assert trade is None
    assert session.events_processed == 1
    assert session.signals_submitted == 1
    assert len(strategy_called) == 1
    assert strategy_called[0] == c1
    assert len(session.pending_signals) == 1
    assert session.pending_signals[0] == sig


@pytest.mark.asyncio
async def test_session_data_provider_passes_data_to_strategy():
    executor = PaperExecutor(make_config())
    stream = MockKlineStream()

    dummy_backtest_data = object()
    provider_called = []
    strategy_called = []
    sig = make_signal(opened=START)

    def mock_provider(candle):
        provider_called.append(candle)
        return dummy_backtest_data

    def mock_strategy(data):
        strategy_called.append(data)
        return [sig]

    session = PaperSession(
        stream=stream,
        executor=executor,
        strategy=mock_strategy,
        data_provider=mock_provider,
    )

    c1 = make_candle(opened=START)
    await session.process_event(c1)

    assert len(provider_called) == 1
    assert provider_called[0] == c1
    assert len(strategy_called) == 1
    assert strategy_called[0] is dummy_backtest_data
    assert session.signals_submitted == 1
    assert len(session.pending_signals) == 1


@pytest.mark.asyncio
async def test_session_ignores_non_1h_candles_for_1h_execution_and_strategy():
    executor = PaperExecutor(make_config())
    stream = MockKlineStream()

    strategy_called = []
    session = PaperSession(
        stream=stream,
        executor=executor,
        strategy=lambda c: strategy_called.append(c) or [],
    )

    four_hour = make_candle(opened=START, interval="4h")
    trade = await session.process_event(four_hour)

    assert trade is None
    assert session.events_processed == 1
    assert len(strategy_called) == 0
    assert session.signals_submitted == 0


@pytest.mark.asyncio
async def test_session_real_time_causality_flow():
    """Verify that signal from candle N waits for open observation of candle N+1."""
    executor = PaperExecutor(make_config())
    stream = MockKlineStream()

    sig = make_signal(opened=START)
    session = PaperSession(stream=stream, executor=executor, strategy=lambda c: [sig])

    # 1. Candle N closes: signal is generated and queued, but NOT filled yet.
    c1 = make_candle(opened=START)
    await session.process_event(c1)
    assert len(session.positions) == 0
    assert len(session.pending_signals) == 1

    # 2. Candle N+1 open observation arrives: fills the pending signal!
    obs = make_observation(opened=START + timedelta(hours=1), price=D("10.5"))
    position = await session.process_event(obs)
    assert position is not None
    assert position.entry_fill.reference_price == D("10.5")
    assert position.entry_fill.timestamp == START + timedelta(hours=1)
    assert len(session.positions) == 1
    assert len(session.pending_signals) == 0

    # 3. Candle N+1 closes hitting target: exit trade is produced!
    c2 = make_candle(
        opened=START + timedelta(hours=1),
        open=D("10.5"),
        high=D("20.0"),  # breaches profit target
        low=D("10.0"),
        close=D("19.0"),
    )
    trade = await session.process_event(c2)
    assert trade is not None
    assert trade.exit_reason == "take_profit"
    assert len(session.positions) == 0
    assert len(session.trades) == 1


@pytest.mark.asyncio
async def test_session_run_consumes_all_stream_events():
    executor = PaperExecutor(make_config())
    c1 = make_candle(opened=START)
    obs = make_observation(opened=START + timedelta(hours=1), price=D("10"))
    stream = MockKlineStream([c1, obs])

    sig = make_signal(opened=START)
    session = PaperSession(stream=stream, executor=executor, strategy=lambda c: [sig])

    await session.run()

    assert session.events_processed == 2
    assert session.signals_submitted == 1
    assert len(session.positions) == 1
    assert not session.is_running


@pytest.mark.asyncio
async def test_session_graceful_shutdown_via_stop():
    executor = PaperExecutor(make_config())
    stream = InfiniteMockStream()
    session = PaperSession(stream=stream, executor=executor)

    run_task = asyncio.create_task(session.run())
    await asyncio.sleep(0.02)
    assert session.is_running

    session.stop()
    await run_task
    assert not session.is_running


@pytest.mark.asyncio
async def test_session_graceful_shutdown_via_cancellation():
    executor = PaperExecutor(make_config())
    stream = InfiniteMockStream()
    session = PaperSession(stream=stream, executor=executor)

    run_task = asyncio.create_task(session.run())
    await asyncio.sleep(0.02)
    assert session.is_running

    run_task.cancel()
    # Does not raise CancelledError to caller
    await run_task
    assert not session.is_running


@pytest.mark.asyncio
async def test_session_graceful_shutdown_via_sigint():
    executor = PaperExecutor(make_config())
    stream = InfiniteMockStream()
    session = PaperSession(stream=stream, executor=executor)

    run_task = asyncio.create_task(session.run())
    await asyncio.sleep(0.02)
    assert session.is_running

    # Send SIGINT to own process to trigger registered signal handler
    os.kill(os.getpid(), signal.SIGINT)
    await run_task
    assert not session.is_running


@pytest.mark.asyncio
async def test_session_rejects_concurrent_runs():
    executor = PaperExecutor(make_config())
    stream = InfiniteMockStream()
    session = PaperSession(stream=stream, executor=executor)

    run_task = asyncio.create_task(session.run())
    await asyncio.sleep(0.02)

    with pytest.raises(RuntimeError, match="session is already running"):
        await session.run()

    session.stop()
    await run_task


@pytest.mark.asyncio
async def test_session_rejects_unsupported_event_type():
    executor = PaperExecutor(make_config())
    stream = MockKlineStream()
    session = PaperSession(stream=stream, executor=executor)

    with pytest.raises(TypeError, match="unsupported event type"):
        await session.process_event("invalid_event")  # type: ignore


@pytest.mark.asyncio
async def test_session_handles_async_data_provider_and_strategy():
    executor = PaperExecutor(make_config())
    stream = MockKlineStream()
    sig = make_signal(opened=START)

    async def async_provider(candle):
        await asyncio.sleep(0.001)
        return {"data": candle}

    async def async_strategy(data):
        await asyncio.sleep(0.001)
        return [sig]

    session = PaperSession(
        stream=stream,
        executor=executor,
        data_provider=async_provider,
        strategy=async_strategy,
    )

    c1 = make_candle(opened=START)
    await session.process_event(c1)

    assert session.signals_submitted == 1
    assert len(session.pending_signals) == 1


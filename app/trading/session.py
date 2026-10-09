"""Asynchronous paper trading session orchestrator.

Coordinates public Binance Kline event streams, deterministic breakout
strategy evaluation, and the real-time PaperExecutor in-memory.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
import inspect
import logging
import signal
from typing import Any

from app.backtest.data import MultiTimeframeBacktestData
from app.backtest.models import Position, Trade
from app.market.models import Candle, CandleOpenObservation
from app.strategy.breakout import BreakoutSignal, generate_breakout_signals
from app.trading.paper import PaperExecutor

logger = logging.getLogger(__name__)


class PaperSession:
    """Orchestrate live paper trading events and deterministic signal execution."""

    def __init__(
        self,
        stream: Any,
        executor: PaperExecutor,
        strategy: Callable[..., Sequence[BreakoutSignal]] | None = None,
        *,
        data_provider: Callable[[Candle], MultiTimeframeBacktestData | None] | None = None,
    ) -> None:
        if not isinstance(executor, PaperExecutor):
            raise TypeError("executor must be PaperExecutor")
        if not hasattr(stream, "events") or not callable(stream.events):
            raise TypeError("stream must provide a callable events() method")

        self.stream = stream
        self.executor = executor
        self.strategy = strategy if strategy is not None else generate_breakout_signals
        self.data_provider = data_provider

        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._events_processed = 0
        self._signals_submitted = 0

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def events_processed(self) -> int:
        return self._events_processed

    @property
    def signals_submitted(self) -> int:
        return self._signals_submitted

    @property
    def positions(self) -> dict[str, Position]:
        return self.executor.positions

    @property
    def trades(self) -> tuple[Trade, ...]:
        return self.executor.trades

    @property
    def pending_signals(self) -> tuple[BreakoutSignal, ...]:
        return self.executor.pending_signals

    async def process_event(
        self, event: Candle | CandleOpenObservation
    ) -> Position | Trade | None:
        """Route a single market event to executor and strategy."""
        if isinstance(event, CandleOpenObservation):
            position = self.executor.on_open_observation(event)
            self._events_processed += 1
            return position

        if isinstance(event, Candle):
            trade = None
            if event.interval == "1h":
                trade = self.executor.on_candle_closed(event)
                signals = await self._generate_signals(event)
                for sig in signals:
                    if self.executor.submit_signal(sig):
                        self._signals_submitted += 1
            self._events_processed += 1
            return trade

        raise TypeError(f"unsupported event type: {type(event).__name__}")

    async def _generate_signals(self, candle: Candle) -> Sequence[BreakoutSignal]:
        """Generate strategy signals for a completed candle."""
        if self.data_provider is not None:
            data = self.data_provider(candle)
            if inspect.iscoroutine(data):
                data = await data
            if data is None:
                return ()
            signals = self.strategy(data)
            if inspect.iscoroutine(signals):
                signals = await signals
            return signals or ()

        if self.strategy is not generate_breakout_signals:
            try:
                res = self.strategy(candle)
            except TypeError:
                res = self.strategy()
            if inspect.iscoroutine(res):
                res = await res
            return res or ()

        return ()

    def stop(self) -> None:
        """Signal the orchestrator to shut down gracefully."""
        self._running = False
        if self._task is not None and not self._task.done() and self._task is not asyncio.current_task():
            self._task.cancel()

    async def run(self) -> None:
        """Run the asynchronous event consumption loop."""
        if self._running:
            raise RuntimeError("session is already running")

        self._running = True
        self._task = asyncio.current_task()

        loop = asyncio.get_running_loop()
        signals_to_handle = (signal.SIGINT, signal.SIGTERM)
        registered_signals: list[signal.Signals] = []

        def _handle_signal() -> None:
            logger.info("Signal received, stopping paper session...")
            self.stop()

        for sig in signals_to_handle:
            try:
                loop.add_signal_handler(sig, _handle_signal)
                registered_signals.append(sig)
            except (NotImplementedError, RuntimeError, ValueError):
                pass

        try:
            async for event in self.stream.events():
                if not self._running:
                    break
                await self.process_event(event)
        except (asyncio.CancelledError, KeyboardInterrupt):
            logger.info("Paper session cancelled or interrupted, exiting gracefully.")
        finally:
            for sig in registered_signals:
                try:
                    loop.remove_signal_handler(sig)
                except (NotImplementedError, RuntimeError, ValueError):
                    pass
            self._running = False
            self._task = None


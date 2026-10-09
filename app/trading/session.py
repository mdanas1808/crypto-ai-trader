"""Asynchronous paper trading session orchestrator.

Coordinates public Binance Kline event streams, deterministic breakout
strategy evaluation, the real-time PaperExecutor, and optional alerting/explanations.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
import inspect
import logging
import signal
from typing import Any

from app.analysis.explainer import TradeExplainer
from app.backtest.data import MultiTimeframeBacktestData
from app.backtest.models import Position, Trade
from app.market.models import Candle, CandleOpenObservation
from app.strategy.breakout import BreakoutSignal, generate_breakout_signals
from app.trading.alerts import AlertHandler, CompositeAlertHandler
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
        alert_handler: AlertHandler | Sequence[AlertHandler] | None = None,
        explainer: TradeExplainer | None = None,
    ) -> None:
        if not isinstance(executor, PaperExecutor):
            raise TypeError("executor must be PaperExecutor")
        if not hasattr(stream, "events") or not callable(stream.events):
            raise TypeError("stream must provide a callable events() method")

        self.stream = stream
        self.executor = executor
        self.strategy = strategy if strategy is not None else generate_breakout_signals
        self.data_provider = data_provider
        if isinstance(alert_handler, (list, tuple, set)):
            self.alert_handler: AlertHandler | None = CompositeAlertHandler(alert_handler)
        else:
            self.alert_handler = alert_handler
        self.explainer = explainer

        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._events_processed = 0
        self._signals_submitted = 0
        self._daily_loss_notified = False

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

    async def _notify(self, method_name: str, *args: Any) -> None:
        """Safely invoke an alert handler callback without interrupting execution."""
        if self.alert_handler is None:
            return
        handler_method = getattr(self.alert_handler, method_name, None)
        if handler_method is None or not callable(handler_method):
            return
        try:
            res = handler_method(*args)
            if inspect.iscoroutine(res):
                await res
        except Exception as exc:
            logger.warning(
                "Error in alert handler callback %s: %s",
                method_name,
                exc,
                exc_info=True,
            )

    async def _check_daily_loss(self) -> None:
        """Trigger callback when daily loss transitions to breached."""
        is_breached = self.executor.daily_loss_breached
        if is_breached and not self._daily_loss_notified:
            self._daily_loss_notified = True
            await self._notify("on_daily_loss_breached", self.executor)
        elif not is_breached and self._daily_loss_notified:
            self._daily_loss_notified = False

    async def process_event(
        self, event: Candle | CandleOpenObservation
    ) -> Position | Trade | None:
        """Route a single market event to executor, strategy, and alert handlers."""
        if isinstance(event, CandleOpenObservation):
            position = self.executor.on_open_observation(event)
            self._events_processed += 1
            if position is not None and self.alert_handler is not None:
                await self._notify("on_entry_filled", position)
            await self._check_daily_loss()
            return position

        if isinstance(event, Candle):
            trade = None
            if event.interval == "1h":
                trade = self.executor.on_candle_closed(event)
                if trade is not None and self.alert_handler is not None:
                    explanation = ""
                    if self.explainer is not None:
                        try:
                            res = self.explainer.explain_trade_exit(trade)
                            if inspect.isawaitable(res):
                                explanation = await res
                            else:
                                explanation = res
                        except Exception as exc:
                            logger.warning(
                                "Failed to generate trade exit explanation: %s", exc
                            )
                    await self._notify("on_exit_filled", trade, explanation)

                signals, context_data = await self._generate_signals_with_context(event)
                for sig in signals:
                    if self.executor.submit_signal(sig):
                        self._signals_submitted += 1
                        if self.alert_handler is not None:
                            explanation = ""
                            if self.explainer is not None:
                                if isinstance(
                                    context_data, MultiTimeframeBacktestData
                                ):
                                    try:
                                        res = self.explainer.explain_signal(
                                            sig, context_data
                                        )
                                        if inspect.isawaitable(res):
                                            explanation = await res
                                        else:
                                            explanation = res
                                    except Exception as exc:
                                        logger.warning(
                                            "Failed to generate signal explanation: %s",
                                            exc,
                                        )
                                else:
                                    explanation = (
                                        f"Breakout Long Signal for {sig.symbol} "
                                        f"({sig.exchange}) at {sig.decision_time.isoformat()}."
                                    )
                            await self._notify(
                                "on_signal_generated", sig, explanation
                            )

                await self._check_daily_loss()

            self._events_processed += 1
            return trade

        raise TypeError(f"unsupported event type: {type(event).__name__}")

    async def _generate_signals(self, candle: Candle) -> Sequence[BreakoutSignal]:
        """Generate strategy signals for a completed candle."""
        signals, _ = await self._generate_signals_with_context(candle)
        return signals

    async def _generate_signals_with_context(
        self, candle: Candle
    ) -> tuple[Sequence[BreakoutSignal], Any]:
        """Generate strategy signals and return context data if available."""
        if self.data_provider is not None:
            data = self.data_provider(candle)
            if inspect.iscoroutine(data):
                data = await data
            if data is None:
                return (), None
            signals = self.strategy(data)
            if inspect.iscoroutine(signals):
                signals = await signals
            return signals or (), data

        if self.strategy is not generate_breakout_signals:
            try:
                res = self.strategy(candle)
            except TypeError:
                res = self.strategy()
            if inspect.iscoroutine(res):
                res = await res
            return res or (), None

        return (), None

    def stop(self) -> None:
        """Signal the orchestrator to shut down gracefully."""
        self._running = False
        if (
            self._task is not None
            and not self._task.done()
            and self._task is not asyncio.current_task()
        ):
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
        except Exception:
            logger.exception("Stream error in PaperSession")
        else:
            if self._running:
                logger.warning("Market stream ended unexpectedly.")
        finally:
            for sig in registered_signals:
                try:
                    loop.remove_signal_handler(sig)
                except (NotImplementedError, RuntimeError, ValueError):
                    pass
            self._running = False
            self._task = None

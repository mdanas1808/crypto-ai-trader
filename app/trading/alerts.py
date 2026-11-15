"""Alerting and event notification interfaces for paper trading sessions.

Provides the AlertHandler protocol, a default LoggingAlertHandler that
formats and logs strategy events, and a CompositeAlertHandler for multiplexing.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

from app.backtest.models import Position, Trade
from app.market.models import Candle
from app.strategy.breakout import BreakoutSignal
from app.trading.paper import PaperExecutor

logger = logging.getLogger(__name__)


@runtime_checkable
class AlertHandler(Protocol):
    """Protocol for receiving real-time paper trading event callbacks."""

    def on_candle_closed(self, candle: Candle) -> None:
        """Called when a completed candle is processed."""
        ...

    def on_signal_generated(self, signal: BreakoutSignal, explanation: str) -> None:
        """Called when a strategy signal is generated and submitted."""
        ...

    def on_entry_filled(self, position: Position) -> None:
        """Called when a pending signal is filled via CandleOpenObservation."""
        ...

    def on_exit_filled(self, trade: Trade, explanation: str) -> None:
        """Called when an open position is closed."""
        ...

    def on_daily_loss_breached(self, executor: PaperExecutor) -> None:
        """Called when portfolio daily loss limit is breached."""
        ...


class LoggingAlertHandler:
    """Default alert handler logging strategy events via standard library logging."""

    def __init__(self, logger: logging.Logger | None = None, level: int = logging.INFO) -> None:
        self.logger = logger or logging.getLogger("app.trading.alerts")
        self.level = level

    def on_candle_closed(self, candle: Candle) -> None:
        """Called when a completed candle is processed."""
        pass

    def on_signal_generated(self, signal: BreakoutSignal, explanation: str) -> None:
        msg = (
            f"SIGNAL GENERATED [{signal.exchange} {signal.symbol} "
            f"{signal.decision_time.isoformat()}]: {signal.side.upper()}"
        )
        if explanation:
            msg += f"\n{explanation}"
        self.logger.log(self.level, msg)

    def on_entry_filled(self, position: Position) -> None:
        self.logger.log(
            self.level,
            "ENTRY FILLED [%s %s]: Qty=%s @ Price=%s (Stop=%s, Target=%s)",
            position.exchange,
            position.symbol,
            position.quantity,
            position.entry_fill.price,
            position.stop_price,
            position.target_price,
        )

    def on_exit_filled(self, trade: Trade, explanation: str) -> None:
        msg = (
            f"EXIT FILLED [{trade.signal.exchange} {trade.signal.symbol} "
            f"{trade.exit_fill.timestamp.isoformat()}]: {trade.exit_reason.upper()}"
        )
        if explanation:
            msg += f"\n{explanation}"
        self.logger.log(self.level, msg)

    def on_daily_loss_breached(self, executor: PaperExecutor) -> None:
        self.logger.log(
            logging.WARNING,
            "DAILY LOSS LIMIT BREACHED: New entries blocked. Cash=%s, Open Positions=%d",
            executor.cash,
            len(executor.positions),
        )


class CompositeAlertHandler:
    """Dispatches alert callbacks to multiple AlertHandler instances."""

    def __init__(self, handlers: Iterable[AlertHandler] | None = None) -> None:
        self.handlers: list[AlertHandler] = list(handlers or [])

    def add_handler(self, handler: AlertHandler) -> None:
        self.handlers.append(handler)

    def _dispatch(self, method_name: str, *args: Any) -> Any:
        coroutines = []
        for handler in self.handlers:
            method = getattr(handler, method_name, None)
            if method is not None and callable(method):
                try:
                    res = method(*args)
                    if inspect.iscoroutine(res):
                        coroutines.append(res)
                except Exception as exc:
                    logger.warning(
                        "Error in composite alert handler %s %s: %s",
                        type(handler).__name__,
                        method_name,
                        exc,
                        exc_info=True,
                    )
        if coroutines:

            async def _await_all() -> None:
                for coro in coroutines:
                    try:
                        await coro
                    except Exception as exc:
                        logger.warning("Error awaiting async alert handler: %s", exc, exc_info=True)

            return _await_all()
        return None

    def on_candle_closed(self, candle: Candle) -> Any:
        return self._dispatch("on_candle_closed", candle)

    def on_signal_generated(self, signal: BreakoutSignal, explanation: str) -> Any:
        return self._dispatch("on_signal_generated", signal, explanation)

    def on_entry_filled(self, position: Position) -> Any:
        return self._dispatch("on_entry_filled", position)

    def on_exit_filled(self, trade: Trade, explanation: str) -> Any:
        return self._dispatch("on_exit_filled", trade, explanation)

    def on_daily_loss_breached(self, executor: PaperExecutor) -> Any:
        return self._dispatch("on_daily_loss_breached", executor)


__all__ = ["AlertHandler", "LoggingAlertHandler", "CompositeAlertHandler"]

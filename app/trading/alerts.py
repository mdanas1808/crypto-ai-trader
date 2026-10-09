"""Alerting and event notification interfaces for paper trading sessions.

Provides the AlertHandler protocol and a default LoggingAlertHandler that
formats and logs strategy events using Python standard logging.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from app.backtest.models import Position, Trade
from app.strategy.breakout import BreakoutSignal
from app.trading.paper import PaperExecutor

logger = logging.getLogger(__name__)


@runtime_checkable
class AlertHandler(Protocol):
    """Protocol for receiving real-time paper trading event callbacks."""

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

    def __init__(
        self, logger: logging.Logger | None = None, level: int = logging.INFO
    ) -> None:
        self.logger = logger or logging.getLogger("app.trading.alerts")
        self.level = level

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


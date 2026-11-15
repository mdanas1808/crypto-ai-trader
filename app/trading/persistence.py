"""Database persistence alert handler for real-time paper trading.

Persists completed trades and generated signals with their AI explanations
to the database asynchronously and non-disruptively.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from decimal import Decimal

from sqlalchemy.orm import Session

from app.backtest.models import Position, Trade
from app.database.paper_models import PaperSignalRecord, PaperTradeRecord
from app.market.models import Candle
from app.market.repository import persist_candles
from app.strategy.breakout import BreakoutSignal
from app.trading.paper import PaperExecutor

logger = logging.getLogger(__name__)


class DatabaseAlertHandler:
    """Alert handler that persists signals and trades into the database."""

    def __init__(self, session_factory: Callable[[], Session] | None = None) -> None:
        if session_factory is None:
            from app.database.database import SessionLocal

            self.session_factory = SessionLocal
        else:
            self.session_factory = session_factory

    def on_candle_closed(self, candle: Candle) -> None:
        """Persist a completed candle into the database."""
        try:
            session = self.session_factory()
            try:
                persist_candles(session, [candle])
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()
        except Exception as exc:
            logger.error(
                "Failed to persist closed candle for %s: %s",
                candle.symbol,
                exc,
                exc_info=True,
            )

    def on_signal_generated(self, signal: BreakoutSignal, explanation: str) -> None:
        """Persist a newly generated breakout signal and its AI explanation."""
        try:
            session = self.session_factory()
            try:
                direction = getattr(
                    signal,
                    "direction",
                    "LONG" if str(signal.side).lower() in ("long", "buy") else "SHORT",
                )
                leverage = int(getattr(signal, "leverage", 1) or 1)
                margin_used = Decimal(str(getattr(signal, "margin_used", Decimal("0")) or "0"))
                stop_loss = Decimal(str(getattr(signal, "stop_loss", Decimal("0")) or "0"))
                take_profit = Decimal(str(getattr(signal, "take_profit", Decimal("0")) or "0"))

                record = PaperSignalRecord(
                    exchange=signal.exchange,
                    symbol=signal.symbol,
                    side=signal.side,
                    direction=direction,
                    leverage=leverage,
                    margin_used=margin_used,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    decision_time=signal.decision_time,
                    signal_open_time=signal.signal_open_time,
                    atr_14=Decimal(str(signal.atr_14)),
                    explanation=explanation or "",
                )
                session.add(record)
                session.commit()
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()
        except Exception as exc:
            logger.error(
                "Failed to persist paper signal for %s: %s",
                signal.symbol,
                exc,
                exc_info=True,
            )

    def on_entry_filled(self, position: Position) -> None:
        """Called when an entry is filled; trade is persisted upon exit."""
        pass

    def on_exit_filled(self, trade: Trade, explanation: str) -> None:
        """Persist a completed trade and its exit metrics."""
        try:
            session = self.session_factory()
            try:
                direction = getattr(
                    trade,
                    "direction",
                    "LONG" if str(trade.signal.side).lower() in ("long", "buy") else "SHORT",
                )
                leverage = int(
                    getattr(trade, "leverage", getattr(trade.signal, "leverage", 1)) or 1
                )
                margin_used = Decimal(
                    str(
                        getattr(
                            trade, "margin_used", getattr(trade.signal, "margin_used", Decimal("0"))
                        )
                        or "0"
                    )
                )
                stop_loss = Decimal(
                    str(
                        getattr(trade, "stop_loss", getattr(trade, "stop_price", Decimal("0")))
                        or "0"
                    )
                )
                take_profit = Decimal(
                    str(
                        getattr(trade, "take_profit", getattr(trade, "target_price", Decimal("0")))
                        or "0"
                    )
                )
                realized_pnl = getattr(trade, "realized_pnl", trade.net_pnl)

                record = PaperTradeRecord(
                    exchange=trade.signal.exchange,
                    symbol=trade.signal.symbol,
                    side=trade.signal.side,
                    direction=direction,
                    leverage=leverage,
                    margin_used=margin_used,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    entry_time=trade.entry_fill.timestamp,
                    exit_time=trade.exit_fill.timestamp,
                    entry_price=trade.entry_fill.price,
                    exit_price=trade.exit_fill.price,
                    quantity=trade.quantity,
                    pnl=trade.net_pnl,
                    realized_pnl=realized_pnl,
                    exit_reason=trade.exit_reason,
                )
                session.add(record)
                session.commit()
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()
        except Exception as exc:
            logger.error(
                "Failed to persist paper trade for %s: %s",
                trade.signal.symbol,
                exc,
                exc_info=True,
            )

    def on_daily_loss_breached(self, executor: PaperExecutor) -> None:
        """Called when portfolio daily loss limit is breached."""
        pass


__all__ = ["DatabaseAlertHandler"]

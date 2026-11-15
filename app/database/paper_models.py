"""SQLAlchemy models for paper trading persistence."""

from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, Integer, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database.database import Base


class PaperTradeRecord(Base):
    """Persisted record of an executed and closed paper trade."""

    __tablename__ = "paper_trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    exchange: Mapped[str] = mapped_column(String(32), nullable=False, default="binance_spot")
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    side: Mapped[str] = mapped_column(String(16), nullable=False, default="long")
    direction: Mapped[str] = mapped_column(String(16), nullable=False, default="LONG")
    leverage: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    margin_used: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False, default=Decimal("0")
    )
    stop_loss: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False, default=Decimal("0")
    )
    take_profit: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False, default=Decimal("0")
    )
    entry_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    exit_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    entry_price: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    exit_price: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    pnl: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    realized_pnl: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False, default=Decimal("0")
    )
    exit_reason: Mapped[str] = mapped_column(String(64), nullable=False)


class PaperSignalRecord(Base):
    """Persisted record of an evaluated strategy signal and its AI explanation."""

    __tablename__ = "paper_signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    exchange: Mapped[str] = mapped_column(String(32), nullable=False, default="binance_spot")
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    side: Mapped[str] = mapped_column(String(16), nullable=False, default="long")
    direction: Mapped[str] = mapped_column(String(16), nullable=False, default="LONG")
    leverage: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    margin_used: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False, default=Decimal("0")
    )
    stop_loss: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False, default=Decimal("0")
    )
    take_profit: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False, default=Decimal("0")
    )
    decision_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    signal_open_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    atr_14: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    explanation: Mapped[str] = mapped_column(Text, nullable=False)


__all__ = ["PaperTradeRecord", "PaperSignalRecord"]

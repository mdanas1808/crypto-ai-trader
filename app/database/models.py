"""SQLAlchemy table definitions."""

from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, Numeric, PrimaryKeyConstraint, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database.database import Base


class CandleRecord(Base):
    __tablename__ = "market_candles"
    __table_args__ = (
        PrimaryKeyConstraint(
            "exchange",
            "symbol",
            "interval",
            "open_time",
            name="pk_market_candles",
        ),
    )

    exchange: Mapped[str] = mapped_column(String(32), nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    interval: Mapped[str] = mapped_column(String(8), nullable=False)
    open_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    open: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    high: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    low: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    close: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    volume: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)


__all__ = ["Base", "CandleRecord"]

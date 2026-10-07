"""PostgreSQL persistence and retrieval for normalized candles."""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.database.models import CandleRecord
from app.market.constants import EXCHANGE
from app.market.models import Candle


@dataclass(frozen=True)
class InsertResult:
    attempted: int
    inserted: int
    duplicates_skipped: int


def persist_candles(session: Session, candles: Iterable[Candle]) -> InsertResult:
    rows = [candle.model_dump() for candle in candles]
    if not rows:
        return InsertResult(attempted=0, inserted=0, duplicates_skipped=0)

    statement = (
        insert(CandleRecord)
        .values(rows)
        .on_conflict_do_nothing(
            index_elements=[
                CandleRecord.exchange,
                CandleRecord.symbol,
                CandleRecord.interval,
                CandleRecord.open_time,
            ]
        )
        .returning(CandleRecord.open_time)
    )
    try:
        inserted = len(session.execute(statement).all())
        session.commit()
    except Exception:
        session.rollback()
        raise

    return InsertResult(
        attempted=len(rows),
        inserted=inserted,
        duplicates_skipped=len(rows) - inserted,
    )


def get_candles(
    session: Session,
    symbol: str,
    interval: str,
    limit: int,
    exchange: str = EXCHANGE,
) -> list[Candle]:
    statement = (
        select(CandleRecord)
        .where(
            CandleRecord.exchange == exchange,
            CandleRecord.symbol == symbol,
            CandleRecord.interval == interval,
        )
        .order_by(CandleRecord.open_time.desc())
        .limit(limit)
    )
    records = session.scalars(statement).all()
    return [Candle.model_validate(record) for record in reversed(records)]


def get_candles_in_range(
    session: Session,
    symbol: str,
    interval: str,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    exchange: str = EXCHANGE,
) -> list[Candle]:
    """Retrieve one candle series within optional inclusive UTC time bounds."""
    if start_time is not None:
        start_time = _as_utc_bound(start_time, "start_time")
    if end_time is not None:
        end_time = _as_utc_bound(end_time, "end_time")
    if start_time is not None and end_time is not None and start_time > end_time:
        raise ValueError("start_time must be less than or equal to end_time")

    statement = select(CandleRecord).where(
        CandleRecord.exchange == exchange,
        CandleRecord.symbol == symbol,
        CandleRecord.interval == interval,
    )
    if start_time is not None:
        statement = statement.where(CandleRecord.open_time >= start_time)
    if end_time is not None:
        statement = statement.where(CandleRecord.open_time <= end_time)
    statement = statement.order_by(CandleRecord.open_time.asc())
    return [Candle.model_validate(record) for record in session.scalars(statement).all()]


def _as_utc_bound(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)

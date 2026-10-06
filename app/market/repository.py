"""PostgreSQL persistence and retrieval for normalized candles."""

from dataclasses import dataclass
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

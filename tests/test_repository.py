import asyncio
from datetime import datetime, timezone
from decimal import Decimal

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.models import CandleRecord
from app.market.ingestion import ingest_candles
from app.market.models import Candle
from app.market.repository import get_candles, get_candles_in_range, persist_candles


def make_candle(open_time: datetime, close: str) -> Candle:
    return Candle(
        exchange="binance_spot",
        symbol="BTCUSDT",
        interval="1h",
        open_time=open_time,
        open="100",
        high="110",
        low="90",
        close=close,
        volume="2.5",
    )


def test_database_insert_and_duplicate_handling(db_session: Session) -> None:
    first = make_candle(datetime(2024, 1, 1, tzinfo=timezone.utc), "105")
    inserted = persist_candles(db_session, [first])
    duplicate = persist_candles(db_session, [first])

    assert inserted.inserted == 1
    assert inserted.duplicates_skipped == 0
    assert duplicate.inserted == 0
    assert duplicate.duplicates_skipped == 1
    stored = db_session.scalars(select(CandleRecord)).all()
    assert len(stored) == 1
    assert stored[0].open == Decimal("100.000000000000000000")


def test_repository_returns_latest_rows_in_chronological_order(db_session: Session) -> None:
    times = [datetime(2024, 1, day, tzinfo=timezone.utc) for day in (1, 2, 3)]
    persist_candles(db_session, [make_candle(time, "105") for time in times])

    candles = get_candles(db_session, "BTCUSDT", "1h", limit=2)
    assert [candle.open_time for candle in candles] == times[1:]


def test_range_retrieval_includes_start_time(db_session: Session) -> None:
    times = [datetime(2024, 1, day, tzinfo=timezone.utc) for day in (1, 2, 3)]
    persist_candles(db_session, [make_candle(time, "105") for time in times])

    candles = get_candles_in_range(db_session, "BTCUSDT", "1h", start_time=times[1])

    assert [candle.open_time for candle in candles] == times[1:]


def test_range_retrieval_includes_end_time(db_session: Session) -> None:
    times = [datetime(2024, 1, day, tzinfo=timezone.utc) for day in (1, 2, 3)]
    persist_candles(db_session, [make_candle(time, "105") for time in times])

    candles = get_candles_in_range(db_session, "BTCUSDT", "1h", end_time=times[1])

    assert [candle.open_time for candle in candles] == times[:2]


def test_range_retrieval_includes_both_boundaries_and_orders_chronologically(
    db_session: Session,
) -> None:
    times = [datetime(2024, 1, day, tzinfo=timezone.utc) for day in (1, 2, 3)]
    persist_candles(db_session, [make_candle(time, "105") for time in reversed(times)])

    candles = get_candles_in_range(
        db_session, "BTCUSDT", "1h", start_time=times[0], end_time=times[2]
    )

    assert [candle.open_time for candle in candles] == times


def test_range_retrieval_returns_empty_when_no_rows_match(db_session: Session) -> None:
    persist_candles(
        db_session,
        [make_candle(datetime(2024, 1, 1, tzinfo=timezone.utc), "105")],
    )

    candles = get_candles_in_range(
        db_session,
        "BTCUSDT",
        "1h",
        start_time=datetime(2024, 1, 2, tzinfo=timezone.utc),
        end_time=datetime(2024, 1, 3, tzinfo=timezone.utc),
    )

    assert candles == []


def test_range_retrieval_rejects_reversed_bounds(db_session: Session) -> None:
    start = datetime(2024, 1, 2, tzinfo=timezone.utc)
    end = datetime(2024, 1, 1, tzinfo=timezone.utc)

    with pytest.raises(ValueError, match="start_time must be less than or equal to end_time"):
        get_candles_in_range(db_session, "BTCUSDT", "1h", start_time=start, end_time=end)


def test_range_retrieval_requires_timezone_aware_bounds(db_session: Session) -> None:
    with pytest.raises(ValueError, match="start_time must be timezone-aware"):
        get_candles_in_range(db_session, "BTCUSDT", "1h", start_time=datetime(2024, 1, 1))


def test_full_ingestion_is_idempotent(db_session: Session) -> None:
    candles = [
        make_candle(datetime(2024, 1, day, tzinfo=timezone.utc), str(105 + day))
        for day in (1, 2, 3)
    ]

    class StubExchangeClient:
        async def fetch_candles(self, symbol: str, interval: str, limit: int) -> list[Candle]:
            assert (symbol, interval, limit) == ("BTCUSDT", "1h", 3)
            return candles

    async def exercise() -> tuple[object, object]:
        client = StubExchangeClient()
        first = await ingest_candles(db_session, "BTCUSDT", "1h", 3, client=client)
        second = await ingest_candles(db_session, "BTCUSDT", "1h", 3, client=client)
        return first, second

    first, second = asyncio.run(exercise())

    assert first.received == 3
    assert first.persistence.inserted == 3
    assert first.persistence.duplicates_skipped == 0
    assert second.received == 3
    assert second.persistence.inserted == 0
    assert second.persistence.duplicates_skipped == 3
    stored = db_session.scalars(select(CandleRecord)).all()
    assert len(stored) == 3

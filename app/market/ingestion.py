"""Explicitly invoked historical candle ingestion service."""

import logging
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.market.constants import EXCHANGE
from app.market.exchange import BinanceSpotClient
from app.market.repository import InsertResult, persist_candles

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionResult:
    received: int
    persistence: InsertResult


async def ingest_candles(
    session: Session,
    symbol: str,
    interval: str,
    limit: int = 100,
    *,
    client: BinanceSpotClient | None = None,
) -> IngestionResult:
    logger.info(
        "historical ingestion started exchange=%s symbol=%s interval=%s requested=%d",
        EXCHANGE,
        symbol,
        interval,
        limit,
    )
    try:
        if client is None:
            async with BinanceSpotClient() as exchange_client:
                candles = await exchange_client.fetch_candles(symbol, interval, limit)
        else:
            candles = await client.fetch_candles(symbol, interval, limit)
        logger.info(
            "historical ingestion received exchange=%s symbol=%s interval=%s received=%d",
            EXCHANGE,
            symbol,
            interval,
            len(candles),
        )
        persistence = persist_candles(session, candles)
    except Exception:
        logger.exception(
            "historical ingestion failed exchange=%s symbol=%s interval=%s",
            EXCHANGE,
            symbol,
            interval,
        )
        raise

    logger.info(
        "historical ingestion completed exchange=%s symbol=%s interval=%s inserted=%d duplicates_skipped=%d",
        EXCHANGE,
        symbol,
        interval,
        persistence.inserted,
        persistence.duplicates_skipped,
    )
    return IngestionResult(received=len(candles), persistence=persistence)

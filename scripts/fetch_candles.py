#!/usr/bin/env python3
"""Explicitly fetch a bounded batch of closed Binance Spot candles."""

import argparse
import asyncio
import logging

from app.database.database import SessionLocal
from app.market.constants import (
    DEFAULT_CANDLE_LIMIT,
    MAX_CANDLE_LIMIT,
    SUPPORTED_INTERVALS,
    SUPPORTED_SYMBOLS,
)
from app.market.ingestion import ingest_candles


def positive_limit(value: str) -> int:
    try:
        limit = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("limit must be an integer") from exc
    if not 1 <= limit <= MAX_CANDLE_LIMIT:
        raise argparse.ArgumentTypeError(f"limit must be between 1 and {MAX_CANDLE_LIMIT}")
    return limit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch recent closed Binance Spot candles")
    parser.add_argument("--symbol", required=True, choices=sorted(SUPPORTED_SYMBOLS))
    parser.add_argument("--interval", required=True, choices=sorted(SUPPORTED_INTERVALS))
    parser.add_argument("--limit", type=positive_limit, default=DEFAULT_CANDLE_LIMIT)
    return parser.parse_args()


async def run(symbol: str, interval: str, limit: int) -> int:
    with SessionLocal() as session:
        result = await ingest_candles(session, symbol, interval, limit)
    print(
        f"exchange=binance_spot symbol={symbol} interval={interval} "
        f"received={result.received} inserted={result.persistence.inserted} "
        f"duplicates_skipped={result.persistence.duplicates_skipped}"
    )
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = parse_args()
    try:
        return asyncio.run(run(args.symbol, args.interval, args.limit))
    except Exception:
        logging.getLogger(__name__).exception("candle fetch command failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

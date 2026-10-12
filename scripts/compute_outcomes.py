"""Standalone script to compute hypothetical forward-looking outcomes for evaluated trades.

Queries SQLite evaluation records missing forward returns (T+1h, T+4h, T+24h),
fetches historical hourly candles, and calculates returns using the NEXT candle's
open price as the hypothetical entry/exit fill price (not the decision-time price).

Usage:
    .venv/bin/python scripts/compute_outcomes.py [--db-path fundamental_eval.db]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Sequence

import httpx

from app.analysis.eval_logger import DEFAULT_DB_PATH
from app.market.constants import FIXED_SYMBOLS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("compute_outcomes")

# Round-trip execution cost hurdles (2 x fee + 2 x slippage)
CORE_COST_HURDLE = 0.003  # 2 x 0.10% fee + 2 x 0.05% slippage = 0.30%
SATELLITE_COST_HURDLE = 0.006  # 2 x 0.10% fee + 2 x 0.20% slippage = 0.60%

BINANCE_KLINES_URL = "https://data-api.binance.vision/api/v3/klines"


def parse_timestamp(ts_val: Any) -> datetime:
    """Parse ISO timestamp or int timestamp into timezone-aware UTC datetime."""
    if isinstance(ts_val, datetime):
        return ts_val if ts_val.tzinfo is not None else ts_val.replace(tzinfo=timezone.utc)
    if isinstance(ts_val, (int, float)):
        return datetime.fromtimestamp(float(ts_val), tz=timezone.utc)
    ts_str = str(ts_val).strip()
    return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))


async def fetch_hourly_candles(
    symbol: str,
    start_time: datetime,
    limit: int = 50,
    *,
    base_url: str = BINANCE_KLINES_URL,
    http_client: httpx.AsyncClient | None = None,
) -> list[list[Any]]:
    """Fetch 1h klines starting from start_time."""
    start_ms = int(start_time.timestamp() * 1000)
    params = {
        "symbol": symbol,
        "interval": "1h",
        "startTime": start_ms,
        "limit": min(limit, 500),
    }

    if http_client is not None:
        resp = await http_client.get(base_url, params=params)
        resp.raise_for_status()
        return resp.json()

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(base_url, params=params)
        resp.raise_for_status()
        return resp.json()


def extract_candle_info(row: Any) -> tuple[datetime, float, float]:
    """Extract (open_time, open_price, close_price) from a kline row or object."""
    if hasattr(row, "open_time") and hasattr(row, "open") and hasattr(row, "close"):
        ot = row.open_time
        if ot.tzinfo is None:
            ot = ot.replace(tzinfo=timezone.utc)
        return ot, float(row.open), float(row.close)
    if isinstance(row, dict):
        ot = parse_timestamp(row.get("open_time") or row.get("time") or row.get("timestamp"))
        return ot, float(row["open"]), float(row["close"])
    if isinstance(row, (list, tuple)) and len(row) >= 5:
        ot = datetime.fromtimestamp(int(row[0]) / 1000.0, tz=timezone.utc)
        return ot, float(row[1]), float(row[4])
    raise ValueError(f"Unrecognized kline format: {row}")


async def compute_outcomes_for_database(
    db_path: str = DEFAULT_DB_PATH,
    *,
    candle_fetcher: Callable[[str, datetime, int], Awaitable[Sequence[Any]] | Sequence[Any]]
    | None = None,
    http_client: httpx.AsyncClient | None = None,
    symbol_filter: str | None = None,
    verdict_filter: str | None = None,
    limit: int | None = None,
    base_url: str = BINANCE_KLINES_URL,
    horizons: tuple[int, ...] = (1, 4, 24),
) -> int:
    """Compute and update forward returns for pending evaluation records.

    Uses the NEXT candle's open price as the hypothetical entry/exit fill price.
    Computes T+1h, T+4h, and T+24h returns for all pending signals including regime-blocked signals.
    Returns the count of successfully processed evaluations.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    query = (
        "SELECT id, timestamp, symbol, current_price, verdict, mode "
        "FROM fundamental_evaluations "
        "WHERE outcome_status = 'PENDING' OR return_24h IS NULL "
    )
    params: list[Any] = []
    if symbol_filter:
        query += "AND symbol = ? "
        params.append(symbol_filter)
    if verdict_filter:
        query += "AND verdict = ? "
        params.append(verdict_filter.upper())
    query += "ORDER BY id ASC"
    if limit is not None and limit > 0:
        query += f" LIMIT {int(limit)}"

    with conn:
        rows = [dict(r) for r in conn.execute(query, params).fetchall()]

    if not rows:
        logger.info("No pending evaluations found requiring outcome calculation.")
        conn.close()
        return 0

    logger.info("Found %d pending evaluations to process.", len(rows))
    processed_count = 0

    for row in rows:
        eval_id = row["id"]
        sym = row["symbol"]
        decision_time = parse_timestamp(row["timestamp"])

        try:
            # 1. Fetch historical candles around decision time
            if candle_fetcher is not None:
                res = candle_fetcher(sym, decision_time, max(horizons) + 5)
                raw_candles = await res if hasattr(res, "__await__") else res
            else:
                raw_candles = await fetch_hourly_candles(
                    sym,
                    decision_time,
                    limit=max(horizons) + 5,
                    base_url=base_url,
                    http_client=http_client,
                )

            if not raw_candles:
                logger.warning("No candles returned for %s at %s", sym, decision_time)
                continue

            parsed_candles = [extract_candle_info(c) for c in raw_candles]
            parsed_candles.sort(key=lambda c: c[0])

            # 2. Find NEXT candle after decision time (hypothetical fill)
            # A decision made at or before the hour mark executes at the next open
            next_candle = None
            for ot, op, cl in parsed_candles:
                if ot >= decision_time:
                    next_candle = (ot, op, cl)
                    break

            if next_candle is None:
                # If no candle with ot >= decision_time, use the first available candle if close enough
                if (
                    parsed_candles
                    and abs((parsed_candles[0][0] - decision_time).total_seconds()) <= 3600
                ):
                    next_candle = parsed_candles[0]
                else:
                    logger.warning(
                        "Could not identify NEXT candle for eval_id %d (%s at %s)",
                        eval_id,
                        sym,
                        decision_time,
                    )
                    continue

            fill_time, fill_price, _ = next_candle
            if fill_price <= 0:
                continue

            prices: dict[int, float | None] = {}
            returns: dict[int, float | None] = {}

            # 3. Calculate forward outcomes at T+1h, T+4h, T+24h from fill_time
            for h in horizons:
                target_time = fill_time + timedelta(hours=h)
                closest = None
                min_diff = float("inf")
                for ot, op, cl in parsed_candles:
                    diff = abs((ot - target_time).total_seconds())
                    if diff < min_diff:
                        min_diff = diff
                        closest = (ot, op, cl)

                if closest is not None and min_diff <= 3600:
                    price_h = closest[2]  # close price of the horizon candle
                    ret_h = (price_h - fill_price) / fill_price
                    prices[h] = price_h
                    returns[h] = ret_h
                else:
                    prices[h] = None
                    returns[h] = None

            cost_hurdle = CORE_COST_HURDLE if sym in FIXED_SYMBOLS else SATELLITE_COST_HURDLE

            p_1h = prices.get(1)
            p_4h = prices.get(4)
            p_24h = prices.get(24)
            r_1h = returns.get(1)
            r_4h = returns.get(4)
            r_24h = returns.get(24)

            # Net-of-cost counterfactual returns
            nr_1h = (r_1h - cost_hurdle) if r_1h is not None else None
            nr_4h = (r_4h - cost_hurdle) if r_4h is not None else None
            nr_24h = (r_24h - cost_hurdle) if r_24h is not None else None

            status = (
                "COMPLETED" if r_24h is not None else ("PARTIAL" if r_1h is not None else "PENDING")
            )

            with conn:
                pragma_cols = {
                    r["name"]
                    for r in conn.execute("PRAGMA table_info(fundamental_evaluations)").fetchall()
                }
                for col in ("net_return_1h", "net_return_4h", "net_return_24h"):
                    if col not in pragma_cols:
                        try:
                            conn.execute(
                                f"ALTER TABLE fundamental_evaluations ADD COLUMN {col} REAL"
                            )
                        except sqlite3.OperationalError:
                            pass

                conn.execute(
                    """
                    UPDATE fundamental_evaluations
                    SET hypothetical_fill_price = ?,
                        price_t_1h = COALESCE(?, price_t_1h),
                        price_t_4h = COALESCE(?, price_t_4h),
                        price_t_24h = COALESCE(?, price_t_24h),
                        return_1h = COALESCE(?, return_1h),
                        return_4h = COALESCE(?, return_4h),
                        return_24h = COALESCE(?, return_24h),
                        net_return_1h = COALESCE(?, net_return_1h),
                        net_return_4h = COALESCE(?, net_return_4h),
                        net_return_24h = COALESCE(?, net_return_24h),
                        outcome_status = ?
                    WHERE id = ?
                    """,
                    (
                        fill_price,
                        p_1h,
                        p_4h,
                        p_24h,
                        r_1h,
                        r_4h,
                        r_24h,
                        nr_1h,
                        nr_4h,
                        nr_24h,
                        status,
                        eval_id,
                    ),
                )

            logger.info(
                "Updated eval_id=%d (%s): fill=%.4f, T+24h Gross=%s, T+24h Net=%s, status=%s",
                eval_id,
                sym,
                fill_price,
                f"{r_24h:+.2%}" if r_24h is not None else "N/A",
                f"{nr_24h:+.2%}" if nr_24h is not None else "N/A",
                status,
            )
            processed_count += 1

        except Exception as exc:
            logger.exception("Error computing outcomes for eval_id %d (%s): %s", eval_id, sym, exc)

    conn.close()
    return processed_count


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute hypothetical outcomes for fundamental evaluations."
    )
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help="Path to SQLite database.")
    parser.add_argument("--symbol", default=None, help="Filter by symbol.")
    parser.add_argument(
        "--verdict",
        default=None,
        help="Filter by verdict (e.g. BLOCKED_BY_REGIME, REJECT, APPROVE).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum records to process.")
    parser.add_argument("--base-url", default=BINANCE_KLINES_URL, help="Binance klines API URL.")

    args = parser.parse_args()
    count = await compute_outcomes_for_database(
        db_path=args.db_path,
        symbol_filter=args.symbol,
        verdict_filter=args.verdict,
        limit=args.limit,
        base_url=args.base_url,
    )
    logger.info("Done. Processed %d evaluations.", count)


if __name__ == "__main__":
    asyncio.run(main())

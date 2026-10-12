"""Evaluation logger for the AI Fundamental Risk Gatekeeper.

Records ENTRY evaluations (symbol, timestamp, price, raw XML news, verdict,
risk level, confidence, and reason) into a SQLite database and dedicated log file.
For rejected trades, triggers background tasks to track forward-looking hypothetical
outcomes at T+1h, T+4h, and T+24h to measure whether the AI successfully dodged a bullet
or missed a profitable breakout.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
import logging
import os
import sqlite3
from typing import Any, Awaitable, Callable

import httpx

from app.market.exchange import BinanceSpotClient

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "fundamental_eval.db"
DEFAULT_LOG_PATH = "fundamental_eval.log"


class FundamentalEvalLogger:
    """Dedicated evaluator logger recording gate evaluations and hypothetical outcomes."""

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        log_file: str = DEFAULT_LOG_PATH,
        binance_client: BinanceSpotClient | None = None,
    ) -> None:
        self.db_path = str(db_path)
        self.log_file = str(log_file)
        self.binance_client = binance_client
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create sqlite tables if they do not exist."""
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fundamental_evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    current_price REAL NOT NULL,
                    raw_inputs TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    reason TEXT NOT NULL,
                    driving_news_id TEXT DEFAULT '',
                    metadata_json TEXT DEFAULT '{}',
                    mode TEXT DEFAULT 'ENTRY',
                    hypothetical_fill_price REAL,
                    price_t_1h REAL,
                    price_t_4h REAL,
                    price_t_24h REAL,
                    return_1h REAL,
                    return_4h REAL,
                    return_24h REAL,
                    outcome_status TEXT DEFAULT 'PENDING'
                );
                """
            )
            for col, col_def in (
                ("mode", "TEXT DEFAULT 'ENTRY'"),
                ("hypothetical_fill_price", "REAL"),
            ):
                try:
                    conn.execute(f"ALTER TABLE fundamental_evaluations ADD COLUMN {col} {col_def}")
                except sqlite3.OperationalError:
                    pass
            conn.commit()

    def _append_to_log_file(self, line: str) -> None:
        """Append line to evaluation log file."""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.log_file)), exist_ok=True)
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as exc:
            logger.warning("Failed to write to %s: %s", self.log_file, exc)

    def log_decision(
        self,
        symbol: str,
        current_price: Decimal | float,
        raw_inputs: str,
        verdict: str,
        risk_level: str,
        confidence: float,
        reason: str,
        mode: str = "ENTRY",
        metadata: dict[str, Any] | None = None,
        timestamp: datetime | None = None,
    ) -> int:
        """Log any fundamental evaluation decision (APPROVE, REJECT, HOLD, REDUCE, EXIT, no-news) to SQLite and log file."""
        ts = timestamp if timestamp is not None else datetime.now(timezone.utc)
        ts_iso = ts.isoformat()
        price_val = float(current_price)
        meta = metadata or {}
        driving_id = str(meta.get("driving_news_id") or "")
        meta_json = json.dumps(meta)
        verdict_upper = str(verdict).strip().upper()
        mode_upper = str(mode).strip().upper()

        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO fundamental_evaluations (
                    timestamp, symbol, current_price, raw_inputs,
                    verdict, risk_level, confidence, reason,
                    driving_news_id, metadata_json, mode, outcome_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts_iso,
                    symbol,
                    price_val,
                    raw_inputs,
                    verdict_upper,
                    risk_level,
                    float(confidence),
                    reason,
                    driving_id,
                    meta_json,
                    mode_upper,
                    "PENDING",
                ),
            )
            eval_id = cur.lastrowid
            conn.commit()

        log_line = (
            f"[{ts_iso}] {mode_upper}_EVAL | id={eval_id} | symbol={symbol} | price={price_val:.4f} | "
            f"verdict={verdict_upper} | risk={risk_level} (conf={confidence:.2f}) | "
            f"reason={reason!r} | driving_news_id={driving_id!r}"
        )
        self._append_to_log_file(log_line)
        logger.info(log_line)

        return eval_id

    def log_entry_evaluation(
        self,
        symbol: str,
        current_price: Decimal | float,
        raw_inputs: str,
        verdict: str,
        risk_level: str,
        confidence: float,
        reason: str,
        metadata: dict[str, Any] | None = None,
        timestamp: datetime | None = None,
    ) -> int:
        """Log an ENTRY signal evaluation (backward-compatible wrapper around log_decision)."""
        return self.log_decision(
            symbol=symbol,
            current_price=current_price,
            raw_inputs=raw_inputs,
            verdict=verdict,
            risk_level=risk_level,
            confidence=confidence,
            reason=reason,
            mode="ENTRY",
            metadata=metadata,
            timestamp=timestamp,
        )

    def update_outcome(
        self,
        eval_id: int,
        hypothetical_fill_price: float | None = None,
        price_t_1h: float | None = None,
        price_t_4h: float | None = None,
        price_t_24h: float | None = None,
        return_1h: float | None = None,
        return_4h: float | None = None,
        return_24h: float | None = None,
        outcome_status: str = "COMPLETED",
    ) -> None:
        """Update evaluation record with calculated hypothetical outcomes."""
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE fundamental_evaluations
                SET hypothetical_fill_price = COALESCE(?, hypothetical_fill_price),
                    price_t_1h = COALESCE(?, price_t_1h),
                    price_t_4h = COALESCE(?, price_t_4h),
                    price_t_24h = COALESCE(?, price_t_24h),
                    return_1h = COALESCE(?, return_1h),
                    return_4h = COALESCE(?, return_4h),
                    return_24h = COALESCE(?, return_24h),
                    outcome_status = ?
                WHERE id = ?
                """,
                (
                    hypothetical_fill_price,
                    price_t_1h,
                    price_t_4h,
                    price_t_24h,
                    return_1h,
                    return_4h,
                    return_24h,
                    outcome_status,
                    eval_id,
                ),
            )
            conn.commit()

    def get_pending_evaluations(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Retrieve evaluation rows that need forward outcome calculations."""
        query = (
            "SELECT * FROM fundamental_evaluations "
            "WHERE outcome_status = 'PENDING' OR return_24h IS NULL "
            "ORDER BY id ASC"
        )
        if limit is not None and limit > 0:
            query += f" LIMIT {int(limit)}"
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(query)
            return [dict(row) for row in cur.fetchall()]

    async def _fetch_price_at_time(
        self,
        symbol: str,
        target_time: datetime,
        price_fetcher: Callable[[str, datetime], Awaitable[float | Decimal | None]] | None = None,
    ) -> float | None:
        """Fetch asset price at a target time via custom fetcher or Binance client."""
        if price_fetcher is not None:
            try:
                res = await price_fetcher(symbol, target_time)
                if res is not None:
                    return float(res)
            except Exception as exc:
                logger.warning("Custom price fetcher failed for %s at %s: %s", symbol, target_time, exc)

        # Fallback to Binance Spot candles
        if self.binance_client is not None:
            try:
                candles = await self.binance_client.fetch_candles(symbol, interval="1h", limit=50)
                closest_candle = None
                min_diff = float("inf")
                for c in candles:
                    diff = abs((c.open_time - target_time).total_seconds())
                    if diff < min_diff:
                        min_diff = diff
                        closest_candle = c
                if closest_candle is not None and min_diff <= 7200:  # within 2 hours
                    return float(closest_candle.close)
            except Exception as exc:
                logger.warning("Binance client failed to fetch candle for %s at %s: %s", symbol, target_time, exc)

        return None

    async def record_hypothetical_outcome(
        self,
        eval_id: int,
        symbol: str,
        entry_price: Decimal | float,
        eval_time: datetime,
        price_fetcher: Callable[[str, datetime], Awaitable[float | Decimal | None]] | None = None,
        wait_if_live: bool = False,
        horizons: tuple[int, ...] = (1, 4, 24),
    ) -> dict[str, Any]:
        """Record hypothetical forward-looking returns for an evaluated trade without sleeping."""
        base_price = float(entry_price)
        if base_price <= 0:
            return {}

        prices: dict[int, float | None] = {}
        returns: dict[int, float | None] = {}

        now = datetime.now(timezone.utc)
        if eval_time.tzinfo is None:
            eval_time = eval_time.replace(tzinfo=timezone.utc)

        for h in horizons:
            target_time = eval_time + timedelta(hours=h)
            price_h = await self._fetch_price_at_time(symbol, target_time, price_fetcher)
            prices[h] = price_h
            if price_h is not None and base_price > 0:
                returns[h] = (price_h - base_price) / base_price
            else:
                returns[h] = None

        p_1h = prices.get(1)
        p_4h = prices.get(4)
        p_24h = prices.get(24)
        r_1h = returns.get(1)
        r_4h = returns.get(4)
        r_24h = returns.get(24)

        # Determine outcome assessment
        primary_return = r_24h if r_24h is not None else (r_4h if r_4h is not None else r_1h)
        if primary_return is not None:
            if primary_return < 0:
                ai_verdict_assessment = f"AI successfully dodged a bullet (loss avoided: {primary_return:+.2%})"
            else:
                ai_verdict_assessment = f"AI missed a profitable trade (missed return: {primary_return:+.2%})"
        else:
            ai_verdict_assessment = "Price data incomplete"

        # Update SQLite record
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE fundamental_evaluations
                SET price_t_1h = ?, price_t_4h = ?, price_t_24h = ?,
                    return_1h = ?, return_4h = ?, return_24h = ?,
                    outcome_status = 'COMPLETED'
                WHERE id = ?
                """,
                (p_1h, p_4h, p_24h, r_1h, r_4h, r_24h, eval_id),
            )
            conn.commit()

        p_1h_str = f"{p_1h:.4f} ({r_1h:+.2%})" if p_1h and r_1h is not None else "N/A"
        p_4h_str = f"{p_4h:.4f} ({r_4h:+.2%})" if p_4h and r_4h is not None else "N/A"
        p_24h_str = f"{p_24h:.4f} ({r_24h:+.2%})" if p_24h and r_24h is not None else "N/A"

        outcome_line = (
            f"[{datetime.now(timezone.utc).isoformat()}] HYPOTHETICAL_OUTCOME | id={eval_id} | symbol={symbol} | "
            f"base_price={base_price:.4f} | T+1h={p_1h_str} | T+4h={p_4h_str} | T+24h={p_24h_str} | "
            f"Assessment: {ai_verdict_assessment}"
        )
        self._append_to_log_file(outcome_line)
        logger.info(outcome_line)

        return {
            "eval_id": eval_id,
            "symbol": symbol,
            "base_price": base_price,
            "price_t_1h": p_1h,
            "price_t_4h": p_4h,
            "price_t_24h": p_24h,
            "return_1h": r_1h,
            "return_4h": r_4h,
            "return_24h": r_24h,
            "assessment": ai_verdict_assessment,
        }

    def get_evaluation(self, eval_id: int) -> dict[str, Any] | None:
        """Retrieve evaluation row by ID."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM fundamental_evaluations WHERE id = ?", (eval_id,))
            row = cur.fetchone()
            if row is not None:
                return dict(row)
        return None

    def get_evaluations(
        self, symbol: str | None = None, verdict: str | None = None
    ) -> list[dict[str, Any]]:
        """Query evaluation records."""
        query = "SELECT * FROM fundamental_evaluations WHERE 1=1"
        params: list[Any] = []
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if verdict:
            query += " AND verdict = ?"
            params.append(verdict)
        query += " ORDER BY id DESC"

        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(query, params)
            return [dict(row) for row in cur.fetchall()]


__all__ = ["DEFAULT_DB_PATH", "DEFAULT_LOG_PATH", "FundamentalEvalLogger"]

"""Read-only dashboard REST API for real-time paper trading sessions.

Exposes portfolio equity, active positions, closed trades, pending signals,
multi-variant experiment equity curves, live BTC regime status, and fundamental
gate evaluation decision logs via standard FastAPI JSON endpoints without mutating state.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from app.backtest.models import Fill, Position, Trade
from app.strategy.breakout import BreakoutSignal

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# -----------------------------------------------------------------------------
# Pydantic Schemas
# -----------------------------------------------------------------------------


class FillResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    symbol: str
    exchange: str
    side: str
    timestamp: datetime
    reference_price: Decimal
    price: Decimal
    quantity: Decimal
    fee: Decimal
    reason: str
    notional: Decimal
    observed_at: datetime | None = None

    @classmethod
    def from_fill(cls, fill: Fill) -> FillResponse:
        return cls(
            symbol=fill.symbol,
            exchange=fill.exchange,
            side=fill.side,
            timestamp=fill.timestamp,
            reference_price=fill.reference_price,
            price=fill.price,
            quantity=fill.quantity,
            fee=fill.fee,
            reason=fill.reason,
            notional=fill.notional,
            observed_at=fill.observed_at,
        )


class SignalResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    exchange: str
    symbol: str
    side: str
    decision_time: datetime
    signal_open_time: datetime
    atr_14: float
    explanation: str | None = None
    status: str | None = None
    direction: str = "LONG"
    leverage: int = 1
    margin_used: Decimal = Decimal("0")
    stop_loss: Decimal | None = None
    take_profit: Decimal | None = None
    score: float = 0.0

    @classmethod
    def from_signal(cls, signal: BreakoutSignal) -> SignalResponse:
        return cls(
            exchange=signal.exchange,
            symbol=signal.symbol,
            side=signal.side,
            decision_time=signal.decision_time,
            signal_open_time=signal.signal_open_time,
            atr_14=signal.atr_14,
            explanation=getattr(signal, "explanation", None),
            status=getattr(signal, "status", "PENDING"),
            direction=getattr(
                signal,
                "direction",
                "LONG" if str(signal.side).lower() in ("long", "buy") else "SHORT",
            ),
            leverage=int(getattr(signal, "leverage", 1) or 1),
            margin_used=Decimal(str(getattr(signal, "margin_used", Decimal("0")) or "0")),
            stop_loss=Decimal(str(getattr(signal, "stop_loss", Decimal("0")) or "0")),
            take_profit=Decimal(str(getattr(signal, "take_profit", Decimal("0")) or "0")),
            score=float(getattr(signal, "score", 0.0) or 0.0),
        )


class PositionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    symbol: str
    exchange: str
    quantity: Decimal
    entry_fill: FillResponse
    stop_price: Decimal
    target_price: Decimal
    initial_risk: Decimal
    signal: SignalResponse
    direction: str = "LONG"
    leverage: int = 1
    margin_used: Decimal = Decimal("0")
    liquidation_price: Decimal = Decimal("0")
    stop_loss: Decimal = Decimal("0")
    take_profit: Decimal = Decimal("0")

    @classmethod
    def from_position(cls, position: Position) -> PositionResponse:
        return cls(
            symbol=position.symbol,
            exchange=position.exchange,
            quantity=position.quantity,
            entry_fill=FillResponse.from_fill(position.entry_fill),
            stop_price=position.stop_price,
            target_price=position.target_price,
            initial_risk=position.initial_risk,
            signal=SignalResponse.from_signal(position.signal),
            direction=getattr(
                position,
                "direction",
                "LONG" if str(position.signal.side).lower() in ("long", "buy") else "SHORT",
            ),
            leverage=int(getattr(position, "leverage", 1) or 1),
            margin_used=Decimal(str(getattr(position, "margin_used", Decimal("0")) or "0")),
            liquidation_price=Decimal(
                str(getattr(position, "liquidation_price", Decimal("0")) or "0")
            ),
            stop_loss=getattr(position, "stop_loss", position.stop_price),
            take_profit=getattr(position, "take_profit", position.target_price),
        )


class TradeResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    symbol: str
    exchange: str
    quantity: Decimal
    entry_fill: FillResponse
    exit_fill: FillResponse
    stop_price: Decimal
    target_price: Decimal
    exit_reason: str
    gross_pnl: Decimal
    net_pnl: Decimal
    signal: SignalResponse
    explanation: str | None = None
    direction: str = "LONG"
    leverage: int = 1
    margin_used: Decimal = Decimal("0")
    liquidation_price: Decimal = Decimal("0")
    stop_loss: Decimal = Decimal("0")
    take_profit: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")

    @classmethod
    def from_trade(cls, trade: Trade) -> TradeResponse:
        return cls(
            symbol=trade.signal.symbol,
            exchange=trade.signal.exchange,
            quantity=trade.quantity,
            entry_fill=FillResponse.from_fill(trade.entry_fill),
            exit_fill=FillResponse.from_fill(trade.exit_fill),
            stop_price=trade.stop_price,
            target_price=trade.target_price,
            exit_reason=trade.exit_reason,
            gross_pnl=trade.gross_pnl,
            net_pnl=trade.net_pnl,
            signal=SignalResponse.from_signal(trade.signal),
            explanation=getattr(trade, "explanation", None),
            direction=getattr(
                trade,
                "direction",
                "LONG" if str(trade.signal.side).lower() in ("long", "buy") else "SHORT",
            ),
            leverage=int(getattr(trade, "leverage", 1) or 1),
            margin_used=Decimal(str(getattr(trade, "margin_used", Decimal("0")) or "0")),
            liquidation_price=Decimal(
                str(getattr(trade, "liquidation_price", Decimal("0")) or "0")
            ),
            stop_loss=getattr(trade, "stop_loss", trade.stop_price),
            take_profit=getattr(trade, "take_profit", trade.target_price),
            realized_pnl=getattr(trade, "realized_pnl", trade.net_pnl),
        )


class PortfolioResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    cash: Decimal
    equity: Decimal
    open_risk: Decimal
    daily_loss_breached: bool


class EquityPointResponse(BaseModel):
    timestamp: str
    equity: float
    cumulative_pnl: float


class VariantEquityResponse(BaseModel):
    variant_id: str
    name: str
    description: str
    color: str
    starting_cash: float = 100.0
    current_cash: float = 100.0
    current_equity: float = 100.0
    cumulative_pnl: float = 0.0
    open_positions_count: int = 0
    closed_trades_count: int = 0
    points: list[EquityPointResponse] = []


class MultiVariantEquityResponse(BaseModel):
    as_of: str
    variants: dict[str, VariantEquityResponse]


class VariantPositionsResponse(BaseModel):
    variant_a: int = 0
    variant_b: int = 0
    variant_c: int = 0
    variant_d: int = 0
    variant_e: int = 0
    total: int = 0


class SystemStatusResponse(BaseModel):
    btc_regime_state: str = "UNKNOWN"
    btc_regime_timestamp: str | None = None
    regime_blocked_recent: bool = False
    fallback_rate_24h: float = 0.0
    fallback_count_24h: int = 0
    total_evals_24h: int = 0
    variant_positions: VariantPositionsResponse = Field(default_factory=VariantPositionsResponse)
    is_day_zero: bool = True


class GateDecisionResponse(BaseModel):
    id: int
    timestamp: str
    symbol: str
    mode: str
    verdict: str
    regime_state: str
    regime_blocked: bool
    confidence: float
    reason: str
    current_price: float | None = None
    risk_level: str | None = None


# -----------------------------------------------------------------------------
# Variant Configurations
# -----------------------------------------------------------------------------

VARIANT_CONFIGS: list[dict[str, Any]] = [
    {
        "id": "variant_a",
        "name": "Variant A (Gate + Regime)",
        "description": "AI Gate + BTC Regime Filter",
        "color": "#6366f1",  # Indigo
        "db_names": ["portfolio_variant_a.db", "portfolio_a.db", "variant_a.db"],
    },
    {
        "id": "variant_b",
        "name": "Variant B (Regime Only)",
        "description": "BTC Regime Filter Only (No AI Gate)",
        "color": "#06b6d4",  # Cyan
        "db_names": ["portfolio_variant_b.db", "portfolio_b.db", "variant_b.db"],
    },
    {
        "id": "variant_c",
        "name": "Variant C (Gate Only)",
        "description": "AI Gate Only (No Regime Filter)",
        "color": "#a855f7",  # Purple
        "db_names": ["portfolio_variant_c.db", "portfolio_c.db", "variant_c.db"],
    },
    {
        "id": "variant_d",
        "name": "Variant D (Baseline)",
        "description": "Baseline (Unfiltered Strategy)",
        "color": "#94a3b8",  # Slate
        "db_names": ["portfolio_variant_d.db", "portfolio_d.db", "variant_d.db"],
    },
    {
        "id": "variant_e",
        "name": "Variant E (Gate + Regime on Core)",
        "description": "AI Gate + Regime Filter on Core Alts too",
        "color": "#f59e0b",  # Amber
        "db_names": ["portfolio_variant_e.db", "portfolio_e.db", "variant_e.db"],
    },
]


# -----------------------------------------------------------------------------
# Read-Only Database Utilities
# -----------------------------------------------------------------------------


def find_sqlite_db(filename: str) -> Path | None:
    """Find a SQLite database file across standard paths and environment variables."""
    # 1. Environment variable override
    env_name = filename.replace(".", "_").upper() + "_PATH"
    val = os.getenv(env_name)
    if val:
        p = Path(val).resolve()
        if p.is_file():
            return p

    if filename == "fundamental_eval.db":
        eval_path = os.getenv("EVAL_DB_PATH") or os.getenv("FUNDAMENTAL_EVAL_DB")
        if eval_path:
            p = Path(eval_path).resolve()
            if p.is_file():
                return p

    portfolio_dir = os.getenv("PORTFOLIO_DB_DIR")
    if portfolio_dir:
        p = (Path(portfolio_dir) / filename).resolve()
        if p.is_file():
            return p

    # 2. Check candidate search paths
    search_dirs = [
        PROJECT_ROOT,
        Path.cwd(),
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "data" / "live",
        Path.cwd() / "data",
    ]
    for directory in search_dirs:
        p = (directory / filename).resolve()
        if p.is_file():
            return p
    return None


def open_readonly_sqlite(db_path: Path | str) -> sqlite3.Connection | None:
    """Open a SQLite database strictly in READ-ONLY mode using SQLite URI."""
    p = Path(db_path).resolve()
    if not p.is_file():
        return None
    try:
        # Enforce read-only mode at SQLite driver level
        uri = f"file:{p.as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as exc:
        logger.warning("Failed to open SQLite database %s in read-only mode: %s", p, exc)
        return None


def read_variant_from_db(
    conn: sqlite3.Connection,
    variant_key: str,
    default_starting_cash: float = 100.0,
) -> dict[str, Any] | None:
    """Attempt to extract equity curve and position stats from a SQLite connection."""
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cur.fetchall()]
    except Exception:
        return None

    open_positions_count = 0

    # 1. Count open positions if table exists
    for pos_table in ("open_positions", "positions", "paper_positions"):
        if pos_table in tables:
            try:
                pragma = [
                    r["name"] for r in conn.execute(f"PRAGMA table_info({pos_table})").fetchall()
                ]
                if "variant" in pragma:
                    row = conn.execute(
                        f"SELECT COUNT(*) as cnt FROM {pos_table} WHERE variant = ? OR variant = ?",
                        (variant_key, variant_key.replace("variant_", "")),
                    ).fetchone()
                else:
                    row = conn.execute(f"SELECT COUNT(*) as cnt FROM {pos_table}").fetchone()
                if row:
                    open_positions_count = int(row["cnt"])
                    break
            except Exception:
                pass

    # 2. Check for dedicated equity_curve table
    for eq_table in ("equity_curve", "equity_snapshots", "portfolio_snapshots"):
        if eq_table in tables:
            try:
                pragma = [
                    r["name"] for r in conn.execute(f"PRAGMA table_info({eq_table})").fetchall()
                ]
                ts_col = (
                    "timestamp" if "timestamp" in pragma else ("time" if "time" in pragma else None)
                )
                eq_col = "equity" if "equity" in pragma else None
                pnl_col = (
                    "cumulative_pnl"
                    if "cumulative_pnl" in pragma
                    else ("pnl" if "pnl" in pragma else None)
                )

                var_filter = ""
                params: list[Any] = []
                if "variant" in pragma:
                    var_filter = "WHERE variant = ? OR variant = ?"
                    params = [variant_key, variant_key.replace("variant_", "")]

                if ts_col and (eq_col or pnl_col):
                    rows = conn.execute(
                        f"SELECT * FROM {eq_table} {var_filter} ORDER BY {ts_col} ASC", params
                    ).fetchall()
                    if rows:
                        points: list[dict[str, Any]] = []
                        for r in rows:
                            ts_str = str(r[ts_col])
                            eq_val = (
                                float(r[eq_col])
                                if eq_col and r[eq_col] is not None
                                else default_starting_cash
                            )
                            if pnl_col and r[pnl_col] is not None:
                                pnl_val = float(r[pnl_col])
                            else:
                                pnl_val = eq_val - default_starting_cash
                            points.append(
                                {
                                    "timestamp": ts_str,
                                    "equity": eq_val,
                                    "cumulative_pnl": pnl_val,
                                }
                            )
                        if points:
                            current_equity = points[-1]["equity"]
                            cumulative_pnl = points[-1]["cumulative_pnl"]
                            return {
                                "starting_cash": default_starting_cash,
                                "current_cash": current_equity,
                                "current_equity": current_equity,
                                "cumulative_pnl": cumulative_pnl,
                                "open_positions_count": open_positions_count,
                                "closed_trades_count": len(points),
                                "points": points,
                            }
            except Exception:
                pass

    # 3. Check for paper_trades or trades table to construct cumulative PnL curve
    for tr_table in ("paper_trades", "trades"):
        if tr_table in tables:
            try:
                pragma = [
                    r["name"] for r in conn.execute(f"PRAGMA table_info({tr_table})").fetchall()
                ]
                time_col = (
                    "exit_time"
                    if "exit_time" in pragma
                    else ("timestamp" if "timestamp" in pragma else None)
                )
                pnl_col = "pnl" if "pnl" in pragma else ("net_pnl" if "net_pnl" in pragma else None)

                var_filter = ""
                params = []
                if "variant" in pragma:
                    var_filter = "WHERE variant = ? OR variant = ?"
                    params = [variant_key, variant_key.replace("variant_", "")]

                if time_col and pnl_col:
                    sql = f"SELECT * FROM {tr_table} {var_filter} ORDER BY {time_col} ASC"
                    rows = conn.execute(sql, params).fetchall()
                    if rows:
                        closed_trades_count = len(rows)
                        running_pnl = 0.0
                        first_ts = str(rows[0][time_col])
                        points = [
                            {
                                "timestamp": first_ts,
                                "equity": default_starting_cash,
                                "cumulative_pnl": 0.0,
                            }
                        ]
                        for r in rows:
                            val = float(r[pnl_col]) if r[pnl_col] is not None else 0.0
                            running_pnl += val
                            running_eq = default_starting_cash + running_pnl
                            points.append(
                                {
                                    "timestamp": str(r[time_col]),
                                    "equity": round(running_eq, 4),
                                    "cumulative_pnl": round(running_pnl, 4),
                                }
                            )
                        cumulative_pnl = running_pnl
                        current_equity = default_starting_cash + cumulative_pnl
                        return {
                            "starting_cash": default_starting_cash,
                            "current_cash": current_equity,
                            "current_equity": current_equity,
                            "cumulative_pnl": round(cumulative_pnl, 4),
                            "open_positions_count": open_positions_count,
                            "closed_trades_count": closed_trades_count,
                            "points": points,
                        }
            except Exception:
                pass

    return None


def get_variant_data(
    cfg: dict[str, Any],
    session: Any | None = None,
) -> VariantEquityResponse:
    """Extract equity curve, cash, and trade stats for a given variant."""
    var_id = cfg["id"]
    name = cfg["name"]
    desc = cfg["description"]
    color = cfg["color"]

    # 1. Try to read from variant-specific SQLite database on disk
    for db_name in cfg["db_names"]:
        p = find_sqlite_db(db_name)
        if p is not None:
            conn = open_readonly_sqlite(p)
            if conn is not None:
                try:
                    with conn:
                        db_res = read_variant_from_db(conn, var_id)
                        if db_res is not None:
                            return VariantEquityResponse(
                                variant_id=var_id,
                                name=name,
                                description=desc,
                                color=color,
                                **db_res,
                            )
                finally:
                    conn.close()

    # 2. Try to read from a shared portfolio database on disk
    for shared_name in ("portfolio.db", "portfolio_variants.db", "paper_trades.db"):
        p = find_sqlite_db(shared_name)
        if p is not None:
            conn = open_readonly_sqlite(p)
            if conn is not None:
                try:
                    with conn:
                        db_res = read_variant_from_db(conn, var_id)
                        if db_res is not None:
                            return VariantEquityResponse(
                                variant_id=var_id,
                                name=name,
                                description=desc,
                                color=color,
                                **db_res,
                            )
                finally:
                    conn.close()

    # 3. Try to read from in-memory session.executors if active
    if session is not None:
        executors = getattr(session, "executors", None)
        executor = None
        if isinstance(executors, dict) and var_id in executors:
            executor = executors[var_id]
        elif var_id == "variant_a":
            executor = _resolve_executor(session)

        if executor is not None:
            open_count = len(getattr(executor, "positions", {}))
            trades = getattr(executor, "trades", ())
            cash = float(getattr(executor, "cash", Decimal("100")))
            portfolio = getattr(executor, "portfolio", None)
            starting_cash = (
                float(getattr(portfolio, "starting_cash", Decimal("100"))) if portfolio else 100.0
            )
            marks = getattr(executor, "_marks", {})
            current_equity = float(portfolio.equity(marks)) if portfolio else cash

            points: list[EquityPointResponse] = []
            cumulative_pnl = 0.0
            if trades:
                first_time = getattr(
                    trades[0].entry_fill, "timestamp", datetime.now(timezone.utc)
                ).isoformat()
                points.append(
                    EquityPointResponse(
                        timestamp=first_time, equity=starting_cash, cumulative_pnl=0.0
                    )
                )
                for t in trades:
                    pnl = float(getattr(t, "net_pnl", Decimal("0")))
                    cumulative_pnl += pnl
                    ts = getattr(t.exit_fill, "timestamp", datetime.now(timezone.utc)).isoformat()
                    points.append(
                        EquityPointResponse(
                            timestamp=ts,
                            equity=round(starting_cash + cumulative_pnl, 4),
                            cumulative_pnl=round(cumulative_pnl, 4),
                        )
                    )
            else:
                now_iso = datetime.now(timezone.utc).isoformat()
                points.append(
                    EquityPointResponse(
                        timestamp=now_iso, equity=current_equity, cumulative_pnl=0.0
                    )
                )

            return VariantEquityResponse(
                variant_id=var_id,
                name=name,
                description=desc,
                color=color,
                starting_cash=starting_cash,
                current_cash=cash,
                current_equity=current_equity,
                cumulative_pnl=round(cumulative_pnl, 4),
                open_positions_count=open_count,
                closed_trades_count=len(trades),
                points=points,
            )

    # 4. Day 0 baseline initial state (missing databases or empty tables handled gracefully)
    now_iso = datetime.now(timezone.utc).isoformat()
    return VariantEquityResponse(
        variant_id=var_id,
        name=name,
        description=desc,
        color=color,
        starting_cash=100.0,
        current_cash=100.0,
        current_equity=100.0,
        cumulative_pnl=0.0,
        open_positions_count=0,
        closed_trades_count=0,
        points=[EquityPointResponse(timestamp=now_iso, equity=100.0, cumulative_pnl=0.0)],
    )


def get_system_status(session: Any | None = None) -> SystemStatusResponse:
    """Extract BTC regime status, 24h fallback rate, and active variant positions."""
    regime_state = "UNKNOWN"
    regime_ts = None
    regime_blocked = False
    fb_rate = 0.0
    fb_count = 0
    total_evals = 0

    # 1. Query fundamental_eval.db in read-only mode
    p = find_sqlite_db("fundamental_eval.db")
    if p is not None:
        conn = open_readonly_sqlite(p)
        if conn is not None:
            try:
                with conn:
                    tables = [
                        r[0]
                        for r in conn.execute(
                            "SELECT name FROM sqlite_master WHERE type='table'"
                        ).fetchall()
                    ]
                    if "fundamental_evaluations" in tables:
                        cols = {
                            r["name"]
                            for r in conn.execute(
                                "PRAGMA table_info(fundamental_evaluations)"
                            ).fetchall()
                        }

                        # Latest BTC regime state
                        if "regime_state" in cols:
                            row = conn.execute(
                                "SELECT regime_state, regime_blocked, timestamp "
                                "FROM fundamental_evaluations "
                                "WHERE regime_state IS NOT NULL AND TRIM(regime_state) != '' "
                                "ORDER BY id DESC LIMIT 1"
                            ).fetchone()
                            if row:
                                regime_state = str(row["regime_state"]).strip().upper()
                                regime_ts = str(row["timestamp"])
                                regime_blocked = (
                                    bool(row["regime_blocked"])
                                    if "regime_blocked" in cols
                                    else False
                                )

                        # 24h Fallback rate
                        cutoff_24h = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
                        fb_sql = (
                            "SELECT COUNT(*) as total_cnt, "
                            "SUM(CASE WHEN verdict = 'FALLBACK_REJECT' OR is_fallback = 1 THEN 1 ELSE 0 END) as fb_cnt "
                            "FROM fundamental_evaluations WHERE timestamp >= ?"
                        )
                        fb_row = conn.execute(fb_sql, (cutoff_24h,)).fetchone()
                        if fb_row:
                            total_evals = int(fb_row["total_cnt"] or 0)
                            fb_count = int(fb_row["fb_cnt"] or 0)
                            if total_evals > 0:
                                fb_rate = round((fb_count / total_evals) * 100.0, 2)
            finally:
                conn.close()

    # Fallback to session regime filter if regime_state is still UNKNOWN
    if regime_state == "UNKNOWN" and session is not None:
        filter_inst = getattr(session, "btc_regime_filter", None)
        if filter_inst is not None and hasattr(filter_inst, "current_regime"):
            regime_state = str(getattr(filter_inst, "current_regime", "UNKNOWN")).strip().upper()

    # 2. Count active open positions for each variant
    pos_counts: dict[str, int] = {}
    for cfg in VARIANT_CONFIGS:
        var_id = cfg["id"]
        cnt = 0
        # Check SQLite db if available
        for db_name in cfg["db_names"]:
            db_p = find_sqlite_db(db_name)
            if db_p:
                c = open_readonly_sqlite(db_p)
                if c:
                    try:
                        with c:
                            t_names = [
                                r[0]
                                for r in c.execute(
                                    "SELECT name FROM sqlite_master WHERE type='table'"
                                ).fetchall()
                            ]
                            for pt in ("open_positions", "positions"):
                                if pt in t_names:
                                    r = c.execute(f"SELECT COUNT(*) as cnt FROM {pt}").fetchone()
                                    if r:
                                        cnt = int(r["cnt"])
                                        break
                    finally:
                        c.close()
            if cnt > 0:
                break

        # Fallback to in-memory session
        if cnt == 0 and session is not None:
            executors = getattr(session, "executors", None)
            if isinstance(executors, dict) and var_id in executors:
                cnt = len(getattr(executors[var_id], "positions", {}))
            elif var_id == "variant_a":
                ex = _resolve_executor(session)
                cnt = len(getattr(ex, "positions", {}))
        pos_counts[var_id] = cnt

    total_pos = sum(pos_counts.values())
    variant_pos = VariantPositionsResponse(
        variant_a=pos_counts.get("variant_a", 0),
        variant_b=pos_counts.get("variant_b", 0),
        variant_c=pos_counts.get("variant_c", 0),
        variant_d=pos_counts.get("variant_d", 0),
        variant_e=pos_counts.get("variant_e", 0),
        total=total_pos,
    )

    is_day_zero = total_evals == 0 and total_pos == 0

    return SystemStatusResponse(
        btc_regime_state=regime_state,
        btc_regime_timestamp=regime_ts,
        regime_blocked_recent=regime_blocked,
        fallback_rate_24h=fb_rate,
        fallback_count_24h=fb_count,
        total_evals_24h=total_evals,
        variant_positions=variant_pos,
        is_day_zero=is_day_zero,
    )


def get_gate_decisions(
    limit: int = 100,
    symbol: str | None = None,
    verdict: str | None = None,
) -> list[GateDecisionResponse]:
    """Retrieve fundamental evaluation decision records from fundamental_eval.db in read-only mode."""
    p = find_sqlite_db("fundamental_eval.db")
    if p is None:
        return []
    conn = open_readonly_sqlite(p)
    if conn is None:
        return []
    try:
        with conn:
            tables = [
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            ]
            if "fundamental_evaluations" not in tables:
                return []
            cols = {
                r["name"]
                for r in conn.execute("PRAGMA table_info(fundamental_evaluations)").fetchall()
            }

            query = "SELECT * FROM fundamental_evaluations WHERE 1=1"
            params: list[Any] = []
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol.strip().upper())
            if verdict:
                v_clean = verdict.strip().upper()
                if v_clean == "FALLBACK_REJECT":
                    query += " AND (verdict = 'FALLBACK_REJECT' OR is_fallback = 1)"
                elif v_clean == "BLOCKED_BY_REGIME":
                    query += " AND (verdict = 'BLOCKED_BY_REGIME' OR regime_blocked = 1)"
                else:
                    query += " AND verdict = ?"
                    params.append(v_clean)

            query += " ORDER BY id DESC LIMIT ?"
            params.append(int(limit))

            rows = conn.execute(query, params).fetchall()
            results: list[GateDecisionResponse] = []
            for r in rows:
                raw_verdict = str(r["verdict"] or "").strip().upper()
                is_fb = bool(r["is_fallback"] if "is_fallback" in cols else 0)
                is_blocked = bool(r["regime_blocked"] if "regime_blocked" in cols else 0)

                effective_verdict = raw_verdict
                if is_fb and (raw_verdict == "REJECT" or not raw_verdict):
                    effective_verdict = "FALLBACK_REJECT"
                elif is_blocked and (raw_verdict == "BLOCKED_BY_REGIME" or not raw_verdict):
                    effective_verdict = "BLOCKED_BY_REGIME"
                elif not effective_verdict:
                    effective_verdict = "UNKNOWN"

                results.append(
                    GateDecisionResponse(
                        id=int(r["id"]),
                        timestamp=str(r["timestamp"]),
                        symbol=str(r["symbol"]),
                        mode=str(r["mode"] if "mode" in cols and r["mode"] else "ENTRY").upper(),
                        verdict=effective_verdict,
                        regime_state=str(
                            r["regime_state"]
                            if "regime_state" in cols and r["regime_state"]
                            else ""
                        ).upper(),
                        regime_blocked=is_blocked,
                        confidence=float(
                            r["confidence"]
                            if "confidence" in cols and r["confidence"] is not None
                            else 1.0
                        ),
                        reason=str(r["reason"] if "reason" in cols and r["reason"] else ""),
                        current_price=float(r["current_price"])
                        if "current_price" in cols and r["current_price"] is not None
                        else None,
                        risk_level=str(r["risk_level"]).upper()
                        if "risk_level" in cols and r["risk_level"]
                        else None,
                    )
                )
            return results
    finally:
        conn.close()


# -----------------------------------------------------------------------------
# Global Dashboard State
# -----------------------------------------------------------------------------


class DashboardState:
    """Thread-safe holder for the global active session/executor reference."""

    def __init__(self) -> None:
        self._session: Any = None

    def set_session(self, session: Any) -> None:
        self._session = session

    def get_session(self) -> Any:
        if self._session is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Paper session is not initialized or running",
            )
        return self._session


dashboard_state = DashboardState()


def get_active_session() -> Any:
    """Dependency returning the currently active dashboard paper session."""
    return dashboard_state.get_session()


def set_dashboard_session(session: Any) -> None:
    """Set the active paper session or executor for dashboard endpoints."""
    dashboard_state.set_session(session)


def _resolve_executor(session_or_executor: Any) -> Any:
    """Extract the PaperExecutor from a session or return target directly."""
    if hasattr(session_or_executor, "executor"):
        return session_or_executor.executor
    return session_or_executor


# -----------------------------------------------------------------------------
# Router Factory
# -----------------------------------------------------------------------------


def create_dashboard_router(
    session_source: Any | Callable[[], Any] | None = None,
) -> APIRouter:
    """Create a FastAPI APIRouter configured for a specific paper session source."""
    router = APIRouter()

    def _get_target_session() -> Any:
        if callable(session_source):
            target = session_source()
        elif session_source is not None:
            target = session_source
        else:
            target = get_active_session()

        if target is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Paper session is not available",
            )
        return target

    def _get_optional_session() -> Any | None:
        if callable(session_source):
            try:
                return session_source()
            except Exception:
                return None
        elif session_source is not None:
            return session_source
        else:
            return dashboard_state._session

    @router.get("/", response_class=FileResponse)
    @router.get("/ui", response_class=FileResponse)
    def serve_ui() -> FileResponse:
        static_index = Path(__file__).resolve().parent.parent / "static" / "index.html"
        if not static_index.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dashboard static index.html not found",
            )
        return FileResponse(static_index, media_type="text/html")

    @router.get("/portfolio", response_model=PortfolioResponse)
    def get_portfolio(
        session: Any = Depends(_get_target_session),
    ) -> PortfolioResponse:
        executor = _resolve_executor(session)
        portfolio = getattr(executor, "portfolio", None)
        marks = getattr(executor, "_marks", {})

        cash = getattr(executor, "cash", Decimal("0"))
        if portfolio is not None:
            equity = portfolio.equity(marks)
            open_risk = portfolio.open_risk
            daily_loss_breached = portfolio.daily_loss_breached
        else:
            equity = getattr(executor, "equity", cash)
            open_risk = getattr(executor, "open_risk", Decimal("0"))
            daily_loss_breached = getattr(executor, "daily_loss_breached", False)

        return PortfolioResponse(
            cash=cash,
            equity=equity,
            open_risk=open_risk,
            daily_loss_breached=daily_loss_breached,
        )

    @router.get("/positions", response_model=list[PositionResponse])
    def get_positions(
        session: Any = Depends(_get_target_session),
    ) -> list[PositionResponse]:
        executor = _resolve_executor(session)
        positions = getattr(executor, "positions", {})
        pos_list = list(positions.values()) if isinstance(positions, dict) else list(positions)
        return [PositionResponse.from_position(p) for p in pos_list]

    @router.get("/trades", response_model=list[TradeResponse])
    def get_trades(
        session: Any = Depends(_get_target_session),
    ) -> list[TradeResponse]:
        executor = _resolve_executor(session)
        trades = getattr(executor, "trades", ())
        return [TradeResponse.from_trade(t) for t in trades]

    @router.get("/signals", response_model=list[SignalResponse])
    def get_signals(
        session: Any = Depends(_get_target_session),
    ) -> list[SignalResponse]:
        signals = getattr(session, "pending_signals", None)
        if signals is None:
            executor = _resolve_executor(session)
            signals = getattr(executor, "pending_signals", ())
        vetoed = getattr(session, "vetoed_signals", ())
        combined = list(signals) + list(vetoed)
        return [SignalResponse.from_signal(s) for s in combined]

    # New 5-Variant Experiment Endpoints
    @router.get("/variants/equity", response_model=MultiVariantEquityResponse)
    @router.get("/portfolio/variants", response_model=MultiVariantEquityResponse)
    def get_variants_equity(
        session: Any | None = Depends(_get_optional_session),
    ) -> MultiVariantEquityResponse:
        now_iso = datetime.now(timezone.utc).isoformat()
        variants_dict = {cfg["id"]: get_variant_data(cfg, session) for cfg in VARIANT_CONFIGS}
        return MultiVariantEquityResponse(as_of=now_iso, variants=variants_dict)

    @router.get("/system/status", response_model=SystemStatusResponse)
    def get_system_status_endpoint(
        session: Any | None = Depends(_get_optional_session),
    ) -> SystemStatusResponse:
        return get_system_status(session)

    @router.get("/gate/decisions", response_model=list[GateDecisionResponse])
    def get_gate_decisions_endpoint(
        limit: int = Query(100, ge=1, le=1000),
        symbol: str | None = Query(None),
        verdict: str | None = Query(None),
    ) -> list[GateDecisionResponse]:
        return get_gate_decisions(limit=limit, symbol=symbol, verdict=verdict)

    return router


def create_dashboard_app(
    session_source: Any | Callable[[], Any] | None = None,
) -> FastAPI:
    """Create a standalone read-only dashboard FastAPI application."""
    app = FastAPI(title="Crypto AI Trader Dashboard")
    router = create_dashboard_router(session_source)
    app.include_router(router)
    return app


# Default router using the global active session dependency
router = create_dashboard_router(None)

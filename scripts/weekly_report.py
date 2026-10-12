"""Weekly performance reporting and fundamental AI gate alpha teardown.

Outputs:
1. Multi-Portfolio Performance Teardown across all 5 variants:
   - Variant A: AI Gate + Regime Filter (Current)
   - Variant B: Regime Filter only (No AI Gate)
   - Variant C: AI Gate only (No Regime Filter)
   - Variant D: Baseline (No Gate, No Regime Filter)
   - Variant E: AI Gate + Regime Filter applied to Core Alts too
   Metrics compared: Total PnL, Max Drawdown, Sharpe Ratio, Sortino Ratio, Win Rate, Total Trades, Excess Return vs BTC.

2. Benchmark Performance:
   - Buy-and-Hold BTC Return.
   - Equal-weight Core basket (BTC, ETH, SOL, BNB, ZEC).

3. AI Fundamental Risk Gatekeeper Evaluation Alpha Analysis:
   - Compares the average and median T+24h returns of signals the AI Approved vs signals the AI Rejected.
   - Separates model rejections from fallback rejections (rate limits/outages excluded from alpha spread).
   - Counterfactual returns reported BOTH Gross and Net of round-trip costs (0.60% hurdle for Satellites, 0.30% for Core).
   - Overlap adjustment: Subsamples signals to at most 1 observation per symbol per rolling 24-hour window.
   - Robust statistics: count, median, worst single return, % below -5% (dodged bullets).
   - Deduplication of repeated rejections by (symbol, catalyst).

Usage:
    .venv/bin/python scripts/weekly_report.py [--db-path fundamental_eval.db]
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sqlite3
import statistics
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Sequence

from app.analysis.eval_logger import DEFAULT_DB_PATH
from app.backtest.models import Trade
from app.market.constants import FIXED_SYMBOLS
from app.trading.paper import PaperExecutor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("weekly_report")

VARIANT_NAMES = {
    "variant_a": "Variant A (AI Gate + Regime)",
    "variant_b": "Variant B (Regime Only)",
    "variant_c": "Variant C (AI Gate Only)",
    "variant_d": "Variant D (Baseline)",
    "variant_e": "Variant E (AI Gate + Regime on Core Alts)",
}

# Realistic round-trip cost hurdles: 2 x fee (0.10%) + 2 x slippage
CORE_COST_HURDLE = 0.003  # 2 x 0.10% + 2 x 0.05% = 0.30%
SATELLITE_COST_HURDLE = 0.006  # 2 x 0.10% + 2 x 0.20% = 0.60%


def parse_timestamp(ts_val: Any) -> datetime | None:
    """Safely parse datetime or ISO timestamp string into timezone-aware UTC datetime."""
    if ts_val is None:
        return None
    if isinstance(ts_val, datetime):
        return ts_val if ts_val.tzinfo is not None else ts_val.replace(tzinfo=timezone.utc)
    if isinstance(ts_val, (int, float)):
        return datetime.fromtimestamp(float(ts_val), tz=timezone.utc)
    try:
        ts_str = str(ts_val).strip()
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except Exception:
        return None


def calculate_sharpe_ratio(
    returns: Sequence[float | Decimal],
    risk_free_rate: float = 0.0,
    annualize: bool = False,
    annualization_factor: float = math.sqrt(365),
) -> float | None:
    """Calculate sample Sharpe ratio from returns (optionally annualized using sqrt(365))."""
    if len(returns) < 2:
        return None
    float_returns = [float(r) for r in returns]
    n = len(float_returns)
    mean_ret = sum(float_returns) / n
    excess_mean = mean_ret - risk_free_rate
    var = sum((r - mean_ret) ** 2 for r in float_returns) / (n - 1)
    if var <= 0.0:
        return 0.0
    std = math.sqrt(var)
    ratio = excess_mean / std if std > 0 else 0.0
    if annualize:
        ratio *= annualization_factor
    return ratio


def calculate_sortino_ratio(
    returns: Sequence[float | Decimal],
    risk_free_rate: float = 0.0,
    annualize: bool = False,
    annualization_factor: float = math.sqrt(365),
) -> float | None:
    """Calculate Sortino ratio using downside standard deviation below risk-free rate."""
    if len(returns) < 2:
        return None
    float_returns = [float(r) for r in returns]
    n = len(float_returns)
    mean_ret = sum(float_returns) / n
    excess_mean = mean_ret - risk_free_rate
    downside_sq_sum = sum(min(0.0, r - risk_free_rate) ** 2 for r in float_returns)
    if downside_sq_sum <= 0.0:
        return None if excess_mean <= 0 else float("inf")
    downside_std = math.sqrt(downside_sq_sum / n)
    ratio = excess_mean / downside_std if downside_std > 0 else 0.0
    if annualize and not math.isinf(ratio):
        ratio *= annualization_factor
    return ratio


def calculate_variant_metrics(
    trades: Sequence[Trade],
    initial_cash: Decimal = Decimal("10000.00"),
    btc_return: float | None = None,
) -> dict[str, Any]:
    """Calculate comprehensive performance metrics for a sequence of completed trades.

    Sharpe and Sortino ratios are calculated using daily mark-to-market equity returns (annualized with sqrt(365)).
    If fewer than 2 distinct calendar days exist, discrete trade returns are used as fallback.
    """
    total_trades = len(trades)
    if total_trades == 0:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "net_pnl": 0.0,
            "net_pnl_pct": 0.0,
            "excess_return_btc": (-btc_return) if btc_return is not None else None,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": None,
            "sortino_ratio": None,
        }

    winning = [t for t in trades if t.net_pnl > Decimal("0")]
    losing = [t for t in trades if t.net_pnl < Decimal("0")]
    win_rate = float(len(winning) / total_trades)

    net_pnl = sum((t.net_pnl for t in trades), Decimal("0"))
    net_pnl_pct = float(net_pnl / initial_cash) if initial_cash > 0 else 0.0
    excess_btc = (net_pnl_pct - btc_return) if btc_return is not None else None

    # Calculate running equity for drawdown
    equity = initial_cash
    peak = equity
    max_dd = Decimal("0")
    max_dd_pct = Decimal("0")

    for t in trades:
        equity += t.net_pnl
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd
            max_dd_pct = dd / peak if peak > Decimal("0") else Decimal("0")

    # Mark-to-market daily returns
    daily_pnl: dict[Any, Decimal] = {}
    for t in trades:
        d = t.exit_fill.timestamp.date()
        daily_pnl[d] = daily_pnl.get(d, Decimal("0")) + t.net_pnl

    sorted_dates = sorted(daily_pnl.keys())
    daily_returns: list[float] = []

    if len(sorted_dates) >= 2:
        cur_date = sorted_dates[0]
        end_date = sorted_dates[-1]
        cur_equity = initial_cash
        while cur_date <= end_date:
            day_pnl = daily_pnl.get(cur_date, Decimal("0"))
            next_equity = cur_equity + day_pnl
            if cur_equity > Decimal("0"):
                daily_returns.append(float(day_pnl / cur_equity))
            else:
                daily_returns.append(0.0)
            cur_equity = next_equity
            cur_date += timedelta(days=1)
    else:
        # Fallback to trade returns if trades occurred on fewer than 2 distinct days
        for t in trades:
            cost = t.entry_fill.price * t.entry_fill.quantity
            if cost > Decimal("0"):
                daily_returns.append(float(t.net_pnl / cost))
            elif initial_cash > Decimal("0"):
                daily_returns.append(float(t.net_pnl / initial_cash))

    sharpe = calculate_sharpe_ratio(daily_returns, annualize=True)
    sortino = calculate_sortino_ratio(daily_returns, annualize=True)

    return {
        "total_trades": total_trades,
        "winning_trades": len(winning),
        "losing_trades": len(losing),
        "win_rate": win_rate,
        "net_pnl": float(net_pnl),
        "net_pnl_pct": net_pnl_pct,
        "excess_return_btc": excess_btc,
        "max_drawdown": float(max_dd),
        "max_drawdown_pct": float(max_dd_pct),
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
    }


def _calculate_return_stats(returns: Sequence[float]) -> dict[str, Any]:
    """Compute count, average, median, worst single return, and percentage below -5%."""
    n = len(returns)
    if n == 0:
        return {
            "count": 0,
            "avg": None,
            "median": None,
            "worst": None,
            "pct_below_neg_5": None,
        }
    float_rets = [float(r) for r in returns]
    avg_val = sum(float_rets) / n
    median_val = float(statistics.median(float_rets))
    worst_val = min(float_rets)
    below_neg_5 = sum(1 for r in float_rets if r < -0.05) / n
    return {
        "count": n,
        "avg": avg_val,
        "median": median_val,
        "worst": worst_val,
        "pct_below_neg_5": below_neg_5,
    }


def _filter_overlapping_24h(rows: list[sqlite3.Row]) -> list[sqlite3.Row]:
    """Subsample signals to at most 1 observation per symbol per rolling 24-hour window."""
    if not rows or "timestamp" not in rows[0].keys():
        return list(rows)

    # Validate that timestamps can be parsed
    parsed_pairs: list[tuple[datetime, sqlite3.Row]] = []
    for r in rows:
        dt = parse_timestamp(r["timestamp"])
        if dt is None:
            return list(rows)
        parsed_pairs.append((dt, r))

    parsed_pairs.sort(key=lambda item: item[0])
    last_ts_by_sym: dict[str, datetime] = {}
    accepted: list[sqlite3.Row] = []

    for dt, r in parsed_pairs:
        sym = str(r["symbol"]).strip() if "symbol" in r.keys() and r["symbol"] else "UNKNOWN"
        last_ts = last_ts_by_sym.get(sym)
        if last_ts is None or (dt - last_ts).total_seconds() >= 24 * 3600:
            accepted.append(r)
            last_ts_by_sym[sym] = dt

    return accepted


def query_fundamental_alpha_comparison(db_path: str = DEFAULT_DB_PATH) -> dict[str, Any]:
    """Query fundamental evaluation records and compare T+24h returns of Approved vs Rejected signals.

    Features:
    - Excludes fallback rejections from alpha spread (reports Fallback Rate and Outage Window Count separately).
    - Subsamples signals to 1 observation per symbol per rolling 24-hour window to avoid auto-correlation.
    - Deduplicates repeated rejections for the same (symbol, catalyst).
    - Computes counterfactuals BOTH Gross and Net of round-trip execution costs.
    """
    if not os.path.exists(db_path):
        return {
            "approved_count": 0,
            "rejected_count": 0,
            "approved_avg_24h": None,
            "rejected_avg_24h": None,
            "alpha_spread": None,
            "approved_net_avg_24h": None,
            "rejected_net_avg_24h": None,
            "net_alpha_spread": None,
            "approved_median_24h": None,
            "rejected_median_24h": None,
            "approved_net_median_24h": None,
            "rejected_net_median_24h": None,
            "median_spread": None,
            "net_median_spread": None,
            "approved_worst_24h": None,
            "rejected_worst_24h": None,
            "approved_pct_below_neg_5": None,
            "rejected_pct_below_neg_5": None,
            "fallback_count": 0,
            "fallback_rate": 0.0,
            "outage_window_count": 0,
            "regime_blocked_count": 0,
            "regime_blocked_avg_24h": None,
            "summary": f"Database file not found: {db_path}",
        }

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    with conn:
        pragma_cols = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(fundamental_evaluations)").fetchall()
        }

        select_cols = ["return_24h"]
        for extra in (
            "symbol",
            "driving_news_id",
            "reason",
            "timestamp",
            "is_fallback",
            "net_return_24h",
            "regime_state",
            "regime_blocked",
        ):
            if extra in pragma_cols:
                select_cols.append(extra)
        cols_sql = ", ".join(select_cols)

        # 1. Approved signals
        approved_rows = conn.execute(
            f"SELECT {cols_sql} FROM fundamental_evaluations WHERE verdict = 'APPROVE' AND return_24h IS NOT NULL"
        ).fetchall()

        # 2. Model Rejections (exclude fallback rejections)
        if "is_fallback" in pragma_cols:
            rej_cond = "(verdict = 'REJECT') AND (is_fallback IS NULL OR is_fallback = 0) AND verdict != 'FALLBACK_REJECT'"
            fb_cond = "verdict = 'FALLBACK_REJECT' OR is_fallback = 1"
        else:
            rej_cond = "verdict = 'REJECT'"
            fb_cond = "verdict = 'FALLBACK_REJECT'"

        rejected_rows = conn.execute(
            f"SELECT {cols_sql} FROM fundamental_evaluations WHERE {rej_cond} AND return_24h IS NOT NULL"
        ).fetchall()

        # 3. Fallback records (for outage & fallback rate calculation)
        fallback_rows = conn.execute(
            f"SELECT {cols_sql} FROM fundamental_evaluations WHERE {fb_cond}"
        ).fetchall()

        total_eval_count = conn.execute(
            "SELECT COUNT(*) as cnt FROM fundamental_evaluations"
        ).fetchone()["cnt"]

        # 4. Regime-blocked signals (check dedicated regime_blocked column or legacy verdict)
        if "regime_blocked" in pragma_cols:
            reg_cond = "(regime_blocked = 1 OR verdict = 'BLOCKED_BY_REGIME')"
        else:
            reg_cond = "verdict = 'BLOCKED_BY_REGIME'"

        regime_rows = conn.execute(
            f"SELECT {cols_sql} FROM fundamental_evaluations WHERE {reg_cond} AND return_24h IS NOT NULL"
        ).fetchall()

    conn.close()

    # Step A: Overlap adjustment (subsample to 1 obs per symbol per 24h rolling window)
    app_overlap_adj = _filter_overlapping_24h(approved_rows)
    rej_overlap_adj = _filter_overlapping_24h(rejected_rows)

    # Step B: Deduplicate repeated rejections for the same symbol and catalyst
    seen_rejections: set[tuple[str, str]] = set()
    rej_deduped: list[sqlite3.Row] = []
    for r in rej_overlap_adj:
        keys = r.keys()
        sym = str(r["symbol"]).strip() if "symbol" in keys and r["symbol"] else ""
        catalyst = ""
        if "driving_news_id" in keys and r["driving_news_id"] and str(r["driving_news_id"]).strip():
            catalyst = str(r["driving_news_id"]).strip()
        elif "reason" in keys and r["reason"] and str(r["reason"]).strip():
            catalyst = str(r["reason"]).strip()

        if sym or catalyst:
            key = (sym, catalyst)
            if key in seen_rejections:
                continue
            seen_rejections.add(key)
        rej_deduped.append(r)

    # Step C: Extract Gross and Net returns for counterfactuals
    def extract_gross_and_net(rows: Sequence[sqlite3.Row]) -> tuple[list[float], list[float]]:
        gross_list: list[float] = []
        net_list: list[float] = []
        for r in rows:
            keys = r.keys()
            sym = str(r["symbol"]).strip() if "symbol" in keys and r["symbol"] else ""
            is_core = sym in FIXED_SYMBOLS
            hurdle = CORE_COST_HURDLE if is_core else SATELLITE_COST_HURDLE

            g = float(r["return_24h"])
            if "net_return_24h" in keys and r["net_return_24h"] is not None:
                n = float(r["net_return_24h"])
            else:
                n = g - hurdle

            gross_list.append(g)
            net_list.append(n)
        return gross_list, net_list

    app_gross, app_net = extract_gross_and_net(app_overlap_adj)
    rej_gross, rej_net = extract_gross_and_net(rej_deduped)
    reg_gross, reg_net = extract_gross_and_net(regime_rows)

    app_stats_g = _calculate_return_stats(app_gross)
    app_stats_n = _calculate_return_stats(app_net)
    rej_stats_g = _calculate_return_stats(rej_gross)
    rej_stats_n = _calculate_return_stats(rej_net)
    reg_stats_g = _calculate_return_stats(reg_gross)

    alpha_spread_g = (
        (app_stats_g["avg"] - rej_stats_g["avg"])
        if (app_stats_g["avg"] is not None and rej_stats_g["avg"] is not None)
        else None
    )
    alpha_spread_n = (
        (app_stats_n["avg"] - rej_stats_n["avg"])
        if (app_stats_n["avg"] is not None and rej_stats_n["avg"] is not None)
        else None
    )

    med_spread_g = (
        (app_stats_g["median"] - rej_stats_g["median"])
        if (app_stats_g["median"] is not None and rej_stats_g["median"] is not None)
        else None
    )
    med_spread_n = (
        (app_stats_n["median"] - rej_stats_n["median"])
        if (app_stats_n["median"] is not None and rej_stats_n["median"] is not None)
        else None
    )

    # Step D: Fallback rate and Outage Windows calculation
    fb_count = len(fallback_rows)
    fb_rate = (fb_count / total_eval_count) if total_eval_count > 0 else 0.0

    outage_window_count = 0
    if fallback_rows and "timestamp" in fallback_rows[0].keys():
        fb_dts = [
            parse_timestamp(r["timestamp"])
            for r in fallback_rows
            if parse_timestamp(r["timestamp"]) is not None
        ]
        fb_dts.sort()
        if fb_dts:
            outage_window_count = 1
            last_dt = fb_dts[0]
            for cur_dt in fb_dts[1:]:
                if (cur_dt - last_dt).total_seconds() > 3600:
                    outage_window_count += 1
                last_dt = cur_dt
    elif fb_count > 0:
        outage_window_count = 1

    # Step E: Cross-tabulation by BTC Regime (UPTREND vs DOWNTREND)
    def _get_row_regime(r: sqlite3.Row) -> str:
        keys = r.keys()
        if "regime_state" in keys and r["regime_state"]:
            return str(r["regime_state"]).strip().upper()
        return ""

    uptrend_app = [r for r in app_overlap_adj if _get_row_regime(r) == "UPTREND"]
    uptrend_rej = [r for r in rej_deduped if _get_row_regime(r) == "UPTREND"]
    downtrend_app = [r for r in app_overlap_adj if _get_row_regime(r) == "DOWNTREND"]
    downtrend_rej = [r for r in rej_deduped if _get_row_regime(r) == "DOWNTREND"]

    def _compute_conditional(
        app_sub: Sequence[sqlite3.Row], rej_sub: Sequence[sqlite3.Row]
    ) -> dict[str, Any]:
        a_g, a_n = extract_gross_and_net(app_sub)
        r_g, r_n = extract_gross_and_net(rej_sub)
        a_s_g = _calculate_return_stats(a_g)
        a_s_n = _calculate_return_stats(a_n)
        r_s_g = _calculate_return_stats(r_g)
        r_s_n = _calculate_return_stats(r_n)
        sp_g = (
            (a_s_g["avg"] - r_s_g["avg"])
            if (a_s_g["avg"] is not None and r_s_g["avg"] is not None)
            else None
        )
        sp_n = (
            (a_s_n["avg"] - r_s_n["avg"])
            if (a_s_n["avg"] is not None and r_s_n["avg"] is not None)
            else None
        )
        return {
            "approved_count": a_s_g["count"],
            "rejected_count": r_s_g["count"],
            "approved_avg": a_s_g["avg"],
            "rejected_avg": r_s_g["avg"],
            "approved_net_avg": a_s_n["avg"],
            "rejected_net_avg": r_s_n["avg"],
            "alpha_spread": sp_g,
            "net_alpha_spread": sp_n,
        }

    uptrend_stats = _compute_conditional(uptrend_app, uptrend_rej)
    downtrend_stats = _compute_conditional(downtrend_app, downtrend_rej)

    if alpha_spread_g is not None:
        dodged_note = ""
        if rej_stats_g["pct_below_neg_5"] is not None and rej_stats_g["pct_below_neg_5"] > 0:
            dodged_note = (
                f" Dodged bullet rate: {rej_stats_g['pct_below_neg_5']:.1%} of vetoed signals fell > 5% "
                f"within 24h (worst: {rej_stats_g['worst']:+.2%})."
            )

        if alpha_spread_g > 0:
            summary = (
                f"POSITIVE ALPHA: AI Gate generated {alpha_spread_g:+.2%} alpha spread (Net: {alpha_spread_n:+.2%}). "
                f"Approved trades averaged {app_stats_g['avg']:+.2%} (median: {app_stats_g['median']:+.2%}), "
                f"while Vetoed trades averaged {rej_stats_g['avg']:+.2%} (median: {rej_stats_g['median']:+.2%}).{dodged_note}"
            )
        else:
            summary = (
                f"NEGATIVE/NEUTRAL ALPHA: AI Gate spread is {alpha_spread_g:+.2%} (Net: {alpha_spread_n:+.2%}). "
                f"Approved trades averaged {app_stats_g['avg']:+.2%} (median: {app_stats_g['median']:+.2%}), "
                f"while Vetoed trades averaged {rej_stats_g['avg']:+.2%} (median: {rej_stats_g['median']:+.2%}).{dodged_note}"
            )
    else:
        summary = "Insufficient completed T+24h return data to compute alpha spread."

    return {
        "approved_count": app_stats_g["count"],
        "rejected_count": rej_stats_g["count"],
        "approved_avg_24h": app_stats_g["avg"],
        "rejected_avg_24h": rej_stats_g["avg"],
        "alpha_spread": alpha_spread_g,
        "approved_net_avg_24h": app_stats_n["avg"],
        "rejected_net_avg_24h": rej_stats_n["avg"],
        "net_alpha_spread": alpha_spread_n,
        "approved_median_24h": app_stats_g["median"],
        "rejected_median_24h": rej_stats_g["median"],
        "approved_net_median_24h": app_stats_n["median"],
        "rejected_net_median_24h": rej_stats_n["median"],
        "median_spread": med_spread_g,
        "net_median_spread": med_spread_n,
        "approved_worst_24h": app_stats_g["worst"],
        "rejected_worst_24h": rej_stats_g["worst"],
        "approved_pct_below_neg_5": app_stats_g["pct_below_neg_5"],
        "rejected_pct_below_neg_5": rej_stats_g["pct_below_neg_5"],
        "fallback_count": fb_count,
        "fallback_rate": fb_rate,
        "outage_window_count": outage_window_count,
        "regime_blocked_count": reg_stats_g["count"],
        "regime_blocked_avg_24h": reg_stats_g["avg"],
        "uptrend_approved_count": uptrend_stats["approved_count"],
        "uptrend_rejected_count": uptrend_stats["rejected_count"],
        "uptrend_alpha_spread": uptrend_stats["alpha_spread"],
        "uptrend_net_alpha_spread": uptrend_stats["net_alpha_spread"],
        "downtrend_approved_count": downtrend_stats["approved_count"],
        "downtrend_rejected_count": downtrend_stats["rejected_count"],
        "downtrend_alpha_spread": downtrend_stats["alpha_spread"],
        "downtrend_net_alpha_spread": downtrend_stats["net_alpha_spread"],
        "by_regime": {
            "UPTREND": uptrend_stats,
            "DOWNTREND": downtrend_stats,
        },
        "summary": summary,
    }


def compute_benchmark_metrics(
    variant_trades: dict[str, Sequence[Trade]] | None = None,
    executors: dict[str, PaperExecutor] | None = None,
    btc_return: float | None = None,
    basket_return: float | None = None,
    db_path: str = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    """Compute benchmark returns for Buy-and-Hold BTC and equal-weight Core basket (BTC, ETH, SOL, BNB, ZEC)."""
    if btc_return is not None and basket_return is not None:
        return {"btc_return": btc_return, "basket_return": basket_return}

    # Collect all trades
    all_trades: list[Trade] = []
    if variant_trades:
        for tr_list in variant_trades.values():
            all_trades.extend(tr_list)
    elif executors:
        for exec_inst in executors.values():
            all_trades.extend(exec_inst.trades)

    # Ingest prices by symbol from trades
    trades_by_sym: dict[str, list[Trade]] = {}
    for t in all_trades:
        trades_by_sym.setdefault(t.signal.symbol, []).append(t)

    # Calculate returns strictly across fixed Core assets (BTC, ETH, SOL, BNB, ZEC)
    core_returns: dict[str, float] = {}
    for sym in FIXED_SYMBOLS:
        if sym in trades_by_sym:
            sym_trades_sorted = sorted(trades_by_sym[sym], key=lambda tr: tr.entry_fill.timestamp)
            first_entry = float(sym_trades_sorted[0].entry_fill.price)
            last_exit = float(sym_trades_sorted[-1].exit_fill.price)
            if first_entry > 0:
                core_returns[sym] = (last_exit - first_entry) / first_entry

    computed_btc = btc_return
    if computed_btc is None:
        if "BTCUSDT" in core_returns:
            computed_btc = core_returns["BTCUSDT"]
        elif "BTCUSDT" in trades_by_sym:
            b_trades = sorted(trades_by_sym["BTCUSDT"], key=lambda tr: tr.entry_fill.timestamp)
            first_entry = float(b_trades[0].entry_fill.price)
            last_exit = float(b_trades[-1].exit_fill.price)
            computed_btc = (last_exit - first_entry) / first_entry if first_entry > 0 else 0.0
        else:
            computed_btc = 0.0

    computed_basket = basket_return
    if computed_basket is None:
        if core_returns:
            computed_basket = sum(core_returns.values()) / len(core_returns)
        else:
            computed_basket = computed_btc

    return {
        "btc_return": computed_btc,
        "basket_return": computed_basket,
    }


def format_weekly_report(
    variant_results: dict[str, dict[str, Any]],
    eval_alpha: dict[str, Any],
    benchmark_metrics: dict[str, Any] | None = None,
) -> str:
    """Format tabular ASCII weekly performance report."""
    lines: list[str] = []
    lines.append("=" * 95)
    lines.append("           CRYPTO AI TRADER - QUANTITATIVE EXPERIMENT WEEKLY REPORT")
    lines.append("=" * 95)
    lines.append("")

    # Section 1: Multi-Portfolio Teardown
    lines.append("1. MULTI-PORTFOLIO PERFORMANCE TEARDOWN")
    lines.append("-" * 95)
    header = (
        f"{'Variant':<35} | {'Trades':>6} | {'WinRate':>7} | {'Net PnL':>10} | "
        f"{'Excess(BTC)':>11} | {'MaxDD':>7} | {'Sharpe':>7} | {'Sortino':>7}"
    )
    lines.append(header)
    lines.append("-" * 95)

    for var_key in ("variant_a", "variant_b", "variant_c", "variant_d", "variant_e"):
        var_label = VARIANT_NAMES.get(var_key, var_key)
        stats = variant_results.get(var_key, {})
        trades_n = stats.get("total_trades", 0)
        win_rate = stats.get("win_rate", 0.0)
        net_pnl = stats.get("net_pnl", 0.0)
        excess_btc = stats.get("excess_return_btc")
        max_dd_pct = stats.get("max_drawdown_pct", 0.0)
        sharpe = stats.get("sharpe_ratio")
        sortino = stats.get("sortino_ratio")

        sharpe_str = f"{sharpe:.2f}" if sharpe is not None and not math.isinf(sharpe) else "N/A"
        sortino_str = f"{sortino:.2f}" if sortino is not None and not math.isinf(sortino) else "N/A"
        pnl_str = f"${net_pnl:+,.2f}"
        excess_btc_str = f"{excess_btc:+.1%}" if excess_btc is not None else "N/A"

        lines.append(
            f"{var_label:<35} | {trades_n:>6} | {win_rate:>6.1%} | {pnl_str:>10} | "
            f"{excess_btc_str:>11} | {max_dd_pct:>6.1%} | {sharpe_str:>7} | {sortino_str:>7}"
        )

    lines.append("-" * 95)
    lines.append("")

    # Benchmarks
    if benchmark_metrics:
        b_btc = benchmark_metrics.get("btc_return")
        b_bsk = benchmark_metrics.get("basket_return")
        b_btc_str = f"{b_btc:+.2%}" if b_btc is not None else "N/A"
        b_bsk_str = f"{b_bsk:+.2%}" if b_bsk is not None else "N/A"
        lines.append("   BENCHMARKS:")
        lines.append(f"   - Buy-and-Hold BTC Return                  : {b_btc_str}")
        lines.append(f"   - Equal-Weight Core Basket (BTC, ETH, ...) : {b_bsk_str}")
        lines.append("")

    # Section 2: AI Gate Alpha Analysis
    lines.append("2. AI FUNDAMENTAL RISK GATEKEEPER ALPHA EVALUATION")
    lines.append("-" * 95)
    app_cnt = eval_alpha.get("approved_count", 0)
    rej_cnt = eval_alpha.get("rejected_count", 0)
    app_avg_g = eval_alpha.get("approved_avg_24h")
    rej_avg_g = eval_alpha.get("rejected_avg_24h")
    app_avg_n = eval_alpha.get("approved_net_avg_24h")
    rej_avg_n = eval_alpha.get("rejected_net_avg_24h")
    spread_g = eval_alpha.get("alpha_spread")
    spread_n = eval_alpha.get("net_alpha_spread")

    app_med_g = eval_alpha.get("approved_median_24h")
    rej_med_g = eval_alpha.get("rejected_median_24h")
    app_med_n = eval_alpha.get("approved_net_median_24h")
    rej_med_n = eval_alpha.get("rejected_net_median_24h")
    app_worst = eval_alpha.get("approved_worst_24h")
    rej_worst = eval_alpha.get("rejected_worst_24h")
    app_below5 = eval_alpha.get("approved_pct_below_neg_5")
    rej_below5 = eval_alpha.get("rejected_pct_below_neg_5")

    fb_rate = eval_alpha.get("fallback_rate", 0.0)
    fb_cnt = eval_alpha.get("fallback_count", 0)
    outage_windows = eval_alpha.get("outage_window_count", 0)

    app_avg_str = f"{app_avg_g:+.2%}" if app_avg_g is not None else "N/A"
    app_net_str = f"{app_avg_n:+.2%}" if app_avg_n is not None else "N/A"
    rej_avg_str = f"{rej_avg_g:+.2%}" if rej_avg_g is not None else "N/A"
    rej_net_str = f"{rej_avg_n:+.2%}" if rej_avg_n is not None else "N/A"
    app_med_str = f"{app_med_g:+.2%}" if app_med_g is not None else "N/A"
    app_med_net_str = f"{app_med_n:+.2%}" if app_med_n is not None else "N/A"
    rej_med_str = f"{rej_med_g:+.2%}" if rej_med_g is not None else "N/A"
    rej_med_net_str = f"{rej_med_n:+.2%}" if rej_med_n is not None else "N/A"
    app_worst_str = f"{app_worst:+.2%}" if app_worst is not None else "N/A"
    rej_worst_str = f"{rej_worst:+.2%}" if rej_worst is not None else "N/A"
    app_below5_str = f"{app_below5:.1%}" if app_below5 is not None else "N/A"
    rej_below5_str = f"{rej_below5:.1%}" if rej_below5 is not None else "N/A"
    spread_g_str = f"{spread_g:+.2%}" if spread_g is not None else "N/A"
    spread_n_str = f"{spread_n:+.2%}" if spread_n is not None else "N/A"
    fb_rate_str = f"{fb_rate:.1%}"

    lines.append(
        f"  AI Approved Signals (T+24h Evaluated) : {app_cnt:>4} signals | Avg Return: {app_avg_str} (Net: {app_net_str}) | "
        f"Median: {app_med_str} (Net: {app_med_net_str}) | Worst: {app_worst_str} | < -5%: {app_below5_str}"
    )
    lines.append(
        f"  AI Rejected Signals (T+24h Evaluated) : {rej_cnt:>4} signals | Avg Return: {rej_avg_str} (Net: {rej_net_str}) | "
        f"Median: {rej_med_str} (Net: {rej_med_net_str}) | Worst: {rej_worst_str} | < -5%: {rej_below5_str} (Dodged Bullets)"
    )
    lines.append(
        f"  Gatekeeper Alpha Spread (Appr - Rej) : {spread_g_str} (Net-of-cost: {spread_n_str})"
    )
    lines.append(
        f"  Gatekeeper Fallback / Outage Metrics : Fallback Rate: {fb_rate_str} ({fb_cnt} events) | Outage Window Count: {outage_windows}"
    )

    reg_cnt = eval_alpha.get("regime_blocked_count", 0)
    if reg_cnt > 0:
        reg_avg = eval_alpha.get("regime_blocked_avg_24h")
        reg_avg_str = f"{reg_avg:+.2%}" if reg_avg is not None else "N/A"
        lines.append(
            f"  Regime-Blocked Signals (T+24h Eval)   : {reg_cnt:>4} signals | Avg Return: {reg_avg_str}"
        )

    by_regime = eval_alpha.get("by_regime", {})
    up_data = by_regime.get("UPTREND", {})
    down_data = by_regime.get("DOWNTREND", {})
    has_regime_breakdown = (
        up_data.get("approved_count", 0) > 0
        or up_data.get("rejected_count", 0) > 0
        or down_data.get("approved_count", 0) > 0
        or down_data.get("rejected_count", 0) > 0
    )
    if has_regime_breakdown:
        lines.append("  Conditional Alpha by BTC Regime:")
        if up_data.get("approved_count", 0) > 0 or up_data.get("rejected_count", 0) > 0:
            up_sp_g = up_data.get("alpha_spread")
            up_sp_n = up_data.get("net_alpha_spread")
            up_g_str = f"{up_sp_g:+.2%}" if up_sp_g is not None else "N/A"
            up_n_str = f"{up_sp_n:+.2%}" if up_sp_n is not None else "N/A"
            lines.append(
                f"    - UPTREND Regime   : Appr: {up_data.get('approved_count', 0):>2} | Rej: {up_data.get('rejected_count', 0):>2} | "
                f"Spread: {up_g_str} (Net: {up_n_str})"
            )
        if down_data.get("approved_count", 0) > 0 or down_data.get("rejected_count", 0) > 0:
            down_sp_g = down_data.get("alpha_spread")
            down_sp_n = down_data.get("net_alpha_spread")
            down_g_str = f"{down_sp_g:+.2%}" if down_sp_g is not None else "N/A"
            down_n_str = f"{down_sp_n:+.2%}" if down_sp_n is not None else "N/A"
            lines.append(
                f"    - DOWNTREND Regime : Appr: {down_data.get('approved_count', 0):>2} | Rej: {down_data.get('rejected_count', 0):>2} | "
                f"Spread: {down_g_str} (Net: {down_n_str})"
            )

    lines.append("")
    lines.append(f"  Assessment: {eval_alpha.get('summary', '')}")
    lines.append("=" * 95)

    return "\n".join(lines)


def generate_weekly_report(
    db_path: str = DEFAULT_DB_PATH,
    executors: dict[str, PaperExecutor] | None = None,
    variant_trades: dict[str, Sequence[Trade]] | None = None,
    btc_return: float | None = None,
    basket_return: float | None = None,
) -> str:
    """Generate weekly performance report string comparing all variants and gate alpha."""
    benchmarks = compute_benchmark_metrics(
        variant_trades=variant_trades,
        executors=executors,
        btc_return=btc_return,
        basket_return=basket_return,
        db_path=db_path,
    )
    b_btc = benchmarks.get("btc_return")

    results: dict[str, dict[str, Any]] = {}

    for var_key in ("variant_a", "variant_b", "variant_c", "variant_d", "variant_e"):
        trades: Sequence[Trade] = ()
        if variant_trades and var_key in variant_trades:
            trades = variant_trades[var_key]
        elif executors and var_key in executors:
            trades = executors[var_key].trades

        results[var_key] = calculate_variant_metrics(trades, btc_return=b_btc)

    alpha_eval = query_fundamental_alpha_comparison(db_path)
    return format_weekly_report(results, alpha_eval, benchmark_metrics=benchmarks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Weekly Multi-Portfolio & AI Gate Alpha Report."
    )
    parser.add_argument(
        "--db-path", default=DEFAULT_DB_PATH, help="Path to evaluation SQLite database."
    )
    parser.add_argument(
        "--btc-return",
        type=float,
        default=None,
        help="Buy-and-Hold BTC return over reporting window.",
    )
    parser.add_argument(
        "--basket-return",
        type=float,
        default=None,
        help="Equal-weight basket return over reporting window.",
    )
    parser.add_argument("--output", default=None, help="Optional file path to write report to.")
    args = parser.parse_args()

    report = generate_weekly_report(
        db_path=args.db_path,
        btc_return=args.btc_return,
        basket_return=args.basket_return,
    )
    print(report)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info("Saved report to %s", args.output)


if __name__ == "__main__":
    main()

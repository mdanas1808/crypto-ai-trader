"""Weekly performance reporting and fundamental AI gate alpha teardown.

Outputs:
1. Multi-Portfolio Performance Teardown across all 5 variants:
   - Variant A: AI Gate + Regime Filter (Current)
   - Variant B: Regime Filter only (No AI Gate)
   - Variant C: AI Gate only (No Regime Filter)
   - Variant D: Baseline (No Gate, No Regime Filter)
   - Variant E: AI Gate + Regime Filter applied to Core Alts too
   Metrics compared: Total PnL, Max Drawdown, Sharpe Ratio, Sortino Ratio, Win Rate, Total Trades.

2. AI Fundamental Risk Gatekeeper Evaluation Alpha Analysis:
   - Compares the average T+24h return of signals the AI Approved vs signals the AI Rejected.
   - Computes alpha spread and decision accuracy.

Usage:
    .venv/bin/python scripts/weekly_report.py [--db-path fundamental_eval.db]
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import json
import logging
import math
import os
import sqlite3
from typing import Any, Sequence

from app.analysis.eval_logger import DEFAULT_DB_PATH
from app.backtest.models import Trade
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


def calculate_sharpe_ratio(
    returns: Sequence[float | Decimal],
    risk_free_rate: float = 0.0,
) -> float | None:
    """Calculate sample Sharpe ratio from returns."""
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
    return excess_mean / std if std > 0 else 0.0


def calculate_sortino_ratio(
    returns: Sequence[float | Decimal],
    risk_free_rate: float = 0.0,
) -> float | None:
    """Calculate Sortino ratio using downside standard deviation below risk-free rate."""
    if len(returns) < 2:
        return None
    float_returns = [float(r) for r in returns]
    n = len(float_returns)
    mean_ret = sum(float_returns) / n
    excess_mean = mean_ret - risk_free_rate
    downside_sq_sum = sum(
        min(0.0, r - risk_free_rate) ** 2 for r in float_returns
    )
    if downside_sq_sum <= 0.0:
        return None if excess_mean <= 0 else float("inf")
    downside_std = math.sqrt(downside_sq_sum / n)
    return excess_mean / downside_std if downside_std > 0 else 0.0


def calculate_variant_metrics(
    trades: Sequence[Trade],
    initial_cash: Decimal = Decimal("10000.00"),
) -> dict[str, Any]:
    """Calculate comprehensive performance metrics for a sequence of completed trades."""
    total_trades = len(trades)
    if total_trades == 0:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "net_pnl": 0.0,
            "net_pnl_pct": 0.0,
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

    # Calculate trade returns and running equity for drawdown
    trade_returns: list[float] = []
    equity = initial_cash
    peak = equity
    max_dd = Decimal("0")
    max_dd_pct = Decimal("0")

    for t in trades:
        cost = t.entry_fill.price * t.entry_fill.quantity
        if cost > Decimal("0"):
            ret = float(t.net_pnl / cost)
            trade_returns.append(ret)
        equity += t.net_pnl
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd
            max_dd_pct = dd / peak if peak > Decimal("0") else Decimal("0")

    sharpe = calculate_sharpe_ratio(trade_returns)
    sortino = calculate_sortino_ratio(trade_returns)

    return {
        "total_trades": total_trades,
        "winning_trades": len(winning),
        "losing_trades": len(losing),
        "win_rate": win_rate,
        "net_pnl": float(net_pnl),
        "net_pnl_pct": net_pnl_pct,
        "max_drawdown": float(max_dd),
        "max_drawdown_pct": float(max_dd_pct),
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
    }


def query_fundamental_alpha_comparison(db_path: str = DEFAULT_DB_PATH) -> dict[str, Any]:
    """Query fundamental evaluation records and compare T+24h returns of Approved vs Rejected signals."""
    if not os.path.exists(db_path):
        return {
            "approved_count": 0,
            "rejected_count": 0,
            "approved_avg_24h": None,
            "rejected_avg_24h": None,
            "alpha_spread": None,
            "summary": f"Database file not found: {db_path}",
        }

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    with conn:
        approved_rows = conn.execute(
            """
            SELECT return_24h FROM fundamental_evaluations
            WHERE verdict = 'APPROVE' AND return_24h IS NOT NULL
            """
        ).fetchall()

        rejected_rows = conn.execute(
            """
            SELECT return_24h FROM fundamental_evaluations
            WHERE verdict = 'REJECT' AND return_24h IS NOT NULL
            """
        ).fetchall()

    conn.close()

    app_returns = [float(r["return_24h"]) for r in approved_rows]
    rej_returns = [float(r["return_24h"]) for r in rejected_rows]

    app_count = len(app_returns)
    rej_count = len(rej_returns)

    app_avg = sum(app_returns) / app_count if app_count > 0 else None
    rej_avg = sum(rej_returns) / rej_count if rej_count > 0 else None

    alpha_spread = (app_avg - rej_avg) if (app_avg is not None and rej_avg is not None) else None

    if alpha_spread is not None:
        if alpha_spread > 0:
            summary = (
                f"POSITIVE ALPHA: AI Gate generated {alpha_spread:+.2%} alpha spread. "
                f"Approved trades averaged {app_avg:+.2%}, while Vetoed trades averaged {rej_avg:+.2%}."
            )
        else:
            summary = (
                f"NEGATIVE/NEUTRAL ALPHA: AI Gate spread is {alpha_spread:+.2%}. "
                f"Approved trades averaged {app_avg:+.2%}, while Vetoed trades averaged {rej_avg:+.2%}."
            )
    else:
        summary = "Insufficient completed T+24h return data to compute alpha spread."

    return {
        "approved_count": app_count,
        "rejected_count": rej_count,
        "approved_avg_24h": app_avg,
        "rejected_avg_24h": rej_avg,
        "alpha_spread": alpha_spread,
        "summary": summary,
    }


def format_weekly_report(
    variant_results: dict[str, dict[str, Any]],
    eval_alpha: dict[str, Any],
) -> str:
    """Format tabular ASCII weekly performance report."""
    lines: list[str] = []
    lines.append("=" * 88)
    lines.append("           CRYPTO AI TRADER - QUANTITATIVE EXPERIMENT WEEKLY REPORT")
    lines.append("=" * 88)
    lines.append("")

    # Section 1: Multi-Portfolio Teardown
    lines.append("1. MULTI-PORTFOLIO PERFORMANCE TEARDOWN")
    lines.append("-" * 88)
    header = (
        f"{'Variant':<35} | {'Trades':>6} | {'WinRate':>7} | {'Net PnL':>10} | "
        f"{'MaxDD':>7} | {'Sharpe':>7} | {'Sortino':>7}"
    )
    lines.append(header)
    lines.append("-" * 88)

    for var_key in ("variant_a", "variant_b", "variant_c", "variant_d", "variant_e"):
        var_label = VARIANT_NAMES.get(var_key, var_key)
        stats = variant_results.get(var_key, {})
        trades_n = stats.get("total_trades", 0)
        win_rate = stats.get("win_rate", 0.0)
        net_pnl = stats.get("net_pnl", 0.0)
        max_dd_pct = stats.get("max_drawdown_pct", 0.0)
        sharpe = stats.get("sharpe_ratio")
        sortino = stats.get("sortino_ratio")

        sharpe_str = f"{sharpe:.2f}" if sharpe is not None and not math.isinf(sharpe) else "N/A"
        sortino_str = f"{sortino:.2f}" if sortino is not None and not math.isinf(sortino) else "N/A"
        pnl_str = f"${net_pnl:+,.2f}"

        lines.append(
            f"{var_label:<35} | {trades_n:>6} | {win_rate:>6.1%} | {pnl_str:>10} | "
            f"{max_dd_pct:>6.1%} | {sharpe_str:>7} | {sortino_str:>7}"
        )

    lines.append("-" * 88)
    lines.append("")

    # Section 2: AI Gate Alpha Analysis
    lines.append("2. AI FUNDAMENTAL RISK GATEKEEPER ALPHA EVALUATION")
    lines.append("-" * 88)
    app_cnt = eval_alpha.get("approved_count", 0)
    rej_cnt = eval_alpha.get("rejected_count", 0)
    app_avg = eval_alpha.get("approved_avg_24h")
    rej_avg = eval_alpha.get("rejected_avg_24h")
    spread = eval_alpha.get("alpha_spread")

    app_avg_str = f"{app_avg:+.2%}" if app_avg is not None else "N/A"
    rej_avg_str = f"{rej_avg:+.2%}" if rej_avg is not None else "N/A"
    spread_str = f"{spread:+.2%}" if spread is not None else "N/A"

    lines.append(f"  AI Approved Signals (T+24h Evaluated) : {app_cnt:>4} signals | Avg Return: {app_avg_str}")
    lines.append(f"  AI Rejected Signals (T+24h Evaluated) : {rej_cnt:>4} signals | Avg Return: {rej_avg_str}")
    lines.append(f"  Gatekeeper Alpha Spread (Appr - Rej) : {spread_str}")
    lines.append("")
    lines.append(f"  Assessment: {eval_alpha.get('summary', '')}")
    lines.append("=" * 88)

    return "\n".join(lines)


def generate_weekly_report(
    db_path: str = DEFAULT_DB_PATH,
    executors: dict[str, PaperExecutor] | None = None,
    variant_trades: dict[str, Sequence[Trade]] | None = None,
) -> str:
    """Generate weekly performance report string comparing all variants and gate alpha."""
    results: dict[str, dict[str, Any]] = {}

    for var_key in ("variant_a", "variant_b", "variant_c", "variant_d", "variant_e"):
        trades: Sequence[Trade] = ()
        if variant_trades and var_key in variant_trades:
            trades = variant_trades[var_key]
        elif executors and var_key in executors:
            trades = executors[var_key].trades

        results[var_key] = calculate_variant_metrics(trades)

    alpha_eval = query_fundamental_alpha_comparison(db_path)
    return format_weekly_report(results, alpha_eval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Weekly Multi-Portfolio & AI Gate Alpha Report.")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help="Path to evaluation SQLite database.")
    parser.add_argument("--output", default=None, help="Optional file path to write report to.")
    args = parser.parse_args()

    report = generate_weekly_report(db_path=args.db_path)
    print(report)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info("Saved report to %s", args.output)


if __name__ == "__main__":
    main()

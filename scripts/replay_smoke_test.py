#!/usr/bin/env python3
"""Multi-week historical replay smoke-test script.

Loads 2-3 weeks (500 hours) of historical/synthetic hourly candles for all 10 symbols
(5 Core + 5 Satellite) through PaperSession.process_event across all 5 variants.
Stubs news fetcher and LLM calls to return immediate clean approvals/no-news to isolate
runtime crashes, uncovered code paths, division-by-zero errors, or unhandled exceptions.

Usage:
    .venv/bin/python scripts/replay_smoke_test.py
"""

from __future__ import annotations

import asyncio
import math
import os
import tempfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from app.analysis.eval_logger import FundamentalEvalLogger
from app.backtest.models import BacktestConfig, Trade
from app.market.constants import DEFAULT_DYNAMIC_SYMBOLS, FIXED_SYMBOLS
from app.market.models import Candle, CandleOpenObservation
from app.strategy.breakout import BreakoutSignal
from app.trading.paper import PaperExecutor
from app.trading.session import PaperSession

SYMBOLS: list[str] = list(FIXED_SYMBOLS) + list(DEFAULT_DYNAMIC_SYMBOLS)
TOTAL_HOURS = 500


class SmokeTestExplainer:
    """Fast synchronous/async stub for LLM gatekeeper."""

    async def evaluate_fundamentals(
        self, symbol: str, news_context: str, mode: str = "ENTRY"
    ) -> tuple[bool, str, dict[str, Any]]:
        if mode == "ENTRY":
            return (
                True,
                "Clean technical momentum with positive ecosystem growth.",
                {
                    "verdict": "APPROVE",
                    "risk_level": "LOW",
                    "confidence": 0.95,
                    "driving_news_id": f"news_cat_{symbol}",
                    "reason": "Clean technical momentum with positive ecosystem growth.",
                },
            )
        return (
            True,
            "No adverse catalyst detected; retain position under protective stops.",
            {
                "verdict": "HOLD",
                "risk_level": "LOW",
                "confidence": 0.90,
                "driving_news_id": f"news_cat_{symbol}",
                "reason": "No adverse catalyst detected; retain position under protective stops.",
            },
        )

    def explain_signal(self, sig: BreakoutSignal, context_data: Any) -> str:
        return f"Breakout Long on {sig.symbol} above rolling resistance."

    def explain_trade_exit(self, trade: Trade) -> str:
        return f"Exit filled on {trade.signal.symbol} via {trade.exit_reason}."


class ToggleBtcRegimeFilter:
    """Regime filter transitioning between UPTREND and DOWNTREND every 150 hours."""

    def __init__(self, toggle_hours: int = 150) -> None:
        self.toggle_hours = toggle_hours
        self.current_hour = 0

    def step(self, hour: int) -> None:
        self.current_hour = hour

    async def is_uptrend(self) -> bool:
        # First 150h UPTREND, next 150h DOWNTREND, next 150h UPTREND, etc.
        phase = (self.current_hour // self.toggle_hours) % 2
        return phase == 0

    def get_current_regime(self) -> str:
        phase = (self.current_hour // self.toggle_hours) % 2
        return "UPTREND" if phase == 0 else "DOWNTREND"


class RollingBreakoutStrategy:
    """Tracks 20-period price channels and generates BreakoutSignals."""

    def __init__(self) -> None:
        self.history: dict[str, list[Decimal]] = {s: [] for s in SYMBOLS}

    def on_candle(self, candle: Candle) -> list[BreakoutSignal]:
        sym = candle.symbol
        prices = self.history.setdefault(sym, [])
        prices.append(candle.close)
        if len(prices) > 40:
            prices.pop(0)

        if len(prices) < 20:
            return []

        # 20-period highest close excluding current
        prev_20_high = max(prices[-21:-1])
        if candle.close > prev_20_high:
            atr = max(Decimal("1.0"), (candle.high - candle.low))
            sig = BreakoutSignal(
                exchange="binance_spot",
                symbol=sym,
                side="long",
                decision_time=candle.open_time + timedelta(hours=1),
                signal_open_time=candle.open_time,
                atr_14=float(atr),
            )
            return [sig]
        return []


def generate_candle_series(
    symbol: str,
    base_price: float,
    start_time: datetime,
    hours: int,
) -> list[Candle]:
    """Generate contiguous, validated 1h candles with realistic oscillations."""
    candles: list[Candle] = []
    price = base_price
    cur_time = start_time

    for h in range(hours):
        # Oscillate price with occasional breakout ramps
        trend_bias = math.sin(h / 30.0) * 0.005 + (0.015 if (h % 70) in range(50, 60) else -0.005)
        step_pct = trend_bias + (math.sin(h * 1.7) * 0.008)
        new_close = max(1.0, price * (1.0 + step_pct))

        c_open = Decimal(f"{price:.4f}")
        c_close = Decimal(f"{new_close:.4f}")
        high_diff = Decimal(f"{abs(new_close - price) * 0.5 + price * 0.005:.4f}")
        low_diff = Decimal(f"{abs(new_close - price) * 0.5 + price * 0.004:.4f}")

        c_high = max(c_open, c_close) + high_diff
        c_low = max(Decimal("0.0001"), min(c_open, c_close) - low_diff)
        c_vol = Decimal("1000.0")

        open_time = cur_time

        c = Candle(
            symbol=symbol,
            exchange="binance_spot",
            interval="1h",
            open_time=open_time,
            open=c_open,
            high=c_high,
            low=c_low,
            close=c_close,
            volume=c_vol,
        )
        candles.append(c)
        price = new_close
        cur_time += timedelta(hours=1)

    return candles


async def run_smoke_test() -> None:
    print(f"=== Starting 500-Hour Historical Replay Smoke Test ({len(SYMBOLS)} symbols) ===")

    # 1. Setup temporary directory for eval db and history
    tmp_dir = tempfile.mkdtemp(prefix="smoke_replay_")
    db_path = os.path.join(tmp_dir, "smoke_eval.db")
    history_file = os.path.join(tmp_dir, "smoke_history.json")
    eval_logger = FundamentalEvalLogger(db_path=db_path)

    # 2. Stub news fetcher in session module
    async def stub_fetch_news(symbol: str, limit: int = 10) -> str:
        return f"<news_data>[1h ago] Routine commentary on {symbol}</news_data>"

    import app.trading.session as session_module

    session_module.fetch_recent_news = stub_fetch_news

    # 3. Initialize components
    config = BacktestConfig()
    executor = PaperExecutor(config)
    explainer = SmokeTestExplainer()
    regime_filter = ToggleBtcRegimeFilter(toggle_hours=120)
    strategy = RollingBreakoutStrategy()

    class NullStream:
        async def events(self):
            if False:
                yield None

    session = PaperSession(
        stream=NullStream(),  # type: ignore
        executor=executor,
        strategy=strategy.on_candle,
        explainer=explainer,  # type: ignore
        btc_regime_filter=regime_filter,  # type: ignore
        eval_logger=eval_logger,
        history_file=history_file,
    )

    # 4. Generate candle sequences for all 10 symbols
    start_time = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    base_prices = {
        "BTCUSDT": 42000.0,
        "ETHUSDT": 2200.0,
        "SOLUSDT": 95.0,
        "BNBUSDT": 305.0,
        "ZECUSDT": 28.0,
        "TAOUSDT": 250.0,
        "FETUSDT": 0.65,
        "RENDERUSDT": 4.5,
        "NEARUSDT": 3.2,
        "INJUSDT": 35.0,
    }

    print("Generating 500 hourly candles for 10 symbols...")
    all_symbol_candles: dict[str, list[Candle]] = {}
    for sym in SYMBOLS:
        all_symbol_candles[sym] = generate_candle_series(
            sym, base_prices[sym], start_time, TOTAL_HOURS
        )

    # 5. Step-by-step chronological replay
    print("Replaying hourly events through PaperSession.process_event...")
    total_events = 0

    for h in range(TOTAL_HOURS):
        regime_filter.step(h)

        # A. Process closed candle of hour h for all symbols
        for sym in SYMBOLS:
            c = all_symbol_candles[sym][h]
            await session.process_event(c)
            total_events += 1

        # B. Process next open observation at h+1 (or close price of current)
        for sym in SYMBOLS:
            c = all_symbol_candles[sym][h]
            next_open = c.open_time + timedelta(hours=1)
            obs = CandleOpenObservation(
                symbol=sym,
                interval="1h",
                open_time=next_open,
                observed_at=next_open,
                open=c.close,
                exchange="binance_spot",
            )
            await session.process_event(obs)
            total_events += 1

        # C. Periodic 24-hour review check
        if h > 0 and h % 24 == 0:
            await session._review_open_positions(cycle_id=f"cycle_{h}")

    print(f"Replay complete! Processed {total_events} events across {TOTAL_HOURS} hours.")

    # 6. Verify variant states and teardown metrics
    print("\n" + "=" * 80)
    print("VARIANT PERFORMANCE TEARDOWN ACROSS REPLAY")
    print("=" * 80)
    print(
        f"{'Variant':<40} | {'Trades':>6} | {'Positions':>9} | {'Final Cash':>12} | {'Net PnL':>10}"
    )
    print("-" * 80)

    for var_key, exec_inst in session.executors.items():
        n_trades = len(exec_inst.trades)
        n_open = len(exec_inst.positions)
        cash = exec_inst.cash
        net_pnl = sum((t.net_pnl for t in exec_inst.trades), Decimal("0"))
        print(
            f"{var_key:<40} | {n_trades:>6} | {n_open:>9} | {float(cash):>12.2f} | {float(net_pnl):>+10.2f}"
        )

        # Sanity assertions
        assert not math.isnan(float(cash)), f"NaN cash encountered in {var_key}"
        assert not math.isinf(float(cash)), f"Inf cash encountered in {var_key}"

    # Verify evaluations in SQLite
    import sqlite3

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*), COUNT(DISTINCT verdict) FROM fundamental_evaluations")
    eval_count, verdict_types = cur.fetchone()
    conn.close()

    print("-" * 80)
    print(f"Total AI Gate Evaluations Logged: {eval_count} (Unique verdicts: {verdict_types})")

    # Cleanup temporary files
    try:
        if os.path.exists(db_path):
            os.remove(db_path)
        if os.path.exists(history_file):
            os.remove(history_file)
        os.rmdir(tmp_dir)
    except Exception:
        pass

    print("\n>>> SMOKE TEST PASSED: 500 hours replayed with 0 exceptions. <<<")


if __name__ == "__main__":
    asyncio.run(run_smoke_test())

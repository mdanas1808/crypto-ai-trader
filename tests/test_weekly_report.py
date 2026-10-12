import sqlite3
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from app.backtest.models import Fill, Trade
from app.strategy.breakout import BreakoutSignal
from scripts.weekly_report import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_variant_metrics,
    compute_benchmark_metrics,
    format_weekly_report,
    generate_weekly_report,
    query_fundamental_alpha_comparison,
)

START = datetime(2024, 1, 2, tzinfo=timezone.utc)


def make_trade(symbol: str, pnl: float, entry_price: float = 100.0) -> Trade:
    sig = BreakoutSignal(
        exchange="binance_spot",
        symbol=symbol,
        side="long",
        decision_time=START,
        signal_open_time=START,
        atr_14=1.0,
    )
    fill_entry = Fill(
        symbol=symbol,
        exchange="binance_spot",
        side="buy",
        timestamp=START,
        price=Decimal(str(entry_price)),
        quantity=Decimal("1.0"),
        fee=Decimal("0.1"),
        reference_price=Decimal(str(entry_price)),
        reason="signal_open",
        observed_at=START,
    )
    exit_price = entry_price + pnl
    fill_exit = Fill(
        symbol=symbol,
        exchange="binance_spot",
        side="sell",
        timestamp=START,
        price=Decimal(str(exit_price)),
        quantity=Decimal("1.0"),
        fee=Decimal("0.1"),
        reference_price=Decimal(str(exit_price)),
        reason="take_profit" if pnl > 0 else "stop_loss",
    )
    return Trade(
        signal=sig,
        quantity=Decimal("1.0"),
        entry_fill=fill_entry,
        exit_fill=fill_exit,
        stop_price=Decimal("90.0"),
        target_price=Decimal("120.0"),
        exit_reason="take_profit" if pnl > 0 else "stop_loss",
        gross_pnl=Decimal(str(pnl)),
        net_pnl=Decimal(str(pnl)),
    )


def test_calculate_sharpe_and_sortino_ratio():
    returns = [0.05, 0.02, -0.01, 0.04, 0.03]
    sharpe = calculate_sharpe_ratio(returns)
    assert sharpe is not None
    assert sharpe > 0.0

    sortino = calculate_sortino_ratio(returns)
    assert sortino is not None
    assert sortino > 0.0

    # Less than 2 returns
    assert calculate_sharpe_ratio([0.05]) is None
    assert calculate_sortino_ratio([0.05]) is None

    # Zero variance returns
    assert calculate_sharpe_ratio([0.05, 0.05]) == 0.0


def test_calculate_variant_metrics():
    trades = [
        make_trade("BTCUSDT", 200.0),
        make_trade("ETHUSDT", -50.0),
        make_trade("SOLUSDT", 150.0),
    ]

    metrics = calculate_variant_metrics(trades, initial_cash=Decimal("10000.00"))
    assert metrics["total_trades"] == 3
    assert metrics["winning_trades"] == 2
    assert metrics["losing_trades"] == 1
    assert metrics["win_rate"] == pytest.approx(2 / 3)
    assert metrics["net_pnl"] == pytest.approx(300.0)
    assert metrics["net_pnl_pct"] == pytest.approx(0.03)
    assert metrics["max_drawdown"] == pytest.approx(50.0)
    assert metrics["sharpe_ratio"] is not None
    assert metrics["sortino_ratio"] is not None


def test_query_fundamental_alpha_comparison(tmp_path):
    db_path = str(tmp_path / "test_weekly_eval.db")
    conn = sqlite3.connect(db_path)
    with conn:
        conn.execute(
            """
            CREATE TABLE fundamental_evaluations (
                id INTEGER PRIMARY KEY,
                verdict TEXT,
                return_24h REAL
            );
            """
        )
        # Approved signals: +6%, +4% (avg +5%)
        conn.execute(
            "INSERT INTO fundamental_evaluations (verdict, return_24h) VALUES ('APPROVE', 0.06)"
        )
        conn.execute(
            "INSERT INTO fundamental_evaluations (verdict, return_24h) VALUES ('APPROVE', 0.04)"
        )
        # Rejected signals: -10%, -6% (avg -8%)
        conn.execute(
            "INSERT INTO fundamental_evaluations (verdict, return_24h) VALUES ('REJECT', -0.10)"
        )
        conn.execute(
            "INSERT INTO fundamental_evaluations (verdict, return_24h) VALUES ('REJECT', -0.06)"
        )
    conn.close()

    alpha_res = query_fundamental_alpha_comparison(db_path)
    assert alpha_res["approved_count"] == 2
    assert alpha_res["rejected_count"] == 2
    assert alpha_res["approved_avg_24h"] == pytest.approx(0.05)
    assert alpha_res["rejected_avg_24h"] == pytest.approx(-0.08)
    assert alpha_res["alpha_spread"] == pytest.approx(0.13)
    assert "POSITIVE ALPHA" in alpha_res["summary"]


def test_generate_weekly_report_output(tmp_path):
    db_path = str(tmp_path / "test_weekly_eval.db")
    conn = sqlite3.connect(db_path)
    with conn:
        conn.execute(
            """
            CREATE TABLE fundamental_evaluations (
                id INTEGER PRIMARY KEY,
                verdict TEXT,
                return_24h REAL
            );
            """
        )
        conn.execute(
            "INSERT INTO fundamental_evaluations (verdict, return_24h) VALUES ('APPROVE', 0.05)"
        )
        conn.execute(
            "INSERT INTO fundamental_evaluations (verdict, return_24h) VALUES ('REJECT', -0.05)"
        )
    conn.close()

    variant_trades = {
        "variant_a": [make_trade("BTCUSDT", 100.0)],
        "variant_b": [make_trade("BTCUSDT", 80.0)],
        "variant_c": [make_trade("BTCUSDT", 90.0)],
        "variant_d": [make_trade("BTCUSDT", 50.0)],
        "variant_e": [make_trade("BTCUSDT", 110.0)],
    }

    report = generate_weekly_report(db_path=db_path, variant_trades=variant_trades)
    assert "QUANTITATIVE EXPERIMENT WEEKLY REPORT" in report
    assert "Variant A" in report
    assert "Variant B" in report
    assert "Variant C" in report
    assert "Variant D" in report
    assert "Variant E" in report
    assert "AI Approved Signals" in report
    assert "AI Rejected Signals" in report
    assert "Gatekeeper Alpha Spread" in report
    assert "+10.00%" in report


def test_query_fundamental_alpha_robust_stats_and_deduplication(tmp_path):
    db_path = str(tmp_path / "test_robust_eval.db")
    conn = sqlite3.connect(db_path)
    with conn:
        conn.execute(
            """
            CREATE TABLE fundamental_evaluations (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                verdict TEXT,
                driving_news_id TEXT,
                reason TEXT,
                return_24h REAL
            );
            """
        )
        # 3 Approved signals: +4%, +6%, +8% -> median: +6%, worst: +4%, pct < -5%: 0%
        conn.execute(
            "INSERT INTO fundamental_evaluations (symbol, verdict, driving_news_id, reason, return_24h) VALUES ('FETUSDT', 'APPROVE', 'news_app_1', 'Strong dev', 0.04)"
        )
        conn.execute(
            "INSERT INTO fundamental_evaluations (symbol, verdict, driving_news_id, reason, return_24h) VALUES ('NEARUSDT', 'APPROVE', 'news_app_2', 'Mainnet ok', 0.06)"
        )
        conn.execute(
            "INSERT INTO fundamental_evaluations (symbol, verdict, driving_news_id, reason, return_24h) VALUES ('SOLUSDT', 'APPROVE', 'news_app_3', 'Ecosystem up', 0.08)"
        )

        # Rejected signals with duplicate (symbol, catalyst):
        # 1. FETUSDT + news_exploit_1 -> -8%
        conn.execute(
            "INSERT INTO fundamental_evaluations (symbol, verdict, driving_news_id, reason, return_24h) VALUES ('FETUSDT', 'REJECT', 'news_exploit_1', 'Bridge exploit', -0.08)"
        )
        # 2. FETUSDT + news_exploit_1 (duplicate review cycle) -> should be deduplicated!
        conn.execute(
            "INSERT INTO fundamental_evaluations (symbol, verdict, driving_news_id, reason, return_24h) VALUES ('FETUSDT', 'REJECT', 'news_exploit_1', 'Bridge exploit duplicate', -0.08)"
        )
        # 3. FETUSDT + news_exploit_2 -> -6% (< -5%)
        conn.execute(
            "INSERT INTO fundamental_evaluations (symbol, verdict, driving_news_id, reason, return_24h) VALUES ('FETUSDT', 'REJECT', 'news_exploit_2', 'Key compromised', -0.06)"
        )
        # 4. NEARUSDT + news_exploit_1 -> -4% (different symbol, kept)
        conn.execute(
            "INSERT INTO fundamental_evaluations (symbol, verdict, driving_news_id, reason, return_24h) VALUES ('NEARUSDT', 'REJECT', 'news_exploit_1', 'Bridge exploit', -0.04)"
        )
    conn.close()

    res = query_fundamental_alpha_comparison(db_path)
    # Deduplicated rejected count: 3 (not 4)
    assert res["rejected_count"] == 3
    assert res["approved_count"] == 3
    assert res["approved_median_24h"] == pytest.approx(0.06)
    assert res["approved_worst_24h"] == pytest.approx(0.04)
    assert res["approved_pct_below_neg_5"] == pytest.approx(0.0)

    # Rejections: -0.08, -0.06, -0.04 -> median: -0.06, worst: -0.08
    assert res["rejected_median_24h"] == pytest.approx(-0.06)
    assert res["rejected_worst_24h"] == pytest.approx(-0.08)
    # Outcomes below -5%: -0.08 and -0.06 (2 out of 3 = 66.7%)
    assert res["rejected_pct_below_neg_5"] == pytest.approx(2 / 3)


def test_calculate_variant_metrics_daily_mtm_and_excess_btc():
    # Make trades on distinct dates
    d1 = START
    d2 = START + timedelta(days=1)
    d3 = START + timedelta(days=2)

    def make_dated_trade(symbol: str, pnl: float, date: datetime) -> Trade:
        sig = BreakoutSignal(
            exchange="binance_spot",
            symbol=symbol,
            side="long",
            decision_time=date,
            signal_open_time=date,
            atr_14=1.0,
        )
        fill_entry = Fill(
            symbol=symbol,
            exchange="binance_spot",
            side="buy",
            timestamp=date,
            price=Decimal("100"),
            quantity=Decimal("1.0"),
            fee=Decimal("0.1"),
            reference_price=Decimal("100"),
            reason="signal_open",
            observed_at=date,
        )
        fill_exit = Fill(
            symbol=symbol,
            exchange="binance_spot",
            side="sell",
            timestamp=date,
            price=Decimal(str(100 + pnl)),
            quantity=Decimal("1.0"),
            fee=Decimal("0.1"),
            reference_price=Decimal(str(100 + pnl)),
            reason="take_profit" if pnl > 0 else "stop_loss",
        )
        return Trade(
            signal=sig,
            quantity=Decimal("1.0"),
            entry_fill=fill_entry,
            exit_fill=fill_exit,
            stop_price=Decimal("90"),
            target_price=Decimal("120"),
            exit_reason="take_profit" if pnl > 0 else "stop_loss",
            gross_pnl=Decimal(str(pnl)),
            net_pnl=Decimal(str(pnl)),
        )

    trades = [
        make_dated_trade("BTCUSDT", 200.0, d1),
        make_dated_trade("ETHUSDT", -50.0, d2),
        make_dated_trade("SOLUSDT", 100.0, d3),
    ]

    # BTC return benchmark: 10%
    metrics = calculate_variant_metrics(trades, initial_cash=Decimal("10000.00"), btc_return=0.10)
    # Net PnL is 250 -> 2.5% return
    assert metrics["net_pnl"] == pytest.approx(250.0)
    assert metrics["net_pnl_pct"] == pytest.approx(0.025)
    # Excess return vs BTC: 2.5% - 10% = -7.5%
    assert metrics["excess_return_btc"] == pytest.approx(-0.075)
    # Annualized Sharpe & Sortino should be non-null numbers
    assert metrics["sharpe_ratio"] is not None
    assert metrics["sortino_ratio"] is not None


def test_weekly_report_with_benchmarks_and_robust_reporting(tmp_path):
    db_path = str(tmp_path / "test_report_bench.db")
    conn = sqlite3.connect(db_path)
    with conn:
        conn.execute(
            """
            CREATE TABLE fundamental_evaluations (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                verdict TEXT,
                driving_news_id TEXT,
                reason TEXT,
                return_24h REAL
            );
            """
        )
        conn.execute(
            "INSERT INTO fundamental_evaluations (symbol, verdict, driving_news_id, reason, return_24h) VALUES ('FETUSDT', 'APPROVE', 'news1', 'Good', 0.05)"
        )
        conn.execute(
            "INSERT INTO fundamental_evaluations (symbol, verdict, driving_news_id, reason, return_24h) VALUES ('FETUSDT', 'REJECT', 'news2', 'Bad', -0.08)"
        )
    conn.close()

    variant_trades = {
        "variant_a": [make_trade("BTCUSDT", 100.0)],
        "variant_b": [make_trade("BTCUSDT", 80.0)],
        "variant_c": [make_trade("BTCUSDT", 90.0)],
        "variant_d": [make_trade("BTCUSDT", 50.0)],
        "variant_e": [make_trade("BTCUSDT", 110.0)],
    }

    report = generate_weekly_report(
        db_path=db_path,
        variant_trades=variant_trades,
        btc_return=0.04,
        basket_return=0.03,
    )
    assert "BENCHMARKS:" in report
    assert "Buy-and-Hold BTC Return                  : +4.00%" in report
    assert "Equal-Weight Core Basket (BTC, ETH, ...) : +3.00%" in report
    assert "Excess(BTC)" in report
    assert "Dodged Bullets" in report
    assert "Worst:" in report
    assert "Median:" in report


def test_weekly_report_excludes_fallback_rejections_and_reports_outage_metrics(tmp_path):
    db_path = str(tmp_path / "test_fallback_report.db")
    conn = sqlite3.connect(db_path)
    with conn:
        conn.execute(
            """
            CREATE TABLE fundamental_evaluations (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                verdict TEXT,
                is_fallback INTEGER,
                driving_news_id TEXT,
                reason TEXT,
                return_24h REAL,
                timestamp TEXT
            );
            """
        )
        t1 = "2026-10-10T10:00:00+00:00"
        t2 = "2026-10-10T12:00:00+00:00"
        t3 = "2026-10-10T14:00:00+00:00"
        conn.execute(
            "INSERT INTO fundamental_evaluations (symbol, verdict, is_fallback, driving_news_id, reason, return_24h, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("FETUSDT", "APPROVE", 0, "n1", "Good", 0.10, t1),
        )
        conn.execute(
            "INSERT INTO fundamental_evaluations (symbol, verdict, is_fallback, driving_news_id, reason, return_24h, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("FETUSDT", "REJECT", 0, "n2", "Bad", -0.05, t2),
        )
        conn.execute(
            "INSERT INTO fundamental_evaluations (symbol, verdict, is_fallback, driving_news_id, reason, return_24h, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("FETUSDT", "FALLBACK_REJECT", 1, "n3", "News fetch error fallback", 0.20, t3),
        )
    conn.close()

    res = query_fundamental_alpha_comparison(db_path)
    assert res["approved_count"] == 1
    assert res["rejected_count"] == 1
    assert res["approved_avg_24h"] == pytest.approx(0.10)
    assert res["rejected_avg_24h"] == pytest.approx(-0.05)
    assert res["alpha_spread"] == pytest.approx(0.15)
    assert res["fallback_count"] == 1
    assert res["fallback_rate"] == pytest.approx(1 / 3)
    assert res["outage_window_count"] == 1

    report = format_weekly_report(
        variant_results={},
        eval_alpha=res,
    )
    assert "Fallback Rate: 33.3% (1 events) | Outage Window Count: 1" in report
    assert "Gatekeeper Alpha Spread (Appr - Rej) : +15.00%" in report


def test_weekly_report_overlap_adjustment_and_net_of_cost(tmp_path):
    db_path = str(tmp_path / "test_overlap_net.db")
    conn = sqlite3.connect(db_path)
    with conn:
        conn.execute(
            """
            CREATE TABLE fundamental_evaluations (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                verdict TEXT,
                is_fallback INTEGER,
                driving_news_id TEXT,
                reason TEXT,
                return_24h REAL,
                net_return_24h REAL,
                timestamp TEXT
            );
            """
        )
        t1 = "2026-10-10T00:00:00+00:00"
        t2 = "2026-10-10T02:00:00+00:00"
        t3 = "2026-10-11T05:00:00+00:00"
        conn.execute(
            "INSERT INTO fundamental_evaluations (symbol, verdict, is_fallback, driving_news_id, reason, return_24h, net_return_24h, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("FETUSDT", "APPROVE", 0, "n1", "ok1", 0.05, 0.044, t1),
        )
        conn.execute(
            "INSERT INTO fundamental_evaluations (symbol, verdict, is_fallback, driving_news_id, reason, return_24h, net_return_24h, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("FETUSDT", "APPROVE", 0, "n2", "ok2", 0.15, 0.144, t2),
        )
        conn.execute(
            "INSERT INTO fundamental_evaluations (symbol, verdict, is_fallback, driving_news_id, reason, return_24h, net_return_24h, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("FETUSDT", "APPROVE", 0, "n3", "ok3", 0.08, 0.074, t3),
        )
        conn.execute(
            "INSERT INTO fundamental_evaluations (symbol, verdict, is_fallback, driving_news_id, reason, return_24h, net_return_24h, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("FETUSDT", "REJECT", 0, "n4", "bad", 0.004, -0.002, t1),
        )
    conn.close()

    res = query_fundamental_alpha_comparison(db_path)
    assert res["approved_count"] == 2
    assert res["approved_avg_24h"] == pytest.approx((0.05 + 0.08) / 2)
    assert res["approved_net_avg_24h"] == pytest.approx((0.044 + 0.074) / 2)
    assert res["rejected_count"] == 1
    assert res["rejected_avg_24h"] == pytest.approx(0.004)
    assert res["rejected_net_avg_24h"] == pytest.approx(-0.002)


def test_weekly_report_equal_weight_core_basket(tmp_path):
    db_path = str(tmp_path / "test_core_basket.db")
    conn = sqlite3.connect(db_path)
    with conn:
        conn.execute(
            """
            CREATE TABLE fundamental_evaluations (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                verdict TEXT,
                return_24h REAL
            );
            """
        )
    conn.close()

    d1 = START
    d2 = START + timedelta(hours=2)

    def make_priced_trade(symbol: str, p1: float, p2: float) -> Trade:
        sig = BreakoutSignal(
            exchange="binance_spot",
            symbol=symbol,
            side="long",
            decision_time=d1,
            signal_open_time=d1,
            atr_14=1.0,
        )
        fill_entry = Fill(
            symbol=symbol,
            exchange="binance_spot",
            side="buy",
            timestamp=d1,
            price=Decimal(str(p1)),
            quantity=Decimal("1.0"),
            fee=Decimal("0.1"),
            reference_price=Decimal(str(p1)),
            reason="signal_open",
            observed_at=d1,
        )
        fill_exit = Fill(
            symbol=symbol,
            exchange="binance_spot",
            side="sell",
            timestamp=d2,
            price=Decimal(str(p2)),
            quantity=Decimal("1.0"),
            fee=Decimal("0.1"),
            reference_price=Decimal(str(p2)),
            reason="take_profit",
        )
        pnl = p2 - p1
        return Trade(
            signal=sig,
            quantity=Decimal("1.0"),
            entry_fill=fill_entry,
            exit_fill=fill_exit,
            stop_price=Decimal("90"),
            target_price=Decimal("120"),
            exit_reason="take_profit",
            gross_pnl=Decimal(str(pnl)),
            net_pnl=Decimal(str(pnl)),
        )

    trades_a = [
        make_priced_trade("BTCUSDT", 100.0, 110.0),
        make_priced_trade("ETHUSDT", 100.0, 106.0),
        make_priced_trade("FETUSDT", 10.0, 20.0),
    ]

    bench = compute_benchmark_metrics(variant_trades={"variant_a": trades_a})
    assert bench["btc_return"] == pytest.approx(0.10)
    assert bench["basket_return"] == pytest.approx(0.08)


def test_weekly_report_regime_cross_tabulation(tmp_path):
    db_path = str(tmp_path / "test_cross_tab.db")
    conn = sqlite3.connect(db_path)
    with conn:
        conn.execute(
            """
            CREATE TABLE fundamental_evaluations (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                verdict TEXT,
                is_fallback INTEGER DEFAULT 0,
                regime_state TEXT DEFAULT '',
                regime_blocked INTEGER DEFAULT 0,
                driving_news_id TEXT,
                reason TEXT,
                return_24h REAL,
                timestamp TEXT
            );
            """
        )
        # UPTREND signals
        conn.execute(
            "INSERT INTO fundamental_evaluations (symbol, verdict, regime_state, regime_blocked, return_24h, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
            ("FETUSDT", "APPROVE", "UPTREND", 0, 0.10, "2026-10-10T00:00:00+00:00"),
        )
        conn.execute(
            "INSERT INTO fundamental_evaluations (symbol, verdict, regime_state, regime_blocked, return_24h, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
            ("NEARUSDT", "REJECT", "UPTREND", 0, -0.05, "2026-10-10T00:00:00+00:00"),
        )
        # DOWNTREND signals (blocked by regime, but verdict remains gate's verdict!)
        conn.execute(
            "INSERT INTO fundamental_evaluations (symbol, verdict, regime_state, regime_blocked, return_24h, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
            ("TAOUSDT", "APPROVE", "DOWNTREND", 1, 0.02, "2026-10-10T00:00:00+00:00"),
        )
        conn.execute(
            "INSERT INTO fundamental_evaluations (symbol, verdict, regime_state, regime_blocked, return_24h, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
            ("INJUSDT", "REJECT", "DOWNTREND", 1, -0.12, "2026-10-10T00:00:00+00:00"),
        )
    conn.close()

    res = query_fundamental_alpha_comparison(db_path)
    # Overall: 2 approved (0.10, 0.02 -> avg 0.06), 2 rejected (-0.05, -0.12 -> avg -0.085)
    assert res["approved_count"] == 2
    assert res["rejected_count"] == 2
    assert res["alpha_spread"] == pytest.approx(0.06 - (-0.085))

    # Cross-tabulation:
    assert "by_regime" in res
    up = res["by_regime"]["UPTREND"]
    down = res["by_regime"]["DOWNTREND"]
    assert up["approved_count"] == 1
    assert up["rejected_count"] == 1
    assert up["alpha_spread"] == pytest.approx(0.10 - (-0.05))

    assert down["approved_count"] == 1
    assert down["rejected_count"] == 1
    assert down["alpha_spread"] == pytest.approx(0.02 - (-0.12))

    assert res["regime_blocked_count"] == 2

    # Formatting verification
    report = format_weekly_report(variant_results={}, eval_alpha=res)
    assert "Conditional Alpha by BTC Regime:" in report
    assert "UPTREND Regime" in report
    assert "DOWNTREND Regime" in report
    assert "Regime-Blocked Signals" in report


def test_filter_overlapping_24h_symmetric():
    from scripts.weekly_report import _filter_overlapping_24h

    # Create synthetic rows: row with symbol, timestamp, verdict
    def make_mock_row(symbol: str, ts_str: str, verdict: str) -> dict:
        return {"symbol": symbol, "timestamp": ts_str, "verdict": verdict}

    app_rows = [
        make_mock_row("FETUSDT", "2026-10-01T00:00:00+00:00", "APPROVE"),
        make_mock_row("FETUSDT", "2026-10-01T06:00:00+00:00", "APPROVE"),  # overlaps (<24h)
        make_mock_row("FETUSDT", "2026-10-02T01:00:00+00:00", "APPROVE"),  # accepted (>24h)
    ]
    rej_rows = [
        make_mock_row("NEARUSDT", "2026-10-01T00:00:00+00:00", "REJECT"),
        make_mock_row("NEARUSDT", "2026-10-01T06:00:00+00:00", "REJECT"),  # overlaps (<24h)
        make_mock_row("NEARUSDT", "2026-10-02T01:00:00+00:00", "REJECT"),  # accepted (>24h)
    ]

    filtered_app = _filter_overlapping_24h(app_rows)
    filtered_rej = _filter_overlapping_24h(rej_rows)

    assert len(filtered_app) == 2
    assert len(filtered_rej) == 2
    assert filtered_app[0]["timestamp"] == "2026-10-01T00:00:00+00:00"
    assert filtered_app[1]["timestamp"] == "2026-10-02T01:00:00+00:00"
    assert filtered_rej[0]["timestamp"] == "2026-10-01T00:00:00+00:00"
    assert filtered_rej[1]["timestamp"] == "2026-10-02T01:00:00+00:00"

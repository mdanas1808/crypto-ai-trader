from datetime import datetime, timezone
from decimal import Decimal
import sqlite3
import pytest

from app.backtest.models import Fill, Trade
from app.strategy.breakout import BreakoutSignal
from scripts.weekly_report import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_variant_metrics,
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
        conn.execute("INSERT INTO fundamental_evaluations (verdict, return_24h) VALUES ('APPROVE', 0.06)")
        conn.execute("INSERT INTO fundamental_evaluations (verdict, return_24h) VALUES ('APPROVE', 0.04)")
        # Rejected signals: -10%, -6% (avg -8%)
        conn.execute("INSERT INTO fundamental_evaluations (verdict, return_24h) VALUES ('REJECT', -0.10)")
        conn.execute("INSERT INTO fundamental_evaluations (verdict, return_24h) VALUES ('REJECT', -0.06)")
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
        conn.execute("INSERT INTO fundamental_evaluations (verdict, return_24h) VALUES ('APPROVE', 0.05)")
        conn.execute("INSERT INTO fundamental_evaluations (verdict, return_24h) VALUES ('REJECT', -0.05)")
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

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from app.backtest.analysis import analyze_backtest
from app.backtest.models import BacktestResult, EquityPoint, Fill, Trade
from app.strategy.breakout import BreakoutSignal

UTC = timezone.utc
T0 = datetime(2025, 1, 1, tzinfo=UTC)
D = Decimal


def make_trade(net, *, reason="end_of_backtest", entry_fee="1", exit_fee="2"):
    net = D(net)
    entry = Fill("BTCUSDT", "binance_spot", "buy", T0, D("100"), D("100"), D("1"), D(entry_fee), "entry")
    exit_fill = Fill("BTCUSDT", "binance_spot", "sell", T0 + timedelta(hours=1), D("100"), D("100"), D("1"), D(exit_fee), reason)
    signal = BreakoutSignal("binance_spot", "BTCUSDT", "long", T0, T0 - timedelta(hours=1), 1.0)
    return Trade(signal, D("1"), entry, exit_fill, D("90"), D("120"), reason, net, net)


def make_result(trades=(), curve=None):
    curve = curve or (
        EquityPoint(T0, D("100"), "start"),
        EquityPoint(T0 + timedelta(hours=1), D("100"), "decision"),
    )
    ending = curve[-1].equity
    return BacktestResult(tuple(trades), ending, ending, (), tuple(curve))


def test_zero_trade_result_has_deterministic_zero_metrics():
    metrics = analyze_backtest(make_result())
    assert metrics.total_trades == metrics.winning_trades == metrics.losing_trades == 0
    assert metrics.win_rate == metrics.net_pnl == metrics.total_return == D("0")
    assert metrics.profit_factor is None
    assert metrics.average_trade_pnl == metrics.average_winning_trade == D("0")
    assert metrics.average_losing_trade == metrics.expectancy == D("0")
    assert metrics.total_fees == metrics.maximum_drawdown == D("0")


def test_empty_backtest_without_timestamps_uses_final_equity_as_baseline():
    result = BacktestResult((), D("100"), D("100"), ())
    metrics = analyze_backtest(result)
    assert metrics.total_return == D("0")
    assert metrics.ending_equity == D("100")


def test_non_positive_starting_equity_is_rejected():
    curve = (EquityPoint(T0, D("0"), "start"), EquityPoint(T0 + timedelta(hours=1), D("0"), "decision"))
    with pytest.raises(ValueError, match="starting equity must be positive"):
        analyze_backtest(make_result(curve=curve))


def test_all_winners_profit_factor_is_none_and_average_is_positive():
    metrics = analyze_backtest(make_result([make_trade("5"), make_trade("3")]))
    assert metrics.total_trades == metrics.winning_trades == 2
    assert metrics.losing_trades == 0
    assert metrics.gross_profit == D("8")
    assert metrics.gross_loss == D("0")
    assert metrics.profit_factor is None
    assert metrics.average_trade_pnl == metrics.average_winning_trade == D("4")


def test_all_losers_have_negative_average_and_finite_profit_factor():
    metrics = analyze_backtest(make_result([make_trade("-4"), make_trade("-2")]))
    assert metrics.winning_trades == 0
    assert metrics.losing_trades == 2
    assert metrics.gross_profit == D("0")
    assert metrics.gross_loss == D("6")
    assert metrics.profit_factor == D("0")
    assert metrics.average_losing_trade == D("-3")


def test_mixed_trade_metrics_count_fees_and_exit_reasons():
    trades = [
        make_trade("10", reason="take_profit"),
        make_trade("-4", reason="stop_loss"),
        make_trade("0", reason="end_of_backtest"),
    ]
    curve = (EquityPoint(T0, D("100"), "start"), EquityPoint(T0 + timedelta(hours=1), D("106"), "decision"))
    metrics = analyze_backtest(make_result(trades, curve))
    assert (metrics.total_trades, metrics.winning_trades, metrics.losing_trades) == (3, 1, 1)
    assert metrics.win_rate == D(1) / D(3)
    assert metrics.net_pnl == D("6")
    assert metrics.total_return == D("0.06")
    assert metrics.gross_profit == D("10")
    assert metrics.gross_loss == D("4")
    assert metrics.profit_factor == D("2.5")
    assert metrics.average_trade_pnl == metrics.expectancy == D("2")
    assert metrics.average_winning_trade == D("10")
    assert metrics.average_losing_trade == D("-4")
    assert metrics.total_fees == D("9")
    assert (metrics.stop_loss_trades, metrics.take_profit_trades, metrics.end_of_backtest_trades) == (1, 1, 1)
    assert all(isinstance(value, Decimal) for value in (
        metrics.win_rate, metrics.net_pnl, metrics.total_return, metrics.profit_factor,
        metrics.average_trade_pnl, metrics.maximum_drawdown_pct,
    ))


def test_maximum_drawdown_uses_equity_curve_including_unrealized_decline():
    curve = (
        EquityPoint(T0, D("100"), "start"),
        EquityPoint(T0 + timedelta(hours=1), D("120"), "decision"),
        EquityPoint(T0 + timedelta(hours=2), D("90"), "decision"),
        EquityPoint(T0 + timedelta(hours=3), D("110"), "decision"),
    )
    metrics = analyze_backtest(make_result(curve=curve))
    assert metrics.maximum_drawdown == D("30")
    assert metrics.maximum_drawdown_pct == D("0.25")


@pytest.mark.parametrize("curve", [
    (EquityPoint(datetime(2025, 1, 1), D("100"), "start"),),
    (EquityPoint(T0, D("100"), "start"), EquityPoint(T0, D("100"), "decision")),
    (EquityPoint(T0, D("100"), "start"), EquityPoint(T0 - timedelta(hours=1), D("100"), "decision")),
])
def test_invalid_timezone_or_chronology_is_rejected(curve):
    with pytest.raises(ValueError):
        analyze_backtest(make_result(curve=curve))


def test_non_decimal_or_non_finite_equity_and_trade_values_are_rejected():
    bad_curve = (EquityPoint(T0, D("100"), "start"), EquityPoint(T0 + timedelta(hours=1), 101.0, "decision"))
    with pytest.raises(ValueError, match="finite Decimal"):
        analyze_backtest(make_result(curve=bad_curve))
    with pytest.raises(ValueError, match="finite Decimal"):
        analyze_backtest(make_result([make_trade("NaN")]))


def test_equity_point_is_immutable():
    point = EquityPoint(T0, D("100"), "start")
    with pytest.raises(FrozenInstanceError):
        point.equity = D("90")

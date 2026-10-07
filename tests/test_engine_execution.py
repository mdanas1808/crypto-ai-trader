from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pandas as pd

from app.backtest.data import MultiTimeframeBacktestData
from app.backtest.engine import run_backtest
from app.backtest.models import BacktestConfig

UTC = timezone.utc
START = datetime(2024, 1, 2, tzinfo=UTC)


def make_stream(symbol="BTCUSDT", *, count=2, opens=None, highs=None, lows=None, closes=None):
    opens = opens or [Decimal("100")] * count
    highs = highs or [Decimal("102")] * count
    lows = lows or [Decimal("99")] * count
    closes = closes or [Decimal("101")] * count
    context_rows = []
    context_index = []
    for offset in range(-20, 0):
        opened = START + timedelta(hours=offset)
        context_index.append(opened + timedelta(hours=1))
        context_rows.append({"exchange": "binance_spot", "symbol": symbol,
                             "open_time": opened, "high": Decimal("100")})
    context = pd.DataFrame(context_rows, index=pd.DatetimeIndex(context_index, tz="UTC", name="decision_time"))
    rows = []
    index = []
    for i in range(count):
        opened = START + timedelta(hours=i)
        index.append(opened + timedelta(hours=1))
        row = {
            "exchange": "binance_spot", "symbol": symbol, "interval": "1h",
            "open_time": opened, "open": opens[i], "high": highs[i], "low": lows[i],
            "close": closes[i], "volume": Decimal("20"),
            "ema_200": 95.0, "rsi_14": 60.0, "volume_sma_20": 10.0, "atr_14": 2.5,
            "4h_exchange": "binance_spot", "4h_symbol": symbol,
            "4h_open_time": START - timedelta(hours=4),
            "4h_completion_time": START,
            "4h_close": 110.0, "4h_ema_20": 105.0,
            "4h_ema_50": 100.0, "4h_ema_200": 100.0,
        }
        rows.append(row)
    trading = pd.DataFrame(rows, index=pd.DatetimeIndex(index, tz="UTC", name="decision_time"))
    return MultiTimeframeBacktestData(context, trading)


def run(streams, *, end=START + timedelta(hours=10), **config_overrides):
    minimum_quantity = config_overrides.pop("minimum_quantity", Decimal("0.000001"))
    config = BacktestConfig(
        minimum_quantity=minimum_quantity, **config_overrides
    )
    return run_backtest(streams, end_time=end, config=config)


def test_signal_enters_only_at_next_candle_open_with_entry_slippage():
    stream = make_stream(count=2, opens=[Decimal("100"), Decimal("103")])
    result = run({"BTCUSDT": stream}, end=START + timedelta(hours=1))

    assert len(result.trades) == 1
    entry = result.trades[0].entry_fill
    assert entry.timestamp == START + timedelta(hours=1)
    assert entry.reference_price == Decimal("103")
    assert entry.price == Decimal("103.0515")


def test_final_row_signal_without_next_candle_is_not_entered():
    result = run({"BTCUSDT": make_stream(count=1)}, end=START)
    assert result.trades == ()
    assert result.final_cash == Decimal("100")


def test_stop_wins_when_stop_and_target_are_both_touched():
    stream = make_stream(
        count=2,
        highs=[Decimal("102"), Decimal("110")],
        lows=[Decimal("99"), Decimal("90")],
    )
    result = run({"BTCUSDT": stream}, end=START + timedelta(hours=1))

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "stop_loss"
    assert trade.exit_fill.reference_price == trade.stop_price
    assert trade.exit_fill.price == trade.stop_price * Decimal("0.9995")


def test_gap_through_stop_fills_at_open_before_same_time_signal():
    stream = make_stream(
        count=3,
        opens=[Decimal("100"), Decimal("100"), Decimal("90")],
        highs=[Decimal("102"), Decimal("104"), Decimal("95")],
        lows=[Decimal("99"), Decimal("97"), Decimal("89")],
        closes=[Decimal("101"), Decimal("103"), Decimal("94")],
    )
    result = run({"BTCUSDT": stream}, end=START + timedelta(hours=2))

    assert [trade.exit_reason for trade in result.trades] == ["stop_loss", "end_of_backtest"]
    first_exit, second_trade = result.trades
    assert first_exit.exit_fill.reference_price == Decimal("90")
    assert first_exit.exit_fill.timestamp == START + timedelta(hours=2)
    assert second_trade.entry_fill.timestamp == first_exit.exit_fill.timestamp


def test_gap_through_target_uses_open_then_adverse_slippage():
    stream = make_stream(
        count=3,
        opens=[Decimal("100"), Decimal("100"), Decimal("110")],
        highs=[Decimal("102"), Decimal("104"), Decimal("111")],
        lows=[Decimal("99"), Decimal("99"), Decimal("109")],
        closes=[Decimal("101"), Decimal("103"), Decimal("110")],
    )
    result = run({"BTCUSDT": stream}, end=START + timedelta(hours=2))

    assert len(result.trades) == 2
    trade = result.trades[0]
    assert trade.exit_reason == "take_profit"
    assert trade.exit_fill.reference_price == Decimal("110")
    assert trade.exit_fill.price == Decimal("110") * Decimal("0.9995")


def test_open_exactly_at_stop_exits_before_same_time_signal():
    stream = make_stream(
        count=3,
        opens=[Decimal("100"), Decimal("100"), Decimal("96.30")],
        highs=[Decimal("102"), Decimal("104"), Decimal("100")],
        lows=[Decimal("99"), Decimal("97"), Decimal("95")],
        closes=[Decimal("101"), Decimal("103"), Decimal("98")],
    )
    result = run({"BTCUSDT": stream}, end=START + timedelta(hours=2))

    first, reentry = result.trades
    assert first.exit_reason == "stop_loss"
    assert first.exit_fill.timestamp == START + timedelta(hours=2)
    assert first.exit_fill.reference_price == Decimal("96.30")
    assert first.exit_fill.price == Decimal("96.30") * Decimal("0.9995")
    assert reentry.entry_fill.timestamp == first.exit_fill.timestamp


def test_open_exactly_at_target_exits_at_open_with_adverse_slippage():
    stream = make_stream(
        count=3,
        opens=[Decimal("100"), Decimal("100"), Decimal("107.55")],
        highs=[Decimal("102"), Decimal("104"), Decimal("110")],
        lows=[Decimal("99"), Decimal("99"), Decimal("105")],
        closes=[Decimal("101"), Decimal("103"), Decimal("108")],
    )
    result = run({"BTCUSDT": stream}, end=START + timedelta(hours=2))

    first = result.trades[0]
    assert first.exit_reason == "take_profit"
    assert first.exit_fill.timestamp == START + timedelta(hours=2)
    assert first.exit_fill.reference_price == Decimal("107.55")
    assert first.exit_fill.price == Decimal("107.55") * Decimal("0.9995")


def test_target_gap_with_low_through_stop_uses_stop_first():
    stream = make_stream(
        count=3,
        opens=[Decimal("100"), Decimal("100"), Decimal("110")],
        highs=[Decimal("102"), Decimal("104"), Decimal("111")],
        lows=[Decimal("99"), Decimal("99"), Decimal("90")],
        closes=[Decimal("101"), Decimal("103"), Decimal("100")],
    )
    result = run({"BTCUSDT": stream}, end=START + timedelta(hours=2))

    first = result.trades[0]
    assert first.exit_reason == "stop_loss"
    assert first.exit_fill.reference_price == first.stop_price
    assert first.exit_fill.price == first.stop_price * Decimal("0.9995")


def test_multi_symbol_positions_share_limits_and_finish_at_end_of_backtest():
    streams = {
        symbol: make_stream(symbol, count=2)
        for symbol in ("DUSDT", "CUSDT", "BUSDT", "AUSDT")
    }
    result = run(
        streams,
        end=START + timedelta(hours=1),
        quantity_step=Decimal("0.1"),
        minimum_quantity=Decimal("0.1"),
    )

    assert len(result.trades) == 3
    assert {trade.signal.symbol for trade in result.trades} == {"AUSDT", "BUSDT", "CUSDT"}
    assert all(trade.exit_reason == "end_of_backtest" for trade in result.trades)
    assert all(
        trade.exit_fill.price == Decimal("101") * Decimal("0.9995")
        for trade in result.trades
    )
    assert result.final_open_positions == ()
    assert result.final_cash == result.final_equity
    assert result.final_cash == Decimal("100") + result.realized_net_pnl


def test_shared_open_risk_limit_is_applied_across_symbols():
    streams = {
        symbol: make_stream(symbol, count=2)
        for symbol in ("AUSDT", "BUSDT", "CUSDT")
    }
    result = run(streams, end=START + timedelta(hours=1))

    # With fine quantity rounding, transaction costs make three individually
    # sized 0.5% risks exceed 1.5% of the then-current shared equity.
    assert len(result.trades) == 2
    assert {trade.signal.symbol for trade in result.trades} == {"AUSDT", "BUSDT"}


def test_multiple_symbols_are_processed_chronologically():
    early = make_stream(
        "BTCUSDT", count=3,
        highs=[Decimal("102"), Decimal("110"), Decimal("104")],
        lows=[Decimal("99"), Decimal("90"), Decimal("99")],
        closes=[Decimal("101"), Decimal("103"), Decimal("101")],
    )
    late = make_stream(
        "ETHUSDT", count=3,
        highs=[Decimal("102"), Decimal("104"), Decimal("110")],
        lows=[Decimal("99"), Decimal("99"), Decimal("99")],
        closes=[Decimal("101"), Decimal("103"), Decimal("105")],
    )
    # Only the first candle in each stream emits an entry signal.
    early.trading_frame.loc[early.trading_frame.index[1:], "rsi_14"] = float("nan")
    late.trading_frame.loc[late.trading_frame.index[1:], "rsi_14"] = float("nan")

    result = run({"ETHUSDT": late, "BTCUSDT": early}, end=START + timedelta(hours=2))

    assert [trade.signal.symbol for trade in result.trades] == ["BTCUSDT", "ETHUSDT"]
    assert [trade.exit_fill.timestamp for trade in result.trades] == [
        START + timedelta(hours=2), START + timedelta(hours=3)
    ]
    assert result.trades[0].exit_reason == "stop_loss"
    assert result.trades[1].exit_reason == "take_profit"


def test_equity_curve_starts_before_first_candle_prices_are_known():
    stream = make_stream(
        count=2,
        highs=[Decimal("10000"), Decimal("102")],
        lows=[Decimal("1"), Decimal("99")],
        closes=[Decimal("5000"), Decimal("101")],
    )
    result = run({"BTCUSDT": stream}, end=START + timedelta(hours=1))

    assert result.equity_curve[0].timestamp == START
    assert result.equity_curve[0].event == "start"
    assert result.equity_curve[0].equity == Decimal("100")


def test_equity_curve_has_one_ordered_decision_per_global_timeline_time():
    result = run(
        {"BTCUSDT": make_stream("BTCUSDT", count=3), "ETHUSDT": make_stream("ETHUSDT", count=3)},
        end=START + timedelta(hours=2),
    )
    points = result.equity_curve

    assert [point.timestamp for point in points] == [
        START,
        START + timedelta(hours=1),
        START + timedelta(hours=2),
        START + timedelta(hours=3),
    ]
    assert [point.event for point in points] == ["start", "decision", "decision", "decision"]
    assert all(point.timestamp.tzinfo is UTC for point in points)
    assert points[-1].equity == result.final_equity


def test_final_decision_point_reflects_forced_close_state():
    result = run({"BTCUSDT": make_stream(count=2)}, end=START + timedelta(hours=1))
    assert result.trades[-1].exit_reason == "end_of_backtest"
    assert result.equity_curve[-1].timestamp == START + timedelta(hours=2)
    assert result.equity_curve[-1].event == "decision"
    assert result.equity_curve[-1].equity == result.final_equity == result.final_cash


def test_opening_gap_fill_time_precedes_completed_equity_observation():
    stream = make_stream(
        count=3,
        opens=[Decimal("100"), Decimal("100"), Decimal("90")],
        highs=[Decimal("102"), Decimal("104"), Decimal("95")],
        lows=[Decimal("99"), Decimal("97"), Decimal("89")],
        closes=[Decimal("101"), Decimal("103"), Decimal("94")],
    )
    result = run({"BTCUSDT": stream}, end=START + timedelta(hours=2))
    gap_exit = result.trades[0].exit_fill

    assert gap_exit.timestamp == START + timedelta(hours=2)
    # The gap handling is an execution event at the open. The point at that
    # boundary remains the preceding completed-candle valuation; the state
    # after resolving the bar is observed at the next completed decision.
    point_at_open = next(point for point in result.equity_curve if point.timestamp == gap_exit.timestamp)
    completed_observation = next(
        point for point in result.equity_curve if point.timestamp == START + timedelta(hours=3)
    )
    assert point_at_open.equity != result.final_equity
    assert completed_observation.equity == result.final_equity


def test_equity_curve_uses_union_of_multi_symbol_decision_times():
    early = make_stream("BTCUSDT", count=2)
    late = make_stream("ETHUSDT", count=2)
    late.trading_frame.index = late.trading_frame.index + pd.Timedelta(hours=2)
    late.trading_frame["open_time"] = late.trading_frame["open_time"].map(
        lambda value: value + timedelta(hours=2)
    )
    late.context_frame.index = late.context_frame.index + pd.Timedelta(hours=2)
    late.context_frame["open_time"] = late.context_frame["open_time"].map(
        lambda value: value + timedelta(hours=2)
    )
    result = run({"BTCUSDT": early, "ETHUSDT": late}, end=START + timedelta(hours=3))

    assert [point.timestamp for point in result.equity_curve] == [
        START, START + timedelta(hours=1), START + timedelta(hours=2),
        START + timedelta(hours=3), START + timedelta(hours=4),
    ]

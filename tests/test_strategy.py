from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from app.backtest.data import MultiTimeframeBacktestData
from app.strategy.breakout import BreakoutSignal, generate_breakout_signals

UTC = timezone.utc
START = datetime(2024, 1, 2, 0, 0, tzinfo=UTC)


def _make_data(
    *,
    context_count: int = 20,
    trading_count: int = 1,
    context_high: Decimal = Decimal("100"),
    trading_close: Decimal = Decimal("101"),
    trading_high: Decimal = Decimal("100"),
    updates: dict[str, object] | None = None,
) -> MultiTimeframeBacktestData:
    context_rows = []
    context_times = []
    for offset in range(-context_count, 0):
        open_time = START + timedelta(hours=offset)
        context_times.append(open_time + timedelta(hours=1))
        context_rows.append(
            {
                "exchange": "binance_spot",
                "symbol": "BTCUSDT",
                "open_time": open_time,
                "high": context_high,
            }
        )
    context = pd.DataFrame(
        context_rows, index=pd.DatetimeIndex(context_times, name="decision_time", tz="UTC")
    )

    trading_rows = []
    trading_times = []
    for offset in range(trading_count):
        open_time = START + timedelta(hours=offset)
        trading_times.append(open_time + timedelta(hours=1))
        row = {
            "exchange": "binance_spot",
            "symbol": "BTCUSDT",
            "interval": "1h",
            "open_time": open_time,
            "open": Decimal("100"),
            "high": trading_high,
            "low": Decimal("99"),
            "close": trading_close,
            "volume": Decimal("15"),
            "ema_20": 100.0,
            "ema_50": 100.0,
            "ema_200": 95.0,
            "sma_20": 100.0,
            "sma_50": 100.0,
            "sma_200": 100.0,
            "rsi_14": 60.0,
            "macd": 1.0,
            "macd_signal": 0.5,
            "macd_histogram": 0.5,
            "atr_14": 2.5,
            "bollinger_upper": 110.0,
            "bollinger_middle": 100.0,
            "bollinger_lower": 90.0,
            "volume_sma_20": 10.0,
            "percent_change": 0.01,
            "4h_exchange": "binance_spot",
            "4h_symbol": "BTCUSDT",
            "4h_interval": "4h",
            "4h_open_time": START - timedelta(hours=4),
            "4h_open": Decimal("95"),
            "4h_high": Decimal("111"),
            "4h_low": Decimal("94"),
            "4h_close": 110.0,
            "4h_volume": Decimal("60"),
            "4h_ema_20": 105.0,
            "4h_ema_50": 100.0,
            "4h_ema_200": 100.0,
            "4h_sma_20": 100.0,
            "4h_sma_50": 100.0,
            "4h_sma_200": 100.0,
            "4h_rsi_14": 60.0,
            "4h_macd": 1.0,
            "4h_macd_signal": 0.5,
            "4h_macd_histogram": 0.5,
            "4h_atr_14": 2.5,
            "4h_bollinger_upper": 110.0,
            "4h_bollinger_middle": 100.0,
            "4h_bollinger_lower": 90.0,
            "4h_volume_sma_20": 10.0,
            "4h_percent_change": 0.01,
            "4h_completion_time": START,
        }
        if updates:
            row.update(updates)
        trading_rows.append(row)
    trading = pd.DataFrame(
        trading_rows,
        index=pd.DatetimeIndex(trading_times, name="decision_time", tz="UTC"),
    )
    return MultiTimeframeBacktestData(context_frame=context, trading_frame=trading)


def test_valid_signal_has_exact_intent_fields_and_is_immutable():
    signals = generate_breakout_signals(_make_data())

    assert len(signals) == 1
    signal = signals[0]
    assert isinstance(signal, BreakoutSignal)
    assert signal.exchange == "binance_spot"
    assert signal.symbol == "BTCUSDT"
    assert signal.side == "long"
    assert signal.decision_time == START + timedelta(hours=1)
    assert signal.signal_open_time == START
    assert signal.atr_14 == 2.5
    with pytest.raises(FrozenInstanceError):
        signal.side = "short"


@pytest.mark.parametrize(
    "updates",
    [
        {"4h_ema_20": 100.0, "4h_ema_50": 100.0},
        {"4h_ema_20": 99.0, "4h_ema_50": 100.0},
    ],
)
def test_four_hour_ema_regime_failure_blocks_signal(updates):
    assert generate_breakout_signals(_make_data(updates=updates)) == []


def test_four_hour_close_not_above_ema200_blocks_signal():
    assert generate_breakout_signals(_make_data(updates={"4h_close": 100.0})) == []


def test_one_hour_close_not_above_ema200_blocks_signal():
    assert generate_breakout_signals(_make_data(updates={"close": Decimal("95")})) == []


@pytest.mark.parametrize("rsi", [54.999, 70.001])
def test_rsi_outside_inclusive_range_blocks_signal(rsi):
    assert generate_breakout_signals(_make_data(updates={"rsi_14": rsi})) == []


@pytest.mark.parametrize("rsi", [55.0, 70.0])
def test_rsi_boundaries_allow_signal(rsi):
    assert len(generate_breakout_signals(_make_data(updates={"rsi_14": rsi}))) == 1


def test_volume_below_threshold_blocks_signal():
    assert generate_breakout_signals(_make_data(updates={"volume": Decimal("14.99")})) == []


def test_volume_exactly_one_and_a_half_times_average_allows_signal():
    assert len(generate_breakout_signals(_make_data(updates={"volume": Decimal("15")}))) == 1


def test_breakout_equality_does_not_signal():
    assert generate_breakout_signals(_make_data(trading_close=Decimal("100"))) == []


def test_close_strictly_above_previous_twenty_high_signals():
    assert (
        len(
            generate_breakout_signals(
                _make_data(context_high=Decimal("100"), trading_close=Decimal("100.01"))
            )
        )
        == 1
    )


def test_current_candle_high_is_excluded_from_breakout_reference():
    signals = generate_breakout_signals(
        _make_data(
            context_high=Decimal("100"), trading_close=Decimal("101"), trading_high=Decimal("999")
        )
    )
    assert len(signals) == 1


def test_fewer_than_twenty_prior_candles_does_not_signal():
    assert generate_breakout_signals(_make_data(context_count=19)) == []


def test_missing_four_hour_row_does_not_signal():
    assert (
        generate_breakout_signals(
            _make_data(updates={"4h_open_time": pd.NaT, "4h_completion_time": pd.NaT})
        )
        == []
    )


@pytest.mark.parametrize("column", ["4h_ema_20", "4h_ema_50", "4h_close", "4h_ema_200"])
def test_nan_four_hour_indicator_does_not_signal(column):
    assert generate_breakout_signals(_make_data(updates={column: np.nan})) == []


@pytest.mark.parametrize("column", ["ema_200", "rsi_14", "volume_sma_20"])
def test_nan_one_hour_indicator_does_not_signal(column):
    assert generate_breakout_signals(_make_data(updates={column: np.nan})) == []


def test_missing_atr_does_not_signal():
    assert generate_breakout_signals(_make_data(updates={"atr_14": np.nan})) == []


@pytest.mark.parametrize("atr", [0.0, -0.1])
def test_nonpositive_atr_does_not_signal(atr):
    assert generate_breakout_signals(_make_data(updates={"atr_14": atr})) == []


def test_empty_trading_frame_produces_no_signals():
    data = _make_data()
    data.trading_frame.drop(index=data.trading_frame.index, inplace=True)
    assert generate_breakout_signals(data) == []


def test_multiple_qualifying_trading_rows_produce_ordered_signals():
    signals = generate_breakout_signals(_make_data(trading_count=3))
    assert [signal.signal_open_time for signal in signals] == [
        START,
        START + timedelta(hours=1),
        START + timedelta(hours=2),
    ]
    assert [signal.decision_time for signal in signals] == [
        START + timedelta(hours=1),
        START + timedelta(hours=2),
        START + timedelta(hours=3),
    ]


def test_context_rows_are_history_only_and_frames_are_not_mutated():
    data = _make_data(context_count=21, trading_count=2)
    context_before = data.context_frame.copy(deep=True)
    trading_before = data.trading_frame.copy(deep=True)

    signals = generate_breakout_signals(data)

    assert [signal.signal_open_time for signal in signals] == [START, START + timedelta(hours=1)]
    assert all(signal.signal_open_time >= START for signal in signals)
    pd.testing.assert_frame_equal(data.context_frame, context_before)
    pd.testing.assert_frame_equal(data.trading_frame, trading_before)


@pytest.mark.parametrize(
    "column,value",
    [("symbol", "ETHUSDT"), ("exchange", "other_exchange"), ("symbol", None)],
)
def test_context_identity_mismatch_produces_no_signals(column, value):
    data = _make_data()
    data.context_frame.loc[data.context_frame.index[-1], column] = value

    assert generate_breakout_signals(data) == []


@pytest.mark.parametrize(
    "column,value",
    [("symbol", "ETHUSDT"), ("exchange", "other_exchange"), ("symbol", None)],
)
def test_mixed_trading_identity_produces_no_signals(column, value):
    data = _make_data(trading_count=2)
    data.trading_frame.loc[data.trading_frame.index[-1], column] = value

    assert generate_breakout_signals(data) == []


@pytest.mark.parametrize(
    "column,value",
    [("4h_symbol", "ETHUSDT"), ("4h_exchange", "other_exchange"), ("4h_symbol", None)],
)
def test_attached_four_hour_identity_mismatch_produces_no_signal(column, value):
    assert generate_breakout_signals(_make_data(updates={column: value})) == []


def test_missing_context_identity_columns_are_rejected():
    data = _make_data()
    data.context_frame.drop(columns="symbol", inplace=True)

    with pytest.raises(ValueError, match="context_frame is missing required columns: symbol"):
        generate_breakout_signals(data)

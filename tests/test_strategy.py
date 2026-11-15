from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from app.backtest.data import MultiTimeframeBacktestData
from app.strategy.breakout import (
    BreakoutSignal,
    compute_dynamic_leverage,
    generate_breakout_signals,
    pick_most_feasible_signal,
    score_signal,
)

UTC = timezone.utc
START = datetime(2024, 1, 2, 0, 0, tzinfo=UTC)


def _make_data(
    *,
    context_count: int = 20,
    trading_count: int = 1,
    context_high: Decimal = Decimal("100"),
    context_low: Decimal = Decimal("90"),
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
                "low": context_low,
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
    assert signal.direction == "LONG"
    assert signal.decision_time == START + timedelta(hours=1)
    assert signal.signal_open_time == START
    assert signal.atr_14 == 2.5
    assert signal.score > 0
    assert signal.stop_loss == Decimal("101.0") - Decimal("1.5") * Decimal("2.5")
    assert signal.take_profit == Decimal("101.0") + Decimal("3.0") * Decimal("2.5")
    with pytest.raises(FrozenInstanceError):
        signal.side = "short"


# --- Long Criteria Unit Tests ---


def test_long_close_above_ema200_required():
    # close = 95 <= ema_200 (95.0) -> blocks long
    assert (
        generate_breakout_signals(_make_data(updates={"close": Decimal("95"), "ema_200": 95.0}))
        == []
    )
    # close = 96 > ema_200 (95.0) -> allows long
    assert (
        len(
            generate_breakout_signals(
                _make_data(updates={"close": Decimal("101"), "ema_200": 95.0})
            )
        )
        == 1
    )


@pytest.mark.parametrize("rsi", [52.0, 51.99, 75.0, 75.01])
def test_long_rsi_outside_strict_bounds_blocks_signal(rsi):
    assert generate_breakout_signals(_make_data(updates={"rsi_14": rsi})) == []


@pytest.mark.parametrize("rsi", [52.01, 60.0, 74.99])
def test_long_rsi_within_strict_bounds_allows_signal(rsi):
    assert len(generate_breakout_signals(_make_data(updates={"rsi_14": rsi}))) == 1


def test_long_volume_must_exceed_volume_sma_20():
    # volume = 10.0 == volume_sma_20 (10.0) -> not strictly greater, blocks
    assert generate_breakout_signals(_make_data(updates={"volume": Decimal("10.0")})) == []
    # volume = 10.01 > volume_sma_20 (10.0) -> allows
    assert len(generate_breakout_signals(_make_data(updates={"volume": Decimal("10.01")}))) == 1


def test_long_breakout_equality_does_not_signal():
    # close == context_high (100) -> blocks
    assert (
        generate_breakout_signals(
            _make_data(context_high=Decimal("100"), trading_close=Decimal("100"))
        )
        == []
    )


def test_long_close_strictly_above_previous_twenty_high_signals():
    assert (
        len(
            generate_breakout_signals(
                _make_data(context_high=Decimal("100"), trading_close=Decimal("100.01"))
            )
        )
        == 1
    )


# --- Short Criteria Unit Tests ---


def test_short_criteria_generates_valid_short_signal():
    # Short criteria: Close < 200 EMA, Close < 20-period Low, Volume > 20-period Volume SMA, bearish RSI (< 48 and > 25)
    data = _make_data(
        context_low=Decimal("95"),
        trading_close=Decimal("89"),
        updates={
            "ema_200": 105.0,  # close 89 < 105
            "low": Decimal("88"),
            "high": Decimal("91"),
            "rsi_14": 40.0,  # 25 < 40 < 48
            "volume": Decimal("15"),
            "volume_sma_20": 10.0,
            "atr_14": 2.0,
        },
    )
    signals = generate_breakout_signals(data)
    assert len(signals) == 1
    sig = signals[0]
    assert sig.side == "short"
    assert sig.direction == "SHORT"
    assert sig.entry_price == Decimal("89")
    # SL is 1.5x ATR above entry, TP is 3.0x ATR below entry
    assert sig.stop_loss == Decimal("89") + Decimal("1.5") * Decimal("2.0")
    assert sig.take_profit == Decimal("89") - Decimal("3.0") * Decimal("2.0")
    assert sig.score > 0


@pytest.mark.parametrize("rsi", [25.0, 24.99, 48.0, 48.01])
def test_short_rsi_outside_strict_bounds_blocks_signal(rsi):
    data = _make_data(
        context_low=Decimal("95"),
        trading_close=Decimal("89"),
        updates={
            "ema_200": 105.0,
            "low": Decimal("88"),
            "rsi_14": rsi,
            "volume": Decimal("15"),
            "volume_sma_20": 10.0,
        },
    )
    assert generate_breakout_signals(data) == []


@pytest.mark.parametrize("rsi", [25.01, 35.0, 47.99])
def test_short_rsi_within_strict_bounds_allows_signal(rsi):
    data = _make_data(
        context_low=Decimal("95"),
        trading_close=Decimal("89"),
        updates={
            "ema_200": 105.0,
            "low": Decimal("88"),
            "rsi_14": rsi,
            "volume": Decimal("15"),
            "volume_sma_20": 10.0,
        },
    )
    assert len(generate_breakout_signals(data)) == 1


def test_short_close_above_ema200_blocks_signal():
    data = _make_data(
        context_low=Decimal("95"),
        trading_close=Decimal("89"),
        updates={
            "ema_200": 85.0,  # Close 89 is above 200 EMA -> blocks short
            "rsi_14": 40.0,
        },
    )
    assert generate_breakout_signals(data) == []


# --- Scoring & Most Feasible Signal Tests ---


def test_momentum_strength_scoring():
    # Long score = (close - ref_high)/atr + volume/volume_sma
    # close=105, ref_high=100, atr=2.5, volume=20, volume_sma=10
    # breakout_dist = 5 -> dist/atr = 5/2.5 = 2.0
    # vol_ratio = 20/10 = 2.0 -> score = 4.0
    score_long = score_signal("long", 105.0, 100.0, 2.5, 20.0, 10.0)
    assert score_long == 4.0

    # Short score = (ref_low - close)/atr + volume/volume_sma
    # ref_low=95, close=90, atr=2.5, volume=15, volume_sma=10
    # breakout_dist = 5 -> dist/atr = 5/2.5 = 2.0
    # vol_ratio = 15/10 = 1.5 -> score = 3.5
    score_short = score_signal("short", 90.0, 95.0, 2.5, 15.0, 10.0)
    assert score_short == 3.5


def test_pick_most_feasible_signal_selects_highest_score():
    sig1 = BreakoutSignal(
        exchange="binance_spot",
        symbol="BTCUSDT",
        side="long",
        decision_time=START + timedelta(hours=1),
        signal_open_time=START,
        atr_14=2.0,
        score=2.5,
    )
    sig2 = BreakoutSignal(
        exchange="binance_spot",
        symbol="ETHUSDT",
        side="short",
        decision_time=START + timedelta(hours=1),
        signal_open_time=START,
        atr_14=1.5,
        score=4.8,  # highest score
    )
    sig3 = BreakoutSignal(
        exchange="binance_spot",
        symbol="SOLUSDT",
        side="long",
        decision_time=START + timedelta(hours=1),
        signal_open_time=START,
        atr_14=0.8,
        score=3.1,
    )
    best = pick_most_feasible_signal([sig1, sig2, sig3])
    assert best is not None
    assert best.symbol == "ETHUSDT"
    assert best.score == 4.8


# --- Dynamic Leverage Tests ---


def test_dynamic_leverage_scaling():
    # Price = 100. ATR = 1.0 (vol_ratio = 0.010 <= 0.015) -> 10x leverage
    assert compute_dynamic_leverage(1.0, 100.0) == 10
    # Price = 100. ATR = 4.0 (vol_ratio = 0.040 >= 0.035) -> 5x leverage
    assert compute_dynamic_leverage(4.0, 100.0) == 5
    # Price = 100. ATR = 2.5 (vol_ratio = 0.025) -> interpolated between 5 and 10 (approx 7 or 8)
    lev = compute_dynamic_leverage(2.5, 100.0)
    assert 5 <= lev <= 10


def test_dynamic_leverage_env_var_overrides(monkeypatch):
    monkeypatch.setenv("DEFAULT_LEVERAGE", "8")
    monkeypatch.setenv("MIN_LEVERAGE", "6")
    monkeypatch.setenv("MAX_LEVERAGE", "12")
    assert compute_dynamic_leverage(0.5, 100.0) == 12  # max
    assert compute_dynamic_leverage(5.0, 100.0) == 6  # min


# --- Legacy Mode Tests ---


@pytest.mark.parametrize(
    "updates",
    [
        {"4h_ema_20": 100.0, "4h_ema_50": 100.0},
        {"4h_ema_20": 99.0, "4h_ema_50": 100.0},
    ],
)
def test_legacy_mode_four_hour_ema_failure_blocks_signal(updates):
    assert generate_breakout_signals(_make_data(updates=updates), legacy_mode=True) == []


def test_legacy_mode_four_hour_close_not_above_ema200_blocks_signal():
    assert (
        generate_breakout_signals(_make_data(updates={"4h_close": 100.0}), legacy_mode=True) == []
    )


def test_legacy_mode_missing_four_hour_row_blocks_signal():
    assert (
        generate_breakout_signals(
            _make_data(updates={"4h_open_time": pd.NaT, "4h_completion_time": pd.NaT}),
            legacy_mode=True,
        )
        == []
    )


# --- Core Integrity & Contiguity Tests ---


def test_current_candle_high_is_excluded_from_breakout_reference():
    signals = generate_breakout_signals(
        _make_data(
            context_high=Decimal("100"), trading_close=Decimal("101"), trading_high=Decimal("999")
        )
    )
    assert len(signals) == 1


def test_fewer_than_twenty_prior_candles_does_not_signal():
    assert generate_breakout_signals(_make_data(context_count=19)) == []


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


def test_missing_context_identity_columns_are_rejected():
    data = _make_data()
    data.context_frame.drop(columns="symbol", inplace=True)
    with pytest.raises(ValueError, match="context_frame is missing required columns: symbol"):
        generate_breakout_signals(data)

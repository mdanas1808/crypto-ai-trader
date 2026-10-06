from datetime import datetime, timedelta, timezone
from decimal import Decimal
from math import sqrt

import numpy as np
import pandas as pd
import pytest

from app.indicators.technical import INDICATOR_COLUMNS, calculate_indicators
from app.market.models import Candle


def make_candles(
    closes: list[int | float], volumes: list[int] | None = None
) -> list[Candle]:
    volumes = volumes if volumes is not None else list(range(1, len(closes) + 1))
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    candles = []
    for index, (close_value, volume_value) in enumerate(zip(closes, volumes, strict=True)):
        close = Decimal(str(close_value))
        candles.append(
            Candle(
                exchange="binance_spot",
                symbol="BTCUSDT",
                interval="1h",
                open_time=start + timedelta(hours=index),
                open=close,
                high=close + Decimal("2"),
                low=close - Decimal("2"),
                close=close,
                volume=Decimal(volume_value),
            )
        )
    return candles


def reference_ema(values: list[float], period: int) -> list[float]:
    result = [float("nan")] * len(values)
    if len(values) < period:
        return result
    seed_index = period - 1
    result[seed_index] = sum(values[:period]) / period
    alpha = 2.0 / (period + 1.0)
    for index in range(period, len(values)):
        result[index] = alpha * values[index] + (1.0 - alpha) * result[index - 1]
    return result


def reference_wilder(values: list[float], period: int) -> list[float]:
    result = [float("nan")] * len(values)
    seed_index = period - 1
    result[seed_index] = sum(values[:period]) / period
    for index in range(period, len(values)):
        result[index] = (
            (period - 1) * result[index - 1] + values[index]
        ) / period
    return result


def test_ema_20_uses_sma_seed_then_recursive_formula() -> None:
    candles = make_candles(list(range(10, 35)))
    result = calculate_indicators(candles)
    expected = reference_ema([float(c.close) for c in candles], 20)

    assert np.isnan(result.loc[18, "ema_20"])
    assert result.loc[19, "ema_20"] == pytest.approx(expected[19])
    assert result.loc[20, "ema_20"] == pytest.approx(expected[20])


def test_ema_50_has_correct_warmup_and_independent_values() -> None:
    result = calculate_indicators(make_candles(list(range(10, 61))))

    assert np.isnan(result.loc[48, "ema_50"])
    assert result.loc[49, "ema_50"] == pytest.approx((10 + 59) / 2)
    # The next close is 60; alpha is 2/51 and the SMA-seeded value is 34.5.
    assert result.loc[50, "ema_50"] == pytest.approx(34.5 + (2 / 51) * (60 - 34.5))


def test_ema_200_has_correct_warmup_and_independent_values() -> None:
    result = calculate_indicators(make_candles(list(range(10, 211))))

    assert np.isnan(result.loc[198, "ema_200"])
    assert result.loc[199, "ema_200"] == pytest.approx((10 + 209) / 2)
    # The next close is 210; alpha is 2/201 and the SMA-seeded value is 109.5.
    assert result.loc[200, "ema_200"] == pytest.approx(109.5 + (2 / 201) * (210 - 109.5))


def test_sma_20_is_trailing_arithmetic_mean() -> None:
    result = calculate_indicators(make_candles(list(range(10, 31))))

    assert np.isnan(result.loc[18, "sma_20"])
    assert result.loc[19, "sma_20"] == pytest.approx(sum(range(10, 30)) / 20)
    assert result.loc[20, "sma_20"] == pytest.approx(sum(range(11, 31)) / 20)


def test_sma_50_has_correct_warmup_and_trailing_window() -> None:
    result = calculate_indicators(make_candles(list(range(10, 61))))

    assert np.isnan(result.loc[48, "sma_50"])
    assert result.loc[49, "sma_50"] == pytest.approx((10 + 59) / 2)
    assert result.loc[50, "sma_50"] == pytest.approx((11 + 60) / 2)


def test_sma_200_has_correct_warmup_and_trailing_window() -> None:
    result = calculate_indicators(make_candles(list(range(10, 211))))

    assert np.isnan(result.loc[198, "sma_200"])
    assert result.loc[199, "sma_200"] == pytest.approx((10 + 209) / 2)
    assert result.loc[200, "sma_200"] == pytest.approx((11 + 210) / 2)


def test_rsi_14_uses_wilder_average_of_gain_and_loss() -> None:
    closes = [100, 102, 101, 104, 103, 102, 105, 107, 106, 108, 105, 107, 109, 108, 111, 110]
    result = calculate_indicators(make_candles(closes))
    changes = [right - left for left, right in zip(closes[:-1], closes[1:], strict=True)]
    gains = [float(max(change, 0)) for change in changes]
    losses = [float(max(-change, 0)) for change in changes]
    avg_gain = sum(gains[:14]) / 14
    avg_loss = sum(losses[:14]) / 14
    expected_at_14 = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    avg_gain = (13 * avg_gain + gains[14]) / 14
    avg_loss = (13 * avg_loss + losses[14]) / 14
    expected_at_15 = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)

    assert np.isnan(result.loc[13, "rsi_14"])
    assert result.loc[14, "rsi_14"] == pytest.approx(expected_at_14)
    assert result.loc[15, "rsi_14"] == pytest.approx(expected_at_15)


@pytest.mark.parametrize(
    ("closes", "expected"),
    [
        pytest.param(list(range(100, 116)), 100.0, id="gains-with-no-losses"),
        pytest.param(list(range(100, 84, -1)), 0.0, id="losses-with-no-gains"),
        pytest.param([100] * 16, 50.0, id="flat-series"),
    ],
)
def test_rsi_edge_cases_first_become_valid_after_14_changes(
    closes: list[int], expected: float
) -> None:
    result = calculate_indicators(make_candles(closes))

    assert result.loc[:13, "rsi_14"].isna().all()
    assert result.loc[14, "rsi_14"] == pytest.approx(expected)


def test_macd_12_26_9_matches_independent_ema_calculation() -> None:
    closes = [100 + ((index * 7) % 13) - index * 0.1 for index in range(45)]
    result = calculate_indicators(make_candles(closes))
    fast = reference_ema(closes, 12)
    slow = reference_ema(closes, 26)
    macd = [fast_value - slow_value for fast_value, slow_value in zip(fast, slow, strict=True)]
    first_macd_index = 25
    signal = [float("nan")] * len(closes)
    signal_seed_index = first_macd_index + 8
    signal[signal_seed_index] = sum(macd[first_macd_index : signal_seed_index + 1]) / 9
    alpha = 2.0 / 10.0
    for index in range(signal_seed_index + 1, len(closes)):
        signal[index] = alpha * macd[index] + (1.0 - alpha) * signal[index - 1]

    assert np.isnan(result.loc[24, "macd"])
    assert result.loc[25, "macd"] == pytest.approx(macd[25])
    assert np.isnan(result.loc[32, "macd_signal"])
    assert result.loc[33, "macd_signal"] == pytest.approx(signal[33])
    assert result.loc[33, "macd_histogram"] == pytest.approx(macd[33] - signal[33])


def test_atr_14_uses_true_range_and_wilder_smoothing() -> None:
    closes = [100 + ((index * 5) % 9) - index * 0.2 for index in range(17)]
    candles = make_candles(closes)
    result = calculate_indicators(candles)
    true_ranges = []
    for index, candle in enumerate(candles):
        high = float(candle.high)
        low = float(candle.low)
        if index == 0:
            true_ranges.append(high - low)
        else:
            previous_close = float(candles[index - 1].close)
            true_ranges.append(
                max(high - low, abs(high - previous_close), abs(low - previous_close))
            )
    expected = reference_wilder(true_ranges, 14)

    assert np.isnan(result.loc[12, "atr_14"])
    assert result.loc[13, "atr_14"] == pytest.approx(expected[13])
    assert result.loc[14, "atr_14"] == pytest.approx(expected[14])


def test_atr_first_true_range_is_high_minus_low_and_seeds_at_index_13() -> None:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    candles = []
    for index in range(15):
        high, low = (104, 96) if index == 0 else (101, 99)
        candles.append(
            Candle(
                exchange="binance_spot",
                symbol="BTCUSDT",
                interval="1h",
                open_time=start + timedelta(hours=index),
                open=Decimal("100"),
                high=Decimal(high),
                low=Decimal(low),
                close=Decimal("100"),
                volume=Decimal("1"),
            )
        )

    result = calculate_indicators(candles)
    first_true_range = float(candles[0].high - candles[0].low)
    expected_first_atr = (first_true_range + 13 * 2.0) / 14

    assert first_true_range == 8.0
    assert np.isnan(result.loc[12, "atr_14"])
    assert result.loc[13, "atr_14"] == pytest.approx(expected_first_atr)


def test_bollinger_bands_use_population_standard_deviation() -> None:
    result = calculate_indicators(make_candles(list(range(10, 30))))
    values = list(range(10, 30))
    middle = sum(values) / 20
    population_std = sqrt(sum((value - middle) ** 2 for value in values) / 20)

    assert result.loc[19, "bb_middle_20"] == pytest.approx(middle)
    assert result.loc[19, "bb_upper_20"] == pytest.approx(middle + 2 * population_std)
    assert result.loc[19, "bb_lower_20"] == pytest.approx(middle - 2 * population_std)


def test_volume_sma_20_uses_trailing_volume_values() -> None:
    volumes = list(range(1, 22))
    result = calculate_indicators(make_candles(list(range(100, 121)), volumes))

    assert np.isnan(result.loc[18, "volume_sma_20"])
    assert result.loc[19, "volume_sma_20"] == pytest.approx(sum(volumes[:20]) / 20)
    assert result.loc[20, "volume_sma_20"] == pytest.approx(sum(volumes[1:]) / 20)


def test_percentage_price_change_is_from_previous_close_in_percent_units() -> None:
    result = calculate_indicators(make_candles([100, 110, 99]))

    assert np.isnan(result.loc[0, "price_change_pct"])
    assert result.loc[1, "price_change_pct"] == pytest.approx(10.0)
    assert result.loc[2, "price_change_pct"] == pytest.approx(-10.0)


def test_insufficient_history_produces_missing_values_without_filling() -> None:
    result = calculate_indicators(make_candles(list(range(10, 20))))
    not_ready = (
        "ema_20",
        "ema_50",
        "ema_200",
        "sma_20",
        "sma_50",
        "sma_200",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_histogram",
        "atr_14",
        "bb_middle_20",
        "bb_upper_20",
        "bb_lower_20",
        "volume_sma_20",
    )

    assert result[list(not_ready)].isna().all().all()
    assert np.isnan(result.loc[0, "price_change_pct"])
    assert result.loc[1:, "price_change_pct"].notna().all()


def test_rejects_non_chronological_input() -> None:
    candles = make_candles([100, 101, 102])

    with pytest.raises(ValueError, match="strictly chronological"):
        calculate_indicators([candles[1], candles[0], candles[2]])


def test_appending_future_candles_does_not_change_past_indicators() -> None:
    closes = [100 + ((index * 11) % 17) - index * 0.05 for index in range(45)]
    prefix = make_candles(closes[:35])
    extended = make_candles(closes[:35] + [5000, 7000, 9000, 11000, 13000, 15000, 17000, 19000, 21000, 23000])
    prefix_result = calculate_indicators(prefix)
    extended_result = calculate_indicators(extended)

    pd.testing.assert_frame_equal(
        prefix_result.loc[:, INDICATOR_COLUMNS],
        extended_result.loc[:34, INDICATOR_COLUMNS].reset_index(drop=True),
        check_exact=False,
        rtol=0,
        atol=0,
    )


def test_identical_input_produces_deterministic_result() -> None:
    candles = make_candles([100 + ((index * 3) % 11) for index in range(40)])

    first = calculate_indicators(candles)
    second = calculate_indicators(candles)

    pd.testing.assert_frame_equal(first, second)


def test_rejects_malformed_and_mixed_candle_input() -> None:
    candle = make_candles([100])[0]

    with pytest.raises(TypeError, match="iterable of Candle"):
        calculate_indicators(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="every input row"):
        calculate_indicators([candle, object()])  # type: ignore[list-item]

    other_symbol = candle.model_copy(update={"symbol": "ETHUSDT"})
    with pytest.raises(ValueError, match="same exchange, symbol, and interval"):
        calculate_indicators([candle, other_symbol])


def test_original_ohlcv_decimal_values_are_preserved_without_mutation() -> None:
    candles = make_candles([100.12345678 + index for index in range(25)])
    original_values = [
        tuple(getattr(candle, field) for field in ("open", "high", "low", "close", "volume"))
        for candle in candles
    ]

    result = calculate_indicators(candles)

    assert [
        tuple(getattr(candle, field) for field in ("open", "high", "low", "close", "volume"))
        for candle in candles
    ] == original_values
    for column in ("open", "high", "low", "close", "volume"):
        assert all(isinstance(value, Decimal) for value in result[column])
    assert result.loc[0, "close"] == original_values[0][3]


def test_empty_input_has_the_documented_columns() -> None:
    result = calculate_indicators([])

    assert result.empty
    assert tuple(result.columns) == (
        "exchange",
        "symbol",
        "interval",
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        *INDICATOR_COLUMNS,
    )

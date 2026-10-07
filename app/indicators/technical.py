"""Pure, causal technical indicators calculated from validated market candles.

Canonical candle values remain :class:`~decimal.Decimal` in the returned frame.
Calculations use NumPy ``float64`` values, and indicator columns are ``float64``
with ``NaN`` until enough history is available.

EMA is seeded with the first ``period``-value simple average, then uses the
recursive formula with alpha ``2 / (period + 1)``. RSI and ATR use Wilder
averages seeded by the first ``period`` gains/losses or true ranges, respectively.
The first candle's true range is ``high - low``; subsequent true ranges include
the previous close. MACD is EMA(12) minus EMA(26), with a seeded EMA(9) signal
and histogram equal to MACD minus signal. Bollinger Bands use a trailing SMA(20)
and population standard deviation (``ddof=0``), multiplied by two. Percentage
price change is ``(close / previous_close - 1) * 100``. No calculation uses a
future candle, and indicator values are never forward-filled.

Input must be aligned, contiguous, chronological, and contain candles for one
exchange, symbol, and interval. The function validates this contract instead
of sorting or repairing input.
"""

from collections.abc import Iterable
from decimal import Decimal
from typing import Final

import numpy as np
import pandas as pd

from app.market.models import Candle
from app.market.validation import validate_candle_sequence

CANDLE_COLUMNS: Final[tuple[str, ...]] = (
    "exchange",
    "symbol",
    "interval",
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
)

INDICATOR_COLUMNS: Final[tuple[str, ...]] = (
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
    "price_change_pct",
)

_DECIMAL_COLUMNS: Final[tuple[str, ...]] = ("open", "high", "low", "close", "volume")


def _recursive_ema(values: np.ndarray, period: int) -> np.ndarray:
    """Return an SMA-seeded recursive EMA, preserving leading NaN values."""
    result = np.full(values.shape, np.nan, dtype=np.float64)
    valid_indices = np.flatnonzero(np.isfinite(values))
    if valid_indices.size == 0:
        return result

    first = int(valid_indices[0])
    if not np.isfinite(values[first:]).all():
        raise ValueError("EMA input may contain leading missing values only")

    seed_index = first + period - 1
    if seed_index >= len(values):
        return result

    result[seed_index] = np.mean(values[first : seed_index + 1])
    alpha = 2.0 / (period + 1.0)
    for index in range(seed_index + 1, len(values)):
        result[index] = alpha * values[index] + (1.0 - alpha) * result[index - 1]
    return result


def _wilder_average(values: np.ndarray, period: int) -> np.ndarray:
    """Return Wilder's RMA, seeded with the first full-period arithmetic mean."""
    result = np.full(values.shape, np.nan, dtype=np.float64)
    valid_indices = np.flatnonzero(np.isfinite(values))
    if valid_indices.size == 0:
        return result

    first = int(valid_indices[0])
    if not np.isfinite(values[first:]).all():
        raise ValueError("Wilder average input may contain leading missing values only")

    seed_index = first + period - 1
    if seed_index >= len(values):
        return result

    result[seed_index] = np.mean(values[first : seed_index + 1])
    for index in range(seed_index + 1, len(values)):
        result[index] = (
            (period - 1) * result[index - 1] + values[index]
        ) / period
    return result


def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    changes = np.full(close.shape, np.nan, dtype=np.float64)
    changes[1:] = np.diff(close)
    gains = np.maximum(changes, 0.0)
    losses = np.maximum(-changes, 0.0)
    gains[0] = np.nan
    losses[0] = np.nan

    average_gain = _wilder_average(gains, period)
    average_loss = _wilder_average(losses, period)
    result = np.full(close.shape, np.nan, dtype=np.float64)

    ready = np.isfinite(average_gain) & np.isfinite(average_loss)
    flat = ready & (average_gain == 0.0) & (average_loss == 0.0)
    no_losses = ready & (average_loss == 0.0) & (average_gain > 0.0)
    ordinary = ready & (average_loss > 0.0)
    result[flat] = 50.0
    result[no_losses] = 100.0
    relative_strength = average_gain[ordinary] / average_loss[ordinary]
    result[ordinary] = 100.0 - (100.0 / (1.0 + relative_strength))
    return result


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    true_range = high - low
    true_range[1:] = np.maximum.reduce(
        (
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1]),
        )
    )
    return _wilder_average(true_range, period)


def _numeric_values(candles: tuple[Candle, ...], column: str) -> np.ndarray:
    values = np.asarray([float(getattr(candle, column)) for candle in candles], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{column} values must be finite and representable as float64")
    return values


def calculate_indicators(candles: Iterable[Candle]) -> pd.DataFrame:
    """Return candle fields plus causal technical indicators in input order.

    Candle OHLCV columns retain their original ``Decimal`` objects. Indicator
    calculations and outputs use NumPy/pandas ``float64``; unavailable warm-up
    values are ``NaN``. Input must contain only ``Candle`` values in strictly
    increasing ``open_time`` order, all for the same exchange, symbol, and
    interval. Empty input returns an empty frame with the complete column set.
    """
    try:
        candle_rows = tuple(candles)
    except TypeError as exc:
        raise TypeError("candles must be an iterable of Candle values") from exc

    columns = (*CANDLE_COLUMNS, *INDICATOR_COLUMNS)
    if not candle_rows:
        return pd.DataFrame(columns=columns)

    if any(not isinstance(candle, Candle) for candle in candle_rows):
        raise TypeError("every input row must be an app.market.models.Candle")

    validate_candle_sequence(candle_rows)

    numeric_values = {
        column: _numeric_values(candle_rows, column) for column in _DECIMAL_COLUMNS
    }
    high = numeric_values["high"]
    low = numeric_values["low"]
    close = numeric_values["close"]
    volume = numeric_values["volume"]

    records = [candle.model_dump() for candle in candle_rows]
    frame = pd.DataFrame.from_records(records, columns=CANDLE_COLUMNS)
    # Preserve exact canonical values rather than allowing pandas to infer a
    # binary-float dtype for Decimal-valued OHLCV columns.
    for column in _DECIMAL_COLUMNS:
        frame[column] = pd.Series(
            [getattr(candle, column) for candle in candle_rows],
            index=frame.index,
            dtype=object,
        )

    close_series = pd.Series(close, dtype=np.float64)
    for period in (20, 50, 200):
        frame[f"ema_{period}"] = _recursive_ema(close, period)
        frame[f"sma_{period}"] = close_series.rolling(period, min_periods=period).mean()

    frame["rsi_14"] = _rsi(close)

    macd = _recursive_ema(close, 12) - _recursive_ema(close, 26)
    signal = _recursive_ema(macd, 9)
    frame["macd"] = macd
    frame["macd_signal"] = signal
    frame["macd_histogram"] = macd - signal

    frame["atr_14"] = _atr(high, low, close)

    rolling_close = close_series.rolling(20, min_periods=20)
    bb_middle = rolling_close.mean()
    bb_width = rolling_close.std(ddof=0) * 2.0
    frame["bb_middle_20"] = bb_middle
    frame["bb_upper_20"] = bb_middle + bb_width
    frame["bb_lower_20"] = bb_middle - bb_width

    frame["volume_sma_20"] = pd.Series(volume, dtype=np.float64).rolling(
        20, min_periods=20
    ).mean()
    frame["price_change_pct"] = close_series.pct_change(fill_method=None) * 100.0

    return frame.loc[:, columns]

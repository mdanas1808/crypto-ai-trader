"""Deterministic breakout strategy signals for prepared backtest data."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
from typing import Literal

import pandas as pd

from app.backtest.data import BREAKOUT_CONTEXT_CANDLES, MultiTimeframeBacktestData


@dataclass(frozen=True)
class BreakoutSignal:
    """Strategy intent emitted at a completed 1H candle decision time."""

    exchange: str
    symbol: str
    side: Literal["long"]
    decision_time: datetime
    signal_open_time: datetime
    atr_14: float


def generate_breakout_signals(
    data: MultiTimeframeBacktestData,
) -> list[BreakoutSignal]:
    """Evaluate trading rows in order and return qualifying long signals.

    Context rows supply history only and are never evaluated as candidates.
    Numeric comparisons use float64 values, matching the indicator calculation
    convention; input DataFrames and canonical Decimal values are not modified.
    """
    if not isinstance(data, MultiTimeframeBacktestData):
        raise TypeError("data must be MultiTimeframeBacktestData")

    context = data.context_frame
    trading = data.trading_frame
    if not isinstance(context, pd.DataFrame) or not isinstance(trading, pd.DataFrame):
        raise TypeError("context_frame and trading_frame must be pandas DataFrames")

    _validate_timeline(context, "context_frame", context_required=True)
    _validate_timeline(trading, "trading_frame", context_required=False)
    required_trading = {
        "exchange", "symbol", "open_time", "high", "close", "volume",
        "ema_200", "rsi_14", "volume_sma_20", "atr_14",
        "4h_exchange", "4h_symbol",
        "4h_open_time", "4h_completion_time", "4h_close",
        "4h_ema_20", "4h_ema_50", "4h_ema_200",
    }
    _require_columns(trading, required_trading, "trading_frame")
    _require_columns(
        context, {"exchange", "symbol", "open_time", "high"}, "context_frame"
    )
    if trading.empty:
        return []

    context_identity = _single_series_identity(context)
    trading_identity = _single_series_identity(trading)
    if (not context.empty and context_identity is None) or trading_identity is None:
        return []
    if context_identity is not None and context_identity != trading_identity:
        return []

    context_open_times = list(context["open_time"])
    trading_open_times = list(trading["open_time"])
    if context_open_times and context_open_times[-1] >= trading_open_times[0]:
        raise ValueError("context_frame must precede trading_frame")

    combined_open_times = context_open_times + trading_open_times
    _validate_hourly_contiguity(combined_open_times)

    history_highs = [_finite_number(value) for value in context["high"]]
    signals: list[BreakoutSignal] = []

    for row_number in range(len(trading)):
        row = trading.iloc[row_number]
        decision_time = trading.index[row_number]
        open_time = _utc_timestamp(row["open_time"])
        decision_utc = _utc_timestamp(decision_time)

        if (
            open_time is not None
            and (row["exchange"], row["symbol"]) == trading_identity
            and decision_utc == open_time + timedelta(hours=1)
        ):
            candidate = _evaluate_candidate(row, decision_utc, open_time, history_highs)
            if candidate is not None:
                signals.append(candidate)

        # Append only after evaluating the current row so it cannot enter its
        # own previous-20-candle breakout window.
        history_highs.append(_finite_number(row["high"]))

    return signals


def _evaluate_candidate(
    row: pd.Series,
    decision_time: datetime,
    open_time: datetime,
    history_highs: list[float | None],
) -> BreakoutSignal | None:
    four_hour_open = _utc_timestamp(row["4h_open_time"])
    four_hour_completion = _utc_timestamp(row["4h_completion_time"])
    if four_hour_open is None or four_hour_completion is None:
        return None
    if row["4h_exchange"] != row["exchange"] or row["4h_symbol"] != row["symbol"]:
        return None
    if four_hour_open + timedelta(hours=4) != four_hour_completion:
        return None
    if four_hour_completion > decision_time:
        return None

    values = {
        name: _finite_number(row[name])
        for name in (
            "close", "ema_200", "rsi_14", "volume", "volume_sma_20", "atr_14",
            "4h_close", "4h_ema_20", "4h_ema_50", "4h_ema_200",
        )
    }
    if any(value is None for value in values.values()):
        return None

    atr = values["atr_14"]
    assert atr is not None
    if atr <= 0:
        return None

    prior_highs = history_highs[-BREAKOUT_CONTEXT_CANDLES:]
    if len(prior_highs) != BREAKOUT_CONTEXT_CANDLES or any(
        value is None for value in prior_highs
    ):
        return None
    reference_high = max(value for value in prior_highs if value is not None)

    close = values["close"]
    ema_200 = values["ema_200"]
    rsi_14 = values["rsi_14"]
    volume = values["volume"]
    volume_sma_20 = values["volume_sma_20"]
    four_hour_close = values["4h_close"]
    four_hour_ema_20 = values["4h_ema_20"]
    four_hour_ema_50 = values["4h_ema_50"]
    four_hour_ema_200 = values["4h_ema_200"]
    assert all(
        value is not None
        for value in (
            close, ema_200, rsi_14, volume, volume_sma_20, four_hour_close,
            four_hour_ema_20, four_hour_ema_50, four_hour_ema_200,
        )
    )

    if not (
        four_hour_ema_20 > four_hour_ema_50
        and four_hour_close > four_hour_ema_200
        and close > ema_200
        and 55.0 <= rsi_14 <= 70.0
        and volume >= 1.5 * volume_sma_20
        and close > reference_high
    ):
        return None

    exchange = row["exchange"]
    symbol = row["symbol"]
    if not isinstance(exchange, str) or not exchange or not isinstance(symbol, str) or not symbol:
        return None
    return BreakoutSignal(
        exchange=exchange,
        symbol=symbol,
        side="long",
        decision_time=decision_time,
        signal_open_time=open_time,
        atr_14=atr,
    )


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(sorted(missing))}")


def _single_series_identity(frame: pd.DataFrame) -> tuple[str, str] | None:
    """Return one valid exchange/symbol identity, or None for empty/invalid data."""
    if frame.empty:
        return None
    identities: set[tuple[str, str]] = set()
    for exchange, symbol in zip(frame["exchange"], frame["symbol"]):
        if (
            not isinstance(exchange, str)
            or not exchange
            or not isinstance(symbol, str)
            or not symbol
        ):
            return None
        identities.add((exchange, symbol))
        if len(identities) > 1:
            return None
    return next(iter(identities))


def _validate_timeline(
    frame: pd.DataFrame, name: str, *, context_required: bool
) -> None:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(f"{name} index must be a UTC DatetimeIndex")
    if str(frame.index.tz) != "UTC":
        raise ValueError(f"{name} index must be timezone-aware UTC")
    if frame.index.hasnans or not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise ValueError(f"{name} index must be strictly chronological")
    if context_required and not frame.empty and "open_time" not in frame.columns:
        raise ValueError("context_frame is missing required columns: open_time")


def _validate_hourly_contiguity(open_times: list[object]) -> None:
    normalized = [_utc_timestamp(value) for value in open_times]
    if any(value is None for value in normalized):
        raise ValueError("hourly open_time values must be timezone-aware timestamps")
    for previous, current in zip(normalized, normalized[1:]):
        assert previous is not None and current is not None
        if current - previous != timedelta(hours=1):
            raise ValueError("context and trading candles must be contiguous 1H rows")


def _utc_timestamp(value: object) -> datetime | None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        return None
    return value.astimezone(timezone.utc)


def _finite_number(value: object) -> float | None:
    try:
        if pd.isna(value):
            return None
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None

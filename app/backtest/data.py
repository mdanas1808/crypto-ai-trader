"""Prepare historical candles and indicators for offline backtests."""

from collections.abc import Iterable
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy.orm import Session

from app.indicators.technical import INDICATOR_COLUMNS, calculate_indicators
from app.market.constants import EXCHANGE
from app.market.repository import get_candles_in_range
from app.market.validation import interval_duration

MINIMUM_INDICATOR_HISTORY_CANDLES = 200
EXECUTION_INTERVAL = "1h"
REGIME_INTERVAL = "4h"


def candle_completion_times(
    open_times: Iterable[datetime], interval: str
) -> pd.DatetimeIndex:
    """Return UTC completion boundaries for aware candle open timestamps."""
    duration = interval_duration(interval)
    normalized: list[datetime] = []
    for value in open_times:
        if not isinstance(value, datetime):
            raise TypeError("candle open times must be datetime values")
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("candle open times must be timezone-aware")
        normalized.append(value.astimezone(timezone.utc) + duration)
    return pd.DatetimeIndex(pd.to_datetime(normalized, utc=True))


def align_timeframes(
    hourly_frame: pd.DataFrame, four_hour_frame: pd.DataFrame
) -> pd.DataFrame:
    """As-of align 4H candles to 1H completion times without future leakage.

    Inputs are expected to be the validated outputs of ``calculate_indicators``.
    The returned index is the UTC 1H decision timeline. All 4H columns are
    prefixed with ``4h_``; rows without an eligible completed 4H candle have
    missing 4H values.
    """
    _validate_indicator_frame(hourly_frame, EXECUTION_INTERVAL, "hourly_frame")
    _validate_indicator_frame(four_hour_frame, REGIME_INTERVAL, "four_hour_frame")
    if not hourly_frame.empty and not four_hour_frame.empty:
        if (hourly_frame["exchange"].iloc[0], hourly_frame["symbol"].iloc[0]) != (
            four_hour_frame["exchange"].iloc[0],
            four_hour_frame["symbol"].iloc[0],
        ):
            raise ValueError("hourly and four-hour frames must describe the same series")

    four_hour_columns = [f"4h_{column}" for column in four_hour_frame.columns]
    four_hour_columns.append("4h_completion_time")
    if hourly_frame.empty:
        result = hourly_frame.copy(deep=True)
        result["decision_time"] = pd.Series(dtype="datetime64[ns, UTC]")
        for column in four_hour_columns:
            result[column] = pd.Series(dtype=object)
        return result.set_index("decision_time", drop=True)

    left = hourly_frame.copy(deep=True).reset_index(drop=True)
    left["open_time"] = pd.to_datetime(left["open_time"], utc=True)
    left["decision_time"] = candle_completion_times(left["open_time"], EXECUTION_INTERVAL)
    right = four_hour_frame.copy(deep=True).reset_index(drop=True)
    right["open_time"] = pd.to_datetime(right["open_time"], utc=True)
    right["4h_completion_time"] = candle_completion_times(
        right["open_time"], REGIME_INTERVAL
    )
    right = right.rename(
        columns={column: f"4h_{column}" for column in four_hour_frame.columns}
    )

    if right.empty:
        aligned = left
        for column in four_hour_columns:
            aligned[column] = pd.Series([None] * len(aligned), dtype=object)
    else:
        aligned = pd.merge_asof(
            left.sort_values("decision_time"),
            right.sort_values("4h_completion_time"),
            left_on="decision_time",
            right_on="4h_completion_time",
            direction="backward",
            allow_exact_matches=True,
        )
    return aligned.set_index("decision_time", drop=True)


def prepare_multi_timeframe_backtest_data(
    session: Session,
    symbol: str,
    start_time: datetime,
    end_time: datetime,
    *,
    warmup_start_time: datetime | None = None,
    exchange: str = EXCHANGE,
) -> pd.DataFrame:
    """Prepare requested 1H decisions and causally aligned 4H indicators.

    Range bounds apply inclusively to candle ``open_time``. An optional
    ``warmup_start_time`` loads earlier data for indicator initialization, but
    returned decision rows still begin at ``start_time``. No warm-up is
    inferred or loaded automatically. EMA 200 first becomes available after
    200 observations (199 earlier candles plus the current candle).
    """
    start_time = _as_utc_bound(start_time, "start_time")
    end_time = _as_utc_bound(end_time, "end_time")
    if start_time > end_time:
        raise ValueError("start_time must be less than or equal to end_time")
    query_start = (
        _as_utc_bound(warmup_start_time, "warmup_start_time")
        if warmup_start_time is not None
        else start_time
    )
    if query_start > start_time:
        raise ValueError("warmup_start_time must be less than or equal to start_time")

    hourly_candles = get_candles_in_range(
        session, symbol, EXECUTION_INTERVAL, query_start, end_time, exchange
    )
    # Include enough preceding 4H opens for the most recent candle completed
    # by the first 1H decision, even when the requested start is mid-4H block.
    four_hour_query_start = query_start - 2 * interval_duration(REGIME_INTERVAL)
    four_hour_candles = get_candles_in_range(
        session, symbol, REGIME_INTERVAL, four_hour_query_start, end_time, exchange
    )
    hourly = calculate_indicators(hourly_candles)
    four_hour = calculate_indicators(four_hour_candles)
    aligned = align_timeframes(hourly, four_hour)

    requested_rows = aligned["open_time"].ge(start_time) & aligned["open_time"].le(end_time)
    return aligned.loc[requested_rows].copy(deep=True)


def _validate_indicator_frame(
    frame: pd.DataFrame, expected_interval: str, name: str
) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    required = {"exchange", "symbol", "open_time", "interval", *INDICATOR_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(sorted(missing))}")
    if not frame.empty:
        if frame["interval"].isna().any() or set(frame["interval"].unique()) != {expected_interval}:
            raise ValueError(f"{name} must contain only {expected_interval} candles")
        if frame["exchange"].isna().any() or frame["exchange"].nunique() != 1:
            raise ValueError(f"{name} must contain exactly one exchange")
        if frame["symbol"].isna().any() or frame["symbol"].nunique() != 1:
            raise ValueError(f"{name} must contain exactly one symbol")
        times = frame["open_time"]
        if times.isna().any() or not times.is_monotonic_increasing or not times.is_unique:
            raise ValueError(f"{name} must be strictly chronological by open_time")
        if any(
            not isinstance(value, datetime)
            or value.tzinfo is None
            or value.utcoffset() is None
            for value in times
        ):
            raise ValueError(f"{name} open_time values must be timezone-aware")


def _as_utc_bound(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def prepare_backtest_data(
    session: Session,
    symbol: str,
    interval: str,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    exchange: str = EXCHANGE,
) -> pd.DataFrame:
    """Read one bounded candle series and calculate its indicators in memory."""
    candles = get_candles_in_range(
        session,
        symbol=symbol,
        interval=interval,
        start_time=start_time,
        end_time=end_time,
        exchange=exchange,
    )
    return calculate_indicators(candles)

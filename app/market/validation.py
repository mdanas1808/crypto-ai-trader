"""Validation for chronological, gap-free candle sequences."""

import re
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone

from app.market.models import Candle

_UTC_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_FIXED_INTERVAL = re.compile(r"([1-9][0-9]*)([mhd])\Z")
_UNIT_SECONDS = {"m": 60, "h": 3600, "d": 86400}


def interval_duration(interval: str) -> timedelta:
    """Convert a fixed Binance minute/hour/day interval into a duration."""
    match = _FIXED_INTERVAL.fullmatch(interval)
    if match is None:
        raise ValueError(f"interval {interval!r} is not a supported fixed-duration interval")
    count, unit = match.groups()
    return timedelta(seconds=int(count) * _UNIT_SECONDS[unit])


class CandleSequenceValidationError(ValueError):
    """Invalid candle sequence with machine-readable context."""

    def __init__(
        self,
        *,
        reason: str,
        exchange: str,
        symbol: str,
        interval: str,
        previous_timestamp: datetime | None,
        next_timestamp: datetime,
        expected_timestamps: tuple[datetime, ...] = (),
    ) -> None:
        self.reason = reason
        self.exchange = exchange
        self.symbol = symbol
        self.interval = interval
        self.previous_timestamp = previous_timestamp
        self.next_timestamp = next_timestamp
        self.expected_timestamps = expected_timestamps
        self.missing_intervals = len(expected_timestamps) if reason == "gap" else 0

        if reason == "gap":
            detail = (
                f"{self.missing_intervals} candle interval(s) missing between "
                f"{previous_timestamp.isoformat()} and {next_timestamp.isoformat()}; "
                f"expected {', '.join(value.isoformat() for value in expected_timestamps)}"
            )
        elif reason == "misaligned":
            detail = (
                f"timestamp {next_timestamp.isoformat()} is not aligned; "
                f"expected {expected_timestamps[0].isoformat()}"
            )
        elif reason == "duplicate":
            detail = f"duplicate candle timestamp {next_timestamp.isoformat()}"
        elif reason == "out_of_order":
            detail = (
                f"out of order: timestamp {next_timestamp.isoformat()} is not after "
                f"{previous_timestamp.isoformat()}"
            )
        else:
            detail = reason
        super().__init__(f"invalid candle sequence for {exchange} {symbol} {interval}: {detail}")


def _aligned_timestamp(timestamp: datetime, duration: timedelta) -> datetime:
    elapsed = timestamp - _UTC_EPOCH
    return _UTC_EPOCH + duration * (elapsed // duration)


def validate_candle_sequence(candles: Iterable[Candle]) -> None:
    """Require same-series candles aligned and contiguous in ascending UTC time.

    Raises :class:`CandleSequenceValidationError` for duplicate, out-of-order,
    misaligned, or missing intervals. No candles are sorted or synthesized.
    """
    rows = tuple(candles)
    if not rows:
        return
    if any(not isinstance(candle, Candle) for candle in rows):
        raise TypeError("every input row must be an app.market.models.Candle")

    first = rows[0]
    exchange, symbol, interval = first.exchange, first.symbol, first.interval
    duration = interval_duration(interval)
    previous: Candle | None = None
    for candle in rows:
        if (candle.exchange, candle.symbol, candle.interval) != (exchange, symbol, interval):
            raise ValueError("all candles must have the same exchange, symbol, and interval")

        aligned = _aligned_timestamp(candle.open_time, duration)
        if candle.open_time != aligned:
            raise CandleSequenceValidationError(
                reason="misaligned",
                exchange=exchange,
                symbol=symbol,
                interval=interval,
                previous_timestamp=previous.open_time if previous else None,
                next_timestamp=candle.open_time,
                expected_timestamps=(aligned,),
            )

        if previous is not None:
            if candle.open_time == previous.open_time:
                reason = "duplicate"
            elif candle.open_time < previous.open_time:
                reason = "out_of_order"
            else:
                reason = ""
            if reason:
                raise CandleSequenceValidationError(
                    reason=reason,
                    exchange=exchange,
                    symbol=symbol,
                    interval=interval,
                    previous_timestamp=previous.open_time,
                    next_timestamp=candle.open_time,
                )

            elapsed = candle.open_time - previous.open_time
            if elapsed != duration:
                steps = elapsed // duration
                missing = tuple(previous.open_time + duration * index for index in range(1, steps))
                raise CandleSequenceValidationError(
                    reason="gap",
                    exchange=exchange,
                    symbol=symbol,
                    interval=interval,
                    previous_timestamp=previous.open_time,
                    next_timestamp=candle.open_time,
                    expected_timestamps=missing,
                )
        previous = candle

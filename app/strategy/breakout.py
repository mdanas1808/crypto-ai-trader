"""Deterministic breakout strategy signals for prepared backtest and live data.

Evaluates bidirectional (Long and Short) breakout setups across tracked symbols.
Scores signals by momentum strength (breakout distance normalized by ATR + volume surge ratio)
and provides dynamic leverage (5x–10x) and explicit Entry, Stop-Loss, and Take-Profit bounds.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Literal, Sequence

import pandas as pd

from app.backtest.data import BREAKOUT_CONTEXT_CANDLES, MultiTimeframeBacktestData


@dataclass(frozen=True)
class BreakoutSignal:
    """Strategy intent emitted at a completed 1H candle decision time."""

    exchange: str
    symbol: str
    side: Literal["long", "short", "LONG", "SHORT"]
    decision_time: datetime
    signal_open_time: datetime
    atr_14: float
    score: float = 0.0
    entry_price: Decimal | None = None
    stop_loss: Decimal | None = None
    take_profit: Decimal | None = None
    leverage: int = 1
    margin_used: Decimal = Decimal("0")

    @property
    def direction(self) -> str:
        return "LONG" if str(self.side).lower() in ("long", "buy") else "SHORT"


def compute_dynamic_leverage(
    atr: float | Decimal,
    price: float | Decimal,
    *,
    default_leverage: int = 7,
    min_leverage: int = 5,
    max_leverage: int = 10,
) -> int:
    """Compute dynamic leverage between min_leverage (5x) and max_leverage (10x).

    Higher ATR volatility relative to price scales leverage down towards min_leverage.
    Strong trend alignment and lower volatility scale leverage up towards max_leverage.
    Configurable via DEFAULT_LEVERAGE, MIN_LEVERAGE, MAX_LEVERAGE env vars.
    """
    try:
        def_lev = int(os.getenv("DEFAULT_LEVERAGE", str(default_leverage)))
        min_lev = int(os.getenv("MIN_LEVERAGE", str(min_leverage)))
        max_lev = int(os.getenv("MAX_LEVERAGE", str(max_leverage)))
        if min_lev > max_lev:
            min_lev, max_lev = max_lev, min_lev
    except Exception:
        def_lev, min_lev, max_lev = 7, 5, 10

    try:
        atr_f = float(atr)
        price_f = float(price)
        if atr_f <= 0 or price_f <= 0:
            return def_lev
        vol_ratio = atr_f / price_f
        # Reference bounds: 1.5% (low volatility/strong trend) -> max_leverage
        #                   3.5% (high volatility)            -> min_leverage
        if vol_ratio <= 0.015:
            return max_lev
        if vol_ratio >= 0.035:
            return min_lev
        ratio = (vol_ratio - 0.015) / (0.035 - 0.015)
        lev = max_lev - ratio * (max_lev - min_lev)
        return int(round(max(min_lev, min(max_lev, lev))))
    except Exception:
        return def_lev


def score_signal(
    side: str,
    close: float,
    reference_level: float,
    atr: float,
    volume: float,
    volume_sma: float,
) -> float:
    """Score a breakout signal by momentum strength:

    momentum strength = (breakout distance normalized by ATR) + volume surge ratio
    """
    if atr <= 0 or volume_sma <= 0:
        return 0.0
    if str(side).lower() == "long":
        breakout_distance = max(0.0, close - reference_level)
    else:
        breakout_distance = max(0.0, reference_level - close)
    normalized_distance = breakout_distance / atr
    volume_surge_ratio = volume / volume_sma
    return normalized_distance + volume_surge_ratio


def pick_most_feasible_signal(signals: Sequence[BreakoutSignal]) -> BreakoutSignal | None:
    """Pick ONLY the single highest-scoring setup ("most feasible trade")."""
    if not signals:
        return None
    return max(signals, key=lambda s: getattr(s, "score", 0.0))


def generate_breakout_signals(
    data: MultiTimeframeBacktestData,
    *,
    bidirectional: bool = True,
    legacy_mode: bool = False,
) -> list[BreakoutSignal]:
    """Evaluate trading rows in order and return qualifying signals.

    Context rows supply history only and are never evaluated as candidates.
    Evaluates both Long and Short setups across closed candles.
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
        "exchange",
        "symbol",
        "open_time",
        "high",
        "close",
        "volume",
        "ema_200",
        "rsi_14",
        "volume_sma_20",
        "atr_14",
    }
    if legacy_mode:
        required_trading.update(
            {
                "4h_exchange",
                "4h_symbol",
                "4h_open_time",
                "4h_completion_time",
                "4h_close",
                "4h_ema_20",
                "4h_ema_50",
                "4h_ema_200",
            }
        )
    _require_columns(trading, required_trading, "trading_frame")
    _require_columns(context, {"exchange", "symbol", "open_time", "high"}, "context_frame")

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
    history_lows = (
        [_finite_number(value) for value in context["low"]]
        if "low" in context.columns
        else [_finite_number(value) for value in context["high"]]
    )
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
            candidate = _evaluate_candidate(
                row,
                decision_utc,
                open_time,
                history_highs,
                history_lows,
                bidirectional=bidirectional,
                legacy_mode=legacy_mode,
            )
            if candidate is not None:
                signals.append(candidate)

        # Append only after evaluating the current row so it cannot enter its
        # own previous-20-candle breakout window.
        history_highs.append(_finite_number(row["high"]))
        cur_low = row["low"] if "low" in row else row["high"]
        history_lows.append(_finite_number(cur_low))

    return signals


def _evaluate_candidate(
    row: pd.Series,
    decision_time: datetime,
    open_time: datetime,
    history_highs: list[float | None],
    history_lows: list[float | None],
    *,
    bidirectional: bool = True,
    legacy_mode: bool = False,
) -> BreakoutSignal | None:
    if legacy_mode:
        return _evaluate_legacy_candidate(row, decision_time, open_time, history_highs)

    values = {
        name: _finite_number(row[name])
        for name in (
            "close",
            "ema_200",
            "rsi_14",
            "volume",
            "volume_sma_20",
            "atr_14",
        )
    }
    if any(value is None for value in values.values()):
        return None

    atr = values["atr_14"]
    assert atr is not None
    if atr <= 0:
        return None

    prior_highs = history_highs[-BREAKOUT_CONTEXT_CANDLES:]
    prior_lows = history_lows[-BREAKOUT_CONTEXT_CANDLES:]
    if (
        len(prior_highs) != BREAKOUT_CONTEXT_CANDLES
        or any(value is None for value in prior_highs)
        or len(prior_lows) != BREAKOUT_CONTEXT_CANDLES
        or any(value is None for value in prior_lows)
    ):
        return None

    reference_high = max(value for value in prior_highs if value is not None)
    reference_low = min(value for value in prior_lows if value is not None)

    close = values["close"]
    ema_200 = values["ema_200"]
    rsi_14 = values["rsi_14"]
    volume = values["volume"]
    volume_sma_20 = values["volume_sma_20"]
    assert all(value is not None for value in (close, ema_200, rsi_14, volume, volume_sma_20))

    exchange = row["exchange"]
    symbol = row["symbol"]
    if not isinstance(exchange, str) or not exchange or not isinstance(symbol, str) or not symbol:
        return None

    # Long criteria: Close > 200 EMA, Close > 20-period High, Volume > 20-period Volume SMA, bullish RSI (> 52 and < 75)
    is_long = (
        close > ema_200
        and close > reference_high
        and volume > volume_sma_20
        and 52.0 < rsi_14 < 75.0
    )

    # Short criteria: Close < 200 EMA, Close < 20-period Low, Volume > 20-period Volume SMA, bearish RSI (< 48 and > 25)
    is_short = (
        bidirectional
        and close < ema_200
        and close < reference_low
        and volume > volume_sma_20
        and 25.0 < rsi_14 < 48.0
    )

    if not is_long and not is_short:
        return None

    side: Literal["long", "short"] = "long" if is_long else "short"
    ref_level = reference_high if is_long else reference_low
    score = score_signal(side, close, ref_level, atr, volume, volume_sma_20)
    leverage = compute_dynamic_leverage(atr, close)

    entry_dec = Decimal(str(close))
    atr_dec = Decimal(str(atr))
    if is_long:
        stop_loss = entry_dec - Decimal("1.5") * atr_dec
        take_profit = entry_dec + Decimal("3.0") * atr_dec
    else:
        stop_loss = entry_dec + Decimal("1.5") * atr_dec
        take_profit = entry_dec - Decimal("3.0") * atr_dec

    if stop_loss <= Decimal("0") or take_profit <= Decimal("0"):
        return None

    return BreakoutSignal(
        exchange=exchange,
        symbol=symbol,
        side=side,
        decision_time=decision_time,
        signal_open_time=open_time,
        atr_14=atr,
        score=score,
        entry_price=entry_dec,
        stop_loss=stop_loss,
        take_profit=take_profit,
        leverage=leverage,
    )


def _evaluate_legacy_candidate(
    row: pd.Series,
    decision_time: datetime,
    open_time: datetime,
    history_highs: list[float | None],
) -> BreakoutSignal | None:
    four_hour_open = _utc_timestamp(row.get("4h_open_time"))
    four_hour_completion = _utc_timestamp(row.get("4h_completion_time"))
    if four_hour_open is None or four_hour_completion is None:
        return None
    if row.get("4h_exchange") != row["exchange"] or row.get("4h_symbol") != row["symbol"]:
        return None
    if four_hour_open + timedelta(hours=4) != four_hour_completion:
        return None
    if four_hour_completion > decision_time:
        return None

    values = {
        name: _finite_number(row.get(name))
        for name in (
            "close",
            "ema_200",
            "rsi_14",
            "volume",
            "volume_sma_20",
            "atr_14",
            "4h_close",
            "4h_ema_20",
            "4h_ema_50",
            "4h_ema_200",
        )
    }
    if any(value is None for value in values.values()):
        return None

    atr = values["atr_14"]
    assert atr is not None
    if atr <= 0:
        return None

    prior_highs = history_highs[-BREAKOUT_CONTEXT_CANDLES:]
    if len(prior_highs) != BREAKOUT_CONTEXT_CANDLES or any(value is None for value in prior_highs):
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
            close,
            ema_200,
            rsi_14,
            volume,
            volume_sma_20,
            four_hour_close,
            four_hour_ema_20,
            four_hour_ema_50,
            four_hour_ema_200,
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

    entry_dec = Decimal(str(close))
    atr_dec = Decimal(str(atr))
    stop_loss = entry_dec - Decimal("1.5") * atr_dec
    take_profit = entry_dec + Decimal("3.0") * atr_dec
    score = score_signal("long", close, reference_high, atr, volume, volume_sma_20)
    leverage = compute_dynamic_leverage(atr, close)

    return BreakoutSignal(
        exchange=exchange,
        symbol=symbol,
        side="long",
        decision_time=decision_time,
        signal_open_time=open_time,
        atr_14=atr,
        score=score,
        entry_price=entry_dec,
        stop_loss=stop_loss,
        take_profit=take_profit,
        leverage=leverage,
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


def _validate_timeline(frame: pd.DataFrame, name: str, *, context_required: bool) -> None:
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

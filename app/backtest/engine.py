"""Causal walk-forward and deterministic offline execution orchestration."""

from collections.abc import Iterator, Mapping
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pandas as pd

from app.backtest.data import MultiTimeframeBacktestData
from app.backtest.models import BacktestConfig, BacktestResult, Position
from app.backtest.portfolio import Portfolio
from app.market.constants import EXCHANGE
from app.strategy.breakout import BreakoutSignal, generate_breakout_signals


def iter_walk_forward(indicator_frame: pd.DataFrame) -> Iterator[pd.DataFrame]:
    """Yield independent chronological prefixes, one current candle per step.

    At step ``i`` the returned frame contains rows ``0`` through ``i`` only.
    The input must already be in strictly increasing ``open_time`` order; it is
    validated rather than sorted. Each yielded prefix is a copy, so consumers
    cannot mutate the original frame through a yielded value.
    """
    if not isinstance(indicator_frame, pd.DataFrame):
        raise TypeError("indicator_frame must be a pandas DataFrame")
    if "open_time" not in indicator_frame.columns:
        raise ValueError("indicator_frame must contain an open_time column")

    open_times = indicator_frame["open_time"]
    if open_times.isna().any() or not open_times.is_monotonic_increasing or not open_times.is_unique:
        raise ValueError("indicator_frame must be strictly chronological by open_time")

    for stop in range(1, len(indicator_frame) + 1):
        yield indicator_frame.iloc[:stop].copy(deep=True)


def run_backtest(
    streams: Mapping[str, MultiTimeframeBacktestData],
    *,
    end_time: datetime,
    config: BacktestConfig,
) -> BacktestResult:
    """Run one shared spot portfolio over prepared per-symbol streams.

    ``end_time`` is an inclusive UTC candle-open bound, matching data
    preparation. Rows beyond it are ignored. Exit evaluation uses the completed
    candle row at each decision time; signals at that time can enter only at
    the next row whose ``open_time`` equals that decision time.
    """
    if not isinstance(streams, Mapping):
        raise TypeError("streams must map symbols to MultiTimeframeBacktestData")
    cutoff = _utc(end_time, "end_time")
    portfolio = Portfolio(config)
    rows_by_time: dict[datetime, dict[str, pd.Series]] = {}
    rows_by_open: dict[str, dict[datetime, pd.Series]] = {}
    final_row_by_symbol: dict[str, pd.Series] = {}
    signals_by_time: dict[datetime, list[BreakoutSignal]] = {}
    symbols_seen: set[str] = set()

    for key, data in streams.items():
        if not isinstance(data, MultiTimeframeBacktestData):
            raise TypeError(f"stream {key!r} must be MultiTimeframeBacktestData")
        frame = data.trading_frame
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"stream {key!r} trading_frame must be a DataFrame")
        if frame.empty:
            continue
        required = {"exchange", "symbol", "open_time", "open", "high", "low", "close"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"stream {key!r} is missing columns: {', '.join(sorted(missing))}")
        if not isinstance(frame.index, pd.DatetimeIndex) or str(frame.index.tz) != "UTC":
            raise ValueError(f"stream {key!r} index must be a UTC DatetimeIndex")
        if frame.index.hasnans or not frame.index.is_monotonic_increasing or not frame.index.is_unique:
            raise ValueError(f"stream {key!r} must be strictly chronological")
        identities = set(zip(frame["exchange"], frame["symbol"]))
        if len(identities) != 1:
            raise ValueError(f"stream {key!r} must contain one exchange and symbol")
        exchange, symbol = next(iter(identities))
        if exchange != EXCHANGE or not isinstance(symbol, str) or not symbol.endswith("USDT"):
            raise ValueError("backtest streams must be Binance Spot USDT symbols")
        if symbol != key:
            raise ValueError(f"stream key {key!r} does not match contained symbol {symbol!r}")
        if symbol in symbols_seen:
            raise ValueError(f"duplicate symbol stream: {symbol}")
        symbols_seen.add(symbol)

        bounded = frame.loc[
            frame["open_time"].map(lambda value: _utc(value, "open_time") <= cutoff)
        ]
        if bounded.empty:
            continue
        signals = generate_breakout_signals(
            MultiTimeframeBacktestData(data.context_frame, bounded.copy(deep=True))
        )
        for signal in signals:
            if signal.symbol != symbol or signal.exchange != exchange:
                raise ValueError("strategy signal identity does not match its stream")
            signals_by_time.setdefault(_utc(signal.decision_time, "decision_time"), []).append(signal)
        for decision_value, row in bounded.iterrows():
            decision_time = _utc(decision_value, "decision_time")
            open_time = _utc(row["open_time"], "open_time")
            rows_by_time.setdefault(decision_time, {})[symbol] = row
            if open_time in rows_by_open.setdefault(symbol, {}):
                raise ValueError(f"duplicate open_time in {symbol} stream")
            rows_by_open[symbol][open_time] = row
            final_row_by_symbol[symbol] = row

    timeline = sorted(rows_by_time)
    if not timeline:
        return BacktestResult((), portfolio.cash, portfolio.cash, ())

    last_marks: dict[str, Decimal] = {}
    for timestamp in timeline:
        current_rows = rows_by_time[timestamp]
        # All exits at this decision boundary are settled before any new signal.
        for symbol in sorted(current_rows):
            row = current_rows[symbol]
            close = _decimal_price(row["close"], "close")
            last_marks[symbol] = close
            position = portfolio.positions.get(symbol)
            if position is None:
                continue
            reason, reference = _exit_condition(position, row)
            if reason is not None:
                portfolio.exit(symbol, timestamp=timestamp, reference_price=reference, reason=reason)

        # Establish a new UTC day's baseline after prior-bar exits but before
        # any opening-gap movement at this timestamp is applied.
        portfolio.begin_day(timestamp, last_marks)

        # Opening gaps occur at the beginning of the next candle, which shares
        # the just-completed candle's decision timestamp. Resolve these before
        # same-time signals, using only the new candle's open (not its future
        # high/low/close).
        opening_rows = {
            symbol: rows.get(timestamp)
            for symbol, rows in rows_by_open.items()
            if timestamp in rows
        }
        for symbol in sorted(opening_rows):
            row = opening_rows[symbol]
            assert row is not None
            if symbol in portfolio.positions:
                position = portfolio.positions[symbol]
                reason, reference = _opening_exit_condition(position, row)
                if reason is not None:
                    portfolio.exit(
                        symbol, timestamp=timestamp, reference_price=reference,
                        reason=reason,
                    )
            last_marks[symbol] = _decimal_price(row["open"], "open")

        portfolio.update_day(timestamp, last_marks)
        # Signal fills happen at the open of the next candle, which begins at
        # this completed candle's decision timestamp.
        for signal in sorted(
            signals_by_time.get(timestamp, []), key=lambda item: item.symbol
        ):
            next_row = rows_by_open.get(signal.symbol, {}).get(timestamp)
            if next_row is None:
                continue
            next_open = _decimal_price(next_row["open"], "open")
            portfolio.enter(
                signal,
                timestamp=timestamp,
                next_open=next_open,
                marks=last_marks,
            )

    portfolio_end = timeline[-1]
    # Close any residual positions at the final available close for their own
    # stream; no stream is extended beyond its last in-range candle.
    for symbol in sorted(tuple(portfolio.positions)):
        final_row = final_row_by_symbol.get(symbol)
        if final_row is None:
            continue
        reference = _decimal_price(final_row["close"], "close")
        portfolio.exit(
            symbol,
            timestamp=portfolio_end,
            reference_price=reference,
            reason="end_of_backtest",
        )

    final_equity = portfolio.equity(last_marks)
    return BacktestResult(
        trades=tuple(portfolio.trades),
        final_cash=portfolio.cash,
        final_equity=final_equity,
        final_open_positions=tuple(portfolio.positions.values()),
    )


def _exit_condition(position: Position, row: pd.Series) -> tuple[str | None, Decimal]:
    open_price = _decimal_price(row["open"], "open")
    high = _decimal_price(row["high"], "high")
    low = _decimal_price(row["low"], "low")
    stop = position.stop_price
    target = position.target_price

    if open_price <= stop:
        return "stop_loss", open_price
    if open_price >= target:
        if low <= stop:
            return "stop_loss", stop
        return "take_profit", open_price
    stop_hit = low <= stop
    target_hit = high >= target
    if stop_hit:  # conservative stop-first behavior, including a dual-touch bar
        return "stop_loss", stop
    if target_hit:
        return "take_profit", target
    return None, open_price


def _opening_exit_condition(
    position: Position, row: pd.Series
) -> tuple[str | None, Decimal]:
    """Resolve opening exits, preserving stop-first dual-touch semantics."""
    open_price = _decimal_price(row["open"], "open")
    low = _decimal_price(row["low"], "low")
    if open_price <= position.stop_price:
        return "stop_loss", open_price
    if open_price >= position.target_price:
        if low <= position.stop_price:
            return "stop_loss", position.stop_price
        return "take_profit", open_price
    return None, open_price


def _utc(value: object, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def _decimal_price(value: object, name: str) -> Decimal:
    if not isinstance(value, Decimal) or not value.is_finite() or value <= 0:
        raise ValueError(f"{name} must be a finite positive Decimal")
    return value

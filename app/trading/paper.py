"""In-memory real-time paper execution for completed breakout signals.

The executor accepts strategy intent separately from market events. Only a
first-seen CandleOpenObservation can fill a pending entry; completed Candle
objects manage existing positions and are never used to reconstruct an entry.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Literal

from app.backtest.models import BacktestConfig, Fill, Position, Trade
from app.backtest.portfolio import Portfolio, get_asset_category
from app.market.constants import EXCHANGE, FIXED_SYMBOLS, SUPPORTED_SYMBOLS
from app.market.models import Candle, CandleOpenObservation
from app.strategy.breakout import BreakoutSignal

_ONE_HOUR = timedelta(hours=1)


class PaperExecutor:
    """Coordinate live paper entries and completed-candle position exits."""

    def __init__(self, config: BacktestConfig) -> None:
        if not isinstance(config, BacktestConfig):
            raise TypeError("config must be BacktestConfig")
        self.config = config
        self.portfolio = Portfolio(config, split_risk=True)
        self._pending: dict[str, BreakoutSignal] = {}
        self._last_closed: dict[str, Candle] = {}
        self._last_open_observation: dict[str, datetime] = {}
        self._marks: dict[str, Decimal] = {}

    @staticmethod
    def get_asset_category(symbol: str) -> Literal["core", "satellite"]:
        """Determine asset category ('core' if symbol in FIXED_SYMBOLS else 'satellite')."""
        return get_asset_category(symbol)

    def calculate_dollar_risk(
        self, symbol: str, equity: Decimal | None = None
    ) -> Decimal:
        """Calculate dollar risk for a symbol using category-specific sizing."""
        current_equity = (
            equity if equity is not None else self.portfolio.equity(self._marks)
        )
        return self.portfolio.calculate_dollar_risk(symbol, current_equity)

    @property
    def pending_signals(self) -> tuple[BreakoutSignal, ...]:
        return tuple(self._pending[symbol] for symbol in sorted(self._pending))

    @property
    def positions(self) -> dict[str, Position]:
        return self.portfolio.positions.copy()

    @property
    def trades(self) -> tuple[Trade, ...]:
        return tuple(self.portfolio.trades)

    @property
    def cash(self) -> Decimal:
        return self.portfolio.cash

    @property
    def daily_loss_breached(self) -> bool:
        return self.portfolio.daily_loss_breached

    @property
    def core_daily_loss_breached(self) -> bool:
        return self.portfolio.core_daily_loss_breached

    @property
    def satellite_daily_loss_breached(self) -> bool:
        return self.portfolio.satellite_daily_loss_breached

    def submit_signal(self, signal: BreakoutSignal) -> bool:
        """Queue a signal only while its next-candle open has not been observed."""
        if not isinstance(signal, BreakoutSignal):
            raise TypeError("signal must be BreakoutSignal")
        if (
            signal.exchange != EXCHANGE
            or signal.symbol not in SUPPORTED_SYMBOLS
            or signal.side != "long"
            or not _aware_utc(signal.decision_time)
            or not _aware_utc(signal.signal_open_time)
            or signal.decision_time.astimezone(timezone.utc)
            != signal.signal_open_time.astimezone(timezone.utc) + _ONE_HOUR
        ):
            return False
        atr = _positive_decimal_from_indicator(signal.atr_14)
        if atr is None:
            return False

        latest = self._last_closed.get(signal.symbol)
        if latest is None or latest.exchange != signal.exchange:
            return False
        if latest.open_time != signal.signal_open_time.astimezone(timezone.utc):
            return False
        if signal.symbol in self.portfolio.positions or signal.symbol in self._pending:
            return False
        previously_observed = self._last_open_observation.get(signal.symbol)
        if previously_observed is not None and previously_observed >= signal.decision_time.astimezone(timezone.utc):
            return False
        self._pending[signal.symbol] = signal
        return True

    def on_open_observation(self, observation: CandleOpenObservation) -> Position | None:
        """Fill a previously queued signal from the first observed next open."""
        if not isinstance(observation, CandleOpenObservation):
            raise TypeError("observation must be CandleOpenObservation")
        if observation.exchange != EXCHANGE or observation.interval != "1h":
            return None
        symbol = observation.symbol
        open_time = observation.open_time.astimezone(timezone.utc)
        previous = self._last_open_observation.get(symbol)
        if previous is not None and open_time <= previous:
            return None
        self._last_open_observation[symbol] = open_time

        # Daily boundaries use only the latest completed-candle marks. The
        # observed forming open is never used to mark risk or portfolio equity.
        self.portfolio.begin_day(open_time, self._marks)
        signal = self._pending.get(symbol)
        if signal is None or signal.decision_time.astimezone(timezone.utc) != open_time:
            if signal is not None and signal.decision_time.astimezone(timezone.utc) < open_time:
                self._pending.pop(symbol, None)
            return None

        position = self.portfolio.enter(
            signal,
            timestamp=open_time,
            next_open=observation.open,
            marks=self._marks,
            observed_at=observation.observed_at,
        )
        self._pending.pop(symbol, None)
        return position

    def on_candle_closed(self, candle: Candle) -> Trade | None:
        """Manage positions using one completed 1H candle; never fill entries."""
        if not isinstance(candle, Candle):
            raise TypeError("candle must be a completed Candle")
        if candle.exchange != EXCHANGE or candle.interval != "1h":
            raise ValueError("paper execution requires Binance Spot 1h candles")
        symbol = candle.symbol
        previous = self._last_closed.get(symbol)
        if previous is not None:
            expected = previous.open_time + _ONE_HOUR
            if candle.open_time <= previous.open_time:
                raise ValueError("completed candles must be strictly chronological")
            if candle.open_time != expected:
                raise ValueError("completed 1h candles must be contiguous")

        decision_time = candle.open_time + _ONE_HOUR
        self.portfolio.begin_day(decision_time, self._marks)

        # A closed candle with no preceding open observation cannot satisfy a
        # pending entry retroactively. Expire its intent before evaluating exits.
        pending = self._pending.get(symbol)
        if pending is not None and pending.decision_time.astimezone(timezone.utc) <= candle.open_time:
            self._pending.pop(symbol, None)

        trade = None
        position = self.portfolio.positions.get(symbol)
        if position is not None:
            reason, reference = _exit_condition(position, candle)
            if reason is not None:
                trade = self.portfolio.exit(
                    symbol,
                    timestamp=decision_time,
                    reference_price=reference,
                    reason=reason,
                )

        self._marks[symbol] = candle.close
        self._last_closed[symbol] = candle
        self.portfolio.update_day(decision_time, self._marks)
        return trade

    def close_position(
        self,
        symbol: str,
        *,
        reason: str = "close",
        timestamp: datetime | None = None,
        reference_price: Decimal | None = None,
    ) -> Trade | None:
        """Close an active position immediately at reference mark or entry price."""
        pos = self.portfolio.positions.get(symbol)
        if pos is None:
            return None
        now = timestamp if timestamp is not None else datetime.now(timezone.utc)
        ref = (
            reference_price
            if reference_price is not None
            else self._marks.get(symbol, pos.entry_fill.price)
        )
        trade = self.portfolio.exit(
            symbol,
            timestamp=now,
            reference_price=ref,
            reason=reason,
        )
        return trade

    def reduce_position(
        self,
        symbol: str,
        *,
        fraction: Decimal = Decimal("0.5"),
        reduce_ratio: Decimal | float | None = None,
        reason: str = "reduce",
        timestamp: datetime | None = None,
        reference_price: Decimal | None = None,
    ) -> Trade | None:
        """Reduce an active position's size or close it if below minimum quantity."""
        if reduce_ratio is not None:
            fraction = Decimal(str(reduce_ratio))
        pos = self.portfolio.positions.get(symbol)
        if pos is None:
            return None
        now = timestamp if timestamp is not None else datetime.now(timezone.utc)
        ref = (
            reference_price
            if reference_price is not None
            else self._marks.get(symbol, pos.entry_fill.price)
        )
        if hasattr(self.portfolio, "reduce") and callable(self.portfolio.reduce):
            return self.portfolio.reduce(
                symbol,
                fraction=fraction,
                reduce_ratio=reduce_ratio,
                timestamp=now,
                reference_price=ref,
                reason=reason,
            )
        return self.portfolio.exit(
            symbol,
            timestamp=now,
            reference_price=ref,
            reason=reason,
        )



def _exit_condition(position: Position, candle: Candle) -> tuple[str | None, Decimal]:
    """Use the same completed-bar, stop-first semantics as the backtest engine."""
    open_price = candle.open
    if open_price <= position.stop_price:
        return "stop_loss", open_price
    if open_price >= position.target_price:
        if candle.low <= position.stop_price:
            return "stop_loss", position.stop_price
        return "take_profit", open_price
    if candle.low <= position.stop_price:
        return "stop_loss", position.stop_price
    if candle.high >= position.target_price:
        return "take_profit", position.target_price
    return None, open_price


def _aware_utc(value: object) -> bool:
    return isinstance(value, datetime) and value.tzinfo is not None and value.utcoffset() is not None


def _positive_decimal_from_indicator(value: object) -> Decimal | None:
    # BreakoutSignal indicators are float64 by design. Convert via str to
    # preserve their displayed numeric value before Decimal execution math.
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return result if result.is_finite() and result > 0 else None

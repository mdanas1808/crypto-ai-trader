"""In-memory real-time paper execution for leveraged Futures breakout trading.

Implements a single-position, bidirectional (Long & Short) leveraged Futures paper-trading system
starting with $100.00 capital, dynamic 5x–10x leverage, 95% margin allocation, 0.04% Binance Futures fees,
explicit 1.5x ATR Stop-Loss and 3.0x ATR Take-Profit, and isolated liquidation protection.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Literal

from app.backtest.execution import round_quantity_down
from app.backtest.models import BacktestConfig, Fill, Position, Trade
from app.market.constants import EXCHANGE, SUPPORTED_SYMBOLS
from app.market.models import Candle, CandleOpenObservation
from app.strategy.breakout import BreakoutSignal, compute_dynamic_leverage

logger = logging.getLogger(__name__)

_ONE_HOUR = timedelta(hours=1)
CORE_SLIPPAGE = Decimal("0.0005")  # 0.05%
SATELLITE_SLIPPAGE = Decimal("0.0020")  # 0.20%
FUTURES_FEE_RATE = Decimal("0.0004")  # 0.04% Binance Futures fee
MARGIN_ALLOCATION_FRACTION = Decimal("0.95")  # 95% margin, 5% fee buffer
DEFAULT_STARTING_CASH = Decimal("100.00")
ATR_STOP_MULTIPLIER = Decimal("1.5")
TARGET_R_MULTIPLIER = Decimal("3.0")


class FuturesPortfolio:
    """Isolated margin leveraged futures portfolio managing a single active trade."""

    def __init__(
        self,
        config: BacktestConfig,
        *,
        starting_cash: Decimal = DEFAULT_STARTING_CASH,
        max_open_positions: int = 1,
        fee_rate: Decimal = FUTURES_FEE_RATE,
        margin_fraction: Decimal = MARGIN_ALLOCATION_FRACTION,
    ) -> None:
        self.config = config
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.max_open_positions = max_open_positions
        self.fee_rate = fee_rate
        self.margin_fraction = margin_fraction
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self.realized_net_pnl = Decimal("0")
        self._day: object | None = None
        self._start_of_day_equity = starting_cash
        self._daily_loss_breached = False

    def equity(self, marks: dict[str, Decimal] | None = None) -> Decimal:
        """Calculate total account equity = cash + margin_used + unrealized_pnl."""
        if marks is None:
            marks = {}
        margin_total = sum((p.margin_used for p in self.positions.values()), Decimal("0"))
        unrealized = Decimal("0")
        for sym, pos in self.positions.items():
            mark = marks.get(sym, pos.entry_fill.price)
            if pos.direction == "LONG":
                unrealized += (mark - pos.entry_fill.price) * pos.quantity
            else:
                unrealized += (pos.entry_fill.price - mark) * pos.quantity
        return max(Decimal("0"), self.cash + margin_total + unrealized)

    @property
    def open_risk(self) -> Decimal:
        return sum((p.initial_risk for p in self.positions.values()), Decimal("0"))

    @property
    def daily_loss_breached(self) -> bool:
        return self._daily_loss_breached

    @property
    def core_daily_loss_breached(self) -> bool:
        return self._daily_loss_breached

    @property
    def satellite_daily_loss_breached(self) -> bool:
        return self._daily_loss_breached

    def begin_day(self, timestamp: datetime, marks: dict[str, Decimal]) -> None:
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("portfolio timestamps must be timezone-aware")
        utc_timestamp = timestamp.astimezone(timezone.utc)
        current_day = utc_timestamp.date()
        if self._day != current_day:
            self._day = current_day
            self._start_of_day_equity = self.equity(marks)
            self._daily_loss_breached = False

    def update_day(self, timestamp: datetime, marks: dict[str, Decimal]) -> None:
        self.begin_day(timestamp, marks)
        current_eq = self.equity(marks)
        daily_loss = self._start_of_day_equity - current_eq
        # Default 5% daily loss limit
        if daily_loss >= self._start_of_day_equity * Decimal(
            str(self.config.core_daily_loss_limit)
        ):
            self._daily_loss_breached = True

    def calculate_dollar_risk(self, symbol: str, equity: Decimal | None = None) -> Decimal:
        eq = equity if equity is not None else self.cash
        return eq * Decimal(str(self.config.core_risk_per_trade))

    def enter(
        self,
        signal: BreakoutSignal,
        *,
        timestamp: datetime,
        next_open: Decimal,
        marks: dict[str, Decimal],
        observed_at: datetime | None = None,
        slippage: Decimal = CORE_SLIPPAGE,
    ) -> Position | None:
        """Enter a leveraged Long or Short position using 95% margin."""
        if len(self.positions) >= self.max_open_positions or signal.symbol in self.positions:
            return None
        if self._daily_loss_breached:
            return None

        current_equity = self.equity(marks)
        if current_equity <= Decimal("0"):
            return None

        # 1. Determine direction and adverse entry fill price
        is_long = str(signal.side).lower() in ("long", "buy")
        if is_long:
            fill_price = next_open * (Decimal("1") + slippage)
            side_str: Literal["buy", "sell"] = "buy"
        else:
            fill_price = next_open * (Decimal("1") - slippage)
            side_str = "sell"

        if fill_price <= Decimal("0"):
            return None

        # 2. Determine dynamic leverage
        leverage = getattr(signal, "leverage", 1)
        if not leverage or leverage <= 1:
            leverage = compute_dynamic_leverage(signal.atr_14, fill_price)
        leverage = max(1, int(leverage))

        # 3. Calculate position sizing: margin = 95% equity, notional = margin * leverage
        allocated_margin = current_equity * self.margin_fraction
        if allocated_margin <= Decimal("0"):
            return None

        notional = allocated_margin * Decimal(leverage)
        quantity = round_quantity_down(notional / fill_price, self.config.quantity_step)
        if quantity < self.config.minimum_quantity:
            return None

        actual_notional = quantity * fill_price
        margin_used = actual_notional / Decimal(leverage)
        entry_fee = actual_notional * self.fee_rate

        # Ensure we have enough cash to allocate margin and pay entry fee
        if margin_used + entry_fee > self.cash:
            # Scale down margin to fit available cash if needed
            available_cash = max(Decimal("0"), self.cash - entry_fee)
            if available_cash <= Decimal("0"):
                return None
            notional = available_cash * Decimal(leverage)
            quantity = round_quantity_down(notional / fill_price, self.config.quantity_step)
            if quantity < self.config.minimum_quantity:
                return None
            actual_notional = quantity * fill_price
            margin_used = actual_notional / Decimal(leverage)
            entry_fee = actual_notional * self.fee_rate
            if margin_used + entry_fee > self.cash:
                return None

        # 4. Compute Stop-Loss, Take-Profit, and Liquidation price
        atr_dec = Decimal(str(signal.atr_14))
        if is_long:
            stop_price = fill_price - ATR_STOP_MULTIPLIER * atr_dec
            target_price = fill_price + TARGET_R_MULTIPLIER * atr_dec
            liquidation_price = fill_price * (Decimal("1") - Decimal("1") / Decimal(leverage))
        else:
            stop_price = fill_price + ATR_STOP_MULTIPLIER * atr_dec
            target_price = fill_price - TARGET_R_MULTIPLIER * atr_dec
            liquidation_price = fill_price * (Decimal("1") + Decimal("1") / Decimal(leverage))

        if (
            stop_price <= Decimal("0")
            or target_price <= Decimal("0")
            or liquidation_price <= Decimal("0")
        ):
            return None

        initial_risk = quantity * (ATR_STOP_MULTIPLIER * atr_dec)

        # 5. Deduct isolated margin and entry fee from cash
        self.cash -= margin_used + entry_fee

        fill = Fill(
            symbol=signal.symbol,
            exchange=signal.exchange,
            side=side_str,
            timestamp=timestamp,
            reference_price=next_open,
            price=fill_price,
            quantity=quantity,
            fee=entry_fee,
            reason="entry",
            observed_at=observed_at,
        )

        position = Position(
            signal=signal,
            quantity=quantity,
            entry_fill=fill,
            stop_price=stop_price,
            target_price=target_price,
            initial_risk=initial_risk,
            leverage=leverage,
            margin_used=margin_used,
            liquidation_price=liquidation_price,
        )
        self.positions[signal.symbol] = position
        return position

    def exit(
        self,
        symbol: str,
        *,
        timestamp: datetime,
        reference_price: Decimal,
        reason: str,
        slippage: Decimal = CORE_SLIPPAGE,
    ) -> Trade | None:
        """Close an active position, calculate bidirectional PnL, deduct fees, and release margin."""
        pos = self.positions.pop(symbol, None)
        if pos is None:
            return None

        is_long = pos.direction == "LONG"
        if is_long:
            exit_fill_price = reference_price * (Decimal("1") - slippage)
            side_str: Literal["buy", "sell"] = "sell"
            gross_pnl = (exit_fill_price - pos.entry_fill.price) * pos.quantity
        else:
            exit_fill_price = reference_price * (Decimal("1") + slippage)
            side_str = "buy"
            gross_pnl = (pos.entry_fill.price - exit_fill_price) * pos.quantity

        exit_notional = exit_fill_price * pos.quantity
        exit_fee = exit_notional * self.fee_rate
        total_fee = pos.entry_fill.fee + exit_fee
        net_pnl = gross_pnl - total_fee

        # Isolated liquidation protection: leveraged losses cannot exceed allocated margin
        if reason == "liquidation":
            net_pnl = max(net_pnl, -pos.margin_used)

        # Release margin and credit realized net PnL back into cash
        self.cash += pos.margin_used + net_pnl
        self.realized_net_pnl += net_pnl

        exit_fill = Fill(
            symbol=pos.symbol,
            exchange=pos.exchange,
            side=side_str,
            timestamp=timestamp,
            reference_price=reference_price,
            price=exit_fill_price,
            quantity=pos.quantity,
            fee=exit_fee,
            reason=reason,
        )

        trade = Trade(
            signal=pos.signal,
            quantity=pos.quantity,
            entry_fill=pos.entry_fill,
            exit_fill=exit_fill,
            stop_price=pos.stop_price,
            target_price=pos.target_price,
            exit_reason=reason,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            leverage=pos.leverage,
            margin_used=pos.margin_used,
            liquidation_price=pos.liquidation_price,
        )
        self.trades.append(trade)
        return trade

    def reduce(
        self,
        symbol: str,
        *,
        fraction: Decimal = Decimal("0.5"),
        reduce_ratio: Decimal | float | None = None,
        timestamp: datetime | None = None,
        reference_price: Decimal | None = None,
        reason: str = "reduce",
        slippage: Decimal = CORE_SLIPPAGE,
    ) -> Trade | None:
        """Reduce active position or close if remaining quantity is below minimum."""
        pos = self.positions.get(symbol)
        if pos is None:
            return None
        now = timestamp if timestamp is not None else datetime.now(timezone.utc)
        ref = reference_price if reference_price is not None else pos.entry_fill.price
        if reduce_ratio is not None:
            fraction = Decimal(str(reduce_ratio))

        reduce_qty = round_quantity_down(pos.quantity * fraction, self.config.quantity_step)
        remaining_qty = pos.quantity - reduce_qty

        if (
            reduce_qty < self.config.minimum_quantity
            or remaining_qty < self.config.minimum_quantity
        ):
            return self.exit(
                symbol, timestamp=now, reference_price=ref, reason=reason, slippage=slippage
            )

        is_long = pos.direction == "LONG"
        if is_long:
            exit_fill_price = ref * (Decimal("1") - slippage)
            side_str: Literal["buy", "sell"] = "sell"
            gross_pnl = (exit_fill_price - pos.entry_fill.price) * reduce_qty
        else:
            exit_fill_price = ref * (Decimal("1") + slippage)
            side_str = "buy"
            gross_pnl = (pos.entry_fill.price - exit_fill_price) * reduce_qty

        exit_notional = exit_fill_price * reduce_qty
        exit_fee = exit_notional * self.fee_rate
        portion = reduce_qty / pos.quantity
        entry_fee_portion = pos.entry_fill.fee * portion
        total_fee = entry_fee_portion + exit_fee
        net_pnl = gross_pnl - total_fee

        released_margin = pos.margin_used * portion
        self.cash += released_margin + net_pnl
        self.realized_net_pnl += net_pnl

        exit_fill = Fill(
            symbol=pos.symbol,
            exchange=pos.exchange,
            side=side_str,
            timestamp=now,
            reference_price=ref,
            price=exit_fill_price,
            quantity=reduce_qty,
            fee=exit_fee,
            reason=reason,
        )

        trade = Trade(
            signal=pos.signal,
            quantity=reduce_qty,
            entry_fill=pos.entry_fill,
            exit_fill=exit_fill,
            stop_price=pos.stop_price,
            target_price=pos.target_price,
            exit_reason=reason,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            leverage=pos.leverage,
            margin_used=released_margin,
            liquidation_price=pos.liquidation_price,
        )
        self.trades.append(trade)

        remaining_entry_fill = Fill(
            symbol=pos.entry_fill.symbol,
            exchange=pos.entry_fill.exchange,
            side=pos.entry_fill.side,
            timestamp=pos.entry_fill.timestamp,
            reference_price=pos.entry_fill.reference_price,
            price=pos.entry_fill.price,
            quantity=remaining_qty,
            fee=pos.entry_fill.fee - entry_fee_portion,
            reason=pos.entry_fill.reason,
            observed_at=pos.entry_fill.observed_at,
        )
        remaining_margin = pos.margin_used - released_margin
        remaining_risk = pos.initial_risk * (Decimal("1") - portion)
        self.positions[symbol] = Position(
            signal=pos.signal,
            quantity=remaining_qty,
            entry_fill=remaining_entry_fill,
            stop_price=pos.stop_price,
            target_price=pos.target_price,
            initial_risk=remaining_risk,
            leverage=pos.leverage,
            margin_used=remaining_margin,
            liquidation_price=pos.liquidation_price,
        )
        return trade


class PaperExecutor:
    """Coordinate single-position leveraged Futures paper entries and exits."""

    def __init__(
        self,
        config: BacktestConfig,
        *,
        max_open_positions: int = 1,
        starting_cash: Decimal = DEFAULT_STARTING_CASH,
    ) -> None:
        if not isinstance(config, BacktestConfig):
            raise TypeError("config must be BacktestConfig")
        self.config = config
        self.max_open_positions = max_open_positions
        self.portfolio = FuturesPortfolio(
            config,
            starting_cash=starting_cash,
            max_open_positions=max_open_positions,
        )
        self._pending: dict[str, BreakoutSignal] = {}
        self._last_closed: dict[str, Candle] = {}
        self._last_open_observation: dict[str, datetime] = {}
        self._marks: dict[str, Decimal] = {}

    @staticmethod
    def get_asset_category(symbol: str) -> Literal["core", "satellite"]:
        return (
            "core"
            if symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ZECUSDT")
            else "satellite"
        )

    @classmethod
    def get_slippage(cls, symbol: str) -> Decimal:
        return CORE_SLIPPAGE if cls.get_asset_category(symbol) == "core" else SATELLITE_SLIPPAGE

    def calculate_dollar_risk(self, symbol: str, equity: Decimal | None = None) -> Decimal:
        return self.portfolio.calculate_dollar_risk(symbol, equity)

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
    def equity(self) -> Decimal:
        return self.portfolio.equity(self._marks)

    @property
    def open_risk(self) -> Decimal:
        return self.portfolio.open_risk

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
        """Queue a signal only while no position is open and next-candle open is not yet observed."""
        if not isinstance(signal, BreakoutSignal):
            raise TypeError("signal must be BreakoutSignal")
        valid_side = str(signal.side).lower() in ("long", "short", "buy", "sell")
        if (
            signal.exchange != EXCHANGE
            or signal.symbol not in SUPPORTED_SYMBOLS
            or not valid_side
            or not _aware_utc(signal.decision_time)
            or not _aware_utc(signal.signal_open_time)
            or signal.decision_time.astimezone(timezone.utc)
            != signal.signal_open_time.astimezone(timezone.utc) + _ONE_HOUR
        ):
            return False
        atr = _positive_decimal_from_indicator(signal.atr_14)
        if atr is None:
            return False

        # Single-trade rule: cannot submit if 1 position is active or pending
        if len(self.portfolio.positions) >= self.max_open_positions:
            return False
        if len(self._pending) >= self.max_open_positions:
            return False

        latest = self._last_closed.get(signal.symbol)
        if latest is None or latest.exchange != signal.exchange:
            return False
        if latest.open_time != signal.signal_open_time.astimezone(timezone.utc):
            return False
        if signal.symbol in self.portfolio.positions or signal.symbol in self._pending:
            return False

        previously_observed = self._last_open_observation.get(signal.symbol)
        if (
            previously_observed is not None
            and previously_observed >= signal.decision_time.astimezone(timezone.utc)
        ):
            return False

        self._pending[signal.symbol] = signal
        return True

    def on_open_observation(self, observation: CandleOpenObservation) -> Position | None:
        """Fill a queued signal from the first observed next open or monitor open position."""
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

        self.portfolio.begin_day(open_time, self._marks)

        # Check if an existing active position hits SL/TP/Liquidation at this open observation
        existing_pos = self.portfolio.positions.get(symbol)
        if existing_pos is not None:
            reason, ref_price = _exit_condition_at_price(existing_pos, observation.open)
            if reason is not None:
                self.portfolio.exit(
                    symbol,
                    timestamp=open_time,
                    reference_price=ref_price,
                    reason=reason,
                    slippage=self.get_slippage(symbol),
                )
                return None

        # Check for pending entry
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
            slippage=self.get_slippage(symbol),
        )
        self._pending.pop(symbol, None)
        return position

    def on_candle_closed(self, candle: Candle) -> Trade | None:
        """Manage active positions using one completed 1H candle; evaluate SL, TP, Liquidation."""
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

        pending = self._pending.get(symbol)
        if (
            pending is not None
            and pending.decision_time.astimezone(timezone.utc) <= candle.open_time
        ):
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
                    slippage=self.get_slippage(symbol),
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
        pos = self.portfolio.positions.get(symbol)
        if pos is None:
            return None
        now = timestamp if timestamp is not None else datetime.now(timezone.utc)
        ref = (
            reference_price
            if reference_price is not None
            else self._marks.get(symbol, pos.entry_fill.price)
        )
        return self.portfolio.exit(
            symbol,
            timestamp=now,
            reference_price=ref,
            reason=reason,
            slippage=self.get_slippage(symbol),
        )

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
        pos = self.portfolio.positions.get(symbol)
        if pos is None:
            return None
        now = timestamp if timestamp is not None else datetime.now(timezone.utc)
        ref = (
            reference_price
            if reference_price is not None
            else self._marks.get(symbol, pos.entry_fill.price)
        )
        return self.portfolio.reduce(
            symbol,
            fraction=fraction,
            reduce_ratio=reduce_ratio,
            timestamp=now,
            reference_price=ref,
            reason=reason,
            slippage=self.get_slippage(symbol),
        )


def _exit_condition(position: Position, candle: Candle) -> tuple[str | None, Decimal]:
    """Evaluate liquidation, stop loss, and take profit with stop-first resolution."""
    is_long = position.direction == "LONG"
    liq_price = position.liquidation_price
    stop_price = position.stop_price
    target_price = position.target_price
    open_price = candle.open

    if is_long:
        # Long position: loss when price drops
        # 1. Liquidation
        if open_price <= liq_price or candle.low <= liq_price:
            return "liquidation", min(open_price, liq_price)

        # 2. Open bar gaps
        if open_price <= stop_price:
            return "stop_loss", open_price
        if open_price >= target_price:
            if candle.low <= stop_price:
                return "stop_loss", stop_price
            return "take_profit", open_price

        # 3. Within bar breaches (stop-first tie break)
        if candle.low <= stop_price:
            return "stop_loss", stop_price
        if candle.high >= target_price:
            return "take_profit", target_price
        return None, open_price
    else:
        # Short position: loss when price rises
        # 1. Liquidation
        if open_price >= liq_price or candle.high >= liq_price:
            return "liquidation", max(open_price, liq_price)

        # 2. Open bar gaps
        if open_price >= stop_price:
            return "stop_loss", open_price
        if open_price <= target_price:
            if candle.high >= stop_price:
                return "stop_loss", stop_price
            return "take_profit", open_price

        # 3. Within bar breaches (stop-first tie break)
        if candle.high >= stop_price:
            return "stop_loss", stop_price
        if candle.low <= target_price:
            return "take_profit", target_price
        return None, open_price


def _exit_condition_at_price(position: Position, price: Decimal) -> tuple[str | None, Decimal]:
    """Check a single price tick against active position's SL, TP, and liquidation."""
    is_long = position.direction == "LONG"
    if is_long:
        if price <= position.liquidation_price:
            return "liquidation", price
        if price <= position.stop_price:
            return "stop_loss", price
        if price >= position.target_price:
            return "take_profit", price
    else:
        if price >= position.liquidation_price:
            return "liquidation", price
        if price >= position.stop_price:
            return "stop_loss", price
        if price <= position.target_price:
            return "take_profit", price
    return None, price


def _aware_utc(value: object) -> bool:
    return (
        isinstance(value, datetime) and value.tzinfo is not None and value.utcoffset() is not None
    )


def _positive_decimal_from_indicator(value: object) -> Decimal | None:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return result if result.is_finite() and result > 0 else None

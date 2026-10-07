"""Shared spot portfolio, risk admission, and position lifecycle."""

from datetime import datetime, timezone
from decimal import Decimal

from app.backtest.execution import (
    decimal_atr,
    entry_fill_price,
    exit_fill_price,
    fee_for_fill,
    round_quantity_down,
    stop_and_target,
)
from app.backtest.models import BacktestConfig, Fill, Position, Trade
from app.strategy.breakout import BreakoutSignal

RISK_PER_POSITION = Decimal("0.005")
MAX_TOTAL_OPEN_RISK = Decimal("0.015")
DAILY_LOSS_LIMIT = Decimal("0.02")
MAX_OPEN_POSITIONS = 3


class Portfolio:
    def __init__(self, config: BacktestConfig) -> None:
        self.config = config
        self.cash = config.starting_cash
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self.realized_net_pnl = Decimal("0")
        self._day: object | None = None
        self._start_of_day_equity = config.starting_cash
        self._daily_loss_breached = False

    def equity(self, marks: dict[str, Decimal]) -> Decimal:
        return self.cash + sum(
            (position.quantity * marks.get(symbol, position.entry_fill.price)
             for symbol, position in self.positions.items()),
            Decimal("0"),
        )

    @property
    def open_risk(self) -> Decimal:
        return sum((position.initial_risk for position in self.positions.values()), Decimal("0"))

    def update_day(self, timestamp: datetime, marks: dict[str, Decimal]) -> None:
        self.begin_day(timestamp, marks)
        current_equity = self.equity(marks)
        daily_loss = self._start_of_day_equity - current_equity
        if daily_loss >= self._start_of_day_equity * DAILY_LOSS_LIMIT:
            self._daily_loss_breached = True

    def begin_day(self, timestamp: datetime, marks: dict[str, Decimal]) -> None:
        """Capture equity at a UTC day boundary before that instant's open gaps."""
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("portfolio timestamps must be timezone-aware")
        utc_timestamp = timestamp.astimezone(timezone.utc)
        current_day = utc_timestamp.date()
        if self._day != current_day:
            self._day = current_day
            self._start_of_day_equity = self.equity(marks)
            self._daily_loss_breached = False

    @property
    def daily_loss_breached(self) -> bool:
        return self._daily_loss_breached

    def enter(
        self,
        signal: BreakoutSignal,
        *,
        timestamp: datetime,
        next_open: Decimal,
        marks: dict[str, Decimal],
        observed_at: datetime | None = None,
    ) -> Position | None:
        if signal.side != "long":
            return None
        if self._daily_loss_breached or signal.symbol in self.positions:
            return None
        if len(self.positions) >= MAX_OPEN_POSITIONS:
            return None

        equity = self.equity(marks)
        if equity <= 0:
            return None
        fill_price = entry_fill_price(next_open)
        atr = decimal_atr(signal.atr_14)
        try:
            stop, target = stop_and_target(fill_price, atr)
        except ValueError:
            # A positive ATR can still imply a non-positive long stop at an
            # extreme price/volatility ratio; such a position is invalid.
            return None
        risk_per_unit = fill_price - stop
        risk_budget = equity * RISK_PER_POSITION
        quantity = round_quantity_down(risk_budget / risk_per_unit, self.config.quantity_step)
        if quantity < self.config.minimum_quantity:
            return None
        position_risk = quantity * risk_per_unit
        if self.open_risk + position_risk > equity * MAX_TOTAL_OPEN_RISK:
            return None

        fee = fee_for_fill(fill_price, quantity)
        required_cash = fill_price * quantity + fee
        if required_cash > self.cash:
            return None

        fill = Fill(
            symbol=signal.symbol,
            exchange=signal.exchange,
            side="buy",
            timestamp=timestamp,
            reference_price=next_open,
            price=fill_price,
            quantity=quantity,
            fee=fee,
            reason="entry",
            observed_at=observed_at,
        )
        position = Position(
            signal=signal,
            quantity=quantity,
            entry_fill=fill,
            stop_price=stop,
            target_price=target,
            initial_risk=position_risk,
        )
        self.cash -= required_cash
        self.positions[signal.symbol] = position
        return position

    def exit(
        self,
        symbol: str,
        *,
        timestamp: datetime,
        reference_price: Decimal,
        reason: str,
    ) -> Trade:
        position = self.positions.pop(symbol)
        fill_price = exit_fill_price(reference_price)
        fee = fee_for_fill(fill_price, position.quantity)
        fill = Fill(
            symbol=symbol,
            exchange=position.exchange,
            side="sell",
            timestamp=timestamp,
            reference_price=reference_price,
            price=fill_price,
            quantity=position.quantity,
            fee=fee,
            reason=reason,
        )
        self.cash += fill_price * position.quantity - fee
        gross_pnl = (fill_price - position.entry_fill.price) * position.quantity
        net_pnl = gross_pnl - position.entry_fill.fee - fee
        trade = Trade(
            signal=position.signal,
            quantity=position.quantity,
            entry_fill=position.entry_fill,
            exit_fill=fill,
            stop_price=position.stop_price,
            target_price=position.target_price,
            exit_reason=reason,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
        )
        self.realized_net_pnl += net_pnl
        self.trades.append(trade)
        return trade

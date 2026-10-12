"""Shared spot portfolio, risk admission, and position lifecycle."""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Literal

from app.backtest.execution import (
    decimal_atr,
    entry_fill_price,
    exit_fill_price,
    fee_for_fill,
    round_quantity_down,
    stop_and_target,
)
from app.backtest.models import BacktestConfig, Fill, Position, Trade
from app.market.constants import FIXED_SYMBOLS
from app.strategy.breakout import BreakoutSignal

RISK_PER_POSITION = Decimal("0.005")
MAX_TOTAL_OPEN_RISK = Decimal("0.015")
DAILY_LOSS_LIMIT = Decimal("0.02")
MAX_OPEN_POSITIONS = 3


def get_asset_category(symbol: str) -> Literal["core", "satellite"]:
    """Determine whether a symbol belongs to the core or satellite basket."""
    return "core" if symbol in FIXED_SYMBOLS else "satellite"


class Portfolio:
    def __init__(self, config: BacktestConfig, *, split_risk: bool = False) -> None:
        self.config = config
        custom_split = (
            config.core_risk_per_trade != 0.05
            or config.satellite_risk_per_trade != 0.04
            or config.core_daily_loss_limit != 0.05
            or config.satellite_daily_loss_limit != 0.04
        )
        self._split_risk = split_risk or custom_split
        self.cash = config.starting_cash
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self.realized_net_pnl = Decimal("0")
        self._day: object | None = None
        self._start_of_day_equity = config.starting_cash
        self._start_of_day_basket_pnl: dict[str, Decimal] = {
            "core": Decimal("0"),
            "satellite": Decimal("0"),
        }
        self._daily_loss_breached = False
        self._core_daily_loss_breached = False
        self._satellite_daily_loss_breached = False

    @staticmethod
    def asset_category(symbol: str) -> Literal["core", "satellite"]:
        return get_asset_category(symbol)

    def calculate_dollar_risk(self, symbol: str, equity: Decimal) -> Decimal:
        """Calculate dollar risk budget using category-specific sizing."""
        category = self.asset_category(symbol)
        if category == "core":
            rate = Decimal(str(self.config.core_risk_per_trade))
        else:
            rate = Decimal(str(self.config.satellite_risk_per_trade))
        return equity * rate

    def _basket_pnl(
        self, category: Literal["core", "satellite"], marks: dict[str, Decimal]
    ) -> Decimal:
        realized = sum(
            (
                trade.net_pnl
                for trade in self.trades
                if self.asset_category(trade.signal.symbol) == category
            ),
            Decimal("0"),
        )
        unrealized = sum(
            (
                position.quantity * marks.get(symbol, position.entry_fill.price)
                - (position.entry_fill.price * position.quantity + position.entry_fill.fee)
                for symbol, position in self.positions.items()
                if self.asset_category(symbol) == category
            ),
            Decimal("0"),
        )
        return realized + unrealized

    def daily_basket_loss(
        self, category: Literal["core", "satellite"], marks: dict[str, Decimal]
    ) -> Decimal:
        return self._start_of_day_basket_pnl[category] - self._basket_pnl(category, marks)

    def equity(self, marks: dict[str, Decimal]) -> Decimal:
        return self.cash + sum(
            (
                position.quantity * marks.get(symbol, position.entry_fill.price)
                for symbol, position in self.positions.items()
            ),
            Decimal("0"),
        )

    @property
    def open_risk(self) -> Decimal:
        return sum((position.initial_risk for position in self.positions.values()), Decimal("0"))

    def update_day(self, timestamp: datetime, marks: dict[str, Decimal]) -> None:
        self.begin_day(timestamp, marks)
        if self._split_risk:
            core_loss = self.daily_basket_loss("core", marks)
            core_limit = self._start_of_day_equity * Decimal(str(self.config.core_daily_loss_limit))
            if core_loss >= core_limit:
                self._core_daily_loss_breached = True

            sat_loss = self.daily_basket_loss("satellite", marks)
            sat_limit = self._start_of_day_equity * Decimal(
                str(self.config.satellite_daily_loss_limit)
            )
            if sat_loss >= sat_limit:
                self._satellite_daily_loss_breached = True

            self._daily_loss_breached = (
                self._core_daily_loss_breached or self._satellite_daily_loss_breached
            )
        else:
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
            self._start_of_day_basket_pnl = {
                "core": self._basket_pnl("core", marks),
                "satellite": self._basket_pnl("satellite", marks),
            }
            self._daily_loss_breached = False
            self._core_daily_loss_breached = False
            self._satellite_daily_loss_breached = False

    @property
    def daily_loss_breached(self) -> bool:
        return self._daily_loss_breached

    @property
    def core_daily_loss_breached(self) -> bool:
        return self._core_daily_loss_breached

    @property
    def satellite_daily_loss_breached(self) -> bool:
        return self._satellite_daily_loss_breached

    def is_category_loss_breached(self, symbol: str) -> bool:
        if not self._split_risk:
            return self._daily_loss_breached
        category = self.asset_category(symbol)
        if category == "core":
            return self._core_daily_loss_breached
        return self._satellite_daily_loss_breached

    def enter(
        self,
        signal: BreakoutSignal,
        *,
        timestamp: datetime,
        next_open: Decimal,
        marks: dict[str, Decimal],
        observed_at: datetime | None = None,
        slippage: Decimal | None = None,
    ) -> Position | None:
        if signal.side != "long":
            return None
        if self.is_category_loss_breached(signal.symbol) or signal.symbol in self.positions:
            return None
        if len(self.positions) >= MAX_OPEN_POSITIONS:
            return None

        equity = self.equity(marks)
        if equity <= 0:
            return None
        if slippage is not None:
            fill_price = entry_fill_price(next_open, slippage=slippage)
        elif self._split_risk:
            cat_slip = (
                Decimal("0.0005")
                if self.asset_category(signal.symbol) == "core"
                else Decimal("0.0020")
            )
            fill_price = entry_fill_price(next_open, slippage=cat_slip)
        else:
            fill_price = entry_fill_price(next_open)
        atr = decimal_atr(signal.atr_14)
        try:
            stop, target = stop_and_target(fill_price, atr)
        except ValueError:
            # A positive ATR can still imply a non-positive long stop at an
            # extreme price/volatility ratio; such a position is invalid.
            return None
        risk_per_unit = fill_price - stop
        if self._split_risk:
            risk_budget = self.calculate_dollar_risk(signal.symbol, equity)
            max_risk_rate = max(
                Decimal(str(self.config.core_risk_per_trade)),
                Decimal(str(self.config.satellite_risk_per_trade)),
            ) * Decimal(str(MAX_OPEN_POSITIONS))
        else:
            risk_budget = equity * RISK_PER_POSITION
            max_risk_rate = MAX_TOTAL_OPEN_RISK

        quantity = round_quantity_down(risk_budget / risk_per_unit, self.config.quantity_step)
        if quantity < self.config.minimum_quantity:
            return None
        position_risk = quantity * risk_per_unit
        if self.open_risk + position_risk > equity * max_risk_rate:
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
        slippage: Decimal | None = None,
    ) -> Trade:
        position = self.positions.pop(symbol)
        if slippage is not None:
            fill_price = exit_fill_price(reference_price, slippage=slippage)
        elif self._split_risk:
            cat_slip = (
                Decimal("0.0005") if self.asset_category(symbol) == "core" else Decimal("0.0020")
            )
            fill_price = exit_fill_price(reference_price, slippage=cat_slip)
        else:
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

    def reduce(
        self,
        symbol: str,
        *,
        fraction: Decimal = Decimal("0.5"),
        reduce_ratio: Decimal | float | None = None,
        timestamp: datetime,
        reference_price: Decimal,
        reason: str,
        slippage: Decimal | None = None,
    ) -> Trade:
        if reduce_ratio is not None:
            fraction = Decimal(str(reduce_ratio))
        position = self.positions.get(symbol)
        if position is None:
            raise KeyError(f"no position for {symbol}")

        reduce_qty = round_quantity_down(position.quantity * fraction, self.config.quantity_step)
        remaining_qty = position.quantity - reduce_qty

        if (
            reduce_qty < self.config.minimum_quantity
            or remaining_qty < self.config.minimum_quantity
        ):
            return self.exit(
                symbol,
                timestamp=timestamp,
                reference_price=reference_price,
                reason=reason,
                slippage=slippage,
            )

        if slippage is not None:
            fill_price = exit_fill_price(reference_price, slippage=slippage)
        elif self._split_risk:
            cat_slip = (
                Decimal("0.0005") if self.asset_category(symbol) == "core" else Decimal("0.0020")
            )
            fill_price = exit_fill_price(reference_price, slippage=cat_slip)
        else:
            fill_price = exit_fill_price(reference_price)
        fee = fee_for_fill(fill_price, reduce_qty)
        fill = Fill(
            symbol=symbol,
            exchange=position.exchange,
            side="sell",
            timestamp=timestamp,
            reference_price=reference_price,
            price=fill_price,
            quantity=reduce_qty,
            fee=fee,
            reason=reason,
        )
        self.cash += fill_price * reduce_qty - fee
        gross_pnl = (fill_price - position.entry_fill.price) * reduce_qty
        entry_fee_portion = position.entry_fill.fee * (reduce_qty / position.quantity)
        net_pnl = gross_pnl - entry_fee_portion - fee
        trade = Trade(
            signal=position.signal,
            quantity=reduce_qty,
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

        remaining_entry_fill = Fill(
            symbol=position.entry_fill.symbol,
            exchange=position.entry_fill.exchange,
            side=position.entry_fill.side,
            timestamp=position.entry_fill.timestamp,
            reference_price=position.entry_fill.reference_price,
            price=position.entry_fill.price,
            quantity=remaining_qty,
            fee=position.entry_fill.fee - entry_fee_portion,
            reason=position.entry_fill.reason,
            observed_at=position.entry_fill.observed_at,
        )
        remaining_risk = position.initial_risk * (remaining_qty / position.quantity)
        self.positions[symbol] = Position(
            signal=position.signal,
            quantity=remaining_qty,
            entry_fill=remaining_entry_fill,
            stop_price=position.stop_price,
            target_price=position.target_price,
            initial_risk=remaining_risk,
        )
        return trade

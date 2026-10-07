"""Value objects for deterministic offline execution and backtest results."""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Literal

from app.strategy.breakout import BreakoutSignal


@dataclass(frozen=True)
class BacktestConfig:
    """Execution settings. Minimum quantity is market-configurable and required."""

    minimum_quantity: Decimal
    starting_cash: Decimal = field(default=Decimal("100"), init=False)
    quantity_step: Decimal = Decimal("0.000001")

    def __post_init__(self) -> None:
        for name in ("minimum_quantity", "starting_cash", "quantity_step"):
            value = getattr(self, name)
            if not isinstance(value, Decimal) or not value.is_finite() or value <= 0:
                raise ValueError(f"{name} must be a finite positive Decimal")


@dataclass(frozen=True)
class Fill:
    symbol: str
    exchange: str
    side: Literal["buy", "sell"]
    timestamp: datetime
    reference_price: Decimal
    price: Decimal
    quantity: Decimal
    fee: Decimal
    reason: str

    @property
    def notional(self) -> Decimal:
        return self.price * self.quantity


@dataclass(frozen=True)
class Position:
    """One open spot long position, with fixed stop and target."""

    signal: BreakoutSignal
    quantity: Decimal
    entry_fill: Fill
    stop_price: Decimal
    target_price: Decimal
    initial_risk: Decimal

    @property
    def symbol(self) -> str:
        return self.signal.symbol

    @property
    def exchange(self) -> str:
        return self.signal.exchange


@dataclass(frozen=True)
class Trade:
    signal: BreakoutSignal
    quantity: Decimal
    entry_fill: Fill
    exit_fill: Fill
    stop_price: Decimal
    target_price: Decimal
    exit_reason: str
    gross_pnl: Decimal
    net_pnl: Decimal


@dataclass(frozen=True)
class BacktestResult:
    trades: tuple[Trade, ...]
    final_cash: Decimal
    final_equity: Decimal
    final_open_positions: tuple[Position, ...]

    @property
    def realized_net_pnl(self) -> Decimal:
        return sum((trade.net_pnl for trade in self.trades), Decimal("0"))

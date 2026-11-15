"""Value objects for deterministic offline execution and backtest results."""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Literal

from app.strategy.breakout import BreakoutSignal


@dataclass(frozen=True)
class BacktestConfig:
    """Execution settings. Minimum quantity is market-configurable and required."""

    minimum_quantity: Decimal = Decimal("0.000001")
    starting_cash: Decimal = field(default=Decimal("100"), init=False)
    quantity_step: Decimal = Decimal("0.000001")
    core_risk_per_trade: float = 0.05
    satellite_risk_per_trade: float = 0.04
    core_daily_loss_limit: float = 0.05
    satellite_daily_loss_limit: float = 0.04

    def __post_init__(self) -> None:
        for name in ("minimum_quantity", "starting_cash", "quantity_step"):
            value = getattr(self, name)
            if not isinstance(value, Decimal) or not value.is_finite() or value <= 0:
                raise ValueError(f"{name} must be a finite positive Decimal")
        for name in (
            "core_risk_per_trade",
            "satellite_risk_per_trade",
            "core_daily_loss_limit",
            "satellite_daily_loss_limit",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
                raise ValueError(f"{name} must be a finite positive number")
            dec_val = Decimal(str(value))
            if not dec_val.is_finite() or dec_val <= 0 or dec_val > 1:
                raise ValueError(f"{name} must be in (0, 1]")


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
    observed_at: datetime | None = None

    @property
    def notional(self) -> Decimal:
        return self.price * self.quantity


@dataclass(frozen=True)
class Position:
    """One open position, with fixed stop, target, leverage, and liquidation."""

    signal: BreakoutSignal
    quantity: Decimal
    entry_fill: Fill
    stop_price: Decimal
    target_price: Decimal
    initial_risk: Decimal
    leverage: int = 1
    margin_used: Decimal = Decimal("0")
    liquidation_price: Decimal = Decimal("0")

    @property
    def symbol(self) -> str:
        return self.signal.symbol

    @property
    def exchange(self) -> str:
        return self.signal.exchange

    @property
    def side(self) -> str:
        return self.signal.side

    @property
    def direction(self) -> str:
        return "LONG" if str(self.signal.side).lower() in ("long", "buy") else "SHORT"

    @property
    def stop_loss(self) -> Decimal:
        return self.stop_price

    @property
    def take_profit(self) -> Decimal:
        return self.target_price


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
    leverage: int = 1
    margin_used: Decimal = Decimal("0")
    liquidation_price: Decimal = Decimal("0")

    @property
    def direction(self) -> str:
        return "LONG" if str(self.signal.side).lower() in ("long", "buy") else "SHORT"

    @property
    def stop_loss(self) -> Decimal:
        return self.stop_price

    @property
    def take_profit(self) -> Decimal:
        return self.target_price

    @property
    def realized_pnl(self) -> Decimal:
        return self.net_pnl


@dataclass(frozen=True)
class EquityPoint:
    """Portfolio valuation at the initial baseline or a completed decision."""

    timestamp: datetime
    equity: Decimal
    event: Literal["start", "decision"]


@dataclass(frozen=True)
class BacktestResult:
    trades: tuple[Trade, ...]
    final_cash: Decimal
    final_equity: Decimal
    final_open_positions: tuple[Position, ...]
    equity_curve: tuple[EquityPoint, ...] = ()

    @property
    def realized_net_pnl(self) -> Decimal:
        return sum((trade.net_pnl for trade in self.trades), Decimal("0"))

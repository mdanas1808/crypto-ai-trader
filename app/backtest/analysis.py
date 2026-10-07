"""Deterministic Decimal-based analysis of completed offline backtests."""

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal

from app.backtest.models import BacktestResult, EquityPoint, Trade

ZERO = Decimal("0")


@dataclass(frozen=True)
class BacktestMetrics:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: Decimal
    net_pnl: Decimal
    total_return: Decimal
    gross_profit: Decimal
    gross_loss: Decimal
    profit_factor: Decimal | None
    average_trade_pnl: Decimal
    average_winning_trade: Decimal
    average_losing_trade: Decimal
    expectancy: Decimal
    maximum_drawdown: Decimal
    maximum_drawdown_pct: Decimal
    ending_equity: Decimal
    total_fees: Decimal
    stop_loss_trades: int
    take_profit_trades: int
    end_of_backtest_trades: int


def analyze_backtest(result: BacktestResult) -> BacktestMetrics:
    """Calculate trade and equity-curve metrics without changing the result.

    An empty result has no timestamped observations. Its final equity is used
    as both the deterministic starting and ending baseline; a non-empty result
    must provide a validated start point followed by decision points.
    """
    if not isinstance(result, BacktestResult):
        raise TypeError("result must be a BacktestResult")
    curve = result.equity_curve
    if not isinstance(curve, tuple):
        raise ValueError("equity_curve must be a tuple")

    ending_equity = _decimal(result.final_equity, "final_equity")

    if curve:
        starting_equity = _validate_curve(curve, ending_equity)
    else:
        if result.trades or result.final_open_positions:
            raise ValueError("a result with trades or open positions requires an equity curve")
        starting_equity = ending_equity
    if starting_equity <= ZERO:
        raise ValueError("starting equity must be positive")

    trades = result.trades
    if not isinstance(trades, tuple):
        raise ValueError("trades must be a tuple")
    for trade in trades:
        _validate_trade(trade)

    total_trades = len(trades)
    winning = [trade for trade in trades if trade.net_pnl > ZERO]
    losing = [trade for trade in trades if trade.net_pnl < ZERO]
    net_pnl = sum((trade.net_pnl for trade in trades), ZERO)
    gross_profit = sum((trade.net_pnl for trade in winning), ZERO)
    gross_loss = -sum((trade.net_pnl for trade in losing), ZERO)
    fees = sum(
        (trade.entry_fill.fee + trade.exit_fill.fee for trade in trades), ZERO
    )

    average_trade = net_pnl / total_trades if total_trades else ZERO
    average_winning = gross_profit / len(winning) if winning else ZERO
    average_losing = -gross_loss / len(losing) if losing else ZERO
    drawdown, drawdown_pct = _drawdown(curve) if curve else (ZERO, ZERO)

    return BacktestMetrics(
        total_trades=total_trades,
        winning_trades=len(winning),
        losing_trades=len(losing),
        win_rate=Decimal(len(winning)) / Decimal(total_trades) if total_trades else ZERO,
        net_pnl=net_pnl,
        total_return=(ending_equity - starting_equity) / starting_equity,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        profit_factor=gross_profit / gross_loss if gross_loss else None,
        average_trade_pnl=average_trade,
        average_winning_trade=average_winning,
        average_losing_trade=average_losing,
        expectancy=average_trade,
        maximum_drawdown=drawdown,
        maximum_drawdown_pct=drawdown_pct,
        ending_equity=ending_equity,
        total_fees=fees,
        stop_loss_trades=sum(trade.exit_reason == "stop_loss" for trade in trades),
        take_profit_trades=sum(trade.exit_reason == "take_profit" for trade in trades),
        end_of_backtest_trades=sum(
            trade.exit_reason == "end_of_backtest" for trade in trades
        ),
    )


def _validate_curve(curve: tuple[EquityPoint, ...], ending_equity: Decimal) -> Decimal:
    if not isinstance(curve[0], EquityPoint):
        raise ValueError("equity_curve entries must be EquityPoint values")
    if curve[0].event != "start":
        raise ValueError("equity_curve must begin with a start point")
    previous = None
    for index, point in enumerate(curve):
        if not isinstance(point, EquityPoint):
            raise ValueError("equity_curve entries must be EquityPoint values")
        _validate_utc(point.timestamp, "equity point timestamp")
        equity = _decimal(point.equity, "equity")
        expected_event = "start" if index == 0 else "decision"
        if point.event != expected_event:
            raise ValueError("equity_curve must contain one start point followed by decisions")
        if previous is not None and point.timestamp <= previous:
            raise ValueError("equity_curve timestamps must be strictly chronological and unique")
        previous = point.timestamp
    starting_equity = curve[0].equity
    if starting_equity <= ZERO:
        raise ValueError("starting equity must be positive")
    if len(curve) < 2:
        raise ValueError("a non-empty equity curve must contain a decision point")
    if curve[-1].equity != ending_equity:
        raise ValueError("final equity must match the final equity_curve point")
    return starting_equity


def _validate_trade(trade: Trade) -> None:
    if not isinstance(trade, Trade):
        raise ValueError("trades must contain Trade values")
    _decimal(trade.net_pnl, "trade net_pnl")
    _decimal(trade.gross_pnl, "trade gross_pnl")
    for name, fill in (("entry", trade.entry_fill), ("exit", trade.exit_fill)):
        _decimal(fill.fee, f"{name} fee")
        if fill.fee < ZERO:
            raise ValueError(f"{name} fee must be non-negative")


def _validate_utc(value: object, name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware UTC")
    if value.utcoffset() != timezone.utc.utcoffset(value):
        raise ValueError(f"{name} must be UTC")


def _decimal(value: object, name: str) -> Decimal:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise ValueError(f"{name} must be a finite Decimal")
    return value


def _drawdown(curve: tuple[EquityPoint, ...]) -> tuple[Decimal, Decimal]:
    peak = curve[0].equity
    maximum = ZERO
    maximum_pct = ZERO
    for point in curve[1:]:
        if point.equity > peak:
            peak = point.equity
        decline = peak - point.equity
        if decline > maximum:
            maximum = decline
            maximum_pct = decline / peak
    return maximum, maximum_pct

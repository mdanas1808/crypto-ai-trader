from datetime import datetime, timezone
from decimal import Decimal

import pytest

from app.backtest.models import BacktestConfig
from app.backtest.portfolio import Portfolio
from app.backtest.execution import entry_fill_price, fee_for_fill, round_quantity_down
from app.strategy.breakout import BreakoutSignal


T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)


def signal(symbol="BTCUSDT", *, atr=1.0, when=T0):
    return BreakoutSignal("binance_spot", symbol, "long", when, when, atr)


def config(**kwargs):
    minimum = kwargs.pop("minimum_quantity", Decimal("0.000001"))
    return BacktestConfig(minimum_quantity=minimum, **kwargs)


def enter(portfolio, symbol="BTCUSDT", price=Decimal("10"), when=T0, marks=None, atr=1.0):
    return portfolio.enter(
        signal(symbol, when=when, atr=atr), timestamp=when, next_open=price,
        marks=marks or {},
    )


def test_initial_cash_risk_budget_quantity_and_entry_costs():
    portfolio = Portfolio(config())
    assert portfolio.cash == Decimal("100")
    position = enter(portfolio)
    assert position is not None
    assert position.entry_fill.price == Decimal("10.0050")
    assert position.stop_price == Decimal("8.5050")
    assert position.target_price == Decimal("13.0050")
    assert position.quantity == Decimal("0.333333")
    assert position.initial_risk == Decimal("0.4999995")
    assert position.entry_fill.fee == position.entry_fill.price * position.quantity * Decimal("0.001")
    assert portfolio.cash == Decimal("100") - position.entry_fill.price * position.quantity - position.entry_fill.fee


def test_open_risk_cap_and_three_position_limit_are_shared():
    portfolio = Portfolio(config())
    assert enter(portfolio, "AUSDT") is not None
    assert enter(portfolio, "BUSDT") is not None
    third = enter(portfolio, "CUSDT")
    # The 1.5% cap is evaluated against current equity and may bind before the
    # hard position-count cap because fees reduce equity.
    assert portfolio.open_risk <= portfolio.equity({}) * Decimal("0.015")
    assert enter(portfolio, "DUSDT") is None
    assert len(portfolio.positions) <= 3
    assert third is None or len(portfolio.positions) == 3


def test_insufficient_cash_includes_entry_fee_and_minimum_quantity_rejects():
    poor = Portfolio(config())
    # A tight stop creates a quantity whose notional exceeds available cash.
    assert enter(poor, atr=0.001) is None

    fee_limited = Portfolio(config())
    expected_fill = entry_fill_price(Decimal("10"))
    expected_quantity = round_quantity_down(
        Decimal("0.5") / (Decimal("1.5") * Decimal("0.03335")),
        fee_limited.config.quantity_step,
    )
    notional = expected_fill * expected_quantity
    assert notional <= fee_limited.cash
    assert notional + fee_for_fill(expected_fill, expected_quantity) > fee_limited.cash
    assert enter(fee_limited, atr=0.03335) is None

    minimum = Portfolio(config(minimum_quantity=Decimal("1")))
    assert enter(minimum) is None


def test_same_symbol_position_rejects_pyramiding():
    portfolio = Portfolio(config())
    assert enter(portfolio) is not None
    assert enter(portfolio, when=T0.replace(hour=1)) is None
    assert len(portfolio.positions) == 1


def test_daily_loss_threshold_is_inclusive_and_existing_position_remains():
    portfolio = Portfolio(config())
    portfolio.update_day(T0, {})
    position = enter(portfolio)
    assert position is not None
    portfolio.update_day(T0, {"BTCUSDT": Decimal("10")})
    exact_mark = (Decimal("98") - portfolio.cash) / position.quantity
    portfolio.update_day(T0.replace(hour=1), {"BTCUSDT": exact_mark})
    assert portfolio.daily_loss_breached
    assert "BTCUSDT" in portfolio.positions
    assert enter(portfolio, "ETHUSDT", when=T0.replace(hour=1)) is None


def test_daily_loss_below_threshold_allows_entry():
    portfolio = Portfolio(config())
    portfolio.update_day(T0, {})
    position = enter(portfolio)
    assert position is not None
    portfolio.update_day(T0, {"BTCUSDT": Decimal("10")})
    mark = (Decimal("99") - portfolio.cash) / position.quantity
    portfolio.update_day(T0.replace(hour=1), {"BTCUSDT": mark})
    assert not portfolio.daily_loss_breached


def test_new_utc_day_baseline_precedes_opening_gap_mark():
    portfolio = Portfolio(config())
    portfolio.begin_day(T0, {})
    position = enter(portfolio)
    assert position is not None
    next_day = T0.replace(day=2)
    # Capture the new day's baseline at the prior close value, then apply an
    # opening gap that causes at least 2% loss.
    portfolio.begin_day(next_day, {"BTCUSDT": Decimal("10")})
    gap_mark = (Decimal("97") - portfolio.cash) / position.quantity
    portfolio.update_day(next_day, {"BTCUSDT": gap_mark})
    assert portfolio.daily_loss_breached
    assert "BTCUSDT" in portfolio.positions


def test_equity_marking_and_realized_pnl_fees_update_cash():
    portfolio = Portfolio(config())
    position = enter(portfolio)
    assert position is not None
    marked = portfolio.equity({"BTCUSDT": Decimal("11")})
    assert marked == portfolio.cash + position.quantity * Decimal("11")
    before = portfolio.cash
    trade = portfolio.exit(
        "BTCUSDT", timestamp=T0.replace(hour=1), reference_price=Decimal("11"), reason="take_profit"
    )
    assert trade.exit_fill.price == Decimal("10.9945")
    assert trade.net_pnl == trade.gross_pnl - trade.entry_fill.fee - trade.exit_fill.fee
    assert portfolio.cash > before
    assert portfolio.realized_net_pnl == trade.net_pnl
    assert portfolio.equity({}) == portfolio.cash

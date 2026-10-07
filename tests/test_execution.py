from decimal import Decimal

import pytest

from app.backtest.execution import (
    decimal_atr,
    entry_fill_price,
    exit_fill_price,
    fee_for_fill,
    round_quantity_down,
    stop_and_target,
)


def test_entry_and_all_exit_fill_slippage_formulas_are_decimal_deterministic():
    price = Decimal("100")
    assert entry_fill_price(price) == Decimal("100.0500")
    assert exit_fill_price(price) == Decimal("99.9500")
    # Stop, target, gap, and forced exits share the same adverse exit formula.
    assert exit_fill_price(Decimal("98.55")) == Decimal("98.5007250")
    assert exit_fill_price(Decimal("104.55")) == Decimal("104.4977250")
    assert exit_fill_price(Decimal("110")) == Decimal("109.9450")


def test_fee_uses_actual_fill_notional():
    assert fee_for_fill(Decimal("100.05"), Decimal("0.25")) == Decimal("0.0250125")


def test_quantity_rounds_down_to_configured_step():
    assert round_quantity_down(Decimal("0.123456789"), Decimal("0.000001")) == Decimal("0.123456")
    assert round_quantity_down(Decimal("1.25"), Decimal("0.1")) == Decimal("1.2")


def test_atr_conversion_and_stop_target_are_decimal_only():
    atr = decimal_atr(0.1)
    assert atr == Decimal("0.1")
    stop, target = stop_and_target(Decimal("100.05"), atr)
    assert stop == Decimal("99.90")
    assert target == Decimal("100.35")
    with pytest.raises(ValueError, match="ATR"):
        decimal_atr(float("inf"))

"""Deterministic Decimal-based fills, fees, and quantity rounding."""

from decimal import ROUND_DOWN, Decimal

ENTRY_SLIPPAGE = Decimal("0.0005")
EXIT_SLIPPAGE = Decimal("0.0005")
CORE_SLIPPAGE = Decimal("0.0005")
SATELLITE_SLIPPAGE = Decimal("0.0020")
FEE_RATE = Decimal("0.001")
ATR_STOP_MULTIPLIER = Decimal("1.5")
TARGET_R_MULTIPLIER = Decimal("2")


def decimal_atr(value: float) -> Decimal:
    """Convert an indicator float through its string representation."""
    result = Decimal(str(value))
    if not result.is_finite() or result <= 0:
        raise ValueError("ATR must be finite and positive")
    return result


def entry_fill_price(open_price: Decimal, slippage: Decimal | None = None) -> Decimal:
    _positive_price(open_price)
    rate = slippage if slippage is not None else ENTRY_SLIPPAGE
    return open_price * (Decimal("1") + rate)


def exit_fill_price(reference_price: Decimal, slippage: Decimal | None = None) -> Decimal:
    _positive_price(reference_price)
    rate = slippage if slippage is not None else EXIT_SLIPPAGE
    return reference_price * (Decimal("1") - rate)


def fee_for_fill(price: Decimal, quantity: Decimal) -> Decimal:
    _positive_price(price)
    if not quantity.is_finite() or quantity <= 0:
        raise ValueError("quantity must be finite and positive")
    return price * quantity * FEE_RATE


def round_quantity_down(quantity: Decimal, step: Decimal) -> Decimal:
    if not quantity.is_finite() or quantity < 0:
        raise ValueError("quantity must be finite and non-negative")
    if not step.is_finite() or step <= 0:
        raise ValueError("quantity step must be finite and positive")
    increments = (quantity / step).to_integral_value(rounding=ROUND_DOWN)
    return increments * step


def stop_and_target(entry_price: Decimal, atr: Decimal) -> tuple[Decimal, Decimal]:
    _positive_price(entry_price)
    if not atr.is_finite() or atr <= 0:
        raise ValueError("ATR must be finite and positive")
    stop = entry_price - ATR_STOP_MULTIPLIER * atr
    if stop <= 0:
        raise ValueError("calculated stop price must be positive")
    risk = entry_price - stop
    return stop, entry_price + TARGET_R_MULTIPLIER * risk


def _positive_price(value: Decimal) -> None:
    if not isinstance(value, Decimal) or not value.is_finite() or value <= 0:
        raise ValueError("price must be a finite positive Decimal")

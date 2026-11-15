"""Leveraged Futures paper trading executor module.

Provides PaperExecutor, FuturesPortfolio, and related execution models.
"""

from app.trading.paper import (
    ATR_STOP_MULTIPLIER,
    CORE_SLIPPAGE,
    DEFAULT_STARTING_CASH,
    FUTURES_FEE_RATE,
    MARGIN_ALLOCATION_FRACTION,
    SATELLITE_SLIPPAGE,
    TARGET_R_MULTIPLIER,
    FuturesPortfolio,
    PaperExecutor,
)

__all__ = [
    "PaperExecutor",
    "FuturesPortfolio",
    "CORE_SLIPPAGE",
    "SATELLITE_SLIPPAGE",
    "FUTURES_FEE_RATE",
    "MARGIN_ALLOCATION_FRACTION",
    "DEFAULT_STARTING_CASH",
    "ATR_STOP_MULTIPLIER",
    "TARGET_R_MULTIPLIER",
]

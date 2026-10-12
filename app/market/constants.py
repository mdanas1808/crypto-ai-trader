"""Shared market-data identifiers and request bounds."""

EXCHANGE = "binance_spot"
FIXED_SYMBOLS: tuple[str, ...] = (
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "ZECUSDT",
)
DEFAULT_DYNAMIC_SYMBOLS: tuple[str, ...] = (
    "TAOUSDT",
    "FETUSDT",
    "RENDERUSDT",
    "NEARUSDT",
    "INJUSDT",
)
SUPPORTED_SYMBOLS: set[str] = set(FIXED_SYMBOLS) | set(DEFAULT_DYNAMIC_SYMBOLS)
SUPPORTED_INTERVALS = frozenset({"1h", "4h"})
MAX_CANDLE_LIMIT = 1000
DEFAULT_CANDLE_LIMIT = 100

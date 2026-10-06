"""Shared market-data identifiers and request bounds."""

EXCHANGE = "binance_spot"
SUPPORTED_SYMBOLS = frozenset({"BTCUSDT", "ETHUSDT", "SOLUSDT"})
SUPPORTED_INTERVALS = frozenset({"1h", "4h"})
MAX_CANDLE_LIMIT = 1000
DEFAULT_CANDLE_LIMIT = 100

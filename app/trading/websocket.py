"""Trading WebSocket stream adapter and helpers.

Provides aliases and helpers for Binance WebSocket streaming, enforcing lowercase
symbol names in stream subscriptions and payloads.
"""

from app.market.websocket import (
    BINANCE_SPOT_WEBSOCKET_URL,
    BinanceSpotKlineStream,
    WebSocketMessageError,
    build_subscription_payload,
    format_kline_stream,
    parse_candle,
    parse_candle_open_observation,
    parse_kline_message,
)

__all__ = [
    "BINANCE_SPOT_WEBSOCKET_URL",
    "BinanceSpotKlineStream",
    "WebSocketMessageError",
    "build_subscription_payload",
    "format_kline_stream",
    "parse_candle",
    "parse_candle_open_observation",
    "parse_kline_message",
]

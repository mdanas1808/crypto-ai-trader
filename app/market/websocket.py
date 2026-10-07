"""Unauthenticated Binance Spot WebSocket adapter for completed klines."""

import json
from collections.abc import AsyncIterator, Iterable
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

import websockets
from pydantic import ValidationError

from app.market.constants import SUPPORTED_INTERVALS, SUPPORTED_SYMBOLS
from app.market.models import Candle

BINANCE_SPOT_WEBSOCKET_URL = "wss://stream.binance.com:9443/ws"
UTC_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


class WebSocketMessageError(ValueError):
    """A Binance WebSocket payload is malformed or inconsistent."""


def _timestamp_ms(value: Any, field: str) -> datetime:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise WebSocketMessageError(f"kline {field} must be a non-negative integer")
    try:
        return UTC_EPOCH + timedelta(milliseconds=value)
    except OverflowError as exc:
        raise WebSocketMessageError(f"kline {field} is outside the supported range") from exc


def _decimal_string(value: Any, field: str) -> Decimal:
    if not isinstance(value, str):
        raise WebSocketMessageError(f"kline {field} must be a decimal string")
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise WebSocketMessageError(f"kline {field} is not a valid decimal") from exc
    if not number.is_finite():
        raise WebSocketMessageError(f"kline {field} must be finite")
    return number


def _decode_payload(payload: str | bytes) -> dict[str, Any]:
    if isinstance(payload, bytes):
        try:
            payload = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise WebSocketMessageError("WebSocket payload must be UTF-8 JSON") from exc
    if not isinstance(payload, str):
        raise WebSocketMessageError("WebSocket payload must be JSON text")
    try:
        decoded = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise WebSocketMessageError("WebSocket payload is not valid JSON") from exc
    if not isinstance(decoded, dict):
        raise WebSocketMessageError("WebSocket payload must be a JSON object")
    return decoded


def parse_kline_message(payload: str | bytes) -> tuple[Candle, bool] | None:
    """Parse a Binance kline event; None denotes a subscription acknowledgement.

    The returned boolean is Binance's k.x closed flag and is authoritative.
    This parser never filters a forming candle; consumers must check the flag.
    """
    message = _decode_payload(payload)

    # Binance WebSocket API subscription acknowledgement:
    # {"result": null, "id": 1}
    if "result" in message and "id" in message and "data" not in message:
        if message["result"] is not None or isinstance(message["id"], bool) or not isinstance(message["id"], int):
            raise WebSocketMessageError("invalid subscription acknowledgement")
        return None

    stream_name = message.get("stream")
    event = message.get("data", message)
    if not isinstance(event, dict):
        raise WebSocketMessageError("kline event data must be a JSON object")
    if event.get("e") != "kline":
        raise WebSocketMessageError("expected a Binance kline event")

    kline = event.get("k")
    if not isinstance(kline, dict):
        raise WebSocketMessageError("kline event must contain a kline object")
    symbol = kline.get("s")
    interval = kline.get("i")
    if not isinstance(symbol, str) or symbol not in SUPPORTED_SYMBOLS:
        raise WebSocketMessageError(f"unsupported symbol: {symbol!r}")
    if not isinstance(interval, str) or interval not in SUPPORTED_INTERVALS:
        raise WebSocketMessageError(f"unsupported interval: {interval!r}")

    outer_symbols = (message.get("s"), event.get("s"))
    if any(outer_symbol is not None and outer_symbol != symbol for outer_symbol in outer_symbols):
        raise WebSocketMessageError("outer and inner symbols do not match")
    if stream_name is not None:
        expected_stream = f"{symbol.lower()}@kline_{interval}"
        if stream_name != expected_stream:
            raise WebSocketMessageError("stream name does not match kline symbol and interval")

    open_time = _timestamp_ms(kline.get("t"), "open time")
    close_time = _timestamp_ms(kline.get("T"), "close time")
    if close_time < open_time:
        raise WebSocketMessageError("kline close time precedes open time")
    closed = kline.get("x")
    if not isinstance(closed, bool):
        raise WebSocketMessageError("kline closed flag must be a boolean")

    values = {
        name: _decimal_string(kline.get(key), name)
        for name, key in (
            ("open", "o"),
            ("high", "h"),
            ("low", "l"),
            ("close", "c"),
            ("volume", "v"),
        )
    }
    try:
        candle = Candle(
            exchange="binance_spot",
            symbol=symbol,
            interval=interval,
            open_time=open_time,
            **values,
        )
    except (ValidationError, ValueError) as exc:
        raise WebSocketMessageError(f"invalid candle data: {exc}") from exc
    return candle, closed


class BinanceSpotKlineStream:
    """Subscribe to configured Binance Spot kline streams and yield closed candles."""

    def __init__(
        self,
        symbols: Iterable[str] = SUPPORTED_SYMBOLS,
        interval: str = "1h",
        *,
        url: str = BINANCE_SPOT_WEBSOCKET_URL,
    ) -> None:
        configured = tuple(symbols)
        if not configured:
            raise ValueError("at least one symbol must be configured")
        if any(not isinstance(symbol, str) for symbol in configured):
            raise ValueError("configured symbols must be strings")
        unsupported = set(configured).difference(SUPPORTED_SYMBOLS)
        if unsupported:
            raise ValueError(f"unsupported symbols: {', '.join(sorted(unsupported))}")
        if not isinstance(interval, str) or interval not in SUPPORTED_INTERVALS:
            raise ValueError(f"unsupported interval: {interval}")
        if not isinstance(url, str) or not url:
            raise ValueError("url must be a non-empty string")
        self.symbols = tuple(sorted(set(configured)))
        self.interval = interval
        self.url = url

    @property
    def stream_names(self) -> tuple[str, ...]:
        return tuple(f"{symbol.lower()}@kline_{self.interval}" for symbol in self.symbols)

    @property
    def subscription_message(self) -> str:
        return json.dumps(
            {"method": "SUBSCRIBE", "params": list(self.stream_names), "id": 1},
            separators=(",", ":"),
        )

    async def candles(self) -> AsyncIterator[Candle]:
        """Connect once, subscribe, and yield only exchange-closed candles.

        Connection failures and malformed events propagate to the caller.
        Retry/reconnection policy intentionally belongs to a higher-level runner.
        """
        async with websockets.connect(self.url) as websocket:
            await websocket.send(self.subscription_message)
            async for payload in websocket:
                parsed = parse_kline_message(payload)
                if parsed is None:
                    continue
                candle, closed = parsed
                if candle.symbol not in self.symbols or candle.interval != self.interval:
                    raise WebSocketMessageError(
                        "received a candle outside the configured symbol/interval streams"
                    )
                if closed:
                    yield candle

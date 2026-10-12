"""Unauthenticated Binance Spot WebSocket adapter for completed klines."""

import json
import logging
from collections.abc import AsyncIterator, Iterable
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

import websockets
from pydantic import ValidationError

from app.market.constants import SUPPORTED_INTERVALS, SUPPORTED_SYMBOLS
from app.market.models import Candle, CandleOpenObservation

logger = logging.getLogger(__name__)

BINANCE_SPOT_WEBSOCKET_URL = "wss://stream.binance.us:9443/ws"
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


def _decode_payload(payload: str | bytes | dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, bytes):
        try:
            payload = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            logger.error("WebSocket payload must be UTF-8 JSON: %s", exc)
            raise WebSocketMessageError("WebSocket payload must be UTF-8 JSON") from exc
    if not isinstance(payload, str):
        logger.error("WebSocket payload must be JSON text or dict, got %s", type(payload).__name__)
        raise WebSocketMessageError("WebSocket payload must be JSON text")
    try:
        decoded = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.error("WebSocket payload is not valid JSON: %s", exc)
        raise WebSocketMessageError("WebSocket payload is not valid JSON") from exc
    if not isinstance(decoded, dict):
        logger.error("WebSocket payload must be a JSON object, got %s", type(decoded).__name__)
        raise WebSocketMessageError("WebSocket payload must be a JSON object")
    return decoded


def parse_kline_message(payload: str | bytes | dict[str, Any]) -> tuple[Candle, bool] | None:
    """Parse a Binance kline event; None denotes a subscription acknowledgement.

    The returned boolean is Binance's k.x closed flag and is authoritative.
    This parser never filters a forming candle; consumers must check the flag.
    """
    message = _decode_payload(payload)

    # Binance WebSocket API subscription acknowledgement:
    # {"result": null, "id": 1}
    if "result" in message and "id" in message and "data" not in message:
        if (
            message["result"] is not None
            or isinstance(message["id"], bool)
            or not isinstance(message["id"], int)
        ):
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


def parse_candle_open_observation(
    payload: str | bytes | dict[str, Any],
) -> CandleOpenObservation | None:
    """Parse only the first-update fields needed for a forming candle's open.

    A closed kline and a subscription acknowledgement return ``None``. The
    parser deliberately does not read forming high, low, close, or volume.
    """
    message = _decode_payload(payload)
    if "result" in message and "id" in message and "data" not in message:
        if (
            message["result"] is not None
            or isinstance(message["id"], bool)
            or not isinstance(message["id"], int)
        ):
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

    closed = kline.get("x")
    if not isinstance(closed, bool):
        raise WebSocketMessageError("kline closed flag must be a boolean")
    if closed:
        return None

    symbol = kline.get("s")
    interval = kline.get("i")
    if not isinstance(symbol, str) or symbol not in SUPPORTED_SYMBOLS:
        raise WebSocketMessageError(f"unsupported symbol: {symbol!r}")
    if not isinstance(interval, str) or interval not in SUPPORTED_INTERVALS:
        raise WebSocketMessageError(f"unsupported interval: {interval!r}")
    outer_symbols = (message.get("s"), event.get("s"))
    if any(outer_symbol is not None and outer_symbol != symbol for outer_symbol in outer_symbols):
        raise WebSocketMessageError("outer and inner symbols do not match")
    if stream_name is not None and stream_name != f"{symbol.lower()}@kline_{interval}":
        raise WebSocketMessageError("stream name does not match kline symbol and interval")

    event_time = event.get("E")
    if event_time is None:
        raise WebSocketMessageError("kline event must contain event time")
    try:
        observation = CandleOpenObservation(
            exchange="binance_spot",
            symbol=symbol,
            interval=interval,
            open_time=_timestamp_ms(kline.get("t"), "open time"),
            observed_at=_timestamp_ms(event_time, "event time"),
            open=_decimal_string(kline.get("o"), "open"),
        )
    except (ValidationError, ValueError) as exc:
        raise WebSocketMessageError(f"invalid candle open observation: {exc}") from exc
    return observation


class BinanceSpotKlineStream:
    """Subscribe to configured Binance Spot kline streams and yield closed candles."""

    def __init__(
        self,
        symbols: Iterable[str] = SUPPORTED_SYMBOLS,
        interval: str = "1h",
        *,
        url: str | None = None,
        base_url: str | None = None,
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
        resolved_url = (
            base_url
            if base_url is not None
            else (url if url is not None else BINANCE_SPOT_WEBSOCKET_URL)
        )
        if not isinstance(resolved_url, str) or not resolved_url:
            raise ValueError("url must be a non-empty string")
        self.symbols = tuple(sorted(set(configured)))
        self.interval = interval
        self.url = resolved_url

    @property
    def base_url(self) -> str:
        return self.url

    @property
    def stream_names(self) -> tuple[str, ...]:
        return tuple(f"{symbol.lower()}@kline_{self.interval}" for symbol in self.symbols)

    @property
    def subscription_message(self) -> str:
        return json.dumps(
            {"method": "SUBSCRIBE", "params": list(self.stream_names), "id": 1},
            separators=(",", ":"),
        )

    async def events(self) -> AsyncIterator[Candle | CandleOpenObservation]:
        """Yield first-seen open observations and completed candles.

        Forming updates are reduced to a CandleOpenObservation before they
        leave the adapter. Later updates for that same candle are suppressed.
        """
        last_open_times: dict[tuple[str, str, str], datetime] = {}
        async with websockets.connect(self.url) as websocket:
            await websocket.send(self.subscription_message)
            async for raw_message in websocket:
                try:
                    if isinstance(raw_message, (str, bytes)):
                        msg = json.loads(raw_message)
                    elif isinstance(raw_message, dict):
                        msg = raw_message
                    else:
                        logger.error(
                            "Unexpected WebSocket message type: %s",
                            type(raw_message).__name__,
                        )
                        continue
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    logger.error("Failed to parse incoming WebSocket JSON message: %s", exc)
                    continue
                except Exception as exc:
                    logger.error("Unexpected error decoding WebSocket message: %s", exc)
                    continue

                if not isinstance(msg, dict):
                    logger.error(
                        "WebSocket payload must be a JSON object, got %s",
                        type(msg).__name__,
                    )
                    continue

                if "data" in msg:
                    payload = msg["data"]
                else:
                    payload = msg

                observation = parse_candle_open_observation(payload)
                if observation is not None:
                    if (
                        observation.symbol not in self.symbols
                        or observation.interval != self.interval
                    ):
                        raise WebSocketMessageError(
                            "received an observation outside the configured symbol/interval streams"
                        )
                    identity = (observation.exchange, observation.symbol, observation.interval)
                    previous_open = last_open_times.get(identity)
                    if previous_open is None or observation.open_time > previous_open:
                        last_open_times[identity] = observation.open_time
                        yield observation
                    continue

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

    async def candles(self) -> AsyncIterator[Candle]:
        """Connect once, subscribe, and yield only exchange-closed candles.

        Connection failures and malformed events propagate to the caller.
        Retry/reconnection policy intentionally belongs to a higher-level runner.
        """
        async with websockets.connect(self.url) as websocket:
            await websocket.send(self.subscription_message)
            async for raw_message in websocket:
                try:
                    if isinstance(raw_message, (str, bytes)):
                        msg = json.loads(raw_message)
                    elif isinstance(raw_message, dict):
                        msg = raw_message
                    else:
                        logger.error(
                            "Unexpected WebSocket message type: %s",
                            type(raw_message).__name__,
                        )
                        continue
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    logger.error("Failed to parse incoming WebSocket JSON message: %s", exc)
                    continue
                except Exception as exc:
                    logger.error("Unexpected error decoding WebSocket message: %s", exc)
                    continue

                if not isinstance(msg, dict):
                    logger.error(
                        "WebSocket payload must be a JSON object, got %s",
                        type(msg).__name__,
                    )
                    continue

                if "data" in msg:
                    payload = msg["data"]
                else:
                    payload = msg

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

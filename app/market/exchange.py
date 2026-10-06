"""Asynchronous public Binance Spot candlestick client."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx

from app.market.constants import (
    EXCHANGE,
    MAX_CANDLE_LIMIT,
    SUPPORTED_INTERVALS,
    SUPPORTED_SYMBOLS,
)
from app.market.models import Candle

BASE_URL = "https://data-api.binance.vision"
KLINES_PATH = "/api/v3/klines"
SERVER_TIME_PATH = "/api/v3/time"
REQUEST_TIMEOUT = httpx.Timeout(10.0, connect=3.0)
UTC_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


class ExchangeClientError(RuntimeError):
    """Base error for public exchange market-data requests."""


class ExchangeTransportError(ExchangeClientError):
    """The request could not be completed because of a network or timeout error."""


class ExchangeHTTPError(ExchangeClientError):
    """The exchange returned a non-success HTTP status."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(f"Binance market-data request returned HTTP {status_code}")


class ExchangeResponseError(ExchangeClientError):
    """The exchange response did not match the documented kline format."""


def _timestamp_ms(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ExchangeResponseError(f"kline {field_name} must be a non-negative integer")
    return value


def _timestamp_from_ms(value: Any, field_name: str) -> datetime:
    try:
        return UTC_EPOCH + timedelta(milliseconds=_timestamp_ms(value, field_name))
    except OverflowError as exc:
        raise ExchangeResponseError(f"kline {field_name} is outside the supported range") from exc


def normalize_binance_kline(row: Any, symbol: str, interval: str) -> Candle:
    """Convert one documented 12-field Binance Spot kline tuple to a Candle."""
    if not isinstance(row, list) or len(row) != 12:
        raise ExchangeResponseError("each kline must be a 12-element JSON array")

    open_time = _timestamp_from_ms(row[0], "open time")
    _timestamp_from_ms(row[6], "close time")
    decimal_values: list[Decimal] = []
    for field_name, value in zip(("open", "high", "low", "close", "volume"), row[1:6]):
        if not isinstance(value, str):
            raise ExchangeResponseError(f"kline {field_name} must be a decimal string")
        try:
            decimal_values.append(Decimal(value))
        except InvalidOperation as exc:
            raise ExchangeResponseError(f"kline {field_name} is not a valid decimal") from exc

    try:
        return Candle(
            exchange=EXCHANGE,
            symbol=symbol,
            interval=interval,
            open_time=open_time,
            open=decimal_values[0],
            high=decimal_values[1],
            low=decimal_values[2],
            close=decimal_values[3],
            volume=decimal_values[4],
        )
    except ValueError as exc:
        raise ExchangeResponseError(f"invalid candle data: {exc}") from exc


class BinanceSpotClient:
    """Public, unauthenticated Binance Spot REST client for historical klines."""

    def __init__(
        self,
        *,
        timeout: httpx.Timeout = REQUEST_TIMEOUT,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=BASE_URL,
            timeout=timeout,
            transport=transport,
        )

    async def __aenter__(self) -> "BinanceSpotClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _get_json(self, path: str, *, params: dict[str, str | int] | None = None) -> Any:
        try:
            response = await self._client.get(path, params=params)
        except httpx.TimeoutException as exc:
            raise ExchangeTransportError("Binance market-data request timed out") from exc
        except httpx.RequestError as exc:
            raise ExchangeTransportError("could not connect to Binance market-data endpoint") from exc

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ExchangeHTTPError(exc.response.status_code) from exc

        try:
            return response.json()
        except ValueError as exc:
            raise ExchangeResponseError("Binance response was not valid JSON") from exc

    async def _server_time_ms(self) -> int:
        payload = await self._get_json(SERVER_TIME_PATH)
        if not isinstance(payload, dict):
            raise ExchangeResponseError("Binance server-time response must be a JSON object")
        server_time = payload.get("serverTime")
        if isinstance(server_time, bool) or not isinstance(server_time, int) or server_time < 0:
            raise ExchangeResponseError("Binance serverTime must be a non-negative integer")
        return server_time

    async def fetch_candles(self, symbol: str, interval: str, limit: int = 100) -> list[Candle]:
        if symbol not in SUPPORTED_SYMBOLS:
            raise ValueError(f"unsupported symbol: {symbol}")
        if interval not in SUPPORTED_INTERVALS:
            raise ValueError(f"unsupported interval: {interval}")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_CANDLE_LIMIT:
            raise ValueError(f"limit must be between 1 and {MAX_CANDLE_LIMIT}")

        # Ask for one extra row so excluding Binance's still-open candle can still
        # return the requested count in the normal case. Binance caps this at 1000.
        request_limit = min(limit + 1, MAX_CANDLE_LIMIT)
        # Sample exchange time before fetching klines. A candle that closes while
        # the kline request is in flight must remain excluded from that response.
        server_time_ms = await self._server_time_ms()
        payload = await self._get_json(
            KLINES_PATH,
            params={"symbol": symbol, "interval": interval, "limit": request_limit},
        )
        if not isinstance(payload, list):
            raise ExchangeResponseError("Binance kline response must be a JSON array")

        # Validate and normalize every response row before deciding which are
        # closed. A malformed row must not be hidden merely because it looks open.
        rows: list[tuple[int, Candle]] = []
        for row in payload:
            if not isinstance(row, list) or len(row) != 12:
                raise ExchangeResponseError("each kline must be a 12-element JSON array")
            close_time_ms = _timestamp_ms(row[6], "close time")
            rows.append((close_time_ms, normalize_binance_kline(row, symbol, interval)))

        # Binance defines this public server-time endpoint in milliseconds.
        # Equality is conservatively treated as not yet closed.
        candles = [candle for close_time_ms, candle in rows if close_time_ms < server_time_ms]

        candles.sort(key=lambda candle: candle.open_time)
        return candles[-limit:]

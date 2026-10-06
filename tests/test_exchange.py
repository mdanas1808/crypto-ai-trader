import asyncio
from datetime import datetime, timezone
from decimal import Decimal

import httpx
import pytest

from app.market.exchange import (
    BinanceSpotClient,
    ExchangeHTTPError,
    ExchangeResponseError,
    ExchangeTransportError,
    normalize_binance_kline,
)


def closed_kline() -> list[object]:
    open_time = 1_704_067_200_000
    close_time = open_time + 3_599_999
    return [
        open_time,
        "42000.12345678",
        "42100.12345678",
        "41900.12345678",
        "42050.12345678",
        "12.34567890",
        close_time,
        "518000.00",
        1200,
        "6.0000",
        "250000.00",
        "0",
    ]


def test_exchange_response_normalizes_to_candle() -> None:
    candle = normalize_binance_kline(closed_kline(), "BTCUSDT", "1h")
    assert candle.exchange == "binance_spot"
    assert candle.open == Decimal("42000.12345678")
    assert candle.open_time == datetime(2024, 1, 1, tzinfo=timezone.utc)


def test_invalid_exchange_response_is_rejected() -> None:
    with pytest.raises(ExchangeResponseError, match="12-element"):
        normalize_binance_kline([1, "1"], "BTCUSDT", "1h")


def test_invalid_timestamp_is_rejected() -> None:
    row = closed_kline()
    row[0] = "1704067200000"
    with pytest.raises(ExchangeResponseError, match="open time"):
        normalize_binance_kline(row, "BTCUSDT", "1h")


@pytest.mark.parametrize(
    ("close_offset_ms", "expected"),
    [
        pytest.param(1, False, id="close-after-server-time-excluded"),
        pytest.param(-1, True, id="close-before-server-time-included"),
        pytest.param(0, False, id="exact-boundary-excluded"),
    ],
)
def test_client_uses_authoritative_server_time_for_candle_closure(
    close_offset_ms: int, expected: bool
) -> None:
    # A candle closes only when close_time < Binance serverTime. Equality is
    # deliberately excluded because the candle is not proven closed at the boundary.
    row = closed_kline()
    close_time_ms = row[6]
    server_time_ms = close_time_ms - close_offset_ms
    request_paths: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        request_paths.append(request.url.path)
        if request.url.path == "/api/v3/klines":
            assert request.url.params["symbol"] == "BTCUSDT"
            assert request.url.params["interval"] == "1h"
            assert request.url.params["limit"] == "2"
            return httpx.Response(200, json=[row])
        assert request.url.path == "/api/v3/time"
        return httpx.Response(200, json={"serverTime": server_time_ms})

    async def exercise() -> list[object]:
        async with BinanceSpotClient(transport=httpx.MockTransport(handler)) as client:
            return await client.fetch_candles("BTCUSDT", "1h", 1)

    candles = asyncio.run(exercise())
    assert bool(candles) is expected
    assert request_paths == ["/api/v3/time", "/api/v3/klines"]


def test_client_never_returns_more_than_requested_limit() -> None:
    first = closed_kline()
    second = list(first)
    third = list(first)
    second[0] += 3_600_000
    second[6] += 3_600_000
    third[0] += 7_200_000
    third[6] += 7_200_000

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v3/klines":
            assert request.url.params["limit"] == "3"
            return httpx.Response(200, json=[first, second, third])
        return httpx.Response(200, json={"serverTime": third[6] + 1})

    async def exercise() -> list[object]:
        async with BinanceSpotClient(transport=httpx.MockTransport(handler)) as client:
            return await client.fetch_candles("BTCUSDT", "1h", 2)

    candles = asyncio.run(exercise())
    assert len(candles) == 2
    assert [candle.open_time for candle in candles] == [
        datetime(2024, 1, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 1, 2, tzinfo=timezone.utc),
    ]


def test_client_reports_http_status_error() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    async def exercise() -> None:
        async with BinanceSpotClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(ExchangeHTTPError, match="HTTP 503"):
                await client.fetch_candles("BTCUSDT", "1h", 1)

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(200, content=b"not-json"), "not valid JSON"),
        (httpx.Response(200, json={"error": "not an array"}), "JSON array"),
    ],
    ids=["malformed-json", "non-array-json"],
)
def test_client_rejects_malformed_exchange_payload(
    response: httpx.Response, message: str
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v3/time":
            return httpx.Response(200, json={"serverTime": 1_704_070_800_000})
        return response

    async def exercise() -> None:
        async with BinanceSpotClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(ExchangeResponseError, match=message):
                await client.fetch_candles("BTCUSDT", "1h", 1)

    asyncio.run(exercise())


def test_client_rejects_invalid_decimal_in_exchange_payload() -> None:
    row = closed_kline()
    row[1] = "not-a-decimal"

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v3/time":
            return httpx.Response(200, json={"serverTime": 1_704_070_800_000})
        return httpx.Response(200, json=[row])

    async def exercise() -> None:
        async with BinanceSpotClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(ExchangeResponseError, match="valid decimal"):
                await client.fetch_candles("BTCUSDT", "1h", 1)

    asyncio.run(exercise())


@pytest.mark.parametrize("error_type", [httpx.ConnectError, httpx.ReadTimeout])
def test_client_reports_transport_and_timeout_errors(error_type: type[httpx.RequestError]) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise error_type("simulated request failure", request=request)

    async def exercise() -> None:
        async with BinanceSpotClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(ExchangeTransportError):
                await client.fetch_candles("BTCUSDT", "1h", 1)

    asyncio.run(exercise())

import asyncio
import json
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from app.market.models import Candle
from app.market.websocket import (
    BINANCE_SPOT_WEBSOCKET_URL,
    BinanceSpotKlineStream,
    WebSocketMessageError,
    parse_kline_message,
)

OPEN_MS = 1_704_067_200_000
CLOSE_MS = OPEN_MS + 3_599_999


def payload(*, closed=True, **kline_updates):
    kline = {
        "t": OPEN_MS,
        "T": CLOSE_MS,
        "s": "BTCUSDT",
        "i": "1h",
        "o": "42000.12345678",
        "h": "42100.12345678",
        "l": "41900.12345678",
        "c": "42050.12345678",
        "v": "12.34567890",
        "x": closed,
    }
    kline.update(kline_updates)
    return json.dumps({"e": "kline", "s": "BTCUSDT", "k": kline})


def test_valid_closed_one_hour_kline_normalizes_candle_and_decimal_values():
    candle, closed = parse_kline_message(payload())

    assert closed is True
    assert isinstance(candle, Candle)
    assert candle.exchange == "binance_spot"
    assert candle.symbol == "BTCUSDT"
    assert candle.interval == "1h"
    assert candle.open_time == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert candle.open == Decimal("42000.12345678")
    assert candle.high == Decimal("42100.12345678")
    assert candle.low == Decimal("41900.12345678")
    assert candle.close == Decimal("42050.12345678")
    assert candle.volume == Decimal("12.34567890")
    assert all(isinstance(getattr(candle, field), Decimal) for field in ("open", "high", "low", "close", "volume"))


def test_forming_kline_is_recognized_as_open():
    candle, closed = parse_kline_message(payload(closed=False))
    assert candle.symbol == "BTCUSDT"
    assert closed is False


def test_binance_closed_flag_is_authoritative():
    assert parse_kline_message(payload(closed=True))[1] is True
    assert parse_kline_message(payload(closed=False))[1] is False


def test_outer_and_inner_symbol_mismatch_is_rejected():
    data = json.loads(payload())
    data["s"] = "ETHUSDT"
    with pytest.raises(WebSocketMessageError, match="symbols do not match"):
        parse_kline_message(json.dumps(data))


def test_combined_stream_outer_symbol_mismatch_is_rejected():
    data = {
        "stream": "ethusdt@kline_1h",
        "s": "ETHUSDT",
        "data": json.loads(payload()),
    }
    with pytest.raises(WebSocketMessageError, match="symbols do not match"):
        parse_kline_message(json.dumps(data))


def test_invalid_ohlc_relationship_is_rejected():
    with pytest.raises(WebSocketMessageError, match="invalid candle data"):
        parse_kline_message(payload(h="42000"))


def test_invalid_close_timestamp_is_rejected():
    with pytest.raises(WebSocketMessageError, match="close time"):
        parse_kline_message(payload(T="1704067200000"))


@pytest.mark.parametrize("symbol", ["BNBUSDT", ""])
def test_unexpected_symbol_is_rejected(symbol):
    with pytest.raises(WebSocketMessageError, match="unsupported symbol"):
        parse_kline_message(payload(s=symbol))


def test_unsupported_interval_is_rejected():
    with pytest.raises(WebSocketMessageError, match="unsupported interval"):
        parse_kline_message(payload(i="15m"))


def test_non_kline_event_is_rejected():
    with pytest.raises(WebSocketMessageError, match="expected a Binance kline event"):
        parse_kline_message(json.dumps({"e": "trade", "s": "BTCUSDT"}))


def test_malformed_kline_object_is_rejected():
    with pytest.raises(WebSocketMessageError, match="kline object"):
        parse_kline_message(json.dumps({"e": "kline", "k": []}))


def test_non_object_json_is_rejected():
    with pytest.raises(WebSocketMessageError, match="JSON object"):
        parse_kline_message("[]")


def test_stream_rejects_unsupported_configured_interval():
    with pytest.raises(ValueError, match="unsupported interval"):
        BinanceSpotKlineStream(symbols=("BTCUSDT",), interval="15m")


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"o": "not-decimal"}, "valid decimal"),
        ({"h": "NaN"}, "finite"),
        ({"c": 42000.0}, "decimal string"),
        ({"v": 12.5}, "decimal string"),
    ],
)
def test_invalid_numeric_values_are_rejected(updates, message):
    with pytest.raises(WebSocketMessageError, match=message):
        parse_kline_message(payload(**updates))


@pytest.mark.parametrize("closed", [0, 1, "true", None])
def test_non_boolean_closed_flag_is_rejected(closed):
    with pytest.raises(WebSocketMessageError, match="closed flag must be a boolean"):
        parse_kline_message(payload(closed=closed))


@pytest.mark.parametrize("timestamp", ["1704067200000", 1.5, True, -1])
def test_invalid_timestamp_types_and_values_are_rejected(timestamp):
    with pytest.raises(WebSocketMessageError, match="open time"):
        parse_kline_message(payload(t=timestamp))


def test_subscription_stream_names_are_deterministic():
    stream = BinanceSpotKlineStream(symbols=("SOLUSDT", "BTCUSDT", "ETHUSDT"))

    assert stream.stream_names == (
        "btcusdt@kline_1h",
        "ethusdt@kline_1h",
        "solusdt@kline_1h",
    )
    assert stream.url == BINANCE_SPOT_WEBSOCKET_URL


def test_subscription_message_is_deterministic_and_valid():
    stream = BinanceSpotKlineStream(symbols=("ETHUSDT", "BTCUSDT"))

    assert stream.subscription_message == (
        '{"method":"SUBSCRIBE","params":["btcusdt@kline_1h","ethusdt@kline_1h"],"id":1}'
    )
    assert json.loads(stream.subscription_message) == {
        "method": "SUBSCRIBE",
        "params": ["btcusdt@kline_1h", "ethusdt@kline_1h"],
        "id": 1,
    }


def test_empty_symbol_configuration_is_rejected():
    with pytest.raises(ValueError, match="at least one symbol"):
        BinanceSpotKlineStream(symbols=())


def test_unsupported_symbol_configuration_is_rejected():
    with pytest.raises(ValueError, match="unsupported symbols"):
        BinanceSpotKlineStream(symbols=("BNBUSDT",))


def test_malformed_json_is_rejected():
    with pytest.raises(WebSocketMessageError, match="not valid JSON"):
        parse_kline_message("{bad-json")


def test_subscription_acknowledgement_is_not_a_market_candle():
    assert parse_kline_message('{"result":null,"id":1}') is None


class FakeWebSocket:
    def __init__(self, messages):
        self.messages = messages
        self.sent = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None

    async def send(self, message):
        self.sent.append(message)

    def __aiter__(self):
        async def messages():
            for item in self.messages:
                yield item
        return messages()


def test_stream_never_yields_forming_candles(monkeypatch):
    fake = FakeWebSocket(
        [
            '{"result":null,"id":1}',
            payload(closed=False),
            payload(closed=True),
        ]
    )
    calls = []

    def fake_connect(url):
        calls.append(url)
        return fake

    monkeypatch.setattr("app.market.websocket.websockets.connect", fake_connect)
    stream = BinanceSpotKlineStream(symbols=("BTCUSDT",))

    async def collect():
        return [candle async for candle in stream.candles()]

    candles = asyncio.run(collect())
    assert calls == [BINANCE_SPOT_WEBSOCKET_URL]
    assert fake.sent == [stream.subscription_message]
    assert len(candles) == 1
    assert candles[0].open_time == datetime(2024, 1, 1, tzinfo=timezone.utc)

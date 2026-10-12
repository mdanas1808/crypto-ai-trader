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
    assert all(
        isinstance(getattr(candle, field), Decimal)
        for field in ("open", "high", "low", "close", "volume")
    )


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


@pytest.mark.parametrize("symbol", ["DOGEUSDT", ""])
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
    assert stream.base_url == "wss://stream.binance.us:9443/ws"


def test_stream_custom_base_url():
    stream = BinanceSpotKlineStream(symbols=("BTCUSDT",), base_url="wss://custom.stream:9443/ws")
    assert stream.url == "wss://custom.stream:9443/ws"
    assert stream.base_url == "wss://custom.stream:9443/ws"


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
        BinanceSpotKlineStream(symbols=("DOGEUSDT",))


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


def event_payload(*, closed=False, event_time=OPEN_MS + 1000, **updates):
    event = json.loads(payload(closed=closed, **updates))
    event["E"] = event_time
    return json.dumps(event)


def test_forming_update_parses_minimal_open_observation_and_ignores_later_ohlcv():
    from app.market.models import CandleOpenObservation
    from app.market.websocket import parse_candle_open_observation

    observation = parse_candle_open_observation(
        event_payload(closed=False, h="not-an-open-field", l=None, c={}, v=object().__str__())
    )

    assert isinstance(observation, CandleOpenObservation)
    assert observation.exchange == "binance_spot"
    assert observation.symbol == "BTCUSDT"
    assert observation.interval == "1h"
    assert observation.open_time == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert observation.observed_at == datetime(2024, 1, 1, 0, 0, 1, tzinfo=timezone.utc)
    assert observation.open == Decimal("42000.12345678")
    assert not hasattr(observation, "high")
    assert not hasattr(observation, "close")
    assert not hasattr(observation, "volume")


def test_open_observation_parser_ignores_ack_and_completed_kline():
    from app.market.websocket import parse_candle_open_observation

    assert parse_candle_open_observation('{"result":null,"id":1}') is None
    assert parse_candle_open_observation(event_payload(closed=True)) is None


def test_event_stream_emits_only_first_open_update_and_keeps_closed_candle(monkeypatch):
    from app.market.models import CandleOpenObservation

    fake = FakeWebSocket(
        [
            '{"result":null,"id":1}',
            event_payload(closed=False, event_time=OPEN_MS + 1000),
            event_payload(closed=False, event_time=OPEN_MS + 2000, o="999", h="invalid"),
            event_payload(closed=True),
        ]
    )
    monkeypatch.setattr("app.market.websocket.websockets.connect", lambda _url: fake)
    stream = BinanceSpotKlineStream(symbols=("BTCUSDT",))

    async def collect():
        return [event async for event in stream.events()]

    events = asyncio.run(collect())
    assert fake.sent == [stream.subscription_message]
    assert len(events) == 2
    assert isinstance(events[0], CandleOpenObservation)
    assert events[0].open == Decimal("42000.12345678")
    assert isinstance(events[1], Candle)
    assert events[1].close == Decimal("42050.12345678")


def test_forming_open_observation_requires_binance_event_time():
    from app.market.websocket import parse_candle_open_observation

    data = json.loads(event_payload(closed=False))
    del data["E"]
    with pytest.raises(WebSocketMessageError, match="event time"):
        parse_candle_open_observation(json.dumps(data))


def test_stream_candles_with_combined_stream_data_payload(monkeypatch):
    combined_message = json.dumps(
        {
            "stream": "btcusdt@kline_1h",
            "data": json.loads(payload(closed=True)),
        }
    )
    fake = FakeWebSocket(
        [
            '{"result":null,"id":1}',
            combined_message,
        ]
    )
    monkeypatch.setattr("app.market.websocket.websockets.connect", lambda _url: fake)
    stream = BinanceSpotKlineStream(symbols=("BTCUSDT",))

    async def collect():
        return [candle async for candle in stream.candles()]

    candles = asyncio.run(collect())
    assert len(candles) == 1
    assert candles[0].symbol == "BTCUSDT"
    assert candles[0].close == Decimal("42050.12345678")


def test_stream_events_with_combined_stream_data_payload(monkeypatch):
    from app.market.models import CandleOpenObservation

    open_combined = json.dumps(
        {
            "stream": "btcusdt@kline_1h",
            "data": json.loads(event_payload(closed=False, event_time=OPEN_MS + 1000)),
        }
    )
    closed_combined = json.dumps(
        {
            "stream": "btcusdt@kline_1h",
            "data": json.loads(event_payload(closed=True)),
        }
    )
    fake = FakeWebSocket(
        [
            '{"result":null,"id":1}',
            open_combined,
            closed_combined,
        ]
    )
    monkeypatch.setattr("app.market.websocket.websockets.connect", lambda _url: fake)
    stream = BinanceSpotKlineStream(symbols=("BTCUSDT",))

    async def collect():
        return [event async for event in stream.events()]

    events = asyncio.run(collect())
    assert len(events) == 2
    assert isinstance(events[0], CandleOpenObservation)
    assert events[0].symbol == "BTCUSDT"
    assert events[0].open == Decimal("42000.12345678")
    assert isinstance(events[1], Candle)
    assert events[1].symbol == "BTCUSDT"
    assert events[1].close == Decimal("42050.12345678")


def test_stream_logs_error_on_invalid_json_and_continues(monkeypatch, caplog):
    import logging

    fake = FakeWebSocket(
        [
            '{"result":null,"id":1}',
            "{malformed-json-here",
            payload(closed=True),
        ]
    )
    monkeypatch.setattr("app.market.websocket.websockets.connect", lambda _url: fake)
    stream = BinanceSpotKlineStream(symbols=("BTCUSDT",))

    async def collect():
        return [candle async for candle in stream.candles()]

    with caplog.at_level(logging.ERROR):
        candles = asyncio.run(collect())

    assert len(candles) == 1
    assert any(
        "Failed to parse incoming WebSocket JSON message" in record.message
        for record in caplog.records
    )


def test_stream_logs_error_on_non_dict_json(monkeypatch, caplog):
    import logging

    fake = FakeWebSocket(
        [
            '{"result":null,"id":1}',
            "12345",
            payload(closed=True),
        ]
    )
    monkeypatch.setattr("app.market.websocket.websockets.connect", lambda _url: fake)
    stream = BinanceSpotKlineStream(symbols=("BTCUSDT",))

    async def collect():
        return [candle async for candle in stream.candles()]

    with caplog.at_level(logging.ERROR):
        candles = asyncio.run(collect())

    assert len(candles) == 1
    assert any(
        "WebSocket payload must be a JSON object" in record.message for record in caplog.records
    )


def test_parse_kline_message_accepts_dict_directly():
    raw_dict = json.loads(payload())
    candle, closed = parse_kline_message(raw_dict)
    assert closed is True
    assert candle.symbol == "BTCUSDT"

    wrapped_dict = {"stream": "btcusdt@kline_1h", "data": raw_dict}
    candle_w, closed_w = parse_kline_message(wrapped_dict)
    assert closed_w is True
    assert candle_w.symbol == "BTCUSDT"


def test_parse_candle_open_observation_accepts_dict_directly():
    from app.market.models import CandleOpenObservation
    from app.market.websocket import parse_candle_open_observation

    raw_dict = json.loads(event_payload(closed=False))
    obs = parse_candle_open_observation(raw_dict)
    assert isinstance(obs, CandleOpenObservation)
    assert obs.symbol == "BTCUSDT"

    wrapped_dict = {"stream": "btcusdt@kline_1h", "data": raw_dict}
    obs_w = parse_candle_open_observation(wrapped_dict)
    assert isinstance(obs_w, CandleOpenObservation)
    assert obs_w.symbol == "BTCUSDT"

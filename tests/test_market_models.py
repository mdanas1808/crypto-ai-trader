from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.market.constants import EXCHANGE
from app.market.models import Candle
from app.market.validation import CandleSequenceValidationError, validate_candle_sequence


def valid_values() -> dict[str, object]:
    return {
        "exchange": EXCHANGE,
        "symbol": "BTCUSDT",
        "interval": "1h",
        "open_time": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "open": "42000.12345678",
        "high": "42100.12345678",
        "low": "41900.12345678",
        "close": "42050.12345678",
        "volume": "12.34567890",
    }


def test_candle_keeps_exact_decimal_values_and_utc_time() -> None:
    candle = Candle.model_validate(valid_values())
    assert candle.open == Decimal("42000.12345678")
    assert candle.volume == Decimal("12.34567890")
    assert candle.open_time.tzinfo == timezone.utc


def test_candle_rejects_binary_float() -> None:
    values = valid_values()
    values["open"] = 42000.12
    with pytest.raises(ValidationError, match="decimal strings or Decimal"):
        Candle.model_validate(values)


def test_candle_rejects_invalid_ohlc() -> None:
    values = valid_values()
    values["high"] = "42010"
    with pytest.raises(ValidationError, match="high must be greater"):
        Candle.model_validate(values)


def test_candle_rejects_unsupported_symbol_and_interval() -> None:
    with pytest.raises(ValidationError, match="unsupported symbol"):
        Candle.model_validate({**valid_values(), "symbol": "DOGEUSDT"})
    with pytest.raises(ValidationError, match="unsupported interval"):
        Candle.model_validate({**valid_values(), "interval": "15m"})


def candles_at(interval: str, hours: tuple[int, ...]) -> list[Candle]:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    step = timedelta(hours=int(interval[:-1]))
    return [
        Candle.model_validate(
            {**valid_values(), "interval": interval, "open_time": start + step * hour}
        )
        for hour in hours
    ]


def test_sequence_validator_accepts_contiguous_hourly_and_four_hour_candles() -> None:
    validate_candle_sequence(candles_at("1h", (0, 1, 2)))
    validate_candle_sequence(candles_at("4h", (0, 1, 2)))


@pytest.mark.parametrize(
    ("interval", "hours", "missing"),
    [
        pytest.param("1h", (0, 2), 1, id="one-hour-gap"),
        pytest.param("1h", (0, 3), 2, id="multiple-hour-gaps"),
        pytest.param("4h", (0, 2), 1, id="four-hour-gap"),
    ],
)
def test_sequence_validator_reports_gap_context(
    interval: str, hours: tuple[int, ...], missing: int
) -> None:
    candles = candles_at(interval, hours)
    with pytest.raises(CandleSequenceValidationError) as captured:
        validate_candle_sequence(candles)

    error = captured.value
    assert error.reason == "gap"
    assert error.exchange == EXCHANGE
    assert error.symbol == "BTCUSDT"
    assert error.interval == interval
    assert error.previous_timestamp == candles[0].open_time
    assert error.next_timestamp == candles[1].open_time
    assert error.missing_intervals == missing
    duration = timedelta(hours=int(interval[:-1]))
    assert error.expected_timestamps == tuple(
        candles[0].open_time + duration * index for index in range(1, missing + 1)
    )


@pytest.mark.parametrize(
    ("hours", "reason"),
    [
        pytest.param((1, 0), "out_of_order", id="unsorted"),
        pytest.param((0, 0), "duplicate", id="duplicate"),
    ],
)
def test_sequence_validator_rejects_non_strict_order(
    hours: tuple[int, int], reason: str
) -> None:
    with pytest.raises(CandleSequenceValidationError) as captured:
        validate_candle_sequence(candles_at("1h", hours))
    assert captured.value.reason == reason


@pytest.mark.parametrize(
    ("interval", "offset"),
    [
        pytest.param("1h", timedelta(minutes=1), id="misaligned-hour"),
        pytest.param("4h", timedelta(hours=1), id="misaligned-four-hour"),
    ],
)
def test_sequence_validator_rejects_misaligned_utc_timestamps(
    interval: str, offset: timedelta
) -> None:
    candle = candles_at(interval, (0,))[0]
    misaligned = candle.model_copy(update={"open_time": candle.open_time + offset})
    with pytest.raises(CandleSequenceValidationError) as captured:
        validate_candle_sequence([misaligned])
    assert captured.value.reason == "misaligned"
    assert captured.value.expected_timestamps == (candle.open_time,)


def valid_open_observation_values() -> dict[str, object]:
    return {
        "exchange": EXCHANGE,
        "symbol": "BTCUSDT",
        "interval": "1h",
        "open_time": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "observed_at": datetime(2024, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
        "open": "42000.12345678",
    }


def test_candle_open_observation_is_immutable_decimal_and_utc_normalized() -> None:
    from datetime import timedelta

    from app.market.models import CandleOpenObservation

    values = valid_open_observation_values()
    values["open_time"] = datetime(2024, 1, 1, 5, tzinfo=timezone(timedelta(hours=5)))
    values["observed_at"] = datetime(2024, 1, 1, 5, 0, 1, tzinfo=timezone(timedelta(hours=5)))
    observation = CandleOpenObservation.model_validate(values)

    assert observation.open_time == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert observation.observed_at == datetime(2024, 1, 1, 0, 0, 1, tzinfo=timezone.utc)
    assert observation.open == Decimal("42000.12345678")
    with pytest.raises(ValidationError):
        observation.open = Decimal("1")


def test_candle_open_observation_rejects_unsupported_identity_and_naive_times() -> None:
    from app.market.models import CandleOpenObservation

    for field, value, message in (
        ("symbol", "BNBUSDT", "unsupported symbol"),
        ("interval", "15m", "unsupported interval"),
        ("open_time", datetime(2024, 1, 1), "timezone"),
        ("observed_at", datetime(2024, 1, 1), "timezone"),
    ):
        with pytest.raises(ValidationError, match=message):
            CandleOpenObservation.model_validate({**valid_open_observation_values(), field: value})


def test_candle_open_observation_rejects_binary_float() -> None:
    from app.market.models import CandleOpenObservation

    with pytest.raises(ValidationError, match="decimal string or Decimal"):
        CandleOpenObservation.model_validate({**valid_open_observation_values(), "open": 42000.5})


def test_candle_open_observation_rejects_forming_ohlcv_extras() -> None:
    from app.market.models import CandleOpenObservation

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CandleOpenObservation.model_validate(
            {**valid_open_observation_values(), "close": "42001"}
        )

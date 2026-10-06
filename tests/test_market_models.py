from datetime import datetime, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.market.constants import EXCHANGE
from app.market.models import Candle


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

"""Normalized, exchange-independent candle values."""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.market.constants import SUPPORTED_INTERVALS, SUPPORTED_SYMBOLS


class Candle(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    exchange: str = Field(min_length=1, max_length=32)
    symbol: str
    interval: str
    open_time: datetime
    open: Decimal = Field(gt=0, max_digits=38, decimal_places=18)
    high: Decimal = Field(gt=0, max_digits=38, decimal_places=18)
    low: Decimal = Field(gt=0, max_digits=38, decimal_places=18)
    close: Decimal = Field(gt=0, max_digits=38, decimal_places=18)
    volume: Decimal = Field(ge=0, max_digits=38, decimal_places=18)

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, value: str) -> str:
        if value not in SUPPORTED_SYMBOLS:
            raise ValueError(f"unsupported symbol: {value}")
        return value

    @field_validator("interval")
    @classmethod
    def validate_interval(cls, value: str) -> str:
        if value not in SUPPORTED_INTERVALS:
            raise ValueError(f"unsupported interval: {value}")
        return value

    @field_validator("open", "high", "low", "close", "volume", mode="before")
    @classmethod
    def reject_binary_float(cls, value: Any) -> Any:
        if isinstance(value, float):
            raise ValueError("candle numeric values must be decimal strings or Decimal values")
        return value

    @field_validator("open_time")
    @classmethod
    def require_aware_utc_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("open_time must include a timezone")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_ohlc_relationships(self) -> "Candle":
        if self.high < max(self.open, self.close):
            raise ValueError("high must be greater than or equal to open and close")
        if self.low > min(self.open, self.close):
            raise ValueError("low must be less than or equal to open and close")
        if self.high < self.low:
            raise ValueError("high must be greater than or equal to low")
        return self


class CandleOpenObservation(BaseModel):
    """Minimal observation of a forming candle's first seen open.

    It intentionally carries no evolving OHLCV values beyond the open.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    exchange: str = Field(min_length=1, max_length=32)
    symbol: str
    interval: str
    open_time: datetime
    observed_at: datetime
    open: Decimal = Field(gt=0, max_digits=38, decimal_places=18)

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, value: str) -> str:
        if value not in SUPPORTED_SYMBOLS:
            raise ValueError(f"unsupported symbol: {value}")
        return value

    @field_validator("interval")
    @classmethod
    def validate_interval(cls, value: str) -> str:
        if value not in SUPPORTED_INTERVALS:
            raise ValueError(f"unsupported interval: {value}")
        return value

    @field_validator("open", mode="before")
    @classmethod
    def reject_binary_float(cls, value: Any) -> Any:
        if isinstance(value, float):
            raise ValueError("open must be a decimal string or Decimal value")
        return value

    @field_validator("open_time", "observed_at")
    @classmethod
    def require_aware_utc_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamps must include a timezone")
        return value.astimezone(timezone.utc)

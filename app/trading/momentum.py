"""BTC Regime Filter for Satellite altcoin momentum entries.

Calculates the daily BTC/USDT trend relative to its 50-day Simple Moving Average (SMA).
Before submitting a new ENTRY signal for any dynamic Satellite altcoins, the session
checks this filter. If BTC is below its 50-day SMA (downtrend regime), new altcoin entries
are blocked. Exits and position reviews for existing positions are never blocked.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Awaitable, Callable, Sequence

import httpx

from app.market.models import Candle

logger = logging.getLogger(__name__)

BINANCE_KLINES_URL = "https://data-api.binance.vision/api/v3/klines"
DEFAULT_BTC_SYMBOL = "BTCUSDT"
DEFAULT_SMA_PERIOD = 50
DEFAULT_HYSTERESIS_PCT = Decimal("0.02")


class BtcRegimeFilter:
    """Evaluates whether BTC/USDT is in an uptrend or downtrend relative to its 50-day SMA.

    Uses completed daily candles and a 2% hysteresis buffer:
    - BTC must close at least 2% below the 50-day SMA to enter a downtrend.
    - BTC must close at least 2% above the 50-day SMA to resume an uptrend.
    - Within the 2% buffer, the previous regime state is maintained to avoid flip-flops.
    """

    def __init__(
        self,
        symbol: str = DEFAULT_BTC_SYMBOL,
        sma_period: int = DEFAULT_SMA_PERIOD,
        *,
        exchange_client: Any | None = None,
        http_client: httpx.AsyncClient | None = None,
        cache_ttl_seconds: float = 3600.0,
        candles_provider: Callable[[], Awaitable[Sequence[Any]] | Sequence[Any]] | None = None,
        hysteresis_pct: Decimal | float = DEFAULT_HYSTERESIS_PCT,
        initial_regime: str | None = None,
    ) -> None:
        self.symbol = symbol
        self.sma_period = max(1, sma_period)
        self.exchange_client = exchange_client
        self.http_client = http_client
        self.cache_ttl_seconds = cache_ttl_seconds
        self.candles_provider = candles_provider
        self.hysteresis_pct = Decimal(str(hysteresis_pct))
        self._current_regime: str | None = initial_regime

        self._cached_closes: list[Decimal] = []
        self._cached_time: datetime | None = None

    @property
    def current_regime(self) -> str | None:
        """Return the current regime state string (e.g. UPTREND or DOWNTREND)."""
        return self._current_regime

    def reset_regime(self, regime: str | None = None) -> None:
        """Reset internal hysteresis regime state."""
        self._current_regime = regime

    def _is_cache_valid(self, now: datetime) -> bool:
        if not self._cached_closes or self._cached_time is None:
            return False
        return (now - self._cached_time).total_seconds() < self.cache_ttl_seconds

    async def initialize(self) -> str:
        """Query historical daily BTC/USDT data on startup and replay the state machine over past 100 candles."""
        closes = await self.fetch_btc_daily_closes()
        if closes:
            self.replay_state_machine(closes, window=100)
        return self._current_regime or "UPTREND"

    def replay_state_machine(
        self,
        daily_closes: Sequence[Decimal | float | str],
        window: int = 100,
    ) -> str:
        """Replay the hysteresis state machine over historical completed daily candles.

        Calculates the 50-day SMA at each step and applies the 2% hysteresis buffer
        to accurately determine the regime state without guessing based purely on
        the most recent candle.
        """
        if not daily_closes:
            return self._current_regime or "UPTREND"

        decimal_closes = [Decimal(str(p)) for p in daily_closes]
        n = len(decimal_closes)
        start_idx = max(0, n - max(1, window))

        for i in range(start_idx, n):
            sample_start = max(0, i - self.sma_period + 1)
            sample = decimal_closes[sample_start : i + 1]
            sma_50 = sum(sample) / Decimal(str(len(sample)))
            price = decimal_closes[i]

            upper_threshold = sma_50 * (Decimal("1") + self.hysteresis_pct)
            lower_threshold = sma_50 * (Decimal("1") - self.hysteresis_pct)

            if self._current_regime is None:
                if price >= sma_50:
                    self._current_regime = "UPTREND"
                else:
                    self._current_regime = "DOWNTREND"
            elif self._current_regime == "UPTREND":
                if price <= lower_threshold:
                    self._current_regime = "DOWNTREND"
            elif self._current_regime == "DOWNTREND":
                if price >= upper_threshold:
                    self._current_regime = "UPTREND"

        return self._current_regime or "UPTREND"

    async def fetch_btc_daily_closes(self) -> list[Decimal]:
        """Fetch completed daily close prices for BTC/USDT."""
        now = datetime.now(timezone.utc)
        if self._is_cache_valid(now):
            return list(self._cached_closes)

        closes: list[Decimal] = []
        now_ms = int(now.timestamp() * 1000)
        fetch_limit = max(150, 100 + self.sma_period + 5)

        # 1. Custom candles provider
        if self.candles_provider is not None:
            try:
                res = self.candles_provider()
                raw_candles = await res if hasattr(res, "__await__") else res
                for c in raw_candles:
                    if isinstance(c, Candle):
                        closes.append(Decimal(str(c.close)))
                    elif isinstance(c, (dict, list, tuple)):
                        val = c["close"] if isinstance(c, dict) else c[4]
                        closes.append(Decimal(str(val)))
            except Exception as exc:
                logger.warning("Error fetching candles from candles_provider: %s", exc)

        # 2. Exchange client if available
        if not closes and self.exchange_client is not None:
            try:
                if hasattr(self.exchange_client, "fetch_candles"):
                    candles = await self.exchange_client.fetch_candles(
                        self.symbol, interval="1d", limit=fetch_limit
                    )
                    closes = [Decimal(str(c.close)) for c in candles]
            except Exception as exc:
                logger.warning("Error fetching daily candles from exchange_client: %s", exc)

        # 3. Direct unauthenticated Binance REST request
        if not closes:
            try:
                params = {
                    "symbol": self.symbol,
                    "interval": "1d",
                    "limit": fetch_limit,
                }
                if self.http_client is not None:
                    response = await self.http_client.get(BINANCE_KLINES_URL, params=params)
                else:
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        response = await client.get(BINANCE_KLINES_URL, params=params)

                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list):
                        for row in data:
                            if isinstance(row, list) and len(row) >= 5:
                                # If Binance 12-field format, check if candle is already closed
                                if len(row) >= 7:
                                    close_time_ms = int(row[6])
                                    if close_time_ms > now_ms:
                                        # Skip currently open, in-progress daily candle
                                        continue
                                closes.append(Decimal(str(row[4])))
                else:
                    logger.warning(
                        "Binance REST API returned HTTP %s for daily BTC klines",
                        response.status_code,
                    )
            except Exception as exc:
                logger.warning("Failed to fetch daily BTC klines: %s", exc)

        if closes:
            self._cached_closes = closes
            self._cached_time = now
            if self._current_regime is None:
                self.replay_state_machine(closes, window=100)

        return list(closes)

    def calculate_regime_sync(
        self,
        daily_closes: Sequence[Decimal | float | str],
        current_price: Decimal | float | str | None = None,
    ) -> tuple[bool, Decimal, Decimal]:
        """Calculate regime synchronously given daily closes and optional current price.

        Applies a 2% hysteresis buffer against the 50-day SMA:
        - To switch from UPTREND to DOWNTREND, price must be <= SMA * (1 - 0.02).
        - To switch from DOWNTREND to UPTREND, price must be >= SMA * (1 + 0.02).
        - Otherwise, retains the previous regime state.

        Returns (is_uptrend, ref_price, sma_50).
        """
        if not daily_closes:
            price = Decimal(str(current_price)) if current_price is not None else Decimal("0")
            return True, price, Decimal("0")

        decimal_closes = [Decimal(str(p)) for p in daily_closes]
        sample = (
            decimal_closes[-self.sma_period :]
            if len(decimal_closes) >= self.sma_period
            else decimal_closes
        )
        sma_50 = sum(sample) / Decimal(str(len(sample)))

        if current_price is not None:
            ref_price = Decimal(str(current_price))
        else:
            ref_price = decimal_closes[-1]

        upper_threshold = sma_50 * (Decimal("1") + self.hysteresis_pct)
        lower_threshold = sma_50 * (Decimal("1") - self.hysteresis_pct)

        if self._current_regime is None:
            if current_price is not None:
                self.replay_state_machine(decimal_closes, window=100)
            elif len(decimal_closes) > 1:
                self.replay_state_machine(decimal_closes[:-1], window=100)

        if self._current_regime is None:
            if ref_price >= sma_50:
                self._current_regime = "UPTREND"
            else:
                self._current_regime = "DOWNTREND"
        else:
            if self._current_regime == "UPTREND":
                if ref_price <= lower_threshold:
                    self._current_regime = "DOWNTREND"
            else:
                if ref_price >= upper_threshold:
                    self._current_regime = "UPTREND"

        is_uptrend = self._current_regime == "UPTREND"
        return is_uptrend, ref_price, sma_50

    async def is_uptrend(
        self,
        current_price: Decimal | float | str | None = None,
        now: datetime | None = None,
    ) -> bool:
        """Return True if BTC is in an uptrend, False if in a downtrend."""
        closes = await self.fetch_btc_daily_closes()
        is_uptrend, ref_price, sma_50 = self.calculate_regime_sync(closes, current_price)
        logger.debug(
            "BTC Regime: price=%s, sma_%d=%s, regime=%s, is_uptrend=%s",
            ref_price,
            self.sma_period,
            sma_50,
            self._current_regime,
            is_uptrend,
        )
        return is_uptrend

    async def get_regime_status(
        self,
        current_price: Decimal | float | str | None = None,
    ) -> dict[str, Any]:
        """Return structured regime status details."""
        closes = await self.fetch_btc_daily_closes()
        is_uptrend, ref_price, sma_50 = self.calculate_regime_sync(closes, current_price)
        upper_threshold = sma_50 * (Decimal("1") + self.hysteresis_pct)
        lower_threshold = sma_50 * (Decimal("1") - self.hysteresis_pct)
        return {
            "symbol": self.symbol,
            "is_uptrend": is_uptrend,
            "regime": self._current_regime or ("UPTREND" if is_uptrend else "DOWNTREND"),
            "current_price": float(ref_price),
            "sma_50": float(sma_50),
            "sma_period": self.sma_period,
            "upper_threshold": float(upper_threshold),
            "lower_threshold": float(lower_threshold),
            "hysteresis_pct": float(self.hysteresis_pct),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


__all__ = [
    "BINANCE_KLINES_URL",
    "BtcRegimeFilter",
    "DEFAULT_BTC_SYMBOL",
    "DEFAULT_HYSTERESIS_PCT",
    "DEFAULT_SMA_PERIOD",
]

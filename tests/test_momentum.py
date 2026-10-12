from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock

import httpx
import pytest

from app.market.models import Candle
from app.trading.momentum import BtcRegimeFilter


def make_candle(close: str | float, open_time: datetime | None = None) -> Candle:
    return Candle(
        exchange="binance_spot",
        symbol="BTCUSDT",
        interval="1h",
        open_time=open_time or datetime(2024, 1, 1, tzinfo=timezone.utc),
        open=Decimal(str(close)),
        high=Decimal(str(close)),
        low=Decimal(str(close)),
        close=Decimal(str(close)),
        volume=Decimal("100"),
    )


@pytest.mark.asyncio
async def test_btc_regime_filter_calculates_50_day_sma():
    filter_ = BtcRegimeFilter(sma_period=50)

    # 50 closes at 100
    closes = [Decimal("100")] * 50
    is_up, price, sma = filter_.calculate_regime_sync(closes, current_price=Decimal("105"))
    assert is_up is True
    assert price == Decimal("105")
    assert sma == Decimal("100")

    # Price below SMA
    is_up_down, price_down, _ = filter_.calculate_regime_sync(closes, current_price=Decimal("95"))
    assert is_up_down is False
    assert price_down == Decimal("95")


@pytest.mark.asyncio
async def test_btc_regime_filter_with_candles_provider():
    # 50 candles with ascending closes from 1 to 50
    candles = [make_candle(close=float(i)) for i in range(1, 51)]
    # Average of 1..50 = 25.5

    filter_ = BtcRegimeFilter(sma_period=50, candles_provider=lambda: candles)

    # Price 30 is above SMA (25.5) -> Uptrend
    assert await filter_.is_uptrend(current_price=Decimal("30")) is True

    # Price 20 is below SMA (25.5) -> Downtrend
    assert await filter_.is_uptrend(current_price=Decimal("20")) is False

    status = await filter_.get_regime_status(current_price=Decimal("20"))
    assert status["regime"] == "DOWNTREND"
    assert status["is_uptrend"] is False
    assert status["sma_50"] == 25.5


@pytest.mark.asyncio
async def test_btc_regime_filter_fetch_via_http_client():
    mock_http = AsyncMock()
    # 55 daily klines from Binance API format
    # row[4] is close price
    kline_rows = [[0, "100", "110", "90", str(50000 + i * 100), "1000"] for i in range(55)]
    mock_http.get = AsyncMock(return_value=httpx.Response(status_code=200, json=kline_rows))

    filter_ = BtcRegimeFilter(sma_period=50, http_client=mock_http)
    closes = await filter_.fetch_btc_daily_closes()
    assert len(closes) == 55
    assert closes[0] == Decimal("50000")
    assert closes[-1] == Decimal("55400")

    # Current price above SMA
    assert await filter_.is_uptrend(current_price=Decimal("60000")) is True
    # Current price below SMA
    assert await filter_.is_uptrend(current_price=Decimal("40000")) is False


@pytest.mark.asyncio
async def test_btc_regime_filter_hysteresis_buffer():
    filter_ = BtcRegimeFilter(sma_period=50, hysteresis_pct=Decimal("0.02"))
    closes = [Decimal("100")] * 50  # SMA is 100.00

    # 1. Start in UPTREND with price 105
    is_up, price, sma = filter_.calculate_regime_sync(closes, current_price=Decimal("105"))
    assert is_up is True
    assert filter_._current_regime == "UPTREND"

    # 2. Price dips to 99 (1% below SMA, inside 2% buffer): MUST REMAIN UPTREND
    is_up, _, _ = filter_.calculate_regime_sync(closes, current_price=Decimal("99"))
    assert is_up is True
    assert filter_._current_regime == "UPTREND"

    # 3. Price drops to 97.5 (2.5% below SMA, <= 98 threshold): TRANSITIONS TO DOWNTREND
    is_up, _, _ = filter_.calculate_regime_sync(closes, current_price=Decimal("97.5"))
    assert is_up is False
    assert filter_._current_regime == "DOWNTREND"

    # 4. Price rebounds to 101 (1% above SMA, inside 2% buffer): MUST REMAIN DOWNTREND
    is_up, _, _ = filter_.calculate_regime_sync(closes, current_price=Decimal("101"))
    assert is_up is False
    assert filter_._current_regime == "DOWNTREND"

    # 5. Price crosses 102.5 (2.5% above SMA, >= 102 threshold): TRANSITIONS TO UPTREND
    is_up, _, _ = filter_.calculate_regime_sync(closes, current_price=Decimal("102.5"))
    assert is_up is True
    assert filter_._current_regime == "UPTREND"


@pytest.mark.asyncio
async def test_btc_regime_filter_drops_unclosed_candle():
    mock_http = AsyncMock()
    # 50 closed daily candles (timestamp in past) + 1 forming candle (close_time far in future)
    past_close_ms = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    future_close_ms = int(datetime(2099, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)

    rows = []
    for i in range(50):
        # 12-element kline format: row[4] = close, row[6] = close_time
        rows.append(
            [
                0,
                "100",
                "110",
                "90",
                str(50000 + i * 10),
                "100",
                past_close_ms,
                "100",
                10,
                "50",
                "50",
                "0",
            ]
        )

    # Open / forming candle
    rows.append(
        [0, "100", "110", "90", "99999", "100", future_close_ms, "100", 10, "50", "50", "0"]
    )

    mock_http.get = AsyncMock(return_value=httpx.Response(status_code=200, json=rows))

    filter_ = BtcRegimeFilter(sma_period=50, http_client=mock_http)
    closes = await filter_.fetch_btc_daily_closes()

    # The 51st candle with close 99999 was in the future and must be discarded
    assert len(closes) == 50
    assert closes[-1] == Decimal("50490")
    assert Decimal("99999") not in closes


@pytest.mark.asyncio
async def test_btc_regime_filter_replays_100_candles_startup():
    # Construct 100 historical completed daily candles:
    # First 50 days at 100.00
    # Day 51 drops to 97.00 (triggers DOWNTREND <= 98.00 threshold)
    # Day 52..100 price rebounds to 101.00 (SMA is ~100.92, price 101 > SMA, but < 102.94 threshold)
    # Without replay: guessing based purely on latest candle (101 > 100.92) would wrongly guess UPTREND!
    # With replay over the past 100 candles: it correctly retains the DOWNTREND state!
    closes = [Decimal("100")] * 50 + [Decimal("97")] + [Decimal("101")] * 49
    assert len(closes) == 100

    filter_ = BtcRegimeFilter(sma_period=50, hysteresis_pct=Decimal("0.02"))
    # Replay past 100 candles
    regime = filter_.replay_state_machine(closes, window=100)
    assert regime == "DOWNTREND"
    assert filter_._current_regime == "DOWNTREND"

    # Now verify async initialize() also queries and replayed state
    candles = [make_candle(close=float(c)) for c in closes]
    filter_init = BtcRegimeFilter(
        sma_period=50,
        hysteresis_pct=Decimal("0.02"),
        candles_provider=lambda: candles,
    )
    init_regime = await filter_init.initialize()
    assert init_regime == "DOWNTREND"
    assert await filter_init.is_uptrend() is False

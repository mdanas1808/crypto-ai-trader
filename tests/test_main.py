import asyncio
import logging
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from app.backtest.models import BacktestConfig
from app.main import app, build_app, main
from app.market.models import Candle, CandleOpenObservation
from app.trading.paper import PaperExecutor
from app.trading.session import PaperSession

UTC = timezone.utc
START = datetime(2024, 1, 2, tzinfo=UTC)
D = Decimal


class MockStream:
    """Mock stream that sleeps until cancelled."""

    async def events(self) -> AsyncIterator[Candle | CandleOpenObservation]:
        while True:
            await asyncio.sleep(0.01)
            if False:
                yield None  # type: ignore


class QuickExitServer:
    """Mock server that immediately exits."""

    def __init__(self) -> None:
        self.should_exit = False
        self.serve_called = False

    async def serve(self) -> None:
        self.serve_called = True
        await asyncio.sleep(0.02)
        self.should_exit = True


class InfiniteServer:
    """Mock server that runs until should_exit is set or cancelled."""

    def __init__(self) -> None:
        self.should_exit = False
        self.serve_called = False

    async def serve(self) -> None:
        self.serve_called = True
        while not self.should_exit:
            await asyncio.sleep(0.01)


def test_app_global_export_health():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_build_app_contains_all_routes():
    executor = PaperExecutor(BacktestConfig(minimum_quantity=D("0.000001")))
    session = PaperSession(stream=MockStream(), executor=executor)
    test_app = build_app(session)
    client = TestClient(test_app)

    # Market routes
    assert client.get("/health").status_code == 200
    # Dashboard routes
    assert client.get("/").status_code == 200
    assert client.get("/ui").status_code == 200
    assert client.get("/portfolio").status_code == 200
    assert client.get("/positions").status_code == 200
    assert client.get("/trades").status_code == 200
    assert client.get("/signals").status_code == 200


@pytest.mark.asyncio
async def test_main_runs_and_shuts_down_when_server_exits():
    executor = PaperExecutor(BacktestConfig(minimum_quantity=D("0.000001")))
    session = PaperSession(stream=MockStream(), executor=executor)
    server = QuickExitServer()

    await main(
        session=session,
        server=server,  # type: ignore
    )

    assert server.serve_called
    assert not session.is_running


@pytest.mark.asyncio
async def test_main_shuts_down_on_task_cancellation():
    executor = PaperExecutor(BacktestConfig(minimum_quantity=D("0.000001")))
    session = PaperSession(stream=MockStream(), executor=executor)
    server = InfiniteServer()

    main_task = asyncio.create_task(
        main(session=session, server=server)  # type: ignore
    )
    await asyncio.sleep(0.03)

    assert session.is_running
    assert server.serve_called

    # Cancel the main task (simulating SIGINT or container stop)
    main_task.cancel()
    await main_task

    assert not session.is_running
    assert server.should_exit is True


@pytest.mark.asyncio
async def test_main_initializes_components_when_session_none():
    server = QuickExitServer()
    mock_stream = MockStream()

    await main(
        stream=mock_stream,
        server=server,  # type: ignore
    )

    assert server.serve_called


class CrashingServer:
    """Mock server that crashes during serve."""

    def __init__(self) -> None:
        self.should_exit = False
        self.serve_called = False

    async def serve(self) -> None:
        self.serve_called = True
        await asyncio.sleep(0.01)
        raise RuntimeError("Uvicorn failed to bind address")


@pytest.mark.asyncio
async def test_main_logs_crashed_task_exception(caplog):
    executor = PaperExecutor(BacktestConfig(minimum_quantity=D("0.000001")))
    session = PaperSession(stream=MockStream(), executor=executor)
    server = CrashingServer()

    with caplog.at_level(logging.ERROR):
        await main(
            session=session,
            server=server,  # type: ignore
        )

    assert "crashed with exception" in caplog.text
    assert "Uvicorn failed to bind address" in caplog.text
    assert not session.is_running


@pytest.mark.asyncio
async def test_fetch_binance_tradable_usdt_symbols_filters_correctly():
    from unittest.mock import AsyncMock, MagicMock

    from app.main import fetch_binance_tradable_usdt_symbols

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "symbols": [
            {"symbol": "BTCUSDT", "status": "TRADING", "quoteAsset": "USDT"},
            {"symbol": "ETHUSDT", "status": "TRADING", "quoteAsset": "USDT"},
            {"symbol": "SOLUSDT", "status": "BREAK", "quoteAsset": "USDT"},  # not TRADING
            {"symbol": "BNBBTC", "status": "TRADING", "quoteAsset": "BTC"},  # quote not USDT
            {"symbol": "ADABUSD", "status": "TRADING", "quoteAsset": "BUSD"},  # quote not USDT
            {"symbol": "DOGEUSDT", "status": "HALT", "quoteAsset": "USDT"},  # not TRADING
            "invalid_entry",  # malformed
            {"symbol": 123, "status": "TRADING", "quoteAsset": "USDT"},  # malformed symbol
        ]
    }
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)

    symbols = await fetch_binance_tradable_usdt_symbols(async_client=mock_client)
    assert symbols == {"BTCUSDT", "ETHUSDT"}


@pytest.mark.asyncio
async def test_fetch_binance_tradable_usdt_symbols_handles_errors(caplog):
    from unittest.mock import AsyncMock, MagicMock

    import httpx

    from app.main import fetch_binance_tradable_usdt_symbols

    # Test HTTP error status
    mock_resp = MagicMock()
    mock_resp.status_code = 451
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)

    with caplog.at_level(logging.WARNING):
        symbols = await fetch_binance_tradable_usdt_symbols(async_client=mock_client)
    assert symbols == set()
    assert "HTTP 451" in caplog.text

    # Test exception
    mock_client.get = AsyncMock(side_effect=httpx.ConnectTimeout("timeout"))
    with caplog.at_level(logging.WARNING):
        symbols = await fetch_binance_tradable_usdt_symbols(async_client=mock_client)
    assert symbols == set()
    assert "Failed to fetch Binance exchange info" in caplog.text


@pytest.mark.asyncio
async def test_fetch_dynamic_trending_symbols_parses_and_filters_fixed(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    import httpx

    from app.main import FIXED_SYMBOLS, fetch_dynamic_trending_symbols

    assert FIXED_SYMBOLS == ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ZECUSDT"]

    async def mock_get(url, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        if "binance" in url:
            resp.json.return_value = {
                "symbols": [
                    {"symbol": "BTCUSDT", "status": "TRADING", "quoteAsset": "USDT"},
                    {"symbol": "TAOUSDT", "status": "TRADING", "quoteAsset": "USDT"},
                    {"symbol": "FETUSDT", "status": "TRADING", "quoteAsset": "USDT"},
                    {"symbol": "RENDERUSDT", "status": "TRADING", "quoteAsset": "USDT"},
                ]
            }
        else:
            resp.json.return_value = {
                "coins": [
                    {"item": {"symbol": "btc"}},  # in FIXED_SYMBOLS -> filtered out
                    {"item": {"symbol": "tao"}},
                    {"item": {"symbol": "fet"}},
                    {"item": {"symbol": "render"}},
                ]
            }
        return resp

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=mock_get)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: mock_client)

    symbols = await fetch_dynamic_trending_symbols(limit=2)
    assert symbols == ["TAOUSDT", "FETUSDT"]


@pytest.mark.asyncio
async def test_fetch_dynamic_trending_symbols_skips_unlisted_and_logs_warning(monkeypatch, caplog):
    from unittest.mock import AsyncMock, MagicMock

    import httpx

    from app.main import fetch_dynamic_trending_symbols

    async def mock_get(url, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        if "binance" in url:
            resp.json.return_value = {
                "symbols": [
                    {"symbol": "NEARUSDT", "status": "TRADING", "quoteAsset": "USDT"},
                    {"symbol": "FETUSDT", "status": "TRADING", "quoteAsset": "USDT"},
                    {"symbol": "RENDERUSDT", "status": "TRADING", "quoteAsset": "USDT"},
                    {"symbol": "INJUSDT", "status": "TRADING", "quoteAsset": "USDT"},
                    {"symbol": "TAOUSDT", "status": "TRADING", "quoteAsset": "USDT"},
                ]
            }
        else:
            resp.json.return_value = {
                "coins": [
                    {"item": {"symbol": "wisp"}},  # Unlisted meme -> skipped with warning
                    {"item": {"symbol": "near"}},  # Listed -> kept (1)
                    {"item": {"symbol": "zylo"}},  # Unlisted meme -> skipped with warning
                    {"item": {"symbol": "fet"}},  # Listed -> kept (2)
                    {"item": {"symbol": "crawler"}},  # Unlisted meme -> skipped with warning
                    {"item": {"symbol": "render"}},  # Listed -> kept (3)
                    {"item": {"symbol": "inj"}},  # Listed -> kept (4)
                    {"item": {"symbol": "tao"}},  # Listed -> kept (5)
                    {"item": {"symbol": "extra"}},  # Shouldn't be needed once 5 are found
                ]
            }
        return resp

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=mock_get)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: mock_client)

    with caplog.at_level(logging.WARNING):
        symbols = await fetch_dynamic_trending_symbols(limit=5)

    assert symbols == ["NEARUSDT", "FETUSDT", "RENDERUSDT", "INJUSDT", "TAOUSDT"]
    assert "Coin 'WISP' (WISPUSDT) skipped: not listed or tradable on Binance" in caplog.text
    assert "Coin 'ZYLO' (ZYLOUSDT) skipped: not listed or tradable on Binance" in caplog.text
    assert "Coin 'CRAWLER' (CRAWLERUSDT) skipped: not listed or tradable on Binance" in caplog.text


@pytest.mark.asyncio
async def test_fetch_dynamic_trending_symbols_fallback_when_not_enough_binance_listed(
    monkeypatch, caplog
):
    from unittest.mock import AsyncMock, MagicMock

    import httpx

    from app.main import FALLBACK_DYNAMIC_SYMBOLS, fetch_dynamic_trending_symbols

    async def mock_get(url, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        if "binance" in url:
            resp.json.return_value = {
                "symbols": [
                    {"symbol": "NEARUSDT", "status": "TRADING", "quoteAsset": "USDT"},
                    {"symbol": "FETUSDT", "status": "TRADING", "quoteAsset": "USDT"},
                ]
            }
        else:
            resp.json.return_value = {
                "coins": [
                    {"item": {"symbol": "wisp"}},  # Unlisted
                    {"item": {"symbol": "near"}},  # Listed (1)
                    {"item": {"symbol": "zylo"}},  # Unlisted
                    {"item": {"symbol": "fet"}},  # Listed (2)
                    {"item": {"symbol": "crawler"}},  # Unlisted
                ]
            }
        return resp

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=mock_get)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: mock_client)

    with caplog.at_level(logging.WARNING):
        symbols = await fetch_dynamic_trending_symbols(limit=5)

    # Only 2 found when 5 needed -> falls back to FALLBACK_DYNAMIC_SYMBOLS
    assert symbols == FALLBACK_DYNAMIC_SYMBOLS
    assert "needed 5" in caplog.text


@pytest.mark.asyncio
async def test_fetch_dynamic_trending_symbols_fallback_on_binance_failure(monkeypatch, caplog):
    from unittest.mock import AsyncMock, MagicMock

    import httpx

    from app.main import FALLBACK_DYNAMIC_SYMBOLS, fetch_dynamic_trending_symbols

    async def mock_get(url, **kwargs):
        resp = MagicMock()
        if "binance" in url:
            resp.status_code = 503
            resp.json.return_value = {"msg": "service unavailable"}
        else:
            resp.status_code = 200
            resp.json.return_value = {"coins": [{"item": {"symbol": "near"}}]}
        return resp

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=mock_get)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: mock_client)

    with caplog.at_level(logging.WARNING):
        symbols = await fetch_dynamic_trending_symbols(limit=5)

    assert symbols == FALLBACK_DYNAMIC_SYMBOLS
    assert "No valid Binance USDT symbols available" in caplog.text


@pytest.mark.asyncio
async def test_fetch_dynamic_trending_symbols_fallback_on_network_error(monkeypatch):
    from unittest.mock import AsyncMock

    import httpx

    from app.main import FALLBACK_DYNAMIC_SYMBOLS, fetch_dynamic_trending_symbols

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=httpx.RequestError("network down"))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: mock_client)

    symbols = await fetch_dynamic_trending_symbols(limit=5)
    assert symbols == FALLBACK_DYNAMIC_SYMBOLS

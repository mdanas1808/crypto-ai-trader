"""Application entry point for live paper trading and dashboard API.

Initializes database models, Binance market WebSocket ingestion, PaperExecutor,
AI trade explainer, alert handlers, and runs the FastAPI dashboard alongside
the PaperSession orchestrator in an asyncio event loop.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable, Sequence
from datetime import timedelta
from decimal import Decimal
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI

from app.analysis.eval_logger import FundamentalEvalLogger
from app.analysis.explainer import TradeExplainer
from app.api.dashboard import create_dashboard_router, set_dashboard_session
from app.api.routes import router as market_router
from app.backtest.data import MultiTimeframeBacktestData, prepare_multi_timeframe_backtest_data
from app.backtest.models import BacktestConfig
from app.database import models as _market_models  # noqa: F401
from app.database import paper_models as _paper_models  # noqa: F401
from app.database.database import Base, SessionLocal, engine
from app.market.constants import SUPPORTED_SYMBOLS
from app.market.exchange import BinanceSpotClient
from app.market.models import Candle
from app.market.repository import persist_candles
from app.market.websocket import BinanceSpotKlineStream
from app.trading.alerts import LoggingAlertHandler
from app.trading.momentum import BtcRegimeFilter
from app.trading.paper import PaperExecutor
from app.trading.persistence import DatabaseAlertHandler
from app.trading.session import PaperSession

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("app.main")

BINANCE_EXCHANGE_INFO_URL = "https://api.binance.com/api/v3/exchangeInfo"
COINGECKO_TRENDING_URL = "https://api.coingecko.com/api/v3/search/trending"
FIXED_SYMBOLS: list[str] = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ZECUSDT"]
FALLBACK_DYNAMIC_SYMBOLS: list[str] = [
    "NEARUSDT",
    "FETUSDT",
    "RENDERUSDT",
    "INJUSDT",
    "TAOUSDT",
]
ALL_SYMBOLS: list[str] = list(dict.fromkeys(FIXED_SYMBOLS + FALLBACK_DYNAMIC_SYMBOLS))


async def fetch_binance_tradable_usdt_symbols(
    *,
    async_client: httpx.AsyncClient | None = None,
) -> set[str]:
    """Fetch all tradable USDT symbol strings from Binance exchangeInfo."""
    try:
        if async_client is not None:
            response = await async_client.get(BINANCE_EXCHANGE_INFO_URL, timeout=5.0)
        else:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(BINANCE_EXCHANGE_INFO_URL)

        if response.status_code != 200:
            logger.warning(
                "Binance exchangeInfo API returned HTTP %s; cannot determine tradable symbols",
                response.status_code,
            )
            return set()

        payload = response.json()
        if not isinstance(payload, dict):
            logger.warning("Binance exchangeInfo API returned non-dict payload")
            return set()

        symbols_data = payload.get("symbols")
        if not isinstance(symbols_data, list):
            logger.warning("Binance exchangeInfo symbols field is missing or not a list")
            return set()

        valid_symbols: set[str] = {
            item["symbol"]
            for item in symbols_data
            if isinstance(item, dict)
            and item.get("status") == "TRADING"
            and item.get("quoteAsset") == "USDT"
            and isinstance(item.get("symbol"), str)
        }
        return valid_symbols
    except Exception as exc:
        logger.warning(
            "Failed to fetch Binance exchange info (%s): %s",
            type(exc).__name__,
            exc,
        )
        return set()


async def fetch_dynamic_trending_symbols(
    limit: int = 5,
    *,
    async_client: httpx.AsyncClient | None = None,
    binance_symbols: set[str] | None = None,
) -> list[str]:
    """Fetch trending coin symbols from CoinGecko, cross-referenced with valid Binance USDT listings."""
    if limit <= 0:
        return []

    fallback = [s for s in FALLBACK_DYNAMIC_SYMBOLS if s not in FIXED_SYMBOLS][:limit]

    async def _execute_fetch(client: httpx.AsyncClient) -> list[str]:
        # 1. Fetch valid tradable USDT symbols from Binance
        valid_binance = (
            binance_symbols
            if binance_symbols is not None
            else await fetch_binance_tradable_usdt_symbols(async_client=client)
        )
        if not valid_binance:
            logger.warning(
                "No valid Binance USDT symbols available; using fallback symbols: %s",
                fallback,
            )
            SUPPORTED_SYMBOLS.update(fallback)
            return fallback

        # 2. Fetch CoinGecko trending list
        cg_response = await client.get(COINGECKO_TRENDING_URL, timeout=5.0)
        if cg_response.status_code != 200:
            logger.warning(
                "CoinGecko trending API returned HTTP %s; using fallback symbols",
                cg_response.status_code,
            )
            SUPPORTED_SYMBOLS.update(fallback)
            return fallback

        payload = cg_response.json()
        if not isinstance(payload, dict):
            logger.warning("CoinGecko trending API returned non-dict payload; using fallback")
            SUPPORTED_SYMBOLS.update(fallback)
            return fallback

        coins = payload.get("coins")
        if not isinstance(coins, list) or not coins:
            logger.warning("CoinGecko trending API returned no coins; using fallback")
            SUPPORTED_SYMBOLS.update(fallback)
            return fallback

        discovered: list[str] = []
        for entry in coins:
            if not isinstance(entry, dict):
                continue
            item = entry.get("item", entry)
            if not isinstance(item, dict):
                continue
            raw_symbol = item.get("symbol")
            if not isinstance(raw_symbol, str) or not raw_symbol.strip():
                continue
            clean_ticker = "".join(ch for ch in raw_symbol.strip().upper() if ch.isalnum())
            if not clean_ticker or clean_ticker in ("USDT", "USDC", "DAI", "FDUSD"):
                continue

            pair = clean_ticker if clean_ticker.endswith("USDT") else f"{clean_ticker}USDT"

            if pair in FIXED_SYMBOLS or pair in discovered:
                continue

            if pair not in valid_binance:
                logger.warning(
                    "Coin '%s' (%s) skipped: not listed or tradable on Binance",
                    clean_ticker,
                    pair,
                )
                continue

            discovered.append(pair)
            if len(discovered) >= limit:
                break

        if len(discovered) < limit:
            logger.warning(
                "CoinGecko trending yielded %d Binance-listed pairs (needed %d); using fallback symbols: %s",
                len(discovered),
                limit,
                fallback,
            )
            SUPPORTED_SYMBOLS.update(fallback)
            return fallback

        SUPPORTED_SYMBOLS.update(discovered)
        return discovered

    try:
        if async_client is not None:
            return await _execute_fetch(async_client)
        else:
            async with httpx.AsyncClient(timeout=5.0) as client:
                return await _execute_fetch(client)
    except Exception as exc:
        logger.warning(
            "Failed to fetch dynamic trending symbols (%s): %s; using fallback",
            type(exc).__name__,
            exc,
        )
        SUPPORTED_SYMBOLS.update(fallback)
        return fallback


def build_app(session: Any | None = None) -> FastAPI:
    """Build the unified FastAPI application with market and dashboard routes."""
    if session is not None:
        set_dashboard_session(session)
    application = FastAPI(title="Crypto AI Trader")
    application.include_router(market_router)
    dashboard_router = create_dashboard_router(session)
    application.include_router(dashboard_router)
    return application


# Global FastAPI application instance for ASGI servers
app = build_app()


async def warmup_historical_candles(
    symbols: Sequence[str],
    limit: int = 500,
    *,
    client: BinanceSpotClient | None = None,
) -> int:
    """Warm up historical 1h and 4h candles into PostgreSQL on startup."""
    logger.info("Starting historical candle warmup for %d symbols...", len(symbols))
    total_inserted = 0

    async def _fetch_and_persist(sym: str, cl: BinanceSpotClient) -> int:
        try:
            c1, c4 = await asyncio.gather(
                cl.fetch_candles(sym, "1h", limit=limit),
                cl.fetch_candles(sym, "4h", limit=limit),
            )
            with SessionLocal() as s:
                r1 = persist_candles(s, c1)
                r4 = persist_candles(s, c4)
                return r1.inserted + r4.inserted
        except Exception as exc:
            logger.warning("Historical warmup failed for %s: %s", sym, exc)
            return 0

    try:
        if client is not None:
            results = await asyncio.gather(*(_fetch_and_persist(s, client) for s in symbols))
        else:
            async with BinanceSpotClient() as exchange_client:
                results = await asyncio.gather(
                    *(_fetch_and_persist(s, exchange_client) for s in symbols)
                )
        total_inserted = sum(results)
        logger.info("Historical candle warmup complete: %d new rows persisted.", total_inserted)
    except Exception as exc:
        logger.warning("Historical candle warmup encountered error: %s", exc)

    return total_inserted


def create_live_data_provider(
    warmup_hours: int = 850,
) -> Callable[[Candle], MultiTimeframeBacktestData | None]:
    """Create data provider returning MultiTimeframeBacktestData for breakout strategy."""

    def live_data_provider(candle: Candle) -> MultiTimeframeBacktestData | None:
        try:
            with SessionLocal() as db_session:
                return prepare_multi_timeframe_backtest_data(
                    db_session,
                    candle.symbol,
                    start_time=candle.open_time,
                    end_time=candle.open_time,
                    warmup_start_time=candle.open_time - timedelta(hours=warmup_hours),
                )
        except Exception as exc:
            logger.warning(
                "Failed to prepare multi-timeframe backtest data for %s: %s",
                candle.symbol,
                exc,
            )
            return None

    return live_data_provider


async def main(
    *,
    host: str = "0.0.0.0",
    port: int = 8000,
    symbols: tuple[str, ...] | list[str] | None = None,
    stream: Any | None = None,
    session: PaperSession | None = None,
    server: uvicorn.Server | None = None,
    warmup_candles: bool | None = None,
) -> None:
    """Initialize all subsystems and run paper session and API concurrently."""
    global ALL_SYMBOLS
    logger.info("Initializing Crypto AI Trader system...")

    # 1. Ensure database tables exist
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database schemas verified.")
    except Exception as exc:
        logger.warning("Database schema verification warning: %s", exc)

    # 2. Initialize Paper Trading components if not injected
    if session is None:
        if symbols is not None:
            resolved_symbols = list(dict.fromkeys(symbols))
            SUPPORTED_SYMBOLS.update(resolved_symbols)
        else:
            dynamic_symbols = await fetch_dynamic_trending_symbols()
            ALL_SYMBOLS = list(dict.fromkeys(FIXED_SYMBOLS + dynamic_symbols))
            SUPPORTED_SYMBOLS.update(ALL_SYMBOLS)
            resolved_symbols = ALL_SYMBOLS

        should_warmup = warmup_candles if warmup_candles is not None else (stream is None)
        if should_warmup:
            try:
                await warmup_historical_candles(resolved_symbols)
            except Exception as exc:
                logger.warning("Historical warmup skipped or failed: %s", exc)

        ws_url = os.getenv("BINANCE_WS_URL")
        kline_stream = stream or BinanceSpotKlineStream(
            symbols=resolved_symbols,
            url=ws_url,
        )
        config = BacktestConfig(
            minimum_quantity=Decimal("0.000001"),
            quantity_step=Decimal("0.000001"),
        )
        executor = PaperExecutor(
            config,
            max_open_positions=1,
            starting_cash=Decimal("100.00"),
        )
        explainer = TradeExplainer()
        logging_handler = LoggingAlertHandler()
        db_handler = DatabaseAlertHandler()

        eval_logger = FundamentalEvalLogger("fundamental_eval.db")
        btc_regime_filter = BtcRegimeFilter()
        try:
            await btc_regime_filter.initialize()
        except Exception as exc:
            logger.warning("BTC regime filter initialization warning: %s", exc)

        data_provider = create_live_data_provider()

        session = PaperSession(
            stream=kline_stream,
            executor=executor,
            alert_handler=[logging_handler, db_handler],
            explainer=explainer,
            eval_logger=eval_logger,
            btc_regime_filter=btc_regime_filter,
            data_provider=data_provider,
            max_open_positions=1,
            tracked_symbols=resolved_symbols,
        )

    set_dashboard_session(session)

    # 3. Build FastAPI application wired to the session
    api_app = build_app(session)

    # 4. Prepare Uvicorn server if not injected
    if server is None:
        server_config = uvicorn.Config(
            app=api_app,
            host=host,
            port=port,
            log_level="info",
        )
        server = uvicorn.Server(server_config)

    # 5. Run PaperSession and FastAPI concurrently with graceful shutdown
    session_task = asyncio.create_task(session.run(), name="paper_session")
    server_task = asyncio.create_task(server.serve(), name="uvicorn_server")

    logger.info("Active trading symbols: %s", ALL_SYMBOLS)
    logger.info("Starting PaperSession and Dashboard API on http://%s:%d", host, port)

    try:
        done, pending = await asyncio.wait(
            [session_task, server_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for completed_task in done:
            task_name = completed_task.get_name()
            if completed_task.cancelled():
                logger.info("Task %s was cancelled.", task_name)
            elif completed_task.exception() is not None:
                exc = completed_task.exception()
                logger.error(
                    "Task %s crashed with exception: %s",
                    task_name,
                    exc,
                    exc_info=exc,
                )
            else:
                logger.info("Task %s exited normally.", task_name)
    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("Termination signal received.")
    finally:
        logger.info("Shutting down paper session and API server...")
        session.stop()
        server.should_exit = True

        for task in (session_task, server_task):
            if not task.done():
                task.cancel()

        await asyncio.gather(session_task, server_task, return_exceptions=True)
        logger.info("All services successfully stopped.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass

# Crypto AI Trader

A staged foundation for crypto market analysis and paper trading. Phase 1 provides FastAPI, environment-based settings, SQLAlchemy, Alembic, and PostgreSQL through Docker Compose. Phase 2A adds explicitly triggered historical candle ingestion and a read-only database API.

## Scope and safety

The system is initially paper-trading only. No real-money trading is enabled. Exchange authentication, WebSocket ingestion, trading, strategy logic, and AI analysis are not implemented. Technical indicators are calculated in memory from validated candle sequences. Historical market data uses the unauthenticated Binance Spot public REST API only.

## Market data

The client uses Binance's documented public market-data-only base URL `https://data-api.binance.vision` and `GET /api/v3/klines`. It requests only closed candles for the currently supported symbols `BTCUSDT`, `ETHUSDT`, and `SOLUSDT` at `1h` or `4h`. Extend the allowlists in `app/market/constants.py` when new symbols or intervals are approved.

Exchange decimal strings are normalized to Python `Decimal` and persisted in PostgreSQL `NUMERIC(38, 18)`. This stores values as exact base-10 decimals, with up to 20 integer digits and 18 fractional digits. Candle open times are stored as timezone-aware UTC timestamps. The composite primary key `(exchange, symbol, interval, open_time)` prevents duplicates and provides the index used by the candle lookups.

Before indicator calculation, candle sequences are validated for UTC interval alignment, strict chronological order, and missing intervals. A duplicate, out-of-order, misaligned, or gapped sequence raises a validation error that identifies the candle series and, for gaps, the missing timestamps and count. Data is never sorted, interpolated, or synthesized; automatic gap backfilling is not implemented.

Run the bounded ingestion command explicitly; it does not run at API startup:

```bash
.venv/bin/python scripts/fetch_candles.py --symbol BTCUSDT --interval 1h --limit 10
```

The read-only endpoint queries PostgreSQL only and never calls the exchange:

```text
GET /market/candles?symbol=BTCUSDT&interval=1h&limit=10
```

## Local development

Python 3.13 is used on the development VM.

```bash
cd ~/crypto-ai-trader
source .venv/bin/activate
```

The checked-in `.env.example` contains placeholders. A local `.env` supplies development settings; never commit `.env` or put real credentials in it.

## PostgreSQL

Start the PostgreSQL 16 container (published only on localhost):

```bash
docker compose up -d
.venv/bin/alembic upgrade head
```

Check its status with `docker compose ps`. Stop PostgreSQL with `docker compose down`. The named volume preserves database data when the container is stopped.

## FastAPI

Start the API locally:

```bash
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Health endpoint: `GET http://127.0.0.1:8000/health`

```bash
curl http://127.0.0.1:8000/health
curl 'http://127.0.0.1:8000/market/candles?symbol=BTCUSDT&interval=1h&limit=10'
```

Run tests with `.venv/bin/pytest -v`. The repository and API database tests use a temporary PostgreSQL schema and do not call the exchange.

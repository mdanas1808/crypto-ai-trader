# GEMINI.md - Crypto AI Trader Instruction & Context Guide

## 1. Project Overview

`crypto-ai-trader` is a robust, modular Python platform for crypto market data ingestion, causal technical analysis, deterministic multi-symbol backtesting, and simulated paper execution.

### Scope and Safety Mandates
- **Offline & Simulation Only:** The system is an offline market analysis, backtesting, and paper trading platform. **No real-money trading or authenticated exchange order execution is enabled.**
- **Public Data Only:** Market data ingestion relies strictly on unauthenticated Binance Spot public endpoints:
  - REST: `https://data-api.binance.vision/api/v3/klines`
  - WebSocket: `wss://stream.binance.com:9443/ws`
- **Supported Assets & Intervals:** Whitelisted in `app/market/constants.py`:
  - Symbols: `BTCUSDT`, `ETHUSDT`, `SOLUSDT`
  - Intervals: `1h`, `4h`

### Core Architectural Principles
1. **Financial Decimal Precision:** All monetary amounts, asset quantities, fees, cash, and equity must use Python `Decimal` and PostgreSQL `NUMERIC(38, 18)`. Floats are strictly prohibited for financial/portfolio accounting and are only permitted for technical indicator math (`numpy.float64`).
2. **Strict Timezone Handling:** All timestamps must be timezone-aware UTC (`timezone.utc`). Naive datetimes are rejected at model and database boundaries.
3. **Causal Integrity (Zero Lookahead):** Indicators, backtest iterators, and multi-timeframe alignments only use data finalized at or before the decision time. Future bars never leak into decision frames.
4. **Strict Sequence Validation:** Candle sequences must be contiguous, sorted chronologically, and aligned to UTC interval boundaries. Missing intervals or misalignments trigger `CandleSequenceValidationError`. Data is **never** silently sorted, interpolated, or synthesized.
5. **Deterministic Execution Simulation:** Fills take adverse slippage, fixed ATR stops (1R), profit targets (2R), and **stop-first** resolution when both boundaries are breached in the same bar.

---

## 2. Architecture & Directory Structure

```
crypto-ai-trader/
├── alembic/                  # Database migration scripts (Alembic)
│   └── versions/             # Versioned schema migrations
├── app/
│   ├── ai/                   # AI market analysis stubs (reserved for future offline analysis)
│   ├── api/                  # FastAPI web routing & dependency injection
│   │   └── routes.py         # /health and read-only /market/candles endpoints
│   ├── backtest/             # Deterministic backtesting engine & metrics
│   │   ├── analysis.py       # Win rate, profit factor, drawdown, trade analysis
│   │   ├── data.py           # Multi-timeframe data preparation (1H trading, 4H context)
│   │   ├── engine.py         # Multi-symbol walk-forward runner & equity curve builder
│   │   ├── execution.py      # Simulation fill models & trade record tracking
│   │   ├── models.py         # Frozen dataclasses (BacktestConfig, Position, Trade, EquityPoint)
│   │   └── portfolio.py      # Shared spot portfolio, risk bounds, daily loss limits
│   ├── config/               # Pydantic settings loading from .env
│   ├── database/             # SQLAlchemy engine, session management, CandleRecord ORM model
│   ├── indicators/           # Pure causal technical indicators (EMA, RSI, MACD, ATR, BB)
│   ├── market/               # Market data domain: Binance client, WS stream, ingestion, validation
│   │   ├── constants.py      # Allowed symbols, intervals, limits
│   │   ├── exchange.py       # Binance REST client
│   │   ├── ingestion.py      # Idempotent persistence pipeline
│   │   ├── models.py         # Pydantic schemas (Candle, CandleOpenObservation)
│   │   ├── repository.py     # Database queries & upsert handling
│   │   ├── validation.py     # Sequence contiguity, gap & alignment validation
│   │   └── websocket.py      # Unauthenticated Binance WebSocket listener
│   ├── risk/                 # Risk management modules (stubs/extensions)
│   ├── strategy/             # Breakout strategy implementation
│   │   └── breakout.py       # Multi-timeframe breakout signal generator
│   └── trading/              # Real-time paper execution
│       └── paper.py          # PaperExecutor reacting to CandleOpenObservation & Candle
├── scripts/
│   └── fetch_candles.py      # CLI script for bounded historical candle ingestion
├── tests/                    # Comprehensive unit and integration test suite (pytest)
├── docker-compose.yml        # PostgreSQL 16 container definition
├── pyproject.toml            # Project metadata, dependencies, ruff and pytest config
└── requirements.txt          # Python dependencies
```

---

## 3. Technology Stack

- **Language:** Python 3.13+
- **Framework:** FastAPI, Uvicorn
- **Database:** PostgreSQL 16, SQLAlchemy 2.0 (declarative), psycopg3 (`postgresql+psycopg`), Alembic
- **Data & Math:** Pandas, NumPy, Python standard `decimal.Decimal`
- **Networking:** HTTPX (async REST client), websockets (async WebSocket client)
- **Validation & Settings:** Pydantic v2, Pydantic-Settings
- **Testing & Quality:** Pytest, pytest-asyncio, Ruff (line length 100, target Python 3.13)

---

## 4. Development Workflow & Commands

### Virtual Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### PostgreSQL Database (Docker)
PostgreSQL is configured to run on `localhost:5432` with user/password/db as `crypto`/`crypto`/`crypto_trader`.
```bash
# Start PostgreSQL container
docker compose up -d

# Check status
docker compose ps

# Apply database migrations
.venv/bin/alembic upgrade head

# Stop database container
docker compose down
```

### Environment Configuration
Copy `.env.example` to `.env` for local overrides:
```bash
cp .env.example .env
```
*Note: Never commit `.env` or store real exchange API keys.*

### Historical Data Ingestion
Ingest closed historical candles via CLI:
```bash
.venv/bin/python scripts/fetch_candles.py --symbol BTCUSDT --interval 1h --limit 100
```

### Running the API Server
```bash
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
- Health Check: `curl http://127.0.0.1:8000/health`
- Read Candles: `curl 'http://127.0.0.1:8000/market/candles?symbol=BTCUSDT&interval=1h&limit=10'`

### Running Tests
The test suite utilizes `pytest` with `asyncio_mode = "auto"`.
```bash
# Run entire test suite (requires PostgreSQL container running for DB integration tests)
.venv/bin/pytest -v

# Run only pure unit tests (runs without PostgreSQL)
.venv/bin/pytest -k "not (repository or test_market_api or test_prepare_backtest_data or test_prepare_multitimeframe_data)"

# Run specific test file
.venv/bin/pytest tests/test_backtest_analysis.py -v
```
Database integration tests use an isolated, dynamically generated schema (`test_market_<uuid>`) that is created and dropped after each test.

### Code Formatting & Linting
```bash
# Lint checks
.venv/bin/ruff check .

# Format code
.venv/bin/ruff format .
```

---

## 5. Coding & Engineering Conventions

### Decimal Handling
- Always parse string numbers into `Decimal(str_val)` before arithmetic.
- Avoid passing floats directly into `Decimal` (e.g., use `Decimal("0.05")`, not `Decimal(0.05)`).
- When persisting or validating quantities and prices, ensure they match the required step/precision.

### Time & Timestamps
- Use `datetime.now(timezone.utc)` or explicit UTC offsets (`datetime(..., tzinfo=timezone.utc)`).
- Never use naive datetimes (`datetime.now()` or `datetime.utcnow()` which is deprecated in Python 3.12+).
- Use `interval_duration(interval)` helper from `app.market.validation` for interval math.

### DataFrames & Indicators
- Maintain separation between canonical candle data (`Decimal`) and indicators (`np.float64`).
- Indicator functions in `app/indicators/technical.py` are pure: they must accept validated candle sequences and return new DataFrames without mutating inputs.
- Indicators do not forward-fill missing values; they leave leading `NaN`s until sufficient observation periods exist.

### Backtesting Rules
- The walk-forward iterator (`iter_walk_forward`) must always produce deep copies of chronological prefixes to avoid state leakage.
- Stop losses are checked before profit targets on bars where both price extremes are touched.
- Positions are forcibly closed at the final available in-range bar's close price at the end of the simulation.

### Tests & Verification
- When modifying backtest, strategy, or indicator logic, add deterministic test cases with synthetic `Candle` sequences covering edge cases (boundary conditions, tie-breaks, slippage, gap handling).
- Do not make external network requests in tests; mock `httpx` or `websockets` if testing ingestion or network adapters.

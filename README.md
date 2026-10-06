# Crypto AI Trader

A staged foundation for crypto market analysis and paper trading. Phase 1 provides a small FastAPI health endpoint, environment-based settings, SQLAlchemy connection setup, Alembic scaffolding, and PostgreSQL through Docker Compose.

## Phase 1 scope

This project is initially paper-trading only. No real-money trading is enabled. Exchange connectivity and AI trading decisions are not implemented. There are no trading endpoints, trading tables, or application migrations yet.

## Local development

Python 3.13 is used on the development VM.

```bash
cd ~/crypto-ai-trader
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The checked-in `.env.example` contains placeholders. A local `.env` is used for development settings; never commit `.env` or put real credentials in it.

## PostgreSQL

Start the PostgreSQL 16 container (published only on localhost):

```bash
docker compose up -d
```

Check its status with `docker compose ps`. Stop PostgreSQL with:

```bash
docker compose down
```

The named volume preserves database data when the container is stopped.

## FastAPI

Activate the virtual environment, then start the API:

```bash
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Health endpoint: `GET http://127.0.0.1:8000/health`

```bash
curl http://127.0.0.1:8000/health
```

Run tests with `pytest`.

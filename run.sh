#!/usr/bin/env bash
set -e

# Start the Crypto AI Trader live paper trading system and dashboard API
export PYTHONPATH=.
exec .venv/bin/python app/main.py "$@"


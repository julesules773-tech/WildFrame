#!/usr/bin/env python3
"""
worker.py — Procrastinate job-queue worker.

Handles BOTH job execution and periodic-task deferral (Procrastinate
starts its periodic deferrer inside the worker), so one process is enough
for the scheduled satellite/FIRMS polls.

Usage:
    python3 worker.py
"""

# Load .env (WILDFRAME_DATABASE_URL etc.) before jobs.py / db.py import.
from pathlib import Path

import os

from dotenv import dotenv_values

# .env is the source of truth, but dotenv refuses to override variables that
# already exist in the environment — including EMPTY ones (e.g. a leftover
# `export NASA_FIRMS_API_KEY=` shadows the real key in .env and the worker
# then reports "NASA_FIRMS_API_KEY not set"). Fill in values for any key
# that is missing OR empty so .env actually applies.
_env_path = Path(__file__).parent / ".env"
for _key, _value in dotenv_values(_env_path).items():
    if _value and not os.environ.get(_key):
        os.environ[_key] = _value

import jobs

if __name__ == "__main__":
    print("🔥 WildFrame job worker (Procrastinate)")
    print("   Listening for jobs + deferring periodic tasks...")
    jobs.app.run_worker()

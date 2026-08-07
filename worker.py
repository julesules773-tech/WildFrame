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

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

import jobs

if __name__ == "__main__":
    print("🔥 WildFrame job worker (Procrastinate)")
    print("   Listening for jobs + deferring periodic tasks...")
    jobs.app.run_worker()

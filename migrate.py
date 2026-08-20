#!/usr/bin/env python3
"""
migrate.py — one-time setup: create the Postgres schema and import the
legacy JSON stores into Postgres.

Usage:
    .venv/bin/python migrate.py

Safe to re-run: tables are created idempotently and each store is only
imported if it's currently empty.
"""

import sys
from pathlib import Path

# Load .env (WILDFRAME_DATABASE_URL etc.) before db.py reads env vars.
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

import db


def main() -> int:
    print("🌲 WildFrame Postgres migration")
    # Mask password in connection string for safe logging
    import re
    _safe_url = re.sub(r'(://[^:]+:)[^@]+@', r'\1***@', db.DATABASE_URL)
    print(f"   Connecting to: {_safe_url}")

    if not db.check_connection():
        print("   ❌ Cannot reach Postgres. Is `brew services start postgresql`"
              " running and the `wildframe` database created?")
        print("      createdb wildframe && psql -d wildframe -c 'CREATE EXTENSION postgis'")
        return 1

    print("   Creating schema (wildframe tables + procrastinate queue tables)...")
    db.init_schema()
    print("   ✅ Schema ready")

    print("   Importing legacy JSON data...")
    summary = db.import_json_data()
    print(f"   ✅ Imported {summary['reports']} production reports, "
          f"{summary['demo_reports']} demo reports, "
          f"{summary['osm_cache']} OSM cache entries")
    if summary["skipped"]:
        print(f"   Skipped: {', '.join(summary['skipped'])}")

    # Seed the global grid-id counter above any ids that already exist so
    # new grids never collide with imported/legacy grid ids.
    seeded = db.seed_grid_counter_from_existing()
    if seeded:
        print(f"   ✅ Grid id counter seeded at {seeded}")

    # Backfill the evidence-age column for grids created before it existed,
    # so the 24h expiry sweep doesn't treat them all as "no evidence".
    backfilled = db.backfill_last_evidence_at()
    if backfilled:
        print(f"   ✅ Backfilled evidence age on {backfilled} existing grid(s)")

    print("Done. Start the server with:  python3 server.py")
    print("Start the job worker with:    python3 worker.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

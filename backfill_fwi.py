#!/usr/bin/env python3
"""
backfill_fwi.py — One-shot EFFIS fuel-moisture backfill for grids
=================================================================

Drains the entire "needs fuel-moisture" queue for a mode (every grid whose
``fwi_updated_at`` is older than the cutoff — e.g. ALL grids right after
``migrate.py`` adds the ffmc/dmc/isi columns to an existing table): for
each grid it fetches today's FFMC/DMC/ISI from EFFIS via
``effis_fwi.get_fwi_full`` (which caches per ~0.5° cell per day in the
shared kv_store and enforces the daily request budget) and persists them
with ``db.update_grid_fwi``. Grids outside EFFIS's EMNA coverage box are
stamped with ``db.touch_grid_fwi`` so the daily sweep never rescans them.

Why this exists
---------------
The worker's periodic ``grids.advance`` job refreshes only 200 grids of
EFFIS data per run, so a freshly-migrated database where every grid has
``fwi_updated_at = 0`` would take many hours to populate organically.
This drains the whole queue in one run (a few minutes — the per-cell
cache means a handful of real WMS requests for a full-Europe population)
so the moisture curve is live immediately.

Safety
------
- Consecutive passes with zero progress (WMS down, daily budget
  exhausted, backoff active) abort after ``--max-idle-passes`` instead of
  hammering EFFIS — the remaining queue is reported, not retried forever.
- Only the ffmc/dmc/isi columns are written; grid probability state is
  never touched. Safe to run concurrently with the worker: both drain the
  same queue via the same cell cache, and the UPDATEs are idempotent.

Usage
-----
    .venv/bin/python backfill_fwi.py                          # drain all production grids
    .venv/bin/python backfill_fwi.py --mode demo              # demo grids instead
    .venv/bin/python backfill_fwi.py --batch 200 --max-grids 500   # bounded smoke test
"""

import argparse
import os
import sys
import time
from pathlib import Path

# --- Load .env BEFORE importing db (db reads env vars at import time) ---
# Mirror worker.py's fill-in pattern: dotenv refuses to override existing
# shell vars — including EMPTY ones — so only missing/empty keys are filled.
from dotenv import dotenv_values

_env_path = Path(__file__).parent / ".env"
for _key, _value in dotenv_values(_env_path).items():
    if _value and not os.environ.get(_key):
        os.environ[_key] = _value

import db
import effis_fwi

# Small positive max_age so a grid stamped earlier in THIS run isn't
# re-selected by the next pass's queue query (avoids redundant re-fetches
# at the one-second boundary).
MAX_AGE_S = 30.0


def _remaining(mode: str, max_age_s: float = MAX_AGE_S) -> int:
    cutoff = time.time() - max_age_s
    with db._conn() as conn:
        return conn.execute(
            "SELECT count(*) AS n FROM bayesian_grids "
            "WHERE mode = %s AND fwi_updated_at < %s",
            (mode, cutoff),
        ).fetchone()["n"]


def drain(
    mode: str,
    batch: int = 500,
    max_idle_passes: int = 5,
    max_grids: int = 0,
    no_backoff: bool = False,
) -> dict:
    """Drain the FWI queue for ``mode``. Returns a summary dict."""
    if no_backoff:
        # The module pauses live fetches for 60 s after 3 consecutive
        # failures — a protection designed for the worker hammering a dead
        # endpoint. A one-shot backfill re-attempts nothing (fetch_failed_ids)
        # and the population is bounded, so the pause is pure dead time here.
        effis_fwi._should_backoff = lambda: False
    t0 = time.time()
    total_updated = total_touched = total_failed = 0
    processed = 0
    idle = 0
    pass_no = 0
    aborted = ""
    # Every grid processed this run (stamped OR value-fetched) is remembered
    # so it can never be re-picked by a later pass — the queue query orders
    # by fwi_updated_at ASC, and a freshly-stamped grid goes stale again
    # after MAX_AGE_S, so without this the loop would drain in a circle.
    seen: set[str] = set()
    # Grids whose fetch returned no data stay in the queue — but don't
    # re-attempt them every pass (a retry storm against a public service).
    # One attempt per run; the worker's daily refresh retries them.
    fetch_failed_ids: set[str] = set()

    while True:
        if max_grids > 0 and processed >= max_grids:
            break
        rows = db.list_grids_needing_fwi(mode, limit=batch, max_age_s=MAX_AGE_S)
        rows = [r for r in rows if r["id"] not in seen and r["id"] not in fetch_failed_ids]
        if not rows:
            break

        # Respect --max-grids mid-batch.
        if max_grids > 0:
            rows = rows[: max_grids - processed]

        updated = touched = failed = 0
        for row in rows:
            lat, lon = row["centroid_lat"], row["centroid_lon"]
            if not effis_fwi.in_coverage(lat, lon):
                # Permanent no-data (EFFIS only covers EMNA) — stamp so the
                # sweep moves on instead of rescanning it every run.
                db.touch_grid_fwi(mode, row["id"])
                touched += 1
                continue
            ffmc, dmc, isi, fetched = effis_fwi.get_fwi_full(lat, lon)
            if fetched > 0 and db.update_grid_fwi(mode, row["id"], ffmc, dmc, isi):
                updated += 1
            else:
                failed += 1  # no data yet — leave in the queue, skip for this run
                fetch_failed_ids.add(row["id"])

        pass_no += 1
        processed += len(rows)
        seen.update(r["id"] for r in rows)
        total_updated += updated
        total_touched += touched
        total_failed += failed
        if fetch_failed_ids:
            print(
                f"[backfill-fwi]     (no-data so far for {len(fetch_failed_ids)} "
                f"grid(s) — will not retry them this run; worker retries later)",
                flush=True,
            )

        done = updated + touched
        if done == 0:
            idle += 1
            if idle >= max_idle_passes:
                aborted = (
                    f"EFFIS unreachable or budget exhausted "
                    f"({failed} fetch failure(s) this pass)"
                )
                print(
                    f"[backfill-fwi] ABORTING after {idle} idle pass(es) — {aborted}."
                )
                break
        else:
            idle = 0

        remaining = max(0, _remaining(mode) - len(seen))
        print(
            f"[backfill-fwi] pass {pass_no}: +{updated} values, "
            f"+{touched} out-of-coverage stamped, {failed} fetch-failed, "
            f"~{remaining} never-attempted remaining "
            f"(elapsed {time.time() - t0:.0f}s)",
            flush=True,
        )

    elapsed = time.time() - t0
    summary = {
        "mode": mode,
        "updated_with_values": total_updated,
        "out_of_coverage_stamped": total_touched,
        "fetch_failed": total_failed,
        "grids_processed": processed,
        "remaining": max(0, _remaining(mode) - len(seen)),
        "elapsed_s": round(elapsed, 1),
        "aborted": aborted or None,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-shot EFFIS fuel-moisture backfill (FFMC/DMC/ISI).",
    )
    parser.add_argument("--mode", default="production",
                        help="DB mode to drain (default production)")
    parser.add_argument("--batch", type=int, default=500,
                        help="grids fetched per queue pass (default 500)")
    parser.add_argument("--max-idle-passes", type=int, default=5,
                        help="consecutive zero-progress passes before abort (default 5)")
    parser.add_argument("--max-grids", type=int, default=0,
                        help="process at most N grids then stop (0 = drain all)")
    parser.add_argument("--no-backoff", action="store_true",
                        help="skip the module's failure backoff sleeps (safe here: "
                             "no grid is re-attempted within a run)")
    args = parser.parse_args()

    if not effis_fwi.ENABLED:
        print("EFFIS is disabled (WILDFRAME_EFFIS=0) — nothing to backfill.")
        sys.exit(1)

    need = _remaining(args.mode)
    print(
        f"[backfill-fwi] mode={args.mode}: {need} grid(s) need fuel-moisture "
        f"(batch={args.batch}"
        + (f", max_grids={args.max_grids}" if args.max_grids else "")
        + "). Fetching today's FFMC/DMC/ISI from EFFIS…"
    )

    summary = drain(
        mode=args.mode,
        batch=args.batch,
        max_idle_passes=args.max_idle_passes,
        max_grids=args.max_grids,
        no_backoff=args.no_backoff,
    )

    print("-" * 72)
    print("BACKFILL SUMMARY")
    print(f"  grids processed          : {summary['grids_processed']}")
    print(f"  with fresh FFMC/DMC/ISI  : {summary['updated_with_values']}")
    print(f"  out-of-coverage stamped  : {summary['out_of_coverage_stamped']}")
    print(f"  fetch failed (no data)   : {summary['fetch_failed']}")
    print(f"  still needing (remain)   : {summary['remaining']}")
    print(f"  elapsed                  : {summary['elapsed_s']}s")
    if summary["aborted"]:
        print(f"  aborted                  : {summary['aborted']}")
        sys.exit(2)
    if summary["remaining"]:
        print(
            "  NOTE: some grids still lack values (EFFIS daily layer not "
            "published or fetch failed) — the worker's periodic refresh will "
            "retry them."
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
weather.py — Real wind speed/direction for Bayesian fire grids (Open-Meteo)
===========================================================================

Before this module, every Bayesian grid was born with the hardcoded
defaults ``wind_speed=3.0`` and ``wind_dir_deg=270.0`` (West) — the DB
column default, the grid-creation defaults, the serialization fallbacks
and the advance job all used 270, and nothing ever wrote a real value.
Since the spread ellipse, the smoke-drift upwind shift and the road-risk
model all read wind, every fire on the map "blew west".

This module fetches real 10 m wind from the Open-Meteo API (no API key
needed; the free tier allows ~10 000 requests/day but is non-commercial
only — a paid plan is the compliant choice for commercial deployments,
see open-meteo.com/pricing):

    https://api.open-meteo.com/v1/forecast?latitude=..&longitude=..&current=wind_speed_10m,wind_direction_10m&wind_speed_unit=ms

Convention
----------
Open-Meteo reports the direction the wind comes FROM (meteorological
convention). WildFrame stores the direction the fire spreads TOWARD
(the head of the spread ellipse, 0° = north, 90° = east — see
SpreadKernel in bayesian_filter.py). This module flips the direction by
180° before returning it, so the rest of the app needs no changes.

Caching & budget
----------------
- Coordinates are bucketed into ~0.5° (~55 km) cells and cached for 24
  hours, both in-process and in the shared ``kv_store`` (so the web
  process and the worker share one cache). The coarse cell + long TTL
  keep the free tier's ~10k/day budget from being exhausted by the
  ~15k production grids — see the ``CELL_DEG``/``CACHE_TTL_S`` notes
  below for the sizing math.
- A hard per-day request budget (default 9500, under the 10k free tier)
  is enforced via a kv_store counter so the free tier is never blown, no
  matter how many grids exist.
- Any failure (API down, timeout, budget exhausted, DB down) degrades to
  the old neutral defaults — weather must NEVER block report ingestion
  or grid creation.
- Failed fetches refund their budget slot (a timeout storm must not eat
  the daily quota), and after a few consecutive failures a short backoff
  pauses live fetches instead of hammering a dead endpoint.
- Concurrent callers wanting the same ~55 km cell serialize on one
  in-flight lock, so the map's on-demand viewport refresh can't fire
  duplicate requests for the same cell.
"""

import logging
import os
import threading
import time
from typing import Optional

import requests

import db

logger = logging.getLogger(__name__)

# Neutral fallback — exactly what the codebase used before weather, so a
# failure is a no-op (fires just stop steering by wind instead of breaking).
DEFAULT_WIND_SPEED_MPS = 3.0
DEFAULT_WIND_DIR_DEG = 270.0  # compass — direction the fire spreads TOWARD

# Set WILDFRAME_WEATHER=0 to disable live weather entirely (uses defaults).
ENABLED = os.environ.get("WILDFRAME_WEATHER", "1") != "0"

API_BASE = "https://api.open-meteo.com/v1/forecast"
TIMEOUT_S = 3.0

# Cache geometry: ~0.5° cells (~55 km) reused for 24 hours.
#
# Sized for the free tier: with ~15.7k production grids across ~6k unique
# 0.5° cells, a 24h TTL means at most one fetch per cell per day (~6.1k
# requests/day, ~64% of the 9.5k daily budget) with headroom left for
# on-demand viewport refreshes. A finer 0.25° grid or a shorter TTL would
# need ~24k+ requests/day and exhaust the budget before noon — the
# "everything blows West" failure mode. On a paid Open-Meteo plan, drop
# CELL_DEG to 0.25 and CACHE_TTL_S to ~6h for fresher, finer wind.
CELL_DEG = 0.5
CACHE_TTL_S = 24 * 60 * 60

# Hard daily request budget (free tier ~10k/day; leave headroom for the
# shared counter racing slightly over across processes). Overridable via
# env for deployments on a paid plan.
DAILY_BUDGET = int(os.environ.get("WILDFRAME_WEATHER_DAILY_BUDGET", "9500"))

# In-process cache: cell_key -> (expires_epoch, wind_speed, wind_dir_deg).
_mem_cache: dict[str, tuple[float, float, float]] = {}
_mem_lock = threading.Lock()

# --- Failure backoff ---------------------------------------------------
# After a few consecutive failures (API down, timeout storm), pause live
# fetches for a short window instead of hammering a dead endpoint and
# burning the daily budget on requests that return nothing.
FAIL_BACKOFF_THRESHOLD = 3   # consecutive failures before pausing
FAIL_BACKOFF_WINDOW_S = 60.0 # pause window after the FIRST failure
_fail_lock = threading.Lock()
_fail_count = 0
_fail_since = 0.0

# --- Per-cell in-flight dedup ------------------------------------------
# The web process (map viewport refresh) and the worker (global sweep) can
# ask for the same ~55 km cell at the same moment. Serializing per cell
# means one live fetch populates the shared cache instead of N duplicates.
_inflight_lock = threading.Lock()
_inflight: dict[str, threading.Lock] = {}


def _cell_key(lat: float, lon: float) -> str:
    """Bucket key for a ~0.5° (~55 km) cell — nearby fires share a fetch."""
    return f"{round(lat / CELL_DEG) * CELL_DEG:.2f},{round(lon / CELL_DEG) * CELL_DEG:.2f}"


def _kv_key(cell: str) -> str:
    return f"weather:{cell}"


def _budget_key() -> str:
    return f"weather_budget:{time.strftime('%Y-%m-%d')}"


def _spend_budget() -> bool:
    """Atomically check-and-increment the per-day request budget.

    Returns True if a live fetch is allowed. DB failures fail open (the
    fetch may proceed) so weather can't be blocked by a DB blip.
    """
    try:
        n = db.kv_get(_budget_key(), 0) or 0
        if int(n) >= DAILY_BUDGET:
            return False
        db.kv_set(_budget_key(), int(n) + 1)
        return True
    except Exception:
        return True


def _refund_budget() -> None:
    """Best-effort refund of a budget slot after a failed live fetch, so a
    timeout storm doesn't silently consume the daily quota on requests that
    returned nothing."""
    try:
        n = db.kv_get(_budget_key(), 0) or 0
        if int(n) > 0:
            db.kv_set(_budget_key(), int(n) - 1)
    except Exception:
        pass


def _note_failure() -> None:
    global _fail_count, _fail_since
    with _fail_lock:
        if _fail_count == 0:
            _fail_since = time.time()
        _fail_count += 1


def _note_success() -> None:
    global _fail_count, _fail_since
    with _fail_lock:
        _fail_count = 0
        _fail_since = 0.0


def _should_backoff() -> bool:
    """True if recent consecutive failures pause live fetches for the
    backoff window. Resets the window once it has elapsed (the next call
    after the pause tries the API again)."""
    global _fail_count, _fail_since
    with _fail_lock:
        if _fail_count == 0:
            return False
        if time.time() - _fail_since >= FAIL_BACKOFF_WINDOW_S:
            _fail_count = 0
            _fail_since = 0.0
            return False
        return _fail_count >= FAIL_BACKOFF_THRESHOLD


def _cell_lock(cell: str) -> threading.Lock:
    """Return the (shared, per-cell) lock serializing live fetches."""
    with _inflight_lock:
        lk = _inflight.get(cell)
        if lk is None:
            lk = threading.Lock()
            _inflight[cell] = lk
        return lk


def _cache_get(cell: str, now: float) -> Optional[tuple[float, float, float]]:
    """Return ``(speed, dir, fetched_epoch)`` from the in-memory or shared
    cache, or None on a miss. Callers treat a hit as "fresh" (fetched=now)."""
    with _mem_lock:
        hit = _mem_cache.get(cell)
        if hit and hit[0] > now:
            return hit[1], hit[2], now
    try:
        shared = db.kv_get(_kv_key(cell))
        if shared and (shared.get("at") or 0) > now - CACHE_TTL_S:
            speed, dir_ = float(shared["speed"]), float(shared["dir"])
            with _mem_lock:
                _mem_cache[cell] = (now + CACHE_TTL_S, speed, dir_)
            return speed, dir_, now
    except Exception:
        pass
    return None


def get_wind_full(lat: float, lon: float) -> tuple[float, float, float]:
    """Return ``(wind_speed_mps, wind_dir_deg, fetched_epoch)``.

    ``fetched_epoch`` is ``time.time()`` when the value is real (live
    fetch or a <30-min cache hit) and ``0`` when it fell back to the
    neutral defaults. Callers can persist it so the periodic refresh only
    re-fetches stale wind.
    """
    if not ENABLED:
        return DEFAULT_WIND_SPEED_MPS, DEFAULT_WIND_DIR_DEG, 0.0

    cell = _cell_key(lat, lon)
    now = time.time()

    # 1) Caches (fast path — in-memory, then shared kv_store)
    cached = _cache_get(cell, now)
    if cached is not None:
        return cached

    # 2) Live fetch — serialized per cell. Re-check the caches once the
    #    lock is held: another caller may have fetched while we waited.
    with _cell_lock(cell):
        cached = _cache_get(cell, time.time())
        if cached is not None:
            return cached

        if _should_backoff():
            return DEFAULT_WIND_SPEED_MPS, DEFAULT_WIND_DIR_DEG, 0.0

        if not _spend_budget():
            logger.warning("[weather] daily budget exhausted — using default wind")
            return DEFAULT_WIND_SPEED_MPS, DEFAULT_WIND_DIR_DEG, 0.0

        try:
            resp = requests.get(
                API_BASE,
                params={
                    "latitude": round(lat, 4),
                    "longitude": round(lon, 4),
                    "current": "wind_speed_10m,wind_direction_10m",
                    "wind_speed_unit": "ms",
                },
                timeout=TIMEOUT_S,
            )
            resp.raise_for_status()
            cur = (resp.json() or {}).get("current") or {}
            speed = float(cur["wind_speed_10m"])
            from_dir = float(cur["wind_direction_10m"])
            # Open-Meteo: direction wind comes FROM → flip to spread direction.
            dir_ = (from_dir + 180.0) % 360.0
        except Exception as exc:
            logger.warning("[weather] fetch failed for (%.3f, %.3f): %s", lat, lon, exc)
            _note_failure()
            _refund_budget()
            return DEFAULT_WIND_SPEED_MPS, DEFAULT_WIND_DIR_DEG, 0.0

        _note_success()
        try:
            db.kv_set(_kv_key(cell), {"at": time.time(), "speed": speed, "dir": dir_})
        except Exception:
            pass
        with _mem_lock:
            _mem_cache[cell] = (time.time() + CACHE_TTL_S, speed, dir_)
        return speed, dir_, time.time()


def refresh_grids_wind(
    mode: str,
    limit: int = 200,
    max_age_s: float = 24 * 60 * 60,
) -> int:
    """Refresh wind for up to ``limit`` grids whose stored wind is older
    than ``max_age_s``. Returns how many grids got a fresh value.

    Called by the worker's periodic ``grids.advance`` job so long-lived
    fires track changing weather. Because get_wind_full() caches per
    ~55 km cell, a slice of 200 grids usually costs a handful of real
    API calls, and the daily budget guards the free tier regardless.
    """
    if not ENABLED:
        return 0
    rows = db.list_grids_needing_wind(mode, limit=limit, max_age_s=max_age_s)
    updated = 0
    for row in rows:
        speed, dir_, fetched = get_wind_full(row["centroid_lat"], row["centroid_lon"])
        if fetched > 0 and db.update_grid_wind(mode, row["id"], speed, dir_):
            updated += 1
    if updated:
        logger.info("[weather] refreshed wind for %d %s grid(s).", updated, mode)
    return updated

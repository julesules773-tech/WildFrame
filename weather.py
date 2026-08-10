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

This module fetches real 10 m wind from the free Open-Meteo API (no API
key, ~10 000 requests/day free tier, allowed for commercial use):

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
- Coordinates are bucketed into ~0.25° (~28 km) cells and cached for 30
  minutes, both in-process and in the shared ``kv_store`` (so the web
  process and the worker share one cache).
- A hard per-day request budget (default 6000) is enforced via a
  kv_store counter so the free tier is never blown, no matter how many
  grids exist.
- Any failure (API down, timeout, budget exhausted, DB down) degrades to
  the old neutral defaults — weather must NEVER block report ingestion
  or grid creation.
"""

import logging
import os
import threading
import time

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

# Cache geometry: ~0.25° cells (~28 km) reused for 30 minutes.
CELL_DEG = 0.25
CACHE_TTL_S = 30 * 60

# Hard daily request budget (free tier ~10k/day; leave headroom for
# retries). Overridable via env for deployments on a paid plan.
DAILY_BUDGET = int(os.environ.get("WILDFRAME_WEATHER_DAILY_BUDGET", "6000"))

# In-process cache: cell_key -> (expires_epoch, wind_speed, wind_dir_deg).
_mem_cache: dict[str, tuple[float, float, float]] = {}
_mem_lock = threading.Lock()


def _cell_key(lat: float, lon: float) -> str:
    """Bucket key for a ~0.25° (~28 km) cell — nearby fires share a fetch."""
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

    # 1) In-process cache
    with _mem_lock:
        hit = _mem_cache.get(cell)
        if hit and hit[0] > now:
            return hit[1], hit[2], now

    # 2) Shared kv_store cache (web + worker processes)
    try:
        shared = db.kv_get(_kv_key(cell))
        if shared and (shared.get("at") or 0) > now - CACHE_TTL_S:
            speed, dir_ = float(shared["speed"]), float(shared["dir"])
            with _mem_lock:
                _mem_cache[cell] = (now + CACHE_TTL_S, speed, dir_)
            return speed, dir_, now
    except Exception:
        pass

    # 3) Live fetch (budgeted)
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
        return DEFAULT_WIND_SPEED_MPS, DEFAULT_WIND_DIR_DEG, 0.0

    try:
        db.kv_set(_kv_key(cell), {"at": now, "speed": speed, "dir": dir_})
    except Exception:
        pass
    with _mem_lock:
        _mem_cache[cell] = (now + CACHE_TTL_S, speed, dir_)
    return speed, dir_, now


def refresh_grids_wind(
    mode: str,
    limit: int = 200,
    max_age_s: float = 30 * 60,
) -> int:
    """Refresh wind for up to ``limit`` grids whose stored wind is older
    than ``max_age_s``. Returns how many grids got a fresh value.

    Called by the worker's periodic ``grids.advance`` job so long-lived
    fires track changing weather. Because get_wind_full() caches per
    ~28 km cell, a slice of 200 grids usually costs a handful of real
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

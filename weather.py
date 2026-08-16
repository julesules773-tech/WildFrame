#!/usr/bin/env python3
"""
weather.py — Real wind speed/direction for Bayesian fire grids
===============================================================

Before this module, every Bayesian grid was born with the hardcoded
defaults ``wind_speed=3.0`` and ``wind_dir_deg=270.0`` (West) — the DB
column default, the grid-creation defaults, the serialization fallbacks
and the advance job all used 270, and nothing ever wrote a real value.
Since the spread ellipse, the smoke-drift upwind shift and the road-risk
model all read wind, every fire on the map "blew west".

Providers
---------
- **WeatherAPI.com** (primary when ``WILDFRAME_WEATHERAPI_KEY`` is set) —
  paid plan (e.g. $7/mo, 3M calls), commercial license, not budget-capped.
  Current wind via ``/v1/current.json?q=lat,lon`` (wind_kph + wind_degree).
- **Open-Meteo** (default, and automatic fallback when WeatherAPI fails) —
  no API key; the free tier allows ~10 000 requests/day but is
  non-commercial only. Budget-gated so the free tier is never blown.

    https://api.open-meteo.com/v1/forecast?latitude=..&longitude=..&current=wind_speed_10m,wind_direction_10m&wind_speed_unit=ms

Convention
----------
Both providers report the direction the wind comes FROM (meteorological
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
- A hard per-day request budget is enforced via kv_store counters, one
  per provider: ``weather_budget`` (default 9500, under Open-Meteo's 10k
  free tier) guards the **fallback**, and ``weatherapi_budget`` (default
  60000, under the paid plan's ~66.7k/day) caps the **primary** so a
  runaway loop can't burn the paid quota. Either cap exhausted falls
  through to the other provider, and if both are out → neutral defaults.
- Any failure (API down, timeout, budget exhausted, DB down) degrades to
  the old neutral defaults — weather must NEVER block report ingestion
  or grid creation.
- Failed fallback fetches refund their budget slot (a timeout storm must
  not eat the daily quota), and after a few consecutive failures a short
  backoff pauses live fetches instead of hammering a dead endpoint.
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

# Primary provider: WeatherAPI.com (paid, commercial license). Fetches are
# NOT budget-capped (own quota), and any failure falls back to Open-Meteo.
# Leave empty to use Open-Meteo directly. Accepts the unprefixed alias too
# (easy to typo), with the WILDFRAME_* name winning.
WEATHERAPI_API_KEY = (
    os.environ.get("WILDFRAME_WEATHERAPI_KEY")
    or os.environ.get("WEATHERAPI_API_KEY")
    or ""
).strip()
# forecast.json returns BOTH current conditions AND the hourly series in one
# call — so every wind fetch also populates the forecast cache (one weatherapi
# call per cell per day serves both needs). Horizon: 3 days is massive
# headroom for road risk (tiers cap at 6h); retain longer only when
# backtesting makes it a real requirement (that wants table storage, not kv).
WEATHERAPI_API_BASE = "https://api.weatherapi.com/v1/forecast.json"
FORECAST_DAYS = 3

# Fallback / default provider: Open-Meteo (free, non-commercial; budget-gated).
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

# Hard daily request budget for the Open-Meteo FALLBACK (free tier
# ~10k/day; leave headroom for the shared counter racing slightly over
# across processes). Overridable via env for deployments on a paid plan.
DAILY_BUDGET = int(os.environ.get("WILDFRAME_WEATHER_DAILY_BUDGET", "9500"))

# Hard daily request budget for WeatherAPI.com (paid plan: ~2M calls/mo
# ≈ 66.7k/day). Cap well under the plan so a runaway loop can't burn the
# paid quota; weatherapi fetches that hit the cap fall back to Open-Meteo
# (which is itself budget-gated). Overridable via env.
WEATHERAPI_DAILY_BUDGET = int(
    os.environ.get("WILDFRAME_WEATHERAPI_DAILY_BUDGET", "60000")
)

# In-process cache: cell_key -> (expires_epoch, wind_speed, wind_dir_deg).
_mem_cache: dict[str, tuple[float, float, float]] = {}
# Forecast-series cache: cell_key -> (expires_epoch, [hour dicts...]).
_mem_fc_cache: dict[str, tuple[float, list]] = {}
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


def _fc_kv_key(cell: str) -> str:
    return f"weather_fc:{cell}"


def _budget_key(prefix: str) -> str:
    return f"{prefix}:{time.strftime('%Y-%m-%d')}"


def _spend_budget(key: str, limit: int) -> bool:
    """Atomically check-and-increment a per-day request budget counter.

    ``key`` is the full kv_store key (e.g. ``_budget_key("weather_budget")``)
    and ``limit`` its per-day cap. Returns True if a live fetch is allowed.
    DB failures fail open (the fetch may proceed) so weather can't be
    blocked by a DB blip.
    """
    try:
        n = db.kv_get(key, 0) or 0
        if int(n) >= limit:
            return False
        db.kv_set(key, int(n) + 1)
        return True
    except Exception:
        return True


def _refund_budget(key: str) -> None:
    """Best-effort refund of a budget slot after a failed live fetch, so a
    timeout storm doesn't silently consume the daily quota on requests that
    returned nothing."""
    try:
        n = db.kv_get(key, 0) or 0
        if int(n) > 0:
            db.kv_set(key, int(n) - 1)
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
    fetch or a fresh cache hit) and ``0`` when it fell back to the
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

        # 2a) Primary: WeatherAPI.com (paid — own daily budget cap).
        if WEATHERAPI_API_KEY:
            if not _spend_budget(_budget_key("weatherapi_budget"), WEATHERAPI_DAILY_BUDGET):
                logger.warning(
                    "[weather] weatherapi daily budget exhausted — falling back to Open-Meteo"
                )
            else:
                try:
                    speed, dir_, series = _fetch_weatherapi(lat, lon)
                    _store_cell(cell, speed, dir_, series)
                    _note_success()
                    return speed, dir_, time.time()
                except Exception as exc:
                    logger.warning(
                        "[weather] weatherapi fetch failed for (%.3f, %.3f): %s — falling back to Open-Meteo",
                        lat, lon, exc,
                    )
                    _refund_budget(_budget_key("weatherapi_budget"))
                    # Note the failure but DON'T return: fall through to Open-Meteo.
                    _note_failure()

        # 2b) Fallback: Open-Meteo (free tier — budget-gated).
        if not _spend_budget(_budget_key("weather_budget"), DAILY_BUDGET):
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
            _refund_budget(_budget_key("weather_budget"))
            return DEFAULT_WIND_SPEED_MPS, DEFAULT_WIND_DIR_DEG, 0.0

        _store_cell(cell, speed, dir_)
        _note_success()
        return speed, dir_, time.time()


def _parse_weatherapi_hour(h: dict) -> dict:
    """Map one WeatherAPI.com forecast hour dict to our compact shape
    (speed in m/s, direction already flipped to spread-TOWARD)."""
    speed_mps = float(h["wind_kph"]) / 3.6
    from_dir = float(h["wind_degree"])
    return {
        "ts": int(h["time_epoch"]),
        "speed": round(speed_mps, 2),
        "dir": round((from_dir + 180.0) % 360.0, 1),
        "precip_mm": round(float(h.get("precip_mm") or 0.0), 2),
        "humidity": round(float(h.get("humidity") or 0.0), 1),
        "temp_c": round(float(h.get("temp_c") or 0.0), 1),
    }


def _fetch_weatherapi(lat: float, lon: float) -> tuple[float, float, list]:
    """Fetch current wind + the hourly forecast series from WeatherAPI.com.

    Returns ``(speed_mps, dir_deg, series)`` with directions already flipped
    to spread-TOWARD, where ``series`` is a list of hourly dicts (ts, speed,
    dir, precip_mm, humidity, temp_c) for the next ``FORECAST_DAYS`` days.
    Raises on any error (HTTP, JSON, missing field) so the caller falls back
    to Open-Meteo for wind (the series is simply unavailable then)."""
    resp = requests.get(
        WEATHERAPI_API_BASE,
        params={
            "key": WEATHERAPI_API_KEY,
            "q": f"{lat:.4f},{lon:.4f}",
            "aqi": "no",
            "days": FORECAST_DAYS,
        },
        timeout=TIMEOUT_S,
    )
    resp.raise_for_status()
    data = resp.json() or {}
    cur = (data.get("current") or {})
    # WeatherAPI.com: wind_kph + wind_degree (direction wind comes FROM).
    speed_mps = float(cur["wind_kph"]) / 3.6
    from_dir = float(cur["wind_degree"])

    series: list = []
    for day in ((data.get("forecast") or {}).get("forecastday") or []):
        for hour in (day.get("hour") or []):
            try:
                series.append(_parse_weatherapi_hour(hour))
            except (KeyError, TypeError, ValueError):
                continue  # skip a malformed hour, keep the rest
    return speed_mps, (from_dir + 180.0) % 360.0, series


def _store_cell(cell: str, speed: float, dir_: float, series: Optional[list] = None) -> None:
    """Persist a freshly-fetched wind value (and forecast series, if any) to
    the shared cache + memory."""
    try:
        db.kv_set(_kv_key(cell), {"at": time.time(), "speed": speed, "dir": dir_})
        if series:
            db.kv_set(_fc_kv_key(cell), {"at": time.time(), "hours": series})
    except Exception:
        pass
    now = time.time()
    with _mem_lock:
        _mem_cache[cell] = (now + CACHE_TTL_S, speed, dir_)
        if series:
            _mem_fc_cache[cell] = (now + CACHE_TTL_S, series)


def _fc_cache_get(cell: str, now: float) -> Optional[list]:
    """Return the cached hourly series for ``cell`` (in-memory or shared kv),
    or None on a miss / stale. Freshness follows ``CACHE_TTL_S`` like wind."""
    with _mem_lock:
        hit = _mem_fc_cache.get(cell)
        if hit and hit[0] > now:
            return hit[1]
    try:
        shared = db.kv_get(_fc_kv_key(cell))
        if shared and (shared.get("at") or 0) > now - CACHE_TTL_S:
            series = shared.get("hours") or []
            with _mem_lock:
                _mem_fc_cache[cell] = (now + CACHE_TTL_S, series)
            return series
    except Exception:
        pass
    return None


def get_forecast_series(lat: float, lon: float) -> Optional[list]:
    """Return the hourly forecast series (ts, speed, dir, precip_mm, humidity,
    temp_c) for the ~55 km cell around ``(lat, lon)``, or None when
    unavailable (weather disabled, no weatherapi key, fetch failed, budget
    exhausted, or the fallback provider is serving wind).

    The series is populated as a side-effect of ``get_wind_full`` (one
    forecast.json call per cell per day serves both) — this is a pure cache
    read. Never raises; callers treat None as "no forecast → current-wind
    road risk only"."""
    if not ENABLED or not WEATHERAPI_API_KEY:
        return None
    cell = _cell_key(lat, lon)
    now = time.time()
    cached = _fc_cache_get(cell, now)
    if cached is not None:
        return cached
    # Series missing while weatherapi is the provider: trigger one live
    # fetch (serialized per cell) so a cell fetched before this feature
    # existed self-heals within a day. Budget-spends like any weatherapi
    # fetch; on failure → None (road risk falls back to current wind).
    with _cell_lock(cell):
        cached = _fc_cache_get(cell, time.time())
        if cached is not None:
            return cached
        if _should_backoff():
            return None
        if not _spend_budget(_budget_key("weatherapi_budget"), WEATHERAPI_DAILY_BUDGET):
            logger.warning("[weather] weatherapi daily budget exhausted — no forecast")
            return None
        try:
            speed, dir_, series = _fetch_weatherapi(lat, lon)
            _store_cell(cell, speed, dir_, series)
            _note_success()
            return series
        except Exception as exc:
            logger.warning("[weather] forecast fetch failed for (%.3f, %.3f): %s", lat, lon, exc)
            _refund_budget(_budget_key("weatherapi_budget"))
            _note_failure()
            return None


def refresh_grids_wind(
    mode: str,
    limit: int = 200,
    max_age_s: float = 24 * 60 * 60,
    max_wall_s: float = 15.0,
) -> int:
    """Refresh wind for up to ``limit`` grids whose stored wind is older
    than ``max_age_s``. Returns how many grids got a fresh value.

    Called by the worker's periodic ``grids.advance`` job so long-lived
    fires track changing weather. Because get_wind_full() caches per
    ~55 km cell, a slice of 200 grids usually costs a handful of real
    API calls, and the daily budget guards the free tier regardless.

    ``max_wall_s`` is a hard wall-clock cap so a degraded weather API
    (network hangs / slow responses) can never stretch the single
    worker's ``grids.advance`` job and starve the other periodic jobs
    behind it (same pattern as effis_fwi.refresh_grids_fwi).
    """
    if not ENABLED:
        return 0
    rows = db.list_grids_needing_wind(mode, limit=limit, max_age_s=max_age_s)
    updated = 0
    deadline = time.monotonic() + max_wall_s
    for row in rows:
        if time.monotonic() >= deadline:
            break
        speed, dir_, fetched = get_wind_full(row["centroid_lat"], row["centroid_lon"])
        if fetched > 0 and db.update_grid_wind(mode, row["id"], speed, dir_):
            updated += 1
    if updated:
        logger.info("[weather] refreshed wind for %d %s grid(s).", updated, mode)
    return updated

#!/usr/bin/env python3
"""
effis_fwi.py — Fuel-moisture / fire-weather indices from EFFIS (Copernicus EMS)
===============================================================================

Why
---
The spread model (SpreadKernel in bayesian_filter.py) was driven by wind
alone. Fuel moisture is the other half of the Rothermel/FWI picture: the same
wind crawls through moist fuels and explodes through cured, bone-dry ones.
EFFIS publishes the full Canadian FWI stack (FFMC, DMC, DC, ISI, BUI, FWI) as
raw-value raster layers over an anonymous WMS — no API key, CC BY 4.0.

Source (verified live 2026-08-12)
---------------------------------
    https://maps.effis.emergency.copernicus.eu/effis   (WMS 1.3.0)

    One GetMap request with FORMAT=image/tiff returns a 32-bit FLOAT,
    multi-band GeoTIFF of RAW index values (not a colour render):

      SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap
      &LAYERS=mf010.ffmc,mf010.dmc,mf010.isi
      &STYLES=&CRS=EPSG:4326
      &BBOX=<minLat>,<minLon>,<maxLat>,<maxLon>
      &WIDTH=24&HEIGHT=24&FORMAT=image/tiff&TIME=YYYY-MM-DD

    Values verified sane for Poland in summer: FFMC ~86-93, DMC ~18-50,
    ISI ~3-11. Gotchas found while probing: must be WMS 1.3.0 with
    CRS=EPSG:4326 (1.1.1 returned nodata for some regions), TIME must be an
    explicit date (no ``current``), and GetFeatureInfo is NOT offered on
    these raster layers — pixel extraction from a tiny centred tile is the
    way to read a value at a point.

Coverage & cadence
------------------
- Daily values; the ``mf010.*`` layers are the Meteo-France ~10 km
  deterministic forecast. ``TIME`` also accepts future days (1-9 day
  forecast), which is how a predictive road-risk pass could use
  tomorrow's moisture.
- Coverage is EMNA only — Europe, Middle East, North Africa (~lat 26.5-72.5,
  lon -18.5-44.5). Outside it this module returns "no data" and the model
  behaves exactly as before (moisture_factor = 1.0). GWIS is the global
  sister system if that ever becomes a need.

How it plugs into the model
---------------------------
- ffmc -> moisture_factor  : scales SpreadKernel's base rate (head/back/
  flank all expand or contract). FFMC is converted to fine-fuel moisture %
  via Van Wagner (1987) and the factor is exponential in FMC, anchored to
  1.0 at FMC 12% (FFMC ~89.5) so the current wind-only behaviour is the
  neutral middle, not an edge.
- dmc  -> decay_scale      : lengthens the probability decay half-life
  (deep duff keeps a fire smouldering after surface fuels are consumed).
- isi                        : stored for display/cross-check only. ISI is
  the FWI system's official wind+moisture spread driver; we keep our own
  Open-Meteo wind and moisture curve, and ISI is the natural calibration
  target for the curve.

Caching & budget (same pattern as weather.py)
---------------------------------------------
- ~0.5° (~55 km) cells cached per DATE in the shared kv_store, plus an
  in-process mirror. EFFIS is ~10 km native and daily, so a coarse cell +
  daily cache keeps request counts tiny (one multiband request per
  (cell, day); a full-Europe sweep is a few dozen requests).
- A modest per-day request budget guards the public service regardless.
- Any failure (WMS down, timeout, budget, DB down) degrades to "no
  moisture" — fire modelling must NEVER block on this.
- Consecutive failures trigger a short backoff (mirrors weather.py).
"""

import io
import logging
import math
import os
import threading
import time
from typing import Optional

import numpy as np
import requests
from PIL import Image

import db

logger = logging.getLogger(__name__)

# Set WILDFRAME_EFFIS=0 to disable fuel-moisture entirely (factor 1.0).
ENABLED = os.environ.get("WILDFRAME_EFFIS", "1") != "0"

WMS_BASE = "https://maps.effis.emergency.copernicus.eu/effis"
TIMEOUT_S = 10.0

# EMNA coverage box (Europe, Middle East, North Africa) — approx bounds.
COVERAGE_BBOX = (-18.5, 26.5, 44.5, 72.5)  # (min_lon, min_lat, max_lon, max_lat)

# ~0.5° (~55 km) cells; EFFIS is ~10 km native and daily, so a coarse cell
# + daily cache is plenty and keeps request counts tiny.
CELL_DEG = 0.5
CACHE_TTL_S = 12 * 3600  # re-fetch at most ~2x/day (AM/PM forecast runs)

DAILY_BUDGET = int(os.environ.get("WILDFRAME_EFFIS_DAILY_BUDGET", "3000"))

# One multiband request per (cell, day): layers are band-ordered.
LAYERS = "mf010.ffmc,mf010.dmc,mf010.isi"
TILE_SPAN_DEG = 1.2  # tile extent around the point
TILE_PX = 24

# Plausible raw index ranges (0 = nodata fill in these float rasters).
# FFMC/DMC are validated strictly (they drive the model); ISI is display
# / cross-check only, so it just needs to be positive — extreme wind-driven
# days push ISI past 50 (sometimes 100+), and those are exactly the days we
# must NOT throw away FFMC/DMC for.
#
# DMC's upper bound: the Canadian FWI DMC code tops out around 800-900,
# and extreme drought pushes it far past the old 300 cap — which silently
# rejected the whole tuple as "insane" and reported NO moisture for the
# driest fires (measured live Aug 2026: DMC 510-600 for Iberia, 758-784
# for the Middle East). 900 keeps the strictness (nodata is 0) while
# accepting real data with headroom.
_VALUE_RANGES = {
    "ffmc": (1.0, 101.0),
    "dmc": (1.0, 900.0),
    "isi": (1.0, 1e9),
}

# Calibrated moisture curve (calibrate_fwi.py, 164 stored grids): the
# canonical Van Wagner (1987) fine-fuel function fF(m) that feeds the FWI
# system's Initial Spread Index (ISI = 0.208 * fF * exp(0.05039 * U_kmh)),
# normalized so factor = 1.0 at the bias-neutral anchor below.
_FW_FMC_REF_PCT = 8.9  # population-mean FMC of observed fires (FFMC ~91.8)
_FWI_K1 = 91.9
_FWI_K2 = 0.1386
_FWI_K3 = 5.31
_FWI_K4 = 4.93e7


def _fwi_ff(m: float) -> float:
    """Van Wagner (1987) fine-fuel function fF(m), moisture in FMC %."""
    return _FWI_K1 * math.exp(-_FWI_K2 * m) * (1.0 + m ** _FWI_K3 / _FWI_K4)


# Precomputed fF at the anchor so every moisture_factor() call is 2 multiplies.
_FW_REF_FF = _fwi_ff(_FW_FMC_REF_PCT)

# --- In-process cache: "effis:{date}:{cell}" -> (expires_epoch, (ffmc, dmc, isi))
_mem_cache: dict[str, tuple[float, tuple[float, float, float]]] = {}
_mem_lock = threading.Lock()

# --- Failure backoff (mirrors weather.py) ---
FAIL_BACKOFF_THRESHOLD = 3   # consecutive failures before pausing
FAIL_BACKOFF_WINDOW_S = 60.0
_fail_lock = threading.Lock()
_fail_count = 0
_fail_since = 0.0

# --- Per-cell in-flight dedup ---
_inflight_lock = threading.Lock()
_inflight: dict[str, threading.Lock] = {}


def _cell_key(lat: float, lon: float) -> str:
    return f"{round(lat / CELL_DEG) * CELL_DEG:.2f},{round(lon / CELL_DEG) * CELL_DEG:.2f}"


def _kv_key(cell: str, date: str) -> str:
    return f"effis:{date}:{cell}"


def _budget_key() -> str:
    return f"effis_budget:{time.strftime('%Y-%m-%d')}"


def in_coverage(lat: float, lon: float) -> bool:
    """True if the point is inside EFFIS's EMNA coverage box."""
    min_lon, min_lat, max_lon, max_lat = COVERAGE_BBOX
    return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon


def _spend_budget() -> bool:
    try:
        n = db.kv_get(_budget_key(), 0) or 0
        if int(n) >= DAILY_BUDGET:
            return False
        db.kv_set(_budget_key(), int(n) + 1)
        return True
    except Exception:
        return True  # fail open — never block modelling on a DB blip


def _refund_budget() -> None:
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
    with _inflight_lock:
        lk = _inflight.get(cell)
        if lk is None:
            lk = threading.Lock()
            _inflight[cell] = lk
        return lk


def _cache_get(cell: str, date: str, now: float) -> Optional[tuple[float, float, float]]:
    key = _kv_key(cell, date)
    with _mem_lock:
        hit = _mem_cache.get(key)
        if hit and hit[0] > now:
            return hit[1]
    try:
        shared = db.kv_get(key)
        if shared and (shared.get("at") or 0) > now - CACHE_TTL_S:
            val = (float(shared["ffmc"]), float(shared["dmc"]), float(shared["isi"]))
            with _mem_lock:
                _mem_cache[key] = (now + CACHE_TTL_S, val)
            return val
    except Exception:
        pass
    return None


def _tiff_magic_ok(content: bytes) -> bool:
    return content[:4] == b"II*\x00" or content[:4] == b"MM\x00*"


def _centre_pixel(arr: np.ndarray) -> float:
    """Median of the non-zero 3x3 patch around the centre pixel (0 = nodata)."""
    h, w = arr.shape[:2]
    cy, cx = h // 2, w // 2
    pad = 1
    patch = arr[max(0, cy - pad):cy + pad + 1, max(0, cx - pad):cx + pad + 1]
    nz = patch[patch > 0.0]
    return float(np.median(nz)) if nz.size else 0.0


def _get_tile(lat: float, lon: float, date: str, layers: str) -> np.ndarray:
    """One WMS GetMap → float32 array. Raises ValueError on non-TIFF replies."""
    half = TILE_SPAN_DEG / 2.0
    resp = requests.get(
        WMS_BASE,
        params={
            "SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetMap",
            "LAYERS": layers, "STYLES": "",
            "CRS": "EPSG:4326",
            "BBOX": f"{lat - half:.4f},{lon - half:.4f},{lat + half:.4f},{lon + half:.4f}",
            "WIDTH": TILE_PX, "HEIGHT": TILE_PX,
            "FORMAT": "image/tiff", "TIME": date,
        },
        timeout=TIMEOUT_S,
    )
    resp.raise_for_status()
    if not _tiff_magic_ok(resp.content):
        # Not a TIFF — the server returned a ServiceException XML.
        raise ValueError("EFFIS WMS returned a non-TIFF response")
    return np.asarray(Image.open(io.BytesIO(resp.content)), dtype=np.float32)


def _fetch_tile(lat: float, lon: float, date: str) -> tuple[float, float, float]:
    """(ffmc, dmc, isi) at the tile centre for a point.

    Tries ONE multiband GetMap (LAYERS=ffmc,dmc,isi); the EFFIS server
    sometimes flattens a multi-layer request to a single band, so on that
    we fall back to three per-layer requests. The tile is centred on
    (lat, lon), so the centre pixel is the point's value regardless of
    WMS 1.3.0 axis order.
    """
    try:
        arr = _get_tile(lat, lon, date, LAYERS)
        if arr.ndim != 3 or arr.shape[2] < 3:
            raise ValueError("single-band TIFF from multi-layer request")
        return _centre_pixel(arr[:, :, 0]), _centre_pixel(arr[:, :, 1]), _centre_pixel(arr[:, :, 2])
    except ValueError:
        return (
            _centre_pixel(_get_tile(lat, lon, date, "mf010.ffmc")),
            _centre_pixel(_get_tile(lat, lon, date, "mf010.dmc")),
            _centre_pixel(_get_tile(lat, lon, date, "mf010.isi")),
        )


def _sane(values: tuple[float, float, float]) -> bool:
    for v, name in zip(values, ("ffmc", "dmc", "isi")):
        lo, hi = _VALUE_RANGES[name]
        if not (lo <= v <= hi):
            return False
    return True


def get_fwi_full(lat: float, lon: float) -> tuple[Optional[float], Optional[float], Optional[float], float]:
    """Return ``(ffmc, dmc, isi, fetched_epoch)`` for a point.

    ``fetched_epoch > 0`` means the values are real (live fetch or fresh
    cache hit). ``(None, None, None, 0.0)`` means "no data" — disabled,
    out of coverage, budget/backoff, fetch failure, or the daily layer
    isn't published yet — and callers must fall back to
    moisture_factor = 1.0 (the pre-EFFIS behaviour).
    """
    if not ENABLED or not in_coverage(lat, lon):
        return None, None, None, 0.0

    cell = _cell_key(lat, lon)
    date = time.strftime("%Y-%m-%d", time.gmtime())
    now = time.time()

    cached = _cache_get(cell, date, now)
    if cached is not None:
        return cached[0], cached[1], cached[2], now

    # Live fetch — serialized per (date, cell).
    lock_key = f"{date}:{cell}"
    with _cell_lock(lock_key):
        cached = _cache_get(cell, date, time.time())
        if cached is not None:
            return cached[0], cached[1], cached[2], time.time()

        if _should_backoff():
            return None, None, None, 0.0
        if not _spend_budget():
            logger.warning("[effis] daily budget exhausted — using no fuel moisture")
            return None, None, None, 0.0

        try:
            values = _fetch_tile(lat, lon, date)
            if not _sane(values):
                # All-nodata or out-of-range — the daily layer likely isn't
                # published yet. Don't cache (allow a retry next call).
                _refund_budget()
                return None, None, None, 0.0
        except Exception as exc:
            logger.warning("[effis] fetch failed for (%.3f, %.3f): %s", lat, lon, exc)
            _note_failure()
            _refund_budget()
            return None, None, None, 0.0

        _note_success()
        try:
            db.kv_set(_kv_key(cell, date), {
                "at": time.time(), "ffmc": values[0], "dmc": values[1], "isi": values[2],
            })
        except Exception:
            pass
        with _mem_lock:
            _mem_cache[_kv_key(cell, date)] = (time.time() + CACHE_TTL_S, values)
        return values[0], values[1], values[2], time.time()


# ---------------------------------------------------------------------------
# Model-facing conversions
# ---------------------------------------------------------------------------

def ffmc_to_fmc_pct(ffmc: float) -> float:
    """Van Wagner (1987): FFMC → approximate fine-fuel moisture content %."""
    f = max(0.0, min(101.0, float(ffmc)))
    return 147.2 * (101.0 - f) / (59.5 + f)


def moisture_factor(ffmc: float) -> float:
    """Spread-rate multiplier from FFMC, calibrated to the canonical FWI curve.

    Replaces the old placeholder exponential (exp(-0.045*m), anchored at
    FMC 12%) with Van Wagner (1987)'s fF(m) — the exact moisture curve that
    drives EFFIS's Initial Spread Index — anchored at the bias-neutral FMC
    8.9% (FFMC ~91.8) found by calibrate_fwi.py over the
    stored grid population, so the average observed fire still spreads at
    factor 1.0. The curve is steeper than the placeholder: FFMC 75 ->
    ~0.20, FFMC 89.5 -> ~0.72, FFMC 92 -> ~1.0, FFMC 96 -> ~1.78.
    Clamped to [0.2, 2.0].
    """
    if not ffmc or ffmc <= 0:
        return 1.0
    fmc = ffmc_to_fmc_pct(ffmc)
    factor = _fwi_ff(fmc) / _FW_REF_FF
    return max(0.2, min(2.0, factor))


def decay_scale(dmc: float) -> float:
    """Probability decay half-life multiplier from DMC (duff moisture).

    Deep duff keeps a fire smouldering: DMC <= 20 leaves the half-life
    unchanged, DMC 60+ doubles it. Clamped to [1.0, 2.0].
    """
    if not dmc or dmc <= 0:
        return 1.0
    return max(1.0, min(2.0, 1.0 + (dmc - 20.0) / 40.0))


# ---------------------------------------------------------------------------
# Periodic refresh (called from jobs.grids.advance)
# ---------------------------------------------------------------------------

def refresh_grids_fwi(
    mode: str,
    limit: int = 200,
    max_age_s: float = 12 * 3600,
) -> int:
    """Refresh fuel-moisture for up to ``limit`` grids whose stored values
    are older than ``max_age_s``. Returns how many grids got fresh values.

    Because get_fwi_full() caches per ~55 km cell per day, a slice of 200
    grids usually costs a handful of real WMS requests. Grids outside the
    EMNA coverage box are stamped as "checked" (fwi_updated_at) so the
    daily sweep doesn't rescan them forever.
    """
    if not ENABLED:
        return 0
    rows = db.list_grids_needing_fwi(mode, limit=limit, max_age_s=max_age_s)
    updated = 0
    for row in rows:
        lat, lon = row["centroid_lat"], row["centroid_lon"]
        if not in_coverage(lat, lon):
            # Permanent no-data (EFFIS doesn't cover this region) — stamp so
            # the sweep moves on instead of rescanning it every run.
            db.touch_grid_fwi(mode, row["id"])
            continue
        ffmc, dmc, isi, fetched = get_fwi_full(lat, lon)
        if fetched > 0 and db.update_grid_fwi(mode, row["id"], ffmc, dmc, isi):
            updated += 1
    if updated:
        logger.info("[effis] refreshed fuel-moisture for %d %s grid(s).", updated, mode)
    return updated

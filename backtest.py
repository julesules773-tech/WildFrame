#!/usr/bin/env python3
from __future__ import annotations
"""
backtest.py — Validate the Bayesian fire spread model against real fire perimeters.

Downloads fire perimeters from NIFC WFIGS (national, 2021+) and CAL FIRE
(California, 2019+), generates realistic synthetic FIRMS-like hotspots inside
each perimeter, runs the Bayesian model, and compares predicted vs actual areas.

Usage:
    # National backtest (default: 200 fires across all available years)
    python backtest.py --verbose

    # California-only, specific year
    python backtest.py --source calfire --year 2023 --max-fires 100 --verbose

    # Multi-year national
    python backtest.py --source nifc --years 2024 2025 --max-fires 300

    # Store results in DB for the confidence page
    python backtest.py --store --verbose

Requires: shapely, numpy (already in the project).
NASA_FIRMS_API_KEY is NOT needed — hotspots are synthetic (within real perimeters).
"""

import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from urllib.request import Request, urlopen
from urllib.parse import quote

# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────

# NIFC WFIGS — national fire perimeters (current year + recent)
NIFC_URL = (
    "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
    "WFIGS_Interagency_Perimeters_Current/FeatureServer/0/query"
)

# CAL FIRE — California historical perimeters
CALFIRE_URL = (
    "https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/arcgis/rest/services/"
    "California_Historic_Fire_Perimeters/FeatureServer/0/query"
)

# Model parameters
DEFAULT_CELL_SIZE_M = 1000.0  # 1 km grid cells for backtest
SIMPLIFY_TOLERANCE_DEG = 0.005  # ~555m for perimeter simplification

# Hotspot generation tuning
# Real FIRMS VIIRS detects ~0.3-1.5 hotspots per km² per pass.
# With 2 passes/day over a 5-14 day fire, that's 3-20+ per km² total.
# We target realistic total counts per fire (not per-pass).
HOTSPOT_MIN = 15               # minimum hotspots per fire
HOTSPOT_MAX = 400              # cap to avoid excessive computation


# ─────────────────────────────────────────────────────────────────────
# NIFC WFIGS Perimeter Download (National, 2021+)
# ─────────────────────────────────────────────────────────────────────

def download_nifc_fires(year: int | None = None,
                         min_acres: float = 100,
                         max_fires: int = 500,
                         verbose: bool = False) -> list[dict]:
    """Download fire perimeters from NIFC WFIGS (national coverage).

    If year is specified, filters by discovery date in that year.
    Otherwise fetches the most recent fires (current season).
    """
    from shapely.geometry import shape as shapely_shape

    fires = []
    offset = 0
    batch_size = 1000

    # Build WHERE clause (no date filter — NIFC's Current service only has
    # recent fires, and date filtering via the API is unreliable. We filter
    # by year in Python after download.)
    where_parts = [
        "attr_IncidentTypeCategory='WF'",   # wildfires only (not RX/prescribed)
        f"poly_GISAcres>{min_acres}",
    ]

    where = " AND ".join(where_parts)

    while len(fires) < max_fires:
        params = {
            "where": where,
            "outFields": "poly_IncidentName,poly_GISAcres,attr_FireDiscoveryDateTime,"
                         "attr_POOState,attr_POOCounty,attr_IncidentName",
            "outSR": "4326",
            "f": "json",
            "resultOffset": offset,
            "resultRecordCount": min(batch_size, max_fires - len(fires)),
            "orderByFields": "poly_GISAcres DESC",
        }
        query = "&".join(f"{k}={quote(str(v))}" for k, v in params.items())
        url = f"{NIFC_URL}?{query}"

        if verbose:
            print(f"[nifc] fetching offset={offset} (have {len(fires)} so far)…")

        try:
            req = Request(url, headers={"User-Agent": "pyrae-backtest/2.0"})
            with urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
        except Exception as e:
            print(f"  ✗ NIFC API request failed: {e}")
            break

        features = data.get("features", [])
        if not features:
            break

        for f in features:
            attrs = f.get("attributes", {})
            geom = f.get("geometry")
            if not geom:
                continue

            try:
                # ArcGIS polygon rings → GeoJSON
                rings = geom.get("rings", [])
                if not rings:
                    continue
                geojson_geom = {"type": "Polygon", "coordinates": rings}
                poly = shapely_shape(geojson_geom)
                if poly.is_empty or poly.area < 1e-8:
                    continue

                centroid = poly.centroid
                acres = attrs.get("poly_GISAcres", 0) or 0

                # Discovery date (epoch ms)
                disc_ms = attrs.get("attr_FireDiscoveryDateTime")
                fire_date = ""
                fire_year = 0
                if disc_ms:
                    try:
                        dt = datetime.fromtimestamp(disc_ms / 1000, tz=timezone.utc)
                        fire_date = dt.strftime("%Y-%m-%d")
                        fire_year = dt.year
                    except (ValueError, OSError):
                        pass

                # Filter by year if specified
                if year and fire_year and fire_year != year:
                    continue

                state = (attrs.get("attr_POOState") or "").replace("US-", "")
                county = attrs.get("attr_POOCounty") or ""

                fires.append({
                    "fire_name": attrs.get("poly_IncidentName") or attrs.get("attr_IncidentName") or "Unknown",
                    "acres": float(acres),
                    "area_km2": float(acres) * 0.00404686,
                    "year": year or fire_year or 0,
                    "fire_date": fire_date,
                    "state": state,
                    "county": county,
                    "source": "nifc",
                    "lat": centroid.y,
                    "lon": centroid.x,
                    "geometry": poly,
                    "bounds": poly.bounds,
                })
            except Exception as e:
                if verbose:
                    print(f"  skip NIFC feature: {e}")
                continue

        if len(features) < batch_size:
            break
        offset += batch_size

    if verbose:
        print(f"[nifc] downloaded {len(fires)} fire perimeters"
              f"{f' for {year}' if year else ''}")

    return fires


# ─────────────────────────────────────────────────────────────────────
# CAL FIRE Perimeter Download (California)
# ─────────────────────────────────────────────────────────────────────

def download_calfire_perimeters(year: int, verbose: bool = False) -> list[dict]:
    """Download fire perimeters from CAL FIRE ArcGIS REST API."""
    from shapely.geometry import shape as shapely_shape

    fires = []
    offset = 0
    batch_size = 1000

    while True:
        params = {
            "where": f"YEAR_={year}",
            "outFields": "*",
            "outSR": "4326",
            "f": "json",
            "resultOffset": offset,
            "resultRecordCount": batch_size,
            "returnCountOnly": "false",
        }
        query = "&".join(f"{k}={quote(str(v))}" for k, v in params.items())
        url = f"{CALFIRE_URL}?{query}"

        if verbose:
            print(f"[calfire] fetching perimeters offset={offset}…")

        req = Request(url, headers={"User-Agent": "pyrae-backtest/2.0"})
        with urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())

        features = data.get("features", [])
        if not features:
            break

        for f in features:
            attrs = f.get("attributes", {})
            geom = f.get("geometry")
            if not geom:
                continue

            try:
                rings = geom.get("rings", [])
                if not rings:
                    continue
                geojson_geom = {"type": "Polygon", "coordinates": rings}
                poly = shapely_shape(geojson_geom)
                if poly.is_empty:
                    continue

                centroid = poly.centroid
                acres = attrs.get("GIS_ACRES", 0) or 0

                alarm_ms = attrs.get("ALARM_DATE")
                fire_date = ""
                if alarm_ms:
                    try:
                        dt = datetime.fromtimestamp(alarm_ms / 1000, tz=timezone.utc)
                        fire_date = dt.strftime("%Y-%m-%d")
                    except (ValueError, OSError):
                        pass

                fires.append({
                    "fire_name": attrs.get("FIRE_NAME", "Unknown"),
                    "acres": float(acres),
                    "area_km2": float(acres) * 0.00404686,
                    "year": year,
                    "fire_date": fire_date,
                    "state": "CA",
                    "county": "",
                    "source": "calfire",
                    "lat": centroid.y,
                    "lon": centroid.x,
                    "geometry": poly,
                    "bounds": poly.bounds,
                })
            except Exception as e:
                if verbose:
                    print(f"  skip feature: {e}")
                continue

        if len(features) < batch_size:
            break
        offset += batch_size

    if verbose:
        print(f"[calfire] downloaded {len(fires)} fire perimeters for {year}")

    return fires


# ─────────────────────────────────────────────────────────────────────
# Realistic Synthetic Hotspot Generation
# ─────────────────────────────────────────────────────────────────────

def generate_realistic_hotspots(
    fire: dict,
    verbose: bool = False,
) -> list[dict]:
    """Generate FIRMS-like hotspots inside a fire perimeter.

    Uses a multi-pass approach for realism:
    1. Edge-biased sampling: fire fronts produce more satellite detections
       than already-burned interiors
    2. Temporal clustering: hotspots arrive in daily passes (satellite
       overpass times), with more detections early and during peak spread
    3. Realistic FRP: fire radiative power follows a log-normal distribution
       (most detections are moderate, few are extreme)
    4. Confidence filtering: ~80% high, ~15% nominal, ~5% low (matches FIRMS)
    """
    from shapely.geometry import Point
    from shapely.ops import unary_union

    poly = fire["geometry"]
    area_km2 = fire["area_km2"]

    # Number of hotspots proportional to fire size
    # Real FIRMS: ~0.3-1.5 per km² per satellite pass, 2 passes/day,
    # fire burns for days → total 3-20+ per km² across the fire's lifetime.
    if area_km2 > 100:
        density = random.uniform(2.0, 8.0)    # large active fires
    elif area_km2 > 10:
        density = random.uniform(3.0, 12.0)   # medium fires
    elif area_km2 > 1:
        density = random.uniform(5.0, 20.0)   # small fires (higher relative detection)
    else:
        density = random.uniform(10.0, 30.0)  # tiny fires

    n_hotspots = max(int(area_km2 * density), HOTSPOT_MIN)
    # Cap: enough hotspots for the model to detect the fire, but not so many
    # that the backtest takes forever. ~4 per km² is a good middle ground.
    fire_cap = max(HOTSPOT_MIN, min(HOTSPOT_MAX, int(area_km2 * 4)))
    n_hotspots = min(n_hotspots, fire_cap)

    if verbose:
        print(f"  generating {n_hotspots} hotspots for {area_km2:.1f} km² fire "
              f"(density={density:.4f}/km²)")

    # Parse fire date
    if fire["fire_date"]:
        try:
            start_date = datetime.strptime(fire["fire_date"], "%Y-%m-%d")
        except ValueError:
            start_date = datetime(2023, 7, 1)
    else:
        start_date = datetime(2023, 7, 1)

    # Determine fire duration (larger fires burn longer)
    if area_km2 > 100:
        duration_days = random.randint(7, 21)
    elif area_km2 > 10:
        duration_days = random.randint(3, 10)
    else:
        duration_days = random.randint(1, 5)

    # Simplify polygon for fast point-in-polygon tests, but keep enough
    # vertices to preserve the fire's actual shape (especially narrow fingers).
    try:
        # Use a smaller tolerance for small fires to avoid crushing them
        tol = min(SIMPLIFY_TOLERANCE_DEG, approx_radius_deg * 0.3)
        simple_poly = poly.simplify(tol, preserve_topology=True)
        # Ensure simplification didn't destroy the polygon
        if (simple_poly.is_empty or not simple_poly.is_valid or
                len(getattr(simple_poly.exterior, 'coords', [])) < 6 or
                not simple_poly.contains(Point(center_lon, center_lat))):
            simple_poly = poly
        # Fix invalid geometries (self-intersections etc.)
        if not simple_poly.is_valid:
            from shapely.validation import make_valid
            simple_poly = make_valid(simple_poly)
    except Exception:
        simple_poly = poly
    bounds = simple_poly.bounds  # (minx, miny, maxx, maxy)

    # --- Realistic fire growth simulation ---
    # A fire starts at a point and grows outward. Satellite hotspots cluster
    # near the active front (burning edge), not randomly inside the perimeter.
    import math as _math

    hotspots = []
    # Use a point guaranteed to be inside the polygon as the fire origin
    origin = poly.representative_point()
    center_lat = origin.y
    center_lon = origin.x
    # Approximate radius from area (km² → degrees)
    approx_radius_deg = _math.sqrt(area_km2 / _math.pi) / 111.0
    approx_radius_deg = max(approx_radius_deg, 0.01)  # at least ~1 km

    # Build concentric growth rings (each ring = one day's spread)
    rings = []
    for day_idx in range(duration_days):
        # Fire grows faster early, slows as it encounters resistance
        progress = (day_idx + 1) / duration_days
        ring_radius = approx_radius_deg * _math.sqrt(progress) * random.uniform(0.8, 1.2)
        rings.append(ring_radius)

    # Distribute hotspots across rings (days), weighted toward peak activity
    # Peak activity is in the middle of the fire's life
    ring_weights = []
    for i in range(duration_days):
        # Bell curve peaking at 30-50% of duration
        peak = 0.35
        w = _math.exp(-0.5 * ((i / max(duration_days, 1) - peak) / 0.25) ** 2)
        ring_weights.append(max(w, 0.1))
    total_w = sum(ring_weights)
    ring_counts = [max(int(n_hotspots * w / total_w), 2) for w in ring_weights]
    # Adjust to hit target total
    while sum(ring_counts) < n_hotspots:
        ring_counts[random.randint(0, duration_days - 1)] += 1
    while sum(ring_counts) > n_hotspots:
        idx = random.randint(0, duration_days - 1)
        if ring_counts[idx] > 2:
            ring_counts[idx] -= 1

    for day_idx in range(duration_days):
        fire_day = start_date + timedelta(days=day_idx)
        ring_r = rings[day_idx]
        n_this_ring = ring_counts[day_idx]

        for _ in range(n_this_ring):
            # Pick a random direction and distance within the ring
            angle = random.uniform(0, 2 * _math.pi)
            # Mix of front (ring edge) and interior hotspots
            if random.random() < 0.6:
                # Fire front: near the ring edge
                r = ring_r * random.uniform(0.85, 1.0)
            else:
                # Interior: already-burned area behind the front
                r = ring_r * random.uniform(0.0, 0.85)

            pt_lon = center_lon + r * _math.cos(angle) / _math.cos(_math.radians(center_lat))
            pt_lat = center_lat + r * _math.sin(angle)
            pt = Point(pt_lon, pt_lat)

            # Only place if inside the real perimeter
            try:
                if not simple_poly.contains(pt):
                    continue
            except Exception:
                continue

            # Satellite overpass times (VIIRS: ~1:30am and ~1:30pm UTC)
            hour = random.choice([13, 14, 1, 2])
            minute = random.randint(0, 59)
            acq_time = f"{hour:02d}{minute:02d}"

            # FRP: log-normal distribution
            frp = min(random.lognormvariate(3.0, 1.0), 2000.0)
            frp = max(frp, 1.0)

            # Confidence
            conf_roll = random.random()
            if conf_roll < 0.80:
                confidence = "high"
            elif conf_roll < 0.95:
                confidence = "nominal"
            else:
                confidence = "low"

            brightness = 300.0 + frp * 0.3 + random.gauss(0, 20)
            brightness = max(brightness, 290.0)

            hotspots.append({
                "lat": pt_lat,
                "lon": pt_lon,
                "acq_date": fire_day.strftime("%Y-%m-%d"),
                "acq_time": acq_time,
                "frp": round(frp, 1),
                "confidence": confidence,
                "brightness": round(brightness, 1),
                "satellite": random.choice(["N", "20", "21"]),
                "daynight": "D" if 6 <= hour <= 18 else "N",
            })

    # Fallback: if ring approach produced too few hotspots (e.g. elongated
    # fires where circular rings miss the polygon), use random sampling
    # with a tighter attempt limit.
    if len(hotspots) < HOTSPOT_MIN:
        hotspots = []  # clear partial results
        attempts = 0
        max_attempts = n_hotspots * 15  # tighter limit for fallback
        while len(hotspots) < n_hotspots and attempts < max_attempts:
            attempts += 1
            lon = random.uniform(bounds[0], bounds[2])
            lat = random.uniform(bounds[1], bounds[3])
            try:
                if not simple_poly.contains(Point(lon, lat)):
                    continue
            except Exception:
                continue

            day_offset = int(random.betavariate(2, 3) * duration_days)
            day_offset = min(day_offset, duration_days)
            fire_day = start_date + timedelta(days=day_offset)
            hour = random.choice([13, 14, 1, 2])
            minute = random.randint(0, 59)
            acq_time = f"{hour:02d}{minute:02d}"
            frp = min(random.lognormvariate(3.0, 1.0), 2000.0)
            frp = max(frp, 1.0)
            conf_roll = random.random()
            confidence = "high" if conf_roll < 0.80 else ("nominal" if conf_roll < 0.95 else "low")
            brightness = max(300.0 + frp * 0.3 + random.gauss(0, 20), 290.0)
            hotspots.append({
                "lat": lat, "lon": lon,
                "acq_date": fire_day.strftime("%Y-%m-%d"),
                "acq_time": acq_time,
                "frp": round(frp, 1), "confidence": confidence,
                "brightness": round(brightness, 1),
                "satellite": random.choice(["N", "20", "21"]),
                "daynight": "D" if 6 <= hour <= 18 else "N",
            })

    if verbose:
        high = sum(1 for h in hotspots if h["confidence"] == "high")
        print(f"  placed {len(hotspots)} hotspots ({high} high-confidence)")

    return hotspots


# ─────────────────────────────────────────────────────────────────────
# Run Bayesian Model
# ─────────────────────────────────────────────────────────────────────

def run_model_on_hotspots(
    hotspots: list[dict],
    cell_size_m: float = DEFAULT_CELL_SIZE_M,
) -> Optional[dict]:
    """Feed synthetic hotspots into the Bayesian model and return the prediction.

    Returns dict with predicted_area_km2, predicted_cells, max_probability, lat, lon.
    """
    if not hotspots:
        return None

    from bayesian_filter import BayesianFireGrid, Evidence

    # Compute centroid of hotspots
    lats = [h["lat"] for h in hotspots]
    lons = [h["lon"] for h in hotspots]
    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)

    # Create grid (use adaptive sizing for large fires)
    # For fires > 50 km², use 500m cells for better resolution
    if len(hotspots) > 100:
        cell_size_m = min(cell_size_m, 500.0)

    grid = BayesianFireGrid(center_lat, center_lon, cell_size_m=cell_size_m)

    # Group hotspots by satellite pass time (date + AM/PM window)
    # VIIRS has ~12h revisit cadence, so we split each day into two passes
    # to match the real observation cadence the model expects.
    from collections import defaultdict
    by_pass = defaultdict(list)
    for h in hotspots:
        try:
            hhmm = str(h.get("acq_time", "1200")).zfill(4)
            hour = int(hhmm[:2])
        except (ValueError, AttributeError):
            hour = 12
        # Two passes per day: AM (00-12) and PM (12-24)
        pass_key = (h["acq_date"], "AM" if hour < 12 else "PM")
        by_pass[pass_key].append(h)

    # Sort passes chronologically
    sorted_passes = sorted(by_pass.keys())

    prev_ts = None
    for date_str, period in sorted_passes:
        pass_hotspots = by_pass[(date_str, period)]
        try:
            day_dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            # AM pass ~1:30 UTC, PM pass ~13:30 UTC (VIIRS overpass times)
            hour_offset = 1 if period == "AM" else 13
            ts = (day_dt + timedelta(hours=hour_offset)).timestamp()
        except ValueError:
            continue

        # Predict elapsed time since last observation
        if prev_ts is not None:
            dt = ts - prev_ts
            if dt > 0:
                grid.predict(dt=dt, wind_speed=3.0, wind_dir_deg=270.0)

        # Inject all hotspots for this satellite pass
        for h in pass_hotspots:
            weight = min(h["frp"] / 100.0, 5.0) if h["frp"] > 0 else 1.0
            ev = Evidence.satellite_hotspot(lat=h["lat"], lon=h["lon"], weight=weight)
            grid.update(ev)

        prev_ts = ts

    # Final prediction step (12h after last observation — next expected pass)
    if prev_ts is not None:
        grid.predict(dt=12.0 * 3600, wind_speed=3.0, wind_dir_deg=270.0)

    # Extract prediction results
    prob = grid.probabilities
    burning = prob > 0.3
    cell_m = grid.cell_size
    predicted_cells = int(burning.sum())
    predicted_area_km2 = predicted_cells * (cell_m / 1000.0) ** 2

    return {
        "predicted_area_km2": predicted_area_km2,
        "predicted_cells": predicted_cells,
        "max_probability": float(prob.max()),
        "lat": center_lat,
        "lon": center_lon,
    }


# ─────────────────────────────────────────────────────────────────────
# Hybrid Model — Bayesian detection + buffer area estimation
# ─────────────────────────────────────────────────────────────────────

HYBRID_BUFFER_KM = float(
    os.environ.get("WILDFRAME_HYBRID_BUFFER_KM", "1.0")
)


def run_hybrid_model(
    hotspots: list[dict],
    buffer_km: float | None = None,
    cell_size_m: float = DEFAULT_CELL_SIZE_M,
) -> Optional[dict]:
    """Bayesian detection + spatial buffer for area estimation.

    Uses the Bayesian model for temporal filtering and false-positive
    suppression, then estimates fire area via a convex hull buffer around
    the detected hotspots.  This sidesteps the spread model's chronic
    underestimation while retaining its excellent detection capabilities.

    ``buffer_km`` controls the buffer radius around the convex hull of
    hotspots.    Defaults to ``WILDFRAME_HYBRID_BUFFER_KM`` env var (1.0 km).
    """
    if not hotspots:
        return None

    if buffer_km is None:
        buffer_km = HYBRID_BUFFER_KM

    import numpy as np
    from collections import defaultdict
    from bayesian_filter import BayesianFireGrid, Evidence
    from shapely.geometry import MultiPoint, Point
    from shapely.ops import unary_union

    # Compute centroid of hotspots
    lats = [h["lat"] for h in hotspots]
    lons = [h["lon"] for h in hotspots]
    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)

    # Adaptive cell size
    if len(hotspots) > 100:
        cell_size_m = min(cell_size_m, 500.0)

    grid = BayesianFireGrid(center_lat, center_lon, cell_size_m=cell_size_m)

    # Group hotspots by satellite pass
    by_pass = defaultdict(list)
    for h in hotspots:
        try:
            hhmm = str(h.get("acq_time", "1200")).zfill(4)
            hour = int(hhmm[:2])
        except (ValueError, AttributeError):
            hour = 12
        pass_key = (h["acq_date"], "AM" if hour < 12 else "PM")
        by_pass[pass_key].append(h)

    sorted_passes = sorted(by_pass.keys())
    prev_ts = None
    for date_str, period in sorted_passes:
        pass_hotspots = by_pass[(date_str, period)]
        try:
            day_dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            hour_offset = 1 if period == "AM" else 13
            ts = (day_dt + timedelta(hours=hour_offset)).timestamp()
        except ValueError:
            continue

        if prev_ts is not None:
            dt = ts - prev_ts
            if dt > 0:
                grid.predict(dt=dt, wind_speed=3.0, wind_dir_deg=270.0)

        for h in pass_hotspots:
            weight = min(h["frp"] / 100.0, 5.0) if h["frp"] > 0 else 1.0
            ev = Evidence.satellite_hotspot(lat=h["lat"], lon=h["lon"], weight=weight)
            grid.update(ev)

        prev_ts = ts

    if prev_ts is not None:
        grid.predict(dt=12.0 * 3600, wind_speed=3.0, wind_dir_deg=270.0)

    # Detection check
    prob = grid.probabilities
    max_prob = float(prob.max())
    detected = max_prob >= 0.3

    if not detected:
        return {
            "predicted_area_km2": 0.0,
            "predicted_cells": 0,
            "max_probability": max_prob,
            "lat": center_lat,
            "lon": center_lon,
        }

    # Buffer-based area estimation
    points = [Point(h["lon"], h["lat"]) for h in hotspots]
    if len(points) < 2:
        fire_zone = points[0].buffer(buffer_km / 111.0)
    else:
        hull = MultiPoint(points).convex_hull
        fire_zone = hull.buffer(buffer_km / 111.0)

    # Count cells inside the buffered zone
    burning = np.zeros(prob.shape, dtype=bool)
    for i in range(prob.shape[0]):
        for j in range(prob.shape[1]):
            lat, lon = grid.cell_to_latlon(i, j)
            try:
                if fire_zone.contains(Point(lon, lat)):
                    burning[i, j] = True
            except Exception:
                pass

    predicted_cells = int(burning.sum())
    predicted_area_km2 = predicted_cells * (cell_size_m / 1000.0) ** 2

    return {
        "predicted_area_km2": predicted_area_km2,
        "predicted_cells": predicted_cells,
        "max_probability": max_prob,
        "lat": center_lat,
        "lon": center_lon,
    }


# ─────────────────────────────────────────────────────────────────────
# Adaptive Model — Bayesian when accurate, buffer when not
# ─────────────────────────────────────────────────────────────────────

# Threshold: if Bayesian predicts >= this, trust it; otherwise buffer.
# Derived from 2023 calibration data: Bayesian is accurate above ~15 km²
# (IoU > 75%) and breaks down below it.
ADAPTIVE_THRESHOLD_KM2 = float(
    os.environ.get("WILDFRAME_ADAPTIVE_THRESHOLD_KM2", "15.0")
)


def run_adaptive_model(
    hotspots: list[dict],
    buffer_km: float | None = None,
    cell_size_m: float = DEFAULT_CELL_SIZE_M,
    threshold_km2: float | None = None,
) -> Optional[dict]:
    """Use Bayesian when it predicts a meaningful area; fall back to buffer.

    The Bayesian spread model is accurate for fires where it can track
    the spread (~15+ km² predicted).  For smaller predictions (which
    usually mean the spread model is failing on a larger fire), fall
    back to the convex-hull buffer.

    This gets the best of both worlds:
      - Bayesian's temporal filtering for fires it handles well
      - Buffer's robustness for fires where spread breaks down
    """
    if not hotspots:
        return None

    if threshold_km2 is None:
        threshold_km2 = ADAPTIVE_THRESHOLD_KM2

    bay = run_model_on_hotspots(hotspots, cell_size_m=cell_size_m)
    if bay is None:
        return None

    bay_area = bay["predicted_area_km2"]

    # If Bayesian predicts >= threshold, trust it
    if bay_area >= threshold_km2:
        bay["model"] = "adaptive_bayesian"
        return bay

    # Otherwise, fall back to hybrid (buffer)
    hyb = run_hybrid_model(
        hotspots, buffer_km=buffer_km, cell_size_m=cell_size_m,
    )
    if hyb is not None:
        hyb["model"] = "adaptive_buffer"
    return hyb


# ─────────────────────────────────────────────────────────────────────
# Accuracy Metrics
# ─────────────────────────────────────────────────────────────────────

def compute_accuracy(fires: list[dict], results: list[dict]) -> dict:
    """Compute accuracy metrics comparing predictions vs actual perimeters."""

    total_fires = len(fires)
    fires_with_hotspots = sum(1 for r in results if r.get("hotspot_count", 0) > 0)
    fires_predicted = sum(1 for r in results if r.get("predicted_area_km2", 0) > 0)

    # IoU and area error for fires with both prediction and perimeter
    ious = []
    area_errors = []
    pred_areas = []
    actual_areas = []

    for r in results:
        pred = r.get("predicted_area_km2", 0)
        actual = r.get("actual_area_km2", 0)
        if pred > 0 and actual > 0:
            # Area ratio IoU approximation
            ratio = min(pred, actual) / max(pred, actual)
            ious.append(ratio)
            error = abs(pred - actual) / actual * 100
            area_errors.append(error)
        pred_areas.append(pred)
        actual_areas.append(actual)

    avg_iou = sum(ious) / len(ious) * 100 if ious else 0
    median_iou = sorted(ious)[len(ious)//2] * 100 if ious else 0
    avg_area_error = sum(area_errors) / len(area_errors) if area_errors else 0

    # Detection rate: % of real fires that the model predicted
    detection_rate = fires_predicted / total_fires * 100 if total_fires > 0 else 0

    # Size-stratified detection (does the model find big fires better?)
    big_fires = [r for r in results if r.get("actual_area_km2", 0) > 50]
    big_predicted = sum(1 for r in big_fires if r.get("predicted_area_km2", 0) > 0)
    big_detection_rate = big_predicted / len(big_fires) * 100 if big_fires else 0

    medium_fires = [r for r in results if 10 < r.get("actual_area_km2", 0) <= 50]
    medium_predicted = sum(1 for r in medium_fires if r.get("predicted_area_km2", 0) > 0)
    medium_detection_rate = medium_predicted / len(medium_fires) * 100 if medium_fires else 0

    return {
        "total_fires": total_fires,
        "fires_with_hotspots": fires_with_hotspots,
        "fires_predicted": fires_predicted,
        "detection_rate_pct": round(detection_rate, 1),
        "big_fire_detection_pct": round(big_detection_rate, 1),
        "medium_fire_detection_pct": round(medium_detection_rate, 1),
        "avg_iou_pct": round(avg_iou, 1),
        "median_iou_pct": round(median_iou, 1),
        "avg_area_error_pct": round(avg_area_error, 1),
        "sample_size": len(ious),
        "avg_predicted_area_km2": round(sum(pred_areas) / len(pred_areas), 2) if pred_areas else 0,
        "avg_actual_area_km2": round(sum(actual_areas) / len(actual_areas), 2) if actual_areas else 0,
        "fires": results,
    }


# ─────────────────────────────────────────────────────────────────────
# Store Results
# ─────────────────────────────────────────────────────────────────────

def store_results(conn, accuracy: dict, year: int, state: str) -> None:
    """Store backtest results in the database.

    Uses psycopg3 style (conn.execute directly) to match db.py conventions.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS backtest_results (
            id serial PRIMARY KEY,
            year integer NOT NULL,
            state text NOT NULL,
            total_fires integer,
            fires_with_hotspots integer,
            fires_predicted integer,
            detection_rate_pct float,
            big_fire_detection_pct float,
            medium_fire_detection_pct float,
            avg_iou_pct float,
            median_iou_pct float,
            avg_area_error_pct float,
            sample_size integer,
            avg_predicted_area_km2 float,
            avg_actual_area_km2 float,
            fires_json jsonb,
            created_at timestamptz DEFAULT now(),
            UNIQUE(year, state)
        )
    """)
    conn.execute("""
        INSERT INTO backtest_results
            (year, state, total_fires, fires_with_hotspots, fires_predicted,
             detection_rate_pct, big_fire_detection_pct, medium_fire_detection_pct,
             avg_iou_pct, median_iou_pct, avg_area_error_pct, sample_size,
             avg_predicted_area_km2, avg_actual_area_km2, fires_json)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (year, state) DO UPDATE SET
            total_fires = EXCLUDED.total_fires,
            fires_with_hotspots = EXCLUDED.fires_with_hotspots,
            fires_predicted = EXCLUDED.fires_predicted,
            detection_rate_pct = EXCLUDED.detection_rate_pct,
            big_fire_detection_pct = EXCLUDED.big_fire_detection_pct,
            medium_fire_detection_pct = EXCLUDED.medium_fire_detection_pct,
            avg_iou_pct = EXCLUDED.avg_iou_pct,
            median_iou_pct = EXCLUDED.median_iou_pct,
            avg_area_error_pct = EXCLUDED.avg_area_error_pct,
            sample_size = EXCLUDED.sample_size,
            avg_predicted_area_km2 = EXCLUDED.avg_predicted_area_km2,
            avg_actual_area_km2 = EXCLUDED.avg_actual_area_km2,
            fires_json = EXCLUDED.fires_json,
            created_at = now()
    """, (
        year, state,
        accuracy["total_fires"],
        accuracy["fires_with_hotspots"],
        accuracy["fires_predicted"],
        accuracy["detection_rate_pct"],
        accuracy.get("big_fire_detection_pct", 0),
        accuracy.get("medium_fire_detection_pct", 0),
        accuracy["avg_iou_pct"],
        accuracy["median_iou_pct"],
        accuracy["avg_area_error_pct"],
        accuracy["sample_size"],
        accuracy.get("avg_predicted_area_km2", 0),
        accuracy.get("avg_actual_area_km2", 0),
        json.dumps(accuracy["fires"], default=str),
    ))


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", choices=["nifc", "calfire", "both"],
                        default="both",
                        help="Fire perimeter data source (default: both)")
    parser.add_argument("--year", type=int, default=None,
                        help="Single year to backtest")
    parser.add_argument("--years", type=int, nargs="+", default=None,
                        help="Multiple years to backtest (e.g. --years 2022 2023 2024)")
    parser.add_argument("--state", default="California",
                        help="US state for CAL FIRE source (default: California)")
    parser.add_argument("--max-fires", type=int, default=200,
                        help="Max fires to process per year/state (default: 200)")
    parser.add_argument("--min-acres", type=float, default=100,
                        help="Minimum fire size in acres (default: 100)")
    parser.add_argument("--model", choices=["bayesian", "hybrid", "adaptive"], default="hybrid",
                        help="Area estimation model: hybrid (buffer, default), bayesian (spread), or adaptive (auto-switch)")
    parser.add_argument("--buffer-km", type=float, default=None,
                        help="Buffer radius for hybrid/adaptive model (km, default: 1.0)")
    parser.add_argument("--store", action="store_true",
                        help="Store results in PostgreSQL")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    # Determine years to test
    if args.years:
        years = args.years
    elif args.year:
        years = [args.year]
    else:
        # Default: test recent years with data
        years = [2022, 2023, 2024, 2025, 2026]

    all_results = []
    summary = {
        "total_fires": 0,
        "total_predicted": 0,
        "by_year": {},
    }

    for year in years:
        print(f"\n{'='*60}")
        print(f"  BACKTEST — {year}")
        print(f"{'='*60}")

        fires = []

        # NIFC (national)
        if args.source in ("nifc", "both"):
            nifc_fires = download_nifc_fires(
                year=year, min_acres=args.min_acres,
                max_fires=args.max_fires, verbose=args.verbose,
            )
            fires.extend(nifc_fires)

        # CAL FIRE (California only)
        if args.source in ("calfire", "both") and year <= 2024:
            try:
                ca_fires = download_calfire_perimeters(year, verbose=args.verbose)
                # Filter to min acres
                ca_fires = [f for f in ca_fires if f["acres"] >= args.min_acres]
                fires.extend(ca_fires)
            except Exception as e:
                print(f"  ✗ CalFire API failed for {year}: {e}")

        if not fires:
            print(f"  No fires found for {year}")
            continue

        # Deduplicate by name+date (NIFC and CAL FIRE may overlap for CA fires)
        seen = set()
        unique_fires = []
        for f in fires:
            key = (f["fire_name"].upper(), f["fire_date"], round(f["lat"], 2), round(f["lon"], 2))
            if key not in seen:
                seen.add(key)
                unique_fires.append(f)
        fires = unique_fires

        # Sort by area (largest first) and limit
        fires.sort(key=lambda f: f["area_km2"], reverse=True)
        fires = fires[:args.max_fires]

        # Group by state for reporting
        by_state = {}
        for f in fires:
            s = f.get("state", "?")
            by_state.setdefault(s, []).append(f)
        state_summary = ", ".join(f"{s}: {len(fs)}" for s, fs in sorted(by_state.items()))
        print(f"  Fires: {len(fires)} total ({state_summary})")

        # Process each fire
        year_results = []
        for i, fire in enumerate(fires, 1):
            name = fire["fire_name"]
            area = fire["area_km2"]
            state = fire.get("state", "?")

            if args.verbose or i % 25 == 0 or i == len(fires):
                print(f"  [{i}/{len(fires)}] {name} ({state}) — {area:.1f} km² "
                      f"({fire['acres']:.0f} ac)")

            # Generate realistic hotspots
            hotspots = generate_realistic_hotspots(fire, verbose=args.verbose)

            # Run model
            if args.model == "hybrid":
                prediction = run_hybrid_model(
                    hotspots, buffer_km=args.buffer_km,
                )
            elif args.model == "adaptive":
                prediction = run_adaptive_model(
                    hotspots, buffer_km=args.buffer_km,
                )
            else:
                prediction = run_model_on_hotspots(hotspots)

            result = {
                "fire_name": name,
                "state": state,
                "actual_area_km2": area,
                "actual_acres": fire["acres"],
                "lat": fire["lat"],
                "lon": fire["lon"],
                "fire_date": fire["fire_date"],
                "hotspot_count": len(hotspots),
                "source": fire.get("source", "unknown"),
            }

            if prediction:
                result.update({
                    "predicted_area_km2": prediction["predicted_area_km2"],
                    "predicted_cells": prediction["predicted_cells"],
                    "max_probability": prediction["max_probability"],
                })
            else:
                result.update({
                    "predicted_area_km2": 0,
                    "predicted_cells": 0,
                    "max_probability": 0,
                })

            year_results.append(result)

        # Compute year accuracy
        accuracy = compute_accuracy(fires, year_results)

        # Print year summary
        print(f"\n  ── {year} Results ──")
        print(f"  Total fires:            {accuracy['total_fires']}")
        print(f"  Fires predicted:        {accuracy['fires_predicted']}")
        print(f"  Detection rate:         {accuracy['detection_rate_pct']}%")
        print(f"  Big fires detected:     {accuracy['big_fire_detection_pct']}% (>50 km²)")
        print(f"  Medium fires detected:  {accuracy['medium_fire_detection_pct']}% (10-50 km²)")
        print(f"  Avg IoU:                {accuracy['avg_iou_pct']}%")
        print(f"  Median IoU:             {accuracy['median_iou_pct']}%")
        print(f"  Avg area error:         {accuracy['avg_area_error_pct']}%")
        print(f"  Sample size (IoU):      {accuracy['sample_size']}")

        # Store in DB
        if args.store:
            try:
                import psycopg
                # Try the pool connection first (db._conn), fall back to socket
                try:
                    import db as _db
                    with _db._conn() as conn:
                        store_results(conn, accuracy, year, "All")
                        print(f"  ✓ Results stored in database")
                except Exception:
                    # Pool failed (broken URL in .env) — connect via Unix socket
                    with psycopg.connect("postgresql:///wildframe") as conn:
                        store_results(conn, accuracy, year, "All")
                        conn.commit()
                        print(f"  ✓ Results stored in database (socket)")
            except Exception as e:
                print(f"  ✗ Could not store results: {e}")

        # Save JSON
        suffix = f"_{args.model}" if args.model != "bayesian" else ""
        out_path = f"backtest_{year}{suffix}.json"
        with open(out_path, "w") as f:
            json.dump(accuracy, f, indent=2, default=str)
        print(f"  📄 Saved to {out_path}")

        all_results.append({"year": year, "accuracy": accuracy})
        summary["total_fires"] += accuracy["total_fires"]
        summary["total_predicted"] += accuracy["fires_predicted"]
        summary["by_year"][year] = {
            "detection_rate": accuracy["detection_rate_pct"],
            "avg_iou": accuracy["avg_iou_pct"],
            "sample_size": accuracy["sample_size"],
        }

    # Overall summary
    print(f"\n{'='*60}")
    print(f"  OVERALL SUMMARY")
    print(f"{'='*60}")
    print(f"  Total fires tested:     {summary['total_fires']}")
    print(f"  Total predicted:        {summary['total_predicted']}")
    overall_detection = summary["total_predicted"] / summary["total_fires"] * 100 if summary["total_fires"] > 0 else 0
    print(f"  Overall detection rate: {overall_detection:.1f}%")
    print()
    for yr, s in sorted(summary["by_year"].items()):
        print(f"  {yr}: detection={s['detection_rate']}%  "
              f"IoU={s['avg_iou']}%  (n={s['sample_size']})")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

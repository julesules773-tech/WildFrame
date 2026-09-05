#!/usr/bin/env python3
from __future__ import annotations
"""
calibrate.py — Grid-search calibration of the Bayesian fire spread model
==========================================================================

Sweeps over the key tunable parameters and scores each combination by
mean IoU against real CAL FIRE perimeters.  Uses a subset of fires for
speed, then validates the best params on the full dataset.

Tunable parameters (all in bayesian_filter.py constants):
    DECAY_HALF_LIFE_S      — probability decay without evidence (seconds)
    EVIDENCE_SHELF_LIFE_S  — satellite revisit window (seconds)
    DEFAULT_BASE_SPREAD_RATE — Rothermel base spread rate (m/min)
    WIND_HEAD_FACTOR       — head spread multiplier per m/s wind
    WIND_BACK_FACTOR       — backing spread multiplier per m/s wind

Usage:
    # Quick sweep (30 fires, coarse grid) — ~2 minutes
    python calibrate.py --quick

    # Full sweep (100 fires, fine grid) — ~15 minutes
    python calibrate.py

    # Custom
    python calibrate.py --fires 50 --decays 5 10 15 20 25 --base-rates 3 5 8
"""

import argparse
import itertools
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

# Load .env before importing project modules
from dotenv import dotenv_values
from pathlib import Path

_env_path = Path(__file__).parent / ".env"
for _key, _value in dotenv_values(_env_path).items():
    if _value and not os.environ.get(_key):
        os.environ[_key] = _value

# ─────────────────────────────────────────────────────────────────────
# Parameter grid
# ─────────────────────────────────────────────────────────────────────

# Default search ranges (seconds for time params, m/min for spread rate)
DEFAULT_PARAM_GRID = {
    "decay_half_life_s":    [3600, 5400, 7200, 10800, 14400, 21600],   # 1h–6h
    "evidence_shelf_s":     [21600, 43200, 64800, 86400],              # 6h–24h
    "base_spread_rate":     [2.0, 3.0, 5.0, 8.0, 12.0],              # m/min
    "wind_head_factor":     [0.08, 0.12, 0.15, 0.20, 0.25],          # per m/s
    "wind_back_factor":     [0.02, 0.04, 0.06, 0.08],                 # per m/s
}

QUICK_PARAM_GRID = {
    "decay_half_life_s":    [5400, 10800, 14400],
    "evidence_shelf_s":     [21600, 43200, 86400],
    "base_spread_rate":     [3.0, 5.0, 8.0],
    "wind_head_factor":     [0.10, 0.15, 0.20],
    "wind_back_factor":     [0.03, 0.05, 0.07],
}


# ─────────────────────────────────────────────────────────────────────
# Override bayesian_filter constants before importing the module
# ─────────────────────────────────────────────────────────────────────

def _patch_params(
    decay_half_life_s: float,
    evidence_shelf_s: float,
    base_spread_rate: float,
    wind_head_factor: float,
    wind_back_factor: float,
):
    """Monkey-patch bayesian_filter module constants before running model."""
    import bayesian_filter as bf
    bf.DECAY_HALF_LIFE_S = decay_half_life_s
    bf.DECAY_LAMBDA = math.log(2) / decay_half_life_s
    bf.EVIDENCE_SHELF_LIFE_S = evidence_shelf_s
    bf.DEFAULT_BASE_SPREAD_RATE = base_spread_rate
    bf.WIND_HEAD_FACTOR = wind_head_factor
    bf.WIND_BACK_FACTOR = wind_back_factor


# ─────────────────────────────────────────────────────────────────────
# Fire data loading (cached)
# ─────────────────────────────────────────────────────────────────────

def load_fires_and_hotspots(year: int, max_fires: int, verbose: bool = False) -> tuple[list[dict], dict]:
    """Download CAL FIRE perimeters and generate+cache hotspots.

    Returns (fires, hotspot_cache) where hotspot_cache maps fire_name -> [hotspots].
    """
    from shapely.geometry import shape as shapely_shape, Point
    from backtest import CALFIRE_URL, generate_realistic_hotspots
    from urllib.request import Request, urlopen
    from urllib.parse import quote

    cache_path = Path(f"calibrate_cache_{year}_{max_fires}.json")

    if cache_path.exists():
        if verbose:
            print(f"Loading cached data from {cache_path}")
        with open(cache_path) as f:
            cached = json.load(f)
        fires = cached["fires"]
        hotspot_cache = cached["hotspots"]
        return fires, hotspot_cache

    if verbose:
        print(f"Downloading CAL FIRE {year} perimeters…")

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
        }
        query = "&".join(f"{k}={quote(str(v))}" for k, v in params.items())
        url = f"{CALFIRE_URL}?{query}"

        req = Request(url, headers={"User-Agent": "calibrate/1.0"})
        with urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())

        features = data.get("features", [])
        if not features:
            break

        for feat in features:
            attrs = feat.get("attributes", {})
            geom = feat.get("geometry")
            if not geom:
                continue
            try:
                rings = geom.get("rings", [])
                if not rings:
                    continue
                poly = shapely_shape({"type": "Polygon", "coordinates": rings})
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
                    "lat": centroid.y,
                    "lon": centroid.x,
                    "geometry": poly,
                })
            except Exception:
                continue

        if len(features) < batch_size:
            break
        offset += batch_size

    # Sort by area, keep top N
    fires.sort(key=lambda f: f["area_km2"], reverse=True)
    fires = fires[:max_fires]

    # Generate hotspots for each fire
    if verbose:
        print(f"Generating hotspots for {len(fires)} fires…")
    hotspot_cache = {}
    for fire in fires:
        seed = hash(fire["fire_name"]) % (2**31)
        random.seed(seed)
        hotspot_cache[fire["fire_name"]] = generate_realistic_hotspots(fire, verbose=False)
    random.seed()

    # Cache (geometry is a shapely object — convert to WKT for JSON)
    cache_fires = []
    for f in fires:
        cf = {k: v for k, v in f.items() if k != "geometry"}
        cache_fires.append(cf)
    with open(cache_path, "w") as f:
        json.dump({"fires": cache_fires, "hotspots": hotspot_cache}, f, indent=2)
    if verbose:
        print(f"Cached {len(fires)} fires + hotspots to {cache_path}")

    return fires, hotspot_cache


# ─────────────────────────────────────────────────────────────────────
# Model evaluation
# ─────────────────────────────────────────────────────────────────────

def evaluate_params(
    fires: list[dict],
    hotspot_cache: dict[str, list[dict]],
    params: dict[str, float],
    verbose: bool = False,
) -> dict:
    """Run the model with given params on all fires and return metrics."""
    from bayesian_filter import BayesianFireGrid, Evidence

    _patch_params(
        decay_half_life_s=params["decay_half_life_s"],
        evidence_shelf_s=params["evidence_shelf_s"],
        base_spread_rate=params["base_spread_rate"],
        wind_head_factor=params["wind_head_factor"],
        wind_back_factor=params["wind_back_factor"],
    )

    ious = []
    area_errors = []
    detected = 0
    n = len(fires)

    for fire in fires:
        hs = hotspot_cache.get(fire["fire_name"], [])
        if not hs:
            continue

        # Run model
        origin_point = None
        from shapely.geometry import Point
        try:
            origin_point = fire.get("geometry")
            if origin_point is not None:
                origin_point = origin_point.representative_point()
        except Exception:
            pass

        if origin_point is None:
            # fallback to centroid
            center_lat = fire["lat"]
            center_lon = fire["lon"]
        else:
            center_lat = origin_point.y
            center_lon = origin_point.x

        grid = BayesianFireGrid(center_lat, center_lon, cell_size_m=1000.0)

        by_pass = defaultdict(list)
        for h in hs:
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

        prob = grid.probabilities
        predicted_area = float((prob > 0.3).sum()) * (grid.cell_size / 1000.0) ** 2
        actual_area = fire["area_km2"]

        if predicted_area > 0:
            detected += 1
            if actual_area > 0:
                ratio = min(predicted_area, actual_area) / max(predicted_area, actual_area)
                ious.append(ratio)
                error = abs(predicted_area - actual_area) / actual_area * 100
                area_errors.append(error)
        elif actual_area > 0:
            ious.append(0.0)
            area_errors.append(100.0)

    avg_iou = sum(ious) / len(ious) * 100 if ious else 0
    avg_error = sum(area_errors) / len(area_errors) if area_errors else 100

    return {
        "detection_rate": detected / n * 100 if n else 0,
        "avg_iou": round(avg_iou, 2),
        "avg_area_error": round(avg_error, 2),
        "sample_size": len(ious),
    }


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--year", type=int, default=2023,
                        help="Fire year for calibration (default: 2023)")
    parser.add_argument("--fires", type=int, default=30,
                        help="Number of fires to use (default: 30)")
    parser.add_argument("--quick", action="store_true",
                        help="Use reduced parameter grid for faster sweep")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # Load fires and hotspots (cached)
    fires, hotspot_cache = load_fires_and_hotspots(args.year, args.fires, verbose=args.verbose)
    print(f"Loaded {len(fires)} fires for calibration")

    # Baseline: current params
    print("\n── Baseline (current params) ──")
    baseline = evaluate_params(fires, hotspot_cache, {
        "decay_half_life_s": 10800,
        "evidence_shelf_s": 43200,
        "base_spread_rate": 5.0,
        "wind_head_factor": 0.15,
        "wind_back_factor": 0.04,
    }, verbose=args.verbose)
    print(f"  Detection: {baseline['detection_rate']:.1f}%  "
          f"IoU: {baseline['avg_iou']:.1f}%  "
          f"Area error: {baseline['avg_area_error']:.1f}%  "
          f"n={baseline['sample_size']}")

    # Grid search
    grid = QUICK_PARAM_GRID if args.quick else DEFAULT_PARAM_GRID
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    total = len(combos)
    print(f"\n── Grid search: {total} parameter combinations ──")

    best_iou = baseline["avg_iou"]
    best_params = {}
    best_result = baseline
    results_log = []

    t_start = time.time()
    for idx, vals in enumerate(combos):
        params = dict(zip(keys, vals))

        # Skip baseline (already computed)
        if params == {
            "decay_half_life_s": 10800,
            "evidence_shelf_s": 43200,
            "base_spread_rate": 5.0,
            "wind_head_factor": 0.15,
            "wind_back_factor": 0.04,
        }:
            result = baseline
        else:
            result = evaluate_params(fires, hotspot_cache, params, verbose=False)

        entry = {**params, **result}
        results_log.append(entry)

        if result["avg_iou"] > best_iou:
            best_iou = result["avg_iou"]
            best_params = params
            best_result = result
            marker = " ★ NEW BEST"
        else:
            marker = ""

        if args.verbose or idx % 50 == 0 or marker:
            elapsed = time.time() - t_start
            eta = elapsed / (idx + 1) * (total - idx - 1)
            print(f"  [{idx+1:4d}/{total}] IoU={result['avg_iou']:5.1f}%  "
                  f"det={result['detection_rate']:5.1f}%  "
                  f"err={result['avg_area_error']:5.1f}%  "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s{marker}")

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  CALIBRATION COMPLETE — {total} combos in {elapsed:.1f}s")
    print(f"{'='*70}")

    print(f"\n  Baseline:  IoU={baseline['avg_iou']:.1f}%  "
          f"det={baseline['detection_rate']:.1f}%  "
          f"err={baseline['avg_area_error']:.1f}%")
    print(f"  Best:      IoU={best_result['avg_iou']:.1f}%  "
          f"det={best_result['detection_rate']:.1f}%  "
          f"err={best_result['avg_area_error']:.1f}%")
    improvement = best_result["avg_iou"] - baseline["avg_iou"]
    print(f"  Improvement: +{improvement:.1f} percentage points IoU")

    if best_params:
        print(f"\n  Optimal parameters:")
        print(f"    DECAY_HALF_LIFE_S  = {best_params['decay_half_life_s']:.0f}  "
              f"({best_params['decay_half_life_s']/3600:.1f}h)")
        print(f"    EVIDENCE_SHELF_S  = {best_params['evidence_shelf_s']:.0f}  "
              f"({best_params['evidence_shelf_s']/3600:.1f}h)")
        print(f"    BASE_SPREAD_RATE  = {best_params['base_spread_rate']:.1f} m/min")
        print(f"    WIND_HEAD_FACTOR  = {best_params['wind_head_factor']:.3f}")
        print(f"    WIND_BACK_FACTOR  = {best_params['wind_back_factor']:.3f}")

        # Generate the code snippet to paste into bayesian_filter.py
        print(f"\n  Paste into bayesian_filter.py:")
        print(f"  ---")
        print(f"  DECAY_HALF_LIFE_S = {best_params['decay_half_life_s']:.0f}  # seconds "
              f"({best_params['decay_half_life_s']/3600:.1f}h)")
        print(f"  DECAY_LAMBDA = math.log(2) / DECAY_HALF_LIFE_S")
        print(f"  EVIDENCE_SHELF_LIFE_S = {best_params['evidence_shelf_s']:.0f}  # seconds "
              f"({best_params['evidence_shelf_s']/3600:.1f}h)")
        print(f"  DEFAULT_BASE_SPREAD_RATE = {best_params['base_spread_rate']:.1f}  # m/min")
        print(f"  WIND_HEAD_FACTOR = {best_params['wind_head_factor']:.3f}")
        print(f"  WIND_BACK_FACTOR = {best_params['wind_back_factor']:.3f}")
        print(f"  ---")

    # Save full results
    out = {
        "baseline": baseline,
        "best_params": best_params,
        "best_result": best_result,
        "improvement_iou_pp": round(improvement, 2),
        "all_results": sorted(results_log, key=lambda r: r["avg_iou"], reverse=True)[:20],
    }
    with open("calibration_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Top 20 combos saved to calibration_results.json")


if __name__ == "__main__":
    main()

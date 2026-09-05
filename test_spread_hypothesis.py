#!/usr/bin/env python3
"""
test_spread_hypothesis.py — Is the Bayesian spread model the bottleneck?
=========================================================================

Compares three area-estimation strategies on the same synthetic hotspots:

  1. **Naive**       — only cells that received a hotspot are "burning"
  2. **Buffer-N km** — every cell within N km of any hotspot is "burning"
  3. **Bayesian**    — the full BayesianFireGrid spread model (current)

If the naive or buffer models score better on IoU / area error, the spread
model is likely *hurting* rather than helping.

Usage:
    python test_spread_hypothesis.py --year 2020 --max-fires 50 --verbose
    python test_spread_hypothesis.py --all-years --max-fires 30
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.request import Request, urlopen
from urllib.parse import quote

import numpy as np

# ── Re-use backtest infrastructure ──────────────────────────────────────
from backtest import (
    download_nifc_fires,
    download_calfire_perimeters,
    generate_realistic_hotspots,
    DEFAULT_CELL_SIZE_M,
    HOTSPOT_MIN,
    HOTSPOT_MAX,
)

from shapely.geometry import Point
from shapely.ops import unary_union


# ─────────────────────────────────────────────────────────────────────────
# Model 1: Naive — hotspot cells only
# ─────────────────────────────────────────────────────────────────────────

def model_naive(
    hotspots: list[dict],
    fire_lat: float,
    fire_lon: float,
    cell_size_m: float = 500.0,
    threshold: float = 0.0,      # any hotspot → burning
) -> dict:
    """Mark every cell that received at least one hotspot as burning.
    No spread, no decay — pure detection."""
    if not hotspots:
        return _empty_result(fire_lat, fire_lon)

    from bayesian_filter import BayesianFireGrid, Evidence

    grid = BayesianFireGrid(fire_lat, fire_lon, cell_size_m=cell_size_m)

    # Just inject all hotspots at once (timing doesn't matter for naive)
    for h in hotspots:
        ev = Evidence.satellite_hotspot(lat=h["lat"], lon=h["lon"])
        grid.update(ev)

    # Any cell that received evidence → burning (p will be high)
    prob = grid.probabilities
    burning = prob > threshold
    predicted_cells = int(burning.sum())
    predicted_area_km2 = predicted_cells * (cell_size_m / 1000.0) ** 2

    return {
        "predicted_area_km2": predicted_area_km2,
        "predicted_cells": predicted_cells,
        "max_probability": float(prob.max()),
        "model": "naive",
    }


# ─────────────────────────────────────────────────────────────────────────
# Model 2: Spatial buffer — cells within N km of any hotspot
# ─────────────────────────────────────────────────────────────────────────

def model_buffer(
    hotspots: list[dict],
    fire_lat: float,
    fire_lon: float,
    buffer_km: float = 2.0,
    cell_size_m: float = 500.0,
) -> dict:
    """Every cell whose centre falls within `buffer_km` of any hotspot
    is marked as burning.  Simple spatial expansion, no temporal model."""
    if not hotspots:
        return _empty_result(fire_lat, fire_lon)

    from bayesian_filter import BayesianFireGrid

    grid = BayesianFireGrid(fire_lat, fire_lon, cell_size_m=cell_size_m)

    # Build a union of buffer circles around all hotspots
    buffer_m = buffer_km * 1000.0
    points = [Point(h["lon"], h["lat"]) for h in hotspots]
    buffers = [p.buffer(buffer_m / 111_000.0) for p in points]  # rough deg buffer
    fire_zone = unary_union(buffers)

    # Check every cell
    prob = grid.probabilities
    burning = np.zeros_like(prob, dtype=bool)
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
        "max_probability": 1.0,
        "model": f"buffer_{buffer_km}km",
    }


# ─────────────────────────────────────────────────────────────────────────
# Model 3: Current Bayesian spread (from backtest.py)
# ─────────────────────────────────────────────────────────────────────────

def model_bayesian(
    hotspots: list[dict],
    fire_lat: float,
    fire_lon: float,
    cell_size_m: float = 500.0,
) -> dict:
    """The existing Bayesian spread model, as used in backtest.py."""
    if not hotspots:
        return _empty_result(fire_lat, fire_lon)

    from bayesian_filter import BayesianFireGrid, Evidence

    grid = BayesianFireGrid(fire_lat, fire_lon, cell_size_m=cell_size_m)

    # Group by satellite pass
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

    prob = grid.probabilities
    burning = prob > 0.3
    predicted_cells = int(burning.sum())
    predicted_area_km2 = predicted_cells * (cell_size_m / 1000.0) ** 2

    return {
        "predicted_area_km2": predicted_area_km2,
        "predicted_cells": predicted_cells,
        "max_probability": float(prob.max()),
        "model": "bayesian",
    }


def _empty_result(lat, lon):
    return {
        "predicted_area_km2": 0.0,
        "predicted_cells": 0,
        "max_probability": 0.0,
        "model": "empty",
    }


# ─────────────────────────────────────────────────────────────────────────
# Model 4: Hybrid — Bayesian detection + buffer area estimation
# ─────────────────────────────────────────────────────────────────────────

def model_hybrid(
    hotspots: list[dict],
    fire_lat: float,
    fire_lon: float,
    buffer_km: float = 1.5,
    cell_size_m: float = 500.0,
    detection_threshold: float = 0.3,
) -> dict:
    """Bayesian model for detection, spatial buffer for area estimation.

    The Bayesian grid tells us *whether* a fire is present (max probability
    above threshold).  The *area* is estimated by buffering the convex hull
    of all hotspots — this sidesteps the spread model's chronic
    underestimation while retaining the Bayesian model's excellent temporal
    filtering (false-positive suppression).

    ``buffer_km`` controls how far the fire perimeter is estimated to extend
    beyond the outermost hotspots.  1.5 km is a good default: large fires
    have hotspots spread across most of the burn area, so the buffer only
    needs to cover the active front.
    """
    if not hotspots:
        return _empty_result(fire_lat, fire_lon)

    from bayesian_filter import BayesianFireGrid, Evidence
    from shapely.geometry import MultiPoint
    from shapely.ops import unary_union

    # ── Step 1: Bayesian detection ──────────────────────────────────────
    grid = BayesianFireGrid(fire_lat, fire_lon, cell_size_m=cell_size_m)

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

    prob = grid.probabilities
    max_prob = float(prob.max())
    detected = max_prob >= detection_threshold

    if not detected:
        return {
            "predicted_area_km2": 0.0,
            "predicted_cells": 0,
            "max_probability": max_prob,
            "model": f"hybrid_{buffer_km}km",
        }

    # ── Step 2: Buffer-based area estimation ─────────────────────────────
    points = [Point(h["lon"], h["lat"]) for h in hotspots]
    if len(points) < 2:
        # Single point — just buffer it
        fire_zone = points[0].buffer(buffer_km / 111.0)
    else:
        # Convex hull of all hotspots, then buffer
        hull = MultiPoint(points).convex_hull
        # Buffer in degrees (~111 km per degree at mid-latitudes)
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
        "model": f"hybrid_{buffer_km}km",
    }


# ─────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────

def compute_metrics(results: list[dict]) -> dict:
    """Compute IoU, area error, detection rate from a list of fire results."""
    total = len(results)
    detected = [r for r in results if r.get("pred_area", 0) > 0]
    detection_rate = len(detected) / total * 100 if total else 0

    ious = []
    area_errors = []
    for r in detected:
        pred = r["pred_area"]
        actual = r["actual_area"]
        if pred > 0 and actual > 0:
            ratio = min(pred, actual) / max(pred, actual)
            ious.append(ratio)
            err = abs(pred - actual) / actual * 100
            area_errors.append(err)

    avg_iou = sum(ious) / len(ious) * 100 if ious else 0
    median_iou = sorted(ious)[len(ious)//2] * 100 if ious else 0
    avg_err = sum(area_errors) / len(area_errors) if area_errors else 0
    avg_pred = sum(r["pred_area"] for r in detected) / len(detected) if detected else 0
    avg_actual = sum(r["actual_area"] for r in detected) / len(detected) if detected else 0

    return {
        "detection_pct": round(detection_rate, 1),
        "avg_iou_pct": round(avg_iou, 1),
        "median_iou_pct": round(median_iou, 1),
        "avg_area_error_pct": round(avg_err, 1),
        "sample_size": len(ious),
        "avg_predicted_km2": round(avg_pred, 2),
        "avg_actual_km2": round(avg_actual, 2),
    }


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, help="Single year to test")
    parser.add_argument("--all-years", action="store_true", help="Test 2020-2024")
    parser.add_argument("--max-fires", type=int, default=50)
    parser.add_argument("--source", default="calfire", choices=["calfire", "nifc"])
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--buffer-km", type=float, nargs="+", default=[1.0, 2.0, 5.0],
                        help="Buffer radii to test (km)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    years = [args.year] if args.year else list(range(2020, 2025))
    all_model_results = defaultdict(list)  # model_name → [fire_results]

    for year in years:
        if args.verbose:
            print(f"\n{'='*60}\n  Year {year}\n{'='*60}")

        # Download perimeters
        if args.source == "calfire":
            fires = download_calfire_perimeters(year, verbose=args.verbose)
        else:
            fires = download_nifc_fires(year=year, max_fires=args.max_fires,
                                        verbose=args.verbose)

        # Limit
        fires = sorted(fires, key=lambda f: f.get("area_km2", 0), reverse=True)
        fires = fires[:args.max_fires]

        if args.verbose:
            print(f"  Testing {len(fires)} fires")

        for fi, fire in enumerate(fires):
            if args.verbose and fi % 10 == 0:
                print(f"    [{fi+1}/{len(fires)}] {fire['fire_name']} "
                      f"({fire.get('area_km2', 0):.0f} km²)")

            # Generate hotspots (fixed seed per fire for reproducibility)
            fire_seed = args.seed + fi * 1000 + year
            random.seed(fire_seed)
            hotspots = generate_realistic_hotspots(fire, verbose=False)

            if not hotspots:
                continue

            actual_area = fire.get("area_km2", 0)
            fire_lat = fire["lat"]
            fire_lon = fire["lon"]

            # Cell size: 500m for fires > 50 km², else 1000m
            cell_m = 500.0 if actual_area > 50 else 1000.0

            # Run all models
            naive = model_naive(hotspots, fire_lat, fire_lon, cell_size_m=cell_m)
            bayesian = model_bayesian(hotspots, fire_lat, fire_lon, cell_size_m=cell_m)

            fire_base = {
                "fire_name": fire["fire_name"],
                "year": year,
                "actual_area": actual_area,
                "hotspot_count": len(hotspots),
            }

            all_model_results["naive"].append({
                **fire_base,
                "pred_area": naive["predicted_area_km2"],
            })
            all_model_results["bayesian"].append({
                **fire_base,
                "pred_area": bayesian["predicted_area_km2"],
            })

            # Buffer models
            for buf_km in args.buffer_km:
                buf = model_buffer(hotspots, fire_lat, fire_lon,
                                   buffer_km=buf_km, cell_size_m=cell_m)
                all_model_results[f"buffer_{buf_km}km"].append({
                    **fire_base,
                    "pred_area": buf["predicted_area_km2"],
                })

            # Hybrid models (Bayesian detection + buffer area)
            for buf_km in args.buffer_km:
                hyb = model_hybrid(hotspots, fire_lat, fire_lon,
                                   buffer_km=buf_km, cell_size_m=cell_m)
                all_model_results[f"hybrid_{buf_km}km"].append({
                    **fire_base,
                    "pred_area": hyb["predicted_area_km2"],
                })

    # ── Print comparison table ──────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  SPREAD MODEL HYPOTHESIS TEST — Results")
    print(f"{'='*72}\n")

    # Header
    models = sorted(all_model_results.keys())
    header = f"{'Model':<20} {'Detect%':>8} {'AvgIoU%':>9} {'MedIoU%':>9} {'AvgErr%':>9} {'n':>5} {'Pred km²':>10} {'Actual km²':>11}"
    print(header)
    print("-" * len(header))

    for model in models:
        m = compute_metrics(all_model_results[model])
        print(f"{model:<20} {m['detection_pct']:>7.1f}% {m['avg_iou_pct']:>8.1f}% "
              f"{m['median_iou_pct']:>8.1f}% {m['avg_area_error_pct']:>8.1f}% "
              f"{m['sample_size']:>5} {m['avg_predicted_km2']:>10.2f} {m['avg_actual_km2']:>11.2f}")

    # ── Per-fire detail for a few sample fires ──────────────────────────
    print(f"\n{'='*72}")
    print("  Sample fires (top 10 by area)")
    print(f"{'='*72}\n")

    # Get unique fires sorted by area
    seen = set()
    sample_fires = []
    for r in all_model_results["bayesian"]:
        key = (r["fire_name"], r["year"])
        if key not in seen:
            seen.add(key)
            sample_fires.append(r)
    sample_fires.sort(key=lambda x: x["actual_area"], reverse=True)

    for fire in sample_fires[:10]:
        print(f"\n  {fire['fire_name']} ({fire['year']}) — {fire['actual_area']:.0f} km² actual, "
              f"{fire['hotspot_count']} hotspots")
        for model in models:
            # Find this fire in model results
            for r in all_model_results[model]:
                if r["fire_name"] == fire["fire_name"] and r["year"] == fire["year"]:
                    ratio = r["pred_area"] / fire["actual_area"] * 100 if fire["actual_area"] > 0 else 0
                    print(f"    {model:<20} {r['pred_area']:>8.2f} km²  ({ratio:>5.1f}% of actual)")
                    break

    # ── Verdict ─────────────────────────────────────────────────────────
    naive_m = compute_metrics(all_model_results["naive"])
    bayesian_m = compute_metrics(all_model_results["bayesian"])

    print(f"\n{'='*72}")
    print("  VERDICT")
    print(f"{'='*72}\n")

    iou_diff = bayesian_m["avg_iou_pct"] - naive_m["avg_iou_pct"]
    err_diff = bayesian_m["avg_area_error_pct"] - naive_m["avg_area_error_pct"]

    if iou_diff > 2:
        print(f"  ✅ Bayesian spread HELPS: +{iou_diff:.1f}pp IoU over naive")
    elif iou_diff < -2:
        print(f"  ❌ Bayesian spread HURTS: {iou_diff:.1f}pp IoU vs naive")
    else:
        print(f"  ⚖️  Bayesian spread is ~neutral: {iou_diff:+.1f}pp IoU vs naive")

    if err_diff < -5:
        print(f"  ✅ Bayesian spread reduces area error by {-err_diff:.1f}pp")
    elif err_diff > 5:
        print(f"  ❌ Bayesian spread increases area error by {err_diff:.1f}pp")

    print(f"\n  Naive avg prediction: {naive_m['avg_predicted_km2']:.2f} km²")
    print(f"  Bayesian avg prediction: {bayesian_m['avg_predicted_km2']:.2f} km²")
    print(f"  Actual avg: {naive_m['avg_actual_km2']:.2f} km²")

    # Save raw results
    output = {
        "summary": {m: compute_metrics(all_model_results[m]) for m in models},
        "per_fire": {m: all_model_results[m] for m in models},
    }
    with open("spread_hypothesis_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Raw results saved to spread_hypothesis_results.json")


if __name__ == "__main__":
    main()

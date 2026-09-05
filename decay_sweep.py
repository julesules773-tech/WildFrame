#!/usr/bin/env python3
"""Sweep DECAY_HALF_LIFE × BASE_SPREAD_RATE grid search."""

import sys, os, json, time, argparse, math, random
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

import shapely
from shapely.geometry import shape
from backtest import download_calfire_perimeters, generate_realistic_hotspots

# ── Model run ──

def run_model(hotspots, actual_area_km2, decay_half_life_s, base_spread):
    import bayesian_filter as bf
    from bayesian_filter import BayesianFireGrid, Evidence
    from datetime import timezone

    # Patch module-level constants
    bf.DECAY_HALF_LIFE_S = decay_half_life_s
    bf.DECAY_LAMBDA = math.log(2) / decay_half_life_s
    bf.DEFAULT_BASE_SPREAD_RATE = base_spread

    center_lat = sum(h["lat"] for h in hotspots) / len(hotspots)
    center_lon = sum(h["lon"] for h in hotspots) / len(hotspots)

    cell_size_m = 1000.0
    if actual_area_km2 > 50:
        cell_size_m = 500.0

    grid = BayesianFireGrid(center_lat, center_lon, cell_size_m=cell_size_m)

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
    cell_m = grid.cell_size
    predicted_cells = int(burning.sum())
    predicted_area_km2 = predicted_cells * (cell_m / 1000.0) ** 2

    return predicted_area_km2


def score(fires, decay_s, base_spread):
    det = 0
    ious = []
    area_errors = []
    pred_areas = []

    for fire in fires:
        hs = fire.get("hotspots", [])
        if not hs:
            continue
        pred_area = run_model(hs, fire["area_km2"], decay_s, base_spread)
        actual = fire["area_km2"]

        pred_areas.append(pred_area)
        if pred_area > 0:
            det += 1
            ratio = min(pred_area, actual) / max(pred_area, actual)
            ious.append(ratio)
            area_errors.append(abs(pred_area - actual) / actual * 100)

    n = len([f for f in fires if f.get("hotspots")])
    det_rate = det / n * 100 if n else 0
    avg_iou = sum(ious) / len(ious) * 100 if ious else 0
    avg_area_err = sum(area_errors) / len(area_errors) if area_errors else 0
    avg_pred = sum(pred_areas) / len(pred_areas) if pred_areas else 0

    return det_rate, avg_iou, avg_area_err, n, avg_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=30)
    parser.add_argument("--year", type=int, default=2024)
    args = parser.parse_args()

    random.seed(42)

    # Download fires
    print(f"Downloading fires from {args.year}...")
    fires = download_calfire_perimeters(args.year, verbose=True)
    fires.sort(key=lambda f: f["area_km2"], reverse=True)
    fires = fires[:args.sample]

    if not fires:
        print("No fires downloaded!")
        sys.exit(1)

    print(f"\nGenerating hotspots for {len(fires)} fires...")
    for fire in fires:
        fire["hotspots"] = generate_realistic_hotspots(fire)
        n_hs = len(fire["hotspots"])
        if n_hs > 0:
            print(f"  {fire['fire_name']:25s} {fire['area_km2']:7.1f} km²  {n_hs} hotspots")

    # ── Grid search: decay × spread ──
    decays = [3600, 7200, 10800, 14400, 21600, 43200, 86400]
    spreads = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0]

    avg_actual = sum(f["area_km2"] for f in fires) / len(fires)
    print(f"\nAverage actual fire size: {avg_actual:.1f} km²")
    print(f"\nGrid: {len(decays)} decay values × {len(spreads)} spread rates = {len(decays)*len(spreads)} combos\n")

    print(f"{'Decay':>7} {'Spread':>7} │ {'Det%':>6} {'IoU%':>7} {'AreaErr%':>9} {'AvgPred':>8}")
    print("─" * 65)

    all_results = []
    for ds in decays:
        for sp in spreads:
            t0 = time.time()
            det, iou, aerr, n, avg_pred = score(fires, ds, sp)
            elapsed = time.time() - t0
            marker = " ←" if ds == 14400 and sp == 3.0 else ""
            print(f"{ds/3600:>6.1f}h {sp:>6.1f}m │ {det:>5.1f}% {iou:>6.1f}% {aerr:>8.1f}% {avg_pred:>7.1f}  {elapsed:.1f}s{marker}")
            all_results.append({"decay_h": ds/3600, "spread": sp, "det": det, "iou": iou, "area_err": aerr, "avg_pred": avg_pred})

    # Find best combos
    best_iou = max(all_results, key=lambda r: r["iou"])
    best_aerr = min(all_results, key=lambda r: r["area_err"])

    # Best balanced: max IoU with AreaErr < 60%
    reasonable = [r for r in all_results if r["area_err"] < 60]
    best_balanced = max(reasonable, key=lambda r: r["iou"]) if reasonable else best_iou

    print(f"\n{'='*65}")
    print(f"Best by IoU:       {best_iou['decay_h']:.0f}h decay, {best_iou['spread']:.0f} m/min spread → IoU {best_iou['iou']:.1f}%, AreaErr {best_iou['area_err']:.1f}%")
    print(f"Best by AreaErr:   {best_aerr['decay_h']:.0f}h decay, {best_aerr['spread']:.0f} m/min spread → IoU {best_aerr['iou']:.1f}%, AreaErr {best_aerr['area_err']:.1f}%")
    if best_balanced != best_iou:
        print(f"Best balanced:     {best_balanced['decay_h']:.0f}h decay, {best_balanced['spread']:.0f} m/min spread → IoU {best_balanced['iou']:.1f}%, AreaErr {best_balanced['area_err']:.1f}%")


if __name__ == "__main__":
    main()

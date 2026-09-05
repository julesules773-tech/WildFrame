#!/usr/bin/env python3
"""Sweep mass transfer cap values at 24h decay using cached fires."""

import subprocess, sys, json, re, time
from pathlib import Path

BF_PATH = Path(__file__).parent / "bayesian_filter.py"

caps = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.75]
decay = 86400

original = BF_PATH.read_text()
pattern = r"fraction_stencil = [\d.]+ \* stencil / mass_transferred"
match = re.search(pattern, original)
if not match:
    print("ERROR: Could not find fraction_stencil line")
    sys.exit(1)
print(f"Found: {match.group()}")
print(f"\nCap sweep: {len(caps)} values × 30 cached fires, decay={decay}s (24h)")
print(f"\n{'Cap':>6} │ {'Det%':>6} {'IoU%':>7} {'AreaErr%':>9} {'AvgPred':>8} {'Time':>6}")
print("─" * 60)

for cap in caps:
    new_source = re.sub(pattern, f"fraction_stencil = {cap} * stencil / mass_transferred", original)
    BF_PATH.write_text(new_source)

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "-c", f"""
import json, random, math
from datetime import datetime, timedelta
from collections import defaultdict
from shapely import wkt as shapely_wkt
import numpy as np

# Load cached fires
with open("cap_sweep_cache.json") as f:
    cached = json.load(f)
fires = cached["fires"]

# Restore geometry from WKT strings
for fire in fires:
    fire["geometry"] = shapely_wkt.loads(fire["geometry"])

import bayesian_filter as bf
bf.DECAY_HALF_LIFE_S = {decay}
bf.DECAY_LAMBDA = math.log(2) / {decay}

from bayesian_filter import BayesianFireGrid, Evidence
from datetime import timezone

det = 0; ious = []; area_errors = []; pred_areas = []

for fire in fires:
    hs = fire.get("hotspots", [])
    if not hs:
        continue

    center_lat = sum(h["lat"] for h in hs) / len(hs)
    center_lon = sum(h["lon"] for h in hs) / len(hs)
    cell_size_m = 500.0 if fire["area_km2"] > 50 else 1000.0
    grid = BayesianFireGrid(center_lat, center_lon, cell_size_m=cell_size_m)

    by_pass = defaultdict(list)
    for h in hs:
        try:
            hhmm = str(h.get("acq_time", "1200")).zfill(4)
            hour = int(hhmm[:2])
        except: hour = 12
        by_pass[(h["acq_date"], "AM" if hour < 12 else "PM")].append(h)

    sorted_passes = sorted(by_pass.keys())
    prev_ts = None
    for date_str, period in sorted_passes:
        pass_hs = by_pass[(date_str, period)]
        try:
            day_dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            ts = (day_dt + timedelta(hours=(1 if period=="AM" else 13))).timestamp()
        except: continue
        if prev_ts is not None and ts > prev_ts:
            grid.predict(dt=ts-prev_ts, wind_speed=3.0, wind_dir_deg=270.0)
        for h in pass_hs:
            w = min(h["frp"]/100.0, 5.0) if h["frp"] > 0 else 1.0
            grid.update(Evidence.satellite_hotspot(lat=h["lat"], lon=h["lon"], weight=w))
        prev_ts = ts
    if prev_ts:
        grid.predict(dt=12*3600, wind_speed=3.0, wind_dir_deg=270.0)

    prob = grid.probabilities
    burning = prob > 0.3
    pred_area = int(burning.sum()) * (grid.cell_size/1000.0)**2
    actual = fire["area_km2"]
    pred_areas.append(pred_area)
    if pred_area > 0:
        det += 1
        ious.append(min(pred_area, actual)/max(pred_area, actual))
        area_errors.append(abs(pred_area-actual)/actual*100)

n = len([f for f in fires if f.get("hotspots")])
det_rate = det/n*100 if n else 0
avg_iou = sum(ious)/len(ious)*100 if ious else 0
avg_aerr = sum(area_errors)/len(area_errors) if area_errors else 0
avg_pred = sum(pred_areas)/len(pred_areas) if pred_areas else 0

print(json.dumps({{"det":det_rate,"iou":avg_iou,"aerr":avg_aerr,"pred":avg_pred}}))
"""],
        capture_output=True, text=True, timeout=120
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  {cap:.0%} │ ERROR: {result.stderr[:300]}")
        continue

    try:
        data = json.loads(result.stdout.strip().split("\n")[-1])
    except:
        print(f"  {cap:.0%} │ PARSE ERROR: stdout={result.stdout[:200]}")
        continue

    marker = " ← current" if cap == 0.30 else ""
    print(f"{cap:>5.0%} │ {data['det']:>5.1f}% {data['iou']:>6.1f}% {data['aerr']:>8.1f}% {data['pred']:>7.1f}  {elapsed:.1f}s{marker}")

BF_PATH.write_text(original)
print(f"\nRestored bayesian_filter.py (cap=30%)")

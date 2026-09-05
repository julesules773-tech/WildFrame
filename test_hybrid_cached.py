#!/usr/bin/env python3
"""Run hybrid vs bayesian on cached hotspot data — no network needed."""
import json
import sys
sys.path.insert(0, ".")

from backtest import run_model_on_hotspots, run_hybrid_model

cache = json.load(open("calibrate_cache_2023_15.json"))
fires = cache["fires"]
hotspots_cache = cache["hotspots"]

print(f"{'Fire':<25} {'Actual':>8}  {'Bayesian':>10} {'Hybrid':>10}  {'Bay IoU':>8} {'Hyb IoU':>8}")
print("-" * 95)

bay_ious, hyb_ious = [], []
bay_errs, hyb_errs = [], []

for fire in fires:
    name = fire["fire_name"]
    actual = fire["area_km2"]
    hs = hotspots_cache.get(name, [])

    bay = run_model_on_hotspots(hs) if hs else None
    hyb = run_hybrid_model(hs) if hs else None

    bay_area = bay["predicted_area_km2"] if bay else 0
    hyb_area = hyb["predicted_area_km2"] if hyb else 0

    def iou(pred, act):
        if pred > 0 and act > 0:
            return min(pred, act) / max(pred, act) * 100
        return 0

    bay_iou = iou(bay_area, actual)
    hyb_iou = iou(hyb_area, actual)

    if bay_area > 0 and actual > 0:
        bay_ious.append(bay_iou)
        bay_errs.append(abs(bay_area - actual) / actual * 100)
    if hyb_area > 0 and actual > 0:
        hyb_ious.append(hyb_iou)
        hyb_errs.append(abs(hyb_area - actual) / actual * 100)

    print(f"{name:<25} {actual:>7.1f}  {bay_area:>9.1f} {hyb_area:>9.1f}  {bay_iou:>7.1f}% {hyb_iou:>7.1f}%")

print("-" * 95)
bay_avg_iou = sum(bay_ious)/len(bay_ious) if bay_ious else 0
hyb_avg_iou = sum(hyb_ious)/len(hyb_ious) if hyb_ious else 0
bay_avg_err = sum(bay_errs)/len(bay_errs) if bay_errs else 0
hyb_avg_err = sum(hyb_errs)/len(hyb_errs) if hyb_errs else 0

print(f"{'AVERAGE':<25} {'':>8}  {'':>10} {'':>10}  {bay_avg_iou:>7.1f}% {hyb_avg_iou:>7.1f}%")
print(f"{'AREA ERROR':<25} {'':>8}  {'':>10} {'':>10}  {bay_avg_err:>7.1f}% {hyb_avg_err:>7.1f}%")
print(f"\nSample size: bayesian={len(bay_ious)}, hybrid={len(hyb_ious)}")

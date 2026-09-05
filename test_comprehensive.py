#!/usr/bin/env python3
"""Comprehensive model comparison on all available cached fire data."""
import json, sys, random
sys.path.insert(0, '.')
from backtest import (
    run_model_on_hotspots, run_hybrid_model, run_adaptive_model,
)

# Load cached data
cache15 = json.load(open('calibrate_cache_2023_15.json'))
cap30 = json.load(open('cap_sweep_cache.json'))

# Build unified fire list
fires = []
for f in cache15['fires']:
    hs = cache15['hotspots'].get(f['fire_name'], [])
    if hs:
        fires.append({'name': f['fire_name'], 'area': f['area_km2'], 'hotspots': hs, 'source': 'cache2023'})

for f in cap30['fires']:
    hs = f.get('hotspots', [])
    if hs:
        fires.append({'name': f['fire_name'], 'area': f['area_km2'], 'hotspots': hs, 'source': 'cap_sweep'})

fires.sort(key=lambda x: x['area'], reverse=True)

# Run all models
hdr = '{:<22} {:>7} {:>8} {:>8} {:>8}  {:>5} {:>5} {:>5}'.format(
    'Fire', 'Actual', 'Bayesian', 'Hybrid', 'Adaptive', 'Bay%', 'Hyb%', 'Ada%')
print(hdr)
print('-' * 85)

bay_ious, hyb_ious, ada_ious = [], [], []
bay_errs, hyb_errs, ada_errs = [], [], []

for f in fires:
    actual = f['area']
    hs = f['hotspots']

    bay = run_model_on_hotspots(hs) if hs else None
    hyb = run_hybrid_model(hs) if hs else None
    ada = run_adaptive_model(hs) if hs else None

    bp = bay['predicted_area_km2'] if bay else 0
    hp = hyb['predicted_area_km2'] if hyb else 0
    ap = ada['predicted_area_km2'] if ada else 0

    def iou(p, a):
        return min(p, a) / max(p, a) * 100 if p > 0 and a > 0 else 0

    bi, hi, ai = iou(bp, actual), iou(hp, actual), iou(ap, actual)

    if bp > 0 and actual > 0:
        bay_ious.append(bi); bay_errs.append(abs(bp - actual) / actual * 100)
    if hp > 0 and actual > 0:
        hyb_ious.append(hi); hyb_errs.append(abs(hp - actual) / actual * 100)
    if ap > 0 and actual > 0:
        ada_ious.append(ai); ada_errs.append(abs(ap - actual) / actual * 100)

    # Show which adaptive chose
    src = ada.get('model', '?') if ada else '?'
    tag = 'B' if 'bayesian' in src else ('H' if 'buffer' in src else '-')

    print('{:<22} {:>7.1f} {:>8.1f} {:>8.1f} {:>8.1f}  {:>4.1f}% {:>4.1f}% {:>4.1f}% {}'.format(
        f['name'], actual, bp, hp, ap, bi, hi, ai, tag))

print('-' * 85)

def avg(lst): return sum(lst)/len(lst) if lst else 0

bay_mi = sorted(bay_ious)[len(bay_ious)//2] if bay_ious else 0
hyb_mi = sorted(hyb_ious)[len(hyb_ious)//2] if hyb_ious else 0
ada_mi = sorted(ada_ious)[len(ada_ious)//2] if ada_ious else 0

print('{:<22} {:>7} {:>8} {:>8} {:>8}  {:>4.1f}% {:>4.1f}% {:>4.1f}%'.format(
    'AVG IoU', '', '', '', '', avg(bay_ious), avg(hyb_ious), avg(ada_ious)))
print('{:<22} {:>7} {:>8} {:>8} {:>8}  {:>4.1f}% {:>4.1f}% {:>4.1f}%'.format(
    'MEDIAN IoU', '', '', '', '', bay_mi, hyb_mi, ada_mi))
print('{:<22} {:>7} {:>8} {:>8} {:>8}  {:>4.1f}% {:>4.1f}% {:>4.1f}%'.format(
    'AVG AREA ERROR', '', '', '', '', avg(bay_errs), avg(hyb_errs), avg(ada_errs)))
print()
print('n: bayesian={}, hybrid={}, adaptive={}'.format(len(bay_ious), len(hyb_ious), len(ada_ious)))

# Threshold sensitivity analysis
print()
print('=== THRESHOLD SENSITIVITY (adaptive IoU by threshold) ===')
for t in [5, 10, 15, 20, 25, 30, 40, 50]:
    test_ious = []
    for f in fires:
        actual = f['area']
        hs = f['hotspots']
        bay = run_model_on_hotspots(hs) if hs else None
        if bay and bay['predicted_area_km2'] >= t:
            bp = bay['predicted_area_km2']
        else:
            hyb = run_hybrid_model(hs) if hs else None
            bp = hyb['predicted_area_km2'] if hyb else 0
        if bp > 0 and actual > 0:
            test_ious.append(min(bp, actual)/max(bp, actual)*100)
    mi = sorted(test_ious)[len(test_ious)//2] if test_ious else 0
    print('  threshold={:>2} km²: avg_iou={:.1f}%  median_iou={:.1f}%  n={}'.format(
        t, avg(test_ious), mi, len(test_ious)))

#!/usr/bin/env python3
"""Compare all models on cached 2023 data."""
import json, sys
sys.path.insert(0, '.')
from backtest import run_model_on_hotspots, run_hybrid_model, run_adaptive_model

cache = json.load(open('calibrate_cache_2023_15.json'))
fires, hs_cache = cache['fires'], cache['hotspots']

hdr = '{:<20} {:>7} {:>8} {:>8} {:>8}  {:>6} {:>6} {:>6}'.format(
    'Fire', 'Actual', 'Bayesian', 'Hybrid', 'Adaptive', 'Bay%', 'Hyb%', 'Ada%')
print(hdr)
print('-' * 85)

bay_ious, hyb_ious, ada_ious = [], [], []
bay_errs, hyb_errs, ada_errs = [], [], []

for f in fires:
    name = f['fire_name']
    actual = f['area_km2']
    hs = hs_cache.get(name, [])

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

    src = getattr(ada, 'get', lambda k, d: d)('model', '?') if ada else '?'
    print('{:<20} {:>7.1f} {:>8.1f} {:>8.1f} {:>8.1f}  {:>5.1f}% {:>5.1f}% {:>5.1f}%  {}'.format(
        name, actual, bp, hp, ap, bi, hi, ai, src))

print('-' * 85)

def avg(lst): return sum(lst)/len(lst) if lst else 0

bay_mi = sorted(bay_ious)[len(bay_ious)//2] if bay_ious else 0
hyb_mi = sorted(hyb_ious)[len(hyb_ious)//2] if hyb_ious else 0
ada_mi = sorted(ada_ious)[len(ada_ious)//2] if ada_ious else 0

print('{:<20} {:>7} {:>8} {:>8} {:>8}  {:>5.1f}% {:>5.1f}% {:>5.1f}%'.format(
    'AVG IoU', '', '', '', '', avg(bay_ious), avg(hyb_ious), avg(ada_ious)))
print('{:<20} {:>7} {:>8} {:>8} {:>8}  {:>5.1f}% {:>5.1f}% {:>5.1f}%'.format(
    'MEDIAN IoU', '', '', '', '', bay_mi, hyb_mi, ada_mi))
print('{:<20} {:>7} {:>8} {:>8} {:>8}  {:>5.1f}% {:>5.1f}% {:>5.1f}%'.format(
    'AVG AREA ERROR', '', '', '', '', avg(bay_errs), avg(hyb_errs), avg(ada_errs)))
print('\nn: bayesian={}, hybrid={}, adaptive={}'.format(len(bay_ious), len(hyb_ious), len(ada_ious)))

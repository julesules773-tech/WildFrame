#!/usr/bin/env python3
"""
backtest_grids.py — Grid-level spread backtest vs observed FIRMS hotspots
=========================================================================

Validates the Bayesian spread model the same way a forecast is scored:
for each production grid, fetch the fire's FULL hotspot history from the
NASA FIRMS API, split it in time at the median, replay the first half
through the actual model (fresh grid + ``Evidence.satellite_hotspot``
updates + ``predict()`` chunks, exactly like production's advance job),
and ask: *at the cutoff, does the predicted probability field cover the
hotspots observed in the second half?*

Why re-fetch instead of reading stored state
--------------------------------------------
The DB keeps only the grid's CURRENT probability field — historical
hotspot observations are fused into it and discarded, so there is no
stored ground-truth timeline to compare against. FIRMS retains ~5-7 days
of detections (VIIRS NRT), and grids die after 24 h without evidence, so
the API gives us the complete observed history of every live fire.

Metrics per grid (evaluated at the median-time cutoff)
------------------------------------------------------
- hit@t        fraction of TEST hotspots inside a cell with P >= t
               (t = 0.05 / 0.10 / 0.20 / 0.30) — "did the model put the
               fire where it actually went?"
- dist_km      mean distance from test hotspots to the nearest cell with
               P >= 0.05 (0 for hotspots already covered)
- area_pred    predicted extent area (km²) at P >= 0.05
- area_obs     observed footprint (km²): convex hull of ALL hotspots
- move_km      distance between the train centroid and test centroid —
               how far the fire actually drifted in the test half
- pers_hit     PERSISTENCE BASELINE: fraction of test hotspots within
               1 km (configurable) of any TRAIN hotspot — "the fire
               didn't move" naive forecast. The model is useful only if
               its hit rate beats this.

Known limitations (read before trusting numbers)
------------------------------------------------
1. The model is replayed with the grid's CURRENT stored wind + EFFIS
   moisture (the DB keeps no per-hour wind history), so multi-day windows
   assume steady wind.
2. The replay injects each detection EXACTLY ONCE. Production re-fetches
   the rolling last-24 h of detections every ~10 min and re-injects them,
   which keeps live probabilities pinned near 1.0. So the numbers here
   are the honest "evidence arrives exactly once" forecast skill — if the
   extent collapses between passes, that is a REAL finding (decay
   half-life 3 h vs VIIRS revisit ~12-24 h), not a harness artifact.
3. FIRMS bbox requests are capped at ~2° per side; larger grids are
   skipped.
4. Hotspot timestamps are the satellite pass times (acq_time, UTC), not
   ground-fire times — the standard FIRMS caveat.
5. One fire can be tracked by several grids (the 10 km grid-match radius
   splits large fires), so near-identical per-grid rows in the sample are
   the same fire counted twice — dedupe by centroid before drawing
   population conclusions.

Usage
-----
    .venv/bin/python backtest_grids.py                    # 20 newest grids, 5 days
    .venv/bin/python backtest_grids.py --limit 50 --days 3
    .venv/bin/python backtest_grids.py --json bt.json --quiet
    .venv/bin/python backtest_grids.py --mode production --min-confidence nominal

Needs NASA_FIRMS_API_KEY in .env (same key the worker uses).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# --- Load .env BEFORE importing db (db reads env vars at import time) ---
# Mirror worker.py's fill-in pattern: dotenv refuses to override existing
# shell vars — including EMPTY ones — so only missing/empty keys are filled.
from dotenv import dotenv_values

_env_path = Path(__file__).parent / ".env"
for _key, _value in dotenv_values(_env_path).items():
    if _value and not os.environ.get(_key):
        os.environ[_key] = _value

import db
import nasa_firms
from bayesian_filter import BayesianFireGrid, Evidence, equirectangular_unproject

# predict() chunk cap — matches production (db.advance_grids caps dt at 600 s)
DEFAULT_CHUNK_S = 600.0
# 0.02 = "any trace above background (0.01)", 0.05 = visible on the map,
# 0.10 / 0.30 = confident extent
THRESHOLDS = (0.02, 0.05, 0.10, 0.30)
# FIRMS area requests: skip grids whose bbox exceeds this per-side cap
MAX_BBOX_DEG = 2.0

R_EARTH = 6_371_000.0
DEG_KM = 111.32  # km per degree (equirectangular approximation for hull area)

_BOLD = "\033[1m"
_RESET = "\033[0m"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_candidate_grids(mode: str, limit: int) -> list[dict]:
    """Newest evidence-bearing grids (guaranteed inside the FIRMS NRT
    window). Entry dicts carry the grid GEOMETRY read straight from the
    state JSONB (nx/ny/cell_size/center are plain fields) — no numpy
    deserialization, since the backtest builds its own fresh replay grid
    anyway."""
    with db._conn() as conn:
        rows = conn.execute(
            "SELECT id, centroid_lat, centroid_lon, wind_speed, wind_dir_deg, "
            "ffmc, dmc, isi, max_p, last_evidence_at, state "
            "FROM bayesian_grids "
            "WHERE mode = %s AND last_evidence_at > 0 "
            "ORDER BY last_evidence_at DESC, max_p DESC "
            "LIMIT %s",
            (mode, limit),
        ).fetchall()
    out = []
    for r in rows:
        st = r["state"] or {}
        out.append({
            "id": r["id"],
            "centroid_lat": r["centroid_lat"],
            "centroid_lon": r["centroid_lon"],
            "wind_speed": r["wind_speed"],
            "wind_dir_deg": r["wind_dir_deg"],
            "ffmc": r["ffmc"],
            "dmc": r["dmc"],
            "isi": r["isi"],
            "max_p": r["max_p"],
            "last_evidence_at": r["last_evidence_at"],
            # geometry lives in the state blob (plain JSON fields)
            "g_nx": int(st.get("nx") or 0),
            "g_ny": int(st.get("ny") or 0),
            "g_cell_size_m": float(st.get("cell_size_m") or 0.0),
            "g_center_lat": float(st.get("center_lat") or r["centroid_lat"]),
            "g_center_lon": float(st.get("center_lon") or r["centroid_lon"]),
        })
    return out


def grid_bbox(entry: dict, margin_m: float = 2000.0) -> tuple[float, float, float, float]:
    """(west, south, east, north) bbox covering the grid + a little slack."""
    hx = entry["g_nx"] * entry["g_cell_size_m"] / 2.0 + margin_m
    hy = entry["g_ny"] * entry["g_cell_size_m"] / 2.0 + margin_m
    sw_lat, sw_lon = equirectangular_unproject(
        -hx, -hy, entry["g_center_lat"], entry["g_center_lon"]
    )
    ne_lat, ne_lon = equirectangular_unproject(
        hx, hy, entry["g_center_lat"], entry["g_center_lon"]
    )
    return sw_lon, sw_lat, ne_lon, ne_lat


# ---------------------------------------------------------------------------
# FIRMS history
# ---------------------------------------------------------------------------

def fetch_history(
    api_key: str,
    entry: dict,
    days: int,
    min_confidence: str,
) -> list[tuple[float, nasa_firms.FIRMSHotspot]]:
    """Hotspots in the grid's bbox over the last ``days``, sorted by
    acquisition time. Raises ValueError for oversized grids."""
    w, s, e, n = grid_bbox(entry)
    if (e - w) > MAX_BBOX_DEG or (n - s) > MAX_BBOX_DEG:
        raise ValueError(
            f"grid extent {(e-w):.1f}x{(n-s):.1f}° exceeds FIRMS bbox cap {MAX_BBOX_DEG}°"
        )
    hotspots = nasa_firms.fetch_fire_data(api_key, (w, s, e, n), day_range=days)
    if min_confidence == "high":
        hotspots = [h for h in hotspots if h.is_high_confidence]
    elif min_confidence == "nominal":
        hotspots = [h for h in hotspots if h.is_nominal_or_higher]

    out: list[tuple[float, nasa_firms.FIRMSHotspot]] = []
    for h in hotspots:
        ts = h.acquired_at
        if ts is None:
            continue
        out.append((ts.timestamp(), h))
    out.sort(key=lambda x: x[0])
    return out


def split_history(
    history: list[tuple[float, nasa_firms.FIRMSHotspot]],
) -> tuple[list[nasa_firms.FIRMSHotspot], list[nasa_firms.FIRMSHotspot], float]:
    """Median-time split: first half trains the model, second half is the
    observed ground truth. Returns (train, test, cutoff_ts)."""
    cutoff = history[len(history) // 2][0]
    train = [h for t, h in history if t <= cutoff]
    test = [h for t, h in history if t > cutoff]
    return train, test, cutoff


# ---------------------------------------------------------------------------
# Model replay
# ---------------------------------------------------------------------------

def _predict_chunked(
    g: BayesianFireGrid,
    dt: float,
    wind_speed: float,
    wind_dir_deg: float,
    moisture_factor: float,
    decay_scale: float,
    chunk_s: float,
) -> None:
    remaining = dt
    while remaining > 0:
        step = min(remaining, chunk_s)
        g.predict(
            dt=step,
            wind_speed=wind_speed,
            wind_dir_deg=wind_dir_deg,
            moisture_factor=moisture_factor,
            decay_scale=decay_scale,
        )
        remaining -= step


def rebuild(
    entry: dict,
    train: list[nasa_firms.FIRMSHotspot],
    cutoff_ts: float,
    chunk_s: float,
) -> BayesianFireGrid:
    """Replay the training hotspots through a fresh grid, predicting
    elapsed time between detections (and to the cutoff) in <= chunk_s
    chunks — the same advance semantics as production."""
    from effis_fwi import decay_scale, moisture_factor  # lazy: effis imports db

    sim = BayesianFireGrid(
        center_lat=entry["g_center_lat"],
        center_lon=entry["g_center_lon"],
        cell_size_m=entry["g_cell_size_m"],
        nx=entry["g_nx"],
        ny=entry["g_ny"],
    )
    ffmc = float(entry.get("ffmc") or 0.0)
    dmc = float(entry.get("dmc") or 0.0)
    mf = moisture_factor(ffmc) if ffmc > 0 else 1.0
    ds = decay_scale(dmc) if dmc > 0 else 1.0
    ws = float(entry.get("wind_speed") or 0.0)
    wd = float(entry.get("wind_dir_deg") or 0.0)

    prev = train[0].acquired_at.timestamp()
    for hs in train:
        t = hs.acquired_at.timestamp()
        dt = t - prev
        if dt > 0:
            _predict_chunked(sim, dt, ws, wd, mf, ds, chunk_s)
        sim.update(Evidence.satellite_hotspot(lat=hs.latitude, lon=hs.longitude))
        prev = t
    if cutoff_ts > prev:
        _predict_chunked(sim, cutoff_ts - prev, ws, wd, mf, ds, chunk_s)
    return sim


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _haversine(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in metres; broadcasts over array lat/lon."""
    p1 = np.radians(np.asarray(lat1, dtype=float))
    p2 = np.radians(np.asarray(lat2, dtype=float))
    dp = np.radians(np.asarray(lat2, dtype=float) - np.asarray(lat1, dtype=float))
    dl = np.radians(np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float))
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2.0 * R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def convex_hull_area_km2(points: list[tuple[float, float]]) -> float:
    """Monotonic-chain hull area in km² (equirectangular approximation).
    Returns NaN when fewer than 3 distinct points."""
    pts = sorted(set((round(p[0], 6), round(p[1], 6)) for p in points))
    if len(pts) < 3:
        return float("nan")

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return float("nan")

    area = 0.0
    for i in range(len(hull)):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % len(hull)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0 * (DEG_KM * DEG_KM)


def evaluate(
    sim: BayesianFireGrid,
    train: list[nasa_firms.FIRMSHotspot],
    test: list[nasa_firms.FIRMSHotspot],
    persistence_radius_m: float,
) -> dict:
    """Score the predicted field against the held-out test hotspots."""
    P = sim.probabilities
    n_test = len(test)

    hits = {}
    for t in THRESHOLDS:
        mask = P >= t
        hits[t] = sum(
            1 for hs in test if mask[sim.latlon_to_cell(hs.latitude, hs.longitude)]
        )

    mask05 = P >= 0.05
    extent_cells = int(mask05.sum())
    cell_area_km2 = (sim.cell_size ** 2) / 1e6
    area_pred_km2 = extent_cells * cell_area_km2

    dists: list[float] = []
    idx = np.argwhere(mask05)
    if len(idx):
        mlat = sim.grid_lat[idx[:, 0], idx[:, 1]]
        mlon = sim.grid_lon[idx[:, 0], idx[:, 1]]
        for hs in test:
            dists.append(float(_haversine(hs.latitude, hs.longitude, mlat, mlon).min()))
    mean_dist_km = (sum(dists) / len(dists)) / 1000.0 if dists else float("nan")

    # Continuous score: mean probability the model assigned to the cells
    # where hotspots were ACTUALLY observed (0.01 background = collapsed).
    p_at_test = []
    for hs in test:
        i, j = sim.latlon_to_cell(hs.latitude, hs.longitude)
        p_at_test.append(float(P[i, j]))
    mean_p_at_test = sum(p_at_test) / len(p_at_test)

    # Persistence baseline: test hotspot within radius of ANY train hotspot
    tlats = np.array([h.latitude for h in train])
    tlons = np.array([h.longitude for h in train])
    pers_hits = 0
    for hs in test:
        if float(_haversine(hs.latitude, hs.longitude, tlats, tlons).min()) <= persistence_radius_m:
            pers_hits += 1

    # Movement: train centroid -> test centroid
    tc = (float(tlats.mean()), float(tlons.mean()))
    ulats = np.array([h.latitude for h in test])
    ulons = np.array([h.longitude for h in test])
    uc = (float(ulats.mean()), float(ulons.mean()))
    move_km = float(_haversine(tc[0], tc[1], uc[0], uc[1])) / 1000.0

    all_pts = [(h.latitude, h.longitude) for h in train] + [
        (h.latitude, h.longitude) for h in test
    ]
    obs_area = convex_hull_area_km2(all_pts)

    return {
        "hit": {t: round(hits[t] / n_test, 3) for t in THRESHOLDS},
        "n_test": n_test,
        "dist_km": round(mean_dist_km, 2) if mean_dist_km == mean_dist_km else None,
        "extent_cells": extent_cells,
        "area_pred_km2": round(area_pred_km2, 2),
        "area_obs_km2": round(obs_area, 2) if obs_area == obs_area else None,
        "move_km": round(move_km, 2),
        "pers_hit": round(pers_hits / n_test, 3) if n_test else None,
        "p_at_test": round(mean_p_at_test, 4),
    }


# ---------------------------------------------------------------------------
# Analysis loop
# ---------------------------------------------------------------------------

def analyze(
    api_key: str,
    mode: str,
    limit: int,
    days: int,
    min_confidence: str,
    chunk_s: float,
    persistence_m: float,
    quiet: bool,
) -> dict:
    entries = load_candidate_grids(mode, limit)
    results = []
    skipped: dict[str, list[str]] = {}

    for entry in entries:
        gid = entry["id"]
        try:
            history = fetch_history(api_key, entry, days, min_confidence)
        except Exception as exc:  # network / oversized bbox / API errors
            reason = f"fetch: {type(exc).__name__}: {str(exc)[:90]}"
            skipped.setdefault(reason, []).append(gid)
            continue
        if len(history) < 4:
            skipped.setdefault(f"<4 hotspots in {days}d window", []).append(gid)
            continue

        train, test, cutoff = split_history(history)
        if len(train) < 2 or len(test) < 2:
            skipped.setdefault("split too small", []).append(gid)
            continue

        sim = rebuild(entry, train, cutoff, chunk_s)
        r = evaluate(sim, train, test, persistence_m)
        r.update(
            id=gid,
            lat=round(entry["centroid_lat"], 4),
            lon=round(entry["centroid_lon"], 4),
            window_h=round((history[-1][0] - history[0][0]) / 3600.0, 1),
            n_train=len(train),
            wind_kmh=round(float(entry.get("wind_speed") or 0.0) * 3.6, 1),
            ffmc=round(float(entry.get("ffmc") or 0.0), 1),
        )
        results.append(r)
        if not quiet:
            d = r["dist_km"] if r["dist_km"] is not None else "—"
            print(
                f"  {gid:12} win={r['window_h']:6.1f}h "
                f"tr={r['n_train']:3d}/te={r['n_test']:3d}  "
                f"hit@.05={r['hit'][0.05]:.2f} hit@.1={r['hit'][0.10]:.2f} "
                f"hit@.3={r['hit'][0.30]:.2f}  dist={d}km "
                f"pers={r['pers_hit']:.2f}  p@test={r['p_at_test']:.3f}"
            )

    return {"results": results, "skipped": skipped, "candidates": len(entries)}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _mean(xs: list[float]) -> float:
    clean = [x for x in xs if x == x]  # drop NaN
    return sum(clean) / len(clean) if clean else float("nan")


def print_report(a: dict) -> None:
    res = a["results"]
    n = len(res)
    print("=" * 84)
    print("  GRID-LEVEL SPREAD BACKTEST — predicted extent vs observed FIRMS hotspots")
    print("=" * 84)
    print(f"  grids evaluated: {n}   candidates: {a['candidates']}   "
          f"skipped: {sum(len(v) for v in a['skipped'].values())}")

    if not res:
        print("\n  No grids met the minimum criteria (>=4 hotspots, train+test split).")
        if a["skipped"]:
            print("  Skip reasons:")
            for reason, gids in sorted(a["skipped"].items()):
                print(f"    - {reason}  ({len(gids)}: {', '.join(gids[:5])}{'…' if len(gids) > 5 else ''})")
        return

    print(f"\n  {'ID':12} {'win.h':>6} {'tr/te':>6} {'hit.02':>6} {'hit.05':>6} "
          f"{'hit.30':>6} {'distkm':>6} {'predkm²':>7} {'obs km²':>7} {'movekm':>6} "
          f"{'pers':>5} {'p@test':>7} {'skill':>6}")
    print("  " + "-" * 90)
    for r in res:
        d = f"{r['dist_km']:6.2f}" if r["dist_km"] is not None else f"{'—':>6}"
        skill = r["hit"][0.05] - r["pers_hit"]
        mark = " ✓" if skill > 0.01 else " ·"
        print(f"  {r['id']:12} {r['window_h']:6.1f} {r['n_train']:3d}/{r['n_test']:<3d} "
              f"{r['hit'][0.02]:6.2f} {r['hit'][0.05]:6.2f} {r['hit'][0.30]:6.2f} "
              f"{d} {r['area_pred_km2']:7.1f} {r['area_obs_km2']:7.1f} "
              f"{r['move_km']:6.1f} {r['pers_hit']:5.2f} {r['p_at_test']:7.3f} "
              f"{skill:+5.2f}{mark}")

    print(f"\n  {'─' * 84}")
    print(f"  {_BOLD}SUMMARY (mean across {n} grids){_RESET}")
    print(f"  {'─' * 84}")
    for t in THRESHOLDS:
        vals = [r["hit"][t] for r in res]
        print(f"  hit@{t:<4} = {_mean(vals):.3f}   "
              f"(median {np.median(vals):.3f})")
    dists = [r["dist_km"] for r in res if r["dist_km"] is not None]
    print(f"  mean dist to extent (km)   : {_mean(dists):.2f}  (n={len(dists)} grids with extent)")
    print(f"  mean predicted area (km²)  : {_mean([r['area_pred_km2'] for r in res]):.1f}")
    print(f"  mean observed area (km²)   : {_mean([r['area_obs_km2'] for r in res]):.1f}")
    print(f"  mean persistence hit       : {_mean([r['pers_hit'] for r in res]):.3f}")
    print(f"  mean P at observed cells   : {_mean([r['p_at_test'] for r in res]):.4f} "
          f"(background is 0.01 — low = field collapsed)")
    no_extent = sum(1 for r in res if r["extent_cells"] == 0)
    print(f"  grids with NO predicted extent at cutoff (P>=0.05): {no_extent}/{n}")
    better = sum(1 for r in res if r["hit"][0.05] - r["pers_hit"] > 0.01)
    print(f"  grids where model beat persistence: {better}/{n} "
          f"({100.0 * better / n:.0f}%)")

    if a["skipped"]:
        print(f"\n  {'─' * 84}")
        print(f"  {_BOLD}SKIPPED GRIDS{_RESET} ({sum(len(v) for v in a['skipped'].values())})")
        for reason, gids in sorted(a["skipped"].items()):
            shown = ", ".join(gids[:4]) + ("…" if len(gids) > 4 else "")
            print(f"    - {reason}  [{len(gids)}: {shown}]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest predicted spread extent vs observed FIRMS hotspots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", default="production", help="DB mode (default production)")
    parser.add_argument("--limit", type=int, default=20, help="max grids to evaluate (newest first)")
    parser.add_argument("--days", type=int, default=5, help="FIRMS lookback window 1-5 (default 5)")
    parser.add_argument("--min-confidence", default="nominal",
                        choices=["low", "nominal", "high"],
                        help="FIRMS confidence filter (default nominal, same as the poller)")
    parser.add_argument("--chunk-s", type=float, default=DEFAULT_CHUNK_S,
                        help="predict() chunk cap in seconds (default 600, matches production)")
    parser.add_argument("--persistence-radius-m", type=float, default=1000.0,
                        help="radius for the 'fire didn't move' baseline (default 1000)")
    parser.add_argument("--json", metavar="FILE", help="export per-grid results to JSON")
    parser.add_argument("--quiet", action="store_true", help="suppress per-grid progress lines")
    args = parser.parse_args()

    days = max(1, min(args.days, nasa_firms.MAX_DAY_RANGE))

    api_key = nasa_firms._get_api_key()
    if not api_key:
        print("NASA_FIRMS_API_KEY is not set in .env — cannot fetch FIRMS history.")
        sys.exit(1)

    print(f"Fetching FIRMS history per grid (days={days}, "
          f"min_confidence={args.min_confidence}) — one API call per grid…\n")
    a = analyze(
        api_key=api_key,
        mode=args.mode,
        limit=args.limit,
        days=days,
        min_confidence=args.min_confidence,
        chunk_s=args.chunk_s,
        persistence_m=args.persistence_radius_m,
        quiet=args.quiet,
    )
    print_report(a)

    if args.json:
        payload = {
            "mode": args.mode,
            "days": days,
            "min_confidence": args.min_confidence,
            "chunk_s": args.chunk_s,
            "persistence_radius_m": args.persistence_radius_m,
            "thresholds": list(THRESHOLDS),
            "n_evaluated": len(a["results"]),
            "skipped": {k: len(v) for k, v in a["skipped"].items()},
            "grids": a["results"],
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n  📄 Exported {len(a['results'])} grids to {args.json}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
calibrate_fwi.py — Calibrate the FFMC moisture curve against stored ISI
========================================================================
The spread model's ``moisture_factor(ffmc)`` is currently a placeholder
exponential (``exp(-0.045·fmc)``). The Canadian FWI system — the same system
EFFIS uses to compute the ISI values we store per grid — has a canonical,
physically-derived fine-fuel moisture factor:

    m   = 147.2·(101 − FFMC) / (59.5 + FFMC)          (Van Wagner 1987)
    fF  = 91.9·exp(−0.1386·m) · (1 + m^5.31 / 4.93e7) (fine-fuel factor)
    ISI = 0.208·fF·exp(0.05039·U)                     (U in km/h)

What this harness does
----------------------
1. Loads every production grid with real (ffmc, isi, wind) from Postgres.
2. Computes the canonical fF curve from stored FFMC, and the *implied*
   fF from stored (ISI, wind). Their ratio is the EFFIS-vs-OpenMeteo wind
   mismatch — the honest signal this data can measure.
3. Verifies the canonical curve end-to-end: predicted ISI (canonical FFMC
   curve + our stored wind) vs stored EFFIS ISI.
4. Picks a *bias-neutral anchor* for moisture_factor: the fuel moisture
   where the canonical factor equals 1.0 over the observed fire population
   (geomean of fF across the sample), so the new curve neither speeds up
   nor slows down fires on average versus today's wind-only model.

Caveat (read before trusting any single number): EFFIS computes ISI from its
OWN model wind; we store Open-Meteo wind. Because the canonical fF shape IS
the function EFFIS used to build ISI, the data cannot independently *invent*
a better curve shape — it can only confirm the canonical one and quantify
how consistent our wind is with EFFIS's. The observed sample is narrow in
FFMC (≈87–97, fires are by definition in dry fuel), so the anchor estimate
is approximate and should be re-run as the grid population grows.

Usage
-----
    .venv/bin/python calibrate_fwi.py                  # report only
    .venv/bin/python calibrate_fwi.py --json fwi_calib.json
    .venv/bin/python calibrate_fwi.py --apply          # write constants
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import db

# ---------------------------------------------------------------------------
# Canonical Canadian FWI equations (Van Wagner & Pickett 1985)
# ---------------------------------------------------------------------------

def fmc_from_ffmc(ffmc: float) -> float:
    """FFMC → fine-fuel moisture content % (Van Wagner 1987)."""
    f = max(0.0, min(101.0, float(ffmc)))
    return 147.2 * (101.0 - f) / (59.5 + f)


def fF_from_fmc(fmc: float) -> float:
    """Fine-fuel moisture factor from moisture % (FWI fine fuel factor)."""
    return 91.9 * math.exp(-0.1386 * fmc) * (1.0 + fmc ** 5.31 / 4.93e7)


def isi_from_fwi(ffmc: float, wind_kmh: float) -> float:
    """Canonical ISI from FFMC + wind speed (km/h, at 10 m)."""
    return 0.208 * fF_from_fmc(fmc_from_ffmc(ffmc)) * math.exp(0.05039 * wind_kmh)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_grids(mode: str, limit: int) -> list[dict]:
    """Grids with real fuel-moisture + wind (all three present).

    Queries the full table directly — list_grid_meta orders by max_p and
    applies a LIMIT window, which would silently under-sample the
    (small) population of grids that actually carry EFFIS data.
    """
    sql = (
        "SELECT id, centroid_lat, centroid_lon, wind_speed, wind_dir_deg, "
        "max_p, wind_updated_at, ffmc, dmc, isi "
        "FROM bayesian_grids WHERE mode = %s AND ffmc > 0 "
        "ORDER BY max_p DESC"
    )
    params: list = [mode]
    if limit and limit > 0:
        sql += " LIMIT %s"
        params.append(int(limit))
    with db._conn() as conn:
        rows = conn.execute(sql, params).fetchall()
    out = []
    for r in rows:
        wind_ms = float(r["wind_speed"] or 0.0)
        has_wind = (r["wind_updated_at"] or 0) > 0
        if wind_ms > 0 and has_wind:
            out.append({
                "id": r["id"],
                "lat": r["centroid_lat"],
                "lon": r["centroid_lon"],
                "ffmc": float(r["ffmc"]),
                "dmc": float(r["dmc"] or 0.0),
                "isi": float(r["isi"] or 0.0),
                "wind_kmh": wind_ms * 3.6,
                "max_p": float(r["max_p"]),
            })
    return out


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _geomean(xs: list[float]) -> float:
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def analyze(grids: list[dict]) -> dict:
    rows = []
    for g in grids:
        m = fmc_from_ffmc(g["ffmc"])
        fF_canon = fF_from_fmc(m)
        # Implied fF from stored (ISI, our wind): ISI = 0.208·fF·exp(0.05039·U)
        # → fF_implied = ISI / (0.208·exp(0.05039·U_ours)). Since EFFIS built
        # ISI with ITS OWN wind, fF_implied = fF_canon · exp(0.05039·ΔU) where
        # ΔU = U_effis − U_ours. The ratio is the wind mismatch.
        fF_implied = g["isi"] / (0.208 * math.exp(0.05039 * g["wind_kmh"]))
        wind_mismatch = fF_implied / fF_canon if fF_canon > 0 else float("nan")
        # Equivalent wind difference in km/h: exp(0.05039·ΔU) = mismatch
        delta_u = math.log(wind_mismatch) / 0.05039 if wind_mismatch > 0 else float("nan")
        rows.append({**g, "fmc_pct": m, "fF_canon": fF_canon, "fF_implied": fF_implied,
                     "wind_mismatch": wind_mismatch, "delta_u_kmh": delta_u})

    # --- Wind-mismatch summary (log-space, so ratios average geometrically) ---
    mismatches = [r["wind_mismatch"] for r in rows if r["wind_mismatch"] > 0]
    geo_mismatch = _geomean(mismatches) if mismatches else float("nan")
    log_mm = [math.log(x) for x in mismatches]
    sd_log = math.sqrt(sum((x - sum(log_mm) / len(log_mm)) ** 2 for x in log_mm) / len(log_mm))

    # --- Bias-neutral anchor: the FMC where fF(m) = geomean(fF) of the
    # observed population, i.e. moisture_factor averages 1.0 over the fires
    # we actually see. Invert fF analytically by grid search. ---
    target_fF = _geomean([r["fF_canon"] for r in rows])
    anchor = None
    for a10 in range(15, 401):  # 1.5% .. 40.0%
        cand = a10 / 10.0
        if fF_from_fmc(cand) <= target_fF:  # fF decreasing in m
            anchor = round(cand, 1)
            break

    # --- Verification: canonical curve + our wind → ISI, vs stored ISI ---
    ver = []
    for r in rows:
        isi_pred = isi_from_fwi(r["ffmc"], r["wind_kmh"])
        ver.append({"id": r["id"], "isi_stored": r["isi"], "isi_pred": isi_pred,
                    "ratio": r["isi"] / isi_pred if isi_pred > 0 else float("nan")})
    within = sum(1 for v in ver if 0.5 <= v["ratio"] <= 2.0)

    # --- What the canonical curve does at reference FFMC values ---
    curve = {str(ffmc): round(mf_canonical(ffmc, anchor), 4) for ffmc in (75, 80, 85, 89.5, 92, 94, 96)}

    return {
        "rows": rows,
        "wind_mismatch_geomean": geo_mismatch,
        "wind_mismatch_sd_log": sd_log,
        "delta_u_geo_kmh": math.log(geo_mismatch) / 0.05039 if geo_mismatch > 0 else float("nan"),
        "anchor_fmc_pct": anchor,
        "anchor_ffmc": _ffmc_at_fmc(anchor) if anchor else None,
        "target_fF": target_fF,
        "verification": ver,
        "verification_within_2x": within,
        "curve_at": curve,
    }


def mf_canonical(ffmc: float, anchor_fmc: float) -> float:
    """Canonical moisture_factor: fF(m) / fF(m_ref), 1.0 at the anchor."""
    m = fmc_from_ffmc(ffmc)
    return fF_from_fmc(m) / fF_from_fmc(anchor_fmc)


def _ffmc_at_fmc(fmc: float) -> float:
    """Invert the Van Wagner formula: moisture % → FFMC."""
    return (147.2 * 101.0 - fmc * 59.5) / (147.2 + fmc)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_BOLD = "\033[1m"
_RESET = "\033[0m"


def print_report(n: int, a: dict) -> None:
    rows = a["rows"]
    print("=" * 80)
    print("  FWI MOISTURE-CURVE CALIBRATION — stored (FFMC, ISI, wind)")
    print("=" * 80)
    print(f"  grids with real ffmc+isi+wind: {n}")

    print(f"\n  {'ID':12} {'FFMC':>6} {'FMC%':>7} {'ISI':>6} {'WINDkm/h':>8}  "
          f"{'fF(canon)':>9} {'fF(impl)':>8} {'windΔkm/h':>9}")
    print("  " + "-" * 74)
    for r in rows:
        d = r["delta_u_kmh"]
        print(f"  {r['id']:12} {r['ffmc']:6.1f} {r['fmc_pct']:7.2f} {r['isi']:6.1f} "
              f"{r['wind_kmh']:8.1f}  {r['fF_canon']:9.2f} {r['fF_implied']:8.2f} "
              f"{d:9.1f}")

    print(f"\n  {'─' * 80}")
    print(f"  {_BOLD}1) WIND CONSISTENCY (EFFIS model wind vs our Open-Meteo){_RESET}")
    print(f"  {'─' * 80}")
    print(f"  geometric-mean mismatch of implied fF vs canonical fF: "
          f"{a['wind_mismatch_geomean']:.3f}")
    print(f"  ≈ EFFIS wind was {a['delta_u_geo_kmh']:+.1f} km/h "
          f"{'stronger' if a['delta_u_geo_kmh'] > 0 else 'weaker'} than ours on average")
    print(f"  log-sd of per-grid mismatch: {a['wind_mismatch_sd_log']:.3f} "
          f"(±{a['wind_mismatch_sd_log'] / 0.05039:.1f} km/h per grid)")

    print(f"\n  {'─' * 80}")
    print(f"  {_BOLD}2) VERIFICATION — canonical curve + our wind vs stored ISI{_RESET}")
    print(f"  {'─' * 80}")
    print(f"    {'ID':12} {'ISI stored':>10} {'ISI pred':>9} {'ratio':>7}")
    for v in a["verification"]:
        good = 0.5 <= v["ratio"] <= 2.0
        mark = "✓" if good else "·"
        print(f"    {v['id']:12} {v['isi_stored']:10.1f} {v['isi_pred']:9.1f} "
              f"{v['ratio']:7.2f} {mark}")
    print(f"    within 0.5×–2× of stored ISI: {a['verification_within_2x']}/{len(a['verification'])}")

    print(f"\n  {'─' * 80}")
    print(f"  {_BOLD}3) BIAS-NEUTRAL ANCHOR (factor = 1.0 at the population mean){_RESET}")
    print(f"  {'─' * 80}")
    print(f"  anchor: FMC {a['anchor_fmc_pct']}%  (FFMC ≈ {a['anchor_ffmc']:.1f})")
    print(f"  canonical curve fF(m)/fF(m_ref):")
    print(f"    {'FFMC':>6} {'FMC%':>7} {'factor':>7}")
    for ffmc in (75, 80, 85, 89.5, 92, 94, 96):
        m = fmc_from_ffmc(ffmc)
        print(f"    {ffmc:6.1f} {m:7.2f} {mf_canonical(ffmc, a['anchor_fmc_pct']):7.3f}")

    print(f"\n  {_BOLD}COMPARED TO TODAY'S PLACEHOLDER{_RESET} "
          f"(exp(−0.045·m), anchor FMC 12%):")
    print(f"    FFMC 75 → 0.50   FFMC 85 → 0.80   FFMC 89.5 → 1.00   "
          f"FFMC 92 → 1.14   FFMC 94 → 1.27   FFMC 96 → 1.43")
    print(f"    (the canonical curve above is steeper — FWI's own shape)")


def recommended_constants(a: dict) -> dict:
    return {
        "anchor_fmc_pct": a["anchor_fmc_pct"],
        "anchor_ffmc": a["anchor_ffmc"],
        "wind_mismatch_geomean": a["wind_mismatch_geomean"],
        "delta_u_geo_kmh": a["delta_u_geo_kmh"],
        "verification_within_2x": a["verification_within_2x"],
        "n": len(a["rows"]),
    }


# ---------------------------------------------------------------------------
# Apply to effis_fwi.py
# ---------------------------------------------------------------------------

def apply_to_effis_fwi(a: dict) -> None:
    anchor = a["anchor_fmc_pct"]
    path = "effis_fwi.py"
    src = open(path).read()
    start = src.index("def moisture_factor(")
    end = src.index("def decay_scale(")

    c96 = mf_canonical(96.0, anchor)
    c89 = mf_canonical(89.5, anchor)
    c75 = mf_canonical(75.0, anchor)
    # Show the clamped values the function actually returns (matches the
    # on-disk docstring in effis_fwi.py so --apply stays byte-idempotent).
    c75 = max(0.2, c75); c89 = max(0.2, min(2.0, c89))
    c96 = max(0.2, min(2.0, c96))
    replacement = f'''def moisture_factor(ffmc: float) -> float:
    """Spread-rate multiplier from FFMC, calibrated to the canonical FWI curve.

    Replaces the old placeholder exponential (exp(-0.045*m), anchored at
    FMC 12%) with Van Wagner (1987)'s fF(m) — the exact moisture curve that
    drives EFFIS's Initial Spread Index — anchored at the bias-neutral FMC
    {anchor}% (FFMC ~{a['anchor_ffmc']:.1f}) found by calibrate_fwi.py over the
    stored grid population, so the average observed fire still spreads at
    factor 1.0. The curve is steeper than the placeholder: FFMC 75 ->
    ~{c75:.2f}, FFMC 89.5 -> ~{c89:.2f}, FFMC 92 -> ~1.0, FFMC 96 -> ~{c96:.2f}.
    Clamped to [0.2, 2.0].
    """
    if not ffmc or ffmc <= 0:
        return 1.0
    fmc = ffmc_to_fmc_pct(ffmc)
    factor = _fwi_ff(fmc) / _FW_REF_FF
    return max(0.2, min(2.0, factor))


'''
    src = src[:start] + replacement + src[end:]
    open(path, "w").write(src)
    print(f"✅ Wrote calibrated moisture_factor (anchor FMC {anchor}%) to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate the FFMC moisture curve against stored ISI values.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", default="production", help="DB mode (default production)")
    parser.add_argument("--limit", type=int, default=10000, help="max grids to consider")
    parser.add_argument("--json", metavar="FILE", help="export analysis to JSON")
    parser.add_argument("--apply", action="store_true", help="write calibrated constants to effis_fwi.py")
    args = parser.parse_args()

    grids = load_grids(args.mode, args.limit)
    if len(grids) < 5:
        print(f"⚠ Only {len(grids)} grids with real fuel-moisture data — "
              f"calibration is approximate below ~10 samples.")
        if not grids:
            sys.exit(1)

    a = analyze(grids)
    print_report(len(grids), a)

    if args.json:
        payload = {
            "mode": args.mode,
            "n": len(grids),
            "grids": [{k: v for k, v in r.items() if k != "wind_mismatch"} | {"wind_mismatch": r["wind_mismatch"]} for r in a["rows"]],
            "analysis": {k: v for k, v in a.items() if k != "rows"},
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n  📄 Exported {len(grids)} grids + analysis to {args.json}")

    if args.apply:
        apply_to_effis_fwi(a)
        print("  ℹ️  Restart services / rerun the worker refresh for the new curve to take effect.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Deterministic tests for forecast-driven road risk (wind-at-arrival).

Covers the peer-reviewed convergence contract in ``_converged_arrival``
(bayesian_filter.py):

1. Critical tier (<30 min) is byte-identical — no iteration, current wind.
2. Wind-at-arrival re-tiering — a road assessed with the forecast wind at
   its arrival time (not the current wind) can flip tier.
3. Tier-stable early stop — arrival converges when the tier holds steady
   across consecutive iterations.
4. Oscillation fallback — a tier that cycles between forecast-hour buckets
   (high -> moderate -> high) must NOT be declared converged; the road
   falls back to the current-wind estimate.
5. The ``rate_modifier`` hook is applied inside the loop.
6. ``compute_road_risk`` plumbing: forecast_series threaded through,
   ``wind_source`` surfaced (forecast/current).

Run:  .venv/bin/python test_road_risk_forecast.py
"""
import sys

sys.path.insert(0, ".")

from bayesian_filter import (  # noqa: E402
    _converged_arrival,
    _rate_toward,
    _risk_tier,
    compute_road_risk,
    Evidence,
    BayesianFireGrid,
)

FAILED = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS {name}")
    else:
        print(f"  FAIL {name}  {detail}")
        FAILED.append(name)


# Rates used by the scenario math below (direction = spread-TOWARD).
head = _rate_toward(270.0, 2.9, 270.0, 1.0)     # head-on toward the road
shifted = _rate_toward(270.0, 2.9, 90.0, 1.0)   # wind shifted to blow away

# Series: hour 0 blows TOWARD the road (270), hours 1-2 blow AWAY (90).
FC = [
    {"ts": 0, "speed": 2.9, "dir": 270.0, "precip_mm": 0, "humidity": 50, "temp_c": 20},
    {"ts": 3600, "speed": 2.9, "dir": 90.0, "precip_mm": 0, "humidity": 50, "temp_c": 20},
    {"ts": 7200, "speed": 2.9, "dir": 90.0, "precip_mm": 0, "humidity": 50, "temp_c": 20},
]
NOW = 1_780_000_000.0

# ---- 1. Critical tier: no iteration, current wind returned ----
print("== 1. critical unchanged ==")
d_crit = 100.0  # 100 m at head rate -> ~14 min (< 30, critical)
t, ws, wd, conv = _converged_arrival(270.0, d_crit, 2.9, 270.0, 1.0, FC, None, NOW)
check("critical returns <30 min", t < 30.0, f"{t:.1f}")
check("critical keeps current wind", (ws, wd) == (2.9, 270.0), f"{ws},{wd}")
check("critical converged=False (no iteration)", conv is False, conv)
check("critical is critical tier", _risk_tier(t) == "critical", _risk_tier(t))

# ---- 2. Re-tiering: current wind -> high; forecast wind away -> moderate ----
print("== 2. re-tiering (wind-at-arrival) ==")
# 850 m: 850/7.18 = 118 min (high, just under the 120 min boundary);
# with the shifted wind 850/5.58 = 152 min (moderate). The forecast wind
# at the arrival time is what actually decides the tier.
d = 850.0
t_cur = d / head
check("scenario: current wind is HIGH tier", _risk_tier(t_cur) == "high",
      f"{t_cur:.0f}min -> {_risk_tier(t_cur)}")
t, ws, wd, conv = _converged_arrival(270.0, d, 2.9, 270.0, 1.0, FC, None, NOW)
check("forecast arrival slower (wind shifted away)", t > t_cur * 1.2, f"{t_cur:.0f} -> {t:.0f}")
check("forecast tier downgraded (high -> moderate/low)",
      _risk_tier(t) in ("moderate", "low"), f"{_risk_tier(t)} vs high")
check("forecast wind used is the shifted one", abs(wd - 90.0) < 0.01, wd)
check("converged=True", conv is True, conv)

# ---- 3. Tier-stable early stop ----
print("== 3. tier-stable stop ==")
FC_STABLE = [{"ts": 0, "speed": 2.9, "dir": 270.0, "precip_mm": 0, "humidity": 50, "temp_c": 20}] * 4
t, ws, wd, conv = _converged_arrival(270.0, d, 2.9, 270.0, 1.0, FC_STABLE, None, NOW)
check("stable wind converges quickly", conv is True, conv)
check("stable wind arrival matches current", abs(t - d / head) < 1.0, f"{t:.1f} vs {d/head:.1f}")

# ---- 4. Oscillation fallback: 2-cycle must NOT converge ----
print("== 4. oscillation fallback ==")
# With a road at d_osc = 700 m: away@3.5 -> 122.8 min (moderate) and
# head@4.5 -> 83.6 min (high). Alternating them produces a genuine
# 2-cycle (high -> moderate -> high) — the tier repeats an EARLIER value,
# so the code must fall back to the current-wind estimate.
d_osc = 700.0
t_cur_osc = d_osc / head
import bayesian_filter as bf  # noqa: E402

real = bf._wind_at_arrival
calls = {"n": 0}


def oscillating(series, now_epoch, t_min):
    calls["n"] += 1
    if calls["n"] % 2 == 1:
        return (3.5, 90.0)   # away-wind -> moderate tier
    return (4.5, 270.0)      # head-wind -> high tier


bf._wind_at_arrival = oscillating
try:
    t, ws, wd, conv = _converged_arrival(270.0, d_osc, 2.9, 270.0, 1.0, FC, None, NOW)
finally:
    bf._wind_at_arrival = real
check("oscillation detected early (cycle repeat)", calls["n"] >= 2, calls["n"])
check("oscillation falls back to current wind", (ws, wd) == (2.9, 270.0), f"{ws},{wd}")
check("oscillation returns the current-wind estimate",
      abs(t - t_cur_osc) < 1.0, f"{t:.1f} vs {t_cur_osc:.1f}")
check("oscillation converged=False", conv is False, conv)

# ---- 5. rate_modifier hook is applied inside the loop ----
print("== 5. rate_modifier hook ==")


def halve(rate, t_min):
    return rate * 0.5


t_mod, ws, wd, conv = _converged_arrival(270.0, d, 2.9, 270.0, 1.0, FC_STABLE, halve, NOW)
check("rate_modifier roughly doubles arrival",
      abs(t_mod - 2.0 * d / head) < 2.0, f"{t_mod:.1f} vs {2 * d / head:.1f}")

# ---- 6. compute_road_risk plumbing ----
print("== 6. compute_road_risk smoke ==")
g = BayesianFireGrid(center_lat=51.106, center_lon=18.941, cell_size_m=2000.0, nx=60, ny=60)
g.update(Evidence.satellite_hotspot(51.106, 18.941))
g.predict(dt=900.0)
import math  # noqa: E402


def point_at(lat, lon, dist, brg):
    R = 6_371_000.0
    br = math.radians(brg)
    lat2 = math.asin(math.sin(lat) * math.cos(dist / R) + math.cos(lat) * math.sin(dist / R) * math.cos(br))
    lon2 = lon + math.atan2(math.sin(br) * math.sin(dist / R) * math.cos(lat),
                            math.cos(dist / R) - math.sin(lat) * math.sin(lat2))
    return (math.degrees(lat2), math.degrees(lon2))


p1 = point_at(51.106, 18.941, 1500.0, 90.0)
p2 = point_at(p1[0], p1[1], 300.0, 90.0)
seg = [p1, p2]
r_no = compute_road_risk(g, [seg], 2.9, 90.0)
r_fc = compute_road_risk(g, [seg], 2.9, 90.0, forecast_series=FC)
check("both runs return a result row", len(r_no) == 1 and len(r_fc) == 1, f"{len(r_no)},{len(r_fc)}")
check("forecast run surfaces wind_source key", "wind_source" in r_fc[0], r_fc[0].get("wind_source"))
check("no-forecast wind_source == current", r_no[0].get("wind_source") == "current",
      r_no[0].get("wind_source"))

print()
if FAILED:
    print(f"{len(FAILED)} FAILURES: {FAILED}")
    sys.exit(1)
print("ALL PASS")

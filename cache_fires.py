#!/usr/bin/env python3
"""Cache fire data so cap_sweep doesn't re-download each time."""

import json, random
from pathlib import Path
from backtest import download_calfire_perimeters, generate_realistic_hotspots

random.seed(42)
print("Downloading fires from 2024...")
fires = download_calfire_perimeters(2024, verbose=True)
fires.sort(key=lambda f: f["area_km2"], reverse=True)
fires = fires[:30]

print(f"\nGenerating hotspots for {len(fires)} fires...")
for fire in fires:
    fire["hotspots"] = generate_realistic_hotspots(fire)
    n = len(fire["hotspots"])
    print(f"  {fire['fire_name']:25s} {fire['area_km2']:7.1f} km²  {n} hotspots")

# Serialize (geometry is a shapely object — convert to wkt)
for fire in fires:
    fire["geometry"] = fire["geometry"].wkt

with open("cap_sweep_cache.json", "w") as f:
    json.dump({"fires": fires, "decay_s": 86400}, f)

print(f"\nCached {len(fires)} fires to cap_sweep_cache.json")

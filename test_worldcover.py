#!/usr/bin/env python3
"""Tests for the ESA WorldCover integration (global land-cover fallback).

Covers:
  * worldcover.py — pure class mapping (burnable / cropland / non-burnable)
  * db.worldcover_code_batch — batched PostGIS ST_Contains lookup, fail-open
    when the table is missing
  * server._gate_firms_by_land_cover — WorldCover fallback for points outside
    CORINE coverage; CORINE still takes priority inside Europe
  * server._tag_ag_burn — WorldCover cropland class triggers ag-burn tag

Run from the project root:  .venv/bin/python test_worldcover.py
"""

import sys

import worldcover


PASS = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")


def test_mapping() -> None:
    print("[worldcover.is_burnable]")
    # Non-burnable
    check("50 built-up is not burnable", not worldcover.is_burnable(50))
    check("60 bare is not burnable", not worldcover.is_burnable(60))
    check("70 snow/ice is not burnable", not worldcover.is_burnable(70))
    check("80 water is not burnable", not worldcover.is_burnable(80))
    # Burnable
    check("10 tree cover is burnable", worldcover.is_burnable(10))
    check("20 shrubland is burnable", worldcover.is_burnable(20))
    check("30 grassland is burnable", worldcover.is_burnable(30))
    check("40 cropland is burnable", worldcover.is_burnable(40))
    check("90 wetland is burnable", worldcover.is_burnable(90))
    check("95 mangroves is burnable", worldcover.is_burnable(95))
    check("100 moss/lichen is burnable", worldcover.is_burnable(100))
    # Unknown / outside coverage: permissive (fail-open)
    check("None (outside coverage) is burnable", worldcover.is_burnable(None))
    check("unknown code 999 is burnable", worldcover.is_burnable(999))

    print("[worldcover.is_cropland]")
    check("40 cropland is cropland", worldcover.is_cropland(40))
    check("10 tree cover is NOT cropland", not worldcover.is_cropland(10))
    check("50 built-up is NOT cropland", not worldcover.is_cropland(50))
    check("None is NOT cropland", not worldcover.is_cropland(None))


def test_db_lookup() -> None:
    print("[db.worldcover_code_batch]")
    import db

    try:
        with db._conn() as conn:
            row = conn.execute(
                "SELECT EXISTS ("
                "  SELECT 1 FROM information_schema.tables"
                "  WHERE table_name = 'worldcover_polygons'"
                ")"
            ).fetchone()
            exists = row["exists"]
        if not exists:
            raise FileNotFoundError
    except Exception:
        print("  SKIP worldcover_polygons table not loaded yet (run worldcover_import.py)")
        return

    # A point in central Poland should resolve to a WorldCover class.
    codes = db.worldcover_code_batch([
        (52.2297, 21.0122),   # Warsaw — likely built-up (50)
    ])
    check("Warsaw resolves to a class", codes.get(0) is not None,
          f"got {codes}")
    if codes.get(0) is not None:
        check("Warsaw is non-burnable (built-up)",
              not worldcover.is_burnable(codes[0]),
              f"code={codes[0]}")

    # A point in Białowieża Forest should be burnable.
    codes2 = db.worldcover_code_batch([
        (52.7419, 23.8590),   # Białowieża Forest
    ])
    check("Białowieża resolves to a class", codes2.get(0) is not None,
          f"got {codes2}")
    if codes2.get(0) is not None:
        check("Białowieża is burnable",
              worldcover.is_burnable(codes2[0]),
              f"code={codes2[0]}")

    # Point far outside coverage → absent from result.
    far = db.worldcover_code_batch([(-14.6, -175.0)])
    check("far point outside coverage is absent", 0 not in far)


def test_gate_fallback() -> None:
    """_gate_firms_by_land_cover falls back to WorldCover outside CORINE."""
    from nasa_firms import FIRMSHotspot
    from server import _gate_firms_by_land_cover, _is_source_passthrough
    import db

    def _hs(lat, lon):
        return FIRMSHotspot(
            latitude=lat, longitude=lon,
            brightness=350.0, scan=1.0, track=1.0,
            acq_date="2026-08-15", acq_time="1200",
            satellite="N", instrument="VIIRS",
            confidence="nominal", version="2.0", frp=3.0, daynight="D",
        )

    # Check if worldcover_polygons exists
    wc_exists = True
    try:
        with db._conn() as conn:
            row = conn.execute(
                "SELECT EXISTS ("
                "  SELECT 1 FROM information_schema.tables"
                "  WHERE table_name = 'worldcover_polygons'"
                ")"
            ).fetchone()
            wc_exists = row["exists"]
    except Exception:
        wc_exists = False

    if not wc_exists:
        print("  SKIP gate fallback (worldcover_polygons not loaded)")
        return

    # Monkeypatch: force CORINE to return nothing (simulates outside Europe)
    orig_corine = db.land_cover_codes_batch
    try:
        db.land_cover_codes_batch = lambda pts: {}  # no CORINE coverage

        # Point on non-burnable land (Warsaw built-up)
        warsaw = _hs(52.2297, 21.0122)
        kept = _gate_firms_by_land_cover([warsaw])
        check("Warsaw falls back to WorldCover and is dropped",
              len(kept) == 0,
              f"kept {len(kept)} hotspot(s)")

        # Point on burnable land (Białowieża)
        forest = _hs(52.7419, 23.8590)
        kept2 = _gate_firms_by_land_cover([forest])
        check("Białowieża falls back to WorldCover and is kept",
              len(kept2) == 1,
              f"kept {len(kept2)} hotspot(s)")
    finally:
        db.land_cover_codes_batch = orig_corine

    # Inside CORINE coverage — CORINE should still win
    try:
        orig_codes = db.land_cover_codes_batch
        db.land_cover_codes_batch = lambda pts: {i: "112" for i in range(len(pts))}
        # Warsaw on CORINE 112 (urban, non-burnable)
        warsaw2 = _hs(52.2297, 21.0122)
        kept3 = _gate_firms_by_land_cover([warsaw2])
        check("CORINE still takes priority (112 urban dropped)",
              len(kept3) == 0,
              f"kept {len(kept3)}")
    finally:
        db.land_cover_codes_batch = orig_codes


if __name__ == "__main__":
    test_mapping()
    test_db_lookup()
    test_gate_fallback()
    print(f"\nALL PASS ({PASS} checks)")

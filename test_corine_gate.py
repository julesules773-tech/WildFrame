#!/usr/bin/env python3
"""Tests for the CORINE land-cover FIRMS gate.

Covers:
  * the pure class mapping in corine.py (burnable / cropland / non-burnable)
  * the batched PostGIS lookup in db.land_cover_codes_batch, gated on the
    `land_cover` table existing (it is populated by corine_import.py)

Run:  .venv/bin/python test_corine_gate.py
"""

import sys

import corine

PASS = 0


def check(name: str, cond: bool) -> None:
    global PASS
    if cond:
        PASS += 1
        print(f"  PASS {name}")
    else:
        print(f"  FAIL {name}")
        sys.exit(1)


def test_mapping() -> None:
    print("[mapping]")
    # Non-burnable: industrial / urban / water / bare
    check("121 industrial is not burnable", not corine.is_burnable("121"))
    check("112 urban is not burnable", not corine.is_burnable("112"))
    check("131 mineral extraction is not burnable", not corine.is_burnable("131"))
    check("512 water is not burnable", not corine.is_burnable("512"))
    check("335 glaciers is not burnable", not corine.is_burnable("335"))
    check("331 beaches is not burnable", not corine.is_burnable("331"))
    # Burnable: forest / shrub / grassland / agriculture / wetlands
    check("311 broad-leaved forest is burnable", corine.is_burnable("311"))
    check("312 coniferous forest is burnable", corine.is_burnable("312"))
    check("324 transitional woodland is burnable", corine.is_burnable("324"))
    check("321 natural grassland is burnable", corine.is_burnable("321"))
    check("231 pasture is burnable", corine.is_burnable("231"))
    check("411 inland marsh is burnable", corine.is_burnable("411"))
    check("412 peat bog is burnable", corine.is_burnable("412"))
    # Unknown / outside coverage: permissive (never silently delete)
    check("None (outside CORINE) is burnable", corine.is_burnable(None))
    check("empty string is burnable", corine.is_burnable(""))
    check("unknown code is burnable", corine.is_burnable("999"))
    # Cropland signals
    check("211 arable is cropland", corine.is_cropland("211"))
    check("213 rice is cropland", corine.is_cropland("213"))
    check("242 complex cultivation is cropland", corine.is_cropland("242"))
    check("231 pasture is NOT cropland", not corine.is_cropland("231"))
    check("311 forest is NOT cropland", not corine.is_cropland("311"))
    check("121 industrial is NOT cropland", not corine.is_cropland("121"))
    check("None is NOT cropland", not corine.is_cropland(None))


def test_db_lookup() -> None:
    print("[db.land_cover_codes_batch]")
    import db

    try:
        # Table exists only after corine_import.py has run.
        with db._conn() as conn:
            conn.execute("SELECT 1 FROM land_cover LIMIT 1").fetchone()
    except Exception:
        print("  SKIP land_cover table not loaded yet (run corine_import.py first)")
        return

    # Warsaw city centre (code 112 discontinuous urban fabric, non-burnable)
    # vs. Bialowieza forest (code 311/312, burnable).
    codes = db.land_cover_codes_batch([
        (52.2297, 21.0122),   # Warsaw
        (52.7419, 23.8590),   # Bialowieza Forest
    ])
    check("Warsaw resolves to a class", codes.get(0) is not None)
    check("Warsaw is non-burnable", codes.get(0) in corine.CORINE_NON_BURNABLE)
    check("Bialowieza resolves to a class", codes.get(1) is not None)
    check("Bialowieza is burnable", corine.is_burnable(codes.get(1)))

    # Point far outside coverage (e.g. mid-Pacific) → absent from result.
    far = db.land_cover_codes_batch([(-14.6, -175.0)])
    check("far point outside coverage is absent", 0 not in far)


if __name__ == "__main__":
    test_mapping()
    test_db_lookup()
    print(f"\nALL PASS ({PASS} checks)")

"""ESA WorldCover 2021 — class semantics for the global FIRMS gate.

WorldCover 2021 (10 m, raster) provides global land-cover classification.
This module maps WorldCover's 11 classes to burnability semantics that the
FIRMS filtering pipeline uses, following the same pattern as ``corine.py``.

The gate is used as a **fallback** when CORINE has no coverage: for points
inside Europe, CORINE (100 m vector) is authoritative; for points outside
Europe (the FIRMS API returns global data), WorldCover fills the gap.

WorldCover class codes (integer, the raster pixel value):

    10  Tree cover
    20  Shrubland
    30  Grassland
    40  Cropland
    50  Built-up
    60  Bare / sparse vegetation
    70  Snow and Ice
    80  Water
    90  Wetland
    95  Mangroves
   100  Moss and Lichen

Reference: https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/docs/WorldCover_PUM_V2.0.pdf
"""

from typing import Optional


# Non-burnable classes: built-up, barren, water, snow/ice.
# A FIRMS detection on any of these is almost certainly not a wildfire.
WORLDCOVER_NON_BURNABLE: set[int] = {
    50,   # built-up (urban, industrial, roads, airports)
    60,   # bare / sparse vegetation (desert, rock, sand)
    70,   # snow and ice
    80,   # water (ocean, lakes, rivers)
}

# Cropland: genuine fires, but frequently agricultural burning.
# Same semantics as CORINE_CROPLAND — the ag-burn downweight signal.
WORLDCOVER_CROPLAND: set[int] = {
    40,   # cropland (arable, orchards, plantations, pastures)
}


def is_burnable(code: Optional[int]) -> bool:
    """True when the WorldCover class is burnable vegetation (or unknown).

    Unknown/None (outside WorldCover coverage, e.g. sea cells) is treated
    as burnable — the gate is permissive where we have no data, so it can
    never silently delete fires in uncovered regions.  This matches the
    fail-open convention in ``corine.is_burnable``.
    """
    if code is None:
        return True
    return code not in WORLDCOVER_NON_BURNABLE


def is_cropland(code: Optional[int]) -> bool:
    """True when the WorldCover class is agricultural cropland."""
    if code is None:
        return False
    return code in WORLDCOVER_CROPLAND

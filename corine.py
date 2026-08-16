"""CORINE Land Cover 2018 — class semantics for the FIRMS gate.

CLC2018 (100 m, vector) is loaded into the `land_cover` PostGIS table by
``corine_import.py`` (currently Poland + margin, the beta footprint; the
import script is bbox-parameterised so coverage can be extended to the
rest of Europe in later runs).

This module maps CLC level-3 codes (``Code_18``) to burnability. It is
pure — no DB access — so the mapping is unit-testable and the gate logic
stays out of the request path.

The guiding rule (per the filtering plan): a FIRMS detection is treated
as a wildfire candidate only when it sits on burnable vegetation —
agricultural land (2xx), forest/shrub/herbaceous (3xx), or fuel-bearing
wetlands (4xx). Detections on artificial surfaces (1xx), water (5xx),
bare rock, beaches, salines or intertidal flats are industrial/urban/
aquatic noise and are excluded outright.
"""

from typing import Optional


# CLC2018 level-3 classes that are NOT burnable vegetation. A FIRMS
# detection on any of these is almost certainly an industrial heat source,
# urban reflection, water glare, or barren ground — not a wildfire.
CORINE_NON_BURNABLE: set[str] = {
    # 1xx — artificial surfaces
    "111",  # continuous urban fabric
    "112",  # discontinuous urban fabric
    "121",  # industrial or commercial units
    "122",  # road and rail networks
    "123",  # port areas
    "124",  # airports
    "131",  # mineral extraction sites
    "132",  # dump sites
    "133",  # construction sites
    "141",  # green urban areas
    "142",  # sport and leisure facilities
    # 3xx — non-vegetated / non-burnable semi-natural
    "331",  # beaches, dunes, sands
    "332",  # bare rock
    "335",  # glaciers and perpetual snow
    # 4xx — wetlands without burnable fuel
    "422",  # salines
    "423",  # intertidal flats
    # 5xx — water bodies
    "511",  # water courses
    "512",  # water bodies
    "521",  # coastal lagoons
    "522",  # estuaries
    "523",  # sea and ocean
}

# Cropland classes: genuine fires, but frequently agricultural burning
# (stubble, slash) rather than wildfires worth full wildfire weight. These
# get the downweight treatment in the FIRMS pass (Step 3 of the plan),
# while non-burnable classes are excluded outright (Step 1).
CORINE_CROPLAND: set[str] = {
    "211",  # non-irrigated arable land
    "212",  # permanently irrigated land
    "213",  # rice fields
    "221",  # vineyards
    "222",  # fruit trees and berry plantations
    "223",  # olive groves
    "241",  # annual crops associated with permanent crops
    "242",  # complex cultivation patterns
    "243",  # land principally occupied by agriculture
}


def is_burnable(code: Optional[str]) -> bool:
    """True when the CLC class is burnable vegetation (or unknown).

    Unknown/None (outside CORINE coverage, e.g. non-European fires) is
    treated as burnable — the gate is permissive where we have no data,
    so it can never silently delete fires in uncovered regions.
    """
    if not code:
        return True
    return code not in CORINE_NON_BURNABLE


def is_cropland(code: Optional[str]) -> bool:
    """True when the CLC class is agricultural cropland (ag-burning signal)."""
    if not code:
        return False
    return code in CORINE_CROPLAND

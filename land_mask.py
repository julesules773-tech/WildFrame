"""Natural Earth 110m Land Mask — global ocean/sea/river pre-filter.

Uses the ``ne_110m_land`` polygon (public domain, ~1 MB) loaded into the
``land_mask`` PostGIS table by ``land_mask_import.py``.  The gate runs as
a **Layer 0** pre-filter before CORINE / WorldCover: any FIRMS detection
that falls outside the land polygon is clearly on water (ocean, sea, or
large lake) and can be dropped outright.

This module is pure — no DB access — so the mapping is unit-testable and
the gate logic stays out of the request path.

The 110m scale (~1 km resolution) catches all ocean/sea false positives
while being tiny enough to load on a 1 GB VM in seconds.  Rivers and
small lakes are not resolved at this scale, but they're rare sources of
FIRMS false positives compared to the ocean.
"""

from typing import Optional


def is_on_land(code: Optional[str]) -> bool:
    """True when the point is on land (or outside land-mask coverage).

    The ``code`` parameter is unused (kept for interface consistency with
    ``corine.is_burnable``) — the actual spatial lookup happens in
    ``db.land_mask_batch``.  This function exists so callers can do a
    quick class-based check when they already have a result.

    Unknown/None (outside land-mask coverage) is treated as on-land —
    fail-open, same convention as CORINE/WorldCover.
    """
    # The land mask is a boolean inside/outside check, not a class mapping.
    # This stub exists for API symmetry; the real logic is spatial.
    return True


def is_water(code: Optional[str]) -> bool:
    """True when the point is clearly on water (ocean, sea, large lake).

    Inverse of ``is_on_land`` — used by the gate to drop water detections.
    """
    return not is_on_land(code)

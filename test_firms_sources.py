"""
Tests for multi-satellite FIRMS fetching (SNPP + NOAA-20 + NOAA-21 merge).

The FIRMS pass fetches every VIIRS 375m instrument and merges them so
fires any single satellite missed are still caught — the combined view
NASA's FIRMS map shows. These tests verify the merge logic in
``nasa_firms.fetch_global_fires`` without hitting the network.
"""
from __future__ import annotations

import sys
from unittest.mock import patch

import nasa_firms
from nasa_firms import FIRMSHotspot


def _hs(lat: float, lon: float, conf: str = "nominal", sat: str = "N",
        day: str = "2026-08-17", tm: str = "1200") -> FIRMSHotspot:
    return FIRMSHotspot(
        latitude=lat, longitude=lon, brightness=320.0, scan=1.0, track=1.0,
        acq_date=day, acq_time=tm, satellite=sat, instrument="VIIRS",
        confidence=conf, version="1", frp=5.0, daynight="D",
    )


def _fake_fetch(records_by_source: dict):
    """Return a fetch_fire_data stand-in that returns per-source records."""
    def _fetch(api_key, bbox, source, day_range):
        return list(records_by_source.get(source, []))
    return _fetch


PASS = 0
FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}")


def run():
    print("== DEFAULT_SOURCES covers all VIIRS 375m instruments ==")
    check("SNPP in DEFAULT_SOURCES",
          nasa_firms.VIIRS_SNPP_NRT in nasa_firms.DEFAULT_SOURCES)
    check("NOAA-20 in DEFAULT_SOURCES",
          nasa_firms.VIIRS_NOAA20_NRT in nasa_firms.DEFAULT_SOURCES)
    check("NOAA-21 in DEFAULT_SOURCES",
          nasa_firms.VIIRS_NOAA21_NRT in nasa_firms.DEFAULT_SOURCES)
    check("MODIS excluded (1km, coarser)",
          nasa_firms.MODIS_NRT not in nasa_firms.DEFAULT_SOURCES)

    print("== merge across sources ==")
    records_by_source = {
        nasa_firms.VIIRS_SNPP_NRT: [_hs(51.0, 19.0, sat="N")],
        nasa_firms.VIIRS_NOAA20_NRT: [_hs(51.1, 19.1, sat="20")],
        nasa_firms.VIIRS_NOAA21_NRT: [_hs(51.2, 19.2, sat="21")],
    }
    with patch.object(
        nasa_firms, "fetch_fire_data",
        side_effect=lambda key, bbox, src, dr: list(records_by_source.get(src, [])),
    ) as m:
        merged = nasa_firms.fetch_global_fires(
            "key", day_range=2, min_confidence="nominal",
            sources=nasa_firms.DEFAULT_SOURCES,
        )
    check("3 sources -> 3 merged detections", len(merged) == 3)
    check("day_range forwarded to every source fetch",
          m.call_count == 3 and all(c.args[3] == 2 for c in m.call_args_list))
    check("each source fetched exactly once",
          sorted(c.args[2] for c in m.call_args_list) == sorted(nasa_firms.DEFAULT_SOURCES))
    sats = sorted(h.satellite for h in merged)
    check("keeps per-source satellite attribution", sats == ["20", "21", "N"])

    print("== single-source default (back-compat) ==")
    fake1 = _fake_fetch({nasa_firms.VIIRS_SNPP_NRT: [_hs(51.0, 19.0)]})
    with patch.object(nasa_firms, "fetch_fire_data", side_effect=fake1):
        one = nasa_firms.fetch_global_fires("key", day_range=1)
    check("no sources arg -> one fetch, one detection", len(one) == 1)

    print("== confidence filter applied across the merged set ==")
    mixed = _fake_fetch({
        nasa_firms.VIIRS_SNPP_NRT: [
            _hs(51.0, 19.0, conf="low"), _hs(51.1, 19.1, conf="nominal"),
        ],
        nasa_firms.VIIRS_NOAA21_NRT: [_hs(51.2, 19.2, conf="high")],
    })
    with patch.object(nasa_firms, "fetch_fire_data", side_effect=mixed):
        nom = nasa_firms.fetch_global_fires(
            "key", day_range=1, min_confidence="nominal",
            sources=nasa_firms.DEFAULT_SOURCES,
        )
    check("nominal filter drops 'low' across sources", len(nom) == 2)
    with patch.object(nasa_firms, "fetch_fire_data", side_effect=mixed):
        allc = nasa_firms.fetch_global_fires(
            "key", day_range=1, min_confidence="low",
            sources=nasa_firms.DEFAULT_SOURCES,
        )
    check("min_confidence=low keeps everything", len(allc) == 3)

    print()
    print(f"{'ALL PASS' if FAIL == 0 else f'{FAIL} FAILURES'} "
          f"({PASS} checks)")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(run())

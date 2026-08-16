#!/usr/bin/env python3
"""Download CORINE Land Cover 2018 (vector) into PostGIS.

Pulls the official EEA "Corine/CLC2018_LAEA" layer (ETRS89-LAEA, EPSG:3035)
via the ArcGIS REST query API for a given bounding box, paginating at the
service's maxRecordCount (1000), then loads each GeoJSON page into a
PostGIS table with ogr2ogr. The service returns geometry already
reprojected to WGS84 (EPSG:4326), so no on-the-fly reprojection is needed
for the load.

Usage:
    python corine_import.py --bbox 4508720,2769703,5446734,3744227 \
        --out /tmp/corine_pl --workers 4 [--table land_cover] [--limit N]

The bbox is in EPSG:3035 (e.g. `ST_Transform` a WGS84 envelope with
PostGIS to get it). --limit caps the number of pages (debug runs).
"""

import argparse
import json
import os
from typing import Optional
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = (
    "https://image.discomap.eea.europa.eu/arcgis/rest/services/"
    "Corine/CLC2018_LAEA/MapServer/0/query"
)
PAGE_SIZE = 1000
RETRIES = 4
BACKOFF_S = 8.0
RESUME_PASSES = 3


def query(params: dict) -> bytes:
    """One GET against the CLC2018 query endpoint, with retries."""
    url = BASE_URL + "?" + urllib.parse.urlencode(params)
    last_exc = None
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "pyrae-corine-import/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001 — network layer, retry everything
            last_exc = exc
            if attempt < RETRIES - 1:
                time.sleep(BACKOFF_S * (attempt + 1))
    raise RuntimeError(f"query failed after {RETRIES} attempts: {last_exc}")


def fetch_count(bbox: list[float]) -> int:
    last_exc: Optional[Exception] = None
    for _ in range(RETRIES):
        try:
            raw = query({
                "where": "1=1",
                "geometry": json.dumps({
                    "xmin": bbox[0], "ymin": bbox[1], "xmax": bbox[2], "ymax": bbox[3],
                    "spatialReference": {"wkid": 3035},
                }),
                "geometryType": "esriGeometryEnvelope",
                "spatialRel": "esriSpatialRelIntersects",
                "returnCountOnly": "true",
                "f": "json",
            })
            return int(json.loads(raw)["count"])
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            last_exc = exc
            time.sleep(BACKOFF_S)
    raise RuntimeError(f"count query failed: {last_exc}")


def fetch_page(bbox: list[float], offset: int, out_path: str) -> None:
    # Resume: never re-download a page that already landed on disk.
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return
    params = {
        "where": "1=1",
        "outFields": "CODE_18,AREA_HA",
        "returnGeometry": "true",
        "geometry": json.dumps({
            "xmin": bbox[0], "ymin": bbox[1], "xmax": bbox[2], "ymax": bbox[3],
            "spatialReference": {"wkid": 3035},
        }),
        "geometryType": "esriGeometryEnvelope",
        "spatialRel": "esriSpatialRelIntersects",
        "resultOffset": str(offset),
        "resultRecordCount": str(PAGE_SIZE),
        "f": "geojson",
    }
    data = query(params)
    with open(out_path, "wb") as f:
        f.write(data)
    n = len(json.loads(data).get("features", []))
    if n == 0:
        raise RuntimeError(f"page offset={offset} came back empty (service drift?)")


def load_pages(out_dir: str, table: str) -> None:
    pages = sorted(p for p in os.listdir(out_dir) if p.startswith("page_") and p.endswith(".geojson"))
    if not pages:
        raise SystemExit("no page files found — nothing to load")
    print(f"[corine] loading {len(pages)} page(s) into {table}")
    for i, page in enumerate(pages):
        path = os.path.join(out_dir, page)
        cmd = [
            "ogr2ogr", "-f", "PostgreSQL", f"PG:dbname=wildframe",
            "-nln", table,
            "-overwrite" if i == 0 else "-append",
            path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        if (i + 1) % 25 == 0 or i + 1 == len(pages):
            print(f"[corine]   loaded {i + 1}/{len(pages)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bbox", required=True,
                        help="EPSG:3035 envelope as xmin,ymin,xmax,ymax")
    parser.add_argument("--out", default="/tmp/corine_import")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--table", default="land_cover")
    parser.add_argument("--limit", type=int, default=0,
                        help="cap the number of pages (debug)")
    args = parser.parse_args()

    bbox = [float(v) for v in args.bbox.split(",")]
    os.makedirs(args.out, exist_ok=True)

    print(f"[corine] counting features in bbox {bbox} …")
    total = fetch_count(bbox)
    pages = (total + PAGE_SIZE - 1) // PAGE_SIZE
    if args.limit:
        pages = min(pages, args.limit)
    print(f"[corine] {total} features → {pages} page(s) of {PAGE_SIZE}")

    # Download with up to RESUME_PASSES full passes: the EEA service
    # intermittently 503s under sustained load, so a failed page is just
    # retried (with the rest of its pass) rather than aborting the import.
    offsets = list(range(0, pages * PAGE_SIZE, PAGE_SIZE))
    for pass_no in range(1, RESUME_PASSES + 1):
        missing = [
            off for off in offsets
            if not (os.path.exists(os.path.join(args.out, f"page_{off:06d}.geojson"))
                    and os.path.getsize(os.path.join(args.out, f"page_{off:06d}.geojson")) > 0)
        ]
        if not missing:
            break
        print(f"[corine] pass {pass_no}/{RESUME_PASSES}: {len(missing)} page(s) to fetch")
        done_before = len(offsets) - len(missing)
        done = done_before
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(fetch_page, bbox, off, os.path.join(args.out, f"page_{off:06d}.geojson")): off
                for off in missing
            }
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as exc:  # noqa: BLE001
                    print(f"[corine] FAILED page offset={futures[fut]}: {exc}")
                done += 1
                if done % 50 == 0 or done == len(offsets):
                    print(f"[corine]   downloaded {done}/{len(offsets)}")
        if pass_no < RESUME_PASSES:
            time.sleep(BACKOFF_S)
    else:
        missing = [
            off for off in offsets
            if not (os.path.exists(os.path.join(args.out, f"page_{off:06d}.geojson"))
                    and os.path.getsize(os.path.join(args.out, f"page_{off:06d}.geojson")) > 0)
        ]
        if missing:
            print(f"[corine] ERROR: still {len(missing)} page(s) missing after {RESUME_PASSES} passes")
            return 1

    load_pages(args.out, args.table)
    print(f"[corine] done — table `{args.table}`")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Download ESA WorldCover 2021 tiles and load into PostGIS as vector polygons.

Since the target PostGIS installation may lack GDAL/raster support
(``ST_FromGDALRaster`` unavailable), this script extracts vector polygons
from the 10 m GeoTIFF tiles using ``rasterio.features.shapes`` and stores
them in a PostGIS geometry table (``worldcover_polygons``) with a spatial
index.

Point queries use:

    SELECT class_code FROM worldcover_polygons
    WHERE ST_Contains(geom, ST_SetSRID(ST_MakePoint(lon, lat), 4326))
    LIMIT 1;

Usage
    python worldcover_import.py \\
        --bbox 12.0,46.9,26.9,56.8 \\
        [--table worldcover_polygons] [--year 2021] [--version v200] \\
        [--simplify 0.001]

The default bbox covers Poland + border margin.  For global coverage,
use --bbox -180,-60,180,85 or a subset.

Requires: rasterio, shapely, psycopg (all already in the project).
"""

import argparse
import gc
import json
import logging
import tempfile
import os
import sys
import tempfile
from urllib.error import HTTPError
from urllib.request import Request, urlopen

logger = logging.getLogger("worldcover_import")

S3_BASE = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
DEFAULT_YEAR = 2021
DEFAULT_VERSION = "v200"
DEFAULT_TABLE = "worldcover_polygons"

# WorldCover tiles are 3° × 3° GeoTIFFs named like:
#   ESA_WorldCover_10m_2021_v200_N51E018_Map.tif
TILE_DEG = 3.0

# WorldCover class codes → our schema
WORLDCOVER_CLASSES = {
    10: "tree_cover",
    20: "shrubland",
    30: "grassland",
    40: "cropland",
    50: "built_up",
    60: "bare",
    70: "snow_ice",
    80: "water",
    90: "wetland",
    95: "mangroves",
    100: "moss_lichen",
}


def _tile_keys_for_bbox(bbox: tuple[float, float, float, float],
                        year: int, version: str) -> list[str]:
    """Return S3 keys for all WorldCover tiles that intersect *bbox*.

    bbox = (west, south, east, north) in WGS84.
    """
    w, s, e, n = bbox
    keys: list[str] = []
    lat = int(s // TILE_DEG) * TILE_DEG
    while lat < n:
        lon = int(w // TILE_DEG) * TILE_DEG
        while lon < e:
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            lat_abs = abs(int(lat))
            lon_abs = abs(int(lon))
            fname = (f"ESA_WorldCover_10m_{year}_{version}"
                     f"_{ns}{lat_abs:02d}{ew}{lon_abs:03d}_Map.tif")
            keys.append(f"{version}/{year}/map/{fname}")
            lon += TILE_DEG
        lat += TILE_DEG
    return keys


def _download_tile(key: str) -> bytes | None:
    """Download a single tile from S3 (public, no auth needed).

    Returns None for 404 (tile doesn't exist — e.g. open ocean) so the
    caller can skip it instead of treating it as a hard error.
    """
    url = f"{S3_BASE}/{key}"
    logger.info("downloading %s …", key)
    req = Request(url, headers={"User-Agent": "pyrae-worldcover-import/1.0"})
    try:
        with urlopen(req, timeout=120) as resp:
            data = resp.read()
    except HTTPError as exc:
        if exc.code == 404:
            logger.info("  %s — 404 (no tile, likely ocean), skipping", key)
            return None
        raise
    # Validate: GeoTIFF starts with either 'II' (little-endian) or 'MM' (big-endian)
    if not data or data[:2] not in (b'II', b'MM'):
        snippet = data[:200].decode('utf-8', errors='replace') if data else '<empty>'
        raise RuntimeError(
            f"Downloaded data for {key} is not a valid GeoTIFF "
            f"({len(data)} bytes, starts with: {snippet!r})"
        )
    return data


def _extract_polygons(tiff_path: str, tile_key: str,
                      simplify: float = 0.005) -> list[tuple[int, str]]:
    """Extract vector polygons from a GeoTIFF tile on disk.

    Uses a temp file instead of MemoryFile to avoid holding both the raw
    bytes AND the rasterio array in memory — the file is read directly
    by rasterio, keeping peak RAM under ~300 MB per tile on a 1 GB VM.

    Returns list of (class_code, wkt_geometry) pairs.
    Polygons are simplified to reduce geometry count.
    """
    import rasterio
    from rasterio.features import shapes
    from shapely.geometry import shape

    # Open directly from disk — avoids loading the raw bytes into Python
    # memory (the 40 MB GeoTIFF stays on disk; rasterio memory-maps it).
    with rasterio.open(tiff_path) as src:
        data = src.read(1)  # band 1 = land cover class codes
        mask = src.dataset_mask()  # alpha mask (0=nodata, 255=valid)
        transform = src.transform

        polygons: list[tuple[int, str]] = []
        for geom, value in shapes(
            data,
            transform=transform,
            mask=mask > 0,  # only polygonize valid pixels
        ):
            class_code = int(value)
            if class_code == 0:
                continue  # nodata

            shp = shape(geom)
            if shp.is_empty:
                continue

            # Simplify to reduce vertex count
            if simplify > 0:
                shp = shp.simplify(simplify, preserve_topology=True)

            if shp.is_empty:
                continue

            # Convert MultiPolygon to individual Polygons
            if shp.geom_type == "MultiPolygon":
                for part in shp.geoms:
                    polygons.append((class_code, part.wkt))
            else:
                polygons.append((class_code, shp.wkt))

        # Free the raster data before returning
        del data, mask

        logger.info("  extracted %d polygons from %s", len(polygons), tile_key)
        return polygons


# -----------------------------------------------------------------------
# Progress tracking (--resume support)
# -----------------------------------------------------------------------
DEFAULT_PROGRESS_FILE = ".worldcover_progress.json"


def _load_progress(path: str) -> dict:
    """Load the progress file (tile_key → 'done' | 'error')."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f).get("tiles", {})
    except Exception:
        return {}


def _save_progress(path: str, tiles_progress: dict,
                   bbox: tuple, table: str) -> None:
    """Atomically write the progress file."""
    data = {
        "bbox": ",".join(str(v) for v in bbox),
        "table": table,
        "tiles": tiles_progress,
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)  # atomic on POSIX


def _ensure_table(cur, table: str, is_resume: bool) -> None:
    """Create the table on first run, or just ensure it exists on resume.

    On first run: DROP + CREATE (clean slate).
    On resume:    CREATE IF NOT EXISTS (keep existing rows, append new ones).
    """
    cur.execute("CREATE EXTENSION IF NOT EXISTS postgis")
    if is_resume:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id serial PRIMARY KEY,
                class_code smallint NOT NULL,
                tile_key text NOT NULL,
                geom geometry(Polygon, 4326) NOT NULL
            )
        """)
    else:
        cur.execute(f"DROP TABLE IF EXISTS {table}")
        cur.execute(f"""
            CREATE TABLE {table} (
                id serial PRIMARY KEY,
                class_code smallint NOT NULL,
                tile_key text NOT NULL,
                geom geometry(Polygon, 4326) NOT NULL
            )
        """)


def _insert_polygons(cur, table: str,
                     polygons: list[tuple[int, str, str]]) -> None:
    """Batch-insert polygons.  Skips duplicates via tile_key uniqueness
    so resume runs never create double rows."""
    BATCH = 2000
    for i in range(0, len(polygons), BATCH):
        batch = polygons[i:i + BATCH]
        values = []
        params = []
        for class_code, wkt, tile_key in batch:
            values.append("( %s, %s, ST_GeomFromText(%s, 4326) )")
            params.extend([class_code, tile_key, wkt])
        sql = (f"INSERT INTO {table} (class_code, tile_key, geom) "
               f"VALUES " + ", ".join(values))
        cur.execute(sql, params)
        if (i + BATCH) % (BATCH * 5) == 0 or i + BATCH >= len(polygons):
            logger.info("  inserted %d/%d",
                        min(i + BATCH, len(polygons)), len(polygons))


def _ensure_indexes(cur, table: str) -> None:
    """Create spatial + class indexes (IF NOT EXISTS for resume)."""
    cur.execute(
        f"CREATE INDEX IF NOT EXISTS {table}_geom_idx "
        f"ON {table} USING GIST (geom)"
    )
    cur.execute(
        f"CREATE INDEX IF NOT EXISTS {table}_class_idx "
        f"ON {table} (class_code)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--bbox", default="12.0,46.9,26.9,56.8",
        help="west,south,east,north WGS84 (default: Poland + margin)",
    )
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument(
        "--simplify", type=float, default=0.005,
        help="polygon simplification tolerance in degrees (default: 0.005 ≈ 555 m)"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="skip tiles already marked done in the progress file")
    parser.add_argument("--progress-file", default=DEFAULT_PROGRESS_FILE,
                        help=f"progress file path (default: {DEFAULT_PROGRESS_FILE})")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="[%(name)s] %(message)s",
    )

    bbox = tuple(float(v) for v in args.bbox.split(","))
    if len(bbox) != 4:
        sys.exit("--bbox must be west,south,east,north")

    keys = _tile_keys_for_bbox(bbox, args.year, args.version)
    if not keys:
        sys.exit("no tiles needed for this bbox")

    # --- Resume support ---
    progress: dict[str, str] = {}
    if args.resume:
        progress = _load_progress(args.progress_file)
        done_count = sum(1 for v in progress.values() if v == "done")
        print(f"[worldcover] resume: {done_count}/{len(keys)} tiles already done")

    print(f"[worldcover] need {len(keys)} tile(s) for bbox {bbox}")

    # --- Process tiles one at a time ---
    # Each tile goes through: download → extract → insert.  Progress is
    # written after each successful tile so a crash mid-import never
    # loses completed work.
    import psycopg

    conninfo = os.environ.get(
        "WILDFRAME_DATABASE_URL",
        "host=localhost dbname=wildframe",
    )
    table = args.table
    table_created = False
    total_polygons = 0

    for i, key in enumerate(keys, 1):
        label = f"[{i}/{len(keys)}]"

        # Skip already-completed tiles
        if progress.get(key) == "done":
            logger.info("%s %s — already done, skipping", label, key)
            continue

        print(f"{label} {key}")

        tiff_path = None
        try:
            # 1. Download to temp file (avoids holding raw bytes + rasterio array in RAM)
            data = _download_tile(key)
            if data is None:
                # Tile doesn't exist (ocean / no data) — mark done so
                # --resume never retries it.
                progress[key] = "done"
                _save_progress(args.progress_file, progress, bbox, table)
                continue

            # Write to temp file, then free the raw bytes immediately
            with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                tmp.write(data)
                tiff_path = tmp.name
            del data  # free ~40 MB of raw GeoTIFF bytes

            # 2. Extract polygons (reads from disk, not from memory)
            polys = _extract_polygons(tiff_path, key, simplify=args.simplify)
            polygons = [(cc, wkt, key) for cc, wkt in polys]
            total_polygons += len(polygons)

            # 3. Insert into PostGIS
            with psycopg.connect(conninfo) as conn:
                with conn.cursor() as cur:
                    if not table_created:
                        _ensure_table(cur, table, args.resume)
                        table_created = True
                    if polygons:
                        _insert_polygons(cur, table, polygons)
                    _ensure_indexes(cur, table)
                    conn.commit()

            # 4. Mark done
            progress[key] = "done"
            _save_progress(args.progress_file, progress, bbox, table)
            logger.info("%s %s — %d polygons inserted", label, key, len(polys))

        except Exception as exc:
            logger.error("FAILED %s %s: %s", label, key, exc)
            progress[key] = "error"
            _save_progress(args.progress_file, progress, bbox, table)
            continue
        finally:
            # Clean up temp file and force GC to free rasterio/shapely memory
            if tiff_path and os.path.exists(tiff_path):
                os.unlink(tiff_path)
            polys = None
            polygons = None
            gc.collect()

    print(f"[worldcover] done — {total_polygons} polygons from "
          f"{len(keys)} tile(s) in '{table}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())

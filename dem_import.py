#!/usr/bin/env python3
"""Download Copernicus GLO-30 DEM tiles, compute slope & aspect, load into PostGIS.

Downloads tiles from the public AWS S3 bucket, reads them with tifffile,
computes terrain attributes (elevation, slope, aspect) at a configurable
resolution, and bulk-loads into a PostGIS table via COPY.

Requires: numpy, tifffile, psycopg (all already in the project venv).

Usage
    python dem_import.py --bbox=-25,35,45,72 --verbose
    python dem_import.py --bbox=-10,45,30,55 --table dem_europe --verbose

The default bbox covers mainland Europe.  Use --bbox for any region.
SLOPE/ASPECT are stored in degrees (aspect: 0=N, 90=E, 180=S, 270=W).
"""

import argparse
import gc
import gzip
import io
import logging
import math
import os
import sys
import tempfile
from urllib.error import HTTPError
from urllib.request import Request, urlopen

logger = logging.getLogger("dem_import")

S3_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"
DEFAULT_TABLE = "dem_terrain"

# Tile size: Copernicus GLO-30 tiles are 1° × 1°
TILE_DEG = 1.0


# ---------------------------------------------------------------------------
# Tile enumeration
# ---------------------------------------------------------------------------

def _tile_keys_for_bbox(west: float, south: float, east: float, north: float):
    """Yield S3 object keys for all 1°×1° tiles intersecting the bbox."""
    lon_start = int(math.floor(west))
    lon_end = int(math.ceil(east))
    lat_start = int(math.floor(south))
    lat_end = int(math.ceil(north))

    for lat in range(lat_start, lat_end):
        for lon in range(lon_start, lon_end):
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            lat_abs = abs(lat)
            lon_abs = abs(lon)
            name = f"Copernicus_DSM_COG_10_{ns}{lat_abs:02d}_00_{ew}{lon_abs:03d}_00_DEM"
            key = f"{name}/{name}.tif"
            yield key, lat, lon


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _download_tile(key: str) -> bytes | None:
    """Download a single COG tile from S3. Returns raw bytes or None (404)."""
    url = f"{S3_BASE}/{key}"
    req = Request(url, headers={"User-Agent": "wildframe-dem-import/1.0"})
    try:
        with urlopen(req, timeout=120) as resp:
            return resp.read()
    except HTTPError as exc:
        if exc.code == 404:
            logger.info("  %s — 404 (no tile, likely ocean), skipping", key)
            return None
        raise


# ---------------------------------------------------------------------------
# Terrain computation (slope & aspect from elevation array)
# ---------------------------------------------------------------------------

def _compute_slope_aspect(elev: "np.ndarray", cell_size_m: float):
    """Compute slope (%) and aspect (degrees) from an elevation array.

    Uses Horn's method (3×3 window).  Returns (slope_pct, aspect_deg) arrays
    of the same shape as *elev*, with border pixels set to 0.

    elev : 2-D numpy array of metres
    cell_size_m : pixel spacing in metres (~30 for GLO-30)
    """
    import numpy as np

    # Pad with edge values to handle borders
    pad = np.pad(elev, 1, mode="edge")

    # Horn's coefficients (Zimmermann 1981)
    dz_dx = ((pad[2:, 2:] + 2 * pad[1:-1, 2:] + pad[:-2, 2:])
             - (pad[2:, :-2] + 2 * pad[1:-1, :-2] + pad[:-2, :-2])) / (8.0 * cell_size_m)
    dz_dy = ((pad[2:, 2:] + 2 * pad[2:, 1:-1] + pad[2:, :-2])
             - (pad[:-2, 2:] + 2 * pad[:-2, 1:-1] + pad[:-2, :-2])) / (8.0 * cell_size_m)

    # Slope in percent (rise/run × 100)
    slope_pct = np.sqrt(dz_dx ** 2 + dz_dy ** 2) * 100.0

    # Aspect: compass bearing (0=N, 90=E, 180=S, 270=W)
    # dz_dy positive = slope rises toward north, dz_dx positive = slope rises toward east
    aspect_rad = np.arctan2(-dz_dx, dz_dy)  # negative dx because north = 0
    aspect_deg = np.degrees(aspect_rad) % 360.0

    # Border pixels (first/last row/col) → 0 slope
    slope_pct[0, :] = 0; slope_pct[-1, :] = 0
    slope_pct[:, 0] = 0; slope_pct[:, -1] = 0
    aspect_deg[0, :] = 0; aspect_deg[-1, :] = 0
    aspect_deg[:, 0] = 0; aspect_deg[:, -1] = 0

    return slope_pct, aspect_deg


# ---------------------------------------------------------------------------
# Process one tile → list of rows
# ---------------------------------------------------------------------------

def _process_tile(tile_key: str, tile_lat: int, tile_lon: int,
                  sample_res: float = 0.001):
    """Download, decompress, compute terrain, and return rows for COPY.

    Returns list of (lon, lat, elevation, slope_pct, aspect_deg) tuples,
    or None if the tile is ocean/missing.
    """
    import numpy as np
    import tifffile

    logger.info("  downloading %s …", tile_key)
    raw = _download_tile(tile_key)
    if raw is None:
        return None

    logger.info("  decompressing %s (%d KB) …", tile_key, len(raw) // 1024)

    # tifffile can read from a BytesIO — it handles COG/DEFLATE natively
    buf = io.BytesIO(raw)
    with tifffile.TiffFile(buf) as tif:
        elev = tif.pages[0].asarray().astype(np.float32)

    # Free the raw bytes immediately
    del raw, buf
    gc.collect()

    # Handle nodata — Copernicus uses -32768 or 0 for ocean
    nodata = -32768.0
    mask = (elev > nodata + 1) & (elev != 0)

    if not mask.any():
        logger.info("  %s — all nodata (ocean), skipping", tile_key)
        return None

    h, w = elev.shape
    pixel_size_deg = TILE_DEG / h  # ~0.000278° for 3600px tiles

    # Compute slope and aspect at full resolution
    # For 30m pixels, cell_size ≈ 30m (varies slightly with latitude)
    cell_size_m = 30.0  # approximate
    logger.info("  computing slope & aspect at full resolution (%d×%d) …", h, w)
    slope_full, aspect_full = _compute_slope_aspect(elev, cell_size_m)

    # Sample at coarser resolution to keep row count manageable
    step = max(1, int(sample_res / pixel_size_deg))
    logger.info("  sampling at ~%.4f° resolution (step=%d) …", sample_res, step)

    rows = []
    for row in range(0, h, step):
        lat = tile_lat + 1.0 - (row + 0.5) * pixel_size_deg  # center of pixel
        for col in range(0, w, step):
            lon = tile_lon + (col + 0.5) * pixel_size_deg

            if not mask[row, col]:
                continue

            e = float(elev[row, col])
            s = float(slope_full[row, col])
            a = float(aspect_full[row, col])

            # Clamp to reasonable ranges
            s = max(0.0, min(s, 9000.0))   # slope 0-9000% (89°)
            a = a % 360.0

            rows.append((lon, lat, e, s, a))

    # Free large arrays
    del elev, slope_full, aspect_full, mask
    gc.collect()

    logger.info("  %s → %d terrain points", tile_key, len(rows))
    return rows


# ---------------------------------------------------------------------------
# PostGIS loading
# ---------------------------------------------------------------------------

def _ensure_table(table: str, conn):
    """Create the dem_terrain table if it doesn't exist."""
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id        BIGSERIAL PRIMARY KEY,
                geom      geometry(Point, 4326) NOT NULL,
                elevation DOUBLE PRECISION NOT NULL DEFAULT 0,
                slope_pct DOUBLE PRECISION NOT NULL DEFAULT 0,
                aspect_deg DOUBLE PRECISION NOT NULL DEFAULT 0
            );
        """)
        # Check if spatial index exists
        cur.execute(f"""
            SELECT indexname FROM pg_indexes
            WHERE tablename = %s AND indexname = %s
        """, (table, f"{table}_geom_idx"))
        if not cur.fetchone():
            logger.info("  creating spatial index on %s …", table)
            cur.execute(f"""
                CREATE INDEX {table}_geom_idx ON {table}
                USING GIST (geom);
            """)
        # Slope index for terrain queries
        cur.execute(f"""
            SELECT indexname FROM pg_indexes
            WHERE tablename = %s AND indexname = %s
        """, (table, f"{table}_slope_idx"))
        if not cur.fetchone():
            cur.execute(f"""
                CREATE INDEX {table}_slope_idx ON {table} (slope_pct);
            """)
        conn.commit()


def _load_rows(rows: list, table: str, conn):
    """Bulk-insert rows via COPY into temp table, then convert to geometry."""
    import io as _io

    if not rows:
        return 0

    # Step 1: COPY raw lon/lat values into a temp table (fast bulk load)
    buf = _io.StringIO()
    for lon, lat, elev, slope, aspect in rows:
        buf.write(f"{lon}\t{lat}\t{elev}\t{slope}\t{aspect}\n")
    buf.seek(0)

    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS _dem_tmp")
        cur.execute(f"""
            CREATE TEMP TABLE _dem_tmp (
                lon DOUBLE PRECISION,
                lat DOUBLE PRECISION,
                elevation DOUBLE PRECISION,
                slope_pct DOUBLE PRECISION,
                aspect_deg DOUBLE PRECISION
            )
        """)
        cur.copy_expert(
            "COPY _dem_tmp FROM STDIN",
            buf,
        )

        # Step 2: Insert into main table with geometry
        cur.execute(f"""
            INSERT INTO {table} (geom, elevation, slope_pct, aspect_deg)
            SELECT ST_SetSRID(ST_MakePoint(lon, lat), 4326),
                   elevation, slope_pct, aspect_deg
            FROM _dem_tmp
        """)
        cur.execute("DROP TABLE IF EXISTS _dem_tmp")
    conn.commit()
    return len(rows)


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

DEFAULT_PROGRESS_FILE = ".dem_progress.json"


def _load_progress(path: str) -> set:
    import json
    if os.path.exists(path):
        with open(path) as f:
            return set(json.load(f))
    return set()


def _save_progress(path: str, done: set):
    import json
    with open(path, "w") as f:
        json.dump(sorted(done), f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bbox", default="-25,35,45,72",
                        help="west,south,east,north (default: all of Europe)")
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--sample-res", type=float, default=0.001,
                        help="Output resolution in degrees (~111m at equator)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel download workers (default: 1)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip tiles already imported")
    parser.add_argument("--progress-file", default=DEFAULT_PROGRESS_FILE)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    # Parse bbox
    parts = [float(x.strip()) for x in args.bbox.split(",")]
    if len(parts) != 4:
        parser.error("--bbox must be west,south,east,north")
    west, south, east, north = parts
    logger.info("[dem] bbox: (%.1f, %.1f, %.1f, %.1f)", west, south, east, north)

    # Enumerate tiles
    tiles = list(_tile_keys_for_bbox(west, south, east, north))
    logger.info("[dem] need %d tile(s) for bbox", len(tiles))

    # Resume support
    progress_path = os.path.join(os.path.dirname(__file__) or ".", args.progress_file)
    done_tiles = _load_progress(progress_path) if args.resume else set()
    if done_tiles:
        logger.info("[dem] resume: %d/%d tiles already done", len(done_tiles), len(tiles))

    # Connect to Postgres
    conninfo = os.environ.get("WILDFRAME_DATABASE_URL", "")
    if not conninfo:
        logger.error("WILDFRAME_DATABASE_URL not set")
        sys.exit(1)

    import psycopg
    with psycopg.connect(conninfo) as conn:
        _ensure_table(args.table, conn)

        total_rows = 0
        tiles_done = 0

        for idx, (key, tile_lat, tile_lon) in enumerate(tiles, 1):
            tile_id = f"{tile_lat}_{tile_lon}"

            if tile_id in done_tiles:
                logger.info("[dem] [%d/%d] %s — already done, skipping",
                            idx, len(tiles), key)
                continue

            logger.info("[dem] [%d/%d] %s", idx, len(tiles), key)

            try:
                rows = _process_tile(key, tile_lat, tile_lon,
                                     sample_res=args.sample_res)
            except Exception as exc:
                logger.warning("  ERROR processing %s: %s — skipping", key, exc)
                continue

            if rows:
                n = _load_rows(rows, args.table, conn)
                total_rows += n
                del rows
                gc.collect()

            # Mark tile as done
            done_tiles.add(tile_id)
            _save_progress(progress_path, done_tiles)
            tiles_done += 1

            if tiles_done % 10 == 0:
                logger.info("[dem] progress: %d/%d tiles, %d total rows",
                            tiles_done, len(tiles), total_rows)

    logger.info("[dem] done — %d tiles processed, %d total rows in %s",
                tiles_done, total_rows, args.table)


if __name__ == "__main__":
    main()

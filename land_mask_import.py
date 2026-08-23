#!/usr/bin/env python3
"""Download Natural Earth 110m land polygon and load into PostGIS.

The ne_110m_land shapefile (~1 MB, public domain) provides a global
land polygon at ~1 km resolution.  Loaded into a PostGIS table
(``land_mask``) with a spatial index for fast point-in-polygon queries.

Usage:
    python land_mask_import.py [--table land_mask] [--verbose]

The script downloads the shapefile from the Natural Earth CDN, converts
it to WGS84 (EPSG:4326) if needed, and loads it via ogr2ogr.

Requires: ogr2ogr (GDAL) — already used by corine_import.py.
"""

import argparse
import os
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

NATURAL_EARTH_URL = (
    "https://naciscdn.org/naturalearth/110m/physical/ne_110m_land.zip"
)
DEFAULT_TABLE = "land_mask"


def download_shapefile(out_dir: str) -> str:
    """Download and extract the ne_110m_land shapefile.

    Returns the path to the extracted .shp file.
    """
    zip_path = os.path.join(out_dir, "ne_110m_land.zip")

    if not os.path.exists(zip_path):
        print(f"[land_mask] downloading {NATURAL_EARTH_URL} ...")
        req = urllib.request.Request(
            NATURAL_EARTH_URL,
            headers={"User-Agent": "pyrae-land-mask-import/1.0"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            with open(zip_path, "wb") as f:
                f.write(resp.read())
        print(f"[land_mask] downloaded {os.path.getsize(zip_path)} bytes")
    else:
        print(f"[land_mask] using cached {zip_path}")

    # Extract
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)

    # Find the .shp file
    for name in os.listdir(out_dir):
        if name.endswith(".shp"):
            return os.path.join(out_dir, name)

    raise FileNotFoundError("no .shp file found in extracted archive")


def load_to_postgis(shp_path: str, table: str, verbose: bool) -> None:
    """Load the shapefile into PostGIS via ogr2ogr.

    First drops the table to ensure a clean load, then creates it fresh.
    """
    conninfo = os.environ.get(
        "WILDFRAME_DATABASE_URL",
        "host=localhost dbname=wildframe",
    )

    # Extract host/dbname from connection string for ogr2ogr
    # Format: postgresql://user:pass@host:port/dbname?params
    import re
    match = re.search(r"@([^:/]+)", conninfo)
    host = match.group(1) if match else "localhost"

    match = re.search(r"/([^?]+)", conninfo)
    dbname = match.group(1) if match else "wildframe"

    # Extract user:pass for ogr2ogr
    match = re.search(r"://([^@]+)@", conninfo)
    auth = match.group(1) if match else ""

    cmd = [
        "ogr2ogr",
        "-f", "PostgreSQL",
        f"PG:dbname={dbname} host={host}" + (f" user={auth.split(':')[0]}" if ':' in auth else ""),
        "-nln", table,
        "-overwrite",
        "-t_srs", "EPSG:4326",
        shp_path,
    ]

    if verbose:
        print(f"[land_mask] running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[land_mask] ogr2ogr stderr: {result.stderr}", file=sys.stderr)
        raise RuntimeError(f"ogr2ogr failed with exit code {result.returncode}")

    if verbose:
        print(f"[land_mask] ogr2ogr stdout: {result.stdout}")


def create_spatial_index(table: str) -> None:
    """Create a GiST spatial index on the land_mask table."""
    import psycopg

    conninfo = os.environ.get(
        "WILDFRAME_DATABASE_URL",
        "host=localhost dbname=wildframe",
    )

    with psycopg.connect(conninfo) as conn:
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS {table}_geom_idx "
            f"ON {table} USING GIST (geom)"
        )
        conn.commit()

    print(f"[land_mask] spatial index created on '{table}'")


def verify_load(table: str) -> int:
    """Quick sanity check: count rows in the loaded table."""
    import psycopg

    conninfo = os.environ.get(
        "WILDFRAME_DATABASE_URL",
        "host=localhost dbname=wildframe",
    )

    with psycopg.connect(conninfo) as conn:
        row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        count = row[0] if row else 0

    print(f"[land_mask] {count} feature(s) loaded into '{table}'")
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table", default=DEFAULT_TABLE,
        help=f"PostGIS table name (default: {DEFAULT_TABLE})",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Download + extract shapefile
        shp_path = download_shapefile(tmpdir)
        if args.verbose:
            print(f"[land_mask] shapefile: {shp_path}")

        # 2. Load into PostGIS
        load_to_postgis(shp_path, args.table, args.verbose)

        # 3. Create spatial index
        create_spatial_index(args.table)

        # 4. Verify
        count = verify_load(args.table)

    print(f"[land_mask] done — {count} feature(s) in '{args.table}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Download Natural Earth 110m land polygon and load into PostGIS.

The ne_110m_land shapefile (~70 KB, public domain) provides a global
land polygon at ~1 km resolution.  Loaded into a PostGIS table
(``land_mask``) with a spatial index for fast point-in-polygon queries.

Uses pyshp + shapely + psycopg (no GDAL/ogr2ogr needed).

Usage:
    python land_mask_import.py [--table land_mask] [--verbose]
"""

import argparse
import os
import sys
import tempfile
import zipfile

import psycopg
from shapely.geometry import shape, MultiPolygon
from shapely.ops import unary_union

NATURAL_EARTH_URL = (
    "https://naciscdn.org/naturalearth/110m/physical/ne_110m_land.zip"
)
DEFAULT_TABLE = "land_mask"


def download_shapefile(out_dir: str) -> str:
    """Download and extract the ne_110m_land shapefile.

    Returns the path to the extracted .shp file.
    """
    import urllib.request

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


def load_to_postgis(shp_path: str, table: str, verbose: bool) -> int:
    """Load the shapefile into PostGIS using pyshp + shapely + psycopg.

    Reads the shapefile, merges all land polygons into a single
    MultiPolygon, and inserts it as one row.  This is much faster
    than inserting thousands of small polygons for a land mask.
    """
    import shapefile as shp

    conninfo = os.environ.get(
        "WILDFRAME_DATABASE_URL",
        "host=localhost dbname=wildframe",
    )

    # Read shapefile
    if verbose:
        print(f"[land_mask] reading {shp_path} ...")
    reader = shp.Reader(shp_path)
    shapes = reader.shapes()

    # Merge all polygons into one MultiPolygon
    polys = []
    for s in shapes:
        geom = shape(s.__geo_interface__)
        if geom.is_empty:
            continue
        if geom.geom_type == "MultiPolygon":
            polys.extend(geom.geoms)
        elif geom.geom_type == "Polygon":
            polys.append(geom)
    if verbose:
        print(f"[land_mask] read {len(polys)} polygon(s)")

    merged = unary_union(polys)
    if verbose:
        print(f"[land_mask] merged into {merged.geom_type} "
              f"({merged.area:.1f} sq deg)")

    wkt = merged.wkt

    # Load into PostGIS
    with psycopg.connect(conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS postgis")
            cur.execute(f"DROP TABLE IF EXISTS {table}")
            cur.execute(f"""
                CREATE TABLE {table} (
                    id serial PRIMARY KEY,
                    geom geometry(MultiPolygon, 4326) NOT NULL
                )
            """)
            cur.execute(
                f"INSERT INTO {table} (geom) VALUES "
                f"(ST_GeomFromText(%s, 4326))",
                [wkt],
            )
            cur.execute(
                f"CREATE INDEX {table}_geom_idx "
                f"ON {table} USING GIST (geom)"
            )
        conn.commit()

    return 1


def verify_load(table: str) -> int:
    """Quick sanity check: count rows and bounding box."""
    conninfo = os.environ.get(
        "WILDFRAME_DATABASE_URL",
        "host=localhost dbname=wildframe",
    )

    with psycopg.connect(conninfo) as conn:
        row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        count = row[0] if row else 0
        bbox = conn.execute(
            f"SELECT ST_AsText(ST_Extent(geom)) FROM {table}"
        ).fetchone()

    print(f"[land_mask] {count} feature(s) in '{table}'")
    if bbox and bbox[0]:
        print(f"[land_mask] extent: {bbox[0]}")
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
        shp_path = download_shapefile(tmpdir)
        if args.verbose:
            print(f"[land_mask] shapefile: {shp_path}")

        load_to_postgis(shp_path, args.table, args.verbose)
        count = verify_load(args.table)

    print(f"[land_mask] done — {count} feature(s) in '{args.table}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())

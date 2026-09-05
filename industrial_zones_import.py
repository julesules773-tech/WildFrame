#!/usr/bin/env python3
from __future__ import annotations
"""Download OSM industrial zone polygons and load into PostGIS.

Queries OpenStreetMap via Overpass API for landuse=industrial polygons
within a bounding box and stores them in the osm_industrial_zones table.

Usage:
    python industrial_zones_import.py --bbox 14.0,49.0,24.2,54.8 -v

    # Or import all of Europe:
    python industrial_zones_import.py --bbox -12.0,35.0,30.0,72.0 -v

Requires: requests, shapely, psycopg (all already in the project).
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Optional

import requests
from shapely.geometry import shape, mapping
from shapely.ops import unary_union

logger = logging.getLogger("industrial_import")

DEFAULT_TABLE = "osm_industrial_zones"

# Multiple Overpass API endpoints for fallback
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

# Rate limiting: minimum seconds between requests
MIN_REQUEST_INTERVAL = 2.0
_last_request_time = 0.0


def _rate_limit():
    """Enforce minimum interval between Overpass requests."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < MIN_REQUEST_INTERVAL:
        time.sleep(MIN_REQUEST_INTERVAL - elapsed)
    _last_request_time = time.time()


def _query_overpass(bbox: tuple[float, float, float, float]) -> list[dict]:
    """Query Overpass API for landuse=industrial polygons.
    
    Args:
        bbox: (south, west, north, east) - Overpass format
        
    Returns:
        List of GeoJSON features
    """
    south, west, north, east = bbox
    
    # Query for ways and relations with landuse=industrial
    query = f"""
    [out:json][timeout:300];
    (
      way["landuse"="industrial"]({south},{west},{north},{east});
      relation["landuse"="industrial"]({south},{west},{north},{east});
    );
    out body;
    >;
    out skel qt;
    """
    
    features = []
    
    for endpoint in OVERPASS_ENDPOINTS:
        _rate_limit()
        logger.info(f"Querying Overpass: {endpoint}")
        
        try:
            resp = requests.post(
                endpoint,
                data={"data": query},
                timeout=600,
                headers={"User-Agent": "wildframe-industrial-import/1.0"}
            )
            resp.raise_for_status()
            data = resp.json()
            
            # Parse OSM JSON to GeoJSON features
            elements = data.get("elements", [])
            
            # Build node lookup
            nodes = {}
            for el in elements:
                if el["type"] == "node":
                    nodes[el["id"]] = (el["lon"], el["lat"])
            
            # Convert ways/relations to polygons
            for el in elements:
                if el["type"] == "way":
                    coords = []
                    for node_id in el.get("nodes", []):
                        if node_id in nodes:
                            coords.append(nodes[node_id])
                    
                    # Close the polygon
                    if coords and coords[0] != coords[-1]:
                        coords.append(coords[0])
                    
                    if len(coords) >= 4:  # Minimum for a valid polygon
                        feature = {
                            "type": "Feature",
                            "properties": {
                                "osm_id": el["id"],
                                "osm_type": "way",
                                "name": el.get("tags", {}).get("name"),
                            },
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [coords]
                            }
                        }
                        features.append(feature)
                
                elif el["type"] == "relation" and el.get("tags", {}).get("type") == "multipolygon":
                    # Handle multipolygon relations
                    members = el.get("members", [])
                    
                    # Collect outer and inner rings
                    outer_ways = []
                    inner_ways = []
                    
                    for member in members:
                        if member["type"] == "way":
                            # Find the way element
                            for way_el in elements:
                                if way_el["type"] == "way" and way_el["id"] == member["ref"]:
                                    coords = []
                                    for node_id in way_el.get("nodes", []):
                                        if node_id in nodes:
                                            coords.append(nodes[node_id])
                                    
                                    if coords and coords[0] != coords[-1]:
                                        coords.append(coords[0])
                                    
                                    if len(coords) >= 4:
                                        if member.get("role") == "outer":
                                            outer_ways.append(coords)
                                        elif member.get("role") == "inner":
                                            inner_ways.append(coords)
                                    break
                    
                    # Build multipolygon geometry
                    if outer_ways:
                        try:
                            # Create outer polygon(s)
                            outer_polys = []
                            for coords in outer_ways:
                                from shapely.geometry import Polygon
                                poly = Polygon(coords)
                                if poly.is_valid:
                                    outer_polys.append(poly)
                            
                            # Create inner polygons (holes)
                            inner_polys = []
                            for coords in inner_ways:
                                from shapely.geometry import Polygon
                                poly = Polygon(coords)
                                if poly.is_valid:
                                    inner_polys.append(poly)
                            
                            if outer_polys:
                                # Union outer polygons and subtract holes
                                outer = unary_union(outer_polys)
                                if inner_polys:
                                    holes = unary_union(inner_polys)
                                    final = outer.difference(holes)
                                else:
                                    final = outer
                                
                                if not final.is_empty and final.geom_type in ("Polygon", "MultiPolygon"):
                                    feature = {
                                        "type": "Feature",
                                        "properties": {
                                            "osm_id": el["id"],
                                            "osm_type": "relation",
                                            "name": el.get("tags", {}).get("name"),
                                        },
                                        "geometry": mapping(final)
                                    }
                                    features.append(feature)
                        except Exception as e:
                            logger.warning(f"Failed to parse relation {el['id']}: {e}")
            
            logger.info(f"Found {len(features)} industrial zones")
            return features
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Overpass request failed ({endpoint}): {e}")
            continue
    
    logger.error("All Overpass endpoints failed")
    return []


def _ensure_table(cur, table: str, is_resume: bool) -> None:
    """Create the table on first run, or just ensure it exists on resume."""
    cur.execute("CREATE EXTENSION IF NOT EXISTS postgis")
    if is_resume:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id serial PRIMARY KEY,
                osm_id bigint NOT NULL,
                osm_type text NOT NULL DEFAULT 'way',
                name text,
                geom geometry(Geometry, 4326) NOT NULL,
                imported_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
    else:
        cur.execute(f"DROP TABLE IF EXISTS {table}")
        cur.execute(f"""
            CREATE TABLE {table} (
                id serial PRIMARY KEY,
                osm_id bigint NOT NULL,
                osm_type text NOT NULL DEFAULT 'way',
                name text,
                geom geometry(Geometry, 4326) NOT NULL,
                imported_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)


def _insert_features(cur, table: str, features: list[dict]) -> int:
    """Insert GeoJSON features into the table. Returns count inserted."""
    inserted = 0
    for feature in features:
        props = feature["properties"]
        geom = json.dumps(feature["geometry"])
        
        try:
            cur.execute(f"""
                INSERT INTO {table} (osm_id, osm_type, name, geom)
                VALUES (%s, %s, %s, ST_GeomFromGeoJSON(%s))
                ON CONFLICT (osm_id, osm_type) DO NOTHING
            """, (props["osm_id"], props["osm_type"], props.get("name"), geom))
            if cur.rowcount > 0:
                inserted += 1
        except Exception as e:
            logger.warning(f"Failed to insert OSM {props['osm_type']} {props['osm_id']}: {e}")
    
    return inserted


def _ensure_indexes(cur, table: str) -> None:
    """Create spatial and uniqueness indexes."""
    cur.execute(f"""
        CREATE INDEX IF NOT EXISTS {table}_geom_idx 
        ON {table} USING GIST (geom)
    """)
    cur.execute(f"""
        CREATE UNIQUE INDEX IF NOT EXISTS {table}_osm_id_idx 
        ON {table} (osm_id, osm_type)
    """)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--bbox", required=True,
        help="south,west,north,east WGS84 (Overpass format)",
    )
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="skip if table already has data")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="[%(name)s] %(message)s",
    )
    
    bbox = tuple(float(v) for v in args.bbox.split(","))
    if len(bbox) != 4:
        sys.exit("--bbox must be south,west,north,east")
    
    # Query Overpass for industrial zones
    logger.info(f"Querying Overpass for landuse=industrial in {bbox}")
    features = _query_overpass(bbox)
    
    if not features:
        logger.warning("No industrial zones found")
        return 0
    
    # Connect to database and insert
    import psycopg
    from psycopg.rows import dict_row
    
    conninfo = os.environ.get(
        "WILDFRAME_DATABASE_URL",
        "host=localhost dbname=wildframe",
    )
    
    with psycopg.connect(conninfo) as conn:
        with conn.cursor() as cur:
            _ensure_table(cur, args.table, args.resume)
            
            inserted = _insert_features(cur, args.table, features)
            _ensure_indexes(cur, args.table)
            conn.commit()
    
    logger.info(f"Done: {inserted} industrial zones inserted into '{args.table}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())

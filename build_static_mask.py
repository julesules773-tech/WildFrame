#!/usr/bin/env python3
"""Build a static thermal-source mask from FIRMS archive detections.

Reads the daily FIRMS archive CSVs (VIIRS 375m, Europe region — the files
the operator downloads from the FIRMS Archive Download page), clips them to
a bounding box (default Poland + margin), bins detections onto a fixed
square grid, and flags cells whose *persistence* indicates a static thermal
source — a refinery flare, power plant or steel mill that burns on nearly
every overpass year-round — as opposed to transient wildfires.

Methodology follows NASA's STA (Static Thermal Anomalies) mask: cumulative
detections binned onto ~400 m cells, flagged when a cell is hot repeatedly
across time. The exact thresholds are configurable so they can be tuned
against the actual data.

Grid: EPSG:3035 (ETRS89-LAEA, metres — the same projection CORINE uses) so
cells are true 400 m squares regardless of latitude. The service's own
coverage gaps (only ~29% of days are present) are absorbed by requiring
persistence across *distinct months* rather than raw day count.

Usage:
    python build_static_mask.py [--dir ~/Downloads/firms_archive] \
        [--bbox 12,46.9,26.9,56.8] [--cell-m 400] \
        [--min-detections 8] [--min-months 3] [--table static_sources]
"""

import argparse
import csv
import logging
import os
import sys
from datetime import datetime

import psycopg

logger = logging.getLogger("build_static_mask")

# WGS84 bbox for Poland + border margin (matches the CORINE import extent).
DEFAULT_BBOX = (12.0, 46.9, 26.9, 56.8)  # west, south, east, north
DEFAULT_CELL_M = 400.0
DEFAULT_MIN_DETECTIONS = 8
DEFAULT_MIN_MONTHS = 3


def parse_archive_files(directory: str, bbox: tuple) -> list[dict]:
    """Parse every daily FIRMS CSV in `directory`, clipped to `bbox`.

    Returns a list of detection dicts (lat, lon, acq_date, frp, daynight,
    satellite, confidence). Only rows inside the bbox are kept — archive
    files are region-scoped (Europe) but the bbox keeps the load small and
    matches the CORINE footprint.
    """
    w, s, e, n = bbox
    detections: list[dict] = []
    files = sorted(
        f for f in os.listdir(directory)
        if f.lower().endswith(".txt") or f.lower().endswith(".csv")
    )
    if not files:
        raise SystemExit(f"no FIRMS archive files found in {directory!r}")
    for fname in files:
        path = os.path.join(directory, fname)
        with open(path, newline="", encoding="utf-8", errors="replace") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames:
                continue
            for row in reader:
                try:
                    lat = float(row["latitude"])
                    lon = float(row["longitude"])
                except (KeyError, ValueError, TypeError):
                    continue
                if not (w <= lon <= e and s <= lat <= n):
                    continue
                detections.append({
                    "lat": lat,
                    "lon": lon,
                    "acq_date": str(row.get("acq_date", "")).strip(),
                    "frp": _as_float(row.get("frp")),
                    "daynight": str(row.get("daynight", "")).strip(),
                    "satellite": str(row.get("satellite", "")).strip(),
                    "confidence": str(row.get("confidence", "")).strip().lower(),
                })
    logger.info("parsed %d detections inside bbox from %d file(s)",
                len(detections), len(files))
    return detections


def _as_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def build_mask(
    detections: list[dict],
    cell_m: float,
    min_detections: int,
    min_months: int,
    table: str,
) -> int:
    """Bin detections in PostGIS, aggregate, and load `table` with flagged cells.

    Returns the number of cells flagged as static sources.
    """
    conninfo = os.environ.get(
        "WILDFRAME_DATABASE_URL",
        "host=localhost dbname=wildframe",
    )
    with psycopg.connect(conninfo) as conn:
        with conn.cursor() as cur:
            # Stage the raw detections as 4326 points (temporary, dropped below).
            cur.execute("DROP TABLE IF EXISTS _firms_archive_raw")
            cur.execute(
                """
                CREATE TEMP TABLE _firms_archive_raw (
                    geom geometry(Point, 4326),
                    acq_date date,
                    frp double precision,
                    daynight text,
                    confidence text
                )
                """
            )
            rows = [
                (
                    f"SRID=4326;POINT({d['lon']} {d['lat']})",
                    d["acq_date"] or None,
                    d["frp"],
                    d["daynight"],
                    d["confidence"],
                )
                for d in detections
            ]
            cur.executemany(
                "INSERT INTO _firms_archive_raw VALUES (%s, %s, %s, %s, %s)",
                rows,
            )

            # Bin onto the cell grid in EPSG:3035 (LAEA, metres), aggregate
            # persistence per cell. ST_SnapToGrid snaps the transformed point
            # to cell-size multiples; expanding by half a cell yields the
            # cell's square polygon (transformed back to 4326 for storage).
            cur.execute(
                """
                DROP TABLE IF EXISTS {table}
                """.format(table=table)
            )
            cur.execute(
                """
                CREATE TABLE {table} AS
                WITH binned AS (
                    SELECT
                        ST_SnapToGrid(ST_Transform(geom, 3035), %s) AS c,
                        acq_date, frp, daynight, confidence
                    FROM _firms_archive_raw
                    WHERE acq_date IS NOT NULL
                ),
                agg AS (
                    SELECT
                        c,
                        count(*)                                         AS detection_count,
                        count(DISTINCT acq_date)                         AS distinct_days,
                        count(DISTINCT date_trunc('month', acq_date))    AS distinct_months,
                        min(acq_date)                                    AS first_date,
                        max(acq_date)                                    AS last_date,
                        round(avg(frp)::numeric, 2)                      AS mean_frp,
                        round(max(frp)::numeric, 2)                      AS max_frp,
                        count(*) FILTER (WHERE daynight = 'D')           AS day_count,
                        count(*) FILTER (WHERE daynight = 'N')           AS night_count,
                        count(*) FILTER (WHERE confidence IN
                            ('nominal', 'high', 'h', 'n'))               AS nominal_count
                    FROM binned
                    GROUP BY c
                )
                SELECT
                    ST_Transform(ST_Expand(c, %s / 2.0, %s / 2.0), 4326) AS geom,
                    ST_X(ST_Transform(c, 4326))                          AS centroid_lon,
                    ST_Y(ST_Transform(c, 4326))                          AS centroid_lat,
                    detection_count,
                    distinct_days,
                    distinct_months,
                    first_date,
                    last_date,
                    mean_frp,
                    max_frp,
                    day_count,
                    night_count,
                    nominal_count,
                    (detection_count >= %s AND distinct_months >= %s)    AS is_static
                FROM agg
                """.format(table=table),
                (cell_m, cell_m, cell_m, min_detections, min_months),
            )
            # Identity + unique cell key. The cell centroid in 3035 IS the
            # snapped grid point (the cell square was expanded around it), so
            # round() recovers the exact snapped coordinate for the key.
            cur.execute(
                """
                ALTER TABLE {table} ADD COLUMN id serial
                """.format(table=table)
            )
            cur.execute(
                """
                ALTER TABLE {table} ADD COLUMN cell_key TEXT
                """.format(table=table)
            )
            cur.execute(
                """
                UPDATE {table} SET cell_key =
                    round(ST_X(ST_Transform(ST_Centroid(geom), 3035)))::bigint || ',' ||
                    round(ST_Y(ST_Transform(ST_Centroid(geom), 3035)))::bigint
                """.format(table=table)
            )
            cur.execute(
                """
                ALTER TABLE {table}
                    ADD CONSTRAINT {table}_pkey PRIMARY KEY (id),
                    ALTER COLUMN cell_key SET NOT NULL
                """.format(table=table)
            )
            cur.execute(
                """
                CREATE UNIQUE INDEX {table}_cell_key_idx ON {table} (cell_key)
                """.format(table=table)
            )
            cur.execute(
                """
                CREATE INDEX {table}_geom_idx ON {table} USING GIST (geom)
                """.format(table=table)
            )
            cur.execute(
                "SELECT count(*) FROM {table} WHERE is_static".format(table=table)
            )
            n_static = cur.fetchone()[0]
            conn.commit()

    logger.info("loaded mask into %s (%d cells flagged static)", table, n_static)
    return n_static


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default=os.path.expanduser("~/Downloads/firms_archive"),
                        help="folder of daily FIRMS archive CSVs")
    parser.add_argument("--bbox", default=",".join(map(str, DEFAULT_BBOX)),
                        help="west,south,east,north WGS84 clip box")
    parser.add_argument("--cell-m", type=float, default=DEFAULT_CELL_M,
                        help="cell size in metres (EPSG:3035)")
    parser.add_argument("--min-detections", type=int, default=DEFAULT_MIN_DETECTIONS,
                        help="minimum detections per cell to flag as static")
    parser.add_argument("--min-months", type=int, default=DEFAULT_MIN_MONTHS,
                        help="minimum distinct months per cell to flag as static")
    parser.add_argument("--table", default="static_sources",
                        help="target PostGIS table (recreated)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="[%(name)s] %(message)s",
    )

    bbox = tuple(float(v) for v in args.bbox.split(","))
    if len(bbox) != 4:
        raise SystemExit("--bbox must be west,south,east,north")

    detections = parse_archive_files(args.dir, bbox)
    if not detections:
        raise SystemExit("no detections in bbox — nothing to build")
    n_static = build_mask(detections, args.cell_m,
                          args.min_detections, args.min_months, args.table)
    print(f"mask built: {args.table} — {n_static} cells flagged static "
          f"(>= {args.min_detections} detections across >= {args.min_months} months)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

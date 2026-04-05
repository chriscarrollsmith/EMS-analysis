#!/usr/bin/env python3
"""
Compare EMS response times in 2024 vs 2025: MTA Congestion Relief geofence ZCTAs
vs the rest of NYC (among ZCTAs whose centroid falls in NYC).

Geofence: data.ny.gov srxy-5nxn (same source as render_heatmap.py).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import sqlite3
from shapely.ops import unary_union

import geopandas as gpd

from ems_sql import incident_year_sql_expr
from render_heatmap import (
    NYC_PLANAR_CRS,
    load_mta_cbd_geofence,
    load_nyc_zctas,
    normalize_zip_key,
)

DEFAULT_DB = Path(__file__).resolve().parent / "ems_incidents.sqlite"
DEFAULT_TABLE = "ems_incidents"

Metric = Literal["incident", "dispatch", "travel"]

METRIC_CONFIG: dict[Metric, tuple[str, str]] = {
    "incident": (
        "incident_response_seconds_qy",
        "valid_incident_rspns_time_indc",
    ),
    "dispatch": (
        "dispatch_response_seconds_qy",
        "valid_dispatch_rspns_time_indc",
    ),
    "travel": (
        "incident_travel_tm_seconds_qy",
        "valid_incident_rspns_time_indc",
    ),
}


def zctas_intersecting_geofence(
    zctas: gpd.GeoDataFrame,
    geofence: gpd.GeoDataFrame,
) -> set[str]:
    z_proj = zctas.to_crs(NYC_PLANAR_CRS)
    cbd_u = unary_union(geofence.to_crs(NYC_PLANAR_CRS).geometry)
    out: set[str] = set()
    for _, row in z_proj.iterrows():
        g = row.geometry
        if g.intersects(cbd_u):
            inter = g.intersection(cbd_u)
            if inter.is_empty:
                continue
            out.add(row["geo_key"])
    return out

def load_incidents_minutes(
    conn: sqlite3.Connection,
    table: str,
    metric: Metric,
    *,
    years: tuple[int, ...],
) -> pd.DataFrame:
    seconds_col, valid_col = METRIC_CONFIG[metric]
    yexpr = incident_year_sql_expr()
    yr_list = ",".join(str(y) for y in years)
    sql = f"""
    SELECT
      TRIM("zipcode") AS raw_zip,
      CAST({yexpr} AS INTEGER) AS year,
      CAST("{seconds_col}" AS REAL) / 60.0 AS minutes
    FROM "{table}"
    WHERE TRIM(COALESCE("zipcode", '')) != ''
      AND "{valid_col}" = 'Y'
      AND CAST("{seconds_col}" AS REAL) > 0
      AND CAST("{seconds_col}" AS REAL) < 86400
      AND TRIM(COALESCE("incident_datetime", '')) != ''
      AND ({yexpr}) GLOB '[0-9][0-9][0-9][0-9]'
      AND CAST({yexpr} AS INTEGER) IN ({yr_list})
    """
    df = pd.read_sql_query(sql, conn)
    df["geo_key"] = df["raw_zip"].map(normalize_zip_key)
    df = df.dropna(subset=["geo_key", "year"])
    df = df[df["year"].between(1990, 2100)]
    df["year"] = df["year"].astype(int)
    return df.drop(columns=["raw_zip"])


def summarize_pool(df: pd.DataFrame) -> dict[str, float | int]:
    m = df["minutes"]
    return {
        "n": int(len(m)),
        "median_min": float(m.median()),
        "mean_min": float(m.mean()),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, default=DEFAULT_DB, help="SQLite database")
    p.add_argument("--table", default=DEFAULT_TABLE, help="Incidents table")
    p.add_argument(
        "--metric",
        choices=("incident", "dispatch", "travel"),
        default="incident",
        help="incident / dispatch / travel (default: incident)",
    )
    p.add_argument(
        "--zcta-year",
        type=int,
        default=2020,
        help="Census year for NYC ZCTA boundaries (default: 2020)",
    )
    args = p.parse_args()

    if not args.db.is_file():
        print(f"Database not found: {args.db}", file=sys.stderr)
        return 1

    print("Loading NYC ZCTAs (pygris)…", file=sys.stderr)
    zctas = load_nyc_zctas(year=args.zcta_year, cb=True)
    nyc_keys = set(zctas["geo_key"].astype(str))

    print("Loading MTA CBD / CRZ geofence…", file=sys.stderr)
    gf = load_mta_cbd_geofence()
    if gf.crs is None:
        gf = gf.set_crs("EPSG:4326")
    crz_keys = zctas_intersecting_geofence(zctas, gf)
    rest_keys = nyc_keys - crz_keys

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    try:
        raw = load_incidents_minutes(conn, args.table, args.metric, years=(2024, 2025))
    finally:
        conn.close()

    raw = raw[raw["geo_key"].isin(nyc_keys)]
    if raw.empty:
        print("No rows for 2024–2025 in NYC ZCTAs.", file=sys.stderr)
        return 1

    raw["zone"] = np.where(raw["geo_key"].isin(crz_keys), "crz", "rest")

    y2024 = raw[raw["year"] == 2024]
    y2025 = raw[raw["year"] == 2025]

    print()
    print("MTA geofence ∩ NYC ZCTA (2020 cb):", len(crz_keys), "ZCTAs in CRZ")
    print("Remaining NYC ZCTAs (rest):", len(rest_keys))
    print(f"Metric: {args.metric} (minutes); valid-time records only")
    print()

    rows = []
    for zone in ("crz", "rest"):
        a = summarize_pool(y2024[y2024["zone"] == zone])
        b = summarize_pool(y2025[y2025["zone"] == zone])
        d_med = b["median_min"] - a["median_min"]
        d_mean = b["mean_min"] - a["mean_min"]
        rows.append(
            {
                "zone": zone,
                "n_2024": a["n"],
                "median_2024": a["median_min"],
                "mean_2024": a["mean_min"],
                "n_2025": b["n"],
                "median_2025": b["median_min"],
                "mean_2025": b["mean_min"],
                "delta_median": d_med,
                "delta_mean": d_mean,
            }
        )

    out = pd.DataFrame(rows).set_index("zone")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print(out.to_string(float_format=lambda x: f"{x:.3f}"))
    print()

    crz, rest = rows[0], rows[1]
    dd_med = crz["delta_median"] - rest["delta_median"]
    dd_mean = crz["delta_mean"] - rest["delta_mean"]
    print(
        "Difference in 2025−2024 changes (CRZ minus rest): "
        f"Δmedian = {dd_med:+.3f} min; Δmean = {dd_mean:+.3f} min"
    )
    if dd_med > 0.005:
        qual = "CRZ worsened more than the rest of NYC (by median)."
    elif dd_med < -0.005:
        qual = "CRZ worsened less than the rest of NYC (by median)."
    else:
        qual = "CRZ and the rest of NYC moved similarly (by median)."
    print(qual)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

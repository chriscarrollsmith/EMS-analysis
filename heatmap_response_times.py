#!/usr/bin/env python3
"""
Choropleth map of EMS response times by NYC geography.

The EMS extract has no coordinates; it includes `zipcode`, `communityschooldistrict`,
`communitydistrict`, etc. Default geography is **Census ZCTA** (ZIP Code Tabulation
Area), which is the standard GIS join target for ZIP-style codes. Boundaries come
from the Census Bureau via `pygris` (cached locally after the first download).

Alternative: `--geo school_district` uses NYC DOE community school districts (32 areas)
with a lightweight GeoJSON URL.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Literal
from urllib.request import urlopen

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import pygris
import sqlite3
from shapely.ops import unary_union

DEFAULT_DB = Path(__file__).resolve().parent / "ems_incidents.sqlite"
DEFAULT_TABLE = "ems_incidents"
DEFAULT_SCHOOL_DISTRICTS_URL = "https://data.mixi.nyc/nyc-school-districts.geojson"

# NYC = Bronx, Kings, New York, Queens, Richmond counties (NY state FIPS 36).
NYC_NY_COUNTYFP = ("005", "047", "061", "081", "085")
NYC_PLANAR_CRS = "EPSG:2263"

Metric = Literal["incident", "dispatch", "travel"]
Geo = Literal["zipcode", "school_district"]


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


def load_boundaries_from_url(url: str) -> gpd.GeoDataFrame:
    with urlopen(url, timeout=120) as resp:
        data = resp.read()
    with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as tmp:
        tmp.write(data)
        tmp.flush()
        gdf = gpd.read_file(tmp.name)
    Path(tmp.name).unlink(missing_ok=True)
    return gdf


def nyc_counties_polygon(year: int) -> object:
    cnt = pygris.counties(state="NY", year=year, cb=True)
    nyc = cnt[cnt["COUNTYFP"].isin(NYC_NY_COUNTYFP)]
    return unary_union(nyc.geometry)


def load_nyc_zctas(*, year: int = 2020, cb: bool = True) -> gpd.GeoDataFrame:
    print(
        "Loading Census ZCTA boundaries via pygris "
        "(first run downloads a national file; it is cached for reuse).",
        file=sys.stderr,
    )
    z = pygris.zctas(year=year, cb=cb)
    u = nyc_counties_polygon(year)
    u_proj = gpd.GeoSeries([u], crs=z.crs).to_crs(NYC_PLANAR_CRS).iloc[0]
    z_proj = z.to_crs(NYC_PLANAR_CRS)
    mask = z_proj.centroid.within(u_proj)
    out = z[mask].copy()
    zip_col = "ZCTA5CE20" if "ZCTA5CE20" in out.columns else "ZCTA5CE10"
    out["geo_key"] = out[zip_col].astype(str).str.strip().str.zfill(5)
    return out


def normalize_zip_key(raw: str | None) -> str | None:
    if raw is None:
        return None
    t = str(raw).strip()
    if not t.isdigit():
        return None
    if len(t) < 5:
        t = t.zfill(5)
    if len(t) != 5:
        return None
    return t


def load_zip_boundaries_from_file(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    for col in ("ZCTA5CE20", "GEOID20", "ZCTA5CE10", "GEOID10", "zcta", "zipcode", "ZIP"):
        if col in gdf.columns:
            gdf = gdf.copy()
            gdf["geo_key"] = gdf[col].astype(str).str.strip().str.zfill(5)
            return gdf
    raise SystemExit(
        f"No ZCTA/ZIP column found in {path} "
        "(expected one of: ZCTA5CE20, GEOID20, ZCTA5CE10, GEOID10, zcta, zipcode, ZIP)."
    )


def load_school_district_boundaries(url: str | None, path: Path | None) -> gpd.GeoDataFrame:
    if path is not None:
        gdf = gpd.read_file(path)
    else:
        assert url is not None
        gdf = load_boundaries_from_url(url)
    if "district" not in gdf.columns:
        raise SystemExit(
            "School-district GeoJSON must include a 'district' property "
            "(NYC community school district number)."
        )
    gdf = gdf.copy()
    gdf["geo_key"] = gdf["district"].astype(str).str.strip()
    return gdf


def fetch_seconds(
    conn: sqlite3.Connection,
    table: str,
    metric: Metric,
    geo: Geo,
) -> pd.DataFrame:
    seconds_col, valid_col = METRIC_CONFIG[metric]
    if geo == "zipcode":
        key_expr = "zipcode"
    else:
        key_expr = "communityschooldistrict"
    sql = f"""
    SELECT
      TRIM("{key_expr}") AS raw_geo,
      CAST("{seconds_col}" AS REAL) AS seconds
    FROM "{table}"
    WHERE TRIM(COALESCE("{key_expr}", '')) != ''
      AND "{valid_col}" = 'Y'
      AND CAST("{seconds_col}" AS REAL) > 0
      AND CAST("{seconds_col}" AS REAL) < 86400
    """
    df = pd.read_sql_query(sql, conn)
    if geo == "zipcode":
        df["geo_key"] = df["raw_geo"].map(normalize_zip_key)
    else:
        df["geo_key"] = df["raw_geo"].astype(str).str.strip()
    return df.dropna(subset=["geo_key"]).drop(columns=["raw_geo"])


def aggregate_by_geo(df: pd.DataFrame) -> pd.DataFrame:
    def mean_min(s: pd.Series) -> float:
        return float(s.mean() / 60.0)

    def med_min(s: pd.Series) -> float:
        return float(s.median() / 60.0)

    return df.groupby("geo_key", as_index=False)["seconds"].agg(
        n_incidents="count",
        mean_minutes=mean_min,
        median_minutes=med_min,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help=f"SQLite database (default: {DEFAULT_DB})",
    )
    p.add_argument("--table", default=DEFAULT_TABLE, help="Incidents table name")
    p.add_argument(
        "--geo",
        choices=("zipcode", "school_district"),
        default="zipcode",
        help="Geography for aggregation and map (default: zipcode / Census ZCTA)",
    )
    p.add_argument(
        "--metric",
        choices=("incident", "dispatch", "travel"),
        default="incident",
        help=(
            "incident: call to on-scene; dispatch: call to first assignment; "
            "travel: assignment to on-scene"
        ),
    )
    p.add_argument(
        "--stat",
        choices=("mean", "median"),
        default="mean",
        help="Aggregation statistic within each area (default: mean)",
    )
    p.add_argument(
        "--zcta-year",
        type=int,
        default=2020,
        metavar="YEAR",
        help="Decennial year for Census ZCTA/county boundaries (default: 2020)",
    )
    p.add_argument(
        "--boundaries-url",
        default=DEFAULT_SCHOOL_DISTRICTS_URL,
        help="GeoJSON URL for school districts (only used when --geo school_district)",
    )
    p.add_argument(
        "--boundaries-file",
        type=Path,
        default=None,
        help=(
            "Local boundary file: for zipcode, ZCTA/ZIP GeoJSON or shapefile; "
            "for school_district, GeoJSON with a 'district' property"
        ),
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default depends on --geo)",
    )
    args = p.parse_args()

    out_path = args.output
    if out_path is None:
        out_path = (
            Path("ems_response_time_by_zipcode.png")
            if args.geo == "zipcode"
            else Path("ems_response_time_by_school_district.png")
        )

    if not args.db.is_file():
        print(f"Database not found: {args.db}", file=sys.stderr)
        return 1

    if args.geo == "zipcode":
        if args.boundaries_file is not None:
            gdf = load_zip_boundaries_from_file(args.boundaries_file)
        else:
            gdf = load_nyc_zctas(year=args.zcta_year, cb=True)
    else:
        gdf = load_school_district_boundaries(
            None if args.boundaries_file else args.boundaries_url,
            args.boundaries_file,
        )

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    try:
        raw = fetch_seconds(conn, args.table, args.metric, args.geo)
    finally:
        conn.close()

    if raw.empty:
        print("No rows matched filters (check DB contents and validity flags).", file=sys.stderr)
        return 1

    sub = aggregate_by_geo(raw)
    value_col = "mean_minutes" if args.stat == "mean" else "median_minutes"

    merged = gdf.merge(sub, on="geo_key", how="left")

    fig, ax = plt.subplots(figsize=(11, 11))
    lw = 0.15 if args.geo == "zipcode" else 0.4
    merged.plot(
        column=value_col,
        ax=ax,
        legend=True,
        cmap="YlOrRd",
        edgecolor="0.35",
        linewidth=lw,
        legend_kwds={"label": f"{args.stat.title()} response (minutes)", "shrink": 0.6},
        missing_kwds={"color": "lightgrey", "label": "No data"},
    )
    geo_label = "ZCTA (ZIP tabulation area)" if args.geo == "zipcode" else "community school district"
    ax.set_title(
        f"NYC EMS {args.metric} response time by {geo_label}\n"
        f"({args.stat}; valid-time records only; n={len(raw):,} incidents)"
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

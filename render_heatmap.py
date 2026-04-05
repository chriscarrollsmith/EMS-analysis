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
import re
import sys
import tempfile
from pathlib import Path
from typing import Literal
from urllib.parse import urlencode
from urllib.request import urlopen

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes as MplAxes
from matplotlib import colormaps
from matplotlib.colors import Normalize, to_hex, to_rgb
import numpy as np
import pandas as pd
import pygris
import sqlite3
from shapely.geometry import Point
from shapely.ops import unary_union

from ems_sql import incident_year_sql_expr

DEFAULT_DB = Path(__file__).resolve().parent / "ems_incidents.sqlite"
DEFAULT_TABLE = "ems_incidents"
MAPS_DIR = Path(__file__).resolve().parent / "maps"
MAPS_BY_YEAR = MAPS_DIR / "by-year"
MAPS_BY_PIN = MAPS_DIR / "by-pin"
DEFAULT_SCHOOL_DISTRICTS_URL = "https://data.mixi.nyc/nyc-school-districts.geojson"

# NYC = Bronx, Kings, New York, Queens, Richmond counties (NY state FIPS 36).
NYC_NY_COUNTYFP = ("005", "047", "061", "081", "085")
# New York County (Manhattan); used for congestion-zone boundary clip.
MANHATTAN_COUNTYFP = "061"
NYC_PLANAR_CRS = "EPSG:2263"
# MTA Central Business District geofence (Congestion Relief Zone polygons), WGS84.
# https://data.ny.gov/Transportation/MTA-Central-Business-District-Geofence-Beginni/srxy-5nxn
MTA_CBD_GEOFENCE_GEOJSON_URL = "https://data.ny.gov/resource/srxy-5nxn.geojson"

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


def slugify_map_basename(text: str) -> str:
    """Lowercase alnum words joined by underscores; empty → 'pin'."""
    t = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower())
    t = re.sub(r"_+", "_", t).strip("_")
    return t or "pin"


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
    *,
    include_year: bool = False,
) -> pd.DataFrame:
    seconds_col, valid_col = METRIC_CONFIG[metric]
    year_expr = incident_year_sql_expr()
    if geo == "zipcode":
        key_expr = "zipcode"
    else:
        key_expr = "communityschooldistrict"
    year_select = ""
    year_where = ""
    if include_year:
        year_select = f", CAST({year_expr} AS INTEGER) AS year"
        year_where = f"""
      AND TRIM(COALESCE("incident_datetime", '')) != ''
      AND ({year_expr}) GLOB '[0-9][0-9][0-9][0-9]'
    """
    sql = f"""
    SELECT
      TRIM("{key_expr}") AS raw_geo,
      CAST("{seconds_col}" AS REAL) AS seconds{year_select}
    FROM "{table}"
    WHERE TRIM(COALESCE("{key_expr}", '')) != ''
      AND "{valid_col}" = 'Y'
      AND CAST("{seconds_col}" AS REAL) > 0
      AND CAST("{seconds_col}" AS REAL) < 86400{year_where}
    """
    df = pd.read_sql_query(sql, conn)
    if geo == "zipcode":
        df["geo_key"] = df["raw_geo"].map(normalize_zip_key)
    else:
        df["geo_key"] = df["raw_geo"].astype(str).str.strip()
    out = df.dropna(subset=["geo_key"]).drop(columns=["raw_geo"])
    if include_year:
        out = out.dropna(subset=["year"])
        out = out[out["year"].between(1990, 2100)]
        out["year"] = out["year"].astype(int)
    return out


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


def aggregate_by_geo_year(df: pd.DataFrame) -> pd.DataFrame:
    def mean_min(s: pd.Series) -> float:
        return float(s.mean() / 60.0)

    def med_min(s: pd.Series) -> float:
        return float(s.median() / 60.0)

    return df.groupby(["geo_key", "year"], as_index=False)["seconds"].agg(
        n_incidents="count",
        mean_minutes=mean_min,
        median_minutes=med_min,
    )


def load_mta_cbd_geofence() -> gpd.GeoDataFrame:
    """Download official MTA CBD / Congestion Relief Zone polygons (small dataset)."""
    q = urlencode({"$limit": 100})
    return load_boundaries_from_url(f"{MTA_CBD_GEOFENCE_GEOJSON_URL}?{q}")


def congestion_zone_gdf_for_map(
    plot_crs: str | object,
    *,
    county_boundary_year: int,
) -> gpd.GeoDataFrame | None:
    """
    MTA geofence polygons clipped to Manhattan, reprojected for plotting.
    For map context only; see dataset documentation for legal definitions.
    """
    gf = load_mta_cbd_geofence()
    if gf.empty:
        return None
    if gf.crs is None:
        gf = gf.set_crs("EPSG:4326")
    cnt = pygris.counties(state="NY", year=county_boundary_year, cb=True)
    manhattan = cnt[cnt["COUNTYFP"] == MANHATTAN_COUNTYFP]
    if manhattan.empty:
        return None
    m_geom = unary_union(manhattan.geometry)
    mask = gpd.GeoDataFrame(geometry=[m_geom], crs=cnt.crs).to_crs(NYC_PLANAR_CRS)
    clipped = gpd.clip(gf.to_crs(NYC_PLANAR_CRS), mask)
    clipped = clipped[~clipped.geometry.is_empty & clipped.geometry.notna()]
    if clipped.empty:
        return None
    return clipped.to_crs(plot_crs)


def _expand_ax_limits_for_xy(
    ax: MplAxes,
    gx: float,
    gy: float,
    *,
    pad_frac: float = 0.03,
) -> None:
    """Ensure x/y limits include the point (e.g. pin slightly outside ZCTA hull)."""
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xw = xmax - xmin or 1.0
    yw = ymax - ymin or 1.0
    pad_x = xw * pad_frac
    pad_y = yw * pad_frac
    ax.set_xlim(min(xmin, gx - pad_x), max(xmax, gx + pad_x))
    ax.set_ylim(min(ymin, gy - pad_y), max(ymax, gy + pad_y))


def value_range_for_maps(agg: pd.DataFrame, value_col: str) -> tuple[float, float]:
    s = agg[value_col].dropna()
    if s.empty:
        return 0.0, 1.0
    lo, hi = float(s.quantile(0.02)), float(s.quantile(0.98))
    if lo >= hi:
        lo, hi = float(s.min()), float(s.max())
    if lo >= hi:
        hi = lo + 1e-6
    return lo, hi


def choropleth_norm_bounds(merged: gpd.GeoDataFrame, value_col: str) -> tuple[float, float]:
    """
    vmin/vmax GeoPandas uses for a numeric choropleth when vmin/vmax are omitted:
    min and max of non-NaN values in ``value_col`` (see geopandas.plotting.plot_series).
    """
    arr = merged[value_col].to_numpy(dtype=float)
    ok = ~np.isnan(arr)
    if not np.any(ok):
        return 0.0, 1.0
    lo, hi = float(arr[ok].min()), float(arr[ok].max())
    if lo >= hi:
        hi = lo + 1e-6
    return lo, hi


def choropleth_fill_for_geo_key(
    merged: gpd.GeoDataFrame,
    *,
    value_col: str,
    geo_key: str,
    cmap_name: str = "YlOrRd",
    missing_color: str = "lightgrey",
) -> tuple[str, tuple[int, int, int], float, float, float | None]:
    """
    Face color for one ``geo_key`` matching ``render_choropleth`` with vmin/vmax left
    auto (full-city NYC map, not --by-year).

    Returns ``(matplotlib_color_hex, (r, g, b) for APIs, vmin, vmax, value_or_none)``.
    Missing data uses ``missing_color`` (same name as choropleth ``missing_kwds``).
    """
    vmin, vmax = choropleth_norm_bounds(merged, value_col)
    rows = merged.loc[merged["geo_key"] == geo_key, value_col]
    if rows.empty:
        raise KeyError(f"geo_key {geo_key!r} not in merged boundaries")
    raw_val = rows.iloc[0]
    if pd.isna(raw_val):
        rgb_f = to_rgb(missing_color)
        hex_c = to_hex(rgb_f)
        rgb = tuple(int(round(c * 255)) for c in rgb_f)
        return hex_c, rgb, vmin, vmax, None
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = colormaps[cmap_name]
    rgba = cmap(norm(float(raw_val)))
    hex_c = to_hex(rgba[:3])
    rgb = (int(round(rgba[0] * 255)), int(round(rgba[1] * 255)), int(round(rgba[2] * 255)))
    return hex_c, rgb, vmin, vmax, float(raw_val)


def render_choropleth(
    merged: gpd.GeoDataFrame,
    *,
    value_col: str,
    stat: str,
    metric: Metric,
    geo: Geo,
    n_incidents: int,
    title_suffix: str,
    out_path: Path,
    vmin: float | None = None,
    vmax: float | None = None,
    congestion_zone: gpd.GeoDataFrame | None = None,
    pin_wgs84: tuple[float, float] | None = None,
    pin_label: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=(11, 11))
    lw = 0.15 if geo == "zipcode" else 0.4
    plot_kw: dict = {
        "column": value_col,
        "ax": ax,
        "legend": True,
        "cmap": "YlOrRd",
        "edgecolor": "0.35",
        "linewidth": lw,
        "legend_kwds": {"label": f"{stat.title()} response (minutes)", "shrink": 0.6},
        "missing_kwds": {"color": "lightgrey", "label": "No data"},
    }
    if vmin is not None and vmax is not None:
        plot_kw["vmin"] = vmin
        plot_kw["vmax"] = vmax
    merged.plot(**plot_kw)
    if congestion_zone is not None and not congestion_zone.empty:
        congestion_zone.plot(
            ax=ax,
            facecolor="none",
            edgecolor="black",
            linestyle=(0, (4, 3)),
            linewidth=1.1,
            zorder=5,
        )
    if pin_wgs84 is not None:
        lat, lon = pin_wgs84
        pin_pt = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(merged.crs)
        g0 = pin_pt.geometry.iloc[0]
        if g0.is_empty:
            print(
                "Pin could not be projected (empty geometry); check --pin-lat / --pin-lon.",
                file=sys.stderr,
            )
        else:
            gx, gy = float(g0.x), float(g0.y)
            for _c in ax.collections:
                _c.set_zorder(min(_c.get_zorder(), 5))
            for _ln in ax.lines:
                _ln.set_zorder(min(_ln.get_zorder(), 6))
            _expand_ax_limits_for_xy(ax, gx, gy)
            ax.scatter(
                [gx],
                [gy],
                s=220,
                c="dodgerblue",
                edgecolors="white",
                linewidths=2.0,
                zorder=100,
                clip_on=False,
            )
            if pin_label:
                ax.annotate(
                    pin_label,
                    xy=(gx, gy),
                    xytext=(8, 8),
                    textcoords="offset points",
                    fontsize=10,
                    fontweight="bold",
                    color="black",
                    bbox={
                        "boxstyle": "round,pad=0.3",
                        "facecolor": "white",
                        "edgecolor": "0.35",
                        "alpha": 0.92,
                    },
                    zorder=101,
                    clip_on=False,
                )
    geo_label = "ZCTA (ZIP tabulation area)" if geo == "zipcode" else "community school district"
    ax.set_title(
        f"NYC EMS {metric} response time by {geo_label}\n"
        f"({stat}; valid-time records only; n={n_incidents:,}{title_suffix})"
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path.resolve()}")


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
        default="median",
        help="Aggregation statistic within each area (default: median)",
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
        help=(
            "Output PNG path (not used with --by-year). "
            "Default: maps/ems_response_time_by_zipcode.png or …_school_district.png; "
            "with --include-pin, maps/by-pin/<slug from --label>.png (slug 'pin' if label empty)."
        ),
    )
    p.add_argument(
        "--by-year",
        action="store_true",
        help="One choropleth per calendar year (incident_datetime) under --maps-dir",
    )
    p.add_argument(
        "--maps-dir",
        type=Path,
        default=MAPS_BY_YEAR,
        help=f"Output directory for --by-year PNGs (default: {MAPS_BY_YEAR})",
    )
    p.add_argument(
        "--show-congestion-zone",
        action="store_true",
        help=(
            "Draw dashed MTA Central Business District / congestion relief geofence "
            "(data.ny.gov srxy-5nxn; clipped to Manhattan). "
            "With --by-year, also drawn automatically for years >= 2025."
        ),
    )
    p.add_argument(
        "--include-pin",
        action="store_true",
        help="Plot a marker at --pin-lat / --pin-lon (WGS84), optional text from --label.",
    )
    p.add_argument(
        "--label",
        default="",
        metavar="TEXT",
        help="Label for the pin when --include-pin is set (omit for marker only).",
    )
    p.add_argument(
        "--pin-lat",
        type=float,
        default=None,
        metavar="LAT",
        help="WGS84 latitude for the pin (required with --include-pin).",
    )
    p.add_argument(
        "--pin-lon",
        type=float,
        default=None,
        metavar="LON",
        help=(
            "WGS84 longitude for the pin (required with --include-pin). "
            "Use --pin-lon=-73.9 if the shell misparses a leading minus."
        ),
    )
    args = p.parse_args()

    if args.include_pin:
        if args.pin_lat is None or args.pin_lon is None:
            p.error("--include-pin requires --pin-lat and --pin-lon")
    elif args.pin_lat is not None or args.pin_lon is not None:
        p.error("--pin-lat / --pin-lon require --include-pin")

    out_path = args.output
    if out_path is None and not args.by_year:
        if args.include_pin:
            out_path = MAPS_BY_PIN / f"{slugify_map_basename(args.label)}.png"
        else:
            out_path = (
                MAPS_DIR / "ems_response_time_by_zipcode.png"
                if args.geo == "zipcode"
                else MAPS_DIR / "ems_response_time_by_school_district.png"
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
        raw = fetch_seconds(
            conn,
            args.table,
            args.metric,
            args.geo,
            include_year=args.by_year,
        )
    finally:
        conn.close()

    if raw.empty:
        print("No rows matched filters (check DB contents and validity flags).", file=sys.stderr)
        return 1

    value_col = "mean_minutes" if args.stat == "mean" else "median_minutes"

    if args.by_year:
        args.maps_dir.mkdir(parents=True, exist_ok=True)
        sub = aggregate_by_geo_year(raw)
        if sub.empty:
            print("No rows after year aggregation.", file=sys.stderr)
            return 1
        vmin, vmax = value_range_for_maps(sub, value_col)
        years_sorted = sorted(int(y) for y in sub["year"].unique())
        print(
            f"By-year: {len(years_sorted)} calendar years ({years_sorted[0]}–{years_sorted[-1]}), "
            f"{len(raw):,} filtered incidents → maps under {args.maps_dir}.",
            file=sys.stderr,
        )
        max_year = years_sorted[-1]
        pin_wgs: tuple[float, float] | None = None
        if args.include_pin:
            pin_wgs = (args.pin_lat, args.pin_lon)
        cz_gdf: gpd.GeoDataFrame | None = None
        if args.show_congestion_zone or max_year >= 2025:
            cz_gdf = congestion_zone_gdf_for_map(
                gdf.crs,
                county_boundary_year=args.zcta_year,
            )
        for year_int in years_sorted:
            sub_y = sub[sub["year"] == year_int]
            n_y = int(sub_y["n_incidents"].sum())
            merged = gdf.merge(sub_y.drop(columns=["year"]), on="geo_key", how="left")
            fname = f"{year_int}.png"
            show_cz = cz_gdf is not None and (
                args.show_congestion_zone or year_int >= 2025
            )
            render_choropleth(
                merged,
                value_col=value_col,
                stat=args.stat,
                metric=args.metric,
                geo=args.geo,
                n_incidents=n_y,
                title_suffix=f"; year={year_int}; shared color scale 2–98% pctl across years",
                out_path=args.maps_dir / fname,
                vmin=vmin,
                vmax=vmax,
                congestion_zone=cz_gdf if show_cz else None,
                pin_wgs84=pin_wgs,
                pin_label=args.label,
            )
        return 0

    sub = aggregate_by_geo(raw)
    merged = gdf.merge(sub, on="geo_key", how="left")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cz_single: gpd.GeoDataFrame | None = None
    if args.show_congestion_zone:
        cz_single = congestion_zone_gdf_for_map(
            gdf.crs,
            county_boundary_year=args.zcta_year,
        )
    pin_single: tuple[float, float] | None = None
    if args.include_pin:
        pin_single = (args.pin_lat, args.pin_lon)
    render_choropleth(
        merged,
        value_col=value_col,
        stat=args.stat,
        metric=args.metric,
        geo=args.geo,
        n_incidents=len(raw),
        title_suffix="",
        out_path=out_path,
        vmin=None,
        vmax=None,
        congestion_zone=cz_single,
        pin_wgs84=pin_single,
        pin_label=args.label,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

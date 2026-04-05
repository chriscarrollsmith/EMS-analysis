#!/usr/bin/env python3
"""
Render one NYC ZCTA (ZIP-style code) in either:

- **heatmap**: matplotlib choropleth-style frame (single solid fill), or
- **google**: Google Static Maps with the same polygon and matching viewport.

Both use **EPSG:3857** (Web Mercator) for framing so the shape aligns with Google’s
basemap. The Static Maps request uses `visible=` corners derived from the same padded
3857 bounding box as the matplotlib axes.

Requires `GOOGLE_MAPS_API_KEY` in the environment for `--mode google` or `both`.

Fill color matches ``render_heatmap.py`` full-city zip choropleth (``YlOrRd``,
min/max of non-NaN values only) unless ``--solid-fill`` is set.
"""

from __future__ import annotations

import argparse
from typing import cast
import os
import re
import sqlite3
import sys
from dotenv import load_dotenv
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box

from render_heatmap import (
    DEFAULT_DB,
    DEFAULT_TABLE,
    MAPS_DIR,
    Metric,
    aggregate_by_geo,
    choropleth_fill_for_geo_key,
    fetch_seconds,
    load_nyc_zctas,
    load_zip_boundaries_from_file,
    normalize_zip_key,
)

load_dotenv()

MAPS_BY_ZIP = MAPS_DIR / "by-zip"
WEB_MERCATOR = "EPSG:3857"
WGS84 = "EPSG:4326"
# Match render_heatmap zipcode styling
ZIP_EDGE_WIDTH = 0.15
ZIP_EDGE_COLOR = "0.35"


def load_zcta_gdf(*, zcta_year: int, boundaries_file: Path | None) -> gpd.GeoDataFrame:
    if boundaries_file is not None:
        return load_zip_boundaries_from_file(boundaries_file)
    return load_nyc_zctas(year=zcta_year, cb=True)


def subset_one_zcta(gdf: gpd.GeoDataFrame, zip_key: str) -> gpd.GeoDataFrame:
    row = gdf[gdf["geo_key"] == zip_key].copy()
    if row.empty:
        raise SystemExit(
            f"No boundary found for ZCTA/ZIP {zip_key!r}. "
            "Check the code is a valid NYC ZCTA or pass --boundaries-file."
        )
    return row


def parse_solid_fill_hex(spec: str) -> tuple[str, tuple[int, int, int]]:
    """``#RRGGBB`` or ``RRGGBB`` → ``('#rrggbb', (r,g,b))``."""
    t = spec.strip()
    if t.startswith("#"):
        t = t[1:]
    if not re.fullmatch(r"[0-9A-Fa-f]{6}", t):
        raise SystemExit(
            f"Invalid --solid-fill {spec!r}; use six hex digits, e.g. #f0a040 or f0a040."
        )
    r, g, b = int(t[0:2], 16), int(t[2:4], 16), int(t[4:6], 16)
    return f"#{t.lower()}", (r, g, b)


def ems_fill_for_zip(
    gdf: gpd.GeoDataFrame,
    zip_key: str,
    *,
    db: Path,
    table: str,
    metric: Metric,
    stat: str,
) -> tuple[str, tuple[int, int, int], float | None]:
    value_col = "mean_minutes" if stat == "mean" else "median_minutes"
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        raw = fetch_seconds(conn, table, metric, "zipcode", include_year=False)
    finally:
        conn.close()
    if raw.empty:
        raise SystemExit("No EMS rows matched filters (same as render_heatmap).")
    sub = aggregate_by_geo(raw)
    merged = gdf.merge(sub, on="geo_key", how="left")
    hex_c, rgb, _vmin, _vmax, val = choropleth_fill_for_geo_key(
        merged, value_col=value_col, geo_key=zip_key
    )
    return hex_c, rgb, val


def largest_polygon(geom):
    if geom.geom_type == "Polygon":
        return geom
    if geom.geom_type == "MultiPolygon":
        return max(geom.geoms, key=lambda g: g.area)
    raise SystemExit(f"Unsupported geometry type: {geom.geom_type}")


def padded_bounds_mercator(gdf_wgs84: gpd.GeoDataFrame, padding: float) -> tuple[float, float, float, float]:
    g3857 = gdf_wgs84.to_crs(WEB_MERCATOR)
    minx, miny, maxx, maxy = g3857.total_bounds
    w, h = maxx - minx, maxy - miny
    if w <= 0 or h <= 0:
        raise SystemExit("Degenerate geometry bounds.")
    px, py = w * padding, h * padding
    return minx - px, miny - py, maxx + px, maxy + py


def mercator_bbox_corners_wgs84(
    minx: float, miny: float, maxx: float, maxy: float,
) -> list[tuple[float, float]]:
    """Four corners of axis-aligned Web Mercator bbox, as (lat, lng) for Google visible=."""
    rect = box(minx, miny, maxx, maxy)
    g = gpd.GeoSeries([rect], crs=WEB_MERCATOR)
    g_wgs = g.to_crs(WGS84)
    xs, ys = g_wgs.geometry.iloc[0].exterior.xy
    ring = list(zip(xs, ys))[:-1]  # (lon, lat); drop closing vertex
    return [(lat, lon) for lon, lat in ring]


def polygon_path_latlng_for_google(
    geom_wgs84,
    *,
    fill_rgb: tuple[int, int, int],
    fill_alpha: int,
    simplify_m: float,
    max_url_chars: int,
) -> str:
    """Build Static Maps `path` value: outline + semi-transparent fill."""
    rr, gg, bb = fill_rgb
    if not (0 <= fill_alpha <= 255):
        raise SystemExit("--google-fill-alpha must be 0–255.")
    g = gpd.GeoDataFrame(geometry=[largest_polygon(geom_wgs84)], crs=WGS84)
    g3857 = g.to_crs(WEB_MERCATOR)
    simp = g3857.geometry.iloc[0].simplify(simplify_m, preserve_topology=True)
    if simp.is_empty:
        simp = g3857.geometry.iloc[0]
    simp_wgs = gpd.GeoSeries([simp], crs=WEB_MERCATOR).to_crs(WGS84).iloc[0]
    poly = largest_polygon(simp_wgs)
    ext = poly.exterior
    pts: list[tuple[float, float]] = []
    for lon, lat in ext.coords[:-1]:
        pts.append((lat, lon))

    def encode_path(points: list[tuple[float, float]]) -> str:
        inner = "|".join(f"{lat:.6f},{lon:.6f}" for lat, lon in points)
        fc = f"0x{rr:02X}{gg:02X}{bb:02X}{fill_alpha:02X}"
        return f"color:0x555555|weight:2|fillcolor:{fc}|{inner}"

    path_val = encode_path(pts)
    while len(path_val) > max_url_chars and simplify_m < 5000:
        simplify_m *= 2
        simp = g3857.geometry.iloc[0].simplify(simplify_m, preserve_topology=True)
        if simp.is_empty:
            break
        simp_wgs = gpd.GeoSeries([simp], crs=WEB_MERCATOR).to_crs(WGS84).iloc[0]
        poly = largest_polygon(simp_wgs)
        ext = poly.exterior
        pts = [(lat, lon) for lon, lat in ext.coords[:-1]]
        path_val = encode_path(pts)
    return path_val


def render_heatmap_mercator(
    gdf_one: gpd.GeoDataFrame,
    *,
    zip_key: str,
    fill_hex: str,
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    out_path: Path,
    fig_inches: float,
    dpi: int,
    subtitle: str,
) -> None:
    g3857 = gdf_one.to_crs(WEB_MERCATOR)
    fig, ax = plt.subplots(figsize=(fig_inches, fig_inches), dpi=dpi)
    g3857.plot(
        ax=ax,
        color=fill_hex,
        edgecolor=ZIP_EDGE_COLOR,
        linewidth=ZIP_EDGE_WIDTH,
    )
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"NYC ZCTA {zip_key}\n{subtitle}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path.resolve()}", file=sys.stderr)


def fetch_google_static_map(
    *,
    path_spec: str,
    visible_latlng: list[tuple[float, float]],
    width: int,
    height: int,
    api_key: str,
    maptype: str,
    out_path: Path,
) -> None:
    params: list[tuple[str, str]] = [
        ("size", f"{width}x{height}"),
        ("maptype", maptype),
        ("key", api_key),
        ("path", path_spec),
    ]
    for lat, lng in visible_latlng:
        params.append(("visible", f"{lat:.6f},{lng:.6f}"))

    url = "https://maps.googleapis.com/maps/api/staticmap?" + urlencode(params, safe="|:,/")
    with urlopen(url, timeout=60) as resp:
        ctype = resp.headers.get("Content-Type", "")
        data = resp.read()
    if "image" not in ctype.lower():
        raise SystemExit(
            "Google Static Maps did not return an image. "
            f"Content-Type={ctype!r}. First 500 bytes: {data[:500]!r}"
        )
    out_path.write_bytes(data)
    print(f"Wrote {out_path.resolve()}", file=sys.stderr)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("zip", help="5-digit ZCTA / ZIP code (e.g. 10031)")
    p.add_argument(
        "--mode",
        choices=("heatmap", "google", "both"),
        default="both",
        help="Output heatmap only, Google Static Map only, or both (default: both)",
    )
    p.add_argument(
        "--padding",
        type=float,
        default=0.12,
        metavar="FRACTION",
        help="Pad around the geometry in Web Mercator (default: 0.12)",
    )
    p.add_argument(
        "--zcta-year",
        type=int,
        default=2020,
        metavar="YEAR",
        help="Decennial year for Census ZCTA boundaries (default: 2020)",
    )
    p.add_argument(
        "--boundaries-file",
        type=Path,
        default=None,
        help="Local ZCTA GeoJSON/shapefile (same as render_heatmap)",
    )
    p.add_argument(
        "--heatmap-out",
        type=Path,
        default=None,
        help=f"PNG path for heatmap mode (default: {MAPS_BY_ZIP}/<zip>_heatmap.png)",
    )
    p.add_argument(
        "--google-out",
        type=Path,
        default=None,
        help=f"PNG path for Google mode (default: {MAPS_BY_ZIP}/<zip>_google.png)",
    )
    p.add_argument(
        "--fig-inches",
        type=float,
        default=8.0,
        help="Square figure size in inches for heatmap (default: 8)",
    )
    p.add_argument("--dpi", type=int, default=150, help="PNG dpi for heatmap (default: 150)")
    p.add_argument("--google-width", type=int, default=640, help="Static map width in px (default: 640)")
    p.add_argument("--google-height", type=int, default=640, help="Static map height in px (default: 640)")
    p.add_argument(
        "--google-maptype",
        default="roadmap",
        choices=("roadmap", "satellite", "terrain", "hybrid"),
        help="Google map type (default: roadmap)",
    )
    p.add_argument(
        "--simplify-m",
        type=float,
        default=25.0,
        metavar="METERS",
        help="Simplify polygon in Web Mercator for Google path (default: 25)",
    )
    p.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help=f"EMS SQLite (default: {DEFAULT_DB}); required for fill unless --solid-fill",
    )
    p.add_argument("--table", default=DEFAULT_TABLE, help="Incidents table name")
    p.add_argument(
        "--metric",
        choices=("incident", "dispatch", "travel"),
        default="incident",
        help="Response-time column (same as render_heatmap; default: incident)",
    )
    p.add_argument(
        "--stat",
        choices=("mean", "median"),
        default="median",
        help="Aggregate within ZCTA (same as render_heatmap; default: median)",
    )
    p.add_argument(
        "--solid-fill",
        type=str,
        default=None,
        metavar="HEX",
        help="Skip EMS; use #RRGGBB or RRGGBB for fill (does not match full-city choropleth)",
    )
    p.add_argument(
        "--google-fill-alpha",
        type=int,
        default=0x66,
        metavar="0-255",
        help="Alpha byte for Google path fillcolor, 0–255 (default: 102 ≈ 40%% opacity)",
    )
    args = p.parse_args()

    if args.mode in ("google", "both") and not os.environ.get("GOOGLE_MAPS_API_KEY", "").strip():
        print(
            "Set GOOGLE_MAPS_API_KEY in the environment for --mode google or both.",
            file=sys.stderr,
        )
        return 1

    z = normalize_zip_key(args.zip)
    if z is None:
        p.error("ZIP must be a 5-digit numeric code.")

    heatmap_out = args.heatmap_out
    google_out = args.google_out
    if heatmap_out is None:
        heatmap_out = MAPS_BY_ZIP / f"{z}_heatmap.png"
    if google_out is None:
        google_out = MAPS_BY_ZIP / f"{z}_google.png"

    gdf_full = load_zcta_gdf(zcta_year=args.zcta_year, boundaries_file=args.boundaries_file)
    gdf_one = subset_one_zcta(gdf_full, z)
    g_wgs = gdf_one.to_crs(WGS84)
    geom_wgs = g_wgs.geometry.iloc[0]

    if args.solid_fill is not None:
        fill_hex, fill_rgb = parse_solid_fill_hex(args.solid_fill)
        subtitle = f"Web Mercator; solid fill {fill_hex} (--solid-fill)"
    else:
        if not args.db.is_file():
            print(
                f"Database not found: {args.db}. "
                "Use --solid-fill #hex or run fetch_ems_to_sqlite.py.",
                file=sys.stderr,
            )
            return 1
        fill_hex, fill_rgb, val = ems_fill_for_zip(
            gdf_full,
            z,
            db=args.db,
            table=args.table,
            metric=cast(Metric, args.metric),
            stat=args.stat,
        )
        if val is None:
            subtitle = (
                f"Web Mercator; {args.stat} {args.metric} — no data; lightgrey "
                "(matches full-map missing_kwds)"
            )
        else:
            subtitle = (
                f"Web Mercator; {args.stat} {args.metric} = {val:.2f} min "
                "(YlOrRd, same min–max scale as full NYC render_heatmap map)"
            )

    minx, miny, maxx, maxy = padded_bounds_mercator(g_wgs, args.padding)
    visible = mercator_bbox_corners_wgs84(minx, miny, maxx, maxy)

    if args.mode in ("heatmap", "both"):
        heatmap_out.parent.mkdir(parents=True, exist_ok=True)
        render_heatmap_mercator(
            gdf_one,
            zip_key=z,
            fill_hex=fill_hex,
            minx=minx,
            miny=miny,
            maxx=maxx,
            maxy=maxy,
            out_path=heatmap_out,
            fig_inches=args.fig_inches,
            dpi=args.dpi,
            subtitle=subtitle,
        )

    if args.mode in ("google", "both"):
        google_out.parent.mkdir(parents=True, exist_ok=True)
        api_key = os.environ["GOOGLE_MAPS_API_KEY"].strip()
        path_spec = polygon_path_latlng_for_google(
            geom_wgs84=geom_wgs,
            fill_rgb=fill_rgb,
            fill_alpha=args.google_fill_alpha,
            simplify_m=args.simplify_m,
            max_url_chars=7500,
        )
        fetch_google_static_map(
            path_spec=path_spec,
            visible_latlng=visible,
            width=args.google_width,
            height=args.google_height,
            api_key=api_key,
            maptype=args.google_maptype,
            out_path=google_out,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

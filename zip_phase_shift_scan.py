#!/usr/bin/env python3
"""
ZIP × year panel: city-demeaned median response times, single change-point per ZIP,
permutation p-value, Benjamini–Hochberg FDR across ZIPs.

For each ZIP with enough years and cell counts, searches the best split of the
ordered time series of residuals r_{z,t} = median_minutes_{z,t} − city_median_t.
Permutation shuffles residuals across years (destroys time order) to calibrate
how often a split this good arises by chance.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import sqlite3

from ems_sql import incident_year_sql_expr

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


def load_incidents_zip_year_minutes(
    conn: sqlite3.Connection,
    table: str,
    metric: Metric,
) -> pd.DataFrame:
    seconds_col, valid_col = METRIC_CONFIG[metric]
    year_expr = incident_year_sql_expr()
    sql = f"""
    SELECT
      TRIM("zipcode") AS raw_zip,
      CAST({year_expr} AS INTEGER) AS year,
      CAST("{seconds_col}" AS REAL) / 60.0 AS minutes
    FROM "{table}"
    WHERE TRIM(COALESCE("zipcode", '')) != ''
      AND "{valid_col}" = 'Y'
      AND CAST("{seconds_col}" AS REAL) > 0
      AND CAST("{seconds_col}" AS REAL) < 86400
      AND TRIM(COALESCE("incident_datetime", '')) != ''
      AND ({year_expr}) GLOB '[0-9][0-9][0-9][0-9]'
    """
    df = pd.read_sql_query(sql, conn)
    df["geo_key"] = df["raw_zip"].map(normalize_zip_key)
    df = df.dropna(subset=["geo_key", "year"])
    df = df[df["year"].between(1990, 2100)]
    df["year"] = df["year"].astype(int)
    df = df.drop(columns=["raw_zip"])
    return df


def rss_one_segment(sum_x: float, sum_x2: float, n: int) -> float:
    if n <= 0:
        return 0.0
    return float(sum_x2 - (sum_x * sum_x) / n)


def best_single_break_rss_drop(
    x: np.ndarray,
    *,
    min_years_segment: int = 1,
) -> tuple[int, float, float]:
    """
    x = ordered residuals (one value per year). Minimize within-phase RSS for
    a single break: phase1 = indices 0..j, phase2 = j+1..K-1.

    Returns (j_best, rss_best, rss_drop) where rss_drop = rss_single_mean - rss_best.
    Only splits with at least min_years_segment years in each phase are allowed.
    """
    x = np.asarray(x, dtype=np.float64)
    k = len(x)
    if k < 2:
        raise ValueError("need at least 2 points")
    if min_years_segment < 1 or k < 2 * min_years_segment:
        raise ValueError("series too short for min_years_segment")
    pref_x = np.cumsum(x)
    pref_x2 = np.cumsum(x * x)
    total = pref_x[-1]
    total2 = pref_x2[-1]
    rss0 = rss_one_segment(total, total2, k)
    n1 = np.arange(1, k, dtype=np.int64)
    s1 = pref_x[:-1]
    ss1 = pref_x2[:-1] - s1 * s1 / n1
    s2 = total - s1
    n2 = (k - n1).astype(np.float64)
    ss2 = (total2 - pref_x2[:-1]) - s2 * s2 / n2
    rss_splits = ss1 + ss2
    valid = (n1 >= min_years_segment) & (n2 >= min_years_segment)
    if not valid.any():
        raise ValueError("no valid split")
    rss_splits = np.where(valid, rss_splits, np.inf)
    j = int(np.argmin(rss_splits))
    rss_b = float(rss_splits[j])
    return j, rss_b, float(rss0 - rss_b)


def permutation_pvalue(
    x: np.ndarray,
    rss_drop_obs: float,
    n_perm: int,
    rng: np.random.Generator,
    *,
    min_years_segment: int,
) -> float:
    """Empirical p-value: P(T* >= T) under shuffle of x across positions."""
    if n_perm <= 0:
        return float("nan")
    ge = 1
    for _ in range(n_perm):
        xs = rng.permutation(x)
        try:
            _, _, drop = best_single_break_rss_drop(xs, min_years_segment=min_years_segment)
        except ValueError:
            continue
        if drop >= rss_drop_obs - 1e-12:
            ge += 1
    return ge / (n_perm + 1)


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """BH-adjusted q-values; NaN p-values stay NaN."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    q = np.full(m, np.nan)
    finite = np.isfinite(pvals) & (pvals >= 0) & (pvals <= 1)
    if not finite.any():
        return q
    pf = pvals[finite]
    order = np.argsort(pf)
    ps = pf[order]
    ranks = np.arange(1, len(ps) + 1, dtype=float)
    bh = ps * len(ps) / ranks
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    bh = np.clip(bh, 0.0, 1.0)
    qf = np.empty_like(ps)
    qf[order] = bh
    q[finite] = qf
    return q


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, default=DEFAULT_DB, help="SQLite path")
    p.add_argument("--table", default=DEFAULT_TABLE, help="Incidents table")
    p.add_argument(
        "--metric",
        choices=("incident", "dispatch", "travel"),
        default="incident",
        help="Response-time column (default: incident)",
    )
    p.add_argument(
        "--min-n-cell",
        type=int,
        default=30,
        metavar="N",
        help="Drop ZIP-year cells with fewer than N incidents (default: 30)",
    )
    p.add_argument(
        "--min-years",
        type=int,
        default=8,
        metavar="T",
        help="Require at least T years with data after cell filter (default: 8)",
    )
    p.add_argument(
        "--min-years-segment",
        type=int,
        default=2,
        metavar="S",
        help="Each phase must span at least S years (default: 2)",
    )
    p.add_argument(
        "--min-total-incidents",
        type=int,
        default=500,
        metavar="N",
        help="Require at least N incidents in ZIP over all years (default: 500)",
    )
    p.add_argument(
        "--n-perm",
        type=int,
        default=499,
        help="Permutation replicates per ZIP (default: 499)",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("zip_phase_shifts.csv"),
        help="Output CSV (default: zip_phase_shifts.csv)",
    )
    args = p.parse_args()

    if args.min_years_segment < 1:
        print("--min-years-segment must be >= 1", file=sys.stderr)
        return 1
    if args.min_years < 2 * args.min_years_segment:
        print(
            "--min-years must be >= 2 * --min-years-segment so both phases can qualify.",
            file=sys.stderr,
        )
        return 1

    if not args.db.is_file():
        print(f"Database not found: {args.db}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    try:
        raw = load_incidents_zip_year_minutes(conn, args.table, args.metric)
    finally:
        conn.close()

    if raw.empty:
        print("No rows after filters.", file=sys.stderr)
        return 1

    city_med = raw.groupby("year", sort=True)["minutes"].median()
    panel = (
        raw.groupby(["geo_key", "year"], sort=True)
        .agg(median_min=("minutes", "median"), n=("minutes", "count"))
        .reset_index()
    )
    panel = panel[panel["n"] >= args.min_n_cell]
    panel["city_median_min"] = panel["year"].map(city_med)
    panel["r"] = panel["median_min"] - panel["city_median_min"]

    rows_out: list[dict] = []
    rng = np.random.default_rng(args.seed)

    for zip_key, g in panel.groupby("geo_key", sort=True):
        g = g.sort_values("year")
        years = g["year"].to_numpy(dtype=int)
        med = g["median_min"].to_numpy(dtype=np.float64)
        r = g["r"].to_numpy(dtype=np.float64)
        n_cell = g["n"].to_numpy(dtype=np.int64)
        k = len(g)
        if k < args.min_years:
            continue
        if int(n_cell.sum()) < args.min_total_incidents:
            continue

        try:
            k_best, _, rss_drop = best_single_break_rss_drop(
                r, min_years_segment=args.min_years_segment
            )
        except ValueError:
            continue
        n1 = k_best + 1
        n2 = k - k_best - 1

        y1 = years[k_best]
        y2 = years[k_best + 1]
        r1 = r[: n1]
        r2 = r[n1:]
        med1 = med[: n1]
        med2 = med[n1:]
        n_before = int(n_cell[:n1].sum())
        n_after = int(n_cell[n1:].sum())

        p_perm = permutation_pvalue(
            r,
            rss_drop,
            args.n_perm,
            rng,
            min_years_segment=args.min_years_segment,
        )

        rows_out.append(
            {
                "geo_key": zip_key,
                "break_after_year": int(y1),
                "phase2_start_year": int(y2),
                "n_years": k,
                "n_years_phase1": n1,
                "n_years_phase2": n2,
                "incidents_phase1": n_before,
                "incidents_phase2": n_after,
                "median_min_phase1": float(np.median(med1)),
                "median_min_phase2": float(np.median(med2)),
                "delta_median_min": float(np.median(med2) - np.median(med1)),
                "mean_r_phase1": float(np.mean(r1)),
                "mean_r_phase2": float(np.mean(r2)),
                "delta_mean_r_min": float(np.mean(r2) - np.mean(r1)),
                "rss_drop": float(rss_drop),
                "p_perm": float(p_perm),
            }
        )

    if not rows_out:
        print("No ZIPs passed filters; relax thresholds or check data.", file=sys.stderr)
        return 1

    out = pd.DataFrame(rows_out)
    out["q_fdr"] = benjamini_hochberg(out["p_perm"].to_numpy())
    out = out.sort_values("p_perm", ascending=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(
        f"Wrote {args.output.resolve()} ({len(out)} ZIPs; "
        f"metric={args.metric}, min_n_cell={args.min_n_cell}, n_perm={args.n_perm})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

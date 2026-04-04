#!/usr/bin/env python3
"""
Download NYC EMS incident response time data from the Socrata SODA API
and load it into a local SQLite database.

Dataset: https://data.cityofnewyork.us/resource/76xm-jjuj.json
Docs: https://dev.socrata.com/foundry/data.cityofnewyork.us/76xm-jjuj
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

BASE_URL = "https://data.cityofnewyork.us/resource/76xm-jjuj.json"
DEFAULT_DB = "ems_incidents.sqlite"
DEFAULT_TABLE = "ems_incidents"
PAGE_SIZE = 50_000
REQUEST_TIMEOUT_S = 120
MAX_RETRIES = 5
BACKOFF_S = 2.0


def fetch_page(
    offset: int,
    limit: int,
    extra_headers: dict[str, str],
) -> list[dict[str, Any]]:
    qs = urllib.parse.urlencode({"$limit": str(limit), "$offset": str(offset)})
    url = f"{BASE_URL}?{qs}"
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers=extra_headers)
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
                raw = resp.read()
            data = json.loads(raw.decode("utf-8"))
            if not isinstance(data, list):
                raise TypeError(f"Expected JSON array, got {type(data).__name__}")
            return data
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TypeError) as e:
            last_exc = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_S * (2**attempt))
    assert last_exc is not None
    raise last_exc


def ensure_table(
    conn: sqlite3.Connection,
    table: str,
    columns: list[str],
) -> None:
    cols_sql = ", ".join(f'"{c}" TEXT' for c in columns)
    conn.execute(f'CREATE TABLE IF NOT EXISTS "{table}" ({cols_sql})')
    conn.execute(
        f'CREATE UNIQUE INDEX IF NOT EXISTS "{table}_cad_incident_id" '
        f'ON "{table}" ("cad_incident_id")'
    )


def row_values(row: dict[str, Any], columns: list[str]) -> tuple[Any, ...]:
    out: list[Any] = []
    for c in columns:
        v = row.get(c)
        if v is None:
            out.append(None)
        elif isinstance(v, (dict, list)):
            out.append(str(v))
        else:
            out.append(v)
    return tuple(out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-o",
        "--output",
        default=DEFAULT_DB,
        help=f"SQLite database file (default: {DEFAULT_DB})",
    )
    p.add_argument(
        "--table",
        default=DEFAULT_TABLE,
        help=f"Table name (default: {DEFAULT_TABLE})",
    )
    p.add_argument(
        "--replace",
        action="store_true",
        help="Drop the table if it exists, then reload from scratch",
    )
    p.add_argument(
        "--page-size",
        type=int,
        default=PAGE_SIZE,
        metavar="N",
        help=f"Rows per API request (default: {PAGE_SIZE})",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        metavar="N",
        help="Stop after importing this many rows (useful for testing)",
    )
    args = p.parse_args()

    headers: dict[str, str] = {"Accept": "application/json"}
    token = os.environ.get("SOCRATA_APP_TOKEN")
    if token:
        headers["X-App-Token"] = token

    db_path = args.output
    conn = sqlite3.connect(db_path)
    try:
        if args.replace:
            conn.execute(f'DROP TABLE IF EXISTS "{args.table}"')

        offset = 0
        total = 0
        columns: list[str] | None = None
        placeholders: str | None = None
        insert_sql: str | None = None

        page_size = max(1, args.page_size)

        while True:
            limit = page_size
            if args.max_rows is not None:
                remaining = args.max_rows - total
                if remaining <= 0:
                    break
                limit = min(limit, remaining)

            rows = fetch_page(offset, limit, headers)
            if not rows:
                break

            if columns is None:
                columns = sorted({k for r in rows for k in r.keys()})
                ensure_table(conn, args.table, columns)
                placeholders = ", ".join("?" * len(columns))
                col_names = ", ".join(f'"{c}"' for c in columns)
                insert_sql = (
                    f'INSERT OR REPLACE INTO "{args.table}" ({col_names}) '
                    f"VALUES ({placeholders})"
                )

            assert columns is not None and placeholders is not None and insert_sql is not None
            conn.executemany(
                insert_sql,
                [row_values(r, columns) for r in rows],
            )
            conn.commit()

            n = len(rows)
            total += n
            print(f"Fetched offset={offset}, rows={n}, total={total}", file=sys.stderr)
            offset += n
            if n < limit:
                break
            if args.max_rows is not None and total >= args.max_rows:
                break

        print(f"Done. Wrote {total} rows to {db_path} table {args.table}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())

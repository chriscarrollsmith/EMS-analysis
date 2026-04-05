#!/usr/bin/env python3
"""
Download NYC EMS incident response time data from the Socrata SODA API (default)
or load from a CSV export, and store rows in a local SQLite database.

Re-running without --replace resumes from the last committed page (checkpoint
in _ems_fetch_checkpoint). Use --replace to drop the table and reload.

Dataset: https://data.cityofnewyork.us/resource/76xm-jjuj.json
Docs: https://dev.socrata.com/foundry/data.cityofnewyork.us/76xm-jjuj
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from ems_sql import normalize_datetime_field

BASE_URL = "https://data.cityofnewyork.us/resource/76xm-jjuj.json"
DEFAULT_DB = "ems_incidents.sqlite"
DEFAULT_TABLE = "ems_incidents"
CHECKPOINT_TABLE = "_ems_fetch_checkpoint"
ORDER_COLUMN = "cad_incident_id"
PAGE_SIZE = 50_000
REQUEST_TIMEOUT_S = 120
MAX_RETRIES = 5
BACKOFF_S = 2.0


def fetch_page(
    offset: int,
    limit: int,
    extra_headers: dict[str, str],
) -> list[dict[str, Any]]:
    qs = urllib.parse.urlencode(
        {
            "$limit": str(limit),
            "$offset": str(offset),
            "$order": ORDER_COLUMN,
        }
    )
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


def ensure_checkpoint_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        f'CREATE TABLE IF NOT EXISTS "{CHECKPOINT_TABLE}" ('
        "table_name TEXT PRIMARY KEY, "
        "next_offset INTEGER NOT NULL)"
    )


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
        (table,),
    )
    return cur.fetchone() is not None


def get_checkpoint(conn: sqlite3.Connection, table: str) -> int | None:
    cur = conn.execute(
        f'SELECT next_offset FROM "{CHECKPOINT_TABLE}" WHERE table_name = ?',
        (table,),
    )
    row = cur.fetchone()
    return int(row[0]) if row else None


def set_checkpoint(conn: sqlite3.Connection, table: str, next_offset: int) -> None:
    ensure_checkpoint_table(conn)
    conn.execute(
        f'INSERT INTO "{CHECKPOINT_TABLE}" (table_name, next_offset) VALUES (?, ?) '
        f'ON CONFLICT(table_name) DO UPDATE SET next_offset = excluded.next_offset',
        (table, next_offset),
    )


def clear_checkpoint(conn: sqlite3.Connection, table: str) -> None:
    ensure_checkpoint_table(conn)
    conn.execute(f'DELETE FROM "{CHECKPOINT_TABLE}" WHERE table_name = ?', (table,))


def row_values(row: dict[str, Any], columns: list[str]) -> tuple[Any, ...]:
    out: list[Any] = []
    for c in columns:
        v = row.get(c)
        if v is None:
            out.append(None)
        elif isinstance(v, (dict, list)):
            out.append(str(v))
        else:
            out.append(normalize_datetime_field(c, v))
    return tuple(out)


def _normalize_csv_fieldname(name: str | None) -> str:
    if not name:
        return ""
    return name.strip().lower()


def open_csv_dict_reader(path: str) -> tuple[csv.DictReader[str], Any]:
    """Open a UTF-8 CSV (with optional BOM) and return a DictReader with lowercase keys."""
    f = open(path, newline="", encoding="utf-8-sig")
    reader = csv.DictReader(f)
    if reader.fieldnames:
        reader.fieldnames = [_normalize_csv_fieldname(n) for n in reader.fieldnames]
    return reader, f


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
    p.add_argument(
        "--csv",
        metavar="PATH",
        default=None,
        help="Load from a CSV file instead of the API (headers normalized to lowercase)",
    )
    args = p.parse_args()

    db_path = args.output
    conn = sqlite3.connect(db_path)
    try:
        ensure_checkpoint_table(conn)
        if args.replace:
            conn.execute(f'DROP TABLE IF EXISTS "{args.table}"')
            clear_checkpoint(conn, args.table)
            conn.commit()

        offset = 0
        if not args.replace and table_exists(conn, args.table):
            cp = get_checkpoint(conn, args.table)
            if cp is not None and cp > 0:
                offset = cp
                n_existing = conn.execute(
                    f'SELECT COUNT(*) FROM "{args.table}"'
                ).fetchone()[0]
                print(
                    f"Resuming from offset={offset} ({n_existing} rows in table)",
                    file=sys.stderr,
                )
        total = 0
        columns: list[str] | None = None
        placeholders: str | None = None
        insert_sql: str | None = None

        page_size = max(1, args.page_size)
        broke_on_empty = False

        if args.csv:
            reader, csv_file = open_csv_dict_reader(args.csv)
            try:
                skipped = 0
                while skipped < offset:
                    if next(reader, None) is None:
                        clear_checkpoint(conn, args.table)
                        conn.commit()
                        print(f"Done. Wrote {total} rows to {db_path} table {args.table}")
                        return 0
                    skipped += 1

                while True:
                    limit = page_size
                    if args.max_rows is not None:
                        remaining = args.max_rows - total
                        if remaining <= 0:
                            break
                        limit = min(limit, remaining)

                    batch: list[dict[str, Any]] = []
                    for _ in range(limit):
                        raw = next(reader, None)
                        if raw is None:
                            break
                        batch.append({k: (v if v != "" else None) for k, v in raw.items()})

                    if not batch:
                        broke_on_empty = True
                        break

                    if columns is None:
                        columns = sorted({k for r in batch for k in r.keys() if k})
                        ensure_table(conn, args.table, columns)
                        placeholders = ", ".join("?" * len(columns))
                        col_names = ", ".join(f'"{c}"' for c in columns)
                        insert_sql = (
                            f'INSERT OR REPLACE INTO "{args.table}" ({col_names}) '
                            f"VALUES ({placeholders})"
                        )

                    assert (
                        columns is not None
                        and placeholders is not None
                        and insert_sql is not None
                    )
                    conn.executemany(
                        insert_sql,
                        [row_values(r, columns) for r in batch],
                    )
                    n = len(batch)
                    next_offset = offset + n
                    set_checkpoint(conn, args.table, next_offset)
                    conn.commit()

                    total += n
                    print(
                        f"CSV offset={offset}, rows={n}, total={total}",
                        file=sys.stderr,
                    )
                    offset = next_offset
                    if n < limit:
                        clear_checkpoint(conn, args.table)
                        conn.commit()
                        break
                    if args.max_rows is not None and total >= args.max_rows:
                        break
            finally:
                csv_file.close()
        else:
            headers: dict[str, str] = {"Accept": "application/json"}
            token = os.environ.get("SOCRATA_APP_TOKEN")
            if token:
                headers["X-App-Token"] = token

            while True:
                limit = page_size
                if args.max_rows is not None:
                    remaining = args.max_rows - total
                    if remaining <= 0:
                        break
                    limit = min(limit, remaining)

                rows = fetch_page(offset, limit, headers)
                if not rows:
                    broke_on_empty = True
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

                assert (
                    columns is not None
                    and placeholders is not None
                    and insert_sql is not None
                )
                conn.executemany(
                    insert_sql,
                    [row_values(r, columns) for r in rows],
                )
                n = len(rows)
                next_offset = offset + n
                set_checkpoint(conn, args.table, next_offset)
                conn.commit()

                total += n
                print(
                    f"Fetched offset={offset}, rows={n}, total={total}",
                    file=sys.stderr,
                )
                offset = next_offset
                if n < limit:
                    clear_checkpoint(conn, args.table)
                    conn.commit()
                    break
                if args.max_rows is not None and total >= args.max_rows:
                    break

        if broke_on_empty:
            clear_checkpoint(conn, args.table)
            conn.commit()

        print(f"Done. Wrote {total} rows to {db_path} table {args.table}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Download the NYC Open Data EMS incident dispatch codebook (XLSX) and emit JSON.

Source: https://data.cityofnewyork.us/api/views/76xm-jjuj/files/5f77cf01-4e52-443b-a718-a3e6567e83f2
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from io import BytesIO
from typing import Any

CODEBOOK_URL = (
    "https://data.cityofnewyork.us/api/views/76xm-jjuj/files/"
    "5f77cf01-4e52-443b-a718-a3e6567e83f2"
    "?download=true&filename=EMS_incident_dispatch_data_description.xlsx"
)

NS_MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
NS_PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
NS_DOC_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

REF_RE = re.compile(r"^([A-Z]+)(\d+)$")


def col_index(letters: str) -> int:
    n = 0
    for c in letters:
        n = n * 26 + (ord(c) - ord("A") + 1)
    return n - 1


def parse_cell_ref(ref: str) -> tuple[int, int]:
    m = REF_RE.match(ref)
    if not m:
        return 0, 0
    return col_index(m.group(1)), int(m.group(2)) - 1


def slug_header(s: str | None, i: int) -> str:
    if not s or not str(s).strip():
        return f"column_{i}"
    t = str(s).strip().lower()
    t = re.sub(r"[^a-z0-9]+", "_", t)
    return t.strip("_") or f"column_{i}"


def read_shared_strings(z: zipfile.ZipFile) -> list[str]:
    try:
        raw = z.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ET.fromstring(raw)
    out: list[str] = []
    for si in root.findall(f"{NS_MAIN}si"):
        parts: list[str] = []
        for te in si.iter(f"{NS_MAIN}t"):
            parts.append(te.text or "")
        out.append("".join(parts))
    return out


def cell_value(c: ET.Element, shared: list[str]) -> str | None:
    t = c.get("t")
    v_el = c.find(f"{NS_MAIN}v")
    if v_el is None or v_el.text is None:
        return None
    raw = v_el.text
    if t == "s":
        return shared[int(raw)]
    return raw


def read_worksheet_rows(z: zipfile.ZipFile, path: str, shared: list[str]) -> dict[int, dict[int, str | None]]:
    root = ET.fromstring(z.read(path))
    rows: dict[int, dict[int, str | None]] = {}
    sd = root.find(f"{NS_MAIN}sheetData")
    if sd is None:
        return rows
    for row in sd.findall(f"{NS_MAIN}row"):
        r_attr = row.get("r")
        row_idx = int(r_attr) - 1 if r_attr else len(rows)
        cells: dict[int, str | None] = {}
        for c in row.findall(f"{NS_MAIN}c"):
            ref = c.get("r")
            if not ref:
                continue
            col_i, _ = parse_cell_ref(ref)
            cells[col_i] = cell_value(c, shared)
        rows[row_idx] = cells
    return rows


def rows_to_records(rows: dict[int, dict[int, str | None]]) -> list[dict[str, str | None]]:
    if not rows:
        return []
    max_col = max(max(c.keys()) for c in rows.values() if c)

    def row_list(ridx: int) -> list[str | None]:
        cells = rows.get(ridx, {})
        return [cells.get(i) for i in range(max_col + 1)]

    sorted_rows = sorted(rows.keys())
    header_idx = sorted_rows[0]
    headers = row_list(header_idx)
    if not any(h is not None and str(h).strip() for h in headers):
        return []

    keys = [slug_header(h, i) for i, h in enumerate(headers)]
    records: list[dict[str, str | None]] = []

    for ridx in sorted_rows[1:]:
        vals = row_list(ridx)
        if all(v is None or (isinstance(v, str) and v.strip() == "") for v in vals):
            continue
        records.append(dict(zip(keys, vals)))
    return records


def sheet_paths_in_order(z: zipfile.ZipFile) -> list[tuple[str, str]]:
    wb = ET.fromstring(z.read("xl/workbook.xml"))
    rels_root = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
    rid_to_target: dict[str, str] = {}
    for rel in rels_root.findall(f"{{{NS_PKG_REL}}}Relationship"):
        if "worksheet" in (rel.get("Type") or ""):
            tid = rel.get("Id")
            tgt = rel.get("Target")
            if tid and tgt:
                rid_to_target[tid] = "xl/" + tgt.lstrip("/")

    sheets_el = wb.find(f"{NS_MAIN}sheets")
    if sheets_el is None:
        return []
    out: list[tuple[str, str]] = []
    for sh in sheets_el.findall(f"{NS_MAIN}sheet"):
        name = sh.get("name") or "Sheet"
        rid = sh.get(f"{{{NS_DOC_REL}}}id")
        if rid and rid in rid_to_target:
            out.append((name, rid_to_target[rid]))
    return out


def xlsx_bytes_to_codebook(data: bytes, source: str) -> dict[str, Any]:
    with zipfile.ZipFile(BytesIO(data)) as z:
        shared = read_shared_strings(z)
        sheets_out: list[dict[str, Any]] = []
        for name, path in sheet_paths_in_order(z):
            grid = read_worksheet_rows(z, path, shared)
            sheets_out.append(
                {
                    "name": name,
                    "records": rows_to_records(grid),
                }
            )
    return {
        "source_url": source,
        "sheets": sheets_out,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-o",
        "--output",
        default="ems_codebook.json",
        help="Output JSON path (default: ems_codebook.json)",
    )
    p.add_argument(
        "--input",
        default=None,
        help="Local .xlsx path instead of downloading",
    )
    args = p.parse_args()

    if args.input:
        with open(args.input, "rb") as f:
            data = f.read()
        src = f"file:{args.input}"
    else:
        req = urllib.request.Request(CODEBOOK_URL, headers={"User-Agent": "EMS-analysis/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
        src = CODEBOOK_URL

    doc = xlsx_bytes_to_codebook(data, src)
    for sh in doc["sheets"]:
        sh["records"] = [
            {k: v for k, v in rec.items() if v is not None}
            for rec in sh["records"]
        ]
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

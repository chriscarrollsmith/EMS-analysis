from __future__ import annotations

from datetime import datetime
from typing import Any

CANONICAL_DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S"

_DATETIME_FORMATS = (
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%m/%d/%Y %I:%M:%S %p",
)


def normalize_datetime_text(value: str | None) -> str | None:
    """Normalize supported timestamp strings to YYYY-MM-DDTHH:MM:SS."""
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    for fmt in _DATETIME_FORMATS:
        try:
            return datetime.strptime(text, fmt).strftime(CANONICAL_DATETIME_FORMAT)
        except ValueError:
            pass
    return text


def normalize_datetime_field(column_name: str, value: Any) -> Any:
    if value is None or not column_name.endswith("_datetime"):
        return value
    if isinstance(value, str):
        return normalize_datetime_text(value)
    return value


def incident_year_sql_expr(column_name: str = "incident_datetime") -> str:
    """SQLite expression that extracts a year from canonical ISO datetime text."""
    text_expr = f'TRIM(COALESCE("{column_name}", \'\'))'
    return f"NULLIF(trim(strftime('%Y', {text_expr})), '')"

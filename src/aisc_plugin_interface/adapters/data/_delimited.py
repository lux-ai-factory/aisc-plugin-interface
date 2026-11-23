"""A shared delimited-text reader: CSV and TSV differ only in their separator."""
from __future__ import annotations

import csv
import io
from pathlib import Path


def read_rows(source: bytes | str | Path, delimiter: str) -> list[dict]:
    """Parse delimited text into a list of dicts, one per row."""
    if isinstance(source, Path):
        text = source.read_text(encoding="utf-8")
    elif isinstance(source, bytes):
        text = source.decode("utf-8")
    else:
        text = source
    return list(csv.DictReader(io.StringIO(text), delimiter=delimiter))


def rows_to_records(rows: list[dict]) -> list[dict]:
    return rows


def rows_to_table(rows: list[dict]):
    import pandas as pd

    return pd.DataFrame(rows)

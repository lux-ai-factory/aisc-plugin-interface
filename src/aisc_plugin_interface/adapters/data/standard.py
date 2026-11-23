"""The standard data adapters (ADR 0009), ported from the plugin interface's input
providers and extended with TSV and JSON Lines.

Each reads one file format and produces the value type a plugin slot accepts
(``table``, ``records``, ``json``, ``onnx-session``, ``file``). Heavy libraries
(pandas, onnxruntime) are imported lazily, so a plugin that reads CSV needs
neither.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from aisc_plugin_interface.adapters.base import DataAdapter
from aisc_plugin_interface.adapters.data._delimited import read_rows, rows_to_table

_SOURCE = bytes | str | Path


class CsvDataAdapter(DataAdapter):
    id = "csv"
    label = "CSV"
    reads = ("csv",)
    produces = ("table", "records")
    delimiter = ","

    def load(self, source: _SOURCE, produce: str) -> Any:
        # ``records`` is raw string rows (as ``csv.DictReader`` gives); ``table``
        # is a **typed** DataFrame via pandas, so numeric columns stay numeric —
        # what an ML plugin expects (and what the old providers gave).
        if produce == "table":
            return self._read_table(source)
        return read_rows(source, self.delimiter)

    def _read_table(self, source: _SOURCE):
        import pandas as pd

        if isinstance(source, Path):
            return pd.read_csv(source, delimiter=self.delimiter)
        if isinstance(source, bytes):
            return pd.read_csv(io.BytesIO(source), delimiter=self.delimiter)
        return pd.read_csv(io.StringIO(source), delimiter=self.delimiter)


class TsvDataAdapter(CsvDataAdapter):
    id = "tsv"
    label = "TSV"
    reads = ("tsv",)
    delimiter = "\t"


class ParquetDataAdapter(DataAdapter):
    id = "parquet"
    label = "Parquet"
    reads = ("parquet",)
    produces = ("table", "records")

    def load(self, source: _SOURCE, produce: str) -> Any:
        import pandas as pd

        if isinstance(source, Path):
            frame = pd.read_parquet(source)
        else:
            frame = pd.read_parquet(io.BytesIO(source))
        if produce == "records":
            return frame.to_dict(orient="records")
        return frame


class JsonLinesDataAdapter(DataAdapter):
    id = "jsonl"
    label = "JSON Lines"
    reads = ("jsonl",)
    produces = ("table", "records")

    def load(self, source: _SOURCE, produce: str) -> Any:
        import json

        if isinstance(source, Path):
            text = source.read_text(encoding="utf-8")
        elif isinstance(source, bytes):
            text = source.decode("utf-8")
        else:
            text = source
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        if produce == "records":
            return rows
        return rows_to_table(rows)


class JsonDataAdapter(DataAdapter):
    id = "json"
    label = "JSON"
    reads = ("json",)
    produces = ("json",)

    def load(self, source: _SOURCE, produce: str) -> Any:
        import json

        if isinstance(source, Path):
            return json.loads(source.read_text(encoding="utf-8"))
        if isinstance(source, bytes):
            return json.loads(source.decode("utf-8"))
        return source


class OnnxDataAdapter(DataAdapter):
    id = "onnx"
    label = "ONNX"
    reads = ("onnx",)
    produces = ("onnx-session",)

    def load(self, source: _SOURCE, produce: str) -> Any:
        import onnxruntime as ort

        if isinstance(source, Path):
            return ort.InferenceSession(str(source))
        # onnxruntime takes the serialized model bytes or a path; it does not
        # accept a BytesIO (the eval runtime always hands us bytes).
        return ort.InferenceSession(source)


class RawFileDataAdapter(DataAdapter):
    """Hand the plugin the bytes (or the path) unchanged, for a slot that accepts
    ``file`` and wants to read the format itself."""

    id = "file"
    label = "Raw file"
    reads = ("*",)
    produces = ("file",)

    def load(self, source: _SOURCE, produce: str) -> Any:
        return source


#: The standard library, in catalogue order.
BUILTIN_DATA_ADAPTERS: tuple[type[DataAdapter], ...] = (
    CsvDataAdapter,
    TsvDataAdapter,
    ParquetDataAdapter,
    JsonLinesDataAdapter,
    JsonDataAdapter,
    OnnxDataAdapter,
    RawFileDataAdapter,
)

__all__ = [
    "CsvDataAdapter",
    "TsvDataAdapter",
    "ParquetDataAdapter",
    "JsonLinesDataAdapter",
    "JsonDataAdapter",
    "OnnxDataAdapter",
    "RawFileDataAdapter",
    "BUILTIN_DATA_ADAPTERS",
]

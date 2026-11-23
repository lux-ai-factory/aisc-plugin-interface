"""Standard data adapters: a file format -> the value a slot accepts."""
from .standard import (
    BUILTIN_DATA_ADAPTERS,
    CsvDataAdapter,
    JsonDataAdapter,
    JsonLinesDataAdapter,
    OnnxDataAdapter,
    ParquetDataAdapter,
    RawFileDataAdapter,
    TsvDataAdapter,
)

__all__ = [
    "BUILTIN_DATA_ADAPTERS",
    "CsvDataAdapter",
    "TsvDataAdapter",
    "ParquetDataAdapter",
    "JsonLinesDataAdapter",
    "JsonDataAdapter",
    "OnnxDataAdapter",
    "RawFileDataAdapter",
]

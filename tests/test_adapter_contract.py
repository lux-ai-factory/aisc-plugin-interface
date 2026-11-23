"""The adapter vocabulary and the standard data adapters (ADR 0009 phase 1)."""
from __future__ import annotations

import io
import json

import pytest

from aisc_plugin_interface import adapters
from aisc_plugin_interface.adapters import registry
from aisc_plugin_interface.adapters.data import (
    CsvDataAdapter,
    JsonDataAdapter,
    JsonLinesDataAdapter,
    ParquetDataAdapter,
    TsvDataAdapter,
)


def test_abi_version_is_set():
    assert registry.ADAPTER_ABI_VERSION == "1"


def test_the_built_in_vocabulary_is_registered():
    assert {"csv", "tsv", "parquet", "jsonl", "json", "onnx"} <= {f.id for f in registry.formats()}
    assert {"table", "records", "json", "onnx-session", "file", "resource"} <= {
        v.id for v in registry.value_types()
    }
    assert {"openai-chat", "agent", "oip-v2", "rest", "url"} <= {p.id for p in registry.protocols()}
    assert {p.id for p in registry.face_protocols()} == {"openai-chat", "agent"}


def test_reregistering_a_protocol_identically_is_a_noop():
    first = registry.register_protocol("custom-x", "Custom X")
    second = registry.register_protocol("custom-x", "Custom X")
    assert first is second
    with pytest.raises(ValueError):
        registry.register_protocol("custom-x", "Different")


def test_csv_and_tsv_share_one_reader():
    data = b"a,b\n1,2\n3,4\n"
    assert CsvDataAdapter().load(data, "records") == [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]
    tsv = b"a\tb\n1\t2\n"
    assert TsvDataAdapter().load(tsv, "records") == [{"a": "1", "b": "2"}]


def test_csv_produces_a_table():
    pd = pytest.importorskip("pandas")
    frame = CsvDataAdapter().load(b"a,b\n1,2\n", "table")
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == ["a", "b"]
    # ``table`` is typed (pandas), so a numeric column stays numeric — unlike
    # ``records``, which are raw strings.
    assert pd.api.types.is_integer_dtype(frame["a"])


def test_jsonl_produces_records_and_table():
    pd = pytest.importorskip("pandas")
    data = b'{"a": 1}\n{"a": 2}\n'
    assert JsonLinesDataAdapter().load(data, "records") == [{"a": 1}, {"a": 2}]
    assert len(JsonLinesDataAdapter().load(data, "table")) == 2


def test_json_produces_a_document():
    assert JsonDataAdapter().load(b'{"a": 1}', "json") == {"a": 1}


def test_parquet_produces_a_table_and_records():
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    buf = io.BytesIO()
    pd.DataFrame({"a": [1, 2]}).to_parquet(buf)
    adapter = ParquetDataAdapter()
    assert list(adapter.load(buf.getvalue(), "table")["a"]) == [1, 2]
    assert adapter.load(buf.getvalue(), "records") == [{"a": 1}, {"a": 2}]


def test_the_raw_file_adapter_hands_back_the_source():
    payload = b"\x00\x01"
    assert adapters.BUILTIN_DATA_ADAPTERS[-1].id == "file"
    assert adapters.BUILTIN_DATA_ADAPTERS[-1]().load(payload, "file") == payload


def test_data_adapters_are_discoverable():
    ids = {a.id for a in adapters.BUILTIN_DATA_ADAPTERS}
    assert ids == {"csv", "tsv", "parquet", "jsonl", "json", "onnx", "file"}


def test_onnx_adapter_passes_bytes_not_a_bytesio(monkeypatch):
    """onnxruntime takes serialized model bytes (or a path); a BytesIO fails with
    `Unable to load from type BytesIO`. The eval runtime always hands us bytes."""
    import sys
    import types

    from aisc_plugin_interface.adapters.data.standard import OnnxDataAdapter

    seen: dict = {}
    fake = types.ModuleType("onnxruntime")

    def _inference_session(arg, *args, **kwargs):
        seen["arg"] = arg
        return object()

    fake.InferenceSession = _inference_session
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)

    OnnxDataAdapter().load(b"model-bytes", "onnx-session")
    assert seen["arg"] == b"model-bytes"
    assert not isinstance(seen["arg"], io.BytesIO)

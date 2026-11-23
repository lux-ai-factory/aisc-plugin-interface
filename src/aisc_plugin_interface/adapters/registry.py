"""
The adapter vocabulary: the protocol, format and value-type ids as **registered
strings**, not schema choices (ADR 0009 D-12).

Adding a protocol, format or value type is a code registration, never a
migration. The backend, the plugin-manager and the eval worker all read the same
registries, so "what is this and what can it talk to" has one answer.
"""
from __future__ import annotations

from dataclasses import dataclass

#: The revision of the adapter contract in this subpackage. A run records this
#: alongside the adapter id and version, so "which contract made this answer" is
#: answered by the run, not by today's code.
ADAPTER_ABI_VERSION = "1"


@dataclass(frozen=True)
class ProtocolInfo:
    """A protocol the platform speaks, on either side of an adapter."""

    id: str
    label: str
    #: The *face* protocols a plugin can be handed: a plugin input declares one of
    #: these in its ``accepts`` list. A target protocol (what the system under test
    #: speaks) need not be a face.
    face: bool = False


@dataclass(frozen=True)
class FormatInfo:
    """A file format a data adapter reads."""

    id: str
    label: str


@dataclass(frozen=True)
class ValueTypeInfo:
    """A value a slot accepts and a data adapter produces (a DataFrame, a list of
    records, a raw file, an ONNX session ...)."""

    id: str
    label: str


_PROTOCOLS: dict[str, ProtocolInfo] = {}
_FORMATS: dict[str, FormatInfo] = {}
_VALUE_TYPES: dict[str, ValueTypeInfo] = {}


def register_protocol(id: str, label: str, *, face: bool = False) -> ProtocolInfo:
    """Register a protocol. Re-registering the same id with the same label is a
    no-op, so an adapter module and the base library can both declare the built-ins
    without clashing."""
    existing = _PROTOCOLS.get(id)
    if existing is not None:
        if existing.label != label or existing.face != face:
            raise ValueError(f"protocol {id!r} is already registered differently")
        return existing
    info = ProtocolInfo(id=id, label=label, face=face)
    _PROTOCOLS[id] = info
    return info


def register_format(id: str, label: str) -> FormatInfo:
    existing = _FORMATS.get(id)
    if existing is not None:
        if existing.label != label:
            raise ValueError(f"format {id!r} is already registered differently")
        return existing
    info = FormatInfo(id=id, label=label)
    _FORMATS[id] = info
    return info


def register_value_type(id: str, label: str) -> ValueTypeInfo:
    existing = _VALUE_TYPES.get(id)
    if existing is not None:
        if existing.label != label:
            raise ValueError(f"value type {id!r} is already registered differently")
        return existing
    info = ValueTypeInfo(id=id, label=label)
    _VALUE_TYPES[id] = info
    return info


def protocol(id: str) -> ProtocolInfo | None:
    return _PROTOCOLS.get(id)


def format(id: str) -> FormatInfo | None:
    return _FORMATS.get(id)


def value_type(id: str) -> ValueTypeInfo | None:
    return _VALUE_TYPES.get(id)


def protocols() -> list[ProtocolInfo]:
    return list(_PROTOCOLS.values())


def face_protocols() -> list[ProtocolInfo]:
    return [info for info in _PROTOCOLS.values() if info.face]


def formats() -> list[FormatInfo]:
    return list(_FORMATS.values())


def value_types() -> list[ValueTypeInfo]:
    return list(_VALUE_TYPES.values())


# The built-in vocabulary. Adapters and plugin packages register more through the
# same functions; nothing below is a special case.
for _id, _label, _face in (
    ("openai-chat", "OpenAI-compatible chat", True),
    ("agent", "A2A agent", True),
    ("oip-v2", "Open Inference Protocol (KServe v2)", False),
    ("rest", "REST API", False),
    ("url", "Raw URL", False),
):
    register_protocol(_id, _label, face=_face)

for _id, _label in (
    ("csv", "CSV"),
    ("tsv", "TSV"),
    ("parquet", "Parquet"),
    ("jsonl", "JSON Lines"),
    ("json", "JSON"),
    ("onnx", "ONNX"),
):
    register_format(_id, _label)

for _id, _label in (
    ("table", "Table (a DataFrame)"),
    ("records", "Records (a list of dicts)"),
    ("json", "JSON document"),
    ("onnx-session", "ONNX inference session"),
    ("file", "Raw file"),
    ("resource", "Resource reference"),
):
    register_value_type(_id, _label)

__all__ = [
    "ADAPTER_ABI_VERSION",
    "ProtocolInfo",
    "FormatInfo",
    "ValueTypeInfo",
    "register_protocol",
    "register_format",
    "register_value_type",
    "protocol",
    "format",
    "value_type",
    "protocols",
    "face_protocols",
    "formats",
    "value_types",
]

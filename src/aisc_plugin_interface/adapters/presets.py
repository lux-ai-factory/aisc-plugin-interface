"""
The task-shaped presets (ADR 0009 D-10): the simple forms "Add LLM", "Add agent",
"Add model API", "Add dataset", "Add model file".

A preset is a starting point, not a mode: it fills a component's columns and
adapter config, and "Custom" opens the adapter-plus-config form for everything
else. Presets are defined in code, so adding one is a registration, not a
migration; a plugin package may ship its own.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PresetField:
    """One field of a preset's form.

    ``target`` is where the value goes: a component column (``url``, ``model``,
    ``secret``) or the adapter's config (``config``, with ``key`` naming the
    config field).
    """

    key: str
    label: str
    target: str = "config"
    config_key: str = ""
    type: str = "string"
    required: bool = False
    secret: bool = False
    default: Any = None


@dataclass(frozen=True)
class Preset:
    id: str
    label: str
    kind: str
    protocol: str | None = None
    format: str | None = None
    adapter: str | None = None
    fields: tuple[PresetField, ...] = field(default_factory=tuple)


#: The endpoint presets carry the common columns; the rest goes to adapter config.
_LLM_FIELDS = (
    PresetField("url", "Endpoint URL", target="url", type="string", required=True),
    PresetField("secret", "API key", target="secret", type="string", secret=True),
    PresetField("model", "Model", target="model", type="string"),
)

PRESETS: dict[str, Preset] = {
    "llm": Preset(id="llm", label="LLM", kind="endpoint", protocol="openai-chat", adapter="openai", fields=_LLM_FIELDS),
    "agent": Preset(
        id="agent",
        label="Agent",
        kind="endpoint",
        protocol="agent",
        adapter="a2a",
        fields=(
            PresetField("url", "Base URL", target="url", required=True),
            PresetField("secret", "API key", target="secret", secret=True),
            PresetField("path", "JSON-RPC path", target="config", config_key="path"),
            PresetField("protocol_version", "A2A version", target="config", config_key="protocol_version", default="1.0"),
        ),
    ),
    "model-api": Preset(
        id="model-api",
        label="Model API",
        kind="endpoint",
        protocol="oip-v2",
        adapter="oip",
        fields=(
            PresetField("url", "Endpoint URL", target="url", required=True),
            PresetField("model", "Model", target="model", required=True),
            PresetField("secret", "API key", target="secret", secret=True),
        ),
    ),
    "dataset": Preset(id="dataset", label="Dataset", kind="dataset"),
    "model-file": Preset(id="model-file", label="Model file", kind="model"),
}


def preset(id: str) -> Preset | None:
    return PRESETS.get(id)


def all_presets() -> list[Preset]:
    return list(PRESETS.values())


__all__ = ["PresetField", "Preset", "PRESETS", "preset", "all_presets"]

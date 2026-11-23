"""
The adapter mechanism (ADR 0009): one place for the protocol/format/value-type
vocabulary, the two adapter kinds, the compatibility rule and the presets.

It is reused, unchanged, by three callers: the backend (the form's candidate list
and "Test"), the plugin-manager (adapter discovery) and the eval worker (resolving
inputs and running the door). The endpoint adapters run behind the run door, which
keeps the target key out of the plugin process.
"""
from __future__ import annotations

from aisc_plugin_interface.adapters.base import (
    Adapter,
    Answer,
    DataAdapter,
    EmptyConfig,
    EndpointAdapter,
    Request,
    Secret,
    Turn,
    context_of,
    json_path,
    parse_json,
    render,
    secret_headers,
)
from aisc_plugin_interface.adapters.compat import (
    accepts_of,
    data_candidates,
    data_fits,
    data_slot_candidates,
    endpoint_candidates,
    endpoint_fits,
    endpoint_slot_candidates,
    is_endpoint_kind,
)
from aisc_plugin_interface.adapters.data import BUILTIN_DATA_ADAPTERS
from aisc_plugin_interface.adapters.endpoint import (
    AGENT_CARD_PATH,
    BUILTIN_ENDPOINT_ADAPTERS,
    agent_card_url,
)
from aisc_plugin_interface.adapters.presets import PRESETS, Preset, PresetField, all_presets, preset
from aisc_plugin_interface.adapters import hosts, registry
from aisc_plugin_interface.adapters.registry import ADAPTER_ABI_VERSION

#: The standard library: data and endpoint adapters together.
BUILTIN_ADAPTERS: tuple[type[Adapter], ...] = (
    *BUILTIN_DATA_ADAPTERS,
    *BUILTIN_ENDPOINT_ADAPTERS,
)

__all__ = [
    "ADAPTER_ABI_VERSION",
    "BUILTIN_ADAPTERS",
    "BUILTIN_DATA_ADAPTERS",
    "BUILTIN_ENDPOINT_ADAPTERS",
    "Adapter",
    "DataAdapter",
    "EndpointAdapter",
    "EmptyConfig",
    "Secret",
    "Request",
    "Answer",
    "Turn",
    "context_of",
    "render",
    "json_path",
    "parse_json",
    "secret_headers",
    "accepts_of",
    "data_fits",
    "endpoint_fits",
    "data_candidates",
    "endpoint_candidates",
    "data_slot_candidates",
    "endpoint_slot_candidates",
    "is_endpoint_kind",
    "Preset",
    "PresetField",
    "PRESETS",
    "preset",
    "all_presets",
    "agent_card_url",
    "AGENT_CARD_PATH",
    "hosts",
    "registry",
]

"""What an evaluation assesses.

In the Configurator every evaluation names its target in its `target` input: the AI system, or one
component of its AI card (a model, a rule engine, its training data, ...). The value is
`target:<platform project pid>/<key>`, where key is `system` or `component:<the card's component
key>`. The platform keeps the targets; results are joined to them through the evaluation's input.
"""
from __future__ import annotations

import re
from typing import Any

from aisc_plugin_interface.connections import BadReference

TARGET_INPUT = "target"
TARGET_PREFIX = "target:"
_KEY = re.compile(r"^(system|component:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$")


def parse_target(value: Any) -> tuple[str, str]:
    """(project pid, target key) of a target reference; BadReference otherwise."""
    if not isinstance(value, str) or not value.startswith(TARGET_PREFIX):
        raise BadReference(f"not a target reference: {value!r}")
    pid, _, key = value[len(TARGET_PREFIX):].partition("/")
    if not pid or not _KEY.match(key or ""):
        raise BadReference(f"not a target reference: {value!r}")
    return pid, key


def is_target(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(TARGET_PREFIX)


def target_of(plugin, input_name: str = TARGET_INPUT) -> dict | None:
    """{"pid", "key"} of the target this run assesses, or None when the input is not bound."""
    payload = plugin.get_input_data(input_name)
    value = payload.get("value") if isinstance(payload, dict) else payload
    if value is None:
        return None
    pid, key = parse_target(value)
    return {"pid": pid, "key": key}

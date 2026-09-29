"""Declare the system under test a plugin's tool calls, and the protocol(s) the tool speaks.

    @system_under_test(protocols=("openai",),
                       fields={"target.base_url": "base_url", "target.api_key": "api_key", "target.model": "model"})
    class MyPlugin(BaseEvaluationPlugin): ...

The plugin gets a resource input (``system`` by default) that the evaluation form fills with a
connection from Manage → Connections. When a run has one bound, the platform issues the run a key for
that connection and the declared config fields and environment variables are set, for the length of
``evaluate``, to the platform's endpoint for the first protocol listed: the tool keeps connecting the
way it always does and reaches the connection. Whatever the connection is (REST, OpenAI-compatible,
A2A agent, OIP model server), the platform translates.

Roles, by protocol:
    aisc    ask_url, api_key                (POST {input, history} -> {output, refused, ...})
    openai  base_url, model, api_key        (Chat Completions, non-streaming)
    a2a     agent_card_url, rpc_url, api_key (A2A 1.0 JSON-RPC; 0.3 method names answered too)
    oip     base_url, model, api_key        (Open Inference Protocol, KServe V2 REST)

A tool written against AISC directly can use ``connections.EndpointClient.for_input`` instead.
"""
from __future__ import annotations

import copy
import functools
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from aisc_plugin_interface.connections import BadReference, reference_of
from aisc_plugin_interface.targets import is_target, parse_target
from aisc_plugin_interface.decorators.evaluation_input import evaluation_input
from aisc_plugin_interface.models.evaluation_input import InputType

#: The evaluation's target (targets plan v2): the system, or one of its AI card's components. A
#: `connection:<pid>/<name>` value (a connection picked before targets) is still honoured.
SYSTEM_INPUT = "target"

ROLES = {
    "aisc": ("ask_url", "api_key"),
    "openai": ("base_url", "model", "api_key"),
    "a2a": ("agent_card_url", "rpc_url", "api_key"),
    "oip": ("base_url", "model", "api_key"),
}


def system_under_test(protocols: tuple[str, ...] | list[str], fields: dict[str, str] | None = None,
                      env: dict[str, str] | None = None, required: bool = True,
                      input_name: str = SYSTEM_INPUT,
                      label: str = "Target of the assessment (the system, or one of its components)"):
    """See the module. ``fields`` maps a config field (dotted for nesting) to a role, ``env`` an
    environment variable to a role; the roles must be ones the first protocol has."""
    protocols = tuple(protocols)
    fields, env = dict(fields or {}), dict(env or {})
    if not protocols or any(p not in ROLES for p in protocols):
        raise ValueError(f"protocols: one or more of {', '.join(ROLES)}")
    if not fields and not env:
        raise ValueError("declare at least one config field or environment variable to fill")
    first = protocols[0]
    for target, role in {**fields, **env}.items():
        if role not in ROLES[first]:
            raise ValueError(f"{target}: {role!r} is not a role of {first} ({', '.join(ROLES[first])})")

    def decorator(cls):
        # Declared optional, so a form without targets (standalone) is not blocked; in the Configurator
        # the engine makes every evaluation's target required (targets plan v2, O2). `required` is
        # whether the tool needs the target's endpoint: without one, an optional tool runs as configured.
        cls = evaluation_input(name=input_name, label=label, input_type=InputType.RESOURCE, required=False)(cls)
        cls.system_under_test_protocols = protocols
        original = cls.evaluate

        @functools.wraps(original)
        def evaluate(self, config_data: dict) -> Any:
            if self.get_input_data(input_name) is None:
                if required:
                    raise ValueError(f"no system under test: bind a connection to input {input_name!r}")
                return original(self, config_data)
            payload = self.get_input_data(input_name)
            value = payload.get("value") if isinstance(payload, dict) else payload
            legacy = None
            try:
                if is_target(value):
                    pid, key = parse_target(value)
                    try:
                        issued = _issue_run_key(pid, f"targets/{urllib.parse.quote(key, safe=':')}", key)
                    except NoEndpoint:
                        if required:
                            raise
                        return original(self, config_data)
                else:
                    pid, legacy = reference_of(self, input_name)
                    issued = _issue_run_key(pid, f"connections/{urllib.parse.quote(legacy)}", legacy)
            except BadReference as exc:
                raise ValueError(f"system under test: {exc}") from None
            name = issued.get("connection") or legacy or "system"
            endpoint = issued["endpoints"][first]
            values = {**endpoint, "api_key": issued["key"]}
            config = copy.deepcopy(config_data) if config_data is not None else {}
            for field, role in fields.items():
                _set_dotted(config, field, values[role])
            self.upload_artifact(f"connection-{name}.json", json.dumps({
                "connection": f"connection:{pid}/{name}", "target": issued.get("target"),
                "protocol": first, "endpoint": endpoint,
                "expires_at": issued.get("expires_at"), "fields": sorted(fields), "env": sorted(env)},
                indent=2).encode())
            before = {var: os.environ.get(var) for var in env}
            try:
                for var, role in env.items():
                    os.environ[var] = values[role]
                return original(self, config)
            finally:
                for var, old in before.items():
                    if old is None:
                        os.environ.pop(var, None)
                    else:
                        os.environ[var] = old

        cls.evaluate = evaluate
        return cls

    return decorator


class NoEndpoint(RuntimeError):
    """The target has no endpoint (or is not a target of the project)."""


def _issue_run_key(pid: str, path: str, name: str) -> dict:
    """A run key from the platform; `path` is `targets/<key>` or, for a legacy value,
    `connections/<name>`. The platform's reason is passed on when it refuses."""
    base = (os.environ.get("PLATFORM_URL") or "").rstrip("/")
    token = os.environ.get("PLATFORM_CONNECTIONS_TOKEN") or ""
    if not base or not token:
        raise RuntimeError("PLATFORM_URL and PLATFORM_CONNECTIONS_TOKEN must be set to reach a system under test")
    req = urllib.request.Request(
        f"{base}/internal/projects/{urllib.parse.quote(pid)}/{path}/run-keys",
        data=b"", method="POST", headers={"X-AISC-Service-Token": token, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read() or b"{}").get("detail")
        except ValueError:
            detail = None
        reason = f": {detail}" if isinstance(detail, str) and detail else f" ({exc.code})"
        error = NoEndpoint if exc.code == 404 else RuntimeError
        raise error(f"the platform refused a run key for {name!r}{reason}") from None
    except (urllib.error.URLError, TimeoutError):
        raise RuntimeError(f"the platform could not be reached for connection {name!r}") from None


def _set_dotted(config: dict, path: str, value: Any) -> None:
    *parents, leaf = path.split(".")
    node = config
    for key in parents:
        node = node.setdefault(key, {})
    node[leaf] = value

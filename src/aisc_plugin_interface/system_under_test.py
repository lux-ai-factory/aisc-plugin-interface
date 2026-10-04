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

#: The evaluation's target: the system, or one of its AI card's components. A
#: ``connection:<pid>/<name>`` value (a connection bound directly, without a target) is accepted too.
SYSTEM_INPUT = "target"

ROLES = {
    "aisc": ("ask_url", "api_key"),
    "openai": ("base_url", "model", "api_key"),
    "a2a": ("agent_card_url", "rpc_url", "api_key"),
    "oip": ("base_url", "model", "api_key"),
}

#: What a plugin may need from its target's endpoint beyond a single answer. The platform's endpoint
#: carries text and a history; a run-key answer may list more (`capabilities`).
CAPABILITIES = ("text", "history", "tools", "logprobs")
CARRIED_BY_DEFAULT = ("text", "history")


def target_access_of(cls) -> str | None:
    """How a plugin class gets what it assesses: "endpoint" (@system_under_test), "inputs"
    (@assesses_inputs), "dataset_through_target" (@dataset_through_target), or None when it declares
    nothing (the conformance check fails it)."""
    return getattr(cls, "target_access", None)


def assesses_inputs():
    """Class decorator for a plugin that only reads its inputs (datasets, models, files) and calls no
    live system: nothing changes at run time, the declaration is what the conformance check reads."""
    def decorator(cls):
        if target_access_of(cls) not in (None, "inputs"):
            raise ValueError(f"{cls.__name__}: declare one of @system_under_test, @assesses_inputs or"
                             " @dataset_through_target, not two")
        cls.target_access = "inputs"
        return cls
    return decorator


def system_under_test(protocols: tuple[str, ...] | list[str], fields: dict[str, str] | None = None,
                      env: dict[str, str] | None = None, required: bool = True,
                      input_name: str = SYSTEM_INPUT,
                      label: str = "Target of the assessment (the system, or one of its components)",
                      needs: tuple[str, ...] | list[str] = ()):
    """Class decorator that points a tool at the run's system under test (see the module docstring).

    ``fields`` maps a config field (dotted for nesting) to a role, ``env`` maps an environment variable
    to a role; every role must belong to the first protocol. ``required`` says whether the tool needs
    an endpoint: when it is False and no target or endpoint is bound, ``evaluate`` runs unchanged.
    ``needs`` lists what the tool needs from the endpoint beyond text (CAPABILITIES): a run whose
    endpoint doesn't carry it is refused before the tool starts, with the reason.
    """
    protocols, needs = tuple(protocols), tuple(needs)
    unknown = [n for n in needs if n not in CAPABILITIES]
    if unknown:
        raise ValueError(f"needs: {', '.join(unknown)} (known: {', '.join(CAPABILITIES)})")
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
        if target_access_of(cls) not in (None, "endpoint"):
            raise ValueError(f"{cls.__name__}: declare one of @system_under_test, @assesses_inputs or"
                             " @dataset_through_target, not two")
        cls.target_access = "endpoint"
        cls.system_under_test_needs = needs
        # The input is declared optional so that a standalone form without targets is not blocked;
        # inside the Configurator the engine makes every evaluation's target required.
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
            carried = tuple(issued.get("capabilities") or CARRIED_BY_DEFAULT)
            missing = [n for n in needs if n not in carried]
            if missing:
                raise ValueError(f"{name} can't be used here: this plugin needs {', '.join(missing)} from its target,"
                                 f" and the endpoint carries {', '.join(carried)}")
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
    """Ask the platform for a run key; ``path`` is ``targets/<key>``, or ``connections/<name>`` for a
    connection bound directly. When the platform refuses, its reason is passed on."""
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


#: the run settings @dataset_through_target reads from a plugin's config, when its form has them
CALLS_AT_ONCE, ROW_LIMIT = "target_calls_at_once", "target_row_limit"
PREFIX = "target."


def dataset_through_target(datasets: tuple[str, ...] | list[str], input_name: str = SYSTEM_INPUT,
                           label: str = "Target of the assessment (the system, or one of its components)"):
    """Class decorator for a plugin that analyses tables: before ``evaluate``, every row of each named
    dataset input is sent to the evaluation's target, through its endpoint (the row as the request's
    ``{{input}}``), and each scalar field of the answer is added to the row as ``target.<field>`` (a text
    answer as ``target.answer``; a refusal as ``target.refused`` and ``target.refusal_reason``; a failed
    call as ``target.error``). The plugin then reads the enriched table as if it had been uploaded so.

    Run settings, read from the plugin's config when its form has them: ``target_calls_at_once``
    (default 1) and ``target_row_limit`` (default 0: every row). The answers are saved with the run as
    ``target-answers-<dataset>.csv``. A run without a target that has an endpoint is refused before the
    plugin starts.
    """
    datasets = tuple(datasets)
    if not datasets:
        raise ValueError("datasets: name at least one dataset input")

    def decorator(cls):
        if target_access_of(cls) not in (None, "dataset_through_target"):
            raise ValueError(f"{cls.__name__}: declare one of @system_under_test, @assesses_inputs or"
                             " @dataset_through_target, not two")
        cls.target_access = "dataset_through_target"
        cls.dataset_through_target_inputs = datasets
        cls = evaluation_input(name=input_name, label=label, input_type=InputType.RESOURCE, required=False)(cls)
        original = cls.evaluate

        @functools.wraps(original)
        def evaluate(self, config_data: dict) -> Any:
            from aisc_plugin_interface.connections import EndpointClient
            if self.get_input_data(input_name) is None:
                raise ValueError("no target: this plugin sends its datasets through the evaluation's target,"
                                 " which needs an endpoint (Manage, Targets and endpoints)")
            client = EndpointClient.for_target(self, input_name)
            settings = config_data or {}
            at_once = max(1, int(settings.get(CALLS_AT_ONCE) or 1))
            limit = max(0, int(settings.get(ROW_LIMIT) or 0))
            for name in datasets:
                table = self.get_input_data(name)
                if table is None:
                    continue
                enriched = _through(client, table, at_once, limit)
                self.set_input_content(name, enriched)
                self.upload_artifact(f"target-answers-{name}.csv", enriched)
            return original(self, config_data)

        cls.evaluate = evaluate
        return cls

    return decorator


def _rows(table) -> list[dict]:
    if hasattr(table, "to_dict"):                              # a pandas DataFrame
        return table.to_dict(orient="records")
    if isinstance(table, list) and all(isinstance(r, dict) for r in table):
        return [dict(r) for r in table]
    raise ValueError("@dataset_through_target needs a table: a pandas DataFrame or a list of rows")


def _plain(value):
    """A JSON value for a cell: numpy numbers as Python numbers, NaN as null."""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and value != value:
        return None
    return value


def _answer_columns(answer) -> dict:
    out = {PREFIX + "refused": bool(answer.refused)}
    if answer.refused:
        out[PREFIX + "refusal_reason"] = answer.refusal_reason
    elif isinstance(answer.text, dict):
        out.update({PREFIX + k: v for k, v in answer.text.items() if isinstance(v, (str, int, float, bool)) or v is None})
    else:
        out[PREFIX + "answer"] = answer.text
    return out


def _through(client, table, at_once: int, limit: int) -> bytes:
    """The table with the target's answer to each row, as CSV."""
    import csv
    import io
    from concurrent.futures import ThreadPoolExecutor
    from aisc_plugin_interface.connections import EndpointError

    rows = _rows(table)
    if limit:
        rows = rows[:limit]

    def one(row):
        try:
            return _answer_columns(client.ask({k: _plain(v) for k, v in row.items()}))
        except EndpointError as exc:
            return {PREFIX + "refused": False, PREFIX + "error": str(exc)}

    with ThreadPoolExecutor(max_workers=at_once) as pool:
        answers = list(pool.map(one, rows))
    if rows and all(PREFIX + "error" in a for a in answers):
        raise RuntimeError(f"the target answered none of the {len(rows)} rows: {answers[0][PREFIX + 'error']}")
    merged = [{**{k: _plain(v) for k, v in row.items()}, **answer} for row, answer in zip(rows, answers)]
    columns = list(dict.fromkeys(k for r in merged for k in r))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    writer.writerows(merged)
    return buffer.getvalue().encode()

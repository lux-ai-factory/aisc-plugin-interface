"""A connection's request and its answer, as data only: the descriptor, the checks of history and params, the
request rendered from its template, the answer read from what the system returned, and the secrets scrubbed from
an error. Nothing here opens a connection (see transport)."""
from __future__ import annotations

import base64
import hashlib
import json
import re
import urllib.parse
import uuid
from dataclasses import dataclass, field
from typing import Any

from .errors import BadReference, BadRequest, EndpointBadResponse, EndpointError


REFERENCE_PREFIX = "connection:"
_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
_PLACEHOLDER = re.compile(r"\{\{\s*(input|history|secret|params\.[A-Za-z0-9_]+)\s*\}\}")
_PATH_PART = re.compile(r"([^.\[\]]+)|\[(\d+)\]")


def parse_reference(value: Any) -> tuple[str, str]:
    """(project pid, connection name) of a ``connection:<pid>/<name>`` value; BadReference otherwise."""
    if not isinstance(value, str) or not value.startswith(REFERENCE_PREFIX):
        raise BadReference(f"not a connection reference: {value!r}")
    pid, _, name = value[len(REFERENCE_PREFIX):].partition("/")
    if not pid or not _NAME.match(name or ""):
        raise BadReference(f"not a connection reference: {value!r}")
    return pid, name


def reference_of(plugin, input_name: str) -> tuple[str, str]:
    """The (pid, name) a plugin's resource input refers to."""
    payload = plugin.get_input_data(input_name)
    value = payload.get("value") if isinstance(payload, dict) else payload
    if value is None:
        raise BadReference(f"input {input_name!r} is not bound to a connection")
    return parse_reference(value)


@dataclass
class Descriptor:
    name: str
    label: str
    kind: str                                # "openai" | "rest" | "a2a" | "oip"
    base_url: str
    method: str = "POST"
    path: str = ""
    headers: dict = field(default_factory=dict)
    secret_header: str | None = None
    body_template: Any = None
    response_path: str | None = None
    refusal: dict | None = None
    model: str | None = None
    timeout_s: int = 60
    secret: str | None = field(default=None, repr=False)   # never in a log or a traceback
    updated_at: str | None = None
    protocol_version: str | None = None      # a2a: "1.0" (default) or "0.3"
    #: the assessment target this connection is the endpoint of (key, kind, label, ...)
    target: dict | None = None
    #: which of ``headers`` carry a secret value: scrubbed from error text like the key; not a stored
    #: setting, so not in FIELDS, public() or the fingerprint
    secret_header_names: list = field(default_factory=list, repr=False)

    FIELDS = ("name", "label", "kind", "base_url", "method", "path", "headers", "secret_header",
              "body_template", "response_path", "refusal", "model", "timeout_s", "secret", "updated_at",
              "protocol_version", "target")

    @classmethod
    def from_dict(cls, d: dict) -> "Descriptor":
        return cls(**{k: d[k] for k in cls.FIELDS if d.get(k) is not None})

    def public(self) -> dict:
        """Every setting except the key, as the platform stores it."""
        return {k: getattr(self, k) for k in self.FIELDS if k != "secret"}

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.public(), sort_keys=True).encode()).hexdigest()


@dataclass
class Answer:
    text: Any                                # the answer: text, or JSON for structured systems
    refused: bool = False
    refusal_reason: str | None = None
    status: int | None = None
    latency_ms: int | None = None
    #: an OpenAI-style agent's tool calls (``choices[0].message.tool_calls``), None when it called none
    tool_calls: list[dict] | None = None


#: what a history message may be: OpenAI's chat messages, an assistant's tool calls and a tool's result
HISTORY_ROLES = ("system", "user", "assistant", "tool")
_MESSAGE_FIELDS = ("role", "content", "name", "tool_calls", "tool_call_id", "refusal")
MAX_HISTORY_MESSAGES = 200
MAX_HISTORY_BYTES = 1_048_576
MAX_PARAMS_BYTES = 262_144


def _good_call(call: Any) -> bool:
    function = call.get("function") if isinstance(call, dict) else None
    return (isinstance(call.get("id") if isinstance(call, dict) else None, str) and call.get("type") == "function"
            and isinstance(function, dict) and isinstance(function.get("name"), str)
            and isinstance(function.get("arguments"), str))


def _check_message(i: int, m: Any) -> None:
    where = f"history[{i}]"
    if not isinstance(m, dict) or "role" not in m:
        raise BadRequest(f"{where}: a message is an object with a role")
    role = m["role"]
    if role not in HISTORY_ROLES:
        raise BadRequest(f"{where}: role {role!r} is not one of {', '.join(HISTORY_ROLES)}")
    for key in m:
        if key not in _MESSAGE_FIELDS or (key == "refusal" and role != "assistant"):
            raise BadRequest(f"{where}: {key!r} is not a message field ({', '.join(_MESSAGE_FIELDS)})")
    if "name" in m and not isinstance(m["name"], str):
        raise BadRequest(f"{where}: name is text")
    if role in ("system", "user") and not isinstance(m.get("content"), str):
        raise BadRequest(f"{where}: a {role} message needs text content")
    if "tool_calls" in m and role != "assistant":
        raise BadRequest(f"{where}: only an assistant message carries tool_calls")
    if "tool_call_id" in m and role != "tool":
        raise BadRequest(f"{where}: only a tool message carries a tool_call_id")
    if role == "assistant":
        calls = m.get("tool_calls")
        if calls is not None and not isinstance(calls, list):
            raise BadRequest(f"{where}: an assistant message needs text content or tool_calls")
        for j, call in enumerate(calls or []):
            if not _good_call(call):
                raise BadRequest(f"{where}: tool_calls[{j}] needs an id, type 'function' and a function with a name"
                                 " and arguments text")
        content = m.get("content")
        if not isinstance(content, str) and not (content is None and calls):
            raise BadRequest(f"{where}: an assistant message needs text content or tool_calls")
        if m.get("refusal") is not None and not isinstance(m["refusal"], str):
            raise BadRequest(f"{where}: refusal is text")
    if role == "tool" and not (isinstance(m.get("tool_call_id"), str) and m["tool_call_id"]
                               and isinstance(m.get("content"), str)):
        raise BadRequest(f"{where}: a tool message needs a tool_call_id and text content")


def check_history(history: list | None) -> list[dict]:
    """``history`` checked and copied: OpenAI chat messages (a system, user, assistant or tool message;
    an assistant's tool calls; a tool's result), at most MAX_HISTORY_MESSAGES and MAX_HISTORY_BYTES as
    JSON. The first fault raises BadRequest naming the message; None is []."""
    if history is None:
        return []
    if not isinstance(history, list):
        raise BadRequest("history: a list of messages")
    if len(history) > MAX_HISTORY_MESSAGES:
        raise BadRequest(f"history: {len(history)} messages, more than {MAX_HISTORY_MESSAGES}")
    try:
        size = len(json.dumps(history).encode())
    except (TypeError, ValueError):
        size = None
    if size is not None and size > MAX_HISTORY_BYTES:
        raise BadRequest(f"history: {size} bytes as JSON, more than {MAX_HISTORY_BYTES}")
    for i, message in enumerate(history):
        _check_message(i, message)
    if size is None:
        raise BadRequest("history: not JSON")
    return list(history)


def check_params(params: dict) -> dict:
    """The params of an ask, refused (BadRequest) when they are not JSON or are larger than
    MAX_PARAMS_BYTES as JSON."""
    try:
        size = len(json.dumps(params).encode())
    except (TypeError, ValueError) as exc:
        raise BadRequest(f"params: not JSON: {str(exc)[:100]}") from None
    if size > MAX_PARAMS_BYTES:
        raise BadRequest(f"params: {size} bytes as JSON, more than {MAX_PARAMS_BYTES}")
    return params


def _ends_with_tool_results(history: list) -> bool:
    return bool(history) and isinstance(history[-1], dict) and history[-1].get("role") == "tool"


def _check_input(d: "Descriptor", input: Any, history: list) -> None:
    """An OpenAI-style call without ``input`` continues a tool loop: the history must end with the tool
    results, and is sent as it is."""
    if d.kind == "openai" and input is None and not _ends_with_tool_results(history):
        raise BadRequest("input is required unless the history ends with tool results")


def _tool_calls_of(label: str, calls: Any) -> list[dict] | None:
    """An answer's tool calls: None when there are none, else a list of objects."""
    if calls is None or calls == []:
        return None
    if not isinstance(calls, list) or not all(isinstance(c, dict) for c in calls):
        raise EndpointBadResponse(f"{label}: the answer's tool_calls are not a list of calls")
    return calls


def extract(obj: Any, path: str) -> Any:
    """``a.b[0].c`` on nested dicts and lists; KeyError when a step is missing. ``$`` (or a path starting
    ``$.``) is the root, so ``$`` alone is the whole answer."""
    current = obj
    path = (path or "").strip()
    if path.startswith("$"):
        path = path[1:].lstrip(".")
    for key, index in _PATH_PART.findall(path):
        if key:
            if not isinstance(current, dict) or key not in current:
                raise KeyError(path)
            current = current[key]
        else:
            if not isinstance(current, list) or int(index) >= len(current):
                raise KeyError(path)
            current = current[int(index)]
    return current


def _render_value(value: Any, input: str, history: list, params: dict) -> Any:
    if isinstance(value, dict):
        return {k: _render_value(v, input, history, params) for k, v in value.items()}
    if isinstance(value, list):
        return [_render_value(v, input, history, params) for v in value]
    if not isinstance(value, str):
        return value

    def lookup(token: str) -> Any:
        if token == "input":
            return input
        if token == "history":
            return history
        if token.startswith("params."):
            key = token[len("params."):]
            if key not in params:
                raise EndpointError(f"the request template needs the parameter {key!r}")
            return params[key]
        raise EndpointError(f"placeholder {{{{{token}}}}} is not allowed in the body")

    whole = _PLACEHOLDER.fullmatch(value.strip())
    if whole:                                 # the whole string is one placeholder: keep its type
        return lookup(whole.group(1))
    return _PLACEHOLDER.sub(lambda m: str(lookup(m.group(1))), value)


def _a2a_legacy(d: Descriptor) -> bool:
    return (d.protocol_version or "1.0").startswith("0.")


def _a2a_part(value: Any, legacy: bool) -> dict:
    if isinstance(value, str):
        return {"kind": "text", "text": value} if legacy else {"text": value}
    return {"kind": "data", "data": value} if legacy else {"data": value}


def a2a_request(d: Descriptor, method: str, params: dict) -> dict:
    return {"jsonrpc": "2.0", "id": uuid.uuid4().hex, "method": method, "params": params}


def render_request(d: Descriptor, input: Any, history: list | None, params: dict) -> tuple[str, str, dict, bytes | None]:
    """(method, url, headers, body) for one call."""
    history = list(history or [])
    headers = {"Content-Type": "application/json", "Accept": "application/json", **(d.headers or {})}
    if d.kind in ("a2a", "oip") and d.secret and not d.secret_header:
        headers["Authorization"] = f"Bearer {d.secret}"
    if d.secret_header and d.secret:          # every kind: a2a and oip return before the end
        name, _, value = d.secret_header.partition(":")
        headers[name.strip()] = value.strip().replace("{{secret}}", d.secret)
    if d.kind == "a2a":
        if history:
            raise EndpointError(f"{d.label}: an A2A agent keeps its own context; a history cannot be sent to it")
        legacy = _a2a_legacy(d)
        url = d.base_url.rstrip("/") + ("/" + d.path.lstrip("/") if d.path else "")
        message = {"messageId": uuid.uuid4().hex, "role": "user" if legacy else "ROLE_USER",
                   "parts": [_a2a_part(input, legacy)]}
        if legacy:
            message["kind"] = "message"
        else:
            headers["A2A-Version"] = d.protocol_version or "1.0"
        body: Any = a2a_request(d, "message/send" if legacy else "SendMessage", {"message": message})
        return "POST", url, headers, json.dumps(body).encode()
    if d.kind == "oip":
        url = d.base_url.rstrip("/") + f"/v2/models/{urllib.parse.quote(d.model or d.name)}/infer"
        if d.body_template is not None:
            body = _render_value(d.body_template, input, history, params)
        else:
            body = {"inputs": [{"name": "input", "shape": [1], "datatype": "BYTES", "data": [input]}]}
        return "POST", url, headers, json.dumps(body).encode()
    if d.kind == "openai":
        _check_input(d, input, history)
        url = d.base_url.rstrip("/") + "/chat/completions"
        # no input continues a tool loop: the history, ending with the tool results, goes as it is
        messages = history if input is None else history + [{"role": "user", "content": input}]
        body: Any = {"model": d.model, "messages": messages, **params}
        method = "POST"
        if d.secret and not d.secret_header:
            headers["Authorization"] = f"Bearer {d.secret}"
    elif d.kind == "rest":
        url = d.base_url.rstrip("/") + ("/" + d.path.lstrip("/") if d.path else "")
        body = _render_value(d.body_template, input, history, params) if d.body_template is not None else None
        method = (d.method or "POST").upper()
        if method == "GET" and isinstance(body, dict):
            url += ("&" if "?" in url else "?") + urllib.parse.urlencode(body)
            body = None
    else:
        raise EndpointError(f"unknown connection kind {d.kind!r}")
    data = json.dumps(body).encode() if body is not None else None
    return method, url, headers, data


def _scrub(text: str, secret: str | None) -> str:
    return text.replace(secret, "***") if secret else text


def _secrets_of(d: Descriptor) -> list[str]:
    """The key and every secret header's value, each in the forms a system may echo it (the platform's
    secret_forms): as it is, base64 (standard and URL-safe), URL-encoded (with and without "/" kept, and with
    "+" for spaces) and JSON-escaped; longest first so a value inside another goes with it. Not taken out:
    hex, unpadded base64, double encodings, case changes and a secret split across calls."""
    names = {n.lower() for n in (getattr(d, "secret_header_names", None) or [])}
    values = [d.secret] + [str(v) for k, v in (d.headers or {}).items() if k.lower() in names]
    forms = set()
    for value in (v for v in values if v):
        raw = value.encode()
        forms.update((value, base64.b64encode(raw).decode(), base64.urlsafe_b64encode(raw).decode(),
                      urllib.parse.quote(value, safe=""), urllib.parse.quote(value),
                      urllib.parse.quote_plus(value), json.dumps(value)[1:-1]))
    return sorted(forms, key=len, reverse=True)


def _scrub_all(text: str, d: Descriptor) -> str:
    for value in _secrets_of(d):
        text = text.replace(value, "***")
    return text


def _refusal_of(d: Descriptor, status: int, payload: Any) -> str | None:
    """The refusal reason when (status, payload) is the connection's declared refusal, else None."""
    r = d.refusal or {}
    if status not in (r.get("status") or []):
        return None
    for path, expected in (r.get("match") or {}).items():
        try:
            if extract(payload, path) != expected:
                return None
        except KeyError:
            return None
    try:
        return str(extract(payload, r["path"])) if r.get("path") else ""
    except KeyError:
        return ""


_A2A_DONE = {"TASK_STATE_COMPLETED": "completed", "completed": "completed",
             "TASK_STATE_REJECTED": "rejected", "rejected": "rejected",
             "TASK_STATE_SUBMITTED": "working", "TASK_STATE_WORKING": "working", "submitted": "working", "working": "working"}


def _parts_value(d: Descriptor, parts: list | None) -> Any:
    """Text parts joined; else the first data part's value; else None. `parts` that is there but not a
    list is a bad response (an int used to raise TypeError and end the dataset run)."""
    if parts is None:
        return None
    if not isinstance(parts, list):
        raise EndpointBadResponse(f"{d.label}: the A2A parts are not a list")
    texts = [p["text"] for p in parts if isinstance(p, dict) and isinstance(p.get("text"), str)]
    if texts:
        return "".join(texts)
    for p in parts:
        if isinstance(p, dict) and "data" in p:
            return p["data"]
    return None


def _object(d: Descriptor, value: Any, what: str) -> dict:
    if not isinstance(value, dict):
        raise EndpointBadResponse(f"{d.label}: the A2A {what} is not an object")
    return value


def _a2a_outcome(d: Descriptor, result: Any) -> tuple[str, Any]:
    """("answer", value) | ("refused", reason) | ("working", task id), else raise."""
    if not isinstance(result, dict):
        raise EndpointBadResponse(f"{d.label}: the A2A result is not an object")
    if "message" in result and "task" not in result and result.get("kind") != "task":
        message = result["message"] if isinstance(result["message"], dict) else {}
        return "answer", _parts_value(d, message.get("parts"))
    if result.get("kind") == "message" or ("parts" in result and "status" not in result):
        return "answer", _parts_value(d, result.get("parts"))
    # the system under test is what is assessed: each level of its answer is checked before it is read
    task = _object(d, result.get("task", result), "task")
    status = _object(d, task.get("status") or {}, "task status")
    state = status.get("state")
    outcome = _A2A_DONE.get(state) if isinstance(state, str) else None
    status_text = _parts_value(d, _object(d, status.get("message") or {}, "status message").get("parts"))
    if outcome == "completed":
        artifacts = task.get("artifacts") or []
        if not isinstance(artifacts, list):
            raise EndpointBadResponse(f"{d.label}: the A2A task's artifacts are not a list")
        for artifact in artifacts:
            value = _parts_value(d, _object(d, artifact, "artifact").get("parts"))
            if value is not None:
                return "answer", value
        return "answer", status_text
    if outcome == "rejected":
        return "refused", status_text or ""
    if outcome == "working":
        return "working", task.get("id")
    raise EndpointBadResponse(f"{d.label}: the A2A task ended in {state}")


def _platform_answer(label: str, payload: Any) -> Answer:
    """The platform's 200 answer to an ask, as the Answer a direct call gives."""
    if not isinstance(payload, dict) or "output" not in payload:
        raise EndpointBadResponse(f"{label}: the platform's answer is not an answer")
    status, latency = payload.get("status"), payload.get("latency_ms")
    return Answer(text=payload["output"], refused=bool(payload.get("refused")),
                  refusal_reason=payload.get("refusal_reason"),
                  status=status if isinstance(status, int) and not isinstance(status, bool) else None,
                  latency_ms=latency if isinstance(latency, int) and not isinstance(latency, bool) else None,
                  tool_calls=_tool_calls_of(label, payload.get("tool_calls")))


def _openai_answer(d: Descriptor, payload: Any, status: int, latency: int) -> Answer:
    """An OpenAI chat completion: the message's text, or its tool calls (with no text, or null text)."""
    try:
        message = extract(payload, "choices[0].message")
    except KeyError:
        message = None
    calls = _tool_calls_of(d.label, message.get("tool_calls")) if isinstance(message, dict) else None
    if not isinstance(message, dict) or ("content" not in message and calls is None):
        raise EndpointBadResponse(f"{d.label}: no answer at 'choices[0].message.content' in the response")
    return Answer(text=message.get("content"), status=status, latency_ms=latency, tool_calls=calls)

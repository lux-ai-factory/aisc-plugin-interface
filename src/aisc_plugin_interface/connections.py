"""Calling a system under test that a project registered under Manage → Connections.

A connection lives on the platform (schema ``connection`` of the project's database). An evaluation
binds it as a ``resource`` input whose value is ``connection:<project pid>/<name>``. A plugin asks
``EndpointClient.for_input(self, "<input name>")`` for a client, then ``client.ask(prompt)``; the
client resolves the connection from the platform, renders the request, calls the system, reads the
answer and turns a declared refusal into an answer marked ``refused``. It records what it assessed
as the run artifact ``connection-<name>.json`` (never the key).

Outbound calls are refused to loopback, private, link-local, multicast, reserved and unspecified
addresses, to any hostname that resolves to one, to schemes other than http(s), and on redirects,
unless the host (or host:port) is listed in ``CONNECTIONS_ALLOWED_HOSTS``. This protects honest
plugins from a misconfigured connection; it is not a sandbox for plugin code.

Standard library only, so no plugin gains a dependency.
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import socket
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

REFERENCE_PREFIX = "connection:"
_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
_PLACEHOLDER = re.compile(r"\{\{\s*(input|history|secret|params\.[A-Za-z0-9_]+)\s*\}\}")
_PATH_PART = re.compile(r"([^.\[\]]+)|\[(\d+)\]")
RETRY_WAITS = (1, 3)                       # seconds before the 2nd and the 3rd try


class EndpointError(RuntimeError):
    """A connection could not be used. The message never carries the key."""


class BadReference(EndpointError):
    """The input is not a ``connection:<pid>/<name>`` reference."""


class BlockedAddress(EndpointError):
    """The URL points at an address the client refuses to call."""


class EndpointAuthError(EndpointError):
    """The system answered 401 or 403."""


class EndpointNotFound(EndpointError):
    """The system answered 404."""


class EndpointTimeout(EndpointError):
    """The system did not answer in time, after the retries."""


class EndpointBadResponse(EndpointError):
    """The system answered, but not with a usable answer."""


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
    secret: str | None = None
    updated_at: str | None = None
    protocol_version: str | None = None      # a2a: "1.0" (default) or "0.3"
    #: the assessment target this connection is the endpoint of (key, kind, label, ...)
    target: dict | None = None

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


def allowed_hosts_from_env() -> list[str]:
    return [h.strip().lower() for h in (os.environ.get("CONNECTIONS_ALLOWED_HOSTS") or "").split(",") if h.strip()]


def _refused_ip(ip: ipaddress._BaseAddress) -> bool:
    return (ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_multicast or ip.is_reserved
            or ip.is_unspecified)


def guard_url(url: str, allowed_hosts: list[str] | None = None, denied_addresses: list[str] | None = None) -> None:
    """Raise BlockedAddress unless ``url`` may be called. ``denied_addresses`` (the stack's own
    services and cloud metadata, from the platform) are refused on the resolved address, even for
    an allowed host, so re-pointing an allowed name does not reach them."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise BlockedAddress(f"scheme {parts.scheme or '(none)'} is not allowed")
    host = (parts.hostname or "").lower()
    if not host:
        raise BlockedAddress("the URL has no host")
    port = parts.port or (443 if parts.scheme == "https" else 80)
    allowed = [h.lower() for h in (allowed_hosts or [])]
    denied = {ipaddress.ip_address(a) for a in (denied_addresses or [])}
    is_allowed = host in allowed or f"{host}:{port}" in allowed
    if is_allowed and not denied:
        return
    addresses = _addresses(host, port)
    if any(ip in denied for ip in addresses):
        raise BlockedAddress(f"{host} is a service of this deployment or a metadata address, never allowed")
    if is_allowed:
        return
    if not addresses or any(_refused_ip(ip) for ip in addresses):
        raise BlockedAddress(f"{host} is an internal address; allow it under Manage, Connections, Allowed internal hosts")


def _addresses(host: str, port: int) -> list:
    try:
        return [ipaddress.ip_address(host)]
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise BlockedAddress(f"{host} does not resolve") from exc
    return [ipaddress.ip_address(info[4][0]) for info in infos]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise _Redirected(code)


class _Redirected(Exception):
    def __init__(self, code):
        self.code = code


_OPENER = urllib.request.build_opener(_NoRedirect)


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
        url = d.base_url.rstrip("/") + "/chat/completions"
        body: Any = {"model": d.model, "messages": history + [{"role": "user", "content": input}], **params}
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
    if d.secret_header and d.secret:
        name, _, value = d.secret_header.partition(":")
        headers[name.strip()] = value.strip().replace("{{secret}}", d.secret)
    data = json.dumps(body).encode() if body is not None else None
    return method, url, headers, data


def _scrub(text: str, secret: str | None) -> str:
    return text.replace(secret, "***") if secret else text


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


def _parts_value(parts: list | None) -> Any:
    """Text parts joined; else the first data part's value; else None."""
    parts = parts or []
    texts = [p["text"] for p in parts if isinstance(p, dict) and isinstance(p.get("text"), str)]
    if texts:
        return "".join(texts)
    for p in parts:
        if isinstance(p, dict) and "data" in p:
            return p["data"]
    return None


def _a2a_outcome(d: Descriptor, result: Any) -> tuple[str, Any]:
    """("answer", value) | ("refused", reason) | ("working", task id), else raise."""
    if not isinstance(result, dict):
        raise EndpointBadResponse(f"{d.label}: the A2A result is not an object")
    if "message" in result and "task" not in result and result.get("kind") != "task":
        message = result["message"] if isinstance(result["message"], dict) else {}
        return "answer", _parts_value(message.get("parts"))
    if result.get("kind") == "message" or ("parts" in result and "status" not in result):
        return "answer", _parts_value(result.get("parts"))
    task = result.get("task", result)
    status = task.get("status") or {}
    state = status.get("state")
    outcome = _A2A_DONE.get(state)
    status_text = _parts_value((status.get("message") or {}).get("parts"))
    if outcome == "completed":
        for artifact in task.get("artifacts") or []:
            value = _parts_value(artifact.get("parts"))
            if value is not None:
                return "answer", value
        return "answer", status_text
    if outcome == "rejected":
        return "refused", status_text or ""
    if outcome == "working":
        return "working", task.get("id")
    raise EndpointBadResponse(f"{d.label}: the A2A task ended in {state}")


def _post_json(d: Descriptor, url: str, headers: dict, body: dict, allowed_hosts, denied_addresses=None) -> Any:
    guard_url(url, allowed_hosts, denied_addresses)
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST", headers=headers)
    try:
        with _OPENER.open(req, timeout=d.timeout_s) as resp:
            return json.loads(resp.read() or b"null")
    except urllib.error.HTTPError as exc:
        raise EndpointBadResponse(f"{d.label}: the agent answered {exc.code} while polling") from None
    except (urllib.error.URLError, TimeoutError, socket.timeout, _Redirected):
        raise EndpointTimeout(f"{d.label}: the agent did not answer while polling") from None


def _a2a_answer(d: Descriptor, payload: Any, url: str, headers: dict, allowed_hosts, sleep, status: int,
                latency: int, denied_addresses=None) -> Answer:
    deadline = time.monotonic() + d.timeout_s
    while True:
        if not isinstance(payload, dict):
            raise EndpointBadResponse(f"{d.label}: not a JSON-RPC response")
        if payload.get("error"):
            err = payload["error"]
            raise EndpointBadResponse(f"{d.label}: the agent answered a JSON-RPC error: {err.get('message', err)}")
        kind, value = _a2a_outcome(d, payload.get("result"))
        if kind == "answer":
            return Answer(text=value, status=status, latency_ms=latency)
        if kind == "refused":
            return Answer(text=None, refused=True, refusal_reason=value, status=status, latency_ms=latency)
        if time.monotonic() >= deadline or not value:
            raise EndpointTimeout(f"{d.label}: the A2A task was still working after {d.timeout_s}s")
        sleep(1)
        legacy = _a2a_legacy(d)
        payload = _post_json(d, url, headers, a2a_request(d, "tasks/get" if legacy else "GetTask", {"id": value}),
                             allowed_hosts, denied_addresses)


def call(d: Descriptor, input: Any, history: list | None = None, params: dict | None = None,
         allowed_hosts: list[str] | None = None, sleep: Callable[[float], None] = time.sleep,
         waits: tuple = RETRY_WAITS, denied_addresses: list[str] | None = None) -> Answer:
    """One question to the system, with the retries (``waits``: seconds before each retry; ``()``
    for a single probe, as the platform's Test button does); see EndpointClient.ask."""
    method, url, headers, data = render_request(d, input, history, dict(params or {}))
    waits = list(waits)
    while True:
        guard_url(url, allowed_hosts if allowed_hosts is not None else allowed_hosts_from_env(), denied_addresses)
        started = time.monotonic()
        retryable: EndpointError
        try:
            req = urllib.request.Request(url, data=data, method=method, headers=headers)
            with _OPENER.open(req, timeout=d.timeout_s) as resp:
                status, raw = resp.status, resp.read()
        except _Redirected as exc:
            raise EndpointBadResponse(f"{d.label}: the system answered with a redirect ({exc.code}), which is not followed")
        except urllib.error.HTTPError as exc:
            status, raw = exc.code, exc.read()
        except (TimeoutError, socket.timeout):
            status, raw = None, b""
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                status, raw = None, b""
            else:
                raise EndpointError(_scrub(f"{d.label}: {exc.reason}", d.secret)) from None
        latency = int((time.monotonic() - started) * 1000)
        try:
            payload = json.loads(raw) if raw else None
        except ValueError:
            payload = raw.decode(errors="replace")
        if status is None:
            retryable = EndpointTimeout(f"{d.label}: no answer within {d.timeout_s}s")
        else:
            reason = _refusal_of(d, status, payload)
            if reason is not None:
                return Answer(text=None, refused=True, refusal_reason=reason, status=status, latency_ms=latency)
            if status in (401, 403):
                raise EndpointAuthError(f"{d.label}: the system refused the credentials ({status})")
            if status == 404:
                raise EndpointNotFound(f"{d.label}: nothing at {urllib.parse.urlsplit(url).path} (404)")
            if 400 <= status < 500:
                # the system's own reason, a short excerpt with the key taken out: without it a 422 is a guess
                said = _scrub((raw or b"").decode("utf-8", errors="replace")[:300], d.secret)
                raise EndpointBadResponse(f"{d.label}: the system answered {status}: {said}")
            if status < 300 and d.kind == "a2a":
                return _a2a_answer(d, payload, url, headers,
                                   allowed_hosts if allowed_hosts is not None else allowed_hosts_from_env(),
                                   sleep, status, latency, denied_addresses)
            if status < 300:
                path = {"openai": "choices[0].message.content", "oip": d.response_path or "outputs[0].data[0]"}.get(
                    d.kind, d.response_path)
                try:
                    return Answer(text=extract(payload, path), status=status, latency_ms=latency)
                except KeyError:
                    raise EndpointBadResponse(f"{d.label}: no answer at {path!r} in the response") from None
            retryable = EndpointBadResponse(f"{d.label}: the system answered {status}")
        if not waits:
            raise retryable
        sleep(waits.pop(0))


class EndpointClient:
    """One system under test, for one plugin run."""

    def __init__(self, descriptor: Descriptor, plugin=None, allowed_hosts: list[str] | None = None,
                 sleep: Callable[[float], None] = time.sleep, denied_addresses: list[str] | None = None):
        self.descriptor = descriptor
        self._plugin = plugin
        self._allowed = allowed_hosts
        self._denied = denied_addresses
        self._sleep = sleep
        self._recorded = False

    @classmethod
    def for_target(cls, plugin, input_name: str = "target", **kwargs) -> "EndpointClient":
        """Resolve the endpoint of the target this run assesses, through the platform."""
        from aisc_plugin_interface.targets import target_of

        found = target_of(plugin, input_name)
        if found is None:
            raise BadReference(f"input {input_name!r} is not bound to a target")
        return cls._resolved(plugin, found["pid"], f"targets/{urllib.parse.quote(found['key'], safe=':')}/connection",
                             found["key"], **kwargs)

    @classmethod
    def for_input(cls, plugin, input_name: str, **kwargs) -> "EndpointClient":
        """Resolve the connection a plugin's resource input refers to, through the platform."""
        pid, name = reference_of(plugin, input_name)
        return cls._resolved(plugin, pid, f"connections/{urllib.parse.quote(name)}", name, **kwargs)

    @classmethod
    def _resolved(cls, plugin, pid: str, path: str, name: str, **kwargs) -> "EndpointClient":
        base = (os.environ.get("PLATFORM_URL") or "").rstrip("/")
        token = os.environ.get("PLATFORM_CONNECTIONS_TOKEN") or ""
        if not base or not token:
            raise EndpointError("PLATFORM_URL and PLATFORM_CONNECTIONS_TOKEN must be set to resolve a connection")
        req = urllib.request.Request(f"{base}/internal/projects/{urllib.parse.quote(pid)}/{path}",
                                     headers={"X-AISC-Service-Token": token, "Accept": "application/json"})
        try:
            with _OPENER.open(req, timeout=30) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            raise EndpointError(f"the platform refused to resolve connection {name!r} ({exc.code})") from None
        except (urllib.error.URLError, TimeoutError, _Redirected) as exc:
            raise EndpointError(f"the platform could not be reached to resolve connection {name!r}") from None
        # The network rule is the platform's, per project: the internal hosts this connection may
        # reach and the addresses it never may. The run's own environment opens nothing.
        return cls(Descriptor.from_dict(data), plugin=plugin, allowed_hosts=list(data.get("allowed_hosts") or []),
                   denied_addresses=list(data.get("denied_addresses") or []), **kwargs)

    def ask(self, input: Any, history: list | None = None, **params) -> Answer:
        """Send ``input`` (text, or a JSON object/list for a structured system) after ``history`` (a list of
        {"role", "content"} turns) and return the answer."""
        answer = call(self.descriptor, input, history, params, allowed_hosts=self._allowed, sleep=self._sleep,
                      denied_addresses=self._denied)
        self._record()
        return answer

    def _record(self) -> None:
        if self._recorded or self._plugin is None:
            return
        doc = {"connection": self.descriptor.public(), "sha256": self.descriptor.fingerprint(),
               "first_call_at": datetime.now(timezone.utc).isoformat()}
        self._plugin.upload_artifact(f"connection-{self.descriptor.name}.json",
                                     json.dumps(doc, indent=2).encode())
        self._recorded = True

"""Calling a system under test that a project registered under Manage → Connections.

A connection lives on the platform (schema ``connection`` of the project's database). An evaluation
binds it as a ``resource`` input whose value is ``connection:<project pid>/<name>``. A plugin asks
``EndpointClient.for_input(self, "<input name>")`` (or ``for_target``) for a client, then
``client.ask(prompt)``. It records what it assessed as the run artifact ``connection-<name>.json``
(never the key).

Two ways the client reaches the system:
- **Through the platform**, when the run has a pass (``PLATFORM_RUN_PASS``): the client resolves the
  connection without its key (a key the platform still returns is dropped), takes a run key for it and
  asks the platform's ``aisc/ask``, which holds the key, renders the request, calls the system with its
  retries and reads the answer. The client makes one attempt; the platform's error codes are raised as
  the exception classes a direct call raises (``FACADE_ERRORS``). The ask URL must be on
  ``PLATFORM_URL``'s origin, or the run key is not sent.
- **Directly**, for a client made by hand from a ``Descriptor`` and one resolved with the service token
  of older platforms and setups (``PLATFORM_CONNECTIONS_TOKEN``, no pass): the client renders the
  request, calls the system, reads the answer and turns a declared refusal into an answer marked
  ``refused``, with the rules below.

Outbound calls are refused to loopback, private, link-local, multicast, reserved and unspecified
addresses, to any hostname that resolves to one, to schemes other than http(s), and on redirects.
For a client resolved through the platform (``for_input``, ``for_target``) the internal hosts that
are allowed come from the platform with the connection: the platform's own ``CONNECTIONS_ALLOWED_HOSTS``
plus the project's Manage, Connections, Allowed internal hosts, minus the addresses it never allows
(the stack's services, cloud metadata). ``CONNECTIONS_ALLOWED_HOSTS`` in the run's own environment
opens nothing for such a client; it is read only by a direct ``call()`` given no ``allowed_hosts``.
This protects honest plugins from a misconfigured connection; it is not a sandbox for plugin code.

Standard library only, so no plugin gains a dependency.
"""
from __future__ import annotations

import base64
import hashlib
import http.client
import ipaddress
import json
import os
import re
import socket
import threading
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
    """A connection could not be used. The message never carries the key; ``status`` is the HTTP status the
    system answered with, or None when it did not answer (unreachable, timed out, refused by the rules)."""

    def __init__(self, message: str = "", status: int | None = None):
        super().__init__(message)
        self.status = status


class BadReference(EndpointError):
    """The input is not a ``connection:<pid>/<name>`` reference."""


class BlockedAddress(EndpointError):
    """The URL points at an address the client refuses to call."""


class EndpointAuthError(EndpointError):
    """The system answered 401 or 403."""


class EndpointNotFound(EndpointError):
    """The system answered 404."""


class TargetHasNoEndpoint(EndpointNotFound):
    """The platform says this target has no endpoint (its 404 with reason no_endpoint): the one 404 that
    lets an optional target be skipped. Any other 404 (no project, no target, no route) is not that."""


def is_no_endpoint(status: int, body: Any) -> bool:
    """Whether a platform answer is its 'this target has no endpoint' 404. Older platforms say so only
    in the detail's words."""
    if status != 404 or not isinstance(body, dict):
        return False
    detail = body.get("detail")
    return body.get("reason") == "no_endpoint" or (isinstance(detail, str) and " has no endpoint" in detail)


class EndpointTimeout(EndpointError):
    """The system did not answer in time, after the retries."""


class EndpointBadResponse(EndpointError):
    """The system answered, but not with a usable answer."""


class BadRequest(EndpointError):
    """The request the plugin built cannot be sent: its history or params."""


#: The platform's ``aisc/ask`` error codes (its 502 answer's ``error``), as the exceptions a direct call
#: raises; ``error`` and any other code are a plain EndpointError.
FACADE_ERRORS: dict[str, type[EndpointError]] = {
    "auth": EndpointAuthError, "not_found": EndpointNotFound, "timeout": EndpointTimeout,
    "bad_response": EndpointBadResponse, "blocked": BlockedAddress,
}


class NoEndpoint(RuntimeError):
    """The target has no endpoint (or is not a target of the project)."""


PLATFORM_AUTH_NEEDED = "PLATFORM_URL and PLATFORM_RUN_PASS (or PLATFORM_CONNECTIONS_TOKEN) must be set"


def platform_auth() -> dict | None:
    """The header that opens the platform's internal routes: the run's pass, which opens only what the
    run names; else the service token of older platforms and setups; None when there is neither."""
    run_pass = os.environ.get("PLATFORM_RUN_PASS") or ""
    if run_pass:
        return {"X-AISC-Run-Pass": run_pass}
    token = os.environ.get("PLATFORM_CONNECTIONS_TOKEN") or ""
    return {"X-AISC-Service-Token": token} if token else None


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


def allowed_hosts_from_env() -> list[str]:
    return [h.strip().lower() for h in (os.environ.get("CONNECTIONS_ALLOWED_HOSTS") or "").split(",") if h.strip()]


def _as_ipv4(ip):
    """The IPv4 address an IPv6 one carries (IPv4-mapped ::ffff:a.b.c.d, 6to4 2002::/16, NAT64
    64:ff9b::/96), else the address itself: a mapped literal reaches the IPv4 address on Linux, so it
    is checked as that address."""
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            return ip.ipv4_mapped
        if ip.sixtofour is not None:
            return ip.sixtofour
        if ip in ipaddress.IPv6Network("64:ff9b::/96"):
            return ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
    return ip


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
    denied = {_as_ipv4(ipaddress.ip_address(a)) for a in (denied_addresses or [])}
    is_allowed = host in allowed or f"{host}:{port}" in allowed
    if is_allowed and not denied:
        return
    addresses = [_as_ipv4(ip) for ip in _addresses(host, port)]
    if not addresses and not is_allowed:
        raise BlockedAddress(f"{host} is an internal address; allow it under Manage, Connections, Allowed internal hosts")
    for ip in addresses:
        _check_address(host, ip, is_allowed, denied)


def _check_address(host: str, ip, is_allowed: bool, denied: set) -> None:
    """One address of `host`, by the rules: a denied one never, else an allowed host, else no internal one."""
    if ip in denied:
        raise BlockedAddress(f"{host} is a service of this deployment or a metadata address, never allowed")
    if not is_allowed and _refused_ip(ip):
        raise BlockedAddress(f"{host} is an internal address; allow it under Manage, Connections, Allowed internal hosts")


#: The largest answer a system's call reads: past it the call is refused, and no more is read into
#: memory (the worker and the platform's Test button share their process with other work).
MAX_ANSWER_BYTES = 10 * 1024 * 1024


def _read_capped(stream, label: str) -> bytes:
    raw = stream.read(MAX_ANSWER_BYTES + 1)
    if len(raw) > MAX_ANSWER_BYTES:
        raise EndpointBadResponse(f"{label}: the answer is larger than {MAX_ANSWER_BYTES} bytes")
    return raw


#: The rules of the call in flight on this thread, (allowed hosts, denied addresses), for the peer check.
_POLICY = threading.local()


class _Policy:
    """While a call to a system runs, the address its socket reaches is checked again by the same rules:
    urllib resolves the name a second time to connect, and a name can answer differently (DNS
    rebinding). The platform's own calls run without one."""

    def __init__(self, allowed_hosts, denied_addresses):
        self.allowed = [h.lower() for h in (allowed_hosts or [])]
        self.denied = {_as_ipv4(ipaddress.ip_address(a)) for a in (denied_addresses or [])}

    def __enter__(self):
        _POLICY.current = self
        return self

    def __exit__(self, *exc):
        _POLICY.current = None
        return False


def _to_proxy(conn) -> bool:
    """Whether the connection goes to one of the configured proxies (not to the system itself). Any
    proxy variable being set is not enough: NO_PROXY alone, or a host it lists, connects directly."""
    host = (conn.host or "").lower()
    for scheme, url in urllib.request.getproxies().items():
        if scheme == "no":
            continue
        if (urllib.parse.urlsplit(url if "://" in url else "//" + url).hostname or "").lower() == host:
            return True
    return False


def _check_peer(conn) -> None:
    """The peer of a connection just made, checked by the policy of the call in flight, before anything
    is sent. Through a proxy the proxy resolves the name, so there is no peer of the system to check."""
    policy = getattr(_POLICY, "current", None)
    if policy is None or conn._tunnel_host or _to_proxy(conn):
        return
    host = (conn.host or "").lower()
    is_allowed = host in policy.allowed or f"{host}:{conn.port}" in policy.allowed
    try:
        _check_address(host, _as_ipv4(ipaddress.ip_address(conn.sock.getpeername()[0])), is_allowed, policy.denied)
    except BlockedAddress:
        conn.sock.close()
        raise


class _GuardedHTTPConnection(http.client.HTTPConnection):
    def connect(self):
        super().connect()
        _check_peer(self)


class _GuardedHTTPSConnection(http.client.HTTPSConnection):
    def connect(self):
        # the TCP connection, the peer checked, then TLS with the name asked for
        http.client.HTTPConnection.connect(self)
        _check_peer(self)
        server_hostname = self._tunnel_host or self.host
        self.sock = self._context.wrap_socket(self.sock, server_hostname=server_hostname)


class _GuardedHTTPHandler(urllib.request.HTTPHandler):
    def http_open(self, req):
        return self.do_open(_GuardedHTTPConnection, req)


class _GuardedHTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req):
        return self.do_open(_GuardedHTTPSConnection, req, context=self._context)


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


_OPENER = urllib.request.build_opener(_NoRedirect, _GuardedHTTPHandler, _GuardedHTTPSHandler)


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


def _post_json(d: Descriptor, url: str, headers: dict, body: dict, allowed_hosts, denied_addresses=None) -> Any:
    guard_url(url, allowed_hosts, denied_addresses)
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST", headers=headers)
    try:
        with _Policy(allowed_hosts, denied_addresses), _OPENER.open(req, timeout=d.timeout_s) as resp:
            return json.loads(_read_capped(resp, d.label) or b"null")
    except urllib.error.HTTPError as exc:
        raise EndpointBadResponse(f"{d.label}: the agent answered {exc.code} while polling") from None
    except (urllib.error.URLError, TimeoutError, socket.timeout, _Redirected):
        raise EndpointTimeout(f"{d.label}: the agent did not answer while polling") from None
    except ValueError:
        raise EndpointBadResponse(f"{d.label}: the agent's answer while polling is not JSON") from None
    except (http.client.HTTPException, ConnectionError) as exc:
        raise EndpointError(f"{d.label}: the connection was dropped while polling ({type(exc).__name__})") from None


def _a2a_answer(d: Descriptor, payload: Any, url: str, headers: dict, allowed_hosts, sleep, status: int,
                latency: int, denied_addresses=None) -> Answer:
    deadline = time.monotonic() + d.timeout_s
    while True:
        if not isinstance(payload, dict):
            raise EndpointBadResponse(f"{d.label}: not a JSON-RPC response")
        if payload.get("error"):
            err = payload["error"]
            said = err.get("message", err) if isinstance(err, dict) else err
            raise EndpointBadResponse(f"{d.label}: the agent answered a JSON-RPC error: {said}")
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
    history = check_history(history)
    method, url, headers, data = render_request(d, input, history, dict(params or {}))
    waits = list(waits)
    while True:
        guard_url(url, allowed_hosts if allowed_hosts is not None else allowed_hosts_from_env(), denied_addresses)
        started = time.monotonic()
        retryable: EndpointError
        try:
            req = urllib.request.Request(url, data=data, method=method, headers=headers)
            with _Policy(allowed_hosts if allowed_hosts is not None else allowed_hosts_from_env(), denied_addresses), \
                    _OPENER.open(req, timeout=d.timeout_s) as resp:
                status, raw = resp.status, _read_capped(resp, d.label)
        except _Redirected as exc:
            raise EndpointBadResponse(f"{d.label}: the system answered with a redirect ({exc.code}), which is not followed")
        except urllib.error.HTTPError as exc:
            status, raw = exc.code, _read_capped(exc, d.label)
        except (TimeoutError, socket.timeout):
            status, raw = None, b""
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                status, raw = None, b""
            else:
                raise EndpointError(_scrub_all(f"{d.label}: {exc.reason}", d)) from None
        except (http.client.HTTPException, ConnectionError) as exc:
            # urllib leaves getresponse() and read() errors unwrapped: a connection the system dropped
            # is retried like a timeout, then an EndpointError
            if not waits:
                raise EndpointError(_scrub_all(f"{d.label}: the connection was dropped ({type(exc).__name__})",
                                               d)) from None
            sleep(waits.pop(0))
            continue
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
                raise EndpointAuthError(f"{d.label}: the system refused the credentials ({status})", status=status)
            if status == 404:
                raise EndpointNotFound(f"{d.label}: nothing at {urllib.parse.urlsplit(url).path} (404)", status=404)
            if 400 <= status < 500:
                # the system's own reason, a short excerpt with the key and every secret header value taken
                # out (before the cut, so a secret across it goes too): without it a 422 is a guess
                said = _scrub_all((raw or b"").decode("utf-8", errors="replace"), d)[:300]
                raise EndpointBadResponse(f"{d.label}: the system answered {status}: {said}", status=status)
            if status < 300 and d.kind == "a2a":
                return _a2a_answer(d, payload, url, headers,
                                   allowed_hosts if allowed_hosts is not None else allowed_hosts_from_env(),
                                   sleep, status, latency, denied_addresses)
            if status < 300 and d.kind == "openai":
                return _openai_answer(d, payload, status, latency)
            if status < 300:
                path = {"oip": d.response_path or "outputs[0].data[0]"}.get(d.kind, d.response_path)
                try:
                    return Answer(text=extract(payload, path), status=status, latency_ms=latency)
                except KeyError:
                    raise EndpointBadResponse(f"{d.label}: no answer at {path!r} in the response") from None
            retryable = EndpointBadResponse(f"{d.label}: the system answered {status}", status=status)
        if not waits:
            raise retryable
        sleep(waits.pop(0))


def _issue_run_key(pid: str, path: str, name: str) -> dict:
    """Ask the platform for a run key; ``path`` is ``targets/<key>``, or ``connections/<name>`` for a
    connection bound directly. When the platform refuses, its reason is passed on."""
    base = (os.environ.get("PLATFORM_URL") or "").rstrip("/")
    auth = platform_auth()
    if not base or auth is None:
        raise RuntimeError(f"{PLATFORM_AUTH_NEEDED} to reach a system under test")
    req = urllib.request.Request(
        f"{base}/internal/projects/{urllib.parse.quote(pid)}/{path}/run-keys",
        data=b"", method="POST", headers={**auth, "Accept": "application/json"})
    try:
        # the opener that follows no redirect: the pass or token goes to the platform and nowhere else
        with _OPENER.open(req, timeout=30) as resp:
            raw = resp.read()
        try:
            return json.loads(raw)
        except ValueError:
            raise RuntimeError(f"the platform's run-key answer was not JSON (connection {name!r})") from None
    except _Redirected as exc:
        raise RuntimeError(f"the platform answered a run key request with a redirect ({exc.code})") from None
    except urllib.error.HTTPError as exc:
        try:
            body = json.loads(exc.read() or b"{}")
        except ValueError:
            body = {}
        detail = body.get("detail") if isinstance(body, dict) else None
        reason = f": {detail}" if isinstance(detail, str) and detail else f" ({exc.code})"
        # only the platform's no-endpoint answer lets an optional target be skipped
        error = NoEndpoint if is_no_endpoint(exc.code, body) else RuntimeError
        raise error(f"the platform refused a run key for {name!r}{reason}") from None
    except (urllib.error.URLError, TimeoutError):
        raise RuntimeError(f"the platform could not be reached for connection {name!r}") from None


@dataclass(frozen=True)
class RunKey:
    """The run key the platform issued for one connection, and where to ask with it."""
    ask_url: str
    key: str = field(repr=False)                 # never in a log or a traceback
    capabilities: tuple[str, ...] = ("text", "history")
    expires_at: str | None = None


def facade_timeout_s(descriptor: Descriptor) -> int:
    """How long an ask through the platform waits: the platform's own call tries three times with up to
    4 s between, and an A2A task may poll for one more timeout."""
    return 4 * descriptor.timeout_s + 15


def _origin(url: str) -> tuple[str, str, int]:
    parts = urllib.parse.urlsplit(url)
    scheme = (parts.scheme or "").lower()
    try:
        port = parts.port
    except ValueError:
        port = -1
    return scheme, (parts.hostname or "").lower(), port or (443 if scheme == "https" else 80)


def _run_key_from(answer: Any, name: str) -> RunKey:
    """The run key of the platform's run-keys answer, checked before anything is sent with it."""
    answer = answer if isinstance(answer, dict) else {}
    key = answer.get("key")
    if not isinstance(key, str) or not key:
        raise EndpointError(f"the platform's run key answer for {name!r} has no key")
    endpoints = answer.get("endpoints") if isinstance(answer.get("endpoints"), dict) else {}
    aisc = endpoints.get("aisc") if isinstance(endpoints.get("aisc"), dict) else {}
    ask_url = aisc.get("ask_url")
    if not isinstance(ask_url, str) or not ask_url:
        raise EndpointError(f"the platform's run key answer for {name!r} has no aisc ask_url")
    # only plain printable ASCII: no lookalike, whitespace or control character the check and the request
    # could read differently
    plain = all("!" <= c <= "~" for c in ask_url)
    try:
        netloc = urllib.parse.urlsplit(ask_url).netloc
        # a userinfo part (or a backslash some parsers read as a path) can make another host the real one
        on_platform = (plain and "@" not in netloc and "\\" not in netloc
                       and _origin(ask_url) == _origin(os.environ.get("PLATFORM_URL") or ""))
    except ValueError:  # a URL the parser cannot read (a lookalike at sign, an unclosed bracket) is not trusted
        on_platform = False
    if not on_platform:
        raise EndpointError(f"the platform's ask URL for {name!r} is not on PLATFORM_URL; the run key is not sent"
                            " there")
    return RunKey(ask_url=ask_url, key=key, capabilities=tuple(answer.get("capabilities") or ("text", "history")),
                  expires_at=answer.get("expires_at"))


class EndpointClient:
    """One system under test, for one plugin run. ``run_key`` set: it asks through the platform (see the
    module docstring); None: it calls the system directly."""

    def __init__(self, descriptor: Descriptor, plugin=None, allowed_hosts: list[str] | None = None,
                 sleep: Callable[[float], None] = time.sleep, denied_addresses: list[str] | None = None,
                 run_key: RunKey | None = None):
        self.descriptor = descriptor
        self._plugin = plugin
        self._allowed = allowed_hosts
        self._denied = denied_addresses
        self._sleep = sleep
        self.run_key = run_key
        self._recorded = False

    @classmethod
    def for_target(cls, plugin, input_name: str = "target", **kwargs) -> "EndpointClient":
        """Resolve the endpoint of the target this run assesses, through the platform."""
        from aisc_plugin_interface.targets import target_of

        found = target_of(plugin, input_name)
        if found is None:
            raise BadReference(f"input {input_name!r} is not bound to a target")
        target = f"targets/{urllib.parse.quote(found['key'], safe=':')}"
        return cls._resolved(plugin, found["pid"], f"{target}/connection", found["key"], target, **kwargs)

    @classmethod
    def for_input(cls, plugin, input_name: str, **kwargs) -> "EndpointClient":
        """Resolve the connection a plugin's resource input refers to, through the platform."""
        pid, name = reference_of(plugin, input_name)
        path = f"connections/{urllib.parse.quote(name)}"
        return cls._resolved(plugin, pid, path, name, path, **kwargs)

    @classmethod
    def _resolved(cls, plugin, pid: str, path: str, name: str, run_key_path: str, **kwargs) -> "EndpointClient":
        base = (os.environ.get("PLATFORM_URL") or "").rstrip("/")
        auth = platform_auth()
        if not base or auth is None:
            raise EndpointError(f"{PLATFORM_AUTH_NEEDED} to resolve a connection")
        req = urllib.request.Request(f"{base}/internal/projects/{urllib.parse.quote(pid)}/{path}",
                                     headers={**auth, "Accept": "application/json"})
        try:
            with _OPENER.open(req, timeout=30) as resp:
                data = json.loads(resp.read())
            if os.environ.get("PLATFORM_RUN_PASS") and isinstance(data, dict):
                # under a run pass the key stays on the platform: a secret it still returns is dropped at
                # once, so no local, exception or traceback frame holds it afterwards
                data.pop("secret", None)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:                    # no such connection, or a target without an endpoint
                try:
                    body = json.loads(exc.read() or b"{}")
                except ValueError:
                    body = {}
                detail = body.get("detail") if isinstance(body, dict) else None
                error = TargetHasNoEndpoint if is_no_endpoint(404, body) else EndpointNotFound
                raise error(detail if isinstance(detail, str) and detail
                            else f"the platform has no connection {name!r}") from None
            raise EndpointError(f"the platform refused to resolve connection {name!r} ({exc.code})") from None
        except (urllib.error.URLError, TimeoutError, _Redirected):
            raise EndpointError(f"the platform could not be reached to resolve connection {name!r}") from None
        # The network rule is the platform's, per project: the internal hosts this connection may
        # reach and the addresses it never may. The run's own environment opens nothing.
        allowed, denied = list(data.get("allowed_hosts") or []), list(data.get("denied_addresses") or [])
        if not os.environ.get("PLATFORM_RUN_PASS"):
            return cls(Descriptor.from_dict(data), plugin=plugin, allowed_hosts=allowed, denied_addresses=denied,
                       **kwargs)
        # under a run pass the key stays on the platform: the client asks it with a run key
        descriptor = Descriptor.from_dict(data)
        try:
            issued = _issue_run_key(pid, run_key_path, descriptor.name)
        except NoEndpoint as exc:
            raise TargetHasNoEndpoint(str(exc)) from None
        except RuntimeError as exc:
            raise EndpointError(str(exc)) from None
        return cls(descriptor, plugin=plugin, allowed_hosts=allowed, denied_addresses=denied,
                   run_key=_run_key_from(issued, descriptor.name), **kwargs)

    def ask(self, input: Any, history: list | None = None, **params) -> Answer:
        """Send ``input`` (text, or a JSON object/list for a structured system) after ``history`` (OpenAI
        chat messages: {"role", "content"} turns, an assistant's ``tool_calls``, a tool's result; see
        ``check_history``) and return the answer. ``params`` fill the template's ``{{params.x}}`` and, for an
        OpenAI-style endpoint, go into the body (``tools``, ``tool_choice``, ...). ``input`` None, after
        a history that ends with tool results, continues the tool loop. A bad history or params raise
        BadRequest before anything is sent."""
        check_params(params)
        if self.run_key is not None:
            answer = self._ask_platform(input, history, params)
        else:
            answer = call(self.descriptor, input, history, params, allowed_hosts=self._allowed, sleep=self._sleep,
                          denied_addresses=self._denied)
        self._record()
        return answer

    def _ask_platform(self, input: Any, history: list | None, params: dict) -> Answer:
        """One POST to the platform's aisc/ask with the run key; the platform calls the system and retries."""
        d, run_key = self.descriptor, self.run_key
        label = d.label
        history = check_history(history)
        _check_input(d, input, history)
        if "tools" in params and "tools" not in run_key.capabilities:
            raise EndpointError(f"{label}: this endpoint does not carry tools (it carries"
                                f" {', '.join(run_key.capabilities)})")
        body = {"input": input, "history": history, "params": dict(params)}
        req = urllib.request.Request(run_key.ask_url, data=json.dumps(body).encode(), method="POST", headers={
            "Authorization": f"Bearer {run_key.key}", "Content-Type": "application/json",
            "Accept": "application/json"})
        timeout = facade_timeout_s(d)
        unreachable = f"{label}: the platform could not be reached to ask {d.name!r}"
        try:
            with _OPENER.open(req, timeout=timeout) as resp:
                status, raw = resp.status, _read_capped(resp, label)
        except _Redirected as exc:
            raise EndpointError(f"{label}: the platform answered with a redirect ({exc.code}), which is not"
                                " followed") from None
        except urllib.error.HTTPError as exc:
            status, raw = exc.code, _read_capped(exc, label)
        except (TimeoutError, socket.timeout):
            raise EndpointTimeout(f"{label}: the platform did not answer within {timeout}s") from None
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise EndpointTimeout(f"{label}: the platform did not answer within {timeout}s") from None
            raise EndpointError(unreachable) from None
        except (http.client.HTTPException, ConnectionError):
            raise EndpointError(unreachable) from None
        try:
            payload = json.loads(raw) if raw else None
        except ValueError:
            payload = None
        if 200 <= status < 300:
            return _platform_answer(label, payload)
        raise _platform_error(label, status, payload, run_key.key)

    def _record(self) -> None:
        """``connection-<name>.json`` after the first answered ask: the settings without the key, and their
        fingerprint, the same whichever way the client asks."""
        if self._recorded or self._plugin is None:
            return
        doc = {"connection": self.descriptor.public(), "sha256": self.descriptor.fingerprint(),
               "first_call_at": datetime.now(timezone.utc).isoformat()}
        self._plugin.upload_artifact(f"connection-{self.descriptor.name}.json",
                                     json.dumps(doc, indent=2).encode())
        self._recorded = True


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


def _platform_error(label: str, status: int, payload: Any, run_key: str) -> EndpointError:
    """The exception for the platform's answer ``status`` to an ask (see FACADE_ERRORS)."""
    body = payload if isinstance(payload, dict) else {}
    detail = body.get("detail")
    detail = _scrub(detail, run_key) if isinstance(detail, str) and detail else None
    if status == 502 and isinstance(body.get("error"), str):
        code = body["error"]
        said = body.get("status")
        return FACADE_ERRORS.get(code, EndpointError)(
            detail[:500] if detail else f"{label}: the system under test failed behind the platform ({code})",
            status=said if isinstance(said, int) and not isinstance(said, bool) else None)
    if status == 401:
        return EndpointAuthError(f"{label}: the platform refused the run key (401); the run's pass may have ended")
    if status == 404:
        return EndpointNotFound(f"{label}: the platform has no endpoint for this connection (404)")
    if status == 410:
        return EndpointNotFound(f"{label}: the connection was deleted (410)")
    if status == 400:
        return BadRequest(f"{label}: the platform refused the request (400){f': {detail}' if detail else ''}")
    return EndpointError(f"{label}: the platform answered {status}")

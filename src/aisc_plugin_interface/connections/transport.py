"""The call itself: every outbound address checked (the guard, on each connection's peer too), redirects refused,
the answer read up to its cap, and the direct call with its retries."""
from __future__ import annotations

import http.client
import ipaddress
import json
import os
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable

from .errors import (BlockedAddress, EndpointAuthError, EndpointBadResponse, EndpointError, EndpointNotFound,
                     EndpointTimeout)
from .render import (Answer, Descriptor, _a2a_legacy, _a2a_outcome, _openai_answer, _refusal_of, _scrub_all,
                     a2a_request, check_history, extract, render_request)


RETRY_WAITS = (1, 3)                       # seconds before the 2nd and the 3rd try


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
    """Whether an address is internal: anything not globally routable (private, loopback, link-local,
    shared CGNAT space, benchmarking, protocol assignments, unspecified, reserved), and multicast."""
    return not ip.is_global or ip.is_multicast


def _is_loopback(ip: ipaddress._BaseAddress) -> bool:
    """Loopback or unspecified (0.0.0.0 and :: reach the machine itself), after mapping to IPv4."""
    return ip.is_loopback or ip.is_unspecified


def _listed(host: str, port: int, hosts: list[str]) -> bool:
    return host in hosts or f"{host}:{port}" in hosts


def guard_url(url: str, allowed_hosts: list[str] | None = None, denied_addresses: list[str] | None = None,
              floor_hosts: list[str] | None = None) -> None:
    """Raise BlockedAddress unless ``url`` may be called. ``denied_addresses`` (the stack's own
    services and cloud metadata, from the platform) are refused on the resolved address, even for
    an allowed host, so re-pointing an allowed name does not reach them. ``floor_hosts`` (the
    deployment's own hosts, from the platform) are the only ones that may reach loopback or the
    unspecified address; None keeps the older rule, where any allowed host may."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise BlockedAddress(f"scheme {parts.scheme or '(none)'} is not allowed")
    host = (parts.hostname or "").lower()
    if not host:
        raise BlockedAddress("the URL has no host")
    port = parts.port or (443 if parts.scheme == "https" else 80)
    allowed = [h.lower() for h in (allowed_hosts or [])]
    if floor_hosts is None and _listed(host, port, allowed) and not denied_addresses:
        return
    checked_addresses(host, port, allowed_hosts, denied_addresses, floor_hosts)


def checked_addresses(host: str, port: int, allowed_hosts: list[str] | None, denied_addresses: list[str] | None,
                      floor_hosts: list[str] | None = None) -> list[ipaddress._BaseAddress]:
    """The addresses ``host`` resolves to, each checked by the rules of guard_url (as IPv4 when an IPv6
    address carries one); BlockedAddress when one of them may not be reached or the name does not resolve."""
    host = host.lower()
    allowed = [h.lower() for h in (allowed_hosts or [])]
    denied = {_as_ipv4(ipaddress.ip_address(a)) for a in (denied_addresses or [])}
    floor = None if floor_hosts is None else [h.lower() for h in floor_hosts]
    is_allowed = _listed(host, port, allowed)
    in_floor = floor is None or _listed(host, port, floor)
    addresses = [_as_ipv4(ip) for ip in _addresses(host, port)]
    if not addresses:
        raise BlockedAddress(f"{host} does not resolve")
    for ip in addresses:
        _check_address(host, ip, is_allowed, denied, in_floor)
    return addresses


def _check_address(host: str, ip, is_allowed: bool, denied: set, in_floor: bool = True) -> None:
    """One address of `host`, by the rules: a denied one never, else an allowed host, else no internal one;
    loopback or unspecified only for a host of the floor."""
    if ip in denied:
        raise BlockedAddress(f"{host} is a service of this deployment or a metadata address, never allowed")
    if not is_allowed and _refused_ip(ip):
        raise BlockedAddress(f"{host} is an internal address; allow it under Manage, Connections, Allowed internal hosts")
    if not in_floor and _is_loopback(ip):
        raise BlockedAddress(f"{host} is loopback or unspecified: only the deployment's CONNECTIONS_ALLOWED_HOSTS"
                             " may reach it")


#: The largest answer a system's call reads: past it the call is refused, and no more is read into
#: memory (the worker and the platform's Test button share their process with other work).
MAX_ANSWER_BYTES = 10 * 1024 * 1024


def _read_capped(stream, label: str) -> bytes:
    raw = stream.read(MAX_ANSWER_BYTES + 1)
    if len(raw) > MAX_ANSWER_BYTES:
        raise EndpointBadResponse(f"{label}: the answer is larger than {MAX_ANSWER_BYTES} bytes")
    return raw


#: The rules of the call in flight on this thread (allowed hosts, denied addresses, floor), for the peer check.
_POLICY = threading.local()


class _Policy:
    """While a call to a system runs, the address its socket reaches is checked again by the same rules:
    urllib resolves the name a second time to connect, and a name can answer differently (DNS
    rebinding). The platform's own calls run without one."""

    def __init__(self, allowed_hosts, denied_addresses, floor_hosts=None):
        self.allowed = [h.lower() for h in (allowed_hosts or [])]
        self.denied = {_as_ipv4(ipaddress.ip_address(a)) for a in (denied_addresses or [])}
        self.floor = None if floor_hosts is None else [h.lower() for h in floor_hosts]

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
    is_allowed = _listed(host, conn.port, policy.allowed)
    in_floor = policy.floor is None or _listed(host, conn.port, policy.floor)
    try:
        _check_address(host, _as_ipv4(ipaddress.ip_address(conn.sock.getpeername()[0])), is_allowed, policy.denied,
                       in_floor)
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


def _post_json(d: Descriptor, url: str, headers: dict, body: dict, allowed_hosts, denied_addresses=None,
               floor_hosts=None) -> Any:
    guard_url(url, allowed_hosts, denied_addresses, floor_hosts)
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST", headers=headers)
    try:
        with _Policy(allowed_hosts, denied_addresses, floor_hosts), _OPENER.open(req, timeout=d.timeout_s) as resp:
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
                latency: int, denied_addresses=None, floor_hosts=None) -> Answer:
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
                             allowed_hosts, denied_addresses, floor_hosts)


def call(d: Descriptor, input: Any, history: list | None = None, params: dict | None = None,
         allowed_hosts: list[str] | None = None, sleep: Callable[[float], None] = time.sleep,
         waits: tuple = RETRY_WAITS, denied_addresses: list[str] | None = None,
         floor_hosts: list[str] | None = None) -> Answer:
    """One question to the system, with the retries (``waits``: seconds before each retry; ``()``
    for a single probe, as the platform's Test button does); see EndpointClient.ask. The network rule
    (``allowed_hosts``, ``denied_addresses``, ``floor_hosts``) is guard_url's."""
    history = check_history(history)
    method, url, headers, data = render_request(d, input, history, dict(params or {}))
    waits = list(waits)
    while True:
        guard_url(url, allowed_hosts if allowed_hosts is not None else allowed_hosts_from_env(), denied_addresses,
                  floor_hosts)
        started = time.monotonic()
        retryable: EndpointError
        try:
            req = urllib.request.Request(url, data=data, method=method, headers=headers)
            with _Policy(allowed_hosts if allowed_hosts is not None else allowed_hosts_from_env(), denied_addresses,
                         floor_hosts), \
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
                                   sleep, status, latency, denied_addresses, floor_hosts)
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

"""The plugin's side of the platform: its credentials, the run key it takes for a connection, and the
EndpointClient a plugin asks with, through the platform's aisc/ask or directly."""
from __future__ import annotations

import http.client
import json
import os
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from .errors import (FACADE_ERRORS, BadReference, BadRequest, BlockedAddress, EndpointAuthError, EndpointBadResponse,
                     EndpointError, EndpointNotFound, EndpointTimeout, NoEndpoint, TargetHasNoEndpoint,
                     is_no_endpoint)
from .render import (Answer, Descriptor, _check_input, _platform_answer, _scrub, check_history, check_params,
                     reference_of)
from .transport import _OPENER, _Redirected, _read_capped, call


PLATFORM_AUTH_NEEDED = "PLATFORM_URL and PLATFORM_RUN_PASS (or PLATFORM_CONNECTIONS_TOKEN) must be set"


def platform_auth() -> dict | None:
    """The header that opens the platform's internal routes: the run's pass, which opens only what the
    run names; else the service token of older platforms and setups; None when there is neither."""
    run_pass = os.environ.get("PLATFORM_RUN_PASS") or ""
    if run_pass:
        return {"X-AISC-Run-Pass": run_pass}
    token = os.environ.get("PLATFORM_CONNECTIONS_TOKEN") or ""
    return {"X-AISC-Service-Token": token} if token else None


def _issue_run_key(pid: str, path: str, name: str) -> dict:
    """Ask the platform for a run key; ``path`` is ``targets/<key>``, or ``connections/<name>`` for a
    connection bound directly. When the platform refuses, its reason is passed on. The answer is checked
    before it is returned (see ``_checked_run_key``): the key goes to every endpoint it names."""
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
            answer = json.loads(raw)
        except ValueError:
            raise EndpointBadResponse(f"the platform's run-key answer was not JSON (connection {name!r})") from None
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
    return _checked_run_key(answer, name)


#: the roles of a run key answer's endpoints that are URLs: the key is sent to each of them
_URL_ROLES = ("ask_url", "base_url", "rpc_url", "agent_card_url")


def _checked_run_key(answer: Any, name: str) -> dict:
    """The platform's run-keys answer, once it is a JSON object with a key and every endpoint URL it
    carries is on PLATFORM_URL's origin: a tool sends the key to whichever of them it is pointed at."""
    if not isinstance(answer, dict):
        raise EndpointBadResponse(f"the platform's run key answer for {name!r} is not a JSON object")
    key = answer.get("key")
    if not isinstance(key, str) or not key:
        raise EndpointBadResponse(f"the platform's run key answer for {name!r} has no key")
    endpoints = answer.get("endpoints") or {}
    if not isinstance(endpoints, dict) or not all(isinstance(e, dict) for e in endpoints.values()):
        raise EndpointBadResponse(f"the platform's run key answer for {name!r} has endpoints that are not objects")
    platform = os.environ.get("PLATFORM_URL") or ""
    for protocol, endpoint in endpoints.items():
        for role in _URL_ROLES:
            if endpoint.get(role) is not None and not _on_platform(endpoint[role], platform):
                raise BlockedAddress(f"the platform's {protocol} {role} for {name!r} is not on PLATFORM_URL;"
                                     " the run key is not sent there")
    return answer


def _on_platform(url: Any, platform: str) -> bool:
    """Whether ``url`` is on the origin of ``platform`` and reads the same to every parser."""
    # only plain printable ASCII: no lookalike, whitespace or control character the check and the request
    # could read differently
    if not isinstance(url, str) or not url or not all("!" <= c <= "~" for c in url):
        return False
    try:
        netloc = urllib.parse.urlsplit(url).netloc
        # a userinfo part (or a backslash some parsers read as a path) can make another host the real one
        return "@" not in netloc and "\\" not in netloc and _origin(url) == _origin(platform)
    except ValueError:  # a URL the parser cannot read (a lookalike at sign, an unclosed bracket) is not trusted
        return False


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


def _run_key_from(answer: dict, name: str) -> RunKey:
    """The run key of a checked run-keys answer (``_issue_run_key``), with the platform's ask URL."""
    ask_url = ((answer.get("endpoints") or {}).get("aisc") or {}).get("ask_url")
    if not ask_url:
        raise EndpointBadResponse(f"the platform's run key answer for {name!r} has no aisc ask_url")
    return RunKey(ask_url=ask_url, key=answer["key"],
                  capabilities=tuple(answer.get("capabilities") or ("text", "history")),
                  expires_at=answer.get("expires_at"))


class EndpointClient:
    """One system under test, for one plugin run. ``run_key`` set: it asks through the platform (see the
    module docstring); None: it calls the system directly."""

    def __init__(self, descriptor: Descriptor, plugin=None, allowed_hosts: list[str] | None = None,
                 sleep: Callable[[float], None] = time.sleep, denied_addresses: list[str] | None = None,
                 run_key: RunKey | None = None, floor_hosts: list[str] | None = None):
        self.descriptor = descriptor
        self._plugin = plugin
        self._allowed = allowed_hosts
        self._denied = denied_addresses
        self._floor = floor_hosts
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
        except ValueError:
            data = None
        if not isinstance(data, dict):
            raise EndpointBadResponse(f"the platform's answer resolving connection {name!r} was not a JSON object")
        # The network rule is the platform's, per project: the internal hosts this connection may
        # reach, the addresses it never may, and the floor (the only hosts that may reach loopback; a
        # platform that sends none keeps the older rule). The run's own environment opens nothing.
        allowed, denied = list(data.get("allowed_hosts") or []), list(data.get("denied_addresses") or [])
        floor = data.get("floor_hosts")
        floor = list(floor) if isinstance(floor, list) else None
        if not os.environ.get("PLATFORM_RUN_PASS"):
            return cls(Descriptor.from_dict(data), plugin=plugin, allowed_hosts=allowed, denied_addresses=denied,
                       floor_hosts=floor, **kwargs)
        # under a run pass the key stays on the platform: the client asks it with a run key
        descriptor = Descriptor.from_dict(data)
        try:
            issued = _issue_run_key(pid, run_key_path, descriptor.name)
        except NoEndpoint as exc:
            raise TargetHasNoEndpoint(str(exc)) from None
        except EndpointError:
            raise
        except RuntimeError as exc:
            raise EndpointError(str(exc)) from None
        return cls(descriptor, plugin=plugin, allowed_hosts=allowed, denied_addresses=denied, floor_hosts=floor,
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
                          denied_addresses=self._denied, floor_hosts=self._floor)
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

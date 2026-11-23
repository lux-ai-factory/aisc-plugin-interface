"""The standard endpoint adapters (ADR 0009), ported from the backend's CON-01
adapters (``openai``, ``oip``, ``rest``, ``a2a``).

They are pure: they build the target's request and read its response. The caller
(the backend's "Test", or the eval door) is the only part that calls out.
"""
from __future__ import annotations

import base64
import json
import uuid
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from aisc_plugin_interface.adapters.base import (
    Answer,
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


class _TargetConfig(BaseModel):
    """Fields shared by the endpoint adapters that place their own secret header."""

    model_config = ConfigDict(extra="forbid")

    headers: dict[str, str] = Field(default_factory=dict)
    secret_header: str = ""


def _base(url: str) -> str:
    return (url or "").rstrip("/")


def _headers(config: _TargetConfig, secret: Secret) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    headers.update(config.headers)
    headers.update(secret_headers(secret, secret_header=config.secret_header))
    return headers


# --------------------------------------------------------------------------- #
# OpenAI-compatible passthrough: the door forwards /v1/* unchanged, only adding
# the key. It performs no conversion.
# --------------------------------------------------------------------------- #
class OpenAiEndpointConfig(_TargetConfig):
    pass


class OpenAiEndpointAdapter(EndpointAdapter):
    id = "openai"
    label = "OpenAI-compatible"
    speaks = "openai-chat"
    serves = ("openai-chat",)
    passthrough = True
    capabilities: ClassVar[dict[str, bool]] = {"chat": True, "streaming": True, "models": True}
    Config = OpenAiEndpointConfig

    def rewrite(
        self,
        target: Any,
        secret: Secret,
        *,
        path: str,
        method: str,
        headers: dict[str, str] | None = None,
        json_body: Any = None,
        text_body: str | None = None,
    ) -> Request:
        base = _base(target.url)
        out: dict[str, str] = {}
        if headers:
            out.update({k: v for k, v in headers.items() if k.lower() != "authorization"})
        out.update(_headers(self.config, secret))
        return Request(
            method=method.upper(),
            url=f"{base}/{path.lstrip('/')}" if path else base,
            headers=out,
            json_body=json_body,
            text_body=text_body,
        )

    def build_request(self, target: Any, secret: Secret, *, turn: Turn) -> Request:
        """The one call a "Test" makes: a minimal chat completion."""
        base = _base(target.url)
        messages = list(turn.history) + [{"role": "user", "content": turn.input}]
        body = {"model": target.model, "messages": messages}
        return Request(method="POST", url=f"{base}/chat/completions", headers=_headers(self.config, secret), json_body=body)

    def parse_response(self, status: int, text: str, target: Any) -> Answer:
        if status >= 400:
            return Answer(ok=False, error="http_status", detail=f"the endpoint answered HTTP {status}")
        document = parse_json(text)
        if not isinstance(document, dict) or not document.get("choices"):
            return Answer(ok=False, error="no_answer", detail="the response carried no choices")
        choice = document["choices"][0]
        message = choice.get("message") or {}
        answer = message.get("content")
        if answer is None and message.get("tool_calls"):
            answer = message["tool_calls"]
        if answer is None:
            return Answer(ok=False, error="no_answer", detail="the choice carried no content")
        return Answer(
            ok=True,
            answer=answer,
            capabilities={
                "tools": bool(message.get("tool_calls")),
                "logprobs": bool(choice.get("logprobs")),
            },
        )


# --------------------------------------------------------------------------- #
# Open Inference Protocol (KServe v2): converts to the common turn format.
# --------------------------------------------------------------------------- #
class OipEndpointConfig(_TargetConfig):
    pass


class OipEndpointAdapter(EndpointAdapter):
    id = "oip"
    label = "Open Inference Protocol"
    speaks = "oip-v2"
    serves = ("openai-chat",)
    Config = OipEndpointConfig

    def build_request(self, target: Any, secret: Secret, *, turn: Turn) -> Request:
        base = _base(target.url)
        text = turn.input if isinstance(turn.input, str) else str(turn.input)
        encoded = base64.b64encode(text.encode()).decode()
        body = {
            "id": str(uuid.uuid4()),
            "inputs": [
                {"name": "text", "shape": [1], "datatype": "BYTES", "data": [encoded]},
            ],
        }
        return Request(
            method="POST",
            url=f"{base}/v2/models/{target.model}/infer",
            headers=_headers(self.config, secret),
            json_body=body,
        )

    def parse_response(self, status: int, text: str, target: Any) -> Answer:
        if status >= 400:
            return Answer(ok=False, error="http_status", detail=f"the server answered HTTP {status}")
        document = parse_json(text)
        outputs = (document or {}).get("outputs") if isinstance(document, dict) else None
        if not outputs:
            return Answer(ok=False, error="no_answer", detail="the response carried no outputs")
        output = outputs[0] or {}
        data = output.get("data")
        value = data[0] if isinstance(data, list) and data else data
        if output.get("datatype") == "BYTES" and isinstance(value, str):
            try:
                value = base64.b64decode(value).decode()
            except (ValueError, UnicodeDecodeError):
                pass
        return Answer(ok=True, answer=value)


# --------------------------------------------------------------------------- #
# REST: a method, a path, a templated body, and where the answer is.
# --------------------------------------------------------------------------- #
class RestEndpointConfig(_TargetConfig):
    path: str = ""
    method: str = "POST"
    body_template: dict[str, Any] = Field(default_factory=dict)
    response_path: str = ""
    refusal: dict[str, Any] = Field(default_factory=dict)


def _matches_refusal(refusal: dict, status: int, document: Any) -> bool:
    criteria = 0
    statuses = refusal.get("status")
    if statuses is not None:
        criteria += 1
        if status not in (statuses if isinstance(statuses, list) else [statuses]):
            return False
    match = refusal.get("match")
    if isinstance(match, dict) and match:
        criteria += 1
        if not all(json_path(document, path) == expected for path, expected in match.items()):
            return False
    path = refusal.get("path")
    if path:
        criteria += 1
        if json_path(document, path) is None and not match:
            return False
    if criteria == 0:
        return status >= 400
    return True


class RestEndpointAdapter(EndpointAdapter):
    id = "rest"
    label = "REST API"
    speaks = "rest"
    serves = ("openai-chat", "agent")
    Config = RestEndpointConfig

    def build_request(self, target: Any, secret: Secret, *, turn: Turn) -> Request:
        config = self.config
        context = context_of(turn.input, turn.history, turn.params)
        base = _base(target.url)
        path = (config.path or "").lstrip("/")
        return Request(
            method=(config.method or "POST").upper(),
            url=f"{base}/{path}" if path else base,
            headers=_headers(config, secret),
            json_body=render(config.body_template or {}, context),
        )

    def parse_response(self, status: int, text: str, target: Any) -> Answer:
        document = parse_json(text)
        refusal = self.config.refusal or None
        if isinstance(refusal, dict) and refusal and _matches_refusal(refusal, status, document):
            path = refusal.get("path")
            reason = json_path(document, path) if path else None
            return Answer(ok=True, refused=True, refusal_reason=reason or refusal.get("reason") or None)
        if status >= 400:
            return Answer(ok=False, error="http_status", detail=f"the system answered HTTP {status}")
        if self.config.response_path:
            answer = json_path(document, self.config.response_path)
        else:
            answer = document if document is not None else text
        if answer is None:
            return Answer(
                ok=False,
                error="no_answer",
                detail=f"nothing at {self.config.response_path!r} in the response",
            )
        return Answer(ok=True, answer=answer)


# --------------------------------------------------------------------------- #
# A2A: JSON-RPC message/send to an agent's interface. One turn per call; no
# conversation id is carried (stateful sessions are later work).
# --------------------------------------------------------------------------- #
AGENT_CARD_PATH = {
    "1.0": "/.well-known/agent-card.json",
    "0.3": "/.well-known/agent.json",
}


def agent_card_url(base_url: str, protocol_version: str = "") -> str:
    base = _base(base_url)
    path = AGENT_CARD_PATH.get(protocol_version or "1.0", AGENT_CARD_PATH["1.0"])
    return f"{base}{path}"


def _text_of(result: dict) -> Any:
    for artifact in result.get("artifacts") or []:
        for part in artifact.get("parts") or []:
            if part.get("text") is not None:
                return part["text"]
    message = (result.get("status") or {}).get("message") or {}
    for part in message.get("parts") or []:
        if part.get("text") is not None:
            return part["text"]
    return None


class A2aEndpointConfig(_TargetConfig):
    path: str = ""
    protocol_version: str = "1.0"


class A2aEndpointAdapter(EndpointAdapter):
    id = "a2a"
    label = "A2A agent"
    speaks = "agent"
    serves = ("openai-chat",)
    Config = A2aEndpointConfig

    def build_request(self, target: Any, secret: Secret, *, turn: Turn) -> Request:
        config = self.config
        base = _base(target.url)
        path = (config.path or "").lstrip("/")
        text = turn.input if isinstance(turn.input, str) else json.dumps(turn.input)
        body = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": text}],
                    "messageId": str(uuid.uuid4()),
                }
            },
        }
        return Request(
            method="POST",
            url=f"{base}/{path}" if path else base,
            headers=_headers(config, secret),
            json_body=body,
        )

    def parse_response(self, status: int, text: str, target: Any) -> Answer:
        if status >= 400:
            return Answer(ok=False, error="http_status", detail=f"the agent answered HTTP {status}")
        document = parse_json(text)
        if not isinstance(document, dict):
            return Answer(ok=False, error="no_answer", detail="the agent's answer was not JSON")
        if document.get("error"):
            error = document["error"] or {}
            return Answer(ok=False, error="rpc_error", detail=error.get("message", "the agent refused the call"))
        result = document.get("result") or {}
        answer = _text_of(result)
        if answer is None:
            answer = result if result else None
        if answer is None:
            return Answer(ok=False, error="no_answer", detail="the agent's result carried no text")
        return Answer(ok=True, answer=answer)


#: The standard endpoint library.
BUILTIN_ENDPOINT_ADAPTERS: tuple[type[EndpointAdapter], ...] = (
    OpenAiEndpointAdapter,
    OipEndpointAdapter,
    RestEndpointAdapter,
    A2aEndpointAdapter,
)

__all__ = [
    "OpenAiEndpointAdapter",
    "OpenAiEndpointConfig",
    "OipEndpointAdapter",
    "OipEndpointConfig",
    "RestEndpointAdapter",
    "RestEndpointConfig",
    "A2aEndpointAdapter",
    "A2aEndpointConfig",
    "agent_card_url",
    "AGENT_CARD_PATH",
    "BUILTIN_ENDPOINT_ADAPTERS",
]

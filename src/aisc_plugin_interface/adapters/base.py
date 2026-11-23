"""
The adapter contract: how a target's protocol becomes a call, and how a file
becomes the value a plugin slot accepts.

Two kinds (ADR 0009):

* a **data adapter** converts a file format to a value type the slot accepts. It
  runs once, in the plugin process.
* an **endpoint adapter** converts a target's protocol to the protocol a plugin
  speaks (its *face*). It runs per call, behind the run door.

Both are pure: they build a request and read a response, or read bytes into a
value. The caller (the backend's "Test", or the eval door) is the only part that
does network I/O, so each protocol is testable on its own.
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, TypeVar

from pydantic import BaseModel, ConfigDict

T = TypeVar("T")

#: A `{{name}}` placeholder in a REST body or path.
_PLACEHOLDER = re.compile(r"\{\{\s*([a-zA-Z0-9_.]+)\s*\}\}")
#: A `data.items[0].text` style JSON path.
_PATH_TOKEN = re.compile(r"([^.\[\]]+)|\[(\d+)\]")


class EmptyConfig(BaseModel):
    """The config of an adapter that takes none. ``extra="forbid"`` makes a typo a
    validation error rather than a silent no-op."""

    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True)
class Secret:
    """A target's key and secret headers, resolved for one call."""

    key: str | None = None
    headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Request:
    method: str = "POST"
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    json_body: Any = None
    text_body: str | None = None


@dataclass(frozen=True)
class Answer:
    ok: bool
    answer: Any = None
    refused: bool = False
    refusal_reason: str | None = None
    error: str | None = None
    detail: str = ""
    capabilities: dict[str, bool] = field(default_factory=dict)


@dataclass(frozen=True)
class Turn:
    """One request from a plugin, in the adapter-neutral shape.

    The *face* (the protocol the plugin speaks) parses the plugin's request into a
    Turn; the adapter builds its target's request from it. ``input`` is the new
    user turn, ``history`` the earlier ones, ``params`` any extra the face carried.
    """

    input: Any = None
    history: list[Any] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    def context(self) -> dict[str, Any]:
        return {"input": self.input, "history": list(self.history), "params": dict(self.params)}


def context_of(input: Any, history: list | None, params: dict | None) -> dict[str, Any]:
    return {"input": input, "history": history or [], "params": params or {}}


def _lookup(name: str, context: dict[str, Any]) -> Any:
    value: Any = context
    for part in name.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return f"{{{{{name}}}}}"
    return value


def render(template: Any, context: dict[str, Any]) -> Any:
    """Substitute `{{input}}`, `{{history}}` and `{{params.x}}` through a JSON body.

    A value that is exactly one placeholder is replaced by the value itself (so
    `{"applicant": "{{params.applicant}}"}` can carry an object); a placeholder
    inside a longer string is interpolated as text.
    """
    if isinstance(template, str):
        stripped = template.strip()
        match = _PLACEHOLDER.fullmatch(stripped)
        if match:
            return _lookup(match.group(1), context)
        return _PLACEHOLDER.sub(lambda m: str(_lookup(m.group(1), context)), template)
    if isinstance(template, dict):
        return {k: render(v, context) for k, v in template.items()}
    if isinstance(template, list):
        return [render(v, context) for v in template]
    return template


def json_path(document: Any, path: str) -> Any:
    """Read `data.items[0].text` out of a decoded JSON document."""
    value = document
    for name, index in _PATH_TOKEN.findall(path or ""):
        if index:
            if not isinstance(value, list) or int(index) >= len(value):
                return None
            value = value[int(index)]
        elif isinstance(value, dict):
            value = value.get(name)
        else:
            return None
    return value


def parse_json(text: str) -> Any:
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return None


def secret_headers(secret: Secret, *, secret_header: str = "") -> dict[str, str]:
    """The Authorization/key header plus any stored secret header.

    `secret_header` is either a literal header line (`X-Token: {{secret}}`) or
    empty, which means the protocol's own default (`Authorization: Bearer <key>`).
    `{{secret}}` is replaced by the key.
    """
    headers: dict[str, str] = {}
    line = (secret_header or "").strip()
    if line:
        name, _, value = line.partition(":")
        value = value.replace("{{secret}}", secret.key or "")
        headers[name.strip()] = value.strip()
    elif secret.key:
        headers["Authorization"] = f"Bearer {secret.key}"
    headers.update(secret.headers)
    return headers


class Adapter(ABC):
    """What every adapter has in common: an id, a version and a label."""

    id: ClassVar[str] = ""
    version: ClassVar[str] = "1"
    label: ClassVar[str] = ""
    #: The pydantic model that validates this adapter's config. Rendered as a form.
    Config: ClassVar[type[BaseModel]] = EmptyConfig

    def __init__(self, config: BaseModel | dict[str, Any] | None = None):
        self.config = self._coerce_config(config)

    @classmethod
    def _coerce_config(cls, config: BaseModel | dict[str, Any] | None) -> BaseModel:
        if config is None:
            return cls.Config()
        if isinstance(config, cls.Config):
            return config
        if isinstance(config, BaseModel):
            config = config.model_dump()
        return cls.Config.model_validate(config)

    @classmethod
    def config_schema(cls) -> dict[str, Any]:
        return cls.Config.model_json_schema()


class DataAdapter(Adapter):
    """A file format -> a value type the slot accepts. Runs once, in the plugin.

    ``reads`` is the set of formats it can read (``"*"`` for any) and ``produces``
    the value types it can hand the plugin; the ``produce`` argument picks one.
    """

    reads: ClassVar[tuple[str, ...]] = ()
    produces: ClassVar[tuple[str, ...]] = ()

    def __init__(self, config: BaseModel | dict[str, Any] | None = None):
        super().__init__(config)

    @abstractmethod
    def load(self, source: bytes | Path | Any, produce: str) -> Any:
        """Read ``source`` and return the value of type ``produce``."""
        raise NotImplementedError


class EndpointAdapter(Adapter):
    """A target's protocol -> the protocol a plugin speaks.

    ``speaks`` is the target protocol (what the system under test answers on);
    ``serves`` the face protocols this adapter can back. A *passthrough* adapter
    speaks and serves the same protocol and forwards requests unchanged
    (``passthrough = True``); it implements :meth:`rewrite` instead of
    :meth:`build_request`/:meth:`parse_response`.
    """

    speaks: ClassVar[str] = ""
    serves: ClassVar[tuple[str, ...]] = ()
    passthrough: ClassVar[bool] = False
    capabilities: ClassVar[dict[str, bool]] = {}
    #: How many calls this adapter may have in flight at once, at the door.
    max_concurrency: ClassVar[int] = 8

    def build_request(self, target: Any, secret: Secret, *, turn: Turn) -> Request:
        """The HTTP request to the target for one plugin turn (converting adapters)."""
        raise NotImplementedError

    def parse_response(self, status: int, text: str, target: Any) -> Answer:
        """The target's response as an :class:`Answer` (converting adapters)."""
        raise NotImplementedError

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
        """A passthrough's rewrite: keep the method, body and query; add the key."""
        raise NotImplementedError


__all__ = [
    "T",
    "EmptyConfig",
    "Secret",
    "Request",
    "Answer",
    "Turn",
    "Adapter",
    "DataAdapter",
    "EndpointAdapter",
    "context_of",
    "render",
    "json_path",
    "parse_json",
    "secret_headers",
]

"""The exceptions a connection raises, each a kind of failure a plugin may tell apart. A message never carries
the key."""
from __future__ import annotations

from typing import Any


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

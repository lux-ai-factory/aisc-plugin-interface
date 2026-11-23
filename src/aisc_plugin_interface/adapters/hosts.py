"""
Allowed internal hosts: which addresses a run may call. (Ported from the backend's
``services/allowed_hosts`` so the backend and the run door share one rule — I3.)

A target may be any **public** address. An **internal** one — a private network,
the Docker host, a machine on the LAN — is refused unless the project lists it or
the stack allows it for every project. Loopback (the platform itself), cloud
metadata and this deployment's own services can never be allowed.

The classification helpers are pure and take no database or framework, so the
rules are testable on their own and importable inside the run door.
"""
from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit

#: Host names a cloud provider answers metadata on. Never reachable.
METADATA_HOSTS = frozenset(
    {
        "169.254.169.254",  # AWS / Azure / GCP / OpenStack
        "169.254.169.253",  # AWS DNS
        "169.254.170.2",  # ECS task metadata
        "100.100.100.200",  # Alibaba Cloud
        "metadata.google.internal",
        "metadata.goog",
        "instance-data",
        "fd00:ec2::254",  # AWS IMDSv2 over IPv6
    }
)

#: Name suffixes that are not on the public internet.
INTERNAL_SUFFIXES = (
    ".local",
    ".localdomain",
    ".internal",
    ".intranet",
    ".corp",
    ".lan",
    ".home",
    ".svc",
    ".cluster.local",
)

#: The host the Docker host itself answers on (resolved inside a container).
DOCKER_HOST_NAMES = frozenset({"host.docker.internal", "gateway.docker.internal"})


def hostname_of(url: str) -> str:
    """The host of an address, lower-cased; '' when it names none."""
    text = (url or "").strip()
    if not text:
        return ""
    if "//" not in text:
        text = "//" + text.split("/", 1)[0]
    try:
        return (urlsplit(text).hostname or "").lower()
    except ValueError:
        return ""


def port_of(url: str) -> str:
    text = (url or "").strip()
    if not text:
        return ""
    if "//" not in text:
        text = "//" + text.split("/", 1)[0]
    try:
        return str(urlsplit(text).port or "")
    except ValueError:
        return ""


def normalize_entry(entry: str) -> str | None:
    """A form entry (`host` or `host:port`, maybe a URL) as a stored key, or None."""
    text = (entry or "").strip().lower()
    if not text:
        return None
    if "//" not in text and "/" in text:
        return None
    if "//" not in text:
        text = "//" + text
    try:
        parts = urlsplit(text)
        host = (parts.hostname or "").lower()
        port = parts.port
    except ValueError:
        return None
    if not host or not host.strip("."):
        return None
    if any(bad in host for bad in " \t\r\n/@?#"):
        return None
    return f"{host}:{port}" if port else host


def is_loopback(host: str) -> bool:
    name = (host or "").lower()
    if not name:
        return False
    if name in {"localhost", "0.0.0.0", "::", "::1", "[::1]"} or name.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


def is_metadata(host: str) -> bool:
    return (host or "").lower() in METADATA_HOSTS


def is_internal(host: str) -> bool:
    """Is this address off the public internet?"""
    name = (host or "").lower().strip(".")
    if not name:
        return True
    if is_loopback(name) or is_metadata(name):
        return True
    if name in DOCKER_HOST_NAMES:
        return True
    try:
        address = ipaddress.ip_address(name)
    except ValueError:
        if "." not in name:
            return True
        return any(name.endswith(suffix) for suffix in INTERNAL_SUFFIXES)
    return not address.is_global or address.is_private or address.is_link_local


def refusal_for(
    url: str, *, allowed: set[str], floor: set[str], stack: set[str]
) -> tuple[str, str] | None:
    """Why a call to `url` is refused, or None when it may go out.

    `allowed` are the project's entries, `floor` the stack's for every project and
    `stack` this deployment's own service names. Both lists are matched on
    `host:port` first, then on the bare host.
    """
    host = hostname_of(url)
    if not host:
        return ("invalid_url", "the address names no host to call")
    if host in {h.lower().strip(".") for h in stack}:
        return (
            "deployment_service",
            f"{host} is one of this deployment's own services; a test reaches the "
            "assessed system, never the platform",
        )
    if is_loopback(host):
        return ("loopback", "loopback addresses (the platform itself) can never be called")
    if is_metadata(host):
        return (
            "metadata",
            "cloud metadata addresses can never be called: they hand out the node's "
            "own credentials",
        )
    if not is_internal(host):
        return None
    entry = f"{host}:{port_of(url)}" if port_of(url) else host
    if entry in allowed or host in allowed:
        return None
    if entry in floor or host in floor:
        return None
    return (
        "internal_not_allowed",
        f"the internal host {host} is not allowed for this project; an owner can add "
        "it under Allowed internal hosts",
    )


def is_never_allowed(url: str, *, stack: set[str]) -> str | None:
    """The reason a host can never be listed, or None."""
    host = hostname_of(url)
    if not host:
        return "the address names no host"
    if host in {h.lower().strip(".") for h in stack}:
        return f"{host} is one of this deployment's own services and can never be reached"
    if is_loopback(host):
        return "loopback addresses (the platform itself) can never be reached"
    if is_metadata(host):
        return "cloud metadata addresses can never be reached"
    return None


__all__ = [
    "METADATA_HOSTS",
    "INTERNAL_SUFFIXES",
    "DOCKER_HOST_NAMES",
    "hostname_of",
    "port_of",
    "normalize_entry",
    "is_loopback",
    "is_metadata",
    "is_internal",
    "refusal_for",
    "is_never_allowed",
]

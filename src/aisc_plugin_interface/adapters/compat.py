"""The one compatibility rule (ADR 0009): which adapters fit a plugin slot.

A slot declares the value types (data) or face protocols (endpoint) it accepts;
a component declares its kind, and for an endpoint its target protocol. Both the
backend (the form's candidate list, the "Test") and the eval worker (resolving an
input) use these functions, so "what may I pick here" has one answer.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

from aisc_plugin_interface.adapters.base import DataAdapter, EndpointAdapter


def accepts_of(slot: Any) -> set[str]:
    return set(getattr(slot, "accepts", None) or [])


def data_fits(
    component_format: str, accepts: set[str], adapter: type[DataAdapter] | DataAdapter
) -> bool:
    reads = getattr(adapter, "reads", ()) or ()
    produces = getattr(adapter, "produces", ()) or ()
    if "*" not in reads and component_format not in reads:
        return False
    return bool(set(produces) & accepts)


def endpoint_fits(
    component_protocol: str, accepts: set[str], adapter: type[EndpointAdapter] | EndpointAdapter
) -> bool:
    speaks = getattr(adapter, "speaks", "")
    serves = getattr(adapter, "serves", ()) or ()
    if component_protocol and speaks != component_protocol:
        return False
    return bool(set(serves) & accepts)


def data_candidates(
    component_format: str, accepts: set[str], adapters: Iterable[type[DataAdapter]]
) -> list[type[DataAdapter]]:
    return [a for a in adapters if data_fits(component_format, accepts, a)]


def endpoint_candidates(
    component_protocol: str, accepts: set[str], adapters: Iterable[type[EndpointAdapter]]
) -> list[type[EndpointAdapter]]:
    return [a for a in adapters if endpoint_fits(component_protocol, accepts, a)]


def _order(candidates: Sequence[type], default_id: str | None) -> list[type]:
    """The slot's declared default first, then the rest in catalogue order."""
    if not default_id:
        return list(candidates)
    first = [a for a in candidates if getattr(a, "id", "") == default_id]
    rest = [a for a in candidates if getattr(a, "id", "") != default_id]
    return first + rest


def data_slot_candidates(
    component_format: str, slot: Any, adapters: Iterable[type[DataAdapter]]
) -> list[type[DataAdapter]]:
    return _order(
        data_candidates(component_format, accepts_of(slot), adapters),
        getattr(slot, "adapter", None),
    )


def endpoint_slot_candidates(
    component_protocol: str, slot: Any, adapters: Iterable[type[EndpointAdapter]]
) -> list[type[EndpointAdapter]]:
    return _order(
        endpoint_candidates(component_protocol, accepts_of(slot), adapters),
        getattr(slot, "adapter", None),
    )


def is_endpoint_kind(kind: str) -> bool:
    return kind == "endpoint"


__all__ = [
    "accepts_of",
    "data_fits",
    "endpoint_fits",
    "data_candidates",
    "endpoint_candidates",
    "data_slot_candidates",
    "endpoint_slot_candidates",
    "is_endpoint_kind",
]

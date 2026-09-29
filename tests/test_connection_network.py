"""The network rule a run gets from the platform (allowlist task 2026-09-29, N1 to N6): the
platform computes, per project, which internal hosts a connection may reach and which addresses
are never reachable (the stack's own services, cloud metadata), and hands both to the client in
the resolve response. The deny list holds on the resolved address, even for an allowed host, so
re-pointing a name at a stack service does not get through."""
import socket

import pytest

from aisc_plugin_interface import connections as c
from tests.test_connections import MCAS, desc as _desc

BODY = {"question": "{{input}}", "history": "{{history}}"}
PLAIN = dict(MCAS, body_template=BODY)


def desc(stub, **over):
    return _desc(stub, body_template=BODY, **over)


def resolving(monkeypatch, table):
    """getaddrinfo answering from `table` (name -> ip) and failing for anything else."""
    def fake(host, port, *a, **k):
        if host not in table:
            raise socket.gaierror(host)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (table[host], port))]
    monkeypatch.setattr(socket, "getaddrinfo", fake)


# ── N1 to N3 guard_url ──────────────────────────────────────────────────────

def test_n1_an_allowed_host_whose_address_is_denied_is_blocked():
    with pytest.raises(c.BlockedAddress, match="never"):
        c.guard_url("http://172.20.0.5:5432/x", ["172.20.0.5:5432"], denied_addresses=["172.20.0.5"])


def test_n2_an_allowed_name_that_resolves_to_a_denied_address_is_blocked(monkeypatch):
    resolving(monkeypatch, {"innocent.internal": "172.20.0.5"})
    with pytest.raises(c.BlockedAddress, match="never"):
        c.guard_url("http://innocent.internal:8500/chat", ["innocent.internal:8500"],
                    denied_addresses=["172.20.0.5"])


def test_n2_an_allowed_name_on_another_address_goes_through(monkeypatch):
    resolving(monkeypatch, {"mcas.internal": "172.30.0.9"})
    c.guard_url("http://mcas.internal:8500/chat", ["mcas.internal:8500"], denied_addresses=["172.20.0.5"])


def test_n3_without_a_deny_list_an_allowed_host_is_not_resolved(monkeypatch):
    resolving(monkeypatch, {})       # would fail to resolve: the old rule never asks
    c.guard_url("http://anything.internal:8500/chat", ["anything.internal:8500"])


# ── N4 a call honours it before sending anything ────────────────────────────

def test_n4_a_denied_address_is_refused_before_any_request(stub):
    stub.route("/chat", (200, {"answer": "should not be asked"}))
    with pytest.raises(c.BlockedAddress):
        c.call(desc(stub), "q", allowed_hosts=[stub.host], denied_addresses=["127.0.0.1"], waits=())
    assert stub.seen == []


def test_n4_the_client_passes_its_deny_list_on(stub):
    stub.route("/chat", (200, {"answer": "ok"}))
    cl = c.EndpointClient(desc(stub), allowed_hosts=[stub.host], denied_addresses=["127.0.0.1"], sleep=lambda s: None)
    with pytest.raises(c.BlockedAddress):
        cl.ask("q")


# ── N5 and N6 a run takes the rule from the platform, not from its own env ───

def _platform(stub, monkeypatch, network):
    stub.route("/internal/projects/pid-1/connections/mcas-chat", (200, dict(PLAIN, base_url=stub.base, **network)))
    stub.route("/chat", (200, {"answer": "Up to 5,000 EUR."}))
    monkeypatch.setenv("PLATFORM_URL", stub.base)
    monkeypatch.setenv("PLATFORM_CONNECTIONS_TOKEN", "tok")


def test_n5_the_workers_own_env_var_no_longer_opens_anything(plugin, stub, monkeypatch):
    _platform(stub, monkeypatch, {"allowed_hosts": [], "denied_addresses": []})
    monkeypatch.setenv("CONNECTIONS_ALLOWED_HOSTS", stub.host)
    cl = c.EndpointClient.for_input(plugin({"system": {"value": "connection:pid-1/mcas-chat"}}), "system",
                                    sleep=lambda s: None)
    with pytest.raises(c.BlockedAddress):
        cl.ask("How much?")


def test_n6_the_platforms_list_opens_the_host_and_its_deny_list_holds(plugin, stub, monkeypatch):
    monkeypatch.delenv("CONNECTIONS_ALLOWED_HOSTS", raising=False)
    _platform(stub, monkeypatch, {"allowed_hosts": [stub.host], "denied_addresses": []})
    p = plugin({"system": {"value": "connection:pid-1/mcas-chat"}})
    assert c.EndpointClient.for_input(p, "system", sleep=lambda s: None).ask("How much?").text == "Up to 5,000 EUR."
    _platform(stub, monkeypatch, {"allowed_hosts": [stub.host], "denied_addresses": ["127.0.0.1"]})
    with pytest.raises(c.BlockedAddress):
        c.EndpointClient.for_input(p, "system", sleep=lambda s: None).ask("How much?")


def test_n6_the_rule_is_not_part_of_the_descriptor(plugin, stub, monkeypatch):
    _platform(stub, monkeypatch, {"allowed_hosts": [stub.host], "denied_addresses": ["10.9.9.9"]})
    cl = c.EndpointClient.for_input(plugin({"system": {"value": "connection:pid-1/mcas-chat"}}), "system")
    assert "allowed_hosts" not in cl.descriptor.public() and "denied_addresses" not in cl.descriptor.public()


def test_n7_a_refusal_says_where_to_allow_the_host(monkeypatch):
    resolving(monkeypatch, {"mcas.internal": "10.1.2.3"})
    with pytest.raises(c.BlockedAddress, match="Allowed internal hosts"):
        c.guard_url("http://mcas.internal:8500/chat", [])

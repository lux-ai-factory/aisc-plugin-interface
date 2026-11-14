"""Loopback and the unspecified address (0.0.0.0, ::, and their IPv4-mapped forms) are reached only by the
hosts of the deployment's own floor, even when a project allows the name; and every address that is not
globally routable (shared CGNAT space, benchmarking, IETF protocol assignments) is internal. The platform's
rule names the floor, and the client a run gets keeps it."""
import ipaddress
import socket

import pytest

from aisc_plugin_interface import connections as c
from tests.test_connection_network import PLAIN, resolving


def resolving_any(monkeypatch, table):
    """Like `resolving`, but answers IPv6 addresses with an IPv6 family."""
    def fake(host, port, *a, **k):
        if host not in table:
            raise socket.gaierror(host)
        ip = table[host]
        family = socket.AF_INET6 if ":" in ip else socket.AF_INET
        return [(family, socket.SOCK_STREAM, 6, "", (ip, port))]
    monkeypatch.setattr(socket, "getaddrinfo", fake)


@pytest.mark.parametrize("ip", ["100.64.0.1", "100.127.255.254", "198.18.0.1", "192.0.0.8"])
def test_non_global_ranges_are_refused_without_an_entry(monkeypatch, ip):
    resolving(monkeypatch, {"svc.example": ip})
    with pytest.raises(c.BlockedAddress, match="internal address"):
        c.guard_url("http://svc.example/x", [], [])


@pytest.mark.parametrize("ip", ["127.0.0.1", "127.0.0.2", "0.0.0.0", "::1", "::", "::ffff:127.0.0.1"])
def test_an_allowed_host_reaching_loopback_is_refused_unless_it_is_in_the_floor(monkeypatch, ip):
    resolving_any(monkeypatch, {"innocent.internal": ip})
    try:
        with pytest.raises(c.BlockedAddress, match="loopback"):
            c.guard_url("http://innocent.internal:8500/x", ["innocent.internal:8500"], [], floor_hosts=[])
        c.guard_url("http://innocent.internal:8500/x", ["innocent.internal:8500"], [],
                    floor_hosts=["innocent.internal:8500"])
    except TypeError as exc:
        pytest.fail(f"the guard takes no floor: {exc}")


def test_a_rule_from_the_platform_carries_its_floor_to_the_client(plugin, stub, monkeypatch):
    monkeypatch.delenv("CONNECTIONS_ALLOWED_HOSTS", raising=False)
    monkeypatch.delenv("PLATFORM_RUN_PASS", raising=False)
    stub.route("/internal/projects/pid-1/connections/mcas-chat",
               (200, dict(PLAIN, base_url=stub.base, allowed_hosts=[stub.host], denied_addresses=[],
                          floor_hosts=[])))
    stub.route("/chat", (200, {"answer": "should not be asked"}))
    monkeypatch.setenv("PLATFORM_URL", stub.base)
    monkeypatch.setenv("PLATFORM_CONNECTIONS_TOKEN", "tok")
    cl = c.EndpointClient.for_input(plugin({"system": {"value": "connection:pid-1/mcas-chat"}}), "system",
                                    sleep=lambda s: None)
    with pytest.raises(c.BlockedAddress):
        cl.ask("How much?")
    assert [s["path"] for s in stub.seen if s["path"].startswith("/chat")] == []


def test_checked_addresses_names_the_addresses_the_guard_lets_through(monkeypatch):
    resolving(monkeypatch, {"api.example": "93.184.216.34"})
    checked = getattr(c, "checked_addresses", None)
    assert checked is not None, "the guard does not say which addresses it let through"
    assert checked("api.example", 443, [], [], []) == [ipaddress.ip_address("93.184.216.34")]

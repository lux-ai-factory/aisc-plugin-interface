"""The network rule a run gets from the platform: the
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


# guard_url

def test_an_allowed_host_whose_address_is_denied_is_blocked():
    with pytest.raises(c.BlockedAddress, match="never"):
        c.guard_url("http://172.20.0.5:5432/x", ["172.20.0.5:5432"], denied_addresses=["172.20.0.5"])


def test_an_allowed_name_that_resolves_to_a_denied_address_is_blocked(monkeypatch):
    resolving(monkeypatch, {"innocent.internal": "172.20.0.5"})
    with pytest.raises(c.BlockedAddress, match="never"):
        c.guard_url("http://innocent.internal:8500/chat", ["innocent.internal:8500"],
                    denied_addresses=["172.20.0.5"])


def test_an_allowed_name_on_another_address_goes_through(monkeypatch):
    resolving(monkeypatch, {"mcas.internal": "172.30.0.9"})
    c.guard_url("http://mcas.internal:8500/chat", ["mcas.internal:8500"], denied_addresses=["172.20.0.5"])


def test_without_a_deny_list_an_allowed_host_is_not_resolved(monkeypatch):
    resolving(monkeypatch, {})       # would fail to resolve: an allowed host is not looked up
    c.guard_url("http://anything.internal:8500/chat", ["anything.internal:8500"])


# a call honours the rule before sending anything

def test_a_denied_address_is_refused_before_any_request(stub):
    stub.route("/chat", (200, {"answer": "should not be asked"}))
    with pytest.raises(c.BlockedAddress):
        c.call(desc(stub), "q", allowed_hosts=[stub.host], denied_addresses=["127.0.0.1"], waits=())
    assert stub.seen == []


def test_the_client_passes_its_deny_list_on(stub):
    stub.route("/chat", (200, {"answer": "ok"}))
    cl = c.EndpointClient(desc(stub), allowed_hosts=[stub.host], denied_addresses=["127.0.0.1"], sleep=lambda s: None)
    with pytest.raises(c.BlockedAddress):
        cl.ask("q")


# a run takes the rule from the platform, not from its own environment

def _platform(stub, monkeypatch, network):
    stub.route("/internal/projects/pid-1/connections/mcas-chat", (200, dict(PLAIN, base_url=stub.base, **network)))
    stub.route("/chat", (200, {"answer": "Up to 5,000 EUR."}))
    monkeypatch.setenv("PLATFORM_URL", stub.base)
    monkeypatch.setenv("PLATFORM_CONNECTIONS_TOKEN", "tok")


def test_the_workers_own_env_var_no_longer_opens_anything(plugin, stub, monkeypatch):
    _platform(stub, monkeypatch, {"allowed_hosts": [], "denied_addresses": []})
    monkeypatch.setenv("CONNECTIONS_ALLOWED_HOSTS", stub.host)
    cl = c.EndpointClient.for_input(plugin({"system": {"value": "connection:pid-1/mcas-chat"}}), "system",
                                    sleep=lambda s: None)
    with pytest.raises(c.BlockedAddress):
        cl.ask("How much?")


def test_the_platforms_list_opens_the_host_and_its_deny_list_holds(plugin, stub, monkeypatch):
    monkeypatch.delenv("CONNECTIONS_ALLOWED_HOSTS", raising=False)
    _platform(stub, monkeypatch, {"allowed_hosts": [stub.host], "denied_addresses": []})
    p = plugin({"system": {"value": "connection:pid-1/mcas-chat"}})
    assert c.EndpointClient.for_input(p, "system", sleep=lambda s: None).ask("How much?").text == "Up to 5,000 EUR."
    _platform(stub, monkeypatch, {"allowed_hosts": [stub.host], "denied_addresses": ["127.0.0.1"]})
    with pytest.raises(c.BlockedAddress):
        c.EndpointClient.for_input(p, "system", sleep=lambda s: None).ask("How much?")


def test_the_rule_is_not_part_of_the_descriptor(plugin, stub, monkeypatch):
    _platform(stub, monkeypatch, {"allowed_hosts": [stub.host], "denied_addresses": ["10.9.9.9"]})
    cl = c.EndpointClient.for_input(plugin({"system": {"value": "connection:pid-1/mcas-chat"}}), "system")
    assert "allowed_hosts" not in cl.descriptor.public() and "denied_addresses" not in cl.descriptor.public()


def test_a_refusal_says_where_to_allow_the_host(monkeypatch):
    resolving(monkeypatch, {"mcas.internal": "10.1.2.3"})
    with pytest.raises(c.BlockedAddress, match="Allowed internal hosts"):
        c.guard_url("http://mcas.internal:8500/chat", [])


# ── where a connection really goes: mapped addresses, rebinding names, proxies ──

@pytest.mark.parametrize("host", ["::ffff:127.0.0.1", "::ffff:169.254.169.254", "::ffff:7f00:1"])
def test_an_ipv4_mapped_address_is_checked_as_the_ipv4_address_it_maps(host):
    """[::ffff:169.254.169.254] was compared with the IPv4 denied list as an IPv6 address, never equal,
    so an allowed mapped literal reached metadata or a stack service."""
    denied = ["127.0.0.1", "169.254.169.254"]
    with pytest.raises(c.BlockedAddress):
        c.guard_url(f"http://[{host}]/x", allowed_hosts=[host], denied_addresses=denied)


def test_a_name_that_rebinds_after_the_check_is_refused_at_connect(stub, monkeypatch):
    """guard_url resolved the name, then urllib resolved it again to connect: a name answering a public
    address for the check and a denied one for the connection got through. The peer the socket
    actually reached is checked before anything is sent."""
    import socket as s

    real = s.getaddrinfo
    calls = []

    def rebinding(host, port, *a, **k):
        if host == "rebind.test":
            calls.append(host)
            ip = "93.184.216.34" if len(calls) == 1 else "127.0.0.1"
            return real(ip, port, *a, **k)
        return real(host, port, *a, **k)

    monkeypatch.setattr(s, "getaddrinfo", rebinding)
    stub.route("/chat", (200, {"answer": "metadata"}))
    d = desc(stub, base_url=f"http://rebind.test:{stub.port}")
    cl = c.EndpointClient(d, allowed_hosts=["rebind.test"], denied_addresses=["127.0.0.1"], sleep=lambda _s: None)
    with pytest.raises(c.BlockedAddress):
        cl.ask("q", lang="en")
    assert stub.seen == []


# Proxy settings do not switch the peer check off for direct connections

@pytest.mark.parametrize("env", [
    {"no_proxy": "internal.example"},                                       # NO_PROXY alone, common on corporate hosts
    {"http_proxy": "http://proxy.invalid:3128", "no_proxy": "rebind.test"},  # the system's host bypasses the proxy
])
def test_a_rebinding_name_is_refused_at_connect_whatever_the_proxy_variables(stub, monkeypatch, env):
    for name in ("http_proxy", "https_proxy", "all_proxy", "no_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    test_a_name_that_rebinds_after_the_check_is_refused_at_connect(stub, monkeypatch)


class _Sock:
    def __init__(self, ip):
        self.ip, self.closed = ip, False

    def getpeername(self):
        return (self.ip, 3128)

    def close(self):
        self.closed = True


class _Conn:
    def __init__(self, host, port, ip):
        self.host, self.port, self._tunnel_host, self.sock = host, port, None, _Sock(ip)


def test_a_connection_to_the_configured_proxy_is_not_checked_as_the_system(monkeypatch):
    """Through a proxy the socket reaches the proxy, often an internal address: that is not the system."""
    monkeypatch.setenv("http_proxy", "http://proxy.corp:3128")
    with c._Policy([], []):
        c._check_peer(_Conn("proxy.corp", 3128, "10.0.0.5"))


def test_a_direct_connection_with_a_proxy_configured_is_still_checked(monkeypatch):
    monkeypatch.setenv("http_proxy", "http://proxy.corp:3128")
    conn = _Conn("sut.example", 80, "10.0.0.5")
    with c._Policy([], []), pytest.raises(c.BlockedAddress):
        c._check_peer(conn)
    assert conn.sock.closed

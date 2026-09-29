"""EndpointClient: one way for every plugin to call a system under test registered under Manage →
Connections (connections plan 2026-09-29, tests I1 to I9)."""
import hashlib
import json

import pytest

from aisc_plugin_interface import connections as c

MCAS = {
    "name": "mcas-chat", "label": "MCAS chat", "kind": "rest", "base_url": "", "method": "POST",
    "path": "/chat", "headers": {"X-Client": "aisc"}, "secret_header": "Authorization: Bearer {{secret}}",
    "body_template": {"question": "{{input}}", "history": "{{history}}", "lang": "{{params.lang}}"},
    "response_path": "answer",
    "refusal": {"status": [502], "path": "detail.reason", "match": {"detail.error": "answer_not_grounded"}},
    "model": None, "timeout_s": 5, "secret": "s3cr3t-value", "updated_at": "2026-09-29T10:00:00Z",
}


def desc(stub, **over):
    d = dict(MCAS, base_url=stub.base)
    d.update(over)
    return c.Descriptor.from_dict(d)


def client(stub, plugin=None, **over):
    return c.EndpointClient(desc(stub, **over), plugin=plugin, allowed_hosts=[stub.host], sleep=lambda s: None)


# ── I1 reference parsing ────────────────────────────────────────────────────

def test_i1_a_reference_names_project_and_connection():
    assert c.parse_reference("connection:0f7c1e2a-aaaa-bbbb-cccc-1234567890ab/mcas-chat") == (
        "0f7c1e2a-aaaa-bbbb-cccc-1234567890ab", "mcas-chat")


@pytest.mark.parametrize("bad", ["", "mcas-chat", "connection:", "connection:/x", "connection:pid/",
                                 "connection:pid/bad name", "storage:pid/x", None, 3])
def test_i1_a_malformed_reference_is_refused(bad):
    with pytest.raises(c.BadReference):
        c.parse_reference(bad)


def test_i1_the_reference_is_read_from_a_resource_input(plugin, stub, monkeypatch):
    p = plugin({"system": {"value": "connection:pid-1/mcas-chat"}})
    assert c.reference_of(p, "system") == ("pid-1", "mcas-chat")
    with pytest.raises(c.BadReference):
        c.reference_of(plugin({}), "system")


# ── I2 resolving through the platform ───────────────────────────────────────

def test_i2_for_input_resolves_through_the_internal_route_with_the_token(plugin, stub, monkeypatch, tmp_path):
    stub.route("/internal/projects/pid-1/connections/mcas-chat", (200, dict(MCAS, base_url=stub.base)))
    monkeypatch.setenv("PLATFORM_URL", stub.base)
    monkeypatch.setenv("PLATFORM_CONNECTIONS_TOKEN", "tok-123")
    monkeypatch.setenv("CONNECTIONS_ALLOWED_HOSTS", stub.host)
    monkeypatch.chdir(tmp_path)
    cl = c.EndpointClient.for_input(plugin({"system": {"value": "connection:pid-1/mcas-chat"}}), "system")
    req = stub.seen[-1]
    assert req["method"] == "GET" and req["headers"]["x-aisc-service-token"] == "tok-123"
    assert cl.descriptor.name == "mcas-chat" and cl.descriptor.secret == "s3cr3t-value"
    assert list(tmp_path.iterdir()) == []                                  # nothing written to disk
    import os
    assert "s3cr3t-value" not in json.dumps(dict(os.environ))              # nor into the environment


def test_i2_without_platform_url_or_token_resolving_fails_clearly(plugin, monkeypatch):
    monkeypatch.delenv("PLATFORM_URL", raising=False)
    monkeypatch.delenv("PLATFORM_CONNECTIONS_TOKEN", raising=False)
    with pytest.raises(c.EndpointError, match="PLATFORM_CONNECTIONS_TOKEN|PLATFORM_URL"):
        c.EndpointClient.for_input(plugin({"system": {"value": "connection:pid-1/x"}}), "system")


def test_i2_a_refused_resolution_names_the_status(plugin, stub, monkeypatch):
    stub.route("/internal/projects/pid-1/connections/x", (403, {"detail": "no"}))
    monkeypatch.setenv("PLATFORM_URL", stub.base)
    monkeypatch.setenv("PLATFORM_CONNECTIONS_TOKEN", "t")
    with pytest.raises(c.EndpointError, match="403"):
        c.EndpointClient.for_input(plugin({"system": {"value": "connection:pid-1/x"}}), "system")


# ── I3 openai kind ──────────────────────────────────────────────────────────

def test_i3_openai_sends_model_and_messages_and_reads_the_choice(stub):
    stub.route("/v1/chat/completions", (200, {"choices": [{"message": {"content": "No, it is not."}}]}))
    cl = client(stub, kind="openai", base_url=stub.base + "/v1", path="", model="gpt-4o-mini",
                body_template=None, response_path=None, refusal=None, secret_header=None, headers={})
    hist = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    a = cl.ask("Is nationality an input?", history=hist, temperature=0)
    req = stub.seen[-1]
    body = json.loads(req["body"])
    assert req["path"] == "/v1/chat/completions" and req["headers"]["authorization"] == "Bearer s3cr3t-value"
    assert body == {"model": "gpt-4o-mini", "messages": hist + [{"role": "user", "content": "Is nationality an input?"}],
                    "temperature": 0}
    assert a.text == "No, it is not." and not a.refused and a.status == 200


# ── I4 rest kind ────────────────────────────────────────────────────────────

def test_i4_rest_renders_input_history_params_and_the_secret_header(stub):
    stub.route("/chat", (200, {"answer": "Yes (POL-FAIR-002)."}))
    hist = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]
    a = client(stub).ask('He said "hi"', history=hist, lang="en")
    req = stub.seen[-1]
    assert json.loads(req["body"]) == {"question": 'He said "hi"', "history": hist, "lang": "en"}
    assert req["headers"]["authorization"] == "Bearer s3cr3t-value" and req["headers"]["x-client"] == "aisc"
    assert a.text == "Yes (POL-FAIR-002)."


def test_i4_history_defaults_to_an_empty_list_and_an_unset_param_is_an_error(stub):
    stub.route("/chat", (200, {"answer": "ok"}))
    client(stub, body_template={"question": "{{input}}", "history": "{{history}}"}).ask("q")
    assert json.loads(stub.seen[-1]["body"])["history"] == []
    with pytest.raises(c.EndpointError, match="lang"):
        client(stub).ask("q")                                           # {{params.lang}} not given


@pytest.mark.parametrize("path,payload,expected", [
    ("answer", {"answer": "x"}, "x"),
    ("data.items[1].text", {"data": {"items": [{"text": "a"}, {"text": "b"}]}}, "b"),
    ("choices[0].message.content", {"choices": [{"message": {"content": "c"}}]}, "c"),
])
def test_i4_the_answer_path_is_dotted_with_indexes(path, payload, expected):
    assert c.extract(payload, path) == expected


def test_i4_a_missing_answer_path_is_a_bad_response(stub):
    stub.route("/chat", (200, {"reply": "x"}))
    with pytest.raises(c.EndpointBadResponse, match="answer"):
        client(stub).ask("q", lang="en")


# ── I5 refusals ─────────────────────────────────────────────────────────────

def test_i5_a_declared_refusal_is_an_answer_not_an_error(stub):
    stub.route("/chat", (502, {"detail": {"error": "answer_not_grounded", "reason": "no clause covers it"}}))
    a = client(stub).ask("Should I buy shares?", lang="en")
    assert a.refused and a.refusal_reason == "no clause covers it" and a.status == 502
    assert len(stub.seen) == 1                                          # a refusal is not retried


def test_i5_the_same_status_without_the_match_is_an_error_after_retries(stub):
    stub.route("/chat", (502, {"detail": {"error": "backend_down"}}))
    with pytest.raises(c.EndpointBadResponse, match="502"):
        client(stub).ask("q", lang="en")
    assert len(stub.seen) == 3                                          # first try + 2 retries


# ── I6 retries ──────────────────────────────────────────────────────────────

def test_i6_a_5xx_then_success_is_retried(stub):
    stub.route("/chat", (503, {"detail": "busy"}), (200, {"answer": "ok"}))
    assert client(stub).ask("q", lang="en").text == "ok" and len(stub.seen) == 2


def test_i6_the_backoff_is_one_then_three_seconds(stub):
    stub.route("/chat", (500, {}), (500, {}), (500, {}))
    waits = []
    cl = c.EndpointClient(desc(stub), allowed_hosts=[stub.host], sleep=waits.append)
    with pytest.raises(c.EndpointBadResponse):
        cl.ask("q", lang="en")
    assert waits == [1, 3]


def test_i6_a_4xx_is_never_retried(stub):
    stub.route("/chat", (422, {"detail": "bad"}))
    with pytest.raises(c.EndpointBadResponse, match="422"):
        client(stub).ask("q", lang="en")
    assert len(stub.seen) == 1


# ── I7 error classes ────────────────────────────────────────────────────────

@pytest.mark.parametrize("status,cls", [(401, c.EndpointAuthError), (403, c.EndpointAuthError), (404, c.EndpointNotFound)])
def test_i7_auth_and_not_found(stub, status, cls):
    stub.route("/chat", (status, {"detail": "x"}))
    with pytest.raises(cls):
        client(stub).ask("q", lang="en")


def test_i7_a_timeout_is_retried_then_reported(stub):
    stub.route("/chat", (200, {"answer": "late"}, 2))
    with pytest.raises(c.EndpointTimeout):
        client(stub, timeout_s=1).ask("q", lang="en")
    assert len(stub.seen) == 3


def test_i7_an_error_message_never_carries_the_secret(stub):
    stub.route("/chat", (401, {"detail": "bad key s3cr3t-value"}))
    with pytest.raises(c.EndpointAuthError) as err:
        client(stub).ask("q", lang="en")
    assert "s3cr3t-value" not in str(err.value)


# ── I8 provenance ───────────────────────────────────────────────────────────

def test_i8_the_first_call_uploads_what_was_assessed_once_without_the_secret(stub, plugin):
    stub.route("/chat", (200, {"answer": "ok"}))
    p = plugin()
    cl = client(stub, plugin=p)
    cl.ask("q1", lang="en")
    cl.ask("q2", lang="en")
    assert list(p.artifacts) == ["connection-mcas-chat.json"]
    doc = json.loads(p.artifacts["connection-mcas-chat.json"])
    assert "s3cr3t-value" not in json.dumps(doc) and "secret" not in doc["connection"]
    public = dict(MCAS, base_url=stub.base, protocol_version=None)
    del public["secret"]
    assert doc["connection"] == public
    assert doc["sha256"] == hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest()
    assert doc["first_call_at"]


# ── I9 outbound safety ──────────────────────────────────────────────────────

@pytest.mark.parametrize("url", [
    "http://127.0.0.1:9/x", "http://localhost/x", "http://10.1.2.3/x", "http://172.16.0.5/x",
    "http://192.168.1.10/x", "http://169.254.169.254/latest/meta-data", "http://[::1]/x",
    "http://[fd00::1]/x", "http://0.0.0.0/x", "file:///etc/passwd", "ftp://example.com/x",
])
def test_i9_internal_addresses_and_other_schemes_are_refused(url):
    with pytest.raises(c.BlockedAddress):
        c.guard_url(url, allowed_hosts=[])


def test_i9_a_hostname_resolving_to_a_private_address_is_refused(monkeypatch):
    monkeypatch.setattr(c.socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("10.0.0.7", 80))])
    with pytest.raises(c.BlockedAddress):
        c.guard_url("http://internal.example.com/x", allowed_hosts=[])


def test_i9_a_public_address_passes(monkeypatch):
    monkeypatch.setattr(c.socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 443))])
    c.guard_url("https://api.example.com/v1", allowed_hosts=[])


def test_i9_an_allowlisted_host_passes(stub):
    c.guard_url(stub.base + "/chat", allowed_hosts=[stub.host])
    c.guard_url("http://host.docker.internal:8500/chat", allowed_hosts=["host.docker.internal:8500"])


def test_i9_a_redirect_is_refused_not_followed(stub):
    stub.route("/chat", (302, "http://169.254.169.254/latest", 0))
    with pytest.raises(c.EndpointBadResponse, match="redirect"):
        client(stub).ask("q", lang="en")
    assert len(stub.seen) == 1


def test_i9_the_guard_runs_on_every_call(stub):
    cl = c.EndpointClient(desc(stub), allowed_hosts=[], sleep=lambda s: None)
    with pytest.raises(c.BlockedAddress):
        cl.ask("q", lang="en")
    assert stub.seen == []


def test_i6_retries_can_be_switched_off_for_a_single_probe(stub):
    stub.route("/chat", (503, {}), (200, {"answer": "ok"}))
    with pytest.raises(c.EndpointBadResponse, match="503"):
        c.call(desc(stub), "q", params={"lang": "en"}, allowed_hosts=[stub.host], waits=())
    assert len(stub.seen) == 1

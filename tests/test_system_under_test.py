"""@system_under_test: a plugin declares which protocol(s) its tool speaks and which of its config
fields or environment variables take the endpoint; with a connection bound, a run gets a key and the
tool is pointed at the platform's endpoint for that protocol."""
import json
import os

import pytest

from aisc_plugin_interface import BaseEvaluationPlugin, InputType
from aisc_plugin_interface.system_under_test import SYSTEM_INPUT, system_under_test

PID = "0f7c1e2a-aaaa-bbbb-cccc-1234567890ab"
ROOT = f"http://platform:8000/internal/facade/{PID}/mcas-chat"
ISSUED = {"key": "aisc-run-abc", "expires_at": "2026-09-30T00:00:00+00:00", "endpoints": {
    "aisc": {"ask_url": f"{ROOT}/aisc/ask"},
    "openai": {"base_url": f"{ROOT}/openai/v1", "model": "mcas-chat"},
    "a2a": {"agent_card_url": f"{ROOT}/a2a/.well-known/agent-card.json", "rpc_url": f"{ROOT}/a2a"},
    "oip": {"base_url": f"{ROOT}/oip", "model": "mcas-chat"}}}


@pytest.fixture
def platform(stub, monkeypatch):
    stub.route(f"/internal/projects/{PID}/connections/mcas-chat/run-keys", (201, ISSUED))
    monkeypatch.setenv("PLATFORM_URL", stub.base)
    monkeypatch.setenv("PLATFORM_CONNECTIONS_TOKEN", "svc-token")
    return stub


def make(**declared):
    @system_under_test(**declared)
    class Tool(BaseEvaluationPlugin):
        def evaluate(self, config_data):
            self.seen = {"config": config_data, "env": {k: os.environ.get(k) for k in ("OPENAI_BASE_URL", "OPENAI_API_KEY")}}
            return "done"

        def export_metrics(self, *a, **k):
            return []

    return Tool


def bound(tool_cls, value=f"connection:{PID}/mcas-chat"):
    t = tool_cls()
    t._set_artifact_callback(lambda name, content: t.__dict__.setdefault("artifacts", {}).__setitem__(name, content))
    if value is not None:
        t.set_input_content(SYSTEM_INPUT, json.dumps({"value": value}).encode())
    return t


# the declaration

def test_s1_adds_the_system_resource_input_and_records_the_protocols():
    Tool = make(protocols=("openai", "a2a"), fields={"target.base_url": "base_url"})
    (d,) = [d for d in Tool().input_definitions if d.name == SYSTEM_INPUT]
    # declared optional: the Configurator engine makes it required
    assert d.input_type == InputType.RESOURCE and d.required is False
    assert Tool.system_under_test_protocols == ("openai", "a2a")


def test_s1_the_input_can_be_optional_and_renamed():
    Tool = make(protocols=("aisc",), fields={"url": "ask_url"}, required=False, input_name="target", label="Target")
    (d,) = [d for d in Tool().input_definitions if d.name == "target"]
    assert d.required is False and d.label == "Target"


@pytest.mark.parametrize("bad", [
    dict(protocols=(), fields={"u": "base_url"}),
    dict(protocols=("grpc",), fields={"u": "base_url"}),
    dict(protocols=("openai",)),
    dict(protocols=("openai",), fields={"u": "rpc_url"}),          # not a role of the first protocol
    dict(protocols=("a2a",), env={"X": "model"}),
])
def test_s1_a_bad_declaration_is_refused_when_the_class_is_made(bad):
    with pytest.raises(ValueError):
        make(**bad)


# a run with a bound connection

def test_s2_fills_the_declared_fields_for_the_first_protocol(platform):
    t = bound(make(protocols=("openai",), fields={"target.base_url": "base_url", "target.api_key": "api_key",
                                                   "target.model": "model"}))
    assert t.evaluate({"target": {"temperature": 0}, "other": 1}) == "done"
    assert t.seen["config"] == {"target": {"temperature": 0, "base_url": f"{ROOT}/openai/v1", "api_key": "aisc-run-abc",
                                           "model": "mcas-chat"}, "other": 1}


def test_s2_the_callers_config_is_not_changed(platform):
    t = bound(make(protocols=("openai",), fields={"base": "base_url"}))
    config = {"x": 1}
    t.evaluate(config)
    assert config == {"x": 1}


def test_s3_issues_the_key_with_the_service_token(platform):
    t = bound(make(protocols=("a2a",), fields={"card": "agent_card_url", "rpc": "rpc_url", "key": "api_key"}))
    t.evaluate({})
    (req,) = [r for r in platform.seen if r["path"].endswith("/run-keys")]
    assert req["method"] == "POST" and req["headers"]["x-aisc-service-token"] == "svc-token"
    assert t.seen["config"] == {"card": f"{ROOT}/a2a/.well-known/agent-card.json", "rpc": f"{ROOT}/a2a", "key": "aisc-run-abc"}


def test_s4_sets_the_declared_env_for_the_run_only(platform, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    t = bound(make(protocols=("openai",), env={"OPENAI_BASE_URL": "base_url", "OPENAI_API_KEY": "api_key"}))
    t.evaluate({})
    assert t.seen["env"] == {"OPENAI_BASE_URL": f"{ROOT}/openai/v1", "OPENAI_API_KEY": "aisc-run-abc"}
    assert os.environ["OPENAI_BASE_URL"] == "https://api.openai.com/v1" and "OPENAI_API_KEY" not in os.environ


def test_s4_the_env_is_restored_when_the_tool_fails(platform, monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    @system_under_test(protocols=("openai",), env={"OPENAI_BASE_URL": "base_url"})
    class Tool(BaseEvaluationPlugin):
        def evaluate(self, config_data):
            raise RuntimeError("tool failed")

    with pytest.raises(RuntimeError):
        bound(Tool).evaluate({})
    assert "OPENAI_BASE_URL" not in os.environ


def test_s5_records_what_the_run_was_pointed_at_without_the_key(platform):
    t = bound(make(protocols=("oip",), fields={"url": "base_url", "m": "model"}))
    t.evaluate({})
    doc = json.loads(t.artifacts["connection-mcas-chat.json"])
    assert doc["connection"] == f"connection:{PID}/mcas-chat" and doc["protocol"] == "oip"
    assert doc["endpoint"] == {"base_url": f"{ROOT}/oip", "model": "mcas-chat"} and doc["expires_at"]
    assert "aisc-run-abc" not in json.dumps(doc)


# no connection, and failures

def test_s6_an_optional_input_left_unbound_runs_the_tool_unchanged(platform):
    t = bound(make(protocols=("openai",), fields={"base": "base_url"}, required=False), value=None)
    t.evaluate({"base": "https://example.test/v1"})
    assert t.seen["config"] == {"base": "https://example.test/v1"}
    assert not [r for r in platform.seen if r["path"].endswith("/run-keys")]


def test_s7_a_required_input_left_unbound_stops_the_run(platform):
    t = bound(make(protocols=("openai",), fields={"base": "base_url"}), value=None)
    with pytest.raises(ValueError, match="system under test"):
        t.evaluate({})


def test_s8_a_refused_key_stops_the_run_before_the_tool(stub, monkeypatch):
    stub.route(f"/internal/projects/{PID}/connections/mcas-chat/run-keys", (404, {"detail": "no connection"}))
    monkeypatch.setenv("PLATFORM_URL", stub.base)
    monkeypatch.setenv("PLATFORM_CONNECTIONS_TOKEN", "svc-token")
    t = bound(make(protocols=("openai",), fields={"base": "base_url"}))
    with pytest.raises(RuntimeError, match="mcas-chat"):
        t.evaluate({})
    assert not hasattr(t, "seen")


def test_s9_without_the_platform_settings_the_run_stops_clearly(monkeypatch):
    monkeypatch.delenv("PLATFORM_URL", raising=False)
    t = bound(make(protocols=("openai",), fields={"base": "base_url"}))
    with pytest.raises(RuntimeError, match="PLATFORM_URL"):
        t.evaluate({})


def test_the_run_key_request_follows_no_redirect(stub, monkeypatch):
    """It used urlopen, which follows a 30x and sends the service token on to wherever it points
    (code review 2026-10-05)."""
    from aisc_plugin_interface import system_under_test as sut

    stub.route("/internal/projects/p1/targets/k/run-keys", (302, f"{stub.base}/elsewhere"))
    stub.route("/elsewhere", (200, {"run_key": "leaked"}))
    monkeypatch.setenv("PLATFORM_URL", stub.base)
    monkeypatch.setenv("PLATFORM_CONNECTIONS_TOKEN", "svc-token")
    with pytest.raises(RuntimeError):
        sut._issue_run_key("p1", "targets/k", "k")
    assert [r["path"] for r in stub.seen] == ["/internal/projects/p1/targets/k/run-keys"]


def test_only_the_no_endpoint_answer_is_no_endpoint(stub, monkeypatch):
    from aisc_plugin_interface import system_under_test as sut

    monkeypatch.setenv("PLATFORM_URL", stub.base)
    monkeypatch.setenv("PLATFORM_CONNECTIONS_TOKEN", "svc-token")
    stub.route("/internal/projects/p1/targets/k/run-keys", (404, {"detail": "no project 'p1'"}))
    with pytest.raises(RuntimeError) as exc:
        sut._issue_run_key("p1", "targets/k", "k")
    assert not isinstance(exc.value, sut.NoEndpoint)
    stub.route("/internal/projects/p1/targets/k/run-keys", (404, {"detail": "x", "reason": "no_endpoint"}))
    with pytest.raises(sut.NoEndpoint):
        sut._issue_run_key("p1", "targets/k", "k")

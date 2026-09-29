"""A plugin names what it assesses (targets plan v2, PI1 to PI4): the evaluation's `target` input
is `target:<platform pid>/<key>`, the system or one component of its AI card. A plugin whose tool
calls a system reaches the endpoint of that target through the platform, and its record says both
what it assessed and what it called."""
import json

import pytest

from aisc_plugin_interface import BaseEvaluationPlugin
from aisc_plugin_interface import connections as c
from aisc_plugin_interface.system_under_test import SYSTEM_INPUT, system_under_test
from aisc_plugin_interface.targets import TARGET_INPUT, parse_target, target_of

PID = "0f7c1e2a-aaaa-bbbb-cccc-1234567890ab"
KEY = "component:0b9c7a1e-0000-4000-8000-000000000001"
ROOT = f"http://platform:8000/internal/facade/{PID}/mcas-scorer"
TARGET = {"key": KEY, "kind": "component", "component_kind": "model", "label": "Scoring model", "last_card_number": 2}
ISSUED = {"key": "aisc-run-abc", "expires_at": "2026-09-30T00:00:00+00:00", "connection": "mcas-scorer",
          "target": TARGET, "endpoints": {"openai": {"base_url": f"{ROOT}/openai/v1", "model": "mcas-scorer"}}}


def test_pi4_the_input_is_named_target_and_its_value_names_project_and_key():
    assert TARGET_INPUT == SYSTEM_INPUT == "target"
    assert parse_target(f"target:{PID}/{KEY}") == (PID, KEY)
    assert parse_target(f"target:{PID}/system") == (PID, "system")
    for bad in ("", "connection:x/y", f"target:{PID}/", f"target:{PID}/component:nope", None):
        with pytest.raises(c.BadReference):
            parse_target(bad)


def test_pi4_target_of_a_plugin(plugin):
    assert target_of(plugin({"target": {"value": f"target:{PID}/{KEY}"}})) == {"pid": PID, "key": KEY}
    assert target_of(plugin({})) is None


@pytest.fixture
def platform(stub, monkeypatch):
    stub.route(f"/internal/projects/{PID}/targets/{KEY}/run-keys", (201, ISSUED))
    monkeypatch.setenv("PLATFORM_URL", stub.base)
    monkeypatch.setenv("PLATFORM_CONNECTIONS_TOKEN", "svc-token")
    return stub


def tool():
    @system_under_test(protocols=("openai",), fields={"target.base_url": "base_url", "target.api_key": "api_key"})
    class Tool(BaseEvaluationPlugin):
        def evaluate(self, config_data):
            self.seen = config_data
            return "done"

    return Tool


def bound(cls, value):
    t = cls()
    t._set_artifact_callback(lambda name, content: t.__dict__.setdefault("artifacts", {}).__setitem__(name, content))
    t.set_input_content("target", json.dumps({"value": value}).encode())
    return t


def test_pi1_the_decorator_declares_target_and_reaches_the_targets_endpoint(platform):
    Tool = tool()
    assert [d.name for d in Tool().input_definitions] == ["target"]
    t = bound(Tool, f"target:{PID}/{KEY}")
    assert t.evaluate({}) == "done"
    assert t.seen["target"] == {"base_url": f"{ROOT}/openai/v1", "api_key": "aisc-run-abc"}
    (req,) = [r for r in platform.seen if r["path"].endswith("/run-keys")]
    assert req["path"] == f"/internal/projects/{PID}/targets/{KEY}/run-keys"


def test_pi2_a_target_without_an_endpoint_stops_the_run_before_the_tool_saying_so(stub, monkeypatch):
    stub.route(f"/internal/projects/{PID}/targets/{KEY}/run-keys",
               (404, {"detail": "Scoring model has no endpoint: set one under Manage, Targets and endpoints"}))
    monkeypatch.setenv("PLATFORM_URL", stub.base)
    monkeypatch.setenv("PLATFORM_CONNECTIONS_TOKEN", "svc-token")
    t = bound(tool(), f"target:{PID}/{KEY}")
    with pytest.raises(RuntimeError, match="Scoring model has no endpoint"):
        t.evaluate({})
    assert not hasattr(t, "seen")


def test_pi3_the_record_says_what_was_assessed_and_what_was_called(platform):
    t = bound(tool(), f"target:{PID}/{KEY}")
    t.evaluate({})
    doc = json.loads(t.artifacts["connection-mcas-scorer.json"])
    assert doc["target"] == TARGET and doc["connection"] == f"connection:{PID}/mcas-scorer"
    assert "aisc-run-abc" not in json.dumps(doc)


def test_pi1_a_legacy_connection_value_still_works(stub, monkeypatch):
    stub.route(f"/internal/projects/{PID}/connections/mcas-chat/run-keys", (201, {**ISSUED, "connection": "mcas-chat"}))
    monkeypatch.setenv("PLATFORM_URL", stub.base)
    monkeypatch.setenv("PLATFORM_CONNECTIONS_TOKEN", "svc-token")
    t = bound(tool(), f"connection:{PID}/mcas-chat")
    assert t.evaluate({}) == "done"


def test_pi1_the_client_resolves_a_targets_endpoint(plugin, stub, monkeypatch):
    from tests.test_connections import MCAS
    stub.route(f"/internal/projects/{PID}/targets/{KEY}/connection",
               (200, dict(MCAS, base_url=stub.base, target=TARGET, allowed_hosts=[stub.host], denied_addresses=[])))
    stub.route("/chat", (200, {"answer": "Approve"}))
    monkeypatch.setenv("PLATFORM_URL", stub.base)
    monkeypatch.setenv("PLATFORM_CONNECTIONS_TOKEN", "svc-token")
    cl = c.EndpointClient.for_target(plugin({"target": {"value": f"target:{PID}/{KEY}"}}), sleep=lambda s: None)
    assert cl.descriptor.name == "mcas-chat" and cl.descriptor.target == TARGET
    assert cl.ask("q", lang="en").text == "Approve"


def test_pi3_the_target_is_part_of_what_the_connection_is():
    from tests.test_connections import MCAS
    a = c.Descriptor.from_dict({**MCAS, "base_url": "http://x"})
    b = c.Descriptor.from_dict({**MCAS, "base_url": "http://x", "target": TARGET})
    assert b.public()["target"] == TARGET and a.fingerprint() != b.fingerprint()


def test_pi5_an_optional_endpoint_lets_the_tool_run_as_configured_when_the_target_has_none(stub, monkeypatch):
    stub.route(f"/internal/projects/{PID}/targets/{KEY}/run-keys", (404, {"detail": "Scoring model has no endpoint"}))
    monkeypatch.setenv("PLATFORM_URL", stub.base)
    monkeypatch.setenv("PLATFORM_CONNECTIONS_TOKEN", "svc-token")

    @system_under_test(protocols=("openai",), fields={"target.base_url": "base_url"}, required=False)
    class Tool(BaseEvaluationPlugin):
        def evaluate(self, config_data):
            self.seen = config_data
            return "done"

    t = bound(Tool, f"target:{PID}/{KEY}")
    assert t.evaluate({"x": 1}) == "done" and t.seen == {"x": 1}


def test_pi5_the_declared_input_is_optional_so_standalone_forms_are_not_blocked():
    # the engine makes it required in configurator mode (O2); the plugin itself stays usable without targets
    assert [(d.name, d.required) for d in tool()().input_definitions] == [("target", False)]

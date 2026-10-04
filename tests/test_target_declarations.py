"""How a plugin gets what it assesses (docs/superpowers/target-standard-2026-10-04/01-plan.md, R1, R4):
every plugin declares it, either @system_under_test (a live target through its endpoint) or
@assesses_inputs (only its inputs); a plugin that needs more of the endpoint than text (tool calls,
logprobs) says so with needs=, and a run is refused at the start when the endpoint doesn't carry it."""
import json
import os

import pytest

from aisc_plugin_interface import BaseEvaluationPlugin
from aisc_plugin_interface.system_under_test import (SYSTEM_INPUT, assesses_inputs, system_under_test,
                                                     target_access_of)

PID = "0f7c1e2a-aaaa-bbbb-cccc-1234567890ab"
ROOT = f"http://platform:8000/internal/facade/{PID}/mcas-chat"
ENDPOINTS = {"openai": {"base_url": f"{ROOT}/openai/v1", "model": "mcas-chat"}, "aisc": {"ask_url": f"{ROOT}/aisc/ask"}}


class Base(BaseEvaluationPlugin):
    def evaluate(self, config_data):
        self.ran = {"env": os.environ.get("AISC_TARGET_BASE_URL")}
        return "done"

    def export_metrics(self, *a, **k):
        return []


def run(cls, issued, stub, monkeypatch):
    stub.route(f"/internal/projects/{PID}/targets/system/run-keys", (201, issued))
    monkeypatch.setenv("PLATFORM_URL", stub.base)
    monkeypatch.setenv("PLATFORM_CONNECTIONS_TOKEN", "svc-token")
    t = cls()
    t._set_artifact_callback(lambda name, content: None)
    t.set_input_content(SYSTEM_INPUT, json.dumps({"value": f"target:{PID}/system"}).encode())
    return t


def test_r1_each_declaration_says_how_the_plugin_gets_what_it_assesses():
    @assesses_inputs()
    class Files(Base):
        pass

    @system_under_test(protocols=("openai",), env={"AISC_TARGET_BASE_URL": "base_url"})
    class Live(Base):
        pass

    class Undeclared(Base):
        pass

    assert target_access_of(Files) == "inputs"
    assert target_access_of(Live) == "endpoint"
    assert target_access_of(Undeclared) is None


def test_r1_inputs_only_changes_nothing_at_run_time():
    @assesses_inputs()
    class Files(Base):
        pass

    t = Files()
    assert t.evaluate({}) == "done" and not [d for d in t.input_definitions if d.name == SYSTEM_INPUT]


def test_r1_a_class_cannot_declare_both():
    with pytest.raises(ValueError):
        @assesses_inputs()
        @system_under_test(protocols=("openai",), env={"AISC_TARGET_BASE_URL": "base_url"})
        class Both(Base):
            pass


def test_r4_needs_names_known_capabilities_only():
    with pytest.raises(ValueError):
        @system_under_test(protocols=("openai",), env={"AISC_TARGET_BASE_URL": "base_url"}, needs=("telepathy",))
        class Bad(Base):
            pass


def test_r4_a_need_the_endpoint_does_not_carry_refuses_the_run_before_it_starts(stub, monkeypatch):
    @system_under_test(protocols=("openai",), env={"AISC_TARGET_BASE_URL": "base_url"}, needs=("tools",))
    class Agent(Base):
        pass

    t = run(Agent, {"key": "k", "endpoints": ENDPOINTS, "connection": "mcas-chat",
                    "capabilities": ["text", "history"]}, stub, monkeypatch)
    with pytest.raises(ValueError) as exc:
        t.evaluate({})
    assert "tools" in str(exc.value) and "mcas-chat" in str(exc.value)
    assert not hasattr(t, "ran")                                    # the tool never started


def test_r4_without_a_capability_list_the_endpoint_carries_text_and_history_only(stub, monkeypatch):
    @system_under_test(protocols=("openai",), env={"AISC_TARGET_BASE_URL": "base_url"}, needs=("logprobs",))
    class Scorer(Base):
        pass

    t = run(Scorer, {"key": "k", "endpoints": ENDPOINTS, "connection": "mcas-chat"}, stub, monkeypatch)
    with pytest.raises(ValueError):
        t.evaluate({})


def test_r4_what_is_carried_runs(stub, monkeypatch):
    @system_under_test(protocols=("openai",), env={"AISC_TARGET_BASE_URL": "base_url"}, needs=("history",))
    class Chat(Base):
        pass

    t = run(Chat, {"key": "k", "endpoints": ENDPOINTS, "connection": "mcas-chat"}, stub, monkeypatch)
    assert t.evaluate({}) == "done" and t.ran["env"] == f"{ROOT}/openai/v1"
    assert Chat.system_under_test_needs == ("history",)

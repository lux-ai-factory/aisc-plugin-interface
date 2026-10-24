"""@helper_model: a plugin declares the models it uses as tools (a judge, an attacker, a grader, an
embedder, a generator), apart from what it assesses. Each role is an LLM input named "helper:<role>"
that the engine fills like any LLM input (endpoint, model, project secret); the plugin reads it with
self.helper(role), the key coming from the secret's environment variable, never from the input."""
import json

import pytest

from aisc_plugin_interface import (
    HELPER_ROLES,
    BaseEvaluationPlugin,
    HelperModel,
    InputType,
    helper_model,
    helper_roles_of,
)

JUDGE_INPUT = {"endpoint_url": "https://llm.example/v1", "secret_key": "judge_key", "model": "gpt-judge"}


class Base(BaseEvaluationPlugin):
    def evaluate(self, config_data):
        return None

    def export_metrics(self, *a, **k):
        return []


@helper_model("judge", "Judge model", required=True)
@helper_model("attacker", "Attacker model")
class RedTeam(Base):
    pass


def bound(plugin, role, payload):
    plugin.set_input_content(f"helper:{role}", json.dumps(payload).encode())
    return plugin


def test_each_role_is_an_llm_input_named_after_it_with_its_label():
    inputs = {d.name: d for d in RedTeam().input_definitions}
    assert inputs["helper:judge"].input_type == InputType.LLM
    assert inputs["helper:judge"].label == "Judge model"
    assert inputs["helper:judge"].required is True
    assert inputs["helper:attacker"].required is False


def test_the_roles_are_recorded_on_the_class():
    assert set(helper_roles_of(RedTeam)) == {"judge", "attacker"}
    assert helper_roles_of(Base) == ()


def test_the_known_roles():
    assert HELPER_ROLES == ("judge", "attacker", "grader", "embedder", "generator")


def test_an_unknown_role_is_refused():
    with pytest.raises(ValueError, match="critic"):
        helper_model("critic", "Critic")


def test_a_bound_helper_gives_endpoint_model_and_the_key_from_the_secret_environment(monkeypatch):
    monkeypatch.setenv("AISC_SECRET_JUDGE_KEY", "sk-judge")
    judge = bound(RedTeam(), "judge", JUDGE_INPUT).helper("judge")
    assert judge == HelperModel(role="judge", endpoint_url="https://llm.example/v1", model="gpt-judge",
                                api_key="sk-judge")


def test_the_key_is_never_read_from_the_input(monkeypatch):
    monkeypatch.delenv("AISC_SECRET_JUDGE_KEY", raising=False)
    judge = bound(RedTeam(), "judge", JUDGE_INPUT).helper("judge")
    assert judge.api_key is None


def test_the_key_is_not_shown_when_the_helper_is_printed(monkeypatch):
    monkeypatch.setenv("AISC_SECRET_JUDGE_KEY", "sk-judge")
    judge = bound(RedTeam(), "judge", JUDGE_INPUT).helper("judge")
    assert "sk-judge" not in repr(judge)
    assert "sk-judge" not in str(judge)
    assert "gpt-judge" in repr(judge)


def test_a_helper_is_read_only_once_resolved(monkeypatch):
    monkeypatch.setenv("AISC_SECRET_JUDGE_KEY", "sk-judge")
    judge = bound(RedTeam(), "judge", JUDGE_INPUT).helper("judge")
    with pytest.raises(AttributeError):
        judge.model = "other"


def test_an_optional_helper_left_unbound_is_none():
    assert RedTeam().helper("attacker") is None


def test_a_required_helper_left_unbound_raises_naming_the_role():
    with pytest.raises(ValueError, match="judge"):
        RedTeam().helper("judge")


def test_a_role_the_plugin_did_not_declare_raises_naming_it():
    with pytest.raises(ValueError, match="grader"):
        RedTeam().helper("grader")


@helper_model("grader", "Grader model")
class Graded(RedTeam):
    pass


def test_a_subclass_keeps_its_base_roles_and_the_base_is_not_changed():
    assert set(helper_roles_of(Graded)) == {"judge", "attacker", "grader"}
    assert set(helper_roles_of(RedTeam)) == {"judge", "attacker"}
    assert "helper:grader" not in [d.name for d in RedTeam().input_definitions]
    assert Graded().helper("grader") is None

"""describe(cls): what a plugin class declares, as plain JSON values, so the engine can store it once per
package version instead of importing the plugin each time it needs to know. One test per decorator."""
import json

from aisc_plugin_interface import (
    BaseEvaluationPlugin,
    ConfigCategory,
    ConfigValueType,
    InputType,
    OnnxInputProvider,
    assesses,
    describe,
    evaluation_input,
    helper_model,
    inputs_access,
    metric,
    model_session,
    onnx,
    project_config,
    system_under_test_access,
)
from aisc_plugin_interface.system_under_test import (
    assesses_inputs,
    dataset_through_target,
    system_under_test,
)


class Base(BaseEvaluationPlugin):
    def evaluate(self, config_data):
        return None


TARGET = {"name": "target", "label": "Target of the assessment (the system, or one of its components)",
          "input_type": "resource", "required": False}


def test_a_class_that_declares_nothing_describes_as_empty():
    class Bare(Base):
        pass

    assert describe(Bare) == {
        "target_kinds": [], "card_access": None, "card_protocols": [], "card_needs_connection": False,
        "local_model_input": None, "helper_roles": [], "inputs": [], "project_configs": [], "metrics": [],
    }


def test_system_under_test_gives_the_card_kind_its_protocols_and_the_target_input():
    @system_under_test(("openai",), env={"OPENAI_BASE_URL": "base_url"})
    class Live(Base):
        pass

    d = describe(Live)
    assert (d["target_kinds"], d["card_access"], d["card_protocols"], d["card_needs_connection"]) == (
        ["card"], "endpoint", ["openai"], True)
    assert d["inputs"] == [TARGET]


def test_dataset_through_target_gives_its_access_and_a_connection_only_when_required():
    @dataset_through_target(("rows",), required=False)
    class Tables(Base):
        pass

    d = describe(Tables)
    assert (d["target_kinds"], d["card_access"], d["card_protocols"], d["card_needs_connection"]) == (
        ["card"], "dataset_through_target", [], False)


def test_assesses_inputs_needs_no_connection():
    @assesses_inputs()
    class Files(Base):
        pass

    d = describe(Files)
    assert (d["target_kinds"], d["card_access"], d["card_needs_connection"], d["inputs"]) == (
        ["card"], "inputs", False, [])


def test_assesses_with_a_card_and_a_local_model_gives_both_kinds_and_the_model_input():
    @assesses(card=system_under_test_access(protocols=("openai",), env={"OPENAI_BASE_URL": "base_url"}),
              local_model=onnx(input="model"))
    @evaluation_input(name="model", label="Model", input_provider_class=OnnxInputProvider,
                      input_type=InputType.MODEL)
    class Both(Base):
        pass

    d = describe(Both)
    assert (d["target_kinds"], d["card_access"], d["card_protocols"], d["local_model_input"]) == (
        ["card", "local_model"], "endpoint", ["openai"], "model")
    assert d["card_needs_connection"] is True


def test_assesses_with_model_session_gives_that_access():
    @assesses(card=model_session(outputs={"label": "label", "probability": "score"}))
    class Scorer(Base):
        pass

    d = describe(Scorer)
    assert (d["target_kinds"], d["card_access"], d["card_needs_connection"]) == (["card"], "model_session", True)


def test_assesses_with_inputs_access_alone():
    @assesses(card=inputs_access())
    class Files(Base):
        pass

    assert describe(Files)["card_access"] == "inputs"


def test_evaluation_input_lists_each_input_in_declaration_order():
    @evaluation_input(name="model", label="Model", input_provider_class=OnnxInputProvider,
                      input_type=InputType.MODEL)
    @evaluation_input(name="judge", label="Judge", input_type=InputType.LLM, required=False)
    class Inputs(Base):
        pass

    assert describe(Inputs)["inputs"] == [
        {"name": "judge", "label": "Judge", "input_type": "llm", "required": False},
        {"name": "model", "label": "Model", "input_type": "model", "required": True},
    ]


def test_project_config_lists_each_setting():
    @project_config(key="api_key", name="API key", category=ConfigCategory.SECRETS)
    @project_config(key="rounds", name="Rounds", category=ConfigCategory.VARIABLES,
                    value_type=ConfigValueType.NUMBER, required=False)
    class Settings(Base):
        pass

    assert describe(Settings)["project_configs"] == [
        {"key": "rounds", "name": "Rounds", "category": "variables", "value_type": "number", "required": False},
        {"key": "api_key", "name": "API key", "category": "secrets", "value_type": None, "required": True},
    ]


def test_helper_model_gives_its_role_and_its_input():
    @helper_model("judge", label="Judge model")
    class Judged(Base):
        pass

    d = describe(Judged)
    assert d["helper_roles"] == ["judge"]
    assert [i["input_type"] for i in d["inputs"]] == ["llm"]


def test_metric_lists_the_metric_names():
    class Scored(Base):
        @metric("accuracy")
        def accuracy(self, *a, **k):
            return []

        @metric("bias")
        def bias(self, *a, **k):
            return []

    assert describe(Scored)["metrics"] == ["accuracy", "bias"]


def test_the_description_is_plain_json():
    @assesses(card=system_under_test_access(protocols=("openai",), env={"OPENAI_BASE_URL": "base_url"}))
    @project_config(key="api_key", name="API key", category=ConfigCategory.SECRETS)
    class Live(Base):
        pass

    d = describe(Live)
    assert json.loads(json.dumps(d)) == d

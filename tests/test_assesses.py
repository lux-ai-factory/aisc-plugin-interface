"""@assesses: a plugin states what it assesses, an AI card target (reached the way a card builder says),
a local model file (an uploaded ONNX it reads through one of its model inputs), or both; the engine
builds the evaluation form from it, so a kind the plugin did not declare cannot be picked."""
import json
import os

import pytest

from aisc_plugin_interface import (
    BaseEvaluationPlugin,
    InputType,
    OnnxInputProvider,
    assesses,
    card_needs_connection,
    card_protocols_of,
    dataset_through_target_access,
    evaluation_input,
    inputs_access,
    model_session,
    onnx,
    system_under_test_access,
    target_kinds_of,
)
from aisc_plugin_interface.system_under_test import (SYSTEM_INPUT, assesses_inputs, dataset_through_target,
                                                     system_under_test, target_access_of)

PID = "0f7c1e2a-aaaa-bbbb-cccc-1234567890ab"
ROOT = f"http://platform:8000/internal/facade/{PID}/mcas-chat"
ISSUED = {"key": "aisc-run-abc", "expires_at": "2026-09-30T00:00:00+00:00",
          "endpoints": {"openai": {"base_url": f"{ROOT}/openai/v1", "model": "mcas-chat"}}}
ENV = {"OPENAI_BASE_URL": "base_url"}


class Base(BaseEvaluationPlugin):
    def evaluate(self, config_data):
        self.seen = os.environ.get("OPENAI_BASE_URL")
        return "done"

    def export_metrics(self, *a, **k):
        return []


def model_input(cls):
    return evaluation_input(name="model", label="Model", input_provider_class=OnnxInputProvider,
                            input_type=InputType.MODEL)(cls)


def bound(cls, value=None):
    t = cls()
    t._set_artifact_callback(lambda name, content: None)
    if value is not None:
        t.set_input_content(SYSTEM_INPUT, json.dumps({"value": value}).encode())
    return t


@pytest.fixture
def platform(stub, monkeypatch):
    stub.route(f"/internal/projects/{PID}/connections/mcas-chat/run-keys", (201, ISSUED))
    monkeypatch.setenv("PLATFORM_URL", stub.base)
    monkeypatch.setenv("PLATFORM_CONNECTIONS_TOKEN", "svc-token")
    return stub


# what is assessed

def test_a_card_builder_records_the_card_kind_and_its_access():
    @assesses(card=system_under_test_access(protocols=("openai",), env=ENV))
    class Live(Base):
        pass

    @assesses(card=dataset_through_target_access(datasets=("rows",)))
    class Tables(Base):
        pass

    @assesses(card=inputs_access())
    class Files(Base):
        pass

    @assesses(card=model_session(outputs={"label": "label", "probability": "score"}))
    class Scorer(Base):
        pass

    assert [target_kinds_of(c) for c in (Live, Tables, Files, Scorer)] == [("card",)] * 4
    assert [c.card_access for c in (Live, Tables, Files, Scorer)] == [
        "endpoint", "dataset_through_target", "inputs", "model_session"]
    assert Scorer.model_session_outputs == {"label": "label", "probability": "score"}
    assert all(c.local_model_input is None for c in (Live, Tables, Files, Scorer))
    assert target_access_of(Scorer) == "model_session"


def test_a_local_model_names_the_model_input_it_reads():
    @assesses(local_model=onnx(input="model"))
    @model_input
    class Local(Base):
        pass

    assert target_kinds_of(Local) == ("local_model",)
    assert Local.local_model_input == "model" and Local.card_access is None
    assert not [d for d in Local().input_definitions if d.name == SYSTEM_INPUT]


def test_onnx_reads_the_input_named_model_by_default():
    assert onnx().input == "model"


def test_a_local_model_needs_a_model_input_of_that_name():
    with pytest.raises(ValueError, match="'model'"):
        @assesses(local_model=onnx(input="model"))
        class NoInput(Base):
            pass

    with pytest.raises(ValueError, match="'model'"):
        @assesses(local_model=onnx(input="model"))
        @evaluation_input(name="model", label="Not a model", input_type=InputType.RESOURCE)
        class WrongType(Base):
            pass


def test_both_kinds_are_recorded_card_first():
    @assesses(card=system_under_test_access(protocols=("openai",), env=ENV), local_model=onnx())
    @model_input
    class Both(Base):
        pass

    assert target_kinds_of(Both) == ("card", "local_model")


def test_neither_is_refused():
    with pytest.raises(ValueError) as exc:
        @assesses()
        class Nothing(Base):
            pass

    assert str(exc.value) == "declare what the plugin assesses: card=, local_model= or both"


# the card builders keep the old decorators' behaviour

def test_a_card_target_alone_is_required_at_run_time():
    @assesses(card=system_under_test_access(protocols=("openai",), env=ENV))
    class Live(Base):
        pass

    with pytest.raises(ValueError, match="no system under test"):
        bound(Live).evaluate({})


def test_the_endpoint_builder_points_the_tool_at_the_platform(platform):
    @assesses(card=system_under_test_access(protocols=("openai",), env=ENV, needs=("history",)))
    class Live(Base):
        pass

    t = bound(Live, f"connection:{PID}/mcas-chat")
    assert t.evaluate({}) == "done" and t.seen == f"{ROOT}/openai/v1"
    assert target_access_of(Live) == "endpoint"


def test_with_a_local_model_too_the_card_access_is_optional():
    @assesses(card=system_under_test_access(protocols=("openai",), env=ENV), local_model=onnx())
    @model_input
    class Live(Base):
        pass

    @assesses(card=dataset_through_target_access(datasets=("rows",)), local_model=onnx())
    @model_input
    class Tables(Base):
        pass

    assert bound(Live).evaluate({}) == "done" and bound(Tables).evaluate({}) == "done"
    assert Live.target_access_required is False and Tables.target_access_required is False


def test_the_dataset_builder_alone_is_required():
    @assesses(card=dataset_through_target_access(datasets=("rows",)))
    class Tables(Base):
        pass

    assert Tables.target_access_required is True
    with pytest.raises(ValueError, match="no target"):
        bound(Tables).evaluate({})


def test_the_inputs_builder_changes_nothing_at_run_time():
    @assesses(card=inputs_access())
    class Files(Base):
        pass

    assert bound(Files).evaluate({}) == "done" and target_access_of(Files) == "inputs"


# one statement of access

def test_assesses_and_an_old_declaration_on_one_class_are_refused():
    old = [
        system_under_test(protocols=("openai",), env=ENV),
        dataset_through_target(datasets=("rows",)),
        assesses_inputs(),
    ]
    for declaration in old:
        with pytest.raises(ValueError):
            @assesses(card=inputs_access())
            @declaration
            class Below(Base):
                pass

        with pytest.raises(ValueError):
            @declaration
            @assesses(card=inputs_access())
            class Above(Base):
                pass


# subclasses

def test_a_subclass_inherits_the_declaration_unchanged():
    @assesses(card=system_under_test_access(protocols=("openai",), env=ENV), local_model=onnx())
    @model_input
    class Live(Base):
        pass

    class Sub(Live):
        pass

    assert target_kinds_of(Sub) == ("card", "local_model") and card_protocols_of(Sub) == ("openai",)
    assert card_needs_connection(Sub) and Sub.local_model_input == "model"


def test_assesses_on_a_subclass_of_an_assesses_base_is_refused():
    @assesses(card=inputs_access())
    class Files(Base):
        pass

    with pytest.raises(ValueError) as exc:
        @assesses(card=inputs_access())
        class Sub(Files):
            pass

    assert str(exc.value) == "Sub: it inherits what Files assesses; declare it on one class only"


def test_an_old_declaration_on_a_subclass_of_an_assesses_base_is_refused():
    @assesses(card=inputs_access())
    class Files(Base):
        pass

    for declaration in (system_under_test(protocols=("openai",), env=ENV),
                        dataset_through_target(datasets=("rows",)), assesses_inputs()):
        with pytest.raises(ValueError) as exc:
            @declaration
            class Sub(Files):
                pass

        assert str(exc.value) == "Sub: it inherits what Files assesses; declare it on one class only"


def test_assesses_on_a_subclass_of_an_old_declaration_is_refused():
    @system_under_test(protocols=("openai",), env=ENV)
    class Live(Base):
        pass

    with pytest.raises(ValueError) as exc:
        @assesses(local_model=onnx())
        @model_input
        class Sub(Live):
            pass

    assert str(exc.value) == "Sub: it inherits what Live assesses; declare it on one class only"


# what the engine reads

def test_a_class_with_an_old_declaration_reports_a_card_target():
    @system_under_test(protocols=("openai",), env=ENV)
    class Live(Base):
        pass

    @dataset_through_target(datasets=("rows",))
    class Tables(Base):
        pass

    @assesses_inputs()
    class Files(Base):
        pass

    class Undeclared(Base):
        pass

    assert [target_kinds_of(c) for c in (Live, Tables, Files)] == [("card",)] * 3
    assert target_kinds_of(Undeclared) == ()


def test_card_protocols_are_those_of_the_endpoint_access_only():
    @assesses(card=system_under_test_access(protocols=("openai", "a2a"), env=ENV))
    class Live(Base):
        pass

    @system_under_test(protocols=("aisc",), env={"ASK": "ask_url"})
    class OldLive(Base):
        pass

    @assesses(card=dataset_through_target_access(datasets=("rows",)))
    class Tables(Base):
        pass

    @assesses(local_model=onnx())
    @model_input
    class Local(Base):
        pass

    assert card_protocols_of(Live) == ("openai", "a2a") and card_protocols_of(OldLive) == ("aisc",)
    assert card_protocols_of(Tables) == () and card_protocols_of(Local) == ()


def test_which_card_accesses_need_a_connection():
    @assesses(card=system_under_test_access(protocols=("openai",), env=ENV), local_model=onnx())
    @model_input
    class Live(Base):
        pass

    @assesses(card=dataset_through_target_access(datasets=("rows",)))
    class Tables(Base):
        pass

    @assesses(card=model_session(outputs={"label": "label"}))
    class Scorer(Base):
        pass

    @assesses(card=inputs_access())
    class Files(Base):
        pass

    @assesses(local_model=onnx())
    @model_input
    class Local(Base):
        pass

    @system_under_test(protocols=("openai",), env=ENV, required=False)
    class OldOptional(Base):
        pass

    @dataset_through_target(datasets=("rows",))
    class OldTables(Base):
        pass

    @assesses_inputs()
    class OldFiles(Base):
        pass

    # a card target picked for a card access that calls it needs a connection, even when the plugin
    # also takes a local model file (that only makes the card target optional in the form)
    assert card_needs_connection(Live) and card_needs_connection(Tables) and card_needs_connection(Scorer)
    assert not card_needs_connection(Files) and not card_needs_connection(Local)
    # an old declaration says it with its required flag
    assert not card_needs_connection(OldOptional) and card_needs_connection(OldTables)
    assert not card_needs_connection(OldFiles)


# an optional card access: the run may have no card target, or one without an endpoint

KEY = "component:0b9c7a1e-0000-4000-8000-000000000002"


def optional_tables():
    @assesses(card=dataset_through_target_access(("a", "b"), required=False))
    class Tables(Base):
        pass

    return Tables


def test_a_card_access_can_be_optional():
    Tables = optional_tables()
    assert target_kinds_of(Tables) == ("card",)
    assert Tables.card_access == "dataset_through_target"
    assert Tables.target_access_required is False
    assert Tables.card_access_required is False
    assert card_needs_connection(Tables) is False


def test_an_optional_dataset_access_runs_on_the_uploaded_data_without_a_target(platform):
    assert bound(optional_tables()).evaluate({}) == "done"
    assert platform.seen == []


def test_an_optional_dataset_access_runs_when_the_target_has_no_endpoint(stub, monkeypatch):
    stub.route(f"/internal/projects/{PID}/targets/{KEY}/connection",
               (404, {"detail": "Training data has no endpoint: set one under Manage, Components and connections",
                      "reason": "no_endpoint"}))
    monkeypatch.setenv("PLATFORM_URL", stub.base)
    monkeypatch.setenv("PLATFORM_CONNECTIONS_TOKEN", "svc-token")
    assert bound(optional_tables(), f"target:{PID}/{KEY}").evaluate({}) == "done"
    assert len(stub.seen) == 1


def test_an_optional_endpoint_access_runs_unchanged_without_a_target(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    @assesses(card=system_under_test_access(protocols=("openai",), env=ENV, required=False))
    class Live(Base):
        pass

    assert Live.target_access_required is False
    assert Live.card_access_required is False
    assert card_needs_connection(Live) is False
    t = bound(Live)
    assert t.evaluate({}) == "done"
    assert t.seen is None


def test_the_card_access_is_required_by_default():
    import inspect

    assert dataset_through_target_access(("a",)).required is True
    assert system_under_test_access(("openai",), env=ENV).required is True
    assert model_session({"label": "l"}).required is True
    assert inputs_access().required is True
    for builder in (dataset_through_target_access, system_under_test_access):
        parameter = inspect.signature(builder).parameters["required"]
        assert parameter.default is True


def test_a_required_flag_on_the_builder_is_recorded():
    access = dataset_through_target_access(("a",), required=False)
    assert access.required is False
    assert access.access == "dataset_through_target"


def test_optional_with_a_local_model_too_needs_no_connection():
    @assesses(card=dataset_through_target_access(("rows",), required=False), local_model=onnx())
    @model_input
    class Tables(Base):
        pass

    assert target_kinds_of(Tables) == ("card", "local_model")
    assert Tables.target_access_required is False
    assert card_needs_connection(Tables) is False


def test_with_a_local_model_and_the_default_flag_a_card_target_still_needs_a_connection():
    @assesses(card=system_under_test_access(protocols=("openai",), env=ENV), local_model=onnx())
    @model_input
    class Live(Base):
        pass

    assert card_needs_connection(Live) is True
    assert Live.card_access_required is True


def test_a_class_without_a_card_has_no_card_access_flag():
    @assesses(local_model=onnx())
    @model_input
    class Local(Base):
        pass

    assert Local.card_access_required is None
    assert card_needs_connection(Local) is False


def test_the_optional_builder_declares_what_the_old_optional_decorator_declares():
    @dataset_through_target(datasets=("r",), required=False)
    class Old(Base):
        pass

    @assesses(card=dataset_through_target_access(("r",), required=False))
    class New(Base):
        pass

    for read in (target_kinds_of, card_needs_connection, card_protocols_of, target_access_of):
        assert read(Old) == read(New), read.__name__
    assert Old.target_access_required == New.target_access_required
    assert [d.name for d in Old().input_definitions] == [d.name for d in New().input_definitions]


def test_the_declarations_keep_no_copy_of_their_arguments_that_nothing_reads():
    @system_under_test(protocols=("openai",), env=ENV, needs=("history",))
    class Live(Base):
        pass

    @dataset_through_target(datasets=("rows",))
    class Tables(Base):
        pass

    assert not hasattr(Live, "system_under_test_needs")
    for attribute in ("dataset_through_target_inputs", "dataset_through_target_required"):
        assert not hasattr(Tables, attribute), attribute

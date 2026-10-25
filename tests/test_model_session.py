"""self.assessed_model(): a plugin that scores rows with a classification model calls one object,
predict(rows) and predict_proba(rows), whether the model is a local ONNX file uploaded to the project
or an AI card target reached through its connection. The plugin never asks where the model lives."""
import json

import pytest

from aisc_plugin_interface import (
    BaseEvaluationPlugin,
    InputType,
    OnnxInputProvider,
    assesses,
    card_needs_connection,
    evaluation_input,
    model_session,
    onnx,
)
from aisc_plugin_interface.connections import EndpointBadResponse, EndpointError
from aisc_plugin_interface.input_providers.base_input_provider import BaseInputProvider
from aisc_plugin_interface.model_session import EndpointSession, ModelSession, OnnxSession
from aisc_plugin_interface.system_under_test import SYSTEM_INPUT

PID = "0f7c1e2a-aaaa-bbbb-cccc-1234567890ab"
KEY = "component:0b9c7a1e-0000-4000-8000-000000000002"
SCORER = {"name": "mcas-score", "label": "MCAS scorer", "kind": "rest", "method": "POST", "path": "/score",
          "headers": {}, "secret_header": "X-Auth-Token: {{secret}}", "body_template": "{{input}}",
          "response_path": "$", "refusal": {"status": [502], "path": "detail.reason"}, "model": None,
          "timeout_s": 5, "secret": "s3cr3t", "updated_at": "2026-10-04T10:00:00Z",
          "target": {"key": KEY, "kind": "component", "label": "Scoring engine"}}
OUTPUTS = {"label": "decision", "probability": "risk.p"}
ROWS = [{"amount": 2500, "months": 12}, {"amount": 4000, "months": 24}, {"amount": 100, "months": 6}]


class Base(BaseEvaluationPlugin):
    def evaluate(self, config_data):
        return "done"

    def export_metrics(self, *a, **k):
        return []


def model_input(cls):
    return evaluation_input(name="model", label="Model", input_provider_class=OnnxInputProvider,
                            input_type=InputType.MODEL)(cls)


def card_scorer(outputs=OUTPUTS):
    @assesses(card=model_session(outputs=outputs))
    class Scorer(Base):
        pass

    return Scorer


def either_scorer():
    @assesses(card=model_session(outputs=OUTPUTS), local_model=onnx(input="model"))
    @model_input
    class Scorer(Base):
        pass

    return Scorer


def bound(cls, target=False, model=None):
    t = cls()
    t._set_artifact_callback(lambda name, content: None)
    if target:
        t.set_input_content(SYSTEM_INPUT, json.dumps({"value": f"target:{PID}/{KEY}"}).encode())
    if model is not None:
        t.set_input_content("model", model)
    return t


def score(request):
    row = json.loads(request["body"])
    p = min(1.0, row["amount"] / 5000)
    return {"decision": "decline" if p > 0.5 else "approve", "risk": {"p": p}}


@pytest.fixture
def system(stub, monkeypatch):
    stub.route(f"/internal/projects/{PID}/targets/{KEY}/connection",
               (200, dict(SCORER, base_url=stub.base, allowed_hosts=[stub.host], denied_addresses=[])))
    stub.route("/score", (200, score))
    monkeypatch.setenv("PLATFORM_URL", stub.base)
    monkeypatch.setenv("PLATFORM_CONNECTIONS_TOKEN", "svc-token")
    return stub


def calls(stub):
    return [r for r in stub.seen if r["path"] == "/score"]


# the declaration

def test_the_builder_declares_the_target_input_required_when_it_is_the_only_kind():
    Scorer = card_scorer()
    assert SYSTEM_INPUT in {d.name for d in Scorer().input_definitions}
    assert Scorer.target_access_required is True
    assert card_needs_connection(Scorer)


def test_with_a_local_model_too_the_target_is_optional():
    Scorer = either_scorer()
    assert {SYSTEM_INPUT, "model"} <= {d.name for d in Scorer().input_definitions}
    assert Scorer.target_access_required is False


def test_the_builder_needs_a_label_or_a_probability_output():
    with pytest.raises(ValueError, match="label"):
        model_session(outputs={})
    with pytest.raises(ValueError, match="colour"):
        model_session(outputs={"label": "decision", "colour": "c"})
    for classes in (1, "2", True, 2.0):
        with pytest.raises(ValueError, match="classes"):
            model_session(outputs={"label": "decision", "classes": classes})


# an AI card target

def test_a_card_target_answers_each_row_once_with_its_label_and_probability(system):
    model = bound(card_scorer(), target=True).assessed_model()
    assert isinstance(model, EndpointSession) and isinstance(model, ModelSession)
    assert model.predict(ROWS) == ["approve", "decline", "approve"]
    assert len(calls(system)) == 3
    assert [json.loads(r["body"]) for r in calls(system)] == ROWS
    assert model.predict_proba(ROWS) == [pytest.approx(p) for p in ([0.5, 0.5], [0.2, 0.8], [0.98, 0.02])]
    assert len(calls(system)) == 3                         # the session keeps the answers it got


def test_a_run_can_give_the_answer_paths_instead_of_the_declared_ones(system):
    system.route("/score", (200, lambda request: {"verdict": "yes", "scores": {"risk": 0.25}}))
    model = bound(card_scorer(), target=True).assessed_model(
        outputs={"label": "verdict", "probability": "scores.risk"})
    assert model.predict(ROWS[:1]) == ["yes"]
    assert model.predict_proba(ROWS[:1]) == [pytest.approx([0.75, 0.25])]
    # this session only: the next one reads the declared paths again
    assert bound(card_scorer(), target=True).assessed_model().outputs == OUTPUTS


def test_per_run_answer_paths_that_are_not_valid_are_refused_naming_the_key(system):
    plugin = bound(card_scorer(), target=True)
    with pytest.raises(ValueError, match="colour"):
        plugin.assessed_model(outputs={"label": "verdict", "colour": "c"})
    with pytest.raises(ValueError, match="label"):
        plugin.assessed_model(outputs={})
    with pytest.raises(ValueError, match="classes"):
        plugin.assessed_model(outputs={"label": "verdict", "classes": 1})


def test_per_run_answer_paths_naming_several_classes_are_refused_on_a_card_target(system):
    with pytest.raises(ValueError, match="3 classes"):
        bound(card_scorer(), target=True).assessed_model(outputs={"label": "verdict", "classes": 3})


def test_probabilities_then_labels_send_each_row_once(system):
    model = bound(card_scorer(), target=True).assessed_model()
    model.predict_proba(ROWS)
    assert model.predict(list(reversed(ROWS))) == ["approve", "decline", "approve"]
    assert len(calls(system)) == 3


def test_without_a_label_output_the_label_is_the_more_probable_class(system):
    model = bound(card_scorer({"probability": "risk.p"}), target=True).assessed_model()
    assert model.predict(ROWS) == [0, 1, 0]


def test_a_missing_probability_in_the_answer_fails_naming_the_path_and_the_row(system):
    system.route("/score", (200, score), (200, {"decision": "approve"}))
    model = bound(card_scorer(), target=True).assessed_model()
    with pytest.raises(EndpointBadResponse, match=r"row 2.*'risk\.p'"):
        model.predict_proba(ROWS)


@pytest.mark.parametrize("label", [None, {"code": 1}, ["approve"]])
def test_a_label_that_is_not_a_value_fails_naming_the_path_and_the_row(system, label):
    system.route("/score", (200, score), (200, {"decision": label, "risk": {"p": 0.1}}))
    model = bound(card_scorer(), target=True).assessed_model()
    with pytest.raises(EndpointBadResponse, match=r"row 2.*'decision'"):
        model.predict(ROWS)


def test_a_probability_that_is_not_one_fails_naming_the_row(system):
    system.route("/score", (200, {"decision": "approve", "risk": {"p": "high"}}))
    model = bound(card_scorer(), target=True).assessed_model()
    with pytest.raises(EndpointBadResponse, match=r"row 1.*'risk\.p'"):
        model.predict_proba(ROWS)


def test_a_refusal_fails_naming_the_row(system):
    system.route("/score", (200, score), (200, score), (502, {"detail": {"reason": "outside the markets"}}))
    model = bound(card_scorer(), target=True).assessed_model()
    with pytest.raises(EndpointError, match="row 3.*outside the markets"):
        model.predict(ROWS)


def test_the_key_echoed_by_the_target_is_not_in_the_errors(system):
    system.route("/score", (502, {"detail": {"reason": "bad key s3cr3t"}}))
    with pytest.raises(EndpointError) as refused:
        bound(card_scorer(), target=True).assessed_model().predict(ROWS)
    system.route("/score", (200, {"decision": "approve", "risk": {"p": "s3cr3t"}}))
    with pytest.raises(EndpointError) as unreadable:
        bound(card_scorer(), target=True).assessed_model().predict_proba(ROWS)
    assert all("s3cr3t" not in str(e.value) for e in (refused, unreadable))


def test_any_failure_while_sending_a_row_names_the_row():
    class Broken:
        def ask(self, row):
            raise ValueError("cannot encode the row")

    with pytest.raises(ValueError, match="row 1: cannot encode the row"):
        EndpointSession(Broken(), OUTPUTS).predict(ROWS)


def test_a_failed_call_fails_naming_the_row(system):
    system.route("/score", (422, {"detail": "bad row"}))
    model = bound(card_scorer(), target=True).assessed_model()
    with pytest.raises(EndpointBadResponse, match="row 1.*422"):
        model.predict(ROWS)


def test_a_multi_class_card_target_is_refused_before_any_call(system):
    t = bound(card_scorer({"label": "decision", "probability": "risk.p", "classes": 3}), target=True)
    with pytest.raises(ValueError, match="binary"):
        t.assessed_model()
    assert system.seen == []


# what the run binds

def test_neither_a_model_file_nor_a_target_raises():
    with pytest.raises(ValueError, match="no model"):
        bound(either_scorer()).assessed_model()


class RawBytes(BaseInputProvider):
    def _read_data(self, file_content):
        return file_content


def test_a_model_file_and_a_target_both_bound_raises(system):
    @assesses(card=model_session(outputs=OUTPUTS), local_model=onnx(input="model"))
    @evaluation_input(name="model", label="Model", input_provider_class=RawBytes, input_type=InputType.MODEL)
    class Scorer(Base):
        pass

    t = bound(Scorer, target=True, model=b"a model file")
    with pytest.raises(ValueError, match="both"):
        t.assessed_model()
    assert system.seen == []


def test_a_plugin_that_declares_no_model_to_score_cannot_ask_for_one():
    @assesses(local_model=onnx())
    @model_input
    class Local(Base):
        pass

    from aisc_plugin_interface import inputs_access

    @assesses(card=inputs_access())
    class Files(Base):
        pass

    with pytest.raises(ValueError, match="declares no model"):
        bound(Files).assessed_model()
    with pytest.raises(ValueError, match="no model"):
        bound(Local).assessed_model()


# a local ONNX file

WEIGHTS = [[0.5, -1.0], [-0.25, 0.75]]
BIAS = [0.1, -0.1]


def tiny_model(onnx_):
    """A two-feature, two-class linear classifier with a label and a probability output, no ZipMap."""
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [helper.make_node("MatMul", ["features", "W"], ["xw"]),
         helper.make_node("Add", ["xw", "B"], ["logits"]),
         helper.make_node("ArgMax", ["logits"], ["label"], axis=1, keepdims=0),
         helper.make_node("Softmax", ["logits"], ["probabilities"], axis=1)],
        "tiny",
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, 2])],
        [helper.make_tensor_value_info("label", TensorProto.INT64, [None]),
         helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [None, 2])],
        initializer=[helper.make_tensor("W", TensorProto.FLOAT, [2, 2], sum(WEIGHTS, [])),
                     helper.make_tensor("B", TensorProto.FLOAT, [2], BIAS)])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    return model.SerializeToString()


@pytest.fixture
def onnx_bytes():
    onnx_ = pytest.importorskip("onnx", reason="the local model session runs an ONNX model built with onnx")
    pytest.importorskip("onnxruntime", reason="the local model session runs its model with onnxruntime")
    return tiny_model(onnx_)


def expected(rows):
    import math

    out = []
    for r in rows:
        x = [r["amount"] / 1000, r["months"] / 10]
        logits = [x[0] * WEIGHTS[0][j] + x[1] * WEIGHTS[1][j] + BIAS[j] for j in range(2)]
        e = [math.exp(v - max(logits)) for v in logits]
        out.append([v / sum(e) for v in e])
    return out


LOCAL_ROWS = [{"amount": r["amount"] / 1000, "months": r["months"] / 10} for r in ROWS]


def test_a_local_model_file_gives_one_label_and_two_probabilities_per_row(onnx_bytes):
    model = bound(either_scorer(), model=onnx_bytes).assessed_model()
    assert isinstance(model, OnnxSession) and isinstance(model, ModelSession)
    proba = model.predict_proba(LOCAL_ROWS)
    assert isinstance(proba, list) and all(isinstance(p, list) and len(p) == 2 for p in proba)
    assert all(isinstance(v, float) for p in proba for v in p)
    assert proba == [pytest.approx(e, abs=1e-6) for e in expected(ROWS)]
    labels = model.predict(LOCAL_ROWS)
    assert labels == [max((0, 1), key=lambda j: e[j]) for e in expected(ROWS)]
    assert all(isinstance(v, int) for v in labels)


def test_the_local_session_takes_the_model_bytes_and_the_tables_plugins_pass_today(onnx_bytes):
    pd = pytest.importorskip("pandas", reason="a DataFrame is one of the tables the local session takes")
    import numpy as np

    model = OnnxSession(onnx_bytes)
    frame = pd.DataFrame(LOCAL_ROWS)
    want = model.predict_proba(LOCAL_ROWS)
    assert model.predict_proba(frame) == want
    assert model.predict_proba(frame.to_numpy()) == want
    assert model.predict(frame) == model.predict(LOCAL_ROWS)


def test_a_row_missing_a_feature_fails_naming_the_row(onnx_bytes):
    with pytest.raises(ValueError, match="row 2.*months"):
        OnnxSession(onnx_bytes).predict([{"amount": 1.0, "months": 1.0}, {"amount": 2.0}])



def test_no_rows_give_no_predictions_without_running_the_model(onnx_bytes):
    model = OnnxSession(onnx_bytes)
    model.session = None                                    # any run would fail
    assert model.predict([]) == [] and model.predict_proba([]) == []


def test_a_local_session_ignores_per_run_answer_paths_but_still_validates_them(onnx_bytes):
    plugin = bound(either_scorer(), model=onnx_bytes)
    assert isinstance(plugin.assessed_model(outputs={"label": "verdict"}), OnnxSession)
    with pytest.raises(ValueError, match="colour"):
        plugin.assessed_model(outputs={"label": "verdict", "colour": "c"})


# a regression's number: the value output

import importlib  # noqa: E402

# the module (the package's own ``model_session`` is the builder of the same name)
sessions = importlib.import_module("aisc_plugin_interface.model_session")
from aisc_plugin_interface.model_session import ModelOutputMissing  # noqa: E402

VALUE_ROWS = [{"a": 1.0, "b": 2.0}, {"a": 3.0, "b": 0.0}]


def tiny_regressor(flat=False):
    """A two-feature linear regressor, one number per row: features @ W + B with W = [[2], [-1]] and
    B = [0.5]; ``flat`` gives W the shape [2], so the output has shape [None] instead of [None, 1]."""
    from onnx import TensorProto, helper

    w_shape, out_shape = ([2], [None]) if flat else ([2, 1], [None, 1])
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["features", "W"], ["xw"]),
         helper.make_node("Add", ["xw", "B"], ["variable"])],
        "tiny_regressor",
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, 2])],
        [helper.make_tensor_value_info("variable", TensorProto.FLOAT, out_shape)],
        initializer=[helper.make_tensor("W", TensorProto.FLOAT, w_shape, [2.0, -1.0]),
                     helper.make_tensor("B", TensorProto.FLOAT, [1], [0.5])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    return model.SerializeToString()


@pytest.fixture
def onnx_runtime():
    pytest.importorskip("onnx", reason="the local model session runs an ONNX model built with onnx")
    pytest.importorskip("onnxruntime", reason="the local model session runs its model with onnxruntime")


def value_scorer(**declared):
    @assesses(card=model_session(outputs={"value": "prediction"}, **declared))
    class Regressor(Base):
        pass

    return Regressor


def test_value_output_keys():
    assert sessions.OUTPUT_KEYS == ("label", "probability", "classes", "value")


def test_value_output_reads_a_number(system):
    system.route("/score", (200, {"prediction": 3}), (200, {"prediction": 2.5}), (200, {"prediction": -1}))
    values = bound(value_scorer(), target=True).assessed_model().predict_value(ROWS)
    assert values == [3.0, 2.5, -1.0]
    assert all(type(v) is float for v in values)
    assert len(calls(system)) == 3


@pytest.mark.parametrize("outputs", [{"value": "p", "probability": "q"}, {"value": "p", "label": "l"},
                                     {"value": "p", "classes": 2}])
def test_value_is_declared_alone(outputs):
    with pytest.raises(ValueError, match="value .*declared alone"):
        model_session(outputs=outputs)


def test_predict_on_a_value_session_gives_the_values(system):
    system.route("/score", (200, {"prediction": 3}), (200, {"prediction": 2.5}), (200, {"prediction": -1}))
    model = bound(value_scorer(), target=True).assessed_model()
    assert model.predict(ROWS) == [3.0, 2.5, -1.0]
    model.predict_value(ROWS)
    assert len(calls(system)) == 3
    with pytest.raises(ValueError, match="no probability output"):
        model.predict_proba(ROWS)


def test_predict_value_needs_a_value_output(system):
    model = bound(card_scorer(), target=True).assessed_model()
    with pytest.raises(ValueError, match="declares no value output"):
        model.predict_value(ROWS)
    assert calls(system) == []


@pytest.mark.parametrize("said", ["high", True, None, float("inf")])
def test_a_value_that_is_not_a_number_fails_naming_the_row(system, said):
    system.route("/score", (200, {"prediction": said}))                    # inf goes out as Infinity
    model = bound(value_scorer(strict=True), target=True).assessed_model()
    with pytest.raises(ModelOutputMissing, match=r"row 1: the value at 'prediction' is not a number"):
        model.predict_value(ROWS)


def test_per_run_value_path(system):
    system.route("/score", (200, {"y": 0.25}))
    model = bound(card_scorer(), target=True).assessed_model(outputs={"value": "y"})
    assert model.predict_value(ROWS[:1]) == [0.25]


def test_onnx_value_is_the_first_output_squeezed(onnx_runtime):
    for flat in (False, True):
        values = OnnxSession(tiny_regressor(flat=flat)).predict_value(VALUE_ROWS)
        assert values == pytest.approx([0.5, 6.5])
        assert all(type(v) is float for v in values)


def test_onnx_value_refuses_a_classifier(onnx_bytes):
    with pytest.raises(ValueError, match="classifier"):
        OnnxSession(onnx_bytes).predict_value(VALUE_ROWS)


def test_onnx_value_of_no_rows_is_empty(onnx_runtime):
    assert OnnxSession(tiny_regressor()).predict_value([]) == []


def test_both_sessions_are_model_sessions_with_predict_value(onnx_runtime):
    class NoClient:
        descriptor = None

    assert isinstance(OnnxSession(tiny_regressor()), ModelSession)
    assert isinstance(EndpointSession(NoClient(), {"label": "l"}), ModelSession)
    assert hasattr(ModelSession, "predict_value")


# under a run pass the session's rows go through the platform

ASK_PATH = f"/internal/facade/{PID}/mcas-score/aisc/ask"


def asked(request):
    """The platform's answer to an ask: the scorer's answer to the posted input."""
    return {"output": score({"body": json.dumps(json.loads(request["body"])["input"]).encode()}),
            "refused": False, "refusal_reason": None, "status": 200, "latency_ms": 7}


def test_a_model_session_under_a_pass_goes_through_the_platform(stub, monkeypatch):
    resolved = {k: v for k, v in SCORER.items() if k != "secret"}
    stub.route(f"/internal/projects/{PID}/targets/{KEY}/connection",
               (200, dict(resolved, has_secret=True, base_url=stub.base, allowed_hosts=[stub.host],
                          denied_addresses=[])))
    stub.route(f"/internal/projects/{PID}/targets/{KEY}/run-keys",
               (201, {"key": "aisc-run-abc", "expires_at": "2026-10-26T00:00:00+00:00", "connection": "mcas-score",
                      "capabilities": ["text", "history"], "target": SCORER["target"],
                      "endpoints": {"aisc": {"ask_url": stub.base + ASK_PATH}}}))
    stub.route(ASK_PATH, (200, asked))
    stub.route("/score", (200, score))
    monkeypatch.setenv("PLATFORM_URL", stub.base)
    monkeypatch.setenv("PLATFORM_RUN_PASS", "pass-123")
    monkeypatch.delenv("PLATFORM_CONNECTIONS_TOKEN", raising=False)
    model = bound(card_scorer(), target=True).assessed_model()
    assert model.predict_proba(ROWS) == [pytest.approx(p) for p in ([0.5, 0.5], [0.2, 0.8], [0.98, 0.02])]
    assert len([r for r in stub.seen if r["path"] == ASK_PATH]) == 3
    assert calls(stub) == []

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


# an AI card target

def test_a_card_target_answers_each_row_once_with_its_label_and_probability(system):
    model = bound(card_scorer(), target=True).assessed_model()
    assert isinstance(model, EndpointSession) and isinstance(model, ModelSession)
    assert model.predict(ROWS) == ["approve", "decline", "approve"]
    assert len(calls(system)) == 3
    assert [json.loads(r["body"]) for r in calls(system)] == ROWS
    assert model.predict_proba(ROWS) == [pytest.approx(p) for p in ([0.5, 0.5], [0.2, 0.8], [0.98, 0.02])]
    assert len(calls(system)) == 6


def test_without_a_label_output_the_label_is_the_more_probable_class(system):
    model = bound(card_scorer({"probability": "risk.p"}), target=True).assessed_model()
    assert model.predict(ROWS) == [0, 1, 0]


def test_a_missing_probability_in_the_answer_fails_naming_the_path_and_the_row(system):
    system.route("/score", (200, score), (200, {"decision": "approve"}))
    model = bound(card_scorer(), target=True).assessed_model()
    with pytest.raises(EndpointBadResponse, match=r"row 2.*'risk\.p'"):
        model.predict_proba(ROWS)


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


def test_a_model_file_and_a_target_both_bound_raises(system):
    onnx_ = pytest.importorskip("onnx", reason="the local model session runs an ONNX model built with onnx")
    pytest.importorskip("onnxruntime", reason="the local model session runs its model with onnxruntime")
    t = bound(either_scorer(), target=True, model=tiny_model(onnx_))
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


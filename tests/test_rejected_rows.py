"""A model session declared ``strict=False`` skips the rows its target rejects (a refusal, a 4xx answer,
an answer it cannot read) and counts them: ``None`` at that row, ``session.rejected`` and the run artifact
``rejected-rows.json``. A target that fails (auth, not found, timeout, 5xx) still stops the run, and so
does a target that rejects every row. The default session is strict and stops on the first rejected row
as before."""
import importlib
import json

import pytest

from aisc_plugin_interface import assesses, model_session
from aisc_plugin_interface.connections import (
    Descriptor, EndpointAuthError, EndpointBadResponse, EndpointClient, EndpointNotFound,
)
from aisc_plugin_interface.model_session import EndpointSession, ModelOutputMissing, OnnxSession
from aisc_plugin_interface.system_under_test import SYSTEM_INPUT

# the module (the package's own ``model_session`` is the builder of the same name)
sessions = importlib.import_module("aisc_plugin_interface.model_session")

from test_model_session import KEY, OUTPUTS, PID, ROWS, SCORER, Base, score, system, tiny_model  # noqa: F401 (system: fixture)

ARTIFACT = "rejected-rows.json"
MISSING_P = "the target's answer has no 'risk.p' (the declared probability output)"


def scorer(outputs=OUTPUTS, **declared):
    @assesses(card=model_session(outputs=outputs, **declared))
    class Scorer(Base):
        pass

    return Scorer


def bound(cls):
    """The plugin bound to the card target, and the list of (name, content) it uploaded."""
    uploads = []
    t = cls()
    t._set_artifact_callback(lambda name, content: uploads.append((name, content)))
    t.set_input_content(SYSTEM_INPUT, json.dumps({"value": f"target:{PID}/{KEY}"}).encode())
    return t, uploads


def rejected_uploads(uploads):
    return [content for name, content in uploads if name == ARTIFACT]


def calls(stub):
    return [r for r in stub.seen if r["path"] == "/score"]


def lenient(stub, *answers, outputs=OUTPUTS):
    """A non-strict session whose /score answers ``answers`` in order, and the uploads list."""
    stub.route("/score", *answers)
    plugin, uploads = bound(scorer(outputs, strict=False))
    return plugin.assessed_model(), uploads


OK = (200, score)
NO_P = (200, {"decision": "approve"})


def test_a_row_whose_answer_lacks_the_probability_is_skipped_and_counted(system):
    model, _ = lenient(system, OK, NO_P, OK)
    assert model.predict_proba(ROWS) == [pytest.approx([0.5, 0.5]), None, pytest.approx([0.98, 0.02])]
    assert model.rejected == [(1, MISSING_P)]


def test_strict_stops_on_the_first_rejected_row(system):
    system.route("/score", OK, NO_P, OK)
    plugin, uploads = bound(scorer(strict=True))
    model = plugin.assessed_model()
    with pytest.raises(ModelOutputMissing, match=r"row 2.*'risk\.p'"):
        model.predict_proba(ROWS)
    assert model.rejected == []
    assert rejected_uploads(uploads) == []


def test_the_default_session_stops_on_a_rejected_row_as_before(system):
    system.route("/score", OK, NO_P, OK)
    plugin, uploads = bound(scorer())
    model = plugin.assessed_model()
    with pytest.raises(ModelOutputMissing, match=r"row 2: the target's answer has no 'risk\.p'"):
        model.predict_proba(ROWS)
    assert model.strict is True
    assert model.rejected == []
    assert rejected_uploads(uploads) == []


def test_the_rejected_rows_artifact_is_uploaded_once(system):
    model, uploads = lenient(system, OK, NO_P, OK)
    model.predict_proba(ROWS)
    model.predict(ROWS)
    (content,) = rejected_uploads(uploads)
    assert json.loads(content) == {"rows_sent": 3, "rows_rejected": 1,
                                   "rejected": [{"index": 1, "reason": MISSING_P}]}


def test_a_rejected_row_is_sent_once(system):
    model, _ = lenient(system, OK, NO_P, OK)
    model.predict_proba(ROWS)
    assert model.predict(ROWS) == ["approve", None, "approve"]
    assert len(calls(system)) == 3


def test_a_refused_row_is_rejected(system):
    model, _ = lenient(system, OK, OK, (502, {"detail": {"reason": "outside the markets"}}))
    assert model.predict(ROWS) == ["approve", "decline", None]
    assert model.rejected == [(2, "the target refused it: outside the markets")]


def test_a_row_the_target_answers_4xx_is_rejected(system):
    model, _ = lenient(system, OK, (422, {"detail": "revolving_utilisation must be <= 1"}), OK)
    assert model.predict_proba(ROWS)[1] is None
    assert model.rejected == [
        (1, 'MCAS scorer: the system answered 422: {"detail": "revolving_utilisation must be <= 1"}')]


def test_the_mcas_error_row_is_rejected(system):
    model, _ = lenient(system, (200, {"p_default": 0.2}),
                       (200, {"p_default": None, "error": "revolving_utilisation above 1"}),
                       (200, {"p_default": 0.4}), outputs={"probability": "p_default"})
    assert model.predict_proba(ROWS) == [pytest.approx([0.8, 0.2]), None, pytest.approx([0.6, 0.4])]
    assert model.rejected == [(1, "the probability at 'p_default' is not a number from 0 to 1: 'None'")]


@pytest.mark.parametrize("status, error", [(401, EndpointAuthError), (404, EndpointNotFound),
                                           (500, EndpointBadResponse)])
def test_target_failures_still_stop_a_non_strict_run(stub, status, error):
    stub.route("/score", (status, {"detail": "no"}))
    client = EndpointClient(Descriptor.from_dict(dict(SCORER, base_url=stub.base)), allowed_hosts=[stub.host],
                            sleep=lambda s: None)
    session = EndpointSession(client, OUTPUTS, strict=False)
    with pytest.raises(error, match="row 1") as raised:
        session.predict(ROWS)
    if status == 500:
        assert raised.value.status == 500
        assert not isinstance(raised.value, sessions.AllRowsRejected)
    assert session.rejected == []


def test_every_row_rejected_raises_all_rows_rejected(system):
    model, uploads = lenient(system, (200, {"decision": "approve", "risk": {"p": "high"}}))
    with pytest.raises(sessions.AllRowsRejected) as raised:
        model.predict_proba(ROWS)
    assert str(raised.value) == ("the target rejected all 3 rows; row 1: the probability at 'risk.p' is not a"
                                 " number from 0 to 1: 'high'")
    assert isinstance(raised.value, EndpointBadResponse)
    (content,) = rejected_uploads(uploads)
    assert json.loads(content)["rows_rejected"] == 3


def test_the_key_is_not_in_the_reasons_or_the_artifact(system):
    model, uploads = lenient(system, OK, (422, {"detail": "bad key s3cr3t"}), OK)
    model.predict(ROWS)
    reason = model.rejected[0][1]
    assert "***" in reason and "s3cr3t" not in reason
    (content,) = rejected_uploads(uploads)
    assert b"s3cr3t" not in content


def test_rejected_is_the_last_calls(system):
    model, uploads = lenient(system, OK, NO_P, OK)
    model.predict_proba(ROWS)
    model.predict_proba(ROWS[:1])
    assert model.rejected == []
    assert len(rejected_uploads(uploads)) == 1


def test_a_long_reason_is_cut(system):
    model, _ = lenient(system, OK, (422, {"detail": "x" * 1000}), OK)
    model.predict(ROWS)
    assert len(model.rejected[0][1]) == 300


def test_a_session_without_a_plugin_uploads_nothing(stub):
    stub.route("/score", OK, NO_P, OK)
    client = EndpointClient(Descriptor.from_dict(dict(SCORER, base_url=stub.base)), allowed_hosts=[stub.host],
                            sleep=lambda s: None)
    session = EndpointSession(client, OUTPUTS, strict=False)
    assert session.predict_proba(ROWS)[1] is None


def test_strict_is_part_of_the_declaration(system):
    assert scorer().model_session_strict is True
    assert scorer(strict=True).model_session_strict is True
    assert scorer(strict=False).model_session_strict is False
    for declared in ({}, {"strict": True}, {"strict": False}):
        cls = scorer(**declared)
        plugin, _ = bound(cls)
        assert plugin.assessed_model().strict == cls.model_session_strict


def test_no_rows_give_no_rejections(system):
    model, _ = lenient(system, OK)
    assert model.predict_proba([]) == []
    assert model.rejected == []
    assert calls(system) == []


def test_a_local_session_rejects_nothing():
    onnx = pytest.importorskip("onnx", reason="the local model session runs an ONNX model built with onnx")
    pytest.importorskip("onnxruntime", reason="the local model session runs its model with onnxruntime")
    model = OnnxSession(tiny_model(onnx))
    model.predict([{"amount": 1.0, "months": 1.0}])
    assert model.rejected == []


def test_the_artifact_name_is_exported():
    assert sessions.REJECTED_ROWS_ARTIFACT == ARTIFACT

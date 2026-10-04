"""@dataset_through_target (the plan's R1, third kind): a plugin that analyses a table can have it sent, row by
row, through the evaluation's target first, so it analyses what the system answered, not only what went
in. Each scalar field of an answer becomes a column `target.<field>`; the plugin reads the enriched table as
if it had been uploaded so."""
import io
import json

import pandas as pd
import pytest

from aisc_plugin_interface import BaseEvaluationPlugin, InputType, evaluation_input
from aisc_plugin_interface.input_providers.base_input_provider import BaseInputProvider
from aisc_plugin_interface.system_under_test import SYSTEM_INPUT, dataset_through_target, target_access_of

PID = "0f7c1e2a-aaaa-bbbb-cccc-1234567890ab"
KEY = "component:0b9c7a1e-0000-4000-8000-000000000002"
SCORER = {"name": "mcas-score", "label": "MCAS scorer", "kind": "rest", "method": "POST", "path": "/score",
          "headers": {}, "secret_header": "X-Auth-Token: {{secret}}", "body_template": "{{input}}",
          "response_path": "", "refusal": {"status": [502], "path": "detail.reason"}, "model": None,
          "timeout_s": 5, "secret": "s3cr3t", "updated_at": "2026-10-04T10:00:00Z",
          "target": {"key": KEY, "kind": "component", "label": "Scoring engine"}}
CSV = b"amount_eur,market\n2500,DE\n4000,FR\n100,NL\n"


class Frames(BaseInputProvider):
    def _read_data(self, file_content):
        return pd.read_csv(io.BytesIO(file_content))


def make(**declared):
    @dataset_through_target(datasets=("reference", "evaluated"), **declared)
    @evaluation_input(name="reference", label="Reference", input_provider_class=Frames, input_type=InputType.DATASET)
    @evaluation_input(name="evaluated", label="Evaluated", input_provider_class=Frames, input_type=InputType.DATASET)
    class Drift(BaseEvaluationPlugin):
        def evaluate(self, config_data):
            self.seen = {n: self.get_input_data(n) for n in ("reference", "evaluated")}
            return "done"

        def export_metrics(self, *a, **k):
            return []

    return Drift


def score(request):
    row = json.loads(request["body"])
    if row["market"] == "NL":
        return {"detail": {"reason": "outside the markets"}}                    # with status 502: a refusal
    return {"score": int(row["amount_eur"]) // 10, "recommendation": "Approve", "factors": [{"x": 1}]}


@pytest.fixture
def system(stub, monkeypatch):
    stub.route(f"/internal/projects/{PID}/targets/{KEY}/connection",
               (200, dict(SCORER, base_url=stub.base, allowed_hosts=[stub.host], denied_addresses=[])))
    stub.route("/score", (200, score))
    monkeypatch.setenv("PLATFORM_URL", stub.base)
    monkeypatch.setenv("PLATFORM_CONNECTIONS_TOKEN", "svc-token")
    return stub


def run(cls, stub, config=None, artifacts=None, target=True):
    t = cls()
    t._set_artifact_callback(lambda name, content: (artifacts if artifacts is not None else {}).__setitem__(name, content))
    if target:
        t.set_input_content(SYSTEM_INPUT, json.dumps({"value": f"target:{PID}/{KEY}"}).encode())
    t.set_input_content("reference", CSV)
    t.set_input_content("evaluated", CSV)
    return t, t.evaluate(config or {})


def test_it_is_its_own_declaration_with_the_target_input():
    Drift = make()
    assert target_access_of(Drift) == "dataset_through_target"
    assert SYSTEM_INPUT in {d.name for d in Drift().input_definitions}


def test_every_row_goes_through_the_target_and_its_answer_becomes_columns(system):
    stub = system
    stub.route("/score", (200, score))
    t, result = run(make(), stub)
    assert result == "done"
    for name in ("reference", "evaluated"):
        df = t.seen[name]
        assert list(df["amount_eur"]) == [2500, 4000, 100]                      # the inputs stay
        assert list(df["target.score"][:2]) == [250, 400]
        assert list(df["target.recommendation"][:2]) == ["Approve", "Approve"]
        assert "target.factors" not in df                                        # only scalar fields
    assert len([r for r in stub.seen if r["path"] == "/score"]) == 6
    assert all(r["headers"].get("x-auth-token") == "s3cr3t" for r in stub.seen if r["path"] == "/score")


def test_a_refusal_and_an_error_are_columns_not_a_crash(system):
    """One call at a time, in row order: the third reference row is refused (502 with a reason, the
    endpoint's refusal rule), the third evaluated row is rejected (422: an error, not retried)."""
    ok = (200, score)
    system.route("/score", ok, ok, (502, {"detail": {"reason": "outside the markets"}}), ok, ok,
                 (422, {"detail": "bad row"}))
    t, result = run(make(), system)
    assert result == "done"
    ref, ev = t.seen["reference"], t.seen["evaluated"]
    assert list(ref["target.refused"]) == [False, False, True]
    assert ref["target.refusal_reason"][2] == "outside the markets"
    assert ev["target.error"][2].startswith("MCAS scorer: the system answered 422")
    assert pd.isna(ev["target.error"][0])


def test_the_answers_are_saved_with_the_run_without_the_key(system):
    artifacts = {}
    run(make(), system, artifacts=artifacts)
    assert {"target-answers-reference.csv", "target-answers-evaluated.csv"} <= set(artifacts)
    assert all(b"s3cr3t" not in v for v in artifacts.values())


def test_the_row_limit_and_calls_at_once_come_from_the_run_settings(system):
    t, _ = run(make(), system, config={"target_row_limit": 2, "target_calls_at_once": 3})
    assert len(t.seen["reference"]) == 2
    assert len([r for r in system.seen if r["path"] == "/score"]) == 4


def test_without_a_target_the_run_is_refused_before_the_plugin_starts(system):
    Drift = make()
    t = Drift()
    t._set_artifact_callback(lambda n, c: None)
    t.set_input_content("reference", CSV)
    t.set_input_content("evaluated", CSV)
    with pytest.raises(ValueError, match="target"):
        t.evaluate({})
    assert not hasattr(t, "seen")

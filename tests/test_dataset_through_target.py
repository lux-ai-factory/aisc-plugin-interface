"""@dataset_through_target (the plan's R1, third kind): a plugin that analyses a table can have it sent, row by
row, through the evaluation's target first, so it analyses what the system answered, not only what went
in. Each scalar field of an answer becomes a column `target.<field>`; the plugin reads the enriched table as
if it had been uploaded so."""
import io
import json

import pytest

# the plugin these tests run reads its CSV with pandas, a table type the decorator takes; pandas is
# not a dependency of the library, so without it this module is skipped, saying why
pd = pytest.importorskip("pandas", reason="the tests of @dataset_through_target read their table with pandas")

from aisc_plugin_interface import BaseEvaluationPlugin, InputType, evaluation_input  # noqa: E402 (after importorskip)
from aisc_plugin_interface.input_providers.base_input_provider import BaseInputProvider  # noqa: E402 (after importorskip)
from aisc_plugin_interface.system_under_test import SYSTEM_INPUT, dataset_through_target, target_access_of  # noqa: E402 (after importorskip)

PID = "0f7c1e2a-aaaa-bbbb-cccc-1234567890ab"
KEY = "component:0b9c7a1e-0000-4000-8000-000000000002"
SCORER = {"name": "mcas-score", "label": "MCAS scorer", "kind": "rest", "method": "POST", "path": "/score",
          "headers": {}, "secret_header": "X-Auth-Token: {{secret}}", "body_template": "{{input}}",
          "response_path": "$", "refusal": {"status": [502], "path": "detail.reason"}, "model": None,
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


@pytest.fixture
def no_endpoint(stub, monkeypatch):
    stub.route(f"/internal/projects/{PID}/targets/{KEY}/connection",
               (404, {"detail": "Training data has no endpoint: set one under Manage, Targets and endpoints"}))
    monkeypatch.setenv("PLATFORM_URL", stub.base)
    monkeypatch.setenv("PLATFORM_CONNECTIONS_TOKEN", "svc-token")
    return stub


def test_optional_a_target_without_an_endpoint_runs_on_the_uploaded_data(no_endpoint):
    """Drift of a component that has no endpoint (training data) is drift of the uploads, as before."""
    t, result = run(make(required=False), no_endpoint)
    assert result == "done"
    assert list(t.seen["reference"].columns) == ["amount_eur", "market"]


def test_required_a_target_without_an_endpoint_is_refused_saying_why(no_endpoint):
    with pytest.raises(ValueError, match="has no endpoint"):
        run(make(), no_endpoint)


def test_optional_a_platform_that_does_not_answer_still_stops_the_run(stub, monkeypatch):
    monkeypatch.setenv("PLATFORM_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("PLATFORM_CONNECTIONS_TOKEN", "svc-token")
    with pytest.raises(Exception, match="could not be reached"):
        run(make(required=False), stub)


# ── code review and security review 2026-10-05 ─────────────────────────────

def test_optional_a_404_that_is_not_no_endpoint_still_stops_the_run(stub, monkeypatch):
    """Any 404 from the platform read as 'this target has no endpoint', so a wrong pid, a missing route
    or an older platform ran an optional plugin on the uploads without a word. Only the platform's
    no-endpoint answer does that now."""
    stub.route(f"/internal/projects/{PID}/targets/{KEY}/connection", (404, {"detail": "no target 'x'"}))
    monkeypatch.setenv("PLATFORM_URL", stub.base)
    monkeypatch.setenv("PLATFORM_CONNECTIONS_TOKEN", "svc-token")
    with pytest.raises(ValueError, match="no target"):
        run(make(required=False), stub)


def test_the_platforms_no_endpoint_reason_is_what_skips_an_optional_target(stub, monkeypatch):
    stub.route(f"/internal/projects/{PID}/targets/{KEY}/connection",
               (404, {"detail": "Training data: none", "reason": "no_endpoint"}))
    monkeypatch.setenv("PLATFORM_URL", stub.base)
    monkeypatch.setenv("PLATFORM_CONNECTIONS_TOKEN", "svc-token")
    t, result = run(make(required=False), stub)
    assert result == "done"


def test_an_unbound_target_input_is_no_target(system):
    """{'value': None} raised BadReference instead of the optional skip."""
    t = make(required=False)()
    t._set_artifact_callback(lambda n, c: None)
    t.set_input_content(SYSTEM_INPUT, json.dumps({"value": None}).encode())
    t.set_input_content("reference", CSV)
    t.set_input_content("evaluated", CSV)
    assert t.evaluate({}) == "done"


def test_a_connection_bound_directly_is_resolved_as_a_connection(stub, monkeypatch):
    """SYSTEM_INPUT accepts connection:<pid>/<name>, but the decorator parsed only target: values."""
    stub.route(f"/internal/projects/{PID}/connections/mcas-score",
               (200, dict(SCORER, base_url=stub.base, allowed_hosts=[stub.host], denied_addresses=[])))
    stub.route("/score", (200, score))
    monkeypatch.setenv("PLATFORM_URL", stub.base)
    monkeypatch.setenv("PLATFORM_CONNECTIONS_TOKEN", "svc-token")
    t = make()()
    t._set_artifact_callback(lambda n, c: None)
    t.set_input_content(SYSTEM_INPUT, json.dumps({"value": f"connection:{PID}/mcas-score"}).encode())
    t.set_input_content("reference", CSV)
    t.set_input_content("evaluated", CSV)
    assert t.evaluate({}) == "done"
    assert "target.score" in t.seen["reference"].columns


def test_calls_at_once_has_a_ceiling(system, monkeypatch):
    """A project member's target_calls_at_once was the thread pool's size as given (security review F4)."""
    from aisc_plugin_interface import system_under_test as sut

    sizes = []
    real = sut._through
    monkeypatch.setattr(sut, "_through", lambda client, table, at_once, limit: sizes.append(at_once)
                        or real(client, table, at_once, limit))
    run(make(), system, config={"target_calls_at_once": 5000})
    assert sizes and max(sizes) == sut.MAX_CALLS_AT_ONCE == 16


def test_a_json_declared_dataset_comes_back_as_rows_not_csv(system):
    """The enriched table went back through set_input_content as CSV whatever the input's provider was,
    so a JSON or Parquet dataset crashed after every row had been sent (code review 2026-10-05). The
    plugin gets its table back in the form it had it, with the target's columns."""
    from aisc_plugin_interface.input_providers.json_input_provider import JsonInputProvider

    @dataset_through_target(datasets=("rows",))
    @evaluation_input(name="rows", label="Rows", input_provider_class=JsonInputProvider, input_type=InputType.DATASET)
    class Rows(BaseEvaluationPlugin):
        def evaluate(self, config_data):
            self.seen = self.get_input_data("rows")
            return "done"

        def export_metrics(self, *a, **k):
            return []

    t = Rows()
    artifacts = {}
    t._set_artifact_callback(artifacts.__setitem__)
    t.set_input_content(SYSTEM_INPUT, json.dumps({"value": f"target:{PID}/{KEY}"}).encode())
    t.set_input_content("rows", json.dumps([{"amount_eur": 2500, "market": "DE"}]).encode())
    assert t.evaluate({}) == "done"
    assert isinstance(t.seen, list) and t.seen[0]["target.score"] == 250
    assert "target-answers-rows.csv" in artifacts

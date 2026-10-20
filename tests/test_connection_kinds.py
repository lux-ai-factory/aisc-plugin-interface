"""Connection kinds beyond openai and rest: an A2A agent (1.0, and the
0.3 dialect) and an Open Inference Protocol model server; and structured input for predictive systems."""
import json

import pytest

from aisc_plugin_interface import connections as c


def client(stub, **d):
    base = {"name": "sut", "label": "System", "base_url": stub.base, "timeout_s": 5, "secret": "k-1"}
    base.update(d)
    return c.EndpointClient(c.Descriptor.from_dict(base), allowed_hosts=[stub.host], sleep=lambda s: None)


def rpc(req):
    return json.loads(req["body"])


# a2a 1.0

def test_a2a_sends_sendmessage_with_a_user_text_part_and_reads_the_message(stub):
    stub.route("/rpc", (200, lambda r: {"jsonrpc": "2.0", "id": rpc(r)["id"],
                                         "result": {"message": {"messageId": "m2", "role": "ROLE_AGENT",
                                                                "parts": [{"text": "No, "}, {"text": "it is not."}]}}}))
    a = client(stub, kind="a2a", path="/rpc").ask("Is nationality an input?")
    req = stub.seen[-1]
    body = rpc(req)
    assert body["jsonrpc"] == "2.0" and body["method"] == "SendMessage"
    msg = body["params"]["message"]
    assert msg["role"] == "ROLE_USER" and msg["parts"] == [{"text": "Is nationality an input?"}] and msg["messageId"]
    assert req["headers"]["a2a-version"] == "1.0" and req["headers"]["authorization"] == "Bearer k-1"
    assert a.text == "No, it is not." and not a.refused


def test_a2a_a_completed_task_answers_with_its_artifacts(stub):
    stub.route("/rpc", (200, lambda r: {"jsonrpc": "2.0", "id": rpc(r)["id"], "result": {"task": {
        "id": "t1", "status": {"state": "TASK_STATE_COMPLETED"},
        "artifacts": [{"artifactId": "a1", "parts": [{"text": "Answer text"}]}]}}}))
    assert client(stub, kind="a2a", path="/rpc").ask("q").text == "Answer text"


def test_a2a_a_rejected_task_is_a_refusal(stub):
    stub.route("/rpc", (200, lambda r: {"jsonrpc": "2.0", "id": rpc(r)["id"], "result": {"task": {
        "id": "t1", "status": {"state": "TASK_STATE_REJECTED",
                               "message": {"messageId": "m", "role": "ROLE_AGENT", "parts": [{"text": "out of scope"}]}}}}}))
    a = client(stub, kind="a2a", path="/rpc").ask("q")
    assert a.refused and a.refusal_reason == "out of scope"


def test_a2a_a_working_task_is_polled_with_gettask_until_it_completes(stub):
    stub.route("/rpc",
               (200, lambda r: {"jsonrpc": "2.0", "id": rpc(r)["id"], "result": {"task": {"id": "t9", "status": {"state": "TASK_STATE_WORKING"}}}}),
               (200, lambda r: {"jsonrpc": "2.0", "id": rpc(r)["id"], "result": {"id": "t9", "status": {"state": "TASK_STATE_COMPLETED",
                   "message": {"messageId": "m", "role": "ROLE_AGENT", "parts": [{"text": "done"}]}}}}))
    a = client(stub, kind="a2a", path="/rpc").ask("q")
    assert a.text == "done"
    second = rpc(stub.seen[-1])
    assert second["method"] == "GetTask" and second["params"]["id"] == "t9"


@pytest.mark.parametrize("state", ["TASK_STATE_FAILED", "TASK_STATE_INPUT_REQUIRED", "TASK_STATE_AUTH_REQUIRED",
                                   "TASK_STATE_CANCELED"])
def test_a2a_other_end_states_are_bad_responses_naming_the_state(stub, state):
    stub.route("/rpc", (200, lambda r: {"jsonrpc": "2.0", "id": rpc(r)["id"], "result": {"task": {"id": "t", "status": {"state": state}}}}))
    with pytest.raises(c.EndpointBadResponse, match=state):
        client(stub, kind="a2a", path="/rpc").ask("q")


def test_a2a_a_jsonrpc_error_is_a_bad_response_with_its_message(stub):
    stub.route("/rpc", (200, lambda r: {"jsonrpc": "2.0", "id": rpc(r)["id"], "error": {"code": -32602, "message": "Invalid params"}}))
    with pytest.raises(c.EndpointBadResponse, match="Invalid params"):
        client(stub, kind="a2a", path="/rpc").ask("q")


def test_a2a_structured_input_goes_as_a_data_part(stub):
    stub.route("/rpc", (200, lambda r: {"jsonrpc": "2.0", "id": rpc(r)["id"], "result": {"message": {"messageId": "m", "role": "ROLE_AGENT",
                                                                                              "parts": [{"data": {"score": 712}}]}}}))
    a = client(stub, kind="a2a", path="/rpc").ask({"fico": 712})
    assert rpc(stub.seen[-1])["params"]["message"]["parts"] == [{"data": {"fico": 712}}]
    assert a.text == {"score": 712}


def test_a2a_refuses_a_history_because_the_agent_keeps_its_own_context(stub):
    with pytest.raises(c.EndpointError, match="history"):
        client(stub, kind="a2a", path="/rpc").ask("q", history=[{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}])


# a2a 0.3 dialect

def test_a2a_0_3_uses_message_send_and_kind_parts(stub):
    stub.route("/rpc", (200, lambda r: {"jsonrpc": "2.0", "id": rpc(r)["id"],
                                         "result": {"kind": "message", "messageId": "m", "role": "agent",
                                                    "parts": [{"kind": "text", "text": "old answer"}]}}))
    a = client(stub, kind="a2a", path="/rpc", protocol_version="0.3").ask("q")
    body = rpc(stub.seen[-1])
    assert body["method"] == "message/send"
    assert body["params"]["message"]["role"] == "user" and body["params"]["message"]["parts"] == [{"kind": "text", "text": "q"}]
    assert a.text == "old answer"


def test_a2a_0_3_rejected_and_working_tasks(stub):
    stub.route("/rpc",
               (200, lambda r: {"jsonrpc": "2.0", "id": rpc(r)["id"], "result": {"kind": "task", "id": "t", "status": {"state": "working"}}}),
               (200, lambda r: {"jsonrpc": "2.0", "id": rpc(r)["id"], "result": {"kind": "task", "id": "t", "status": {"state": "rejected",
                   "message": {"kind": "message", "role": "agent", "parts": [{"kind": "text", "text": "no"}]}}}}))
    a = client(stub, kind="a2a", path="/rpc", protocol_version="0.3").ask("q")
    assert a.refused and a.refusal_reason == "no" and rpc(stub.seen[-1])["method"] == "tasks/get"


# oip

def test_oip_sends_a_bytes_tensor_and_reads_the_first_output(stub):
    stub.route("/v2/models/scorer/infer", (200, {"model_name": "scorer", "id": "x",
                                               "outputs": [{"name": "output", "shape": [1], "datatype": "BYTES", "data": ["Approve"]}]}))
    a = client(stub, kind="oip", model="scorer").ask("applicant 42")
    body = json.loads(stub.seen[-1]["body"])
    assert body["inputs"] == [{"name": "input", "shape": [1], "datatype": "BYTES", "data": ["applicant 42"]}]
    assert a.text == "Approve"


def test_oip_a_request_template_sends_feature_tensors_and_a_path_reads_a_named_output(stub):
    stub.route("/v2/models/credit/infer", (200, {"model_name": "credit", "outputs": [
        {"name": "label", "shape": [1], "datatype": "BYTES", "data": ["Review"]},
        {"name": "score", "shape": [1], "datatype": "FP32", "data": [0.61]}]}))
    tpl = {"inputs": [{"name": "features", "shape": [1, 3], "datatype": "FP32", "data": "{{input}}"}]}
    a = client(stub, kind="oip", model="credit", body_template=tpl, response_path="outputs[1].data[0]").ask([712, 0.21, 7])
    assert json.loads(stub.seen[-1]["body"])["inputs"][0]["data"] == [712, 0.21, 7]
    assert a.text == 0.61


def test_oip_an_error_object_is_a_bad_response(stub):
    stub.route("/v2/models/scorer/infer", (400, {"error": "unexpected input shape"}))
    with pytest.raises(c.EndpointBadResponse, match="400"):
        client(stub, kind="oip", model="scorer").ask("x")


# structured input for templated REST

def test_rest_a_whole_value_placeholder_keeps_a_structured_input(stub):
    stub.route("/score", (200, {"recommendation": "Approve"}))
    a = client(stub, kind="rest", path="/score", body_template={"applicant": "{{input}}"},
               response_path="recommendation").ask({"fico_score": 712, "dti": 0.21})
    assert json.loads(stub.seen[-1]["body"]) == {"applicant": {"fico_score": 712, "dti": 0.21}}
    assert a.text == "Approve"


# A key header on every kind

@pytest.mark.parametrize("over", [{"kind": "a2a"}, {"kind": "a2a", "protocol_version": "0.3"}, {"kind": "oip", "model": "m"}])
def test_a_secret_header_is_sent_by_a2a_and_oip_and_replaces_the_bearer(over):
    d = c.Descriptor.from_dict({"name": "sut", "label": "System", "base_url": "https://sut.example", "secret": "k-1",
                                "secret_header": "X-API-Key: {{secret}}", **over})
    _, _, headers, _ = c.render_request(d, "q", None, {})
    assert headers.get("X-API-Key") == "k-1"
    assert "Authorization" not in headers


# Odd answers of an A2A agent are bad responses, not crashes

WORKING = {"task": {"id": "t9", "status": {"state": "TASK_STATE_WORKING"}}}


@pytest.mark.parametrize("payload", [
    {"error": "boom"},
    {"result": {"task": "t1"}},
    {"result": {"task": {"id": "t", "status": "done"}}},
    {"result": {"task": {"id": "t", "status": {"state": "TASK_STATE_REJECTED", "message": "no"}}}},
    {"result": {"task": {"id": "t", "status": {"state": "TASK_STATE_COMPLETED"}, "artifacts": ["x"]}}},
    {"result": {"task": {"id": "t", "status": {"state": "TASK_STATE_COMPLETED"}, "artifacts": "x"}}},
])
def test_a2a_a_malformed_answer_is_a_bad_response(stub, payload):
    stub.route("/rpc", (200, lambda r: {"jsonrpc": "2.0", "id": rpc(r)["id"], **payload}))
    with pytest.raises(c.EndpointBadResponse):
        client(stub, kind="a2a", path="/rpc").ask("q")


# `parts` that is not a list raised TypeError (an int) out of call(), which ended the whole
# dataset run, or was read as no answer (a dict, a string). Each is now a bad response, like any other.
@pytest.mark.parametrize("result", [
    {"parts": 5},
    {"message": {"parts": 7}},
    {"status": {"state": "completed", "message": {"parts": 3}}},
    {"message": {"parts": {"text": "hello"}}},
    {"kind": "message", "parts": "hello"},
    {"task": {"id": "t", "status": {"state": "TASK_STATE_COMPLETED"}, "artifacts": [{"parts": {"text": "x"}}]}},
], ids=["int", "message-int", "status-message-int", "message-dict", "kind-message-str", "artifact-dict"])
def test_a2a_parts_that_are_not_a_list_are_a_bad_response(stub, result):
    stub.route("/rpc", (200, lambda r: {"jsonrpc": "2.0", "id": rpc(r)["id"], "result": result}))
    with pytest.raises(c.EndpointBadResponse, match="parts"):
        client(stub, kind="a2a", path="/rpc").ask("q")


def test_a2a_missing_parts_are_still_no_answer_text(stub):
    """Only a present `parts` must be a list: a task completed with no status message still answers."""
    done = {"task": {"id": "t", "status": {"state": "TASK_STATE_COMPLETED"},
                     "artifacts": [{"parts": [{"text": "fine"}]}]}}
    stub.route("/rpc", (200, lambda r: {"jsonrpc": "2.0", "id": rpc(r)["id"], "result": done}))
    assert client(stub, kind="a2a", path="/rpc").ask("q").text == "fine"


def test_a2a_a_poll_answer_that_is_not_json_is_a_bad_response(stub):
    stub.route("/rpc", (200, lambda r: {"jsonrpc": "2.0", "id": rpc(r)["id"], "result": WORKING}),
               (200, b"<html>busy</html>"))
    with pytest.raises(c.EndpointBadResponse, match="polling"):
        client(stub, kind="a2a", path="/rpc").ask("q")


def test_a2a_a_dropped_connection_while_polling_is_an_endpoint_error(stub, monkeypatch):
    import http.client

    stub.route("/rpc", (200, lambda r: {"jsonrpc": "2.0", "id": rpc(r)["id"], "result": WORKING}))
    real, calls = c._OPENER.open, []

    def open_(req, timeout=None):
        calls.append(req)
        if len(calls) > 1:
            raise http.client.RemoteDisconnected("gone")
        return real(req, timeout=timeout)

    monkeypatch.setattr(c._OPENER, "open", open_)
    with pytest.raises(c.EndpointError, match="polling"):
        client(stub, kind="a2a", path="/rpc").ask("q")

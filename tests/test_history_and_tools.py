"""An agent's answer may be tool calls instead of text (``Answer.tool_calls``), and a history may carry the
full OpenAI messages of such a turn (an assistant's tool calls, the tool's result). The history and the
params are checked and size-capped before anything is sent, the same way in a direct call and through
the platform."""
import json

import pytest

from aisc_plugin_interface import connections as c

from test_endpoint_client_facade import ASK_PATH, OK, RUN_KEYS, asks, facade, issued  # noqa: F401 (facade: fixture)
from test_endpoint_client_facade import client as facade_client

CALL = {"id": "call_1", "type": "function", "function": {"name": "get_balance", "arguments": "{\"iban\": \"LU12\"}"}}
CHAT = "/v1/chat/completions"
MCAS = {
    "name": "mcas-chat", "label": "MCAS chat", "kind": "rest", "base_url": "", "method": "POST",
    "path": "/chat", "headers": {"X-Client": "aisc"}, "secret_header": "Authorization: Bearer {{secret}}",
    "body_template": {"question": "{{input}}", "history": "{{history}}"},
    "response_path": "answer", "refusal": None, "model": None, "timeout_s": 5, "secret": "s3cr3t-value",
    "updated_at": "2026-09-29T10:00:00Z",
}
OPENAI = dict(kind="openai", path="", model="gpt-4o-mini", body_template=None, response_path=None, refusal=None,
              secret_header=None, headers={})


def desc(stub, **over):
    d = dict(MCAS, base_url=stub.base)
    d.update(over)
    return c.Descriptor.from_dict(d)


def openai_desc(stub):
    return desc(stub, **dict(OPENAI, base_url=stub.base + "/v1"))


def client(stub, **over):
    return c.EndpointClient(desc(stub, **over), allowed_hosts=[stub.host], sleep=lambda s: None)


def openai_client(stub):
    return client(stub, **dict(OPENAI, base_url=stub.base + "/v1"))


def answered(message):
    return (200, {"choices": [{"message": message, "finish_reason": "stop"}]})


HIST = [{"role": "system", "content": "You are a bank agent."},
        {"role": "user", "content": "My balance?"},
        {"role": "assistant", "content": None, "tool_calls": [CALL]},
        {"role": "tool", "tool_call_id": "call_1", "content": "{\"balance\": 10}"}]


# tool calls in the answer

def test_an_openai_tool_call_with_null_content(stub):
    stub.route(CHAT, (200, {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [CALL]},
                                         "finish_reason": "tool_calls"}]}))
    answer = openai_client(stub).ask("My balance?")
    assert answer.text is None
    assert answer.tool_calls == [CALL]
    assert answer.refused is False
    assert answer.status == 200


def test_a_text_answer_has_no_tool_calls(stub):
    stub.route(CHAT, (200, {"choices": [{"message": {"content": "hi"}}]}))
    answer = openai_client(stub).ask("q")
    assert answer.text == "hi"
    assert answer.tool_calls is None
    stub.route(CHAT, answered({"content": "hi", "tool_calls": []}))
    assert openai_client(stub).ask("q").tool_calls is None


def test_a_tool_call_without_a_content_key(stub):
    stub.route(CHAT, answered({"tool_calls": [CALL]}))
    answer = openai_client(stub).ask("q")
    assert answer.text is None
    assert answer.tool_calls == [CALL]
    stub.route(CHAT, answered({}))
    with pytest.raises(c.EndpointBadResponse, match=r"no answer at 'choices\[0\]\.message\.content'"):
        openai_client(stub).ask("q")


@pytest.mark.parametrize("calls", ["x", ["x"], {"id": 1}])
def test_tool_calls_that_are_not_calls_are_a_bad_response(stub, calls):
    stub.route(CHAT, answered({"content": None, "tool_calls": calls}))
    with pytest.raises(c.EndpointBadResponse, match="tool_calls are not a list of calls"):
        openai_client(stub).ask("q")


def test_tools_go_into_the_openai_body(stub):
    stub.route(CHAT, answered({"content": "ok"}))
    tools = [{"type": "function", "function": {"name": "get_balance", "parameters": {"type": "object"}}}]
    openai_client(stub).ask("q", tools=tools, tool_choice="auto")
    body = json.loads(stub.seen[-1]["body"])
    assert body["tools"] == tools
    assert body["tool_choice"] == "auto"


# full OpenAI messages in the history

def test_a_history_with_a_tool_turn_round_trips(stub):
    stub.route(CHAT, answered({"content": "You're welcome."}))
    openai_client(stub).ask("Thanks", history=HIST)
    assert json.loads(stub.seen[-1]["body"])["messages"] == HIST + [{"role": "user", "content": "Thanks"}]
    stub.route("/chat", (200, {"answer": "ok"}))
    client(stub).ask("Thanks", history=HIST)
    assert json.loads(stub.seen[-1]["body"])["history"] == HIST


def test_an_assistant_message_as_openai_returns_it_is_accepted():
    for message in ({"role": "assistant", "content": "ok", "refusal": None},
                    {"role": "assistant", "content": None, "tool_calls": [CALL], "refusal": None}):
        assert c.check_history([message]) == [message]


FIELDS = "(role, content, name, tool_calls, tool_call_id, refusal)"
BAD_HISTORIES = [
    ({"role": "user"}, "history: a list of messages"),
    (["hi"], "history[0]: a message is an object with a role"),
    ([{"role": "developer", "content": "x"}], "history[0]: role 'developer' is not one of system, user, assistant, tool"),
    ([{"role": "user", "content": "x", "extra": 1}], f"history[0]: 'extra' is not a message field {FIELDS}"),
    ([{"role": "user", "content": None}], "history[0]: a user message needs text content"),
    ([{"role": "user", "content": [{"type": "text", "text": "x"}]}], "history[0]: a user message needs text content"),
    ([{"role": "user", "content": "x", "name": 3}], "history[0]: name is text"),
    ([{"role": "assistant", "content": None}], "history[0]: an assistant message needs text content or tool_calls"),
    ([{"role": "assistant", "content": None, "tool_calls": [{"id": "c", "type": "function", "function": {"name": "f"}}]}],
     "history[0]: tool_calls[0] needs an id, type 'function' and a function with a name and arguments text"),
    ([{"role": "user", "content": "x", "tool_calls": [CALL]}], "history[0]: only an assistant message carries tool_calls"),
    ([{"role": "tool", "content": "x"}], "history[0]: a tool message needs a tool_call_id and text content"),
    ([{"role": "user", "content": "x", "tool_call_id": "c"}], "history[0]: only a tool message carries a tool_call_id"),
    ([{"role": "user", "content": "x", "refusal": None}], f"history[0]: 'refusal' is not a message field {FIELDS}"),
    ([{"role": "user", "content": "a"}, "b"], "history[1]: a message is an object with a role"),
]


@pytest.mark.parametrize("history, message", BAD_HISTORIES)
def test_a_bad_history_is_refused_before_any_request(stub, history, message):
    stub.route(CHAT, answered({"content": "ok"}))
    with pytest.raises(c.EndpointError) as raised:
        openai_client(stub).ask("q", history=history)
    assert isinstance(raised.value, c.BadRequest)
    assert str(raised.value) == message
    assert stub.seen == []


def test_the_history_caps():
    short = [{"role": "user", "content": "x"}] * 200
    assert c.check_history(short) == short
    with pytest.raises(c.BadRequest) as raised:
        c.check_history(short + [{"role": "user", "content": "x"}])
    assert str(raised.value) == "history: 201 messages, more than 200"
    big = [{"role": "user", "content": "x" * 1_048_576}]
    with pytest.raises(c.BadRequest) as raised:
        c.check_history(big)
    assert str(raised.value) == f"history: {len(json.dumps(big).encode())} bytes as JSON, more than 1048576"


def test_the_params_cap(stub):
    stub.route(CHAT, answered({"content": "ok"}))
    with pytest.raises(c.EndpointError) as raised:
        openai_client(stub).ask("q", tools=["x" * 262_144])
    assert isinstance(raised.value, c.BadRequest)
    assert raised.match(r"^params: \d+ bytes as JSON, more than 262144$")
    with pytest.raises(c.EndpointError) as raised:
        openai_client(stub).ask("q", when=object())
    assert isinstance(raised.value, c.BadRequest)
    assert raised.match("^params: not JSON")
    assert stub.seen == []


def test_call_checks_the_history_too(stub):
    with pytest.raises(c.EndpointError) as raised:
        c.call(desc(stub), "q", history=[{"role": "bot", "content": "x"}], allowed_hosts=[stub.host])
    assert isinstance(raised.value, c.BadRequest)
    assert stub.seen == []


# through the platform

def test_a_facade_answer_carries_tool_calls(facade):
    facade.route(RUN_KEYS, (201, issued(facade, capabilities=("text", "history", "tools"))))
    facade.route(ASK_PATH, (200, {"output": None, "refused": False, "refusal_reason": None, "status": 200,
                                  "latency_ms": 3, "tool_calls": [CALL]}))
    answer = facade_client().ask("My balance?")
    assert answer.tool_calls == [CALL]
    assert answer.text is None
    facade.route(ASK_PATH, (200, {"output": None, "refused": False, "refusal_reason": None, "status": 200,
                                  "latency_ms": 3, "tool_calls": "x"}))
    with pytest.raises(c.EndpointBadResponse, match="tool_calls are not a list of calls"):
        facade_client().ask("My balance?")


def test_a_history_through_the_platform_is_sent_as_given(facade):
    hist = HIST + [{"role": "assistant", "content": "Your balance is 10."}, {"role": "user", "content": "Thanks"}]
    assert len(hist) == 6
    facade.route(ASK_PATH, (200, OK))
    cl = facade_client()
    cl.ask("Anything else?", history=hist)
    (req,) = asks(facade)
    assert json.loads(req["body"])["history"] == hist
    with pytest.raises(c.EndpointError) as raised:
        cl.ask("q", history=[{"role": "bot", "content": "x"}])
    assert isinstance(raised.value, c.BadRequest)
    assert len(asks(facade)) == 1


def test_an_answer_built_positionally_keeps_its_fields():
    answer = c.Answer("t", False, None, 200, 5)
    assert answer.tool_calls is None
    assert answer.latency_ms == 5


# the tool loop goes on after a tool result: no new user message

TOOL_TURN = [{"role": "user", "content": "My balance?"},
             {"role": "assistant", "content": None, "tool_calls": [CALL]},
             {"role": "tool", "tool_call_id": "call_1", "content": "{\"balance\": 10}"}]


def test_after_tool_results_no_input_sends_the_history_as_is(stub):
    _, _, _, body = c.render_request(openai_desc(stub), None, TOOL_TURN, {})
    assert json.loads(body)["messages"] == TOOL_TURN
    stub.route(CHAT, answered({"content": "Your balance is 10."}))
    assert openai_client(stub).ask(None, history=TOOL_TURN).text == "Your balance is 10."
    assert json.loads(stub.seen[-1]["body"])["messages"] == TOOL_TURN


@pytest.mark.parametrize("history", [[{"role": "user", "content": "My balance?"}],
                                     [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]])
def test_no_input_needs_a_history_that_ends_with_tool_results(stub, history):
    stub.route(CHAT, answered({"content": "ok"}))
    with pytest.raises(c.EndpointError) as raised:
        openai_client(stub).ask(None, history=history)
    assert isinstance(raised.value, c.BadRequest)
    assert str(raised.value) == "input is required unless the history ends with tool results"
    assert stub.seen == []


def test_an_input_after_a_tool_result_is_a_new_user_message(stub):
    stub.route(CHAT, answered({"content": "ok"}))
    openai_client(stub).ask("x", history=TOOL_TURN)
    assert json.loads(stub.seen[-1]["body"])["messages"] == TOOL_TURN + [{"role": "user", "content": "x"}]

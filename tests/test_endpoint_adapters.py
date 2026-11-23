"""The standard endpoint adapters, ported from the backend CON-01 adapters."""
from __future__ import annotations

import base64
from dataclasses import dataclass, field

from aisc_plugin_interface.adapters import BUILTIN_ENDPOINT_ADAPTERS, Turn
from aisc_plugin_interface.adapters.base import Secret, json_path, render, secret_headers
from aisc_plugin_interface.adapters.endpoint import (
    A2aEndpointAdapter,
    A2aEndpointConfig,
    OpenAiEndpointAdapter,
    OipEndpointAdapter,
    RestEndpointAdapter,
    RestEndpointConfig,
    agent_card_url,
)


@dataclass
class Target:
    url: str = ""
    model: str = ""
    protocol: str = ""


def _adapter(cls, **config):
    return cls(config or None)


def test_openai_is_a_passthrough():
    adapter = _adapter(OpenAiEndpointAdapter)
    assert adapter.passthrough is True
    assert adapter.speaks == "openai-chat"
    assert "openai-chat" in adapter.serves


def test_openai_rewrite_swaps_the_key_and_keeps_the_path():
    adapter = _adapter(OpenAiEndpointAdapter)
    request = adapter.rewrite(
        Target(url="https://llm.example.com/v1"),
        Secret(key="sk-real"),
        path="/v1/chat/completions",
        method="post",
        headers={"Authorization": "Bearer run-token", "X-Trace": "1"},
        json_body={"model": "gpt"},
    )
    assert request.method == "POST"
    assert request.url == "https://llm.example.com/v1/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer sk-real"
    assert request.headers["X-Trace"] == "1"
    assert request.json_body == {"model": "gpt"}


def test_openai_test_builds_a_chat_completion():
    adapter = _adapter(OpenAiEndpointAdapter)
    request = adapter.build_request(Target(url="https://llm.example.com/v1", model="gpt-4o-mini"), Secret(key="sk"), turn=Turn(input="hello"))
    assert request.url == "https://llm.example.com/v1/chat/completions"
    assert request.json_body["model"] == "gpt-4o-mini"
    assert request.json_body["messages"][-1] == {"role": "user", "content": "hello"}
    assert request.headers["Authorization"] == "Bearer sk"


def test_openai_reads_content_and_capabilities():
    adapter = _adapter(OpenAiEndpointAdapter)
    body = '{"choices": [{"message": {"content": "hi", "tool_calls": [{"id": "1"}]}, "logprobs": {"a": 1}}]}'
    answer = adapter.parse_response(200, body, Target())
    assert answer.ok and answer.answer == "hi"
    assert answer.capabilities["tools"] and answer.capabilities["logprobs"]


def test_rest_builds_a_templated_request():
    adapter = _adapter(
        RestEndpointAdapter,
        path="/chat",
        method="post",
        body_template={"question": "{{input}}", "history": "{{history}}"},
        headers={"X-Client": "aisc"},
    )
    request = adapter.build_request(
        Target(url="https://system.example.com/"),
        Secret(key="k1", headers={"X-Token": "t"}),
        turn=Turn(input="why?", history=[{"role": "user", "content": "hi"}]),
    )
    assert request.method == "POST"
    assert request.url == "https://system.example.com/chat"
    assert request.json_body["question"] == "why?"
    assert request.json_body["history"] == [{"role": "user", "content": "hi"}]
    assert request.headers["X-Client"] == "aisc"
    assert request.headers["Authorization"] == "Bearer k1"
    assert request.headers["X-Token"] == "t"


def test_rest_reads_the_answer_at_the_response_path():
    adapter = _adapter(RestEndpointAdapter, response_path="data.items[0].text")
    answer = adapter.parse_response(200, '{"data": {"items": [{"text": "the answer"}]}}', Target())
    assert answer.ok and answer.answer == "the answer"


def test_rest_refusal_is_not_an_error():
    adapter = _adapter(
        RestEndpointAdapter,
        response_path="detail.reason",
        refusal={"status": [502], "path": "detail.reason", "match": {"detail.error": "answer_not_grounded"}},
    )
    answer = adapter.parse_response(502, '{"detail": {"error": "answer_not_grounded", "reason": "ungrounded"}}', Target())
    assert answer.ok and answer.refused and answer.refusal_reason == "ungrounded"


def test_rest_non_matching_is_an_error():
    adapter = _adapter(RestEndpointAdapter, response_path="answer")
    answer = adapter.parse_response(500, "upstream exploded", Target())
    assert not answer.ok and answer.error == "http_status"


def test_a2a_builds_a_message_send():
    adapter = _adapter(A2aEndpointAdapter, path="/a2a", protocol_version="1.0")
    request = adapter.build_request(Target(url="https://agent.example.com"), Secret(), turn=Turn(input="hi"))
    assert request.url == "https://agent.example.com/a2a"
    assert request.json_body["method"] == "message/send"
    assert request.json_body["params"]["message"]["parts"][0]["text"] == "hi"


def test_a2a_reads_artifact_text_and_reports_rpc_errors():
    adapter = _adapter(A2aEndpointAdapter)
    body = '{"jsonrpc": "2.0", "result": {"artifacts": [{"parts": [{"kind": "text", "text": "42"}]}]}}'
    assert adapter.parse_response(200, body, Target()).answer == "42"
    error = '{"jsonrpc": "2.0", "error": {"code": -32601, "message": "no such method"}}'
    answer = adapter.parse_response(200, error, Target())
    assert not answer.ok and answer.error == "rpc_error" and answer.detail == "no such method"


def test_the_agent_card_url_follows_the_version():
    assert agent_card_url("https://agent.example.com/", "1.0") == "https://agent.example.com/.well-known/agent-card.json"
    assert agent_card_url("https://agent.example.com", "0.3") == "https://agent.example.com/.well-known/agent.json"


def test_oip_builds_and_decodes():
    adapter = _adapter(OipEndpointAdapter)
    request = adapter.build_request(Target(url="https://kserve.example.com", model="m1"), Secret(), turn=Turn(input="ping"))
    assert request.url == "https://kserve.example.com/v2/models/m1/infer"
    assert request.json_body["inputs"][0]["datatype"] == "BYTES"
    encoded = base64.b64encode(b"decoded answer").decode()
    answer = adapter.parse_response(200, '{"outputs": [{"datatype": "BYTES", "data": ["%s"]}]}' % encoded, Target())
    assert answer.ok and answer.answer == "decoded answer"


def test_helpers_render_and_json_path():
    assert render({"a": "{{params.x}}"}, {"params": {"x": {"id": 7}}}) == {"a": {"id": 7}}
    assert json_path({"a": {"b": [1, {"c": "x"}]}}, "a.b[1].c") == "x"
    assert json_path({"a": 1}, "a.b.c") is None
    assert secret_headers(Secret(key="abc"), secret_header="X-Token: {{secret}}") == {"X-Token": "abc"}


def test_the_endpoint_library_serves_four_protocols():
    assert {a.id for a in BUILTIN_ENDPOINT_ADAPTERS} == {"openai", "oip", "rest", "a2a"}

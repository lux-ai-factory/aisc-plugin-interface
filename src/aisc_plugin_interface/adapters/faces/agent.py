"""The `agent` face: an A2A JSON-RPC request/response <-> a Turn/Answer.

A plugin that speaks `agent` sends `message/send` JSON-RPC to the door; the face
turns it into a Turn and the Answer back into a JSON-RPC result. One turn per
call: no context id is carried (stateful sessions are later work, ADR 0009).
"""
from __future__ import annotations

import json
import uuid
from typing import Any

from aisc_plugin_interface.adapters.base import Answer, Turn


def parse_agent_request(body: dict[str, Any]) -> Turn:
    params = body.get("params") or {}
    message = params.get("message") or {}
    text = ""
    for part in message.get("parts") or []:
        if part.get("text") is not None:
            text += part["text"]
    return Turn(input=text, history=[], params={"message_id": message.get("messageId")})


def _result_text(answer: Answer) -> str:
    if answer.refused:
        return answer.refusal_reason or "the agent refused the request"
    if isinstance(answer.answer, str):
        return answer.answer
    return json.dumps(answer.answer)


def render_agent_response(answer: Answer, *, request_id: str | None = None) -> dict[str, Any]:
    """An :class:`Answer` -> an A2A `message/send` JSON-RPC response."""
    if not answer.ok and not answer.refused:
        return {
            "jsonrpc": "2.0",
            "id": request_id or str(uuid.uuid4()),
            "error": {"code": -32603, "message": answer.detail or answer.error or "no answer"},
        }
    return {
        "jsonrpc": "2.0",
        "id": request_id or str(uuid.uuid4()),
        "result": {
            "kind": "task",
            "artifacts": [{"parts": [{"kind": "text", "text": _result_text(answer)}]}],
        },
    }


__all__ = ["parse_agent_request", "render_agent_response"]

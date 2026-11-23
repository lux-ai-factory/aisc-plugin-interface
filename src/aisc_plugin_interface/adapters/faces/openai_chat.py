"""The `openai-chat` face: an OpenAI-compatible request/response <-> a Turn/Answer.

A plugin that speaks `openai-chat` (most LLM plugins) sends a chat-completions
request to the door. The face turns it into a
:class:`~aisc_plugin_interface.adapters.base.Turn` for the adapter, and the
adapter's :class:`~aisc_plugin_interface.adapters.base.Answer` back into a
chat-completions response.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any

from aisc_plugin_interface.adapters.base import Answer, Turn


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    return "" if content is None else str(content)


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return str(value)


def parse_chat_request(body: dict[str, Any]) -> Turn:
    """A chat-completions body -> the last user turn plus the earlier messages."""
    messages = list(body.get("messages") or [])
    history = [
        {"role": m.get("role", "user"), "content": _content_text(m.get("content"))}
        for m in messages[:-1]
    ]
    last = messages[-1] if messages else {}
    params = {
        key: body[key]
        for key in ("temperature", "max_tokens", "top_p", "tools", "tool_choice", "stop")
        if key in body
    }
    return Turn(input=_content_text(last.get("content")), history=history, params=params)


def render_chat_response(
    answer: Answer, *, model: str = "", request_id: str | None = None
) -> dict[str, Any]:
    """An :class:`Answer` -> an OpenAI chat-completion response body."""
    if answer.refused:
        content = answer.refusal_reason or "the system refused the request"
    elif answer.ok:
        content = _stringify(answer.answer)
    else:
        content = answer.detail or answer.error or "the system returned no answer"
    return {
        "id": f"chatcmpl-{request_id or uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def render_models(models: list[str]) -> dict[str, Any]:
    return {
        "object": "list",
        "data": [{"id": name, "object": "model", "owned_by": "aisc"} for name in models],
    }


__all__ = ["parse_chat_request", "render_chat_response", "render_models"]

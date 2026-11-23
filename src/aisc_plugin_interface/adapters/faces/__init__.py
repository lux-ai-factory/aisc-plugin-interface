"""Face translation: the protocol a plugin speaks <-> the adapter-neutral Turn/Answer."""
from .agent import parse_agent_request, render_agent_response
from .openai_chat import parse_chat_request, render_chat_response, render_models

#: face protocol id -> (parse request, render response)
FACES = {
    "openai-chat": (parse_chat_request, render_chat_response),
    "agent": (parse_agent_request, render_agent_response),
}

__all__ = [
    "parse_agent_request",
    "render_agent_response",
    "parse_chat_request",
    "render_chat_response",
    "render_models",
    "FACES",
]

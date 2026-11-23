"""Standard endpoint adapters: a target's protocol -> the protocol a plugin speaks."""
from .standard import (
    AGENT_CARD_PATH,
    BUILTIN_ENDPOINT_ADAPTERS,
    A2aEndpointAdapter,
    A2aEndpointConfig,
    OipEndpointAdapter,
    OipEndpointConfig,
    OpenAiEndpointAdapter,
    OpenAiEndpointConfig,
    RestEndpointAdapter,
    RestEndpointConfig,
    agent_card_url,
)

__all__ = [
    "BUILTIN_ENDPOINT_ADAPTERS",
    "OpenAiEndpointAdapter",
    "OpenAiEndpointConfig",
    "OipEndpointAdapter",
    "OipEndpointConfig",
    "RestEndpointAdapter",
    "RestEndpointConfig",
    "A2aEndpointAdapter",
    "A2aEndpointConfig",
    "agent_card_url",
    "AGENT_CARD_PATH",
]

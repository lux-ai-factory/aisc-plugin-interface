from pydantic import BaseModel, ConfigDict


class EndpointConfig(BaseModel):
    """The endpoint a plugin receives for a non-LLM endpoint slot.

    ``url`` is the run door's local URL (never the target's real URL) and
    ``token_key`` is the project config key whose value is the per-run token. The
    real key stays with the door. An ``openai-chat`` slot keeps the
    :class:`~aisc_plugin_interface.models.llm.LLMConfig` shape instead (D-16).
    """

    model_config = ConfigDict(extra="forbid")

    url: str = ""
    protocol: str = ""
    model: str = ""
    token_key: str = ""

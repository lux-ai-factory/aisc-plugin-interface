from pydantic import BaseModel, ConfigDict


class LLMConfig(BaseModel):
    """Defines the ``json_value`` of an ``llm`` AIComponent.

    ``endpoint_url`` and ``secret_key`` (the project config key holding the API
    key) are the endpoint configuration. ``model`` is intentionally left empty
    on the component; it is filled in at evaluation/plugin run with the
    per-run selection.
    """

    model_config = ConfigDict(extra="forbid")

    endpoint_url: str = ""
    secret_key: str = ""
    model: str = ""

from pydantic import BaseModel, ConfigDict


class ResourceConfig(BaseModel):
    """Defines the ``json_value`` of a ``resource`` AIComponent.

    Deliberately generic: a single string reference (e.g. a HuggingFace model
    id such as ``"user/repo"``) for a plugin to resolve.
    """

    model_config = ConfigDict(extra="forbid")

    value: str = ""

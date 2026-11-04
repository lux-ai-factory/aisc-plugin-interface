import enum

from pydantic import BaseModel


class InputType(str, enum.Enum):
    MODEL = "model"
    DATASET = "dataset"
    LLM = "llm"
    DATASHAPE = "datashape"
    RESOURCE = "resource"


class InputDefinition(BaseModel):
    """Maps an `InputDefinition` to an `AIComponent` of the assessed AISystem.

    ``input_types`` and ``output_type`` are optional advisory hints used by
    the frontend to detect potentially incompatible adapter chains.
    """

    name: str
    label: str
    input_type: InputType
    required: bool = True
    input_types: list[str] | None = None
    output_type: str | None = None

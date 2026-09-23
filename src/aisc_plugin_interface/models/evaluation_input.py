import enum

from pydantic import BaseModel


class InputType(str, enum.Enum):
    MODEL = "model"
    DATASET = "dataset"
    LLM = "llm"
    DATASHAPE = "datashape"
    RESOURCE = "resource"


class InputDefinition(BaseModel):
    """Maps an `InputDefinition` to an `AIComponent` of the assessed AISystem."""

    name: str
    label: str
    input_type: InputType
    required: bool = True

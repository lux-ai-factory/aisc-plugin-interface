import enum

from pydantic import BaseModel


class InputType(str, enum.Enum):
    MODEL = "model"
    DATASET = "dataset"
    LLM = "llm"
    REST = "rest"
    DATASHAPE = "datashape"


class InputDefinition(BaseModel):
    """Maps an `InputDefinition` to an `AIComponent` of the assessed AISystem."""

    name: str
    label: str
    input_type: InputType
    required: bool = True

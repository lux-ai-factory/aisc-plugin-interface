import enum

from pydantic import BaseModel, Field


class InputType(str, enum.Enum):
    MODEL = "model"
    DATASET = "dataset"
    LLM = "llm"
    DATASHAPE = "datashape"
    RESOURCE = "resource"


class InputRole(str, enum.Enum):
    """What the thing bound to a slot is, in the run.

    ``subject`` is the system under test, ``instrument`` a judge/grader the plugin
    calls, ``input`` any other data or model it uses.
    """

    SUBJECT = "subject"
    INSTRUMENT = "instrument"
    INPUT = "input"


class InputDefinition(BaseModel):
    """Maps a plugin slot to an `AIComponent` of the project (ADR 0009 D-14).

    The binding (which component, which adapter) is chosen at evaluation time.
    ``accepts`` lists the value types (data) or face protocols (endpoint) this
    slot can use; ``role`` says what the bound component is; ``adapter`` names the
    default adapter and ``adapter_required`` locks the choice. The fields default,
    so definitions serialised before this contract still parse.
    """

    name: str
    label: str
    input_type: InputType
    required: bool = True
    accepts: list[str] = Field(default_factory=list)
    role: str = InputRole.INPUT.value
    adapter: str | None = None
    adapter_required: bool = False
    env: dict[str, str] = Field(default_factory=dict)

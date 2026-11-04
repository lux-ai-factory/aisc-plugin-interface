from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, ClassVar, TypeVar, Generic

T = TypeVar("T")


class BaseInputAdapter(ABC, Generic[T]):
    """Converts an evaluation input into the shape a plugin provider expects.

    The first adapter in a chain receives the raw input file.
    Every subsequent adapter receives the previous adapter's output.

    Adapters may optionally declare type hints used by the UI to flag
    potentially incompatible chains. ``input_types`` is a list of expected
    input patterns (``*.csv``, ``pd.dataframe``); ``output_type`` is the
    concrete type produced by the adapter. Hints are advisory only.
    """

    input_types: ClassVar[list[str] | None] = None
    output_type: ClassVar[str | None] = None

    @abstractmethod
    def adapt(self, data: bytes | Path | Any) -> T:
        raise NotImplementedError

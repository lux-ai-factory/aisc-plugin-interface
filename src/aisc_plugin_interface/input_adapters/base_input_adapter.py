from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, TypeVar, Generic

T = TypeVar("T")


class BaseInputAdapter(ABC, Generic[T]):
    """Converts an evaluation input into the shape a plugin provider expects.

    The first adapter in a chain receives the raw input file.
    Every subsequent adapter receives the previous adapter's output.
    """

    @abstractmethod
    def adapt(self, data: bytes | Path | Any) -> T:
        raise NotImplementedError

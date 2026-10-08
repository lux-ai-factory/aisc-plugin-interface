from pathlib import Path
from abc import ABC, abstractmethod
from typing import TypeVar, Generic

T = TypeVar("T")


class BaseInputAdapter(ABC, Generic[T]):
    @abstractmethod
    def adapt(self, file_content: bytes | Path) -> T:
        raise NotImplementedError

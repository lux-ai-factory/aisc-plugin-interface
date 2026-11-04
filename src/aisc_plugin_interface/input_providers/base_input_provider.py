from pathlib import Path
from abc import ABC, abstractmethod
from typing import ClassVar, TypeVar, Generic

T = TypeVar("T")


class BaseInputProvider(ABC, Generic[T]):
    """
    Abstract base class for data loaders used by plugins.
    Input providers handle the transformation of raw bytes (from S3/Storage)
    into a structured format that the plugin can process.

    Providers may optionally declare type hints used by the UI to flag
    potentially incompatible adapter chains. ``input_types`` lists the
    accepted raw formats or adapter outputs (``*.csv``, ``pd.dataframe``);
    ``output_type`` is the concrete type handed to the plugin. Hints are
    advisory only.
    """

    input_types: ClassVar[list[str] | None] = None
    output_type: ClassVar[str | None] = None

    def __init__(self, file_content: bytes | Path | T):
        """
        Initializes the provider and triggers the data reading process.
        :param file_content: Raw bytes of the file to be processed.
        """
        self._data = self._read_data(file_content)

    @abstractmethod
    def _read_data(self, file_content: bytes | Path | T) -> T:
        """
        Parses raw bytes into an object (e.g., list, dict, DataFrame, ONNX session etc).
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def get_data(self) -> T:
        """
        Returns the parsed data stored in the provider.
        """
        return self._data

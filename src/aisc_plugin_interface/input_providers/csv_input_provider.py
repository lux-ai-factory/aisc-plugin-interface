import csv
import io
from pathlib import Path

from .base_input_provider import BaseInputProvider


class CsvInputProvider(BaseInputProvider[list[dict]]):
    """
    A concrete implementation of BaseInputProvider for CSV files.
    Parses the file content into a list of dictionaries, where each dict represents a row.
    """

    def _read_data(self, file_content: bytes | Path | list[dict]) -> list[dict]:
        """
        Converts CSV bytes (or a path to a CSV file) into a list of dictionaries.
        """
        if isinstance(file_content, list):
            return file_content
        if isinstance(file_content, Path):
            file_content = file_content.read_bytes()
        file_stream = io.BytesIO(file_content)
        wrapper = io.TextIOWrapper(file_stream, encoding="utf-8")
        reader = csv.DictReader(wrapper)
        return list(reader)

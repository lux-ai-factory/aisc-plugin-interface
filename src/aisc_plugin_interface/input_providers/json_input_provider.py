import json

from .base_input_provider import BaseInputProvider


class JsonInputProvider(BaseInputProvider):
    """
    Parses JSON file bytes into a dictionary.

    Used for components whose config is serialised as JSON in ``json_value``
    (datashape, llm, resource).
    """

    def _read_data(self, file_content: bytes) -> dict:
        return json.loads(file_content.decode("utf-8"))

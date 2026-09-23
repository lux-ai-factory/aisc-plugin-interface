import io

from .base_input_provider import BaseInputProvider


class ParquetInputProvider(BaseInputProvider):
    """
    Parses Parquet file bytes into a pandas DataFrame.

    `pandas` is imported lazily so plugins that don't use parquet files don't
    need it installed.
    """

    def _read_data(self, file_content: bytes):
        import pandas as pd

        return pd.read_parquet(io.BytesIO(file_content))

import io

from .base_input_provider import BaseInputProvider


class OnnxInputProvider(BaseInputProvider):
    """
    Parses a serialised ONNX model's bytes into an `onnxruntime.InferenceSession`,
    the object a plugin would expect for inference.

    `onnxruntime` is imported lazily so plugins that don't run models don't need
    it installed.
    """

    def _read_data(self, file_content: bytes):
        import onnxruntime as ort

        return ort.InferenceSession(io.BytesIO(file_content))

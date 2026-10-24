"""One model to score rows with, wherever it lives: what ``self.assessed_model()`` returns.

    @assesses(card=model_session(outputs={"label": "decision", "probability": "risk.p"}), local_model=onnx())
    @evaluation_input(name="model", label="Model", input_provider_class=OnnxInputProvider, input_type=InputType.MODEL)
    class ScoringPlugin(BaseEvaluationPlugin):
        def evaluate(self, config_data):
            model = self.assessed_model()
            labels, proba = model.predict(rows), model.predict_proba(rows)

Both sessions take ``rows`` as a list of dicts (feature name -> value) and return plain Python lists:
``predict`` one label per row, ``predict_proba`` one list of class probabilities per row.

- A local model file: ``OnnxSession`` over the ONNX the project uploaded (its bytes, or the
  ``onnxruntime.InferenceSession`` an ONNX input provider made of them). It reads its outputs the way the
  catalogue plugins' ``OnnxModelSession`` does: the probability output is the first whose name says
  prob (else the second output), a label is the more probable class: labels are class indices only
  when the model has a probability output. It also takes what that session
  takes, a pandas DataFrame or a numpy array of rows in the model's feature order; a list of dicts is
  read in the first row's key order.
- An AI card target: ``EndpointSession`` sends each row once, in order, through the target's
  connection (the row as the request's ``{{input}}``) and reads the declared ``outputs`` paths in the
  answer: ``label`` (when not declared: the more probable class, 0 or 1) and ``probability`` (of the
  positive class; ``predict_proba`` gives ``[1-p, p]``). Binary classification only: ``outputs["classes"]``
  above 2 makes ``assessed_model()`` refuse a card target. Without a probability path ``predict_proba``
  raises; with only a probability path ``predict`` returns 0 or 1, the more probable class.
"""
from __future__ import annotations

import json
from typing import Any, Protocol, runtime_checkable

from aisc_plugin_interface.connections import EndpointBadResponse, EndpointError, _scrub, extract

#: what ``model_session(outputs=...)`` may declare
OUTPUT_KEYS = ("label", "probability", "classes")


@runtime_checkable
class ModelSession(Protocol):
    def predict(self, rows) -> list: ...

    def predict_proba(self, rows) -> list[list[float]]: ...


def check_outputs(outputs: dict) -> dict:
    """The ``outputs`` a model_session declaration may give; ValueError otherwise."""
    outputs = dict(outputs)
    unknown = [k for k in outputs if k not in OUTPUT_KEYS]
    if unknown:
        raise ValueError(f"outputs: {', '.join(map(repr, unknown))} unknown (known: {', '.join(OUTPUT_KEYS)})")
    if not outputs.get("label") and not outputs.get("probability"):
        raise ValueError("outputs: declare where the answer carries the label, the probability or both")
    classes = outputs.get("classes")
    if classes is not None and (isinstance(classes, bool) or not isinstance(classes, int) or classes < 2):
        raise ValueError(f"outputs: classes is the number of classes, a whole number from 2: {classes!r}")
    return outputs


class ModelOutputMissing(EndpointBadResponse):
    """The target's answer to a row lacks a declared output, or carries one the session cannot read."""


class RowRefused(EndpointError):
    """The target refused to answer a row."""


class EndpointSession:
    """A card target's model, reached through its connection (see the module docstring). The session
    keeps the answers it got: a row (the same values) is sent once, however often it is scored."""

    def __init__(self, client, outputs: dict):
        self.client = client
        self.outputs = check_outputs(outputs)
        self._answered: dict[str, Any] = {}
        descriptor = getattr(client, "descriptor", None)
        self._secret = getattr(descriptor, "secret", None)

    def predict(self, rows) -> list:
        if not self.outputs.get("label"):
            return [int(p[1] > p[0]) for p in self.predict_proba(rows)]
        labels = []
        for n, answer in self._answers(rows):
            label = self._output(answer, n, "label")
            if label is None or isinstance(label, (dict, list)):
                raise ModelOutputMissing(f"row {n}: the label at {self.outputs['label']!r} is"
                                         f" {'empty' if label is None else 'not a single value'}")
            labels.append(label)
        return labels

    def predict_proba(self, rows) -> list[list[float]]:
        if not self.outputs.get("probability"):
            raise ValueError("outputs: this model session declares no probability output")
        proba = []
        for n, answer in self._answers(rows):
            p = self._output(answer, n, "probability")
            if isinstance(p, bool) or not isinstance(p, (int, float)) or not 0 <= p <= 1:
                said = _scrub(str(p), self._secret)[:100]
                raise ModelOutputMissing(f"row {n}: the probability at {self.outputs['probability']!r} is"
                                         f" not a number from 0 to 1: {said!r}")
            proba.append([1.0 - float(p), float(p)])
        return proba

    def _answers(self, rows):
        """(row number from 1, the answer's value) for each row, in order; a row not answered yet is sent."""
        from aisc_plugin_interface.system_under_test import _sent

        for n, row in enumerate(rows, start=1):
            try:
                sent = {k: _sent(v) for k, v in dict(row).items()}
                key = json.dumps(sent, sort_keys=True, default=str)
                if key not in self._answered:
                    self._answered[key] = self.client.ask(sent)
            except EndpointError as exc:
                raise type(exc)(f"row {n}: {exc}", status=exc.status) from None
            except Exception as exc:
                try:
                    named = type(exc)(f"row {n}: {exc}")
                except Exception:
                    named = RuntimeError(f"row {n}: {type(exc).__name__}: {exc}")
                raise named from exc
            answer = self._answered[key]
            if answer.refused:
                reason = _scrub(str(answer.refusal_reason), self._secret)
                raise RowRefused(f"row {n}: the target refused it: {reason}", status=answer.status)
            yield n, answer.text

    def _output(self, answer, n: int, name: str) -> Any:
        path = self.outputs[name]
        try:
            return extract(answer, path)
        except KeyError:
            raise ModelOutputMissing(f"row {n}: the target's answer has no {path!r} (the declared {name}"
                                     " output)") from None


class OnnxSession:
    """A local ONNX model (see the module docstring); onnxruntime and numpy are imported when it is made."""

    def __init__(self, model):
        if isinstance(model, (bytes, bytearray)):
            import onnxruntime

            model = onnxruntime.InferenceSession(bytes(model))
        self.session = model
        inputs, outputs = model.get_inputs(), model.get_outputs()
        self._input = inputs[0]
        self._output_names = [o.name for o in outputs]
        self._probability = next(
            (i for i, o in enumerate(outputs) if "prob" in o.name.lower()), 1 if len(outputs) >= 2 else None)

    def predict(self, rows) -> list:
        import numpy as np

        if len(rows) == 0:
            return []
        if self._probability is not None:
            return [int(i) for i in np.argmax(self._proba(rows), axis=1)]
        labels = np.asarray(self._run(rows)[0])
        if labels.ndim > 1 and labels.shape[-1] == 1:
            labels = labels.squeeze(-1)
        return labels.tolist()

    def predict_proba(self, rows) -> list[list[float]]:
        if len(rows) == 0:
            return []
        return self._proba(rows).tolist()

    def _proba(self, rows):
        import numpy as np

        raw = self._run(rows)[self._probability if self._probability is not None else 0]
        if isinstance(raw, list) and raw and isinstance(raw[0], dict):           # a ZipMap output
            keys = sorted(raw[0])
            return np.array([[r.get(k, 0.0) for k in keys] for r in raw], dtype=np.float32)
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim == 1:
            arr = np.column_stack((1.0 - arr, arr))
        elif arr.ndim == 2 and arr.shape[1] == 1:
            arr = np.column_stack((1.0 - arr[:, 0], arr[:, 0]))
        if arr.ndim == 2 and arr.shape[1] == 2 and np.allclose(arr.sum(axis=1), 0, atol=1e-4):
            # binary scores centred on 0 (zipmap=False): column 1 is the positive class's probability
            positive = np.clip(arr[:, 1:2], 0, 1)
            arr = np.concatenate([1.0 - positive, positive], axis=1)
        elif not (arr.ndim == 2 and np.all(arr >= 0) and np.all(arr <= 1)
                  and np.allclose(arr.sum(axis=1), 1.0, atol=1e-4)):
            arr = np.exp(arr - np.max(arr, axis=1, keepdims=True))
            arr = arr / np.sum(arr, axis=1, keepdims=True)
        return arr.astype(np.float32, copy=False)

    def _run(self, rows):
        return self.session.run(self._output_names, {self._input.name: self._array(rows)})

    def _array(self, rows):
        import numpy as np

        if hasattr(rows, "to_numpy"):                                            # a pandas DataFrame
            arr = rows.to_numpy()
        elif isinstance(rows, np.ndarray):
            arr = rows
        elif isinstance(rows, list) and all(isinstance(r, dict) for r in rows):
            features = list(rows[0]) if rows else []
            for n, row in enumerate(rows, start=1):
                missing = [f for f in features if f not in row]
                if missing:
                    raise ValueError(f"row {n}: no {', '.join(map(repr, missing))} (the first row's features)")
            arr = np.array([[row[f] for f in features] for row in rows])
        else:
            raise ValueError(f"rows: a list of dicts, a pandas DataFrame or a numpy array, not {type(rows).__name__}")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return np.ascontiguousarray(arr, dtype=_NUMPY_TYPES.get(self._input.type, "float32"))


_NUMPY_TYPES = {"tensor(float)": "float32", "tensor(double)": "float64", "tensor(float16)": "float16",
                "tensor(int64)": "int64", "tensor(int32)": "int32", "tensor(int16)": "int16",
                "tensor(int8)": "int8", "tensor(uint8)": "uint8", "tensor(bool)": "bool"}


def session_for(plugin, outputs: dict | None = None) -> ModelSession:
    """The model the run assesses, for ``plugin.assessed_model()``: its local model file when the run
    bound one, its card target when the run bound that; never one in place of the other. ``outputs``, when
    given, replaces the declared card outputs for this session (validated like a declaration, also when
    the session is local, which ignores it)."""
    from aisc_plugin_interface.connections import BadReference, EndpointClient, EndpointNotFound
    from aisc_plugin_interface.system_under_test import SYSTEM_INPUT
    from aisc_plugin_interface.targets import is_target

    cls = type(plugin)
    if outputs is not None:
        outputs = check_outputs(outputs)
    card = getattr(cls, "target_access", None) == "model_session"
    local_input = getattr(cls, "local_model_input", None)
    if not card and local_input is None:
        raise ValueError(f"{cls.__name__} declares no model to score: declare @assesses(card=model_session(...))"
                         " or @assesses(local_model=onnx(...))")
    local = plugin.get_input_data(local_input) if local_input is not None else None
    payload = plugin.get_input_data(SYSTEM_INPUT) if card else None
    target = payload.get("value") if isinstance(payload, dict) else payload
    if local is not None and target is not None:
        raise ValueError("the run binds both a local model file and an AI card target: a run assesses one of them")
    if local is not None:
        return OnnxSession(local)
    if target is None:
        raise ValueError("the run binds no model: neither a local model file"
                         f"{f' in input {local_input!r}' if local_input else ''} nor an AI card target")
    if outputs is None:
        outputs = cls.model_session_outputs
    if int(outputs.get("classes") or 2) > 2:
        raise ValueError(f"a card target is scored as a binary classifier; this plugin declares"
                         f" {outputs['classes']} classes")
    try:
        client = (EndpointClient.for_target(plugin, SYSTEM_INPUT) if is_target(target)
                  else EndpointClient.for_input(plugin, SYSTEM_INPUT))
    except (EndpointNotFound, BadReference) as exc:
        raise ValueError(f"target: {exc}") from None
    return EndpointSession(client, outputs)

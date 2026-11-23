from typing import Sequence, Type

from aisc_plugin_interface.input_providers.base_input_provider import BaseInputProvider
from aisc_plugin_interface.input_providers.json_input_provider import JsonInputProvider
from aisc_plugin_interface.models.evaluation_input import (
    InputDefinition,
    InputRole,
    InputType,
)

_JSON_INPUT_TYPES = {InputType.LLM, InputType.DATASHAPE, InputType.RESOURCE}

#: input_type -> (accepts, role) when nothing more specific is given (D-15).
_SHORTHAND: dict[InputType, tuple[list[str], str]] = {
    InputType.LLM: (["openai-chat"], InputRole.SUBJECT.value),
    InputType.DATASET: (["records"], InputRole.INPUT.value),
    InputType.MODEL: (["onnx-session"], InputRole.INPUT.value),
    InputType.DATASHAPE: (["json"], InputRole.INPUT.value),
    InputType.RESOURCE: (["resource"], InputRole.INPUT.value),
}

#: The value type an input provider produces, so the shorthand can be precise.
_PROVIDER_VALUE_TYPES = {
    "CsvInputProvider": "records",
    "ParquetInputProvider": "table",
    "OnnxInputProvider": "onnx-session",
    "JsonInputProvider": "json",
}


def evaluation_input(
    name: str,
    label: str,
    input_provider_class: Type[BaseInputProvider] | None = None,
    input_type: InputType = InputType.MODEL,
    required: bool = True,
    *,
    accepts: Sequence[str] | None = None,
    role: str | InputRole | None = None,
    adapter: str | None = None,
    adapter_required: bool = False,
    env: dict[str, str] | None = None,
):
    """Declare a plugin input slot (ADR 0009 D-14).

    ``input_provider_class`` may be omitted for JSON-backed component types
    (llm, datashape, resource); it then defaults to :class:`JsonInputProvider`.
    It is required for file-backed types (dataset, model).

    ``accepts`` names the value types (data) or face protocols (endpoint) the
    slot will use; when omitted it is derived from ``input_type`` and, for a known
    provider, from what that provider produces. ``role`` is ``subject``,
    ``instrument`` or ``input``. ``adapter`` names the default adapter and
    ``adapter_required`` makes it the only choice. ``env`` maps environment
    variable names to templates (``{base_url}``, ``{api_key}``, ``{model}``) that
    the runtime fills for the plugin process.
    """

    if input_provider_class is None:
        if input_type not in _JSON_INPUT_TYPES:
            raise ValueError(
                f"input_provider_class is required for input_type {input_type}"
            )
        input_provider_class = JsonInputProvider

    resolved_accepts: list[str]
    if accepts is not None:
        resolved_accepts = list(accepts)
    else:
        resolved_accepts = list(_SHORTHAND[input_type][0])
        # A file-backed slot's value type is whatever its provider produces.
        if input_type in (InputType.DATASET, InputType.MODEL):
            produced = _PROVIDER_VALUE_TYPES.get(input_provider_class.__name__)
            if produced:
                resolved_accepts = [produced]

    resolved_role = role.value if isinstance(role, InputRole) else role
    resolved_role = resolved_role or _SHORTHAND[input_type][1]

    def decorator(cls):
        if "_input_definitions" not in cls.__dict__:
            cls._input_definitions = []
        if "_input_provider_types" not in cls.__dict__:
            cls._input_provider_types = {}

        if not any(d.name == name for d in cls._input_definitions):
            cls._input_definitions.append(
                InputDefinition(
                    name=name,
                    label=label,
                    input_type=input_type,
                    required=required,
                    accepts=resolved_accepts,
                    role=resolved_role,
                    adapter=adapter,
                    adapter_required=adapter_required,
                    env=dict(env or {}),
                )
            )
        cls._input_provider_types[name] = input_provider_class
        return cls

    return decorator

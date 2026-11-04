from typing import Type

from aisc_plugin_interface.input_providers.base_input_provider import BaseInputProvider
from aisc_plugin_interface.input_providers.json_input_provider import JsonInputProvider
from aisc_plugin_interface.models.evaluation_input import InputDefinition, InputType

_JSON_INPUT_TYPES = {InputType.LLM, InputType.DATASHAPE, InputType.RESOURCE}


def evaluation_input(
    name: str,
    label: str,
    input_provider_class: Type[BaseInputProvider] | None = None,
    input_type: InputType = InputType.MODEL,
    required: bool = True,
    input_types: list[str] | None = None,
    output_type: str | None = None,
):
    """
    Decorator to create input definitions and their provider.

    ``input_provider_class`` may be omitted for JSON-backed component types
    (llm, datashape, resource); it then defaults to :class:`JsonInputProvider`.
    It is required for file-backed types (dataset, model).

    ``input_types`` and ``output_type`` are optional advisory hints for the
    evaluation UI. When omitted, values declared on the provider class are
    used.
    """

    if input_provider_class is None:
        if input_type not in _JSON_INPUT_TYPES:
            raise ValueError(
                f"input_provider_class is required for input_type {input_type}"
            )
        input_provider_class = JsonInputProvider

    resolved_input_types = input_types
    if resolved_input_types is None:
        resolved_input_types = getattr(input_provider_class, "input_types", None)
    if resolved_input_types is not None:
        resolved_input_types = list(resolved_input_types)

    resolved_output_type = output_type
    if resolved_output_type is None:
        resolved_output_type = getattr(input_provider_class, "output_type", None)

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
                    input_types=resolved_input_types,
                    output_type=resolved_output_type,
                )
            )
        cls._input_provider_types[name] = input_provider_class
        return cls

    return decorator

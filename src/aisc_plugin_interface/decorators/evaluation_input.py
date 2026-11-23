from typing import Sequence

from aisc_plugin_interface.models.evaluation_input import (
    InputDefinition,
    InputRole,
    InputType,
)

#: input_type -> (accepts, role) when nothing more specific is given (ADR 0009 D-15).
_SHORTHAND: dict[InputType, tuple[list[str], str]] = {
    InputType.LLM: (["openai-chat"], InputRole.SUBJECT.value),
    InputType.DATASET: (["records"], InputRole.INPUT.value),
    InputType.MODEL: (["onnx-session"], InputRole.INPUT.value),
    InputType.DATASHAPE: (["json"], InputRole.INPUT.value),
    InputType.RESOURCE: (["resource"], InputRole.INPUT.value),
}


def evaluation_input(
    name: str,
    label: str,
    input_type: InputType = InputType.MODEL,
    required: bool = True,
    *,
    accepts: Sequence[str] | None = None,
    role: str | InputRole | None = None,
    adapter: str | None = None,
    adapter_required: bool = False,
    env: dict[str, str] | None = None,
):
    """Declare a plugin input slot (ADR 0009 D-14 / D-22).

    A slot names the **value types** it uses with ``accepts``; the runtime
    resolves a data adapter for the component's format and hands the produced
    value to :meth:`BaseEvaluationPlugin.get_input_data`. There is one data
    mechanism — adapters (standard or plugin-shipped); input providers were
    removed (D-22).

    ``accepts`` defaults from ``input_type``: ``records`` for a dataset, ``table``
    for a dataset a plugin wants as a DataFrame (set it explicitly), ``onnx-session``
    for a model, ``json``/``resource`` for the JSON-backed kinds, ``openai-chat``
    for an endpoint. ``role`` is ``subject``, ``instrument`` or ``input``;
    ``adapter`` names the default adapter and ``adapter_required`` makes it the
    only choice; ``env`` maps environment variables to templates the runtime fills.
    """
    resolved_accepts = (
        list(accepts) if accepts is not None else list(_SHORTHAND[input_type][0])
    )
    resolved_role = role.value if isinstance(role, InputRole) else role
    resolved_role = resolved_role or _SHORTHAND[input_type][1]

    def decorator(cls):
        if "_input_definitions" not in cls.__dict__:
            cls._input_definitions = []
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
        return cls

    return decorator

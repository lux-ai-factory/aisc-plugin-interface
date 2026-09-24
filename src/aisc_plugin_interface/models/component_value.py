from typing import Any

from aisc_plugin_interface.models.datashape import DataShape
from aisc_plugin_interface.models.llm import LLMConfig
from aisc_plugin_interface.models.resource import ResourceConfig

_COMPONENT_MODELS: dict[str, type] = {
    "datashape": DataShape,
    "llm": LLMConfig,
    "resource": ResourceConfig,
}


def parse_component_value(component_type: str, json_value: Any) -> Any:
    """
    Parse a component's ``json_value`` into the pydantic model that defines its
    type. The model is selected by the AIComponent type.

    Returns the raw value untouched when no typed model is registered for the
    component type or parsing fails.
    """
    model_cls = _COMPONENT_MODELS.get(component_type)
    if model_cls is None or json_value is None:
        return json_value
    try:
        return model_cls.model_validate(json_value)
    except Exception:
        return json_value

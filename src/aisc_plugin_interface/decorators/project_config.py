from aisc_plugin_interface.models.project_config_definition import (
    ProjectConfigDefinition,
    ConfigCategory,
    ConfigValueType,
)


def project_config(
        key: str,
        name: str,
        category: ConfigCategory,
        value_type: ConfigValueType | None = None,
        required: bool = True,
):
    def decorator(cls):
        if "_project_config_definitions" not in cls.__dict__:
            cls._project_config_definitions = []
        if not any(definition.key == key for definition in cls._project_config_definitions):
            cls._project_config_definitions.append(
                ProjectConfigDefinition(
                    key=key,
                    name=name,
                    category=category,
                    value_type=value_type,
                    required=required,
                )
            )
        return cls

    return decorator

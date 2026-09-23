import enum

from pydantic import BaseModel, ConfigDict, field_validator


class ConfigCategory(str, enum.Enum):
    SECRETS = "secrets"
    VARIABLES = "variables"


class ConfigValueType(str, enum.Enum):
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    JSON = "json"


class ProjectConfigDefinition(BaseModel):
    """Describes one of a plugin's required project configs.

    Maps to a project-level `ProjectConfig` (a secret, a variable, an API
    endpoint, or a datashape) that the platform stores on the project.
    """

    model_config = ConfigDict(extra="forbid")

    key: str
    name: str
    category: ConfigCategory
    value_type: ConfigValueType | None = None
    required: bool = True

    @field_validator("key")
    @classmethod
    def valid_name(cls, value: str) -> str:
        if not value or not value.replace("_", "").isalnum() or not value[0].isalpha():
            raise ValueError("setting key must start with a letter and contain only letters, numbers, and underscores")
        return value

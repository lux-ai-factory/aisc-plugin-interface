from typing import Any

from pydantic import BaseModel

from aisc_plugin_interface import BaseEvaluationPlugin, Measure, metric


# Define the configuration form schema
class ConfigFormSchema(BaseModel):
    pass


class {{ plugin_name }}(BaseEvaluationPlugin[ConfigFormSchema]):
    def evaluate(self, config_data: dict) -> Any:
        pass

    @metric("my-metric")
    def my_metric(self, evaluation_output: Any) -> list[Measure]:
        pass

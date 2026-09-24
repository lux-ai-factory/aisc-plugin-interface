# AISC Plugin Interface

This repository provides the base interface for writing evaluation plugins.

If you are new to the platform, start here:

- [Plugin Developer Guide](PLUGIN_DEVELOPER_GUIDE.md)

That guide covers:

- the plugin contract and discovery model
- how to create a plugin project
- how to implement configuration, inputs, metrics, and progress reporting
- optional integration hooks and current extension points
- common failure modes and architectural feedback areas

## Quick Start

Create a new plugin project:

```bash
mkdir my-aisc-plugin
cd my-aisc-plugin
uv init --lib --python 3.12
uv add aisc-plugin-interface
```

Implement a plugin class that inherits from `BaseEvaluationPlugin[T]`, then export it from your package `__init__.py`.

Minimal example:

```python
from typing import Any

from pydantic import BaseModel, Field

from aisc_plugin_interface import BaseEvaluationPlugin, Measure, metric


class ConfigFormSchema(BaseModel):
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class MyPlugin(BaseEvaluationPlugin[ConfigFormSchema]):
    def evaluate(self, config_data: dict) -> Any:
        # Dependencies only used during evaluation should be imported locally here
        import numpy as np

        # Access configuration form data
        config: ConfigFormSchema = self.validate_config_form_data(config_data)
        threshold: float = config.threshold
        
        # Your evaluation logic here

        # Use self.logger for logging
        self.logger.info("Evaluation completed successfully")
        
        return {"MyMetric": [0.99, 0.5, 0.67], "OtherMetric": [0.01, 0.22, 0.77]}


    @metric("MyMetric")
    def my_metric(self, evaluation_output: Any) -> list[Measure]:
        if not evaluation_output:
            return []
        values = evaluation_output.get("MyMetric", [])
        return [Measure(name="MyMetric", score=value) for value in values]
```

Export it:

```python
from .plugin import MyPlugin

__all__ = ["MyPlugin"]
```

For the full development workflow, use the guide above.

---

##  License

This project is licensed under the [Apache License 2.0](LICENSE).  
© 2024–2026 Université du Luxembourg and Luxembourg Institute of Science and Technology (LIST). Originally developed within the SerVal Research Group and the Interdisciplinary Centre for Security, Reliability and Trust (SnT).

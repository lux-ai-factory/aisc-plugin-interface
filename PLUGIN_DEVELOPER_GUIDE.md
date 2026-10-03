# Plugin Developer Guide

This guide is for developers who write evaluation plugins for AISC. It describes the contract a plugin
implements, how the platform finds and runs it, and the optional hooks.

## 1. What a plugin does

Evaluation logic lives outside the core application, in plugin packages loaded at runtime. A plugin:

- defines its configuration as typed data (a Pydantic model);
- declares the inputs it needs (datasets, models, LLMs, data shapes, resources) and parses them;
- runs the evaluation;
- emits structured measurements;
- optionally reports progress, uploads artifacts and adjusts its form while the user fills it in.

## 2. Architecture

Four parts are involved:

- a **plugin project**, which contains one or more plugin classes;
- this **interface package**, which defines the base class and the shared models;
- the **plugin loader** ([aisc-plugin-manager](https://github.com/lux-ai-factory/aisc-plugin-manager)),
  which finds plugin packages and imports them;
- two **host runtimes**: the execution engine imports a plugin to read its form, inputs and metrics;
  the evaluation worker imports it again to run an evaluation.

Because the engine imports the module only to inspect it, import heavy dependencies inside
`evaluate`, not at module level.

## 3. Discovery

The loader looks in a plugin folder (`PLUGIN_PATH` of the engine and the worker, mounted at
`/app/plugins`) and, separately, lists the packages of the stack's package index (devpi). In the
plugin folder, every subfolder with a `pyproject.toml` is a candidate. It is accepted when:

- `pyproject.toml` has a `[project]` `name` and `version`;
- one of its dependencies is `aisc-plugin-interface`;
- the code is in a folder named after the package name with dashes turned into underscores, either
  under `src/` or at the project root;
- that module imports without error and exposes at least one non-abstract subclass of
  `BaseEvaluationPlugin`.

`src` layout:

```text
my-aisc-plugin/
├── pyproject.toml          name = "my-aisc-plugin"
└── src/
    └── my_aisc_plugin/
        ├── __init__.py
        └── plugin.py
```

Flat layout:

```text
my-aisc-plugin/
├── pyproject.toml
└── my_aisc_plugin/
    ├── __init__.py
    └── plugin.py
```

The package exports its plugin classes from `__init__.py`. The loader registers each class under its
Python class name; the name shown in the UI is `plugin_name` when set, otherwise the class name. The
project folder name does not have to match the package name. A local package is listed under its
version with the local label `+local`, so it never shadows a version of the same package on the index.

## 4. Creating a plugin project

```bash
mkdir my-aisc-plugin && cd my-aisc-plugin
uv init --lib
uv add aisc-plugin-interface
uv run aisc-plugin-interface init-plugin
```

`init-plugin` asks for a class name and a file path, writes a plugin template into
`src/<package>/` and adds the class to `__init__.py`. `--force` overwrites an existing file.

PyPI's `aisc-plugin-interface` 0.3.0 has no connection client (section 13). A plugin that calls a
system under test installs the `feat/unified-modules` branch instead:

```bash
uv add git+https://github.com/lux-ai-factory/aisc-plugin-interface --branch feat/unified-modules
```

To try the plugin in a running AISC stack, put the project folder in the stack's plugin folder.

## 5. The contract

Every plugin inherits from `BaseEvaluationPlugin[T]`, where `T` is the Pydantic model of its
configuration form.

- `evaluate(config_data)`: the evaluation; required.
- `validate_config_form_data(config_data)`: validates the configuration against `T`.
- `@metric("...")`: marks a method that turns the result of `evaluate` into `list[Measure]`.
- `export_metrics(output)`: called by the runtime; runs every `@metric` method and joins the results.
- `@evaluation_input(...)`: declares an input and the provider class that parses it.
- `get_input_data(name)`: the parsed value of an input, or `None` when it was not provided.
- `@project_config(...)`, `get_project_setting(key)`, `get_secret(key)`: project-level settings and
  secrets the plugin needs.
- `progress_bar(...)` and `report_progress(TaskProgress(...))`: progress reporting.
- `upload_artifact(name, content)`: store a file with the run.
- `self.logger`: a logger named after the plugin class.

A run goes: the runtime sets the inputs and project settings, calls `evaluate` with the
configuration, then calls `export_metrics` with what `evaluate` returned.

## 6. Minimal example

This plugin reads a CSV dataset and computes two aggregate metrics.

`src/my_aisc_plugin/plugin.py`:

```python
from typing import Any

from pydantic import BaseModel, Field

from aisc_plugin_interface import BaseEvaluationPlugin, InputType, Measure, evaluation_input, metric
from aisc_plugin_interface.input_providers.csv_input_provider import CsvInputProvider


class ConfigSchema(BaseModel):
    score_column: str = Field(..., description="Name of the CSV column containing numeric scores.")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0,
                             description="Scores greater than or equal to this value count as passing.")


@evaluation_input(name="dataset", label="Dataset", input_provider_class=CsvInputProvider,
                  input_type=InputType.DATASET, required=True)
class ExampleCsvPlugin(BaseEvaluationPlugin[ConfigSchema]):
    plugin_name = "Example CSV plugin"

    def evaluate(self, config_data: dict) -> Any:
        config = self.validate_config_form_data(config_data)
        rows = self.get_input_data("dataset")
        if rows is None:
            raise ValueError("This plugin requires a dataset file")

        scores: list[float] = []
        for row in self.progress_bar(rows, desc="Scoring rows"):
            try:
                scores.append(float(row.get(config.score_column)))
            except (TypeError, ValueError):
                continue

        passing = [s for s in scores if s >= config.threshold]
        return {
            "average_score": sum(scores) / len(scores) if scores else 0.0,
            "pass_rate": len(passing) / len(scores) if scores else 0.0,
        }

    @metric("Average score")
    def average_score_metric(self, evaluation_output: dict) -> list[Measure]:
        return [Measure(name="Average score", score=float(evaluation_output["average_score"]))]

    @metric("Pass rate")
    def pass_rate_metric(self, evaluation_output: dict) -> list[Measure]:
        return [Measure(name="Pass rate", score=float(evaluation_output["pass_rate"]), unit="ratio")]
```

`src/my_aisc_plugin/__init__.py`:

```python
from .plugin import ExampleCsvPlugin

__all__ = ["ExampleCsvPlugin"]
```

## 7. Configuration

The configuration is a Pydantic model. The engine sends its JSON schema to the web form
(`get_config_form_schema`), and `validate_config_form_data` turns the submitted data back into the
model. Validate inside `evaluate` before using the configuration.

The `form_ui_schema` class attribute customises how the form renders: it is a
[react-jsonschema-form](https://rjsf-team.github.io/react-jsonschema-form/) UI schema keyed by field
name.

```python
class MyPlugin(BaseEvaluationPlugin[ConfigSchema]):
    form_ui_schema = {
        "threshold": {"ui:widget": "range"},
        "notes": {"ui:widget": "textarea", "ui:placeholder": "Optional notes"},
    }
```

Project-level settings are declared with `@project_config(key, name, category, value_type, required)`
(`ConfigCategory.SECRETS` or `ConfigCategory.VARIABLES`). At run time `get_project_setting(key)` and
`require_project_setting(key)` return variables; `get_secret(key)` and `require_secret(key)` return
secrets, which the worker passes to the plugin process as environment variables `AISC_SECRET_<KEY>`.

## 8. Inputs

Declare each input with the `@evaluation_input` class decorator:

- `name`: the key you read it back with (`get_input_data(name)`);
- `label`: shown on the evaluation form;
- `input_type`: `InputType.DATASET`, `MODEL`, `LLM`, `DATASHAPE` or `RESOURCE`;
- `input_provider_class`: the class that parses the bytes; required for datasets and models,
  `JsonInputProvider` by default for the other types;
- `required`: whether the form insists on it.

Built-in providers: `CsvInputProvider` (a list of dicts, one per row), `JsonInputProvider`,
`ParquetInputProvider` (a pandas DataFrame; needs pandas) and `OnnxInputProvider` (an
`onnxruntime.InferenceSession`; needs onnxruntime). Typed helpers read the JSON-backed inputs:
`get_llm_config(name)`, `get_resource_config(name)` and `get_input_datashape(name)`.

For another format, subclass `BaseInputProvider` and implement `_read_data`:

```python
from aisc_plugin_interface import BaseInputProvider


class YamlInputProvider(BaseInputProvider):
    def _read_data(self, file_content: bytes):
        import yaml
        return yaml.safe_load(file_content)
```

## 9. Metrics and measurements

`evaluate` may return any object; it is passed to every method decorated with `@metric`. Each of those
returns `list[Measure]`. A `Measure` has `name`, `score`, and optionally `description`, `unit`,
`time`, `error`, `dimensions` and `direction` (`MetricDirection.HIGHER_IS_BETTER`, `LOWER_IS_BETTER`
or `NEUTRAL`).

To report a metric per slice (per split, per class), emit several measures with the same `name`, each
with scalar `dimensions` values, and turn on the dimensions view with `feature_flags`:

```python
from aisc_plugin_interface import PluginFeatureFlags


class MyPlugin(BaseEvaluationPlugin[ConfigSchema]):
    @property
    def feature_flags(self) -> PluginFeatureFlags:
        return PluginFeatureFlags(show_dimensions_visualisation=True)

    @metric("accuracy")
    def accuracy_metric(self, evaluation_output) -> list[Measure]:
        return [Measure(name="accuracy", score=float(score), dimensions={"split": split})
                for split, score in evaluation_output.get("accuracy_by_split", {}).items()]
```

`get_metric_visualizations(config_data)` says how the results page shows the metrics. By default it
is one table with every metric; override it to add charts. Metric names must match the `@metric`
names.

```python
from aisc_plugin_interface import ChartType, MetricVisualization


class MyPlugin(BaseEvaluationPlugin[ConfigSchema]):
    def get_metric_visualizations(self, config_data: dict) -> list[MetricVisualization]:
        return [MetricVisualization(chart_type=ChartType.TABLE, metrics=self.get_metrics()),
                MetricVisualization(chart_type=ChartType.BARS, metrics=["Pass rate"], title="Pass rate")]
```

## 10. Progress and artifacts

For loops, `self.progress_bar(iterable, desc=...)` reports progress as items are processed:

```python
for row in self.progress_bar(rows, desc="Evaluating"):
    ...
```

For long steps without an iterable:

```python
self.report_progress(TaskProgress(progress=0.25, extra={"stage": "loading"}))
```

`progress` is between 0.0 and 1.0; `extra` holds any plugin-defined data. Reporting is optional and
does nothing outside a run. `_set_progress_callback` belongs to the runtime: do not override it.

`self.upload_artifact(name, content)` stores a file (bytes) with the run.

## 11. Optional hooks

### `on_config_change`

Called whenever the user changes a form value, with the partial form data (which may be incomplete,
so do not validate it). It returns the data, the JSON schema and the UI schema, so it can fill
derived values, change drop-downs or hide fields:

```python
class MyPlugin(BaseEvaluationPlugin[ConfigSchema]):
    def on_config_change(self, form_data):
        import copy

        schema, ui_schema = self.get_full_schema()
        data = form_data.model_dump() if form_data else {}
        ui_schema = copy.deepcopy(ui_schema)
        if data.get("task") != "multiclass":
            ui_schema["num_classes"] = {"ui:widget": "hidden"}
        return form_data, schema, ui_schema
```

### `parse_config_from_dataset` and `feature_flags`

With `PluginFeatureFlags(can_parse_config_from_dataset=True)` the form shows a dataset drop-down, and
`parse_config_from_dataset(file_content)` may return a configuration derived from the chosen file
(or `None`):

```python
class MyPlugin(BaseEvaluationPlugin[ConfigSchema]):
    @property
    def feature_flags(self) -> PluginFeatureFlags:
        return PluginFeatureFlags(can_parse_config_from_dataset=True)

    def parse_config_from_dataset(self, file_content: bytes) -> dict | None:
        from aisc_plugin_interface import CsvInputProvider

        rows = CsvInputProvider(file_content).get_data()
        return {"score_column": next(iter(rows[0]), "")} if rows else None
```

### `plugin_name`, `ui_icon` and `description`

- `plugin_name`: the name shown in the UI (the class name otherwise). A subclass of another plugin
  does not inherit it.
- `ui_icon`: a [Material icon](https://fonts.google.com/icons) name for the plugin list
  (`extension` by default); or override the `display_icon` property.
- `description`: Markdown shown on the plugin's configuration page (GitHub-flavoured: headings,
  lists, bold, links, code). Common indentation is removed, so a triple-quoted string inside the class
  is not rendered as a code block. Empty by default, which hides the block.

```python
class MyPlugin(BaseEvaluationPlugin[ConfigSchema]):
    plugin_name = "My Evaluator"
    ui_icon = "table_chart"
    description = """
        ## What this plugin does

        Evaluates a candidate response against a reference answer.

        - Provide a **dataset** with `reference` and `candidate` columns.
        - Optionally set a passing `threshold`.
    """
```

## 12. Testing a plugin

A plugin can be tested without the platform: instantiate it, feed inputs with
`set_input_content(name, file_bytes)`, call `evaluate` and then `export_metrics`:

```python
plugin = ExampleCsvPlugin()
plugin.set_input_content("dataset", b"score\n0.2\n0.9\n")
output = plugin.evaluate({"score_column": "score", "threshold": 0.5})
assert [m.name for m in plugin.export_metrics(output)] == ["Average score", "Pass rate"]
```

## 13. Calling a system under test (Manage, Connections)

This section needs the `feat/unified-modules` version of the library (see section 4).

A project admin registers the AI systems the project assesses over the network under **Manage,
Connections** on the project page: an OpenAI-compatible endpoint, an A2A agent, an Open Inference
Protocol (KServe V2) model server, or any REST API with a request template. There are two ways for a
plugin to use one.

### 13.1 A tool that already speaks a standard protocol

Most tools call a system their own way: an OpenAI client, an A2A client, an OIP client. Declare which
protocol(s) the tool speaks and where it takes the endpoint; the tool is then pointed at the platform,
which translates to whatever the connection is:

```python
from aisc_plugin_interface.system_under_test import system_under_test


@system_under_test(protocols=("openai",),
                   fields={"target.base_url": "base_url", "target.api_key": "api_key", "target.model": "model"})
class MyPlugin(BaseEvaluationPlugin[MyConfig]):
    def evaluate(self, config_data):
        ...   # config_data["target"] points at the system under test
```

This adds the `target` input: what the evaluation assesses, the system or one component of its AI
card. When a run has a target with an endpoint, the platform issues the run a key for that endpoint
(valid 12 hours, stored hashed), and the declared config fields (dotted for nesting) and environment
variables (`env={"OPENAI_BASE_URL": "base_url"}`, set for the length of `evaluate` only) are filled
for the first protocol listed.

| Protocol | Roles |
|---|---|
| `aisc` | `ask_url`, `api_key` |
| `openai` | `base_url`, `model`, `api_key` |
| `a2a` | `agent_card_url`, `rpc_url`, `api_key` |
| `oip` | `base_url`, `model`, `api_key` |

Only fill what reaches the system under test: a tool that also uses an LLM as a judge keeps that
client on its own key. With `required=False` the tool runs unchanged when no endpoint is bound.

Every evaluation in the Configurator names its target, whether or not the tool calls a system: the
engine adds a required `target` input to every plugin's form, and results are joined to it. A plugin
that declares the input (the decorator does) reads it with
`aisc_plugin_interface.targets.target_of(self)`; `EndpointClient.for_target(self)` reaches that
target's endpoint.

### 13.2 A tool written for AISC

Declare a resource input and ask the client:

```python
from aisc_plugin_interface import BaseEvaluationPlugin, InputType, evaluation_input
from aisc_plugin_interface.connections import EndpointClient


@evaluation_input(name="system", label="System under test", input_type=InputType.RESOURCE, required=True)
class MyPlugin(BaseEvaluationPlugin[MyConfig]):
    def evaluate(self, config_data):
        system = EndpointClient.for_input(self, "system")
        answer = system.ask("Is my nationality an input?", history=[...])   # history is optional
        if answer.refused:
            ...                          # the system declined; answer.refusal_reason says why
        else:
            ...                          # answer.text
```

The client resolves the connection from the platform, renders the request, calls the system, reads
the answer and records what was assessed as the artifact `connection-<name>.json` (without the key).
Errors are `EndpointAuthError`, `EndpointNotFound`, `EndpointTimeout`, `EndpointBadResponse` and
`BlockedAddress`, all subclasses of `EndpointError`. Internal addresses are refused unless the
project allows them (Manage, Connections, Allowed internal hosts, or the deployment's
`CONNECTIONS_ALLOWED_HOSTS`); the platform hands the run that rule with the connection, so the
plugin needs no setting of its own.

## License

This guide is part of the AISC project, licensed under the [Apache License 2.0](LICENSE.md).
© 2024–2026 Université du Luxembourg and Luxembourg Institute of Science and Technology (LIST).

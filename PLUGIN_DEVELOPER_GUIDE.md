# Plugin Developer Guide

This guide is for developers who want to implement evaluation plugins.
The goal is to explain the contract exposed to plugin authors, clarify the current extension points, and make it easier to identify which additional hooks would be useful.

## 0. Upgrading an existing plugin

Before changing a plugin that already works against a released version, read the migration guide for
that hop — it lists what changed in the plugin interface, the plugin manager, the evaluation runtime
and the backend, and which changes are breaking:

- [`aisc` repo → `docs/migrations/`](https://github.com/lux-ai-factory/aisc/tree/master/docs/migrations)
  — one `vX-to-vY.md` per release, plus a `README` explaining how they are written.

The short version for the v1.3.0 → v1.4.0 hop: a plugin that reads its inputs from the declared
slots, takes all settings from plugin config / project settings, and does not call the
backend mid-run keeps working. The one thing that breaks a plugin is depending on **environment
variables** the worker used to inherit — the child environment is now a closed allowlist.

## 1. Purpose

The system is built around a plugin model in which evaluation logic is implemented outside the core application and loaded dynamically at runtime.

A plugin author should be able to:

- define configuration as typed data
- parse datasets and models into domain-specific objects
- execute evaluation logic
- emit structured measurements
- report progress during long-running work
- optionally influence host-specific integration behavior through extra hooks

The core question behind this guide is not how to use the surrounding platform, but whether the plugin contract is expressive enough for real evaluation workloads.

## 2. Architecture Overview

At a high level, the architecture has four moving parts:

- a **plugin project** that contains one or more plugin classes
- an **interface package** that defines the base classes and shared models
- a **plugin loader** that discovers and imports plugin packages
- one or more **host runtimes** that instantiate plugins for discovery, validation, or execution

The same plugin can be loaded in more than one context. In the current reference implementation, one runtime imports plugins to inspect metadata and configuration, while another runtime imports them to execute evaluation jobs.

## 3. Discovery Model

Plugins are discovered from a configured root directory. The loader scans each top-level folder in that directory and looks for a Python package in one of these layouts.

### `src` layout

```text
my-plugin-project/
├── pyproject.toml
└── src/
    └── my_plugin/
        ├── __init__.py
        └── plugin.py
```

### Direct package layout

```text
my-plugin-project/
├── pyproject.toml
└── my_plugin/
    ├── __init__.py
    └── plugin.py
```

The package must export the plugin class from `__init__.py`. The loader imports the package and registers every class that inherits from `BaseEvaluationPlugin`.

Important details:

- the exposed plugin name is the Python class name, for example `MyPlugin`
- the project folder name and package name do not have to match
- any import error during discovery prevents the plugin from being registered

## 4. Local Development Assumptions

The current reference setup expects a plugin root directory mounted into the runtimes that load plugins. A typical local arrangement looks like this:

```text
/absolute/path/to/plugins/
└── my-plugin-project/
```

As long as the host runtime points its plugin search path at `/absolute/path/to/plugins`, the plugin can be discovered.

You do not need to understand the surrounding platform to implement a plugin, but you do need to know two practical constraints:

- your project must live under the configured plugin root
- both the discovery runtime and the execution runtime must be able to import the package

## 5. Creating a Plugin Project

Create a new project in your plugin workspace:

```bash
mkdir -p /absolute/path/to/plugins
cd /absolute/path/to/plugins
mkdir my-aisc-plugin
cd my-aisc-plugin
uv init --lib --python 3.12
uv add aisc-plugin-interface
```

Recommended structure:

```text
my-aisc-plugin/
├── pyproject.toml
├── README.md
├── src/
│   └── my_aisc_plugin/
│       ├── __init__.py
│       └── plugin.py
└── uv.lock
```

## 6. Core Contract

Every plugin must inherit from `BaseEvaluationPlugin[T]`, where `T` is a Pydantic model representing plugin configuration.

The core contract is centered around these capabilities:

- `evaluate(config_data)`: main execution logic
- `validate_config_form_data(config_form_data)`: validates incoming configuration against your typed model
- `@metric("...")`: marks metric export methods
- `export_metrics(...)`: runs all metric exporters and aggregates their `Measure` outputs
- `@evaluation_input(...)`: declares a dataset/model input slot and its accepted value types
- `get_input_data(name)`: returns parsed input data for slots declared with `@evaluation_input`
- `progress_bar(...)` / `report_progress(TaskProgress(...))`: optional progress reporting hooks
- `upload_artifact(name, content)`: optional artifact upload hook

These pieces are sufficient for the basic plugin lifecycle:

1. receive configuration
2. parse input artifacts
3. execute evaluation logic
4. emit measurements

## 7. Minimal Example

This example reads a CSV dataset and computes two aggregate metrics.

### `src/my_aisc_plugin/plugin.py`

```python
from typing import Any

from pydantic import BaseModel, Field

from aisc_plugin_interface import BaseEvaluationPlugin, InputType, Measure, evaluation_input, metric


class ConfigSchema(BaseModel):
    score_column: str = Field(
        ...,
        description="Name of the CSV column containing numeric scores.",
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Scores greater than or equal to this value count as passing.",
    )


@evaluation_input(
    name="dataset",
    label="Dataset",
    input_type=InputType.DATASET,
    accepts=["records"],
    required=True,
)
class ExampleCsvPlugin(BaseEvaluationPlugin[ConfigSchema]):
    def evaluate(self, config_data: dict) -> Any:
        config = self.validate_config_form_data(config_data)
        rows = self.get_input_data("dataset")
        if rows is None:
            raise ValueError("This plugin requires a dataset file")

        scores: list[float] = []

        for row in self.progress_bar(rows, desc="Scoring rows"):
            raw_value = row.get(config.score_column)
            if raw_value is None or raw_value == "":
                continue

            try:
                scores.append(float(raw_value))
            except (TypeError, ValueError):
                continue

        passing = [score for score in scores if score >= config.threshold]

        return {
            "average_score": (sum(scores) / len(scores)) if scores else 0.0,
            "pass_rate": (len(passing) / len(scores)) if scores else 0.0,
        }

    @metric("Average score")
    def average_score_metric(self, evaluation_output: dict) -> list[Measure]:
        if not evaluation_output:
            return []
        return [
            Measure(
                name="Average score",
                score=float(evaluation_output["average_score"]),
            )
        ]

    @metric("Pass rate")
    def pass_rate_metric(self, evaluation_output: dict) -> list[Measure]:
        if not evaluation_output:
            return []
        return [
            Measure(
                name="Pass rate",
                score=float(evaluation_output["pass_rate"]),
                unit="ratio",
            )
        ]
```

### `src/my_aisc_plugin/__init__.py`

```python
from .plugin import ExampleCsvPlugin

__all__ = ["ExampleCsvPlugin"]
```

## 8. Configuration as Typed Data

Plugin configuration is defined as a Pydantic model.

That gives you:

- explicit configuration structure
- validation rules close to the plugin implementation
- a single source of truth for default values and constraints

Example:

```python
class ConfigSchema(BaseModel):
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
```

Inside `evaluate`, validate incoming configuration before using it:

```python
config = self.validate_config_form_data(config_data)
```

Even if the host application also validates configuration, the plugin should treat validated config as the boundary between transport data and business logic.

The `form_ui_schema` class attribute customizes how the form renders (an RJSF UI schema keyed by field name, served via `get_config_form_ui_schema()`). Use it to pick widgets per field (keys are your config model's field names — the example below is illustrative); `on_config_change` can further adjust the returned UI schema at runtime, e.g. hiding irrelevant fields with `{"ui:widget": "hidden"}`.

Example:

```python
class MyPlugin(BaseEvaluationPlugin[ConfigSchema]):
    form_ui_schema = {
        "threshold": {"ui:widget": "range"},
        "metrics": {"ui:widget": "checkboxes"},
        "notes": {"ui:widget": "textarea", "ui:placeholder": "Optional notes"},
    }
```

## 9. Input Parsing

The runtime resolves a **data adapter** for each input slot and hands the plugin
the produced value. There is one input mechanism — adapters (standard ones, or
ones a plugin ships); the old input-provider classes were removed (ADR 0009 D-22).

Declare each slot with `@evaluation_input`, naming the value types it accepts with
`accepts`. Inside `evaluate`, read the value with `get_input_data(name)`, which
returns `None` when the input was not provided (for an `openai-chat` slot, use
`get_llm_config(name)` instead).

```python
from aisc_plugin_interface import BaseEvaluationPlugin, InputType, evaluation_input


@evaluation_input(
    name="dataset",
    label="Dataset",
    input_type=InputType.DATASET,
    accepts=["records"],   # a list of dicts; use ["table"] for a pandas DataFrame
    required=True,
)
class MyPlugin(BaseEvaluationPlugin[ConfigSchema]):
    def evaluate(self, config_data):
        rows = self.get_input_data("dataset")   # list[dict]
        ...
```

**A plugin may ship its own data adapter** for a format the standard library does
not cover: subclass `DataAdapter` in the package's module folder and give the slot
its id (`adapter="<id>"`). The platform resolves and runs it like any other. This
replaces the old `input_provider_class=`; there are no input providers any more.

### Inputs and adapters (0.5.0)

A slot declares the value types (data) or face protocols (endpoint) it can use
with `accepts`, and what the bound component *is* with `role`. The `input_type`
shorthand fills `accepts`/`role`:

| `input_type` | shorthand `accepts` | shorthand `role` |
|---|---|---|
| `LLM` | `["openai-chat"]` | `subject` |
| `DATASET` | `["records"]` (set `accepts=["table"]` for a DataFrame) | `input` |
| `MODEL` | `["onnx-session"]` | `input` |
| `DATASHAPE` | `["json"]` | `input` |
| `RESOURCE` | `["resource"]` | `input` |

There are two kinds of adapter, both in `aisc_plugin_interface.adapters`:

**Data adapters** turn a file format into the value the slot accepts, once, in the
plugin process. The standard library reads `csv`, `tsv`, `parquet`, `jsonl`,
`json` and `onnx` and produces `table`, `records`, `json`, `onnx-session` or
`file`. A plugin that wants CSV records from a Parquet upload simply declares
`accepts=["records"]`; the platform picks the adapter.

**Endpoint adapters** turn a target's protocol into the protocol the plugin
speaks (its *face*), per call. The standard library has `openai` (a passthrough),
`oip`, `rest` and `a2a`. An `openai-chat` slot keeps the `LLMConfig` shape; the
plugin receives the run door's local URL and a run token, never the real key:

```python
@evaluation_input(name="target", label="Target", input_type=InputType.LLM)
class MyPlugin(BaseEvaluationPlugin[ConfigSchema]):
    def evaluate(self, config_data):
        llm = self.get_llm_config("target")          # url is the door, secret_key names the token
        api_key = self.get_secret(llm.secret_key)    # the run token, not the target key
```

A non-LLM endpoint slot reads `self.get_endpoint(name)`, returning an
`EndpointConfig(url, protocol, model, token_key)`. A plugin that reaches a
bespoke system can declare the environment its library already reads, and the
runtime fills it from the door:

```python
@evaluation_input(
    name="target", label="Target", input_type=InputType.LLM,
    env={"OPENAI_BASE_URL": "{base_url}", "OPENAI_API_KEY": "{api_key}"},
)
```

`{base_url}`, `{api_key}` and `{model}` are filled per run.

#### Shipping an adapter with your plugin

An adapter is a small class next to your plugin class; the plugin-manager finds it
in the same scan. A data adapter subclasses `DataAdapter`; an endpoint adapter
subclasses `EndpointAdapter`. Both declare an `id`, a `label`, a pydantic `Config`
and their types:

```python
from aisc_plugin_interface.adapters import DataAdapter


class BankStatementAdapter(DataAdapter):
    id = "bank-statement"
    label = "Bank statement"
    reads = ("csv",)
    produces = ("records",)

    def load(self, source, produce: str):
        ...  # return list[dict]
```

Then name it as a slot's default, and optionally make it the only choice:

```python
@evaluation_input(
    name="target", label="Target", input_type=InputType.LLM,
    adapter="my_pkg:BankPortalAdapter", adapter_required=True,
)
```

A shipped endpoint adapter still runs behind the run door, not in the plugin
process. Adapters are matched to slots by type, so a shipped adapter is offered
wherever it fits, not only in the plugin that shipped it.

The **presets** (`llm`, `agent`, `model-api`, `dataset`, `model-file`) are the
simple "Add …" forms; a preset is a starting point, not a mode, and "Custom"
opens the adapter-plus-config form.

## 10. Metrics and Measurements

The `evaluate` method may return any intermediate object. That object is then passed to each method decorated with `@metric`.

Metric methods must return `list[Measure]`.

`Measure` contains:

- `name`
- `description`
- `unit`
- `score`
- `time`
- `error`
- `dimensions`
- `direction`

One plugin can export one metric or many metrics. This separation is useful because it lets the evaluation step produce a shared intermediate result and keeps metric extraction methods small and focused.

To report a metric per slice (e.g. per split or class), emit several `Measure`s sharing the same `name`, each with scalar `dimensions` values (`dict[str, str | int | bool]`), and opt in to the dimensions visualisation via `feature_flags`.

Example:

```python
from aisc_plugin_interface import PluginFeatureFlags


class MyPlugin(BaseEvaluationPlugin[ConfigSchema]):
    @property
    def feature_flags(self) -> PluginFeatureFlags:
        return PluginFeatureFlags(show_dimensions_visualisation=True)

    @metric(name="accuracy")
    def accuracy_metric(self, evaluation_output: Any) -> list[Measure]:
        if not evaluation_output:
            return []
        return [Measure(name="accuracy", score=float(score), dimensions={"split": split})
                for split, score in evaluation_output.get("accuracy_by_split", {}).items()]
```

Rules for dimensions: every per-slice `Measure` shares the same `name`; each `dimensions` value must be a scalar; use config-declared keys (e.g. the split names from the dataset) so the host can filter and group by them.

Example:

```python
class MyPlugin(BaseEvaluationPlugin[ConfigSchema]):
    @metric("Accuracy")
    def accuracy_metric(self, evaluation_output) -> list[Measure]:
        ...


    @metric("F1")
    def f1_metric(self, evaluation_output) -> list[Measure]:
        ...
```

## 11. Progress Reporting

Long-running plugins can report progress during evaluation.

For loops, prefer `self.progress_bar(...)`, which reports progress automatically as items are processed:

```python
for row in self.progress_bar(rows, desc="Evaluating"):
    ...
```

For single long calls without an iterable, use:

```python
self.report_progress(TaskProgress(progress=0.25, extra={"stage": "loading"}))
```

Rules:

- `progress` must be between `0.0` and `1.0`
- `extra` may contain plugin-defined metadata
- progress reporting is optional

Do not override `_set_progress_callback`; that is managed by the execution runtime.

Upload intermediate or final files with `upload_artifact(name, content)`.

## 12. Optional Integration Hooks

The core plugin model is small, but the base class also exposes several optional hooks that are integration-oriented rather than strictly evaluation-oriented.

These hooks are important because they reveal where the architecture already allows extension and where future hooks may be needed.

### `on_config_change`

This hook receives incomplete or partially edited configuration and can return:

- updated config data
- updated schema
- updated UI schema

It is useful for dynamic configuration, conditional fields, and derived defaults.

From a pure plugin-architecture perspective, this hook can be read more generally as:

- a way to react to evolving configuration state
- a place to derive secondary configuration fields
- a place to resolve configuration against partially known inputs

Example: hide an irrelevant field based on the current selection (`form_data` may be incomplete, so never validate it here):

```python
class MyPlugin(BaseEvaluationPlugin[ConfigSchema]):
    def on_config_change(self, form_data: ConfigSchema | None) -> tuple[ConfigSchema | None, dict, dict]:
        import copy

        schema, ui_schema = self.get_full_schema()
        data = form_data.model_dump() if form_data else {}
        ui_schema = copy.deepcopy(ui_schema)
        if data.get("task") != "multiclass":
            ui_schema["num_classes"] = {"ui:widget": "hidden"}
        return form_data, schema, ui_schema
```

### `feature_flags`

This hook exposes plugin-specific capabilities to the host application.

At the moment it is mostly used for host-specific behavior, but the pattern is potentially useful for broader capability negotiation between plugin and runtime.

Example: opt in to dataset-driven config (pairs with `parse_config_from_dataset` below):

```python
from aisc_plugin_interface import PluginFeatureFlags


class MyPlugin(BaseEvaluationPlugin[ConfigSchema]):
    @property
    def feature_flags(self) -> PluginFeatureFlags:
        return PluginFeatureFlags(can_parse_config_from_dataset=True)
```

### `parse_config_from_dataset`

This hook attempts to infer configuration from the current dataset.

It is useful when configuration depends on data shape, column names, task type, or metadata inferred from the dataset.

From an extensibility perspective, this suggests a broader family of possible hooks:

- infer config from model artifacts
- infer config from project metadata
- infer config from external registries or schemas

Example: infer the score column from the uploaded CSV header (requires `feature_flags` above):

```python
class MyPlugin(BaseEvaluationPlugin[ConfigSchema]):
    def parse_config_from_dataset(self, file_content: bytes) -> dict | None:
        import csv
        import io

        rows = list(csv.DictReader(io.StringIO(file_content.decode("utf-8"))))
        if not rows:
            return None
        return {"score_column": next(iter(rows[0]), "")}
```

For the full dynamic-form patterns (conditional field groups, dataset-derived enums wired through `on_config_change`), see the `aisc-plugin-builder` skill.

### `get_metric_visualizations`

This hook returns visualization metadata associated with exported metrics.

It is not part of the core evaluation algorithm, but it is currently the way a plugin can communicate preferred result structure to a host application.

If the goal is to keep the plugin contract implementation-focused, this hook can be treated as optional integration metadata rather than a required plugin concern.

The default implementation returns a single `TABLE` with all metrics; override it to add charts or restrict which metrics each visualization shows. Metric names must match the `@metric(name=...)` strings.

Example:

```python
from aisc_plugin_interface import ChartType, MetricVisualization

### `description`

Optional Markdown text that the host renders on the plugin config page. Use it to describe
what the plugin does, list its expected inputs, or add usage notes.

It is a plain string attribute, so keep it simple: author it inline with a multi-line string,
or assign it an imported constant if you prefer to keep it in a separate util module.

```python
class MyPlugin(BaseEvaluationPlugin[ConfigForm]):
    plugin_name = "My Plugin"
    description = """## What this plugin does

Evaluates a candidate response against a reference answer.

### Usage

- Provide a **dataset** containing `reference` and `candidate` columns.
- Optionally set a passing `threshold`.

See `coding_assistant.md` in this package for details.
"""
```

Markdown is rendered with standard GFM (headings, lists, bold, links, inline code and code
blocks). Leave it empty (the default) to hide the block on the config page.

### `display_icon`

class MyPlugin(BaseEvaluationPlugin[ConfigSchema]):
    def get_metric_visualizations(self, config_data: dict) -> list[MetricVisualization]:
        return [MetricVisualization(chart_type=ChartType.TABLE, metrics=self.get_metrics()),
                MetricVisualization(chart_type=ChartType.BARS, metrics=["Pass rate"], title="Pass rate")]
```

### `plugin_name` / `ui_icon`

These class attributes are entirely presentation-oriented. They are useful only if the host application wants plugins to contribute presentation metadata.

Set `plugin_name` to override the displayed name (otherwise the class name is shown), and `ui_icon` to a Material icon name.

Example:

```python
class MyPlugin(BaseEvaluationPlugin[ConfigSchema]):
    plugin_name = "My Evaluator"
    ui_icon = "table_chart"  # a Material icon name, see https://fonts.google.com/icons
```

If the document should remain strictly focused on implementation concerns, this hook is peripheral.

---

##  License

This guide is part of the AISC project, licensed under the [Apache License 2.0](LICENSE).  
© 2024–2026 Université du Luxembourg and Luxembourg Institute of Science and Technology (LIST).




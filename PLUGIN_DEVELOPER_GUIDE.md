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
uv init --lib --python 3.12
uv add aisc-plugin-interface
uv run aisc-plugin-interface init-plugin
```

`init-plugin` asks for a class name and a file path, writes a plugin template into
`src/<package>/` and adds the class to `__init__.py`. `--force` overwrites an existing file.

This branch is 0.4.0, above PyPI's latest release 0.3.0. PyPI's 0.3.0 has none of `connections`,
`system_under_test`, `targets`, `assesses`, `helpers` or `model_session` (section 13). A plugin that
needs them pins `aisc-plugin-interface>=0.4.0` and installs it from the stack's package index (devpi)
or from git (`--branch feat/unified-modules`), never from PyPI until 0.4.0 is released there:

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

Rules for dimensions: every per-slice `Measure` shares the same `name`; each `dimensions` value must be a scalar; use config-declared keys (e.g. the split names from the dataset) so the host can filter and group by them.

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

## 13. What a plugin assesses, and the models that help it

This section needs the `feat/unified-modules` version of the library (see section 4), from version 0.2.9.

Two kinds of model meet in a plugin, and the library keeps them apart:

- The **target** is what the evaluation assesses: the AI system, or one component of its AI card (the
  chat assistant, the scoring model, the training data), or a model file the user uploaded to the
  project. A target reached over the network has an **endpoint**, set once under **Manage, Components
  and connections**.
- A **helper model** is a model the plugin uses as a tool: a judge, an attacker, a grader, an
  embedder or a generator. It is never what is assessed.

### 13.1 What a plugin assesses: `@assesses`

A plugin states what it assesses with `@assesses(card=..., local_model=...)`; at least one of the two.

- `card=` says how the plugin reaches a target defined on the **AI card** (the system or one of its
  components). It takes one of these builders:

  | Builder | The plugin |
  |---|---|
  | `system_under_test_access(protocols=("openai",), fields=..., env=..., needs=..., required=True)` | calls its target, only through the target's endpoint (13.5); as `@system_under_test` |
  | `dataset_through_target_access(datasets=("reference-dataset",), required=True)` | analyses tables; each row is first sent through the target's endpoint, as `@dataset_through_target` (13.6) |
  | `inputs_access()` | only reads its inputs (datasets, models, files) and calls no system; as `@assesses_inputs` |
  | `model_session(outputs={...})` | scores rows with a model, whichever it is (13.3) |

- `local_model=onnx(input="model")` says the plugin can assess a **local model file**: the ONNX file a
  person uploaded to the project, read through the plugin's own model input (`InputType.MODEL`). Declare
  that input below `@assesses`; the declaration fails at import if it is missing.

`required=False` on `system_under_test_access` or `dataset_through_target_access` makes the card access
optional: a run with no card target, or with a card target that has no endpoint (training data, say),
runs the plugin on its uploaded inputs unchanged, as the older decorators do with `required=False`; any
other failure to reach the platform still stops the run. The form then lets such a plugin pick a card
target without a connection (`card_needs_connection(cls)` is False; `cls.card_access_required` records
the flag, None when there is no `card=`).

```python
from aisc_plugin_interface import (
    BaseEvaluationPlugin, InputType, OnnxInputProvider, assesses, evaluation_input, onnx,
    system_under_test_access,
)


@assesses(card=system_under_test_access(protocols=("openai",),
                                        env={"AISC_TARGET_BASE_URL": "base_url", "AISC_TARGET_API_KEY": "api_key",
                                             "AISC_TARGET_MODEL": "model"}))
class BiasPlugin(BaseEvaluationPlugin):
    ...                                   # calls the card target through its endpoint


@assesses(local_model=onnx(input="model"))
@evaluation_input(name="model", label="Model", input_provider_class=OnnxInputProvider,
                  input_type=InputType.MODEL, required=True)
class DriftPlugin(BaseEvaluationPlugin):
    ...                                   # reads an uploaded model file
```

**Declare exactly one.** A class carries one declaration: `@assesses`, or one of the older decorators
(13.7), never two. A subclass inherits the declaration of its base and must not declare again; a second
declaration on the subclass raises `ValueError` at import.

### 13.2 What the evaluation form offers

The engine reads the declaration (`target_kinds_of`, `card_protocols_of`, `card_needs_connection`) and
builds the form from it, so a kind the plugin does not declare cannot be picked:

| Declared | The form asks for | Cannot be picked |
|---|---|---|
| `card=` only | one required target: "What is assessed (AI card)" | a local model file |
| `local_model=` only | the plugin's model input: "What is assessed (local model file)" | a card target (no `target` input is added) |
| both | two optional inputs, the AI card target or a local model file; an evaluation with neither or both is refused | |
| neither (a plugin that has not adopted) | a required `target`, as before | |

A card target for a plugin whose card access calls the target (`system_under_test_access`,
`dataset_through_target_access`, `model_session`) must have a connection; the form shows one without as
"not connected" and the evaluation is refused. A plugin that declares both kinds is run on one of them:
a run on a local model file is listed apart ("Local, not part of the assessment record"), is not in the
AI card's results page or reports, and is labelled "Local" in the dashboards. There is no silent fallback
from a card target to the plugin's own model.

### 13.3 Scoring plugins: `model_session` and `assessed_model()`

A plugin that scores rows with a model wants one object whichever target the run has.
`self.assessed_model()` returns it: `predict(rows)` gives one label per row, `predict_proba(rows)` one
list of class probabilities per row. `rows` is a list of dicts (feature name to value).

```python
@assesses(card=model_session(outputs={"label": "decision", "probability": "risk.p"}), local_model=onnx())
@evaluation_input(name="model", label="Model", input_provider_class=OnnxInputProvider,
                  input_type=InputType.MODEL)
class ScoringPlugin(BaseEvaluationPlugin):
    def evaluate(self, config_data):
        model = self.assessed_model()
        labels = model.predict(rows)
        proba = model.predict_proba(rows)
```

- With a **local model file**, the session runs the uploaded ONNX model; labels are class indices
  (0, 1, ...) when the ONNX model has a probability output. Any number of classes works.
- With a **card target**, the session sends each row once, in order, through the target's connection (the
  row is the request's `{{input}}`) and reads the answer at the paths declared in `outputs`: `label`
  and `probability` (of the positive class; `predict_proba` gives `[1-p, p]`). `predict` returns the
  target's own label values. Declare `label`, `probability` or both. `predict_proba` on a card session
  without a probability path raises; `predict` with only a probability path returns 0 or 1, the more
  probable class. Declaring `outputs["classes"]` above 2 makes `assessed_model()` refuse a card target:
  **binary classification only on card targets**. An answer lacking a declared path, a
  probability outside 0 to 1 and a refused row stop the run with the row number and the path.
- A run that binds both a model file and a card target, or neither, raises `ValueError`.
- When the answer fields differ per system, give the paths per run: `assessed_model(outputs={...})`
  replaces the declared `outputs` for that session only. It is checked like a declaration (an unknown
  key or no label and probability raises `ValueError` naming it); a local model ignores it but still
  checks it.

```python
    def evaluate(self, config_data):
        model = self.assessed_model(outputs={"label": config_data["label_field"],
                                             "probability": config_data["probability_field"]})
```

### 13.4 Helper models: `@helper_model` and `self.helper(role)`

```python
from aisc_plugin_interface import BaseEvaluationPlugin, assesses, helper_model, system_under_test_access


@helper_model("judge", "Judge model", required=True)
@helper_model("attacker", "Attacker model")
@assesses(card=system_under_test_access(protocols=("openai",), env={"AISC_TARGET_BASE_URL": "base_url"}))
class JailbreakPlugin(BaseEvaluationPlugin):
    def evaluate(self, config_data):
        judge = self.helper("judge")          # HelperModel(role, endpoint_url, model, api_key)
        attacker = self.helper("attacker")    # None when the run binds none to this optional role
```

The roles are `judge`, `attacker`, `grader`, `embedder` and `generator` (`HELPER_ROLES`). Each role is
an LLM input named `helper:<role>` that the form shows as "Helper model · <Role> (not assessed)": the
person picks the endpoint, the model and the project secret that holds the key. Rules:

1. **Never put a helper model's key, address or model in a config field.** `self.helper(role)` reads
   them, the key from the project secret's environment variable (as `get_secret`); the repr of the result
   never shows the key.
2. **Never use the target's key or address for a helper model.** A judge for a card target is its own
   model with its own key: the key of the target stays on the target (the `AISC_TARGET_*` variables of
   13.5 are not what a judge client reads), and a helper's key is never sent to the target.
3. A role that is declared required and bound to nothing, or a model bound without an endpoint or model
   name, or whose secret the run does not provide, raises `ValueError` naming the role.
4. Pass the helper's values to its client explicitly: no fallback to a shared environment variable such
   as `OPENAI_API_KEY`.

Config fields that select or key the model under test (`target_model`, `target_credential`,
`target_endpoint`, `model_under_test`) have no place in a plugin that declares `card=`: the target comes
from the platform. Remove them from the plugin's config when adopting `@assesses`.

### 13.5 The rules for a card target

1. **Never take the target from the form.** With a target bound, its address, key and model come from
   the platform for the length of the run. Write them into the plugin's own target fields, or into
   variables named for the target (`AISC_TARGET_BASE_URL`, `AISC_TARGET_API_KEY`, `AISC_TARGET_MODEL`),
   never into variables a helper model also reads (`OPENAI_API_KEY`, `OPENAI_BASE_URL` and the like).
2. **Say what the plugin needs from the endpoint beyond text**: `needs=("tools",)` or
   `needs=("logprobs",)`. The platform's endpoint carries text and a history; a run whose endpoint
   doesn't carry a need is refused before the tool starts, with the reason. A target without an
   endpoint is refused the same way ("<target> has no endpoint: set one under Manage, Components and
   connections").
3. **The run records what it reached**: the library saves `connection-<name>.json` with the connection,
   the target and the protocol (never the key), and the platform records which connection served the run.

The names in the evaluation form say what each target is: "AI card · System: <label> · reached through
<protocol>" for the system and "AI card · Component: <label> (<kind>) · not connected" for a component
without a connection.

A project admin registers the AI systems the project assesses over the network under **Manage,
Components and connections** on the project page: an OpenAI-compatible endpoint, an A2A agent, an Open
Inference Protocol (KServe V2) model server, or any REST API with a request template. There are two ways
for a plugin to use one.

#### A tool that already speaks a standard protocol

Most tools call a system their own way: an OpenAI client, an A2A client, an OIP client. Declare which
protocol(s) the tool speaks and where it takes the endpoint; the tool is then pointed at the platform,
which translates to whatever the connection is:

```python
@assesses(card=system_under_test_access(
    protocols=("openai",),
    fields={"target.base_url": "base_url", "target.api_key": "api_key", "target.model": "model"}))
class MyPlugin(BaseEvaluationPlugin):
    def evaluate(self, config_data):
        ...   # config_data["target"] points at the system under test
```

This adds the `target` input: what the evaluation assesses. When a run has a target with an endpoint,
the platform issues the run a key for that endpoint (valid 12 hours, stored hashed), and the declared
config fields (dotted for nesting) and environment variables (`env={"AISC_TARGET_BASE_URL": "base_url"}`,
set for the length of `evaluate` only) are filled for the first protocol listed.

| Protocol | Roles |
|---|---|
| `aisc` | `ask_url`, `api_key` |
| `openai` | `base_url`, `model`, `api_key` |
| `a2a` | `agent_card_url`, `rpc_url`, `api_key` |
| `oip` | `base_url`, `model`, `api_key` |

Only fill what reaches the target (rule 1): a tool that also uses an LLM as a judge keeps that client on
its helper model (13.4). A plugin whose card access is optional (`required=False` in the older
decorators, or `local_model=` declared next to `card=`) runs unchanged when no endpoint is bound. A plugin
reads the target with `aisc_plugin_interface.targets.target_of(self)`; `EndpointClient.for_target(self)`
reaches its endpoint.

#### A tool written for AISC

Declare a resource input and ask the client:

```python
from aisc_plugin_interface import BaseEvaluationPlugin, InputType, evaluation_input
from aisc_plugin_interface.connections import EndpointClient


@evaluation_input(name="system", label="System under test", input_type=InputType.RESOURCE, required=True)
class MyPlugin(BaseEvaluationPlugin):
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
Errors are `EndpointAuthError`, `EndpointNotFound` (and its `TargetHasNoEndpoint`), `EndpointTimeout`,
`EndpointBadResponse` and `BlockedAddress`, all subclasses of `EndpointError`. A connection the system
drops is retried like a timeout; an answer past `MAX_ANSWER_BYTES` (10 MiB) is refused, not read whole.
An A2A answer of the wrong shape (a task, status, message or artifact that is not an object, artifacts
that are not a list, a reply that is not JSON) is an `EndpointBadResponse`, never a crash of the plugin.
A connection with a key header sends the key in that header whatever its kind, A2A and OIP included;
without one, A2A and OIP send it as `Authorization: Bearer`.
Internal addresses are refused unless the project allows them (Manage, Components and connections,
Allowed internal hosts, or the deployment's `CONNECTIONS_ALLOWED_HOSTS`); the platform hands the run that
rule with the connection, so the plugin needs no setting of its own. The rule is applied to the address
the connection actually reaches, after any second name lookup, and an IPv4-mapped address counts as the
IPv4 address it carries. Only a socket to a configured proxy's own host skips that check (the proxy
resolves the name); a proxy variable being set, or `NO_PROXY` alone, does not switch it off.

### 13.6 Tables sent through the target

`dataset_through_target_access(datasets=("reference-dataset", ...))` is for a plugin that analyses
tables: each row is first sent through the target's endpoint, and its answer becomes columns
`target.<field>` (a refusal `target.refused` and `target.refusal_reason`, a failed call `target.error`; a
field of the system's own answer named `refused`, `refusal_reason`, `error` or `answer` goes under
`target.answer.<field>`, so it is never taken for the wrapper's flag). Dates and times are sent as ISO
text (a missing one as null) and decimals as numbers. The plugin reads the enriched table in the form it
had it (a DataFrame stays a DataFrame, rows stay rows, whatever its input's provider); the answers are
saved as `target-answers-<dataset>.csv`, where a text cell or column name starting with `=`, `+`, `-`,
`@`, a tab or a carriage return gets a leading `'` so a spreadsheet does not run it as a formula
(numbers are written as they are). The run settings `target_calls_at_once` (default 1, at most 16) and
`target_row_limit` (0: every row) are read from the plugin's config when its form has them. Which fields
of a row are sent is the endpoint's business: its body template (`{{input}}` sends the whole row). A
connection bound directly (`connection:<pid>/<name>`) works as well as a target. With a target the
platform says has no endpoint (training data, say) and `required=False` (on the builder or the older
decorator), the
plugin runs on the uploads as they are; any other failure to resolve it (no such target or project, a
platform that does not answer) still stops the run, and so does a target whose every call failed (one
failed row is only that row's `target.error`).

### 13.7 Plugins that have not adopted `@assesses`

`@system_under_test(...)`, `@dataset_through_target(...)` and `@assesses_inputs()` keep working
unchanged, alone: they read as `card=` with no local model, and the form adds a required `target` as
before. Use them with the same arguments (`required=`, `needs=`, `fields=`, `env=`). A plugin with none
of them and no `@assesses` has no target kinds; the platform's plugin check reports it. Reading the
declaration: `target_kinds_of(cls)` (`("card",)`, `("local_model",)`, both, or `()`),
`card_protocols_of(cls)`, `card_needs_connection(cls)`, `helper_roles_of(cls)`; `target_access_of(cls)`
still reads the older declaration.

## License

This guide is part of the AISC project, licensed under the [Apache License 2.0](LICENSE.md).
© 2024–2026 Université du Luxembourg and Luxembourg Institute of Science and Technology (LIST).

# AISC Plugin Interface

`aisc-plugin-interface` is the Python library every AISC test is written against. AISC (the AI Assessment
Sandbox Configurator) runs an assessment in four steps: 1 Qualify the AI system, 2 Set control objectives,
3 Choose tests and checklists, 4 Collect evidence. A **test** is chosen in step 3 and runs in step 4; it is a
plugin, a Python package with one or more classes that inherit from `BaseEvaluationPlugin`, defined here.
The library also holds the models those classes exchange with the engine (inputs, measures, progress, project
settings), the declarations of what a test assesses and of the helper models it uses, and the client a test uses
to reach the system being assessed through the platform.

It is a library, not a service: nothing here listens on a port.

## How it works

A plugin class declares:

- its configuration form, as a Pydantic model given as the type parameter
  (`BaseEvaluationPlugin[MyConfig]`); the engine turns it into a JSON schema for the web form;
- its inputs (datasets, models, LLMs, data shapes, resources) with `@evaluation_input`, and the
  project settings it needs with `@project_config`;
- what it assesses, with `@assesses` (a target of the AI card, a local model file, or both), and the helper
  models it uses as tools, with `@helper_model`;
- `evaluate(config_data)`, which does the work and returns any intermediate result;
- one or more `@metric("...")` methods that turn that result into a list of `Measure`.

Two services of the engine import plugins through this library:

- `aisc-backend` (in `apps/backend`) loads a plugin to read its form, inputs, declarations and metrics, and
  builds the run form from them;
- `aisc-eval-worker` (in `apps/eval`) runs it in its own process: it feeds the inputs (`set_input_content`), the
  project settings and the secrets (environment variables `AISC_SECRET_<KEY>`, the helper models' keys
  included), calls `evaluate`, and collects the measures, progress reports and artifacts.

Both find plugins with [aisc-plugin-manager](https://github.com/lux-ai-factory/aisc-plugin-manager).
The platform service also mounts this library, to test a connection with the same client plugins use.

**What is assessed and helper models.** A test never holds the secrets of the system it assesses. During a run
the worker gives it a run pass (`PLATFORM_RUN_PASS`); with it the client asks the platform for a run key and
calls one of the platform's doors, and the platform adds the connection's secrets and forwards the call. A
helper model (judge, attacker, grader, embedder, generator) is called directly: `self.helper(role)` returns its
endpoint, the model picked for the run and its key. For the whole flow, see `docs/architecture.md` ("Running
tests") and `docs/appendices/manage.md` ("Assessed system: Components and connections", "Helper models") in the
aisc repository; [PLUGIN_DEVELOPER_GUIDE.md](PLUGIN_DEVELOPER_GUIDE.md) section 13 gives the rules for plugin
authors.

Main parts of `src/aisc_plugin_interface/`:

| Module | What it holds |
|---|---|
| `base_evaluation_plugin.py` | `BaseEvaluationPlugin`, `PluginFeatureFlags`, `ProgressBar` |
| `decorators/` | `@metric`, `@evaluation_input`, `@project_config` |
| `models/` | `Measure`, `MetricVisualization`, `TaskProgress`, `InputDefinition`, `DataShape`, `LLMConfig`, `ResourceConfig`, ... |
| `input_providers/` | readers that turn input bytes into Python objects: CSV, JSON, Parquet (needs pandas), ONNX (needs onnxruntime) |
| `connections.py` | `EndpointClient`: reach a system registered under Manage, Components and connections (OpenAI-compatible, REST, A2A, Open Inference Protocol); with a run pass it asks through the platform, which also serves the adapter plugins' protocols |
| `system_under_test.py` | how a plugin gets what it assesses: `@system_under_test` (a target through its connection, with `needs=`), `@assesses_inputs`, or `@dataset_through_target` (each row of a table through the target first); the rules are in PLUGIN_DEVELOPER_GUIDE.md section 13 |
| `assesses.py` | `@assesses(card=..., local_model=...)`: what a plugin assesses (an AI card target, a local ONNX model file, or both), with the card builders `system_under_test_access`, `dataset_through_target_access`, `inputs_access`, `model_session`, and `onnx`; `target_kinds_of`, `card_protocols_of` and `card_needs_connection` read it |
| `helpers.py` | `@helper_model(role, label)`: the judge, attacker, grader, embedder or generator a plugin uses as a tool (`HELPER_ROLES`); `self.helper(role)` returns a `HelperModel` |
| `model_session.py` | `self.assessed_model()`: one `predict` / `predict_proba` over a local ONNX file or an AI card target's connection |
| `targets.py` | the evaluation's `target` input (`target:<project pid>/<key>`): the AI system or one component of its AI card |
| `model_listing.py` | `list_openai_models`: list the models of an OpenAI-compatible endpoint |
| `cli.py`, `templates/` | the `aisc-plugin-interface init-plugin` command |

The connection client uses the standard library only, so a plugin gains no dependency from it. It
refuses loopback, private and other internal addresses unless the project allows them.

## Install

Requires Python 3.12 or newer. The only runtime dependencies are `pydantic` and `rich`.

From PyPI (the released version, built from the `master` branch):

```bash
uv add aisc-plugin-interface        # or: pip install aisc-plugin-interface
```

This branch is 0.4.0, above PyPI's latest release 0.3.0. PyPI's 0.3.0 has none of `connections`,
`system_under_test`, `targets`, `assesses`, `helpers` or `model_session`, which exist only on the
`pilot` branch that the AISC stack uses. A plugin that needs them pins
`aisc-plugin-interface>=0.4.0` and installs it from the stack's package index (devpi), from a vendored wheel, or
from git, never from PyPI until 0.4.0 is released there:

```bash
uv add git+https://github.com/lux-ai-factory/aisc-plugin-interface --branch pilot
```

A test's run installs the plugin and its dependencies into a fresh environment, from PyPI and the stack's
package index (devpi). `plugin-publisher` publishes this library to that index too, and uv takes a package from
the first index that has it, the extra index (devpi) first: so a run gets this branch's library, not PyPI's,
whatever the version numbers say.

### Inside the AISC stack

There is no compose service for this library. It comes up with the stack (`uv run scripts/aisc.py start` from
the aisc repo root):

- the engine's services in `docker-compose.development.yml` (`aisc-backend`, `aisc-backend-migrate`,
  `aisc-backend-ledger`, `aisc-eval-worker`, `aisc-eval-flower`) mount `./shared` and, on start, run
  `uv pip install --no-deps -e /app/shared/plugin-interface`, so they use this checkout;
- the `platform` service mounts `shared/plugin-interface/src` read-only and puts it on its `PYTHONPATH`;
- `plugin-publisher` mounts this folder and publishes it to devpi, where test runs take it from.

A change here reaches those services when they restart. Images built on their own, and
`docker-compose.engine-standalone.yml`, use the wheel locked in `apps/backend` and `apps/eval` (0.4.0, under their
`vendor/`, built from this checkout by the aisc repo's `scripts/vendor-plugin-interface.sh`).

## Writing a plugin

Create a project and add the library (from git or a vendored wheel for the 0.4.0 features, see Install):

```bash
mkdir my-aisc-plugin && cd my-aisc-plugin
uv init --lib --python 3.12
uv add aisc-plugin-interface
uv run aisc-plugin-interface init-plugin     # writes a plugin class template into src/<package>/
```

A minimal plugin:

```python
from typing import Any

from pydantic import BaseModel, Field

from aisc_plugin_interface import BaseEvaluationPlugin, Measure, metric


class ConfigFormSchema(BaseModel):
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class MyPlugin(BaseEvaluationPlugin[ConfigFormSchema]):
    plugin_name = "My plugin"

    def evaluate(self, config_data: dict) -> Any:
        # Import heavy dependencies here, not at module level: the engine imports the
        # module only to read the form and the metrics.
        import numpy as np

        config = self.validate_config_form_data(config_data)
        self.logger.info("threshold %s", config.threshold)
        return {"MyMetric": [0.99, 0.5, 0.67]}

    @metric("MyMetric")
    def my_metric(self, evaluation_output: Any) -> list[Measure]:
        return [Measure(name="MyMetric", score=v) for v in evaluation_output.get("MyMetric", [])]
```

Export it from the package's `__init__.py`:

```python
from .plugin import MyPlugin

__all__ = ["MyPlugin"]
```

For the loader to accept the package, its `pyproject.toml` must have a `name` and a `version`, list
`aisc-plugin-interface` in its dependencies, and the code must sit in a folder named after the
package (dashes become underscores), under `src/` or at the project root.

The [Plugin Developer Guide](PLUGIN_DEVELOPER_GUIDE.md) covers inputs, metrics, visualisations,
progress, the optional hooks, and what a plugin assesses and the helper models it uses. To run your own plugin
in AISC, put its folder in `local_plugins/` of the aisc repository (see `local_plugins/README.md` there).

## Configuration

The library reads these environment variables. In AISC the evaluation worker sets them for the plugin
process; a plugin author does not set them.

| Variable | Read by | Meaning | Default |
|---|---|---|---|
| `PLATFORM_URL` | `connections.EndpointClient`, `@system_under_test` | Base URL of the platform service, which resolves connections and targets and issues run keys | none: resolving a connection fails without it |
| `PLATFORM_RUN_PASS` | same | The run's pass, sent as `X-AISC-Run-Pass` to the platform's internal routes; it opens only the targets and connections the run names. The worker mints it before the plugin starts and revokes it after. With a pass, the client never holds the connection's key: it asks the platform with a run key | none |
| `PLATFORM_CONNECTIONS_TOKEN` | same | Service token sent as `X-AISC-Service-Token` when there is no `PLATFORM_RUN_PASS` (older platforms and setups); the client then calls the system directly. The AISC worker keeps this token and never gives it to a plugin | none: without it or a pass, resolving a connection fails |
| `CONNECTIONS_ALLOWED_HOSTS` | `connections.call` | Comma-separated hosts (or `host:port`) that may be called although they are internal. Used only when no allowlist is passed in; a client resolved through the platform takes the project's rule from the platform | empty |
| `AISC_SECRET_<KEY>` | `get_secret`, `require_secret`, `self.helper` | A project secret, or a helper model's key. `<KEY>` is the setting key in upper case, other characters replaced by `_` | none |

## Tests

The tests need no database and no network beyond local stub servers on 127.0.0.1. The project
declares no dev dependencies, so pytest and pandas (the tests of `@dataset_through_target` read their
table with it) are added for the run; use `python -m pytest` so that the `tests` package can be
imported:

```bash
uv run --with pytest --with pandas python -m pytest -q
```

## Layout

```text
src/aisc_plugin_interface/   the library (see the table above)
tests/                       unit tests: connections, targets, @system_under_test, @assesses, helper models,
                             model sessions, input providers, measures
PLUGIN_DEVELOPER_GUIDE.md    how to write a plugin
```

## Contributing

`pilot` is the branch the AISC stack uses and the only one to work on. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the contributor licence terms.

## License

This project is licensed under the [Apache License 2.0](LICENSE.md).
© 2024–2026 Université du Luxembourg and Luxembourg Institute of Science and Technology (LIST).

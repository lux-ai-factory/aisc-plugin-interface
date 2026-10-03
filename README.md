# AISC Plugin Interface

`aisc-plugin-interface` is the Python library every AISC evaluation plugin is written against. AISC
(the AI Assessment Sandbox Configurator) runs an assessment in six steps: qualification, control
objectives, installing plugins (the evaluation tools), executing tests and addressing controls,
analysing results on the dashboard, and composing the report. Steps 3 and 4 run plugins, and a plugin
is a Python package with one or more classes that inherit from `BaseEvaluationPlugin`, defined here.
The library also holds the models those classes exchange with the execution engine (inputs,
measures, progress, project settings) and the client a plugin uses to call the system under test
that a project registered under Manage, Connections.

It is a library, not a service: nothing here listens on a port.

## How it works

A plugin class declares:

- its configuration form, as a Pydantic model given as the type parameter
  (`BaseEvaluationPlugin[MyConfig]`); the engine turns it into a JSON schema for the web form;
- its inputs (datasets, models, LLMs, data shapes, resources) with `@evaluation_input`, and the
  project settings it needs with `@project_config`;
- `evaluate(config_data)`, which does the work and returns any intermediate result;
- one or more `@metric("...")` methods that turn that result into a list of `Measure`.

Two AISC services import plugins through this library:

- the execution engine (`aisc-backend`, in `apps/backend`) loads a plugin to read its form, inputs and
  metrics;
- the evaluation worker (`aisc-eval-worker`, in `apps/eval`) runs it in its own process: it feeds the
  inputs (`set_input_content`), the project settings and the secrets (environment variables
  `AISC_SECRET_<KEY>`), calls `evaluate`, collects the measures, progress reports and artifacts.

Both find plugins with [aisc-plugin-manager](https://github.com/lux-ai-factory/aisc-plugin-manager).
The platform service also mounts this library, to test a connection with the same client plugins use.

Main parts of `src/aisc_plugin_interface/`:

| Module | What it holds |
|---|---|
| `base_evaluation_plugin.py` | `BaseEvaluationPlugin`, `PluginFeatureFlags`, `ProgressBar` |
| `decorators/` | `@metric`, `@evaluation_input`, `@project_config` |
| `models/` | `Measure`, `MetricVisualization`, `TaskProgress`, `InputDefinition`, `DataShape`, `LLMConfig`, `ResourceConfig`, ... |
| `input_providers/` | readers that turn input bytes into Python objects: CSV, JSON, Parquet (needs pandas), ONNX (needs onnxruntime) |
| `connections.py` | `EndpointClient`: call a system registered under Manage, Connections (OpenAI-compatible, REST, A2A, Open Inference Protocol) |
| `system_under_test.py` | `@system_under_test`: point a tool that already speaks a standard protocol at the system under test |
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

PyPI's latest release is 0.3.0. It does not contain `connections.py`, `system_under_test.py` or
`targets.py`, which exist only on the `feat/unified-modules` branch that the AISC stack uses. A plugin
that needs them installs from git:

```bash
uv add git+https://github.com/lux-ai-factory/aisc-plugin-interface --branch feat/unified-modules
```

The `version` in this branch's `pyproject.toml` still reads 0.2.6, so a version constraint such as
`>=0.3.0` is not met by a git install of this branch; constrain on the git source instead.

### Inside the AISC stack

There is no compose service for this library. In `docker-compose.development.yml` (from the aisc
repo root) the services `aisc-backend`, `aisc-backend-migrate` and `aisc-eval-worker` mount `./shared`
and, on start, run `uv pip install --no-deps -e /app/shared/plugin-interface`, so they use this
checkout. The `platform` service mounts `shared/plugin-interface/src` read-only and puts it on its
`PYTHONPATH`. The library therefore comes up with the stack (run `./scripts/secrets.sh` once first,
then the `docker compose ... up` command in the aisc README); a change here reaches those services
when they restart. Images built on their own, and `docker-compose.engine-standalone.yml`, use the
version locked in `apps/backend/uv.lock` and `apps/eval/uv.lock` (0.3.0 from PyPI) instead.

## Writing a plugin

Create a project and add the library:

```bash
mkdir my-aisc-plugin && cd my-aisc-plugin
uv init --lib
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
progress, the optional hooks and calling a system under test.

## Configuration

The library reads these environment variables. In AISC the evaluation worker sets them for the plugin
process; a plugin author does not set them.

| Variable | Read by | Meaning | Default |
|---|---|---|---|
| `PLATFORM_URL` | `connections.EndpointClient`, `@system_under_test` | Base URL of the platform service, which resolves connections and targets and issues run keys | none: resolving a connection fails without it |
| `PLATFORM_CONNECTIONS_TOKEN` | same | Service token sent as `X-AISC-Service-Token` to the platform's internal routes | none: resolving a connection fails without it |
| `CONNECTIONS_ALLOWED_HOSTS` | `connections.call` | Comma-separated hosts (or `host:port`) that may be called although they are internal. Used only when no allowlist is passed in; a client resolved through the platform takes the project's rule from the platform | empty |
| `AISC_SECRET_<KEY>` | `get_secret`, `require_secret` | A project secret. `<KEY>` is the setting key in upper case, other characters replaced by `_` | none |

## Tests

The tests need no database and no network beyond local stub servers on 127.0.0.1. The project
declares no dev dependencies, so pytest is added for the run; use `python -m pytest` so that the
`tests` package can be imported:

```bash
uv run --with pytest python -m pytest -q
```

## Layout

```text
src/aisc_plugin_interface/   the library (see the table above)
tests/                       unit tests for connections, targets and @system_under_test
PLUGIN_DEVELOPER_GUIDE.md    how to write a plugin
```

## Contributing

`feat/unified-modules` is the branch the AISC stack uses and the only one to work on. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the contributor licence terms.

## License

This project is licensed under the [Apache License 2.0](LICENSE.md).
© 2024–2026 Université du Luxembourg and Luxembourg Institute of Science and Technology (LIST).

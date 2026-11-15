"""A plugin's manifest: everything the engine shows and configures of a plugin, as plain JSON, so the engine
never imports plugin code. The worker reads it once, in the plugin's own environment and as a user of its
own, and the engine stores it on the plugin's row.

The three form hooks (HOOKS) are the only plugin code the engine ever asks for after that: the worker runs
them the same way, one call at a time.

    python -m aisc_plugin_interface.manifest describe <module>
    python -m aisc_plugin_interface.manifest hook          (reads {"module", "plugin", "hook", "args"} on stdin)

Each prints one JSON document on stdout, or exits 1 with a one-line reason on stderr. What the plugin itself
prints goes to stderr, so it never mixes with the answer. The file runs as a script too (the worker copies
it next to a run), so it imports only by absolute names.
"""

import base64
import contextlib
import importlib
import inspect
import json
import sys
from typing import Any

from aisc_plugin_interface import BaseEvaluationPlugin
from aisc_plugin_interface.assesses import describe
from aisc_plugin_interface.models.measure import ChartType, MetricVisualization

#: the plugin code the engine may ask for once a plugin is installed: its form hooks, nothing else
HOOKS = ("on_config_change", "parse_config_from_dataset", "get_metric_visualizations")


def _json(value: Any) -> Any:
    """A pydantic model or a list of them as plain JSON values; anything else as it is."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    return value


def _overridden(cls, hook: str) -> bool:
    return getattr(cls, hook, None) is not getattr(BaseEvaluationPlugin, hook)


def _form(plugin) -> tuple[dict | None, dict | None]:
    """The form the engine shows first: the hook's answer with no data, else the declared schema; (None, None)
    when neither can be built (no configuration model, or a form that needs what the sandbox does not have), so
    the engine can say so instead of showing an empty form."""
    try:
        _config, schema, ui = plugin.on_config_change(None)
    except Exception:  # noqa: BLE001 - a hook that needs data or a network falls back to the declared form
        try:
            schema, ui = plugin.get_full_schema()
        except Exception:  # noqa: BLE001 - said as None
            return None, None
    return (_json(schema) if isinstance(schema, dict) else None), (
        _json(ui) if isinstance(ui, dict) else {}
    )


def manifest_of(cls) -> dict:
    """What the plugin class declares, as plain JSON values."""
    plugin = cls()
    declaration = describe(cls)
    # read from the class: an instance lists its metrics through every property, a failing one included
    metrics = declaration["metrics"]
    try:
        charts = _json(list(plugin.get_metric_visualizations({})))
    except Exception:  # noqa: BLE001 - charts that need a configuration fall back to one table of every metric
        charts = [
            _json(MetricVisualization(chart_type=ChartType.TABLE, metrics=metrics))
        ]
    schema, ui = _form(plugin)
    return {
        "name": cls.__name__,
        "display_name": cls.display_name,
        "help_text": plugin.help_text,
        "display_icon": plugin.display_icon,
        "feature_flags": _json(plugin.feature_flags),
        "input_definitions": _json(list(plugin.input_definitions)),
        "project_config_definitions": _json(list(plugin.project_config_definitions)),
        "form_schema": schema,
        "form_ui_schema": ui,
        "metrics": metrics,
        "metric_visualizations": charts,
        "declaration": declaration,
        "adopted": getattr(cls, "target_kinds", None) is not None,
        "hooks": sorted(hook for hook in HOOKS if _overridden(cls, hook)),
    }


def _plugin_classes(module) -> dict[str, type]:
    return {
        obj.__name__: obj
        for _name, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, BaseEvaluationPlugin)
        and not getattr(obj.evaluate, "__isabstractmethod__", False)
    }


def manifest_of_module(module_name: str) -> dict:
    """{"plugins": {class name: manifest}} of every plugin class of the module; ValueError when it has none."""
    classes = _plugin_classes(importlib.import_module(module_name))
    if not classes:
        raise ValueError(
            f"{module_name} holds no plugin class (a subclass of BaseEvaluationPlugin)"
        )
    return {
        "plugins": {name: manifest_of(cls) for name, cls in sorted(classes.items())}
    }


def run_hook(module_name: str, plugin: str, hook: str, args: dict) -> Any:
    """One form hook of one plugin class, with JSON arguments, answering JSON. Only HOOKS are run."""
    if hook not in HOOKS:
        raise ValueError(
            f"{hook} is not a form hook: only {', '.join(HOOKS)} are run for the engine"
        )
    cls = _plugin_classes(importlib.import_module(module_name)).get(plugin)
    if cls is None:
        raise ValueError(f"{module_name} has no plugin class {plugin}")
    instance = cls()
    if hook == "parse_config_from_dataset":
        return _json(
            instance.parse_config_from_dataset(
                base64.b64decode(args.get("content_b64") or "")
            )
        )
    if hook == "on_config_change":
        config, schema, ui = instance.on_config_change(args.get("config"))
        return [_json(config), _json(schema), _json(ui)]
    return _json(list(instance.get_metric_visualizations(args.get("config") or {})))


def main(argv: list[str]) -> int:
    answer = sys.stdout
    try:
        # the plugin's own prints go to stderr: stdout carries the answer only
        with contextlib.redirect_stdout(sys.stderr):
            if argv[:1] == ["describe"] and len(argv) == 2:
                result = manifest_of_module(argv[1])
            elif argv == ["hook"]:
                asked = json.loads(sys.stdin.read())
                result = run_hook(
                    asked["module"],
                    asked["plugin"],
                    asked["hook"],
                    asked.get("args") or {},
                )
            else:
                raise ValueError(
                    "usage: describe <module> | hook (a JSON request on stdin)"
                )
            text = json.dumps(result)
    except Exception as exc:  # noqa: BLE001 - every failure is one line on stderr and exit 1
        reason = " ".join(f"{type(exc).__name__}: {exc}".split())
        print(reason[:2000], file=sys.stderr)
        return 1
    answer.write(text)
    answer.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

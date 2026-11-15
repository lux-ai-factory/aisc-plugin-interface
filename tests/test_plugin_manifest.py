"""A plugin's manifest: everything the engine needs to show and configure a plugin (its declarations, form,
metrics and charts, and which form hooks it overrides), as plain JSON, read once in a sandbox so the engine
never imports plugin code. The three form hooks run from the command line the same way, and nothing else does."""
import importlib
import json
import os
import subprocess
import sys
import textwrap

import pytest

from aisc_plugin_interface import BaseEvaluationPlugin, describe


def manifest():
    try:
        return importlib.import_module("aisc_plugin_interface.manifest")
    except ImportError:
        pytest.fail("the interface cannot describe a plugin as JSON (aisc_plugin_interface.manifest)")


class CardOnly(BaseEvaluationPlugin):
    def evaluate(self, config_data):
        return None


class WithDatasetParser(CardOnly):
    def parse_config_from_dataset(self, file_content):
        return {"rows": len(file_content.splitlines())}


class ChartsNeedAConfig(CardOnly):
    def get_metric_visualizations(self, config_data):
        return config_data["charts"]  # a KeyError with the empty config the manifest reads it with


DEMO = textwrap.dedent('''
    from aisc_plugin_interface import BaseEvaluationPlugin

    class DemoPlugin(BaseEvaluationPlugin):
        def evaluate(self, config_data):
            return None
''')


@pytest.fixture
def demo_module(tmp_path):
    (tmp_path / "demo_plugin").mkdir()
    (tmp_path / "demo_plugin" / "__init__.py").write_text(DEMO)
    return tmp_path


def test_the_manifest_is_plain_json_of_what_the_class_declares():
    got = manifest().manifest_of(CardOnly)
    assert json.loads(json.dumps(got)) == got
    assert got["declaration"] == describe(CardOnly) and got["hooks"] == []
    assert set(got) >= {"name", "display_name", "help_text", "display_icon", "feature_flags",
                        "input_definitions", "project_config_definitions", "form_schema", "form_ui_schema",
                        "metrics", "metric_visualizations", "declaration", "adopted", "hooks"}


def test_an_overridden_hook_is_named():
    assert manifest().manifest_of(WithDatasetParser)["hooks"] == ["parse_config_from_dataset"]


def test_visualizations_that_need_a_config_fall_back_to_one_table():
    charts = manifest().manifest_of(ChartsNeedAConfig)["metric_visualizations"]
    assert len(charts) == 1 and json.dumps(charts[0]).upper().count("TABLE") >= 1


def test_the_command_describes_a_module(demo_module):
    manifest()
    out = subprocess.run([sys.executable, "-m", "aisc_plugin_interface.manifest", "describe", "demo_plugin"],
                         env={**os.environ, "PYTHONPATH": str(demo_module)}, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-500:]
    assert "DemoPlugin" in json.loads(out.stdout)["plugins"]


def test_a_module_without_a_plugin_is_refused_with_a_reason(tmp_path):
    manifest()
    (tmp_path / "empty_plugin.py").write_text("x = 1\n")
    out = subprocess.run([sys.executable, "-m", "aisc_plugin_interface.manifest", "describe", "empty_plugin"],
                         env={**os.environ, "PYTHONPATH": str(tmp_path)}, capture_output=True, text=True)
    assert out.returncode == 1 and out.stdout == "" and out.stderr.strip()


def test_a_hook_other_than_the_three_is_refused(demo_module, monkeypatch):
    m = manifest()
    monkeypatch.syspath_prepend(str(demo_module))
    with pytest.raises(ValueError, match="evaluate"):
        m.run_hook("demo_plugin", "DemoPlugin", "evaluate", {})


def test_the_three_form_hooks_are_the_only_ones():
    assert manifest().HOOKS == ("on_config_change", "parse_config_from_dataset", "get_metric_visualizations")


class FormThatDoesNotBuild(CardOnly):
    def get_full_schema(self):
        raise RuntimeError("the form needs a server")


def test_a_form_that_cannot_be_built_is_said_so_not_shown_empty():
    got = manifest().manifest_of(FormThatDoesNotBuild)
    assert got["form_schema"] is None and got["form_ui_schema"] is None

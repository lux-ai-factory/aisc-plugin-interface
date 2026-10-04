"""A plugin class built on a base that declares inputs (data drift and data anomaly share their two datasets)
keeps them when the subclass declares one more: the target input added by @system_under_test or
@dataset_through_target must not wipe the inherited datasets."""
from aisc_plugin_interface import BaseEvaluationPlugin, InputType, evaluation_input
from aisc_plugin_interface.input_providers.json_input_provider import JsonInputProvider


@evaluation_input(name="reference", label="Reference", input_provider_class=JsonInputProvider, input_type=InputType.DATASET)
class Base(BaseEvaluationPlugin):
    def evaluate(self, config_data):
        return None

    def export_metrics(self, *a, **k):
        return []


@evaluation_input(name="target", label="Target", input_type=InputType.RESOURCE, required=False)
class Child(Base):
    pass


def test_a_subclass_keeps_the_inputs_it_inherits():
    assert [d.name for d in Child().input_definitions] == ["reference", "target"]
    child = Child()
    child.set_input_content("reference", b'{"a": 1}')
    assert child.get_input_data("reference") == {"a": 1}


def test_the_base_is_not_changed_by_its_subclass():
    assert [d.name for d in Base().input_definitions] == ["reference"]

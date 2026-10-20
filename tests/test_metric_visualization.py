"""A chart a plugin declares can say how its values read: a percent chart's scores are ratios 0..1."""
import pytest
from pydantic import ValidationError

from aisc_plugin_interface import ChartType, MetricVisualization, ValueFormat


def test_a_chart_reads_as_plain_numbers_unless_it_says_otherwise():
    chart = MetricVisualization(chart_type=ChartType.BARS, metrics=["Pass rate"])
    assert chart.value_format is None


def test_a_chart_of_ratios_reads_as_percent():
    chart = MetricVisualization(chart_type=ChartType.BARS, metrics=["Pass rate"], value_format="percent")
    assert chart.value_format == ValueFormat.PERCENT
    assert chart.model_dump(mode="json")["value_format"] == "percent"


def test_an_unknown_format_is_refused():
    with pytest.raises(ValidationError):
        MetricVisualization(chart_type=ChartType.BARS, metrics=["x"], value_format="permille")

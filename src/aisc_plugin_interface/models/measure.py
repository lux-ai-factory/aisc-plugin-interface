from datetime import datetime
import enum

from pydantic import BaseModel


class MetricDirection(str, enum.Enum):
    HIGHER_IS_BETTER = "higher"
    LOWER_IS_BETTER = "lower"
    NEUTRAL = "neutral"

class Measure(BaseModel):
    name: str
    description: str | None = None
    unit: str | None = None
    score: float
    time: datetime = datetime.now()
    error: str | None = None
    dimensions: dict[str, str | int | bool] | None = None
    direction: MetricDirection | None = None


class ChartType(str, enum.Enum):
    TABLE = "table"
    LINE = "line"
    RADAR = "radar"
    SCATTER = "scatter"
    KDE = "kde"
    BARS = "bars"
    PIE = "pie"
    CSV = "csv"


class ValueFormat(str, enum.Enum):
    """How a chart's values read. PERCENT: the scores are ratios from 0 to 1, shown as 0% to 100%."""
    NUMBER = "number"
    PERCENT = "percent"


class MetricVisualization(BaseModel):
    chart_type: ChartType
    metrics: list[str]
    title: str | None = None
    description: str | None = None
    filter_dimensions: dict[str, list[str | int | bool]] | None = None
    metric_label_dimension: str | None = None
    group_by_dimensions: list[str] | None = None
    #: None reads as plain numbers, as before the field
    value_format: ValueFormat | None = None

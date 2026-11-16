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


class MetricVisualization(BaseModel):
    chart_type: ChartType
    metrics: list[str]
    title: str | None = None
    description: str | None = None
    filter_dimensions: dict[str, list[str | int | bool]] | None = None
    metric_label_dimension: str | None = None
    group_by_dimensions: list[str] | None = None
    #: Split the chart into one plot per value of these dimensions ("iterate" a
    #: dimension): within each plot, group_by_dimensions remain the series.
    facet_by_dimension: list[str] | None = None

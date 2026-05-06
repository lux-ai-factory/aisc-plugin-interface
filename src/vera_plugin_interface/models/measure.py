from datetime import datetime
import enum

from pydantic import BaseModel


class Measure(BaseModel):
    name: str
    description: str | None = None
    unit: str | None = None
    score: float
    time: datetime = datetime.now()
    error: str | None = None
    dimensions: dict[str, str | int | bool] | None = None


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

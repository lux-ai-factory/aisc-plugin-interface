import copy
import inspect
import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sized
from typing import (
    Any,
    get_args,
    get_origin,
    Callable,
    Tuple,
    TypeAlias,
    TypeVar,
    Generic,
    final,
    Type,
    Self,
)

from pydantic import BaseModel, Field

from aisc_plugin_interface.decorators.evaluation_input import InputDefinition
from aisc_plugin_interface.utils import classproperty
from aisc_plugin_interface.models.task import TaskProgress
from aisc_plugin_interface.input_providers.base_input_provider import BaseInputProvider
from aisc_plugin_interface.models.measure import Measure, MetricVisualization, ChartType

ProgressCallback: TypeAlias = Callable[[TaskProgress], None]
ArtifactCallback: TypeAlias = Callable[[str, bytes], None]


class PluginFeatureFlags(BaseModel):
    can_parse_config_from_dataset: bool = Field(
        False, description="Show the dataset dropdown"
    )
    show_dimensions_visualisation: bool = Field(
        False, description="Show the dimensions visualisation on the results page"
    )
    extra: dict = Field({}, description="Additional feature flags")


class BaseEvaluationPlugin[T: BaseModel](ABC):
    """
    Abstract Base Class for evaluation plugins.
    Plugins should inherit from this class and provide a Pydantic model for their configuration.

    Example:
        class MyPlugin(BaseEvaluationPlugin[MyConfigModel]):
            ...
    """

    # UI Schema for RJSF (react-jsonschema-form) to customize form appearance
    form_ui_schema: dict = {}

    # The plugin name that will be displayed in the UI
    plugin_name: str | None = None

    # Controls the icon displayed in the plugin list.
    # Use a Material Design icon name https://fonts.google.com/icons
    ui_icon: str | None = None

    _input_definitions: list[InputDefinition] = []
    _input_provider_types: dict[str, Type[BaseInputProvider]] = {}

    def __init__(self):
        self._input_provider_instances: dict[str, BaseInputProvider] = {}
        self._progress_callback: ProgressCallback | None = None
        self._artifact_callback: ArtifactCallback | None = None
        self._logger = None

    @classmethod
    def is_direct_subclass(cls: type[Self]) -> bool:
        return BaseEvaluationPlugin in cls.__bases__

    @classproperty
    def display_name(cls: type[Self]) -> str:
        """
        Returns a display name for the plugin.

        If the class defines a `plugin_name` attribute, its value is returned.
        If not, the class's name (`cls.__name__`) is used as a fallback.
        """
        # NOTE: use cls.plugin_name only if cls is direct subclass of BaseEvaluationPlugin
        plugin_name = cls.plugin_name
        default = cls.__name__

        if cls.is_direct_subclass():
            # cls is a direct subclass of BaseInputProvider
            return plugin_name or default
        else:
            if plugin_name is None:
                return default
            # avoid using the plugin name that was set for the parent plugin
            for base in cls.__bases__:
                if (
                    issubclass(base, BaseEvaluationPlugin)
                    and plugin_name == base.plugin_name
                ):
                    return default
            return plugin_name

    @property
    def logger(self):
        """
        Returns the cached logger for this plugin class.

        The logger is shared across all instances of the class (per-class caching)
        and named as "<module> - __name__".
        """
        if self._logger is None:
            cls = self.__class__
            self._logger = logging.getLogger(f"{cls.__module__} - {cls.__name__}")
        return self._logger

    @property
    def feature_flags(self) -> PluginFeatureFlags:
        """
        Controls UI behavior on the frontend.
        Override this property in your subclass to change defaults.
        """
        return PluginFeatureFlags()

    @property
    def input_definitions(self) -> list[InputDefinition]:
        """
        Controls the evaluation input form at evaluation creation
        """
        return self._input_definitions

    @property
    def display_icon(self) -> str:
        """
        Controls the icon displayed in the plugin list.
        Use a Material Design icon name
        https://fonts.google.com/icons

        For a new plugin, either redefine this property, or set the class attribute `ui_icon`
        """
        return self.ui_icon or "extension"

    @property
    def config_type(self) -> type[T]:
        """
        Retrieves the Pydantic model type used for plugin configuration.
        """
        for base in getattr(self.__class__, "__orig_bases__", []):
            if get_origin(base) is BaseEvaluationPlugin:
                return get_args(base)[0]
        raise TypeError("Could not determine Config type T")

    def get_metrics(self) -> list[str]:
        """
        Returns a list of all metric names defined in this plugin via the @metric decorator.
        """
        metrics = []
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if hasattr(method, "metric_name"):
                metrics.append(method.metric_name)
        return metrics

    def get_metric_visualizations(self, config_data: dict) -> list[MetricVisualization]:
        """
        Returns a list of MetricVisualization objects to render a list of
        visualizations on the front end and the metrics to display for each

        The config_data could be used to define specific visualizations from the config

        By default, returns a single visualization (TABLE) with all metrics
        """
        return [
            MetricVisualization(chart_type=ChartType.TABLE, metrics=self.get_metrics())
        ]

    def export_metrics(self, *args, **kwargs) -> list[Measure]:
        """
        Executes all methods decorated with @metric and aggregates their results.
        """
        results: list[Measure] = []
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if hasattr(method, "metric_name"):
                metric_measures: list[Measure] = method(*args, **kwargs)
                results.extend(metric_measures)
        return results

    @abstractmethod
    def evaluate(self, config_data: dict) -> Any:
        """
        The main execution logic of the plugin.
        Developers should process the dataset/model here and return an intermediate result
        that will be passed to the metric methods.
        """
        raise NotImplementedError

    @final
    def _set_progress_callback(
        self, progress_callback: ProgressCallback | None
    ) -> None:
        """
        Internal: called by the evaluation runtime (eval module) to feedback progress reporting.
        Plugin implementations should not call or override this.
        """
        if progress_callback is not None and not callable(progress_callback):
            raise TypeError("Progress sink must be callable or None")
        self._progress_callback = progress_callback

    @final
    def report_progress(self, task_progress: TaskProgress) -> None:
        """
        Public, stable API for plugin authors.
        No-op if no sink is configured by the evaluation runtime.
        """
        if self._progress_callback is None:
            return
        self._progress_callback(task_progress)

    @final
    def progress_bar(
        self,
        iterable,
        *,
        total: int | None = None,
        desc: str | None = None,
        start: int = 0,
        with_index: bool = False,
        extra: dict | None = None,
    ) -> "ProgressBar":
        """
        Wrap an iterable to yield items (optionally with index)
        while emitting progress for each item that is processed.

        Returns a ProgressBar yielding items (or index, item) with automatic progress reporting.

        Args:
            iterable: Items to iterate.
            total: Optional total length.
            desc: Label.
            start: Enumeration start.
            with_index: Yield index with items.
            extra: Extra progress fields.
        """
        return ProgressBar(
            iterable=iterable,
            plugin=self,
            total=total,
            desc=desc,
            start=start,
            with_index=with_index,
            extra=extra,
        )

    @final
    def _set_artifact_callback(
        self, artifact_callback: ArtifactCallback | None
    ) -> None:
        """Internal: called by the evaluation runtime to hook into artifact uploading."""
        self._artifact_callback = artifact_callback

    @final
    def upload_artifact(self, name: str, content: bytes) -> None:
        """
        Public API for plugin authors to upload arbitrary files.
        """
        if self._artifact_callback:
            self._artifact_callback(name, content)
        else:
            self.logger.warning(
                f"Artifact callback not configured. Dropping artifact: {name}"
            )

    def set_input_content(self, name: str, file_content: bytes | None) -> None:
        """
        Called by the runtime. Instantiates the provider mapped via @input.
        """
        provider_cls = self._input_provider_types.get(name)
        if provider_cls and file_content is not None:
            self._input_provider_instances[name] = provider_cls(file_content)

    def get_input_data(self, name: str) -> Any | None:
        """
        Get data from InputProvider using name set in evaluation_input decorator
        """
        provider = self._input_provider_instances.get(name)
        if provider is None:
            return None
        return provider.get_data()

    def get_config_form_schema(self) -> dict:
        """
        Generates a JSON Schema from the Pydantic config model for the frontend UI.
        """
        return self.config_type.model_json_schema(mode="validation")

    def validate_config_form_data(self, config_form_data: dict) -> T:
        """
        Validates incoming form data from frontend UI/backend DB against the Pydantic config model.
        """
        return self.config_type.model_validate(config_form_data)

    def get_config_form_ui_schema(self) -> dict:
        """
        Returns a deep copy of the UI schema for form customization.
        """
        return copy.deepcopy(self.form_ui_schema)

    def form_schema_to_internal(self, form_schema: T) -> dict:
        """
        Optional: Converts the validated Pydantic model used for the UI form
                  into a dict for internal use
        This can be overridden to add/change the structure of the input config data
        for use in the evaluate method
        """
        return form_schema.model_dump()

    def get_full_schema(self) -> Tuple[dict, dict]:
        """Helper to get the fresh, static baseline."""
        return self.get_config_form_schema(), self.get_config_form_ui_schema()

    # form_data passed here may be incomplete, so we don't validate and use MyConfigModel
    # It is the developer's responsibility to check for and use data accordingly here
    def on_config_change(self, form_data: T | None) -> Tuple[T | None, dict, dict]:
        """
        Hook called whenever the user changes a form value.
        Allows the plugin to dynamically update the schema (e.g. drop downs),
        the data (e.g. auto-fill), or the UI (e.g. hide fields).
        """
        # Default: Do nothing, just return what came in
        schema, ui_schema = self.get_full_schema()
        return form_data, schema, ui_schema

    def parse_config_from_dataset(self, file_content: bytes) -> dict | None:
        """
        Optional: Try to parse a valid config from the dataset.
        Use a InputProvider to read the file contents
        """
        return None


Item = TypeVar("Item")


class ProgressBar(Generic[Item]):
    """
    Iterator that wraps an iterable, yielding each item (and optionally its index)
    and reporting progress via a plugin.

    Each emit reports the progress for the current item that is processed.

    Args:
        iterable: Items to iterate over.
        plugin: Receives progress reports (must implement `report_progress`).
        total: Length of iteration (inferred if not given).
        desc: Progress label.
        start: Enumeration start index (min 1 if negative).
        with_index: Yield (index, item) pairs if True.
        extra: Extra fields for progress payloads.
    """

    def __init__(
        self,
        iterable: Iterable[Item],
        *,
        plugin: BaseEvaluationPlugin,
        total: int | None = None,
        desc: str | None = None,
        start: int = 0,
        with_index: bool = False,
        extra: dict | None = None,
    ):
        self._start = start
        self._with_index = with_index

        # infer total if not provided
        if total is None:
            if isinstance(iterable, Sized):
                total = len(iterable)
            else:
                iterable = list(iterable)
                total = len(iterable)
        self._total = total

        self._desc = desc or ""
        self._extra = {} if extra is None else extra.copy()

        self._report_progress = getattr(plugin, "report_progress", None)

        # prepare enumerate on iterable with correct start index
        self._enumerated_iter = enumerate(iterable, start=self._start)

        # start emitting at 0
        self.emit(0)

    @property
    def desc(self) -> str:
        return self._desc

    def set_description(self, description: str):
        self._desc = description

    @property
    def extra(self) -> dict:
        return self._extra

    def set_extra(self, fields: dict) -> None:
        self._extra.update(fields)

    def emit(self, i: int):
        """
        Report progress for index `i` using the plugin.
        """
        if self._report_progress and self._total:
            progress = min(float(min(i, self._total)) / self._total, 1.0)
            payload = {
                "iteration": int(i),
                "total": self._total,
            }
            if self._desc:
                payload["desc"] = self._desc
            if self._extra:
                payload.update(self._extra)
            self._report_progress(
                TaskProgress(
                    progress=progress,
                    extra=payload,
                )
            )

    def __len__(self) -> int:
        return self._total

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> Item | tuple[int, Item]:
        """
        Yield the next item (and index if requested),
        emitting progress for the current item being processed.

        Raises:
            StopIteration when exhausted.
        """
        idx, item = next(self._enumerated_iter)

        self.emit(idx - self._start + 1)
        return (idx, item) if self._with_index else item

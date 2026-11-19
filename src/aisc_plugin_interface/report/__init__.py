"""The report ABI: the contract an eval plugin implements to draw its own report section.

An eval plugin returns a :class:`BaseToolRenderer` from
:meth:`aisc_plugin_interface.base_evaluation_plugin.BaseEvaluationPlugin.get_report_renderer`.
The backend hands that renderer a read-only :class:`ToolRun` — one plugin run of the engine,
with no database session — and prints the HTML fragment it returns, in its own report document.

A plugin that does not implement the method (or returns ``None``) is **not** left out: the
report falls back to its generic measurements table, so every plugin, including one written
before this contract existed, still renders something useful.

This subpackage is the reusable half of the pilot's report-plugin interface
(``aisc_report_plugin_interface``): the tool-renderer contract, the i18n catalogues, the
package templating and the inline-SVG chart helper. It is imported only by the report engine
and by plugins that draw a report section, so it lives behind the ``[report]`` extra of this
distribution.
"""
from .i18n import fill
from .templating import render_package_template
from .tools import (
    BaseToolRenderer,
    ToolRun,
    headline_items,
    normalise_tool_name,
    tool_keys,
)

#: version of the tool-renderer contract
INTERFACE_VERSION = 2

__all__ = [
    "BaseToolRenderer",
    "ToolRun",
    "headline_items",
    "normalise_tool_name",
    "tool_keys",
    "render_package_template",
    "fill",
    "INTERFACE_VERSION",
]

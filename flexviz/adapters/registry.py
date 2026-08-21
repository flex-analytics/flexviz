"""Renderer registry and capability validation for FlexViz adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import AbstractAdapter
    from ..spec import DashboardSpec, FigureSpec


@dataclass(frozen=True)
class RendererCapabilities:
    """Declarative capability set for one renderer."""

    name: str
    supported_trace_types: frozenset[str]

    def supports_trace_type(self, trace_type: str) -> bool:
        return trace_type in self.supported_trace_types


@dataclass(frozen=True)
class RendererDefinition:
    """Adapter lookup + capabilities for one renderer."""

    name: str
    adapter_import_path: str
    capabilities: RendererCapabilities

    def adapter_cls(self) -> type["AbstractAdapter"]:
        module_name, _, class_name = self.adapter_import_path.rpartition(".")
        module = __import__(module_name, fromlist=[class_name])
        return getattr(module, class_name)

    def build_adapter(self) -> "AbstractAdapter":
        adapter_cls = self.adapter_cls()
        return adapter_cls()


PLOTLY_TRACE_TYPES = frozenset(
    {
        "line",
        "histogram",
        "box",
        "bar",
        "pie",
        "treemap",
        "histogram2d",
        "corr_heatmap",
        "geo_histogram2d",
        "geo_line",
    }
)

ECHARTS_TRACE_TYPES = frozenset(
    {
        "line",
        "histogram",
        "box",
        "bar",
        "pie",
        "treemap",
        "histogram2d",
        "corr_heatmap",
    }
)

_RENDERERS: dict[str, RendererDefinition] = {
    "plotly": RendererDefinition(
        name="plotly",
        adapter_import_path="flexviz.adapters.plotly_adapter.PlotlyAdapter",
        capabilities=RendererCapabilities(
            name="plotly",
            supported_trace_types=PLOTLY_TRACE_TYPES,
        ),
    ),
    "echarts": RendererDefinition(
        name="echarts",
        adapter_import_path="flexviz.adapters.echarts_adapter.EChartsAdapter",
        capabilities=RendererCapabilities(
            name="echarts",
            supported_trace_types=ECHARTS_TRACE_TYPES,
        ),
    ),
}


def supported_renderers() -> tuple[str, ...]:
    return tuple(_RENDERERS)


def normalize_renderer_name(renderer: str) -> str:
    name = str(renderer).strip().lower()
    if name not in _RENDERERS:
        supported = ", ".join(repr(item) for item in supported_renderers())
        raise ValueError(f"Unknown renderer {renderer!r}. Supported: {supported}.")
    return name


def get_renderer_definition(renderer: str) -> RendererDefinition:
    return _RENDERERS[normalize_renderer_name(renderer)]


def build_adapter(renderer: str) -> "AbstractAdapter":
    return get_renderer_definition(renderer).build_adapter()


def _figure_label(fig_spec: "FigureSpec") -> str:
    title = (fig_spec.layout or {}).get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    if isinstance(title, dict):
        text = title.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
    return fig_spec.uid


def validate_dashboard_renderer(renderer: str, spec: "DashboardSpec") -> str:
    definition = get_renderer_definition(renderer)
    for fig_spec in spec.figures:
        for ts in fig_spec.traces:
            if definition.capabilities.supports_trace_type(ts.trace_type):
                continue
            label = _figure_label(fig_spec)
            supported_alternatives = [
                other.name
                for other in _RENDERERS.values()
                if other.capabilities.supports_trace_type(ts.trace_type)
            ]
            alternative_text = ""
            if supported_alternatives:
                alternatives = ", ".join(repr(item) for item in supported_alternatives)
                alternative_text = (
                    f" Use {alternatives} or remove the unsupported trace."
                )
            raise ValueError(
                f"renderer={definition.name!r} does not support trace type "
                f"{ts.trace_type!r} in figure {label!r}.{alternative_text}"
            )
    return definition.name

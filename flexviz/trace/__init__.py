"""Trace registry and factory for flexviz.

All concrete FlexTrace subclasses register themselves here so that the
server can reconstruct trace objects from a TraceSpec at request time.
"""

from __future__ import annotations

from .bar import BarPlot
from .base import FlexTrace
from .box import BoxPlot
from .corr_heatmap import CorrHeatmap
from .geo_hist2d import GeoHistogram2D
from .hist import Histogram
from .hist2d import Histogram2D
from .line import LinePlot
from .pie import PiePlot
from .geo_line import GeoLine
from .treemap import TreeMap
from ..spec import TraceSpec

# ---------------------------------------------------------------------------
# Trace type registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[FlexTrace]] = {
    LinePlot.trace_type: LinePlot,
    Histogram.trace_type: Histogram,
    BoxPlot.trace_type: BoxPlot,
    BarPlot.trace_type: BarPlot,
    PiePlot.trace_type: PiePlot,
    Histogram2D.trace_type: Histogram2D,
    GeoHistogram2D.trace_type: GeoHistogram2D,
    CorrHeatmap.trace_type: CorrHeatmap,
    GeoLine.trace_type: GeoLine,
    TreeMap.trace_type: TreeMap,
}


def recompute_policy_table() -> str:
    """Render the per-trace zoom re-aggregation policy as a Markdown table.

    Drives the ``<!-- recompute-policy -->`` block in ``Architecture.md`` so the
    documented policy can never silently drift from the registered traces (a
    test regenerates this and compares).  Each trace's ``recompute_policy_doc``
    is the single source of the human-readable description; the binding axes
    themselves live in ``_default_recompute_axes``.
    """
    rows = [
        "| `trace_type` | re-aggregates on zoom of |",
        "|---|---|",
    ]
    for trace_type in sorted(_REGISTRY):
        cls = _REGISTRY[trace_type]
        rows.append(f"| `{trace_type}` | {cls.recompute_policy_doc} |")
    return "\n".join(rows)


def select_policy_table() -> str:
    """Render the per-trace selection geometry as a Markdown table.

    Drives the ``<!-- select-policy -->`` block in ``Architecture.md`` (a test
    regenerates this and compares).  Each trace's ``select_policy_doc`` is the
    single source of the human-readable description; the machine-readable
    geometry lives in ``_make_selection_spec`` → ``TraceSpec.selection`` (a
    ``TraceSelectionSpec``) which the generic client runtime interprets.
    """
    rows = [
        "| `trace_type` | selects on |",
        "|---|---|",
    ]
    for trace_type in sorted(_REGISTRY):
        cls = _REGISTRY[trace_type]
        rows.append(f"| `{trace_type}` | {cls.select_policy_doc} |")
    return "\n".join(rows)


def build_trace_from_spec(spec: TraceSpec) -> FlexTrace:
    """Reconstruct a FlexTrace from a TraceSpec (called server-side per request).

    Parameters
    ----------
    spec:
        TraceSpec from the incoming VisualizationSpec.

    Returns
    -------
    FlexTrace
        A concrete trace instance with the uid and settings from the spec.

    Raises
    ------
    ValueError
        If ``spec.trace_type`` has no registered implementation.
    """
    cls = _REGISTRY.get(spec.trace_type)
    if cls is None:
        registered = list(_REGISTRY)
        raise ValueError(
            f"Unknown trace type {spec.trace_type!r}. "
            f"Registered types: {registered}"
        )
    return cls.from_trace_spec(spec)


__all__ = [
    "BarPlot",
    "BoxPlot",
    "FlexTrace",
    "GeoHistogram2D",
    "Histogram",
    "CorrHeatmap",
    "Histogram2D",
    "LinePlot",
    "PiePlot",
    "GeoLine",
    "TreeMap",
    "build_trace_from_spec",
    "recompute_policy_table",
    "select_policy_table",
]

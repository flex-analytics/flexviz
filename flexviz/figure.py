"""User-facing Figure API for flexviz.

Example usage::

    import polars as pl
    from flexviz.figure import Figure

    df = pl.read_parquet("sensors.parquet")
    fig = Figure(df)
    fig.add_line(x="timestamp", y="temperature", name="Temp", color="#e74c3c")
    fig.add_line(x="timestamp", y="humidity",    name="Hum",  color="#3498db")
    fig.update_layout(title="Sensor data")
    fig.show(renderer="plotly")
"""

from __future__ import annotations

from collections.abc import Sequence
import math
import warnings
import threading
from typing import Any, Dict, List, Literal
from uuid import uuid4

import polars as pl

from .LF import LFQueryBuilder, polars_lf_from
from .spec import DashboardSpec, FigureSpec, VisualizationSpec
from .trace.bar import BarPlot
from .trace.base import FlexTrace, _composite_label
from .trace.box import BoxPlot
from .trace.corr_heatmap import CorrHeatmap
from .trace.geo_hist2d import GeoHistogram2D
from .trace.hist import Histogram
from .trace.hist2d import Histogram2D
from .trace.line import LinePlot
from .trace.pie import PiePlot
from .trace.geo_line import GeoLine
from .trace.treemap import TreeMap

_HEATMAP_TRACE_TYPES = frozenset({"histogram2d", "corr_heatmap"})
_HEATMAP_STYLE_INVARIANT_ERROR = "Generated heatmap specs must include explicit color_scale and color_range defaults."


def _derive_axis_labels(trace_specs: list) -> dict[str, str | None]:
    """Infer xlabel/ylabel from trace backend_data when all cartesian traces agree.

    Non-cartesian traces (``axes is None``) are ignored.  If cartesian traces
    disagree on the column used for an axis, that axis label is left as ``None``.
    """
    x_cols: set[str] = set()
    y_cols: set[str] = set()
    for ts in trace_specs:
        if ts.axes is None:
            continue
        bd = ts.backend_data or {}
        p = ts.params or {}
        tt = ts.trace_type
        if tt == "bar":
            # "labels" maps to the label axis; "values" (if present) maps to the
            # value axis.  Orientation controls which physical axis is which.
            raw_labels = bd["labels"]
            label_tuple = (
                tuple(raw_labels) if isinstance(raw_labels, list) else (raw_labels,)
            )
            label_str = _composite_label(label_tuple)
            values_col = bd.get("values")
            if p["orientation"] == "v":
                x_cols.add(label_str)
                if values_col is None:
                    y_cols.add("count")
                else:
                    y_cols.add(f"{p['agg']} of {values_col}")
            else:
                y_cols.add(label_str)
                if values_col is None:
                    x_cols.add("count")
                else:
                    x_cols.add(f"{p['agg']} of {values_col}")
        else:
            if "x" in bd:
                x_cols.add(bd["x"])
                if tt == "histogram":
                    y_cols.add(p["histnorm"])
            if "y" in bd:
                y_cols.add(bd["y"])
                if tt == "histogram":
                    x_cols.add(p["histnorm"])
    return {
        "xlabel": next(iter(x_cols)) if len(x_cols) == 1 else None,
        "ylabel": next(iter(y_cols)) if len(y_cols) == 1 else None,
    }


def _derive_auto_title(trace_specs: list) -> str | None:
    """Infer a figure title from aggregated traces when all agree.

    Contributing trace types: ``pie``, ``histogram2d``,``geo_histogram2d``.
    If all contributing traces produce the same candidate
    string, that string is returned; otherwise ``None`` is returned.
    Non-aggregated traces (line, box, corr_heatmap) are ignored.
    """
    candidates: set[str] = set()
    for ts in trace_specs:
        tt = ts.trace_type
        bd = ts.backend_data or {}
        p = ts.params or {}
        if tt == "pie":
            agg = p["agg"]
            values_col = bd.get("values")
            raw_labels = bd.get("labels", "")
            label_tuple = (
                tuple(raw_labels) if isinstance(raw_labels, list) else (raw_labels,)
            )
            labels_str = _composite_label(label_tuple)
            candidate = (
                f"count of {labels_str}"
                if values_col is None
                else f"{agg} of {values_col}"
            )
        elif tt in ("histogram2d", "geo_histogram2d"):
            histfunc = p.get("histfunc")  # None = implicit count
            histnorm = p.get("histnorm")  # None = no normalization
            z_col = bd.get("z")
            if z_col is None:
                candidate = histnorm if histnorm is not None else "count"
            else:
                candidate = f"{histfunc} of {z_col}"
                if histnorm is not None:
                    candidate += f" ({histnorm})"
        else:
            candidates.add(None)  # non-aggregated traces do not contribute
            continue
        candidates.add(candidate)
    return next(iter(candidates)) if len(candidates) == 1 else None


def _validate_bar_modes(trace_specs: list) -> None:
    bar_modes = {
        ts.display.get("bar_mode", ts.params.get("bar_mode", "group"))
        for ts in trace_specs
        if ts.trace_type == "bar"
    }
    if len(bar_modes) > 1:
        raise ValueError(
            "All bar traces in one figure must share the same bar_mode because "
            "renderers treat bar_mode as a figure-level setting."
        )


def _effective_heatmap_style(trace_spec: Any) -> tuple[str, tuple[float, float] | str]:
    if (
        "color_scale" not in trace_spec.display
        or "color_range" not in trace_spec.display
    ):
        raise ValueError(_HEATMAP_STYLE_INVARIANT_ERROR)

    scale = trace_spec.display["color_scale"]
    if not isinstance(scale, str) or not scale:
        raise TypeError("heatmap color_scale must be a non-empty string")

    color_range = trace_spec.display["color_range"]
    if color_range != "auto":
        if not (isinstance(color_range, (list, tuple)) and len(color_range) == 2):
            raise TypeError(
                "heatmap color_range must be 'auto' or a (min, max) numeric tuple"
            )
        lo = float(color_range[0])
        hi = float(color_range[1])
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise ValueError("heatmap color_range values must be finite numbers")
        if lo >= hi:
            raise ValueError("heatmap color_range must satisfy min < max")
        color_range = (lo, hi)
    return (scale, color_range)


def _validate_heatmap_styles(trace_specs: list) -> None:
    heatmap_styles = {
        _effective_heatmap_style(ts)
        for ts in trace_specs
        if ts.trace_type in _HEATMAP_TRACE_TYPES
    }
    if len(heatmap_styles) > 1:
        raise ValueError(
            "All heatmap traces in one figure must share the same effective "
            "color_scale and color_range because renderers treat the heatmap "
            "colorbar as a figure-level setting."
        )


class Figure:
    """Low-code, renderer-agnostic figure.

    Owns:
    - An optional ``LFQueryBuilder`` (the shared backend LazyFrame)
    - A list of ``FlexTrace`` instances
    - A layout config dict (renderer hints, e.g. ``title``)

    The Figure does not know about Plotly, Echarts, or any other renderer.
    Rendering is delegated to an adapter chosen via ``show(renderer=...)``.
    """

    def __init__(
        self,
        data: pl.DataFrame | pl.LazyFrame | None = None,
        cache: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        data:
            A Polars DataFrame or LazyFrame (or anything accepted by
            ``polars_lf_from``).  Pass ``None`` for in-memory figures
            where each trace carries its own data.
        cache:
            Opt this figure's source into init-load caching.  Asserts the
            data is static for the process lifetime (no invalidation yet;
            see ``register_source`` and issue #27).  Can be overridden per
            call in :meth:`show`.
        """
        self._backend_lf: LFQueryBuilder | None = None
        if data is not None:
            self._backend_lf = LFQueryBuilder(polars_lf_from(data))

        self._uid: str = str(uuid4())
        self._traces: List[FlexTrace] = []
        self._layout: Dict[str, Any] = {}
        self._cache_enabled: bool = cache

    # ------------------------------------------------------------------
    # Trace builders
    # ------------------------------------------------------------------

    def _add_trace(self, trace: FlexTrace) -> "Figure":
        """Assign a uid and register a trace."""
        trace.uid = str(uuid4())
        self._traces.append(trace)
        return self

    def add_line(
        self,
        x: str,
        y: str,
        name: str | None = None,
        color: str | None = None,
        color_map: dict | None = None,
        n_points: int = 1000,
        downsample: Literal["minmax", "lttb", "fpcs", "nth"] = "minmax",
        add_gaps: bool = True,
        axes: tuple[str, ...] = ("x", "y"),
        assume_sorted_x: bool = False,
        group_by: str | Sequence[str] | None = None,
    ) -> "Figure":
        """Add a line trace backed by the figure's shared LazyFrame.

        Parameters
        ----------
        x:
            Column name for x-axis data.  For an ungrouped ``"minmax"``,
            ``"lttb"`` or ``"fpcs"`` line it must be sorted ascending: those
            buckets are equal in x width.
        y:
            Column name for y-axis data.
        name:
            Series name shown in the legend.  Defaults to the column name.
        color:
            CSS colour string (e.g. ``"#e74c3c"``).
        n_points:
            Target number of visible points per viewport, between 2 and 25000.
            ``"lttb"`` returns exactly this many when the prefetch holds more,
            and fewer when x gaps leave buckets empty.  For FPCS it is not a
            hard cap; output may be up to roughly twice this value.
        add_gaps:
            When True (default), insert None values at detected gaps in x so
            renderers show a broken line across empty periods.
        downsample:
            Downsampling algorithm: ``"minmax"`` (default), ``"lttb"``
            (MinMaxLTTB, ungrouped only), ``"fpcs"``, or ``"nth"``.
        axes:
            Axis anchor tuple.  Defaults to ``("x", "y")`` for a single
            cartesian axis.  Use ``("x2", "y2")`` for a second axis.
        assume_sorted_x:
            For an ungrouped `"minmax"`, `"lttb"` or `"fpcs"` line the engine
            verifies `x` on the first request. On a resident frame it makes one
            pass over the column: no nulls or NaN, sorted ascending. On a scan
            source it checks the dtype only. It raises `ValueError` when the
            column fails. A cached figure pays that once per source and column.
            An uncached one pays it on every unzoomed request. Set True to skip
            the check by marking the column sorted via `set_sorted`. Only use it
            if you guarantee `x` meets the contract. A column that does not then
            gives wrong results.
        """
        trace = LinePlot(
            x=x,
            y=y,
            name=name,
            color=color,
            color_map=color_map,
            n_points=n_points,
            downsample=downsample,
            axes=axes,
            add_gaps=add_gaps,
            group_by=group_by,
        )
        if assume_sorted_x and self._backend_lf is not None:
            # Opt-in: caller guarantees sortedness. This avoids an expensive collect.
            self._backend_lf.assume_sorted(x)
        return self._add_trace(trace)

    def add_histogram(
        self,
        x: str | None = None,
        y: str | None = None,
        bins: int = 20,
        histnorm: str = "count",
        name: str | None = None,
        color: str | None = None,
        color_map: dict | None = None,
        axes: tuple[str, ...] = ("x", "y"),
        group_by: str | Sequence[str] | None = None,
    ) -> "Figure":
        """Add a histogram trace backed by the figure's shared LazyFrame.

        Parameters
        ----------
        x:
            Column name for the data axis (produces vertical bars).
        y:
            Column name for the data axis (produces horizontal bars).
        bins:
            Number of bins.
        histnorm:
            Normalization: ``"count"``, ``"percent"``, ``"probability"``,
            ``"density"``, or ``"probability density"``.
        name:
            Legend label.  Defaults to the column name.
        color:
            CSS colour string.
        axes:
            Axis anchor tuple, e.g. ``("x", "y")``.
        """
        return self._add_trace(
            Histogram(
                x=x,
                y=y,
                bins=bins,
                histnorm=histnorm,
                name=name,
                color=color,
                color_map=color_map,
                axes=axes,
                group_by=group_by,
            )
        )

    def add_boxplot(
        self,
        x: str | None = None,
        y: str | None = None,
        name: str | None = None,
        color: str | None = None,
        color_map: dict | None = None,
        axes: tuple[str, ...] = ("x", "y"),
        group_by: str | Sequence[str] | None = None,
    ) -> "Figure":
        """Add a box plot trace backed by the figure's shared LazyFrame."""
        return self._add_trace(
            BoxPlot(
                x=x,
                y=y,
                name=name,
                color=color,
                color_map=color_map,
                axes=axes,
                group_by=group_by,
            )
        )

    def add_bar(
        self,
        labels: str | Sequence[str],
        values: str | None = None,
        agg: Literal["sum", "mean", "median", "min", "max", "n_unique"] = "sum",
        name: str | None = None,
        color: str | None = None,
        orientation: Literal["v", "h"] = "v",
        bar_mode: Literal["group", "stack"] = "group",
        group_by: str | Sequence[str] | None = None,
        color_map: dict | None = None,
        axes: tuple[str, ...] = ("x", "y"),
    ) -> "Figure":
        """Add a bar trace backed by the figure's shared LazyFrame.

        Parameters
        ----------
        labels:
            Column name for the label / category axis (primary grouping key).
        values:
            Column name for the values to aggregate.  When ``None`` (default)
            the trace counts rows per label.
        agg:
            Aggregation function applied to ``values``: ``"sum"`` (default),
            ``"mean"``, ``"median"``, ``"min"``, ``"max"``, or ``"n_unique"``.
            Ignored when ``values`` is ``None``.
        name:
            Legend label.
        color:
            CSS colour string (used when there is no ``group_by``).
        orientation:
            ``"v"`` (default) for vertical bars; ``"h"`` for horizontal bars.
        bar_mode:
            ``"group"`` (default, side-by-side) or ``"stack"`` (stacked bars).
            Only meaningful when ``group_by`` is set.
        group_by:
            Column name for a second grouping dimension (hue / color split).
        color_map:
            Optional ``{group_value: css_color}`` override.
        axes:
            Axis anchor tuple, e.g. ``("x", "y")``.
        """
        return self._add_trace(
            BarPlot(
                labels=labels,
                values=values,
                agg=agg,
                name=name,
                color=color,
                orientation=orientation,
                bar_mode=bar_mode,
                group_by=group_by,
                color_map=color_map,
                axes=axes,
            )
        )

    def add_pie(
        self,
        labels: str | Sequence[str],
        values: str | None = None,
        agg: Literal["sum", "mean", "median", "min", "max", "n_unique"] = "sum",
        name: str | None = None,
        hole: float = 0.0,
        color_map: dict | None = None,
    ) -> "Figure":
        """Add a pie (or donut) trace backed by the figure's shared LazyFrame.

        Parameters
        ----------
        labels:
            Column name for the slice labels / categories.
        values:
            Column name for the values to aggregate per slice.  When ``None``
            (default) the trace counts rows per label.
        agg:
            Aggregation function applied to ``values``: ``"sum"`` (default),
            ``"mean"``, ``"median"``, ``"min"``, ``"max"``, or ``"n_unique"``.
            Ignored when ``values`` is ``None``.
        name:
            Legend label.
        hole:
            Fraction of the radius to cut out (0 = pie, 0.4–0.6 = donut).
        color_map:
            Optional ``{label: css_color}`` override.
        """
        return self._add_trace(
            PiePlot(
                labels=labels,
                values=values,
                agg=agg,
                name=name,
                hole=hole,
                color_map=color_map,
            )
        )

    def add_treemap(
        self,
        path: list[str],
        values: str | None = None,
        agg: Literal["sum", "mean", "median", "min", "max", "n_unique"] = "sum",
        name: str | None = None,
        color_map: dict | None = None,
    ) -> "Figure":
        """Add a treemap trace backed by a Polars LazyFrame.

        Parameters
        ----------
        path:
            Ordered list of column names from root to leaf,
            e.g. ``["continent", "country"]``.
        values:
            Column name to aggregate per leaf node.  When ``None`` (default)
            the trace counts rows.
        agg:
            Aggregation function applied to ``values``: ``"sum"`` (default),
            ``"mean"``, ``"median"``, ``"min"``, ``"max"``, or ``"n_unique"``.
            Ignored when ``values`` is ``None``.
        name:
            Series name shown in the chart legend.
        color_map:
            Optional ``{label: css_color}`` dict for categorical colouring.
        """
        trace = TreeMap(
            path=path, values=values, agg=agg, name=name, color_map=color_map
        )
        return self._add_trace(trace)

    def add_histogram2d(
        self,
        x: str,
        y: str,
        x_bins: int = 20,
        y_bins: int = 20,
        z: str | None = None,
        histfunc: Literal["sum", "mean", "min", "max"] | None = None,
        histnorm: str | None = None,
        name: str | None = None,
        color_scale: str | None = None,
        color_range: tuple[float, float] | Literal["auto"] | None = None,
        axes: tuple[str, ...] = ("x", "y"),
    ) -> "Figure":
        """Add a 2D histogram / heatmap trace.

        Parameters
        ----------
        x:
            Column name for the horizontal axis.
        y:
            Column name for the vertical axis.
        x_bins:
            Number of bins along x (default 20).
        y_bins:
            Number of bins along y (default 20).
        z:
            Column name for the value to aggregate per bin.  When ``None``
            (default) the trace counts rows per bin.
        histfunc:
            Aggregation function applied to ``z``.  Required when ``z`` is
            given; one of ``"sum"``, ``"mean"``, ``"min"``, ``"max"``.
            Forbidden when ``z`` is ``None``.
        histnorm:
            Normalization applied after aggregation.  ``None`` (default) means
            no normalization.
        name:
            Legend label.
        color_scale:
            Heatmap color scale name understood by the active renderer.
        color_range:
            Fixed heatmap color range, or ``"auto"`` for dynamic scaling.
        axes:
            Axis anchor tuple, e.g. ``("x", "y")``.
        """
        return self._add_trace(
            Histogram2D(
                x=x,
                y=y,
                x_bins=x_bins,
                y_bins=y_bins,
                z=z,
                histfunc=histfunc,
                histnorm=histnorm,
                name=name,
                color_scale=color_scale,
                color_range=color_range,
                axes=axes,
            )
        )

    def add_geo_histogram2d(
        self,
        lat: str,
        lon: str,
        lat_bins: int = 64,
        lon_bins: int = 64,
        z: str | None = None,
        histfunc: Literal["sum", "mean", "min", "max"] | None = None,
        histnorm: str | None = None,
        name: str | None = None,
        color_scale: str | None = None,
        color_range: tuple[float, float] | Literal["auto"] | None = None,
        bin_boundaries: Literal["data", "viewport"] = "data",
    ) -> "Figure":
        """Add a geospatial 2D histogram (choropleth) trace.

        Parameters
        ----------
        lat:
            Column name for latitude.
        lon:
            Column name for longitude.
        lat_bins:
            Number of bins along latitude (default 64).
        lon_bins:
            Number of bins along longitude (default 64).
        z:
            Column name for the value to aggregate per bin.  When ``None``
            (default) the trace counts rows per bin.
        histfunc:
            Aggregation function applied to ``z``.  Required when ``z`` is
            given; one of ``"sum"``, ``"mean"``, ``"min"``, ``"max"``
            (computed by the ``flexviz_polars`` Rust kernel).  Forbidden when
            ``z`` is ``None``.
        histnorm:
            Normalization applied after aggregation.  ``None`` (default) means
            no normalization.
        name:
            Legend label.
        color_scale:
            Color scale name understood by the active renderer.
        color_range:
            Fixed color range, or ``"auto"`` for dynamic scaling.
        bin_boundaries:
            ``"data"`` uses visible data extent for bin edges; ``"viewport"``
            anchors bins to the map bounds (stable grid when panning/zooming).
        """
        return self._add_trace(
            GeoHistogram2D(
                lat=lat,
                lon=lon,
                lat_bins=lat_bins,
                lon_bins=lon_bins,
                z=z,
                histfunc=histfunc,
                histnorm=histnorm,
                name=name,
                color_scale=color_scale,
                color_range=color_range,
                bin_boundaries=bin_boundaries,
            )
        )

    def add_geo_line(
        self,
        lat: str,
        lon: str,
        n_points: int = 1000,
        name: str | None = None,
        color: str | None = None,
        add_gaps: bool = True,
    ) -> "Figure":
        """Add a scalable geo line trace.

        Parameters
        ----------
        lat:
            Column name for latitude.
        lon:
            Column name for longitude.
        n_points:
            Maximum number of points returned per map viewport update.
        name:
            Legend label.
        color:
            Line colour hint (CSS string).
        add_gaps:
            When ``True`` (default), inserts breaks in the line where
            consecutive points are unusually far apart.
        """
        return self._add_trace(
            GeoLine(
                lat=lat,
                lon=lon,
                n_points=n_points,
                name=name,
                color=color,
                add_gaps=add_gaps,
            )
        )

    def add_corr_heatmap(
        self,
        columns: list[str] | None = None,
        method: Literal["pearson", "spearman"] = "pearson",
        triangular: bool = False,
        absolute: bool = False,
        name: str | None = None,
        color_scale: str | None = None,
        color_range: tuple[float, float] | Literal["auto"] | None = None,
    ) -> "Figure":
        """Add a correlation heatmap trace.

        Parameters
        ----------
        columns:
            Subset of numeric columns to correlate.  ``None`` = all numeric.
        method:
            ``"pearson"`` (default) or ``"spearman"``.
        triangular:
            If ``True``, keep only one triangle (incl. diagonal).
        absolute:
            If ``True``, report ``abs(corr)``.
        name:
            Legend label.
        color_scale:
            Heatmap color scale name understood by the active renderer.
        color_range:
            Fixed heatmap color range, or ``"auto"`` for dynamic scaling.
        """
        return self._add_trace(
            CorrHeatmap(
                columns=columns,
                method=method,
                triangular=triangular,
                absolute=absolute,
                name=name,
                color_scale=color_scale,
                color_range=color_range,
            )
        )

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def title(self, text: str) -> "Figure":
        """Set a renderer-agnostic figure title."""
        self._layout["title"] = text
        return self

    def xlabel(self, text: str) -> "Figure":
        """Set a renderer-agnostic x-axis label."""
        self._layout["xlabel"] = text
        return self

    def ylabel(self, text: str) -> "Figure":
        """Set a renderer-agnostic y-axis label."""
        self._layout["ylabel"] = text
        return self

    def legend(self, show: bool = True) -> "Figure":
        """Toggle legend visibility."""
        self._layout["legend"] = show
        return self

    def update_layout(self, **kwargs: Any) -> "Figure":
        """Merge layout hints (title, width, height, …)."""
        self._layout.update(kwargs)
        return self

    # ------------------------------------------------------------------
    # Spec serialisation
    # ------------------------------------------------------------------

    def to_spec(self, source: str | None = None) -> VisualizationSpec:
        """Serialise the figure to a ``VisualizationSpec``.

        Parameters
        ----------
        source:
            Data source name as registered with ``register_source()`` on the
            server.  Clients include this in every ``/update`` request so the
            server knows which ``LFQueryBuilder`` to use.
        """
        domain_source = source or self._uid
        trace_specs = [
            t.to_trace_spec(domain_source=domain_source) for t in self._traces
        ]
        _validate_bar_modes(trace_specs)
        _validate_heatmap_styles(trace_specs)
        layout = dict(self._layout)
        derived = _derive_axis_labels(trace_specs)
        if "xlabel" not in layout and derived.get("xlabel"):
            layout["xlabel"] = derived["xlabel"]
        if "ylabel" not in layout and derived.get("ylabel"):
            layout["ylabel"] = derived["ylabel"]
        derived_title = _derive_auto_title(trace_specs)
        if "title" not in layout and derived_title:
            layout["title"] = derived_title
        figure_spec = FigureSpec(
            uid=self._uid,
            source=source,
            layout=layout,
            traces=trace_specs,
        )
        return VisualizationSpec(figure=figure_spec)

    def save_spec(self, path: str, source: str | None = None) -> None:
        """Serialise the figure spec to a JSON file.

        Parameters
        ----------
        path:
            Filesystem path to write (creates or overwrites the file).
        source:
            Data source name forwarded to :meth:`to_spec`.
        """
        import pathlib

        spec = self.to_spec(source=source)
        pathlib.Path(path).write_text(spec.model_dump_json(indent=2), encoding="utf-8")

    @staticmethod
    def load_spec(path: str) -> VisualizationSpec:
        """Load a ``VisualizationSpec`` from a JSON file written by :meth:`save_spec`.

        Parameters
        ----------
        path:
            Filesystem path to read.

        Returns
        -------
        VisualizationSpec
        """
        import json
        import pathlib

        data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        return VisualizationSpec.model_validate(data)

    @classmethod
    def from_spec(
        cls,
        spec: VisualizationSpec,
        backend_lf: "LFQueryBuilder | None" = None,
    ) -> "Figure":
        """Reconstruct a ``Figure`` from a ``VisualizationSpec``.

        The Figure's uid and all trace uids are restored from the spec so
        that subsequent calls to :meth:`show` use the same uids that were
        saved — guaranteeing that the server routes deltas to the correct
        trace divs.

        Parameters
        ----------
        spec:
            A ``VisualizationSpec`` loaded from a file or produced by
            :func:`flexviz.spec.decode_spec`.
        backend_lf:
            Provide the data source here (a ``LFQueryBuilder`` or raw data
            accepted by ``polars_lf_from``), or register it on the server
            separately via ``register_source()`` before calling :meth:`show`.

        Returns
        -------
        Figure
        """
        from .trace import build_trace_from_spec  # local import avoids cycle

        fig = cls.__new__(cls)
        fig._uid = spec.figure.uid
        if backend_lf is not None and not isinstance(backend_lf, LFQueryBuilder):
            fig._backend_lf = LFQueryBuilder(polars_lf_from(backend_lf))
        else:
            fig._backend_lf = backend_lf
        fig._layout = dict(spec.figure.layout)
        fig._traces = []
        for ts in spec.figure.traces:
            trace = build_trace_from_spec(ts)
            trace.uid = ts.uid  # restore original uid so delta routing matches
            fig._traces.append(trace)
        return fig

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def show(
        self,
        renderer: str = "plotly",
        source_name: str | None = None,
        host: str = "127.0.0.1",
        port: int | str = 8000,
        cache: bool | None = None,
        live_brush: Literal["auto", "off"] | None = None,
        **kwargs: Any,
    ) -> None:
        """Start the FastAPI backend and render through the chosen adapter.

        Parameters
        ----------
        renderer:
            ``"plotly"`` (default) or ``"echarts"``.
        source_name:
            Name under which the figure's backend LazyFrame is registered
            with the server's data-source registry.  Defaults to the
            figure's uid so multiple ``show()`` calls never collide.
        host:
            Server bind address.
        port:
            Server port.  Pass ``"auto"`` to pick a random free port.
        cache:
            Opt this figure's source into init-load caching (asserts the data
            is static; see ``register_source``).  Defaults to the value passed
            to the ``Figure`` constructor.
        live_brush:
            Cube live-brush mode, stored on the client state (the server never
            reads it).  ``"auto"`` enables drag-time cube slicing where
            available; ``"off"`` restores mouseup-only selection.  ``None``
            (default) resolves to ``"auto"``.  Live brushing requires
            ``cache=True`` (cubes are only built for cacheable sources), so when
            caching is off this is forced to ``"off"`` — silently for the
            default, with a warning if ``"auto"`` was passed explicitly.
        **kwargs:
            Forwarded to the adapter's ``show_dashboard()`` method.
        """
        if port == "auto":
            port = _pick_free_port(host)

        if source_name is None:
            # TODO: see TODO in dashboard.show()
            source_name = self._uid

        effective_cache = self._cache_enabled if cache is None else cache
        _register_source_if_needed(source_name, self._backend_lf, cache=effective_cache)
        _start_server_thread(host, port)

        src = source_name if self._backend_lf is not None else None
        spec = self.to_spec(source=src)
        dash_spec = DashboardSpec(figures=[spec.figure], state=spec.state)
        dash_spec.client_state.live_brush = _effective_live_brush(
            live_brush, effective_cache
        )
        _render_dashboard(renderer, dash_spec, f"http://{host}:{port}", **kwargs)


def _registered_sources() -> list[str]:
    """Return names of already-registered sources (avoids double-registration)."""
    from .server import _sources

    return list(_sources)


def _effective_live_brush(live_brush: str | None, effective_cache: bool) -> str:
    """Couple ``live_brush`` to caching: cubes are only built for ``cache=True``
    sources (the "data is static" contract), so live-brush cannot function
    without it.  When ``effective_cache`` is False we force ``"off"`` — otherwise
    the client binds ``plotly_selecting`` and fires a wasted ``cube_request`` per
    drag that the server answers with an empty bundle, then degrades.

    ``live_brush=None`` is the unset default (resolves to ``"auto"``); a caller
    that *explicitly* asked for live brushing while uncacheable gets a one-time
    warning explaining why it won't engage.  The default path is silent.
    """
    explicit = live_brush is not None
    requested = live_brush or "auto"
    if effective_cache:
        return requested
    if explicit and requested != "off":
        warnings.warn(
            f"live_brush={requested!r} needs cache=True to serve cubes; "
            "forcing live_brush='off'."
        )
    return "off"


def _register_source_if_needed(
    source_name: str, backend_lf: "LFQueryBuilder | None", cache: bool = False
) -> None:
    """Register *backend_lf* under *source_name*.

    Idempotent: silently skips if the same object is already registered under
    that name.  Warns and overwrites only when a *different* object is being
    registered under an existing name (explicit name collision).

    ``cache`` opts the source into init-load caching (see
    ``register_source``).  It is applied even on the idempotent path so a
    later ``show(cache=True)`` on an already-registered source still takes
    effect.
    """
    if backend_lf is None:
        return
    from .cache import set_source_cacheable
    from .server import _sources, register_source

    existing = _sources.get(source_name)
    if existing is not None:
        if existing is not backend_lf:
            warnings.warn(
                f"source_name={source_name!r} is already registered, "
                "will override existing source"
            )
            register_source(source_name, backend_lf, cache=cache)
        else:
            # same object — idempotent; mirror the caller's current cache choice
            # so a later show(cache=False) can opt out after an earlier opt-in.
            set_source_cacheable(source_name, bool(cache))
    else:
        register_source(source_name, backend_lf, cache=cache)


_started_servers: set[tuple[str, int]] = set()


def _start_server_thread(host: str, port: int) -> None:
    """Start the flexviz FastAPI server in a daemon thread (idempotent per port)."""
    if (host, port) in _started_servers:
        return

    from .server import app

    import uvicorn

    threading.Thread(
        target=uvicorn.run,
        args=(app,),
        kwargs={"host": host, "port": port, "log_level": "warning"},
        daemon=True,
    ).start()
    _started_servers.add((host, port))


def _render_dashboard(
    renderer: str, spec: DashboardSpec, server_url: str, **kwargs: Any
) -> None:
    """Pick the adapter for *renderer* and call ``show_dashboard``."""
    from .adapters import build_adapter, validate_dashboard_renderer

    renderer_name = validate_dashboard_renderer(renderer, spec)
    adapter = build_adapter(renderer_name)
    adapter.show_dashboard(spec, server_url=server_url, **kwargs)


def _pick_free_port(host: str = "127.0.0.1") -> int:
    """Ask the OS for a free TCP port on *host*."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]

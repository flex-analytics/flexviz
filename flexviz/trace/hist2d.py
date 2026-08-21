"""Histogram2D — renderer-agnostic 2D histogram / heatmap trace.

Bins two numeric columns into a 2D grid and counts occurrences (or applies
``"sum"``, ``"mean"``, ``"min"``, or ``"max"`` via *histfunc*).  Returns
``{x: [...centers], y: [...centers], z: [[counts]]}`` — the standard format
for Plotly ``heatmap`` and ECharts ``heatmap`` series.

When *z* is omitted the trace counts rows per bin (implicit count).  When *z*
is given, *histfunc* is required and must be one of ``"sum"``, ``"mean"``,
``"min"``, ``"max"``.

Supported *histnorm* values: ``None`` (no normalization, default), ``"percent"``,
``"probability"``, ``"density"``, ``"probability density"``.

Viewport filtering
------------------
``recompute_axes = (x, y)`` — the histogram is recomputed on each viewport
change (same behaviour as 1D Histogram).  Bin edges always span the
viewport range (or the data range on init).
"""

from __future__ import annotations

from typing import Any, Dict

import polars as pl

from ..cube import CubeTargetSpec, FreeAxisSpec, MeasureSpec, TargetDimSpec
from ..LF import AggregationSpec
from ..spec import TraceHoverSpec, TraceSpec
from .base import (
    FlexTrace,
    TraceResult,
    _CUBE_RESERVED_COLS,
    _dtype_for_col,
    _phys_epoch_ms_factor,
    _physical_bound_expr,
    _physical_to_temporal_series,
    _temporal_dtype_for_col,
    _typed_range_bounds,
)
from ._hist_helpers import (
    HeatmapColorRange,
    _HISTNORM_OPTIONS,
    apply_histnorm,
    normalize_heatmap_color_scale,
    normalize_heatmap_color_range,
)

import flexviz_polars  # noqa: F401 — registers pl.Expr.flexviz namespace

_DEFAULT_COLOR_SCALE = "viridis"
_DEFAULT_COLOR_RANGE: HeatmapColorRange = "auto"
_HIST2D_HISTFUNC_OPTIONS = ("sum", "mean", "min", "max")


class Histogram2D(FlexTrace):
    """Scalable 2D histogram / heatmap trace.

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
        given; must be one of ``"sum"``, ``"mean"``, ``"min"``, ``"max"``.
        Forbidden when ``z`` is ``None``.
    histnorm:
        Normalization applied after aggregation.  ``None`` (default) means
        no normalization; other options: ``"percent"``, ``"probability"``,
        ``"density"``, ``"probability density"``.
    name:
        Legend / series name.
    """

    trace_type: str = "histogram2d"
    select_policy_doc: str = "both axes (x, y) — 2-D box select"
    recompute_policy_doc: str = "both axes (x, y) — re-bins to viewport"
    overlay_style: str = "filtered_only"

    def __init__(
        self,
        x: str,
        y: str,
        x_bins: int = 20,
        y_bins: int = 20,
        z: str | None = None,
        histfunc: str | None = None,
        histnorm: str | None = None,
        name: str | None = None,
        color_scale: str | None = None,
        color_range: tuple[float, float] | str | None = None,
        axes: tuple[str, ...] = ("x", "y"),
    ) -> None:
        if z is None and histfunc is not None:
            raise ValueError("histfunc is only meaningful when z is given.")
        if z is not None and histfunc is None:
            raise ValueError("histfunc is required when z is given.")
        if z is not None and histfunc not in _HIST2D_HISTFUNC_OPTIONS:
            raise ValueError(f"histfunc must be one of {_HIST2D_HISTFUNC_OPTIONS}.")
        if histnorm not in _HISTNORM_OPTIONS:
            raise ValueError(f"histnorm must be one of {_HISTNORM_OPTIONS}.")

        backend_data: Dict[str, str] = {"x": x, "y": y}
        if z is not None:
            backend_data["z"] = z

        super().__init__(
            backend_data=backend_data,
            display={
                "name": name or f"{y} vs {x}",
                "color_scale": normalize_heatmap_color_scale(
                    color_scale, _DEFAULT_COLOR_SCALE, trace_name="Histogram2D"
                ),
                "color_range": normalize_heatmap_color_range(
                    color_range, _DEFAULT_COLOR_RANGE, trace_name="Histogram2D"
                ),
            },
            params={
                "x_bins": x_bins,
                "y_bins": y_bins,
                "histfunc": histfunc,
                "histnorm": histnorm,
            },
            axes=axes,
        )
        # Resolved per-request in get_aggregation_spec when x/y are temporal
        # (binning runs on the physical representation); read back in _to_update
        # to restore datetime bin centers. None ⇒ that axis is non-temporal.
        self._x_temporal_dtype: pl.DataType | None = None
        self._y_temporal_dtype: pl.DataType | None = None

    def _default_recompute_axes(self) -> tuple[str, ...]:
        return tuple(self._axes)  # both axes bin the 2-D histogram

    def _make_selection_spec(self):
        return self._range_selection_spec()

    def _make_hover_spec(self) -> "TraceHoverSpec":
        return TraceHoverSpec(
            source_modes=["cell"],
            target_modes=["cell", "axis"],
        )

    @property
    def x_col(self) -> str:
        return self._backend_data["x"]

    @property
    def y_col(self) -> str:
        return self._backend_data["y"]

    @property
    def z_col(self) -> str | None:
        return self._backend_data.get("z")

    @property
    def x_bins(self) -> int:
        return self._params["x_bins"]

    @property
    def y_bins(self) -> int:
        return self._params["y_bins"]

    @property
    def histfunc(self) -> str | None:
        return self._params["histfunc"]

    @property
    def histnorm(self) -> str | None:
        return self._params["histnorm"]

    @property
    def color_scale(self) -> str:
        return self._display["color_scale"]

    @property
    def color_range(self) -> HeatmapColorRange:
        return self._display["color_range"]

    # ------------------------------------------------------------------
    # Cube descriptors (cross-filter pre-aggregation)
    # ------------------------------------------------------------------

    def get_cube_source_spec(
        self,
        axis_range: tuple[float, float] | None,
        schema: pl.Schema | None = None,
    ) -> FreeAxisSpec | None:
        """A box-select on a 2-D histogram defines a **box2d** free axis on its
        ``(x_col, y_col)`` pair (contract H).

        ``column`` is the x column (the primary ``active_source.column`` join
        key); ``columns = (x_col, y_col)``; ``p = P₂D = 128`` per axis. The
        per-axis ``domains`` are resolved by the **engine** (box2d domain
        resolution is two-axis: this method's ``axis_range`` is only the
        x-anchor viewport, so it cannot fill both), exactly as the 1-D
        temporal block has the engine set ``unit``/``domains``. Both columns
        must be numeric or temporal; an unsuitable dtype (when a schema is
        available) gates to ``None`` — the box2d branch in ``_locate_free_axis``
        validates the per-axis temporal units and resolves the two viewports.
        """
        x_col, y_col = self.x_col, self.y_col
        if not isinstance(x_col, str) or not isinstance(y_col, str):
            return None
        # Gate non-suitable dtypes when a schema is available. Temporal axes
        # are allowed; the engine's box2d block validates each axis's unit
        # (Datetime("ns")/Time gate to no cube) and sets per-axis units.
        for col in (x_col, y_col):
            dtype = _dtype_for_col(schema, col)
            if dtype is not None and not (dtype.is_numeric() or dtype.is_temporal()):
                return None
        return FreeAxisSpec(
            column=x_col,
            kind="box2d",
            p=128,
            columns=(x_col, y_col),
            domains=None,
        )

    def get_cube_target_spec(
        self,
        axis_range: tuple[float, float] | None,
        schema: pl.Schema | None = None,
    ) -> "CubeTargetSpec | None":
        """A 2-D histogram is a ``count``/reduce cube target (contract K).

        Its grouping dims are its own ``(x_col, y_col)`` bin axes (order pinned:
        x first, y second) with ``bin_variant="hist2d"`` so the cube bins
        bit-equally to the ``fixed_hist2d`` kernel (the ``+1e-10`` span eps).
        The measure is a count when ``z_col`` is ``None``, else the ``histfunc``
        reduction over ``z_col``. ``histnorm`` is NOT part of the cube — it is a
        client-side display normalization applied per-slice (two hist2ds
        differing only in ``histnorm`` share one cube).

        Gates (any failure ⇒ ``None`` ⇒ legacy server recompute):

        * **Full-data only**: ``axis_range is not None`` (the cube anchor axis,
          x, is zoomed) ⇒ ``None``. The cube hist2d target is served only when
          BOTH axes span the full data range so the resolved full-data domains
          give bit-equal binning. A zoom on either axis falls back to a POST.
        * A reduce target requires a numeric ``z_col`` (a ``schema`` is
          therefore required for the reduce case); a non-numeric ``z_col`` ⇒
          ``None``.
        * ``x_col``/``y_col``/``z_col`` must not collide with a reserved cube
          partial-column name (``count``/``sum``/``min``/``max``/``free_bin``).
        * The engine resolves each axis's temporal ``unit`` and applies the
          ``Datetime("ns")``/``Time`` gate uniformly in
          ``_resolved_target_dims`` (so it is not duplicated here).

        Like every heatmap target this is ``filtered_only`` (no background
        layer) and skipPost-eligible: when it is the only target the brush is
        fully cube-served and the commit need not POST.
        """
        if axis_range is not None:
            return None
        x_col, y_col, z_col = self.x_col, self.y_col, self.z_col
        if not isinstance(x_col, str) or not isinstance(y_col, str):
            return None
        if x_col in _CUBE_RESERVED_COLS or y_col in _CUBE_RESERVED_COLS:
            return None
        if z_col is not None:
            if z_col in _CUBE_RESERVED_COLS:
                return None
            z_dtype = _dtype_for_col(schema, z_col)
            if z_dtype is None or not z_dtype.is_numeric():
                return None
            assert self.histfunc is not None
            measure = MeasureSpec(agg=self.histfunc, value_col=z_col)
        else:
            measure = MeasureSpec(agg="count")
        return CubeTargetSpec(
            target_dims=(
                TargetDimSpec(
                    column=x_col,
                    kind="binned",
                    bins=self.x_bins,
                    domain=None,
                    bin_variant="hist2d",
                ),
                TargetDimSpec(
                    column=y_col,
                    kind="binned",
                    bins=self.y_bins,
                    domain=None,
                    bin_variant="hist2d",
                ),
            ),
            measure=measure,
        )

    # ------------------------------------------------------------------
    # FlexTrace interface
    # ------------------------------------------------------------------

    def get_aggregation_spec(
        self,
        update_range: Dict[str, Any],
        schema: pl.Schema | None = None,
    ) -> AggregationSpec:
        x_range = update_range.get("x")
        y_range = update_range.get("y")
        # Temporal axes bin on their physical representation; _to_update restores
        # datetime centers afterward.
        self._x_temporal_dtype = _temporal_dtype_for_col(self.x_col, schema)
        self._y_temporal_dtype = _temporal_dtype_for_col(self.y_col, schema)
        if self.z_col is None:
            expr, global_stats_cols = _hist2d_count_expr(
                self.x_col,
                self.y_col,
                self.x_bins,
                self.y_bins,
                x_range,
                y_range,
                self.uid,
                schema,
            )
        else:
            assert self.histfunc is not None
            expr, global_stats_cols = _hist2d_reduce_expr(
                self.x_col,
                self.y_col,
                self.z_col,
                self.x_bins,
                self.y_bins,
                x_range,
                y_range,
                self.uid,
                self.histfunc,
                schema,
            )
        return AggregationSpec(
            expr=expr, uid=self.uid, global_stats_cols=global_stats_cols
        )

    def _to_update(self, df: pl.DataFrame) -> TraceResult:
        raw = df[self.uid][0]
        nb_x = self.x_bins
        nb_y = self.y_bins

        # Rust kernel output: Struct{z_flat, x_lo, x_hi, y_lo, y_hi}
        z_flat_raw: list = raw["z_flat"]
        x_lo: float = raw["x_lo"]
        x_hi: float = raw["x_hi"]
        y_lo: float = raw["y_lo"]
        y_hi: float = raw["y_hi"]

        if self.z_col is None:
            # Count kernel returns UInt32; 0 indicates an empty bin →
            # emit None to match the documented gap-rendering contract.
            z_flat: list = [None if v == 0 else float(v) for v in z_flat_raw]
        else:
            # Reducer kernel returns nullable Float64 values directly.
            z_flat = [None if v is None else float(v) for v in z_flat_raw]

        x_centers = _centers(x_lo, x_hi, nb_x)
        y_centers = _centers(y_lo, y_hi, nb_y)

        if self.histnorm is not None:
            x_step = (x_hi - x_lo) / nb_x
            y_step = (y_hi - y_lo) / nb_y
            z_series = pl.Series("value", z_flat, dtype=pl.Float64)
            z_df = apply_histnorm(
                pl.DataFrame({"value": z_series}),
                "value",
                self.histnorm,
                x_step * y_step,
            )
            z_flat = z_df["value"].to_list()

        z = [z_flat[j * nb_x : (j + 1) * nb_x] for j in range(nb_y)]

        # Temporal axes are binned in physical space: restore datetime centers
        # (so the renderer draws a date axis) and express hover-band edges in
        # epoch-ms (Plotly's numeric date coordinate, what hover matching uses).
        x_out = self._axis_centers(x_centers, self._x_temporal_dtype, self.x_col)
        y_out = self._axis_centers(y_centers, self._y_temporal_dtype, self.y_col)
        x_edge = self._edge_scale(self._x_temporal_dtype)
        y_edge = self._edge_scale(self._y_temporal_dtype)

        # -- hover_bounds: 2D array of cell bounds matching z shape
        x_step = (x_hi - x_lo) / nb_x
        y_step = (y_hi - y_lo) / nb_y
        hover_bounds = []
        for row in range(nb_y):
            hover_bounds.append([])
            for col in range(nb_x):
                x0 = x_lo + col * x_step
                x1 = x0 + x_step
                y0 = y_lo + row * y_step
                y1 = y0 + y_step
                hover_bounds[-1].append(
                    {
                        "x0": x_edge(x0),
                        "x1": x_edge(x1),
                        "y0": y_edge(y0),
                        "y1": y_edge(y1),
                    }
                )

        return TraceResult(
            updates={
                "x": x_out,
                "y": y_out,
                "z": z,
                "hover_bounds": hover_bounds,
            }
        )

    @staticmethod
    def _axis_centers(
        centers: list[float], dtype: pl.DataType | None, col: str
    ) -> list:
        """Physical bin centers → datetime objects for a temporal axis, else the
        raw float centers unchanged."""
        if dtype is None:
            return centers
        return _physical_to_temporal_series(centers, dtype, col).to_list()

    @staticmethod
    def _edge_scale(dtype: pl.DataType | None):
        """A function mapping a physical edge to the renderer's axis coordinate:
        epoch-ms for a temporal axis, the plain float otherwise."""
        if dtype is None:
            return lambda v: float(v)
        factor = _phys_epoch_ms_factor(dtype)
        return lambda v: float(v) * factor

    # ------------------------------------------------------------------
    # Spec reconstruction
    # ------------------------------------------------------------------

    @classmethod
    def from_trace_spec(cls, spec: TraceSpec) -> "Histogram2D":
        z = spec.backend_data.get("z")
        raw_histfunc = spec.params.get("histfunc")
        # Backward compat: old specs stored histfunc="count" when z was None.
        if raw_histfunc == "count" or raw_histfunc is None:
            histfunc = None
        elif raw_histfunc in ("median", "n_unique"):
            raise ValueError(
                f"histfunc={raw_histfunc!r} is no longer supported by Histogram2D "
                f"(removed in favour of the Rust kernel). "
                f"Use one of: {_HIST2D_HISTFUNC_OPTIONS}."
            )
        else:
            histfunc = raw_histfunc
        trace = cls(
            x=spec.backend_data["x"],
            y=spec.backend_data["y"],
            x_bins=spec.params.get("x_bins", 20),
            y_bins=spec.params.get("y_bins", 20),
            z=z,
            histfunc=histfunc if z is not None else None,
            histnorm=spec.params.get("histnorm"),
            name=spec.display.get("name"),
            color_scale=spec.display.get("color_scale"),
            color_range=spec.display.get("color_range"),
            axes=spec.axes or ("x", "y"),
        )
        trace.uid = spec.uid
        return trace


def _centers(lo: float, hi: float, n: int) -> list[float]:
    step = (hi - lo) / n
    return [lo + (i + 0.5) * step for i in range(n)]


def _hist2d_phys_col(col: str, schema: pl.Schema | None) -> pl.Expr:
    """The data expression to feed the numeric kernel: physical representation
    for a temporal column (the kernel needs numeric data), else the raw column.
    """
    dtype = _temporal_dtype_for_col(col, schema)
    return pl.col(col).to_physical() if dtype is not None else pl.col(col)


def _hist2d_bound_lits(
    range_: tuple, dtype: pl.DataType | None
) -> tuple[pl.Expr, pl.Expr]:
    """Viewport bin-edge literals matching the kernel's data space: physical
    units for a temporal axis, plain floats otherwise."""
    if dtype is not None:
        return (
            _physical_bound_expr(range_[0], dtype),
            _physical_bound_expr(range_[1], dtype),
        )
    return pl.lit(float(range_[0])), pl.lit(float(range_[1]))


def _hist2d_bounds(
    x_col: str,
    y_col: str,
    x_range: tuple | None,
    y_range: tuple | None,
    schema: pl.Schema | None = None,
) -> tuple:
    """Resolve viewport bounds and the viewport mask for the hist2d kernels.

    When a viewport is given, returns scalar ``pl.lit`` bound expressions and
    the inclusive viewport mask.  When no viewport is given, returns lazy
    column expressions that read from the global-stats columns injected by
    ``LFQueryBuilder``.

    Returns ``(x_lo, x_hi, y_lo, y_hi, viewport_mask_or_None, global_stats_cols)``.

    Boundary note: ``is_between`` is inclusive on both ends, so a value equal
    to ``x_hi`` passes the filter and is placed in the last bin by the Rust
    kernel's ``min(xi, max_xi)`` clamp — both sides must agree on this
    inclusive-right semantics.
    """
    # Histogram2D has both x and y axes; the engine always supplies both
    # or neither. Partial viewport (one axis only) cannot occur here.
    if x_range is not None and y_range is not None:
        x_dtype = _temporal_dtype_for_col(x_col, schema)
        y_dtype = _temporal_dtype_for_col(y_col, schema)
        # Cast bounds to column dtype to avoid upcasting Float32 columns to
        # Float64 during the filter (which doubles memory bandwidth and disables
        # Float32 SIMD). Falls back to plain pl.lit() when schema is unavailable.
        x_filter = _typed_range_bounds(x_col, (x_range[0], x_range[1]), schema)
        y_filter = _typed_range_bounds(y_col, (y_range[0], y_range[1]), schema)
        viewport_mask = pl.col(x_col).is_between(*x_filter) & pl.col(y_col).is_between(
            *y_filter
        )
        # Bin edges must be in the same space as the (possibly physical) data
        # columns the kernel sees: physical units for temporal axes, the raw
        # numeric range otherwise.
        x_lo, x_hi = _hist2d_bound_lits(x_range, x_dtype)
        y_lo, y_hi = _hist2d_bound_lits(y_range, y_dtype)
        return (x_lo, x_hi, y_lo, y_hi, viewport_mask, ())
    else:
        x_lo_expr = pl.col(f"__hist_lo_{x_col}__").first().fill_null(0.0)
        x_hi_expr = pl.col(f"__hist_hi_{x_col}__").first().fill_null(1.0)
        y_lo_expr = pl.col(f"__hist_lo_{y_col}__").first().fill_null(0.0)
        y_hi_expr = pl.col(f"__hist_hi_{y_col}__").first().fill_null(1.0)
        return (x_lo_expr, x_hi_expr, y_lo_expr, y_hi_expr, None, (x_col, y_col))


def _hist2d_count_expr(
    x_col: str,
    y_col: str,
    nb_x: int,
    nb_y: int,
    x_range: tuple | None,
    y_range: tuple | None,
    uid: str,
    schema: pl.Schema | None = None,
) -> tuple[pl.Expr, tuple[str, ...]]:
    """Build a count-only 2D histogram expression using fixed_hist2d.

    Returns ``(expr, global_stats_cols)``.  When no viewport is given the
    data bounds are derived via ``global_stats_cols``; when a viewport is
    given the bounds are known scalars and data is pre-filtered.

    The Rust kernel adds its own internal EPS to ``(x_hi - x_lo)`` when
    computing the bin scale, so callers must pass raw bounds (not EPS-adjusted).
    """
    x_lo, x_hi, y_lo, y_hi, mask, global_stats_cols = _hist2d_bounds(
        x_col, y_col, x_range, y_range, schema
    )
    x_phys = _hist2d_phys_col(x_col, schema)
    y_phys = _hist2d_phys_col(y_col, schema)
    x_expr = x_phys.filter(mask) if mask is not None else x_phys
    y_expr = y_phys.filter(mask) if mask is not None else y_phys
    expr = x_expr.flexviz.fixed_hist2d(
        y_expr,
        x_lo,
        x_hi,
        y_lo,
        y_hi,
        nb_x,
        nb_y,
    ).alias(uid)
    return expr, global_stats_cols


def _hist2d_reduce_expr(
    x_col: str,
    y_col: str,
    z_col: str,
    nb_x: int,
    nb_y: int,
    x_range: tuple | None,
    y_range: tuple | None,
    uid: str,
    histfunc: str,
    schema: pl.Schema | None = None,
) -> tuple[pl.Expr, tuple[str, ...]]:
    """Build a z-reduced 2D histogram expression using fixed_hist2d_reduce."""
    x_lo, x_hi, y_lo, y_hi, mask, global_stats_cols = _hist2d_bounds(
        x_col, y_col, x_range, y_range, schema
    )
    x_phys = _hist2d_phys_col(x_col, schema)
    y_phys = _hist2d_phys_col(y_col, schema)
    x_expr = x_phys.filter(mask) if mask is not None else x_phys
    y_expr = y_phys.filter(mask) if mask is not None else y_phys
    z_expr = pl.col(z_col).filter(mask) if mask is not None else pl.col(z_col)
    expr = x_expr.flexviz.fixed_hist2d_reduce(
        y_expr,
        z_expr,
        x_lo,
        x_hi,
        y_lo,
        y_hi,
        nb_x,
        nb_y,
        histfunc,
    ).alias(uid)
    return expr, global_stats_cols

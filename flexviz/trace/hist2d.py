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
change (same behaviour as 1D Histogram).  Unzoomed, the bin edges span the
engine-resolved data range.  Zoomed, they span the viewport snapped outward to
a fixed lattice, so the grid stands still while the user pans.
"""

from __future__ import annotations

from collections.abc import Mapping
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
    _physical_to_temporal_series,
    _temporal_dtype_for_col,
)
from ._hist_helpers import (
    HeatmapColorRange,
    _HISTNORM_OPTIONS,
    _hist2d_agg_spec,
    apply_histnorm,
    normalize_heatmap_color_scale,
    normalize_heatmap_color_range,
)

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
        # The grid the last request actually binned on: a zoomed request snaps
        # its edges to a lattice, which can add one bin per axis. _to_update
        # unpacks z_flat with this, not with the configured bin counts.
        self._grid: tuple[int, int] = (x_bins, y_bins)

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

    def domain_cols(self, update_range: Dict[str, Any]) -> tuple[str, ...]:
        # Each axis re-bins on its own, so only an axis the viewport leaves
        # out still needs its unfiltered domain.
        return tuple(
            col
            for axis, col in (("x", self.x_col), ("y", self.y_col))
            if update_range.get(axis) is None
        )

    def get_aggregation_spec(
        self,
        update_range: Dict[str, Any],
        schema: pl.Schema | None = None,
        *,
        domains: Mapping[str, tuple[Any, Any]] | None = None,
        scan_source: bool = False,
        sorted_cols: frozenset[str] = frozenset(),
    ) -> AggregationSpec:
        """Return the 2-D histogram aggregation spec (see ``_hist2d_agg_spec``)."""
        # Temporal axes bin on their physical representation; _to_update restores
        # datetime centers afterward.
        self._x_temporal_dtype = _temporal_dtype_for_col(self.x_col, schema)
        self._y_temporal_dtype = _temporal_dtype_for_col(self.y_col, schema)
        spec, self._grid = _hist2d_agg_spec(
            self.x_col,
            self.y_col,
            self.z_col,
            self.histfunc,
            update_range.get("x"),
            update_range.get("y"),
            self.x_bins,
            self.y_bins,
            self.uid,
            domains,
            schema,
            scan_source,
        )
        return spec

    def _to_update(self, df: pl.DataFrame) -> TraceResult:
        raw = df[self.uid][0]
        nb_x, nb_y = self._grid

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

        x_step = (x_hi - x_lo) / nb_x
        y_step = (y_hi - y_lo) / nb_y

        if self.histnorm is not None:
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

        # The client derives one {x0,x1,y0,y1} per cell from these triples;
        # sending the per-cell objects instead is most of a hist2d response.
        return TraceResult(
            updates={
                "x": x_out,
                "y": y_out,
                "z": z,
                "x_edges": [x_edge(x_lo), x_edge(x_step), nb_x],
                "y_edges": [y_edge(y_lo), y_edge(y_step), nb_y],
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

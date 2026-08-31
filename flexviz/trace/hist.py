"""Histogram — renderer-agnostic 1-D histogram trace.

Computes binned counts (or normalized variants) lazily using Polars'
built-in ``hist`` expression.  Either x or y must be provided (not both);
the unspecified axis receives the computed counts, giving a vertical or
horizontal bar chart respectively.

Supported ``histnorm`` values
-----------------------------
``"count"``             — raw bin counts (default)
``"percent"``           — count / total * 100
``"probability"``       — count / total
``"density"``           — count / bin_width
``"probability density"`` — count / (total * bin_width)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict

import polars as pl

import flexviz_polars  # noqa: F401 — registers pl.Expr.flexviz namespace

from ..cube import CubeTargetSpec, FreeAxisSpec, MeasureSpec, TargetDimSpec
from ..LF import AggregationSpec, GroupedAggregationSpec
from ..spec import TraceHoverSpec, TraceSpec
from .base import (
    FlexTrace,
    GroupedChildResult,
    TraceResult,
    _categorical_dims_ok,
    _child_uid_for_group,
    _dtype_for_col,
    _group_value_key,
    _group_values_from_frame,
    _phys_epoch_ms_factor,
    _physical_bound_expr,
    _physical_to_temporal_series,
    _temporal_dtype_for_col,
    _to_col_tuple,
    _range_filter_expr,
)
from ._hist_helpers import _HISTNORM_OPTIONS as _HIST2D_HISTNORM_OPTIONS
from ._hist_helpers import kernel_bin_index

# For 1-D histograms "histnorm" describes what the count-axis displays, so
# "count" (raw bin counts) is a meaningful, natural value — not a no-op.
_HISTNORM_OPTIONS = ("count",) + _HIST2D_HISTNORM_OPTIONS[1:]

# ---------------------------------------------------------------------------
# Bin-edge helpers
# ---------------------------------------------------------------------------

#: Small offset added to the upper bound so the maximum data point always
#: falls inside the last bin and bin edges are never degenerate.
_HIST_BIN_EPSILON: float = 1e-10


def _streaming_bin_expr(value: pl.Expr, lo: float, hi: float, bins: int) -> pl.Expr:
    """The 1D kernel's bin index for a scan source.

    Only the scale is 1D-specific: ``hi == lo`` gives ``scale = 0``, so every
    value lands in bin 0, which is what the kernel's ``count_degenerate`` does.
    ``hi < lo`` is rejected by the kernel and must be rejected before reaching
    here. The rest is ``kernel_bin_index``, shared with the 2D plan so the two
    cannot drift.
    """
    scale = bins / (hi - lo) if hi > lo else 0.0
    return kernel_bin_index(value, lo, scale, bins)


def _resolve_hist_bounds(
    lo_expr: pl.Expr,
    hi_expr: pl.Expr,
    needs_stats: bool,
    stats_row: pl.DataFrame | None,
) -> tuple[float, float]:
    """Evaluate the *same* bound expressions the kernel path is given.

    Not a reimplementation of ``_histogram_bounds_exprs`` -- deliberately the
    expressions it returns, evaluated. Recomputing the bounds in Python drifts
    from the kernel in ways that are invisible until they are not: on a
    Float32 column the ``+ _HIST_BIN_EPSILON`` in ``hi_expr`` is swallowed by
    Float32 precision, so recomputing it in Float64 keeps an epsilon the
    kernel never had and shifts every bin edge by ~1e-16.

    A plan only receives the *filtered* frame, and the broadcast
    ``__hist_lo_*`` columns must be read off the unfiltered one or bin edges
    would move with every cross-filter. ``LFQueryBuilder`` resolves them once
    and hands the row over.
    """
    if needs_stats:
        if stats_row is None:
            raise ValueError(
                "the streaming histogram needs global stats when there is no "
                "viewport range; LFQueryBuilder did not supply a stats row"
            )
        frame = stats_row
    else:
        # Viewport bounds are literals; any single row will do as a carrier.
        frame = pl.DataFrame({"__unused": [0]})
    row = frame.select(lo_expr.alias("__lo"), hi_expr.alias("__hi"))
    return float(row["__lo"][0]), float(row["__hi"][0])


def _streaming_hist_plan(
    data_col: str,
    bins: int,
    uid: str,
    lo_expr: pl.Expr,
    hi_expr: pl.Expr,
    *,
    needs_stats: bool,
    temporal_dtype: pl.DataType | None = None,
    filter_expr: pl.Expr | None = None,
    group_cols: tuple[str, ...] | None = None,
):
    """Streaming histogram for scan sources, ungrouped or grouped.

    The plugin kernel takes a contiguous ``Series`` and cannot stream. This
    plan uses ``group_by(bin).len()`` (or ``group_by(group, bin).len()`` when
    grouped) on the streaming engine, so peak memory does not grow with the
    file.

    When ``group_cols`` is ``None``, returns a one-column imploded struct
    (for ``AggregationSpec``). When set, returns ``group_cols`` plus the uid
    column (for ``GroupedAggregationSpec``).
    """

    def run(
        filtered_ldf: pl.LazyFrame, stats_row: pl.DataFrame | None = None
    ) -> pl.DataFrame:
        lo, hi = _resolve_hist_bounds(lo_expr, hi_expr, needs_stats, stats_row)
        if lo > hi:
            raise ValueError(f"histogram bounds are inverted: lo={lo} > hi={hi}")

        value = (
            pl.col(data_col).to_physical()
            if temporal_dtype is not None
            else pl.col(data_col)
        )
        src = filtered_ldf if filter_expr is None else filtered_ldf.filter(filter_expr)
        bin_expr = _streaming_bin_expr(value, lo, hi, bins).alias("__bin")

        if group_cols is None:
            counted = src.group_by(bin_expr).len("count").collect(engine="streaming")
            return _densify_and_format(counted, lo, hi, bins, uid)

        counted = (
            src.select(*[pl.col(c) for c in group_cols], bin_expr)
            .group_by(*group_cols, "__bin")
            .len("count")
            .collect(engine="streaming")
        )
        return _densify_grouped(counted, group_cols, lo, hi, bins, uid)

    return run


def _densify_and_format(
    counted: pl.DataFrame, lo: float, hi: float, bins: int, uid: str
) -> pl.DataFrame:
    """Sparse bin counts -> the kernel's imploded struct format."""
    counts = (
        pl.DataFrame({"__bin": range(bins)}, schema={"__bin": pl.Int32})
        .join(counted, on="__bin", how="left")
        .sort("__bin")["count"]
        .fill_null(0)
        .cast(pl.UInt32)
    )
    step = (hi - lo) / bins if hi > lo else 0.0
    return pl.DataFrame(
        {
            "breakpoint": [lo + (i + 1.0) * step for i in range(bins)],
            "count": counts,
        },
        schema={"breakpoint": pl.Float64, "count": pl.UInt32},
    ).select(pl.struct("breakpoint", "count").implode().alias(uid))


def _densify_grouped(
    counted: pl.DataFrame,
    group_cols: tuple[str, ...],
    lo: float,
    hi: float,
    bins: int,
    uid: str,
) -> pl.DataFrame:
    """Sparse (group, bin) counts -> per-group imploded struct format."""
    groups = counted.select(*group_cols).unique().sort(*group_cols)
    all_bins = pl.DataFrame({"__bin": range(bins)}, schema={"__bin": pl.Int32})
    dense = (
        groups.join(all_bins, how="cross")
        # nulls_equal: a null group value is a real group the kernel counted,
        # and a left join drops it by default.
        .join(counted, on=[*group_cols, "__bin"], how="left", nulls_equal=True)
        .with_columns(pl.col("count").fill_null(0).cast(pl.UInt32))
        .sort(*group_cols, "__bin")
    )
    step = (hi - lo) / bins if hi > lo else 0.0
    # Derived from __bin rather than positionally, so the result does not
    # depend on the join emitting rows group-major and bin-ascending.
    dense = dense.with_columns(
        (lo + (pl.col("__bin") + 1.0) * step).cast(pl.Float64).alias("breakpoint")
    )
    return (
        dense.group_by(*group_cols, maintain_order=True)
        .agg(pl.struct("breakpoint", "count").implode().alias(uid))
        .sort(*group_cols)
    )


class Histogram(FlexTrace):
    """Scalable 1-D histogram trace backed by a Polars LazyFrame.

    Parameters
    ----------
    x:
        Column name for the data axis when orientation is vertical.
        Provide either ``x`` or ``y``, not both.
    y:
        Column name for the data axis when orientation is horizontal.
    bins:
        Number of bins.
    histnorm:
        Normalization mode.  One of ``"count"``, ``"percent"``,
        ``"probability"``, ``"density"``, ``"probability density"``.
    name:
        Legend / series name.
    color:
        Bar colour hint (CSS string), passed to the renderer.
    axes:
        Axis anchors, e.g. ``("x", "y")``.
    """

    trace_type: str = "histogram"
    select_policy_doc: str = "data (prop) axis only — orthogonal range dropped"
    recompute_policy_doc: str = (
        "binned axis (x or y by orientation) — re-bins to viewport"
    )

    def __init__(
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
    ) -> None:
        if (x is None) == (y is None):
            raise ValueError("Provide either x or y, not both (or neither).")
        if histnorm not in _HISTNORM_OPTIONS:
            raise ValueError(f"histnorm must be one of {_HISTNORM_OPTIONS}.")
        group_cols = (
            _to_col_tuple(group_by, "group_by") if group_by is not None else None
        )

        col = x if x is not None else y
        prop_key = "x" if x is not None else "y"
        # Set before super().__init__ so _default_recompute_axes can read it.
        self._prop_key = prop_key
        # Resolved per-request in get_aggregation_spec when the data column is
        # temporal (binning runs on the physical representation); read back in
        # _to_update to restore datetime bin centers. None ⇒ non-temporal.
        self._data_temporal_dtype: pl.DataType | None = None

        super().__init__(
            backend_data={prop_key: col},
            display={
                "name": name or col,
                **({"color": color} if color is not None else {}),
                **({"color_map": color_map} if color_map is not None else {}),
            },
            params={
                "bins": bins,
                "histnorm": histnorm,
                **({"group_by": list(group_cols)} if group_cols is not None else {}),
            },
            axes=axes,
        )

    def _default_recompute_axes(self) -> tuple[str, ...]:
        # Re-bins on the binned (data) axis only; the count axis is decorative.
        anchor = self._axes[0] if self._prop_key == "x" else self._axes[1]
        return (anchor,)

    def _default_select_axes(self) -> tuple[str, ...]:
        # Select only on the data (prop) axis; the orthogonal count axis carries
        # no selectable data, so a brush there is never emitted as a clause.
        anchor = self._axes[0] if self._prop_key == "x" else self._axes[1]
        return (anchor,)

    def _make_selection_spec(self):
        return self._range_selection_spec()

    def _make_hover_spec(self) -> "TraceHoverSpec":
        return TraceHoverSpec(
            source_modes=["axis", "cell"],
            target_modes=["axis", "cell"],
        )

    # ------------------------------------------------------------------
    # Properties (convenience access)
    # ------------------------------------------------------------------

    @property
    def prop_key(self) -> str:
        return self._prop_key

    @property
    def data_col(self) -> str:
        return self._backend_data[self._prop_key]

    @property
    def bins(self) -> int:
        return self._params["bins"]

    @property
    def histnorm(self) -> str:
        return self._params["histnorm"]

    # ------------------------------------------------------------------
    # Cube descriptors (cross-filter pre-aggregation)
    # ------------------------------------------------------------------

    def get_cube_source_spec(
        self,
        axis_range: tuple[float, float] | None,
        schema: pl.Schema | None = None,
    ) -> FreeAxisSpec | None:
        """A brush on a histogram defines a 1-D free axis on its data column.

        The kind is ``"temporal"`` when the schema says ``data_col`` is a
        temporal dtype (Date/Datetime/Time), else ``"continuous"`` — including
        when no schema is available. Grouped histograms are still valid
        sources: the brush is on the shared data axis, independent of the
        grouping. ``domain`` is the viewport range verbatim (``None`` =
        unzoomed; the engine resolves it to the full data domain).
        """
        dtype = _dtype_for_col(schema, self.data_col)
        kind = "temporal" if dtype is not None and dtype.is_temporal() else "continuous"
        return FreeAxisSpec(column=self.data_col, kind=kind, p=2048, domain=axis_range)

    def get_cube_target_spec(
        self,
        axis_range: tuple[float, float] | None,
        schema: pl.Schema | None = None,
    ) -> CubeTargetSpec | None:
        """An ungrouped histogram is a binned-count cube target.

        ``domain`` is ``axis_range`` verbatim — ``None`` when unzoomed, the
        raw viewport tuple when zoomed. The trace never adds
        ``_HIST_BIN_EPSILON`` here: the **engine** epsilon-pads the upper
        bound uniformly when resolving domains (both ``None``-resolved full
        domains and zoomed viewports), mirroring
        ``_histogram_bounds_exprs`` so cube bins align with display bins.

        A **grouped** histogram appends one categorical dim per ``group_by``
        column after the binned dim (pinned order — contract C); the client
        splits slice cells by those dims into per-child deltas. Every group
        column must pass the string-dtype + reserved-name gate (contracts
        A/B — a schema is therefore required for the grouped case), else
        ``None``. ``histnorm`` never gates target-ability: normalization is
        client-side arithmetic over the counts.
        """
        group_cols = self.group_by_cols or ()
        if group_cols and not _categorical_dims_ok(schema, group_cols):
            return None
        return CubeTargetSpec(
            target_dims=(
                TargetDimSpec(
                    column=self.data_col,
                    kind="binned",
                    bins=self.bins,
                    domain=axis_range,
                ),
                *(TargetDimSpec(column=c, kind="categorical") for c in group_cols),
            ),
            measure=MeasureSpec(agg="count"),
        )

    # ------------------------------------------------------------------
    # FlexTrace interface
    # ------------------------------------------------------------------

    def get_aggregation_spec(
        self,
        update_range: Dict[str, Any],
        schema: pl.Schema | None = None,
        *,
        histogram_domain_cols: Sequence[str] | None = None,
        scan_source: bool = False,
    ) -> AggregationSpec | GroupedAggregationSpec:
        """Return either a regular or grouped histogram aggregation spec.

        Bin edges are always explicit so that multiple histogram traces on the
        same figure, or bg/fg layers in overlay mode, produce aligned bins:

        - When a viewport axis range is present in ``update_range`` the edges
          are ``pl.lit(lo)`` / ``pl.lit(hi)`` scalar expressions derived from
          that range.  Both bg and fg overlay layers use the same spec (and
          therefore the same edges).
        - When no viewport range is available (e.g. ``init`` with no prior
          zoom) the edges are derived lazily from unfiltered global stats.
          The engine can pass same-figure sibling ``histogram_domain_cols`` so
          related histogram traces share one min/max domain without an extra
          collect.
        """
        filter_expr = _range_filter_expr(
            self.data_col, update_range.get(self.prop_key), schema=schema
        )
        axis_range = update_range.get(self.prop_key)

        # Temporal data columns are binned on their physical representation (the
        # numeric kernel rejects temporal dtypes); _to_update restores datetimes.
        self._data_temporal_dtype = _temporal_dtype_for_col(self.data_col, schema)
        temporal = self._data_temporal_dtype

        group_by_cols = self.group_by_cols
        if group_by_cols is not None:
            # ------------------------------------------------------------------
            # Grouped path: a streaming plan, on every source kind. The
            # viewport filter runs inside the plan, before the group_by split.
            # Bin edges are shared across groups so all groups align.
            # ------------------------------------------------------------------
            lo_expr, hi_expr, global_stats_cols = self._histogram_bounds_exprs(
                axis_range, histogram_domain_cols, temporal
            )

            batch_key = (
                self.prop_key,
                self.data_col,
                (
                    tuple(update_range.get(self.prop_key))
                    if update_range.get(self.prop_key) is not None
                    else None
                ),
            )

            # The plugin expression is opaque, so group_by().agg() must
            # materialize per-group value lists. A flat group_by(group, bin)
            # on the streaming engine avoids this on both source kinds.
            return GroupedAggregationSpec(
                uid=self.uid,
                group_cols=group_by_cols,
                sort_cols=group_by_cols,
                agg_exprs=(),
                pre_group_filters=(),
                batch_key=batch_key,
                global_stats_cols=global_stats_cols,
                plan=_streaming_hist_plan(
                    self.data_col,
                    self.bins,
                    self.uid,
                    lo_expr,
                    hi_expr,
                    needs_stats=bool(global_stats_cols),
                    temporal_dtype=temporal,
                    filter_expr=filter_expr,
                    group_cols=group_by_cols,
                ),
            )

        # ----------------------------------------------------------------------
        # Ungrouped path: apply viewport filter directly inside the expression.
        # ----------------------------------------------------------------------
        lo_expr, hi_expr, global_stats = self._histogram_bounds_exprs(
            axis_range, histogram_domain_cols, temporal
        )

        if scan_source:
            return AggregationSpec(
                expr=pl.lit(None).alias(self.uid),
                uid=self.uid,
                global_stats_cols=global_stats,
                plan=_streaming_hist_plan(
                    self.data_col,
                    self.bins,
                    self.uid,
                    lo_expr,
                    hi_expr,
                    needs_stats=bool(global_stats),
                    temporal_dtype=temporal,
                    filter_expr=filter_expr,
                ),
            )

        data_expr = (
            pl.col(self.data_col).to_physical()
            if temporal is not None
            else pl.col(self.data_col)
        )
        if filter_expr is not None:
            data_expr = data_expr.filter(filter_expr)
        hist_expr = data_expr.flexviz.fixed_hist(lo_expr, hi_expr, n_bins=self.bins)
        return AggregationSpec(
            expr=hist_expr.implode().alias(self.uid),
            uid=self.uid,
            global_stats_cols=global_stats,
        )

    def _histogram_bounds_exprs(
        self,
        axis_range: Any,
        histogram_domain_cols: Sequence[str] | None,
        temporal_dtype: pl.DataType | None = None,
    ) -> tuple[pl.Expr, pl.Expr, tuple[str, ...]]:
        if axis_range is not None:
            if temporal_dtype is not None:
                # Viewport bounds arrive as date strings / epoch-ms; convert to
                # the column's physical unit so the bin edges align with the
                # physical (to_physical) data column. _typed_temporal_lit handles
                # naive, Z and offset (±HH:MM) spellings against any column tz —
                # the same conversion the viewport filter already applied.
                lo = _physical_bound_expr(axis_range[0], temporal_dtype)
                hi = _physical_bound_expr(axis_range[1], temporal_dtype)
                return (lo, hi + _HIST_BIN_EPSILON, ())
            return (
                pl.lit(float(axis_range[0])),
                pl.lit(float(axis_range[1]) + _HIST_BIN_EPSILON),
                (),
            )

        domain_cols = self._histogram_domain_cols(histogram_domain_cols)
        # Aggregate after the horizontal reduction: polars rejects horizontal
        # functions over length-1 inputs inside group_by(). Equivalent, since
        # the stats columns are whole-frame constants.
        lo_expr = (
            pl.min_horizontal(*[pl.col(f"__hist_lo_{col}__") for col in domain_cols])
            .first()
            .fill_null(0.0)
        )
        hi_expr = (
            pl.max_horizontal(*[pl.col(f"__hist_hi_{col}__") for col in domain_cols])
            .first()
            .fill_null(1.0)
            + _HIST_BIN_EPSILON
        )
        return lo_expr, hi_expr, domain_cols

    def _histogram_domain_cols(
        self,
        histogram_domain_cols: Sequence[str] | None,
    ) -> tuple[str, ...]:
        cols = list(dict.fromkeys(histogram_domain_cols or (self.data_col,)))
        # Ensure this trace's own column is always in global_stats_cols so that
        # __hist_lo_{data_col}__ / __hist_hi_{data_col}__ exist in the DataFrame.
        if self.data_col not in cols:
            cols.append(self.data_col)
        return tuple(cols)

    def _to_update(
        self,
        df_agg: pl.DataFrame,
    ) -> TraceResult:
        """Unpack the histogram struct and apply normalization."""
        raw: pl.Series = df_agg[self.uid].item()
        df_hist = (
            raw.explode()
            .struct.unnest()
            .with_columns(pl.col("breakpoint") - pl.col("breakpoint").diff().mean() / 2)
            .rename({"breakpoint": "center"})
        )

        # -- normalize the counts (if needed)
        bin_width = df_hist["center"].diff().drop_nulls().mean()
        if bin_width is None or bin_width == 0.0:
            # Degenerate case: bins=1, or all data in a single bin.
            # Use 1.0 so density-normalized values stay finite.
            bin_width = 1.0
        total = df_hist["count"].sum()

        if self.histnorm == "percent":
            df_hist = df_hist.with_columns(pl.col("count") / total * 100)
        elif self.histnorm == "probability":
            df_hist = df_hist.with_columns(pl.col("count") / total)
        elif self.histnorm == "density":
            df_hist = df_hist.with_columns(pl.col("count") / bin_width)
        elif self.histnorm == "probability density":
            df_hist = df_hist.with_columns(pl.col("count") / (total * bin_width))

        # -- hover_bounds: explicit bin edges for cell hover
        half_w = bin_width / 2.0
        centers = df_hist["center"].to_list()

        # Temporal data axis: emit datetime bin centers (so the renderer auto-
        # detects a date axis) and epoch-ms hover bounds (Plotly's numeric date
        # coordinate, what hover-band matching compares against).
        temporal = self._data_temporal_dtype
        if temporal is not None:
            data_axis = _physical_to_temporal_series(
                df_hist["center"], temporal, self.data_col
            )
            factor = _phys_epoch_ms_factor(temporal)

            def _edge(v: Any) -> float | None:
                return None if v is None else float(v) * factor

        else:
            data_axis = df_hist["center"]

            def _edge(v: Any) -> float | None:
                return None if v is None else float(v)

        if self.prop_key == "x":
            hover_bounds = [
                (
                    {"x0": _edge(c - half_w), "x1": _edge(c + half_w)}
                    if c is not None
                    else {"x0": None, "x1": None}
                )
                for c in centers
            ]
            return TraceResult(
                updates={
                    "x": data_axis,
                    "y": df_hist["count"],
                    "hover_bounds": hover_bounds,
                }
            )

        assert self.prop_key == "y"
        hover_bounds = [
            (
                {"y0": _edge(c - half_w), "y1": _edge(c + half_w)}
                if c is not None
                else {"y0": None, "y1": None}
            )
            for c in centers
        ]
        return TraceResult(
            updates={
                "x": df_hist["count"],
                "y": data_axis,
                "orientation": "h",
                "hover_bounds": hover_bounds,
            }
        )

    def _to_grouped_update(self, df_grouped: pl.DataFrame) -> TraceResult:
        """Unpack grouped histogram output into one child result per group."""
        group_by_cols = self.group_by_cols
        assert group_by_cols is not None, "Grouped histogram requires group_by"
        group_results: list[GroupedChildResult] = []
        for i, gv in enumerate(_group_values_from_frame(df_grouped, group_by_cols)):
            child_df = df_grouped.select(pl.col(self.uid).slice(i, 1))
            child_result = self._to_update(child_df)
            group_results.append(
                GroupedChildResult(
                    child_uid=_child_uid_for_group(self.uid, gv),
                    group_value_key=_group_value_key(gv),
                    updates=child_result.updates,
                )
            )
        return TraceResult(group_results=group_results)

    # ------------------------------------------------------------------
    # Spec reconstruction (server-side)
    # ------------------------------------------------------------------

    @classmethod
    def from_trace_spec(cls, spec: TraceSpec) -> "Histogram":
        prop_key = next(iter(spec.backend_data))  # "x" or "y"
        col = spec.backend_data[prop_key]
        trace = cls(
            **{prop_key: col},
            bins=spec.params["bins"],
            histnorm=spec.params["histnorm"],
            name=spec.display.get("name"),
            color=spec.display.get("color"),
            color_map=spec.display.get("color_map"),
            axes=spec.axes or ("x", "y"),
            group_by=spec.params.get("group_by"),
        )
        if "group_domain_key" in spec.params:
            trace._params["group_domain_key"] = spec.params["group_domain_key"]
        trace.uid = spec.uid
        return trace

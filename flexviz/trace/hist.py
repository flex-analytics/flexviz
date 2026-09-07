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

from collections.abc import Mapping, Sequence
from typing import Any, Dict

import polars as pl

import flexviz_polars  # noqa: F401 — registers pl.Expr.flexviz namespace

from ..cube import (
    CubeTargetSpec,
    FreeAxisSpec,
    MeasureSpec,
    TargetDimSpec,
    _fixed_hist_bin_expr,
)
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
    _physical_to_temporal_series,
    _temporal_dtype_for_col,
    _to_col_tuple,
)
from ._hist_helpers import _HISTNORM_OPTIONS as _HIST2D_HISTNORM_OPTIONS
from .batch_fold import hist1d_fold_plan
from .bin_grid import snap_range, snapped_axis

# For 1-D histograms "histnorm" describes what the count-axis displays, so
# "count" (raw bin counts) is a meaningful, natural value — not a no-op.
_HISTNORM_OPTIONS = ("count",) + _HIST2D_HISTNORM_OPTIONS[1:]

# ---------------------------------------------------------------------------
# Bin-edge helpers
# ---------------------------------------------------------------------------

#: Small offset added to the upper bound (``hi``) so the maximum data point
#: always falls inside the last bin and bin edges are never degenerate. On a
#: zoomed axis it pads the *snapped* ``hi``, so a value sitting exactly on it
#: still lands in the last bin. Distinct from the cube's
#: ``_FIXED_HIST_ROUND_EPS``, which pads the bin index instead of ``hi``.
_HIST_BIN_EPSILON: float = 1e-10


def _streaming_hist_plan(
    value_expr: pl.Expr,
    lo_expr: pl.Expr,
    hi_expr: pl.Expr,
    bins: int,
    uid: str,
    group_cols: tuple[str, ...],
):
    """The grouped kernel histogram as a streaming ``group_by``.

    The ``fixed_hist`` kernel materializes the whole column, and a grouped
    histogram would hold every group's column at once; this reads the frame in
    batches instead. The bin index comes from the cube's
    ``_fixed_hist_bin_expr``, the one Python mirror of the kernel arithmetic;
    its non-strict cast turns NaN into null so NaN and null both drop out at
    the dense join. Empty bins come back as zero, ordered: only the counts are
    emitted, because the trace derives its bin centers from the bounds.

    It returns one row per group, sorted by group value, each holding that
    group's bins: the shape the fused grouped query returns. A null group value
    keeps its own child, because the dense join compares null keys as equal. The
    viewport arrives through ``pre_group_filters``, before the group split, so
    the plan takes no filter of its own.

    The plan's own columns share the frame with the group columns, so they carry
    a ``__fv_`` prefix: a group column named ``count`` would otherwise collide.
    The emitted struct still calls its field ``count``, the name the kernel uses
    and ``_to_update`` reads.
    """

    def run(filtered_ldf: pl.LazyFrame) -> pl.DataFrame:
        # Both bounds are literal expressions, so this selects no data.
        lo, hi = pl.select(lo_expr.alias("lo"), hi_expr.alias("hi")).row(0)
        # The kernel reads its bounds as f64 and refuses an inverted span.
        lo = 0.0 if lo is None else float(lo)
        hi = 1.0 if hi is None else float(hi)
        if hi < lo:
            raise ValueError(f"histogram bounds are inverted: lo={lo} > hi={hi}")

        bin_idx = _fixed_hist_bin_expr(value_expr, lo, hi, bins, "__fv_b")
        all_bins = pl.DataFrame({"__fv_b": range(bins)}, schema={"__fv_b": pl.Int32})

        cols = list(group_cols)
        counted = (
            filtered_ldf.group_by([*cols, bin_idx])
            .agg(pl.len().alias("__fv_count"))
            .collect(engine="streaming")
        )
        # No rows leaves no groups, so the chain returns a zero-row frame of the
        # right shape and needs no branch of its own.
        dense = counted.select(cols).unique().join(all_bins, how="cross")
        return (
            dense.join(counted, on=[*cols, "__fv_b"], how="left", nulls_equal=True)
            .sort([*cols, "__fv_b"])
            .with_columns(pl.col("__fv_count").fill_null(0).cast(pl.UInt32))
            .group_by(cols, maintain_order=True)
            .agg(pl.struct(pl.col("__fv_count").alias("count")).alias(uid))
            .sort(cols)
        )

    return run


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
        Number of bins. A zoomed axis can show one more, because the grid snaps to a
        fixed lattice.
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
        # Resolved per-request in get_aggregation_spec: the exact bounds and bin
        # count the kernel bins with. _to_update derives the bin centers and the
        # wire-format bin edges from them.
        self._bin_edges: tuple[float, float, int] = (0.0, 1.0, bins)

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

        ``domain`` is ``None`` when unzoomed. Zoomed it is the viewport
        snapped to the display lattice, with the bin count the snap yields
        (``bins`` or ``bins + 1``): the client derives bar centers from
        ``domain`` and ``bins``, so a cube-served bar would otherwise miss the
        server's bar. ``axis_range`` is already physical here
        (``FlexEngine._cube_axis_range``), so only the lattice rule applies.
        The trace never adds ``_HIST_BIN_EPSILON``: the **engine**
        epsilon-pads the upper bound uniformly when resolving domains (both
        ``None``-resolved full domains and snapped viewports), mirroring
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
        bins, domain = self.bins, None
        if axis_range is not None:
            lo, hi, bins = snap_range(
                float(axis_range[0]), float(axis_range[1]), self.bins
            )
            domain = (lo, hi)
        return CubeTargetSpec(
            target_dims=(
                TargetDimSpec(
                    column=self.data_col,
                    kind="binned",
                    bins=bins,
                    domain=domain,
                ),
                *(TargetDimSpec(column=c, kind="categorical") for c in group_cols),
            ),
            measure=MeasureSpec(agg="count"),
        )

    # ------------------------------------------------------------------
    # FlexTrace interface
    # ------------------------------------------------------------------

    def domain_cols(self, update_range: Dict[str, Any]) -> tuple[str, ...]:
        if update_range.get(self.prop_key) is not None:
            return ()
        return (self.data_col,)

    def get_aggregation_spec(
        self,
        update_range: Dict[str, Any],
        schema: pl.Schema | None = None,
        *,
        domains: Mapping[str, tuple[Any, Any]] | None = None,
        scan_source: bool = False,
        **_: Any,
    ) -> AggregationSpec | GroupedAggregationSpec:
        """Return either a regular or grouped histogram aggregation spec.

        Bin edges are always explicit so that multiple histogram traces on the
        same figure, or bg/fg layers in overlay mode, produce aligned bins:

        - When a viewport axis range is present in ``update_range`` the edges
          are that range snapped outward to a fixed lattice, so the grid stands
          still while the user pans; the count is then ``bins`` or ``bins + 1``.
          Both bg and fg overlay layers use the same spec (and therefore the
          same edges).
        - When no viewport range is available (e.g. ``init`` with no prior
          zoom) the edges come from ``domains``: the unfiltered ``(min, max)``
          the engine resolved for this trace. It holds one entry per
          same-figure sibling column, so related histograms bin over one
          shared domain.

        An unzoomed trace requires ``data_col`` in ``domains``; a
        ``(None, None)`` entry means an empty or all-null column.

        ``scan_source`` says the rows come from storage rather than a resident
        frame. The ``fixed_hist`` kernel needs the whole column in memory, so an
        ungrouped histogram on a scan takes ``hist1d_fold_plan`` instead: the
        same kernel, run per streamed batch and summed, at bounded memory. A
        grouped histogram takes the streaming ``group_by`` plan on both source
        kinds, because one kernel per group would hold every group's column at
        once.
        """
        axis_range = update_range.get(self.prop_key)

        # Temporal data columns are binned on their physical representation (the
        # numeric kernel rejects temporal dtypes); _to_update restores datetimes.
        self._data_temporal_dtype = _temporal_dtype_for_col(self.data_col, schema)
        data_col_expr = (
            pl.col(self.data_col).to_physical()
            if self._data_temporal_dtype is not None
            else pl.col(self.data_col)
        )
        lo, hi, n_bins, filter_expr = self._histogram_bounds_exprs(
            axis_range, domains, schema
        )
        self._bin_edges = (lo, hi, n_bins)
        lo_expr, hi_expr = pl.lit(lo), pl.lit(hi)

        group_by_cols = self.group_by_cols
        if group_by_cols is not None:
            # ------------------------------------------------------------------
            # Grouped path: the viewport mask runs as a pre_group_filter, before
            # the group_by split, and every group shares the same bin edges.
            #
            # One plan per grouped histogram, on both source kinds: the kernel
            # would hold every group's column in memory at once. A plan spec
            # runs alone and never joins the fused query, so it carries no
            # batch_key / pre_group_filter_key (LF.aggregate only reads those
            # for expression specs).
            return GroupedAggregationSpec(
                uid=self.uid,
                group_cols=group_by_cols,
                sort_cols=group_by_cols,
                agg_exprs=(),
                pre_group_filters=(filter_expr,) if filter_expr is not None else (),
                plan=_streaming_hist_plan(
                    data_col_expr,
                    lo_expr,
                    hi_expr,
                    n_bins,
                    self.uid,
                    group_by_cols,
                ),
            )

        # ----------------------------------------------------------------------
        # Ungrouped path: the viewport mask runs inside the kernel expression;
        # the scan fold filters the frame, so the scan itself rejects the rows.
        # ----------------------------------------------------------------------
        if scan_source:
            return AggregationSpec(
                uid=self.uid,
                plan=hist1d_fold_plan(
                    data_col_expr, lo, hi, n_bins, self.uid, filter_expr
                ),
            )

        data_expr = data_col_expr
        if filter_expr is not None:
            data_expr = data_expr.filter(filter_expr)
        hist_expr = data_expr.flexviz.fixed_hist(lo_expr, hi_expr, n_bins=n_bins)
        return AggregationSpec(expr=hist_expr.implode().alias(self.uid), uid=self.uid)

    def _histogram_bounds_exprs(
        self,
        axis_range: Any,
        domains: Mapping[str, tuple[Any, Any]] | None,
        schema: pl.Schema | None = None,
    ) -> tuple[float, float, int, pl.Expr | None]:
        """Bin edges in the kernel's data space, the bin count, and the viewport
        mask (``None`` when unzoomed).

        Zoomed, the edges are the viewport snapped outward to the lattice of
        width ``(hi - lo) / bins``, so the grid stands still while the user pans,
        and the mask restricts the rows to that same span, so an edge bin is
        complete. Snapping costs at most one extra bin, which is why the count
        comes back too.
        """
        if axis_range is not None:
            lo, hi, n, mask = snapped_axis(self.data_col, axis_range, self.bins, schema)
            return lo, hi + _HIST_BIN_EPSILON, n, mask

        # The trace's own column must be a resolved key; a missing key means
        # the caller violated the unzoomed-domains contract.
        resolved = domains or {}
        if self.data_col not in resolved:
            raise KeyError(self.data_col)

        # Siblings sharing a domain widen it: the lowest low and the highest
        # high across every column the engine resolved for this trace.
        bounds = resolved.values()
        los = [lo for lo, _ in bounds if lo is not None]
        his = [hi for _, hi in bounds if hi is not None]
        lo = min(los) if los else 0.0
        hi = max(his) if his else 1.0
        return lo, hi + _HIST_BIN_EPSILON, self.bins, None

    def _to_update(
        self,
        df_agg: pl.DataFrame,
    ) -> TraceResult:
        """Unpack the histogram struct and apply normalization."""
        counts: pl.Series = df_agg[self.uid].item().explode().struct.field("count")

        lo, hi, n_bins = self._bin_edges
        step = (hi - lo) / n_bins
        # hi == lo is the kernel's degenerate span: its epsilon pad vanishes at
        # the column's magnitude. Use 1.0 so density norms stay finite.
        bin_width = step if step > 0.0 else 1.0
        centers = (pl.int_range(0, n_bins, eager=True) + 0.5) * step + lo

        total = counts.sum()
        if self.histnorm == "percent":
            counts = counts / total * 100
        elif self.histnorm == "probability":
            counts = counts / total
        elif self.histnorm == "density":
            counts = counts / bin_width
        elif self.histnorm == "probability density":
            counts = counts / (total * bin_width)

        # Temporal data axis: emit datetime bin centers (so the renderer auto-
        # detects a date axis) and epoch-ms bin edges (Plotly's numeric date
        # coordinate, what hover-band matching compares against).
        temporal = self._data_temporal_dtype
        if temporal is not None:
            data_axis = _physical_to_temporal_series(centers, temporal, self.data_col)
            factor = _phys_epoch_ms_factor(temporal)
        else:
            data_axis = centers
            factor = 1.0

        # The client derives one {x0,x1} per bin from this triple; sending the
        # per-bin objects instead is most of a histogram response.
        edges = [float(lo) * factor, step * factor, n_bins]

        if self.prop_key == "x":
            return TraceResult(updates={"x": data_axis, "y": counts, "x_edges": edges})

        assert self.prop_key == "y"
        return TraceResult(
            updates={
                "x": counts,
                "y": data_axis,
                "orientation": "h",
                "y_edges": edges,
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

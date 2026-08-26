"""LinePlot — renderer-agnostic line / time-series trace.

Downsamples a sorted series in the current viewport.

Downsampling strategies:

* ``"minmax"`` (default) — min-max envelope downsampling: splits the viewport into
  ``n_points // 2`` equal buckets and keeps the argmin + argmax of ``y`` within each
  bucket, yielding at most ``n_points`` output points that preserve extrema and spikes.

* ``"fpcs"`` — Feature-Preserving Compensated Sampling: first applies the same
  index-bucket MinMax reduction to interior points, then runs the FPCS compensation
  pass and appends the first and last point.  ``n_points`` is a target, not a hard
  cap; output may be up to roughly ``2 * n_points``.  X-aware buckets are planned
  for a later version.

* ``"nth"`` — uniform stride gather: every ``max(1, n // n_points)``-th row.
  Stride is computed inside the Rust kernel (no Polars ``len()`` expression
  dependency), enabling full parallelism across N grouped sub-traces in a single
  ``select()``.

"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict, Literal

import polars as pl

from ..cube import (
    CubeTargetSpec,
    FreeAxisSpec,
    MeasureSpec,
    TargetDimSpec,
    temporal_unit,
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
    _to_col_tuple,
    _range_filter_expr,
    _typed_range_bounds,
)

import flexviz_polars as _fvp  # noqa: F401 — registers pl.Expr.flexviz namespace

LineDownsample = Literal["minmax", "fpcs", "nth"]

# A viewport restriction: either an ``is_between`` mask, or a zero-copy
# ``(offset, length)`` slice when x is known sorted.
# TODO: should we not always assume that x is sorted? (KISS)
Viewport = pl.Expr | tuple[pl.Expr, pl.Expr]


def _apply_viewport(expr: pl.Expr, vp: Viewport | None) -> pl.Expr:
    """Restrict ``expr`` to the viewport.

    ``filter`` materialises the surviving rows; ``slice`` is zero-copy, so the
    downsample kernel reads the original buffer at an offset. The gap widens as
    the user zooms in — at 100M rows, a 60% viewport is 3x faster and a 1%
    viewport 35x, against a flat ~1 MB instead of ~75 MB (measured at 100M
    rows on a 16-core desktop).
    """
    if vp is None:
        return expr
    if isinstance(vp, tuple):
        return expr.slice(*vp)
    return expr.filter(vp)


def _viewport_window(
    x_col: str,
    x_range: tuple | None,
    schema: pl.Schema | None,
    x_sorted: bool,
) -> Viewport | None:
    """Build the viewport restriction for a *select-level* line aggregation.

    Sorted x makes the viewport a contiguous row range, so it can be expressed
    as a binary-searched slice instead of a mask. ``search_sorted`` stays inside
    the expression on purpose: cross-filter predicates are applied to the frame
    before these expressions run, and since filtering preserves order the
    bounds must be resolved against the *already filtered* column, not the
    original one.
    """
    if x_range is None:
        return None
    if not x_sorted:
        return _range_filter_expr(x_col, x_range, schema)
    bounds = _typed_range_bounds(x_col, x_range, schema)
    if bounds is None:  # untyped column — no safe literal to search for
        return _range_filter_expr(x_col, x_range, schema)
    lo, hi = bounds
    x = pl.col(x_col)
    start = x.search_sorted(lo, "left")
    return (start, x.search_sorted(hi, "right") - start)


# ---------------------------------------------------------------------------
# Out-of-core envelope (native Polars, no kernel)
# ---------------------------------------------------------------------------


def _uniform_bucket_expr(ri: pl.Expr, n_rows: int, n_out: int) -> pl.Expr:
    """Row index -> bucket, matching the kernel's ``uniform_offsets`` exactly.

    The kernel splits ``n_rows`` into ``n_out`` windows whose first ``n_rows %
    n_out`` are one element longer. The naive ``ri * n_out // n_rows`` does
    *not* reproduce that (it drifts whenever the remainder is > 1), so invert
    the window layout directly: rows below ``split`` sit in the long windows,
    the rest in the short ones.
    """
    base, rem = divmod(n_rows, n_out)
    split = rem * (base + 1)
    return (
        pl.when(ri < split).then(ri // (base + 1)).otherwise(rem + (ri - split) // base)
    )


def _native_envelope_plan(
    x_col: str,
    y_col: str,
    n_points: int,
    uid: str,
    vp_expr: pl.Expr | None,
):
    """Bit-identical `arg_min_max` replacement that streams.

    The kernel materializes the whole column, so it OOMs on a scan at every cap
    tested. This is the same envelope as two bounded passes:

    1. per-bucket min/max of y — ``min``/``max`` are order-independent, so the
       result does not depend on how the engine schedules morsels;
    2. keep only rows sitting *on* a bucket extremum, then take the **first**
       row index of each — ``min`` over indices is order-independent too.

    Both properties matter. The obvious one-pass form, ``ri.min_by(y)``, breaks
    them: on an exact plateau it picks an arbitrary member, and *which* member
    changes with `POLARS_MAX_THREADS` — so the same data would render
    differently on machines with different core counts. The other one-pass
    form, ``ri.filter(y == y.min()).min()``, is correct but buffers each group
    and OOMs at every cap.

    The bucket extrema are collected between the passes and fed back as literal
    lookups rather than joined: the join builds on the 100M-row side and costs
    ~1.8x the peak (2.76 GB vs 1.51 GB at 100M).
    """

    def run(
        filtered_ldf: pl.LazyFrame, stats_row: pl.DataFrame | None = None
    ) -> pl.DataFrame:
        src = filtered_ldf if vp_expr is None else filtered_ldf.filter(vp_expr)
        src = src.select(x_col, y_col)
        # The bucket layout needs the post-filter row count up front. This is
        # the only blocking step, and it is a count, not a materialization.
        n_rows = int(src.select(pl.len()).collect(engine="streaming").item())
        if n_rows == 0:
            return pl.DataFrame(
                {uid: [[]]},
                schema={uid: pl.List(pl.Struct({x_col: pl.Null, y_col: pl.Null}))},
            )
        n_out = max(min(n_points // 2, n_rows), 1)
        ri = pl.col("__ri").cast(pl.Int64)
        y = pl.col(y_col)
        base = src.with_row_index("__ri").with_columns(
            _uniform_bucket_expr(ri, n_rows, n_out).alias("__b")
        )
        ext = (
            base.group_by("__b")
            .agg(__lo=y.min(), __hi=y.max())
            .sort("__b")
            .collect(engine="streaming")
        )
        lo, hi = pl.lit(ext["__lo"]), pl.lit(ext["__hi"])
        b = pl.col("__b")
        at_lo, at_hi = y == lo.gather(b), y == hi.gather(b)
        x = pl.col(x_col)
        # Pass 2: only rows sitting ON a bucket extremum survive the frame-level
        # filter, so the per-group work below is over a handful of rows. Taking
        # `min_by(ri)` is safe precisely because row indices are unique — the
        # tie-break that makes `min_by(y)` thread-dependent cannot arise here.
        picked = (
            base.filter(at_lo | at_hi)
            .group_by("__b")
            .agg(
                __imin=ri.filter(at_lo).min(),
                __imax=ri.filter(at_hi).min(),
                __xmin=x.filter(at_lo).min_by(ri.filter(at_lo)),
                __xmax=x.filter(at_hi).min_by(ri.filter(at_hi)),
            )
            .collect(engine="streaming")
        )
        # Reassemble into index order: each bucket contributes its argmin and
        # argmax point, deduplicated where a bucket's extremum is a single row.
        # `picked` comes out of a group_by, so its bucket order is arbitrary —
        # look the extrema up by bucket rather than pairing them positionally.
        pts = pl.concat(
            [
                picked.select(i=pl.col("__imin"), xv=pl.col("__xmin"), yv=lo.gather(b)),
                picked.select(i=pl.col("__imax"), xv=pl.col("__xmax"), yv=hi.gather(b)),
            ]
        ).drop_nulls("i")
        pts = pts.unique(subset=["i"], keep="first").sort("i")
        # Field names must be the source column names, not "x"/"y" —
        # `_to_update` looks them up by `self.x_col` / `self.y_col`.
        return pts.select(
            pl.struct(
                **{
                    x_col: pl.col("xv").alias(x_col),
                    y_col: pl.col("yv").alias(y_col),
                }
            )
            .implode()
            .alias(uid)
        )

    return run


# ---------------------------------------------------------------------------
# Aggregation expression builders
# ---------------------------------------------------------------------------


def _plugin_nth_agg_expr(
    x_col: str,
    y_col: str,
    vp: Viewport | None,
    n_points: int,
    uid: str,
) -> pl.Expr:
    """Single-pass every-nth using the flexviz_polars Rust kernel.

    Stride is computed inside the kernel — no ``len()`` expression dependency,
    so N sub-traces inside one ``select()`` can parallelize on a single scan.
    """
    x = _apply_viewport(pl.col(x_col), vp)
    y = _apply_viewport(pl.col(y_col), vp)
    return (
        pl.struct(
            **{
                x_col: x.flexviz.every_nth(n_points),
                y_col: y.flexviz.every_nth(n_points),
            }
        )
        .implode()
        .alias(uid)
    )


def _plugin_minmax_agg_expr(
    x_col: str,
    y_col: str,
    vp: Viewport | None,
    n_points: int,
    uid: str,
) -> pl.Expr:
    """Min-max envelope downsampling using the flexviz_polars Rust kernel.

    Splits the (filtered) y-column into ``n_points // 2`` equal buckets and
    gathers x and y at the argmin and argmax positions within each bucket.
    Preserves extrema and spikes that uniform-stride subsampling would miss.

    Index selection and both gathers happen in one kernel call: Polars does not
    CSE opaque plugin expressions, so the two-gather form
    (``x.gather(idx), y.gather(idx)``) ran the whole argmin/argmax scan twice.
    """
    y_expr = _apply_viewport(pl.col(y_col), vp)
    x_expr = _apply_viewport(pl.col(x_col), vp)
    return (
        _fvp._minmax_line(x_expr, y_expr, n_points, x_name=x_col, y_name=y_col)
        .implode()
        .alias(uid)
    )


def _plugin_fpcs_agg_expr(
    x_col: str,
    y_col: str,
    vp: Viewport | None,
    n_points: int,
    uid: str,
) -> pl.Expr:
    """FPCS downsampling using one Rust kernel for index selection and gather."""
    y_expr = _apply_viewport(pl.col(y_col), vp)
    x_expr = _apply_viewport(pl.col(x_col), vp)
    return (
        _fvp._fpcs_line(x_expr, y_expr, n_points, x_name=x_col, y_name=y_col)
        .implode()
        .alias(uid)
    )


def _plugin_line_agg_expr(
    x_col: str,
    y_col: str,
    vp: Viewport | None,
    n_points: int,
    downsample: LineDownsample,
    uid: str,
) -> pl.Expr:
    if downsample == "minmax":
        return _plugin_minmax_agg_expr(x_col, y_col, vp, n_points, uid)
    if downsample == "fpcs":
        return _plugin_fpcs_agg_expr(x_col, y_col, vp, n_points, uid)
    if downsample == "nth":
        return _plugin_nth_agg_expr(x_col, y_col, vp, n_points, uid)
    raise ValueError(f"unknown downsample strategy {downsample!r}")


def _range_cube_source_spec(
    column: str,
    axis_range: tuple[float, float] | None,
    schema: pl.Schema | None,
) -> FreeAxisSpec | None:
    """Shared 1-D range cube-source descriptor (hist-shaped; cube plan step 8).

    Continuous/temporal kind from the schema dtype; temporal axes carry their
    physical ``unit`` (contract G) and unsupported temporal dtypes
    (``Datetime("ns")``, ``Time``) gate to ``None``, as do non-numeric dtypes.
    Without a schema the kind defaults to ``"continuous"``. ``domain`` is the
    viewport range verbatim (``None`` = unzoomed; engine-resolved).

    A twin lives in ``box.py`` — keep the two in sync (a shared home in
    ``base.py`` is deliberately out of this step's file scope).
    """
    dtype = _dtype_for_col(schema, column)
    if dtype is not None:
        if dtype.is_temporal():
            unit = temporal_unit(dtype)
            if unit is None:
                return None
            return FreeAxisSpec(
                column=column, kind="temporal", p=2048, domain=axis_range, unit=unit
            )
        if not dtype.is_numeric():
            return None
    return FreeAxisSpec(column=column, kind="continuous", p=2048, domain=axis_range)


# ---------------------------------------------------------------------------
# LinePlot trace
# ---------------------------------------------------------------------------


class LinePlot(FlexTrace):
    """A scalable line trace backed by a Polars LazyFrame.

    Parameters
    ----------
    x:
        Column name for the x-axis values.  Should be sorted (ascending)
        for efficient range queries.
    y:
        Column name for the y-axis values.
    name:
        Legend / series name passed to the renderer.
    color:
        Line colour hint (CSS string), passed to the renderer.
    n_points:
        Target number of points returned per viewport after downsampling.
        ``"minmax"`` and ``"nth"`` return at most this many points.  ``"fpcs"``
        follows FPCS target semantics and can return up to roughly
        ``2 * n_points`` points.
    downsample:
        Downsampling strategy.  ``"minmax"`` (default) uses the min-max
        envelope algorithm which requires the ``flexviz_polars`` plugin;
        ``"fpcs"`` uses Feature-Preserving Compensated Sampling; ``"nth"``
        selects every nth row.
    axes:
        Tuple of axis anchors, e.g. ``("x", "y")``.  Used by the engine to
        match viewport events to this trace.
    update_on_zoom:
        When ``True`` (default) the engine re-aggregates this trace whenever
        its axes appear in a viewport event.
    """

    trace_type: str = "line"
    select_policy_doc: str = "x anchor only — vertical band across all series"
    recompute_policy_doc: str = (
        "x anchor — downsample window (frozen if update_on_zoom=False)"
    )

    def __init__(
        self,
        x: str,
        y: str,
        name: str | None = None,
        color: str | None = None,
        color_map: dict | None = None,
        n_points: int = 1000,
        downsample: LineDownsample = "minmax",
        add_gaps: bool = True,
        axes: tuple[str, ...] = ("x", "y"),
        update_on_zoom: bool = True,
        group_by: str | Sequence[str] | None = None,
    ) -> None:
        group_cols = (
            _to_col_tuple(group_by, "group_by") if group_by is not None else None
        )
        super().__init__(
            backend_data={"x": x, "y": y},
            display={
                "name": name or y,
                **({"color": color} if color is not None else {}),
                **({"color_map": color_map} if color_map is not None else {}),
            },
            params={
                "n_points": n_points,
                "downsample": downsample,
                "add_gaps": add_gaps,
                **({"group_by": list(group_cols)} if group_cols is not None else {}),
            },
            axes=axes,
            # x-bound: minmax/nth downsampling is windowed on the x range only.
            # The escape hatch (``update_on_zoom=False``) freezes the trace.
            recompute_axes=None if update_on_zoom else (),
        )

    def _default_recompute_axes(self) -> tuple[str, ...]:
        return (self._axes[0],)

    def _default_select_axes(self) -> tuple[str, ...]:
        # x-only: an x-range brush cross-filters every series sharing the x
        # axis; a 2-D box on a multi-series line would be ambiguous (whose y?).
        return (self._axes[0],)

    def _make_selection_spec(self):
        return self._range_selection_spec()

    def _make_hover_spec(self) -> "TraceHoverSpec":
        return TraceHoverSpec(
            source_modes=["axis"],
            target_modes=["axis"],
        )

    # ------------------------------------------------------------------
    # Cube descriptors (cross-filter pre-aggregation)
    # ------------------------------------------------------------------

    def get_cube_source_spec(
        self,
        axis_range: tuple[float, float] | None,
        schema: pl.Schema | None = None,
    ) -> FreeAxisSpec | None:
        """A brush on a line defines a 1-D free axis on its **x** column.

        Line selection is x-only (locked v0.2 decision, matching Mosaic
        ``intervalX``): ``select_axes = (x anchor,)``, so committed line
        predicates carry only the x-column clause and the cube free axis is
        the x column. Identical shape to the histogram's descriptor —
        P=2048, ``domain`` = the x-viewport range verbatim (``None`` =
        unzoomed; engine-resolved). Temporal x columns carry their physical
        ``unit`` (contract G); ``Datetime("ns")``/``Time`` and non-numeric
        dtypes gate to ``None``. Source-ability is selection geometry, so it
        is independent of ``downsample`` (the minmax-only gate belongs to
        the line *target* descriptor, contract J) and of grouping.
        """
        return _range_cube_source_spec(self.x_col, axis_range, schema)

    def get_cube_target_spec(
        self,
        axis_range: tuple[float, float] | None,
        schema: pl.Schema | None = None,
    ) -> CubeTargetSpec | None:
        """A minmax-downsampled line is a ``line_env`` cube target (contract J).

        The free axis is the brushed source column; this target's grouping
        dims are the line's own **x bucket axis** (a single binned dim) plus
        one categorical dim per ``group_by`` column (grouped lines), and the
        measure is the exact min/max envelope of the **y** column over each
        bucket. Mirrors ``Histogram.get_cube_target_spec`` shape-for-shape.

        Gates:

        * ``downsample == "minmax"`` only — ``"nth"``/``"fpcs"`` decide which
          rows survive the filter, which is not decomposable (spec §4), so
          they get no cube and fall back to a server recompute.
        * The y (value) column must be numeric — a ``schema`` is therefore
          required (``None`` ⇒ no target).
        * The x (bucket) column must be numeric or temporal; the engine
          resolves the temporal ``unit`` and applies the ``Datetime("ns")``/
          ``Time`` gate uniformly in ``_resolved_target_dims`` (exactly as for
          histogram targets), so it is not duplicated here.
        * Grouped lines are allowed: every ``group_by`` column must pass the
          same string-dtype + reserved-name gate the histogram uses
          (``_categorical_dims_ok``), else ``None``.

        ``n_buckets = max(1, n_points // 2)`` matches the legacy minmax bucket
        count (two extrema points per bucket). ``domain`` is ``axis_range``
        verbatim (``None`` when unzoomed); the engine epsilon-pads it
        uniformly when resolving binned domains.

        Caveat (no size guard): a grouped line runs the envelope kernel once
        per series, so the cube payload multiplies by the group cardinality —
        accepted under the no-size-guard decision (contract J). Line targets
        are **live-only**: the envelope tracks the brush during a drag, but
        every commit POSTs and the legacy delta replaces it (contract J;
        marked ``postRequired`` on the client, never skipPost).
        """
        if self.downsample != "minmax":
            return None
        y_dtype = _dtype_for_col(schema, self.y_col)
        if y_dtype is None or not y_dtype.is_numeric():
            return None
        group_cols = self.group_by_cols or ()
        if group_cols and not _categorical_dims_ok(schema, group_cols):
            return None
        n_buckets = max(1, self.n_points // 2)
        return CubeTargetSpec(
            target_dims=(
                TargetDimSpec(
                    column=self.x_col,
                    kind="binned",
                    bins=n_buckets,
                    domain=axis_range,
                ),
                *(TargetDimSpec(column=c, kind="categorical") for c in group_cols),
            ),
            measure=MeasureSpec(agg="line_env", value_col=self.y_col),
        )

    # ------------------------------------------------------------------
    # Properties (convenience access)
    # ------------------------------------------------------------------

    @property
    def x_col(self) -> str:
        return self._backend_data["x"]

    @property
    def y_col(self) -> str:
        return self._backend_data["y"]

    @property
    def n_points(self) -> int:
        return self._params["n_points"]

    @property
    def add_gaps(self) -> bool:
        return self._params["add_gaps"]

    @property
    def downsample(self) -> LineDownsample:
        return self._params["downsample"]

    # ------------------------------------------------------------------
    # FlexTrace interface
    # ------------------------------------------------------------------

    def get_aggregation_spec(
        self,
        update_range: Dict[str, Any],
        schema: pl.Schema | None = None,
        x_sorted: bool = False,
        scan_source: bool = False,
    ) -> AggregationSpec | GroupedAggregationSpec:
        """Return either a regular or grouped line aggregation spec.

        ``x_sorted`` is the caller's guarantee that the x column is ascending
        (set by ``assume_sorted_x`` / ``check_sorted``). It only enables a
        faster viewport restriction; correctness of the output is unchanged.

        ``scan_source`` says the rows come from storage rather than a resident
        frame. The kernel needs the whole column in memory, so on a scan the
        minmax envelope switches to a streaming plan that computes the *same*
        points (see ``_native_envelope_plan``). Output is identical either way
        — there is a test that asserts it — so this only picks a formulation.
        """
        x_range = update_range.get("x")

        group_by_cols = self.group_by_cols
        if group_by_cols is not None:
            # Grouped lines restrict the frame *before* grouping, and a
            # frame-level filter is not a slice, so this path keeps the mask.
            vp_expr = (
                _range_filter_expr(self.x_col, x_range, schema=schema)
                if x_range is not None
                else None
            )
            batch_key = (
                self.x_col,
                tuple(x_range) if x_range is not None else None,
            )
            grouped_expr = _plugin_line_agg_expr(
                self.x_col,
                self.y_col,
                None,
                self.n_points,
                self.downsample,
                self.uid,
            )
            return GroupedAggregationSpec(
                uid=self.uid,
                group_cols=group_by_cols,
                sort_cols=group_by_cols,
                agg_exprs=(grouped_expr,),
                pre_group_filters=(vp_expr,) if vp_expr is not None else (),
                pre_group_filter_key=(
                    ("x_range", tuple(x_range)) if x_range is not None else None
                ),
                batch_key=batch_key,
                streaming_safe=False,  # line downsample kernel inside the agg
            )

        if scan_source and self.downsample == "minmax":
            # `nth` already streams (a stride needs no state) and `fpcs` has no
            # streaming formulation at all, so only minmax needs the swap.
            return AggregationSpec(
                expr=pl.lit(None).alias(self.uid),
                uid=self.uid,
                plan=_native_envelope_plan(
                    self.x_col,
                    self.y_col,
                    self.n_points,
                    self.uid,
                    (
                        _range_filter_expr(self.x_col, x_range, schema=schema)
                        if x_range is not None
                        else None
                    ),
                ),
            )

        expr = _plugin_line_agg_expr(
            self.x_col,
            self.y_col,
            _viewport_window(self.x_col, x_range, schema, x_sorted),
            self.n_points,
            self.downsample,
            self.uid,
        )

        return AggregationSpec(expr=expr, uid=self.uid)

    def _to_update(
        self,
        df_agg: pl.DataFrame,
    ) -> TraceResult:
        """Unpack the aggregated + imploded struct column -> ``{"x": series, "y": series}``.

        Gaps (null breaks across large x jumps) are a client-side display
        concern, inserted at render time by ``fvApplyLineGaps``
        (``adapters/js/plotly/traces.js``); the server emits gapless x/y.
        """
        raw: pl.Series = df_agg[self.uid].item()
        df_line = raw.explode().struct.unnest()
        return TraceResult(updates={"x": df_line[self.x_col], "y": df_line[self.y_col]})

    def _to_grouped_update(self, df_grouped: pl.DataFrame) -> TraceResult:
        """Unpack one grouped line result row into one child result per group."""
        group_by_cols = self.group_by_cols
        assert group_by_cols is not None, "Grouped line requires group_by"
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
    def from_trace_spec(cls, spec: TraceSpec) -> "LinePlot":  # type: ignore[override]
        trace = cls(
            x=spec.backend_data.get("x", ""),
            y=spec.backend_data.get("y", ""),
            name=spec.display.get("name"),
            color=spec.display.get("color"),
            color_map=spec.display.get("color_map"),
            n_points=spec.params["n_points"],
            downsample=spec.params["downsample"],
            add_gaps=spec.params["add_gaps"],
            axes=spec.axes or ("x", "y"),
            # Frozen only when the spec carries an explicit empty tuple; ``None``
            # (omitted) or a populated tuple both mean "x-bound as usual".
            update_on_zoom=spec.recompute_axes != (),
            group_by=spec.params.get("group_by"),
        )
        if spec.recompute_axes is not None:
            trace._recompute_axes = tuple(spec.recompute_axes)
        if "group_domain_key" in spec.params:
            trace._params["group_domain_key"] = spec.params["group_domain_key"]
        trace.uid = spec.uid
        return trace

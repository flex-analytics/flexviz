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

from collections.abc import Mapping, Sequence
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
# Out-of-core envelope (streaming, single collect)
# ---------------------------------------------------------------------------


def _streaming_envelope_plan(
    x_col: str,
    y_col: str,
    n_points: int,
    uid: str,
    vp_filter: pl.Expr | None,
    x_range: tuple | None,
    x_domain: tuple | None,
    schema: pl.Schema | None,
):
    """Streaming min-max envelope using equal-width buckets in x.

    Replaces the two-pass ``_native_envelope_plan``. One streaming collect, no
    intermediate collects.

    Buckets partition the x range into ``n_points // 2`` equal-width bins.
    ``min_by``/``max_by`` locate the x value at each y extremum in a single
    associative pass, so the group never buffers.

    When zoomed, the viewport provides the x bounds. When unzoomed, ``x_domain``
    carries the unfiltered ``(min, max)`` the engine resolved, in physical
    units, so a cross-filter cannot move the bucket edges.

    On an exact y plateau, ``min_by`` picks an arbitrary member, and which
    member can vary with ``POLARS_MAX_THREADS``. Any member is a valid envelope
    point. This is a deliberate trade: the previous two-pass plan paid 2.9x
    runtime and 39x memory for deterministic tie-breaking.
    """
    import math

    n_out = max(n_points // 2, 1)
    empty = pl.DataFrame(
        {uid: [[]]},
        schema={uid: pl.List(pl.Struct({x_col: pl.Null, y_col: pl.Null}))},
    )

    def run(filtered_ldf: pl.LazyFrame) -> pl.DataFrame:
        src = filtered_ldf if vp_filter is None else filtered_ldf.filter(vp_filter)
        src = src.select(x_col, y_col)

        dtype = schema.get(x_col) if schema else None
        is_temporal = dtype is not None and dtype.is_temporal()

        # Bucket arithmetic runs on the physical representation so that
        # temporal columns reduce to plain integer division.
        phys_expr = pl.col(x_col).to_physical() if is_temporal else pl.col(x_col)
        if x_range is not None:
            x_lo, x_hi = x_range[0], x_range[1]
            if dtype is not None and dtype.is_integer():
                x_lo = math.ceil(x_lo) if isinstance(x_lo, float) else x_lo
                x_hi = math.floor(x_hi) if isinstance(x_hi, float) else x_hi
            elif is_temporal:
                x_lo, x_hi = int(x_lo), int(x_hi)
        else:
            x_lo, x_hi = x_domain if x_domain is not None else (None, None)

        if x_lo is None or x_hi is None:
            return empty

        span = x_hi - x_lo
        # Float columns need true division: integer ceiling division rounds a
        # sub-1 width up to 1 (0.002 / 500 -> 1), collapsing every row into
        # one bucket. Integer and temporal columns use ceiling division to keep
        # the width whole. The check is on the column dtype, not the Python
        # type of span, because JSON deserializes 100.0 as int.
        use_float_div = dtype is not None and dtype.is_float()
        if span <= 0:
            bsz = 1
        elif use_float_div:
            bsz = span / n_out
        else:
            bsz = -(-span // n_out)

        lo_lit = pl.lit(x_lo)
        bsz_lit = pl.lit(bsz)

        y = pl.col(y_col)
        result = (
            src.group_by(
                # True division lands x_hi exactly on n_out; fold that lone
                # top row back into the last bucket instead of letting it open
                # an n_out + 1-th one and overrun the n_points budget.
                ((phys_expr - lo_lit) // bsz_lit)
                .clip(upper_bound=n_out - 1)
                .alias("__b")
            )
            .agg(
                pl.col([x_col, y_col]).min_by(y).name.prefix("__lo_"),
                pl.col([x_col, y_col]).max_by(y).name.prefix("__hi_"),
            )
            .collect(engine="streaming")
        )

        if result.is_empty():
            return empty

        pts = pl.concat(
            [
                result.select(
                    pl.col(f"__lo_{x_col}").alias("__x"),
                    pl.col(f"__lo_{y_col}").alias("__y"),
                ),
                result.select(
                    pl.col(f"__hi_{x_col}").alias("__x"),
                    pl.col(f"__hi_{y_col}").alias("__y"),
                ),
            ]
        ).drop_nulls("__x")
        pts = pts.unique().sort("__x")

        return pts.select(
            pl.struct(
                **{
                    x_col: pl.col("__x").alias(x_col),
                    y_col: pl.col("__y").alias(y_col),
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

    def domain_cols(
        self, update_range: Dict[str, Any], *, scan_source: bool = False
    ) -> tuple[str, ...]:
        # Only the ungrouped out-of-core minmax envelope bins in x; every other
        # line formulation reads its bucket grid off the rows themselves.
        if (
            not scan_source
            or self.downsample != "minmax"
            or self.group_by_cols is not None
            or update_range.get("x") is not None
        ):
            return ()
        return (self.x_col,)

    def get_aggregation_spec(
        self,
        update_range: Dict[str, Any],
        schema: pl.Schema | None = None,
        x_sorted: bool = False,
        scan_source: bool = False,
        *,
        domains: Mapping[str, tuple[Any, Any]] | None = None,
    ) -> AggregationSpec | GroupedAggregationSpec:
        """Return either a regular or grouped line aggregation spec.

        ``x_sorted`` is the caller's guarantee that the x column is ascending
        (set by ``assume_sorted_x`` / ``check_sorted``). It only enables a
        faster viewport restriction; correctness of the output is unchanged.

        ``scan_source`` says the rows come from storage rather than a resident
        frame. The kernel needs the whole column in memory, so on a scan the
        minmax envelope switches to a streaming plan (``_streaming_envelope_plan``)
        that uses equal-width buckets in x with ``min_by``/``max_by``. The output
        is a valid envelope but not bit-identical to the kernel on exact plateaus.

        An unzoomed streaming-envelope trace requires ``x_col`` in ``domains``;
        a ``(None, None)`` entry means an empty or all-null column.
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
            )

        if scan_source and self.downsample == "minmax":
            # `nth` already streams (a stride needs no state) and `fpcs` has no
            # streaming formulation at all, so only minmax needs the swap.
            return AggregationSpec(
                expr=pl.lit(None).alias(self.uid),
                uid=self.uid,
                plan=_streaming_envelope_plan(
                    self.x_col,
                    self.y_col,
                    self.n_points,
                    self.uid,
                    (
                        _range_filter_expr(self.x_col, x_range, schema=schema)
                        if x_range is not None
                        else None
                    ),
                    x_range,
                    (domains or {})[self.x_col] if x_range is None else None,
                    schema,
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

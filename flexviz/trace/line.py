"""LinePlot — renderer-agnostic line / time-series trace.

Downsamples a sorted series in the current viewport.

The three x-width strategies share one stage 1: one
``(x_min, y_min, x_max, y_max)`` pair per non-empty equal-x-width bucket, on the
one global grid (``_bucket_grid``).  A resident ungrouped line runs the
``minmax_pairs_line`` kernel, every other case ``_pairs_plan``.  ``_to_update``
turns those pairs into points, so the strategies differ only in stage 2.

Downsampling strategies:

* ``"minmax"`` (default) — min-max envelope downsampling: ``n_points // 2``
  buckets, and both extrema of each are kept, so at most ``n_points`` output
  points preserve the extrema and spikes.  A grouped line bins every group on the
  one global grid, so a series covering a tenth of the x domain gets a tenth of
  the points.

* ``"lttb"`` — MinMaxLTTB: the pair pass with a 4x budget, then the
  Largest-Triangle-Three-Buckets rule over the prefetched points.  Returns exactly
  ``n_points`` points when the prefetch holds more.  Smoother than ``"minmax"`` on
  noisy data, at the price of one Python pass over the prefetch.  Grouped, the
  thinning runs once per child.

* ``"fpcs"`` — Feature-Preserving Compensated Sampling: ``n_points - 2`` buckets,
  then a compensation walk that carries a deferred extremum across bucket
  boundaries.  ``n_points`` is a target, not a hard cap: the walk emits up to
  roughly ``2 * n_points`` points, fewer on gappy x.

* ``"nth"`` — uniform stride gather: every ``max(1, n // n_points)``-th row.
  Stride is computed inside the Rust kernel (no Polars ``len()`` expression
  dependency), enabling full parallelism across N grouped sub-traces in a single
  ``select()``.

"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Dict, Literal, get_args

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
    _physical_bound_expr,
    _to_col_tuple,
    _range_filter_expr,
    _typed_range_bounds,
)

import flexviz_polars as _fvp  # noqa: F401 — registers pl.Expr.flexviz namespace

LineDownsample = Literal["minmax", "lttb", "fpcs", "nth"]

# Downsamplers that bucket by x width, grouped or not. They share the x
# contract (see ``LinePlot.buckets_by_x_width``): a dtype the grid can bucket,
# and ungrouped also sorted, finite, null-free x.
_X_WIDTH_DOWNSAMPLES = ("minmax", "lttb", "fpcs")

# LTTB stage 1 prefetch, as a multiple of ``n_points``: four candidates per
# output point. Fewer starves the triangle rule, more only grows the Python
# pass. Not a public knob until someone needs to tune it.
_LTTB_MINMAX_RATIO = 4

# The four fixed struct fields the pair kernel and the pair plan both emit.
_PAIR_FIELDS = ("x_min", "y_min", "x_max", "y_max")

# Group dtypes whose id in a packed group key is their categorical physical.
_STRING_LIKE = (pl.String, pl.Categorical, pl.Enum)

# Two points make a line, and 25k already exceeds the pixel width of any screen
# the browser draws them on.
_N_POINTS_MIN = 2
_N_POINTS_MAX = 25_000

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
# Equal-x-width bucket grid (shared by both minmax formulations)
# ---------------------------------------------------------------------------


def _bucket_grid(
    x_col: str,
    x_range: tuple | None,
    x_domain: tuple | None,
    dtype: pl.DataType | None,
    n_buckets: int,
) -> tuple[Any, Any, Any] | None:
    """``(lo, bucket_width, hi)`` of the equal-x-width grid, in physical units.

    Zoomed, the grid spans the client viewport. Unzoomed it spans ``x_domain``,
    the unfiltered ``(min, max)`` the engine resolved, so a cross-filter cannot
    move the bucket edges. ``None`` means there is no grid: an empty or all-null
    x column, or a temporal viewport bound that failed to parse. An infinite
    bound raises: the edges are found by binary search, and an infinite span has
    no finite bucket width.

    Float columns need true division: integer ceiling division rounds a sub-1
    width up to 1 (0.002 / 500 -> 1), collapsing every row into one bucket.
    Integer and temporal columns divide with a ceiling to keep the width whole.
    The branch is on the column dtype, because the Python type of the span is no
    dtype signal: a JSON viewport bound of ``100`` arrives as an int on a float
    column.
    """
    if x_range is not None:
        lo, hi = x_range[0], x_range[1]
        if dtype is not None and dtype.is_integer():
            lo = math.ceil(lo) if isinstance(lo, float) else lo
            hi = math.floor(hi) if isinstance(hi, float) else hi
        elif dtype is not None and dtype.is_temporal():
            # A viewport bound arrives as a date string or epoch-ms number, so
            # it needs the same parse the typed filters use before it can be
            # read as a physical unit.
            lo, hi = pl.select(
                _physical_bound_expr(lo, dtype).alias("lo"),
                _physical_bound_expr(hi, dtype).alias("hi"),
            ).row(0)
    else:
        lo, hi = x_domain if x_domain is not None else (None, None)

    if lo is None or hi is None:
        return None
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError(
            f"x column '{x_col}' has an infinite bound ({lo}, {hi}). A minmax "
            f"line needs a finite x. Filter the frame with is_finite() first."
        )

    span = hi - lo
    if span <= 0:  # a constant x column still gets one bucket, of width 1
        return (lo, 1, lo + 1)
    if dtype is not None and dtype.is_float():
        return (lo, span / n_buckets, hi)
    return (lo, -(-span // n_buckets), hi)


# ---------------------------------------------------------------------------
# Bucket extrema and the stage-1 plan (streaming, single collect)
# ---------------------------------------------------------------------------


def _grouped_bucket_keys(
    group_cols: tuple[str, ...],
    schema: pl.Schema | None,
    domains: Mapping[str, tuple[Any, Any]],
    n_buckets: int,
    bucket: pl.Expr,
) -> tuple[pl.Expr, list[str], list[pl.Expr]]:
    """The ``group_by`` key of a grouped bucket query.

    Returns ``(key expression aliased "__b", group_by columns, extra
    aggregations)``. Within one group the key is monotone in the bucket, so
    ``__b`` orders the buckets in either form.

    The packed form turns every group column and the bucket into one Int64
    key: one hash column instead of n + 1, measured 1.4x faster on a third of
    the memory at 100M rows. The group values come back through ``first()``.

    Each group column becomes an integer id. A string-like column takes its
    categorical physical, which is below 2^32 but carries no resolved bound, so
    it must lead and only one of them fits. Every other id is ``value - min +
    1`` over the resolved bounds, with null in slot 0.

    The multi-column form ``group_by(*group_cols, bucket)`` is the fallback,
    for a dtype with no id, a second string-like column, a column whose bounds
    were not resolved, or a width product that would overflow the key.
    """
    multi = (bucket.alias("__b"), [*group_cols, "__b"], [])
    lead: pl.Expr | None = None
    factors: list[tuple[pl.Expr, int]] = []
    for col in group_cols:
        dtype = _dtype_for_col(schema, col)
        if dtype is None:
            return multi
        if dtype in _STRING_LIKE:
            if lead is not None:
                return multi
            phys = pl.col(col)
            if dtype == pl.String:
                phys = phys.cast(pl.Categorical)
            # Physicals are >= 0, so -1 is a free slot for a null group value.
            lead = phys.to_physical().cast(pl.Int64).fill_null(-1)
            continue
        if not (dtype.is_integer() or dtype.is_temporal() or dtype == pl.Boolean):
            return multi
        lo, hi = domains.get(col, (None, None))
        if lo is None or hi is None:
            return multi
        value = pl.col(col)
        if dtype.is_temporal():
            value = value.to_physical()
        lo, hi = int(lo), int(hi)
        factors.append(((value.cast(pl.Int64) - lo + 1).fill_null(0), hi - lo + 2))

    width = n_buckets
    for _, w in factors:
        width *= w
    # The leading physical is unbounded below 2^32, so it needs the rest of the
    # key to stay under 2^31. Without one the whole product is the bound.
    if width > (2**31 if lead is not None else 2**62):
        return multi

    ids = factors if lead is None else [(lead, 0), *factors]
    key = ids[0][0]
    for id_expr, w in ids[1:]:
        key = key * w + id_expr
    return (
        (key * n_buckets + bucket.cast(pl.Int64, strict=False)).alias("__b"),
        ["__b"],
        [pl.col(c).first() for c in group_cols],
    )


def _bucket_extrema(
    filtered_ldf: pl.LazyFrame,
    x_col: str,
    y_col: str,
    n_buckets: int,
    vp_filter: pl.Expr | None,
    x_range: tuple | None,
    x_domain: tuple | None,
    schema: pl.Schema | None,
    *,
    group_cols: tuple[str, ...] | None = None,
    domains: Mapping[str, tuple[Any, Any]] | None = None,
) -> pl.DataFrame | None:
    """One row per non-empty equal-x-width bucket, in group-by order.

    Columns: ``__b``, ``__lo_<x>``, ``__lo_<y>``, ``__hi_<x>``, ``__hi_<y>``,
    plus the group columns when ``group_cols`` is given: every group then bins
    on the one global grid. ``None`` means there is no grid (an empty or
    all-null x column). ``_pairs_plan`` orders the rows by bucket.

    One streaming collect, no intermediate collects. ``min_by``/``max_by``
    locate the x value at each y extremum in a single associative pass, so the
    group never buffers.

    When zoomed, the viewport provides the x bounds. When unzoomed, ``x_domain``
    carries the unfiltered ``(min, max)`` the engine resolved, in physical
    units, so a cross-filter cannot move the bucket edges.

    On an exact y plateau, ``min_by`` picks an arbitrary member, and which
    member can vary with ``POLARS_MAX_THREADS``. The kernel picks a member of
    its own, so the two formulations can differ there. Any member is a valid
    envelope point, and a pair is consumed as a pair. This is a deliberate
    trade: deterministic tie-breaking here would cost a second pass and far
    more memory for no visible difference.
    """
    src = filtered_ldf if vp_filter is None else filtered_ldf.filter(vp_filter)
    src = src.select(list(dict.fromkeys((*(group_cols or ()), x_col, y_col))))

    dtype = schema.get(x_col) if schema else None

    # Bucket arithmetic runs on the physical representation so that temporal
    # columns reduce to plain integer division.
    phys_expr = (
        pl.col(x_col).to_physical()
        if dtype is not None and dtype.is_temporal()
        else pl.col(x_col)
    )
    grid = _bucket_grid(x_col, x_range, x_domain, dtype, n_buckets)
    if grid is None:
        return None
    x_lo, bsz, _ = grid

    # A fractional width multiplies by the reciprocal instead of dividing:
    # Polars lowers a float `//` to that same multiply, and spelling it out is
    # what lets the kernel copy the arithmetic bit for bit. A whole width keeps
    # integer division, which is exact past 2**53.
    offset = phys_expr - pl.lit(x_lo)
    raw = (
        (offset * pl.lit(1.0 / bsz)).floor()
        if isinstance(bsz, float)
        else offset // pl.lit(bsz)
    )
    # The Int64 cast is what drops a null or NaN x: neither has a bucket, and
    # both become a null key that the filter removes. A row at x_hi lands on
    # n_buckets; the clip folds it back into the last bucket instead of letting
    # it open an n_buckets + 1-th one and overrun the budget.
    bucket = (
        raw.cast(pl.Int64, strict=False).clip(upper_bound=n_buckets - 1).alias("__b")
    )
    src = src.with_columns(bucket).filter(pl.col("__b").is_not_null())

    y = pl.col(y_col)
    extrema = (
        pl.col([x_col, y_col]).min_by(y).name.prefix("__lo_"),
        pl.col([x_col, y_col]).max_by(y).name.prefix("__hi_"),
    )
    if group_cols is None:
        return src.group_by("__b").agg(*extrema).collect(engine="streaming")

    key, by, group_aggs = _grouped_bucket_keys(
        group_cols, schema, domains or {}, n_buckets, pl.col("__b")
    )
    return (
        src.with_columns(key)
        .group_by(by)
        .agg(*group_aggs, *extrema)
        .collect(engine="streaming")
    )


def _no_groups(
    group_cols: tuple[str, ...],
    schema: pl.Schema | None,
    uid: str,
) -> pl.DataFrame:
    """The zero-row shape of a grouped plan: no group, so no child."""
    return pl.DataFrame(
        schema={
            **{c: (_dtype_for_col(schema, c) or pl.Null) for c in group_cols},
            uid: pl.List(pl.Struct({f: pl.Null for f in _PAIR_FIELDS})),
        }
    )


def _pairs_plan(
    x_col: str,
    y_col: str,
    n_buckets: int,
    uid: str,
    vp_filter: pl.Expr | None,
    x_range: tuple | None,
    x_domain: tuple | None,
    schema: pl.Schema | None,
    *,
    group_cols: tuple[str, ...] | None = None,
    domains: Mapping[str, tuple[Any, Any]] | None = None,
):
    """Stage 1 of every x-width strategy: the bucket extrema kept as pairs.

    Ungrouped it returns one row holding every pair in bucket order. Grouped,
    one row per group, sorted by group value, each holding that group's pairs
    in bucket order. Every group bins on the same grid as an ungrouped line
    over the same x, so a series covering a tenth of the x domain gets a tenth
    of the points.

    The pair is the unit ``_to_update`` consumes, so the four columns stay on
    one row and are never deduplicated here.
    """
    pair_exprs = (
        pl.col(f"__lo_{x_col}").alias("x_min"),
        pl.col(f"__lo_{y_col}").alias("y_min"),
        pl.col(f"__hi_{x_col}").alias("x_max"),
        pl.col(f"__hi_{y_col}").alias("y_max"),
    )

    def run(filtered_ldf: pl.LazyFrame) -> pl.DataFrame:
        result = _bucket_extrema(
            filtered_ldf,
            x_col,
            y_col,
            n_buckets,
            vp_filter,
            x_range,
            x_domain,
            schema,
            group_cols=group_cols,
            domains=domains,
        )
        if group_cols is not None:
            if result is None:
                return _no_groups(group_cols, schema, uid)
            cols = list(group_cols)
            # The drop is scoped to the pair so a null group value keeps its
            # own child.
            pairs = result.select(*cols, "__b", *pair_exprs).drop_nulls(_PAIR_FIELDS)
            # Sorted after the implode: the packed key orders by categorical
            # physical, not by group value.
            return (
                pairs.group_by(cols)
                .agg(pl.struct(*_PAIR_FIELDS).sort_by("__b").alias(uid))
                .sort(cols)
            )

        if result is None:
            return pl.DataFrame(
                {uid: [[]]},
                schema={uid: pl.List(pl.Struct({f: pl.Null for f in _PAIR_FIELDS}))},
            )
        # An empty frame implodes to one row holding a typed empty list, which
        # is the sentinel shape, so no separate empty branch is needed.
        return (
            result.sort("__b")
            .select(*pair_exprs)
            .drop_nulls()
            .select(pl.struct(*_PAIR_FIELDS).implode().alias(uid))
        )

    return run


# ---------------------------------------------------------------------------
# Stage 2: pure-Python thinning of the bucket output
# ---------------------------------------------------------------------------


def _lttb(x: list, y: list, n_out: int) -> tuple[list, list]:
    """Largest-Triangle-Three-Buckets over ``n_out`` points.

    Stage 2 of MinMaxLTTB. ``_to_update`` drops null and NaN rows first.

    ``x`` is shifted by ``x[0]`` before the area math: a nanosecond epoch loses
    precision in float64, a difference of one viewport does not. No other
    normalization is needed, because scaling x scales every area by the same
    factor and leaves the argmax unchanged.
    """
    n = len(x)
    if n <= n_out:
        return x, y
    if n_out < 3:  # no interior buckets to pick from
        return [x[0], x[-1]], [y[0], y[-1]]

    x0 = x[0]
    xs = [float(v - x0) for v in x]
    out_x, out_y = [x[0]], [y[0]]

    every = (n - 2) / (n_out - 2)
    a = 0
    for i in range(n_out - 2):
        # The mean of the next bucket. The last bucket of all reduces to the
        # final point, which is the anchor the paper uses there.
        lo = int((i + 1) * every) + 1
        hi = min(int((i + 2) * every) + 1, n)
        span = hi - lo
        avg_x = sum(xs[lo:hi]) / span
        avg_y = sum(y[lo:hi]) / span

        ax, ay = xs[a], y[a]
        best_area = -1.0
        for j in range(int(i * every) + 1, lo):
            area = abs((ax - avg_x) * (y[j] - ay) - (ax - xs[j]) * (avg_y - ay))
            if area > best_area:
                best_area, a = area, j
        out_x.append(x[a])
        out_y.append(y[a])

    out_x.append(x[-1])
    out_y.append(y[-1])
    return out_x, out_y


def _fpcs_walk(x_min: list, y_min: list, x_max: list, y_max: list) -> tuple[list, list]:
    """Feature-Preserving Compensated Sampling over per-bucket ``(min, max)`` pairs.

    Stage 2 of FPCS. Each bucket emits the earlier of its two extrema and
    defers the later one. When the next bucket emits the same kind of
    extremum again, the deferred point goes out first, so the turn between
    them is not lost. Earlier means smaller x.

    The comparisons are inverted (``not (a > b)``) on purpose: a NaN compares
    false against everything, so it always takes the extremum, which is the
    NaN-propagating policy the bucket kernel already applies.

    Unlike a row-index formulation this keeps no first and last point: x-width
    buckets have no interior. Output is ordered by x and holds at most
    ``2 * len(x_min) + 1`` points.
    """
    out_x: list = []
    out_y: list = []

    def push(point):
        # A deferred point is often the extremum the next bucket re-emits. On
        # sorted x the same x is the same row, and comparing y as well would
        # let a NaN through twice.
        if not out_x or out_x[-1] != point[0]:
            out_x.append(point[0])
            out_y.append(point[1])

    if not x_min:
        return out_x, out_y

    first = (x_min[0], y_min[0]) if x_min[0] <= x_max[0] else (x_max[0], y_max[0])
    min_point = max_point = potential_point = first
    previous_flag: str | None = None

    for i in range(len(x_min)):
        lo_point = (x_min[i], y_min[i])
        hi_point = (x_max[i], y_max[i])
        if not (max_point[1] > hi_point[1]):
            max_point = hi_point
        if not (min_point[1] <= lo_point[1]):
            min_point = lo_point

        if min_point[0] < max_point[0]:
            if previous_flag == "min" and min_point[0] != potential_point[0]:
                push(potential_point)
            push(min_point)
            potential_point = min_point = max_point
            previous_flag = "min"
        else:
            if previous_flag == "max" and max_point[0] != potential_point[0]:
                push(potential_point)
            push(max_point)
            potential_point = max_point = min_point
            previous_flag = "max"

    # The last bucket defers a point too; without this it would be dropped.
    if previous_flag is not None:
        push(potential_point)
    return out_x, out_y


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


def _plugin_pairs_agg_expr(
    x_col: str,
    y_col: str,
    vp: Viewport | None,
    n_buckets: int,
    uid: str,
    x_domain: tuple[Any, Any],
) -> pl.Expr:
    """Stage 1 on a resident frame: one pair per bucket, from the Rust kernel.

    ``x_domain`` is the ``(lo, hi)`` the buckets span. The kernel rebuilds the
    same equal-x-width grid ``_bucket_grid`` holds over it, which needs x sorted
    ascending, and drops rows outside ``[lo, hi]``. It reads the bucket of a
    row with the arithmetic ``_bucket_extrema`` uses, so the two formulations
    place a row at a bucket edge in the same bucket.

    Bucket selection and all four gathers happen in one kernel call: Polars does
    not CSE opaque plugin expressions, so a two-gather form would run the whole
    argmin/argmax scan twice.
    """
    y_expr = _apply_viewport(pl.col(y_col), vp)
    x_expr = _apply_viewport(pl.col(x_col), vp)
    return (
        _fvp._minmax_pairs_line(x_expr, y_expr, n_buckets, x_domain=x_domain)
        .implode()
        .alias(uid)
    )


def _bucket_budget(n_points: int, downsample: LineDownsample) -> int:
    """How many equal-x-width buckets one series gets.

    The only budget formula: the kernel and the plan both take a bucket count,
    the same one grouped and ungrouped, so every line lands on the same edges.
    ``minmax`` keeps both extrema of a bucket, so half the points are buckets.
    ``lttb`` prefetches ``_LTTB_MINMAX_RATIO`` times that. The ``fpcs`` count is
    kept from the row-index formulation, so its output stays between about
    ``n_points`` and ``2 * n_points``.
    """
    if downsample == "fpcs":
        return max(n_points - 2, 1)
    if downsample == "lttb":
        return max(n_points * _LTTB_MINMAX_RATIO // 2, 1)
    return max(n_points // 2, 1)


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
        Column name for the x-axis values.  Must be sorted ascending for an
        ungrouped ``"minmax"``, ``"lttb"`` or ``"fpcs"`` line: their buckets are
        equal in x width, so the engine binary-searches the bucket edges.
        Sorted x also turns every viewport into a zero-copy slice.  A grouped
        line reads x in no order and needs only a dtype the grid can bucket.
    y:
        Column name for the y-axis values.
    name:
        Legend / series name passed to the renderer.
    color:
        Line colour hint (CSS string), passed to the renderer.
    n_points:
        Target number of points returned per viewport after downsampling,
        between 2 and 25000.
        ``"minmax"`` and ``"nth"`` return at most this many points.  ``"lttb"``
        returns exactly this many when the prefetch holds more, and fewer when
        x gaps leave buckets empty.  ``"fpcs"`` follows FPCS target semantics
        and can return up to roughly ``2 * n_points`` points.
    downsample:
        Downsampling strategy.  ``"minmax"`` (default) uses the min-max
        envelope algorithm; ``"lttb"`` thins a 4x min-max prefetch with the
        MinMaxLTTB triangle rule; ``"fpcs"`` uses
        Feature-Preserving Compensated Sampling; ``"nth"`` selects every nth
        row.  All of them need the ``flexviz_polars`` plugin.  The first three
        bucket by x width, and grouped they share one grid across the groups,
        so the budget follows the x span of each series.
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
        # The trust boundary: the client posts the spec on every update, and a
        # decoded spec builds the trace through here as well.
        if not _N_POINTS_MIN <= n_points <= _N_POINTS_MAX:
            raise ValueError(
                f"n_points must be between {_N_POINTS_MIN} and {_N_POINTS_MAX}, "
                f"got {n_points}."
            )
        if downsample not in get_args(LineDownsample):
            raise ValueError(
                f"downsample must be one of {get_args(LineDownsample)}, "
                f"got {downsample!r}."
            )
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

        * ``downsample == "minmax"`` only — every other strategy decides which
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

    @property
    def buckets_by_x_width(self) -> bool:
        """Whether the buckets are equal in x width, which is the x contract.

        Grouped or not, the x-width strategies share it. ``nth`` gathers a
        stride and carries no grid.
        """
        return self.downsample in _X_WIDTH_DOWNSAMPLES

    def check_schema(self, schema: pl.Schema) -> None:
        """Raise when the schema cannot feed this line. Reads dtypes, never collects."""
        if self.y_col not in schema:
            raise ValueError(f"Column '{self.y_col}' not in schema")
        if self.downsample != "lttb":
            return
        # The triangle rule does arithmetic on y.
        y_dtype = schema[self.y_col]
        if not (y_dtype.is_numeric() or y_dtype.is_temporal() or y_dtype == pl.Boolean):
            raise ValueError(
                f"y column '{self.y_col}' must be numeric, temporal or Boolean "
                f"for an lttb line, got {y_dtype}. Cast the column "
                f"first, or use downsample='minmax'."
            )

    def domain_cols(
        self,
        update_range: Dict[str, Any],
        *,
        schema: pl.Schema | None = None,
        scan_source: bool = False,
    ) -> tuple[str, ...]:
        # Every x-width line bins in x, on both source kinds. A zoomed one
        # takes its grid from the viewport, and ``nth`` needs no grid at all.
        if not self.buckets_by_x_width:
            return ()
        cols = () if update_range.get("x") is not None else (self.x_col,)
        group_cols = self.group_by_cols
        if group_cols is None:
            return cols
        # The packed group key places each id over its own (min, max), so a
        # grouped line needs those bounds on every request, zoomed too. A
        # string-like id needs none, and any other dtype falls back to the
        # multi-column key.
        return tuple(
            dict.fromkeys(
                cols
                + tuple(
                    c
                    for c in group_cols
                    if (dtype := _dtype_for_col(schema, c)) is not None
                    and (
                        dtype.is_integer() or dtype.is_temporal() or dtype == pl.Boolean
                    )
                )
            )
        )

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
        (set by ``assume_sorted_x`` / ``check_line_x``). It only enables a
        faster viewport restriction. The output is unchanged.
        The ungrouped x-width strategies need it for a second reason: their
        buckets are equal in x width, which the engine validates once per
        source.

        ``scan_source`` says the rows come from storage rather than a resident
        frame. The kernel needs the whole column in memory, so on a scan every
        ungrouped x-width strategy switches to ``_pairs_plan``, which builds the
        same equal-x-width grid (``_bucket_grid``) as the kernel and reads a
        bucket with the same floor division. So both source kinds return the
        same ``y`` multiset and differ only in which member of an exact ``y``
        plateau they pick. ``fpcs`` can emit a different point set
        on such a plateau, because the walk orders each pair by x.

        A grouped x-width line takes ``_pairs_plan`` on both source kinds, on the
        same grid as an ungrouped line over the same x. Its plateau tie-break is
        the arbitrary one of the ungrouped scan plan. Grouped ``nth`` keeps its
        kernel inside the ``group_by``.

        An unzoomed x-width trace requires ``x_col`` in ``domains``, and a
        grouped one the bounds of its packed group columns. A ``(None, None)``
        entry means an empty or all-null column.
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
            spec = dict(
                uid=self.uid,
                group_cols=group_by_cols,
                sort_cols=group_by_cols,
                pre_group_filters=(vp_expr,) if vp_expr is not None else (),
                pre_group_filter_key=(
                    ("x_range", tuple(x_range)) if x_range is not None else None
                ),
                batch_key=batch_key,
            )
            if self.buckets_by_x_width:
                # One plan per grouped line, on both source kinds: the kernel
                # would hold every column of every group in memory at once.
                return GroupedAggregationSpec(
                    **spec,
                    agg_exprs=(),
                    plan=_pairs_plan(
                        self.x_col,
                        self.y_col,
                        _bucket_budget(self.n_points, self.downsample),
                        self.uid,
                        None,
                        x_range,
                        None if x_range is not None else (domains or {})[self.x_col],
                        schema,
                        group_cols=group_by_cols,
                        domains=domains or {},
                    ),
                )
            return GroupedAggregationSpec(
                **spec,
                agg_exprs=(
                    _plugin_nth_agg_expr(
                        self.x_col, self.y_col, None, self.n_points, self.uid
                    ),
                ),
            )

        if self.buckets_by_x_width:
            n_buckets = _bucket_budget(self.n_points, self.downsample)
            x_domain = None if x_range is not None else (domains or {})[self.x_col]

            if scan_source:
                # The kernel needs the whole column in memory, so a scan runs
                # the streaming plan instead. `nth` already streams: a stride
                # needs no state.
                return AggregationSpec(
                    expr=pl.lit(None).alias(self.uid),
                    uid=self.uid,
                    plan=_pairs_plan(
                        self.x_col,
                        self.y_col,
                        n_buckets,
                        self.uid,
                        (
                            _range_filter_expr(self.x_col, x_range, schema=schema)
                            if x_range is not None
                            else None
                        ),
                        x_range,
                        x_domain,
                        schema,
                    ),
                )

            x_dtype = schema.get(self.x_col) if schema else None
            grid = _bucket_grid(self.x_col, x_range, x_domain, x_dtype, n_buckets)
            # The kernel rebuilds the same width from ``(lo, hi)`` and reads a
            # bucket with the same floor division, so grid and kernel never
            # disagree. It drops rows outside ``[lo, hi]``, so the viewport
            # slice agrees with the grid. A ``None`` grid (empty or all-null x)
            # becomes a zero-span sentinel: the kernel returns no points.
            kernel_domain = (grid[0], grid[2]) if grid is not None else (0.0, 0.0)
            return AggregationSpec(
                expr=_plugin_pairs_agg_expr(
                    self.x_col,
                    self.y_col,
                    _viewport_window(self.x_col, x_range, schema, x_sorted),
                    n_buckets,
                    self.uid,
                    kernel_domain,
                ),
                uid=self.uid,
            )

        expr = _plugin_nth_agg_expr(
            self.x_col,
            self.y_col,
            _viewport_window(self.x_col, x_range, schema, x_sorted),
            self.n_points,
            self.uid,
        )

        return AggregationSpec(expr=expr, uid=self.uid)

    def _to_update(
        self,
        df_agg: pl.DataFrame,
    ) -> TraceResult:
        """Turn the aggregated struct column into ``{"x": series, "y": series}``.

        The one place a line's points are produced. Every x-width strategy gets
        the same stage 1, one pair per bucket, and picks its points from it here:
        ``minmax`` and ``lttb`` flatten the pairs, ``fpcs`` walks them. ``lttb``
        and ``fpcs`` then run their stage 2 in Python, so their values go through
        their physical representation and are cast back after.

        Gaps (null breaks across large x jumps) are a client-side display
        concern, inserted at render time by ``fvApplyLineGaps``
        (``adapters/js/plotly/traces.js``); the server emits gapless x/y.
        """
        raw: pl.Series = df_agg[self.uid].item()
        df_line = raw.explode().struct.unnest()
        if self.downsample == "nth":
            return TraceResult(
                updates={"x": df_line[self.x_col], "y": df_line[self.y_col]}
            )

        # A bucket whose y is all null or all NaN still emits that value as its
        # extremum, and a line skips a null or NaN point on every path.
        df_line = df_line.drop_nulls()
        if any(dtype.is_float() for dtype in df_line.dtypes):
            df_line = df_line.filter(
                pl.all_horizontal(pl.col(pl.Float32, pl.Float64).is_not_nan())
            )

        if self.downsample == "fpcs":
            x_src, y_src = df_line["x_min"], df_line["y_min"]
            x_phys, y_phys = x_src.to_physical(), y_src
            xs, ys = _fpcs_walk(
                x_phys.to_list(),
                y_phys.to_list(),
                df_line["x_max"].to_physical().to_list(),
                df_line["y_max"].to_list(),
            )
            return TraceResult(
                updates={
                    "x": pl.Series(self.x_col, xs, dtype=x_phys.dtype).cast(
                        x_src.dtype
                    ),
                    "y": pl.Series(self.y_col, ys, dtype=y_phys.dtype).cast(
                        y_src.dtype
                    ),
                }
            )

        # Flatten every pair to its two points. At most two rows per bucket, so
        # this stays in Polars on a frame the size of the budget.
        points = (
            pl.concat(
                [
                    df_line.select(
                        pl.col("x_min").alias(self.x_col),
                        pl.col("y_min").alias(self.y_col),
                    ),
                    df_line.select(
                        pl.col("x_max").alias(self.x_col),
                        pl.col("y_max").alias(self.y_col),
                    ),
                ]
            )
            .unique()
            .sort(self.x_col)
        )
        x_src, y_src = points[self.x_col], points[self.y_col]
        if self.downsample == "minmax":
            return TraceResult(updates={"x": x_src, "y": y_src})

        # LTTB does arithmetic on y as well as on x, so a temporal y also goes
        # through its physical representation and is cast back.
        x_phys, y_phys = x_src.to_physical(), y_src.to_physical()
        xs, ys = _lttb(x_phys.to_list(), y_phys.to_list(), self.n_points)
        return TraceResult(
            updates={
                "x": pl.Series(self.x_col, xs, dtype=x_phys.dtype).cast(x_src.dtype),
                "y": pl.Series(self.y_col, ys, dtype=y_phys.dtype).cast(y_src.dtype),
            }
        )

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

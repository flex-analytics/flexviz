"""Stage 1 of an x-width line: the bucket grid and the bucket plan.

The one equal-x-width grid builder (``bucket_grid``) and the one stage-1 plan
(``pairs_plan``) every x-width line uses when it does not run the Rust kernel.
The kernel path in ``line.py`` builds the same grid over the same bounds.

Naming: what ``line.py`` imports is public, the rest keeps a leading underscore.
"""

from __future__ import annotations

import math
from typing import Any

import polars as pl

from .base import _dtype_for_col, _physical_bound_expr

# The four fixed struct fields the pair kernel and the pair plan both emit.
_PAIR_FIELDS = ("x_min", "y_min", "x_max", "y_max")

# Every column the plan adds carries this prefix, so a user column can only
# collide by using the prefix itself. The plan groups by the user's columns
# next to its own, and a name clash there is a DuplicateError, not a wrong
# answer that a caller could notice.
_ALIAS_PREFIX = "__fv_"
_BUCKET = _ALIAS_PREFIX + "b"

# Group dtypes whose id in a packed group key is their categorical physical.
_STRING_LIKE = (pl.String, pl.Categorical, pl.Enum)


# ---------------------------------------------------------------------------
# Equal-x-width bucket grid (shared by both minmax formulations)
# ---------------------------------------------------------------------------


def bucket_grid(
    x_col: str,
    x_range: tuple | None,
    x_domain: tuple | None,
    dtype: pl.DataType | None,
) -> tuple[Any, Any]:
    """``(lo, hi)`` bounds of the equal-x-width grid, in physical units.

    Zoomed, the grid spans the client viewport. Unzoomed it spans ``x_domain``,
    the unfiltered ``(min, max)`` the engine resolved, so a cross-filter cannot
    move the bucket edges. With no bounds (an empty or all-null x column, an
    all-NaN one, or a temporal viewport bound that failed to parse) the span is
    one unit typed for the dtype, and both formulations fall out empty. A span
    that is not finite raises: the edges are found by binary search, and there
    is no finite bucket width to search over.
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

    # A NaN bound is no bound: an all-NaN x resolves to ``(nan, nan)``, and a
    # NaN x row has no bucket on either formulation, so the line falls out
    # empty rather than raising.
    if lo is None or hi is None or lo != lo or hi != hi:
        integral = dtype is not None and (dtype.is_integer() or dtype.is_temporal())
        return (0, 1) if integral else (0.0, 1.0)
    # The span, not each bound: two finite bounds can still differ by infinity.
    if not math.isfinite(hi - lo):
        raise ValueError(
            f"x column '{x_col}' has no finite span ({lo}, {hi}). A minmax "
            f"line needs a finite x. Filter the frame with is_finite() first."
        )

    if hi - lo <= 0:  # a constant x column still gets one bucket
        return (lo, lo + 1)
    return (lo, hi)


# ---------------------------------------------------------------------------
# Bucket extrema and the stage-1 plan (streaming, single collect)
# ---------------------------------------------------------------------------


def _grouped_bucket_keys(
    group_cols: tuple[str, ...],
    schema: pl.Schema | None,
    n_buckets: int,
    bucket: pl.Expr,
) -> tuple[pl.Expr, list[str], list[pl.Expr]]:
    """The ``group_by`` key of a grouped bucket query.

    Returns ``(key expression aliased ``_BUCKET``, group_by columns, extra
    aggregations)``. Within one group the key is monotone in the bucket, so
    the key orders the buckets in either form.

    One string-like group column packs with the bucket into a single Int64
    key, so the ``group_by`` hashes one column instead of two. It packs
    because its categorical physical is a small id the frame already carries,
    needing no resolved bounds. The group value comes back through
    ``first()``. Every other shape groups by the columns themselves.
    """
    dtype = _dtype_for_col(schema, group_cols[0]) if len(group_cols) == 1 else None
    if dtype not in _STRING_LIKE:
        return (bucket.alias(_BUCKET), [*group_cols, _BUCKET], [])

    phys = pl.col(group_cols[0])
    if dtype == pl.String:
        phys = phys.cast(pl.Categorical)
    # Physicals are >= 0, so -1 is a free slot for a null group value.
    key = phys.to_physical().cast(pl.Int64).fill_null(-1)
    return (
        (key * n_buckets + bucket).alias(_BUCKET),
        [_BUCKET],
        [pl.col(group_cols[0]).first()],
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
) -> pl.DataFrame:
    """One row per non-empty equal-x-width bucket, in group-by order.

    Columns: ``__fv_b``, ``__fv_lo_<x>``, ``__fv_lo_<y>``, ``__fv_hi_<x>``,
    ``__fv_hi_<y>``, plus the group columns when ``group_cols`` is given: every
    group then bins on the one global grid. ``pairs_plan`` orders the rows by
    bucket.

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
    x_lo, x_hi = bucket_grid(x_col, x_range, x_domain, dtype)
    # Float columns need true division: integer ceiling division rounds a sub-1
    # width up to 1 (0.002 / 500 -> 1), collapsing every row into one bucket.
    # Integer and temporal columns divide with a ceiling to keep the width
    # whole. The branch is on the column dtype, because the Python type of the
    # span is no dtype signal: a JSON viewport bound of ``100`` arrives as an
    # int on a float column.
    span = x_hi - x_lo
    bsz = (
        span / n_buckets
        if dtype is not None and dtype.is_float()
        else -(-span // n_buckets)
    )

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
        raw.cast(pl.Int64, strict=False).clip(upper_bound=n_buckets - 1).alias(_BUCKET)
    )
    src = src.with_columns(bucket).filter(pl.col(_BUCKET).is_not_null())

    y = pl.col(y_col)
    extrema = (
        pl.col([x_col, y_col]).min_by(y).name.prefix(_ALIAS_PREFIX + "lo_"),
        pl.col([x_col, y_col]).max_by(y).name.prefix(_ALIAS_PREFIX + "hi_"),
    )
    if group_cols is None:
        return src.group_by(_BUCKET).agg(*extrema).collect(engine="streaming")

    key, by, group_aggs = _grouped_bucket_keys(
        group_cols, schema, n_buckets, pl.col(_BUCKET)
    )
    return (
        src.with_columns(key)
        .group_by(by)
        .agg(*group_aggs, *extrema)
        .collect(engine="streaming")
    )


def pairs_plan(
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
):
    """Stage 1 of every x-width strategy: the bucket extrema kept as pairs.

    Ungrouped it returns one row holding every pair in bucket order. Grouped,
    one row per group, sorted by group value, each holding that group's pairs
    in bucket order. Every group bins on the same grid as an ungrouped line
    over the same x, so a series covering a tenth of the x domain gets a tenth
    of the points.

    The pair is the unit ``_to_update`` consumes, so the four columns stay on
    one row and are never deduplicated here.

    The pair names live only as struct fields: grouped, the frame still carries
    the user's group columns, and a top-level ``x_min`` beside a group column of
    that name would collide.
    """
    pair_cols = (
        f"{_ALIAS_PREFIX}lo_{x_col}",
        f"{_ALIAS_PREFIX}lo_{y_col}",
        f"{_ALIAS_PREFIX}hi_{x_col}",
        f"{_ALIAS_PREFIX}hi_{y_col}",
    )
    pair_struct = pl.struct(
        *(pl.col(c).alias(f) for c, f in zip(pair_cols, _PAIR_FIELDS))
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
        )
        if group_cols is not None:
            cols = list(group_cols)
            # The drop is scoped to the pair so a null group value keeps its
            # own child.
            pairs = result.drop_nulls(pair_cols)
            # Sorted after the implode: the packed key orders by categorical
            # physical, not by group value.
            return (
                pairs.group_by(cols)
                .agg(pair_struct.sort_by(_BUCKET).alias(uid))
                .sort(cols)
            )

        # An empty frame implodes to one row holding a typed empty list, which
        # is the sentinel shape, so no separate empty branch is needed.
        return (
            result.drop_nulls(pair_cols)
            .sort(_BUCKET)
            .select(pair_struct.implode().alias(uid))
        )

    return run

"""Shared helpers for the histogram traces.

Histogram, Histogram2D, GeoHistogram2D and CorrHeatmap all bin through here.

Holds the histfunc/histnorm constants and type aliases, the ``apply_histnorm``
normalization function, the color-scale/color-range validation helpers, and the
2D histogram binning itself: bin-edge resolution, the kernel expressions, and
the streamed batch fold a scan source runs instead. ``Histogram2D`` and
``GeoHistogram2D`` bin the same way over different column pairs, so both drive
this one path; ``Histogram`` shares the batch fold and the lattice snap.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any, Callable, Literal

import numpy as np
import polars as pl

from ..LF import AggregationSpec
from .base import (
    _dtype_for_col,
    _physical_bound_expr,
    _temporal_dtype_for_col,
    _typed_range_bounds,
)

import flexviz_polars  # noqa: F401 — registers pl.Expr.flexviz namespace

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_AGG_FUNCTIONS: dict[str, Any] = {
    "sum": lambda col: pl.col(col).sum(),
    "mean": lambda col: pl.col(col).mean(),
    "median": lambda col: pl.col(col).median(),
    "min": lambda col: pl.col(col).min(),
    "max": lambda col: pl.col(col).max(),
    "n_unique": lambda col: pl.col(col).n_unique(),
}

_HISTFUNC_OPTIONS = tuple(_AGG_FUNCTIONS)
_HISTNORM_OPTIONS = (
    None,  # no normalization
    "percent",  # count / total * 100
    "probability",  # count / total
    "density",  # count / bin_width
    "probability density",  # count / (total * bin_width)
)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

HeatmapColorRange = tuple[float, float] | Literal["auto"]

# ---------------------------------------------------------------------------
# Shared color-normalization helpers (used by all heatmap trace classes)
# ---------------------------------------------------------------------------


def normalize_heatmap_color_scale(
    color_scale: str | None,
    default: str,
    *,
    trace_name: str,
) -> str:
    """Validate and normalise a ``color_scale`` value.

    Returns *default* when *color_scale* is ``None``; otherwise validates that
    it is a non-empty string and returns it unchanged.
    """
    if color_scale is None:
        return default
    if not isinstance(color_scale, str) or not color_scale:
        raise TypeError(f"{trace_name} color_scale must be a non-empty string")
    return color_scale


def normalize_heatmap_color_range(
    color_range: Any,
    default: HeatmapColorRange,
    *,
    trace_name: str,
) -> HeatmapColorRange:
    """Validate and normalise a ``color_range`` value.

    Accepts ``None`` (→ *default*), ``"auto"``, or a two-element numeric
    sequence ``(min, max)`` with ``min < max``.
    """
    if color_range is None:
        return default
    if color_range == "auto":
        return "auto"
    if isinstance(color_range, (list, tuple)) and len(color_range) == 2:
        lo = float(color_range[0])
        hi = float(color_range[1])
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise ValueError(f"{trace_name} color_range values must be finite numbers")
        if lo >= hi:
            raise ValueError(f"{trace_name} color_range must satisfy min < max")
        return (lo, hi)
    raise TypeError(
        f"{trace_name} color_range must be 'auto' or a (min, max) numeric tuple"
    )


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def apply_histnorm(
    df: pl.DataFrame,
    value_col: str,
    histnorm: str,
    bin_area: float | None = None,
) -> pl.DataFrame:
    """Apply *histnorm* normalization to *value_col* in *df*.

    For ``"density"`` and ``"probability density"`` the caller must supply
    *bin_area* (``bin_width_a * bin_width_b``).
    """
    if histnorm is None:
        return df
    if histnorm == "percent":
        return df.with_columns(pl.col(value_col) / pl.col(value_col).sum() * 100)
    if histnorm == "probability":
        return df.with_columns(pl.col(value_col) / pl.col(value_col).sum())
    if histnorm == "density":
        assert bin_area is not None and bin_area > 0
        return df.with_columns(pl.col(value_col) / bin_area)
    if histnorm == "probability density":
        assert bin_area is not None and bin_area > 0
        total = df[value_col].sum()
        return df.with_columns(pl.col(value_col) / (total * bin_area))
    raise ValueError(f"Unknown histnorm: {histnorm!r}")


# ---------------------------------------------------------------------------
# 2D binning: bin edges, kernel expressions, batch fold
# ---------------------------------------------------------------------------


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


#: The (x_lo, x_hi, y_lo, y_hi) bin-edge literal expressions from _hist2d_bounds.
_Edges = tuple[pl.Expr, pl.Expr, pl.Expr, pl.Expr]


def _snap_range(lo: float, hi: float, n: int) -> tuple[float, float, int]:
    """Snap ``[lo, hi]`` outward to the lattice of bin width ``(hi - lo) / n``.

    A pan keeps the span, so the width and the lattice stay fixed and every cell
    keeps its place: the grid stands still under the data instead of sliding
    with the viewport. The snapped range covers the viewport, so it holds ``n``
    or ``n + 1`` bins. The epsilon keeps a bound that is a lattice multiple up
    to float error from buying an extra bin. A degenerate span has no lattice,
    so it is returned unchanged.
    """
    width = (hi - lo) / n
    if width <= 0:
        return lo, hi, n
    k0 = math.floor(lo / width + 1e-9)
    k1 = max(math.ceil(hi / width - 1e-9), k0 + 1)
    return k0 * width, k1 * width, k1 - k0


def _snapped_axis_mask(
    col: str, lo: float, hi: float, dtype: pl.DataType | None, schema: pl.Schema | None
) -> pl.Expr:
    """Rows inside the snapped bin rectangle on one axis.

    A temporal column compares on its physical representation, where the
    snapped bounds live. Physical temporal values are whole units, so rounding
    each bound inward keeps membership exact and the comparison in the column's
    own type instead of widening it to Float64.
    """
    if dtype is not None:
        return (
            pl.col(col)
            .to_physical()
            .is_between(pl.lit(math.ceil(lo)), pl.lit(math.floor(hi)))
        )
    return pl.col(col).is_between(*_typed_range_bounds(col, (lo, hi), schema))


def _snapped_axis(
    col: str, range_: tuple, n: int, schema: pl.Schema | None
) -> tuple[float, float, int, pl.DataType | None]:
    """A zoomed axis in the kernel's data space: snapped bounds, bin count, and
    the column's temporal dtype (``None`` when numeric).

    A viewport bound can be a date string or an epoch number, so it is evaluated
    into that space first. One lattice rule then serves temporal and numeric
    axes alike. This is the only place a viewport is snapped, so a 1-D display
    grid, a 2-D display grid, and a cube target cannot land on different edges.
    """
    dtype = _temporal_dtype_for_col(col, schema)
    # Both bounds are literals, so this selects no data.
    lo_expr, hi_expr = _hist2d_bound_lits(range_, dtype)
    raw = pl.select(lo_expr.alias("l"), hi_expr.alias("h")).row(0)
    lo, hi, n = _snap_range(float(raw[0]), float(raw[1]), n)
    return lo, hi, n, dtype


def _axis_edges(
    col: str,
    range_: tuple | None,
    n: int,
    domains: Mapping[str, tuple[Any, Any]] | None,
    schema: pl.Schema | None,
) -> tuple[pl.Expr, pl.Expr, pl.Expr | None, int]:
    """One axis's bin-edge literals, its viewport mask, and its bin count.

    Without a range the axis spans the engine-resolved unfiltered ``(min, max)``
    and needs no mask, so cross-filtering cannot move its bin edges. With one,
    the bounds are the viewport snapped outward to a fixed lattice and the mask
    restricts the data to that same span, so every edge bin is complete.
    Snapping costs at most one extra bin, which is why the count comes back too.
    """
    if range_ is None:
        # The kernel adds its own EPS to the span, so pass the raw bounds.
        lo_lit, hi_lit = _domain_lits(domains, col)
        return lo_lit, hi_lit, None, n
    lo, hi, n, dtype = _snapped_axis(col, range_, n, schema)
    return pl.lit(lo), pl.lit(hi), _snapped_axis_mask(col, lo, hi, dtype, schema), n


def _hist2d_bounds(
    x_col: str,
    y_col: str,
    x_range: tuple | None,
    y_range: tuple | None,
    nb_x: int,
    nb_y: int,
    domains: Mapping[str, tuple[Any, Any]] | None,
    schema: pl.Schema | None = None,
) -> tuple[_Edges, pl.Expr | None, tuple[int, int]]:
    """Resolve the bin edges, the viewport mask, and the grid for the kernels.

    ``update_range`` holds any subset of the figure's axes, because the client
    sends only the axes a zoom moved. So each axis resolves on its own through
    ``_axis_edges``: a zoom on x alone re-bins x to the viewport and leaves y on
    its full domain. The mask is the conjunction of the masks that exist, and
    the grid comes back with the edges because snapping can add a bin per axis.

    Boundary note: ``is_between`` is inclusive on both ends, so a value equal
    to ``x_hi`` passes the filter and is placed in the last bin by the Rust
    kernel's ``min(xi, max_xi)`` clamp — both sides must agree on this
    inclusive-right semantics.
    """
    x_lo, x_hi, x_mask, nb_x = _axis_edges(x_col, x_range, nb_x, domains, schema)
    y_lo, y_hi, y_mask, nb_y = _axis_edges(y_col, y_range, nb_y, domains, schema)
    masks = [m for m in (x_mask, y_mask) if m is not None]
    mask = masks[0] & masks[1] if len(masks) == 2 else (masks[0] if masks else None)
    return (x_lo, x_hi, y_lo, y_hi), mask, (nb_x, nb_y)


def _domain_lits(
    domains: Mapping[str, tuple[Any, Any]] | None, col: str
) -> tuple[pl.Expr, pl.Expr]:
    lo, hi = (domains or {})[col]
    return pl.lit(0.0 if lo is None else lo), pl.lit(1.0 if hi is None else hi)


def _hist2d_count_expr(
    x_col: str,
    y_col: str,
    nb_x: int,
    nb_y: int,
    edges: _Edges,
    mask: pl.Expr | None,
    alias: str,
    schema: pl.Schema | None = None,
) -> pl.Expr:
    """Build a count-only 2D histogram expression using fixed_hist2d.

    ``edges`` are the raw bounds from ``_hist2d_bounds``: the Rust kernel adds
    its own internal EPS to ``(x_hi - x_lo)`` when computing the bin scale, so
    they must not be EPS-adjusted. ``mask`` restricts the rows inside the
    expression; the batch fold passes None and filters the frame instead.
    """
    x_phys = _hist2d_phys_col(x_col, schema)
    y_phys = _hist2d_phys_col(y_col, schema)
    x_expr = x_phys.filter(mask) if mask is not None else x_phys
    y_expr = y_phys.filter(mask) if mask is not None else y_phys
    return x_expr.flexviz.fixed_hist2d(y_expr, *edges, nb_x, nb_y).alias(alias)


def _hist2d_reduce_expr(
    x_col: str,
    y_col: str,
    z_col: str,
    nb_x: int,
    nb_y: int,
    edges: _Edges,
    mask: pl.Expr | None,
    alias: str,
    histfunc: str,
    schema: pl.Schema | None = None,
) -> pl.Expr:
    """Build a z-reduced 2D histogram expression using fixed_hist2d_reduce.

    Same ``edges`` and ``mask`` contract as ``_hist2d_count_expr``.
    """
    x_phys = _hist2d_phys_col(x_col, schema)
    y_phys = _hist2d_phys_col(y_col, schema)
    x_expr = x_phys.filter(mask) if mask is not None else x_phys
    y_expr = y_phys.filter(mask) if mask is not None else y_phys
    z_expr = pl.col(z_col).filter(mask) if mask is not None else pl.col(z_col)
    return x_expr.flexviz.fixed_hist2d_reduce(
        y_expr, z_expr, *edges, nb_x, nb_y, histfunc
    ).alias(alias)


# ---------------------------------------------------------------------------
# Batch fold (scan sources)
# ---------------------------------------------------------------------------

#: Rows per streamed batch. About 64 MB for a two-column Float64 batch. Larger
#: chunks only cost memory; smaller ones cost per-batch Python and kernel
#: overhead.
_FOLD_CHUNK_ROWS = 4_000_000


def _fold_result_frame(
    uid: str, z_flat: pl.Series, bounds: tuple[float, float, float, float]
) -> pl.DataFrame:
    """The one-row, one-column frame the kernel expression would have produced.

    ``z_flat`` already carries the kernel's dtype (UInt32 counts, nullable
    Float64 reducer cells), so the struct is built from the Series itself: a
    Python list of a million cells costs more than the fold.
    """
    x_lo, x_hi, y_lo, y_hi = bounds
    return pl.select(
        pl.struct(
            z_flat.implode().alias("z_flat"),
            pl.lit(x_lo).alias("x_lo"),
            pl.lit(x_hi).alias("x_hi"),
            pl.lit(y_lo).alias("y_lo"),
            pl.lit(y_hi).alias("y_hi"),
        ).alias(uid)
    )


def _finite_z_expr(z_col: str, schema: pl.Schema | None) -> pl.Expr:
    """Rows whose ``z`` the reduce kernel accepts: not null, and not NaN for a
    float column."""
    usable = pl.col(z_col).is_not_null()
    dtype = _dtype_for_col(schema, z_col)
    if dtype is not None and dtype.is_float():
        usable = usable & pl.col(z_col).is_not_nan()
    return usable


def _hist1d_fold_plan(
    value_expr: pl.Expr,
    lo: float,
    hi: float,
    bins: int,
    uid: str,
    mask: pl.Expr | None,
) -> Callable[[pl.LazyFrame], pl.DataFrame]:
    """The ``fixed_hist`` kernel folded over streamed batches, for a scan source.

    The 2-D fold one dimension down: the kernel materializes the whole column,
    so a scan runs it on one batch at a time and sums the counts in NumPy, over
    the batches in flight rather than the whole column. Peak memory is the
    reader's row-group prefetch window (``POLARS_ROW_GROUP_PREFETCH_SIZE``),
    which does not grow with the file. Counts add exactly, so the grid equals
    the kernel's. A streaming ``group_by`` on the bin index is
    the obvious alternative and was measured at about twice the time, and worse
    as the bin count grows, where the fold is flat.

    The frame is filtered before the batches, so the scan itself rejects the
    rows outside the viewport, and the imploded ``{breakpoint, count}`` struct
    matches the kernel's own output shape.
    """
    hist_expr = pl.col("v").flexviz.fixed_hist(pl.lit(lo), pl.lit(hi), n_bins=bins)
    # The kernel's breakpoints, computed its way: a degenerate span has no step.
    step = (hi - lo) / bins if hi > lo else 0.0
    breakpoints = lo + np.arange(1, bins + 1, dtype=np.float64) * step

    def run(filtered_ldf: pl.LazyFrame) -> pl.DataFrame:
        src = filtered_ldf if mask is None else filtered_ldf.filter(mask)
        acc = np.zeros(bins, dtype=np.int64)
        for batch in src.select(value_expr.alias("v")).collect_batches(
            chunk_size=_FOLD_CHUNK_ROWS, maintain_order=False, engine="streaming"
        ):
            counts = batch.select(hist_expr.alias("h"))["h"].struct.field("count")
            acc += counts.to_numpy()
        return pl.select(
            pl.struct(
                pl.Series("breakpoint", breakpoints),
                pl.Series("count", acc, dtype=pl.UInt32),
            )
            .implode()
            .alias(uid)
        )

    return run


def _batch_fold_plan(
    x_col: str,
    y_col: str,
    z_col: str | None,
    nb_x: int,
    nb_y: int,
    histfunc: str | None,
    edges: _Edges,
    mask: pl.Expr | None,
    uid: str,
    schema: pl.Schema | None = None,
) -> Callable[[pl.LazyFrame], pl.DataFrame]:
    """The kernel grid folded over streamed batches, for a scan source.

    As one expression the kernels need whole Series, so a scan materializes
    both axis columns before binning. This runs the same kernels on one batch
    at a time and accumulates the grid in NumPy, so peak memory is one batch
    plus the grid. A streaming ``group_by`` on the flattened cell key is the
    obvious alternative, but it is unbounded above the Polars hot table size
    (measured), and ``collect_batches`` is the Polars escape hatch for custom
    logic over streamed batches.

    Count, min and max are exact. Sum and mean fold in a different order than
    the kernel, so they agree to about 1e-13 relative.

    ``collect_batches`` is marked unstable in Polars; a semantics test pins the
    chunking this relies on.
    """
    # Both bounds are literal expressions, so this selects no data. The kernel
    # reads them as f64, and the result frame echoes them back.
    bounds = tuple(
        float(v)
        for v in pl.select(
            *(e.alias(name) for e, name in zip(edges, ("xl", "xh", "yl", "yh")))
        ).row(0)
    )
    lits = tuple(pl.lit(v) for v in bounds)
    n_cells = nb_x * nb_y
    cols = [c for c in (x_col, y_col, z_col) if c is not None]

    if z_col is None:
        exprs = [_hist2d_count_expr(x_col, y_col, nb_x, nb_y, lits, None, "g", schema)]
    else:
        assert histfunc is not None
        exprs = [
            _hist2d_reduce_expr(
                x_col,
                y_col,
                z_col,
                nb_x,
                nb_y,
                lits,
                None,
                "g",
                "sum" if histfunc == "mean" else histfunc,
                schema,
            )
        ]
        if histfunc == "mean":
            # The reduce kernel skips null/NaN z, so the mean denominator is the
            # count of the rows with a usable z, not the count of all rows.
            exprs.append(
                _hist2d_count_expr(
                    x_col,
                    y_col,
                    nb_x,
                    nb_y,
                    lits,
                    _finite_z_expr(z_col, schema),
                    "c",
                    schema,
                )
            )

    def run(filtered_ldf: pl.LazyFrame) -> pl.DataFrame:
        src = filtered_ldf if mask is None else filtered_ldf.filter(mask)
        acc = np.zeros(n_cells, dtype=np.int64 if z_col is None else np.float64)
        seen = np.zeros(n_cells, dtype=bool)
        cnt = np.zeros(n_cells, dtype=np.int64)
        for batch in src.select(cols).collect_batches(
            chunk_size=_FOLD_CHUNK_ROWS, maintain_order=False, engine="streaming"
        ):
            out = batch.select(exprs)
            # Through Arrow, not Python: ``out["g"][0]`` would build a dict and
            # a list of every cell once per batch.
            grid = out["g"].struct.field("z_flat").item().to_numpy()
            if z_col is None:
                acc += grid
                continue
            if histfunc == "mean":
                cnt += out["c"].struct.field("z_flat").item().to_numpy()
            # An empty cell comes back null, which lands here as NaN.
            filled = ~np.isnan(grid)
            if histfunc in ("sum", "mean"):
                acc[filled] += grid[filled]
            elif histfunc == "min":
                acc[filled] = np.where(
                    seen[filled], np.fmin(acc[filled], grid[filled]), grid[filled]
                )
            else:
                acc[filled] = np.where(
                    seen[filled], np.fmax(acc[filled], grid[filled]), grid[filled]
                )
            seen |= filled

        # No batches (an empty frame) leaves the zero grid, which is what the
        # kernel returns for empty input: zero counts, null reducer cells.
        # An empty reducer cell is NaN here and null in the result, as in the
        # kernel's own output.
        if z_col is None:
            z_flat = pl.Series(acc, dtype=pl.UInt32)
        elif histfunc == "mean":
            mean = np.full(n_cells, np.nan)
            np.divide(acc, cnt, out=mean, where=cnt > 0)
            z_flat = pl.Series(mean).fill_nan(None)
        else:
            z_flat = pl.Series(np.where(seen, acc, np.nan)).fill_nan(None)
        return _fold_result_frame(uid, z_flat, bounds)

    return run


def _hist2d_agg_spec(
    x_col: str,
    y_col: str,
    z_col: str | None,
    histfunc: str | None,
    x_range: tuple | None,
    y_range: tuple | None,
    nb_x: int,
    nb_y: int,
    uid: str,
    domains: Mapping[str, tuple[Any, Any]] | None,
    schema: pl.Schema | None,
    scan_source: bool,
) -> tuple[AggregationSpec, tuple[int, int]]:
    """The 2-D histogram aggregation spec, plus the grid it bins on.

    Both 2-D traces come through here over their own column pair. Each axis
    resolves on its own: an axis without a range needs its column in
    ``domains``, where a ``(None, None)`` entry means an empty or all-null
    column, and an axis with one snaps its edges to a lattice, which can add a
    bin. So the caller must keep the returned grid and unpack ``z_flat`` with
    it.

    ``scan_source`` says the rows come from storage rather than a resident
    frame. The kernels need both axis columns in memory at once, so a scan takes
    ``_batch_fold_plan`` instead: the same kernels, run over streamed batches and
    folded in NumPy, at bounded memory. A resident frame keeps the plain kernel
    expression, which joins the shared select.
    """
    edges, mask, grid = _hist2d_bounds(
        x_col, y_col, x_range, y_range, nb_x, nb_y, domains, schema
    )
    nb_x, nb_y = grid

    if scan_source:
        plan = _batch_fold_plan(
            x_col, y_col, z_col, nb_x, nb_y, histfunc, edges, mask, uid, schema
        )
        return AggregationSpec(uid=uid, plan=plan), grid

    if z_col is None:
        expr = _hist2d_count_expr(x_col, y_col, nb_x, nb_y, edges, mask, uid, schema)
    else:
        assert histfunc is not None
        expr = _hist2d_reduce_expr(
            x_col, y_col, z_col, nb_x, nb_y, edges, mask, uid, histfunc, schema
        )
    return AggregationSpec(expr=expr, uid=uid), grid

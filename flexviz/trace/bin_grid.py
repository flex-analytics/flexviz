"""The bin grid a binned trace bins on: edges, viewport mask, kernel calls.

``Histogram``, ``Histogram2D`` and ``GeoHistogram2D`` resolve their grid here,
so a 1-D display grid, a 2-D display grid and a cube target cannot land on
different edges.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import polars as pl

from .base import (
    _physical_bound_expr,
    _temporal_dtype_for_col,
    _typed_range_bounds,
)

import flexviz_polars  # noqa: F401 — registers pl.Expr.flexviz namespace

#: The (x_lo, x_hi, y_lo, y_hi) bin-edge literal expressions from axis_edges.
Edges = tuple[pl.Expr, pl.Expr, pl.Expr, pl.Expr]


def snap_range(lo: float, hi: float, n: int) -> tuple[float, float, int]:
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


def snapped_axis(
    col: str, range_: tuple, n: int, schema: pl.Schema | None
) -> tuple[float, float, int, pl.Expr]:
    """A zoomed axis in the kernel's data space: snapped bounds, bin count, and
    the mask selecting the rows inside them.

    A viewport bound can be a date string or an epoch number, so it is evaluated
    into that space first: physical units for a temporal column, plain floats
    otherwise. One lattice rule then serves temporal and numeric axes alike.
    This is the only place a viewport is snapped.

    Physical temporal values are whole units, so rounding each bound inward
    keeps membership exact and the comparison in the column's own type instead
    of widening it to Float64. ``is_between`` is inclusive on both ends, so a
    value equal to the upper bound passes the mask and the kernel's top clamp
    puts it in the last bin: both sides agree on inclusive-right.
    """
    dtype = _temporal_dtype_for_col(col, schema)
    if dtype is not None:
        lo_lit = _physical_bound_expr(range_[0], dtype)
        hi_lit = _physical_bound_expr(range_[1], dtype)
    else:
        lo_lit, hi_lit = pl.lit(float(range_[0])), pl.lit(float(range_[1]))
    # Both bounds are literals, so this selects no data.
    raw = pl.select(lo_lit.alias("l"), hi_lit.alias("h")).row(0)
    lo, hi, n = snap_range(float(raw[0]), float(raw[1]), n)

    if dtype is not None:
        mask = (
            pl.col(col)
            .to_physical()
            .is_between(pl.lit(math.ceil(lo)), pl.lit(math.floor(hi)))
        )
    else:
        mask = pl.col(col).is_between(*_typed_range_bounds(col, (lo, hi), schema))
    return lo, hi, n, mask


def axis_edges(
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
        lo, hi = (domains or {})[col]
        return (
            pl.lit(0.0 if lo is None else lo),
            pl.lit(1.0 if hi is None else hi),
            None,
            n,
        )
    lo, hi, n, mask = snapped_axis(col, range_, n, schema)
    return pl.lit(lo), pl.lit(hi), mask, n


def hist2d_phys_col(col: str, schema: pl.Schema | None) -> pl.Expr:
    """The data expression to feed the numeric kernel: physical representation
    for a temporal column (the kernel needs numeric data), else the raw column.
    """
    dtype = _temporal_dtype_for_col(col, schema)
    return pl.col(col).to_physical() if dtype is not None else pl.col(col)


def hist2d_count_expr(
    x_col: str,
    y_col: str,
    nb_x: int,
    nb_y: int,
    edges: Edges,
    mask: pl.Expr | None,
    alias: str,
    schema: pl.Schema | None = None,
) -> pl.Expr:
    """Build a count-only 2D histogram expression using fixed_hist2d.

    ``edges`` are the raw bounds from ``axis_edges``: the Rust kernel adds its
    own internal EPS to ``(x_hi - x_lo)`` when computing the bin scale, so they
    must not be EPS-adjusted. ``mask`` restricts the rows inside the
    expression; the batch fold passes None and filters the frame instead.
    """
    x_phys = hist2d_phys_col(x_col, schema)
    y_phys = hist2d_phys_col(y_col, schema)
    x_expr = x_phys.filter(mask) if mask is not None else x_phys
    y_expr = y_phys.filter(mask) if mask is not None else y_phys
    return x_expr.flexviz.fixed_hist2d(y_expr, *edges, nb_x, nb_y).alias(alias)


def hist2d_reduce_expr(
    x_col: str,
    y_col: str,
    z_col: str,
    nb_x: int,
    nb_y: int,
    edges: Edges,
    mask: pl.Expr | None,
    alias: str,
    histfunc: str,
    schema: pl.Schema | None = None,
) -> pl.Expr:
    """Build a z-reduced 2D histogram expression using fixed_hist2d_reduce.

    Same ``edges`` and ``mask`` contract as ``hist2d_count_expr``.
    """
    x_phys = hist2d_phys_col(x_col, schema)
    y_phys = hist2d_phys_col(y_col, schema)
    x_expr = x_phys.filter(mask) if mask is not None else x_phys
    y_expr = y_phys.filter(mask) if mask is not None else y_phys
    z_expr = pl.col(z_col).filter(mask) if mask is not None else pl.col(z_col)
    return x_expr.flexviz.fixed_hist2d_reduce(
        y_expr, z_expr, *edges, nb_x, nb_y, histfunc
    ).alias(alias)

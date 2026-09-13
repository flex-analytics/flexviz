"""The histogram kernels folded over streamed batches, for a scan source.

The kernels materialize their whole input, so a scan runs them one batch at a
time and accumulates in NumPy, at bounded memory. Each plan is carried on an
``AggregationSpec.plan`` and returns the frame the kernel expression would.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl

import flexviz_polars  # noqa: F401 — registers pl.Expr.flexviz namespace

from .base import _dtype_for_col
from .bin_grid import Edges, hist2d_count_expr, hist2d_reduce_expr

#: Rows per streamed batch. With a consumer that keeps up, peak live memory is
#: one to two batches plus the output grid (measured 19 / 25 / 48 MB at
#: 250k / 1M / 4M rows of one Float64 column, 32 threads), so the chunk sets the
#: working set without hard-bounding Polars' streaming pipeline.
#: The impact on the runtime is very small < 3% -- for the 2-D fold, whose
#: per-batch kernel is heavy enough to dominate the per-batch dispatch cost.
_FOLD_CHUNK_ROWS = 1_000_000

#: The 1-D count kernel is cheap, so the per-batch dispatch cost (one kernel
#: call + Arrow->NumPy + accumulate per batch) dominates instead, and a 1M
#: chunk -- ~4x more batches -- made the scan histogram ~1.3x slower at scale
#: than the 4M it used to run. The larger fixed chunk buys the speed back while
#: keeping memory bounded independently of source row count. Kept separate from
#: the 2-D constant, which stays 1M for its smaller working set.
_FOLD_CHUNK_ROWS_1D = 4_000_000


def _fold_result_frame(uid: str, z_flat: pl.Series, bounds: Edges) -> pl.DataFrame:
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


def hist1d_fold_plan(
    value_expr: pl.Expr,
    lo: float,
    hi: float,
    bins: int,
    uid: str,
    mask: pl.Expr | None,
) -> Callable[[pl.LazyFrame], pl.DataFrame]:
    """The ``fixed_hist`` kernel folded over streamed batches, for a scan source.

    The 2-D fold one dimension down. The kernel needs its whole input as one
    Series, so a scan runs it on one batch at a time and sums the counts in
    NumPy. Peak live memory is the current batch, bin grid, and Polars streaming
    pipeline, bounded independently of the file's row count. A streaming
    ``group_by`` on the bin index is the obvious alternative, but it is slower
    and its memory grows with the number of bin keys. Counts add exactly, so
    the grid equals the kernel's.

    The frame is filtered before the batches, so the scan itself rejects the
    rows outside the viewport. Only the counts are emitted: the trace derives
    its bin centers from the bounds it binned with.

    The plan handed to ``collect_batches`` projects the source column as stored
    and nothing else; ``value_expr`` (a temporal column's physical cast) runs
    on each batch. An alias or cast between the scan and the batch sink puts
    Polars on a buffered path that holds row groups per thread ahead of a
    consumer that falls behind (see Architecture.md, Execution decisions).
    """
    hist_expr = value_expr.flexviz.fixed_hist(pl.lit(lo), pl.lit(hi), n_bins=bins)
    cols = value_expr.meta.root_names()

    def run(filtered_ldf: pl.LazyFrame) -> pl.DataFrame:
        src = filtered_ldf if mask is None else filtered_ldf.filter(mask)
        acc = np.zeros(bins, dtype=np.int64)
        for batch in src.select(cols).collect_batches(
            chunk_size=_FOLD_CHUNK_ROWS_1D, maintain_order=False, engine="streaming"
        ):
            counts = batch.select(hist_expr.alias("h"))["h"].struct.field("count")
            acc += counts.to_numpy()
        return pl.select(
            pl.struct(pl.Series("count", acc, dtype=pl.UInt32)).implode().alias(uid)
        )

    return run


def hist2d_fold_plan(
    x_col: str,
    y_col: str,
    z_col: str | None,
    nb_x: int,
    nb_y: int,
    histfunc: str | None,
    edges: Edges,
    mask: pl.Expr | None,
    uid: str,
    schema: pl.Schema | None = None,
) -> Callable[[pl.LazyFrame], pl.DataFrame]:
    """The kernel grid folded over streamed batches, for a scan source.

    As one expression the kernels need whole Series, so a scan materializes
    both axis columns before binning. This runs the same kernels on one batch
    at a time and accumulates the grid in NumPy, so peak memory is one batch
    plus the grid. A streaming ``group_by`` on the flattened cell key is the
    obvious alternative, but it is slower and its memory grows with the number
    of cell keys. ``collect_batches`` is the Polars escape hatch for custom
    logic over streamed batches.

    Count, min and max are exact. Sum and mean fold in a different order than
    the kernel, so they agree to about 1e-13 relative.

    ``collect_batches`` is marked unstable in Polars; a semantics test pins the
    chunking this relies on.
    """
    n_cells = nb_x * nb_y
    # z may be one of the axis columns, and a select refuses a duplicate.
    cols = list(dict.fromkeys(c for c in (x_col, y_col, z_col) if c is not None))

    if z_col is None:
        exprs = [hist2d_count_expr(x_col, y_col, nb_x, nb_y, edges, None, "g", schema)]
    else:
        assert histfunc is not None
        exprs = [
            hist2d_reduce_expr(
                x_col,
                y_col,
                z_col,
                nb_x,
                nb_y,
                edges,
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
                hist2d_count_expr(
                    x_col,
                    y_col,
                    nb_x,
                    nb_y,
                    edges,
                    _finite_z_expr(z_col, schema),
                    "c",
                    schema,
                )
            )

    def run(filtered_ldf: pl.LazyFrame) -> pl.DataFrame:
        src = filtered_ldf if mask is None else filtered_ldf.filter(mask)
        # Keep the batch projection in source order. Polars 1.44 routes a pure
        # reorder through a buffered stage (~7 row groups per thread), while
        # the kernels address columns by name and do not care about order.
        # ``collect_schema()`` is a metadata-only fallback when none was given.
        source_schema = schema or filtered_ldf.collect_schema()
        batch_cols = [col for col in source_schema if col in cols]
        acc = np.zeros(n_cells, dtype=np.int64 if z_col is None else np.float64)
        seen = np.zeros(n_cells, dtype=bool)
        cnt = np.zeros(n_cells, dtype=np.int64)
        for batch in src.select(batch_cols).collect_batches(
            chunk_size=_FOLD_CHUNK_ROWS, maintain_order=False, engine="streaming"
        ):
            out = batch.select(exprs)
            # Through Arrow, not Python: ``out["g"][0]`` would build a dict and
            # a list of every cell once per batch.
            cell = out["g"].struct.field("z_flat").item()
            grid = cell.to_numpy()
            if z_col is None:
                acc += grid
                continue
            if histfunc == "mean":
                cnt += out["c"].struct.field("z_flat").item().to_numpy()
            # The kernel marks an empty cell null, and ``to_numpy`` writes a null
            # as NaN. Only the validity bitmap separates an empty cell from a
            # cell whose reduction is genuinely NaN (z holding +inf and -inf).
            filled = cell.is_not_null().to_numpy()
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
        return _fold_result_frame(uid, z_flat, edges)

    return run

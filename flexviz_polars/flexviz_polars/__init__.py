from __future__ import annotations

import numbers
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

from flexviz_polars._internal import __version__ as __version__

if TYPE_CHECKING:
    from flexviz_polars.typing import IntoExprColumn

LIB = Path(__file__).parent


def _every_nth(expr: IntoExprColumn, n_points: int) -> pl.Expr:
    return register_plugin_function(
        args=[expr],
        plugin_path=LIB,
        function_name="every_nth",
        kwargs={"n_points": n_points},
        is_elementwise=False,
    )


def _kernel_bound(value) -> int | float:
    """Normalize one ``x_domain`` bound to the Python type the kernel reads.

    An integral bound stays a Python ``int``: the kernel searches integer x on
    exact integers, and ``float()`` would round a nanosecond epoch. Everything
    else goes as ``float``.
    """
    return int(value) if isinstance(value, numbers.Integral) else float(value)


def _minmax_pairs_line(
    x_expr: "IntoExprColumn",
    y_expr: "IntoExprColumn",
    n_buckets: int,
    x_domain: tuple[int | float, int | float],
) -> pl.Expr:
    """One ``(x_min, y_min, x_max, y_max)`` row per non-empty bucket.

    The buckets are equal in x width over ``x_domain``, which needs x sorted
    ascending; rows outside it are dropped. A zero span returns no rows. The
    pairing is kept, so the caller can flatten the pairs or walk them.
    """
    return register_plugin_function(
        args=[x_expr, y_expr],
        plugin_path=LIB,
        function_name="minmax_pairs_line",
        kwargs={
            "n_buckets": n_buckets,
            "x_domain": tuple(_kernel_bound(v) for v in x_domain),
        },
        is_elementwise=False,
    )


def _fixed_hist(
    expr: "IntoExprColumn",
    lo_expr: pl.Expr,
    hi_expr: pl.Expr,
    n_bins: int,
) -> pl.Expr:
    return register_plugin_function(
        args=[expr, lo_expr, hi_expr],
        plugin_path=LIB,
        function_name="fixed_hist",
        kwargs={"n_bins": n_bins},
        is_elementwise=False,
    )


def _fixed_hist2d(
    x_expr: "IntoExprColumn",
    y_expr: "IntoExprColumn",
    x_lo_expr: pl.Expr,
    x_hi_expr: pl.Expr,
    y_lo_expr: pl.Expr,
    y_hi_expr: pl.Expr,
    nb_x: int,
    nb_y: int,
) -> pl.Expr:
    return register_plugin_function(
        args=[x_expr, y_expr, x_lo_expr, x_hi_expr, y_lo_expr, y_hi_expr],
        plugin_path=LIB,
        function_name="fixed_hist2d",
        kwargs={"nb_x": nb_x, "nb_y": nb_y},
        is_elementwise=False,
    )


def _fixed_hist2d_reduce(
    x_expr: "IntoExprColumn",
    y_expr: "IntoExprColumn",
    z_expr: "IntoExprColumn",
    x_lo_expr: pl.Expr,
    x_hi_expr: pl.Expr,
    y_lo_expr: pl.Expr,
    y_hi_expr: pl.Expr,
    nb_x: int,
    nb_y: int,
    histfunc: str,
) -> pl.Expr:
    return register_plugin_function(
        args=[x_expr, y_expr, z_expr, x_lo_expr, x_hi_expr, y_lo_expr, y_hi_expr],
        plugin_path=LIB,
        function_name="fixed_hist2d_reduce",
        kwargs={"nb_x": nb_x, "nb_y": nb_y, "histfunc": histfunc},
        is_elementwise=False,
    )


def _fixed_line_envelope2d(
    x_expr: "IntoExprColumn",
    y_expr: "IntoExprColumn",
    free_expr: "IntoExprColumn",
    x_lo_expr: pl.Expr,
    x_hi_expr: pl.Expr,
    free_lo_expr: pl.Expr,
    free_hi_expr: pl.Expr,
    n_buckets: int,
    p: int,
) -> pl.Expr:
    return register_plugin_function(
        args=[
            x_expr,
            y_expr,
            free_expr,
            x_lo_expr,
            x_hi_expr,
            free_lo_expr,
            free_hi_expr,
        ],
        plugin_path=LIB,
        function_name="fixed_line_envelope2d",
        kwargs={"n_buckets": n_buckets, "p": p},
        is_elementwise=False,
    )


@pl.api.register_expr_namespace("flexviz")
class FlexvizExprNamespace:
    """Polars expression namespace for flexviz downsampling kernels.

    Activated by ``import flexviz_polars``. After that, any Polars expression
    supports ``.flexviz.every_nth(n)`` and
    ``.flexviz.fixed_hist(lo, hi, n_bins)``.
    """

    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def every_nth(self, n_points: int) -> pl.Expr:
        """Return every nth element of the series (stride computed from series length).

        Parameters
        ----------
        n_points:
            Target number of output points.  The kernel computes
            ``stride = max(1, len // n_points)`` internally — no Polars
            ``len()`` expression dependency.  When ``n_points >= len``,
            the full series is returned unchanged.
        """
        return _every_nth(self._expr, n_points)

    def fixed_hist(self, lo_expr: pl.Expr, hi_expr: pl.Expr, n_bins: int) -> pl.Expr:
        """Compute a fixed-bin 1D histogram using O(n) direct floor-division indexing.

        Parameters
        ----------
        lo_expr:
            Polars expression evaluating to a scalar (length-1 Series) that is the
            left edge of the first bin.  Pass it as ``pl.lit(lo)``; a null or
            missing value falls back to 0.0.
        hi_expr:
            Right edge of the last bin (caller must add epsilon before calling).
        n_bins:
            Number of bins.

        Returns
        -------
        pl.Expr
            A ``Struct{breakpoint: Float64, count: UInt32}`` Series of length *n_bins*.
            Identical output format to ``polars.hist(..., include_breakpoint=True)``.
        """
        return _fixed_hist(self._expr, lo_expr, hi_expr, n_bins)

    def fixed_hist2d(
        self,
        y_expr: "IntoExprColumn",
        x_lo_expr: pl.Expr,
        x_hi_expr: pl.Expr,
        y_lo_expr: pl.Expr,
        y_hi_expr: pl.Expr,
        nb_x: int,
        nb_y: int,
    ) -> pl.Expr:
        """Compute a 2D count histogram via a single O(n) Rust pass.

        Uses typed dispatch (avoids f64 cast for same-type pairs) and a
        contiguous-slice fast path for null-free single-chunk inputs.
        Returns a length-1 ``Struct{z_flat: List(UInt32), x_lo, x_hi, y_lo, y_hi}``.
        ``self`` is the x expression; pass y as the first positional argument.
        Pass every bound as ``pl.lit(val)``.
        z_flat is row-major (y-first): ``z_flat[yi * nb_x + xi]`` is bin (xi, yi).
        Empty bins have count 0.
        """
        return _fixed_hist2d(
            self._expr,
            y_expr,
            x_lo_expr,
            x_hi_expr,
            y_lo_expr,
            y_hi_expr,
            nb_x,
            nb_y,
        )

    def fixed_hist2d_reduce(
        self,
        y_expr: "IntoExprColumn",
        z_expr: "IntoExprColumn",
        x_lo_expr: pl.Expr,
        x_hi_expr: pl.Expr,
        y_lo_expr: pl.Expr,
        y_hi_expr: pl.Expr,
        nb_x: int,
        nb_y: int,
        histfunc: str,
    ) -> pl.Expr:
        """Compute a z-reduced 2D histogram for sum, mean, min, and max.

        Returns a length-1 ``Struct{z_flat: List(Float64), x_lo, x_hi, y_lo, y_hi}``.
        Empty bins are null. ``self`` is the x expression; pass y and z as the first
        two positional arguments.
        """
        return _fixed_hist2d_reduce(
            self._expr,
            y_expr,
            z_expr,
            x_lo_expr,
            x_hi_expr,
            y_lo_expr,
            y_hi_expr,
            nb_x,
            nb_y,
            histfunc,
        )

    def fixed_line_envelope2d(
        self,
        y_expr: "IntoExprColumn",
        free_expr: "IntoExprColumn",
        x_lo_expr: pl.Expr,
        x_hi_expr: pl.Expr,
        free_lo_expr: pl.Expr,
        free_hi_expr: pl.Expr,
        n_buckets: int,
        p: int,
    ) -> pl.Expr:
        """One-pass exact argmin/argmax-by-y line envelope per (bucket, free_bin)
        cell (cube line-envelope target — contract J / issue #36).

        ``self`` is the line's x expression; pass y and the free (active
        brush) column as the first two positional arguments. Bounds are scalar
        expressions (``pl.lit(val)``), same convention as ``fixed_hist2d``.

        Bin arithmetic is the cube **shared arithmetic** on both axes —
        natural floor ``floor((v - lo) / (hi - lo) * n)`` with NO epsilon and
        NO clip: rows outside ``[lo, hi]`` on either axis are filtered out,
        and a value exactly at the domain max lands in the degenerate top
        bin, so bucket indices run ``0..=n_buckets`` and free indices
        ``0..=p`` inclusive. Rows with a null or NaN in x, y, or free are
        filtered. Ties (equal y within a cell) keep the FIRST row in scan
        order for both min and max — deterministic.

        Any primitive numeric dtype is accepted for x/y/free; values are
        cast to Float64 internally (temporal columns must be cast to their
        physical representation by the caller, as for ``fixed_hist``). x and
        y come back as exact f64 — no quantization (encode-time f32/u16
        packing is the codec's job).

        Returns
        -------
        pl.Expr
            A ``Struct{bucket: UInt32, free_bin: UInt32, y_min: Float64,
            x_at_ymin: Float64, y_max: Float64, x_at_ymax: Float64}`` Series
            with one row per **non-empty** cell, sorted by
            ``(free_bin, bucket)`` ascending — unnest it for the long-format
            cube frame.
        """
        return _fixed_line_envelope2d(
            self._expr,
            y_expr,
            free_expr,
            x_lo_expr,
            x_hi_expr,
            free_lo_expr,
            free_hi_expr,
            n_buckets,
            p,
        )

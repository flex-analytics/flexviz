"""Shared helpers for 2D histogram traces (Histogram2D, GeoHistogram2D, CorrHeatmap).

Provides constants, type aliases, aggregation expression builders, color-normalization
helpers, and a pure-Polars grid generator — used by the heatmap trace classes to
avoid duplication.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import polars as pl

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPSILON = 1e-10  # prevents off-by-one when a data point equals the range maximum

#: The Rust kernels add this *inside* the bin index, before truncating, so a
#: value sitting a hair below a bin boundary lands in the bin above rather than
#: the one below (``FIXED_HIST_ROUND_EPS`` in ``flexviz_polars/src/expressions.rs``,
#: shared by ``fixed_hist`` and ``fixed_hist2d``). The streaming plans have to
#: reproduce it exactly, so it lives here rather than once per trace module.
#: ``tests/test_trace_hist_streaming.py`` pins the behaviour.
_FIXED_HIST_ROUND_EPS: float = 1e-9

#: The 2D kernels widen the span before computing the scale, so a value exactly
#: at ``hi`` lands in the top bin (``FIXED_HIST2D_SPAN_EPS`` in
#: ``expressions.rs``). The 1D kernel has no equivalent: ``hist.py`` widens
#: ``hi`` itself with ``_HIST_BIN_EPSILON``. Not the same job as ``_EPSILON``
#: above, which belongs to the pure-Polars ``bin_2d`` path.
_FIXED_HIST2D_SPAN_EPS: float = 1e-10

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
# Streaming bin index (mirrors the Rust kernels)
# ---------------------------------------------------------------------------


def kernel_bin_index(value: pl.Expr, lo: float, scale: float, n_bins: int) -> pl.Expr:
    """The kernels' bin arithmetic as an expression, for streaming plans.

    Mirrors ``count_value`` / ``count_2d_slices`` in ``expressions.rs``::

        counts[(((v - lo) * scale + FIXED_HIST_ROUND_EPS) as usize).min(max_idx)] += 1

    Returns the bin index, or null for a value the kernels skip. Callers drop
    the null group and zero-fill the bins nothing landed in. *scale* differs
    between 1D and 2D (the 2D kernels widen the span first), so it is the
    caller's to compute; everything after it must not.

    Three choices here are deliberate.

    **Clip before the cast, not after.** Both orders agree on in-range data --
    truncation and floor differ only for negatives, and the clip maps those to
    0 either way -- but clipping first also bounds the value before it reaches
    ``Int32``, which matters when a narrow viewport and a far outlier make
    ``(v - lo) * scale`` overflow it.

    **No ``.floor()``.** It is redundant in front of a clip and a truncating
    cast. It is not a performance question either: the two forms measure
    within 0.5 percent of each other at 500M rows.

    **A non-strict cast is the null/NaN guard.** The kernels skip both, and a
    strict cast of NaN raises, so a guard is not optional. The clip has already
    bounded the value into ``Int32`` range, so NaN is the only conversion this
    cast can fail on: ``strict=False`` nulls exactly those rows and nothing
    else. Nulls reach here as nulls on their own, because the arithmetic
    propagates them. This costs nothing measurable, where an explicit
    ``fill_nan``/``is_nan`` pass costs about 6 percent of the scan at 200M rows.

    **The cast to Float64 is load-bearing, not decoration.** Polars keeps
    ``Float32 - <python float>`` in Float32, and the kernels widen to f64
    before doing any arithmetic. Without the cast, a Float32 column bins at
    Float32 precision and drifts from the kernel.
    """
    return (
        ((value.cast(pl.Float64) - lo) * scale + _FIXED_HIST_ROUND_EPS)
        .clip(0.0, float(n_bins - 1))
        .cast(pl.Int32, strict=False)
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
# Aggregation expression
# ---------------------------------------------------------------------------


def histfunc_agg_expr(z_col: str, histfunc: str) -> pl.Expr:
    """Return the Polars aggregation expression for *histfunc*.

    *z_col* is always required; the caller is responsible for routing the
    count (z=None) case to ``pl.len()`` directly.
    """
    return _AGG_FUNCTIONS[histfunc](z_col)


# ---------------------------------------------------------------------------
# Pure-Polars grid of all (bin_a, bin_b) combinations
# ---------------------------------------------------------------------------


def build_all_bins_grid(
    nb_a: int, nb_b: int, col_a: str = "x_bin", col_b: str = "y_bin"
) -> pl.DataFrame:
    """Return a DataFrame with the cross-product of bin indices.

    ``col_a`` ranges ``[0, nb_a)`` and ``col_b`` ranges ``[0, nb_b)``.
    All values are ``Float64`` to match the bin-index dtype from ``//``.
    """
    return pl.DataFrame({col_a: [float(i) for i in range(nb_a)]}).join(
        pl.DataFrame({col_b: [float(j) for j in range(nb_b)]}), how="cross"
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
# 2D binning pipeline (pure Polars, used inside map_batches)
# ---------------------------------------------------------------------------


def bin_2d(
    df: pl.DataFrame,
    col_a: str,
    col_b: str,
    nb_a: int,
    nb_b: int,
    histfunc: str | None = None,
    histnorm: str | None = None,
    z_col: str | None = None,
    a_range: tuple[float, float] | None = None,
    b_range: tuple[float, float] | None = None,
    fill_value: float | None = None,
    bin_col_a: str = "x_bin",
    bin_col_b: str = "y_bin",
) -> tuple[list, list, list, float, float]:
    """Compute a 2D histogram grid using pure Polars operations.

    Returns ``(a_centers, b_centers, z_flat, a_step, b_step)`` where
    *z_flat* is row-major (b-major) order: ``z_flat[j * nb_a + i]`` is
    bin ``(i, j)``.  *a_step* and *b_step* are the bin widths along each
    axis; callers can use them to derive bin edges without re-reading the
    source data.

    After ``sort([bin_col_b, bin_col_a])`` the result has a regular layout:
    rows ``0..nb_a-1`` hold all *a*-centers (at the smallest *b*-center),
    so ``result["a_center"][:nb_a]`` and ``result["b_center"][::nb_a]``
    give the unique sorted center lists directly, without a separate
    ``unique`` + ``sort`` pass.
    """
    valid_mask = pl.col(col_a).is_not_null() & pl.col(col_b).is_not_null()
    df = df.filter(valid_mask)

    if df.height == 0:
        a_lo = a_range[0] if a_range else 0.0
        a_hi = a_range[1] if a_range else 1.0
        b_lo = b_range[0] if b_range else 0.0
        b_hi = b_range[1] if b_range else 1.0
        a_step = (a_hi - a_lo) / nb_a
        b_step = (b_hi - b_lo) / nb_b
        a_centers = [a_lo + (i + 0.5) * a_step for i in range(nb_a)]
        b_centers = [b_lo + (j + 0.5) * b_step for j in range(nb_b)]
        return a_centers, b_centers, [None] * (nb_a * nb_b), a_step, b_step

    a_min = a_range[0] if a_range else df[col_a].min()
    a_max = a_range[1] if a_range else df[col_a].max()
    b_min = b_range[0] if b_range else df[col_b].min()
    b_max = b_range[1] if b_range else df[col_b].max()

    a_full_range = _EPSILON + (a_max - a_min)
    b_full_range = _EPSILON + (b_max - b_min)

    a_step = a_full_range / nb_a
    b_step = b_full_range / nb_b
    bin_area = a_step * b_step

    binned = (
        df.lazy()
        .with_columns(
            **{
                bin_col_a: ((pl.col(col_a) - a_min) // a_step),
                bin_col_b: ((pl.col(col_b) - b_min) // b_step),
            }
        )
        .group_by([bin_col_a, bin_col_b])
        .agg(
            (
                pl.len() if histfunc is None else histfunc_agg_expr(z_col, histfunc)
            ).alias("value")
        )
    )

    all_bins = build_all_bins_grid(nb_a, nb_b, bin_col_a, bin_col_b)

    value_expr = pl.col("value").cast(pl.Float64)
    fill_expr = (
        value_expr.fill_null(fill_value) if fill_value is not None else value_expr
    )

    result = (
        binned.join(all_bins.lazy(), on=[bin_col_a, bin_col_b], how="right")
        .sort([bin_col_b, bin_col_a])
        .select(
            fill_expr,
            a_center=(pl.col(bin_col_a) * a_full_range / nb_a) + a_min + (a_step / 2),
            b_center=(pl.col(bin_col_b) * b_full_range / nb_b) + b_min + (b_step / 2),
        )
        .collect()
    )

    if histnorm is not None:
        result = apply_histnorm(result, "value", histnorm, bin_area)

    # After sort([bin_col_b, bin_col_a]) the layout is regular:
    # rows 0..nb_a-1 have all a-centers at b_center[0]; every nb_a-th row
    # advances to the next b-center.  Direct slicing is O(1) vs sort+unique.
    a_centers = result["a_center"][:nb_a].to_list()
    b_centers = result["b_center"][::nb_a].to_list()
    z_flat = result["value"].to_list()

    return a_centers, b_centers, z_flat, a_step, b_step

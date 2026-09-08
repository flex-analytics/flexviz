"""Shared helpers for 2D histogram traces (Histogram2D, GeoHistogram2D, CorrHeatmap).

Provides histfunc/histnorm constants and type aliases, the ``apply_histnorm``
normalization function, and color-scale/color-range validation helpers. The
heatmap trace classes share them to avoid duplication.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import polars as pl

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

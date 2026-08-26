"""Cross-filter **cube** pre-aggregation (count/sum/mean/min/max measures,
1-D continuous/temporal/categorical free axis).

A cube is a pre-aggregation grouped by a target trace's own dims × the *active*
selection dimension binned to a fixed resolution ``P``. A brush then **slices**
the cube along the free (active) axis instead of rescanning raw data — turning an
O(n) cross-filter recompute into an O(cells) slice.

This module is deliberately **trace-agnostic**: it consumes a ``CubeSpec``
descriptor (free axis + target dims + measure) and never sees a ``trace_type``.
Traces contribute only the descriptor (see ``FlexTrace.get_cube_*_spec``).

Scope (see ``Architecture.md``, "Cube Pre-Aggregation & Live Brushing"):
the count/sum/mean/min/max partial algebra (Phase 2 contract A) plus the
``line_env`` packed envelope measure (Phase 4 contract J) and the corr-partial
measure, a 1-D continuous/temporal/categorical free axis, and
categorical/binned target dims.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence, Tuple

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Descriptor
# ---------------------------------------------------------------------------

FreeAxisKind = Literal["continuous", "temporal", "categorical", "box2d"]
TargetDimKind = Literal["categorical", "binned"]
MeasureAgg = Literal["count", "sum", "mean", "min", "max", "line_env", "corr"]

#: Partial columns per measure: (name, codec dtype) in build/encode order.
#: ``mean`` ships a sum (f64) plus a **non-null** value count (u32); the
#: combine is Σ/Σ and the finalize divides (0-count ⇒ null cell).
#: ``line_env`` is built by the Rust ``fixed_line_envelope2d`` kernel (not by
#: group_by aggregation); its frame holds exact f64 ``y_min``/``x_at_ymin``/
#: ``y_max``/``x_at_ymax`` and the codec packs them as f32 y values plus u16
#: in-bucket x offsets (contract J — quantization at ENCODE time only).
_MEASURE_PARTIALS: dict[str, tuple[tuple[str, str], ...]] = {
    "count": (("count", "u32"),),
    "sum": (("sum", "f64"),),
    "mean": (("sum", "f64"), ("count", "u32")),
    "min": (("min", "f64"),),
    "max": (("max", "f64"),),
    "line_env": (
        ("y_min", "f32"),
        ("y_max", "f32"),
        ("x_argmin", "u16"),
        ("x_argmax", "u16"),
    ),
}

#: Codec dtype → numpy little-endian dtype string.
_DTYPE_NP: dict[str, str] = {"u32": "<u4", "f64": "<f8", "f32": "<f4", "u16": "<u2"}

#: Per-pair statistic suffixes for a ``corr`` cube, in build/encode order:
#: ``n`` (u32 jointly-non-null count) then the five f64 centered sums.
_CORR_STATS: tuple[tuple[str, str], ...] = (
    ("n", "u32"),
    ("sx", "f64"),
    ("sy", "f64"),
    ("sxy", "f64"),
    ("sxx", "f64"),
    ("syy", "f64"),
)


def _corr_pairs(n_cols: int) -> list[tuple[int, int]]:
    """All ``(i, j)`` column-index pairs with ``i < j`` in column order."""
    return [(i, j) for i in range(n_cols) for j in range(i + 1, n_cols)]


def _corr_partial_name(i: int, j: int, stat: str) -> str:
    """Partial-column name for pair ``(i, j)``, statistic ``stat`` (contract I).

    e.g. ``__corr__0_1__sxy``. The double-underscore prefix keeps these out of
    any user/categorical-dim namespace (``_CUBE_RESERVED_COLS`` /
    ``_categorical_dims_ok`` reject ``__``-prefixed columns)."""
    return f"__corr__{i}_{j}__{stat}"


#: Physical units supported for temporal cube axes (contract G).
TemporalUnit = Literal["us", "ms", "day"]


def temporal_unit(dtype: pl.DataType | None) -> str | None:
    """The cube physical unit for a temporal dtype, or ``None`` when the
    dtype is unsupported. ``Datetime("ns")`` is deliberately not supported:
    the string round-trip is µs-precision (documented gate). ``Time`` has no
    date-string representation. Non-temporal dtypes return ``None``."""
    if dtype == pl.Date:
        return "day"
    if isinstance(dtype, pl.Datetime) and dtype.time_unit in ("us", "ms"):
        return dtype.time_unit
    return None


def day_grid(lo: float, hi: float, p: int) -> tuple[int, int]:
    """The integer-day snap grid for a ``unit="day"`` free axis.

    A fixed P=2048 grid would yield fractional-day edges that ``YYYY-MM-DD``
    cannot represent, breaking the round-trip contract. Instead the bin
    width is ``w = max(1, ceil(span_days / p))`` whole days and the bin
    count ``P' = ceil(span_days / w)`` (≤ p) — every snap edge
    ``lo + k*w`` is an integer day, so date strings round-trip bit-exactly.
    All other arithmetic is the shared arithmetic with ``P = P'`` (natural
    floor, filter-don't-clip, degenerate top bin ``P'``)."""
    span = hi - lo
    w = max(1, math.ceil(span / p))
    p_eff = max(0, math.ceil(span / w))
    return w, p_eff


@dataclass(frozen=True)
class FreeAxisSpec:
    """The active (brushed) dimension → the cube's free axis.

    The axis is binned into ``p`` uniform bins over ``domain`` (the source
    figure's viewport, or the full data domain when unzoomed). The brush range
    is snapped to this grid at slice time. A ``"temporal"`` axis bins on the
    column's physical representation (e.g. µs for ``Datetime("us")``, days for
    ``Date``); ``domain`` is in those same physical units.

    Traces may emit ``domain=None`` (= "full data domain", the unzoomed case);
    the engine resolves it to concrete floats before ``build_cube``, which
    requires a resolved domain.

    A ``"categorical"`` axis (bar/pie/treemap source) is not binned at all: the
    free key is the tuple of ``columns`` values (ordered — bar/pie label cols,
    or the full treemap ``path``), with ``columns[0] == column`` (the primary
    column, the ``active_source.column`` join key). It takes no ``domain`` and
    emitters pin ``p=0`` by convention.

    A ``"box2d"`` axis (hist2d source, contract H) is a 2-D rectangular brush:
    ``columns = (x_col, y_col)`` with ``columns[0] == column`` (x is the
    primary ``active_source.column`` join key), ``p = P₂D = 128`` per axis, and
    the per-axis domains live in ``domains = ((lox,hix),(loy,hiy))`` (the
    single-axis ``domain`` stays ``None``). Each axis is binned with the shared
    arithmetic; the composite free bin is ``bin_y * (p+1) + bin_x``.
    ``unit`` is per-axis for box2d — encoded as a 2-tuple ``(unit_x, unit_y)``
    — and is set by the engine from the schema dtypes.
    """

    column: str
    kind: FreeAxisKind = "continuous"
    p: int = 2048
    domain: Tuple[float, float] | None = None
    columns: Tuple[str, ...] | None = None
    # Physical unit for kind="temporal" (contract G); the engine sets it from
    # the schema dtype (Datetime("ns") gates to no cube at all). unit="day"
    # switches the snap grid to integer days (see ``day_grid``). For box2d
    # (contract H) this is a per-axis 2-tuple ``(unit_x, unit_y)`` (each
    # element None or a TemporalUnit), set by the engine.
    unit: Any = None
    # Per-axis domains for kind="box2d" (contract H): ((lox,hix),(loy,hiy)).
    # None until the engine resolves the two viewports / full-data ranges.
    domains: Tuple[Tuple[float, float], Tuple[float, float]] | None = None

    def __post_init__(self) -> None:
        if self.kind == "categorical":
            if not self.columns:
                raise ValueError("categorical free axis requires columns")
            if self.columns[0] != self.column:
                raise ValueError(
                    "categorical free axis: columns[0] must equal column "
                    "(the primary active_source join key)"
                )
            if self.domain is not None:
                raise ValueError("categorical free axis takes no domain")
            if self.domains is not None:
                raise ValueError("categorical free axis takes no domains")
        elif self.kind == "box2d":
            if not self.columns or len(self.columns) != 2:
                raise ValueError("box2d free axis requires (x_col, y_col) columns")
            if self.columns[0] != self.column:
                raise ValueError(
                    "box2d free axis: columns[0] must equal column "
                    "(the primary active_source join key — the x column)"
                )
            if self.domain is not None:
                raise ValueError("box2d free axis takes no single domain (use domains)")
        else:
            if self.columns is not None:
                raise ValueError(f"free axis kind {self.kind!r} takes no columns")
            if self.domains is not None:
                raise ValueError(f"free axis kind {self.kind!r} takes no domains")


@dataclass(frozen=True)
class TargetDimSpec:
    """One target grouping dimension (a categorical column or a binned numeric).

    A binned dim may be emitted by a trace with ``domain=None`` (= "full data
    domain"); the engine resolves it before ``build_cube``, which requires
    resolved domains.

    ``unit`` marks a temporal binned dim (contract G): binning runs on the
    column's physical representation in that unit and the header dim entry
    carries it. The engine sets it from the schema dtype.

    ``bin_variant`` selects the binning arithmetic for a binned dim
    (contract K): ``"hist1d"`` (default) bins bit-equal to the Rust
    ``fixed_hist`` kernel (``scale = n/(hi-lo)``); ``"hist2d"`` reproduces the
    ``fixed_hist2d`` kernel's ``+1e-10`` span epsilon (``scale =
    n/(hi-lo+1e-10)``) so a hist2d target's z-matrix is bit-equal to a server
    delta. The default keeps every existing (line/bar/pie/corr/hist1d)
    descriptor and content key byte-identical.
    """

    column: str
    kind: TargetDimKind
    bins: int | None = None
    domain: Tuple[float, float] | None = None
    unit: str | None = None
    bin_variant: Literal["hist1d", "hist2d"] = "hist1d"

    def __post_init__(self) -> None:
        if self.kind == "binned" and self.bins is None:
            raise ValueError("binned target dim requires bins")


@dataclass(frozen=True)
class MeasureSpec:
    """The cube's measure: an aggregation plus its value column.

    ``value_col`` is required for every agg except ``count`` and ``corr`` and
    must be ``None`` for ``count`` (count aggregates rows, not a column) and
    ``corr`` (corr aggregates column *pairs*, not a single value column).

    ``line_env`` (contract J): ``value_col`` is the line's **y** column; the
    line's x (bucket) axis is the spec's single **binned** target dim, whose
    ``bins``/``domain``/``unit`` define the bucket grid the header ships to
    the client. Any other target dims must be categorical (grouped lines).

    ``corr`` (contract I): ``columns`` is the ordered (≥2) list of numeric
    columns to correlate; ``value_col`` is ``None`` and there are no target
    dims. The build emits decomposable per-pair partials (n, Σx̃, Σỹ, Σx̃ỹ,
    Σx̃², Σỹ²) over mean-centered columns; finalize recovers Pearson r per
    pair. ``columns`` is pinned to user/param order (never sorted) — the order
    determines the matrix labels and the content key.
    """

    agg: MeasureAgg = "count"
    value_col: str | None = None
    columns: Tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.agg == "corr":
            if self.value_col is not None:
                raise ValueError("measure 'corr' takes no value_col")
            if self.columns is None or len(self.columns) < 2:
                raise ValueError("measure 'corr' requires columns (>= 2)")
            return
        if self.columns is not None:
            raise ValueError(f"measure {self.agg!r} takes no columns")
        if self.agg == "count":
            if self.value_col is not None:
                raise ValueError("measure 'count' takes no value_col")
        elif self.value_col is None:
            raise ValueError(f"measure {self.agg!r} requires a value_col")


@dataclass(frozen=True)
class CubeTargetSpec:
    """A target trace's contribution to a cube: its grouping dims + measure.

    Returned by ``FlexTrace.get_cube_target_spec``; the engine combines it with
    the active source's ``FreeAxisSpec`` into a full ``CubeSpec``.
    """

    target_dims: Tuple[TargetDimSpec, ...]
    measure: MeasureSpec = field(default_factory=MeasureSpec)


@dataclass(frozen=True)
class CubeSpec:
    """Everything that determines a cube's contents (its content-address)."""

    source_name: str
    free: FreeAxisSpec
    target_dims: Tuple[TargetDimSpec, ...]
    measure: MeasureSpec = field(default_factory=MeasureSpec)
    # Phase 1: no passive filters. Kept in the identity so later phases that bake
    # passive predicates collide correctly across sessions.
    passive_key: Any = None

    @property
    def target_columns(self) -> Tuple[str, ...]:
        return tuple(d.column for d in self.target_dims)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class CubeResult:
    """A built cube: long-format partials keyed by (target dims…, free_bin).

    ``frame`` columns: one per target dim (raw category value or ``__bin__{col}``
    integer bin index), the free key, and the measure partial columns
    (``_MEASURE_PARTIALS``). For a range (continuous/temporal) free axis the
    free key is ``free_bin`` (Int32); the client slices by combining partials
    over the free bins inside the snapped brush range. Valid free bins are
    ``0..P`` *inclusive*: a value exactly equal to the domain max lands in the
    degenerate top bin ``P`` (natural un-clamped floor, Mosaic-style).

    For a **categorical** free axis the free key is the tuple of typed
    ``__free__{col}`` columns (``free_key_cols``); there is no ``free_bin``
    column at build time — the encoder assigns codes. Slicing selects exact
    category tuples (``slice_agg_keys``).
    """

    spec: CubeSpec
    frame: pl.DataFrame
    # Actual target group-column names in ``frame`` (categorical → raw column;
    # binned → ``__bin__{col}``). Set by ``build_cube``.
    group_cols: Tuple[str, ...] = ()
    # Free-axis key column names in ``frame`` (``__free__{col}``) for a
    # categorical free axis; empty for range axes. Set by ``build_cube``.
    free_key_cols: Tuple[str, ...] = ()
    # corr measure only (contract I): the global per-column means over the
    # passive-filtered build frame, in ``spec.measure.columns`` order. Shipped
    # informationally in the header (finalize is shift-invariant). Set by
    # ``_build_corr_cube``.
    corr_means: Tuple[float, ...] = ()

    @property
    def n_cells(self) -> int:
        return self.frame.height

    def slice_count(self, free_lo: float, free_hi: float) -> pl.DataFrame:
        """Reference (server-side) slice: count per target cell over a free range.

        Mirrors what ``cube.js`` does client-side — provided here for parity tests
        and as the fallback path. ``[free_lo, free_hi]`` is snapped to the P-grid.
        """
        lo_bin, hi_bin = self._snap(free_lo, free_hi)
        tgt = list(self.group_cols)
        sliced = self.frame.filter(pl.col("free_bin").is_between(lo_bin, hi_bin))
        out = sliced.group_by(tgt).agg(pl.col("count").sum().alias("count"))
        return out.sort(tgt) if tgt else out

    def slice_agg(self, free_lo: float, free_hi: float) -> pl.DataFrame:
        """Measure-aware reference slice: combine + finalize partials over a
        snapped free range, returning ``[*group_cols, "value"]``.

        Combine/finalize per the shared contract: ``count``/``sum`` sum their
        partials; ``mean`` sums both partials and divides (a combined count of
        0 finalizes to null → the cell is **omitted**, matching the legacy
        grouped output where labels with no surviving rows are absent);
        ``min``/``max`` skip null partials (all-null cells likewise omitted).
        Mirrors what ``cube.js`` does client-side — provided for parity tests
        and as the fallback path.

        ``line_env`` returns ``[*group_cols, y_min, x_at_ymin, y_max,
        x_at_ymax]`` instead (see ``_combine_line_env``).
        """
        lo_bin, hi_bin = self._snap(free_lo, free_hi)
        sliced = self.frame.filter(pl.col("free_bin").is_between(lo_bin, hi_bin))
        if self.spec.measure.agg == "line_env":
            return self._combine_line_env(sliced)
        if self.spec.measure.agg == "corr":
            return self._combine_corr(sliced)
        return self._combine_partials(sliced)

    def slice_agg_box2d(
        self, x_lo: float, x_hi: float, y_lo: float, y_hi: float
    ) -> pl.DataFrame:
        """Measure-aware reference rectangle slice over a **box2d** free axis
        (contract H): snap each axis to its grid, accumulate every composite
        bin ``by*S + bx`` for ``bx`` in ``[lx..hx]``, ``by`` in ``[ly..hy]``,
        then combine + finalize like ``slice_agg``. Mirrors the client's
        rectangle slice (``fvCubeSliceRect``)."""
        (lx, hx), (ly, hy) = self._snap_box2d(x_lo, x_hi, y_lo, y_hi)
        s = box2d_composite_stride(self.spec.free.p)
        codes: list[int] = []
        for by in range(ly, hy + 1):
            row = by * s
            codes.extend(range(row + lx, row + hx + 1))
        sliced = self.frame.filter(pl.col("free_bin").is_in(codes))
        if self.spec.measure.agg == "line_env":
            return self._combine_line_env(sliced)
        if self.spec.measure.agg == "corr":
            return self._combine_corr(sliced)
        return self._combine_partials(sliced)

    def _snap_box2d(
        self, x_lo: float, x_hi: float, y_lo: float, y_hi: float
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """Per-axis natural-floor bin pairs of the box's corners, each clamped
        to ``[0, p_eff]`` (degenerate top bin included). Mirrors ``fvCubeSnap``
        per axis against each axis's resolved physical domain / day-grid."""
        free = self.spec.free
        (lox, hix), (loy, hiy) = free.domains  # type: ignore[misc]
        unit_x, unit_y = _box2d_units(free)

        def _axis(lo: float, hi: float, p: int, unit: str | None, a: float, b: float):
            if unit == "day":
                w, p_eff = day_grid(lo, hi, p)
                lo_b = max(0, min(p_eff, int((a - lo) / w)))
                hi_b = max(0, min(p_eff, int((b - lo) / w)))
            else:
                span = (hi - lo) or 1.0
                lo_b = max(0, min(p, int((a - lo) / span * p)))
                hi_b = max(0, min(p, int((b - lo) / span * p)))
            if hi_b < lo_b:
                lo_b, hi_b = hi_b, lo_b
            return lo_b, hi_b

        return (
            _axis(lox, hix, free.p, unit_x, x_lo, x_hi),
            _axis(loy, hiy, free.p, unit_y, y_lo, y_hi),
        )

    def corr_matrix(
        self,
        free_lo: float,
        free_hi: float,
        *,
        absolute: bool = False,
        triangular: bool = False,
    ) -> dict[str, list]:
        """Finalize a ``corr`` slice into the heatmap delta shape that mirrors
        ``CorrHeatmap._to_update`` (contract I): diag 1.0, symmetric fill,
        ``absolute``/``triangular`` applied at finalize (display params — NOT
        cube-determining), then rows and y labels reversed. Returns
        ``{"x": cols, "y": reversed(cols), "z": reversed(matrix)}``.

        ``absolute`` and ``triangular`` are passed here (not stored in the
        cube) so two heatmaps differing only in those params share one cube.
        """
        lo_bin, hi_bin = self._snap(free_lo, free_hi)
        sliced = self.frame.filter(pl.col("free_bin").is_between(lo_bin, hi_bin))
        per_pair = self._combine_corr(sliced)
        cols = list(self.spec.measure.columns or ())
        return _corr_assemble_matrix(cols, per_pair, absolute, triangular)

    def slice_agg_keys(self, keys: Sequence[tuple[Any, ...]]) -> pl.DataFrame:
        """Measure-aware reference slice over a **categorical** free axis:
        combine + finalize partials over exact category tuples, returning
        ``[*group_cols, "value"]``.

        ``keys`` are full tuples over ``free_key_cols`` (prefix predicates —
        treemap paths — are expanded to full tuples by the caller). The union
        of keys models an OR'd multi-select. Same combine/finalize semantics
        as ``slice_agg``; mirrors what ``cube.js`` does client-side.
        """
        key_cols = list(self.free_key_cols)
        for key in keys:
            if len(key) != len(key_cols):
                raise ValueError(
                    f"key {key!r} does not match free key arity {len(key_cols)}"
                )
        preds = [
            pl.all_horizontal([pl.col(c) == v for c, v in zip(key_cols, key)])
            for key in keys
        ]
        sliced = (
            self.frame.filter(pl.any_horizontal(*preds))
            if preds
            else self.frame.clear()
        )
        if self.spec.measure.agg == "line_env":
            return self._combine_line_env(sliced)
        if self.spec.measure.agg == "corr":
            return self._combine_corr(sliced)
        return self._combine_partials(sliced)

    def _combine_partials(self, sliced: pl.DataFrame) -> pl.DataFrame:
        """Shared combine + finalize (contract A) over already-sliced cells:
        group by the target cols and reduce the measure partials to ``value``."""
        tgt = list(self.group_cols)
        agg = self.spec.measure.agg
        if agg in ("count", "sum"):
            out = sliced.group_by(tgt).agg(pl.col(agg).sum().alias("value"))
        elif agg == "mean":
            out = (
                sliced.group_by(tgt)
                .agg(pl.col("sum").sum(), pl.col("count").sum())
                .with_columns(
                    pl.when(pl.col("count") > 0)
                    .then(pl.col("sum") / pl.col("count"))
                    .otherwise(None)  # never 0/0 NaN
                    .alias("value")
                )
                .drop("sum", "count")
                .filter(pl.col("value").is_not_null())
            )
        elif agg in ("min", "max"):
            combined = getattr(pl.col(agg), agg)()  # null partials skipped
            out = (
                sliced.group_by(tgt)
                .agg(combined.alias("value"))
                .filter(pl.col("value").is_not_null())
            )
        else:  # pragma: no cover — unreachable past MeasureSpec validation
            raise NotImplementedError(f"measure {agg!r} not supported")
        return out.sort(tgt) if tgt else out

    def _combine_line_env(self, sliced: pl.DataFrame) -> pl.DataFrame:
        """Line-envelope combine (contract J) over already-sliced cells: per
        target cell (group dims × bucket), min/max over the **quantized** f32
        y values — exactly what the client computes from the decoded blob
        (quantize-then-combine equivalence on both sides). NaN y partials are
        skipped; ties keep the earlier ``free_bin`` row (``arg_min``/
        ``arg_max`` return the first occurrence after the free_bin sort).
        Returns ``[*group_cols, y_min, x_at_ymin, y_max, x_at_ymax]`` with x
        dequantized from its u16 in-bucket offset, mirroring the client.
        """
        tgt = list(self.group_cols)
        bucket = _line_env_bucket_dim(self.spec)
        lo, hi = bucket.domain
        width = (hi - lo) / bucket.bins
        q = _line_env_quantized_buffers(sliced, self.spec)
        b = sliced[f"__bin__{bucket.column}"].to_numpy().astype(np.float64)
        bucket_lo = lo + b * width
        key_cols = list(self.free_key_cols) if self.free_key_cols else ["free_bin"]
        quantized = sliced.select(*tgt, *key_cols).with_columns(
            pl.Series("y_min", q["y_min"].astype(np.float64)),
            pl.Series(
                "x_at_ymin",
                bucket_lo + (q["x_argmin"].astype(np.float64) / 65535.0) * width,
            ),
            pl.Series("y_max", q["y_max"].astype(np.float64)),
            pl.Series(
                "x_at_ymax",
                bucket_lo + (q["x_argmax"].astype(np.float64) / 65535.0) * width,
            ),
        )
        out = (
            quantized.filter(pl.col("y_min").is_not_nan(), pl.col("y_max").is_not_nan())
            # Cells are unique per (target cell, free key), so this sort is a
            # total order within every group — ties on quantized y then keep
            # the earlier free-key row via first-occurrence arg_min/arg_max.
            .sort(key_cols)
            .group_by(tgt)
            .agg(
                pl.col("y_min").min(),
                pl.col("x_at_ymin").gather(pl.col("y_min").arg_min()).first(),
                pl.col("y_max").max(),
                pl.col("x_at_ymax").gather(pl.col("y_max").arg_max()).first(),
            )
        )
        return out.sort(tgt)

    def _combine_corr(self, sliced: pl.DataFrame) -> pl.DataFrame:
        """Correlation combine + finalize (contract I) over already-sliced
        cells: Σ each per-pair partial across the sliced free bins (one global
        accumulator per pair — corr has no target dims), then finalize Pearson
        ``r`` per pair. Non-finite (``n < 2`` or zero variance) ⇒ ``0.0``,
        mirroring ``_corr_expr._pack``'s ``None → 0.0``. ``absolute``/
        ``triangular`` are NOT applied here (display params, applied in
        ``corr_matrix``). Returns ``[i, j, r]`` rows (raw r), one per pair in
        column order — the reference shape parity tests compare directly."""
        cols = self.spec.measure.columns or ()
        rows: list[dict] = []
        for i, j in _corr_pairs(len(cols)):
            n = float(sliced[_corr_partial_name(i, j, "n")].sum() or 0)
            sx = float(sliced[_corr_partial_name(i, j, "sx")].sum() or 0.0)
            sy = float(sliced[_corr_partial_name(i, j, "sy")].sum() or 0.0)
            sxy = float(sliced[_corr_partial_name(i, j, "sxy")].sum() or 0.0)
            sxx = float(sliced[_corr_partial_name(i, j, "sxx")].sum() or 0.0)
            syy = float(sliced[_corr_partial_name(i, j, "syy")].sum() or 0.0)
            rows.append({"i": i, "j": j, "r": _corr_finalize(n, sx, sy, sxy, sxx, syy)})
        return pl.DataFrame(
            rows, schema={"i": pl.Int64, "j": pl.Int64, "r": pl.Float64}
        )

    def _snap(self, free_lo: float, free_hi: float) -> tuple[int, int]:
        """Natural-floor bin indices of the brush endpoints, clamped to
        ``[0, P]`` — ``P`` included so a brush reaching the domain max selects
        the degenerate top bin (same arithmetic as the build side)."""
        lo, hi = self.spec.free.domain
        p = self.spec.free.p
        span = (hi - lo) or 1.0
        lo_bin = max(0, min(p, int((free_lo - lo) / span * p)))
        hi_bin = max(0, min(p, int((free_hi - lo) / span * p)))
        if hi_bin < lo_bin:
            lo_bin, hi_bin = hi_bin, lo_bin
        return lo_bin, hi_bin


# ---------------------------------------------------------------------------
# Build (Polars, streaming)
# ---------------------------------------------------------------------------


#: Mirrors ``FIXED_HIST_ROUND_EPS`` in the Rust kernel: keeps values exactly on
#: a bin edge from falling into the previous bin through float rounding.
_FIXED_HIST_ROUND_EPS: float = 1e-9


#: Significand bits kept in a corr centering constant. The streaming engine's
#: parallel f64 reduction combines partials in completion order, so the raw mean
#: is not reproducible run to run; dropping the low bits makes it so. Centering
#: is shift-invariant, so any constant near the mean is equally correct — it
#: only has to be *stable*, which the raw mean is not.
_CENTER_KEEP_BITS: int = 32


def _stable_center(mean: float) -> float:
    """Truncate ``mean`` to a build-reproducible centering constant."""
    if mean == 0.0 or not math.isfinite(mean):
        return 0.0
    frac, exp = math.frexp(mean)
    scale = 1 << _CENTER_KEEP_BITS
    return math.ldexp(math.trunc(frac * scale) / scale, exp)


def _corr_finalize(
    n: float, sx: float, sy: float, sxy: float, sxx: float, syy: float
) -> float:
    """Finalize Pearson ``r`` from the combined centered partials (contract I).

    ``m_x = sx/n``, ``cov = sxy/n − m_x·m_y``, ``var_x = sxx/n − m_x²``
    (likewise y), ``r = cov / sqrt(var_x·var_y)``. Non-finite (``n < 2`` or a
    non-positive variance, i.e. zero-variance column) ⇒ ``0.0``, mirroring
    ``_corr_expr._pack``'s ``None → 0.0``."""
    if n < 2:
        return 0.0
    m_x = sx / n
    m_y = sy / n
    cov = sxy / n - m_x * m_y
    var_x = sxx / n - m_x * m_x
    var_y = syy / n - m_y * m_y
    if var_x <= 0.0 or var_y <= 0.0:
        return 0.0
    r = cov / math.sqrt(var_x * var_y)
    if not math.isfinite(r):
        return 0.0
    return r


def _corr_assemble_matrix(
    cols: list[str],
    per_pair: pl.DataFrame,
    absolute: bool,
    triangular: bool,
) -> dict[str, list]:
    """Assemble a corr heatmap delta from per-pair raw r, mirroring
    ``CorrHeatmap._to_update`` exactly (contract I): diag 1.0, symmetric fill,
    ``absolute ⇒ |r|`` per cell (applied as each pair value is packed, like
    ``_corr_expr``), ``triangular`` nulls the upper half
    (``z[j][i] = None for i in range(j, n)``), then rows + y labels reversed."""
    n = len(cols)
    matrix: list[list] = [[0.0] * n for _ in range(n)]
    for d in range(n):
        matrix[d][d] = 1.0
    for row in per_pair.iter_rows(named=True):
        i, j, r = row["i"], row["j"], float(row["r"])
        if absolute:
            r = abs(r)
        matrix[i][j] = r
        matrix[j][i] = r
    if triangular:
        for j in range(n):
            for i in range(j, n):
                matrix[j][i] = None
    y = list(reversed(cols))
    z = list(reversed(matrix))
    return {"x": list(cols), "y": y, "z": z}


def _free_value_expr(free: FreeAxisSpec) -> pl.Expr:
    """The free-axis value as Float64 (temporal → physical representation)."""
    if free.kind == "temporal":
        return pl.col(free.column).to_physical().cast(pl.Float64)
    return pl.col(free.column)


def _free_bin_expr(value: pl.Expr, lo: float, hi: float, p: int) -> pl.Expr:
    """Natural (un-clamped) floor bin: ``floor((v-lo)/(hi-lo)*p)``.

    Rows are pre-filtered to ``[lo, hi]``, so indices are ``0..p`` inclusive —
    ``p`` being the degenerate top bin for ``v == hi``. Int32 holds bin ``p``
    and matches the descriptor spec's wire type.
    """
    span = (hi - lo) or 1.0
    return ((value - lo) / span * p).floor().cast(pl.Int32).alias("free_bin")


def _day_free_bin_expr(value: pl.Expr, lo: float, w: int) -> pl.Expr:
    """Integer-day grid bin (contract G): ``floor((v - lo) / w)`` with whole-
    day width ``w``. Values and edges are integer days, so the division is
    exact; the degenerate top bin is ``P'`` for ``v == hi``."""
    return ((value - lo) / w).floor().cast(pl.Int32).alias("free_bin")


def _box2d_axis(column: str, unit: str | None) -> pl.Expr:
    """A box2d axis value as Float64 — physical representation for a temporal
    unit (contract G, per axis), the raw column otherwise."""
    val = pl.col(column)
    if unit is not None:
        val = val.to_physical()
    return val.cast(pl.Float64)


def _box2d_axis_bin_expr(value: pl.Expr, lo: float, hi: float, p: int) -> pl.Expr:
    """Natural floor bin on one box2d axis — the shared range arithmetic
    (verbatim ``_free_bin_expr``), un-aliased so the two axes compose into the
    composite ``free_bin``. ``unit="day"`` axes pass ``hi``/``p`` derived from
    ``day_grid`` so the per-axis arithmetic stays the integer-day grid."""
    span = (hi - lo) or 1.0
    return ((value - lo) / span * p).floor().cast(pl.Int32)


def _box2d_units(free: FreeAxisSpec) -> tuple[str | None, str | None]:
    """The per-axis temporal units of a box2d free axis as a 2-tuple. ``unit``
    is stored as a tuple/list ``(unit_x, unit_y)`` (engine-set) or ``None``."""
    u = free.unit
    if u is None:
        return None, None
    return (u[0], u[1])


def _fixed_hist_bin_expr(
    value: pl.Expr, lo: float, hi: float, n: int, alias: str
) -> pl.Expr:
    """Bin index bit-equal to the Rust ``fixed_hist`` kernel for in-domain rows:
    ``min(floor((v-lo)*(n/(hi-lo)) + eps), n-1)`` — same operation order, same
    round-epsilon, same top clamp (the epsilon can push ``v == hi`` to ``n``)."""
    scale = n / (hi - lo) if hi > lo else 0.0
    return (
        ((value.cast(pl.Float64) - lo) * scale + _FIXED_HIST_ROUND_EPS)
        .floor()
        .clip(0, n - 1)
        .cast(pl.Int32)
        .alias(alias)
    )


#: Span epsilon of the Rust ``fixed_hist2d`` kernel (``expressions.rs``): the
#: 2-D kernel adds it to ``(hi - lo)`` before computing the bin scale, so the
#: cube hist2d target must reproduce it to bin bit-equally (contract K).
_FIXED_HIST2D_SPAN_EPS: float = 1e-10


def _fixed_hist2d_bin_expr(
    value: pl.Expr, lo: float, hi: float, n: int, alias: str
) -> pl.Expr:
    """Bin index bit-equal to the Rust ``fixed_hist2d`` kernel for in-domain
    rows (per axis): ``min(floor((v-lo)*(n/(hi-lo+EPS)) + eps), n-1)``.

    Differs from ``_fixed_hist_bin_expr`` ONLY by the ``+1e-10`` span epsilon
    in the scale denominator (the 2-D kernel's ``EPS``); the epsilon makes the
    denominator strictly positive so no ``hi > lo`` guard is needed. Same
    operation order, same round-epsilon, same top clamp."""
    scale = n / (hi - lo + _FIXED_HIST2D_SPAN_EPS)
    return (
        ((value.cast(pl.Float64) - lo) * scale + _FIXED_HIST_ROUND_EPS)
        .floor()
        .clip(0, n - 1)
        .cast(pl.Int32)
        .alias(alias)
    )


def _target_dim_value_expr(d: TargetDimSpec) -> pl.Expr:
    """A binned target dim's value: physical representation for temporal dims
    (contract G), the raw column otherwise."""
    if d.unit is not None:
        return pl.col(d.column).to_physical().cast(pl.Float64)
    return pl.col(d.column)


def _target_group_exprs(
    spec: CubeSpec,
) -> tuple[list[pl.Expr], list[str], list[pl.Expr]]:
    """Return (bin exprs for binned dims, group-by column names, domain filters)."""
    pre: list[pl.Expr] = []
    group_cols: list[str] = []
    filters: list[pl.Expr] = []
    for d in spec.target_dims:
        if d.kind == "categorical":
            group_cols.append(d.column)
        else:  # binned
            assert d.bins is not None and d.domain is not None
            col = f"__bin__{d.column}"
            val = _target_dim_value_expr(d)
            bin_expr = (
                _fixed_hist2d_bin_expr
                if d.bin_variant == "hist2d"
                else _fixed_hist_bin_expr
            )
            pre.append(bin_expr(val, d.domain[0], d.domain[1], d.bins, col))
            filters.append(val.is_between(d.domain[0], d.domain[1]))
            group_cols.append(col)
    return pre, group_cols, filters


def _measure_exprs(measure: MeasureSpec) -> list[pl.Expr]:
    """Partial aggregation expressions per cube cell (shared contract A).

    ``sum``/``min``/``max`` cast the value column to Float64 (the f64 wire
    dtype); ``mean`` ships a sum partial plus the **non-null** value count.
    Polars ``sum`` over an all-null group is 0 (legacy parity); ``min``/``max``
    over an all-null group is null, encoded as NaN and skipped at combine time.
    """
    v = measure.value_col
    if measure.agg == "count":
        return [pl.len().alias("count")]
    if measure.agg == "sum":
        return [pl.col(v).cast(pl.Float64).sum().alias("sum")]
    if measure.agg == "mean":
        return [
            pl.col(v).cast(pl.Float64).sum().alias("sum"),
            pl.col(v).count().alias("count"),
        ]
    if measure.agg == "min":
        return [pl.col(v).cast(pl.Float64).min().alias("min")]
    if measure.agg == "max":
        return [pl.col(v).cast(pl.Float64).max().alias("max")]
    raise NotImplementedError(f"measure {measure.agg!r} not supported")


def _line_env_bucket_dim(spec: CubeSpec) -> TargetDimSpec:
    """The single **binned** target dim of a ``line_env`` cube = the line's x
    bucket axis (contract J). All other target dims must be categorical group
    dims (grouped lines)."""
    binned = [d for d in spec.target_dims if d.kind == "binned"]
    if len(binned) != 1:
        raise ValueError(
            "line_env requires exactly one binned target dim "
            f"(the x bucket axis); got {len(binned)}"
        )
    return binned[0]


def _empty_envelope_frame() -> pl.DataFrame:
    """Zero-row frame with the kernel's static output schema."""
    return pl.DataFrame(
        schema={
            "bucket": pl.UInt32,
            "free_bin": pl.UInt32,
            "y_min": pl.Float64,
            "x_at_ymin": pl.Float64,
            "y_max": pl.Float64,
            "x_at_ymax": pl.Float64,
        }
    )


def _native_envelope_cells(
    base: pl.LazyFrame,
    x_lo: float,
    x_hi: float,
    f_lo: float,
    f_hi: float,
    n_buckets: int,
    p: int,
) -> pl.DataFrame:
    """Streaming ``fixed_line_envelope2d`` replacement for scan sources.

    ``base`` must already be projected to ``__x``/``__y``/``__f`` (Float64,
    day remap applied) — the same frame the kernel path collects. The kernel
    materializes it, which is O(rows) on a scan; this computes identical cells
    in two bounded passes, the same construction as the line trace's
    ``_native_envelope_plan``:

    1. per-cell ``y`` min/max — order-independent aggregations;
    2. rows sitting ON a cell extremum, keeping the FIRST in source order
       (min row index — unique, so no thread-dependent tie-break) and its x.

    That reproduces the kernel's strict-comparison first-row-wins tie rule
    exactly. Row semantics mirror ``envelope_scan``: closed-domain
    ``is_between`` on x and free (NaN fails, null propagates to a dropped
    row), NaN/null y dropped, natural floor bins with NO epsilon and NO clip,
    a domain-max value landing in the degenerate top bin.

    Division hazard: the kernel uses true IEEE division; on this Polars
    version the in-memory engine rewrites division by a constant into
    multiplication by the reciprocal (1 ulp off at exact bin edges) while the
    streaming engine does not. Both collects here are ``engine="streaming"``,
    and the scan-seam tests pin domain-max/bin-edge values so an engine
    fallback or rewrite change breaks loudly instead of shifting bins
    silently.
    """
    x_span = x_hi - x_lo if x_hi > x_lo else 1.0
    f_span = f_hi - f_lo if f_hi > f_lo else 1.0
    stride = n_buckets + 1  # buckets 0..=n_buckets (degenerate top bin)
    n_cells = stride * (p + 1)

    keep = (
        pl.col("__x").is_between(x_lo, x_hi)
        & pl.col("__f").is_between(f_lo, f_hi)
        & pl.col("__y").is_not_nan()
    )
    bx = ((pl.col("__x") - x_lo) / x_span * float(n_buckets)).floor().cast(pl.Int64)
    bf = ((pl.col("__f") - f_lo) / f_span * float(p)).floor().cast(pl.Int64)
    binned = base.filter(keep).with_columns((bf * stride + bx).alias("__cell"))
    y = pl.col("__y")
    # Pass 1 carries no row index: `with_row_index` is an ordered computation
    # the streaming engine must serialize, and only the pick pass needs it.
    ext = (
        binned.group_by("__cell")
        .agg(__lo=y.min(), __hi=y.max())
        .collect(engine="streaming")
    )
    if ext.height == 0:
        return _empty_envelope_frame()

    # Dense per-cell extrema so pass 2 can look them up positionally
    # (`gather(__cell)`); empty cells stay NaN and are unreachable — every row
    # belongs to a cell that contains at least itself. 16 bytes/cell: the
    # cube's own cell-count wall (#47) bounds this long before it matters.
    lo_arr = np.full(n_cells, np.nan)
    hi_arr = np.full(n_cells, np.nan)
    lo_arr[ext["__cell"].to_numpy()] = ext["__lo"].to_numpy()
    hi_arr[ext["__cell"].to_numpy()] = ext["__hi"].to_numpy()
    lo_lit = pl.lit(pl.Series("__lo", lo_arr))
    hi_lit = pl.lit(pl.Series("__hi", hi_arr))

    cell = pl.col("__cell")
    ri = pl.col("__ri")
    x = pl.col("__x")
    at_lo = y == lo_lit.gather(cell)
    at_hi = y == hi_lit.gather(cell)
    picked = (
        binned.with_row_index("__ri")
        .filter(at_lo | at_hi)
        .group_by("__cell")
        .agg(
            __xlo=x.filter(at_lo).min_by(ri.filter(at_lo)),
            __xhi=x.filter(at_hi).min_by(ri.filter(at_hi)),
        )
        .collect(engine="streaming")
    )
    # `__cell = free_bin * stride + bucket`, so sorting by cell IS the
    # kernel's ascending (free_bin, bucket) compaction order.
    return picked.sort("__cell").select(
        (pl.col("__cell") % stride).cast(pl.UInt32).alias("bucket"),
        (pl.col("__cell") // stride).cast(pl.UInt32).alias("free_bin"),
        lo_lit.gather(pl.col("__cell")).alias("y_min"),
        pl.col("__xlo").alias("x_at_ymin"),
        hi_lit.gather(pl.col("__cell")).alias("y_max"),
        pl.col("__xhi").alias("x_at_ymax"),
    )


def _build_line_env_cube(
    ldf: pl.LazyFrame, spec: CubeSpec, scan_source: bool = False
) -> CubeResult:
    """Build a line-envelope cube via the Rust ``fixed_line_envelope2d``
    kernel (contract J): one pass over (x, y, free) producing per
    (bucket, free_bin) cell the exact argmin/argmax-by-y — no group_by
    aggregation. Categorical group dims (grouped lines) run the kernel once
    per partition. Temporal x and free columns run on their physical Float64
    representation (contract G). The kernel returns exact f64; the f32/u16
    quantization happens at encode time only (``_line_env_quantized_buffers``).

    ``scan_source`` is the residency seam: the kernel needs the projected
    frame in memory, so a scan swaps in ``_native_envelope_cells`` — identical
    cells from two bounded streaming passes. Only the numeric-free,
    no-group-dims shape has the streaming form; the categorical variants
    partition a collected frame and still materialize on a scan.
    """
    import flexviz_polars  # noqa: F401 — registers pl.Expr.flexviz namespace

    free = spec.free
    if free.kind == "box2d":
        raise ValueError("line_env does not support a box2d free axis")
    bucket = _line_env_bucket_dim(spec)
    if bucket.domain is None:
        raise ValueError(
            "build_cube requires resolved domains; the engine resolves "
            "domain=None (full data domain) before building"
        )
    cat_cols = [d.column for d in spec.target_dims if d.kind == "categorical"]
    x_lo, x_hi = bucket.domain

    if free.kind == "categorical":
        if not free.columns:
            raise ValueError("categorical free axis requires columns")
        free_cols = list(free.columns)
        free_key_cols = tuple(f"__free__{c}" for c in free_cols)
        part_cols = [*free_key_cols, *cat_cols]
        df = (
            ldf.filter(*[pl.col(c).is_not_null() for c in free_cols])
            .select(
                *[pl.col(c).alias(k) for c, k in zip(free_cols, free_key_cols)],
                *[pl.col(c) for c in cat_cols],
                _target_dim_value_expr(bucket).cast(pl.Float64).alias("__x"),
                pl.col(spec.measure.value_col).cast(pl.Float64).alias("__y"),
                pl.lit(0.0).alias("__f"),
            )
            .collect(engine="streaming")
        )

        def _env_part(part: pl.DataFrame) -> pl.DataFrame:
            # Degenerate 1-bin free axis: every row maps to free_bin 0, so the
            # kernel yields the exact per-x-bucket envelope of this partition.
            return (
                part.select(
                    pl.col("__x").flexviz.fixed_line_envelope2d(
                        pl.col("__y"),
                        pl.col("__f"),
                        pl.lit(float(x_lo)),
                        pl.lit(float(x_hi)),
                        pl.lit(0.0),
                        pl.lit(1.0),
                        bucket.bins,
                        1,
                    )
                )
                .to_series()
                .struct.unnest()
            )

        if df.height:
            frames = [
                _env_part(part).with_columns(
                    *[
                        pl.lit(v, dtype=df.schema[c]).alias(c)
                        for c, v in zip(
                            part_cols, (key if isinstance(key, tuple) else (key,))
                        )
                    ]
                )
                for key, part in df.partition_by(part_cols, as_dict=True).items()
            ]
            frame = pl.concat(frames)
        else:
            frame = _empty_envelope_frame().with_columns(
                *[pl.lit(None, dtype=df.schema[c]).alias(c) for c in part_cols]
            )

        bucket_col = f"__bin__{bucket.column}"
        frame = frame.rename({"bucket": bucket_col}).with_columns(
            pl.col(bucket_col).cast(pl.Int32)
        )
        group_cols = tuple(
            d.column if d.kind == "categorical" else bucket_col
            for d in spec.target_dims
        )
        frame = frame.select(
            *group_cols, *free_key_cols, "y_min", "x_at_ymin", "y_max", "x_at_ymax"
        )
        return CubeResult(
            spec=spec, frame=frame, group_cols=group_cols, free_key_cols=free_key_cols
        )

    if free.domain is None:
        raise ValueError(
            "build_cube requires resolved domains; the engine resolves "
            "domain=None (full data domain) before building"
        )
    f_lo, f_hi = free.domain
    free_val = _free_value_expr(free)
    filters: list[pl.Expr] = []
    if free.unit == "day":
        # The integer-day snap grid (contract G) is not a uniform P-grid over
        # the domain, so the kernel cannot bin raw day values directly. Remap
        # each row to its day-grid bin MIDPOINT ``b + 0.5`` over domain
        # ``[0, P'+1]`` with ``p = P'+1``: ``floor((b+0.5)/(P'+1)*(P'+1)) == b``
        # is robust to the double rounding of the kernel's true division (the
        # value sits half a bin from any edge), and the resulting free_bin
        # equals ``_day_free_bin_expr``'s bit-exactly. Out-of-domain rows are
        # pre-filtered: a row just above the domain max could otherwise remap
        # back into ``0..P'`` (the grid may overshoot the span).
        w, p_eff = day_grid(f_lo, f_hi, free.p)
        day_bin = _day_free_bin_expr(free_val, f_lo, w)
        free_expr = (day_bin.cast(pl.Float64) + 0.5).alias("__f")
        k_lo, k_hi, k_p = 0.0, float(p_eff + 1), p_eff + 1
        filters.append(free_val.is_between(f_lo, f_hi))
    else:
        free_expr = free_val.cast(pl.Float64).alias("__f")
        k_lo, k_hi, k_p = float(f_lo), float(f_hi), free.p

    base = ldf.filter(*filters) if filters else ldf
    proj = base.select(
        *[pl.col(c) for c in cat_cols],
        _target_dim_value_expr(bucket).cast(pl.Float64).alias("__x"),
        pl.col(spec.measure.value_col).cast(pl.Float64).alias("__y"),
        free_expr,
    )

    def _envelope(part: pl.DataFrame) -> pl.DataFrame:
        return (
            part.select(
                pl.col("__x").flexviz.fixed_line_envelope2d(
                    pl.col("__y"),
                    pl.col("__f"),
                    pl.lit(float(x_lo)),
                    pl.lit(float(x_hi)),
                    pl.lit(k_lo),
                    pl.lit(k_hi),
                    bucket.bins,
                    k_p,
                )
            )
            .to_series()
            .struct.unnest()
        )

    if scan_source and not cat_cols:
        frame = _native_envelope_cells(
            proj, float(x_lo), float(x_hi), k_lo, k_hi, bucket.bins, k_p
        )
    elif not cat_cols:
        df = proj.collect(engine="streaming")
        frame = _envelope(df) if df.height else _empty_envelope_frame()
    else:
        df = proj.collect(engine="streaming")
        frames = []
        for key, part in df.partition_by(cat_cols, as_dict=True).items():
            key_t = key if isinstance(key, tuple) else (key,)
            frames.append(
                _envelope(part).with_columns(
                    *[
                        pl.lit(v, dtype=df.schema[c]).alias(c)
                        for c, v in zip(cat_cols, key_t)
                    ]
                )
            )
        if frames:
            frame = pl.concat(frames)
        else:
            frame = _empty_envelope_frame().with_columns(
                *[pl.lit(None, dtype=df.schema[c]).alias(c) for c in cat_cols]
            )

    bucket_col = f"__bin__{bucket.column}"
    frame = frame.rename({"bucket": bucket_col}).with_columns(
        pl.col(bucket_col).cast(pl.Int32), pl.col("free_bin").cast(pl.Int32)
    )
    group_cols = tuple(
        d.column if d.kind == "categorical" else bucket_col for d in spec.target_dims
    )
    frame = frame.select(
        *group_cols, "free_bin", "y_min", "x_at_ymin", "y_max", "x_at_ymax"
    )
    return CubeResult(spec=spec, frame=frame, group_cols=group_cols)


def _build_corr_cube(ldf: pl.LazyFrame, spec: CubeSpec) -> CubeResult:
    """Build a correlation cube (contract I) — empty target dims, per-pair
    mean-centered partials keyed only by ``free_bin``.

    The free axis is whatever the corr source is (a range axis — corr is a
    *target*, its free axis comes from the brushed source). Categorical free
    axes and unresolved domains are rejected like ``line_env``.

    Means pre-pass: one batched collect of ``pl.col(c).mean()`` for every
    column over ``ldf`` — the passive-filtered build frame (contract E). The
    means are kept (informational) AND used to center the partials so the f64
    sums stay precise on large-offset columns (epoch-like values): centering
    is mathematically shift-invariant, but pre-centering avoids the
    catastrophic cancellation a raw ``Σxy − Σx·Σy/n`` one-pass would suffer.

    Per pair ``(i, j)``, ``i < j`` in column order, with pairwise mask
    ``m = col_i.is_not_null() & col_j.is_not_null()`` and centered
    ``x̃ = col_i − mean_i``, ``ỹ = col_j − mean_j``, grouped by ``free_bin``:
    ``n = m.sum()`` (u32), ``sx = (x̃).filter(m).sum()`` (f64), likewise
    ``sy``, ``sxy = (x̃·ỹ).filter(m).sum()``, ``sxx``, ``syy``. ``.filter(m)``
    excludes rows where EITHER column is null (pairwise-complete deletion,
    matching ``pl.corr``)."""
    free = spec.free
    if free.kind == "box2d":
        raise ValueError("corr does not support a box2d free axis")
    if spec.target_dims:
        raise ValueError("corr cube takes no target dims")
    cols = list(spec.measure.columns or ())
    if len(cols) < 2:
        raise ValueError("corr measure requires >= 2 columns")

    # Means pre-pass over the passive-filtered build frame (contract E);
    # shift-invariant, so identical for every free-axis kind.
    #
    # Streaming, like every other collect in this file: the default engine
    # materializes the columns, which is O(rows) and the only reason a corr
    # cube could not be built from a scan (100M rows: 2.73 GB and dead under a
    # 1.5 GiB cap, vs 0.69 GB streaming).
    means_row = ldf.select(
        *[pl.col(c).cast(pl.Float64).mean().alias(c) for c in cols]
    ).collect(engine="streaming")
    means = tuple(
        _stable_center(means_row[c][0]) if means_row[c][0] is not None else 0.0
        for c in cols
    )

    # Per-pair mean-centered partials (free-kind-independent).
    agg_exprs: list[pl.Expr] = []
    centered: dict[int, pl.Expr] = {
        idx: (pl.col(c).cast(pl.Float64) - means[idx]) for idx, c in enumerate(cols)
    }
    for i, j in _corr_pairs(len(cols)):
        ci, cj = cols[i], cols[j]
        m = pl.col(ci).is_not_null() & pl.col(cj).is_not_null()
        xt, yt = centered[i], centered[j]
        agg_exprs.extend(
            [
                m.sum().cast(pl.UInt32).alias(_corr_partial_name(i, j, "n")),
                xt.filter(m).sum().alias(_corr_partial_name(i, j, "sx")),
                yt.filter(m).sum().alias(_corr_partial_name(i, j, "sy")),
                (xt * yt).filter(m).sum().alias(_corr_partial_name(i, j, "sxy")),
                (xt * xt).filter(m).sum().alias(_corr_partial_name(i, j, "sxx")),
                (yt * yt).filter(m).sum().alias(_corr_partial_name(i, j, "syy")),
            ]
        )

    if free.kind == "categorical":
        if not free.columns:
            raise ValueError("categorical free axis requires columns")
        free_key_cols = tuple(f"__free__{c}" for c in free.columns)
        free_exprs = [pl.col(c).alias(f"__free__{c}") for c in free.columns]
        null_filters = [pl.col(c).is_not_null() for c in free.columns]
        frame = (
            ldf.filter(*null_filters)
            .with_columns(*free_exprs)
            .group_by(*free_key_cols)
            .agg(*agg_exprs)
            .collect(engine="streaming")
        )
        return CubeResult(
            spec=spec,
            frame=frame,
            group_cols=(),
            free_key_cols=free_key_cols,
            corr_means=means,
        )

    if free.domain is None:
        raise ValueError(
            "build_cube requires resolved domains; the engine resolves "
            "domain=None (full data domain) before building"
        )
    free_val = _free_value_expr(free)
    lo, hi = free.domain
    if free.unit == "day":
        w, _ = day_grid(lo, hi, free.p)
        free_bin = _day_free_bin_expr(free_val, lo, w)
    else:
        free_bin = _free_bin_expr(free_val, lo, hi, free.p)
    frame = (
        ldf.filter(free_val.is_between(lo, hi))
        .with_columns(free_bin)
        .group_by("free_bin")
        .agg(*agg_exprs)
        .collect(engine="streaming")
    )
    return CubeResult(spec=spec, frame=frame, group_cols=(), corr_means=means)


def box2d_composite_stride(p: int) -> int:
    """The composite-index stride ``S = p + 1`` (contract H): the free bin is
    ``bin_y * S + bin_x``, with ``S`` large enough for the degenerate top bin
    ``p`` on the x axis. ``P₂D = 128 ⇒ S = 129``."""
    return p + 1


def _box2d_axis_grid(
    lo: float, hi: float, p: int, unit: str | None
) -> tuple[int | float, int]:
    """The (width-or-hi, effective-p) per box2d axis. A ``unit="day"`` axis
    uses the integer-day snap grid (contract G) so its arithmetic is
    ``floor((v-lo)/w)`` with whole-day width ``w`` and ``p_eff`` bins; every
    other axis is the uniform P-grid (``floor((v-lo)/(hi-lo)*p)``)."""
    if unit == "day":
        w, p_eff = day_grid(lo, hi, p)
        # Re-express the integer-day grid as a uniform-P arithmetic over
        # [lo, lo + p_eff*w]: floor((v-lo)/(p_eff*w)*p_eff) == floor((v-lo)/w).
        return float(lo + p_eff * w), p_eff
    return float(hi), p


def _build_box2d_cube(ldf: pl.LazyFrame, spec: CubeSpec) -> CubeResult:
    """Build a 2-D box (hist2d source) cube (contract H).

    A range-like build: each of the two axes is binned with the shared
    arithmetic (natural floor, filter-don't-clip via ``is_between`` per axis,
    degenerate top bin ``p`` per axis). The composite free key is
    ``free_bin = bin_y * S + bin_x`` with ``S = p + 1`` (so every composite
    index ``0..S²-1`` is reachable). Temporal axes run on their physical
    representation per axis (contract G), and a ``unit="day"`` axis uses the
    integer-day snap grid. Reuses ``_target_group_exprs`` / ``_measure_exprs``
    (a pure source build has no target dims, but the path stays general).
    """
    free = spec.free
    if free.domains is None:
        raise ValueError(
            "build_cube requires resolved domains; the engine resolves "
            "domains=None (full data domain) before building"
        )
    if any(d.kind == "binned" and d.domain is None for d in spec.target_dims):
        raise ValueError(
            "build_cube requires resolved domains; the engine resolves "
            "domain=None (full data domain) before building"
        )
    cx, cy = free.columns  # type: ignore[misc]
    (lox, hix), (loy, hiy) = free.domains
    unit_x, unit_y = _box2d_units(free)
    p = free.p
    s = box2d_composite_stride(p)

    val_x = _box2d_axis(cx, unit_x)
    val_y = _box2d_axis(cy, unit_y)
    hix_eff, px = _box2d_axis_grid(lox, hix, p, unit_x)
    hiy_eff, py = _box2d_axis_grid(loy, hiy, p, unit_y)

    bin_x = _box2d_axis_bin_expr(val_x, lox, hix_eff, px)
    bin_y = _box2d_axis_bin_expr(val_y, loy, hiy_eff, py)
    free_bin = (bin_y * s + bin_x).cast(pl.Int32).alias("free_bin")

    pre, group_cols, filters = _target_group_exprs(spec)
    frame = (
        ldf.filter(
            val_x.is_between(lox, hix),
            val_y.is_between(loy, hiy),
            *filters,
        )
        .with_columns(*pre, free_bin)
        .group_by(*group_cols, "free_bin")
        .agg(*_measure_exprs(spec.measure))
        .collect(engine="streaming")
    )
    return CubeResult(spec=spec, frame=frame, group_cols=tuple(group_cols))


def cube_target_buildable(free: FreeAxisSpec, measure: MeasureSpec) -> bool:
    """Whether a cube for ``measure`` can be built against ``free``'s axis kind.

    The kernel-based ``line_env`` (contract J) and ``corr`` (contract I) builds
    bin/scan the free axis as a numeric range OR partition it by category, so
    they require a range (continuous/temporal) **or categorical** free axis —
    only a **box2d** (hist2d) source is rejected: a box2d line/corr cube is a
    cell-count/wire-size wall (#47), so those targets fall back to the
    per-commit recompute, exactly like the ``box``/``median`` targets that get
    no cube while compatible targets are still served. Every other agg
    (``count``/``sum``/``mean``/``min``/``max``) works against any free kind.
    """
    if measure.agg in ("line_env", "corr"):
        return free.kind in ("continuous", "temporal", "categorical")
    return True


def build_cube(
    ldf: pl.LazyFrame, spec: CubeSpec, scan_source: bool = False
) -> CubeResult:
    """Build a cube from a (already passive-filtered) LazyFrame.

    Rows outside the free-axis domain or any binned target dim's domain are
    **dropped** (filter, don't clip — clipping would contaminate edge bins on
    zoomed domains). The domain filters are ``is_between`` (closed on both
    ends) and null-propagating, so rows with a null (or NaN) in the free-axis
    column or any binned target column are absent from the cube — consistent
    with the legacy predicate path and the ``fixed_hist`` kernel.

    A **categorical** free axis has no domain (and no domain-resolution
    requirement): the free key is the typed tuple of ``columns`` values aliased
    to ``__free__{col}`` (the alias avoids collision when a target dim uses
    the same column). Rows with a null in **any** free column are dropped —
    consistent with ``is_in`` null semantics. No ``free_bin`` column exists at
    build time; the encoder assigns category codes.

    Streaming engine per the benchmarks (2–4× over the default). The result is
    long-format; empty cells are simply absent (the group_by only emits populated
    groups), which keeps the payload sparse.

    A ``line_env`` measure dispatches to the Rust kernel build path
    (``_build_line_env_cube``) instead of group_by aggregation; on a scan
    source (``scan_source=True``, decided by the caller) it swaps in a bounded
    streaming formulation. All other targets are plain streaming group_bys and
    need no seam.
    """
    if spec.measure.agg == "line_env":
        return _build_line_env_cube(ldf, spec, scan_source)
    if spec.measure.agg == "corr":
        return _build_corr_cube(ldf, spec)
    if spec.free.kind == "box2d":
        return _build_box2d_cube(ldf, spec)
    free = spec.free
    if any(d.kind == "binned" and d.domain is None for d in spec.target_dims):
        raise ValueError(
            "build_cube requires resolved domains; the engine resolves "
            "domain=None (full data domain) before building"
        )
    pre, group_cols, filters = _target_group_exprs(spec)

    if free.kind == "categorical":
        if free.domain is not None:
            raise ValueError("categorical free axis takes no domain")
        if not free.columns:
            raise ValueError("categorical free axis requires columns")
        free_key_cols = tuple(f"__free__{c}" for c in free.columns)
        free_exprs = [pl.col(c).alias(f"__free__{c}") for c in free.columns]
        null_filters = [pl.col(c).is_not_null() for c in free.columns]
        frame = (
            ldf.filter(*null_filters, *filters)
            .with_columns(*pre, *free_exprs)
            .group_by(*group_cols, *free_key_cols)
            .agg(*_measure_exprs(spec.measure))
            .collect(engine="streaming")
        )
        return CubeResult(
            spec=spec,
            frame=frame,
            group_cols=tuple(group_cols),
            free_key_cols=free_key_cols,
        )

    if free.domain is None:
        raise ValueError(
            "build_cube requires resolved domains; the engine resolves "
            "domain=None (full data domain) before building"
        )
    free_val = _free_value_expr(free)
    lo, hi = free.domain
    if free.unit == "day":
        w, _ = day_grid(lo, hi, free.p)
        free_bin = _day_free_bin_expr(free_val, lo, w)
    else:
        free_bin = _free_bin_expr(free_val, lo, hi, free.p)

    frame = (
        ldf.filter(free_val.is_between(lo, hi), *filters)
        .with_columns(*pre, free_bin)
        .group_by(*group_cols, "free_bin")
        .agg(*_measure_exprs(spec.measure))
        .collect(engine="streaming")
    )
    return CubeResult(spec=spec, frame=frame, group_cols=tuple(group_cols))


# ---------------------------------------------------------------------------
# Codec (FVCube v1 binary blob)
# ---------------------------------------------------------------------------
#
# One blob per cube, little-endian throughout:
#
#   offset 0   : magic  b"FVCUBE"            (6 bytes)
#   offset 6   : version u8 = 1
#   offset 7   : reserved u8 = 0
#   offset 8   : header_len u32
#   offset 12  : header — UTF-8 JSON, exactly header_len bytes
#   then       : buffer section; each buffer 8-byte aligned (zero padding
#                between buffers); header offsets are relative to the START of
#                the buffer section.
#
# Dtypes: ``free_bin``, binned-target bin indices, dictionary codes for
# categorical targets, and ``count`` partials are u32 (per-cell counts are
# bounded by 2³²−1; ``encode_fvcube`` raises beyond that). ``sum``/``min``/
# ``max`` partials are f64 (``"<f8"``, null encoded as NaN); ``mean`` ships a
# f64 ``sum`` plus a u32 ``count``. ``line_env`` packs f32 (``"<f4"``) y
# values plus u16 (``"<u2"``) in-bucket x offsets (contract J) — buffers are
# 8-byte aligned throughout and the codec stays version 1.

_FVCUBE_MAGIC = b"FVCUBE"
_FVCUBE_VERSION = 1


def _line_env_quantized_buffers(
    frame: pl.DataFrame, spec: CubeSpec
) -> dict[str, np.ndarray]:
    """Encode-time quantization of the exact f64 envelope (contract J).

    Per cell: ``y_min``/``y_max`` are f32-rounded (numpy float32 cast =
    JS ``Math.fround``); x positions pack as u16 offsets within their bucket:
    ``x_off = rint((x - bucket_lo) / bucket_width * 65535)`` clamped to
    [0, 65535], with ``bucket_width = (hi - lo) / bins`` (true division) and
    ``bucket_lo = lo + b * bucket_width``. The client decode mirrors
    ``x = bucket_lo + (x_off / 65535) * bucket_width``. The degenerate top
    bucket (``b == bins``, only ``x == domain max``) packs to offset 0.
    """
    bucket = _line_env_bucket_dim(spec)
    lo, hi = bucket.domain
    width = (hi - lo) / bucket.bins
    b = frame[f"__bin__{bucket.column}"].to_numpy().astype(np.float64)
    bucket_lo = lo + b * width

    def _off(col: str) -> np.ndarray:
        x = frame[col].to_numpy().astype(np.float64)
        if width > 0:
            off = np.rint((x - bucket_lo) / width * 65535.0)
        else:  # degenerate zero-width bucket domain
            off = np.zeros(len(x))
        return np.clip(off, 0.0, 65535.0).astype("<u2")

    return {
        "y_min": frame["y_min"].to_numpy().astype("<f4"),
        "y_max": frame["y_max"].to_numpy().astype("<f4"),
        "x_argmin": _off("x_at_ymin"),
        "x_argmax": _off("x_at_ymax"),
    }


def encode_fvcube(result: CubeResult, cube_id: str) -> bytes:
    """Encode a built cube as an FVCube v1 blob.

    Rows are sorted by ``free_bin`` ascending, then by the target group
    columns — so the client can build CSR offsets in one pass and the same
    ``CubeResult`` always encodes to identical bytes (``build_cube``'s
    group_by order is nondeterministic). Rebuilding the cube is byte-exact in
    structure only (header, row order, non-f64 buffers): the streaming
    engine's morsel boundaries vary between runs and f64 addition is not
    associative, so ``sum``/``mean``/``corr`` partials can land on different
    last bits. Do not byte-compare, hash, or ETag blobs across builds.
    Categorical target columns are dictionary-encoded: the
    header lists typed numeric categories in numeric order, other categories
    in sorted string order, and the column ships u32 codes into that list.

    A **categorical free axis** is dictionary-encoded the same way: the
    distinct category *tuples* over the ``__free__`` columns are listed in
    Python-sorted order in the header free block, preserving numeric values as
    JSON numbers
    (``{"kind": "categorical", "cols": [...], "categories": [[part, ...], ...]}``
    — no ``p``, no ``domain``) and the standard u32 ``free_bin`` column holds
    the code (= position in that sorted list). The byte layout is otherwise
    unchanged — the codec stays version 1.
    """
    spec = result.spec
    group_cols = list(result.group_cols)
    if spec.free.kind == "categorical":
        key_cols = list(result.free_key_cols)
        row_tuples = (
            list(zip(*(result.frame[c].to_list() for c in key_cols)))
            if result.frame.height
            else []
        )
        categories = sorted(set(row_tuples))
        code_of = {t: i for i, t in enumerate(categories)}
        frame = result.frame.with_columns(
            pl.Series("free_bin", [code_of[t] for t in row_tuples], dtype=pl.UInt32)
        ).sort(["free_bin", *group_cols])
        free_block: dict = {
            "kind": "categorical",
            "cols": list(spec.free.columns or ()),
            "categories": [list(t) for t in categories],
        }
    elif spec.free.kind == "box2d":
        # The composite free_bin is already a u32 (build cast it to Int32); the
        # encode below treats it like any range free_bin. The header free block
        # carries per-axis domains and the composite stride is recovered from
        # "p" (S = p + 1). Per-axis temporal handling mirrors the single-axis
        # block but as 2-element lists: "units" = [unit_x|null, unit_y|null]
        # and "grids" = [{w, p_eff}|null, ...] (a grid entry is present only
        # for a "day" axis). The decoder rebuilds the per-axis grids from these.
        frame = result.frame.sort(["free_bin", *group_cols])
        (lox, hix), (loy, hiy) = spec.free.domains  # type: ignore[misc]
        unit_x, unit_y = _box2d_units(spec.free)
        free_block = {
            "kind": "box2d",
            "cols": list(spec.free.columns or ()),
            "p": spec.free.p,
            "domains": [[lox, hix], [loy, hiy]],
        }
        if unit_x is not None or unit_y is not None:
            free_block["units"] = [unit_x, unit_y]
            grids: list = []
            for unit, lo, hi in ((unit_x, lox, hix), (unit_y, loy, hiy)):
                if unit == "day":
                    w, p_eff = day_grid(lo, hi, spec.free.p)
                    grids.append({"w": w, "p_eff": p_eff})
                else:
                    grids.append(None)
            free_block["grids"] = grids
    else:
        frame = result.frame.sort(["free_bin", *group_cols])
        free_block = {
            "kind": spec.free.kind,
            "p": spec.free.p,
            "domain": list(spec.free.domain),
        }
        if spec.free.unit is not None:
            free_block["unit"] = spec.free.unit
        if spec.free.unit == "day":
            w, p_eff = day_grid(spec.free.domain[0], spec.free.domain[1], spec.free.p)
            free_block["w"] = w
            free_block["p_eff"] = p_eff
    is_corr = spec.measure.agg == "corr"
    if is_corr:
        # corr partials are DYNAMIC (6 per pair) — generated from the pairs,
        # not a fixed ``_MEASURE_PARTIALS`` tuple. The target-dim loop above is
        # a no-op (corr has empty target_dims).
        corr_cols = list(spec.measure.columns or ())
        corr_pairs = _corr_pairs(len(corr_cols))
        partials = tuple(
            (_corr_partial_name(i, j, stat), dtype)
            for (i, j) in corr_pairs
            for (stat, dtype) in _CORR_STATS
        )
    else:
        partials = _MEASURE_PARTIALS[spec.measure.agg]

    if (
        any(name == "count" for name, _ in partials)
        and frame.height
        and frame["count"].max() >= 2**32
    ):
        raise ValueError("FVCube v1 stores counts as u32; a cell count >= 2**32")

    # --- target dim header entries + column buffers ------------------------
    target_dims: list[dict] = []
    buffers: list[tuple[str, bytes, str]] = [
        ("free_bin", frame["free_bin"].to_numpy().astype("<u4").tobytes(), "u32")
    ]
    for d, col in zip(spec.target_dims, group_cols):
        if d.kind == "categorical":
            if frame.schema[col].is_integer() or frame.schema[col].is_float():
                # Numeric categories keep their type and NUMERIC order so the
                # cube's emitted labels byte-match the server's typed,
                # numerically-sorted grouped/ungrouped output. This avoids
                # Python/JS string-format drift for floats such as 1.0 vs 1.
                categories = sorted(frame[col].unique().to_list())
                codes = {v: i for i, v in enumerate(categories)}
                buf = np.fromiter(
                    (codes[v] for v in frame[col].to_list()),
                    dtype="<u4",
                    count=frame.height,
                ).tobytes()
            else:
                categories = sorted(str(v) for v in frame[col].unique().to_list())
                codes = {v: i for i, v in enumerate(categories)}
                buf = np.fromiter(
                    (codes[str(v)] for v in frame[col].to_list()),
                    dtype="<u4",
                    count=frame.height,
                ).tobytes()
            target_dims.append(
                {"name": d.column, "kind": "categorical", "categories": categories}
            )
        else:  # binned — `__bin__{col}` Int32, always >= 0 after domain filters
            dim_entry = {
                "name": d.column,
                "kind": "binned",
                "bins": d.bins,
                "domain": list(d.domain),
            }
            if d.unit is not None:
                dim_entry["unit"] = d.unit
            target_dims.append(dim_entry)
            buf = frame[col].to_numpy().astype("<u4").tobytes()
        buffers.append((col, buf, "u32"))
    if spec.measure.agg == "line_env":
        quantized = _line_env_quantized_buffers(frame, spec)
        for name, dtype in partials:
            buffers.append((name, quantized[name].tobytes(), dtype))
    else:
        for name, dtype in partials:
            arr = frame[name].to_numpy()
            # f64 partials: null → NaN (numpy conversion of nullable floats).
            buf = arr.astype(_DTYPE_NP[dtype]).tobytes()
            buffers.append((name, buf, dtype))

    # --- buffer section: 8-byte aligned offsets ----------------------------
    columns: list[dict] = []
    section = bytearray()
    for name, buf, dtype in buffers:
        pad = -len(section) % 8
        section += b"\x00" * pad
        columns.append(
            {
                "name": name,
                "dtype": dtype,
                "offset": len(section),
                "byte_len": len(buf),
            }
        )
        section += buf

    measure_block: dict = {
        "agg": spec.measure.agg,
        "value_col": spec.measure.value_col,
    }
    if is_corr:
        measure_block["columns"] = corr_cols
        measure_block["pairs"] = [[i, j] for (i, j) in corr_pairs]
        measure_block["means"] = {
            c: result.corr_means[idx] for idx, c in enumerate(corr_cols)
        }
    header = {
        "v": _FVCUBE_VERSION,
        "cube_id": cube_id,
        "rows": frame.height,
        "sorted_by": "free_bin",
        "free": free_block,
        "target_dims": target_dims,
        "measure": measure_block,
        "columns": columns,
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # Pad the header (trailing spaces are valid JSON whitespace) so the buffer
    # section starts 8-byte aligned in the blob — relative offsets then stay
    # aligned in absolute terms too (zero-copy TypedArray views client-side).
    header_bytes += b" " * (-(12 + len(header_bytes)) % 8)

    return (
        _FVCUBE_MAGIC
        + struct.pack("<BBI", _FVCUBE_VERSION, 0, len(header_bytes))
        + header_bytes
        + bytes(section)
    )


def decode_fvcube_header(blob: bytes) -> dict:
    """Validate an FVCube blob's prelude and return the parsed header dict.

    Server/test-side only — the full column decode lives in the JS runtime.
    Raises ``ValueError`` on bad magic, unsupported version, or a truncated
    blob (header or buffer section cut short).
    """
    if len(blob) < 12:
        raise ValueError("FVCube blob truncated: shorter than the 12-byte prelude")
    if blob[:6] != _FVCUBE_MAGIC:
        raise ValueError(f"bad FVCube magic: {blob[:6]!r}")
    version, _reserved, header_len = struct.unpack_from("<BBI", blob, 6)
    if version != _FVCUBE_VERSION:
        raise ValueError(f"unsupported FVCube version: {version}")
    if len(blob) < 12 + header_len:
        raise ValueError("FVCube blob truncated inside the header")
    header = json.loads(blob[12 : 12 + header_len].decode("utf-8"))
    section_len = len(blob) - 12 - header_len
    for col in header["columns"]:
        if col["offset"] + col["byte_len"] > section_len:
            raise ValueError(
                f"FVCube blob truncated: column {col['name']!r} exceeds buffer section"
            )
    return header


# ---------------------------------------------------------------------------
# Cube bundle (multi-blob binary transport envelope)
# ---------------------------------------------------------------------------
#
# A ``cube_request`` serves several FVCube blobs at once (one per distinct
# target cube) plus the ``trace_cubes`` map (target trace uid → blob index).
# Rather than base64-encode each blob into a JSON body — which inflates the
# binary 33% and turns the payload into high-entropy text that gzip then burns
# CPU on — the blobs ride raw inside a thin binary envelope sent as
# ``application/octet-stream``. Little-endian throughout:
#
#   offset 0  : magic   b"FVCBNDL\0"     (8 bytes)
#   offset 8  : version u8 = 1
#   offset 9  : reserved u8 = 0
#   offset 10 : reserved u16 = 0
#   offset 12 : meta_len u32
#   offset 16 : meta — UTF-8 JSON, exactly meta_len bytes, space-padded so the
#               blob section starts 8-byte aligned. Shape:
#               ``{"trace_cubes": {uid: idx}, "lengths": [len0, len1, ...]}``
#   then      : the FVCube blobs concatenated in index order (each ``lengths[i]``
#               bytes, no inter-blob padding — the client copies each out into
#               its own buffer, which restores 8-byte alignment).

_FVCBNDL_MAGIC = b"FVCBNDL\x00"
_FVCBNDL_VERSION = 1


def encode_cube_bundle(blobs: Sequence[bytes], trace_cubes: dict[str, int]) -> bytes:
    """Pack FVCube blobs + the ``trace_cubes`` map into one binary bundle.

    Deterministic for a given (blobs, trace_cubes): the meta JSON preserves the
    map's insertion order and the blobs are concatenated verbatim, so identical
    cube requests produce byte-identical bundles.
    """
    meta = {"trace_cubes": trace_cubes, "lengths": [len(b) for b in blobs]}
    meta_bytes = json.dumps(meta, separators=(",", ":")).encode("utf-8")
    # Pad the meta (trailing JSON whitespace) so the blob section starts 8-byte
    # aligned in the bundle (the 16-byte prelude precedes it).
    meta_bytes += b" " * (-(16 + len(meta_bytes)) % 8)
    prelude = _FVCBNDL_MAGIC + struct.pack(
        "<BBHI", _FVCBNDL_VERSION, 0, 0, len(meta_bytes)
    )
    return prelude + meta_bytes + b"".join(blobs)


def decode_cube_bundle(bundle: bytes) -> tuple[list[bytes], dict[str, int]]:
    """Inverse of ``encode_cube_bundle``: return ``(blobs, trace_cubes)``.

    Server/test-side mirror of the JS ``decodeCubeBundle``. Raises
    ``ValueError`` on bad magic, unsupported version, or a truncated bundle.
    """
    if len(bundle) < 16:
        raise ValueError("cube bundle truncated: shorter than the 16-byte prelude")
    if bundle[:8] != _FVCBNDL_MAGIC:
        raise ValueError(f"bad cube bundle magic: {bundle[:8]!r}")
    version, _r8, _r16, meta_len = struct.unpack_from("<BBHI", bundle, 8)
    if version != _FVCBNDL_VERSION:
        raise ValueError(f"unsupported cube bundle version: {version}")
    if len(bundle) < 16 + meta_len:
        raise ValueError("cube bundle truncated inside the meta block")
    meta = json.loads(bundle[16 : 16 + meta_len].decode("utf-8"))
    blobs: list[bytes] = []
    off = 16 + meta_len
    for length in meta["lengths"]:
        end = off + length
        if end > len(bundle):
            raise ValueError("cube bundle truncated inside the blob section")
        blobs.append(bundle[off:end])
        off = end
    return blobs, meta["trace_cubes"]


# ---------------------------------------------------------------------------
# Content key (cube identity)
# ---------------------------------------------------------------------------


def _measure_content_block(measure: MeasureSpec) -> dict:
    """The ``"m"`` block of a cube content key. Non-corr measures stay exactly
    ``{"a", "v"}`` (byte-identical to Phases 1–4 keys); a ``corr`` measure adds
    ``"cc"`` = the column list in spec/param order (NOT sorted — order
    determines the matrix labels). ``absolute``/``triangular`` are display
    params, NOT cube-determining, so they are absent (``MeasureSpec`` does not
    carry them) — two heatmaps differing only there share one cube."""
    block: dict = {"a": measure.agg, "v": measure.value_col}
    if measure.agg == "corr":
        block["cc"] = list(measure.columns or ())
    return block


def cube_content_key(spec: CubeSpec) -> str:
    """Content-addressed key for a cube. Two specs that determine the same cube
    contents hash identically — so a built cube is cached and shared across
    sessions. Excludes nothing cube-determining; includes the passive key so
    later passive-baked cubes collide correctly."""
    if spec.free.kind == "categorical":
        # "c" is the ordered column list; "p"/"d" pinned to 0/None (contract C).
        free_payload: dict = {
            "c": list(spec.free.columns or ()),
            "k": spec.free.kind,
            "p": 0,
            "d": None,
        }
    elif spec.free.kind == "box2d":
        # "c" is the (x, y) column tuple; "d" is the per-axis domain pair
        # ([dx|null, dy|null]) — contract H. Mirrors the client key's free slot.
        domains = spec.free.domains
        free_payload = {
            "c": list(spec.free.columns or ()),
            "k": spec.free.kind,
            "p": spec.free.p,
            "d": [list(domains[0]), list(domains[1])] if domains else None,
        }
    else:
        free_payload = {
            "c": spec.free.column,
            "k": spec.free.kind,
            "p": spec.free.p,
            "d": list(spec.free.domain) if spec.free.domain else None,
        }
    payload = {
        "s": spec.source_name,
        "free": free_payload,
        "t": [
            {
                "c": d.column,
                "k": d.kind,
                "b": d.bins,
                "d": list(d.domain) if d.domain else None,
                # Only present for non-default bin variants — keeps existing
                # (line/bar/pie/corr/hist1d) keys byte-identical.
                **({"bv": d.bin_variant} if d.bin_variant != "hist1d" else {}),
            }
            for d in spec.target_dims
        ],
        "m": _measure_content_block(spec.measure),
        "p": spec.passive_key,
        "v": 1,  # cube schema version
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.blake2b(canonical.encode("utf-8"), digest_size=16).hexdigest()

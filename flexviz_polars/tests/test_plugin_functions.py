"""Unit tests for flexviz_polars plugin functions: every_nth and arg_min_max."""

from __future__ import annotations

import polars as pl
import pytest

import flexviz_polars  # registers pl.Expr.flexviz namespace

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _every_nth(series: pl.Series, n_points: int) -> pl.Series:
    return pl.select(pl.lit(series).flexviz.every_nth(n_points)).to_series()


def _arg_min_max(series: pl.Series, n_points: int) -> pl.Series:
    return pl.select(pl.lit(series).flexviz.arg_min_max(n_points)).to_series()


def _fpcs(series: pl.Series, n_points: int) -> pl.Series:
    return pl.select(pl.lit(series).flexviz.fpcs(n_points)).to_series()


def _fpcs_line(x: pl.Series, y: pl.Series, n_points: int) -> pl.Series:
    return pl.select(
        flexviz_polars._fpcs_line(
            pl.lit(x), pl.lit(y), n_points, x_name=x.name, y_name=y.name
        )
    ).to_series()


def _minmax_line(x: pl.Series, y: pl.Series, n_points: int) -> pl.Series:
    return pl.select(
        flexviz_polars._minmax_line(
            pl.lit(x), pl.lit(y), n_points, x_name=x.name, y_name=y.name
        )
    ).to_series()


def _fixed_hist(
    series: pl.Series,
    lo: float,
    hi: float,
    n_bins: int,
) -> pl.Series:
    """Call fixed_hist on a plain Series, return the Struct Series of length n_bins."""
    return pl.select(
        pl.lit(series).flexviz.fixed_hist(pl.lit(lo), pl.lit(hi), n_bins)
    ).to_series()


def _fixed_hist_counts(
    series: pl.Series, lo: float, hi: float, n_bins: int
) -> list[int]:
    """Return just the count field as a Python list."""
    return _fixed_hist(series, lo, hi, n_bins).struct.field("count").to_list()


def _fixed_hist_breakpoints(
    series: pl.Series, lo: float, hi: float, n_bins: int
) -> list[float]:
    """Return just the breakpoint field as a Python list."""
    return _fixed_hist(series, lo, hi, n_bins).struct.field("breakpoint").to_list()


def _fixed_hist2d(
    x: pl.Series,
    y: pl.Series,
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
    nb_x: int,
    nb_y: int,
) -> pl.Series:
    """Call fixed_hist2d; return the length-1 Struct Series."""
    return pl.select(
        pl.lit(x).flexviz.fixed_hist2d(
            pl.lit(y),
            pl.lit(x_lo),
            pl.lit(x_hi),
            pl.lit(y_lo),
            pl.lit(y_hi),
            nb_x,
            nb_y,
        )
    ).to_series()


def _fixed_hist2d_counts(
    x: pl.Series,
    y: pl.Series,
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
    nb_x: int,
    nb_y: int,
) -> list[int]:
    """Return z_flat (row-major: yi * nb_x + xi) as a Python list of ints."""
    result = _fixed_hist2d(x, y, x_lo, x_hi, y_lo, y_hi, nb_x, nb_y)
    return result[0]["z_flat"]


def _fixed_hist2d_reduce(
    x: pl.Series,
    y: pl.Series,
    z: pl.Series,
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
    nb_x: int,
    nb_y: int,
    histfunc: str,
) -> pl.Series:
    """Call fixed_hist2d_reduce; return the length-1 Struct Series."""
    return pl.select(
        pl.lit(x).flexviz.fixed_hist2d_reduce(
            pl.lit(y),
            pl.lit(z),
            pl.lit(x_lo),
            pl.lit(x_hi),
            pl.lit(y_lo),
            pl.lit(y_hi),
            nb_x,
            nb_y,
            histfunc,
        )
    ).to_series()


def _fixed_line_envelope2d(
    x: pl.Series,
    y: pl.Series,
    free: pl.Series,
    x_lo: float,
    x_hi: float,
    free_lo: float,
    free_hi: float,
    n_buckets: int,
    p: int,
) -> pl.DataFrame:
    """Call fixed_line_envelope2d; return the unnested long-format DataFrame."""
    return (
        pl.select(
            pl.lit(x).flexviz.fixed_line_envelope2d(
                pl.lit(y),
                pl.lit(free),
                pl.lit(x_lo),
                pl.lit(x_hi),
                pl.lit(free_lo),
                pl.lit(free_hi),
                n_buckets,
                p,
            )
        )
        .to_series()
        .struct.unnest()
    )


def _envelope_reference(
    x: pl.Series,
    y: pl.Series,
    free: pl.Series,
    x_lo: float,
    x_hi: float,
    free_lo: float,
    free_hi: float,
    n_buckets: int,
    p: int,
) -> pl.DataFrame:
    """Pure-Polars reference for fixed_line_envelope2d (contract J, plan
    2026-06-11 — pins kernel parity bit-exactly).

    Shared-arithmetic semantics on BOTH axes:
      * natural floor bin = ``floor((v - lo) / span * n)`` with true IEEE
        division (see the divisor note below); NO epsilon, NO clip;
      * rows outside ``[lo, hi]`` on either axis are FILTERED (not clipped);
      * a value exactly at the domain max lands in the degenerate top bin, so
        indices run ``0..=n_buckets`` and ``0..=p`` inclusive;
      * null or NaN in x, y, or free ⇒ row filtered;
      * ties (equal y within a cell): FIRST row in scan order wins for both
        min and max (Polars arg_min/arg_max return the first occurrence).
    Output: one row per non-empty cell, sorted by (free_bin, bucket).

    The span divisors are materialized as real columns: Polars rewrites
    float division *by a scalar* into multiplication by the reciprocal
    (e.g. ``49.0 / lit(49.0)`` → ``0.999…`` on frames with ≥2 rows), which
    is not IEEE division and can shift a domain-max value out of its
    degenerate top bin. The kernel — like the JS client — uses true
    division, and column/column division in Polars is true division too,
    so dividing by a materialized column pins exactly that.
    """
    x_span = (x_hi - x_lo) or 1.0
    f_span = (free_hi - free_lo) or 1.0
    n_rows = len(x)
    return (
        pl.DataFrame(
            {
                "__x": x.cast(pl.Float64),
                "__y": y.cast(pl.Float64),
                "__f": free.cast(pl.Float64),
                "__xspan": pl.Series([x_span] * n_rows, dtype=pl.Float64),
                "__fspan": pl.Series([f_span] * n_rows, dtype=pl.Float64),
            }
        )
        .lazy()
        .filter(
            pl.col("__x").is_between(x_lo, x_hi),
            pl.col("__f").is_between(free_lo, free_hi),
            pl.col("__y").is_not_null(),
            pl.col("__y").is_not_nan(),
        )
        .with_columns(
            ((pl.col("__x") - x_lo) / pl.col("__xspan") * float(n_buckets))
            .floor()
            .cast(pl.UInt32)
            .alias("bucket"),
            ((pl.col("__f") - free_lo) / pl.col("__fspan") * float(p))
            .floor()
            .cast(pl.UInt32)
            .alias("free_bin"),
        )
        .group_by("bucket", "free_bin")
        .agg(
            pl.col("__y").min().alias("y_min"),
            pl.col("__x").gather(pl.col("__y").arg_min()).first().alias("x_at_ymin"),
            pl.col("__y").max().alias("y_max"),
            pl.col("__x").gather(pl.col("__y").arg_max()).first().alias("x_at_ymax"),
        )
        .sort("free_bin", "bucket")
        .select("bucket", "free_bin", "y_min", "x_at_ymin", "y_max", "x_at_ymax")
        .collect()
        # On zero groups Polars materializes the gather/first aggs as Null
        # dtype (even with an in-agg cast); restore the static schema eagerly.
        .cast({"x_at_ymin": pl.Float64, "x_at_ymax": pl.Float64})
    )


def _fixed_hist2d_reduce_values(
    x: pl.Series,
    y: pl.Series,
    z: pl.Series,
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
    nb_x: int,
    nb_y: int,
    histfunc: str,
) -> list[float | None]:
    result = _fixed_hist2d_reduce(x, y, z, x_lo, x_hi, y_lo, y_hi, nb_x, nb_y, histfunc)
    return result[0]["z_flat"]


# ---------------------------------------------------------------------------
# every_nth
# ---------------------------------------------------------------------------


class TestEveryNth:
    def test_empty_series(self):
        s = pl.Series("x", [], dtype=pl.Int64)
        result = _every_nth(s, n_points=10)
        assert result.is_empty()
        assert result.dtype == pl.Int64

    def test_len_less_than_n_points(self):
        s = pl.Series("x", [1, 2, 3, 4, 5])
        result = _every_nth(s, n_points=100)
        # stride = max(1, 5//100) = 1, so full series returned
        assert result.to_list() == [1, 2, 3, 4, 5]

    def test_len_equal_n_points(self):
        s = pl.Series("x", list(range(10)))
        result = _every_nth(s, n_points=10)
        # stride = max(1, 10//10) = 1
        assert len(result) == 10
        assert result.to_list() == list(range(10))

    def test_stride_2(self):
        s = pl.Series("x", list(range(100)))
        result = _every_nth(s, n_points=50)
        # stride = max(1, 100//50) = 2 → elements at 0, 2, 4, ...
        assert len(result) == 50
        assert result.to_list() == list(range(0, 100, 2))

    def test_output_count_le_n_points(self):
        for n_rows, n_points in [(1000, 200), (999, 100), (1, 50), (500, 500)]:
            s = pl.Series("x", list(range(n_rows)))
            result = _every_nth(s, n_points=n_points)
            assert len(result) <= n_points, f"n_rows={n_rows}, n_points={n_points}"

    def test_output_dtype_preserved_float(self):
        s = pl.Series("x", [1.0, 2.0, 3.0, 4.0], dtype=pl.Float64)
        result = _every_nth(s, n_points=2)
        assert result.dtype == pl.Float64

    def test_output_dtype_preserved_int32(self):
        s = pl.Series("x", [1, 2, 3, 4], dtype=pl.Int32)
        result = _every_nth(s, n_points=2)
        assert result.dtype == pl.Int32

    def test_single_element(self):
        s = pl.Series("x", [42])
        result = _every_nth(s, n_points=10)
        assert result.to_list() == [42]

    def test_large_stride(self):
        s = pl.Series("x", list(range(1000)))
        result = _every_nth(s, n_points=10)
        # stride = 100
        assert len(result) == 10
        assert result[0] == 0
        assert result[1] == 100


# ---------------------------------------------------------------------------
# arg_min_max
# ---------------------------------------------------------------------------


class TestArgMinMax:
    def test_empty_series(self):
        s = pl.Series("y", [], dtype=pl.Float64)
        result = _arg_min_max(s, n_points=10)
        assert result.is_empty()
        assert result.dtype == pl.UInt32

    def test_single_element(self):
        s = pl.Series("y", [5.0])
        result = _arg_min_max(s, n_points=10)
        # One bucket, argmin == argmax == 0, deduped to 1 index
        assert result.to_list() == [0]

    def test_output_dtype_is_uint32(self):
        s = pl.Series("y", [1.0, 2.0, 3.0, 4.0])
        result = _arg_min_max(s, n_points=4)
        assert result.dtype == pl.UInt32

    def test_output_count_le_n_points(self):
        for n_rows, n_points in [(100, 20), (1000, 50), (5, 10), (2, 4)]:
            s = pl.Series("y", list(range(n_rows)), dtype=pl.Float64)
            result = _arg_min_max(s, n_points=n_points)
            assert len(result) <= n_points, f"n_rows={n_rows}, n_points={n_points}"

    def test_output_is_sorted(self):
        s = pl.Series("y", [float(i % 7) for i in range(100)])
        result = _arg_min_max(s, n_points=20)
        assert result.is_sorted(), "indices must be in ascending order"

    def test_dedup_constant_series(self):
        # Constant series: argmin == argmax in every bucket → deduped to n_buckets indices
        s = pl.Series("y", [3.14] * 100)
        n_points = 20
        result = _arg_min_max(s, n_points=n_points)
        n_buckets = n_points // 2  # = 10
        # Each bucket contributes 1 (deduplicated) index → at most n_buckets indices
        assert len(result) <= n_buckets

    def test_spike_preserved(self):
        # Flat series with one clear spike at position 499
        vals = [0.0] * 499 + [1000.0] + [0.0] * 500
        s = pl.Series("y", vals)
        result = _arg_min_max(s, n_points=20)
        # The spike at index 499 must appear as an argmax for its bucket
        assert 499 in result.to_list(), "spike index 499 must be preserved"

    def test_no_duplicate_indices(self):
        s = pl.Series("y", list(range(100)), dtype=pl.Float64)
        result = _arg_min_max(s, n_points=20)
        assert len(result) == len(result.unique()), "output indices must be unique"

    def test_indices_valid_for_gather(self):
        s = pl.Series("y", list(range(50)), dtype=pl.Float64)
        indices = _arg_min_max(s, n_points=10)
        # All indices must be within [0, len(s) - 1]
        assert indices.max() < len(s)
        assert indices.min() >= 0

    def test_monotone_series_preserves_endpoints(self):
        # Monotone increasing: per bucket, min is first element, max is last
        s = pl.Series("y", list(range(100)), dtype=pl.Float64)
        indices = _arg_min_max(s, n_points=20)
        # First bucket: argmin=0, argmax=9 (for 10 buckets of 10)
        idx_list = indices.to_list()
        assert 0 in idx_list, "first element (global min of first bucket) must appear"
        assert 99 in idx_list, "last element (global max of last bucket) must appear"

    def test_float16_output_count_le_n_points(self):
        s = pl.Series("y", [float(i % 17) / 17 for i in range(1000)]).cast(pl.Float16)
        result = _arg_min_max(s, n_points=50)
        assert result.dtype == pl.UInt32
        assert len(result) <= 50
        assert result.is_sorted()

    def test_float16_spike_preserved(self):
        vals = [0.0] * 499 + [10.0] + [0.0] * 500
        s = pl.Series("y", vals).cast(pl.Float16)
        result = _arg_min_max(s, n_points=20)
        assert 499 in result.to_list(), "float16 spike index must be preserved"

    def test_filtered_multichunk_series_skips_empty_chunks(self):
        frames = []
        for chunk_idx in range(20):
            start = chunk_idx * 1000
            frames.append(
                pl.DataFrame(
                    {
                        "x": list(range(start, start + 1000)),
                        "y": [float(i % 100) for i in range(start, start + 1000)],
                    }
                )
            )
        df = pl.concat(frames, rechunk=False)

        result = (
            df.lazy()
            .select(
                pl.col("y")
                .filter(pl.col("x").is_between(5000, 12000))
                .flexviz.arg_min_max(50)
            )
            .collect()
        )

        indices = result["y"].to_list()
        assert indices
        assert min(indices) >= 0
        assert max(indices) <= 7000
        assert len(indices) <= 50


# ---------------------------------------------------------------------------
# fpcs
# ---------------------------------------------------------------------------


class TestFPCS:
    def test_empty_series(self):
        s = pl.Series("y", [], dtype=pl.Float64)
        result = _fpcs(s, n_points=10)
        assert result.is_empty()
        assert result.dtype == pl.UInt32

    def test_one_and_two_rows_return_full_input(self):
        assert _fpcs(pl.Series("y", [5.0]), n_points=3).to_list() == [0]
        assert _fpcs(pl.Series("y", [5.0, 6.0]), n_points=3).to_list() == [0, 1]

    def test_n_points_must_be_at_least_three(self):
        s = pl.Series("y", [1.0, 2.0, 3.0])
        with pytest.raises(Exception, match="at least 3"):
            _fpcs(s, n_points=2)

    def test_len_less_than_n_points_returns_full_input(self):
        s = pl.Series("y", [1.0, 3.0, 2.0, 4.0])
        result = _fpcs(s, n_points=10)
        assert result.to_list() == [0, 1, 2, 3]

    def test_preserves_endpoints(self):
        vals = [float((i * 17) % 101) for i in range(1000)]
        result = _fpcs(pl.Series("y", vals), n_points=100)
        idx = result.to_list()
        assert idx[0] == 0
        assert idx[-1] == len(vals) - 1

    def test_output_can_exceed_target_but_stays_near_two_x(self):
        vals = [0.0]
        for i in range(80):
            vals.extend([10.0 + i, -10.0 - i])
        vals.append(0.0)
        n_points = 20
        result = _fpcs(pl.Series("y", vals), n_points=n_points)
        assert len(result) > n_points
        assert len(result) <= 2 * n_points

    def test_spike_preserved(self):
        vals = [0.0] * 499 + [1000.0] + [0.0] * 500
        result = _fpcs(pl.Series("y", vals), n_points=40)
        assert 499 in result.to_list(), "spike index 499 must be preserved"

    def test_integer_dtype_smoke(self):
        s = pl.Series("y", [i % 13 for i in range(1000)], dtype=pl.Int32)
        result = _fpcs(s, n_points=50)
        assert result.dtype == pl.UInt32
        assert result[0] == 0
        assert result[-1] == len(s) - 1

    def test_float16_dtype_smoke(self):
        s = pl.Series("y", [float(i % 17) / 17 for i in range(1000)]).cast(pl.Float16)
        result = _fpcs(s, n_points=50)
        assert result.dtype == pl.UInt32
        assert result[0] == 0
        assert result[-1] == len(s) - 1

    def test_null_and_nan_smoke(self):
        s = pl.Series("y", [0.0, None, float("nan"), 5.0, -1.0, 2.0] * 50)
        result = _fpcs(s, n_points=20)
        assert result.dtype == pl.UInt32
        assert result[0] == 0
        assert result[-1] == len(s) - 1

    def test_filtered_multichunk_series(self):
        frames = []
        for chunk_idx in range(20):
            start = chunk_idx * 1000
            frames.append(
                pl.DataFrame(
                    {
                        "x": list(range(start, start + 1000)),
                        "y": [float(i % 100) for i in range(start, start + 1000)],
                    }
                )
            )
        df = pl.concat(frames, rechunk=False)

        result = (
            df.lazy()
            .select(
                pl.col("y").filter(pl.col("x").is_between(5000, 12000)).flexviz.fpcs(50)
            )
            .collect()
        )

        indices = result["y"].to_list()
        assert indices
        assert min(indices) >= 0
        assert max(indices) <= 7000
        assert len(indices) <= 100

    def test_no_duplicate_indices(self):
        # Monotone ascending: first bucket's argmin is at index 0, which was
        # already pushed as the start point — guard must prevent a duplicate.
        for vals in [
            list(range(500)),
            [float(i) for i in range(500)],
            [0.0] * 500,
        ]:
            result = _fpcs(pl.Series("y", vals), n_points=40)
            lst = result.to_list()
            assert len(lst) == len(set(lst)), f"duplicate indices for {vals[:4]=}"

    def test_last_window_potential_emitted(self):
        # Spike at the very last bucket: the trailing potential_point must be
        # emitted before the endpoint rather than silently dropped.
        vals = [0.0] * 490 + [1000.0] + [0.0] * 9
        result = _fpcs(pl.Series("y", vals), n_points=40)
        assert (
            490 in result.to_list()
        ), "spike at last-window index 490 must be preserved"

    def test_fpcs_line_matches_gathered_indices(self):
        x = pl.Series("ts", list(range(1000)), dtype=pl.Int64)
        y = pl.Series("val", [float((i * 31) % 97) for i in range(1000)])
        idx = _fpcs(y, n_points=50)
        line = _fpcs_line(x, y, n_points=50).struct.unnest()

        assert line["ts"].to_list() == x.gather(idx).to_list()
        assert line["val"].to_list() == y.gather(idx).to_list()


# ---------------------------------------------------------------------------
# fixed_hist
# ---------------------------------------------------------------------------

_EPS = 1e-10  # matches _HIST_BIN_EPSILON in hist.py


class TestFixedHist:
    # ---- output structure -------------------------------------------------------

    def test_struct_field_names(self):
        s = pl.Series("v", [1.0, 2.0, 3.0], dtype=pl.Float64)
        result = _fixed_hist(s, 0.0, 4.0 + _EPS, n_bins=4)
        assert result.dtype == pl.Struct({"breakpoint": pl.Float64, "count": pl.UInt32})

    def test_output_length_equals_n_bins(self):
        s = pl.Series("v", list(range(100)), dtype=pl.Float64)
        for n_bins in [1, 5, 10, 20]:
            result = _fixed_hist(s, 0.0, 100.0 + _EPS, n_bins=n_bins)
            assert len(result) == n_bins, f"n_bins={n_bins}"

    # ---- correctness ------------------------------------------------------------

    def test_uniform_data_equal_counts(self):
        """100 values spread evenly over 10 bins → each bin has count 10."""
        vals = [float(i) for i in range(100)]  # 0..99
        s = pl.Series("v", vals, dtype=pl.Float64)
        lo, hi = 0.0, 100.0 + _EPS
        counts = _fixed_hist_counts(s, lo, hi, n_bins=10)
        assert counts == [10] * 10

    def test_total_count_equals_non_null_len(self):
        s = pl.Series("v", [1.0, 2.0, None, 4.0, 5.0], dtype=pl.Float64)
        counts = _fixed_hist_counts(s, 0.0, 6.0 + _EPS, n_bins=3)
        assert sum(counts) == 4  # 4 non-null values

    def test_breakpoints_formula(self):
        """breakpoint[i] == lo + (i+1) * step where step = (hi - lo) / n_bins."""
        lo, hi = 2.0, 12.0
        n_bins = 5
        step = (hi - lo) / n_bins
        expected = [lo + (i + 1) * step for i in range(n_bins)]
        bps = _fixed_hist_breakpoints(pl.Series("v", [5.0, 7.0]), lo, hi, n_bins)
        for got, want in zip(bps, expected):
            assert abs(got - want) < 1e-9, f"got={got}, want={want}"

    def test_counts_match_polars_hist(self):
        """Fixed_hist counts must exactly match polars hist(bins=edges) for Float64."""
        import random

        rng = random.Random(42)
        vals = [rng.uniform(0.0, 100.0) for _ in range(500)]
        s = pl.Series("v", vals, dtype=pl.Float64)
        lo, hi = 0.0, 100.0
        n_bins = 20
        eps = _EPS
        hi_eps = hi + eps
        step = (hi_eps - lo) / n_bins
        edges = [lo + i * step for i in range(n_bins + 1)]

        polars_counts = (
            pl.select(pl.lit(s).hist(bins=edges, include_breakpoint=True))
            .to_series()
            .struct.field("count")
            .to_list()
        )
        plugin_counts = _fixed_hist_counts(s, lo, hi_eps, n_bins=n_bins)
        assert plugin_counts == polars_counts

    # ---- dtype support ----------------------------------------------------------

    def test_dtype_int32(self):
        s = pl.Series("v", list(range(50)), dtype=pl.Int32)
        counts = _fixed_hist_counts(s, 0.0, 50.0 + _EPS, n_bins=5)
        assert sum(counts) == 50
        assert counts == [10] * 5

    def test_dtype_int64(self):
        s = pl.Series("v", list(range(50)), dtype=pl.Int64)
        counts = _fixed_hist_counts(s, 0.0, 50.0 + _EPS, n_bins=5)
        assert counts == [10] * 5

    def test_dtype_int8(self):
        s = pl.Series("v", list(range(-10, 10)), dtype=pl.Int8)
        counts = _fixed_hist_counts(s, -10.0, 10.0 + _EPS, n_bins=4)
        assert sum(counts) == 20

    def test_dtype_uint32(self):
        s = pl.Series("v", list(range(20)), dtype=pl.UInt32)
        counts = _fixed_hist_counts(s, 0.0, 20.0 + _EPS, n_bins=4)
        assert counts == [5] * 4

    def test_dtype_float32(self):
        s = pl.Series("v", [float(i) for i in range(40)], dtype=pl.Float32)
        counts = _fixed_hist_counts(s, 0.0, 40.0 + _EPS, n_bins=4)
        assert counts == [10] * 4

    # ---- edge cases -------------------------------------------------------------

    def test_empty_series_all_zero_counts(self):
        s = pl.Series("v", [], dtype=pl.Float64)
        counts = _fixed_hist_counts(s, 0.0, 1.0 + _EPS, n_bins=5)
        assert counts == [0] * 5

    def test_all_nulls_all_zero_counts(self):
        s = pl.Series("v", [None, None, None], dtype=pl.Float64)
        counts = _fixed_hist_counts(s, 0.0, 1.0 + _EPS, n_bins=5)
        assert counts == [0] * 5

    def test_single_bin(self):
        s = pl.Series("v", [1.0, 2.0, 3.0], dtype=pl.Float64)
        counts = _fixed_hist_counts(s, 0.0, 4.0 + _EPS, n_bins=1)
        assert counts == [3]

    def test_values_at_boundary_clamped(self):
        """Values exactly at lo and hi must land in the first and last bin."""
        lo, hi = 0.0, 10.0
        n_bins = 5
        # lo lands in bin 0, hi lands in bin n_bins-1 (after clamping)
        s = pl.Series("v", [lo, hi], dtype=pl.Float64)
        counts = _fixed_hist_counts(s, lo, hi + _EPS, n_bins=n_bins)
        assert counts[0] == 1, "value at lo must be in first bin"
        assert counts[-1] == 1, "value at hi must be in last bin"


# ---------------------------------------------------------------------------
# fixed_hist — the parallel path
# ---------------------------------------------------------------------------

#: The kernel runs single-threaded below this many rows (`MIN_PAR` in
#: expressions.rs). Every other fixed_hist test in this file uses 3-500 rows, so
#: without these the parallel code path is never executed at all.
_MIN_PAR = 1 << 17


def _ref_hist_counts(vals, lo: float, hi: float, n_bins: int) -> list[int]:
    """Reference binning, independent of the kernel.

    Mirrors the Rust arithmetic exactly: ``(v - lo) * n_bins/(hi - lo)`` plus
    ``FIXED_HIST_ROUND_EPS``, truncated, clamped to ``[0, n_bins - 1]``. Rust's
    float->usize cast saturates, so a value below ``lo`` lands in bin 0. Nulls
    and NaN are skipped.

    Deliberately not a comparison against the scalar kernel: this has to keep
    working after the scalar entry point is gone.
    """
    scale = n_bins / (hi - lo)
    counts = [0] * n_bins
    for v in vals:
        if v is None:
            continue
        f = float(v)
        if f != f:  # NaN
            continue
        b = (f - lo) * scale + 1e-9
        counts[0 if b < 0 else min(int(b), n_bins - 1)] += 1
    return counts


class TestFixedHistParallel:
    """Cases at or above `MIN_PAR`, where the rayon path actually runs.

    Each case is checked against `_ref_hist_counts`, and every count vector must
    also sum to the number of non-null, non-NaN inputs.
    """

    @pytest.mark.parametrize("dtype", [pl.Float64, pl.Float32, pl.Int64, pl.Int32])
    def test_dispatched_dtypes(self, dtype):
        n = _MIN_PAR + 1234
        vals = [(i * 7919) % 1000 for i in range(n)]
        s = pl.Series("v", vals, dtype=dtype)
        lo, hi = 0.0, 1000.0 + _EPS
        assert _fixed_hist_counts(s, lo, hi, 256) == _ref_hist_counts(vals, lo, hi, 256)

    @pytest.mark.parametrize("dtype", [pl.UInt8, pl.UInt16])
    def test_undispatched_dtypes_fall_back_correctly(self, dtype):
        """Rarer dtypes take the scalar fallback — still exact."""
        n = _MIN_PAR + 7
        vals = [i % 200 for i in range(n)]
        s = pl.Series("v", vals, dtype=dtype)
        lo, hi = 0.0, 200.0 + _EPS
        assert _fixed_hist_counts(s, lo, hi, 64) == _ref_hist_counts(vals, lo, hi, 64)

    def test_nan_is_skipped(self):
        n = _MIN_PAR + 500
        vals = [float("nan") if i % 1000 == 0 else float(i % 997) for i in range(n)]
        s = pl.Series("v", vals, dtype=pl.Float64)
        lo, hi = 0.0, 997.0 + _EPS
        counts = _fixed_hist_counts(s, lo, hi, 128)
        assert counts == _ref_hist_counts(vals, lo, hi, 128)
        assert sum(counts) == sum(1 for v in vals if v == v)

    def test_nulls_force_the_scalar_fallback_and_stay_exact(self):
        """A chunk with nulls is refused by the parallel path — result unchanged."""
        n = _MIN_PAR + 321
        vals = [None if i % 500 == 0 else float(i % 313) for i in range(n)]
        s = pl.Series("v", vals, dtype=pl.Float64)
        lo, hi = 0.0, 313.0 + _EPS
        counts = _fixed_hist_counts(s, lo, hi, 64)
        assert counts == _ref_hist_counts(vals, lo, hi, 64)
        assert sum(counts) == sum(1 for v in vals if v is not None)

    def test_all_null(self):
        s = pl.Series("v", [None] * (_MIN_PAR + 3), dtype=pl.Float64)
        assert _fixed_hist_counts(s, 0.0, 10.0, 8) == [0] * 8

    @pytest.mark.parametrize("n_chunks_in", [3, 4])
    def test_multi_chunk_matches_single_chunk(self, n_chunks_in):
        """Concatenated frames are the normal case; they must not fall back."""
        n = _MIN_PAR + 999
        vals = [float((i * 31) % 500) for i in range(n)]
        lo, hi = 0.0, 500.0 + _EPS
        one = pl.Series("v", vals, dtype=pl.Float64)
        # Uneven cuts, so the work-splitting sees runs of different sizes.
        cuts = (
            [0, 7, n // 3, n] if n_chunks_in == 3 else [0, 13, n // 5, n // 2 + 11, n]
        )
        many = pl.concat(
            [
                pl.Series("v", vals[a:b], dtype=pl.Float64)
                for a, b in zip(cuts, cuts[1:])
            ]
        )
        assert many.n_chunks() == n_chunks_in
        expected = _ref_hist_counts(vals, lo, hi, 128)
        assert _fixed_hist_counts(one, lo, hi, 128) == expected
        assert _fixed_hist_counts(many, lo, hi, 128) == expected

    def test_many_small_chunks_stay_exact(self):
        """Fragmented input makes more work units than the worker cap; units
        are folded in groups (one private table per group), and the grouping
        must not change a single count."""
        n = 2 * _MIN_PAR
        vals = [float((i * 13) % 353) for i in range(n)]
        lo, hi = 0.0, 353.0 + _EPS
        step = n // 40
        cuts = list(range(0, n, step)) + [n]
        many = pl.concat(
            [
                pl.Series("v", vals[a:b], dtype=pl.Float64)
                for a, b in zip(cuts, cuts[1:])
            ]
        )
        assert many.n_chunks() >= 40
        assert _fixed_hist_counts(many, lo, hi, 64) == _ref_hist_counts(
            vals, lo, hi, 64
        )

    def test_bins_above_budget_fall_back_and_stay_exact(self):
        """One table above MAX_PRIVATE_BYTES (32 MiB / 4 B ~ 8.4M bins) must
        not be multiplied by the worker count — the kernel stays scalar. The
        counts are asserted bin-exact, not just in total."""
        n = _MIN_PAR + 33
        n_bins = 9_000_000
        vals = [float((i * 7) % 1000) for i in range(n)]
        lo, hi = 0.0, 1000.0 + _EPS
        s = pl.Series("v", vals, dtype=pl.Float64)
        assert _fixed_hist_counts(s, lo, hi, n_bins) == _ref_hist_counts(
            vals, lo, hi, n_bins
        )

    @pytest.mark.parametrize("n", [_MIN_PAR - 1, _MIN_PAR, _MIN_PAR + 1])
    def test_threshold_boundary(self, n):
        """Both sides of MIN_PAR must agree — the split is an optimisation."""
        vals = [float(i % 251) for i in range(n)]
        s = pl.Series("v", vals, dtype=pl.Float64)
        lo, hi = 0.0, 251.0 + _EPS
        assert _fixed_hist_counts(s, lo, hi, 32) == _ref_hist_counts(vals, lo, hi, 32)

    def test_values_outside_domain_clamp(self):
        n = _MIN_PAR + 64
        vals = [float(i % 300) - 100.0 for i in range(n)]  # spans [-100, 199]
        s = pl.Series("v", vals, dtype=pl.Float64)
        lo, hi = 0.0, 100.0 + _EPS
        counts = _fixed_hist_counts(s, lo, hi, 10)
        assert counts == _ref_hist_counts(vals, lo, hi, 10)
        assert sum(counts) == n, "out-of-domain values clamp, they are not dropped"

    def test_constant_column(self):
        n = _MIN_PAR + 11
        s = pl.Series("v", [5.0] * n, dtype=pl.Float64)
        counts = _fixed_hist_counts(s, 0.0, 10.0, 4)
        assert sum(counts) == n
        assert counts[2] == n, "5.0 in [0,10) over 4 bins is bin 2"

    def test_degenerate_domain(self):
        """lo == hi: every row lands somewhere, nothing is lost."""
        n = _MIN_PAR + 5
        s = pl.Series("v", [3.0] * n, dtype=pl.Float64)
        assert sum(_fixed_hist_counts(s, 3.0, 3.0, 4)) == n

    def test_single_bin_holds_everything(self):
        n = _MIN_PAR + 17
        vals = [float(i % 1000) for i in range(n)]
        s = pl.Series("v", vals, dtype=pl.Float64)
        assert _fixed_hist_counts(s, 0.0, 1000.0 + _EPS, 1) == [n]


# ---------------------------------------------------------------------------
# fixed_hist2d
# ---------------------------------------------------------------------------

_EPS2D = 1e-10  # matches internal EPS in the fixed_hist2d Rust kernel


class TestFixedHist2D:
    # ---- output structure ---------------------------------------------------

    def test_struct_field_names(self):
        x = pl.Series("x", [1.0, 2.0, 3.0], dtype=pl.Float64)
        y = pl.Series("y", [1.0, 2.0, 3.0], dtype=pl.Float64)
        result = _fixed_hist2d(x, y, 0.0, 4.0, 0.0, 4.0, 2, 2)
        assert result.dtype == pl.Struct(
            {
                "z_flat": pl.List(pl.UInt32),
                "x_lo": pl.Float64,
                "x_hi": pl.Float64,
                "y_lo": pl.Float64,
                "y_hi": pl.Float64,
            }
        )

    def test_output_is_length_one(self):
        x = pl.Series("x", [0.5, 1.5], dtype=pl.Float64)
        y = pl.Series("y", [0.5, 1.5], dtype=pl.Float64)
        result = _fixed_hist2d(x, y, 0.0, 2.0, 0.0, 2.0, 2, 2)
        assert len(result) == 1

    def test_z_flat_length_equals_nb_x_times_nb_y(self):
        x = pl.Series("x", list(range(100)), dtype=pl.Float64)
        y = pl.Series("y", list(range(100)), dtype=pl.Float64)
        for nb_x, nb_y in [(1, 1), (5, 4), (10, 10), (20, 30)]:
            counts = _fixed_hist2d_counts(x, y, 0.0, 100.0, 0.0, 100.0, nb_x, nb_y)
            assert len(counts) == nb_x * nb_y, f"nb_x={nb_x}, nb_y={nb_y}"

    def test_bounds_echo_in_struct(self):
        x = pl.Series("x", [1.0, 2.0], dtype=pl.Float64)
        y = pl.Series("y", [3.0, 4.0], dtype=pl.Float64)
        result = _fixed_hist2d(x, y, 1.0, 5.0, 3.0, 7.0, 2, 2)
        s = result[0]
        assert s["x_lo"] == pytest.approx(1.0)
        assert s["x_hi"] == pytest.approx(5.0)
        assert s["y_lo"] == pytest.approx(3.0)
        assert s["y_hi"] == pytest.approx(7.0)

    # ---- correctness --------------------------------------------------------

    def test_total_count_equals_non_null_len(self):
        x = pl.Series("x", [0.5, 1.5, None, 3.5, 4.5], dtype=pl.Float64)
        y = pl.Series("y", [0.5, 1.5, 2.5, 3.5, 4.5], dtype=pl.Float64)
        counts = _fixed_hist2d_counts(x, y, 0.0, 5.0, 0.0, 5.0, 5, 5)
        assert sum(counts) == 4  # one null in x

    def test_diagonal_data_counts_match_groupby(self):
        """fixed_hist2d must match a polars group_by for f64 uniform data."""
        import random

        rng = random.Random(42)
        n = 400
        x_vals = [rng.uniform(0.0, 10.0) for _ in range(n)]
        y_vals = [rng.uniform(0.0, 10.0) for _ in range(n)]
        x = pl.Series("x", x_vals, dtype=pl.Float64)
        y = pl.Series("y", y_vals, dtype=pl.Float64)

        nb_x, nb_y = 5, 4
        x_lo, x_hi = 0.0, 10.0
        y_lo, y_hi = 0.0, 10.0

        # Plugin result (row-major: z_flat[yi * nb_x + xi])
        plugin_counts = _fixed_hist2d_counts(x, y, x_lo, x_hi, y_lo, y_hi, nb_x, nb_y)

        # Polars native — compute the same bin indices the Rust kernel uses
        x_scale = nb_x / (x_hi - x_lo + _EPS2D)
        y_scale = nb_y / (y_hi - y_lo + _EPS2D)
        df = pl.DataFrame({"x": x, "y": y})
        native = (
            df.with_columns(
                x_bin=((pl.col("x") - x_lo) * x_scale).cast(pl.Int32).clip(0, nb_x - 1),
                y_bin=((pl.col("y") - y_lo) * y_scale).cast(pl.Int32).clip(0, nb_y - 1),
            )
            .group_by(["x_bin", "y_bin"])
            .agg(pl.len().alias("count"))
        )
        count_map = {
            (row["x_bin"], row["y_bin"]): row["count"] for row in native.to_dicts()
        }
        expected = [
            count_map.get((xi, yi), 0) for yi in range(nb_y) for xi in range(nb_x)
        ]
        assert plugin_counts == expected

    def test_uniform_grid_equal_counts(self):
        """100 x 100 grid data into 10×10 bins → each bin gets exactly 100 counts."""
        xs, ys = [], []
        for i in range(10):
            for j in range(10):
                for _ in range(100):
                    xs.append(float(i) + 0.5)
                    ys.append(float(j) + 0.5)
        x = pl.Series("x", xs, dtype=pl.Float64)
        y = pl.Series("y", ys, dtype=pl.Float64)
        counts = _fixed_hist2d_counts(x, y, 0.0, 10.0, 0.0, 10.0, 10, 10)
        assert counts == [100] * 100

    # ---- dtype support ------------------------------------------------------

    def test_dtype_float32(self):
        x = pl.Series("x", [float(i) for i in range(40)], dtype=pl.Float32)
        y = pl.Series("y", [float(i) for i in range(40)], dtype=pl.Float32)
        counts = _fixed_hist2d_counts(x, y, 0.0, 40.0, 0.0, 40.0, 4, 4)
        assert sum(counts) == 40

    def test_dtype_int32(self):
        x = pl.Series("x", list(range(50)), dtype=pl.Int32)
        y = pl.Series("y", list(range(50)), dtype=pl.Int32)
        counts = _fixed_hist2d_counts(x, y, 0.0, 50.0, 0.0, 50.0, 5, 5)
        assert sum(counts) == 50

    def test_dtype_int64(self):
        x = pl.Series("x", list(range(50)), dtype=pl.Int64)
        y = pl.Series("y", list(range(50)), dtype=pl.Int64)
        counts = _fixed_hist2d_counts(x, y, 0.0, 50.0, 0.0, 50.0, 5, 5)
        assert sum(counts) == 50

    def test_dtype_uint32(self):
        x = pl.Series("x", list(range(20)), dtype=pl.UInt32)
        y = pl.Series("y", list(range(20)), dtype=pl.UInt32)
        counts = _fixed_hist2d_counts(x, y, 0.0, 20.0, 0.0, 20.0, 4, 4)
        assert sum(counts) == 20

    def test_mixed_dtypes_fallback(self):
        """Float64 x + Int32 y: falls back to cast path, still correct."""
        x = pl.Series("x", [0.5, 1.5, 2.5], dtype=pl.Float64)
        y = pl.Series("y", [0, 1, 2], dtype=pl.Int32)
        counts = _fixed_hist2d_counts(x, y, 0.0, 3.0, 0, 3, 3, 3)
        assert sum(counts) == 3

    # ---- edge cases ---------------------------------------------------------

    def test_empty_series_all_zero_counts(self):
        x = pl.Series("x", [], dtype=pl.Float64)
        y = pl.Series("y", [], dtype=pl.Float64)
        counts = _fixed_hist2d_counts(x, y, 0.0, 1.0, 0.0, 1.0, 5, 4)
        assert counts == [0] * 20

    def test_all_nulls_all_zero_counts(self):
        x = pl.Series("x", [None, None], dtype=pl.Float64)
        y = pl.Series("y", [None, None], dtype=pl.Float64)
        counts = _fixed_hist2d_counts(x, y, 0.0, 1.0, 0.0, 1.0, 3, 3)
        assert counts == [0] * 9

    def test_single_bin(self):
        x = pl.Series("x", [1.0, 2.0, 3.0], dtype=pl.Float64)
        y = pl.Series("y", [1.0, 2.0, 3.0], dtype=pl.Float64)
        counts = _fixed_hist2d_counts(x, y, 0.0, 4.0, 0.0, 4.0, 1, 1)
        assert counts == [3]

    def test_boundary_clamped(self):
        """Values exactly at lo and hi fall into the first and last bin."""
        lo, hi = 0.0, 10.0
        # x_lo value: should land in x_bin 0
        # x_hi value: scale = nb_x / (10 + EPS), index ≈ nb_x - tiny → clamped to nb_x-1
        x = pl.Series("x", [lo, hi], dtype=pl.Float64)
        y = pl.Series("y", [lo, hi], dtype=pl.Float64)
        counts = _fixed_hist2d_counts(x, y, lo, hi, lo, hi, 5, 5)
        # (lo, lo) → (xi=0, yi=0) → z_flat[0]
        # (hi, hi) → (xi=4, yi=4) → z_flat[4*5+4 = 24]
        assert counts[0] == 1, "lower-left corner should have count 1"
        assert counts[24] == 1, "upper-right corner should have count 1"
        assert sum(counts) == 2

    def test_nan_skipped_like_null(self):
        x = pl.Series("x", [0.5, float("nan"), 1.5], dtype=pl.Float64)
        y = pl.Series("y", [0.5, 0.5, 1.5], dtype=pl.Float64)
        counts = _fixed_hist2d_counts(x, y, 0.0, 2.0, 0.0, 2.0, 2, 2)
        # NaN row skipped; two valid points → total count 2
        assert sum(counts) == 2

    # ---- chunked performance -----------------------------------------------

    def test_multichunk_result_matches_single_chunk(self):
        """Multi-chunk input (no rechunk) must produce the same counts as a single chunk."""
        import random

        rng = random.Random(99)
        n = 2000
        x_vals = [rng.uniform(0.0, 100.0) for _ in range(n)]
        y_vals = [rng.uniform(0.0, 100.0) for _ in range(n)]

        # Single-chunk baseline
        x_single = pl.Series("x", x_vals, dtype=pl.Float64)
        y_single = pl.Series("y", y_vals, dtype=pl.Float64)
        single_counts = _fixed_hist2d_counts(
            x_single, y_single, 0.0, 100.0, 0.0, 100.0, 10, 10
        )

        # Multi-chunk: build via concat without rechunk
        chunk_size = 100
        chunks = [
            pl.DataFrame(
                {
                    "x": pl.Series(
                        "x",
                        x_vals[i * chunk_size : (i + 1) * chunk_size],
                        dtype=pl.Float64,
                    ),
                    "y": pl.Series(
                        "y",
                        y_vals[i * chunk_size : (i + 1) * chunk_size],
                        dtype=pl.Float64,
                    ),
                }
            )
            for i in range(n // chunk_size)
        ]
        multi_df = pl.concat(chunks, rechunk=False)

        multi_counts = (
            multi_df.lazy()
            .select(
                pl.col("x").flexviz.fixed_hist2d(
                    pl.col("y"),
                    pl.lit(0.0),
                    pl.lit(100.0),
                    pl.lit(0.0),
                    pl.lit(100.0),
                    10,
                    10,
                )
            )
            .collect()
            .to_series()[0]["z_flat"]
        )
        assert multi_counts == single_counts

    def test_filtered_multichunk_correct_count(self):
        """After filtering, kernel must count only the surviving rows across chunks."""
        chunk_size = 500
        chunks = [
            pl.DataFrame(
                {
                    "x": pl.Series(
                        "x",
                        [float(i % 10 + 0.5) for i in range(chunk_size)],
                        dtype=pl.Float64,
                    ),
                    "y": pl.Series(
                        "y",
                        [float(i % 10 + 0.5) for i in range(chunk_size)],
                        dtype=pl.Float64,
                    ),
                }
            )
            for _ in range(10)
        ]
        df = pl.concat(chunks, rechunk=False)
        n_total = len(df)  # 5000

        # Filter to x in [0, 5] — keeps rows where x_val < 5.5 (i.e., i%10 in 0..4)
        result = (
            df.lazy()
            .select(
                pl.col("x")
                .filter(pl.col("x") < 5.5)
                .flexviz.fixed_hist2d(
                    pl.col("y").filter(pl.col("x") < 5.5),
                    pl.lit(0.0),
                    pl.lit(10.0),
                    pl.lit(0.0),
                    pl.lit(10.0),
                    10,
                    10,
                )
            )
            .collect()
            .to_series()
        )
        z_flat = result[0]["z_flat"]
        # Exactly half of n_total rows survive (x_val in 0.5, 1.5, 2.5, 3.5, 4.5)
        assert sum(z_flat) == n_total // 2


def _ref_hist2d_counts(xs, ys, x_lo, x_hi, y_lo, y_hi, nb_x, nb_y) -> list[int]:
    """Reference 2D binning, independent of the kernel.

    Mirrors the Rust arithmetic: the span carries `FIXED_HIST2D_SPAN_EPS`
    (added by the kernel, not the caller), the index carries
    `FIXED_HIST_ROUND_EPS`, and both axes clamp — Rust's saturating float->usize
    cast puts anything below `lo` in bin 0. A NaN on either axis drops the row.
    """
    x_scale = nb_x / (x_hi - x_lo + 1e-10)
    y_scale = nb_y / (y_hi - y_lo + 1e-10)
    z = [0] * (nb_x * nb_y)
    for xv, yv in zip(xs, ys):
        if xv is None or yv is None:
            continue
        xf, yf = float(xv), float(yv)
        if xf != xf or yf != yf:
            continue
        bx = (xf - x_lo) * x_scale + 1e-9
        by = (yf - y_lo) * y_scale + 1e-9
        xi = 0 if bx < 0 else min(int(bx), nb_x - 1)
        yi = 0 if by < 0 else min(int(by), nb_y - 1)
        z[yi * nb_x + xi] += 1
    return z


class TestFixedHist2DParallel:
    """Cases at or above `MIN_PAR`, where the rayon 2D path actually runs.

    The other fixed_hist2d tests in this file are all far below that threshold,
    so without these the parallel path is never executed.
    """

    @pytest.mark.parametrize(
        "xdt,ydt",
        [
            (pl.Float64, pl.Float64),
            (pl.Float32, pl.Float32),
            (pl.Int64, pl.Int64),
            (pl.Int32, pl.Float64),  # mixed pair: the kernel dispatches axes apart
            (pl.UInt32, pl.Float64),  # Enum/Categorical physical codes
        ],
    )
    def test_dispatched_pairs(self, xdt, ydt):
        n = _MIN_PAR + 777
        xs = [(i * 13) % 100 for i in range(n)]
        ys = [(i * 7) % 80 for i in range(n)]
        x = pl.Series("x", xs, dtype=xdt)
        y = pl.Series("y", ys, dtype=ydt)
        got = _fixed_hist2d_counts(x, y, 0.0, 100.0, 0.0, 80.0, 32, 16)
        assert got == _ref_hist2d_counts(xs, ys, 0.0, 100.0, 0.0, 80.0, 32, 16)
        assert sum(got) == n

    def test_nan_drops_the_row(self):
        n = _MIN_PAR + 300
        xs = [float("nan") if i % 700 == 0 else float(i % 90) for i in range(n)]
        ys = [float(i % 60) for i in range(n)]
        x = pl.Series("x", xs, dtype=pl.Float64)
        y = pl.Series("y", ys, dtype=pl.Float64)
        got = _fixed_hist2d_counts(x, y, 0.0, 90.0, 0.0, 60.0, 16, 8)
        assert got == _ref_hist2d_counts(xs, ys, 0.0, 90.0, 0.0, 60.0, 16, 8)
        assert sum(got) == sum(1 for v in xs if v == v)

    def test_nulls_force_the_scalar_fallback_and_stay_exact(self):
        n = _MIN_PAR + 55
        xs = [None if i % 400 == 0 else float(i % 90) for i in range(n)]
        ys = [float(i % 60) for i in range(n)]
        x = pl.Series("x", xs, dtype=pl.Float64)
        y = pl.Series("y", ys, dtype=pl.Float64)
        got = _fixed_hist2d_counts(x, y, 0.0, 90.0, 0.0, 60.0, 16, 8)
        assert got == _ref_hist2d_counts(xs, ys, 0.0, 90.0, 0.0, 60.0, 16, 8)
        assert sum(got) == sum(1 for v in xs if v is not None)

    def test_differently_chunked_axes_are_aligned(self):
        """x and y may arrive with different chunk layouts; the kernel aligns."""
        n = _MIN_PAR + 640
        xs = [float((i * 17) % 120) for i in range(n)]
        ys = [float((i * 23) % 90) for i in range(n)]
        xcuts, ycuts = [0, 11, n // 2, n], [0, n // 4, n - 3, n]
        x = pl.concat(
            [
                pl.Series("x", xs[a:b], dtype=pl.Float64)
                for a, b in zip(xcuts, xcuts[1:])
            ]
        )
        y = pl.concat(
            [
                pl.Series("y", ys[a:b], dtype=pl.Float64)
                for a, b in zip(ycuts, ycuts[1:])
            ]
        )
        assert x.n_chunks() > 1 and y.n_chunks() > 1
        got = _fixed_hist2d_counts(x, y, 0.0, 120.0, 0.0, 90.0, 24, 18)
        assert got == _ref_hist2d_counts(xs, ys, 0.0, 120.0, 0.0, 90.0, 24, 18)

    @pytest.mark.parametrize("n", [_MIN_PAR - 1, _MIN_PAR, _MIN_PAR + 1])
    def test_threshold_boundary(self, n):
        xs = [float(i % 70) for i in range(n)]
        ys = [float(i % 50) for i in range(n)]
        x = pl.Series("x", xs, dtype=pl.Float64)
        y = pl.Series("y", ys, dtype=pl.Float64)
        got = _fixed_hist2d_counts(x, y, 0.0, 70.0, 0.0, 50.0, 14, 10)
        assert got == _ref_hist2d_counts(xs, ys, 0.0, 70.0, 0.0, 50.0, 14, 10)

    def test_values_outside_domain_clamp(self):
        n = _MIN_PAR + 128
        xs = [float(i % 200) - 50.0 for i in range(n)]
        ys = [float(i % 150) - 30.0 for i in range(n)]
        x = pl.Series("x", xs, dtype=pl.Float64)
        y = pl.Series("y", ys, dtype=pl.Float64)
        got = _fixed_hist2d_counts(x, y, 0.0, 100.0, 0.0, 80.0, 10, 8)
        assert got == _ref_hist2d_counts(xs, ys, 0.0, 100.0, 0.0, 80.0, 10, 8)
        assert sum(got) == n, "out-of-domain values clamp, they are not dropped"

    def test_raster_shape(self):
        """800x500 = 400k bins — the Mosaic-style raster grid, well past the old
        2^18 per-table cutoff that used to force this shape back to scalar."""
        n = _MIN_PAR + 999
        xs = [float((i * 37) % 800) for i in range(n)]
        ys = [float((i * 41) % 500) for i in range(n)]
        x = pl.Series("x", xs, dtype=pl.Float64)
        y = pl.Series("y", ys, dtype=pl.Float64)
        got = _fixed_hist2d_counts(x, y, 0.0, 800.0, 0.0, 500.0, 800, 500)
        assert got == _ref_hist2d_counts(xs, ys, 0.0, 800.0, 0.0, 500.0, 800, 500)
        assert sum(got) == n

    def test_huge_grid_falls_back_and_stays_exact(self):
        """Above ~4M bins the private tables no longer fit the byte budget, so
        the kernel goes scalar. Documented ceiling — the counts must not move,
        asserted cell-exact (a total alone would pass with every row misbinned).
        """
        n = _MIN_PAR + 32
        xs = [float(i % 2049) for i in range(n)]
        ys = [float(i % 2048) for i in range(n)]
        x = pl.Series("x", xs, dtype=pl.Float64)
        y = pl.Series("y", ys, dtype=pl.Float64)
        got = _fixed_hist2d_counts(x, y, 0.0, 2049.0, 0.0, 2048.0, 2049, 2048)
        assert got == _ref_hist2d_counts(xs, ys, 0.0, 2049.0, 0.0, 2048.0, 2049, 2048)

    def test_many_small_chunks_stay_exact(self):
        """Fragmented input makes more work units than the worker cap; units
        are folded in groups (one private table per group), and the grouping
        must not change a single count."""
        n = 2 * _MIN_PAR
        xs = [float((i * 17) % 120) for i in range(n)]
        ys = [float((i * 23) % 90) for i in range(n)]
        step = n // 40
        cuts = list(range(0, n, step)) + [n]
        x = pl.concat(
            [pl.Series("x", xs[a:b], dtype=pl.Float64) for a, b in zip(cuts, cuts[1:])]
        )
        y = pl.concat(
            [pl.Series("y", ys[a:b], dtype=pl.Float64) for a, b in zip(cuts, cuts[1:])]
        )
        assert x.n_chunks() >= 40
        got = _fixed_hist2d_counts(x, y, 0.0, 120.0, 0.0, 90.0, 24, 18)
        assert got == _ref_hist2d_counts(xs, ys, 0.0, 120.0, 0.0, 90.0, 24, 18)

    def test_single_cell_holds_everything(self):
        n = _MIN_PAR + 9
        xs = [float(i % 500) for i in range(n)]
        ys = [float(i % 300) for i in range(n)]
        x = pl.Series("x", xs, dtype=pl.Float64)
        y = pl.Series("y", ys, dtype=pl.Float64)
        assert _fixed_hist2d_counts(x, y, 0.0, 500.0, 0.0, 300.0, 1, 1) == [n]


class TestFixedHist2DReduce:
    def test_struct_field_names(self):
        x = pl.Series("x", [0.5, 1.5], dtype=pl.Float64)
        y = pl.Series("y", [0.5, 1.5], dtype=pl.Float64)
        z = pl.Series("z", [2.0, 4.0], dtype=pl.Float64)
        result = _fixed_hist2d_reduce(x, y, z, 0.0, 2.0, 0.0, 2.0, 2, 2, "sum")
        assert result.dtype == pl.Struct(
            {
                "z_flat": pl.List(pl.Float64),
                "x_lo": pl.Float64,
                "x_hi": pl.Float64,
                "y_lo": pl.Float64,
                "y_hi": pl.Float64,
            }
        )

    @pytest.mark.parametrize(
        ("histfunc", "expected"),
        [
            ("sum", [3.0, None, None, 12.0]),
            ("mean", [1.5, None, None, 6.0]),
            ("min", [1.0, None, None, 4.0]),
            ("max", [2.0, None, None, 8.0]),
        ],
    )
    def test_reducers_simple_grid(self, histfunc, expected):
        x = pl.Series("x", [0.25, 0.25, 1.25, 1.25], dtype=pl.Float64)
        y = pl.Series("y", [0.25, 0.25, 1.25, 1.25], dtype=pl.Float64)
        z = pl.Series("z", [1.0, 2.0, 4.0, 8.0], dtype=pl.Float64)
        values = _fixed_hist2d_reduce_values(
            x, y, z, 0.0, 2.0, 0.0, 2.0, 2, 2, histfunc
        )
        assert values == expected

    def test_sum_zero_value_is_not_empty(self):
        x = pl.Series("x", [0.25, 1.25], dtype=pl.Float64)
        y = pl.Series("y", [0.25, 1.25], dtype=pl.Float64)
        z = pl.Series("z", [0.0, 5.0], dtype=pl.Float64)
        values = _fixed_hist2d_reduce_values(x, y, z, 0.0, 2.0, 0.0, 2.0, 2, 2, "sum")
        assert values == [0.0, None, None, 5.0]

    def test_null_and_nan_rows_are_skipped(self):
        x = pl.Series("x", [0.25, None, 1.25, 1.25], dtype=pl.Float64)
        y = pl.Series("y", [0.25, 0.25, float("nan"), 1.25], dtype=pl.Float64)
        z = pl.Series("z", [2.0, 3.0, 4.0, float("nan")], dtype=pl.Float64)
        values = _fixed_hist2d_reduce_values(x, y, z, 0.0, 2.0, 0.0, 2.0, 2, 2, "sum")
        assert values == [2.0, None, None, None]

    def test_integer_z_is_accepted(self):
        x = pl.Series("x", [0.25, 0.25, 1.25], dtype=pl.Float64)
        y = pl.Series("y", [0.25, 0.25, 1.25], dtype=pl.Float64)
        z = pl.Series("z", [1, 2, 3], dtype=pl.Int64)
        values = _fixed_hist2d_reduce_values(x, y, z, 0.0, 2.0, 0.0, 2.0, 2, 2, "sum")
        assert values == [3.0, None, None, 3.0]

    def test_invalid_histfunc_raises(self):
        x = pl.Series("x", [0.25], dtype=pl.Float64)
        y = pl.Series("y", [0.25], dtype=pl.Float64)
        z = pl.Series("z", [1.0], dtype=pl.Float64)
        with pytest.raises(Exception, match="histfunc"):
            _fixed_hist2d_reduce_values(x, y, z, 0.0, 1.0, 0.0, 1.0, 1, 1, "median")


# ---------------------------------------------------------------------------
# fixed_line_envelope2d
# ---------------------------------------------------------------------------


_ENVELOPE_SCHEMA = pl.Struct(
    {
        "bucket": pl.UInt32,
        "free_bin": pl.UInt32,
        "y_min": pl.Float64,
        "x_at_ymin": pl.Float64,
        "y_max": pl.Float64,
        "x_at_ymax": pl.Float64,
    }
)


def _assert_envelope_parity(
    x: pl.Series,
    y: pl.Series,
    free: pl.Series,
    x_lo: float,
    x_hi: float,
    free_lo: float,
    free_hi: float,
    n_buckets: int,
    p: int,
) -> pl.DataFrame:
    """Kernel output must be bit-exactly equal to the pure-Polars reference."""
    from polars.testing import assert_frame_equal

    got = _fixed_line_envelope2d(x, y, free, x_lo, x_hi, free_lo, free_hi, n_buckets, p)
    want = _envelope_reference(x, y, free, x_lo, x_hi, free_lo, free_hi, n_buckets, p)
    assert_frame_equal(got, want, check_exact=True)
    return got


class TestFixedLineEnvelope2D:
    # ---- output structure ---------------------------------------------------

    def test_struct_field_names(self):
        x = pl.Series("x", [0.5, 1.5], dtype=pl.Float64)
        y = pl.Series("y", [1.0, 2.0], dtype=pl.Float64)
        f = pl.Series("f", [0.5, 1.5], dtype=pl.Float64)
        result = pl.select(
            pl.lit(x).flexviz.fixed_line_envelope2d(
                pl.lit(y),
                pl.lit(f),
                pl.lit(0.0),
                pl.lit(2.0),
                pl.lit(0.0),
                pl.lit(2.0),
                2,
                2,
            )
        ).to_series()
        assert result.dtype == _ENVELOPE_SCHEMA

    def test_rows_sorted_by_free_bin_then_bucket(self):
        import random

        rng = random.Random(3)
        n = 500
        x = pl.Series("x", [rng.uniform(0.0, 10.0) for _ in range(n)])
        y = pl.Series("y", [rng.gauss(0.0, 1.0) for _ in range(n)])
        f = pl.Series("f", [rng.uniform(0.0, 5.0) for _ in range(n)])
        out = _fixed_line_envelope2d(x, y, f, 0.0, 10.0, 0.0, 5.0, 8, 6)
        keys = list(zip(out["free_bin"].to_list(), out["bucket"].to_list()))
        assert keys == sorted(keys), "rows must be sorted by (free_bin, bucket)"
        assert len(keys) == len(set(keys)), "one row per cell"

    # ---- parity with the pure-Polars reference -------------------------------

    def test_dense_data_parity(self):
        import random

        rng = random.Random(7)
        n = 2000
        x = pl.Series("x", [rng.uniform(0.0, 100.0) for _ in range(n)])
        y = pl.Series("y", [rng.gauss(0.0, 10.0) for _ in range(n)])
        f = pl.Series("f", [rng.uniform(0.0, 10.0) for _ in range(n)])
        out = _assert_envelope_parity(x, y, f, 0.0, 100.0, 0.0, 10.0, 8, 16)
        # Dense data: nearly every cell populated.
        assert out.height > 100

    def test_sparse_data_empty_cells_absent(self):
        # Two isolated points → exactly two cells; everything else absent.
        x = pl.Series("x", [0.5, 9.5], dtype=pl.Float64)
        y = pl.Series("y", [1.0, 2.0], dtype=pl.Float64)
        f = pl.Series("f", [0.5, 7.5], dtype=pl.Float64)
        out = _assert_envelope_parity(x, y, f, 0.0, 10.0, 0.0, 8.0, 10, 8)
        assert out.height == 2

    def test_nulls_in_x_y_free_filtered(self):
        x = pl.Series("x", [0.5, None, 1.5, 2.5, 3.5], dtype=pl.Float64)
        y = pl.Series("y", [1.0, 2.0, None, 3.0, 4.0], dtype=pl.Float64)
        f = pl.Series("f", [0.5, 0.5, 0.5, None, 1.5], dtype=pl.Float64)
        out = _assert_envelope_parity(x, y, f, 0.0, 4.0, 0.0, 2.0, 4, 2)
        # Only rows 0 and 4 survive (each its own cell).
        assert out.height == 2
        assert out["y_min"].to_list() == [1.0, 4.0]

    def test_nan_rows_filtered(self):
        nan = float("nan")
        x = pl.Series("x", [0.5, nan, 1.5, 2.5, 3.5], dtype=pl.Float64)
        y = pl.Series("y", [1.0, 2.0, nan, 3.0, 4.0], dtype=pl.Float64)
        f = pl.Series("f", [0.5, 0.5, 0.5, nan, 1.5], dtype=pl.Float64)
        out = _assert_envelope_parity(x, y, f, 0.0, 4.0, 0.0, 2.0, 4, 2)
        assert out.height == 2
        assert out["y_min"].to_list() == [1.0, 4.0]

    def test_degenerate_top_bin_both_axes(self):
        # Values exactly at the domain max land in bin n_buckets / bin p.
        x = pl.Series("x", [0.0, 10.0, 10.0], dtype=pl.Float64)
        y = pl.Series("y", [1.0, 2.0, 3.0], dtype=pl.Float64)
        f = pl.Series("f", [0.0, 0.0, 5.0], dtype=pl.Float64)
        out = _assert_envelope_parity(x, y, f, 0.0, 10.0, 0.0, 5.0, 5, 4)
        cells = set(zip(out["bucket"].to_list(), out["free_bin"].to_list()))
        assert (0, 0) in cells
        assert (5, 0) in cells, "x == x_hi must land in degenerate bucket 5"
        assert (5, 4) in cells, "free == free_hi must land in degenerate bin 4"

    def test_out_of_domain_filtered_not_clipped(self):
        # Out-of-domain rows vanish entirely instead of contaminating edge bins.
        x = pl.Series("x", [-0.1, 5.0, 10.1], dtype=pl.Float64)
        y = pl.Series("y", [100.0, 1.0, 100.0], dtype=pl.Float64)
        f = pl.Series("f", [1.0, 1.0, 1.0], dtype=pl.Float64)
        out = _assert_envelope_parity(x, y, f, 0.0, 10.0, 0.0, 2.0, 4, 2)
        assert out.height == 1
        assert out["y_min"].to_list() == [1.0]
        # Free axis out-of-domain likewise.
        f2 = pl.Series("f", [-1.0, 1.0, 3.0], dtype=pl.Float64)
        x2 = pl.Series("x", [5.0, 5.0, 5.0], dtype=pl.Float64)
        out2 = _assert_envelope_parity(x2, y, f2, 0.0, 10.0, 0.0, 2.0, 4, 2)
        assert out2.height == 1
        assert out2["y_min"].to_list() == [1.0]

    def test_all_rows_out_of_domain_empty_output(self):
        x = pl.Series("x", [11.0, 12.0], dtype=pl.Float64)
        y = pl.Series("y", [1.0, 2.0], dtype=pl.Float64)
        f = pl.Series("f", [1.0, 1.0], dtype=pl.Float64)
        out = _assert_envelope_parity(x, y, f, 0.0, 10.0, 0.0, 2.0, 4, 2)
        assert out.height == 0

    def test_empty_series(self):
        x = pl.Series("x", [], dtype=pl.Float64)
        y = pl.Series("y", [], dtype=pl.Float64)
        f = pl.Series("f", [], dtype=pl.Float64)
        out = _assert_envelope_parity(x, y, f, 0.0, 10.0, 0.0, 2.0, 4, 2)
        assert out.height == 0

    def test_single_row(self):
        x = pl.Series("x", [3.3], dtype=pl.Float64)
        y = pl.Series("y", [7.0], dtype=pl.Float64)
        f = pl.Series("f", [1.1], dtype=pl.Float64)
        out = _assert_envelope_parity(x, y, f, 0.0, 10.0, 0.0, 2.0, 4, 2)
        assert out.height == 1
        row = out.row(0, named=True)
        assert row["y_min"] == row["y_max"] == 7.0
        assert row["x_at_ymin"] == row["x_at_ymax"] == 3.3

    def test_tie_first_row_in_scan_order_wins(self):
        # Duplicate y extrema within one cell: first occurrence wins for both
        # min and max — deterministic.
        x = pl.Series("x", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=pl.Float64)
        y = pl.Series("y", [5.0, 1.0, 1.0, 9.0, 9.0, 5.0], dtype=pl.Float64)
        f = pl.Series("f", [0.5] * 6, dtype=pl.Float64)
        out = _assert_envelope_parity(x, y, f, 0.0, 10.0, 0.0, 1.0, 1, 1)
        row = out.row(0, named=True)
        assert row["y_min"] == 1.0
        assert row["x_at_ymin"] == 1.0, "first y==1.0 row (x=1) must win the tie"
        assert row["y_max"] == 9.0
        assert row["x_at_ymax"] == 3.0, "first y==9.0 row (x=3) must win the tie"

    def test_negative_domains(self):
        import random

        rng = random.Random(11)
        n = 500
        x = pl.Series("x", [rng.uniform(-100.0, -50.0) for _ in range(n)])
        y = pl.Series("y", [rng.gauss(-5.0, 2.0) for _ in range(n)])
        f = pl.Series("f", [rng.uniform(-5.0, -1.0) for _ in range(n)])
        out = _assert_envelope_parity(x, y, f, -100.0, -50.0, -5.0, -1.0, 6, 8)
        assert out.height > 0

    def test_integer_dtypes_parity(self):
        # Int x / Int free / Float y — the kernel casts to f64 exactly like the
        # reference (temporal columns are cast to physical by the caller).
        x = pl.Series("x", list(range(50)), dtype=pl.Int64)
        y = pl.Series("y", [float((i * 13) % 17) for i in range(50)])
        f = pl.Series("f", [i % 5 for i in range(50)], dtype=pl.Int32)
        _assert_envelope_parity(x, y, f, 0.0, 49.0, 0.0, 4.0, 5, 4)

    def test_property_randomized_parity(self):
        # Property-style: random mix of in/out-of-domain, nulls, and NaNs on
        # every column — fixed seed, bit-exact parity.
        import random

        rng = random.Random(1234)
        n = 5000

        def messy(lo: float, hi: float) -> list[float | None]:
            vals: list[float | None] = []
            for _ in range(n):
                r = rng.random()
                if r < 0.02:
                    vals.append(None)
                elif r < 0.04:
                    vals.append(float("nan"))
                else:
                    # Spill ~20% outside the domain on both sides.
                    span = hi - lo
                    vals.append(rng.uniform(lo - 0.2 * span, hi + 0.2 * span))
            return vals

        x = pl.Series("x", messy(0.0, 100.0), dtype=pl.Float64)
        y = pl.Series("y", messy(-10.0, 10.0), dtype=pl.Float64)
        f = pl.Series("f", messy(0.0, 10.0), dtype=pl.Float64)
        out = _assert_envelope_parity(x, y, f, 0.0, 100.0, 0.0, 10.0, 16, 32)
        assert 0 < out.height <= 17 * 33

    def test_filtered_multichunk_input(self):
        # Lazy filter → multi-chunk, possibly null-masked input series.
        frames = [
            pl.DataFrame(
                {
                    "x": [float(i) for i in range(start, start + 100)],
                    "y": [float((i * 7) % 23) for i in range(start, start + 100)],
                    "f": [float(i % 10) for i in range(start, start + 100)],
                }
            )
            for start in range(0, 1000, 100)
        ]
        df = pl.concat(frames, rechunk=False)
        out = (
            df.lazy()
            .select(
                pl.col("x")
                .filter(pl.col("x") < 500.0)
                .flexviz.fixed_line_envelope2d(
                    pl.col("y").filter(pl.col("x") < 500.0),
                    pl.col("f").filter(pl.col("x") < 500.0),
                    pl.lit(0.0),
                    pl.lit(1000.0),
                    pl.lit(0.0),
                    pl.lit(9.0),
                    10,
                    5,
                )
            )
            .collect()
            .to_series()
            .struct.unnest()
        )
        filtered = df.filter(pl.col("x") < 500.0)
        want = _envelope_reference(
            filtered["x"],
            filtered["y"],
            filtered["f"],
            0.0,
            1000.0,
            0.0,
            9.0,
            10,
            5,
        )
        from polars.testing import assert_frame_equal

        assert_frame_equal(out, want, check_exact=True)

    # ---- validation -----------------------------------------------------------

    def test_mismatched_lengths_raise(self):
        x = pl.Series("x", [1.0, 2.0], dtype=pl.Float64)
        y = pl.Series("y", [1.0], dtype=pl.Float64)
        f = pl.Series("f", [1.0, 2.0], dtype=pl.Float64)
        with pytest.raises(Exception, match="same length"):
            _fixed_line_envelope2d(x, y, f, 0.0, 10.0, 0.0, 2.0, 4, 2)

    def test_inverted_domain_raises(self):
        x = pl.Series("x", [1.0], dtype=pl.Float64)
        y = pl.Series("y", [1.0], dtype=pl.Float64)
        f = pl.Series("f", [1.0], dtype=pl.Float64)
        with pytest.raises(Exception, match="lo"):
            _fixed_line_envelope2d(x, y, f, 10.0, 0.0, 0.0, 2.0, 4, 2)

    def test_zero_buckets_raise(self):
        x = pl.Series("x", [1.0], dtype=pl.Float64)
        y = pl.Series("y", [1.0], dtype=pl.Float64)
        f = pl.Series("f", [1.0], dtype=pl.Float64)
        with pytest.raises(Exception, match="greater than 0"):
            _fixed_line_envelope2d(x, y, f, 0.0, 10.0, 0.0, 2.0, 0, 2)
        with pytest.raises(Exception, match="greater than 0"):
            _fixed_line_envelope2d(x, y, f, 0.0, 10.0, 0.0, 2.0, 4, 0)


# ---------------------------------------------------------------------------
# Kernel thread pool
# ---------------------------------------------------------------------------


class TestKernelThreadPool:
    """The kernels run on their own rayon pool, sized from POLARS_MAX_THREADS.

    A cdylib plugin cannot join Polars' pool (pola-rs/polars#19650), so a second
    pool is unavoidable. What must not happen is that second pool ignoring the
    thread limit the deployment set — on a CPU-quota'd container that is the
    oversubscription that gets the whole cgroup throttled.
    """

    @staticmethod
    def _thread_count(env_limit: str | None) -> int:
        """Threads alive after running the parallel kernel in a fresh process."""
        import os
        import subprocess
        import sys
        import textwrap

        script = textwrap.dedent("""
            import os
            import polars as pl
            import flexviz_polars  # noqa: F401

            # Above MIN_PAR, so the parallel path really does start the pool.
            n = (1 << 17) + 64
            s = pl.Series("v", [float(i % 997) for i in range(n)], dtype=pl.Float64)
            pl.select(pl.lit(s).flexviz.fixed_hist(pl.lit(0.0), pl.lit(997.0), 64))
            print(len(os.listdir("/proc/self/task")))
            """)
        env = dict(os.environ)
        env.pop("POLARS_MAX_THREADS", None)
        env.pop("RAYON_NUM_THREADS", None)
        if env_limit is not None:
            env["POLARS_MAX_THREADS"] = env_limit
        out = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, env=env
        )
        assert out.returncode == 0, out.stderr
        return int(out.stdout.split()[-1])

    def test_pool_respects_polars_max_threads(self):
        import os
        import sys

        if not sys.platform.startswith("linux"):
            pytest.skip("thread count is read from /proc")
        cores = os.cpu_count() or 1
        if cores < 8:
            pytest.skip("needs enough cores for the limit to be distinguishable")

        limited = self._thread_count("2")
        unlimited = self._thread_count(None)
        # The limited process must not have spun up a core-sized kernel pool on
        # top of its 2 Polars threads.
        assert limited < unlimited, (
            f"POLARS_MAX_THREADS=2 gave {limited} threads, "
            f"unlimited gave {unlimited} — the kernel pool ignored the limit"
        )
        assert (
            limited < cores
        ), f"{limited} threads under POLARS_MAX_THREADS=2 on a {cores}-core box"


# ---------------------------------------------------------------------------
# arg_min_max — parallel window path
#
# `par_by_window` only engages above MIN_PAR (1 << 17) total rows, so every test
# above this point exercises the serial path only. These run over that threshold.
# ---------------------------------------------------------------------------

PAR_ROWS = 300_000  # > MIN_PAR (131_072)


def _distinct_values(n: int) -> list[float]:
    """Distinct, non-monotonic values.

    Multiplication by an odd constant mod 2**23 is a bijection, so no two entries
    tie. Ties matter: SIMD argminmax may pick any index among equal values, so a
    tie would make the oracle comparison legitimately ambiguous. 2**23 also keeps
    every value exact in Float32 and in range for Int32.
    """
    return [float((i * 2654435761) % (2**23)) for i in range(n)]


def _expected_arg_min_max(values: list[float], n_points: int) -> list[int]:
    """Independent oracle: pure-Python argmin/argmax per uniform window."""
    n = len(values)
    n_out = max(min(max(n_points // 2, 1), n), 1)
    base, remainder = divmod(n, n_out)
    indices: set[int] = set()
    start = 0
    for i in range(n_out):
        window_len = base + (1 if i < remainder else 0)
        if window_len:
            window = values[start : start + window_len]
            indices.add(start + min(range(window_len), key=window.__getitem__))
            indices.add(start + max(range(window_len), key=window.__getitem__))
        start += window_len
    return sorted(indices)


class TestArgMinMaxParallel:
    @pytest.mark.parametrize("dtype", [pl.Float64, pl.Float32, pl.Int64, pl.Int32])
    def test_single_chunk_matches_oracle(self, dtype):
        values = _distinct_values(PAR_ROWS)
        s = pl.Series("y", values, dtype=pl.Float64).cast(dtype)
        assert s.n_chunks() == 1
        result = _arg_min_max(s, n_points=1000).to_list()
        assert result == _expected_arg_min_max(s.to_list(), 1000)

    @pytest.mark.parametrize("n_points", [4, 1000, 20_001])
    def test_multi_chunk_matches_single_chunk(self, n_points):
        """The per-worker chunk cursor must be seeded from its own first window.

        A worker that inherited a cursor starting at row 0 would mis-locate every
        window it owns; with a single chunk there is nothing to mis-locate, so this
        pairing is what catches it.
        """
        values = _distinct_values(PAR_ROWS)
        one = pl.Series("y", values, dtype=pl.Float64)
        # Deliberately uneven, including length-1 chunks, so a worker's first
        # window can start mid-chunk and the cursor cannot rely on a uniform stride.
        bounds = [0, 1, 7919, 100_000, 100_001, 233_333, PAR_ROWS]
        many = pl.concat(
            [
                pl.Series("y", values[a:b], dtype=pl.Float64)
                for a, b in zip(bounds, bounds[1:])
            ],
            rechunk=False,
        )
        assert many.n_chunks() == len(bounds) - 1
        assert many.to_list() == values

        expected = _expected_arg_min_max(values, n_points)
        assert _arg_min_max(one, n_points=n_points).to_list() == expected
        assert _arg_min_max(many, n_points=n_points).to_list() == expected

    # -- null fallback -----------------------------------------------------
    #
    # `simd_argminmax` requires null_count == 0, so a single null routes the whole
    # column to `fallback_window_argminmax`. That is also split by window, so these
    # exercise the parallel fallback rather than the SIMD paths.

    @staticmethod
    def _expected_with_nulls(values: list[float | None], n_points: int) -> list[int]:
        n = len(values)
        n_out = max(min(max(n_points // 2, 1), n), 1)
        base, remainder = divmod(n, n_out)
        indices: set[int] = set()
        start = 0
        for i in range(n_out):
            window_len = base + (1 if i < remainder else 0)
            live = [
                start + k for k in range(window_len) if values[start + k] is not None
            ]
            # An all-null window yields no index at all (both args are None and the
            # pair is dropped), matching `arg_min_max_pairs`.
            if live:
                indices.add(min(live, key=lambda j: values[j]))
                indices.add(max(live, key=lambda j: values[j]))
            start += window_len
        return sorted(indices)

    @pytest.mark.parametrize("n_chunks_in", [1, 5])
    def test_scattered_nulls_match_oracle(self, n_chunks_in):
        values: list[float | None] = list(_distinct_values(PAR_ROWS))
        for i in range(0, PAR_ROWS, 7):  # ~14% nulls, none of them window-aligned
            values[i] = None
        if n_chunks_in == 1:
            s = pl.Series("y", values, dtype=pl.Float64)
        else:
            step = PAR_ROWS // n_chunks_in
            s = pl.concat(
                [
                    pl.Series("y", values[a : a + step], dtype=pl.Float64)
                    for a in range(0, PAR_ROWS, step)
                ],
                rechunk=False,
            )
        assert s.null_count() > 0 and s.n_chunks() == n_chunks_in
        assert _arg_min_max(s, n_points=1000).to_list() == self._expected_with_nulls(
            values, 1000
        )

    def test_all_null_window_contributes_no_index(self):
        values: list[float | None] = list(_distinct_values(PAR_ROWS))
        # 500 windows of 600 rows; blank window 3 entirely.
        for i in range(3 * 600, 4 * 600):
            values[i] = None
        s = pl.Series("y", values, dtype=pl.Float64)
        result = _arg_min_max(s, n_points=1000).to_list()
        assert result == self._expected_with_nulls(values, 1000)
        assert not any(3 * 600 <= i < 4 * 600 for i in result)

    def test_all_null_series(self):
        s = pl.Series("y", [None] * PAR_ROWS, dtype=pl.Float64)
        assert _arg_min_max(s, n_points=1000).to_list() == []

    def test_window_count_below_worker_count_stays_correct(self):
        """n_points=2 gives one window: below the 2-window floor, so serial."""
        values = _distinct_values(PAR_ROWS)
        s = pl.Series("y", values, dtype=pl.Float64)
        assert _arg_min_max(s, n_points=2).to_list() == _expected_arg_min_max(values, 2)


# ---------------------------------------------------------------------------
# minmax_line — differential against the two-gather form
#
# `minmax_line` exists because Polars does not CSE opaque plugin expressions:
# the pre-fusion form `x.gather(idx), y.gather(idx)` ran the whole arg_min_max
# scan twice per trace. Both formulations are still compiled into the plugin,
# so the old form is the reference. A differential is stronger than an oracle
# here: on ties and NaN the SIMD kernel may pick any of several valid indices,
# which an independent oracle cannot predict, but both formulations run the
# identical index selection, so equality is exact by construction.
# ---------------------------------------------------------------------------


def _two_gather(x: pl.Series, y: pl.Series, n_points: int) -> pl.Series:
    """The pre-fusion formulation of the minmax line aggregation."""
    idx = pl.lit(y).flexviz.arg_min_max(n_points)
    return pl.select(
        pl.struct(
            **{x.name: pl.lit(x).gather(idx), y.name: pl.lit(y).gather(idx)}
        ).alias("out")
    ).to_series()


def _assert_fused_matches(x: pl.Series, y: pl.Series, n_points: int) -> None:
    from polars.testing import assert_series_equal

    fused = _minmax_line(x, y, n_points).rename("out")
    assert_series_equal(fused, _two_gather(x, y, n_points))


class TestMinmaxLineDifferential:
    @pytest.mark.parametrize("n_rows", [1_000, PAR_ROWS])  # serial and parallel paths
    @pytest.mark.parametrize(
        "dtype", [pl.Float64, pl.Float32, pl.Float16, pl.Int64, pl.Int32]
    )
    def test_dtypes(self, n_rows, dtype):
        # The f16 cast collapses distinct values into plateaus; for a
        # differential that is a feature, not a problem (see class comment).
        y = pl.Series("y", _distinct_values(n_rows), dtype=pl.Float64).cast(dtype)
        x = pl.Series("x", range(n_rows), dtype=pl.Float64)
        _assert_fused_matches(x, y, 1000)

    @pytest.mark.parametrize("n_points", [1, 2, 3, 1000, 25_000])
    def test_n_points_extremes(self, n_points):
        n = 10_000
        y = pl.Series("y", _distinct_values(n))
        x = pl.Series("x", range(n), dtype=pl.Float64)
        _assert_fused_matches(x, y, n_points)

    def test_plateaus(self):
        """Ties everywhere: which index wins is kernel-defined, but both forms
        must pick the same one."""
        n = 10_000
        y = pl.Series("y", [float(i // 500) for i in range(n)])
        x = pl.Series("x", range(n), dtype=pl.Float64)
        _assert_fused_matches(x, y, 100)

    def test_nan(self):
        values = _distinct_values(10_000)
        values[10::100] = [float("nan")] * len(values[10::100])
        y = pl.Series("y", values)
        x = pl.Series("x", range(len(values)), dtype=pl.Float64)
        _assert_fused_matches(x, y, 1000)

    @pytest.mark.parametrize("n_rows", [10_000, PAR_ROWS])
    def test_y_nulls_route_to_fallback(self, n_rows):
        values: list[float | None] = list(_distinct_values(n_rows))
        for i in range(0, n_rows, 7):
            values[i] = None
        y = pl.Series("y", values, dtype=pl.Float64)
        x = pl.Series("x", range(n_rows), dtype=pl.Float64)
        _assert_fused_matches(x, y, 1000)

    def test_x_nulls_survive_the_gather(self):
        n = 10_000
        y = pl.Series("y", _distinct_values(n))
        x = pl.Series(
            "x", [None if i % 3 == 0 else float(i) for i in range(n)], dtype=pl.Float64
        )
        _assert_fused_matches(x, y, 1000)

    def test_datetime_x(self):
        from datetime import datetime, timedelta

        n = 10_000
        t0 = datetime(2026, 1, 1)
        x = pl.Series("x", [t0 + timedelta(seconds=i) for i in range(n)])
        y = pl.Series("y", _distinct_values(n))
        _assert_fused_matches(x, y, 1000)

    def test_multi_chunk(self):
        values = _distinct_values(PAR_ROWS)
        bounds = [0, 1, 7919, 100_000, 233_333, PAR_ROWS]
        y = pl.concat(
            [
                pl.Series("y", values[a:b], dtype=pl.Float64)
                for a, b in zip(bounds, bounds[1:])
            ],
            rechunk=False,
        )
        x = pl.Series("x", range(PAR_ROWS), dtype=pl.Float64)
        assert y.n_chunks() == len(bounds) - 1
        _assert_fused_matches(x, y, 1000)

    def test_empty(self):
        _assert_fused_matches(
            pl.Series("x", [], dtype=pl.Float64),
            pl.Series("y", [], dtype=pl.Float64),
            1000,
        )

    def test_single_row(self):
        _assert_fused_matches(pl.Series("x", [1.0]), pl.Series("y", [2.0]), 1000)

    def test_length_mismatch_raises(self):
        with pytest.raises(Exception, match="same length"):
            _minmax_line(pl.Series("x", [1.0, 2.0]), pl.Series("y", [1.0]), 10)

    def test_n_points_zero_raises(self):
        with pytest.raises(Exception, match="greater than 0"):
            _minmax_line(pl.Series("x", [1.0]), pl.Series("y", [1.0]), 0)

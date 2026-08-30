"""Equivalence tests for the streaming (scan-source) 1D histogram.

The scan path and the resident path answer the *same* question, so unlike the
line envelope -- where two bucket definitions produce two valid answers -- the
bar here is bit-identical output.

The one thing that makes that non-trivial is an epsilon that lives only in
Rust. ``flexviz_polars/src/expressions.rs`` adds ``FIXED_HIST_ROUND_EPS``
(1e-9) before truncating the bin index, so a value sitting a hair below a bin
boundary lands in the bin above. A plan written as the obvious
``floor((v - lo) / bin_width)`` disagrees with the kernel on exactly those
values. ``TestRoundingEpsilon`` pins the behaviour rather than the constant,
so it keeps working if the Rust source is not on disk.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from flexviz.engine import FlexEngine, TraceInfo
from flexviz.events import InteractionEvent
from flexviz.LF import GroupedAggregationSpec, LFQueryBuilder
from flexviz.trace.hist import (
    Histogram,
    _FIXED_HIST_ROUND_EPS,
    _streaming_bin_expr,
)

# The epsilon hist.py adds to `hi` so the maximum value stays in the last bin.
# Different constant, different job -- see hist.py.
_HI_EPS = 1e-10


def _kernel_counts(df: pl.DataFrame, col: str, lo: float, hi: float, bins: int):
    expr = pl.col(col).flexviz.fixed_hist(pl.lit(lo), pl.lit(hi), bins)
    return df.select(expr.alias("h")).unnest("h")["count"].to_list()


def _streaming_counts(df: pl.DataFrame, col: str, lo: float, hi: float, bins: int):
    """Group-by on the streaming bin index, densified the way the plan does."""
    counted = (
        df.lazy()
        .group_by(_streaming_bin_expr(pl.col(col), lo, hi, bins).alias("__b"))
        .agg(pl.len().alias("count"))
        .collect()
    )
    dense = (
        pl.DataFrame({"__b": range(bins)}, schema={"__b": pl.Int32})
        .join(counted, on="__b", how="left")
        .with_columns(pl.col("count").fill_null(0).cast(pl.UInt32))
        .sort("__b")
    )
    return dense["count"].to_list()


_VALS = [0.0, 1.0, 2.5, 3.0, 4.0, 5.5, 7.9, 8.0]


def _f(vals, dtype=pl.Float64):
    return pl.DataFrame({"v": vals}, schema={"v": dtype})


# ---- bin arithmetic --------------------------------------------------------


class TestBinArithmeticMatchesKernel:
    """Every case here is a way the two formulations could drift apart."""

    @pytest.mark.parametrize(
        "name,df,lo,hi,bins",
        [
            ("f64", _f(_VALS), 0.0, 8.0 + _HI_EPS, 8),
            ("f64_nulls", _f(_VALS + [None]), 0.0, 8.0 + _HI_EPS, 8),
            ("f64_nan", _f(_VALS + [float("nan")]), 0.0, 8.0 + _HI_EPS, 8),
            ("f32", _f(_VALS, pl.Float32), 0.0, 8.0 + _HI_EPS, 8),
            ("i64", _f([0, 1, 2, 3, 4, 5, 7, 8], pl.Int64), 0.0, 8.0 + _HI_EPS, 8),
            (
                "i32_nulls",
                _f([0, 1, 2, None, 4, 5, 7, 8], pl.Int32),
                0.0,
                8.0 + _HI_EPS,
                8,
            ),
            ("u32", _f([0, 1, 2, 3, 4, 5, 7, 8], pl.UInt32), 0.0, 8.0 + _HI_EPS, 8),
            # hi == lo: the kernel counts every non-null into bin 0.
            ("degenerate", _f(_VALS), 3.0, 3.0, 8),
            # A viewport narrower than the data: values fall outside on both
            # sides and must clamp, not wrap or drop.
            ("viewport_inside", _f(_VALS), 2.0, 6.0 + _HI_EPS, 8),
            ("all_below_lo", _f([-5.0, -3.0]), 0.0, 8.0 + _HI_EPS, 8),
            ("all_above_hi", _f([50.0, 90.0]), 0.0, 8.0 + _HI_EPS, 8),
            ("single_bin", _f(_VALS), 0.0, 8.0 + _HI_EPS, 1),
            ("empty", _f([]), 0.0, 8.0 + _HI_EPS, 8),
            ("many_bins", _f(_VALS), 0.0, 8.0 + _HI_EPS, 257),
        ],
    )
    def test_counts_are_identical(self, name, df, lo, hi, bins):
        assert _streaming_counts(df, "v", lo, hi, bins) == _kernel_counts(
            df, "v", lo, hi, bins
        )

    def test_temporal_on_the_physical_column(self):
        """Temporal columns bin on to_physical(), which is what the plan does."""
        series = pl.datetime_range(
            dt.datetime(2020, 1, 1), dt.datetime(2020, 1, 9), interval="1d", eager=True
        )
        df = pl.DataFrame({"v": series}).select(pl.col("v").to_physical())
        lo = float(df["v"].min())
        hi = float(df["v"].max()) + _HI_EPS
        assert _streaming_counts(df, "v", lo, hi, 8) == _kernel_counts(
            df, "v", lo, hi, 8
        )

    def test_every_row_is_counted_exactly_once(self):
        """No value may be dropped or double-counted by the clamping."""
        vals = [float(i % 97) / 3.0 for i in range(5_000)]
        df = _f(vals)
        counts = _streaming_counts(df, "v", 0.0, 32.0 + _HI_EPS, 64)
        assert sum(counts) == len(vals)

    def test_nulls_and_nans_are_skipped_not_binned(self):
        clean = _f([1.0, 2.0, 3.0])
        dirty = _f([1.0, 2.0, 3.0, None, float("nan")])
        lo, hi = 0.0, 4.0 + _HI_EPS
        assert _streaming_counts(dirty, "v", lo, hi, 4) == _streaming_counts(
            clean, "v", lo, hi, 4
        )
        assert sum(_streaming_counts(dirty, "v", lo, hi, 4)) == 3


# ---- the epsilon -----------------------------------------------------------


class TestRoundingEpsilon:
    """Pin the boundary behaviour the Rust epsilon produces.

    Pinned behaviourally, not by reading expressions.rs: the Rust source is not
    guaranteed to be on disk next to an installed wheel, and the behaviour is
    what actually has to agree.
    """

    #: lo=0, hi=10, bins=10 puts bin boundaries on the integers, so
    #: (v - lo) * scale == v and the arithmetic is easy to reason about.
    LO, HI, BINS = 0.0, 10.0, 10

    def test_a_hair_below_a_boundary_lands_in_the_upper_bin(self):
        """This is the whole reason _FIXED_HIST_ROUND_EPS exists."""
        v = 3.0 - 5e-10  # closer to the boundary than the epsilon
        kernel = _kernel_counts(_f([v]), "v", self.LO, self.HI, self.BINS)
        assert kernel.index(1) == 3, "kernel must round the boundary value up"
        assert _streaming_counts(_f([v]), "v", self.LO, self.HI, self.BINS) == kernel

    def test_the_naive_form_disagrees(self):
        """Guard against someone 'simplifying' the epsilon away.

        A plain floor((v - lo) / bin_width) is the obvious translation and it
        is wrong. If this ever starts passing, the epsilon has been dropped.
        """
        v = 3.0 - 5e-10
        bin_width = (self.HI - self.LO) / self.BINS
        naive = int((v - self.LO) // bin_width)
        assert naive == 2, "sanity: the naive form floors into the lower bin"

        kernel = _kernel_counts(_f([v]), "v", self.LO, self.HI, self.BINS)
        assert kernel.index(1) != naive, "kernel and naive form must differ here"

    def test_the_constant_matches_the_kernel(self):
        """A value displaced by exactly the epsilon must cross the boundary."""
        assert _FIXED_HIST_ROUND_EPS == 1e-9
        just_under = 3.0 - _FIXED_HIST_ROUND_EPS / 2
        counts = _kernel_counts(_f([just_under]), "v", self.LO, self.HI, self.BINS)
        assert counts.index(1) == 3

    def test_a_value_further_below_stays_put(self):
        """The epsilon must not be so large that it moves ordinary values."""
        v = 2.5
        counts = _kernel_counts(_f([v]), "v", self.LO, self.HI, self.BINS)
        assert counts.index(1) == 2
        assert _streaming_counts(_f([v]), "v", self.LO, self.HI, self.BINS) == counts


# ---- the seam --------------------------------------------------------------


class TestScanSelectsTheStreamingPlan:
    """The seam must swap formulations, not just report a flag."""

    def test_scan_brings_a_plan_and_resident_does_not(self):
        hist = Histogram(x="val", bins=10)
        assert hist.get_aggregation_spec({}, scan_source=False).plan is None
        assert hist.get_aggregation_spec({}, scan_source=True).plan is not None

    def test_grouped_scan_uses_streaming_plan(self):
        """Grouped + scan_source uses a streaming plan instead of the plugin
        kernel, avoiding per-group list materialization."""
        hist = Histogram(x="val", bins=10, group_by="cat")
        spec = hist.get_aggregation_spec({}, scan_source=True)
        assert isinstance(spec, GroupedAggregationSpec)
        assert spec.plan is not None, "grouped scan must carry a streaming plan"
        assert not spec.agg_exprs, "streaming plan replaces agg_exprs"

    def test_resident_lazyframe_is_not_a_scan(self):
        df = pl.DataFrame({"val": [1.0, 2.0, 3.0]})
        assert LFQueryBuilder(df.lazy()).is_scan is False


class TestScanMatchesResident:
    """End to end: the same figure over a frame and over a scan of that frame.

    This is the gate. Both paths answer the same question, so the bar is
    identical output -- counts, centres and hover bounds.
    """

    @staticmethod
    def _updates(src, bins=16, histnorm="count", x_range=None, col="val"):
        lf = LFQueryBuilder(src)
        hist = Histogram(x=col, bins=bins, histnorm=histnorm)
        engine = FlexEngine(backend_lf=lf, scalable_traces={hist.uid: hist})
        infos = [TraceInfo(uid=hist.uid, axes=("x", "y"), trace_type="histogram")]
        event = (
            InteractionEvent(type="init", force_update=True)
            if x_range is None
            else InteractionEvent(
                type="viewport", force_update=True, axis_ranges={"x": list(x_range)}
            )
        )
        deltas = engine.process(event, infos)
        return deltas[0].updates, lf.is_scan

    def _both(self, tmp_path, df, name, **kw):
        path = tmp_path / f"{name}.parquet"
        df.write_parquet(path)
        resident, res_is_scan = self._updates(df, **kw)
        scanned, scan_is_scan = self._updates(pl.scan_parquet(path), **kw)
        assert res_is_scan is False, "the frame must keep the kernel"
        assert scan_is_scan is True, "the scan must take the streaming plan"
        return resident, scanned

    @staticmethod
    def _same(a, b):
        assert list(a["y"]) == list(b["y"]), "counts differ"
        # Compare raw, not rounded: the point is bit-identical output, and the
        # centres may be datetimes (temporal axis) or None (bins=1, where
        # breakpoint.diff().mean() is null on BOTH paths -- a pre-existing
        # quirk of _to_update, reproduced here rather than papered over).
        assert list(a["x"]) == list(b["x"]), "bin centres differ"
        assert a["hover_bounds"] == b["hover_bounds"], "hover bounds differ"

    @pytest.mark.parametrize(
        "name,vals,dtype",
        [
            ("floats", [((i * 7919) % 1000) / 7.0 for i in range(20_011)], pl.Float64),
            ("ints", [(i * 13) % 511 for i in range(20_000)], pl.Int64),
            ("f32", [((i * 31) % 97) / 7.0 for i in range(20_000)], pl.Float32),
            ("constant", [3.0] * 20_000, pl.Float64),
            ("two_level", [1.0, 2.0] * 10_000, pl.Float64),
            ("tiny", [float(i % 31) for i in range(137)], pl.Float64),
            (
                "with_nulls",
                [None if i % 501 == 0 else float(i % 89) for i in range(20_000)],
                pl.Float64,
            ),
            (
                "with_nan",
                [
                    float("nan") if i % 307 == 0 else float(i % 89)
                    for i in range(20_000)
                ],
                pl.Float64,
            ),
        ],
    )
    def test_identical_across_dtypes_and_shapes(self, tmp_path, name, vals, dtype):
        df = pl.DataFrame({"val": vals}, schema={"val": dtype})
        self._same(*self._both(tmp_path, df, name))

    @pytest.mark.parametrize(
        "histnorm",
        ["count", "percent", "probability", "density", "probability density"],
    )
    def test_identical_for_every_histnorm(self, tmp_path, histnorm):
        df = pl.DataFrame({"val": [((i * 7919) % 1000) / 7.0 for i in range(20_011)]})
        self._same(*self._both(tmp_path, df, "norm", histnorm=histnorm))

    @pytest.mark.parametrize("x_range", [(10.0, 90.0), (0.0, 50.0), (-20.0, 500.0)])
    def test_identical_under_a_viewport(self, tmp_path, x_range):
        df = pl.DataFrame({"val": [((i * 7919) % 1000) / 7.0 for i in range(20_011)]})
        self._same(*self._both(tmp_path, df, "vp", x_range=x_range))

    @pytest.mark.parametrize("bins", [1, 2, 7, 100, 257])
    def test_identical_across_bin_counts(self, tmp_path, bins):
        df = pl.DataFrame({"val": [float(i % 991) for i in range(20_000)]})
        self._same(*self._both(tmp_path, df, "bins", bins=bins))

    def test_identical_on_a_temporal_column(self, tmp_path):
        series = pl.datetime_range(
            dt.datetime(2020, 1, 1),
            dt.datetime(2020, 3, 1),
            interval="1h",
            eager=True,
        )
        df = pl.DataFrame({"val": series})
        self._same(*self._both(tmp_path, df, "temporal", bins=24))

    def test_integer_bins_are_not_shifted_down(self, tmp_path):
        """The case the rounding epsilon exists for.

        Integer data on integer-aligned bin edges. Without the epsilon every
        non-zero value lands one bin low, on both paths -- which would still
        be self-consistent, so equality alone would not catch it. Assert the
        answer, not just the agreement.
        """
        df = pl.DataFrame({"val": list(range(100))}, schema={"val": pl.Int64})
        resident, scanned = self._both(tmp_path, df, "intbins", bins=100)
        self._same(resident, scanned)
        assert list(scanned["y"]) == [1] * 100, "each integer needs its own bin"

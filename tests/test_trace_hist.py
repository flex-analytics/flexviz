"""Unit tests for Histogram trace."""

from __future__ import annotations

from collections.abc import Sequence
import math

import polars as pl
import pytest

from flexviz.LF import LFQueryBuilder
from flexviz.spec import TraceSpec
from flexviz.trace import _hist_helpers as helpers_mod
from flexviz.trace._hist_helpers import _snap_range
from flexviz.trace.hist import _HIST_BIN_EPSILON, Histogram, _streaming_hist_plan

# ---- helpers ---------------------------------------------------------------


def _domains(lf: LFQueryBuilder, trace, update_range: dict) -> dict:
    """Resolve a trace's unfiltered bounds the way ``FlexEngine`` does."""
    cols = trace.domain_cols(update_range)
    return lf.physical_minmax(list(cols), lf.schema, memoize=False) if cols else {}


def _aggregate_hist(
    df: pl.DataFrame,
    bins: int = 20,
    histnorm: str = "count",
    x: str | None = "val",
    y: str | None = None,
    x_range: tuple[float, float] | None = None,
) -> dict:
    """Run a full histogram aggregation pipeline and return the update dict."""
    lf = LFQueryBuilder(df)
    trace = Histogram(x=x, y=y, bins=bins, histnorm=histnorm)
    vp_key = "x" if x is not None else "y"
    update_range = {vp_key: x_range} if x_range else {}
    agg_spec = trace.get_aggregation_spec(
        update_range, schema=lf.schema, domains=_domains(lf, trace, update_range)
    )
    df_agg, _ = lf.aggregate([], [agg_spec])
    return trace._to_update(df_agg).updates


def _aggregate_grouped_hist(
    df: pl.DataFrame,
    x_range: tuple[float, float] | None = None,
    group_by: str | Sequence[str] | None = "cat",
) -> list:
    lf = LFQueryBuilder(df)
    trace = Histogram(x="val", bins=10, histnorm="count", group_by=group_by)
    update_range = {"x": x_range} if x_range is not None else {}
    agg_spec = trace.get_aggregation_spec(
        update_range, schema=lf.schema, domains=_domains(lf, trace, update_range)
    )
    _, grouped_dfs = lf.aggregate([], [agg_spec])
    return trace._to_grouped_update(grouped_dfs[trace.uid]).group_results or []


# ---- bin count -------------------------------------------------------------


class TestHistogramBinCount:
    def test_bin_count_matches(self, small_df: pl.DataFrame):
        update = _aggregate_hist(small_df, bins=10)
        assert len(update["x"]) == 10
        assert len(update["y"]) == 10

    def test_bin_count_30(self, small_df: pl.DataFrame):
        update = _aggregate_hist(small_df, bins=30)
        assert len(update["x"]) == 30


# ---- viewport filter -------------------------------------------------------


class TestHistogramViewport:
    def test_viewport_reduces_total_count(self, small_df: pl.DataFrame):
        full = _aggregate_hist(small_df, bins=20, histnorm="count")
        zoomed = _aggregate_hist(
            small_df, bins=20, histnorm="count", x_range=(200, 400)
        )

        full_total = sum(full["y"])
        zoomed_total = sum(zoomed["y"])
        assert zoomed_total < full_total


# ---- histnorm modes -------------------------------------------------------


class TestHistogramNormalization:
    def test_count(self, small_df: pl.DataFrame):
        update = _aggregate_hist(small_df, bins=20, histnorm="count")
        total = sum(update["y"])
        assert total == 1000

    def test_percent(self, small_df: pl.DataFrame):
        update = _aggregate_hist(small_df, bins=20, histnorm="percent")
        total = sum(update["y"])
        assert abs(total - 100.0) < 0.01

    def test_probability(self, small_df: pl.DataFrame):
        update = _aggregate_hist(small_df, bins=20, histnorm="probability")
        total = sum(update["y"])
        assert abs(total - 1.0) < 0.01

    def test_density(self, small_df: pl.DataFrame):
        update = _aggregate_hist(small_df, bins=20, histnorm="density")
        assert all(v >= 0 for v in update["y"])
        assert len(update["y"]) == 20

    def test_probability_density(self, small_df: pl.DataFrame):
        update = _aggregate_hist(small_df, bins=20, histnorm="probability density")
        assert all(v >= 0 for v in update["y"])
        assert len(update["y"]) == 20

    def test_density_bins_1_does_not_crash(self):
        """bins=1 must not raise TypeError from None bin_width — regression test."""
        df = pl.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0]})
        update = _aggregate_hist(df, bins=1, histnorm="density")
        assert len(update["y"]) == 1
        assert all(math.isfinite(v) for v in update["y"])

    def test_probability_density_bins_1_does_not_crash(self):
        """bins=1 with probability density must produce a finite result."""
        df = pl.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0]})
        update = _aggregate_hist(df, bins=1, histnorm="probability density")
        assert len(update["y"]) == 1
        assert all(math.isfinite(v) for v in update["y"])


# ---- horizontal orientation -----------------------------------------------


class TestHistogramOrientation:
    def test_horizontal(self, small_df: pl.DataFrame):
        update = _aggregate_hist(small_df, x=None, y="val", bins=10)
        assert "orientation" in update
        assert update["orientation"] == "h"
        assert len(update["x"]) == 10
        assert len(update["y"]) == 10

    def test_vertical_no_orientation_key(self, small_df: pl.DataFrame):
        update = _aggregate_hist(small_df, x="val", y=None, bins=10)
        assert "orientation" not in update


# ---- invalid construction --------------------------------------------------


class TestHistogramValidation:
    def test_both_x_y_raises(self):
        with pytest.raises(ValueError, match="either x or y"):
            Histogram(x="a", y="b")

    def test_neither_x_y_raises(self):
        with pytest.raises(ValueError, match="either x or y"):
            Histogram()

    def test_invalid_histnorm(self):
        with pytest.raises(ValueError, match="histnorm"):
            Histogram(x="a", histnorm="invalid")


# ---- from_trace_spec round-trip --------------------------------------------


class TestHistogramFromTraceSpec:
    def test_roundtrip(self):
        original = Histogram(
            x="val", bins=30, histnorm="percent", name="H", color="#f00"
        )
        spec = original.to_trace_spec()

        assert isinstance(spec, TraceSpec)
        assert spec.trace_type == "histogram"

        restored = Histogram.from_trace_spec(spec)
        assert restored.uid == original.uid
        assert restored.data_col == "val"
        assert restored.bins == 30
        assert restored.histnorm == "percent"
        assert restored._display.get("name") == "H"
        assert restored._display.get("color") == "#f00"


# ---- bin alignment ----------------------------------------------------------


class TestHistogramBinAlignment:
    """Verify that multiple histogram traces on the same figure share bin edges."""

    def test_ungrouped_two_traces_share_bin_edges(self, small_df: pl.DataFrame):
        """Two ungrouped histograms on the same data column must produce identical
        bin centers when evaluated together (no viewport range — dynamic path)."""
        lf = LFQueryBuilder(small_df)
        bins = 10
        t1 = Histogram(x="val", bins=bins)
        t2 = Histogram(x="val", bins=bins)
        domains = _domains(lf, t1, {})
        spec1 = t1.get_aggregation_spec({}, schema=lf.schema, domains=domains)
        spec2 = t2.get_aggregation_spec({}, schema=lf.schema, domains=domains)
        df_agg, _ = lf.aggregate([], [spec1, spec2])
        centers1 = t1._to_update(df_agg).updates["x"]
        centers2 = t2._to_update(df_agg).updates["x"]
        assert list(centers1) == list(centers2)

    def test_shared_domain_aligns_edges_across_different_columns(self):
        """Sibling histograms handed one shared domain land on identical edges."""
        df = pl.DataFrame(
            {
                "y_pos": [0.0, 0.25, 0.5, 0.75, 1.0],
                "y_neg": [1e-9, 0.25 + 1e-9, 0.5 + 1e-9, 0.75 + 1e-9, 1.0 + 1e-9],
            }
        )
        lf = LFQueryBuilder(df)
        bins = 4
        t_pos = Histogram(x="y_pos", bins=bins)
        t_neg = Histogram(x="y_neg", bins=bins)
        shared = lf.physical_minmax(["y_pos", "y_neg"], lf.schema, memoize=False)

        spec_pos = t_pos.get_aggregation_spec({}, schema=lf.schema, domains=shared)
        spec_neg = t_neg.get_aggregation_spec({}, schema=lf.schema, domains=shared)
        df_agg, _ = lf.aggregate([], [spec_pos, spec_neg])

        centers_pos = list(t_pos._to_update(df_agg).updates["x"])
        centers_neg = list(t_neg._to_update(df_agg).updates["x"])
        assert centers_pos == centers_neg

    def test_viewport_range_produces_correct_bin_centers(self, small_df: pl.DataFrame):
        """With a viewport range the bin centers must equal lo + (i + 0.5) * step
        over the SNAPPED range, which can hold one bin more than asked for."""
        lf = LFQueryBuilder(small_df)
        bins = 5
        trace = Histogram(x="val", bins=bins)
        spec = trace.get_aggregation_spec({"x": [100.0, 400.0]}, schema=lf.schema)
        df_agg, _ = lf.aggregate([], [spec])
        centers = list(trace._to_update(df_agg).updates["x"])
        lo, hi, n = _snap_range(100.0, 400.0, bins)
        assert n == bins + 1
        step = (hi - lo + _HIST_BIN_EPSILON) / n
        expected = [lo + (i + 0.5) * step for i in range(n)]
        assert len(centers) == n
        for got, want in zip(centers, expected):
            assert abs(got - want) < 1e-6

    def test_explicit_bins_all_bins_returned(self, small_df: pl.DataFrame):
        """All N bins are returned including empty ones (no gaps)."""
        bins = 15
        update = _aggregate_hist(small_df, bins=bins)
        assert len(update["x"]) == bins

    def test_ungrouped_empty_data_does_not_crash(self):
        """Filtering to an empty range must not crash — fill_null guards min/max."""
        df = pl.DataFrame({"val": [1.0, 2.0, 3.0]})
        lf = LFQueryBuilder(df)
        trace = Histogram(x="val", bins=5)
        # x_range that excludes all data
        spec = trace.get_aggregation_spec({"x": [100.0, 200.0]}, schema=lf.schema)
        df_agg, _ = lf.aggregate([], [spec])
        result = trace._to_update(df_agg).updates
        assert len(result["x"]) == 5
        assert all(c == 0 for c in result["y"])

    def test_bins_stable_under_cross_filter(self):
        """Bin centers must not change when a cross-filter reduces the data range."""
        df = pl.DataFrame({"val": list(range(100))})
        lf = LFQueryBuilder(df)
        trace = Histogram(x="val", bins=10)
        spec = trace.get_aggregation_spec(
            {}, schema=lf.schema, domains=_domains(lf, trace, {})
        )

        # Baseline: no cross-filter
        df_agg_full, _ = lf.aggregate([], [spec])
        centers_full = list(trace._to_update(df_agg_full).updates["x"])

        # Cross-filter that restricts the value range
        cross_filter = [pl.col("val") < 50]
        df_agg_filtered, _ = lf.aggregate(cross_filter, [spec])
        centers_filtered = list(trace._to_update(df_agg_filtered).updates["x"])

        assert centers_full == centers_filtered, (
            "Bin centers must be derived from the full (unfiltered) data range, "
            "not from the cross-filtered range."
        )

    def test_two_traces_each_stable_under_cross_filter(self):
        """Each of two histogram traces on different columns must keep its own bin
        centers stable when a cross-filter asymmetrically reduces each column's range.

        This is the primary regression test for the bug: before the fix, bin edges
        were derived from the cross-filtered column range, so each trace re-binned
        its data after every cross-filter update, making bins inconsistent over time.
        """
        import math as _math

        df = pl.DataFrame(
            {
                "sin": [_math.sin(i * 0.1) for i in range(200)],
                "cos": [_math.cos(i * 0.1) for i in range(200)],
                "flag": [i % 3 != 0 for i in range(200)],
            }
        )
        lf = LFQueryBuilder(df)
        bins = 10
        t_sin = Histogram(x="sin", bins=bins)
        t_cos = Histogram(x="cos", bins=bins)
        spec_sin = t_sin.get_aggregation_spec(
            {}, schema=lf.schema, domains=_domains(lf, t_sin, {})
        )
        spec_cos = t_cos.get_aggregation_spec(
            {}, schema=lf.schema, domains=_domains(lf, t_cos, {})
        )

        # Baseline — no cross-filter
        df_base, _ = lf.aggregate([], [spec_sin, spec_cos])
        centers_sin_base = list(t_sin._to_update(df_base).updates["x"])
        centers_cos_base = list(t_cos._to_update(df_base).updates["x"])

        # After cross-filter that asymmetrically restricts each column's range
        df_filtered, _ = lf.aggregate([pl.col("flag")], [spec_sin, spec_cos])
        centers_sin_filtered = list(t_sin._to_update(df_filtered).updates["x"])
        centers_cos_filtered = list(t_cos._to_update(df_filtered).updates["x"])

        assert (
            centers_sin_base == centers_sin_filtered
        ), "sin histogram bin centers must not change under cross-filtering"
        assert (
            centers_cos_base == centers_cos_filtered
        ), "cos histogram bin centers must not change under cross-filtering"


class TestHistogramViewportSnap:
    """A zoomed histogram bins on the viewport snapped outward to a lattice."""

    _DF = pl.DataFrame({"val": [float(i) / 4 for i in range(400)]})

    def _centers(self, x_range, bins=10, df=None, x="val"):
        return list(
            _aggregate_hist(
                self._DF if df is None else df, bins=bins, x=x, x_range=x_range
            )["x"]
        )

    def test_lattice_aligned_viewport_is_unchanged(self):
        # 20..60 over 10 bins has width 4 and both bounds are multiples of it.
        assert _snap_range(20.0, 60.0, 10) == (20.0, 60.0, 10)
        centers = self._centers((20.0, 60.0))
        assert len(centers) == 10
        assert centers[0] == pytest.approx(22.0)

    def test_offset_viewport_snaps_outward_and_gains_a_bin(self):
        lo, hi, n = _snap_range(21.0, 61.0, 10)
        assert (lo, hi, n) == (20.0, 64.0, 11)
        centers = self._centers((21.0, 61.0))
        assert len(centers) == 11
        assert centers[0] == pytest.approx(22.0)  # same lattice as the aligned case

    def test_half_bin_pan_keeps_the_shared_edges(self):
        # Panning by half a bin keeps the span, so the lattice does not move:
        # every edge the two viewports have in common stays put.
        before = self._centers((20.0, 60.0))
        after = self._centers((22.0, 62.0))
        for c in before:
            assert min(abs(c - o) for o in after) < 1e-6

    def test_counts_equal_the_kernel_at_the_snapped_edges(self):
        x_range = (21.0, 61.0)
        counts = list(_aggregate_hist(self._DF, bins=10, x_range=x_range)["y"])
        lo, hi, n = _snap_range(*x_range, 10)
        inside = self._DF.filter(pl.col("val").is_between(lo, hi))
        ref = (
            inside.select(
                pl.col("val")
                .flexviz.fixed_hist(
                    pl.lit(lo), pl.lit(hi + _HIST_BIN_EPSILON), n_bins=n
                )
                .implode()
                .alias("u")
            )["u"]
            .item()
            .explode()
            .struct.field("count")
            .to_list()
        )
        assert counts == ref

    def test_temporal_axis_snaps_in_physical_units(self):
        import datetime

        ts = pl.datetime_range(
            datetime.datetime(2020, 1, 1),
            datetime.datetime(2020, 1, 5),
            interval="1h",
            eager=True,
        ).rename("t")
        df = pl.DataFrame([ts])
        # A window offset by 30 minutes from the 6-hour lattice: 12 bins over
        # 3 days is a 6-hour width, so the snap must widen it by one bin.
        update = _aggregate_hist(
            df,
            bins=12,
            x="t",
            x_range=("2020-01-01 00:30:00", "2020-01-04 00:30:00"),
        )
        centers = update["x"].to_list()
        assert len(centers) == 13
        # Snapped edges are whole 6-hour steps from the epoch, so every center
        # sits on a 3-hour offset: 03:00, 09:00, ...
        assert all(c.minute == 0 and c.hour % 6 == 3 for c in centers)

    def test_grouped_snaps_the_same_for_every_group(self):
        df = pl.DataFrame(
            {"val": [float(i) for i in range(40)], "cat": ["A", "B"] * 20}
        )
        lf = LFQueryBuilder(df)
        trace = Histogram(x="val", bins=10, group_by="cat")
        update_range = {"x": [1.0, 21.0]}
        spec = trace.get_aggregation_spec(
            update_range, schema=lf.schema, domains=_domains(lf, trace, update_range)
        )
        _, grouped = lf.aggregate([], [spec])
        results = trace._to_grouped_update(grouped[trace.uid]).group_results
        lo, hi, n = _snap_range(1.0, 21.0, 10)
        assert n == 11
        centers = [list(cr.updates["x"]) for cr in results]
        assert len(centers) == 2 and centers[0] == centers[1]
        assert len(centers[0]) == n
        assert centers[0][0] == pytest.approx(lo + (hi - lo) / n / 2)

    def test_scan_plan_matches_the_kernel_under_a_snapped_viewport(self):
        lf = LFQueryBuilder(self._DF)
        trace = Histogram(x="val", bins=10)
        update_range = {"x": [21.0, 61.0]}
        out = []
        for scan_source in (False, True):
            spec = trace.get_aggregation_spec(
                update_range, schema=lf.schema, scan_source=scan_source
            )
            df_agg, _ = lf.aggregate([], [spec])
            updates = trace._to_update(df_agg).updates
            out.append((updates["x"].to_list(), updates["y"].to_list()))
        assert out[0] == out[1]
        assert len(out[0][0]) == 11


class TestHistogramGroupedBinAlignment:
    """Grouped histograms must share bin edges across groups."""

    def _grouped_bin_edges(
        self,
        df: pl.DataFrame,
        x_range: tuple[float, float] | None = None,
    ) -> list[list[float]]:
        """Return the breakpoint lists for each group."""
        lf = LFQueryBuilder(df)
        trace = Histogram(x="val", bins=5, group_by="cat")
        update_range = {"x": list(x_range)} if x_range is not None else {}
        spec = trace.get_aggregation_spec(
            update_range, schema=lf.schema, domains=_domains(lf, trace, update_range)
        )
        _, grouped_dfs = lf.aggregate([], [spec])
        results = trace._to_grouped_update(grouped_dfs[trace.uid]).group_results
        return [list(cr.updates["x"]) for cr in results]

    def test_grouped_groups_share_bin_edges_no_viewport(self):
        """Without a viewport range groups with different data ranges must align."""
        df = pl.DataFrame(
            {
                "val": list(range(10)) + list(range(100, 110)),
                "cat": ["A"] * 10 + ["B"] * 10,
            }
        )
        centers = self._grouped_bin_edges(df)
        assert len(centers) == 2
        assert centers[0] == centers[1]

    def test_grouped_groups_share_bin_edges_with_viewport(self):
        """Viewport-range path must also align all groups."""
        df = pl.DataFrame(
            {
                "val": list(range(10)) + list(range(100, 110)),
                "cat": ["A"] * 10 + ["B"] * 10,
            }
        )
        centers = self._grouped_bin_edges(df, x_range=(0.0, 110.0))
        assert len(centers) == 2
        assert centers[0] == centers[1]


class TestHistogramGrouped:
    def test_grouped_returns_child_results(self):
        df = pl.DataFrame(
            {
                "val": list(range(20)) + list(range(100, 120)),
                "cat": ["A"] * 20 + ["B"] * 20,
            }
        )
        results = _aggregate_grouped_hist(df)
        assert len(results) == 2
        assert {cr.group_value_key for cr in results} == {"A", "B"}

    def test_grouped_viewport_controls_visible_groups(self):
        df = pl.DataFrame(
            {
                "val": list(range(20)) + list(range(100, 120)),
                "cat": ["A"] * 20 + ["B"] * 20,
            }
        )
        results = _aggregate_grouped_hist(df, x_range=(0, 25))
        assert [cr.group_value_key for cr in results] == ["A"]

    def test_grouped_by_two_columns_returns_composite_children(self):
        df = pl.DataFrame(
            {
                "val": list(range(10)) + list(range(100, 110)) + list(range(200, 210)),
                "cat": ["A"] * 10 + ["A"] * 10 + ["B"] * 10,
                "site": ["north"] * 10 + ["south"] * 10 + ["north"] * 10,
            }
        )
        results = _aggregate_grouped_hist(df, group_by=("cat", "site"))
        assert {cr.group_value_key for cr in results} == {
            '["A","north"]',
            '["A","south"]',
            '["B","north"]',
        }


# ---- dtype support -----------------------------------------------------------


class TestHistogramDtypeSupport:
    """fixed_hist must handle integer and float32 columns without caller casting."""

    def test_int32_column(self):
        df = pl.DataFrame({"val": pl.Series(list(range(100)), dtype=pl.Int32)})
        update = _aggregate_hist(df, bins=10)
        assert len(update["x"]) == 10
        assert sum(update["y"]) == 100

    def test_int64_column(self):
        df = pl.DataFrame({"val": pl.Series(list(range(100)), dtype=pl.Int64)})
        update = _aggregate_hist(df, bins=10)
        assert len(update["x"]) == 10
        assert sum(update["y"]) == 100

    def test_float32_column(self):
        df = pl.DataFrame(
            {"val": pl.Series([float(i) for i in range(100)], dtype=pl.Float32)}
        )
        update = _aggregate_hist(df, bins=10)
        assert len(update["x"]) == 10
        assert sum(update["y"]) == 100

    def test_grouped_int32_column(self):
        df = pl.DataFrame(
            {
                "val": pl.Series(list(range(40)), dtype=pl.Int32),
                "cat": ["A"] * 20 + ["B"] * 20,
            }
        )
        results = _aggregate_grouped_hist(df)
        assert len(results) == 2
        total = sum(sum(cr.updates["y"]) for cr in results)
        assert total == 40


# ---- temporal x support (regression: fixed_hist panicked on temporal) -------


class TestHistogramTemporal:
    """A temporal x column must bin via its physical representation.

    Regression: ``Histogram(x="t")`` handed the temporal column (and temporal
    min/max stats) straight to the numeric ``fixed_hist`` kernel, which panicked
    (``not implemented``) for naive AND tz-aware ``Datetime`` and for ``Date``.
    The fix bins in physical space and returns datetime bin centers (so Plotly
    renders a date axis, like the line trace).
    """

    @staticmethod
    def _hourly_df(tz: str | None = None) -> pl.DataFrame:
        import datetime as dt

        base = dt.datetime(2020, 1, 1)
        ts = [base + dt.timedelta(hours=i) for i in range(100)]
        s = pl.Series("t", ts, dtype=pl.Datetime("us"))
        if tz is not None:
            s = s.dt.replace_time_zone(tz)
        return pl.DataFrame({"t": s})

    def test_naive_datetime_bins_uniformly(self):
        # 100 hourly points over 10 bins → exactly 10 per bin.
        update = _aggregate_hist(self._hourly_df(), bins=10, x="t")
        assert len(update["x"]) == 10
        assert update["y"].to_list() == [10] * 10

    def test_utc_datetime_bins_uniformly(self):
        update = _aggregate_hist(self._hourly_df(tz="UTC"), bins=10, x="t")
        assert update["y"].to_list() == [10] * 10

    def test_named_tz_with_offset_bins_uniformly(self):
        update = _aggregate_hist(self._hourly_df(tz="Europe/Brussels"), bins=10, x="t")
        assert update["y"].to_list() == [10] * 10

    def test_date_column_counts_all(self):
        import datetime as dt

        days = [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(50)]
        df = pl.DataFrame({"t": pl.Series("t", days, dtype=pl.Date)})
        update = _aggregate_hist(df, bins=10, x="t")
        assert len(update["x"]) == 10
        assert sum(update["y"]) == 50

    def test_centers_are_temporal_for_date_axis(self):
        # Centers must be a temporal Series (not raw epoch ints) so the renderer
        # auto-detects a date axis, consistent with the line trace.
        update = _aggregate_hist(self._hourly_df(tz="UTC"), bins=10, x="t")
        assert update["x"].dtype.is_temporal()
        centers = update["x"].to_list()
        assert centers == sorted(centers)
        assert all(c.year == 2020 and c.month == 1 for c in centers)

    def test_grouped_temporal_counts_all(self):
        df = self._hourly_df(tz="UTC").with_columns(pl.Series("cat", ["A", "B"] * 50))
        lf = LFQueryBuilder(df)
        trace = Histogram(x="t", bins=10, histnorm="count", group_by="cat")
        spec = trace.get_aggregation_spec(
            {}, schema=lf.schema, domains=_domains(lf, trace, {})
        )
        _, grouped = lf.aggregate([], [spec])
        results = trace._to_grouped_update(grouped[trace.uid]).group_results or []
        assert len(results) == 2
        assert sum(sum(cr.updates["y"]) for cr in results) == 100
        assert all(cr.updates["x"].dtype.is_temporal() for cr in results)

    def test_viewport_temporal_reduces_count(self):
        # Zoom to the first 24 hours via a Plotly date-axis range string.
        update = _aggregate_hist(
            self._hourly_df(),
            bins=10,
            x="t",
            x_range=("2020-01-01 00:00:00", "2020-01-01 23:00:00"),
        )
        assert sum(update["y"]) == 24
        assert update["x"].dtype.is_temporal()

    def test_viewport_tz_offset_bound_on_utc_column(self):
        """A tz-aware (offset) viewport bound against a UTC temporal column must
        bin the zoomed window — not raise. Regression: the viewport FILTER path
        (_range_filter_expr → _typed_temporal_lit) raised a tz-mismatch
        TypeError before the bin-edge fallback could run."""
        # 100 hourly UTC points from 2020-01-01T00:00Z; zoom to the first 24 h
        # expressed with a +02:00 offset (== 00:00Z .. 23:00Z).
        update = _aggregate_hist(
            self._hourly_df(tz="UTC"),
            bins=10,
            x="t",
            x_range=("2020-01-01T02:00:00+02:00", "2020-01-02T01:00:00+02:00"),
        )
        assert sum(update["y"]) == 24


# ---- bin-center stability under cross-filter (regression) --------------------


class TestHistogramBinStabilityRegression:
    """Bin centers must not shift when fixed_hist is used (no-viewport path)."""

    def test_viewport_bin_centers_stable_across_calls(self):
        """Same viewport → identical centers on two separate calls."""
        df = pl.DataFrame({"val": [float(i) for i in range(500)]})
        lo, hi = 0.0, 500.0
        c1 = list(_aggregate_hist(df, bins=10, x_range=(lo, hi))["x"])
        c2 = list(_aggregate_hist(df, bins=10, x_range=(lo, hi))["x"])
        assert c1 == c2

    def test_grouped_bins_stable_under_cross_filter(self):
        """Grouped histogram: bin centers identical with and without cross-filter."""
        df = pl.DataFrame(
            {
                "val": list(range(100)) + list(range(100, 200)),
                "cat": ["A"] * 100 + ["B"] * 100,
            }
        )
        lf_full = LFQueryBuilder(df)
        lf_filtered = LFQueryBuilder(df)

        trace = Histogram(x="val", bins=10, group_by="cat")
        spec = trace.get_aggregation_spec(
            {}, schema=lf_full.schema, domains=_domains(lf_full, trace, {})
        )

        _, grouped_full = lf_full.aggregate([], [spec])
        _, grouped_filtered = lf_filtered.aggregate([pl.col("val") < 150], [spec])

        results_full = trace._to_grouped_update(grouped_full[trace.uid]).group_results
        results_filtered = trace._to_grouped_update(
            grouped_filtered[trace.uid]
        ).group_results

        centers_full = {
            cr.group_value_key: list(cr.updates["x"]) for cr in results_full
        }
        centers_filtered = {
            cr.group_value_key: list(cr.updates["x"]) for cr in results_filtered
        }

        for key in centers_full:
            if key in centers_filtered:
                assert (
                    centers_full[key] == centers_filtered[key]
                ), f"Group {key}: bin centers shifted under cross-filter"


class TestHistogramHoverSpec:
    def test_histogram_has_axis_and_cell_source_modes(self):
        from flexviz.trace.hist import Histogram

        t = Histogram(x="val")
        spec = t.to_trace_spec()
        assert "axis" in spec.hover.source_modes
        assert "cell" in spec.hover.source_modes

    def test_histogram_has_axis_and_cell_target_modes(self):
        from flexviz.trace.hist import Histogram

        t = Histogram(x="val")
        spec = t.to_trace_spec()
        assert "axis" in spec.hover.target_modes
        assert "cell" in spec.hover.target_modes


class TestHistogramBinEdges:
    """The wire format sends one ``[lo, step, n]`` triple per binned axis."""

    def test_histogram_vertical_has_x_edges(self):
        """_to_update must include the x_edges triple."""
        t = Histogram(x="val", bins=5)
        df = pl.DataFrame({"val": list(range(20))})
        lf = LFQueryBuilder(df.lazy())
        agg_spec = t.get_aggregation_spec(
            update_range={}, schema=df.schema, domains=_domains(lf, t, {})
        )
        result_df, _ = lf.aggregate(filter_exprs=[], agg_specs=[agg_spec])
        updates = t._to_update(result_df).updates
        lo, step, n = updates["x_edges"]
        assert n == 5, "the triple carries the bin count"
        assert step > 0
        assert lo == pytest.approx(0.0)
        assert lo + n * step == pytest.approx(19.0 + _HIST_BIN_EPSILON)
        assert "y_edges" not in updates

    def test_histogram_horizontal_has_y_edges(self):
        """A horizontal histogram bins on y, so the triple lands on y_edges."""
        t = Histogram(y="val", bins=4)
        df = pl.DataFrame({"val": list(range(16))})
        lf = LFQueryBuilder(df.lazy())
        agg_spec = t.get_aggregation_spec(
            update_range={}, schema=df.schema, domains=_domains(lf, t, {})
        )
        result_df, _ = lf.aggregate(filter_exprs=[], agg_specs=[agg_spec])
        updates = t._to_update(result_df).updates
        assert "x_edges" not in updates
        lo, step, n = updates["y_edges"]
        assert n == 4
        assert lo + n * step == pytest.approx(15.0 + _HIST_BIN_EPSILON)

    def test_bin_edges_reproduce_the_bin_centers(self):
        """``lo + (i + 0.5) * step`` is exactly the emitted center of bin i."""
        updates = _aggregate_hist(
            pl.DataFrame({"val": [0.0, 1.0, 2.5, 3.0, 4.0, 5.5, 7.9, 8.0]}), bins=8
        )
        lo, step, n = updates["x_edges"]
        assert updates["x"].to_list() == pytest.approx(
            [lo + (i + 0.5) * step for i in range(n)]
        )

    @pytest.mark.parametrize("scan_source", [False, True])
    def test_bin_centers_match_the_kernel_breakpoints(self, scan_source):
        """The derived centers sit half a bin below the kernel's breakpoints."""
        df = pl.DataFrame({"val": [0.0, 1.0, 2.5, 3.0, 4.0, 5.5, 7.9, 8.0]})
        lf = LFQueryBuilder(df)
        t = Histogram(x="val", bins=8)
        agg_spec = t.get_aggregation_spec(
            {}, schema=lf.schema, domains=_domains(lf, t, {}), scan_source=scan_source
        )
        df_agg, _ = lf.aggregate([], [agg_spec])
        breakpoints = (
            df_agg[t.uid].item().explode().struct.field("breakpoint").to_list()
        )
        lo, step, n = t._to_update(df_agg).updates["x_edges"]
        assert [lo + (i + 0.5) * step for i in range(n)] == pytest.approx(
            [bp - step / 2 for bp in breakpoints]
        )

    def test_bins_1_edges_and_center_are_finite(self):
        """bins=1 is no longer degenerate: one real center, one real triple."""
        updates = _aggregate_hist(pl.DataFrame({"val": [1.0, 2.0, 3.0]}), bins=1)
        assert updates["x"].to_list() == pytest.approx([2.0])
        lo, step, n = updates["x_edges"]
        assert n == 1
        assert math.isfinite(lo) and math.isfinite(step) and step > 0

    def test_temporal_edges_are_epoch_ms(self):
        """A temporal axis sends its triple in Plotly's epoch-ms coordinate."""
        import datetime

        series = pl.datetime_range(
            datetime.datetime(2020, 1, 1),
            datetime.datetime(2020, 1, 9),
            interval="1d",
            eager=True,
            time_unit="ns",
        ).rename("val")
        df = pl.DataFrame([series])
        lf = LFQueryBuilder(df)
        t = Histogram(x="val", bins=4)
        agg_spec = t.get_aggregation_spec(
            {}, schema=lf.schema, domains=_domains(lf, t, {})
        )
        df_agg, _ = lf.aggregate([], [agg_spec])
        updates = t._to_update(df_agg).updates
        lo, step, n = updates["x_edges"]
        # Physical ns bounds, epoch-ms on the wire: 1e-6 per ns.
        phys_lo, phys_hi, phys_n = t._bin_edges
        assert (lo, n) == (phys_lo * 1e-6, phys_n)
        assert lo + n * step == pytest.approx(phys_hi * 1e-6)
        epoch = datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc)
        assert lo == epoch.timestamp() * 1000
        assert step == pytest.approx(2 * 86_400_000.0)
        # The centers stay datetimes so the renderer draws a date axis.
        assert updates["x"].dtype.is_temporal()


# ---- cube descriptors (Phase 1) ----------------------------------------------


class TestCubeDescriptors:
    """Histogram cube source/target descriptors (cross-filter pre-aggregation)."""

    # -- source (free axis) --------------------------------------------------

    def test_source_spec_unzoomed_domain_none(self):
        from flexviz.cube import FreeAxisSpec

        trace = Histogram(x="val", bins=10)
        spec = trace.get_cube_source_spec(None)
        assert isinstance(spec, FreeAxisSpec)
        assert spec.column == "val"
        assert spec.kind == "continuous"
        assert spec.p == 2048
        assert spec.domain is None

    def test_source_spec_zoomed_domain_is_axis_range(self):
        trace = Histogram(x="val", bins=10)
        spec = trace.get_cube_source_spec((10.0, 50.0))
        assert spec is not None
        assert tuple(spec.domain) == (10.0, 50.0)

    def test_source_spec_temporal_kind_from_schema(self):
        schema = pl.Schema({"ts": pl.Datetime("us")})
        trace = Histogram(x="ts", bins=10)
        spec = trace.get_cube_source_spec(None, schema=schema)
        assert spec is not None
        assert spec.kind == "temporal"

    def test_source_spec_continuous_kind_from_float_schema(self):
        schema = pl.Schema({"val": pl.Float64})
        trace = Histogram(x="val", bins=10)
        spec = trace.get_cube_source_spec(None, schema=schema)
        assert spec is not None
        assert spec.kind == "continuous"

    def test_source_spec_defaults_to_continuous_without_schema(self):
        trace = Histogram(x="val", bins=10)
        spec = trace.get_cube_source_spec(None, schema=None)
        assert spec is not None
        assert spec.kind == "continuous"

    def test_grouped_hist_is_still_a_source(self):
        # The brush is on the shared data axis, so grouping does not affect
        # source-ability.
        trace = Histogram(x="val", bins=10, group_by="cat")
        spec = trace.get_cube_source_spec((0.0, 1.0))
        assert spec is not None
        assert spec.column == "val"

    def test_y_oriented_source_uses_data_col(self):
        trace = Histogram(y="val", bins=10)
        spec = trace.get_cube_source_spec((0.0, 1.0))
        assert spec is not None
        assert spec.column == "val"

    # -- target (grouping + measure) -----------------------------------------

    def test_target_spec_ungrouped(self):
        from flexviz.cube import CubeTargetSpec

        trace = Histogram(x="val", bins=25)
        spec = trace.get_cube_target_spec((2.0, 8.0))
        assert isinstance(spec, CubeTargetSpec)
        assert len(spec.target_dims) == 1
        dim = spec.target_dims[0]
        assert dim.column == "val"
        assert dim.kind == "binned"
        lo, hi, n = _snap_range(2.0, 8.0, 25)
        assert dim.bins == n
        assert tuple(dim.domain) == (lo, hi)
        assert spec.measure.agg == "count"

    def test_target_spec_unzoomed_domain_none(self):
        trace = Histogram(x="val", bins=10)
        spec = trace.get_cube_target_spec(None)
        assert spec is not None
        assert spec.target_dims[0].domain is None

    def test_target_domain_is_the_snapped_range_no_epsilon(self):
        # The ENGINE adds _HIST_BIN_EPSILON when resolving domains; the trace
        # emits the snapped viewport, the display grid, and nothing else.
        trace = Histogram(x="val", bins=10)
        spec = trace.get_cube_target_spec((100.0, 400.0))
        assert spec is not None
        lo, hi, n = _snap_range(100.0, 400.0, 10)
        assert tuple(spec.target_dims[0].domain) == (lo, hi)
        assert spec.target_dims[0].bins == n

    def test_target_grid_equals_the_display_grid(self):
        # The client derives bar centers from (domain, bins), so a cube-served
        # bar lands on the server's bar only if both grids agree.
        trace = Histogram(x="val", bins=10)
        axis_range = (100.0, 400.0)
        spec = trace.get_cube_target_spec(axis_range)
        lo, hi, n_bins, _ = trace._histogram_bounds_exprs(axis_range, None)
        dim = spec.target_dims[0]
        # The engine pads the cube dim's hi exactly like the display path.
        assert (dim.domain[0], dim.domain[1] + _HIST_BIN_EPSILON) == (lo, hi)
        assert dim.bins == n_bins

    @pytest.mark.parametrize(
        "viewport",
        [
            ("2024-01-02 03:00:00", "2024-01-05 07:30:00"),
            (1704164400000.0, 1704439800000.0),  # the same instants, epoch-ms
        ],
        ids=["date_string", "epoch_ms"],
    )
    def test_temporal_target_grid_equals_the_display_grid(self, viewport):
        # A temporal viewport reaches the descriptor through the engine, which
        # converts it to physical units. Both spellings must land on the
        # display grid, not one unit factor away from it.
        from flexviz.engine import FlexEngine

        schema = pl.Schema({"t": pl.Datetime("us")})
        trace = Histogram(x="t", bins=7)
        cube_range = FlexEngine._cube_axis_range(
            {"fig": {"x": list(viewport)}}, "fig", "x", schema=schema, column="t"
        )
        dim = trace.get_cube_target_spec(cube_range, schema=schema).target_dims[0]
        lo, hi, n_bins, _ = trace._histogram_bounds_exprs(viewport, None, schema)
        assert (dim.domain[0], dim.domain[1] + _HIST_BIN_EPSILON) == (lo, hi)
        assert dim.bins == n_bins

    def test_grouped_hist_target_dims_binned_then_groups(self):
        # Pinned target-dim order (Phase 2): binned data col first, then the
        # group_by columns as categorical dims, in group_by order.
        schema = pl.Schema({"val": pl.Float64, "cat": pl.String, "sub": pl.String})
        trace = Histogram(x="val", bins=10, group_by=["cat", "sub"])
        spec = trace.get_cube_target_spec((0.0, 1.0), schema=schema)
        assert spec is not None
        assert [(d.column, d.kind) for d in spec.target_dims] == [
            ("val", "binned"),
            ("cat", "categorical"),
            ("sub", "categorical"),
        ]
        assert spec.target_dims[0].bins == 10
        assert tuple(spec.target_dims[0].domain) == (0.0, 1.0)
        assert spec.measure.agg == "count"

    def test_grouped_hist_unzoomed_domain_none(self):
        schema = pl.Schema({"val": pl.Float64, "cat": pl.String})
        trace = Histogram(x="val", bins=10, group_by="cat")
        spec = trace.get_cube_target_spec(None, schema=schema)
        assert spec is not None
        assert spec.target_dims[0].domain is None

    def test_grouped_hist_numeric_group_col_not_a_target(self):
        # String-dtype gate: the codec stringifies categorical dims, so numeric
        # group columns would silently mismatch client-side.
        schema = pl.Schema({"val": pl.Float64, "cat": pl.Int64})
        trace = Histogram(x="val", bins=10, group_by="cat")
        assert trace.get_cube_target_spec((0.0, 1.0), schema=schema) is None

    def test_grouped_hist_without_schema_not_a_target(self):
        # Categorical capability requires a schema to verify the string gate.
        trace = Histogram(x="val", bins=10, group_by="cat")
        assert trace.get_cube_target_spec((0.0, 1.0)) is None
        assert trace.get_cube_target_spec(None) is None

    def test_grouped_hist_reserved_group_col_not_a_target(self):
        # A group column named after a measure partial would collide in the
        # cube frame (contract A reserved names).
        schema = pl.Schema({"val": pl.Float64, "count": pl.String})
        trace = Histogram(x="val", bins=10, group_by="count")
        assert trace.get_cube_target_spec(None, schema=schema) is None

    def test_ungrouped_target_needs_no_schema(self):
        # Regression: the ungrouped path has no categorical dims and stays
        # schema-independent (unchanged from Phase 1).
        trace = Histogram(x="val", bins=10)
        assert trace.get_cube_target_spec(None) is not None

    def test_y_oriented_target_uses_data_col(self):
        trace = Histogram(y="val", bins=10)
        spec = trace.get_cube_target_spec(None)
        assert spec is not None
        assert spec.target_dims[0].column == "val"

    def test_histnorm_percent_is_still_a_target(self):
        # Normalization is client-side from counts; any histnorm stays
        # cube-eligible.
        trace = Histogram(x="val", bins=10, histnorm="percent")
        spec = trace.get_cube_target_spec(None)
        assert spec is not None
        assert spec.measure.agg == "count"


class TestHistogramScanPlanEquivalence:
    """The batch fold a scan source takes must equal the kernel exactly.

    ``scan_source`` only picks a formulation, never a result. The fold runs the
    same kernel per batch, so it counts exactly; it streams a resident frame
    too, which is fine here: the point is the arithmetic, not the source kind.
    """

    _VALUES = [0.0, 1.0, 2.5, 3.0, 4.0, 5.5, 7.9, 8.0]

    @staticmethod
    def _both_updates(
        df: pl.DataFrame,
        bins: int = 8,
        x_range: tuple[float, float] | None = None,
        histnorm: str = "count",
    ) -> tuple[dict, dict]:
        lf = LFQueryBuilder(df)
        trace = Histogram(x="v", bins=bins, histnorm=histnorm)
        update_range = {"x": x_range} if x_range is not None else {}
        kwargs = dict(schema=lf.schema, domains=_domains(lf, trace, update_range))
        out = []
        for scan_source in (False, True):
            spec = trace.get_aggregation_spec(
                update_range, scan_source=scan_source, **kwargs
            )
            assert (spec.plan is not None) is scan_source
            df_agg, _ = lf.aggregate([], [spec])
            updates = trace._to_update(df_agg).updates
            out.append(
                {
                    "x": updates["x"].to_list(),
                    "y": updates["y"].to_list(),
                    "x_edges": updates["x_edges"],
                }
            )
        return out[0], out[1]

    @pytest.mark.parametrize(
        "name,series,bins,x_range,histnorm",
        [
            ("f64", pl.Series("v", _VALUES), 8, None, "count"),
            ("f64_nulls", pl.Series("v", _VALUES + [None]), 8, None, "count"),
            ("f64_nan", pl.Series("v", _VALUES + [float("nan")]), 8, None, "count"),
            ("f32", pl.Series("v", _VALUES, dtype=pl.Float32), 8, None, "count"),
            (
                "i64",
                pl.Series("v", [0, 1, 2, 3, 4, 5, 7, 8], dtype=pl.Int64),
                8,
                None,
                "count",
            ),
            (
                "i32_nulls",
                pl.Series("v", [0, 1, 2, None, 4, 5, 7, 8], dtype=pl.Int32),
                8,
                None,
                "count",
            ),
            (
                "u32",
                pl.Series("v", [0, 1, 2, 3, 4, 5, 7, 8], dtype=pl.UInt32),
                8,
                None,
                "count",
            ),
            # A constant column is the narrowest span the trace can build: the
            # engine pads hi by _HIST_BIN_EPSILON, so the span never inverts.
            ("constant", pl.Series("v", [3.0] * 8), 8, None, "count"),
            ("viewport", pl.Series("v", _VALUES), 8, (2.0, 6.0), "count"),
            ("all_below_lo", pl.Series("v", [-5.0, -3.0]), 8, (0.0, 8.0), "count"),
            ("one_bin", pl.Series("v", _VALUES), 1, None, "count"),
            ("empty", pl.Series("v", [], dtype=pl.Float64), 8, None, "count"),
            ("density", pl.Series("v", _VALUES), 8, None, "probability density"),
        ],
    )
    def test_plan_matches_kernel(self, name, series, bins, x_range, histnorm):
        resident, scanned = self._both_updates(
            pl.DataFrame([series]), bins=bins, x_range=x_range, histnorm=histnorm
        )
        assert resident == scanned

    def test_datetime_column(self):
        import datetime

        series = pl.datetime_range(
            datetime.datetime(2020, 1, 1),
            datetime.datetime(2020, 1, 9),
            interval="1d",
            eager=True,
        ).rename("v")
        resident, scanned = self._both_updates(pl.DataFrame([series]))
        assert resident == scanned

    def test_fold_merges_across_batches(self, monkeypatch):
        """A frame larger than one chunk must fold to the single-batch counts."""
        monkeypatch.setattr(helpers_mod, "_FOLD_CHUNK_ROWS", 7)
        seen: list[tuple[int, int]] = []
        original = pl.LazyFrame.collect_batches

        def spy(self, *args, **kwargs):
            batches = list(original(self, *args, **kwargs))
            seen.append((kwargs["chunk_size"], len(batches)))
            return batches

        monkeypatch.setattr(pl.LazyFrame, "collect_batches", spy)

        df = pl.DataFrame({"v": [float(i % 13) for i in range(50)]})
        resident, scanned = self._both_updates(df)
        assert resident == scanned
        assert seen == [(7, 8)], seen

    def test_scan_source_with_a_viewport_matches_the_resident_kernel(self, tmp_path):
        """The fold rejects the out-of-viewport rows in the scan; the kernel
        filters inside its expression. Both must count the same rows."""
        df = pl.DataFrame({"v": [float(i) / 4 for i in range(200)]})
        path = tmp_path / "d.parquet"
        df.write_parquet(path)
        update_range = {"x": (7.3, 31.8)}

        out = []
        for src in (df, pl.scan_parquet(path)):
            lf = LFQueryBuilder(src)
            trace = Histogram(x="v", bins=8)
            spec = trace.get_aggregation_spec(
                update_range,
                schema=lf.schema,
                domains=_domains(lf, trace, update_range),
                scan_source=lf.is_scan,
            )
            assert (spec.plan is not None) is lf.is_scan
            df_agg, _ = lf.aggregate([], [spec])
            updates = trace._to_update(df_agg).updates
            out.append((updates["x"].to_list(), updates["y"].to_list()))

        assert out[0] == out[1]
        # The viewport must actually drop rows, else the test proves nothing.
        assert sum(out[0][1]) < df.height


class TestHistogramGroupedPlanEquivalence:
    """A grouped histogram runs the plan on both source kinds.

    The reference is the fused grouped query the kernel used to run: the
    ``fixed_hist`` expression inside ``group_by().agg()``. Children are
    compared after ``_to_grouped_update``, which is what the engine sends.
    """

    _VALUES = [0.0, 1.0, 2.5, 3.0, 4.0, 5.5, 7.9, 8.0]

    @staticmethod
    def _children(result) -> list[dict]:
        return [
            {
                "child_uid": c.child_uid,
                "group_value_key": c.group_value_key,
                "x": c.updates["x"].to_list(),
                "y": c.updates["y"].to_list(),
                "x_edges": c.updates["x_edges"],
            }
            for c in result.group_results
        ]

    def _both(
        self,
        df: pl.DataFrame,
        group_by,
        bins: int = 8,
        x_range: tuple[float, float] | None = None,
        histnorm: str = "count",
    ) -> tuple[list[dict], list[dict]]:
        lf = LFQueryBuilder(df)
        trace = Histogram(x="v", bins=bins, histnorm=histnorm, group_by=group_by)
        update_range = {"x": x_range} if x_range is not None else {}
        domains = _domains(lf, trace, update_range)

        specs = [
            trace.get_aggregation_spec(
                update_range, schema=lf.schema, domains=domains, scan_source=scan
            )
            for scan in (False, True)
        ]
        for spec in specs:
            assert spec.plan is not None
            assert spec.agg_exprs == ()
        _, grouped = lf.aggregate([], [specs[0]])
        got = self._children(trace._to_grouped_update(grouped[trace.uid]))

        cols = list(trace.group_by_cols)
        lo_expr, hi_expr, n_bins, mask = trace._histogram_bounds_exprs(
            x_range, domains, lf.schema
        )
        ldf = df.lazy()
        if mask is not None:
            ldf = ldf.filter(mask)
        ref_df = (
            ldf.group_by(cols)
            .agg(
                pl.col("v")
                .flexviz.fixed_hist(lo_expr, hi_expr, n_bins=n_bins)
                .implode()
                .alias(trace.uid)
            )
            .sort(cols)
            .collect()
        )
        return got, self._children(trace._to_grouped_update(ref_df))

    @pytest.mark.parametrize(
        "name,frame,group_by,x_range,histnorm",
        [
            (
                "int_groups",
                {"v": _VALUES, "g": [0, 1, 0, 1, 2, 2, 0, 1]},
                "g",
                None,
                "count",
            ),
            (
                "string_groups",
                {"v": _VALUES, "g": list("abcabcab")},
                "g",
                None,
                "count",
            ),
            (
                "two_group_cols",
                {"v": _VALUES, "g": list("aabbaabb"), "h": [0, 1] * 4},
                ["g", "h"],
                None,
                "count",
            ),
            (
                "null_group",
                {"v": _VALUES, "g": ["a", None, "b", None, "a", "b", None, "a"]},
                "g",
                None,
                "count",
            ),
            (
                "nan_and_null_values",
                {
                    "v": [0.0, float("nan"), 2.5, None, 4.0, float("nan"), 7.9, None],
                    "g": list("aabbaabb"),
                },
                "g",
                None,
                "count",
            ),
            (
                "viewport",
                {"v": _VALUES, "g": list("abcabcab")},
                "g",
                (2.0, 6.0),
                "count",
            ),
            (
                "density",
                {"v": _VALUES, "g": list("abcabcab")},
                "g",
                None,
                "probability density",
            ),
            (
                "empty_viewport",
                {"v": _VALUES, "g": list("abcabcab")},
                "g",
                (100.0, 200.0),
                "count",
            ),
        ],
    )
    def test_plan_matches_kernel(self, name, frame, group_by, x_range, histnorm):
        got, ref = self._both(
            pl.DataFrame(frame), group_by, x_range=x_range, histnorm=histnorm
        )
        assert got == ref

    def test_empty_viewport_yields_no_children(self):
        got, ref = self._both(
            pl.DataFrame({"v": self._VALUES, "g": list("abcabcab")}),
            "g",
            x_range=(100.0, 200.0),
        )
        assert got == [] and ref == []


class TestHistogramStreamingPlanArithmetic:
    """Bounds the trace itself cannot build, checked plan against kernel.

    The plan serves the grouped path only, so it bins one constant group here:
    the bin arithmetic under test is the same either way.
    """

    @staticmethod
    def _plan_rows(df: pl.DataFrame, lo: float, hi: float, bins: int) -> list:
        run = _streaming_hist_plan(
            pl.col("v"), pl.lit(lo), pl.lit(hi), bins, "u", ("g",)
        )
        out = run(df.with_columns(g=pl.lit("a")).lazy())
        return out["u"].item().struct.unnest().rows()

    @staticmethod
    def _kernel_rows(df: pl.DataFrame, lo: float, hi: float, bins: int) -> list:
        expr = pl.col("v").flexviz.fixed_hist(pl.lit(lo), pl.lit(hi), n_bins=bins)
        agg = df.select(expr.implode().alias("u"))
        return agg["u"].item().explode().struct.unnest().rows()

    @pytest.mark.parametrize(
        "name,values,lo,hi,bins",
        [
            # lo == hi: every value lands in bin 0 and every breakpoint is lo.
            ("degenerate", [0.0, 1.0, 2.5, 3.0, 8.0], 3.0, 3.0, 8),
            ("degenerate_nan", [1.0, float("nan"), 3.0], 3.0, 3.0, 8),
            ("below_lo", [-5.0, -3.0], 0.0, 8.0, 8),
            ("above_hi", [11.0, 42.0], 0.0, 8.0, 8),
        ],
    )
    def test_matches_kernel(self, name, values, lo, hi, bins):
        df = pl.DataFrame({"v": values}, schema={"v": pl.Float64})
        assert self._plan_rows(df, lo, hi, bins) == self._kernel_rows(df, lo, hi, bins)

    def test_inverted_bounds_raise(self):
        # The kernel raises on lo > hi; the plan must not silently bin instead.
        df = pl.DataFrame({"v": [1.0, 2.0]}, schema={"v": pl.Float64})
        with pytest.raises(ValueError, match="inverted"):
            self._plan_rows(df, 5.0, 1.0, 8)

    def test_bin_edge_epsilon_is_load_bearing(self):
        """Values on a bin edge of a non-round domain need the round epsilon.

        lo=0.1, hi=0.7, bins=6: without the epsilon 0.3 falls back into bin 1,
        so bin 1 counts 2 and bin 2 counts 0.
        """
        lo, hi, bins = 0.1, 0.7, 6
        values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.15, 0.65]
        df = pl.DataFrame({"v": values}, schema={"v": pl.Float64})

        expected = [2, 1, 1, 1, 1, 3]
        assert [c for _, c in self._kernel_rows(df, lo, hi, bins)] == expected
        assert [c for _, c in self._plan_rows(df, lo, hi, bins)] == expected

        # The same plan with the epsilon dropped.
        scale = bins / (hi - lo)
        no_eps = (
            ((pl.col("v").cast(pl.Float64) - lo) * scale + 0.0)
            .clip(0, bins - 1)
            .cast(pl.Int32, strict=False)
        )
        counted = dict(
            df.lazy()
            .group_by(no_eps.alias("__b"))
            .agg(pl.len().alias("count"))
            .collect()
            .iter_rows()
        )
        assert [counted.get(i, 0) for i in range(bins)] == [2, 2, 0, 1, 1, 3]

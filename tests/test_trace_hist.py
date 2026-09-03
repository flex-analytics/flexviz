"""Unit tests for Histogram trace."""

from __future__ import annotations

from collections.abc import Sequence
import math

import polars as pl
import pytest

from flexviz.LF import LFQueryBuilder
from flexviz.spec import TraceSpec
from flexviz.trace.hist import Histogram

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
        """With a viewport range the bin centers must equal lo + (i + 0.5) * step."""
        from flexviz.trace.hist import _HIST_BIN_EPSILON

        lf = LFQueryBuilder(small_df)
        bins = 5
        lo, hi = 100.0, 400.0
        trace = Histogram(x="val", bins=bins)
        spec = trace.get_aggregation_spec({"x": [lo, hi]}, schema=lf.schema)
        df_agg, _ = lf.aggregate([], [spec])
        centers = list(trace._to_update(df_agg).updates["x"])
        step = (hi - lo + _HIST_BIN_EPSILON) / bins
        expected = [lo + (i + 0.5) * step for i in range(bins)]
        assert len(centers) == bins
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


class TestHistogramHoverBounds:
    def test_histogram_vertical_has_hover_bounds(self):
        """_to_update must include hover_bounds with x0/x1 per bar."""
        t = Histogram(x="val", bins=5)
        df = pl.DataFrame({"val": list(range(20))})
        lf = LFQueryBuilder(df.lazy())
        schema = df.schema
        agg_spec = t.get_aggregation_spec(
            update_range={}, schema=schema, domains=_domains(lf, t, {})
        )
        result_df, _ = lf.aggregate(filter_exprs=[], agg_specs=[agg_spec])
        tr = t._to_update(result_df)
        assert "hover_bounds" in tr.updates, "hover_bounds must be in updates"
        bounds = tr.updates["hover_bounds"]
        assert isinstance(bounds, list), "hover_bounds must be a list"
        assert len(bounds) == 5, "one bounds entry per bin"
        for b in bounds:
            assert "x0" in b and "x1" in b, "each entry must have x0, x1"
            assert b["x1"] > b["x0"], "x1 must be > x0"

    def test_histogram_horizontal_has_hover_bounds_y(self):
        """Horizontal histogram must use y0/y1."""
        t = Histogram(y="val", bins=4)
        df = pl.DataFrame({"val": list(range(16))})
        lf = LFQueryBuilder(df.lazy())
        schema = df.schema
        agg_spec = t.get_aggregation_spec(
            update_range={}, schema=schema, domains=_domains(lf, t, {})
        )
        result_df, _ = lf.aggregate(filter_exprs=[], agg_specs=[agg_spec])
        tr = t._to_update(result_df)
        bounds = tr.updates["hover_bounds"]
        assert len(bounds) == 4
        for b in bounds:
            assert (
                "y0" in b and "y1" in b
            ), "horizontal histogram bounds must have y0, y1"
            assert "x0" not in b

    def test_histogram_hover_bounds_not_in_x_or_y(self):
        """hover_bounds must not shadow the rendered x/y arrays."""
        t = Histogram(x="val", bins=3)
        df = pl.DataFrame({"val": list(range(12))})
        lf = LFQueryBuilder(df.lazy())
        result_df, _ = lf.aggregate(
            filter_exprs=[],
            agg_specs=[
                t.get_aggregation_spec({}, df.schema, domains=_domains(lf, t, {}))
            ],
        )
        tr = t._to_update(result_df)
        assert "x" in tr.updates, "x must still be present"
        assert "y" in tr.updates, "y (counts) must still be present"
        assert tr.updates["hover_bounds"] is not tr.updates["x"]

    def test_histogram_hover_bounds_length_matches_x_for_bins_1(self):
        """hover_bounds length must equal len(x) even for bins=1 degenerate case."""
        t = Histogram(x="val", bins=1)
        df = pl.DataFrame({"val": [1.0, 2.0, 3.0]})
        lf = LFQueryBuilder(df.lazy())
        result_df, _ = lf.aggregate(
            filter_exprs=[],
            agg_specs=[
                t.get_aggregation_spec({}, df.schema, domains=_domains(lf, t, {}))
            ],
        )
        tr = t._to_update(result_df)
        x_len = len(tr.updates["x"])
        bounds_len = len(tr.updates["hover_bounds"])
        assert (
            bounds_len == x_len
        ), f"hover_bounds length ({bounds_len}) must equal x length ({x_len})"


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
        assert dim.bins == 25
        assert tuple(dim.domain) == (2.0, 8.0)
        assert spec.measure.agg == "count"

    def test_target_spec_unzoomed_domain_none(self):
        trace = Histogram(x="val", bins=10)
        spec = trace.get_cube_target_spec(None)
        assert spec is not None
        assert spec.target_dims[0].domain is None

    def test_target_domain_is_axis_range_verbatim_no_epsilon(self):
        # The ENGINE adds _HIST_BIN_EPSILON when resolving domains; the trace
        # must pass the viewport through unchanged.
        trace = Histogram(x="val", bins=10)
        spec = trace.get_cube_target_spec((100.0, 400.0))
        assert spec is not None
        assert tuple(spec.target_dims[0].domain) == (100.0, 400.0)

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

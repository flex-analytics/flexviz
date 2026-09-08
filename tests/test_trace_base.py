"""Unit tests for shared helpers in trace.base."""

from __future__ import annotations

import datetime as dt
import re

import polars as pl
import pytest

from flexviz.trace.base import (
    _child_uid_for_group,
    _dtype_for_col,
    _group_value_key,
    _range_filter_expr,
    _typed_range_bounds,
    _typed_temporal_lit,
)


class TestDtypeForCol:
    def test_returns_dtype_for_present_column(self):
        schema = pl.Schema({"x": pl.Int64, "ts": pl.Datetime("us")})
        assert _dtype_for_col(schema, "x") == pl.Int64
        assert _dtype_for_col(schema, "ts") == pl.Datetime("us")

    def test_returns_none_for_missing_column(self):
        schema = pl.Schema({"x": pl.Int64})
        assert _dtype_for_col(schema, "missing") is None

    def test_returns_none_when_schema_is_none(self):
        assert _dtype_for_col(None, "x") is None


class TestTypedTemporalLit:
    def test_date_only_string(self):
        dtype = pl.Datetime("us")
        expr = _typed_temporal_lit("2020-01-15", dtype)
        # Should evaluate without error and produce a scalar
        result = pl.select(expr).item()
        assert result == dt.datetime(2020, 1, 15, 0, 0, 0)

    def test_datetime_with_space_separator(self):
        dtype = pl.Datetime("us")
        expr = _typed_temporal_lit("2020-06-01 12:30:45", dtype)
        result = pl.select(expr).item()
        assert result == dt.datetime(2020, 6, 1, 12, 30, 45)

    def test_datetime_with_T_separator(self):
        dtype = pl.Datetime("us")
        expr = _typed_temporal_lit("2020-06-01T12:30:45", dtype)
        result = pl.select(expr).item()
        assert result == dt.datetime(2020, 6, 1, 12, 30, 45)

    def test_datetime_with_microseconds(self):
        dtype = pl.Datetime("us")
        expr = _typed_temporal_lit("2020-06-01 12:30:45.123456", dtype)
        result = pl.select(expr).item()
        assert result == dt.datetime(2020, 6, 1, 12, 30, 45, 123456)

    def test_datetime_with_utc_z_suffix(self):
        dtype = pl.Datetime("us", "UTC")
        expr = _typed_temporal_lit("2020-06-01T12:30:45.123456Z", dtype)
        result = pl.select(expr).item()
        assert result == dt.datetime(
            2020, 6, 1, 12, 30, 45, 123456, tzinfo=dt.timezone.utc
        )

    def test_python_datetime_object(self):
        dtype = pl.Datetime("us")
        val = dt.datetime(2021, 3, 15, 9, 0, 0)
        expr = _typed_temporal_lit(val, dtype)
        result = pl.select(expr).item()
        assert result == val

    def test_epoch_milliseconds_number(self):
        dtype = pl.Datetime("us", "UTC")
        epoch_ms = 1_704_067_200_000.0  # 2024-01-01T00:00:00Z
        expr = _typed_temporal_lit(epoch_ms, dtype)
        result = pl.select(expr).item()
        assert result == dt.datetime(2024, 1, 1, 0, 0, tzinfo=dt.timezone.utc)

    def test_utc_z_suffix_on_naive_column_is_filterable(self):
        """A Z-suffixed string against a NAIVE Datetime column must produce a
        naive literal that compares without a tz-mismatch SchemaError."""
        dtype = pl.Datetime("us")
        expr = _typed_temporal_lit("2020-06-01T12:30:45Z", dtype)
        result = pl.select(expr).item()
        assert result == dt.datetime(2020, 6, 1, 12, 30, 45)
        assert result.tzinfo is None

    def test_iso_offset_on_naive_column_is_filterable(self):
        dtype = pl.Datetime("us")
        expr = _typed_temporal_lit("2020-06-01T12:30:45+00:00", dtype)
        result = pl.select(expr).item()
        assert result == dt.datetime(2020, 6, 1, 12, 30, 45)
        assert result.tzinfo is None

    def test_epoch_milliseconds_on_naive_column_is_filterable(self):
        dtype = pl.Datetime("us")
        epoch_ms = 1_704_067_200_000.0  # 2024-01-01T00:00:00Z
        expr = _typed_temporal_lit(epoch_ms, dtype)
        result = pl.select(expr).item()
        assert result == dt.datetime(2024, 1, 1, 0, 0)
        assert result.tzinfo is None

    def test_epoch_milliseconds_on_naive_date_column(self):
        dtype = pl.Date
        epoch_ms = 1_704_067_200_000.0  # 2024-01-01T00:00:00Z
        expr = _typed_temporal_lit(epoch_ms, dtype)
        result = pl.select(expr).item()
        assert result == dt.date(2024, 1, 1)

    #: µs epoch of 2020-01-01T00:00:00Z — the instant all the offset spellings
    #: below denote.
    _EPOCH_2020_US = 1_577_836_800_000_000

    def test_positive_offset_on_utc_column_converts_to_instant(self):
        """A tz-aware offset string against a tz-aware (UTC) column must convert
        to the true UTC instant — not raise a tz-mismatch TypeError. Regression:
        ``pl.lit(datetime(+02:00), dtype=Datetime('us','UTC'))`` raised because
        the value's offset differs from the column tz."""
        dtype = pl.Datetime("us", "UTC")
        # 2020-01-01T02:00:00+02:00 == 2020-01-01T00:00:00Z
        expr = _typed_temporal_lit("2020-01-01T02:00:00+02:00", dtype)
        assert pl.select(expr.to_physical()).item() == self._EPOCH_2020_US

    def test_negative_offset_on_utc_column_converts_to_instant(self):
        dtype = pl.Datetime("us", "UTC")
        # 2019-12-31T18:30:00-05:30 == 2020-01-01T00:00:00Z
        expr = _typed_temporal_lit("2019-12-31T18:30:00-05:30", dtype)
        assert pl.select(expr.to_physical()).item() == self._EPOCH_2020_US

    def test_offset_on_named_tz_column_converts_to_instant(self):
        """The same conversion must work for a named (non-UTC) tz column; the
        instant is tz-independent so the physical epoch is identical."""
        dtype = pl.Datetime("us", "Europe/Brussels")
        expr = _typed_temporal_lit("2020-01-01T02:00:00+02:00", dtype)
        assert pl.select(expr.to_physical()).item() == self._EPOCH_2020_US


class TestTypedRangeBounds:
    def test_numeric_bounds_with_schema(self):
        schema = pl.Schema({"x": pl.Int64})
        bounds = _typed_range_bounds("x", (1, 10), schema=schema)
        assert bounds is not None
        lo, hi = bounds
        df = pl.DataFrame({"x": [0, 5, 11]})
        got = df.select(pl.col("x").is_between(lo, hi).alias("ok"))["ok"].to_list()
        assert got == [False, True, False]

    def test_datetime_string_bounds_with_schema(self):
        schema = pl.Schema({"ts": pl.Datetime("us")})
        start = "2020-01-01 00:00:10"
        end = "2020-01-01 00:00:20"
        bounds = _typed_range_bounds("ts", (start, end), schema=schema)
        assert bounds is not None
        lo, hi = bounds

        ts = [dt.datetime(2020, 1, 1) + dt.timedelta(seconds=i) for i in range(30)]
        df = pl.DataFrame({"ts": ts})
        out = df.select(pl.col("ts").is_between(lo, hi).alias("ok"))["ok"]
        assert out.sum() == 11

    def test_datetime_epoch_millisecond_bounds_with_schema(self):
        schema = pl.Schema({"ts": pl.Datetime("us", "UTC")})
        base = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
        ts = [base + dt.timedelta(hours=i) for i in range(48)]
        df = pl.DataFrame(
            {"ts": pl.Series("ts", ts, dtype=pl.Datetime("us", time_zone="UTC"))}
        )
        start_ms = (base + dt.timedelta(hours=10)).timestamp() * 1000.0
        end_ms = (base + dt.timedelta(hours=20)).timestamp() * 1000.0
        bounds = _typed_range_bounds("ts", (start_ms, end_ms), schema=schema)
        assert bounds is not None
        lo, hi = bounds
        out = df.select(pl.col("ts").is_between(lo, hi).alias("ok"))["ok"]
        assert out.sum() == 11

    def test_naive_datetime_z_string_bounds_filter(self):
        """A range brush sending Z-suffixed ISO strings on a NAIVE datetime
        axis must compile to a filterable expr (no tz-mismatch SchemaError)."""
        schema = pl.Schema({"ts": pl.Datetime("us")})
        ts = [dt.datetime(2020, 1, 1) + dt.timedelta(seconds=i) for i in range(30)]
        df = pl.DataFrame({"ts": ts})
        bounds = _typed_range_bounds(
            "ts", ("2020-01-01T00:00:10Z", "2020-01-01T00:00:20Z"), schema=schema
        )
        assert bounds is not None
        lo, hi = bounds
        out = df.select(pl.col("ts").is_between(lo, hi).alias("ok"))["ok"]
        assert out.sum() == 11

    def test_naive_datetime_epoch_ms_bounds_filter(self):
        """A range brush sending numeric epoch-ms on a NAIVE datetime axis must
        compile to a filterable expr (no tz-mismatch SchemaError)."""
        schema = pl.Schema({"ts": pl.Datetime("us")})
        base = dt.datetime(2024, 1, 1)
        ts = [base + dt.timedelta(hours=i) for i in range(48)]
        df = pl.DataFrame({"ts": ts})
        # epoch-ms interpreted as UTC wall-clock, matching the naive column
        start_ms = (base + dt.timedelta(hours=10)).replace(
            tzinfo=dt.timezone.utc
        ).timestamp() * 1000.0
        end_ms = (base + dt.timedelta(hours=20)).replace(
            tzinfo=dt.timezone.utc
        ).timestamp() * 1000.0
        bounds = _typed_range_bounds("ts", (start_ms, end_ms), schema=schema)
        assert bounds is not None
        lo, hi = bounds
        out = df.select(pl.col("ts").is_between(lo, hi).alias("ok"))["ok"]
        assert out.sum() == 11

    def test_none_range_returns_none(self):
        assert _typed_range_bounds("x", None, schema=None) is None

    def test_float_bounds_on_int64_column(self):
        """Float bounds use ceil(lo)/floor(hi) so that is_between matches the viewport.
        lo=99.9 → ceil → 100, hi=100.1 → floor → 100: only x=100 included.
        x=99 is correctly excluded because 99 < 99.9."""
        schema = pl.Schema({"x": pl.Int64})
        bounds = _typed_range_bounds("x", (99.9, 100.1), schema=schema)
        assert bounds is not None
        lo, hi = bounds
        df = pl.DataFrame({"x": [99, 100, 101]})
        filtered = (
            df.lazy().filter(pl.col("x").is_between(lo, hi)).collect()["x"].to_list()
        )
        assert filtered == [100]  # ceil(99.9)=100, floor(100.1)=100

    def test_exact_float_on_int64_is_included(self):
        schema = pl.Schema({"x": pl.Int64})
        bounds = _typed_range_bounds("x", (100.0, 200.0), schema=schema)
        assert bounds is not None
        lo, hi = bounds
        df = pl.DataFrame({"x": [100, 150, 200, 201]})
        filtered = (
            df.lazy().filter(pl.col("x").is_between(lo, hi)).collect()["x"].to_list()
        )
        assert 100 in filtered
        assert 200 in filtered
        assert 201 not in filtered

    def test_negative_float_on_int64(self):
        schema = pl.Schema({"x": pl.Int64})
        bounds = _typed_range_bounds("x", (-5.5, 5.5), schema=schema)
        assert bounds is not None
        lo, hi = bounds
        df = pl.DataFrame({"x": [-6, -5, 0, 5, 6]})
        filtered = (
            df.lazy().filter(pl.col("x").is_between(lo, hi)).collect()["x"].to_list()
        )
        assert filtered == [-5, 0, 5]

    def test_positive_fractional_lo_excludes_integer_below(self):
        """lo=0.25 → ceil → 1: x=0 must be excluded (0 < 0.25)."""
        schema = pl.Schema({"x": pl.Int64})
        bounds = _typed_range_bounds("x", (0.25, 10.0), schema=schema)
        assert bounds is not None
        lo, hi = bounds
        df = pl.DataFrame({"x": [-1, 0, 1, 10, 11]})
        filtered = (
            df.lazy().filter(pl.col("x").is_between(lo, hi)).collect()["x"].to_list()
        )
        assert 0 not in filtered  # 0 < 0.25 → must be excluded
        assert 1 in filtered
        assert 10 in filtered
        assert 11 not in filtered

    def test_negative_fractional_hi_excludes_integer_above(self):
        """hi=-0.25 → floor → -1: x=0 must be excluded (0 > -0.25)."""
        schema = pl.Schema({"x": pl.Int64})
        bounds = _typed_range_bounds("x", (-10.0, -0.25), schema=schema)
        assert bounds is not None
        lo, hi = bounds
        df = pl.DataFrame({"x": [-11, -10, -1, 0, 1]})
        filtered = (
            df.lazy().filter(pl.col("x").is_between(lo, hi)).collect()["x"].to_list()
        )
        assert 0 not in filtered  # 0 > -0.25 → must be excluded
        assert -1 in filtered
        assert -10 in filtered
        assert -11 not in filtered

    def test_fractional_bounds_both_sides_integer_only(self):
        """lo=0.9, hi=1.1 → ceil/floor → is_between(1, 1): only x=1 matches."""
        schema = pl.Schema({"x": pl.Int64})
        bounds = _typed_range_bounds("x", (0.9, 1.1), schema=schema)
        assert bounds is not None
        lo, hi = bounds
        df = pl.DataFrame({"x": [0, 1, 2]})
        filtered = (
            df.lazy().filter(pl.col("x").is_between(lo, hi)).collect()["x"].to_list()
        )
        assert filtered == [1]

    def test_exact_integer_float_bounds_unchanged(self):
        """Exact floats like 5.0 round to the same integer under ceil/floor."""
        schema = pl.Schema({"x": pl.Int64})
        bounds = _typed_range_bounds("x", (5.0, 10.0), schema=schema)
        assert bounds is not None
        lo, hi = bounds
        df = pl.DataFrame({"x": [4, 5, 10, 11]})
        filtered = (
            df.lazy().filter(pl.col("x").is_between(lo, hi)).collect()["x"].to_list()
        )
        assert filtered == [5, 10]

    def test_int32_column_fractional_bounds(self):
        """ceil/floor fix applies to i32 columns too."""
        schema = pl.Schema({"x": pl.Int32})
        bounds = _typed_range_bounds("x", (0.25, 5.75), schema=schema)
        assert bounds is not None
        lo, hi = bounds
        df = pl.DataFrame({"x": pl.Series([0, 1, 5, 6], dtype=pl.Int32)})
        filtered = (
            df.lazy().filter(pl.col("x").is_between(lo, hi)).collect()["x"].to_list()
        )
        assert 0 not in filtered  # 0 < 0.25
        assert 1 in filtered
        assert 5 in filtered
        assert 6 not in filtered  # 6 > 5.75

    def test_no_schema_falls_back_to_untyped_literals(self):
        bounds = _typed_range_bounds("x", (1.0, 9.0), schema=None)
        assert bounds is not None
        lo, hi = bounds
        df = pl.DataFrame({"x": [0.5, 5.0, 9.5]})
        filtered = (
            df.lazy().filter(pl.col("x").is_between(lo, hi)).collect()["x"].to_list()
        )
        assert filtered == [5.0]


class TestRangeFilterExpr:
    def test_returns_none_when_range_none(self):
        assert _range_filter_expr("x", None, schema=None) is None

    def test_builds_filter_expr(self):
        expr = _range_filter_expr("x", (2, 4), schema=pl.Schema({"x": pl.Int64}))
        assert expr is not None
        df = pl.DataFrame({"x": [1, 2, 3, 4, 5]})
        filtered = df.lazy().filter(expr).collect()["x"].to_list()
        assert filtered == [2, 3, 4]


class TestRangeFilterExprsMethod:
    """Tests for the FlexTrace._range_filter_exprs convenience instance method."""

    def test_returns_empty_list_for_none_range(self):
        from flexviz.trace.line import LinePlot

        trace = LinePlot(x="ts", y="val")
        exprs = trace._range_filter_exprs("ts", None)
        assert exprs == []

    def test_returns_single_element_list_for_valid_range(self):
        from flexviz.trace.line import LinePlot

        trace = LinePlot(x="ts", y="val")
        exprs = trace._range_filter_exprs(
            "ts", (10, 50), schema=pl.Schema({"ts": pl.Int64})
        )
        assert len(exprs) == 1
        df = pl.DataFrame({"ts": [5, 10, 30, 50, 55]})
        filtered = df.lazy().filter(*exprs).collect()["ts"].to_list()
        assert filtered == [10, 30, 50]


class TestTraceResult:
    """Tests for the TraceResult container."""

    def test_returns_trace_result_from_line(self):
        from flexviz.LF import LFQueryBuilder
        from flexviz.trace.base import TraceResult
        from flexviz.trace.line import LinePlot

        df = pl.DataFrame(
            {"ts": list(range(100)), "val": [float(i) for i in range(100)]}
        )
        lf = LFQueryBuilder(df)
        trace = LinePlot(x="ts", y="val", n_points=50)
        agg = trace.get_aggregation_spec(
            {}, schema=lf.schema, domains=lf.physical_minmax(["ts"], memoize=False)
        )
        df_agg, _ = lf.aggregate([], [agg])
        result = trace._to_update(df_agg)
        assert isinstance(result, TraceResult)
        assert "x" in result.updates
        assert "y" in result.updates

    def test_returns_trace_result_from_histogram(self):
        from flexviz.LF import LFQueryBuilder
        from flexviz.trace.base import TraceResult
        from flexviz.trace.hist import Histogram

        df = pl.DataFrame({"val": [float(i) for i in range(100)]})
        lf = LFQueryBuilder(df)
        trace = Histogram(x="val", bins=10)
        domains = lf.physical_minmax(
            list(trace.domain_cols({})), lf.schema, memoize=False
        )
        agg = trace.get_aggregation_spec({}, schema=lf.schema, domains=domains)
        df_agg, _ = lf.aggregate([], [agg])
        result = trace._to_update(df_agg)
        assert isinstance(result, TraceResult)
        assert isinstance(result.updates["x"], (list, pl.Series))

    def test_returns_trace_result_from_boxplot(self):
        from flexviz.LF import LFQueryBuilder
        from flexviz.trace.base import TraceResult
        from flexviz.trace.box import BoxPlot

        df = pl.DataFrame({"val": [float(i) for i in range(100)]})
        lf = LFQueryBuilder(df)
        trace = BoxPlot(y="val")
        agg = trace.get_aggregation_spec({}, schema=lf.schema)
        df_agg, _ = lf.aggregate([], [agg])
        result = trace._to_update(df_agg)
        assert isinstance(result, TraceResult)
        assert "median" in result.updates


class TestGroupValueKeyConsistency:
    """group_value_key in GroupedChildResult must match _group_value_key(gv)."""

    @pytest.fixture()
    def grouped_df(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "ts": list(range(100)),
                "val": [float(i) for i in range(100)],
                "grp": [i % 3 for i in range(100)],
            }
        )

    def _run_grouped(self, trace, df):
        from flexviz.LF import GroupedAggregationSpec, LFQueryBuilder

        lf = LFQueryBuilder(df)
        cols = trace.domain_cols({})
        # Only Histogram/Histogram2D/LinePlot accept `domains`; box and bar
        # never need it (domain_cols is always empty for them).
        kwargs = (
            {"domains": lf.physical_minmax(list(cols), lf.schema, memoize=False)}
            if cols
            else {}
        )
        spec = trace.get_aggregation_spec({}, schema=lf.schema, **kwargs)

        if isinstance(spec, GroupedAggregationSpec):
            _, grouped = lf.aggregate([], [spec])
            return trace._to_grouped_update(grouped[trace.uid])
        agg_specs, _ = lf.aggregate([], [spec])
        return trace._to_grouped_update(agg_specs)

    def test_line_int_group_values(self, grouped_df):
        from flexviz.trace.line import LinePlot

        trace = LinePlot(x="ts", y="val", group_by="grp")
        result = self._run_grouped(trace, grouped_df)
        for child in result.group_results:
            gv = int(child.group_value_key)
            assert child.group_value_key == _group_value_key(gv)
            assert child.child_uid == _child_uid_for_group(trace.uid, gv)

    def test_hist_int_group_values(self, grouped_df):
        from flexviz.trace.hist import Histogram

        trace = Histogram(x="val", group_by="grp")
        result = self._run_grouped(trace, grouped_df)
        for child in result.group_results:
            gv = int(child.group_value_key)
            assert child.group_value_key == _group_value_key(gv)

    def test_box_int_group_values(self, grouped_df):
        from flexviz.trace.box import BoxPlot

        trace = BoxPlot(y="val", group_by="grp")
        result = self._run_grouped(trace, grouped_df)
        for child in result.group_results:
            gv = int(child.group_value_key)
            assert child.group_value_key == _group_value_key(gv)

    def test_bar_int_group_values(self):
        from flexviz.trace.bar import BarPlot

        df = pl.DataFrame(
            {"cat": ["A", "A", "B", "B"], "grp": [1, 2, 1, 2], "val": [10, 20, 30, 40]}
        )
        trace = BarPlot(labels="cat", values="val", group_by="grp")
        result = self._run_grouped(trace, df)
        for child in result.group_results:
            gv = int(child.group_value_key)
            assert child.group_value_key == _group_value_key(gv)

    @pytest.mark.parametrize("parent_uid", ["parent", "123-parent"])
    def test_composite_child_uid_is_css_selector_safe(self, parent_uid):
        child_uid = _child_uid_for_group(parent_uid, ("solar", "BE"))

        assert re.fullmatch(r"fv_[A-Za-z0-9_-]+", child_uid)
        assert "solar_BE" in child_uid


class TestDomainKeyViaToTraceSpec:
    """to_trace_spec(domain_source=...) sets group_domain_key without Figure."""

    def test_sets_group_domain_key(self):
        from flexviz.trace.line import LinePlot

        trace = LinePlot(x="ts", y="val", group_by="cat")
        spec = trace.to_trace_spec(domain_source="my_source")
        assert spec.params["group_domain_key"] == "my_source::cat"

    def test_no_domain_source_omits_key(self):
        from flexviz.trace.line import LinePlot

        trace = LinePlot(x="ts", y="val", group_by="cat")
        spec = trace.to_trace_spec()
        assert "group_domain_key" not in spec.params

    def test_no_group_by_omits_key(self):
        from flexviz.trace.line import LinePlot

        trace = LinePlot(x="ts", y="val")
        spec = trace.to_trace_spec(domain_source="src")
        assert "group_domain_key" not in spec.params


class TestAggregationSpecUid:
    """AggregationSpec.uid carries the trace uid."""

    def test_line_agg_spec_has_uid(self):
        from flexviz.trace.line import LinePlot

        trace = LinePlot(x="ts", y="val")
        spec = trace.get_aggregation_spec({}, schema=None, domains={"ts": (0.0, 1.0)})
        assert spec.uid == trace.uid

    def test_hist_agg_spec_has_uid(self):
        from flexviz.trace.hist import Histogram

        trace = Histogram(x="val")
        spec = trace.get_aggregation_spec({}, schema=None, domains={"val": (0.0, 1.0)})
        assert spec.uid == trace.uid

    def test_box_agg_spec_has_uid(self):
        from flexviz.trace.box import BoxPlot

        trace = BoxPlot(y="val")
        spec = trace.get_aggregation_spec({}, schema=None)
        assert spec.uid == trace.uid

    def test_bar_grouped_agg_spec_has_uid(self):
        from flexviz.trace.bar import BarPlot

        trace = BarPlot(labels="cat", values="val", group_by="grp")
        spec = trace.get_aggregation_spec({}, schema=None)
        assert spec.uid == trace.uid


class TestOverlayStyle:
    """overlay_style class/instance attribute is readable on all traces."""

    def test_line_full(self):
        from flexviz.trace.line import LinePlot

        assert LinePlot(x="ts", y="val").overlay_style == "full"

    def test_hist_full(self):
        from flexviz.trace.hist import Histogram

        assert Histogram(x="val").overlay_style == "full"

    def test_box_filtered_only(self):
        from flexviz.trace.box import BoxPlot

        assert BoxPlot(y="val").overlay_style == "filtered_only"

    def test_bar_no_group_full(self):
        from flexviz.trace.bar import BarPlot

        assert BarPlot(labels="cat", values="val").overlay_style == "full"

    def test_bar_with_group_filtered_only(self):
        from flexviz.trace.bar import BarPlot

        assert (
            BarPlot(labels="cat", values="val", group_by="grp").overlay_style
            == "filtered_only"
        )


class TestSelectionSpec:
    """Declarative TraceSelectionSpec emitted per trace (descriptor contract)."""

    def test_line_range_x_only(self):
        from flexviz.trace.line import LinePlot

        sel = LinePlot("ts", "val")._make_selection_spec()
        assert sel.kind == "range" and sel.axis_columns == {"x": "ts"}

    def test_horizontal_histogram_maps_y_anchor(self):
        from flexviz.trace.hist import Histogram

        sel = Histogram(y="v")._make_selection_spec()
        assert sel.kind == "range" and sel.axis_columns == {"y": "v"}

    def test_hist2d_range_both_axes(self):
        from flexviz.trace.hist2d import Histogram2D

        sel = Histogram2D("a", "b")._make_selection_spec()
        assert sel.kind == "range" and sel.axis_columns == {"x": "a", "y": "b"}

    def test_bar_categorical_replace(self):
        from flexviz.trace.bar import BarPlot

        sel = BarPlot(labels=["source", "country"], values="v")._make_selection_spec()
        assert sel.kind == "categorical"
        assert sel.label_columns == ["source", "country"]
        assert sel.multi == "replace"

    def test_pie_categorical(self):
        from flexviz.trace.pie import PiePlot

        sel = PiePlot(labels="c", values="v")._make_selection_spec()
        assert sel.kind == "categorical" and sel.label_columns == ["c"]

    def test_treemap_path(self):
        from flexviz.trace.treemap import TreeMap

        sel = TreeMap(path=["a", "b"], values="v")._make_selection_spec()
        assert sel.kind == "path" and sel.path_columns == ["a", "b"]
        assert sel.multi == "path"

    def test_geo_box_lon_lat(self):
        from flexviz.trace.geo_hist2d import GeoHistogram2D

        sel = GeoHistogram2D(lat="la", lon="lo")._make_selection_spec()
        assert sel.kind == "geo_box"
        assert (sel.lon_column, sel.lat_column) == ("lo", "la")

    def test_geo_line_is_not_a_source(self):
        # A scattermap line exposes points, not GeoJSON features, so it cannot
        # emit a selection — it must not advertise geo_box.
        from flexviz.trace.geo_line import GeoLine

        assert GeoLine(lat="la", lon="lo")._make_selection_spec().kind == "none"

    def test_corr_heatmap_is_not_a_source(self):
        from flexviz.trace.corr_heatmap import CorrHeatmap

        assert CorrHeatmap(columns=["a", "b"])._make_selection_spec().kind == "none"


class TestCubeDescriptorDefaults:
    """Base FlexTrace cube descriptor methods default to None (not cube-capable).

    Since Phase 4 (plan step 8) line and box override the *source* descriptor
    (1-D range sources, like hist); their *target* descriptors still inherit
    the base ``None`` default. Traces without any override (e.g. GeoLine) are
    neither a source nor a target.
    """

    def test_line_is_a_cube_source_but_not_a_target(self):
        from flexviz.trace.line import LinePlot

        trace = LinePlot(x="ts", y="val")
        spec = trace.get_cube_source_spec((0.0, 1.0))
        assert spec is not None and spec.column == "ts"
        assert trace.get_cube_target_spec((0.0, 1.0)) is None

    def test_box_is_a_cube_source_but_not_a_target(self):
        from flexviz.trace.box import BoxPlot

        trace = BoxPlot(y="val")
        spec = trace.get_cube_source_spec(None)
        assert spec is not None and spec.column == "val"
        assert trace.get_cube_target_spec(None) is None

    def test_defaults_accept_schema_keyword(self):
        from flexviz.trace.geo_line import GeoLine

        trace = GeoLine(lat="la", lon="lo")
        schema = pl.Schema({"la": pl.Float64, "lo": pl.Float64})
        assert trace.get_cube_source_spec(None, schema=schema) is None
        assert trace.get_cube_target_spec(None, schema=schema) is None

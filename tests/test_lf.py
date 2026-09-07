"""Unit tests for LFQueryBuilder and AggregationSpec."""

from __future__ import annotations

import sys
from datetime import date, datetime, time, timedelta

import polars as pl
import pytest
from flexviz.LF import AggregationSpec, GroupedAggregationSpec, LFQueryBuilder
from flexviz.trace.line import LinePlot

# ---- AggregationSpec -------------------------------------------------------


class TestAggregationSpec:
    def test_expr_stored(self):
        e = pl.col("x").sum()
        spec = AggregationSpec(e)
        assert spec.expr is e

    def test_neither_expr_nor_plan_raises(self):
        with pytest.raises(ValueError, match="either an expr or a plan"):
            AggregationSpec()


# ---- LFQueryBuilder.aggregate ---------------------------------------------


class TestLFQueryBuilderAggregate:
    def test_expr_only(self, backend_lf: LFQueryBuilder):
        agg = AggregationSpec(pl.col("val").sum().alias("total"))
        result, grouped = backend_lf.aggregate([], [agg])
        assert result.shape[0] == 1
        assert result["total"][0] == sum(range(1_000))
        assert grouped == {}

    def test_expr_with_filter_on_other_column(self, backend_lf: LFQueryBuilder):
        agg = AggregationSpec(
            pl.col("val").filter(pl.col("ts") < 500).sum().alias("total")
        )
        result, _ = backend_lf.aggregate([], [agg])
        assert result["total"][0] == sum(range(500))

    def test_multiple_exprs_one_select(self, backend_lf: LFQueryBuilder):
        expr_agg = AggregationSpec(pl.col("val").sum().alias("total"))
        cnt_agg = AggregationSpec(pl.len().alias("cnt"))
        result, _ = backend_lf.aggregate([], [expr_agg, cnt_agg])
        assert result["total"][0] == sum(range(1_000))
        assert result["cnt"][0] == 1_000

    def test_empty_filter_exprs(self, backend_lf: LFQueryBuilder):
        agg = AggregationSpec(pl.len().alias("cnt"))
        result, _ = backend_lf.aggregate([], [agg])
        assert result["cnt"][0] == 1_000

    def test_filter_reduces_rows(self, backend_lf: LFQueryBuilder):
        agg = AggregationSpec(pl.len().alias("cnt"))
        result, _ = backend_lf.aggregate([pl.col("ts").is_between(100, 199)], [agg])
        assert result["cnt"][0] == 100

    def test_empty_agg_specs(self, backend_lf: LFQueryBuilder):
        result, grouped = backend_lf.aggregate([], [])
        assert result.shape == (0, 0)
        assert grouped == {}

    def test_two_line_traces_different_n_points_single_row_df_agg(self):
        """Multi-trace `select()` requires equal column heights; line uses implode."""
        n = 10_000
        df = pl.DataFrame(
            {
                "ts": list(range(n)),
                "a": [float(i) for i in range(n)],
                "b": [float(i * 2) for i in range(n)],
            }
        )
        lf = LFQueryBuilder(df)
        t_high = LinePlot(x="ts", y="a", n_points=2000)
        t_low = LinePlot(x="ts", y="b", n_points=500)
        schema = lf.schema
        domains = lf.physical_minmax(["ts"])
        result, _ = lf.aggregate(
            [],
            [
                t_high.get_aggregation_spec({}, schema=schema, domains=domains),
                t_low.get_aggregation_spec({}, schema=schema, domains=domains),
            ],
        )
        assert result.height == 1
        assert len(result.columns) == 2
        assert {t_high.uid, t_low.uid} == set(result.columns)

        up_a = t_high._to_update(result).updates
        up_b = t_low._to_update(result).updates
        assert len(up_a["x"]) <= 2000
        assert len(up_a["y"]) <= 2000
        assert len(up_a["x"]) == len(up_a["y"])
        assert len(up_b["x"]) <= 500
        assert len(up_b["y"]) <= 500
        assert len(up_b["x"]) == len(up_b["y"])


# ---- GroupedAggregationSpec ------------------------------------------------


class TestGroupedAggregationSpec:
    def test_simple_groupby(self, grouped_backend_lf: LFQueryBuilder):
        """GroupedAggregationSpec executes group_by().agg().sort() and returns a DataFrame."""
        spec = GroupedAggregationSpec(
            uid="test_bar",
            group_cols=("cat",),
            agg_exprs=(pl.col("val").sum().alias("test_bar"),),
            sort_cols=("cat",),
        )
        _, grouped = grouped_backend_lf.aggregate([], [spec])
        assert "test_bar" in grouped
        df = grouped["test_bar"]
        assert set(df.columns) == {"cat", "test_bar"}
        assert df.height == 2  # "A" and "B" categories

    def test_groupby_with_filter(self, grouped_backend_lf: LFQueryBuilder):
        """Filter expressions reduce rows before group_by aggregation."""
        spec = GroupedAggregationSpec(
            uid="test_bar_filtered",
            group_cols=("cat",),
            agg_exprs=(pl.col("val").sum().alias("test_bar_filtered"),),
            sort_cols=("cat",),
            pre_group_filters=(pl.col("cat") == "A",),
            pre_group_filter_key=("cat", "A"),
        )
        _, grouped = grouped_backend_lf.aggregate([], [spec])
        df = grouped["test_bar_filtered"]
        assert df.height == 1
        assert df["cat"][0] == "A"

    def test_mixed_specs(self, grouped_backend_lf: LFQueryBuilder):
        """AggregationSpec and GroupedAggregationSpec can coexist in one aggregate() call."""
        reg_spec = AggregationSpec(pl.len().alias("cnt"))
        grp_spec = GroupedAggregationSpec(
            uid="grp_uid",
            group_cols=("cat",),
            agg_exprs=(pl.col("val").mean().alias("grp_uid"),),
            sort_cols=("cat",),
        )
        result, grouped = grouped_backend_lf.aggregate([], [reg_spec, grp_spec])
        assert result["cnt"][0] == 500
        assert "grp_uid" in grouped
        assert grouped["grp_uid"].height == 2

    def test_same_batch_reuses_one_grouped_df(self, grouped_backend_lf: LFQueryBuilder):
        spec_a = GroupedAggregationSpec(
            uid="sum_uid",
            group_cols=("cat",),
            agg_exprs=(pl.col("val").sum().alias("sum_uid"),),
            sort_cols=("cat",),
        )
        spec_b = GroupedAggregationSpec(
            uid="mean_uid",
            group_cols=("cat",),
            agg_exprs=(pl.col("val").mean().alias("mean_uid"),),
            sort_cols=("cat",),
        )
        _, grouped = grouped_backend_lf.aggregate([], [spec_a, spec_b])
        assert grouped["sum_uid"] is grouped["mean_uid"]
        assert {"cat", "sum_uid", "mean_uid"} <= set(grouped["sum_uid"].columns)

    def test_same_batch_with_missing_pre_group_filter_key_raises(
        self, grouped_backend_lf: LFQueryBuilder
    ):
        spec_a = GroupedAggregationSpec(
            uid="sum_uid",
            group_cols=("cat",),
            agg_exprs=(pl.col("val").sum().alias("sum_uid"),),
            sort_cols=("cat",),
            pre_group_filters=(pl.col("cat") == "A",),
        )
        spec_b = GroupedAggregationSpec(
            uid="mean_uid",
            group_cols=("cat",),
            agg_exprs=(pl.col("val").mean().alias("mean_uid"),),
            sort_cols=("cat",),
            pre_group_filters=(pl.col("cat") == "B",),
        )

        with pytest.raises(ValueError, match="must provide pre_group_filter_key"):
            grouped_backend_lf.aggregate([], [spec_a, spec_b])

    def test_same_batch_with_mismatched_pre_group_filter_keys_raises(
        self, grouped_backend_lf: LFQueryBuilder
    ):
        spec_a = GroupedAggregationSpec(
            uid="sum_uid",
            group_cols=("cat",),
            agg_exprs=(pl.col("val").sum().alias("sum_uid"),),
            sort_cols=("cat",),
            pre_group_filters=(pl.col("cat") == "A",),
            pre_group_filter_key=("cat", "A"),
        )
        spec_b = GroupedAggregationSpec(
            uid="mean_uid",
            group_cols=("cat",),
            agg_exprs=(pl.col("val").mean().alias("mean_uid"),),
            sort_cols=("cat",),
            pre_group_filters=(pl.col("cat") == "B",),
            pre_group_filter_key=("cat", "B"),
        )

        with pytest.raises(ValueError, match="Extend batch_key"):
            grouped_backend_lf.aggregate([], [spec_a, spec_b])

    def test_plan_spec_result_stored(self, grouped_backend_lf: LFQueryBuilder):
        """A grouped spec with a plan lands in grouped_dfs with the plan's frame."""

        def plan(ldf: pl.LazyFrame) -> pl.DataFrame:
            return (
                ldf.group_by("cat")
                .agg(pl.col("val").max().alias("plan_uid"))
                .sort("cat")
                .collect()
            )

        spec = GroupedAggregationSpec(
            uid="plan_uid",
            group_cols=("cat",),
            sort_cols=("cat",),
            agg_exprs=(),
            plan=plan,
        )
        _, grouped = grouped_backend_lf.aggregate([], [spec])
        df = grouped["plan_uid"]
        assert set(df.columns) == {"cat", "plan_uid"}
        assert df.to_dicts() == [
            {"cat": "A", "plan_uid": 498.0},
            {"cat": "B", "plan_uid": 499.0},
        ]

    def test_plan_spec_sees_pre_group_filters(self, grouped_backend_lf: LFQueryBuilder):
        """The plan gets the cross-filter and the spec's pre-group filters."""
        seen: list[pl.DataFrame] = []

        def plan(ldf: pl.LazyFrame) -> pl.DataFrame:
            frame = ldf.collect()
            seen.append(frame)
            return frame.group_by("cat").agg(pl.len().alias("plan_uid")).sort("cat")

        spec = GroupedAggregationSpec(
            uid="plan_uid",
            group_cols=("cat",),
            sort_cols=("cat",),
            agg_exprs=(),
            pre_group_filters=(pl.col("cat") == "A",),
            plan=plan,
        )
        _, grouped = grouped_backend_lf.aggregate([pl.col("ts") < 100], [spec])
        # 100 rows pass the cross-filter, half of them are category "A".
        assert seen[0]["cat"].unique().to_list() == ["A"]
        assert seen[0].height == 50
        assert grouped["plan_uid"].to_dicts() == [{"cat": "A", "plan_uid": 50}]

    def test_plan_and_expr_specs_coexist(self, grouped_backend_lf: LFQueryBuilder):
        """Expression specs still fuse while a plan spec runs on its own."""

        def plan(ldf: pl.LazyFrame) -> pl.DataFrame:
            return (
                ldf.group_by("cat")
                .agg(pl.col("val").max().alias("plan_uid"))
                .sort("cat")
                .collect()
            )

        plan_spec = GroupedAggregationSpec(
            uid="plan_uid",
            group_cols=("cat",),
            sort_cols=("cat",),
            agg_exprs=(),
            plan=plan,
        )
        sum_spec = GroupedAggregationSpec(
            uid="sum_uid",
            group_cols=("cat",),
            sort_cols=("cat",),
            agg_exprs=(pl.col("val").sum().alias("sum_uid"),),
        )
        mean_spec = GroupedAggregationSpec(
            uid="mean_uid",
            group_cols=("cat",),
            sort_cols=("cat",),
            agg_exprs=(pl.col("val").mean().alias("mean_uid"),),
        )
        _, grouped = grouped_backend_lf.aggregate([], [plan_spec, sum_spec, mean_spec])
        assert grouped["sum_uid"] is grouped["mean_uid"]
        assert set(grouped["sum_uid"].columns) == {"cat", "sum_uid", "mean_uid"}
        assert set(grouped["plan_uid"].columns) == {"cat", "plan_uid"}

    def test_plan_spec_never_fuses(self, grouped_backend_lf: LFQueryBuilder):
        """A plan spec shares the batch key but stays out of the fused query.

        Its pre-group filters carry no ``pre_group_filter_key``, so fusion with
        the expression spec would be rejected.
        """

        def plan(ldf: pl.LazyFrame) -> pl.DataFrame:
            return (
                ldf.group_by("cat")
                .agg(pl.len().alias("plan_uid"))
                .sort("cat")
                .collect()
            )

        plan_spec = GroupedAggregationSpec(
            uid="plan_uid",
            group_cols=("cat",),
            sort_cols=("cat",),
            agg_exprs=(),
            pre_group_filters=(pl.col("cat") == "A",),
            plan=plan,
        )
        expr_spec = GroupedAggregationSpec(
            uid="sum_uid",
            group_cols=("cat",),
            sort_cols=("cat",),
            agg_exprs=(pl.col("val").sum().alias("sum_uid"),),
            pre_group_filters=(pl.col("ts") < 100,),
            pre_group_filter_key=("ts", 100),
        )
        _, grouped = grouped_backend_lf.aggregate([], [plan_spec, expr_spec])
        assert grouped["plan_uid"].to_dicts() == [{"cat": "A", "plan_uid": 250}]
        assert set(grouped["sum_uid"].columns) == {"cat", "sum_uid"}


class TestLFQueryBuilderAssumeSorted:
    def test_assume_sorted_does_not_collect_or_raise(self):
        df = pl.DataFrame({"a": [3, 2, 1], "b": [1, 2, 3]})
        lf = LFQueryBuilder(df)

        # Should not raise, even though `a` is not actually sorted.
        # (Caller responsibility; this only sets the sorted flag.)
        lf.assume_sorted("a")

        # Idempotent
        lf.assume_sorted("a")


# ---- LFQueryBuilder.physical_minmax ---------------------------------------


class TestPhysicalMinMax:
    def test_basic_minmax(self):
        b = LFQueryBuilder(pl.DataFrame({"a": [1.0, 5.0, 3.0]}).lazy())
        assert b.physical_minmax(["a"]) == {"a": (1.0, 5.0)}

    def test_temporal_uses_physical(self):
        b = LFQueryBuilder(
            pl.DataFrame({"t": [date(2020, 1, 1), date(2020, 1, 11)]}).lazy()
        )
        lo, hi = b.physical_minmax(["t"])["t"]
        # Date physical = days since epoch; the span is 10 days.
        assert hi - lo == 10

    def test_memoized_no_second_collect(self):
        """Second call must not re-collect — sabotage the LazyFrame to prove
        the value comes from the memo (the cube cache-hit TTFB guarantee)."""
        b = LFQueryBuilder(pl.DataFrame({"a": [1.0, 2.0, 3.0]}).lazy())
        assert b.physical_minmax(["a"]) == {"a": (1.0, 3.0)}
        b._ldf = None  # any further .collect() would raise
        assert b.physical_minmax(["a"]) == {"a": (1.0, 3.0)}

    def test_partial_memo_only_collects_missing(self):
        b = LFQueryBuilder(pl.DataFrame({"a": [1.0, 3.0], "c": [10.0, 40.0]}).lazy())
        assert b.physical_minmax(["a"]) == {"a": (1.0, 3.0)}
        # "a" is memoized; "c" is new — both returned, "a" not recomputed.
        assert b.physical_minmax(["a", "c"]) == {
            "a": (1.0, 3.0),
            "c": (10.0, 40.0),
        }

    def test_all_null_column_yields_none(self):
        b = LFQueryBuilder(
            pl.DataFrame({"a": pl.Series([None, None], dtype=pl.Float64)}).lazy()
        )
        assert b.physical_minmax(["a"]) == {"a": (None, None)}

    def test_duplicate_columns_deduped(self):
        """The same column in several roles (free axis == target dim) must not
        build duplicate select aliases (Polars DuplicateError)."""
        b = LFQueryBuilder(pl.DataFrame({"a": [1.0, 4.0], "b": [2.0, 8.0]}).lazy())
        out = b.physical_minmax(["a", "a", "b", "a"])
        assert out == {"a": (1.0, 4.0), "b": (2.0, 8.0)}


# ---- LFQueryBuilder.check_line_x -------------------------------------------


class TestCheckLineX:
    """The data check of an ungrouped x-width line on a resident frame."""

    @staticmethod
    def _lf(xs, dtype=pl.Float64) -> LFQueryBuilder:
        return LFQueryBuilder(
            pl.DataFrame(
                {"ts": pl.Series("ts", xs, dtype=dtype), "val": [0.0] * len(xs)}
            )
        )

    def test_sorted_numeric_x_passes_and_is_flagged(self):
        lf = self._lf([1.0, 2.0, 3.0])
        lf.check_line_x("ts")
        assert "ts" in lf.sorted_cols

    def test_null_x_is_rejected(self):
        lf = self._lf([1.0, None, 3.0])
        with pytest.raises(ValueError, match="null values"):
            lf.check_line_x("ts")

    def test_trailing_nan_x_is_rejected(self):
        # A NaN sorts last, so this column passes `is_sorted`.
        lf = self._lf([1.0, 2.0, float("nan")])
        with pytest.raises(ValueError, match="NaN values"):
            lf.check_line_x("ts")

    def test_nan_in_the_middle_is_rejected(self):
        # A NaN sorts last, so it breaks the order first. Either message names
        # a real defect of the column.
        lf = self._lf([1.0, float("nan"), 3.0])
        with pytest.raises(ValueError, match="not sorted ascending"):
            lf.check_line_x("ts")

    def test_failing_column_is_not_memoized(self):
        lf = self._lf([1.0, 2.0, float("nan")])
        for _ in range(2):
            with pytest.raises(ValueError, match="NaN values"):
                lf.check_line_x("ts")
        assert "ts" not in lf.sorted_cols

    def test_unsorted_x_is_rejected(self):
        lf = self._lf([3.0, 1.0, 2.0])
        with pytest.raises(ValueError, match="not sorted ascending"):
            lf.check_line_x("ts")

    def test_memoized_column_is_checked_once(self, monkeypatch):
        lf = self._lf([1.0, 2.0, 3.0])
        lf.check_line_x("ts")
        monkeypatch.setattr(
            pl.LazyFrame, "collect", lambda *a, **k: pytest.fail("collected twice")
        )
        lf.check_line_x("ts")

    def test_an_uncached_scan_keeps_nothing(self, tmp_path):
        # An uncached scan may have changed since the last request, so the pass
        # is not remembered and the next call collects again.
        path = tmp_path / "x.parquet"
        pl.DataFrame({"ts": [1.0, 2.0, 3.0], "val": [0.0] * 3}).write_parquet(path)
        lf = LFQueryBuilder(pl.scan_parquet(str(path)))
        lf.check_line_x("ts")
        assert lf.sorted_cols == frozenset()
        lf.check_line_x("ts")  # collects again, still passes


# ---- physical_minmax: the Parquet footer path ------------------------------


def _no_collect(*args, **kwargs):
    raise AssertionError("physical_minmax must not collect here")


def _footer_sample_frame() -> pl.DataFrame:
    base = datetime(2024, 1, 2, 3, 4, 5, 123456)
    df = pl.DataFrame(
        {
            "i64": [3, 1, 2],
            "u64": pl.Series([1, 2**63 + 7, 5], dtype=pl.UInt64),
            "f32": pl.Series([1.5, 3.25, 2.0], dtype=pl.Float32),
            "f64": [1.5, 3.25, 2.0],
            "date": [date(2024, 1, 1), date(2025, 1, 1), date(2024, 6, 1)],
            "dt_ms": pl.Series(
                [base, base + timedelta(days=1), base], dtype=pl.Datetime("ms")
            ),
            "dt_us": pl.Series(
                [base, base + timedelta(days=1), base], dtype=pl.Datetime("us")
            ),
            # Not whole microseconds: the footer must keep the nanoseconds.
            "dt_ns": pl.Series(
                [1704164645123456789, 1704251045987654321, 1704164645123456790],
                dtype=pl.Int64,
            ).cast(pl.Datetime("ns")),
            "time": [time(1, 2, 3), time(23, 59, 59), time(0, 0, 1)],
            "dur": pl.Series([1, 3, 2], dtype=pl.Int64).cast(pl.Duration("us")),
        }
    )
    return df.with_columns(
        dt_bru=pl.col("dt_us").dt.replace_time_zone("Europe/Brussels"),
        dt_utc=pl.col("dt_us").dt.replace_time_zone("UTC"),
        dt_ns_bru=pl.col("dt_ns").dt.replace_time_zone("Europe/Brussels"),
    )


class TestParquetFooterMinMax:
    def test_footer_matches_collect(self, tmp_path, monkeypatch):
        """Every supported dtype: the footer must give what the collect gives,
        and it must answer without reading the column data."""
        df = _footer_sample_frame()
        cols = list(df.columns)
        path = tmp_path / "s.parquet"
        df.write_parquet(path)

        expected = LFQueryBuilder(df.lazy()).physical_minmax(cols)

        b = LFQueryBuilder(pl.scan_parquet(str(path)))
        assert b._parquet_path == str(path)
        b.schema  # resolve the schema before the collect is sabotaged
        monkeypatch.setattr(pl.LazyFrame, "collect", _no_collect)
        assert b.physical_minmax(cols) == expected

    def test_polars_writes_statistics(self, tmp_path):
        """The footer path rests on Polars writing min/max. Pin that."""
        pq = pytest.importorskip("pyarrow.parquet")
        path = tmp_path / "s.parquet"
        pl.DataFrame({"a": [1.0, 3.0]}).write_parquet(path)
        stats = pq.read_metadata(str(path)).row_group(0).column(0).statistics
        assert stats is not None and stats.has_min_max

    def test_float16_statistics_are_decoded(self, tmp_path):
        """A Float16 statistic arrives as its raw 2-byte half, so folding the
        row groups on the raw bytes would order the negative values wrong."""
        path = tmp_path / "f16.parquet"
        df = pl.DataFrame({"a": pl.Series([-3.5, 2.5, -7.25, 9.0], dtype=pl.Float16)})
        df.write_parquet(path, row_group_size=2)
        expected = LFQueryBuilder(df.lazy()).physical_minmax(["a"])
        assert expected == {"a": (-7.25, 9.0)}
        assert (
            LFQueryBuilder(pl.scan_parquet(str(path))).physical_minmax(["a"])
            == expected
        )

    def test_folds_over_row_groups(self, tmp_path):
        path = tmp_path / "many.parquet"
        pl.DataFrame({"a": [float(i) for i in range(2000)]}).write_parquet(
            path, row_group_size=250
        )
        b = LFQueryBuilder(pl.scan_parquet(str(path)))
        assert b.physical_minmax(["a"]) == {"a": (0.0, 1999.0)}

    def test_mixed_footer_and_collect(self, tmp_path):
        """A String column the footer skips must still be answered."""
        path = tmp_path / "mixed.parquet"
        pl.DataFrame({"a": [1.0, 3.0], "s": ["b", "a"]}).write_parquet(path)
        b = LFQueryBuilder(pl.scan_parquet(str(path)))
        assert b.physical_minmax(["a", "s"]) == {
            "a": (1.0, 3.0),
            "s": ("a", "b"),
        }

    def test_nan_column_falls_back(self, tmp_path):
        """Parquet statistics leave NaN out, so Polars writes no min/max for a
        NaN column and the collect answers. Were a writer to store the finite
        bounds anyway, the finite value is the one the histogram kernels want:
        they skip NaN rows, and a NaN bound would break the binning."""
        path = tmp_path / "nan.parquet"
        pl.DataFrame({"a": [1.0, float("nan"), 3.0]}).write_parquet(path)
        b = LFQueryBuilder(pl.scan_parquet(str(path)))
        assert b.physical_minmax(["a"]) == {"a": (1.0, 3.0)}

    def test_all_nan_column_falls_back(self, tmp_path, monkeypatch):
        path = tmp_path / "allnan.parquet"
        pl.DataFrame({"a": [float("nan"), float("nan")]}).write_parquet(path)
        b = LFQueryBuilder(pl.scan_parquet(str(path)))
        b.schema
        monkeypatch.setattr(pl.LazyFrame, "collect", _no_collect)
        with pytest.raises(AssertionError):
            b.physical_minmax(["a"])

    def test_all_null_column_falls_back(self, tmp_path):
        path = tmp_path / "null.parquet"
        pl.DataFrame({"a": pl.Series([None, None], dtype=pl.Float64)}).write_parquet(
            path
        )
        b = LFQueryBuilder(pl.scan_parquet(str(path)))
        assert b.physical_minmax(["a"]) == {"a": (None, None)}

    def test_memoizes_footer_results(self, tmp_path):
        path = tmp_path / "s.parquet"
        pl.DataFrame({"a": [1.0, 3.0]}).write_parquet(path)
        b = LFQueryBuilder(pl.scan_parquet(str(path)), cache=True)
        assert b.physical_minmax(["a"]) == {"a": (1.0, 3.0)}
        b._ldf = None  # neither the footer nor a collect can run now
        assert b.physical_minmax(["a"]) == {"a": (1.0, 3.0)}

    def test_no_memo_rereads_the_footer(self, tmp_path):
        path = tmp_path / "s.parquet"
        pl.DataFrame({"a": [1.0, 3.0]}).write_parquet(path)
        b = LFQueryBuilder(pl.scan_parquet(str(path)))
        assert b.physical_minmax(["a"]) == {"a": (1.0, 3.0)}
        assert b._minmax_memo == {}
        pl.DataFrame({"a": [7.0, 9.0]}).write_parquet(path)
        assert b.physical_minmax(["a"]) == {"a": (7.0, 9.0)}

    def test_pyarrow_missing_falls_back(self, tmp_path, monkeypatch):
        path = tmp_path / "s.parquet"
        pl.DataFrame({"a": [1.0, 3.0]}).write_parquet(path)
        b = LFQueryBuilder(pl.scan_parquet(str(path)))
        monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)
        assert b.physical_minmax(["a"]) == {"a": (1.0, 3.0)}
        b.schema
        monkeypatch.setattr(pl.LazyFrame, "collect", _no_collect)
        with pytest.raises(AssertionError):
            b.physical_minmax(["a"])

    def test_statistics_disabled_falls_back(self, tmp_path, monkeypatch):
        path = tmp_path / "nostats.parquet"
        pl.DataFrame({"a": [1.0, 3.0]}).write_parquet(path, statistics=False)
        b = LFQueryBuilder(pl.scan_parquet(str(path)))
        assert b.physical_minmax(["a"]) == {"a": (1.0, 3.0)}
        b.schema
        monkeypatch.setattr(pl.LazyFrame, "collect", _no_collect)
        with pytest.raises(AssertionError):
            b.physical_minmax(["a"])

    def test_nested_column_file_falls_back(self, tmp_path):
        """A struct adds leaves, so a field index is no longer a leaf index."""
        path = tmp_path / "nested.parquet"
        pl.DataFrame({"s": [{"x": 1}], "a": [2.0]}).write_parquet(path)
        b = LFQueryBuilder(pl.scan_parquet(str(path)))
        assert b._parquet_path == str(path)
        assert b.physical_minmax(["a"]) == {"a": (2.0, 2.0)}


class TestParquetPathDetection:
    """Only a bare single-file local Parquet scan may use the footer."""

    def test_single_file(self, tmp_path):
        path = tmp_path / "one.parquet"
        pl.DataFrame({"a": [1.0]}).write_parquet(path)
        assert LFQueryBuilder(pl.scan_parquet(str(path)))._parquet_path == str(path)

    def test_glob_and_file_list(self, tmp_path):
        one, two = tmp_path / "one.parquet", tmp_path / "two.parquet"
        pl.DataFrame({"a": [1.0]}).write_parquet(one)
        pl.DataFrame({"a": [2.0]}).write_parquet(two)
        glob = LFQueryBuilder(pl.scan_parquet(str(tmp_path / "*.parquet")))
        assert glob._parquet_path is None
        assert glob.physical_minmax(["a"]) == {"a": (1.0, 2.0)}
        assert (
            LFQueryBuilder(pl.scan_parquet([str(one), str(two)]))._parquet_path is None
        )

    def test_hive_directory(self, tmp_path):
        pl.DataFrame({"a": [1.0, 2.0], "h": ["x", "y"]}).write_parquet(
            tmp_path / "hive", partition_by="h"
        )
        b = LFQueryBuilder(
            pl.scan_parquet(str(tmp_path / "hive"), hive_partitioning=True)
        )
        assert b._parquet_path is None

    def test_node_above_the_scan(self, tmp_path):
        path = tmp_path / "one.parquet"
        pl.DataFrame({"a": [1.0, 5.0]}).write_parquet(path)
        for lf in (
            pl.scan_parquet(str(path)).filter(pl.col("a") > 2),
            pl.scan_parquet(str(path)).with_columns(pl.col("a") * 2),
            pl.scan_parquet(str(path)).select("a"),
        ):
            assert LFQueryBuilder(lf)._parquet_path is None

    def test_sorted_hint(self, tmp_path):
        """A sorted hint leaves the same rows behind the footer statistics."""
        path = tmp_path / "one.parquet"
        pl.DataFrame({"a": [1.0, 2.0]}).write_parquet(path)
        b = LFQueryBuilder(pl.scan_parquet(str(path)).set_sorted("a"))
        assert b._parquet_path == str(path)

    def test_assume_sorted_hint(self, tmp_path):
        path = tmp_path / "one.parquet"
        pl.DataFrame({"a": [1.0, 2.0]}).write_parquet(path)
        b = LFQueryBuilder(pl.scan_parquet(str(path)))
        b.assume_sorted("a")
        assert b._parquet_path == str(path)

    def test_stacked_sorted_hints(self, tmp_path):
        path = tmp_path / "one.parquet"
        pl.DataFrame({"a": [1.0, 2.0], "b": [1.0, 2.0]}).write_parquet(path)
        b = LFQueryBuilder(pl.scan_parquet(str(path)).set_sorted("a").set_sorted("b"))
        assert b._parquet_path == str(path)

    def test_sorted_hint_over_a_slice(self, tmp_path):
        path = tmp_path / "one.parquet"
        pl.DataFrame({"a": [1.0, 2.0, 3.0]}).write_parquet(path)
        b = LFQueryBuilder(pl.scan_parquet(str(path), n_rows=2).set_sorted("a"))
        assert b._parquet_path is None

    def test_sorted_hint_answers_from_the_footer(self, tmp_path, monkeypatch):
        path = tmp_path / "one.parquet"
        pl.DataFrame({"a": [1.0, 3.0]}).write_parquet(path)
        b = LFQueryBuilder(pl.scan_parquet(str(path)).set_sorted("a"))
        b.schema  # resolve the schema before the collect is sabotaged
        monkeypatch.setattr(pl.LazyFrame, "collect", _no_collect)
        assert b.physical_minmax(["a"]) == {"a": (1.0, 3.0)}

    def test_filtered_scan_answers_from_the_rows(self, tmp_path):
        """The footer describes the file, not the rows a filter keeps."""
        path = tmp_path / "one.parquet"
        pl.DataFrame({"a": [1.0, 5.0, 9.0]}).write_parquet(path)
        b = LFQueryBuilder(pl.scan_parquet(str(path)).filter(pl.col("a") > 2))
        assert b.physical_minmax(["a"]) == {"a": (5.0, 9.0)}

    def test_sliced_scan(self, tmp_path):
        """``n_rows`` lives inside the scan node, so the plan head still reads
        like a bare scan while the footer describes rows the query drops."""
        path = tmp_path / "one.parquet"
        pl.DataFrame({"a": [1.0, 5.0, 9.0]}).write_parquet(path)
        b = LFQueryBuilder(pl.scan_parquet(str(path), n_rows=2))
        assert b._parquet_path is None
        assert b.physical_minmax(["a"]) == {"a": (1.0, 5.0)}

    def test_csv_scan(self, tmp_path):
        path = tmp_path / "one.csv"
        pl.DataFrame({"a": [1.0, 3.0]}).write_csv(path)
        b = LFQueryBuilder(pl.scan_csv(str(path)))
        assert b._parquet_path is None
        assert b.physical_minmax(["a"]) == {"a": (1.0, 3.0)}

    def test_resident_frame(self):
        assert LFQueryBuilder(pl.DataFrame({"a": [1.0]}).lazy())._parquet_path is None

    def test_missing_file(self, tmp_path):
        """A path Polars accepts lazily but that is not on this machine."""
        b = LFQueryBuilder(pl.scan_parquet(str(tmp_path / "gone.parquet")))
        assert b._parquet_path is None

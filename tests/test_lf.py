"""Unit tests for LFQueryBuilder and AggregationSpec."""

from __future__ import annotations

from datetime import date

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
        domains = lf.physical_minmax(["ts"], memoize=False)
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
        assert b.physical_minmax(["a"], memoize=True) == {"a": (1.0, 5.0)}

    def test_temporal_uses_physical(self):
        b = LFQueryBuilder(
            pl.DataFrame({"t": [date(2020, 1, 1), date(2020, 1, 11)]}).lazy()
        )
        lo, hi = b.physical_minmax(["t"], memoize=True)["t"]
        # Date physical = days since epoch; the span is 10 days.
        assert hi - lo == 10

    def test_memoized_no_second_collect(self):
        """Second call must not re-collect — sabotage the LazyFrame to prove
        the value comes from the memo (the cube cache-hit TTFB guarantee)."""
        b = LFQueryBuilder(pl.DataFrame({"a": [1.0, 2.0, 3.0]}).lazy())
        assert b.physical_minmax(["a"], memoize=True) == {"a": (1.0, 3.0)}
        b._ldf = None  # any further .collect() would raise
        assert b.physical_minmax(["a"], memoize=True) == {"a": (1.0, 3.0)}

    def test_partial_memo_only_collects_missing(self):
        b = LFQueryBuilder(pl.DataFrame({"a": [1.0, 3.0], "c": [10.0, 40.0]}).lazy())
        assert b.physical_minmax(["a"], memoize=True) == {"a": (1.0, 3.0)}
        # "a" is memoized; "c" is new — both returned, "a" not recomputed.
        assert b.physical_minmax(["a", "c"], memoize=True) == {
            "a": (1.0, 3.0),
            "c": (10.0, 40.0),
        }

    def test_all_null_column_yields_none(self):
        b = LFQueryBuilder(
            pl.DataFrame({"a": pl.Series([None, None], dtype=pl.Float64)}).lazy()
        )
        assert b.physical_minmax(["a"], memoize=True) == {"a": (None, None)}

    def test_duplicate_columns_deduped(self):
        """The same column in several roles (free axis == target dim) must not
        build duplicate select aliases (Polars DuplicateError)."""
        b = LFQueryBuilder(pl.DataFrame({"a": [1.0, 4.0], "b": [2.0, 8.0]}).lazy())
        out = b.physical_minmax(["a", "a", "b", "a"], memoize=True)
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
        lf.check_line_x("ts", memoize=True)
        assert lf.is_sorted("ts")

    def test_null_x_is_rejected(self):
        lf = self._lf([1.0, None, 3.0])
        with pytest.raises(ValueError, match="null values"):
            lf.check_line_x("ts", memoize=True)

    def test_trailing_nan_x_is_rejected(self):
        # A NaN sorts last, so this column passes `is_sorted`.
        lf = self._lf([1.0, 2.0, float("nan")])
        with pytest.raises(ValueError, match="NaN values"):
            lf.check_line_x("ts", memoize=True)

    def test_nan_in_the_middle_is_rejected(self):
        # A NaN sorts last, so it breaks the order first. Either message names
        # a real defect of the column.
        lf = self._lf([1.0, float("nan"), 3.0])
        with pytest.raises(ValueError, match="not sorted ascending"):
            lf.check_line_x("ts", memoize=True)

    def test_failing_column_is_not_memoized(self):
        lf = self._lf([1.0, 2.0, float("nan")])
        for _ in range(2):
            with pytest.raises(ValueError, match="NaN values"):
                lf.check_line_x("ts", memoize=True)
        assert not lf.is_sorted("ts")

    def test_unsorted_x_is_rejected(self):
        lf = self._lf([3.0, 1.0, 2.0])
        with pytest.raises(ValueError, match="not sorted ascending"):
            lf.check_line_x("ts", memoize=True)

    def test_memoized_column_is_checked_once(self, monkeypatch):
        lf = self._lf([1.0, 2.0, 3.0])
        lf.check_line_x("ts", memoize=True)
        monkeypatch.setattr(
            pl.LazyFrame, "collect", lambda *a, **k: pytest.fail("collected twice")
        )
        lf.check_line_x("ts", memoize=True)

    def test_without_memoize_nothing_is_kept(self):
        # An uncached source may have changed since the last request, so the
        # pass is not remembered and the next call collects again.
        lf = self._lf([1.0, 2.0, 3.0])
        lf.check_line_x("ts", memoize=False)
        assert not lf.is_sorted("ts")
        assert lf._sorted_cols == set()
        lf.check_line_x("ts", memoize=False)  # collects again, still passes

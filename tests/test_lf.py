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
        result, _ = lf.aggregate(
            [],
            [
                t_high.get_aggregation_spec({}, schema=schema),
                t_low.get_aggregation_spec({}, schema=schema),
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


class TestLFQueryBuilderAssumeSorted:
    def test_assume_sorted_does_not_collect_or_raise(self):
        df = pl.DataFrame({"a": [3, 2, 1], "b": [1, 2, 3]})
        lf = LFQueryBuilder(df)

        # Should not raise, even though `a` is not actually sorted.
        # (Caller responsibility; this only sets the sorted flag.)
        lf.assume_sorted("a")

        # Idempotent
        lf.assume_sorted("a")


# ---- LFQueryBuilder.physical_minmax (cube domain memo) --------------------


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
        assert b.physical_minmax(["a", "c"]) == {"a": (1.0, 3.0), "c": (10.0, 40.0)}

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


# ---- scan-source collect engine --------------------------------------------


class TestScanCollectEngine:
    """On a scan, aggregate() must run on the streaming engine and resolve the
    global stats in one hoisted pass; a resident frame must keep the default
    engine. Polars falls back to the in-memory engine silently, so the engine
    choice is only observable white-box: capture the collect() kwargs."""

    @staticmethod
    def _capture(monkeypatch):
        calls: list[str] = []
        orig = pl.LazyFrame.collect

        def spy(self, *args, **kwargs):
            calls.append(kwargs.get("engine", "default"))
            return orig(self, *args, **kwargs)

        monkeypatch.setattr(pl.LazyFrame, "collect", spy)
        return calls

    @staticmethod
    def _df() -> pl.DataFrame:
        return pl.DataFrame(
            {"val": [float(i % 7) for i in range(100)], "g": ["a", "b"] * 50}
        )

    def test_scan_select_streams(self, tmp_path, monkeypatch):
        path = tmp_path / "d.parquet"
        self._df().write_parquet(path)
        b = LFQueryBuilder(pl.scan_parquet(path))
        calls = self._capture(monkeypatch)
        out, _ = b.aggregate([], [AggregationSpec(pl.col("val").sum().alias("t"))])
        assert out["t"][0] == self._df()["val"].sum()
        assert calls == ["streaming"]

    def test_resident_select_keeps_default_engine(self, monkeypatch):
        b = LFQueryBuilder(self._df())
        calls = self._capture(monkeypatch)
        b.aggregate([], [AggregationSpec(pl.col("val").sum().alias("t"))])
        assert calls == ["default"]

    def test_scan_grouped_streams(self, tmp_path, monkeypatch):
        path = tmp_path / "d.parquet"
        self._df().write_parquet(path)
        b = LFQueryBuilder(pl.scan_parquet(path))
        calls = self._capture(monkeypatch)
        _, grouped = b.aggregate(
            [],
            [
                GroupedAggregationSpec(
                    uid="u",
                    group_cols=("g",),
                    sort_cols=("g",),
                    agg_exprs=(pl.col("val").sum().alias("u"),),
                )
            ],
        )
        assert grouped["u"]["g"].to_list() == ["a", "b"]
        assert calls == ["streaming"]

    def test_plan_specs_get_hoisted_stats_row(self, tmp_path, monkeypatch):
        """A plan spec with global_stats_cols on a scan receives the resolved
        stats as one row (from a single extra streaming pass); expression
        specs keep reading the broadcast columns — pre-resolving those into
        literal columns costs +2.3 GB in a grouped collect (measured)."""
        path = tmp_path / "d.parquet"
        self._df().write_parquet(path)
        b = LFQueryBuilder(pl.scan_parquet(path))
        calls = self._capture(monkeypatch)
        seen: dict = {}

        def plan(filtered_ldf, stats_row=None):
            seen["stats"] = stats_row
            return pl.DataFrame({"u": [1.0]})

        b.aggregate(
            [],
            [
                AggregationSpec(
                    pl.lit(None), uid="u", global_stats_cols=("val",), plan=plan
                )
            ],
        )
        assert seen["stats"]["__hist_lo_val__"][0] == 0.0
        assert seen["stats"]["__hist_hi_val__"][0] == 6.0
        assert calls == ["streaming"]  # exactly one hoisted stats pass

    def test_plan_specs_without_stats_skip_the_stats_pass(self, tmp_path, monkeypatch):
        """A zoomed plan (literal bounds, no global_stats_cols) must not pay a
        stats scan."""
        path = tmp_path / "d.parquet"
        self._df().write_parquet(path)
        b = LFQueryBuilder(pl.scan_parquet(path))
        calls = self._capture(monkeypatch)
        b.aggregate(
            [],
            [
                AggregationSpec(
                    pl.lit(None),
                    uid="u",
                    plan=lambda ldf, stats_row=None: pl.DataFrame({"u": [1.0]}),
                )
            ],
        )
        assert calls == []

    def test_kernel_grouped_batch_keeps_default_engine(self, tmp_path, monkeypatch):
        """A grouped batch flagged streaming_safe=False (kernel agg) must stay
        on the default engine even on a scan: the streaming fallback costs
        ~1.75x the in-memory engine's peak on kernel aggs (measured)."""
        path = tmp_path / "d.parquet"
        self._df().write_parquet(path)
        b = LFQueryBuilder(pl.scan_parquet(path))
        calls = self._capture(monkeypatch)
        b.aggregate(
            [],
            [
                GroupedAggregationSpec(
                    uid="u",
                    group_cols=("g",),
                    sort_cols=("g",),
                    agg_exprs=(pl.col("val").sum().alias("u"),),
                    streaming_safe=False,
                )
            ],
        )
        assert calls == ["default"]

    def test_physical_minmax_streams_on_scan(self, tmp_path, monkeypatch):
        """The cube domain pre-pass must not load full columns on a scan."""
        path = tmp_path / "d.parquet"
        self._df().write_parquet(path)
        b = LFQueryBuilder(pl.scan_parquet(path))
        calls = self._capture(monkeypatch)
        res = b.physical_minmax(["val"])
        assert res == {"val": (0.0, 6.0)}
        assert calls == ["streaming"]

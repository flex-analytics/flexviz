"""Unit tests for BarPlot trace."""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl
import pytest

from flexviz.LF import LFQueryBuilder, GroupedAggregationSpec
from flexviz.trace.bar import BarPlot
from flexviz.trace.base import TraceResult

# ---- helpers ---------------------------------------------------------------


def _aggregate_bar(
    df: pl.DataFrame,
    labels: str | Sequence[str] = "cat",
    values: str | None = "val",
    agg: str = "sum",
    group_by: str | Sequence[str] | None = None,
    orientation: str = "v",
    color_map: dict | None = None,
) -> TraceResult:
    """Run a full bar aggregation pipeline and return the TraceResult."""
    lf = LFQueryBuilder(df)
    trace = BarPlot(
        labels=labels,
        values=values,
        agg=agg,
        group_by=group_by,
        orientation=orientation,
        color_map=color_map,
    )
    spec = trace.get_aggregation_spec({}, schema=lf.schema)
    _, grouped_dfs = lf.aggregate([], [spec])
    return trace._to_update(grouped_dfs[trace.uid])


@pytest.fixture()
def cat_df() -> pl.DataFrame:
    """50-row DataFrame with cat (A/B) and val columns."""
    n = 50
    return pl.DataFrame(
        {
            "cat": ["A" if i % 2 == 0 else "B" for i in range(n)],
            "val": [float(i) for i in range(n)],
        }
    )


@pytest.fixture()
def two_cat_df() -> pl.DataFrame:
    """DataFrame with two categorical columns and a value column."""
    return pl.DataFrame(
        {
            "label": ["X", "X", "Y", "Y"],
            "region": ["North", "South", "North", "South"],
            "val": [10.0, 20.0, 30.0, 40.0],
        }
    )


# ---- constructor validation ------------------------------------------------


class TestBarPlotInit:
    def test_invalid_agg_raises(self):
        with pytest.raises(ValueError, match="agg must be one of"):
            BarPlot(labels="cat", values="val", agg="invalid")

    def test_count_is_not_valid_agg(self):
        """'count' is implicit (values=None), not a valid agg string."""
        with pytest.raises(ValueError, match="agg must be one of"):
            BarPlot(labels="cat", values="val", agg="count")

    def test_default_trace_type(self):
        trace = BarPlot(labels="cat")
        assert trace.trace_type == "bar"

    def test_update_on_zoom_false(self):
        trace = BarPlot(labels="cat")
        assert trace.recompute_axes == ()
        assert trace.update_on_zoom is False

    def test_count_only_stores_count_agg(self):
        trace = BarPlot(labels="cat")
        assert trace.agg == "count"
        assert trace.values_col is None

    def test_with_values_stores_agg(self):
        trace = BarPlot(labels="cat", values="val", agg="sum")
        assert trace.agg == "sum"
        assert trace.values_col == "val"

    def test_group_by_stored_in_params(self):
        trace = BarPlot(labels="cat", values="val", group_by="region")
        assert trace.group_by_cols == ("region",)

    def test_color_map_stored_in_display(self):
        cm = {"A": "#ff0000", "B": "#0000ff"}
        trace = BarPlot(labels="cat", values="val", color_map=cm)
        assert trace._display["color_map"] == cm

    def test_label_cols_property(self):
        trace = BarPlot(labels="cat", values="val")
        assert trace.label_cols == ("cat",)


# ---- aggregation spec type -------------------------------------------------


class TestBarPlotAggSpec:
    def test_returns_group_by_aggregation_spec(self, cat_df: pl.DataFrame):
        lf = LFQueryBuilder(cat_df)
        trace = BarPlot(labels="cat", values="val")
        spec = trace.get_aggregation_spec({}, schema=lf.schema)
        assert isinstance(spec, GroupedAggregationSpec)

    def test_simple_group_cols(self, cat_df: pl.DataFrame):
        lf = LFQueryBuilder(cat_df)
        trace = BarPlot(labels="cat", values="val")
        spec = trace.get_aggregation_spec({}, schema=lf.schema)
        assert spec.group_cols == ("cat",)
        assert spec.sort_cols == ("cat",)

    def test_grouped_group_cols(self, two_cat_df: pl.DataFrame):
        lf = LFQueryBuilder(two_cat_df)
        trace = BarPlot(labels="label", values="val", group_by="region")
        spec = trace.get_aggregation_spec({}, schema=lf.schema)
        assert spec.group_cols == ("label", "region")
        assert spec.sort_cols == ("region", "label")

    def test_grouped_group_cols_accept_multiple_columns(self):
        df = pl.DataFrame(
            {
                "label": ["X", "X", "Y", "Y"],
                "region": ["North", "South", "North", "South"],
                "site": ["a", "a", "b", "b"],
                "val": [10.0, 20.0, 30.0, 40.0],
            }
        )
        lf = LFQueryBuilder(df)
        trace = BarPlot(labels="label", values="val", group_by=("region", "site"))
        spec = trace.get_aggregation_spec({}, schema=lf.schema)
        assert spec.group_cols == ("label", "region", "site")
        assert spec.sort_cols == ("region", "site", "label")

    def test_multi_label_group_cols(self):
        df = pl.DataFrame(
            {
                "continent": ["Europe", "Europe"],
                "country": ["Germany", "France"],
                "val": [10.0, 20.0],
            }
        )
        lf = LFQueryBuilder(df)
        trace = BarPlot(labels=["continent", "country"], values="val")
        spec = trace.get_aggregation_spec({}, schema=lf.schema)
        assert spec.group_cols == ("continent", "country")
        assert spec.sort_cols == ("continent", "country")


# ---- simple bar (no group_by) ----------------------------------------------


class TestBarPlotSimple:
    def test_vertical_bar_returns_labels_and_values(self, cat_df: pl.DataFrame):
        result = _aggregate_bar(cat_df, labels="cat", values="val", agg="sum")
        assert result.group_results is None
        assert "x" in result.updates
        assert "y" in result.updates

    def test_vertical_bar_two_categories(self, cat_df: pl.DataFrame):
        result = _aggregate_bar(cat_df, labels="cat", values="val", agg="sum")
        assert len(result.updates["x"]) == 2
        assert set(result.updates["x"]) == {"A", "B"}

    def test_sum_agg(self, cat_df: pl.DataFrame):
        result = _aggregate_bar(cat_df, labels="cat", values="val", agg="sum")
        total = sum(result.updates["y"])
        assert abs(total - sum(range(50))) < 1e-6

    def test_mean_agg(self, cat_df: pl.DataFrame):
        result = _aggregate_bar(cat_df, labels="cat", values="val", agg="mean")
        assert all(v > 0 for v in result.updates["y"])

    def test_median_agg(self, cat_df: pl.DataFrame):
        result = _aggregate_bar(cat_df, labels="cat", values="val", agg="median")
        a_idx = result.updates["x"].index("A")
        b_idx = result.updates["x"].index("B")
        assert result.updates["y"][a_idx] == 24.0
        assert result.updates["y"][b_idx] == 25.0

    def test_count_only_no_values(self, cat_df: pl.DataFrame):
        """values=None → count rows per label."""
        result = _aggregate_bar(cat_df, labels="cat", values=None)
        assert all(v == 25 for v in result.updates["y"])

    def test_min_agg(self, cat_df: pl.DataFrame):
        result = _aggregate_bar(cat_df, labels="cat", values="val", agg="min")
        # A is even indices (0,2,4,...) → min = 0
        a_idx = result.updates["x"].index("A")
        assert result.updates["y"][a_idx] == 0.0

    def test_max_agg(self, cat_df: pl.DataFrame):
        result = _aggregate_bar(cat_df, labels="cat", values="val", agg="max")
        # B is odd indices (1,3,5,...,49) → max = 49
        b_idx = result.updates["x"].index("B")
        assert result.updates["y"][b_idx] == 49.0

    def test_n_unique_agg(self, cat_df: pl.DataFrame):
        result = _aggregate_bar(cat_df, labels="cat", values="val", agg="n_unique")
        assert all(v == 25 for v in result.updates["y"])

    def test_horizontal_bar_flips_xy(self, cat_df: pl.DataFrame):
        result = _aggregate_bar(cat_df, labels="cat", values="val", orientation="h")
        assert result.updates.get("orientation") == "h"
        # x should be values (numbers), y should be labels (strings)
        assert all(isinstance(v, (int, float)) for v in result.updates["x"])
        assert all(isinstance(v, str) for v in result.updates["y"])

    def test_count_horizontal_bar(self, cat_df: pl.DataFrame):
        result = _aggregate_bar(cat_df, labels="cat", values=None, orientation="h")
        assert result.updates.get("orientation") == "h"
        assert all(v == 25 for v in result.updates["x"])

    def test_multi_label_bar_returns_composite_labels(self):
        df = pl.DataFrame(
            {
                "continent": ["Europe", "Europe", "Europe", "Asia"],
                "country": ["Germany", "Germany", "France", "Japan"],
                "val": [10.0, 20.0, 5.0, 30.0],
            }
        )
        result = _aggregate_bar(df, labels=["continent", "country"], values="val")
        by_label = dict(zip(result.updates["x"], result.updates["y"]))
        assert by_label == {
            '["Asia","Japan"]': 30.0,
            '["Europe","France"]': 5.0,
            '["Europe","Germany"]': 30.0,
        }

    def test_horizontal_multi_label_bar_uses_composite_y_labels(self):
        df = pl.DataFrame(
            {
                "continent": ["Europe", "Asia"],
                "country": ["Germany", "Japan"],
                "val": [10.0, 30.0],
            }
        )
        result = _aggregate_bar(
            df, labels=["continent", "country"], values="val", orientation="h"
        )
        assert result.updates["x"] == [30.0, 10.0]
        assert result.updates["y"] == ['["Asia","Japan"]', '["Europe","Germany"]']


# ---- grouped bar (group_by) ------------------------------------------------


class TestBarPlotGrouped:
    def test_grouped_returns_group_results(self, two_cat_df: pl.DataFrame):
        result = _aggregate_bar(
            two_cat_df, labels="label", values="val", group_by="region"
        )
        assert result.updates == {}
        assert len(result.group_results) == 2

    def test_group_results_have_child_uid(self, two_cat_df: pl.DataFrame):
        result = _aggregate_bar(
            two_cat_df, labels="label", values="val", group_by="region"
        )
        for cr in result.group_results:
            assert cr.child_uid is not None
            assert cr.group_value_key is not None

    def test_group_results_have_correct_values(self, two_cat_df: pl.DataFrame):
        result = _aggregate_bar(
            two_cat_df, labels="label", values="val", group_by="region"
        )
        by_group = {cr.group_value_key: cr for cr in result.group_results}
        assert "North" in by_group
        assert "South" in by_group
        north = by_group["North"]
        assert set(north.updates["x"]) == {"X", "Y"}
        # North: X=10, Y=30
        x_idx = north.updates["x"].index("X")
        assert north.updates["y"][x_idx] == 10.0

    def test_grouped_horizontal_flips_xy(self, two_cat_df: pl.DataFrame):
        result = _aggregate_bar(
            two_cat_df, labels="label", values="val", group_by="region", orientation="h"
        )
        for cr in result.group_results:
            assert cr.updates.get("orientation") == "h"
            assert all(isinstance(v, (int, float)) for v in cr.updates["x"])
            assert all(isinstance(v, str) for v in cr.updates["y"])

    def test_child_uids_are_deterministic(self, two_cat_df: pl.DataFrame):
        lf = LFQueryBuilder(two_cat_df)
        trace = BarPlot(labels="label", values="val", group_by="region")
        spec = trace.get_aggregation_spec({}, schema=lf.schema)
        _, grouped_dfs = lf.aggregate([], [spec])
        result1 = trace._to_update(grouped_dfs[trace.uid])
        result2 = trace._to_update(grouped_dfs[trace.uid])
        uids1 = {cr.child_uid for cr in result1.group_results}
        uids2 = {cr.child_uid for cr in result2.group_results}
        assert uids1 == uids2

    def test_grouped_by_two_columns_returns_composite_children(self):
        df = pl.DataFrame(
            {
                "label": ["X", "X", "Y", "Y", "X"],
                "region": ["North", "South", "North", "South", "North"],
                "site": ["a", "a", "b", "b", "a"],
                "val": [10.0, 20.0, 30.0, 40.0, 5.0],
            }
        )
        result = _aggregate_bar(
            df, labels="label", values="val", group_by=["region", "site"]
        )
        by_group = {cr.group_value_key: cr for cr in result.group_results}
        assert set(by_group) == {
            '["North","a"]',
            '["North","b"]',
            '["South","a"]',
            '["South","b"]',
        }
        north_a = by_group['["North","a"]']
        assert north_a.updates["x"] == ["X"]
        assert north_a.updates["y"] == [15.0]

    def test_multi_labels_and_multi_group_by(self):
        df = pl.DataFrame(
            {
                "continent": ["Europe", "Europe", "Europe", "Asia"],
                "country": ["Germany", "Germany", "France", "Japan"],
                "source": ["A", "A", "A", "B"],
                "scenario": ["base", "base", "alt", "base"],
                "val": [10.0, 20.0, 5.0, 30.0],
            }
        )
        result = _aggregate_bar(
            df,
            labels=["continent", "country"],
            values="val",
            group_by=["source", "scenario"],
        )
        by_group = {cr.group_value_key: cr for cr in result.group_results}
        assert set(by_group) == {'["A","alt"]', '["A","base"]', '["B","base"]'}
        assert by_group['["A","base"]'].updates["x"] == ['["Europe","Germany"]']
        assert by_group['["A","base"]'].updates["y"] == [30.0]


# ---- from_trace_spec round-trip --------------------------------------------


class TestBarPlotSpec:
    def test_round_trip_simple(self):
        original = BarPlot(
            labels="cat", values="revenue", agg="sum", name="Rev", color="#e74c3c"
        )
        spec = original.to_trace_spec()
        restored = BarPlot.from_trace_spec(spec)
        assert restored.label_cols == original.label_cols
        assert restored.values_col == original.values_col
        assert restored.agg == original.agg
        assert restored._display["name"] == "Rev"
        assert restored._display["color"] == "#e74c3c"

    def test_round_trip_count_only(self):
        original = BarPlot(labels="cat")
        spec = original.to_trace_spec()
        restored = BarPlot.from_trace_spec(spec)
        assert restored.label_cols == ("cat",)
        assert restored.values_col is None
        assert restored.agg == "count"

    def test_round_trip_grouped(self):
        original = BarPlot(
            labels="label",
            values="val",
            agg="mean",
            orientation="h",
            bar_mode="stack",
            group_by="region",
        )
        spec = original.to_trace_spec()
        restored = BarPlot.from_trace_spec(spec)
        assert restored.group_by_cols == ("region",)
        assert restored.orientation == "h"
        assert restored.bar_mode == "stack"

    def test_round_trip_multi_labels(self):
        original = BarPlot(labels=("continent", "country"), values="val")
        spec = original.to_trace_spec()
        assert spec.backend_data["labels"] == ["continent", "country"]
        restored = BarPlot.from_trace_spec(spec)
        assert restored.label_cols == ("continent", "country")

    def test_round_trip_color_map(self):
        cm = {"A": "#ff0000"}
        original = BarPlot(labels="cat", values="val", color_map=cm)
        spec = original.to_trace_spec()
        restored = BarPlot.from_trace_spec(spec)
        assert restored._display.get("color_map") == cm

    def test_uid_preserved(self):
        original = BarPlot(labels="cat", values="val")
        spec = original.to_trace_spec()
        restored = BarPlot.from_trace_spec(spec)
        assert restored.uid == original.uid

    def test_backward_compat_old_spec_with_xy(self):
        """Old specs stored backend_data as {"x": ..., "y": ...}."""
        from flexviz.spec import TraceSpec

        old_spec = TraceSpec(
            uid="test",
            trace_type="bar",
            backend_data={"x": "category", "y": "revenue"},
            display={"name": "old", "bar_mode": "group"},
            params={"agg": "sum", "orientation": "v"},
            axes=("x", "y"),
        )
        restored = BarPlot.from_trace_spec(old_spec)
        assert restored.label_cols == ("category",)
        assert restored.values_col == "revenue"
        assert restored.agg == "sum"

    def test_backward_compat_old_count_spec(self):
        """Old specs with agg='count' → values_col becomes None."""
        from flexviz.spec import TraceSpec

        old_spec = TraceSpec(
            uid="test",
            trace_type="bar",
            backend_data={"x": "category", "y": "ignored_col"},
            display={"name": "old", "bar_mode": "group"},
            params={"agg": "count", "orientation": "v"},
            axes=("x", "y"),
        )
        restored = BarPlot.from_trace_spec(old_spec)
        assert restored.label_cols == ("category",)
        assert restored.values_col is None
        assert restored.agg == "count"


class TestBarColorMap:
    @pytest.fixture()
    def df(self) -> pl.DataFrame:
        return pl.DataFrame({"cat": ["A", "B", "C"], "val": [10.0, 20.0, 30.0]})

    def test_vertical_bar_color_map_populates_marker_color(self, df):
        color_map = {"A": "#ff0000", "B": "#00ff00", "C": "#0000ff"}
        result = _aggregate_bar(df, color_map=color_map, orientation="v")
        assert "marker" in result.updates
        colors = result.updates["marker"]["color"]
        labels = result.updates["x"]  # vertical: labels on x
        assert len(colors) == len(labels)
        for label, color in zip(labels, colors):
            assert color == color_map[label]

    def test_horizontal_bar_color_map_populates_marker_color(self, df):
        color_map = {"A": "#ff0000", "B": "#00ff00", "C": "#0000ff"}
        result = _aggregate_bar(df, color_map=color_map, orientation="h")
        assert "marker" in result.updates
        colors = result.updates["marker"]["color"]
        labels = result.updates["y"]  # horizontal: labels on y
        assert len(colors) == len(labels)
        for label, color in zip(labels, colors):
            assert color == color_map[label]

    def test_no_color_map_no_marker_in_updates(self, df):
        result = _aggregate_bar(df, color_map=None)
        assert "marker" not in result.updates

    def test_empty_color_map_produces_marker_with_none_colors(self, df):
        result = _aggregate_bar(df, color_map={})
        assert "marker" in result.updates
        assert all(c is None for c in result.updates["marker"]["color"])

    def test_label_not_in_color_map_gets_none(self, df):
        color_map = {"A": "#ff0000"}  # B and C are missing
        result = _aggregate_bar(df, color_map=color_map)
        labels = result.updates["x"]
        colors = result.updates["marker"]["color"]
        for label, color in zip(labels, colors):
            if label == "A":
                assert color == "#ff0000"
            else:
                assert color is None

    def test_multi_label_color_map_uses_composite_keys(self):
        df = pl.DataFrame(
            {
                "continent": ["Europe", "Asia"],
                "country": ["Germany", "Japan"],
                "val": [10.0, 30.0],
            }
        )
        color_map = {
            '["Asia","Japan"]': "#ff0000",
            '["Europe","Germany"]': "#0000ff",
        }
        result = _aggregate_bar(
            df, labels=["continent", "country"], values="val", color_map=color_map
        )
        labels = result.updates["x"]
        colors = result.updates["marker"]["color"]
        assert colors == [color_map[label] for label in labels]

    def test_grouped_bar_color_map_not_in_group_results(self):
        """For grouped bars, color_map is resolved by the JS ensureGroupColor,
        not baked into server-side updates per group-child."""
        two_cat_df = pl.DataFrame(
            {
                "label": ["X", "X", "Y", "Y"],
                "region": ["North", "South", "North", "South"],
                "val": [10.0, 20.0, 30.0, 40.0],
            }
        )
        lf = LFQueryBuilder(two_cat_df)
        color_map = {"North": "#ff0000", "South": "#0000ff"}
        trace = BarPlot(
            labels="label", values="val", group_by="region", color_map=color_map
        )
        spec = trace.get_aggregation_spec({}, schema=lf.schema)
        _, grouped_dfs = lf.aggregate([], [spec])
        result = trace._to_update(grouped_dfs[trace.uid])
        assert result.group_results is not None
        for cr in result.group_results:
            assert "marker" not in cr.updates

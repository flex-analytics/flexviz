"""Unit tests for PiePlot trace."""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl
import pytest

from flexviz.LF import LFQueryBuilder
from flexviz.trace.pie import PiePlot
from flexviz.trace.base import TraceResult


def _aggregate_pie(
    df: pl.DataFrame,
    labels: str | Sequence[str] = "cat",
    values: str | None = "val",
    agg: str = "sum",
    color_map: dict | None = None,
) -> TraceResult:
    lf = LFQueryBuilder(df)
    trace = PiePlot(labels=labels, values=values, agg=agg, color_map=color_map)
    spec = trace.get_aggregation_spec({}, schema=lf.schema)
    _, grouped_dfs = lf.aggregate([], [spec])
    return trace._to_update(grouped_dfs[trace.uid])


@pytest.fixture()
def cat_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "cat": ["A", "A", "B", "B", "C"],
            "val": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )


class TestPieConstructor:
    def test_defaults_with_values(self):
        t = PiePlot(labels="cat", values="val")
        assert t.trace_type == "pie"
        assert t.label_cols == ("cat",)
        assert t.values_col == "val"
        assert t.agg == "sum"
        assert t.hole == 0.0
        assert t._axes is None
        assert t.recompute_axes == ()
        assert t.update_on_zoom is False
        assert t.overlay_style == "filtered_only"

    def test_defaults_count_only(self):
        t = PiePlot(labels="cat")
        assert t.label_cols == ("cat",)
        assert t.values_col is None
        assert t.agg == "count"

    def test_donut_hole(self):
        t = PiePlot(labels="cat", values="val", hole=0.4)
        assert t.hole == 0.4

    def test_invalid_agg(self):
        with pytest.raises(ValueError, match="agg must be one of"):
            PiePlot(labels="cat", values="val", agg="invalid")

    def test_count_not_valid_agg_string(self):
        """'count' is implicit (values=None), not a valid agg string."""
        with pytest.raises(ValueError, match="agg must be one of"):
            PiePlot(labels="cat", values="val", agg="count")


class TestPieAggregation:
    def test_sum(self, cat_df):
        result = _aggregate_pie(cat_df, agg="sum")
        assert "labels" in result.updates
        assert "values" in result.updates
        assert set(result.updates["labels"]) == {"A", "B", "C"}
        idx_a = result.updates["labels"].index("A")
        assert result.updates["values"][idx_a] == 30.0

    def test_count_only_no_values_column(self, cat_df):
        """values=None → count rows per label."""
        result = _aggregate_pie(cat_df, values=None)
        idx_a = result.updates["labels"].index("A")
        assert result.updates["values"][idx_a] == 2

    def test_mean(self, cat_df):
        result = _aggregate_pie(cat_df, agg="mean")
        idx_a = result.updates["labels"].index("A")
        assert result.updates["values"][idx_a] == 15.0

    def test_median(self, cat_df):
        result = _aggregate_pie(cat_df, agg="median")
        idx_b = result.updates["labels"].index("B")
        assert result.updates["values"][idx_b] == 35.0

    def test_n_unique(self, cat_df):
        result = _aggregate_pie(cat_df, agg="n_unique")
        idx_a = result.updates["labels"].index("A")
        assert result.updates["values"][idx_a] == 2

    def test_output_shape(self, cat_df):
        result = _aggregate_pie(cat_df)
        assert len(result.updates["labels"]) == len(result.updates["values"])
        assert result.group_results is None

    def test_multi_label_pie_returns_composite_labels(self):
        df = pl.DataFrame(
            {
                "continent": ["Europe", "Europe", "Europe", "Asia"],
                "country": ["Germany", "Germany", "France", "Japan"],
                "val": [10.0, 20.0, 5.0, 30.0],
            }
        )
        result = _aggregate_pie(df, labels=["continent", "country"], values="val")
        by_label = dict(zip(result.updates["labels"], result.updates["values"]))
        assert by_label == {
            '["Asia","Japan"]': 30.0,
            '["Europe","France"]': 5.0,
            '["Europe","Germany"]': 30.0,
        }


class TestPieSpec:
    def test_roundtrip_with_values(self):
        t = PiePlot(labels="cat", values="val", agg="mean", name="My Pie", hole=0.5)
        spec = t.to_trace_spec()
        assert spec.trace_type == "pie"
        assert spec.backend_data == {"labels": ["cat"], "values": "val"}
        assert spec.params["agg"] == "mean"
        assert spec.params["hole"] == 0.5
        t2 = PiePlot.from_trace_spec(spec)
        assert t2.label_cols == ("cat",)
        assert t2.values_col == "val"
        assert t2.agg == "mean"
        assert t2.hole == 0.5

    def test_roundtrip_count_only(self):
        t = PiePlot(labels="cat", name="Count Pie")
        spec = t.to_trace_spec()
        assert spec.backend_data == {"labels": ["cat"]}
        assert spec.params["agg"] == "count"
        t2 = PiePlot.from_trace_spec(spec)
        assert t2.label_cols == ("cat",)
        assert t2.values_col is None
        assert t2.agg == "count"

    def test_roundtrip_multi_labels(self):
        t = PiePlot(labels=("continent", "country"), values="val")
        spec = t.to_trace_spec()
        assert spec.backend_data["labels"] == ["continent", "country"]
        t2 = PiePlot.from_trace_spec(spec)
        assert t2.label_cols == ("continent", "country")

    def test_backward_compat_old_count_spec(self):
        """Old specs had both labels+values with agg='count'; values_col becomes None."""
        from flexviz.spec import TraceSpec

        old_spec = TraceSpec(
            uid="test",
            trace_type="pie",
            backend_data={"labels": "cat", "values": "ignored"},
            display={"name": "old"},
            params={"agg": "count", "hole": 0.0},
            axes=None,
        )
        restored = PiePlot.from_trace_spec(old_spec)
        assert restored.label_cols == ("cat",)
        assert restored.values_col is None
        assert restored.agg == "count"


class TestPieColorMap:
    @pytest.fixture()
    def df(self) -> pl.DataFrame:
        return pl.DataFrame({"cat": ["A", "B", "C"], "val": [10.0, 20.0, 30.0]})

    def test_color_map_populates_marker_colors(self, df):
        color_map = {"A": "#ff0000", "B": "#00ff00", "C": "#0000ff"}
        result = _aggregate_pie(df, color_map=color_map)
        assert "marker" in result.updates
        colors = result.updates["marker"]["colors"]
        labels = result.updates["labels"]
        assert len(colors) == len(labels)
        for label, color in zip(labels, colors):
            assert color == color_map[label]

    def test_no_color_map_no_marker_key(self, df):
        result = _aggregate_pie(df, color_map=None)
        assert "marker" not in result.updates

    def test_empty_color_map_produces_marker_with_none_colors(self, df):
        result = _aggregate_pie(df, color_map={})
        assert "marker" in result.updates
        assert all(c is None for c in result.updates["marker"]["colors"])

    def test_label_not_in_color_map_gets_none(self, df):
        color_map = {"A": "#ff0000"}  # B and C are missing
        result = _aggregate_pie(df, color_map=color_map)
        labels = result.updates["labels"]
        colors = result.updates["marker"]["colors"]
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
        result = _aggregate_pie(
            df, labels=["continent", "country"], values="val", color_map=color_map
        )
        labels = result.updates["labels"]
        colors = result.updates["marker"]["colors"]
        assert colors == [color_map[label] for label in labels]

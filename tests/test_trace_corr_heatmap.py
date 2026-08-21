"""Unit tests for CorrHeatmap trace."""

from __future__ import annotations

import polars as pl
import pytest

from flexviz.LF import LFQueryBuilder
from flexviz.spec import TraceSpec
from flexviz.trace.corr_heatmap import CorrHeatmap
from flexviz.trace.base import TraceResult


def _aggregate_corr(
    df: pl.DataFrame,
    columns: list[str] | None = None,
    method: str = "pearson",
    triangular: bool = False,
    absolute: bool = False,
) -> TraceResult:
    lf = LFQueryBuilder(df)
    trace = CorrHeatmap(
        columns=columns,
        method=method,
        triangular=triangular,
        absolute=absolute,
    )
    spec = trace.get_aggregation_spec({}, schema=lf.schema)
    regular_df, _ = lf.aggregate([], [spec])
    return trace._to_update(regular_df)


@pytest.fixture()
def numeric_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [5.0, 4.0, 3.0, 2.0, 1.0],
            "c": [1.0, 2.0, 3.0, 4.0, 5.0],
            "cat": ["x", "y", "x", "y", "x"],
        }
    )


class TestCorrHeatmapConstructor:
    def test_defaults(self):
        t = CorrHeatmap()
        assert t.trace_type == "corr_heatmap"
        assert t._axes is None
        assert t.recompute_axes == ()
        assert t.update_on_zoom is False
        assert t.overlay_style == "filtered_only"
        assert t.method == "pearson"
        assert t.triangular is False
        assert t.absolute is False
        assert t.color_scale == "rdbu"
        assert t.color_range == (-1.0, 1.0)

    def test_custom_params(self):
        t = CorrHeatmap(
            columns=["a", "b"], method="spearman", triangular=True, absolute=True
        )
        assert t._columns == ["a", "b"]
        assert t.method == "spearman"
        assert t.triangular is True
        assert t.absolute is True
        assert t.color_scale == "viridis"
        assert t.color_range == (0.0, 1.0)

    def test_dynamic_color_range_can_be_requested(self):
        t = CorrHeatmap(columns=["a", "b"], color_range="auto")
        assert t.color_range == "auto"


class TestCorrHeatmapAggregation:
    def test_auto_columns(self, numeric_df):
        result = _aggregate_corr(numeric_df)
        assert set(result.updates["x"]) == {"a", "b", "c"}
        assert result.updates["y"] == list(reversed(result.updates["x"]))
        n = len(result.updates["x"])
        assert len(result.updates["z"]) == n
        assert len(result.updates["z"][0]) == n

    def test_explicit_columns(self, numeric_df):
        result = _aggregate_corr(numeric_df, columns=["a", "b"])
        assert result.updates["x"] == ["a", "b"]
        assert result.updates["y"] == ["b", "a"]
        assert len(result.updates["z"]) == 2

    def test_rows_are_reversed(self, numeric_df):
        result = _aggregate_corr(numeric_df, columns=["a", "b"])
        z = result.updates["z"]
        assert result.updates["x"] == ["a", "b"]
        assert result.updates["y"] == ["b", "a"]
        assert z[0] == pytest.approx([-1.0, 1.0])
        assert z[1] == pytest.approx([1.0, -1.0])

    def test_diagonal_is_one(self, numeric_df):
        result = _aggregate_corr(numeric_df, columns=["a", "b", "c"])
        x = result.updates["x"]
        for y_label, row in zip(result.updates["y"], result.updates["z"]):
            diag_idx = x.index(y_label)
            assert abs(row[diag_idx] - 1.0) < 1e-10

    def test_perfect_negative_correlation(self, numeric_df):
        result = _aggregate_corr(numeric_df, columns=["a", "b"])
        rows = {
            label: row for label, row in zip(result.updates["y"], result.updates["z"])
        }
        assert abs(rows["a"][1] - (-1.0)) < 1e-10
        assert abs(rows["b"][0] - (-1.0)) < 1e-10

    def test_perfect_positive_correlation(self, numeric_df):
        result = _aggregate_corr(numeric_df, columns=["a", "c"])
        rows = {
            label: row for label, row in zip(result.updates["y"], result.updates["z"])
        }
        assert abs(rows["a"][1] - 1.0) < 1e-10
        assert abs(rows["c"][0] - 1.0) < 1e-10

    def test_absolute(self, numeric_df):
        result = _aggregate_corr(numeric_df, columns=["a", "b"], absolute=True)
        rows = {
            label: row for label, row in zip(result.updates["y"], result.updates["z"])
        }
        assert abs(rows["a"][1] - 1.0) < 1e-10
        assert abs(rows["b"][0] - 1.0) < 1e-10

    def test_triangular_masks_upper_half(self, numeric_df):
        result = _aggregate_corr(numeric_df, columns=["a", "b", "c"], triangular=True)
        rows = {
            label: row for label, row in zip(result.updates["y"], result.updates["z"])
        }
        assert rows["a"][0] is None
        assert rows["a"][1] is None
        assert rows["a"][2] is None
        assert rows["b"][0] == pytest.approx(-1.0)
        assert rows["b"][1] is None
        assert rows["b"][2] is None
        assert rows["c"][0] == pytest.approx(1.0)
        assert rows["c"][1] == pytest.approx(-1.0)
        assert rows["c"][2] is None

    def test_too_few_columns_raises(self):
        df = pl.DataFrame({"a": [1.0, 2.0]})
        with pytest.raises(ValueError, match="at least 2 numeric columns"):
            _aggregate_corr(df, columns=["a"])


class TestCorrHeatmapSpec:
    def test_roundtrip(self):
        t = CorrHeatmap(
            columns=["a", "b"],
            method="spearman",
            triangular=True,
            absolute=True,
            name="Corr",
            color_scale="cividis",
            color_range="auto",
        )
        spec = t.to_trace_spec()
        assert spec.trace_type == "corr_heatmap"
        assert spec.params["method"] == "spearman"
        assert spec.params["triangular"] is True
        assert spec.params["absolute"] is True
        assert spec.params["columns"] == ["a", "b"]
        assert spec.display["color_scale"] == "cividis"
        assert spec.display["color_range"] == "auto"

        t2 = CorrHeatmap.from_trace_spec(spec)
        assert t2._columns == ["a", "b"]
        assert t2.method == "spearman"
        assert t2.triangular is True
        assert t2.absolute is True
        assert t2.color_scale == "cividis"
        assert t2.color_range == "auto"

    def test_legacy_spec_without_display_gets_semantic_defaults(self):
        spec = TraceSpec(
            uid="corr",
            trace_type="corr_heatmap",
            backend_data={},
            params={
                "method": "pearson",
                "triangular": False,
                "absolute": True,
                "columns": ["a", "b"],
            },
            display={"name": "Corr"},
            recompute_axes=(),
        )
        trace = CorrHeatmap.from_trace_spec(spec)
        assert trace.color_scale == "viridis"
        assert trace.color_range == (0.0, 1.0)

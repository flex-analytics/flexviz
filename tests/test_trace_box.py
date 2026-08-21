"""Unit tests for BoxPlot trace."""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl
import pytest


from flexviz.LF import LFQueryBuilder
from flexviz.spec import TraceSpec
from flexviz.trace.box import BoxPlot

# ---- helpers ---------------------------------------------------------------


def _aggregate_box(
    df: pl.DataFrame,
    x: str | None = None,
    y: str | None = None,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
) -> dict:
    """Run a full box aggregation pipeline and return the update dict."""
    lf = LFQueryBuilder(df)
    trace = BoxPlot(x=x, y=y)
    vp_key = "x" if x is not None else "y"
    r = x_range if x is not None else y_range
    update_range = {vp_key: r} if r is not None else {}
    agg_spec = trace.get_aggregation_spec(update_range, schema=lf.schema)
    df_agg, _ = lf.aggregate([], [agg_spec])
    return trace._to_update(df_agg).updates


def _aggregate_grouped_box(
    df: pl.DataFrame,
    y: str = "val",
    group_by: str | Sequence[str] | None = "region",
) -> list:
    lf = LFQueryBuilder(df)
    trace = BoxPlot(y=y, group_by=group_by)
    agg_spec = trace.get_aggregation_spec({}, schema=lf.schema)
    _, grouped_dfs = lf.aggregate([], [agg_spec])
    return trace._to_grouped_update(grouped_dfs[trace.uid]).group_results or []


# ---- stats correctness -----------------------------------------------------


class TestBoxPlotStats:
    def test_quartiles_match_polars(self):
        vals = list(range(1, 11))
        df = pl.DataFrame({"val": vals})
        # BoxPlot uses quantile() which defaults to nearest-rank interpolation.
        # Use the same interpolation here to get a consistent comparison.
        expected = df.select(
            [
                pl.col("val").min().alias("min"),
                pl.col("val").quantile(0.25, interpolation="nearest").alias("q1"),
                pl.col("val").quantile(0.50, interpolation="nearest").alias("median"),
                pl.col("val").quantile(0.75, interpolation="nearest").alias("q3"),
                pl.col("val").max().alias("max"),
            ]
        ).to_dicts()[0]

        update = _aggregate_box(df, y="val")

        assert update["orientation"] == "v"
        assert "x0" in update
        assert update["median"][0] == pytest.approx(expected["median"])
        assert update["q1"][0] == pytest.approx(expected["q1"])
        assert update["q3"][0] == pytest.approx(expected["q3"])
        assert update["lowerfence"][0] == pytest.approx(
            max(
                expected["q1"] - 1.5 * (expected["q3"] - expected["q1"]),
                expected["min"],
            )
        )
        assert update["upperfence"][0] == pytest.approx(
            min(
                expected["q3"] + 1.5 * (expected["q3"] - expected["q1"]),
                expected["max"],
            )
        )

    def test_vertical_trace_uses_x0(self):
        df = pl.DataFrame({"val": [1.0, 2.0, 3.0]})
        update = _aggregate_box(df, y="val")
        assert update["x0"] == "val"


class TestBoxPlotOrientation:
    def test_horizontal(self, small_df: pl.DataFrame):
        update = _aggregate_box(small_df, x="val")
        assert update["orientation"] == "h"
        assert "y0" in update
        assert "x0" not in update
        assert len(update["median"]) == 1

    def test_vertical(self, small_df: pl.DataFrame):
        update = _aggregate_box(small_df, y="val")
        assert update["orientation"] == "v"
        assert "x0" in update
        assert "y0" not in update


# ---- viewport filter -------------------------------------------------------


class TestBoxPlotViewport:
    def test_viewport_does_not_change_median(self, small_df: pl.DataFrame):
        full = _aggregate_box(small_df, y="val")
        zoomed = _aggregate_box(small_df, y="val", y_range=(200.0, 400.0))

        assert full["median"][0] == zoomed["median"][0]


# ---- invalid construction --------------------------------------------------


class TestBoxPlotValidation:
    def test_both_x_y_raises(self):
        with pytest.raises(ValueError, match="either x or y"):
            BoxPlot(x="a", y="b")

    def test_neither_x_y_raises(self):
        with pytest.raises(ValueError, match="either x or y"):
            BoxPlot()


# ---- from_trace_spec round-trip --------------------------------------------


class TestBoxPlotFromTraceSpec:
    def test_roundtrip(self):
        original = BoxPlot(y="val", name="B", color="#00f")
        spec = original.to_trace_spec()

        assert isinstance(spec, TraceSpec)
        assert spec.trace_type == "box"

        restored = BoxPlot.from_trace_spec(spec)
        assert restored.uid == original.uid
        assert restored.data_col == "val"
        assert restored.prop_key == "y"
        assert restored._display.get("name") == "B"
        assert restored._display.get("color") == "#00f"


class TestBoxPlotGrouped:
    def test_grouped_returns_child_results(self):
        df = pl.DataFrame(
            {
                "val": [1.0, 2.0, 10.0, 12.0],
                "region": ["N", "N", "S", "S"],
            }
        )
        results = _aggregate_grouped_box(df)
        assert len(results) == 2
        assert {cr.group_value_key for cr in results} == {"N", "S"}

    def test_grouped_children_use_group_label(self):
        df = pl.DataFrame(
            {
                "val": [1.0, 2.0, 10.0, 12.0],
                "region": ["N", "N", "S", "S"],
            }
        )
        results = _aggregate_grouped_box(df)
        by_group = {cr.group_value_key: cr for cr in results}
        assert by_group["N"].updates["x0"] == "N"
        assert by_group["S"].updates["x0"] == "S"

    def test_grouped_by_two_columns_uses_composite_group_label(self):
        df = pl.DataFrame(
            {
                "val": [1.0, 2.0, 10.0, 12.0, 20.0, 22.0],
                "region": ["N", "N", "S", "S", "N", "N"],
                "site": ["a", "a", "a", "a", "b", "b"],
            }
        )
        results = _aggregate_grouped_box(df, group_by=["region", "site"])
        by_group = {cr.group_value_key: cr for cr in results}
        assert set(by_group) == {'["N","a"]', '["N","b"]', '["S","a"]'}
        assert by_group['["N","a"]'].updates["x0"] == '["N","a"]'


# ---- cube source descriptor (Phase 4, plan step 8) ---------------------------


class TestBoxCubeSource:
    """BoxPlot cube source descriptor — identical shape to the histogram's.

    A brush on a box plot defines a 1-D free axis on its data (prop) column;
    the orthogonal axis is decorative. Temporal dtypes carry their physical
    ``unit`` (contract G); ``Datetime("ns")``/``Time`` and non-numeric dtypes
    gate to ``None``.
    """

    def test_unzoomed_domain_none(self):
        from flexviz.cube import FreeAxisSpec

        trace = BoxPlot(y="val")
        spec = trace.get_cube_source_spec(None)
        assert isinstance(spec, FreeAxisSpec)
        assert spec.column == "val"
        assert spec.kind == "continuous"
        assert spec.p == 2048
        assert spec.domain is None

    def test_zoomed_domain_is_axis_range_verbatim(self):
        trace = BoxPlot(y="val")
        spec = trace.get_cube_source_spec((10.0, 50.0))
        assert spec is not None
        assert tuple(spec.domain) == (10.0, 50.0)

    def test_x_oriented_uses_data_col(self):
        trace = BoxPlot(x="val")
        spec = trace.get_cube_source_spec((0.0, 1.0))
        assert spec is not None
        assert spec.column == "val"

    @pytest.mark.parametrize(
        "dtype,unit",
        [(pl.Datetime("us"), "us"), (pl.Datetime("ms"), "ms"), (pl.Date, "day")],
    )
    def test_temporal_kind_and_unit_from_schema(self, dtype, unit):
        schema = pl.Schema({"ts": dtype})
        spec = BoxPlot(y="ts").get_cube_source_spec(None, schema=schema)
        assert spec is not None
        assert spec.kind == "temporal"
        assert spec.unit == unit

    @pytest.mark.parametrize("dtype", [pl.Datetime("ns"), pl.Time])
    def test_ns_and_time_gate_to_none(self, dtype):
        # Datetime("ns") has no exact string round-trip (µs precision); Time
        # has no date-string representation at all (contract G).
        schema = pl.Schema({"ts": dtype})
        assert BoxPlot(y="ts").get_cube_source_spec(None, schema=schema) is None

    def test_non_numeric_gate(self):
        schema = pl.Schema({"val": pl.String})
        assert BoxPlot(y="val").get_cube_source_spec(None, schema=schema) is None

    def test_defaults_to_continuous_without_schema(self):
        spec = BoxPlot(y="val").get_cube_source_spec(None, schema=None)
        assert spec is not None
        assert spec.kind == "continuous"

    def test_grouped_box_is_still_a_source(self):
        # The brush is on the shared data axis, independent of grouping.
        schema = pl.Schema({"val": pl.Float64, "region": pl.String})
        spec = BoxPlot(y="val", group_by="region").get_cube_source_spec(
            (0.0, 1.0), schema=schema
        )
        assert spec is not None
        assert spec.column == "val"

    def test_target_spec_still_none(self):
        assert BoxPlot(y="val").get_cube_target_spec(None) is None

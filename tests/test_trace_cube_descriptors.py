"""Cube descriptor tests for bar, pie, and treemap traces (cube Phase 2).

Histogram descriptor tests (Phase 1, plus the grouped-target extension) live
in ``tests/test_trace_hist.py``. This file covers the categorical
source/target descriptors and the bar ≡ pie descriptor-sharing property.
"""

from __future__ import annotations

import polars as pl

from flexviz.cube import (
    CubeSpec,
    CubeTargetSpec,
    FreeAxisSpec,
    cube_content_key,
)
from flexviz.trace.bar import BarPlot
from flexviz.trace.pie import PiePlot
from flexviz.trace.treemap import TreeMap

#: Baseline schema: string label/group columns, numeric value column.
_SCHEMA = pl.Schema(
    {
        "cat": pl.String,
        "sub": pl.String,
        "grp": pl.String,
        "val": pl.Float64,
        "ts": pl.Datetime("us"),
        "num": pl.Int64,
        "fnum": pl.Float64,
    }
)


# ---- bar source (categorical free axis) -------------------------------------


class TestBarCubeSource:
    def test_source_spec_single_label(self):
        trace = BarPlot(labels="cat")
        spec = trace.get_cube_source_spec(None, schema=_SCHEMA)
        assert isinstance(spec, FreeAxisSpec)
        assert spec.kind == "categorical"
        assert spec.column == "cat"
        assert spec.columns == ("cat",)
        assert spec.p == 0
        assert spec.domain is None

    def test_source_spec_composite_labels(self):
        trace = BarPlot(labels=["cat", "sub"])
        spec = trace.get_cube_source_spec(None, schema=_SCHEMA)
        assert spec is not None
        assert spec.column == "cat"
        assert spec.columns == ("cat", "sub")

    def test_axis_range_is_ignored(self):
        # Categorical selection geometry is viewport-independent: the covered
        # label set never depends on the visible portion of the label axis.
        trace = BarPlot(labels="cat")
        assert trace.get_cube_source_spec(
            (0.0, 1.0), schema=_SCHEMA
        ) == trace.get_cube_source_spec(None, schema=_SCHEMA)

    def test_categorical_and_enum_label_dtypes_ok(self):
        schema = pl.Schema(
            {"cat": pl.Categorical(), "sub": pl.Enum(["a", "b"]), "val": pl.Float64}
        )
        trace = BarPlot(labels=["cat", "sub"])
        assert trace.get_cube_source_spec(None, schema=schema) is not None

    def test_integer_label_col_is_a_source(self):
        # An integer label column is a valid cube source, mirroring the target
        # gate: numeric free categories stay typed in the cube header and the
        # committed ``is_in`` values cast back to the integer column exactly
        # (e.g. the demo's hour-of-day bar driving a live brush).
        trace = BarPlot(labels="num")
        spec = trace.get_cube_source_spec(None, schema=_SCHEMA)
        assert isinstance(spec, FreeAxisSpec)
        assert spec.kind == "categorical"
        assert spec.column == "num"
        assert spec.columns == ("num",)

    def test_float_label_col_is_a_source(self):
        trace = BarPlot(labels="fnum")
        spec = trace.get_cube_source_spec(None, schema=_SCHEMA)
        assert isinstance(spec, FreeAxisSpec)
        assert spec.kind == "categorical"
        assert spec.column == "fnum"
        assert spec.columns == ("fnum",)

    def test_missing_schema_not_a_source(self):
        trace = BarPlot(labels="cat")
        assert trace.get_cube_source_spec(None, schema=None) is None

    def test_label_col_absent_from_schema_not_a_source(self):
        trace = BarPlot(labels="missing")
        assert trace.get_cube_source_spec(None, schema=_SCHEMA) is None


# ---- bar target (categorical dims + measure) ---------------------------------


class TestBarCubeTarget:
    def test_target_dims_ungrouped(self):
        trace = BarPlot(labels=["cat", "sub"], values="val", agg="sum")
        spec = trace.get_cube_target_spec(None, schema=_SCHEMA)
        assert isinstance(spec, CubeTargetSpec)
        assert [(d.column, d.kind) for d in spec.target_dims] == [
            ("cat", "categorical"),
            ("sub", "categorical"),
        ]
        assert spec.measure.agg == "sum"
        assert spec.measure.value_col == "val"

    def test_target_dims_grouped_order_labels_then_groups(self):
        trace = BarPlot(labels="cat", values="val", agg="mean", group_by=["grp", "sub"])
        spec = trace.get_cube_target_spec(None, schema=_SCHEMA)
        assert spec is not None
        assert [d.column for d in spec.target_dims] == ["cat", "grp", "sub"]
        assert all(d.kind == "categorical" for d in spec.target_dims)
        assert spec.measure.agg == "mean"

    def test_count_measure_when_values_omitted(self):
        trace = BarPlot(labels="cat")
        spec = trace.get_cube_target_spec(None, schema=_SCHEMA)
        assert spec is not None
        assert spec.measure.agg == "count"
        assert spec.measure.value_col is None

    def test_axis_range_is_ignored(self):
        # Bar aggregation never depends on the viewport (no zoom re-binning).
        trace = BarPlot(labels="cat", values="val", agg="sum")
        assert trace.get_cube_target_spec(
            (0.0, 1.0), schema=_SCHEMA
        ) == trace.get_cube_target_spec(None, schema=_SCHEMA)

    def test_median_agg_not_a_target(self):
        trace = BarPlot(labels="cat", values="val", agg="median")
        assert trace.get_cube_target_spec(None, schema=_SCHEMA) is None

    def test_n_unique_agg_not_a_target(self):
        trace = BarPlot(labels="cat", values="val", agg="n_unique")
        assert trace.get_cube_target_spec(None, schema=_SCHEMA) is None

    def test_temporal_value_col_not_a_target(self):
        trace = BarPlot(labels="cat", values="ts", agg="max")
        assert trace.get_cube_target_spec(None, schema=_SCHEMA) is None

    def test_string_value_col_not_a_target(self):
        trace = BarPlot(labels="cat", values="sub", agg="min")
        assert trace.get_cube_target_spec(None, schema=_SCHEMA) is None

    def test_integer_label_col_is_a_target(self):
        # An integer label column is a valid cube target: the codec preserves
        # typed, numerically-ordered labels.
        trace = BarPlot(labels="num", values="val", agg="sum")
        spec = trace.get_cube_target_spec(None, schema=_SCHEMA)
        assert isinstance(spec, CubeTargetSpec)
        assert [(d.column, d.kind) for d in spec.target_dims] == [
            ("num", "categorical")
        ]

    def test_integer_label_with_string_group_is_a_target(self):
        # The demo's hour_of_day / month bars: integer labels, string group_by.
        trace = BarPlot(labels="num", values="val", agg="sum", group_by="grp")
        spec = trace.get_cube_target_spec(None, schema=_SCHEMA)
        assert isinstance(spec, CubeTargetSpec)
        assert [d.column for d in spec.target_dims] == ["num", "grp"]

    def test_integer_group_col_not_a_target(self):
        # group_by stays string-only: a grouped child's identity is
        # ``_group_value_key``-stringified server-side (``5 -> "5"``) while the
        # client cube path keys children by the raw category value, so an
        # integer group would never reconcile — demote instead.
        trace = BarPlot(labels="cat", values="val", agg="sum", group_by="num")
        assert trace.get_cube_target_spec(None, schema=_SCHEMA) is None

    def test_float_label_col_is_a_target(self):
        trace = BarPlot(labels="fnum", values="val", agg="sum")
        spec = trace.get_cube_target_spec(None, schema=_SCHEMA)
        assert isinstance(spec, CubeTargetSpec)
        assert [(d.column, d.kind) for d in spec.target_dims] == [
            ("fnum", "categorical")
        ]

    def test_float_group_col_not_a_target(self):
        trace = BarPlot(labels="cat", values="val", agg="sum", group_by="fnum")
        assert trace.get_cube_target_spec(None, schema=_SCHEMA) is None

    def test_reserved_label_name_not_a_target(self):
        # A categorical dim column literally named "sum" would collide with the
        # measure partial columns in the cube frame (contract A).
        schema = pl.Schema({"sum": pl.String, "val": pl.Float64})
        trace = BarPlot(labels="sum", values="val", agg="sum")
        assert trace.get_cube_target_spec(None, schema=schema) is None

    def test_reserved_group_col_not_a_target(self):
        schema = pl.Schema({"cat": pl.String, "free_bin": pl.String})
        trace = BarPlot(labels="cat", group_by="free_bin")
        assert trace.get_cube_target_spec(None, schema=schema) is None

    def test_missing_schema_not_a_target(self):
        trace = BarPlot(labels="cat", values="val", agg="sum")
        assert trace.get_cube_target_spec(None, schema=None) is None


# ---- pie ---------------------------------------------------------------------


class TestPieCubeDescriptors:
    def test_source_spec_single_label(self):
        trace = PiePlot(labels="cat")
        spec = trace.get_cube_source_spec(None, schema=_SCHEMA)
        assert isinstance(spec, FreeAxisSpec)
        assert spec.kind == "categorical"
        assert spec.column == "cat"
        assert spec.columns == ("cat",)
        assert spec.p == 0
        assert spec.domain is None

    def test_integer_label_col_is_a_source(self):
        # Integer pie labels are valid sources too (mirrors the pie target gate
        # and the bar source gate).
        trace = PiePlot(labels="num")
        spec = trace.get_cube_source_spec(None, schema=_SCHEMA)
        assert isinstance(spec, FreeAxisSpec)
        assert spec.kind == "categorical"
        assert spec.column == "num"

    def test_float_label_col_is_a_source(self):
        trace = PiePlot(labels="fnum")
        spec = trace.get_cube_source_spec(None, schema=_SCHEMA)
        assert isinstance(spec, FreeAxisSpec)
        assert spec.kind == "categorical"
        assert spec.column == "fnum"

    def test_missing_schema_not_a_source(self):
        trace = PiePlot(labels="cat")
        assert trace.get_cube_source_spec(None, schema=None) is None

    def test_target_dims_are_label_cols(self):
        trace = PiePlot(labels=["cat", "sub"], values="val", agg="sum")
        spec = trace.get_cube_target_spec(None, schema=_SCHEMA)
        assert isinstance(spec, CubeTargetSpec)
        assert [(d.column, d.kind) for d in spec.target_dims] == [
            ("cat", "categorical"),
            ("sub", "categorical"),
        ]
        assert spec.measure.agg == "sum"
        assert spec.measure.value_col == "val"

    def test_count_measure_when_values_omitted(self):
        trace = PiePlot(labels="cat")
        spec = trace.get_cube_target_spec(None, schema=_SCHEMA)
        assert spec is not None
        assert spec.measure.agg == "count"
        assert spec.measure.value_col is None

    def test_median_agg_not_a_target(self):
        trace = PiePlot(labels="cat", values="val", agg="median")
        assert trace.get_cube_target_spec(None, schema=_SCHEMA) is None

    def test_string_value_col_not_a_target(self):
        trace = PiePlot(labels="cat", values="sub", agg="sum")
        assert trace.get_cube_target_spec(None, schema=_SCHEMA) is None

    def test_reserved_label_name_not_a_target(self):
        schema = pl.Schema({"count": pl.String})
        trace = PiePlot(labels="count")
        assert trace.get_cube_target_spec(None, schema=schema) is None

    def test_missing_schema_not_a_target(self):
        trace = PiePlot(labels="cat", values="val", agg="sum")
        assert trace.get_cube_target_spec(None, schema=None) is None


# ---- bar ≡ pie descriptor sharing ---------------------------------------------


class TestBarPieSharing:
    def test_same_labels_values_agg_share_one_cube(self):
        bar = BarPlot(labels=["cat", "sub"], values="val", agg="mean")
        pie = PiePlot(labels=["cat", "sub"], values="val", agg="mean")
        bar_target = bar.get_cube_target_spec(None, schema=_SCHEMA)
        pie_target = pie.get_cube_target_spec(None, schema=_SCHEMA)
        assert bar_target is not None and pie_target is not None
        assert bar_target.target_dims == pie_target.target_dims
        assert bar_target.measure == pie_target.measure

        free = bar.get_cube_source_spec(None, schema=_SCHEMA)
        assert free is not None
        specs = [
            CubeSpec(
                source_name="src",
                free=free,
                target_dims=t.target_dims,
                measure=t.measure,
            )
            for t in (bar_target, pie_target)
        ]
        assert specs[0] == specs[1]
        assert cube_content_key(specs[0]) == cube_content_key(specs[1])

    def test_count_variant_shares_too(self):
        bar = BarPlot(labels="cat")
        pie = PiePlot(labels="cat")
        bar_target = bar.get_cube_target_spec(None, schema=_SCHEMA)
        pie_target = pie.get_cube_target_spec(None, schema=_SCHEMA)
        assert bar_target == pie_target


# ---- treemap -------------------------------------------------------------------


class TestTreeMapCubeDescriptors:
    def test_source_spec_from_path(self):
        trace = TreeMap(path=["cat", "sub"], values="val")
        spec = trace.get_cube_source_spec(None, schema=_SCHEMA)
        assert isinstance(spec, FreeAxisSpec)
        assert spec.kind == "categorical"
        assert spec.columns == ("cat", "sub")
        assert spec.column == "cat"
        assert spec.p == 0
        assert spec.domain is None

    def test_axis_range_is_ignored(self):
        trace = TreeMap(path=["cat"])
        assert trace.get_cube_source_spec(
            (0.0, 1.0), schema=_SCHEMA
        ) == trace.get_cube_source_spec(None, schema=_SCHEMA)

    def test_numeric_path_level_not_a_source(self):
        trace = TreeMap(path=["cat", "num"])
        assert trace.get_cube_source_spec(None, schema=_SCHEMA) is None

    def test_missing_schema_not_a_source(self):
        trace = TreeMap(path=["cat"])
        assert trace.get_cube_source_spec(None, schema=None) is None

    def test_treemap_target_dims_are_leaf_path(self):
        # Step 15: a treemap target is a categorical cube on its leaf path; the
        # client finalizes leaf cells then sums them up every level. See
        # tests/test_cube.py::TestTreeMapTargetSpec for the full gate matrix.
        trace = TreeMap(path=["cat", "sub"], values="val", agg="sum")
        spec = trace.get_cube_target_spec(None, schema=_SCHEMA)
        assert spec is not None
        assert [d.column for d in spec.target_dims] == ["cat", "sub"]
        assert all(d.kind == "categorical" for d in spec.target_dims)
        assert spec.measure.agg == "sum"
        # axis_range is ignored (no cartesian viewport).
        assert spec == trace.get_cube_target_spec((0.0, 1.0), schema=_SCHEMA)

"""Unit tests for LinePlot trace."""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl
import pytest

import datetime as dt

from flexviz.LF import LFQueryBuilder
from flexviz.spec import TraceSpec
from flexviz.trace.line import LinePlot

# ---- helpers ---------------------------------------------------------------


def _aggregate_line(
    df: pl.DataFrame,
    n_points: int = 100,
    x_range: tuple[float, float] | None = None,
    downsample: str = "minmax",
) -> dict:
    """Run a full aggregation pipeline for a LinePlot and return the update dict."""
    lf = LFQueryBuilder(df)
    trace = LinePlot(x="ts", y="val", n_points=n_points, downsample=downsample)
    update_range = {"x": x_range} if x_range else {}
    agg_spec = trace.get_aggregation_spec(update_range, schema=lf.schema)
    df_agg, _ = lf.aggregate([], [agg_spec])
    return trace._to_update(df_agg).updates


def _aggregate_grouped_line(
    df: pl.DataFrame,
    x_range: tuple[float, float] | None = None,
    group_by: str | Sequence[str] | None = "sensor",
    downsample: str = "minmax",
) -> list:
    lf = LFQueryBuilder(df)
    trace = LinePlot(
        x="ts", y="val", n_points=100, group_by=group_by, downsample=downsample
    )
    update_range = {"x": x_range} if x_range is not None else {}
    agg_spec = trace.get_aggregation_spec(update_range, schema=lf.schema)
    _, grouped_dfs = lf.aggregate([], [agg_spec])
    return trace._to_grouped_update(grouped_dfs[trace.uid]).group_results or []


# ---- row count <= n_points -------------------------------------------------


class TestLinePlotRowCount:
    @pytest.mark.parametrize("n_rows", [1_000, 100_000])
    @pytest.mark.parametrize("downsample", ["minmax", "nth"])
    def test_result_count_le_n_points(self, n_rows: int, downsample: str):
        n_points = 200
        df = pl.DataFrame(
            {
                "ts": list(range(n_rows)),
                "val": [float(i) for i in range(n_rows)],
            }
        )
        update = _aggregate_line(df, n_points=n_points, downsample=downsample)
        assert len(update["x"]) <= n_points

    @pytest.mark.slow
    def test_result_count_le_n_points_1m(self):
        n_rows = 1_000_000
        n_points = 500
        df = pl.DataFrame(
            {
                "ts": list(range(n_rows)),
                "val": [float(i) for i in range(n_rows)],
            }
        )
        update = _aggregate_line(df, n_points=n_points)
        assert len(update["x"]) <= n_points


# ---- viewport filter -------------------------------------------------------


class TestLinePlotViewport:
    @pytest.mark.parametrize("downsample", ["minmax", "nth"])
    def test_viewport_reduces_count(self, small_df: pl.DataFrame, downsample: str):
        full = _aggregate_line(small_df, n_points=1000, downsample=downsample)
        zoomed = _aggregate_line(
            small_df, n_points=1000, x_range=(200, 400), downsample=downsample
        )
        assert len(zoomed["x"]) < len(full["x"])

    @pytest.mark.parametrize("downsample", ["minmax", "nth"])
    def test_viewport_data_within_range(self, small_df: pl.DataFrame, downsample: str):
        update = _aggregate_line(
            small_df, n_points=1000, x_range=(100, 300), downsample=downsample
        )
        assert all(100 <= v <= 300 for v in update["x"])

    def test_no_viewport_returns_full_dataset(self, small_df: pl.DataFrame):
        update = _aggregate_line(small_df, n_points=5000)
        assert len(update["x"]) == 1000


# ---- _to_update output shape -----------------------------------------------


class TestLinePlotToUpdate:
    def test_returns_x_y(self, small_df: pl.DataFrame):
        update = _aggregate_line(small_df, n_points=50)
        assert "x" in update
        assert "y" in update
        assert len(update["x"]) == len(update["y"])
        assert isinstance(update["x"], (list, pl.Series))


# ---- typed-literal datetime filtering --------------------------------------


class TestLinePlotDatetimeAxes:
    def test_viewport_filter_datetime_with_schema(self):
        ts = [dt.datetime(2020, 1, 1) + dt.timedelta(seconds=i) for i in range(200)]
        df = pl.DataFrame({"ts": ts, "val": [float(i) for i in range(200)]})
        lf = LFQueryBuilder(df)

        trace = LinePlot(x="ts", y="val", n_points=50)
        start = ts[50]
        end = ts[120]
        start_s = start.strftime("%Y-%m-%d %H:%M:%S.%f")
        end_s = end.strftime("%Y-%m-%d %H:%M:%S.%f")

        update_range = {"x": (start_s, end_s)}
        agg_spec = trace.get_aggregation_spec(update_range, schema=lf.schema)
        df_agg, _ = lf.aggregate([], [agg_spec])
        updates = trace._to_update(df_agg).updates

        assert len(updates["x"]) > 0  # non-empty
        assert all(start <= v <= end for v in updates["x"])

    def test_viewport_filter_utc_datetime_with_schema(self):
        base = dt.datetime(2026, 4, 1, tzinfo=dt.timezone.utc)
        frames = []
        for chunk_idx in range(20):
            start_i = chunk_idx * 1000
            ts = [
                base + dt.timedelta(minutes=i) for i in range(start_i, start_i + 1000)
            ]
            frames.append(
                pl.DataFrame(
                    {
                        "ts": pl.Series(
                            "ts", ts, dtype=pl.Datetime("us", time_zone="UTC")
                        ),
                        "val": [float(i % 100) for i in range(start_i, start_i + 1000)],
                    }
                )
            )
        df = pl.concat(frames, rechunk=False)
        lf = LFQueryBuilder(df)

        trace = LinePlot(x="ts", y="val", n_points=50)
        start = base + dt.timedelta(days=6, hours=4, minutes=25, seconds=36)
        end = base + dt.timedelta(days=10, hours=7, minutes=35, seconds=46)
        start_s = "2026-04-07 04:25:36.1694"
        end_s = "2026-04-11 07:35:46.6646"

        update_range = {"x": (start_s, end_s)}
        agg_spec = trace.get_aggregation_spec(update_range, schema=lf.schema)
        df_agg, _ = lf.aggregate([], [agg_spec])
        updates = trace._to_update(df_agg).updates

        assert len(updates["x"]) > 0
        assert all(start <= v <= end for v in updates["x"])


# ---- from_trace_spec round-trip --------------------------------------------


class TestLinePlotFromTraceSpec:
    def test_roundtrip(self):
        original = LinePlot(
            x="ts", y="val", name="Sig", n_points=250, color="#abc", downsample="nth"
        )
        spec = original.to_trace_spec()

        assert isinstance(spec, TraceSpec)
        assert spec.trace_type == "line"

        restored = LinePlot.from_trace_spec(spec)
        assert restored.uid == original.uid
        assert restored.x_col == "ts"
        assert restored.y_col == "val"
        assert restored.n_points == 250
        assert restored.downsample == "nth"
        assert restored._display.get("name") == "Sig"
        assert restored._display.get("color") == "#abc"

    def test_roundtrip_preserves_fpcs(self):
        original = LinePlot(x="ts", y="val", n_points=250, downsample="fpcs")
        spec = original.to_trace_spec()
        restored = LinePlot.from_trace_spec(spec)
        assert restored.downsample == "fpcs"

    def test_roundtrip_preserves_add_gaps(self):
        original = LinePlot(x="ts", y="val", add_gaps=False)
        spec = original.to_trace_spec()
        restored = LinePlot.from_trace_spec(spec)
        assert restored._params.get("add_gaps") is False

    def test_from_trace_spec_preserves_explicit_recompute_axes(self):
        spec = TraceSpec(
            uid="line-custom-recompute",
            trace_type="line",
            axes=("x", "y"),
            backend_data={"x": "ts", "y": "val"},
            params={"n_points": 250, "downsample": "nth", "add_gaps": True},
            recompute_axes=("y",),
        )

        restored = LinePlot.from_trace_spec(spec)

        assert restored.recompute_axes == ("y",)


# ---- plugin tests --------------------------------------------------


class TestLinePlotPlugin:
    """Tests that exercise the flexviz_polars plugin paths specifically."""

    def test_minmax_preserves_spike(self):
        """A spike in the data must appear in the minmax output."""
        vals = [0.0] * 499 + [1000.0] + [0.0] * 500
        df = pl.DataFrame({"ts": list(range(1000)), "val": vals})
        update = _aggregate_line(df, n_points=20, downsample="minmax")
        assert 1000.0 in update["y"].to_list(), "spike must be preserved by minmax"

    def test_nth_plugin_count_le_n_points(self):
        n_points = 100
        df = pl.DataFrame(
            {
                "ts": list(range(999)),  # non-round number to catch stride edge cases
                "val": [float(i) for i in range(999)],
            }
        )
        update = _aggregate_line(df, n_points=n_points, downsample="nth")
        assert len(update["x"]) <= n_points

    def test_minmax_count_le_n_points(self):
        n_points = 50
        df = pl.DataFrame(
            {
                "ts": list(range(10_000)),
                "val": [float(i % 100) for i in range(10_000)],
            }
        )
        update = _aggregate_line(df, n_points=n_points, downsample="minmax")
        assert len(update["x"]) <= n_points

    def test_minmax_float16_preserves_spike(self):
        vals = [0.0] * 499 + [10.0] + [0.0] * 500
        df = pl.DataFrame({"ts": list(range(1000)), "val": vals}).with_columns(
            pl.col("val").cast(pl.Float16)
        )
        update = _aggregate_line(df, n_points=20, downsample="minmax")
        assert len(update["x"]) <= 20
        assert 10.0 in update["y"].to_list()

    def test_minmax_agg_expr_runs_kernel_once(self):
        """Pin the fusion: one minmax_line call, no bare arg_min_max.

        Polars does not CSE opaque plugin expressions, so the two-gather form
        (x.gather(idx), y.gather(idx)) evaluated the whole argmin/argmax scan
        twice per trace. Output-based tests are blind to that: reverting to the
        two-gather form changes no result, only the work. The optimized plan is
        where the property is visible.
        """
        from flexviz.trace.line import _plugin_minmax_agg_expr

        lf = pl.DataFrame({"ts": [1.0, 2.0], "val": [3.0, 4.0]}).lazy()
        plan = lf.select(_plugin_minmax_agg_expr("ts", "val", None, 100, "u")).explain(
            optimized=True
        )
        assert plan.count("minmax_line") == 1
        assert "arg_min_max" not in plan

    def test_fpcs_preserves_spike(self):
        vals = [0.0] * 499 + [1000.0] + [0.0] * 500
        df = pl.DataFrame({"ts": list(range(1000)), "val": vals})
        update = _aggregate_line(df, n_points=40, downsample="fpcs")
        assert 1000.0 in update["y"].to_list(), "spike must be preserved by FPCS"
        assert update["x"][0] == 0
        assert update["x"][-1] == 999

    def test_fpcs_target_is_not_a_hard_cap(self):
        vals = [0.0]
        for i in range(80):
            vals.extend([10.0 + i, -10.0 - i])
        vals.append(0.0)
        n_points = 20
        df = pl.DataFrame({"ts": list(range(len(vals))), "val": vals})
        update = _aggregate_line(df, n_points=n_points, downsample="fpcs")
        assert len(update["x"]) > n_points
        assert len(update["x"]) <= 2 * n_points

    def test_fpcs_viewport_data_within_range(self):
        df = pl.DataFrame(
            {
                "ts": list(range(1000)),
                "val": [float((i * 31) % 97) for i in range(1000)],
            }
        )
        update = _aggregate_line(df, n_points=50, x_range=(100, 300), downsample="fpcs")
        assert len(update["x"]) > 0
        assert all(100 <= v <= 300 for v in update["x"])


# ---- add_gaps / gap mask ----------------------------------------------------


class TestLinePlotAddGaps:
    @pytest.mark.parametrize("add_gaps", [True, False])
    def test_to_update_never_emits_none(self, add_gaps):
        # A clear x gap (0..49 then 10_000..10_049). Gaps are now a client
        # render-time concern (fvApplyLineGaps), so _to_update must stay gapless
        # regardless of add_gaps, with x order preserved.
        x = list(range(50)) + list(range(10_000, 10_050))
        y = [float(i) for i in range(len(x))]
        df = pl.DataFrame({"ts": x, "val": y})
        lf = LFQueryBuilder(df)

        trace = LinePlot(x="ts", y="val", n_points=5000, add_gaps=add_gaps)
        agg_spec = trace.get_aggregation_spec({}, schema=lf.schema)
        df_agg, _ = lf.aggregate([], [agg_spec])
        update = trace._to_update(df_agg).updates

        xs = update["x"].to_list()
        ys = update["y"].to_list()
        assert len(xs) == len(ys)
        assert all(v is not None for v in xs), "server must emit gapless x"
        assert all(v is not None for v in ys), "server must emit gapless y"
        assert xs == sorted(xs), "x order preserved (no interleaved breaks)"


class TestLinePlotGrouped:
    def test_grouped_returns_child_results(self):
        df = pl.DataFrame(
            {
                "ts": list(range(20)) * 2,
                "val": [float(i) for i in range(20)] * 2,
                "sensor": ["A"] * 20 + ["B"] * 20,
            }
        )
        results = _aggregate_grouped_line(df)
        assert len(results) == 2
        assert {cr.group_value_key for cr in results} == {"A", "B"}

    def test_grouped_viewport_controls_visible_groups(self):
        df = pl.DataFrame(
            {
                "ts": list(range(20)) + list(range(100, 120)),
                "val": [float(i) for i in range(40)],
                "sensor": ["A"] * 20 + ["B"] * 20,
            }
        )
        results = _aggregate_grouped_line(df, x_range=(0, 40))
        assert [cr.group_value_key for cr in results] == ["A"]

    def test_grouped_empty_viewport_returns_no_children(self):
        df = pl.DataFrame(
            {
                "ts": list(range(20)) * 2,
                "val": [float(i) for i in range(20)] * 2,
                "sensor": ["A"] * 20 + ["B"] * 20,
            }
        )
        results = _aggregate_grouped_line(df, x_range=(100, 120))
        assert results == []

    def test_grouped_fpcs_returns_child_results(self):
        df = pl.DataFrame(
            {
                "ts": list(range(200)) * 2,
                "val": [float((i * 31) % 97) for i in range(200)] * 2,
                "sensor": ["A"] * 200 + ["B"] * 200,
            }
        )
        results = _aggregate_grouped_line(df, group_by="sensor", downsample="fpcs")
        assert len(results) == 2
        assert {cr.group_value_key for cr in results} == {"A", "B"}
        assert all(cr.updates["x"][0] == 0 for cr in results)
        assert all(cr.updates["x"][-1] == 199 for cr in results)

    def test_grouped_by_two_columns_returns_composite_children(self):
        df = pl.DataFrame(
            {
                "ts": list(range(10)) * 3,
                "val": [float(i) for i in range(30)],
                "sensor": ["A"] * 10 + ["A"] * 10 + ["B"] * 10,
                "site": ["north"] * 10 + ["south"] * 10 + ["north"] * 10,
            }
        )
        results = _aggregate_grouped_line(df, group_by=["sensor", "site"])
        assert {cr.group_value_key for cr in results} == {
            '["A","north"]',
            '["A","south"]',
            '["B","north"]',
        }


class TestGroupedLineEmptyResult:
    """The streaming envelope builds its own empty frame, so its schema is
    hand-written rather than derived from the data."""

    @staticmethod
    def _empty_grouped(df, group_by="sensor"):
        lf = LFQueryBuilder(df)
        trace = LinePlot(x="ts", y="val", n_points=100, group_by=group_by)
        spec = trace.get_aggregation_spec({}, schema=lf.schema)
        # A filter nothing survives, so the plan takes its empty branch.
        _, grouped = lf.aggregate([pl.col("ts") > 10**9], [spec])
        return trace, grouped[trace.uid]

    def test_group_column_keeps_its_dtype_when_empty(self):
        """An Int group column must not come back as Utf8 just because the
        result has no rows."""
        df = pl.DataFrame(
            {
                "ts": list(range(20)),
                "val": [float(i) for i in range(20)],
                "sensor": [i % 2 for i in range(20)],
            }
        )
        _, out = self._empty_grouped(df)
        assert out.height == 0
        assert out.schema["sensor"] == pl.Int64

    def test_string_group_column_is_unchanged(self):
        df = pl.DataFrame(
            {
                "ts": list(range(20)),
                "val": [float(i) for i in range(20)],
                "sensor": ["A", "B"] * 10,
            }
        )
        _, out = self._empty_grouped(df)
        assert out.height == 0
        assert out.schema["sensor"] == pl.String

    def test_empty_result_yields_no_children(self):
        df = pl.DataFrame(
            {
                "ts": list(range(20)),
                "val": [float(i) for i in range(20)],
                "sensor": [i % 2 for i in range(20)],
            }
        )
        trace, out = self._empty_grouped(df)
        assert (trace._to_grouped_update(out).group_results or []) == []


class TestLinePlotHoverSpec:
    def test_line_has_axis_source_mode(self):
        from flexviz.trace.line import LinePlot

        t = LinePlot(x="ts", y="val")
        spec = t.to_trace_spec()
        assert "axis" in spec.hover.source_modes

    def test_line_has_axis_target_mode(self):
        from flexviz.trace.line import LinePlot

        t = LinePlot(x="ts", y="val")
        spec = t.to_trace_spec()
        assert "axis" in spec.hover.target_modes

    def test_line_hover_roundtrips_through_spec(self):
        import json
        from flexviz.trace.line import LinePlot
        from flexviz.spec import TraceSpec

        t = LinePlot(x="ts", y="val")
        spec = t.to_trace_spec()
        restored = TraceSpec.model_validate(json.loads(spec.model_dump_json()))
        assert restored.hover.source_modes == spec.hover.source_modes
        assert restored.hover.target_modes == spec.hover.target_modes


# ---- cube source descriptor (Phase 4, plan step 8) ---------------------------


class TestLineCubeSource:
    """LinePlot cube source descriptor — x-only (locked v0.2 decision).

    A line's selection geometry is x-only (``select_axes = (x anchor,)``,
    matching Mosaic ``intervalX``), so its cube free axis is the x column.
    Identical shape to the histogram's descriptor: P=2048, domain = the
    x-viewport range verbatim. Temporal dtypes carry their physical ``unit``
    (contract G); ``Datetime("ns")``/``Time`` and non-numeric dtypes gate to
    ``None``.
    """

    def test_unzoomed_x_column(self):
        from flexviz.cube import FreeAxisSpec

        trace = LinePlot(x="ts", y="val")
        spec = trace.get_cube_source_spec(None)
        assert isinstance(spec, FreeAxisSpec)
        assert spec.column == "ts"
        assert spec.kind == "continuous"
        assert spec.p == 2048
        assert spec.domain is None

    def test_zoomed_domain_is_x_viewport_verbatim(self):
        trace = LinePlot(x="ts", y="val")
        spec = trace.get_cube_source_spec((10.0, 50.0))
        assert spec is not None
        assert tuple(spec.domain) == (10.0, 50.0)

    def test_selection_spec_is_x_only(self):
        # The emitted selection predicate carries only the x-axis column —
        # committed line predicates have no y clause (user-visible v0.2 lock).
        sel = LinePlot(x="ts", y="val")._make_selection_spec()
        assert sel.kind == "range"
        assert sel.axis_columns == {"x": "ts"}

    @pytest.mark.parametrize(
        "dtype,unit",
        [(pl.Datetime("us"), "us"), (pl.Datetime("ms"), "ms"), (pl.Date, "day")],
    )
    def test_temporal_kind_and_unit_from_schema(self, dtype, unit):
        schema = pl.Schema({"ts": dtype, "val": pl.Float64})
        spec = LinePlot(x="ts", y="val").get_cube_source_spec(None, schema=schema)
        assert spec is not None
        assert spec.kind == "temporal"
        assert spec.unit == unit

    @pytest.mark.parametrize("dtype", [pl.Datetime("ns"), pl.Time])
    def test_ns_and_time_gate_to_none(self, dtype):
        # Datetime("ns") has no exact string round-trip (µs precision); Time
        # has no date-string representation at all (contract G).
        schema = pl.Schema({"ts": dtype, "val": pl.Float64})
        trace = LinePlot(x="ts", y="val")
        assert trace.get_cube_source_spec(None, schema=schema) is None

    def test_non_numeric_x_gate(self):
        schema = pl.Schema({"ts": pl.String, "val": pl.Float64})
        trace = LinePlot(x="ts", y="val")
        assert trace.get_cube_source_spec(None, schema=schema) is None

    def test_defaults_to_continuous_without_schema(self):
        spec = LinePlot(x="ts", y="val").get_cube_source_spec(None, schema=None)
        assert spec is not None
        assert spec.kind == "continuous"

    @pytest.mark.parametrize("downsample", ["minmax", "fpcs", "nth"])
    def test_source_independent_of_downsample(self, downsample):
        # Source-ability is selection geometry, not aggregation: the minmax
        # gate belongs to the line *target* descriptor (contract J), not here.
        trace = LinePlot(x="ts", y="val", downsample=downsample)
        assert trace.get_cube_source_spec(None) is not None

    def test_grouped_line_is_still_a_source(self):
        schema = pl.Schema({"ts": pl.Float64, "val": pl.Float64, "sensor": pl.String})
        trace = LinePlot(x="ts", y="val", group_by="sensor")
        spec = trace.get_cube_source_spec((0.0, 1.0), schema=schema)
        assert spec is not None
        assert spec.column == "ts"

    def test_target_spec_still_none(self):
        # The line-envelope target lands in a later step (contract J).
        assert LinePlot(x="ts", y="val").get_cube_target_spec(None) is None


# ---- sorted-x viewport slice ------------------------------------------------


class TestSortedViewportSlice:
    """``x_sorted`` picks a zero-copy slice over an ``is_between`` mask.

    It is a performance path only: every case here asserts the slice returns
    exactly what the mask returns.
    """

    @staticmethod
    def _agg(df: pl.DataFrame, x_range, x_sorted: bool, downsample="minmax") -> dict:
        lf = LFQueryBuilder(df)
        trace = LinePlot(x="ts", y="val", n_points=100, downsample=downsample)
        spec = trace.get_aggregation_spec(
            {"x": x_range} if x_range else {},
            schema=lf.schema,
            x_sorted=x_sorted,
        )
        df_agg, _ = lf.aggregate([], [spec])
        return trace._to_update(df_agg).updates

    @staticmethod
    def _norm(update) -> dict:
        """updates hold Series/arrays, so `==` on the dicts is elementwise."""
        return {
            k: (list(v) if hasattr(v, "__iter__") else v) for k, v in update.items()
        }

    @classmethod
    def _same(cls, a, b) -> bool:
        return cls._norm(a) == cls._norm(b)

    @staticmethod
    def _df(n: int = 5_000, dtype=pl.Int64) -> pl.DataFrame:
        return pl.DataFrame(
            {"ts": list(range(n)), "val": [float((i * 37) % 101) for i in range(n)]}
        ).with_columns(ts=pl.col("ts").cast(dtype))

    @pytest.mark.parametrize("downsample", ["minmax", "fpcs", "nth"])
    @pytest.mark.parametrize(
        "x_range",
        [
            (100, 900),
            (0, 4999),  # whole domain
            (0.5, 1234.7),  # fractional bounds on an integer column
            (-50, 10**9),  # bounds outside the data on both sides
            (2500, 2500),  # single point
        ],
    )
    def test_slice_matches_mask(self, x_range, downsample):
        df = self._df()
        assert self._same(
            self._agg(df, x_range, True, downsample),
            self._agg(df, x_range, False, downsample),
        )

    def test_slice_matches_mask_float_x(self):
        df = self._df(dtype=pl.Float64)
        assert self._same(
            self._agg(df, (10.25, 3999.75), True),
            self._agg(df, (10.25, 3999.75), False),
        )

    def test_unzoomed_is_untouched(self):
        df = self._df()
        assert self._same(self._agg(df, None, True), self._agg(df, None, False))

    def test_empty_viewport(self):
        # A range past the end of the data slices to length 0, same as a mask
        # that matches nothing.
        df = self._df()
        assert self._same(
            self._agg(df, (10**8, 10**9), True), self._agg(df, (10**8, 10**9), False)
        )

    def test_grouped_path_ignores_x_sorted(self):
        # Grouped lines restrict the frame before grouping, so they keep the
        # mask; passing x_sorted must not change their output.
        df = self._df(n=2_000).with_columns(
            sensor=pl.Series(["a", "b"] * 1_000),
        )
        lf = LFQueryBuilder(df)
        out = []
        for flag in (False, True):
            trace = LinePlot(x="ts", y="val", n_points=100, group_by="sensor")
            spec = trace.get_aggregation_spec(
                {"x": (100, 1900)}, schema=lf.schema, x_sorted=flag
            )
            _, grouped = lf.aggregate([], [spec])
            out.append(
                [
                    self._norm(g.updates)
                    for g in (
                        trace._to_grouped_update(grouped[trace.uid]).group_results or []
                    )
                ]
            )
        assert out[0] == out[1]
        assert out[0], "grouped viewport should not be empty"

    def test_slice_correct_under_cross_filter(self):
        # A cross-filter removes rows before the expression runs. Order is
        # preserved, so search_sorted must resolve against the FILTERED column;
        # if it resolved against the original, the window would be wrong.
        df = self._df()
        lf = LFQueryBuilder(df)
        keep = pl.col("val") > 20.0
        got = []
        for flag in (False, True):
            trace = LinePlot(x="ts", y="val", n_points=100)
            spec = trace.get_aggregation_spec(
                {"x": (500, 3000)}, schema=lf.schema, x_sorted=flag
            )
            df_agg, _ = lf.aggregate([keep], [spec])
            got.append(self._norm(trace._to_update(df_agg).updates))
        assert got[0] == got[1]
        assert got[0]["x"], "cross-filtered viewport should not be empty"

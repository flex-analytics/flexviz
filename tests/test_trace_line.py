"""Unit tests for LinePlot trace."""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl
import pytest

import datetime as dt
import math
import random

from flexviz.engine import FlexEngine, TraceInfo
from flexviz.events import InteractionEvent
from flexviz.figure import Figure
from flexviz.LF import LFQueryBuilder
from flexviz.spec import TraceSpec
from flexviz.trace.line import LinePlot
from flexviz.trace.line_buckets import _grouped_bucket_keys, pairs_plan

# ---- helpers ---------------------------------------------------------------


def _domains(lf: LFQueryBuilder, trace: LinePlot, update_range: dict) -> dict:
    """The unfiltered bounds the engine would resolve for this trace."""
    cols = trace.domain_cols(update_range)
    return lf.physical_minmax(list(cols), memoize=False) if cols else {}


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
    agg_spec = trace.get_aggregation_spec(
        update_range, schema=lf.schema, domains=_domains(lf, trace, update_range)
    )
    df_agg, _ = lf.aggregate([], [agg_spec])
    return trace._to_update(df_agg).updates


def _aggregate_grouped_line(
    df: pl.DataFrame,
    x_range: tuple[float, float] | None = None,
    group_by: str | Sequence[str] | None = "sensor",
    downsample: str = "minmax",
    n_points: int = 100,
) -> list:
    lf = LFQueryBuilder(df)
    trace = LinePlot(
        x="ts", y="val", n_points=n_points, group_by=group_by, downsample=downsample
    )
    update_range = {"x": x_range} if x_range is not None else {}
    agg_spec = trace.get_aggregation_spec(
        update_range, schema=lf.schema, domains=_domains(lf, trace, update_range)
    )
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

    @pytest.mark.parametrize("downsample", ["fpcs", "lttb"])
    def test_roundtrip_preserves_downsample(self, downsample):
        original = LinePlot(x="ts", y="val", n_points=250, downsample=downsample)
        spec = original.to_trace_spec()
        restored = LinePlot.from_trace_spec(spec)
        assert restored.downsample == downsample

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

    def test_pairs_agg_expr_runs_kernel_once(self):
        """Pin the fusion: one kernel call per trace.

        Polars does not CSE opaque plugin expressions, so a per-field form would
        evaluate the whole argmin/argmax scan once per gather. Output-based tests
        are blind to that: it changes no result, only the work. The optimized
        plan is where the property is visible.
        """
        from flexviz.trace.line import _plugin_pairs_agg_expr

        lf = pl.DataFrame({"ts": [1.0, 2.0], "val": [3.0, 4.0]}).lazy()
        plan = lf.select(
            _plugin_pairs_agg_expr("ts", "val", None, 50, "u", (1.0, 2.0))
        ).explain(optimized=True)
        assert plan.count("minmax_pairs_line") == 1

    def test_fpcs_preserves_spike(self):
        vals = [0.0] * 499 + [1000.0] + [0.0] * 500
        df = pl.DataFrame({"ts": list(range(1000)), "val": vals})
        update = _aggregate_line(df, n_points=40, downsample="fpcs")
        assert 1000.0 in update["y"].to_list(), "spike must be preserved by FPCS"
        assert update["x"][0] == 0
        assert update["x"][-1] == 999

    def test_fpcs_target_is_not_a_hard_cap(self):
        # A sawtooth: every bucket defers an extremum that the next one flushes,
        # so the walk emits close to two points per bucket.
        n = 2_000
        df = pl.DataFrame(
            {"ts": list(range(n)), "val": [float((i * 7) % 53) for i in range(n)]}
        )
        n_points = 20
        update = _aggregate_line(df, n_points=n_points, downsample="fpcs")
        assert len(update["x"]) > n_points
        assert len(update["x"]) <= 2 * n_points

    def test_fpcs_gappy_x_returns_fewer_points(self):
        # x-width buckets over a gap are empty, and an empty bucket emits
        # nothing. Row-count buckets would spend the whole budget.
        xs = list(range(500)) + list(range(500_000, 500_500))
        df = pl.DataFrame(
            {"ts": xs, "val": [float((i * 31) % 97) for i in range(len(xs))]},
            schema={"ts": pl.Int64, "val": pl.Float64},
        )
        update = _aggregate_line(df, n_points=200, downsample="fpcs")
        assert 0 < len(update["x"]) < 200

    def test_fpcs_walk_matches_a_hand_checked_trace(self):
        # Four row-count buckets of five rows over x = 0..19, y distinct.
        # Traced by hand against the compensation rules: bucket 2 flushes the
        # deferred max of bucket 1, and the trailing potential point closes it.
        from flexviz.trace.line import _fpcs_walk

        xs, ys = _fpcs_walk(
            [0, 9, 11, 19],
            [10.0, 21.0, 5.0, 15.0],
            [4, 5, 14, 15],
            [14.0, 25.0, 32.0, 19.0],
        )
        assert xs == [0, 4, 5, 11, 14, 19]
        assert ys == [10.0, 14.0, 25.0, 5.0, 32.0, 15.0]

    def test_fpcs_keeps_both_extrema_at_one_x(self):
        # Two rows share a timestamp and hold the min and the max of their
        # bucket. Deduplicating on x alone dropped one of the two.
        df = pl.DataFrame(
            {
                "ts": [0, 0, 1, 2, 3, 4, 5, 6],
                "val": [-10.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            }
        )
        update = _aggregate_line(df, n_points=4, downsample="fpcs")
        ys = update["y"].to_list()
        assert 10.0 in ys and -10.0 in ys

    def test_fpcs_emits_a_repeated_point_once(self):
        # Two single-row buckets: the first defers the point it just emitted,
        # and the second re-emits it.
        from flexviz.trace.line import _fpcs_walk

        assert _fpcs_walk([0, 1], [5.0, 1.0], [0, 1], [5.0, 1.0]) == (
            [0, 1],
            [5.0, 1.0],
        )

    def test_fpcs_resident_matches_the_scan_plan(self, tmp_path):
        df = _gappy_frame()
        path = tmp_path / "fpcs.parquet"
        df.write_parquet(path)
        scan_lf = LFQueryBuilder(pl.scan_parquet(path))
        assert scan_lf.is_scan

        trace = LinePlot(x="ts", y="val", n_points=200, downsample="fpcs")
        resident = _minmax_points(LFQueryBuilder(df), trace)
        scanned = _minmax_points(scan_lf, trace)

        assert len(resident["x"]) > 0
        assert resident["x"].to_list() == scanned["x"].to_list()
        assert resident["y"].to_list() == scanned["y"].to_list()

    def test_fpcs_scan_with_an_all_null_y_returns_a_typed_empty_x(self, tmp_path):
        df = pl.DataFrame(
            {"ts": list(range(100)), "val": [None] * 100},
            schema={"ts": pl.Int64, "val": pl.Float64},
        )
        path = tmp_path / "null_y.parquet"
        df.write_parquet(path)
        scan_lf = LFQueryBuilder(pl.scan_parquet(path))
        assert scan_lf.is_scan

        out = _minmax_points(
            scan_lf, LinePlot(x="ts", y="val", n_points=20, downsample="fpcs")
        )
        assert len(out["x"]) == 0
        assert out["x"].dtype == pl.Int64

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
        agg_spec = trace.get_aggregation_spec(
            {}, schema=lf.schema, domains=_domains(lf, trace, {})
        )
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
        # The walk keeps x ordered inside each group.
        for cr in results:
            xs = cr.updates["x"].to_list()
            assert 0 < len(xs) <= 2 * 100
            assert xs == sorted(xs)

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

    @pytest.mark.parametrize("downsample", ["minmax", "lttb", "fpcs", "nth"])
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
    exactly what the mask returns. The flag decides for ``nth`` alone; an
    x-width line is sorted by contract and always slices.
    """

    @staticmethod
    def _agg(df: pl.DataFrame, x_range, x_sorted: bool, downsample="nth") -> dict:
        lf = LFQueryBuilder(df)
        trace = LinePlot(x="ts", y="val", n_points=100, downsample=downsample)
        update_range = {"x": x_range} if x_range else {}
        spec = trace.get_aggregation_spec(
            update_range,
            schema=lf.schema,
            x_sorted=x_sorted,
            domains=_domains(lf, trace, update_range),
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
    def test_slice_matches_mask(self, x_range):
        df = self._df()
        assert self._same(
            self._agg(df, x_range, True),
            self._agg(df, x_range, False),
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
            trace = LinePlot(x="ts", y="val", n_points=100, downsample="nth")
            spec = trace.get_aggregation_spec(
                {"x": (500, 3000)}, schema=lf.schema, x_sorted=flag
            )
            df_agg, _ = lf.aggregate([keep], [spec])
            got.append(self._norm(trace._to_update(df_agg).updates))
        assert got[0] == got[1]
        assert got[0]["x"], "cross-filtered viewport should not be empty"


# ---- equal-x-width buckets (ungrouped minmax) -------------------------------


def _minmax_points(lf: LFQueryBuilder, trace: LinePlot, x_range=None) -> dict:
    """Aggregate one ungrouped line the way the engine would."""
    update_range = {"x": x_range} if x_range is not None else {}
    spec = trace.get_aggregation_spec(
        update_range,
        schema=lf.schema,
        x_sorted=True,
        scan_source=lf.is_scan,
        domains=_domains(lf, trace, update_range),
    )
    df_agg, _ = lf.aggregate([], [spec])
    return trace._to_update(df_agg).updates


def _gappy_frame() -> pl.DataFrame:
    """Sorted Int64 x with a large gap. Every y is distinct, so no plateau ties.

    The span (100_000) is a whole multiple of the 100 buckets, so the streaming
    plan's integer-ceil width equals the kernel's true-division width and the
    two grids coincide exactly.
    """
    xs = (
        list(range(1000))  # dense head
        + list(range(20_000, 60_000, 40))  # medium block after the gap
        + list(range(60_000, 100_001, 500))  # sparse tail
    )
    n = len(xs)
    return pl.DataFrame(
        {"ts": xs, "val": [float((i * 7919) % n) for i in range(n)]},
        schema={"ts": pl.Int64, "val": pl.Float64},
    )


class TestLineXWidthBuckets:
    @pytest.mark.parametrize(
        "kwargs,update_range,expected",
        [
            ({}, {}, ("ts",)),
            ({}, {"x": (0, 10)}, ()),
            ({"group_by": "sensor"}, {}, ("ts",)),
            ({"group_by": "sensor"}, {"x": (0, 10)}, ()),
            ({"group_by": ["sensor", "gi"], "downsample": "fpcs"}, {}, ("ts",)),
            ({"group_by": "sensor", "downsample": "nth"}, {}, ()),
            ({"downsample": "nth"}, {}, ()),
            ({"downsample": "fpcs"}, {}, ("ts",)),
            ({"downsample": "lttb"}, {}, ("ts",)),
            ({"downsample": "fpcs"}, {"x": (0, 10)}, ()),
        ],
        ids=[
            "unzoomed",
            "zoomed",
            "grouped",
            "grouped_zoomed",
            "grouped_two_cols",
            "grouped_nth",
            "nth",
            "fpcs",
            "lttb",
            "fpcs_zoomed",
        ],
    )
    def test_domain_cols(self, kwargs, update_range, expected):
        # The x domain is requested on both source kinds: every x-width
        # formulation buckets by x width. Grouping adds no column: the group
        # key needs no bounds.
        trace = LinePlot(x="ts", y="val", **kwargs)
        assert trace.domain_cols(update_range) == expected

    @staticmethod
    def _resident_and_scan(df, tmp_path, n_points, x_range=None):
        """The same line aggregated on a resident frame and on a scan."""
        path = tmp_path / "x_width.parquet"
        df.write_parquet(path)
        scan_lf = LFQueryBuilder(pl.scan_parquet(path))
        assert scan_lf.is_scan
        return (
            _minmax_points(
                LFQueryBuilder(df),
                LinePlot(x="ts", y="val", n_points=n_points),
                x_range=x_range,
            ),
            _minmax_points(
                scan_lf,
                LinePlot(x="ts", y="val", n_points=n_points),
                x_range=x_range,
            ),
        )

    def test_resident_kernel_matches_the_scan_plan(self, tmp_path):
        df = _gappy_frame()
        path = tmp_path / "gappy.parquet"
        df.write_parquet(path)

        resident = _minmax_points(
            LFQueryBuilder(df), LinePlot(x="ts", y="val", n_points=200)
        )
        scan_lf = LFQueryBuilder(pl.scan_parquet(path))
        assert scan_lf.is_scan
        scanned = _minmax_points(scan_lf, LinePlot(x="ts", y="val", n_points=200))

        assert len(resident["x"]) == len(scanned["x"])
        assert sorted(resident["y"].to_list()) == sorted(scanned["y"].to_list())

    def test_bursty_x_spends_its_budget_on_the_tail(self):
        # 1000 rows packed into [0, 999], then a sparse tail out to 999_000.
        xs = list(range(1000)) + list(range(1000, 1_000_000, 1000))
        n = len(xs)
        df = pl.DataFrame(
            {"ts": xs, "val": [float((i * 7919) % n) for i in range(n)]},
            schema={"ts": pl.Int64, "val": pl.Float64},
        )
        out = _minmax_points(
            LFQueryBuilder(df), LinePlot(x="ts", y="val", n_points=200)
        )

        xs_out = out["x"].to_list()
        assert len(xs_out) > 0
        in_tail = sum(v >= 1000 for v in xs_out)
        assert in_tail >= 0.9 * len(xs_out)

    def test_zoom_buckets_span_the_viewport(self):
        xs = list(range(1000)) + list(range(1000, 1_000_000, 1000))
        n = len(xs)
        df = pl.DataFrame(
            {"ts": xs, "val": [float((i * 7919) % n) for i in range(n)]},
            schema={"ts": pl.Int64, "val": pl.Float64},
        )
        lf = LFQueryBuilder(df)
        trace = LinePlot(x="ts", y="val", n_points=200)

        unzoomed = _minmax_points(lf, trace)
        zoomed = _minmax_points(lf, trace, x_range=(500_000, 999_000))

        xs_out = zoomed["x"].to_list()
        assert all(500_000 <= v <= 999_000 for v in xs_out)
        # 100 buckets over the viewport, ~5 rows each: near the full budget.
        assert len(xs_out) >= 180
        # The same span holds far fewer points when the buckets span the data.
        assert sum(v >= 500_000 for v in unzoomed["x"].to_list()) < len(xs_out)

    def test_all_null_x_breaks_the_x_contract(self):
        # The engine rejects a null x before the aggregation runs. The
        # aggregation itself also yields nothing.
        df = pl.DataFrame(
            {"ts": pl.Series("ts", [None, None], dtype=pl.Int64), "val": [1.0, 2.0]}
        )
        lf = LFQueryBuilder(df)
        with pytest.raises(ValueError, match="null values"):
            lf.check_line_x("ts", memoize=True)
        out = _minmax_points(lf, LinePlot(x="ts", y="val", n_points=20))
        assert len(out["x"]) == 0 or all(v is None for v in out["x"].to_list())

    def test_constant_x_still_returns_its_envelope(self):
        df = pl.DataFrame(
            {"ts": [7] * 100, "val": [float(i) for i in range(100)]},
            schema={"ts": pl.Int64, "val": pl.Float64},
        )
        out = _minmax_points(LFQueryBuilder(df), LinePlot(x="ts", y="val", n_points=20))
        assert sorted(out["y"].to_list()) == [0.0, 99.0]

    def test_true_max_x_survives_on_a_float_domain(self):
        # lo + width * n_buckets rounds below this max, so only a closed last
        # bucket keeps it.
        df = pl.DataFrame(
            {"ts": [0.667425, 99.849514], "val": [0.0, 100.0]},
            schema={"ts": pl.Float64, "val": pl.Float64},
        )
        out = _minmax_points(LFQueryBuilder(df), LinePlot(x="ts", y="val", n_points=20))
        assert out["x"].to_list() == [0.667425, 99.849514]

    def test_max_x_row_survives_on_random_float_domains(self):
        # y strictly increasing, so the max-x row is the global y max and must
        # appear in any correct envelope.
        rng = random.Random(20260903)
        for _ in range(300):
            k = rng.choice([2, 3, 5, 10, 50, 200])
            xs = sorted({rng.uniform(-1e3, 1e3) for _ in range(k)})
            if len(xs) < 2:
                continue
            df = pl.DataFrame(
                {"ts": xs, "val": [float(i) for i in range(len(xs))]},
                schema={"ts": pl.Float64, "val": pl.Float64},
            )
            n_points = rng.choice([2, 4, 20, 100, 1000])
            out = _minmax_points(
                LFQueryBuilder(df), LinePlot(x="ts", y="val", n_points=n_points)
            )
            assert xs[-1] in out["x"].to_list(), (xs[-1], n_points)

    @pytest.mark.parametrize(
        "base", [2**53, 1_700_000_000_000_000_000], ids=["2**53", "ns_epoch"]
    )
    def test_large_integer_x_matches_the_scan_plan(self, base, tmp_path):
        # Past 2**53 these 20 x values share f64 representations.
        df = pl.DataFrame(
            {"ts": [base + i for i in range(20)], "val": [float(i) for i in range(20)]},
            schema={"ts": pl.Int64, "val": pl.Float64},
        )
        resident, scanned = self._resident_and_scan(df, tmp_path, n_points=20)
        assert len(resident["x"]) == 20
        assert resident["x"].to_list() == scanned["x"].to_list()

    def test_float_x_on_a_bucket_edge_matches_the_scan_plan(self, tmp_path):
        # 10 buckets of width 0.1 over (0.0, 1.0), so many rows sit exactly on
        # an edge. The kernel and the plan must place each of them in the same
        # bucket: `7 * 0.1` rounds above 0.7, so an edge comparison put that row
        # one bucket below the plan.
        # y rises with x, so each bucket's extrema are its first and last row:
        # one row moving bucket changes the output.
        n = 1000
        df = pl.DataFrame(
            {
                "ts": [i / n for i in range(n + 1)],
                "val": [float(i) for i in range(n + 1)],
            },
            schema={"ts": pl.Float64, "val": pl.Float64},
        )
        resident, scanned = self._resident_and_scan(df, tmp_path, n_points=20)
        assert resident["x"].to_list() == scanned["x"].to_list()
        assert resident["y"].to_list() == scanned["y"].to_list()

    @pytest.mark.parametrize("unit", ["ns", "us"])
    @pytest.mark.parametrize("zoomed", [False, True], ids=["unzoomed", "zoomed"])
    def test_fine_grained_datetime_matches_the_scan_plan(self, unit, zoomed, tmp_path):
        # 20 samples one time-unit apart: at "ns" the physical values are past
        # 2**53 and only integer edges separate them.
        t0 = dt.datetime(2023, 1, 1, tzinfo=dt.timezone.utc)
        phys0 = int(t0.timestamp() * (1e9 if unit == "ns" else 1e6))
        df = pl.DataFrame(
            {
                "ts": [phys0 + i for i in range(20)],
                "val": [float(i) for i in range(20)],
            },
            schema={"ts": pl.Int64, "val": pl.Float64},
        ).with_columns(pl.col("ts").cast(pl.Datetime(unit)))
        # A viewport bound is a date string, which carries microseconds at most,
        # so the "ns" window is 1000x wider than its 19 ns of data and one
        # bucket swallows every row.
        x_range = ("2023-01-01T00:00:00", "2023-01-01T00:00:00.000019")
        expected = 2 if zoomed and unit == "ns" else 20

        resident, scanned = self._resident_and_scan(
            df, tmp_path, n_points=20, x_range=x_range if zoomed else None
        )
        assert len(resident["x"]) == expected
        assert resident["x"].to_list() == scanned["x"].to_list()


# ---- equal-x-width buckets (grouped) ----------------------------------------


def _grouped_points(lf: LFQueryBuilder, trace: LinePlot, x_range=None) -> dict:
    """Aggregate one grouped line the way the engine would, keyed by group."""
    update_range = {"x": x_range} if x_range is not None else {}
    spec = trace.get_aggregation_spec(
        update_range,
        schema=lf.schema,
        scan_source=lf.is_scan,
        domains=_domains(lf, trace, update_range),
    )
    _, grouped = lf.aggregate([], [spec])
    results = trace._to_grouped_update(grouped[trace.uid]).group_results or []
    return {cr.group_value_key: cr.updates for cr in results}


def _grouped_frame(n: int = 400) -> pl.DataFrame:
    """Two series over the same sorted x, every y distinct: no plateau ties."""
    return pl.DataFrame(
        {
            "ts": list(range(n)) * 2,
            "val": [float(i) for i in range(2 * n)],
            "g": ["a"] * n + ["b"] * n,
            "gi": [3] * n + [8] * n,
            "gc": pl.Series(["a"] * n + ["b"] * n, dtype=pl.Categorical),
            "site": ["north"] * n + ["south"] * n,
        }
    )


def _points(updates) -> tuple[list, list]:
    return updates["x"].to_list(), updates["y"].to_list()


class TestGroupedXWidthBuckets:
    """A grouped line bins on the same global grid as an ungrouped one."""

    @pytest.mark.parametrize("downsample", ["minmax", "fpcs"])
    def test_one_child_per_group(self, downsample):
        out = _grouped_points(
            LFQueryBuilder(_grouped_frame()),
            LinePlot(
                x="ts", y="val", n_points=100, group_by="g", downsample=downsample
            ),
        )
        assert set(out) == {"a", "b"}
        assert all(len(u["x"]) > 0 for u in out.values())

    def test_one_group_matches_the_ungrouped_line(self):
        # Same grid, same points: the group column only splits the rows.
        df = _gappy_frame().with_columns(g=pl.lit("a"))
        lf = LFQueryBuilder(df)
        ungrouped = _minmax_points(lf, LinePlot(x="ts", y="val", n_points=200))
        grouped = _grouped_points(
            lf, LinePlot(x="ts", y="val", n_points=200, group_by="g")
        )
        assert _points(grouped["a"]) == _points(ungrouped)

    @pytest.mark.parametrize(
        "group_by", ["g", "gi", "gc", ["g", "gi"]], ids=["str", "int", "cat", "two"]
    )
    def test_every_key_form_gives_the_same_envelopes(self, group_by):
        lf = LFQueryBuilder(_grouped_frame())
        out = _grouped_points(
            lf, LinePlot(x="ts", y="val", n_points=100, group_by=group_by)
        )
        reference = _grouped_points(
            lf, LinePlot(x="ts", y="val", n_points=100, group_by="g")
        )
        assert [_points(u) for u in out.values()] == [
            _points(u) for u in reference.values()
        ]

    @pytest.mark.parametrize("downsample", ["minmax", "fpcs"])
    def test_resident_and_scan_agree(self, tmp_path, downsample):
        df = _grouped_frame()
        path = tmp_path / "grouped.parquet"
        df.write_parquet(path)
        scan_lf = LFQueryBuilder(pl.scan_parquet(path))
        assert scan_lf.is_scan

        def run(lf):
            return {
                k: _points(u)
                for k, u in _grouped_points(
                    lf,
                    LinePlot(
                        x="ts",
                        y="val",
                        n_points=100,
                        group_by="g",
                        downsample=downsample,
                    ),
                ).items()
            }

        assert run(LFQueryBuilder(df)) == run(scan_lf)

    def test_temporal_x_keeps_its_dtype(self):
        n = 200
        df = pl.DataFrame(
            {
                "ts": [
                    dt.datetime(2023, 1, 1) + dt.timedelta(minutes=i) for i in range(n)
                ]
                * 2,
                "val": [float(i) for i in range(2 * n)],
                "g": ["a"] * n + ["b"] * n,
            }
        )
        out = _grouped_points(
            LFQueryBuilder(df), LinePlot(x="ts", y="val", n_points=50, group_by="g")
        )
        assert set(out) == {"a", "b"}
        assert out["a"]["x"].dtype == pl.Datetime("us")

    def test_float_x_works(self):
        df = _grouped_frame().with_columns(ts=pl.col("ts").cast(pl.Float64) / 3.0)
        out = _grouped_points(
            LFQueryBuilder(df), LinePlot(x="ts", y="val", n_points=100, group_by="g")
        )
        assert set(out) == {"a", "b"}
        assert out["b"]["x"].dtype == pl.Float64

    def test_zoom_restricts_the_groups(self):
        n = 100
        df = pl.DataFrame(
            {
                "ts": list(range(n)) + list(range(1000, 1000 + n)),
                "val": [float(i) for i in range(2 * n)],
                "g": ["a"] * n + ["b"] * n,
            }
        )
        out = _grouped_points(
            LFQueryBuilder(df),
            LinePlot(x="ts", y="val", n_points=50, group_by="g"),
            x_range=(0, 50),
        )
        assert set(out) == {"a"}
        assert all(0 <= v <= 50 for v in out["a"]["x"].to_list())

    def test_an_empty_viewport_gives_no_children(self):
        out = _grouped_points(
            LFQueryBuilder(_grouped_frame()),
            LinePlot(x="ts", y="val", n_points=50, group_by="g"),
            x_range=(10_000, 20_000),
        )
        assert out == {}

    @pytest.mark.parametrize(
        "group_by,key", [("g", "null"), (["g", "site"], '[null,"north"]')]
    )
    def test_a_null_group_value_keeps_its_own_child(self, group_by, key):
        # Two string columns fall back to the multi-column key, so both forms
        # are covered.
        df = _grouped_frame().with_columns(
            g=pl.when(pl.col("ts") < 100).then(None).otherwise(pl.col("g")),
            site=pl.lit("north"),
        )
        out = _grouped_points(
            LFQueryBuilder(df),
            LinePlot(x="ts", y="val", n_points=100, group_by=group_by),
        )
        assert key in out
        assert len(out[key]["x"]) > 0

    def test_a_null_integer_group_value_keeps_its_own_child(self):
        df = _grouped_frame().with_columns(
            gi=pl.when(pl.col("ts") < 100).then(None).otherwise(pl.col("gi"))
        )
        out = _grouped_points(
            LFQueryBuilder(df), LinePlot(x="ts", y="val", n_points=100, group_by="gi")
        )
        assert set(out) == {"null", "3", "8"}

    def test_null_x_rows_fall_out(self):
        df = _grouped_frame().with_columns(
            ts=pl.when(pl.col("ts") < 100).then(None).otherwise(pl.col("ts"))
        )
        out = _grouped_points(
            LFQueryBuilder(df), LinePlot(x="ts", y="val", n_points=100, group_by="g")
        )
        assert all(v is not None for u in out.values() for v in u["x"].to_list())

    def test_nan_x_does_not_raise(self):
        df = _grouped_frame().with_columns(
            ts=pl.when(pl.col("ts") == 5)
            .then(float("nan"))
            .otherwise(pl.col("ts").cast(pl.Float64))
        )
        out = _grouped_points(
            LFQueryBuilder(df), LinePlot(x="ts", y="val", n_points=100, group_by="g")
        )
        assert set(out) == {"a", "b"}
        assert not any(math.isnan(v) for u in out.values() for v in u["x"].to_list())

    def test_a_nan_only_group_gets_no_child(self):
        # Every key form must drop a NaN x. "site" is a second string column,
        # so that group_by takes the multi-column key and "g" the packed one.
        df = _grouped_frame().with_columns(
            ts=pl.when(pl.col("g") == "b")
            .then(float("nan"))
            .otherwise(pl.col("ts").cast(pl.Float64)),
            site=pl.lit("north"),
        )
        lf = LFQueryBuilder(df)
        packed = _grouped_points(
            lf, LinePlot(x="ts", y="val", n_points=100, group_by="g")
        )
        multi = _grouped_points(
            lf, LinePlot(x="ts", y="val", n_points=100, group_by=["g", "site"])
        )
        assert set(packed) == {"a"}
        assert set(multi) == {'["a","north"]'}

    def test_a_tenth_of_the_domain_gets_a_tenth_of_the_points(self):
        # The whole point of one shared grid (issue #16). Row-count buckets
        # would spend the full budget on the short series.
        n = 1000
        df = pl.DataFrame(
            {
                "ts": list(range(n // 10)) + list(range(n)),
                "val": [float(i) for i in range(n // 10 + n)],
                "g": ["a"] * (n // 10) + ["b"] * n,
            }
        )
        out = _grouped_points(
            LFQueryBuilder(df), LinePlot(x="ts", y="val", n_points=200, group_by="g")
        )
        assert len(out["b"]["x"]) >= 180
        assert 0.05 < len(out["a"]["x"]) / len(out["b"]["x"]) < 0.2

    @pytest.mark.parametrize(
        "group_cols,expected",
        [
            (("g",), ["__b"]),
            (("gc",), ["__b"]),
            (("gi",), ["gi", "__b"]),
            (("val",), ["val", "__b"]),
            (("g", "site"), ["g", "site", "__b"]),
            (("g", "gi"), ["g", "gi", "__b"]),
        ],
        ids=["str", "cat", "int", "float", "two_str", "str_int"],
    )
    def test_the_key_form(self, group_cols, expected):
        keys = _grouped_bucket_keys(
            group_cols, _grouped_frame().schema, 50, pl.col("ts")
        )
        assert keys[1] == expected


class TestPlanDropsNaNAndNullX:
    """No bucket holds a null or a NaN x, so every plan form drops those rows."""

    @staticmethod
    def _pairs(group_cols=None):
        df = pl.DataFrame(
            {
                "ts": [0.0, 1.0, 2.0, 3.0, float("nan"), None],
                "val": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "gi": [1] * 6,
                "gs": ["a"] * 6,
            },
            schema={
                "ts": pl.Float64,
                "val": pl.Float64,
                "gi": pl.Int64,
                "gs": pl.String,
            },
        )
        run = pairs_plan(
            "ts",
            "val",
            2,
            "u",
            None,
            None,
            (0.0, 3.0),
            df.schema,
            group_cols=group_cols,
        )
        return run(df.lazy())["u"].to_list()[0]

    def test_every_key_form_gives_the_same_pairs(self):
        # The packed key drops both rows through its null key. The ungrouped
        # plan and the multi-column fallback must agree with it.
        expected = [
            {"x_min": 0.0, "y_min": 1.0, "x_max": 1.0, "y_max": 2.0},
            {"x_min": 2.0, "y_min": 3.0, "x_max": 3.0, "y_max": 4.0},
        ]
        assert self._pairs() == expected
        assert self._pairs(("gs",)) == expected
        assert self._pairs(("gi",)) == expected


# ---- lttb (MinMaxLTTB) ------------------------------------------------------


def _lttb_frame(n: int = 10_000) -> pl.DataFrame:
    rng = random.Random(20260904)
    return pl.DataFrame(
        {"ts": list(range(n)), "val": [rng.random() * 100 for _ in range(n)]},
        schema={"ts": pl.Int64, "val": pl.Float64},
    )


def _walk(n: int) -> list[int]:
    """A random walk whose y range is close to the float64 ulp at a ns epoch."""
    rng = random.Random(20260905)
    value = 0
    out = []
    for _ in range(n):
        value += rng.randint(-5, 5)
        out.append(value)
    return out


class TestLineLTTB:
    def test_returns_exactly_n_points(self):
        out = _minmax_points(
            LFQueryBuilder(_lttb_frame()),
            LinePlot(x="ts", y="val", n_points=500, downsample="lttb"),
        )
        assert len(out["x"]) == 500
        assert len(out["y"]) == 500

    def test_passes_the_prefetch_through_when_it_is_small(self):
        df = _lttb_frame(300)
        out = _minmax_points(
            LFQueryBuilder(df),
            LinePlot(x="ts", y="val", n_points=1_000, downsample="lttb"),
        )
        assert len(out["x"]) == 300

    def test_keeps_the_first_and_last_prefetched_point(self):
        df = _lttb_frame()
        lf = LFQueryBuilder(df)
        n_points = 200
        thinned = _minmax_points(
            LFQueryBuilder(df),
            LinePlot(x="ts", y="val", n_points=n_points, downsample="lttb"),
        )
        # The stage-1 prefetch: the same minmax envelope at the 4x budget.
        prefetch = _minmax_points(lf, LinePlot(x="ts", y="val", n_points=4 * n_points))
        assert thinned["x"][0] == prefetch["x"][0]
        assert thinned["x"][-1] == prefetch["x"][-1]
        assert thinned["x"].to_list() == sorted(thinned["x"].to_list())

    def test_resident_matches_the_scan_plan(self, tmp_path):
        df = _gappy_frame()
        path = tmp_path / "lttb.parquet"
        df.write_parquet(path)
        scan_lf = LFQueryBuilder(pl.scan_parquet(path))
        assert scan_lf.is_scan

        trace = LinePlot(x="ts", y="val", n_points=100, downsample="lttb")
        resident = _minmax_points(LFQueryBuilder(df), trace)
        scanned = _minmax_points(scan_lf, trace)

        assert resident["x"].to_list() == scanned["x"].to_list()
        assert resident["y"].to_list() == scanned["y"].to_list()

    def test_viewport_zoom_respects_the_range(self):
        out = _minmax_points(
            LFQueryBuilder(_lttb_frame()),
            LinePlot(x="ts", y="val", n_points=100, downsample="lttb"),
            x_range=(2_000, 3_000),
        )
        assert len(out["x"]) == 100
        assert all(2_000 <= v <= 3_000 for v in out["x"])

    def test_temporal_x_round_trips_its_dtype(self):
        n = 5_000
        t0 = dt.datetime(2023, 1, 1)
        rng = random.Random(11)
        df = pl.DataFrame(
            {
                "ts": pl.Series(
                    "ts",
                    [t0 + dt.timedelta(microseconds=i) for i in range(n)],
                    dtype=pl.Datetime("ns"),
                ),
                "val": [rng.random() for _ in range(n)],
            }
        )
        out = _minmax_points(
            LFQueryBuilder(df),
            LinePlot(x="ts", y="val", n_points=100, downsample="lttb"),
        )
        assert out["x"].dtype == pl.Datetime("ns")
        assert len(out["x"]) == 100
        assert out["x"][0] >= t0

    def test_a_large_y_offset_picks_the_same_points(self):
        # The triangle areas average y, so a y past 2**53 loses precision in
        # float64 and the argmax flips. A nanosecond epoch sits near 1.7e18.
        from flexviz.trace.line import _lttb

        n = 4_000
        y = _walk(n)
        x = list(range(n))
        assert _lttb(x, y, 100)[0] == _lttb(x, [v + 2**60 for v in y], 100)[0]

    def test_temporal_y_picks_the_same_points_as_its_physical(self):
        # A Datetime("ns") y goes through its physical, which is that same
        # large offset.
        n = 4_000
        ints = pl.DataFrame(
            {"ts": list(range(n)), "val": pl.Series(_walk(n), dtype=pl.Int64)}
        )
        stamps = ints.with_columns(
            val=(pl.col("val") + 1_700_000_000_000_000_000).cast(pl.Datetime("ns"))
        )
        trace = LinePlot(x="ts", y="val", n_points=100, downsample="lttb")
        from_ints = _minmax_points(LFQueryBuilder(ints), trace)
        from_stamps = _minmax_points(LFQueryBuilder(stamps), trace)
        assert from_stamps["x"].to_list() == from_ints["x"].to_list()

    def test_temporal_y_round_trips_its_dtype(self):
        n = 5_000
        t0 = dt.datetime(2023, 1, 1)
        df = pl.DataFrame(
            {
                "ts": list(range(n)),
                "val": pl.Series(
                    "val",
                    [t0 + dt.timedelta(milliseconds=(i * 7) % 1_000) for i in range(n)],
                    dtype=pl.Datetime("ms"),
                ),
            }
        )
        out = _minmax_points(
            LFQueryBuilder(df),
            LinePlot(x="ts", y="val", n_points=100, downsample="lttb"),
        )
        assert out["y"].dtype == pl.Datetime("ms")
        assert len(out["y"]) == 100

    def test_grouped_returns_exactly_n_points_per_child(self):
        n_points = 50
        n_rows = 500
        rng = random.Random(7)
        sensors = ["A", "B", "C"]
        df = pl.DataFrame(
            {
                "ts": list(range(n_rows)) * len(sensors),
                "val": [
                    rng.random() + i * 10
                    for i in range(len(sensors))
                    for _ in range(n_rows)
                ],
                "sensor": [s for s in sensors for _ in range(n_rows)],
            }
        )
        results = _aggregate_grouped_line(df, downsample="lttb", n_points=n_points)
        assert {cr.group_value_key for cr in results} == set(sensors)
        for cr in results:
            xs = cr.updates["x"].to_list()
            ys = cr.updates["y"].to_list()
            assert len(xs) == n_points
            assert len(ys) == n_points
            assert xs == sorted(xs)


class TestLTTBFunction:
    """The stage-2 rule on its own, away from any bucket pass."""

    @staticmethod
    def _points(n: int):
        return list(range(n)), [float((i * 37) % 101) for i in range(n)]

    @pytest.mark.parametrize("n_out", [3, 10, 999])
    def test_output_length_is_exact(self, n_out):
        from flexviz.trace.line import _lttb

        x, y = self._points(5_000)
        out_x, out_y = _lttb(x, y, n_out)
        assert len(out_x) == n_out
        assert len(out_y) == n_out
        assert out_x[0] == x[0] and out_x[-1] == x[-1]
        assert out_x == sorted(out_x)

    def test_short_input_passes_through(self):
        from flexviz.trace.line import _lttb

        x, y = self._points(10)
        assert _lttb(x, y, 10) == (x, y)
        assert _lttb(x, y, 50) == (x, y)

    def test_two_points_keeps_the_ends(self):
        from flexviz.trace.line import _lttb

        x, y = self._points(100)
        assert _lttb(x, y, 2) == ([0, 99], [y[0], y[99]])

    def test_a_spike_survives(self):
        from flexviz.trace.line import _lttb

        x = list(range(1_000))
        y = [0.0] * 500 + [1_000.0] + [0.0] * 499
        _, out_y = _lttb(x, y, 100)
        assert 1_000.0 in out_y


# ---- n_points bounds and the add_line x gate --------------------------------


class TestLineNPointsBounds:
    @pytest.mark.parametrize("n_points", [1, 0, -5, 25_001])
    def test_out_of_range_is_rejected(self, n_points):
        with pytest.raises(ValueError, match="n_points must be between 2 and 25000"):
            LinePlot(x="ts", y="val", n_points=n_points)

    @pytest.mark.parametrize("n_points", [2, 25_000])
    def test_bounds_are_inclusive(self, n_points):
        assert LinePlot(x="ts", y="val", n_points=n_points).n_points == n_points

    def test_a_decoded_spec_is_bounded_too(self):
        # The client posts the spec on every update, so decoding is a trust
        # boundary.
        spec = LinePlot(x="ts", y="val", n_points=1000).to_trace_spec()
        spec.params["n_points"] = 25_001
        with pytest.raises(ValueError, match="n_points must be between 2 and 25000"):
            LinePlot.from_trace_spec(spec)


class TestBucketsByXWidth:
    """The property that tells the engine which lines carry the x contract."""

    @pytest.mark.parametrize("downsample", ["minmax", "lttb", "fpcs"])
    def test_ungrouped_x_width_lines(self, downsample):
        assert LinePlot(x="ts", y="val", downsample=downsample).buckets_by_x_width

    def test_nth_buckets_by_row_count(self):
        assert not LinePlot(x="ts", y="val", downsample="nth").buckets_by_x_width

    def test_a_grouped_line_buckets_by_x_width_too(self):
        # One grid across the groups, so a series covering a tenth of the x
        # domain gets a tenth of the points.
        assert LinePlot(x="ts", y="val", group_by="sensor").buckets_by_x_width

    def test_a_grouped_nth_line_buckets_by_row_count(self):
        assert not LinePlot(
            x="ts", y="val", group_by="sensor", downsample="nth"
        ).buckets_by_x_width


def _run_line(lf: LFQueryBuilder, trace: LinePlot) -> list:
    """Drive one line trace through the engine, the way a request does."""
    engine = FlexEngine(backend_lf=lf, scalable_traces={trace.uid: trace})
    infos = [TraceInfo(uid=trace.uid, axes=trace._axes, trace_type="line")]
    return engine.process(InteractionEvent(type="init", force_update=True), infos)


def _both_sources(df: pl.DataFrame, path) -> list[LFQueryBuilder]:
    """The same frame as a resident source and as a file source."""
    df.write_parquet(path)
    return [LFQueryBuilder(df), LFQueryBuilder(pl.scan_parquet(path))]


class TestCheckSchema:
    """The dtype half of the line contract. Reads the schema, never collects."""

    @staticmethod
    def _schema(x_dtype, y_dtype=pl.Float64) -> pl.Schema:
        return pl.Schema({"ts": x_dtype, "val": y_dtype})

    @pytest.mark.parametrize("dtype", [pl.Int64, pl.Float32, pl.Datetime("us")])
    def test_a_bucketable_x_passes(self, dtype):
        LinePlot(x="ts", y="val").check_schema(self._schema(dtype))

    @pytest.mark.parametrize("x,y", [("nope", "val"), ("ts", "nope")])
    def test_a_missing_column_is_rejected(self, x, y):
        with pytest.raises(ValueError, match="not in schema"):
            LinePlot(x=x, y=y).check_schema(self._schema(pl.Int64))

    @pytest.mark.parametrize("downsample", ["minmax", "lttb", "fpcs"])
    def test_string_x_is_rejected(self, downsample):
        with pytest.raises(ValueError, match="numeric or a temporal"):
            LinePlot(x="ts", y="val", downsample=downsample).check_schema(
                self._schema(pl.String)
            )

    @pytest.mark.parametrize("dtype", [pl.Int128, pl.Decimal(10, 2)])
    def test_a_wider_numeric_x_is_rejected(self, dtype):
        # Wider than the kernel's i64 edge search.
        with pytest.raises(ValueError, match="64-bit-or-smaller"):
            LinePlot(x="ts", y="val").check_schema(self._schema(dtype))

    def test_a_grouped_line_gates_its_x_too(self):
        # The grid is arithmetic on x, grouped as well.
        with pytest.raises(ValueError, match="numeric or a temporal"):
            LinePlot(x="ts", y="val", group_by="ts").check_schema(
                self._schema(pl.String)
            )

    def test_an_nth_line_takes_any_x(self):
        # A stride carries no grid, so it needs no bucketable x.
        LinePlot(x="ts", y="val", downsample="nth").check_schema(
            self._schema(pl.String)
        )

    def test_lttb_rejects_a_non_numeric_y(self):
        with pytest.raises(ValueError, match="numeric, temporal or Boolean"):
            LinePlot(x="ts", y="val", downsample="lttb").check_schema(
                self._schema(pl.Int64, y_dtype=pl.String)
            )

    @pytest.mark.parametrize("downsample", ["minmax", "lttb", "fpcs"])
    @pytest.mark.parametrize(
        "dtype", [pl.Decimal(10, 2), pl.Int128, pl.Categorical(), pl.Enum(["a"])]
    )
    def test_a_y_the_pair_kernel_cannot_take_is_rejected(self, dtype, downsample):
        with pytest.raises(ValueError, match="must not be a Decimal"):
            LinePlot(x="ts", y="val", downsample=downsample).check_schema(
                self._schema(pl.Int64, y_dtype=dtype)
            )

    def test_an_nth_line_takes_any_y(self):
        # A stride gathers rows and compares nothing.
        LinePlot(x="ts", y="val", downsample="nth").check_schema(
            self._schema(pl.Int64, y_dtype=pl.Decimal(10, 2))
        )

    @pytest.mark.parametrize("downsample", ["minmax", "lttb"])
    def test_a_decimal_y_is_rejected_on_both_source_kinds(self, tmp_path, downsample):
        # The kernel panics on a Decimal y and the scan plan accepts it, so
        # without the gate residency would decide the outcome.
        df = pl.DataFrame({"ts": list(range(100))}).with_columns(
            val=pl.col("ts").cast(pl.Decimal(10, 2))
        )
        for lf in _both_sources(df, tmp_path / "decimal.parquet"):
            with pytest.raises(ValueError, match="must not be a Decimal"):
                _run_line(lf, LinePlot(x="ts", y="val", n_points=10))

    @pytest.mark.parametrize(
        "dtype",
        [
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
            pl.Float32,
            pl.Float64,
            pl.Boolean,
            pl.Date,
            pl.Datetime("us"),
            pl.Datetime("ns"),
            pl.Duration("ms"),
            pl.Time,
            pl.String,
        ],
    )
    def test_an_accepted_y_runs_on_both_source_kinds(self, tmp_path, dtype):
        df = pl.DataFrame({"ts": list(range(100))}).with_columns(
            val=pl.col("ts").cast(dtype)
        )
        for lf in _both_sources(df, tmp_path / "y.parquet"):
            deltas = _run_line(lf, LinePlot(x="ts", y="val", n_points=10))
            assert len(deltas[0].updates["x"]) > 0


class TestLineDownsampleValidation:
    """An unknown strategy used to fall through to ``nth`` without a word."""

    def test_an_unknown_strategy_is_rejected(self):
        fig = Figure(pl.DataFrame({"ts": [1, 2], "val": [1.0, 2.0]}))
        with pytest.raises(ValueError, match="downsample must be one of"):
            fig.add_line(x="ts", y="val", downsample="bogus")

    def test_a_decoded_spec_is_validated_too(self):
        # The client posts the spec on every update, so decoding is a trust
        # boundary.
        spec = LinePlot(x="ts", y="val").to_trace_spec()
        spec.params["downsample"] = "bogus"
        with pytest.raises(ValueError, match="downsample must be one of"):
            LinePlot.from_trace_spec(spec)


class TestStageTwoDropsNaN:
    """A whole NaN bucket still emits its NaN extremum from stage 1."""

    @pytest.mark.parametrize("downsample", ["lttb", "fpcs"])
    def test_nan_y_never_reaches_the_output(self, downsample):
        n = 2_000
        vals = [float((i * 37) % 101) for i in range(n)]
        for i in range(600, 800):  # wide enough to fill whole buckets
            vals[i] = float("nan")
        df = pl.DataFrame(
            {"ts": list(range(n)), "val": vals},
            schema={"ts": pl.Int64, "val": pl.Float64},
        )
        out = _minmax_points(
            LFQueryBuilder(df),
            LinePlot(x="ts", y="val", n_points=100, downsample=downsample),
        )
        xs, ys = out["x"].to_list(), out["y"].to_list()
        assert ys and not any(math.isnan(v) for v in ys)
        assert len(set(xs)) == len(xs)
        assert xs == sorted(xs)

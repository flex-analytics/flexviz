"""Unit tests for FlexEngine.process across all event types."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from flexviz.engine import FlexEngine, TraceInfo
from flexviz.events import InteractionEvent
from flexviz.LF import LFQueryBuilder
from flexviz.spec import ClauseFilter, SelectionPredicate, SelectionState
from flexviz.trace.bar import BarPlot
from flexviz.trace.box import BoxPlot
from flexviz.trace.hist import Histogram
from flexviz.trace.line import LinePlot

# ---- helpers ---------------------------------------------------------------


def _build_engine(
    df: pl.DataFrame,
) -> tuple[FlexEngine, list[TraceInfo], dict[str, LinePlot | Histogram]]:
    """Build an engine with one line trace and one histogram on the same data."""
    lf = LFQueryBuilder(df)
    line = LinePlot(x="ts", y="val", n_points=100)
    hist = Histogram(x="val", bins=20)
    traces = {line.uid: line, hist.uid: hist}
    engine = FlexEngine(backend_lf=lf, scalable_traces=traces)
    infos = [
        TraceInfo(uid=line.uid, axes=("x", "y"), trace_type="line"),
        TraceInfo(uid=hist.uid, axes=("x", "y"), trace_type="histogram"),
    ]
    return engine, infos, traces


# ---- init event ------------------------------------------------------------


class TestEngineInit:
    def test_init_returns_deltas_for_all_traces(self, small_df: pl.DataFrame):
        engine, infos, traces = _build_engine(small_df)
        event = InteractionEvent(type="init", force_update=True)
        deltas = engine.process(event, infos)
        assert len(deltas) == 2
        uids = {d.uid for d in deltas}
        assert uids == set(traces.keys())

    def test_init_deltas_have_data(self, small_df: pl.DataFrame):
        engine, infos, _ = _build_engine(small_df)
        event = InteractionEvent(type="init", force_update=True)
        deltas = engine.process(event, infos)
        for delta in deltas:
            assert len(delta.updates.get("x", [])) > 0
            assert len(delta.updates.get("y", [])) > 0


# ---- viewport event --------------------------------------------------------


class TestEngineViewport:
    def test_viewport_matching_axes(self, small_df: pl.DataFrame):
        engine, infos, _ = _build_engine(small_df)
        event = InteractionEvent(
            type="viewport",
            axis_ranges={"x": [200, 400]},
        )
        deltas = engine.process(event, infos)
        assert len(deltas) > 0

    def test_viewport_no_matching_axes(self, small_df: pl.DataFrame):
        engine, infos, _ = _build_engine(small_df)
        event = InteractionEvent(
            type="viewport",
            axis_ranges={"x99": [200, 400]},
        )
        deltas = engine.process(event, infos)
        assert len(deltas) == 0

    def test_viewport_fewer_points_than_init(self, small_df: pl.DataFrame):
        engine, infos, _ = _build_engine(small_df)

        init_event = InteractionEvent(type="init", force_update=True)
        engine.process(init_event, infos)
        vp_event = InteractionEvent(
            type="viewport",
            axis_ranges={"x": [200, 400]},
        )
        vp_deltas = engine.process(vp_event, infos)
        assert len(vp_deltas) > 0

    def test_viewport_shared_x_dual_y(self, small_df: pl.DataFrame):
        """axes=("x","y2") trace must re-aggregate on an x-only viewport event."""
        lf = LFQueryBuilder(small_df)
        line = LinePlot(x="ts", y="val", n_points=100, axes=("x", "y2"))
        traces = {line.uid: line}
        engine = FlexEngine(backend_lf=lf, scalable_traces=traces)
        infos = [
            TraceInfo(uid=line.uid, axes=("x", "y2"), trace_type="line"),
        ]
        event = InteractionEvent(
            type="viewport",
            axis_ranges={"x": [200, 400]},
        )
        deltas = engine.process(event, infos)
        assert len(deltas) == 1
        assert deltas[0].uid == line.uid
        assert all(200 <= v <= 400 for v in deltas[0].updates["x"])


# ---- reset event -----------------------------------------------------------


class TestEngineReset:
    def test_reset_returns_all_traces(self, small_df: pl.DataFrame):
        engine, infos, traces = _build_engine(small_df)
        event = InteractionEvent(type="reset", force_update=True)
        deltas = engine.process(event, infos)
        assert len(deltas) == 2
        assert {d.uid for d in deltas} == set(traces.keys())


# ---- deselect event --------------------------------------------------------


class TestEngineDeselect:
    def test_deselect_returns_all_traces(self, small_df: pl.DataFrame):
        engine, infos, traces = _build_engine(small_df)
        event = InteractionEvent(type="deselect", force_update=True)
        deltas = engine.process(event, infos)
        assert len(deltas) == 2


# ---- selection event (cross-filter) ----------------------------------------


class TestEngineSelection:
    def test_source_figure_excluded(self, small_df: pl.DataFrame):
        """Traces belonging to the source figure must not be re-aggregated."""
        lf = LFQueryBuilder(small_df)
        line_a = LinePlot(x="ts", y="val", n_points=100)
        line_b = LinePlot(x="ts", y="val", n_points=100)
        traces = {line_a.uid: line_a, line_b.uid: line_b}
        engine = FlexEngine(backend_lf=lf, scalable_traces=traces)

        infos = [
            TraceInfo(
                uid=line_a.uid,
                axes=("x", "y"),
                trace_type="line",
                figure_uid="fig_a",
            ),
            TraceInfo(
                uid=line_b.uid,
                axes=("x", "y"),
                trace_type="line",
                figure_uid="fig_b",
            ),
        ]

        event = InteractionEvent(
            type="selection",
            force_update=True,
            selections=[
                SelectionState(
                    source_figure_uid="fig_a",
                    predicates=[
                        SelectionPredicate(
                            clauses=[ClauseFilter(column="ts", range=(100, 500))]
                        )
                    ],
                ),
            ],
        )
        deltas = engine.process(event, infos)
        delta_uids = {d.uid for d in deltas}
        assert line_a.uid not in delta_uids, "source figure trace should be excluded"
        assert line_b.uid in delta_uids, "target figure trace should receive a delta"

    def test_selection_filters_data(self, small_df: pl.DataFrame):
        """Selection on fig_a should filter data returned for fig_b."""
        lf = LFQueryBuilder(small_df)
        line_a = LinePlot(x="ts", y="val", n_points=1000)
        line_b = LinePlot(x="ts", y="val", n_points=1000)
        traces = {line_a.uid: line_a, line_b.uid: line_b}
        engine = FlexEngine(backend_lf=lf, scalable_traces=traces)

        infos = [
            TraceInfo(
                uid=line_a.uid,
                axes=("x", "y"),
                trace_type="line",
                figure_uid="fig_a",
            ),
            TraceInfo(
                uid=line_b.uid,
                axes=("x", "y"),
                trace_type="line",
                figure_uid="fig_b",
            ),
        ]

        event = InteractionEvent(
            type="selection",
            force_update=True,
            selections=[
                SelectionState(
                    source_figure_uid="fig_a",
                    predicates=[
                        SelectionPredicate(
                            clauses=[ClauseFilter(column="ts", range=(100, 300))]
                        )
                    ],
                ),
            ],
        )
        deltas = engine.process(event, infos)
        b_delta = next(d for d in deltas if d.uid == line_b.uid)
        assert all(100 <= v <= 300 for v in b_delta.updates["x"])

    def test_y_range_selection(self, small_df: pl.DataFrame):
        """y-range-only selection must filter target data by the y column."""
        lf = LFQueryBuilder(small_df)
        line_a = LinePlot(x="ts", y="val", n_points=1000)
        line_b = LinePlot(x="ts", y="val", n_points=1000)
        traces = {line_a.uid: line_a, line_b.uid: line_b}
        engine = FlexEngine(backend_lf=lf, scalable_traces=traces)

        infos = [
            TraceInfo(
                uid=line_a.uid,
                axes=("x", "y"),
                trace_type="line",
                figure_uid="fig_a",
            ),
            TraceInfo(
                uid=line_b.uid,
                axes=("x", "y"),
                trace_type="line",
                figure_uid="fig_b",
            ),
        ]

        event = InteractionEvent(
            type="selection",
            force_update=True,
            selections=[
                SelectionState(
                    source_figure_uid="fig_a",
                    predicates=[
                        SelectionPredicate(
                            clauses=[ClauseFilter(column="val", range=(200.0, 500.0))]
                        )
                    ],
                ),
            ],
        )
        deltas = engine.process(event, infos)
        b_delta = next(d for d in deltas if d.uid == line_b.uid)
        assert all(200 <= v <= 500 for v in b_delta.updates["y"])

    def test_combined_xy_selection(self, small_df: pl.DataFrame):
        """Combined x+y selection must satisfy both bounds on target data."""
        lf = LFQueryBuilder(small_df)
        line_a = LinePlot(x="ts", y="val", n_points=1000)
        line_b = LinePlot(x="ts", y="val", n_points=1000)
        traces = {line_a.uid: line_a, line_b.uid: line_b}
        engine = FlexEngine(backend_lf=lf, scalable_traces=traces)

        infos = [
            TraceInfo(
                uid=line_a.uid,
                axes=("x", "y"),
                trace_type="line",
                figure_uid="fig_a",
            ),
            TraceInfo(
                uid=line_b.uid,
                axes=("x", "y"),
                trace_type="line",
                figure_uid="fig_b",
            ),
        ]

        event = InteractionEvent(
            type="selection",
            force_update=True,
            selections=[
                SelectionState(
                    source_figure_uid="fig_a",
                    predicates=[
                        SelectionPredicate(
                            clauses=[
                                ClauseFilter(column="ts", range=(100, 400)),
                                ClauseFilter(column="val", range=(100.0, 400.0)),
                            ]
                        )
                    ],
                ),
            ],
        )
        deltas = engine.process(event, infos)
        b_delta = next(d for d in deltas if d.uid == line_b.uid)
        assert all(100 <= v <= 400 for v in b_delta.updates["x"])
        assert all(100 <= v <= 400 for v in b_delta.updates["y"])

    def test_histogram_source_to_line_target(self, small_df: pl.DataFrame):
        """Histogram source with x-selection must cross-filter a line target."""
        lf = LFQueryBuilder(small_df)
        hist = Histogram(x="val", bins=20)
        line = LinePlot(x="ts", y="val", n_points=1000)
        traces = {hist.uid: hist, line.uid: line}
        engine = FlexEngine(backend_lf=lf, scalable_traces=traces)

        infos = [
            TraceInfo(
                uid=hist.uid,
                axes=("x", "y"),
                trace_type="histogram",
                figure_uid="fig_hist",
            ),
            TraceInfo(
                uid=line.uid,
                axes=("x", "y"),
                trace_type="line",
                figure_uid="fig_line",
            ),
        ]

        event = InteractionEvent(
            type="selection",
            force_update=True,
            selections=[
                SelectionState(
                    source_figure_uid="fig_hist",
                    predicates=[
                        SelectionPredicate(
                            clauses=[ClauseFilter(column="val", range=(200, 500))]
                        )
                    ],
                ),
            ],
        )
        deltas = engine.process(event, infos)
        delta_uids = {d.uid for d in deltas}
        assert hist.uid not in delta_uids
        assert line.uid in delta_uids
        line_delta = next(d for d in deltas if d.uid == line.uid)
        assert all(200 <= v <= 500 for v in line_delta.updates["y"])


class TestEngineOverlay:
    def test_overlay_selection_returns_fg_for_target_only(self):
        df = pl.DataFrame({"ts": list(range(10)), "val": [float(i) for i in range(10)]})
        lf = LFQueryBuilder(df)
        line_a = LinePlot(x="ts", y="val", n_points=1000)
        line_b = LinePlot(x="ts", y="val", n_points=1000)
        engine = FlexEngine(
            backend_lf=lf,
            scalable_traces={line_a.uid: line_a, line_b.uid: line_b},
        )
        infos = [
            TraceInfo(
                uid=line_a.uid,
                axes=("x", "y"),
                trace_type="line",
                figure_uid="fig_a",
            ),
            TraceInfo(
                uid=line_b.uid,
                axes=("x", "y"),
                trace_type="line",
                figure_uid="fig_b",
            ),
        ]

        event = InteractionEvent(
            type="selection",
            force_update=True,
            selections=[
                SelectionState(
                    source_figure_uid="fig_a",
                    predicates=[
                        SelectionPredicate(
                            clauses=[ClauseFilter(column="ts", range=(2, 4))]
                        )
                    ],
                )
            ],
        )
        deltas = engine.process(event, infos, cross_filter_mode="overlay")
        target_deltas = [d for d in deltas if d.uid == line_b.uid]
        assert [d.uid for d in deltas].count(line_a.uid) == 0
        assert {d.layer for d in target_deltas} == {"fg"}

        fg = next(d for d in target_deltas if d.layer == "fg")
        assert all(2 <= v <= 4 for v in fg.updates["x"])

    def test_overlay_viewport_with_active_selection_returns_bg_and_fg(self):
        df = pl.DataFrame({"ts": list(range(10)), "val": [float(i) for i in range(10)]})
        lf = LFQueryBuilder(df)
        line_a = LinePlot(x="ts", y="val", n_points=1000)
        line_b = LinePlot(x="ts", y="val", n_points=1000)
        engine = FlexEngine(
            backend_lf=lf,
            scalable_traces={line_a.uid: line_a, line_b.uid: line_b},
        )
        infos = [
            TraceInfo(
                uid=line_a.uid,
                axes=("x", "y"),
                trace_type="line",
                figure_uid="fig_a",
            ),
            TraceInfo(
                uid=line_b.uid,
                axes=("x", "y"),
                trace_type="line",
                figure_uid="fig_b",
            ),
        ]

        event = InteractionEvent(
            type="viewport",
            axis_ranges={"x": (0, 6)},
            selections=[
                SelectionState(
                    source_figure_uid="fig_a",
                    predicates=[
                        SelectionPredicate(
                            clauses=[ClauseFilter(column="ts", range=(2, 4))]
                        )
                    ],
                )
            ],
            figure_uid="fig_b",
        )
        deltas = engine.process(event, infos, cross_filter_mode="overlay")
        target_deltas = [d for d in deltas if d.uid == line_b.uid]
        assert {d.layer for d in target_deltas} == {"bg", "fg"}
        bg = next(d for d in target_deltas if d.layer == "bg")  # noqa: F841
        fg = next(d for d in target_deltas if d.layer == "fg")
        assert all(0 <= v <= 6 for v in bg.updates["x"])
        assert all(2 <= v <= 4 for v in fg.updates["x"])

    def test_overlay_reset_returns_bg_only(self):
        df = pl.DataFrame({"ts": list(range(10)), "val": [float(i) for i in range(10)]})
        lf = LFQueryBuilder(df)
        line = LinePlot(x="ts", y="val", n_points=1000)
        engine = FlexEngine(backend_lf=lf, scalable_traces={line.uid: line})
        infos = [TraceInfo(uid=line.uid, axes=("x", "y"), trace_type="line")]

        event = InteractionEvent(type="reset", force_update=True)
        deltas = engine.process(event, infos, cross_filter_mode="overlay")
        assert {d.layer for d in deltas} == {"bg"}


class TestBarPlotCrossFilter:
    """BarPlot as a cross-filter source using the categories field."""

    @pytest.fixture()
    def cat_lf(self) -> LFQueryBuilder:
        df = pl.DataFrame(
            {
                "cat": ["A", "A", "B", "B", "C", "C"],
                "ts": [1, 2, 3, 4, 5, 6],
                "val": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            }
        )
        return LFQueryBuilder(df)

    def _make_event(self, figure_uid: str, categories: list[str]) -> InteractionEvent:
        return InteractionEvent(
            type="selection",
            force_update=True,
            selections=[
                SelectionState(
                    source_figure_uid=figure_uid,
                    predicates=[
                        SelectionPredicate(
                            clauses=[ClauseFilter(column="cat", values=categories)]
                        )
                    ],
                )
            ],
        )

    def test_bar_to_line_cross_filter(self, cat_lf: LFQueryBuilder):
        """Selecting bars on fig_bar should filter the line on fig_line to matching rows."""
        bar = BarPlot(labels="cat")
        line = LinePlot(x="ts", y="val", n_points=1000)
        engine = FlexEngine(
            backend_lf=cat_lf,
            scalable_traces={bar.uid: bar, line.uid: line},
        )
        infos = [
            TraceInfo(
                uid=bar.uid, axes=("x", "y"), trace_type="bar", figure_uid="fig_bar"
            ),
            TraceInfo(
                uid=line.uid, axes=("x", "y"), trace_type="line", figure_uid="fig_line"
            ),
        ]

        event = self._make_event("fig_bar", ["A"])
        deltas = engine.process(event, infos)

        delta_uids = {d.uid for d in deltas}
        assert bar.uid not in delta_uids, "Source figure trace must not be re-processed"
        assert line.uid in delta_uids

        line_delta = next(d for d in deltas if d.uid == line.uid)
        assert line_delta.updates["y"] == [10.0, 20.0], "Only cat=A rows (val 10, 20)"

    def test_bar_to_histogram_cross_filter(self, cat_lf: LFQueryBuilder):
        """Selecting bars should filter a Histogram on another figure."""
        bar = BarPlot(labels="cat")
        hist = Histogram(x="val", bins=10)
        engine = FlexEngine(
            backend_lf=cat_lf,
            scalable_traces={bar.uid: bar, hist.uid: hist},
        )
        infos = [
            TraceInfo(
                uid=bar.uid, axes=("x", "y"), trace_type="bar", figure_uid="fig_bar"
            ),
            TraceInfo(
                uid=hist.uid,
                axes=("x", "y"),
                trace_type="histogram",
                figure_uid="fig_hist",
            ),
        ]

        event = self._make_event("fig_bar", ["B", "C"])
        deltas = engine.process(event, infos)

        hist_delta = next(d for d in deltas if d.uid == hist.uid)
        # Only 4 rows remain (B+C), so total count across all bins must be 4.
        total_count = sum(hist_delta.updates["y"])
        assert total_count == 4

    def test_bar_source_not_reprocessed(self, cat_lf: LFQueryBuilder):
        """Source figure's BarPlot trace must not appear in the response deltas."""
        bar = BarPlot(labels="cat")
        line = LinePlot(x="ts", y="val", n_points=1000)
        engine = FlexEngine(
            backend_lf=cat_lf,
            scalable_traces={bar.uid: bar, line.uid: line},
        )
        infos = [
            TraceInfo(
                uid=bar.uid, axes=("x", "y"), trace_type="bar", figure_uid="fig_bar"
            ),
            TraceInfo(
                uid=line.uid, axes=("x", "y"), trace_type="line", figure_uid="fig_line"
            ),
        ]

        event = self._make_event("fig_bar", ["A", "B"])
        deltas = engine.process(event, infos)
        assert all(d.uid != bar.uid for d in deltas)

    def test_bar_as_cross_filter_target(self, cat_lf: LFQueryBuilder):
        """A BarPlot on fig_bar should receive a cross-filter from a LinePlot selection."""
        line = LinePlot(x="ts", y="val", n_points=1000)
        bar = BarPlot(labels="cat")
        engine = FlexEngine(
            backend_lf=cat_lf,
            scalable_traces={line.uid: line, bar.uid: bar},
        )
        infos = [
            TraceInfo(
                uid=line.uid, axes=("x", "y"), trace_type="line", figure_uid="fig_line"
            ),
            TraceInfo(
                uid=bar.uid, axes=("x", "y"), trace_type="bar", figure_uid="fig_bar"
            ),
        ]

        # Select ts range 1-2 on the line (only cat=A rows).
        event = InteractionEvent(
            type="selection",
            force_update=True,
            selections=[
                SelectionState(
                    source_figure_uid="fig_line",
                    predicates=[
                        SelectionPredicate(
                            clauses=[ClauseFilter(column="ts", range=(1, 2))]
                        )
                    ],
                )
            ],
        )
        deltas = engine.process(event, infos)

        bar_delta = next(d for d in deltas if d.uid == bar.uid)
        # After filtering to ts∈[1,2], only cat=A rows remain → bar has one label "A".
        assert bar_delta.updates["x"] == ["A"]

    def test_bar_deselect_clears_filter(self, cat_lf: LFQueryBuilder):
        """A deselect event must return unfiltered aggregation for all traces."""
        bar = BarPlot(labels="cat")
        line = LinePlot(x="ts", y="val", n_points=1000)
        engine = FlexEngine(
            backend_lf=cat_lf,
            scalable_traces={bar.uid: bar, line.uid: line},
        )
        infos = [
            TraceInfo(
                uid=bar.uid, axes=("x", "y"), trace_type="bar", figure_uid="fig_bar"
            ),
            TraceInfo(
                uid=line.uid, axes=("x", "y"), trace_type="line", figure_uid="fig_line"
            ),
        ]

        event = InteractionEvent(type="deselect", force_update=True, selections=[])
        deltas = engine.process(event, infos)

        line_delta = next(d for d in deltas if d.uid == line.uid)
        assert len(line_delta.updates["y"]) == 6, "All 6 rows returned after deselect"

    def test_grouped_bar_cross_filter(self, cat_lf: LFQueryBuilder):
        """group_by does not change which column the is_in filter targets."""
        bar = BarPlot(labels="cat", group_by="ts")
        line = LinePlot(x="ts", y="val", n_points=1000)
        engine = FlexEngine(
            backend_lf=cat_lf,
            scalable_traces={bar.uid: bar, line.uid: line},
        )
        infos = [
            TraceInfo(
                uid=bar.uid, axes=("x", "y"), trace_type="bar", figure_uid="fig_bar"
            ),
            TraceInfo(
                uid=line.uid, axes=("x", "y"), trace_type="line", figure_uid="fig_line"
            ),
        ]

        event = self._make_event("fig_bar", ["C"])
        deltas = engine.process(event, infos)

        line_delta = next(d for d in deltas if d.uid == line.uid)
        assert line_delta.updates["y"] == [50.0, 60.0], "Only cat=C rows"

    def test_overlay_grouped_target_selection_returns_fg_parent_delta(self):
        df = pl.DataFrame(
            {
                "ts": [0, 1, 2, 3, 4, 5],
                "label": ["X", "X", "Y", "Y", "X", "Y"],
                "region": ["N", "S", "N", "S", "N", "S"],
                "val": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            }
        )
        lf = LFQueryBuilder(df)
        line = LinePlot(x="ts", y="val", n_points=100)
        from flexviz.trace.bar import BarPlot

        bar = BarPlot(labels="label", values="val", agg="sum", group_by="region")
        engine = FlexEngine(
            backend_lf=lf,
            scalable_traces={line.uid: line, bar.uid: bar},
        )
        infos = [
            TraceInfo(
                uid=line.uid,
                axes=("x", "y"),
                trace_type="line",
                figure_uid="fig_src",
            ),
            TraceInfo(
                uid=bar.uid,
                axes=("x", "y"),
                trace_type="bar",
                figure_uid="fig_tgt",
            ),
        ]

        event = InteractionEvent(
            type="selection",
            force_update=True,
            selections=[
                SelectionState(
                    source_figure_uid="fig_src",
                    predicates=[
                        SelectionPredicate(
                            clauses=[ClauseFilter(column="ts", range=(0, 2))]
                        )
                    ],
                )
            ],
        )
        deltas = engine.process(event, infos, cross_filter_mode="overlay")
        bar_deltas = [d for d in deltas if d.uid == bar.uid]
        assert {d.layer for d in bar_deltas} == {"fg"}
        assert all(d.group_results is not None for d in bar_deltas)

    def test_overlay_init_returns_bg_only(self):
        df = pl.DataFrame({"ts": list(range(10)), "val": [float(i) for i in range(10)]})
        lf = LFQueryBuilder(df)
        line = LinePlot(x="ts", y="val", n_points=1000)
        engine = FlexEngine(backend_lf=lf, scalable_traces={line.uid: line})
        infos = [TraceInfo(uid=line.uid, axes=("x", "y"), trace_type="line")]

        event = InteractionEvent(type="init", force_update=True)
        deltas = engine.process(event, infos, cross_filter_mode="overlay")
        assert all(d.layer == "bg" for d in deltas)
        assert not any(d.layer == "fg" for d in deltas)

    def test_overlay_deselect_returns_bg_only(self):
        df = pl.DataFrame({"ts": list(range(10)), "val": [float(i) for i in range(10)]})
        lf = LFQueryBuilder(df)
        line = LinePlot(x="ts", y="val", n_points=1000)
        engine = FlexEngine(backend_lf=lf, scalable_traces={line.uid: line})
        infos = [TraceInfo(uid=line.uid, axes=("x", "y"), trace_type="line")]

        event = InteractionEvent(type="deselect", force_update=True, selections=[])
        deltas = engine.process(event, infos, cross_filter_mode="overlay")
        assert all(d.layer == "bg" for d in deltas)
        assert not any(d.layer == "fg" for d in deltas)

    def test_overlay_viewport_no_sel_returns_bg_only(self):
        df = pl.DataFrame({"ts": list(range(10)), "val": [float(i) for i in range(10)]})
        lf = LFQueryBuilder(df)
        line = LinePlot(x="ts", y="val", n_points=1000)
        engine = FlexEngine(backend_lf=lf, scalable_traces={line.uid: line})
        infos = [TraceInfo(uid=line.uid, axes=("x", "y"), trace_type="line")]

        event = InteractionEvent(
            type="viewport", axis_ranges={"x": (2, 7)}, selections=[]
        )
        deltas = engine.process(event, infos, cross_filter_mode="overlay")
        assert all(d.layer == "bg" for d in deltas)
        assert not any(d.layer == "fg" for d in deltas)

    def test_update_mode_selection_no_layer(self):
        df = pl.DataFrame({"ts": list(range(10)), "val": [float(i) for i in range(10)]})
        lf = LFQueryBuilder(df)
        line_a = LinePlot(x="ts", y="val", n_points=1000)
        line_b = LinePlot(x="ts", y="val", n_points=1000)
        engine = FlexEngine(
            backend_lf=lf,
            scalable_traces={line_a.uid: line_a, line_b.uid: line_b},
        )
        infos = [
            TraceInfo(
                uid=line_a.uid, axes=("x", "y"), trace_type="line", figure_uid="fig_a"
            ),
            TraceInfo(
                uid=line_b.uid, axes=("x", "y"), trace_type="line", figure_uid="fig_b"
            ),
        ]

        event = InteractionEvent(
            type="selection",
            force_update=True,
            selections=[
                SelectionState(
                    source_figure_uid="fig_a",
                    predicates=[
                        SelectionPredicate(
                            clauses=[ClauseFilter(column="ts", range=(2, 4))]
                        )
                    ],
                )
            ],
        )
        deltas = engine.process(event, infos, cross_filter_mode="update")
        assert all(d.layer is None for d in deltas)


class TestCategoriesColumnCrossFilter:
    """Integration tests: categories_column routes to the correct path column."""

    @pytest.fixture()
    def geo_df(self) -> LFQueryBuilder:
        df = pl.DataFrame(
            {
                "continent": ["Europe", "Europe", "Asia", "Asia", "Europe"],
                "country": ["Germany", "France", "Japan", "China", "Germany"],
                "population": [83.0, 67.0, 125.0, 1400.0, 83.0],
            }
        )
        return LFQueryBuilder(df)

    def _make_click_event(
        self,
        figure_uid: str,
        clauses: list[ClauseFilter] | None = None,
    ) -> InteractionEvent:
        predicates = [SelectionPredicate(clauses=clauses)] if clauses else []
        return InteractionEvent(
            type="selection",
            force_update=True,
            selections=[
                SelectionState(
                    source_figure_uid=figure_uid,
                    predicates=predicates,
                )
            ],
        )

    def test_treemap_continent_click_filters_bar(self, geo_df: LFQueryBuilder):
        """Clicking 'Europe' on a treemap cross-filters a bar in the same dashboard."""
        from flexviz.trace.treemap import TreeMap
        from flexviz.trace.bar import BarPlot

        treemap = TreeMap(path=["continent", "country"], values="population")
        bar = BarPlot(labels="country", values="population")
        engine = FlexEngine(
            backend_lf=geo_df,
            scalable_traces={treemap.uid: treemap, bar.uid: bar},
        )
        infos = [
            TraceInfo(
                uid=treemap.uid, axes=None, trace_type="treemap", figure_uid="fig_tree"
            ),
            TraceInfo(
                uid=bar.uid, axes=("x", "y"), trace_type="bar", figure_uid="fig_bar"
            ),
        ]
        event = self._make_click_event(
            figure_uid="fig_tree",
            clauses=[ClauseFilter(column="continent", values=["Europe"])],
        )
        deltas = engine.process(event, infos)
        # fig_bar should be recomputed with continent=Europe filter
        bar_delta = next((d for d in deltas if d.uid == bar.uid), None)
        assert bar_delta is not None
        # Only European countries should appear
        countries = set(bar_delta.updates.get("x", []))
        assert "China" not in countries
        assert "Japan" not in countries
        assert "Germany" in countries or "France" in countries

    def test_treemap_country_click_filters_bar(self, geo_df: LFQueryBuilder):
        """Clicking 'Germany' (leaf level) cross-filters correctly via categories_column='country'."""
        from flexviz.trace.treemap import TreeMap
        from flexviz.trace.bar import BarPlot

        treemap = TreeMap(path=["continent", "country"], values="population")
        bar = BarPlot(labels="continent", values="population")
        engine = FlexEngine(
            backend_lf=geo_df,
            scalable_traces={treemap.uid: treemap, bar.uid: bar},
        )
        infos = [
            TraceInfo(
                uid=treemap.uid, axes=None, trace_type="treemap", figure_uid="fig_tree"
            ),
            TraceInfo(
                uid=bar.uid, axes=("x", "y"), trace_type="bar", figure_uid="fig_bar"
            ),
        ]
        event = self._make_click_event(
            figure_uid="fig_tree",
            clauses=[ClauseFilter(column="country", values=["Germany"])],
        )
        deltas = engine.process(event, infos)
        bar_delta = next((d for d in deltas if d.uid == bar.uid), None)
        assert bar_delta is not None
        # Only Europe (where Germany is) should appear
        continents = set(bar_delta.updates.get("x", []))
        assert continents == {"Europe"}

    def test_pie_click_cross_filters_bar(self, geo_df: LFQueryBuilder):
        """Clicking a pie slice cross-filters a bar (categories_column is None for pie)."""
        from flexviz.trace.pie import PiePlot
        from flexviz.trace.bar import BarPlot

        pie = PiePlot(labels="continent", values="population")
        bar = BarPlot(labels="country", values="population")
        engine = FlexEngine(
            backend_lf=geo_df,
            scalable_traces={pie.uid: pie, bar.uid: bar},
        )
        infos = [
            TraceInfo(uid=pie.uid, axes=None, trace_type="pie", figure_uid="fig_pie"),
            TraceInfo(
                uid=bar.uid, axes=("x", "y"), trace_type="bar", figure_uid="fig_bar"
            ),
        ]
        event = self._make_click_event(
            figure_uid="fig_pie",
            clauses=[ClauseFilter(column="continent", values=["Europe"])],
        )
        deltas = engine.process(event, infos)
        bar_delta = next((d for d in deltas if d.uid == bar.uid), None)
        assert bar_delta is not None
        countries = set(bar_delta.updates.get("x", []))
        assert "China" not in countries
        assert "Japan" not in countries
        assert "Germany" in countries or "France" in countries

    def test_no_categories_column_on_treemap_returns_empty_filter(
        self, geo_df: LFQueryBuilder
    ):
        """If categories_column is missing, treemap produces no filter (all data shown)."""
        from flexviz.trace.treemap import TreeMap
        from flexviz.trace.bar import BarPlot

        treemap = TreeMap(path=["continent", "country"], values="population")
        bar = BarPlot(labels="country", values="population")
        engine = FlexEngine(
            backend_lf=geo_df,
            scalable_traces={treemap.uid: treemap, bar.uid: bar},
        )
        infos = [
            TraceInfo(
                uid=treemap.uid, axes=None, trace_type="treemap", figure_uid="fig_tree"
            ),
            TraceInfo(
                uid=bar.uid, axes=("x", "y"), trace_type="bar", figure_uid="fig_bar"
            ),
        ]
        # No predicates → no filter applied → all data shown
        event = self._make_click_event(figure_uid="fig_tree", clauses=None)
        deltas = engine.process(event, infos)
        bar_delta = next((d for d in deltas if d.uid == bar.uid), None)
        assert bar_delta is not None
        # All countries present since no filter was applied
        countries = set(bar_delta.updates.get("x", []))
        assert "China" in countries
        assert "Japan" in countries
        assert "Germany" in countries


# ---- bar trace -------------------------------------------------------------


class TestEngineBar:
    def test_bar_delta_has_x_and_y(self):
        """Simple bar trace returns a delta with x and y lists."""
        df = pl.DataFrame({"cat": ["A", "A", "B", "B"], "val": [1.0, 2.0, 3.0, 4.0]})
        lf = LFQueryBuilder(df)
        from flexviz.trace.bar import BarPlot

        bar = BarPlot(labels="cat", values="val", agg="sum")
        engine = FlexEngine(backend_lf=lf, scalable_traces={bar.uid: bar})
        infos = [TraceInfo(uid=bar.uid, axes=("x", "y"), trace_type="bar")]
        event = InteractionEvent(type="init", force_update=True)
        deltas = engine.process(event, infos)
        assert len(deltas) == 1
        d = deltas[0]
        assert d.uid == bar.uid
        assert set(d.updates["x"]) == {"A", "B"}
        a_idx = d.updates["x"].index("A")
        assert d.updates["y"][a_idx] == 3.0

    def test_grouped_bar_returns_group_results(self):
        """Grouped bar delta carries child deltas in group_results."""
        df = pl.DataFrame(
            {
                "label": ["X", "X", "Y", "Y"],
                "region": ["North", "South", "North", "South"],
                "val": [10.0, 20.0, 30.0, 40.0],
            }
        )
        lf = LFQueryBuilder(df)
        from flexviz.trace.bar import BarPlot

        bar = BarPlot(labels="label", values="val", agg="sum", group_by="region")
        engine = FlexEngine(backend_lf=lf, scalable_traces={bar.uid: bar})
        infos = [TraceInfo(uid=bar.uid, axes=("x", "y"), trace_type="bar")]
        event = InteractionEvent(type="init", force_update=True)
        deltas = engine.process(event, infos)
        assert len(deltas) == 1
        d = deltas[0]
        assert d.uid == bar.uid
        assert len(d.group_results) == 2
        group_keys = {cr.group_value_key for cr in d.group_results}
        assert group_keys == {"North", "South"}
        for cr in d.group_results:
            assert cr.parent_uid == bar.uid
            assert cr.uid is not None
            assert set(cr.updates["x"]) == {"X", "Y"}

    def test_bar_not_updated_on_viewport(self):
        """Bar trace has empty recompute_axes — viewport event skips it."""
        df = pl.DataFrame(
            {"ts": list(range(100)), "cat": ["A"] * 100, "val": [1.0] * 100}
        )
        lf = LFQueryBuilder(df)
        from flexviz.trace.bar import BarPlot

        bar = BarPlot(labels="cat", values="val")
        line = LinePlot(x="ts", y="val", n_points=50)
        engine = FlexEngine(
            backend_lf=lf, scalable_traces={bar.uid: bar, line.uid: line}
        )
        infos = [
            TraceInfo(uid=bar.uid, axes=("x", "y"), trace_type="bar"),
            TraceInfo(uid=line.uid, axes=("x", "y"), trace_type="line"),
        ]
        event = InteractionEvent(type="viewport", axis_ranges={"x": [0, 50]})
        deltas = engine.process(event, infos)
        delta_uids = {d.uid for d in deltas}
        assert bar.uid not in delta_uids
        assert line.uid in delta_uids

    def test_line_not_updated_on_y_only_viewport(self):
        """A line is x-bound: zooming only y leaves its data unchanged, so the
        engine emits no delta for it (but an x zoom still does)."""
        df = pl.DataFrame(
            {"ts": list(range(100)), "val": [float(i) for i in range(100)]}
        )
        lf = LFQueryBuilder(df)
        line = LinePlot(x="ts", y="val", n_points=50)
        engine = FlexEngine(backend_lf=lf, scalable_traces={line.uid: line})
        infos = [TraceInfo(uid=line.uid, axes=("x", "y"), trace_type="line")]

        y_only = InteractionEvent(type="viewport", axis_ranges={"y": [0, 10]})
        assert engine.process(y_only, infos) == []

        x_zoom = InteractionEvent(type="viewport", axis_ranges={"x": [0, 50]})
        assert {d.uid for d in engine.process(x_zoom, infos)} == {line.uid}

    def test_vertical_histogram_not_updated_on_count_axis_zoom(self):
        """A vertical histogram bins on x; zooming the count axis (y) must not
        re-bin it, while zooming the binned axis (x) must."""
        df = pl.DataFrame({"val": [float(i) for i in range(100)]})
        lf = LFQueryBuilder(df)
        hist = Histogram(x="val", bins=10)
        engine = FlexEngine(backend_lf=lf, scalable_traces={hist.uid: hist})
        infos = [TraceInfo(uid=hist.uid, axes=("x", "y"), trace_type="histogram")]

        count_zoom = InteractionEvent(type="viewport", axis_ranges={"y": [0, 5]})
        assert engine.process(count_zoom, infos) == []

        bin_zoom = InteractionEvent(type="viewport", axis_ranges={"x": [0, 50]})
        assert {d.uid for d in engine.process(bin_zoom, infos)} == {hist.uid}


# ---- grouped line (group_by) -----------------------------------------------


class TestEngineGroupedLine:
    def test_grouped_line_returns_child_deltas(self):
        """Grouped LinePlot parent returns child deltas with correct group_value_key."""
        df = pl.DataFrame(
            {
                "ts": list(range(50)) * 2,
                "val": [float(i) for i in range(50)] * 2,
                "sensor": ["A"] * 50 + ["B"] * 50,
            }
        )
        lf = LFQueryBuilder(df)
        line = LinePlot(x="ts", y="val", n_points=20, group_by="sensor")
        engine = FlexEngine(backend_lf=lf, scalable_traces={line.uid: line})
        infos = [TraceInfo(uid=line.uid, axes=("x", "y"), trace_type="line")]
        event = InteractionEvent(type="init", force_update=True)
        deltas = engine.process(event, infos)
        assert len(deltas) == 1
        d = deltas[0]
        assert d.uid == line.uid
        assert len(d.group_results) == 2
        group_keys = {cr.group_value_key for cr in d.group_results}
        assert group_keys == {"A", "B"}
        for cr in d.group_results:
            assert cr.parent_uid == line.uid
            assert "x" in cr.updates
            assert "y" in cr.updates

    def test_grouped_line_child_uids_are_deterministic(self):
        df = pl.DataFrame(
            {
                "ts": list(range(20)) * 2,
                "val": [float(i) for i in range(20)] * 2,
                "sensor": ["X"] * 20 + ["Y"] * 20,
            }
        )
        lf = LFQueryBuilder(df)
        line = LinePlot(x="ts", y="val", n_points=10, group_by="sensor")
        engine = FlexEngine(backend_lf=lf, scalable_traces={line.uid: line})
        infos = [TraceInfo(uid=line.uid, axes=("x", "y"), trace_type="line")]
        event = InteractionEvent(type="init", force_update=True)
        d1 = engine.process(event, infos)[0]
        d2 = engine.process(event, infos)[0]
        uids1 = {cr.uid for cr in d1.group_results}
        uids2 = {cr.uid for cr in d2.group_results}
        assert uids1 == uids2

    def test_grouped_line_viewport_removes_hidden_group(self):
        df = pl.DataFrame(
            {
                "ts": list(range(10)) + list(range(100, 110)),
                "val": [float(i) for i in range(20)],
                "sensor": ["A"] * 10 + ["B"] * 10,
            }
        )
        lf = LFQueryBuilder(df)
        line = LinePlot(x="ts", y="val", n_points=100, group_by="sensor")
        engine = FlexEngine(backend_lf=lf, scalable_traces={line.uid: line})
        infos = [TraceInfo(uid=line.uid, axes=("x", "y"), trace_type="line")]
        event = InteractionEvent(type="viewport", axis_ranges={"x": [0, 20]})
        deltas = engine.process(event, infos)
        grp_delta = deltas[0]
        assert [cr.group_value_key for cr in grp_delta.group_results] == ["A"]

    def test_grouped_line_empty_groups_after_filter(self):
        """Viewport outside all rows still emits an empty grouped parent delta."""
        df = pl.DataFrame(
            {
                "ts": list(range(10)) * 2,
                "val": [float(i) for i in range(10)] * 2,
                "sensor": ["A"] * 10 + ["B"] * 10,
            }
        )
        lf = LFQueryBuilder(df)
        line = LinePlot(x="ts", y="val", n_points=100, group_by="sensor")
        engine = FlexEngine(backend_lf=lf, scalable_traces={line.uid: line})
        infos = [TraceInfo(uid=line.uid, axes=("x", "y"), trace_type="line")]
        event = InteractionEvent(type="viewport", axis_ranges={"x": [100, 120]})
        deltas = engine.process(event, infos)
        grp_delta = deltas[0]
        assert grp_delta.group_results == []

    def test_grouped_line_selection_keeps_only_visible_groups(self):
        """Cross-filter selection can reduce grouped children without special engine discovery."""
        df = pl.DataFrame(
            {
                "ts": list(range(10)) + list(range(10)),
                "val": [float(i) for i in range(10)]
                + [float(i + 100) for i in range(10)],
                "sensor": ["A"] * 10 + ["B"] * 10,
            }
        )
        lf = LFQueryBuilder(df)
        line_src = LinePlot(x="ts", y="val", n_points=100)
        line_grp = LinePlot(x="ts", y="val", n_points=100, group_by="sensor")
        engine = FlexEngine(
            backend_lf=lf,
            scalable_traces={line_src.uid: line_src, line_grp.uid: line_grp},
        )
        infos = [
            TraceInfo(
                uid=line_src.uid, axes=("x", "y"), trace_type="line", figure_uid="fig_a"
            ),
            TraceInfo(
                uid=line_grp.uid, axes=("x", "y"), trace_type="line", figure_uid="fig_b"
            ),
        ]
        # Select a range that includes only ts=0..4 (both sensors have data there)
        event = InteractionEvent(
            type="selection",
            force_update=True,
            selections=[
                SelectionState(
                    source_figure_uid="fig_a",
                    predicates=[
                        SelectionPredicate(
                            clauses=[
                                ClauseFilter(column="ts", range=(0, 4)),
                                ClauseFilter(column="val", range=(0.0, 50.0)),
                            ]
                        )
                    ],
                ),
            ],
        )
        deltas = engine.process(event, infos)
        grp_delta = next((d for d in deltas if d.uid == line_grp.uid), None)
        assert grp_delta is not None
        assert [cr.group_value_key for cr in grp_delta.group_results] == ["A"]


class TestEngineGroupedHistAndBox:
    def test_grouped_histogram_returns_group_results(self):
        df = pl.DataFrame(
            {
                "val": list(range(10)) + list(range(100, 110)),
                "cat": ["A"] * 10 + ["B"] * 10,
            }
        )
        lf = LFQueryBuilder(df)
        hist = Histogram(x="val", bins=5, group_by="cat")
        engine = FlexEngine(backend_lf=lf, scalable_traces={hist.uid: hist})
        infos = [TraceInfo(uid=hist.uid, axes=("x", "y"), trace_type="histogram")]
        deltas = engine.process(InteractionEvent(type="init", force_update=True), infos)
        assert {cr.group_value_key for cr in deltas[0].group_results} == {"A", "B"}

    def test_grouped_box_returns_group_results(self):
        df = pl.DataFrame(
            {
                "val": [1.0, 2.0, 10.0, 12.0],
                "region": ["N", "N", "S", "S"],
            }
        )
        lf = LFQueryBuilder(df)
        box = BoxPlot(y="val", group_by="region")
        engine = FlexEngine(backend_lf=lf, scalable_traces={box.uid: box})
        infos = [TraceInfo(uid=box.uid, axes=("x", "y"), trace_type="box")]
        deltas = engine.process(InteractionEvent(type="init", force_update=True), infos)
        assert {cr.group_value_key for cr in deltas[0].group_results} == {"N", "S"}


# ---- overlay policy (Phase 2) ----------------------------------------------


class TestEngineOverlayPolicy:
    """Filtered-only traces need an unfiltered layer, but no duplicate bg."""

    def test_overlay_boxplot_init_gets_unfiltered_bg(self):
        """Init needs a sole unfiltered layer even for filtered-only traces."""
        df = pl.DataFrame(
            {
                "ts": list(range(100)),
                "val": [float(i) for i in range(100)],
            }
        )
        lf = LFQueryBuilder(df)
        line = LinePlot(x="ts", y="val", n_points=100)
        box = BoxPlot(y="val")
        engine = FlexEngine(
            backend_lf=lf,
            scalable_traces={line.uid: line, box.uid: box},
        )
        infos = [
            TraceInfo(
                uid=line.uid, axes=("x", "y"), trace_type="line", figure_uid="fig_a"
            ),
            TraceInfo(
                uid=box.uid, axes=("x", "y"), trace_type="box", figure_uid="fig_b"
            ),
        ]
        event = InteractionEvent(type="init", force_update=True)
        deltas = engine.process(event, infos, cross_filter_mode="overlay")
        box_deltas = [d for d in deltas if d.uid == box.uid]
        assert {d.layer for d in box_deltas} == {"bg"}
        line_deltas = [d for d in deltas if d.uid == line.uid]
        assert any(
            d.layer == "bg" for d in line_deltas
        ), "LinePlot should still get bg layer"

    def test_overlay_grouped_bar_init_gets_unfiltered_bg(self):
        """Grouped filtered-only traces also need their initial data."""
        df = pl.DataFrame(
            {
                "ts": list(range(20)),
                "val": [float(i) for i in range(20)],
                "cat": ["A", "B"] * 10,
                "grp": ["X", "Y"] * 10,
            }
        )
        lf = LFQueryBuilder(df)
        line = LinePlot(x="ts", y="val", n_points=100)
        from flexviz.trace.bar import BarPlot

        bar = BarPlot(labels="cat", values="val", agg="sum", group_by="grp")
        engine = FlexEngine(
            backend_lf=lf,
            scalable_traces={line.uid: line, bar.uid: bar},
        )
        infos = [
            TraceInfo(
                uid=line.uid, axes=("x", "y"), trace_type="line", figure_uid="fig_a"
            ),
            TraceInfo(
                uid=bar.uid, axes=("x", "y"), trace_type="bar", figure_uid="fig_b"
            ),
        ]
        event = InteractionEvent(type="init", force_update=True)
        deltas = engine.process(event, infos, cross_filter_mode="overlay")
        bar_deltas = [d for d in deltas if d.uid == bar.uid]
        assert {d.layer for d in bar_deltas} == {"bg"}

    def test_overlay_boxplot_active_viewport_omits_bg(self):
        """With a filtered foreground, filtered-only still suppresses bg work."""
        df = pl.DataFrame(
            {"ts": list(range(100)), "val": [float(i) for i in range(100)]}
        )
        lf = LFQueryBuilder(df)
        line = LinePlot(x="ts", y="val", n_points=100)
        box = BoxPlot(y="val")
        engine = FlexEngine(
            backend_lf=lf,
            scalable_traces={line.uid: line, box.uid: box},
        )
        infos = [
            TraceInfo(
                uid=line.uid, axes=("x", "y"), trace_type="line", figure_uid="fig_a"
            ),
            TraceInfo(
                uid=box.uid, axes=("x", "y"), trace_type="box", figure_uid="fig_b"
            ),
        ]
        event = InteractionEvent(
            type="viewport",
            force_update=True,
            selections=[
                SelectionState(
                    source_figure_uid="fig_a",
                    predicates=[
                        SelectionPredicate(
                            clauses=[ClauseFilter(column="ts", range=(20, 40))]
                        )
                    ],
                )
            ],
            figure_uid="fig_b",
        )

        deltas = engine.process(event, infos, cross_filter_mode="overlay")
        assert {d.layer for d in deltas if d.uid == box.uid} == {"fg"}

    def test_overlay_ungrouped_bar_gets_bg(self):
        """Non-grouped BarPlot has overlay_style='full' — gets bg normally."""
        df = pl.DataFrame(
            {
                "ts": list(range(20)),
                "val": [float(i) for i in range(20)],
                "cat": ["A", "B"] * 10,
            }
        )
        lf = LFQueryBuilder(df)
        line = LinePlot(x="ts", y="val", n_points=100)
        from flexviz.trace.bar import BarPlot

        bar = BarPlot(labels="cat", values="val", agg="sum")
        engine = FlexEngine(
            backend_lf=lf,
            scalable_traces={line.uid: line, bar.uid: bar},
        )
        infos = [
            TraceInfo(
                uid=line.uid, axes=("x", "y"), trace_type="line", figure_uid="fig_a"
            ),
            TraceInfo(
                uid=bar.uid, axes=("x", "y"), trace_type="bar", figure_uid="fig_b"
            ),
        ]
        event = InteractionEvent(type="init", force_update=True)
        deltas = engine.process(event, infos, cross_filter_mode="overlay")
        bar_deltas = [d for d in deltas if d.uid == bar.uid]
        assert any(
            d.layer == "bg" for d in bar_deltas
        ), "ungrouped BarPlot should get bg layer"

    def test_overlay_line_still_gets_bg_and_fg(self):
        """LinePlot (overlay_style='full') still gets both layers."""
        df = pl.DataFrame({"ts": list(range(10)), "val": [float(i) for i in range(10)]})
        lf = LFQueryBuilder(df)
        line_a = LinePlot(x="ts", y="val", n_points=1000)
        line_b = LinePlot(x="ts", y="val", n_points=1000)
        engine = FlexEngine(
            backend_lf=lf,
            scalable_traces={line_a.uid: line_a, line_b.uid: line_b},
        )
        infos = [
            TraceInfo(
                uid=line_a.uid, axes=("x", "y"), trace_type="line", figure_uid="fig_a"
            ),
            TraceInfo(
                uid=line_b.uid, axes=("x", "y"), trace_type="line", figure_uid="fig_b"
            ),
        ]
        event = InteractionEvent(
            type="viewport",
            axis_ranges={"x": (0, 6)},
            selections=[
                SelectionState(
                    source_figure_uid="fig_a",
                    predicates=[
                        SelectionPredicate(
                            clauses=[ClauseFilter(column="ts", range=(2, 4))]
                        )
                    ],
                )
            ],
            figure_uid="fig_b",
        )
        deltas = engine.process(event, infos, cross_filter_mode="overlay")
        target_deltas = [d for d in deltas if d.uid == line_b.uid]
        assert {d.layer for d in target_deltas} == {"bg", "fg"}


# ---- histogram bin alignment ------------------------------------------------


def _make_hist_engine(df: pl.DataFrame, traces: dict):
    lf = LFQueryBuilder(df)
    engine = FlexEngine(backend_lf=lf, scalable_traces=traces)
    return engine, lf


class TestEngineHistogramBinAlignment:
    """Engine-level: multiple histogram traces on the same figure share bin edges."""

    def test_multi_hist_same_figure_aligned_on_init(self):
        """Two ungrouped histogram traces on the same figure must produce identical
        bin centers on an init event (no stored viewport)."""
        df = pl.DataFrame({"val": list(range(500))})
        h1 = Histogram(x="val", bins=10)
        h2 = Histogram(x="val", bins=10)
        engine, _ = _make_hist_engine(df, {h1.uid: h1, h2.uid: h2})
        infos = [
            TraceInfo(
                uid=h1.uid, axes=("x", "y"), trace_type="histogram", figure_uid="fig"
            ),
            TraceInfo(
                uid=h2.uid, axes=("x", "y"), trace_type="histogram", figure_uid="fig"
            ),
        ]
        event = InteractionEvent(type="init", force_update=True)
        deltas = engine.process(event, infos)
        d1 = next(d for d in deltas if d.uid == h1.uid)
        d2 = next(d for d in deltas if d.uid == h2.uid)
        assert d1.updates["x"] == d2.updates["x"]

    def test_multi_hist_same_figure_different_columns_aligned_on_init(self):
        """Same-figure histograms use one no-viewport domain across sibling columns."""
        df = pl.DataFrame(
            {
                "y_pos": [0.0, 0.25, 0.5, 0.75, 1.0],
                "y_neg": [
                    1e-9,
                    0.25 + 1e-9,
                    0.5 + 1e-9,
                    0.75 + 1e-9,
                    1.0 + 1e-9,
                ],
            }
        )
        h1 = Histogram(x="y_pos", bins=4)
        h2 = Histogram(x="y_neg", bins=4)
        engine, _ = _make_hist_engine(df, {h1.uid: h1, h2.uid: h2})
        infos = [
            TraceInfo(
                uid=h1.uid, axes=("x", "y"), trace_type="histogram", figure_uid="fig"
            ),
            TraceInfo(
                uid=h2.uid, axes=("x", "y"), trace_type="histogram", figure_uid="fig"
            ),
        ]
        event = InteractionEvent(type="init", force_update=True)
        deltas = engine.process(event, infos)
        d1 = next(d for d in deltas if d.uid == h1.uid)
        d2 = next(d for d in deltas if d.uid == h2.uid)
        assert d1.updates["x"] == d2.updates["x"]

    def test_multi_hist_different_figures_do_not_share_init_domain(self):
        """Histogram domains are scoped by figure so separate figures stay independent."""
        df = pl.DataFrame(
            {
                "y_pos": [0.0, 0.25, 0.5, 0.75, 1.0],
                "y_neg": [
                    1e-9,
                    0.25 + 1e-9,
                    0.5 + 1e-9,
                    0.75 + 1e-9,
                    1.0 + 1e-9,
                ],
            }
        )
        h1 = Histogram(x="y_pos", bins=4)
        h2 = Histogram(x="y_neg", bins=4)
        engine, _ = _make_hist_engine(df, {h1.uid: h1, h2.uid: h2})
        infos = [
            TraceInfo(
                uid=h1.uid, axes=("x", "y"), trace_type="histogram", figure_uid="fig_a"
            ),
            TraceInfo(
                uid=h2.uid, axes=("x", "y"), trace_type="histogram", figure_uid="fig_b"
            ),
        ]
        event = InteractionEvent(type="init", force_update=True)
        deltas = engine.process(event, infos)
        d1 = next(d for d in deltas if d.uid == h1.uid)
        d2 = next(d for d in deltas if d.uid == h2.uid)
        assert d1.updates["x"] != d2.updates["x"]

    def test_incompatible_coordinate_units_do_not_share_init_domain(self):
        """Sibling domains must not mix numeric or temporal physical scales."""
        base = dt.datetime(2020, 1, 1)
        df = pl.DataFrame(
            {
                "value": list(range(100)),
                "day": [base.date() + dt.timedelta(days=i) for i in range(100)],
                "stamp_us": [base + dt.timedelta(hours=i) for i in range(100)],
            }
        ).with_columns(pl.col("stamp_us").cast(pl.Datetime("ms")).alias("stamp_ms"))
        traces = [Histogram(x=col) for col in ("value", "day", "stamp_us", "stamp_ms")]
        engine, _ = _make_hist_engine(df, {trace.uid: trace for trace in traces})
        infos = [
            TraceInfo(
                uid=trace.uid,
                axes=("x", "y"),
                trace_type="histogram",
                figure_uid="fig",
            )
            for trace in traces
        ]

        deltas = engine.process(InteractionEvent(type="init", force_update=True), infos)
        deltas_by_uid = {delta.uid: delta for delta in deltas}
        centers = {
            trace.data_col: deltas_by_uid[trace.uid].updates["x"] for trace in traces
        }

        assert 0 <= min(centers["value"]) <= max(centers["value"]) <= 99
        assert (
            base.date()
            <= min(centers["day"])
            <= max(centers["day"])
            <= (base.date() + dt.timedelta(days=99))
        )
        for col in ("stamp_us", "stamp_ms"):
            assert (
                base
                <= min(centers[col])
                <= max(centers[col])
                <= (base + dt.timedelta(hours=99))
            )

    def test_grouped_hist_aligned_on_init(self):
        """Grouped histogram: all child groups share identical bin centers on init."""
        df = pl.DataFrame(
            {
                "val": list(range(10)) + list(range(100, 110)),
                "cat": ["A"] * 10 + ["B"] * 10,
            }
        )
        hist = Histogram(x="val", bins=5, group_by="cat")
        engine, _ = _make_hist_engine(df, {hist.uid: hist})
        infos = [
            TraceInfo(
                uid=hist.uid, axes=("x", "y"), trace_type="histogram", figure_uid="fig"
            ),
        ]
        event = InteractionEvent(type="init", force_update=True)
        deltas = engine.process(event, infos)
        assert len(deltas) == 1
        child_deltas = deltas[0].group_results
        assert child_deltas is not None and len(child_deltas) == 2
        centers_a = child_deltas[0].updates["x"]
        centers_b = child_deltas[1].updates["x"]
        assert centers_a == centers_b

    def test_hist_aligned_on_viewport(self):
        """Viewport event: two traces use the same literal bin edges from the range."""
        df = pl.DataFrame({"val": list(range(500))})
        h1 = Histogram(x="val", bins=8)
        h2 = Histogram(x="val", bins=8)
        engine, _ = _make_hist_engine(df, {h1.uid: h1, h2.uid: h2})
        infos = [
            TraceInfo(
                uid=h1.uid, axes=("x", "y"), trace_type="histogram", figure_uid="fig"
            ),
            TraceInfo(
                uid=h2.uid, axes=("x", "y"), trace_type="histogram", figure_uid="fig"
            ),
        ]
        event = InteractionEvent(
            type="viewport", axis_ranges={"x": (50, 300)}, force_update=True
        )
        deltas = engine.process(event, infos)
        d1 = next(d for d in deltas if d.uid == h1.uid)
        d2 = next(d for d in deltas if d.uid == h2.uid)
        assert d1.updates["x"] == d2.updates["x"]
        assert len(d1.updates["x"]) == 8

    def test_hist_aligned_on_autorange_null_viewport(self):
        """A double-click autorange posts ``axis_ranges={"x": None}`` (unzoomed).
        Sibling histograms on different columns must keep their shared domain —
        a ``None`` range must not be read as "zoomed" and drop them to per-column
        domains (that misaligned the bars; see the anchor/null review finding)."""
        df = pl.DataFrame(
            {
                "wide": [float(i % 100) for i in range(500)],
                "narrow": [float(i % 10) for i in range(500)],
            }
        )
        h1 = Histogram(x="wide", bins=10)
        h2 = Histogram(x="narrow", bins=10)
        engine, _ = _make_hist_engine(df, {h1.uid: h1, h2.uid: h2})
        infos = [
            TraceInfo(
                uid=h1.uid, axes=("x", "y"), trace_type="histogram", figure_uid="fig"
            ),
            TraceInfo(
                uid=h2.uid, axes=("x", "y"), trace_type="histogram", figure_uid="fig"
            ),
        ]
        event = InteractionEvent(
            type="viewport", axis_ranges={"x": None}, force_update=True
        )
        deltas = engine.process(event, infos)
        d1 = next(d for d in deltas if d.uid == h1.uid)
        d2 = next(d for d in deltas if d.uid == h2.uid)
        assert d1.updates["x"] == d2.updates["x"]


class TestEngineHistogramOverlayAlignment:
    """Overlay mode: bg and fg histogram layers must share identical bin edges."""

    def test_overlay_viewport_bg_fg_share_bin_centers(self):
        """On a viewport event with an active selection both bg and fg layers of a
        histogram target must have identical bin centers."""
        n = 200
        df = pl.DataFrame(
            {
                "ts": list(range(n)),
                "val": [float(i) for i in range(n)],
            }
        )
        lf = LFQueryBuilder(df)
        line_src = LinePlot(x="ts", y="val", n_points=1000)
        hist_tgt = Histogram(x="val", bins=10)
        engine = FlexEngine(
            backend_lf=lf,
            scalable_traces={line_src.uid: line_src, hist_tgt.uid: hist_tgt},
        )
        infos = [
            TraceInfo(
                uid=line_src.uid,
                axes=("x", "y"),
                trace_type="line",
                figure_uid="fig_src",
            ),
            TraceInfo(
                uid=hist_tgt.uid,
                axes=("x", "y"),
                trace_type="histogram",
                figure_uid="fig_tgt",
            ),
        ]
        event = InteractionEvent(
            type="viewport",
            axis_ranges={"x": (0, n - 1)},
            figure_uid="fig_tgt",
            selections=[
                SelectionState(
                    source_figure_uid="fig_src",
                    predicates=[
                        SelectionPredicate(
                            clauses=[ClauseFilter(column="ts", range=(50, 100))]
                        )
                    ],
                )
            ],
        )
        deltas = engine.process(event, infos, cross_filter_mode="overlay")
        hist_deltas = [d for d in deltas if d.uid == hist_tgt.uid]
        assert {d.layer for d in hist_deltas} == {"bg", "fg"}
        bg = next(d for d in hist_deltas if d.layer == "bg")
        fg = next(d for d in hist_deltas if d.layer == "fg")
        assert bg.updates["x"] == fg.updates["x"], "bg and fg bin centers must match"


# ---- overlay event-sequence tests ------------------------------------------


class TestEngineOverlaySequences:
    """Verify the overlay state-machine across init → selection → deselect."""

    def _two_figure_setup(self):
        df = pl.DataFrame({"ts": list(range(20)), "val": [float(i) for i in range(20)]})
        lf = LFQueryBuilder(df)
        src = LinePlot(x="ts", y="val", n_points=1000)
        tgt = LinePlot(x="ts", y="val", n_points=1000)
        engine = FlexEngine(backend_lf=lf, scalable_traces={src.uid: src, tgt.uid: tgt})
        infos = [
            TraceInfo(
                uid=src.uid, axes=("x", "y"), trace_type="line", figure_uid="fig_src"
            ),
            TraceInfo(
                uid=tgt.uid, axes=("x", "y"), trace_type="line", figure_uid="fig_tgt"
            ),
        ]
        return engine, infos, src, tgt

    def test_init_overlay_emits_bg_only(self):
        """On init with no selection, overlay mode emits only bg for all traces."""
        engine, infos, src, tgt = self._two_figure_setup()
        event = InteractionEvent(type="init", force_update=True)
        deltas = engine.process(event, infos, cross_filter_mode="overlay")
        layers = {d.layer for d in deltas}
        assert "fg" not in layers, "init should not produce fg deltas"
        assert "bg" in layers

    def test_deselect_overlay_returns_bg_only(self):
        """After a deselect, overlay mode emits only bg (clears the fg layer)."""
        engine, infos, src, tgt = self._two_figure_setup()
        event = InteractionEvent(
            type="deselect",
            force_update=True,
            selections=[],
        )
        deltas = engine.process(event, infos, cross_filter_mode="overlay")
        layers = {d.layer for d in deltas}
        assert "fg" not in layers
        assert "bg" in layers

    def test_reset_overlay_returns_bg_only(self):
        """After a reset, overlay mode emits only bg, same as init."""
        engine, infos, src, tgt = self._two_figure_setup()
        event = InteractionEvent(type="reset", force_update=True, selections=[])
        deltas = engine.process(event, infos, cross_filter_mode="overlay")
        layers = {d.layer for d in deltas}
        assert "fg" not in layers
        assert "bg" in layers


# ---- viewport edge cases ---------------------------------------------------


class TestEngineViewportEdgeCases:
    def test_viewport_outside_data_range_returns_empty_delta(self):
        """A viewport that does not intersect the data still returns a delta (empty)."""
        df = pl.DataFrame({"ts": list(range(10)), "val": [float(i) for i in range(10)]})
        lf = LFQueryBuilder(df)
        line = LinePlot(x="ts", y="val", n_points=100)
        engine = FlexEngine(backend_lf=lf, scalable_traces={line.uid: line})
        infos = [TraceInfo(uid=line.uid, axes=("x", "y"), trace_type="line")]
        event = InteractionEvent(type="viewport", axis_ranges={"x": [1000, 2000]})
        deltas = engine.process(event, infos)
        assert len(deltas) == 1
        assert deltas[0].updates["x"] == []

    def test_self_viewport_does_not_cross_filter(self):
        """A viewport event from figure A must NOT filter figure A's own data."""
        df = pl.DataFrame({"ts": list(range(20)), "val": [float(i) for i in range(20)]})
        lf = LFQueryBuilder(df)
        line_a = LinePlot(x="ts", y="val", n_points=1000)
        line_b = LinePlot(x="ts", y="val", n_points=1000)
        engine = FlexEngine(
            backend_lf=lf, scalable_traces={line_a.uid: line_a, line_b.uid: line_b}
        )
        infos = [
            TraceInfo(
                uid=line_a.uid, axes=("x", "y"), trace_type="line", figure_uid="fig_a"
            ),
            TraceInfo(
                uid=line_b.uid, axes=("x", "y"), trace_type="line", figure_uid="fig_a"
            ),
        ]
        # viewport from fig_a with a selection on fig_a — both traces are in fig_a
        event = InteractionEvent(
            type="viewport",
            axis_ranges={"x": (0, 5)},
            figure_uid="fig_a",
            selections=[
                SelectionState(
                    source_figure_uid="fig_a",
                    predicates=[
                        SelectionPredicate(
                            clauses=[ClauseFilter(column="ts", range=(0, 5))]
                        )
                    ],
                )
            ],
        )
        deltas = engine.process(event, infos)
        # fig_a traces should not be cross-filtered — both traces get viewport data
        assert len(deltas) >= 1
        for d in deltas:
            # viewport restricts to x=0..5, but cross-filter from self is skipped
            assert len(d.updates["x"]) > 0

    def test_histogram_update_on_zoom_false_skipped_on_viewport(self):
        """Histogram with empty recompute_axes must NOT be included in viewport deltas."""
        df = pl.DataFrame({"ts": list(range(50)), "val": [float(i) for i in range(50)]})
        lf = LFQueryBuilder(df)
        hist = Histogram(x="val", bins=10)
        hist._recompute_axes = ()  # override post-construction for test
        line = LinePlot(x="ts", y="val", n_points=100)
        engine = FlexEngine(
            backend_lf=lf, scalable_traces={hist.uid: hist, line.uid: line}
        )
        infos = [
            TraceInfo(uid=hist.uid, axes=("x", "y"), trace_type="histogram"),
            TraceInfo(uid=line.uid, axes=("x", "y"), trace_type="line"),
        ]
        event = InteractionEvent(type="viewport", axis_ranges={"x": [10, 40]})
        deltas = engine.process(event, infos)
        delta_uids = {d.uid for d in deltas}
        assert hist.uid not in delta_uids
        assert line.uid in delta_uids


# ---- grouped trace edge cases ----------------------------------------------


class TestGroupedTracesEdgeCases:
    def test_group_domain_key_format_in_trace_spec(self):
        """to_trace_spec(domain_source=...) sets group_domain_key = '{source}::{col}'."""
        line = LinePlot(x="ts", y="val", n_points=100, group_by="sensor")
        spec = line.to_trace_spec(domain_source="mysource")
        assert spec.params["group_domain_key"] == "mysource::sensor"

    def test_group_domain_key_not_overwritten_if_already_set(self):
        """If group_domain_key is already in params, to_trace_spec must not overwrite it."""
        line = LinePlot(x="ts", y="val", n_points=100, group_by="sensor")
        line._params["group_domain_key"] = "original::key"
        spec = line.to_trace_spec(domain_source="other_source")
        assert spec.params["group_domain_key"] == "original::key"

    def test_grouped_bar_cross_filter_with_categories(self):
        """Grouped bar receives only matching categories after cross-filter selection."""
        df = pl.DataFrame(
            {
                "cat": ["A", "A", "B", "B", "C", "C"],
                "grp": ["X", "Y", "X", "Y", "X", "Y"],
                "val": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )
        lf = LFQueryBuilder(df)

        bar_src = BarPlot(labels="cat", values="val", agg="sum")
        bar_tgt = BarPlot(labels="cat", values="val", agg="sum", group_by="grp")

        engine = FlexEngine(
            backend_lf=lf,
            scalable_traces={bar_src.uid: bar_src, bar_tgt.uid: bar_tgt},
        )
        infos = [
            TraceInfo(
                uid=bar_src.uid, axes=("x", "y"), trace_type="bar", figure_uid="fig_src"
            ),
            TraceInfo(
                uid=bar_tgt.uid, axes=("x", "y"), trace_type="bar", figure_uid="fig_tgt"
            ),
        ]
        event = InteractionEvent(
            type="selection",
            force_update=True,
            selections=[
                SelectionState(
                    source_figure_uid="fig_src",
                    predicates=[
                        SelectionPredicate(
                            clauses=[ClauseFilter(column="cat", values=["A", "B"])]
                        )
                    ],
                )
            ],
        )
        deltas = engine.process(event, infos)
        tgt_delta = next(d for d in deltas if d.uid == bar_tgt.uid)
        # grouped bar yields group_results; all child cats should be A or B only
        for child in tgt_delta.group_results:
            for cat in child.updates.get("x", []):
                assert cat in ("A", "B")


class TestEngineSelectionFilterExprs:
    def test_predicates_filter_target_traces(self):
        """Selections with predicates filter target traces' aggregation."""
        import polars as pl
        from flexviz.engine import FlexEngine, TraceInfo
        from flexviz.events import InteractionEvent
        from flexviz.LF import LFQueryBuilder
        from flexviz.spec import ClauseFilter, SelectionPredicate, SelectionState
        from flexviz.trace.bar import BarPlot

        df = pl.DataFrame(
            {
                "country": ["NL", "BE", "NL", "DE"],
                "source": ["solar", "solar", "wind", "wind"],
                "v": [1.0, 1.0, 1.0, 1.0],
            }
        )
        lf = LFQueryBuilder(df)
        trace_a = BarPlot(labels="country", values="v", agg="sum")
        trace_b = BarPlot(labels="source", values="v", agg="sum")
        trace_a.uid = "a"
        trace_b.uid = "b"

        engine = FlexEngine(backend_lf=lf, scalable_traces={"a": trace_a, "b": trace_b})
        event = InteractionEvent(
            type="selection",
            force_update=True,
            selections=[
                SelectionState(
                    source_figure_uid="figA",
                    predicates=[
                        SelectionPredicate(
                            clauses=[ClauseFilter(column="country", values=["NL"])]
                        )
                    ],
                )
            ],
        )
        deltas = engine.process(
            event=event,
            trace_infos=[
                TraceInfo(
                    uid="a", axes=("x", "y"), trace_type="bar", figure_uid="figA"
                ),
                TraceInfo(
                    uid="b", axes=("x", "y"), trace_type="bar", figure_uid="figB"
                ),
            ],
        )
        deltas_by_uid = {d.uid: d for d in deltas}
        # Source figure "figA" is not re-aggregated on its own selection
        assert "a" not in deltas_by_uid
        # Target figure "figB" sees only NL rows: solar and wind, both sum=1.0
        target = deltas_by_uid["b"].group_results
        assert target is None
        assert sorted(deltas_by_uid["b"].updates["x"]) == ["solar", "wind"]
        assert all(v == 1.0 for v in deltas_by_uid["b"].updates["y"])


# ---- residency seam --------------------------------------------------------


class TestResidencySeam:
    """A scan source must render exactly what a resident frame renders.

    Columns are deliberately not called x/y: the streaming plan builds a struct
    whose field names must follow the *source* columns, and naming them x/y
    hides that.

    The kernel needs the whole column in memory, so a scan swaps in a streaming
    plan (`_native_envelope_plan`). That is a formulation change, never a
    result change — if these ever diverge, the same figure would look different
    depending on where its rows came from.
    """

    @staticmethod
    def _line_deltas(src, n_points: int = 1000):
        lf = LFQueryBuilder(src)
        line = LinePlot(x="ts", y="val", n_points=n_points)
        engine = FlexEngine(backend_lf=lf, scalable_traces={line.uid: line})
        infos = [TraceInfo(uid=line.uid, axes=("x", "y"), trace_type="line")]
        deltas = engine.process(InteractionEvent(type="init", force_update=True), infos)
        return deltas[0].updates, lf.is_scan

    @pytest.mark.parametrize(
        "name,n_rows,ymaker",
        [
            # Plain float data: no ties, the easy case.
            ("random", 20_011, lambda n: [((i * 7919) % 1000) / 7.0 for i in range(n)]),
            # Quantised data puts exact plateaus under the bucket extrema, which
            # is where a `min_by` formulation silently picks a different row.
            (
                "plateau",
                20_000,
                lambda n: [round(((i * 31) % 97) / 7.0, 1) for i in range(n)],
            ),
            ("constant", 20_000, lambda n: [3.0] * n),
            ("two_level", 20_000, lambda n: [1.0, 2.0] * (n // 2)),
            # len % n_buckets != 0 exercises the long/short window split.
            ("odd_len", 20_003, lambda n: [float((i * 13) % 511) for i in range(n)]),
            # Fewer rows than requested points: every row is its own bucket.
            ("tiny", 137, lambda n: [float((i * 5) % 31) for i in range(n)]),
        ],
    )
    def test_scan_matches_resident(self, tmp_path, name, n_rows, ymaker):
        df = pl.DataFrame(
            {"ts": list(range(n_rows)), "val": ymaker(n_rows)},
            schema={"ts": pl.Int64, "val": pl.Float64},
        )
        path = tmp_path / f"{name}.parquet"
        df.write_parquet(path)

        resident, resident_is_scan = self._line_deltas(df)
        scanned, scan_is_scan = self._line_deltas(pl.scan_parquet(path))

        assert resident_is_scan is False and scan_is_scan is True, (
            "the two sides must actually take different paths for this to mean "
            "anything"
        )
        assert list(scanned["x"]) == list(resident["x"])
        assert list(scanned["y"]) == list(resident["y"])

    def test_nulls_and_nan_match(self, tmp_path):
        n = 20_000
        y = [
            None if i % 501 == 0 else (float("nan") if i % 307 == 0 else float(i % 89))
            for i in range(n)
        ]
        df = pl.DataFrame(
            {"ts": list(range(n)), "val": y}, schema={"ts": pl.Int64, "val": pl.Float64}
        )
        path = tmp_path / "nulls.parquet"
        df.write_parquet(path)
        resident, _ = self._line_deltas(df)
        scanned, _ = self._line_deltas(pl.scan_parquet(path))
        assert list(scanned["x"]) == list(resident["x"])
        # NaN != NaN, so compare the bit patterns rather than the values.
        assert [repr(v) for v in scanned["y"]] == [repr(v) for v in resident["y"]]

    def test_resident_lazyframe_is_not_treated_as_a_scan(self):
        """`df.lazy()` is still resident — it must keep the kernel path."""
        df = pl.DataFrame({"ts": [1, 2, 3], "val": [1.0, 2.0, 3.0]})
        assert LFQueryBuilder(df.lazy()).is_scan is False

    def test_scan_selects_the_streaming_plan(self):
        """The seam must actually swap formulations, not just report a flag.

        Cheap portable guard: the full capped boundedness gate (an
        out-of-core canary needing Linux cgroups and a 60M-row fixture) is
        too heavy for the unit suite. This catches the regression that would
        silently disable it — `is_scan` wired up but the spec still carrying
        the materializing kernel expression.
        """
        line = LinePlot(x="ts", y="val", n_points=1000)
        resident = line.get_aggregation_spec({}, scan_source=False)
        scanned = line.get_aggregation_spec({}, scan_source=True)
        assert resident.plan is None, "a resident source must keep the kernel"
        assert scanned.plan is not None, "a scan source must bring its own plan"

    @pytest.mark.parametrize("downsample", ["nth", "fpcs"])
    def test_only_minmax_swaps(self, downsample):
        """`nth` already streams; `fpcs` has no streaming formulation."""
        line = LinePlot(x="ts", y="val", n_points=1000, downsample=downsample)
        assert line.get_aggregation_spec({}, scan_source=True).plan is None


class TestHistResidencySeam:
    """Scan-source histograms swap the fixed_hist kernel for a streaming plan.

    Same contract as the line seam above: a formulation change, never a result
    change. The fixtures deliberately cover the kernel's edge semantics — null
    and NaN rows are skipped, out-of-viewport values clamp into the edge bins,
    an all-identical column keeps the epsilon-padded non-degenerate path, and
    a constant column at 1e20 underflows the epsilon into the kernel's
    degenerate everything-into-bin-0 path.
    """

    @staticmethod
    def _deltas(src, traces: list, event: InteractionEvent | None = None):
        lf = LFQueryBuilder(src)
        engine = FlexEngine(backend_lf=lf, scalable_traces={t.uid: t for t in traces})
        infos = [
            TraceInfo(
                uid=t.uid, axes=("x", "y"), trace_type=t.trace_type, figure_uid="f1"
            )
            for t in traces
        ]
        event = event or InteractionEvent(type="init", force_update=True)
        deltas = engine.process(event, infos)
        by_uid = {d.uid: d for d in deltas}
        # Order by construction order: each side builds its own trace
        # instances, so uids never match across sides — positions do.
        return [by_uid[t.uid] for t in traces], lf.is_scan

    @staticmethod
    def _roundtrip(tmp_path, df: pl.DataFrame, traces_maker, event=None):
        path = tmp_path / "data.parquet"
        df.write_parquet(path)
        resident, res_scan = TestHistResidencySeam._deltas(df, traces_maker(), event)
        scanned, scan_scan = TestHistResidencySeam._deltas(
            pl.scan_parquet(path), traces_maker(), event
        )
        assert res_scan is False and scan_scan is True
        return resident, scanned

    @pytest.mark.parametrize(
        "name,values",
        [
            ("random", [((i * 7919) % 1009) / 7.0 for i in range(20_011)]),
            (
                "nulls_and_nan",
                [
                    (
                        None
                        if i % 501 == 0
                        else (float("nan") if i % 307 == 0 else float(i % 89))
                    )
                    for i in range(20_000)
                ],
            ),
            ("constant", [3.0] * 5_000),
            # 1e20 + _HIST_BIN_EPSILON == 1e20: the kernel's hi <= lo
            # degenerate branch, everything lands in bin 0.
            ("degenerate_eps_underflow", [1e20] * 5_000),
        ],
    )
    def test_scan_matches_resident(self, tmp_path, name, values):
        df = pl.DataFrame({"val": values}, schema={"val": pl.Float64})

        def make():
            return [Histogram(x="val", bins=13)]

        resident, scanned = self._roundtrip(tmp_path, df, make)
        (r,) = resident
        (s,) = scanned
        assert s.updates.keys() == r.updates.keys()
        for key in r.updates:
            assert list(s.updates[key]) == list(r.updates[key]), key

    def test_integer_dtype_matches(self, tmp_path):
        df = pl.DataFrame(
            {"val": [(i * 13) % 257 for i in range(10_000)]},
            schema={"val": pl.Int32},
        )
        resident, scanned = self._roundtrip(
            tmp_path, df, lambda: [Histogram(x="val", bins=16)]
        )
        (r,), (s,) = resident, scanned
        for key in r.updates:
            assert list(s.updates[key]) == list(r.updates[key]), key

    def test_temporal_dtype_matches(self, tmp_path):
        base = dt.datetime(2024, 1, 1)
        df = pl.DataFrame(
            {
                "when": [
                    base + dt.timedelta(minutes=(i * 37) % 999) for i in range(5_000)
                ]
            }
        )
        resident, scanned = self._roundtrip(
            tmp_path, df, lambda: [Histogram(x="when", bins=10)]
        )
        (r,), (s,) = resident, scanned
        for key in r.updates:
            assert list(s.updates[key]) == list(r.updates[key]), key

    def test_viewport_zoom_matches(self, tmp_path):
        """Zoomed bounds are literals; values outside the viewport are dropped
        by the filter on both paths, and boundary values clamp identically."""
        df = pl.DataFrame({"val": [float(i % 977) for i in range(20_000)]})
        event = InteractionEvent(
            type="viewport", axis_ranges={"x": [100.0, 500.0]}, force_update=True
        )
        resident, scanned = self._roundtrip(
            tmp_path, df, lambda: [Histogram(x="val", bins=17)], event
        )
        (r,), (s,) = resident, scanned
        for key in r.updates:
            assert list(s.updates[key]) == list(r.updates[key]), key

    def test_sibling_histograms_share_domain_on_scan(self, tmp_path):
        """Same-figure histogram siblings share one min/max domain
        (histogram_domain_cols). The hoisted stats row must preserve that."""
        df = pl.DataFrame(
            {
                "a": [float(i % 100) for i in range(10_000)],
                "b": [float(i % 700) for i in range(10_000)],
            }
        )

        def make():
            return [Histogram(x="a", bins=12), Histogram(x="b", bins=12)]

        resident, scanned = self._roundtrip(tmp_path, df, make)
        assert len(resident) == len(scanned) == 2
        for i, (r, s) in enumerate(zip(resident, scanned)):
            for key in r.updates:
                assert list(s.updates[key]) == list(r.updates[key]), (i, key)

    def test_grouped_hist_matches_on_scan(self, tmp_path):
        """The grouped path keeps the kernel on scans; its bin bounds now come
        from the constant stats columns LFQueryBuilder injects. Same children,
        same bins, either residency."""
        df = pl.DataFrame(
            {
                "val": [float(i % 89) for i in range(10_000)],
                "g": [("g" + str(i % 3)) for i in range(10_000)],
            }
        )

        def make():
            return [Histogram(x="val", bins=9, group_by="g")]

        resident, scanned = self._roundtrip(tmp_path, df, make)
        (r,), (s,) = resident, scanned
        # Child uids derive from the (per-instance) parent uid; the group value
        # key is the residency-independent identity.
        r_children = {c.group_value_key: c for c in r.group_results or []}
        s_children = {c.group_value_key: c for c in s.group_results or []}
        assert r_children.keys() == s_children.keys() and r_children
        for gv, rc in r_children.items():
            sc = s_children[gv]
            for key in rc.updates:
                assert list(sc.updates[key]) == list(rc.updates[key]), (gv, key)

    def test_cross_filtered_matches(self, tmp_path):
        """The streaming plan runs on the cross-filtered frame while its bins
        stay derived from the unfiltered stats — identical to the kernel path."""
        df = pl.DataFrame(
            {
                "ts": [float(i) for i in range(10_000)],
                "val": [float(i % 89) for i in range(10_000)],
            }
        )
        path = tmp_path / "data.parquet"
        df.write_parquet(path)
        event = InteractionEvent(
            type="selection",
            force_update=True,
            selections=[
                SelectionState(
                    source_figure_uid="fig_src",
                    predicates=[
                        SelectionPredicate(
                            clauses=[ClauseFilter(column="ts", range=(1000, 4000))]
                        )
                    ],
                ),
            ],
        )

        def deltas(src):
            lf = LFQueryBuilder(src)
            hist = Histogram(x="val", bins=11)
            engine = FlexEngine(backend_lf=lf, scalable_traces={hist.uid: hist})
            infos = [
                TraceInfo(
                    uid=hist.uid,
                    axes=("x", "y"),
                    trace_type="histogram",
                    figure_uid="fig_target",
                )
            ]
            return engine.process(event, infos)[0].updates

        resident = deltas(df)
        scanned = deltas(pl.scan_parquet(path))
        for key in resident:
            assert list(scanned[key]) == list(resident[key]), key

    def test_empty_source_matches(self, tmp_path):
        df = pl.DataFrame({"val": []}, schema={"val": pl.Float64})
        resident, scanned = self._roundtrip(
            tmp_path, df, lambda: [Histogram(x="val", bins=5)]
        )
        (r,), (s,) = resident, scanned
        for key in r.updates:
            assert list(s.updates[key]) == list(r.updates[key]), key

    def test_scan_selects_the_streaming_plan(self):
        hist = Histogram(x="val", bins=10)
        assert hist.get_aggregation_spec({}, scan_source=False).plan is None
        assert hist.get_aggregation_spec({}, scan_source=True).plan is not None

    def test_grouped_spec_keeps_the_kernel(self):
        from flexviz.LF import GroupedAggregationSpec

        hist = Histogram(x="val", bins=10, group_by="g")
        schema = pl.Schema({"val": pl.Float64, "g": pl.String})
        spec = hist.get_aggregation_spec({}, schema=schema, scan_source=True)
        assert isinstance(spec, GroupedAggregationSpec)


class TestHist2DResidencySeam:
    """Scan-source 2D histograms swap fixed_hist2d(_reduce) for a streaming
    plan. Counts and min/max must match exactly; sum/mean accumulate in a
    different order, so the fixtures use integer-valued z where every partial
    sum is exactly representable and equality stays exact.
    """

    @staticmethod
    def _deltas(src, trace, event=None):
        from flexviz.trace.hist2d import Histogram2D  # noqa: F401

        lf = LFQueryBuilder(src)
        engine = FlexEngine(backend_lf=lf, scalable_traces={trace.uid: trace})
        infos = [
            TraceInfo(
                uid=trace.uid,
                axes=("x", "y"),
                trace_type=trace.trace_type,
                figure_uid="f1",
            )
        ]
        event = event or InteractionEvent(type="init", force_update=True)
        deltas = engine.process(event, infos)
        return deltas[0].updates, lf.is_scan

    @classmethod
    def _roundtrip(cls, tmp_path, df, trace_maker, event=None):
        path = tmp_path / "data.parquet"
        df.write_parquet(path)
        resident, res_scan = cls._deltas(df, trace_maker(), event)
        scanned, scan_scan = cls._deltas(pl.scan_parquet(path), trace_maker(), event)
        assert res_scan is False and scan_scan is True
        assert scanned.keys() == resident.keys()
        for key in resident:
            assert scanned[key] == resident[key], key

    @staticmethod
    def _df(n: int = 20_000) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "px": [
                    (
                        None
                        if i % 501 == 0
                        else (float("nan") if i % 307 == 0 else ((i * 13) % 199) / 3.0)
                    )
                    for i in range(n)
                ],
                "py": [
                    None if i % 401 == 0 else float((i * 29) % 173) for i in range(n)
                ],
                # Integer-valued floats: sums are exact in any accumulation order.
                "pz": [None if i % 703 == 0 else float((i * 7) % 50) for i in range(n)],
            }
        )

    def test_count_matches(self, tmp_path):
        from flexviz.trace.hist2d import Histogram2D

        self._roundtrip(
            tmp_path,
            self._df(),
            lambda: Histogram2D(x="px", y="py", x_bins=15, y_bins=11),
        )

    @pytest.mark.parametrize("histfunc", ["sum", "mean", "min", "max"])
    def test_reduce_matches(self, tmp_path, histfunc):
        from flexviz.trace.hist2d import Histogram2D

        self._roundtrip(
            tmp_path,
            self._df(),
            lambda: Histogram2D(
                x="px", y="py", x_bins=9, y_bins=7, z="pz", histfunc=histfunc
            ),
        )

    def test_viewport_zoom_matches(self, tmp_path):
        from flexviz.trace.hist2d import Histogram2D

        event = InteractionEvent(
            type="viewport",
            axis_ranges={"x": [10.0, 50.0], "y": [20.0, 120.0]},
            force_update=True,
        )
        self._roundtrip(
            tmp_path,
            self._df(),
            lambda: Histogram2D(x="px", y="py", x_bins=8, y_bins=8),
            event,
        )

    def test_temporal_axis_matches(self, tmp_path):
        from flexviz.trace.hist2d import Histogram2D

        base = dt.datetime(2024, 3, 1)
        df = pl.DataFrame(
            {
                "when": [
                    base + dt.timedelta(minutes=(i * 37) % 999) for i in range(5_000)
                ],
                "py": [float((i * 29) % 173) for i in range(5_000)],
            }
        )
        self._roundtrip(
            tmp_path,
            df,
            lambda: Histogram2D(x="when", y="py", x_bins=6, y_bins=6),
        )

    def test_scan_selects_the_streaming_plan(self):
        from flexviz.trace.hist2d import Histogram2D

        h = Histogram2D(x="px", y="py")
        assert h.get_aggregation_spec({}, scan_source=False).plan is None
        assert h.get_aggregation_spec({}, scan_source=True).plan is not None


class TestGeoHist2DResidencySeam:
    """Scan-source geo heatmaps swap the kernels for a streaming plan; the
    "data" bin-boundaries mode additionally derives its bounds from the
    filtered extent in an extra bounded pass."""

    @staticmethod
    def _deltas(src, trace, viewports=None):
        lf = LFQueryBuilder(src)
        engine = FlexEngine(backend_lf=lf, scalable_traces={trace.uid: trace})
        infos = [
            TraceInfo(
                uid=trace.uid,
                axes=("coordinates",),
                trace_type=trace.trace_type,
                figure_uid="f1",
            )
        ]
        deltas = engine.process(
            InteractionEvent(type="init", force_update=True),
            infos,
            viewports_by_figure={"f1": viewports or {}},
        )
        return deltas[0].updates, lf.is_scan

    @classmethod
    def _roundtrip(cls, tmp_path, df, trace_maker, viewports=None):
        path = tmp_path / "geo.parquet"
        df.write_parquet(path)
        resident, res_scan = cls._deltas(df, trace_maker(), viewports)
        scanned, scan_scan = cls._deltas(
            pl.scan_parquet(path), trace_maker(), viewports
        )
        assert res_scan is False and scan_scan is True
        assert scanned.keys() == resident.keys()
        for key in resident:
            assert scanned[key] == resident[key], key

    @staticmethod
    def _df(n: int = 10_000) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "lat": [
                    None if i % 401 == 0 else 40.0 + ((i * 13) % 100) / 10.0
                    for i in range(n)
                ],
                "lon": [3.0 + ((i * 29) % 140) / 20.0 for i in range(n)],
                "zv": [float((i * 7) % 30) for i in range(n)],
            }
        )

    def test_data_mode_count_matches(self, tmp_path):
        from flexviz.trace.geo_hist2d import GeoHistogram2D

        self._roundtrip(
            tmp_path,
            self._df(),
            lambda: GeoHistogram2D(lat="lat", lon="lon", lat_bins=9, lon_bins=7),
        )

    @pytest.mark.parametrize("histfunc", ["sum", "mean", "min", "max"])
    def test_data_mode_reduce_matches(self, tmp_path, histfunc):
        from flexviz.trace.geo_hist2d import GeoHistogram2D

        self._roundtrip(
            tmp_path,
            self._df(),
            lambda: GeoHistogram2D(
                lat="lat",
                lon="lon",
                lat_bins=6,
                lon_bins=6,
                z="zv",
                histfunc=histfunc,
            ),
        )

    @pytest.mark.parametrize("bin_boundaries", ["data", "viewport"])
    def test_map_viewport_matches(self, tmp_path, bin_boundaries):
        from flexviz.trace.geo_hist2d import GeoHistogram2D

        viewports = {
            "coordinates": [[3.5, 41.0], [9.0, 41.0], [9.0, 48.5], [3.5, 48.5]]
        }
        self._roundtrip(
            tmp_path,
            self._df(),
            lambda: GeoHistogram2D(
                lat="lat",
                lon="lon",
                lat_bins=8,
                lon_bins=8,
                bin_boundaries=bin_boundaries,
            ),
            viewports,
        )

    def test_scan_selects_the_streaming_plan(self):
        from flexviz.trace.geo_hist2d import GeoHistogram2D

        g = GeoHistogram2D(lat="lat", lon="lon")
        assert g.get_aggregation_spec({}, scan_source=False).plan is None
        assert g.get_aggregation_spec({}, scan_source=True).plan is not None


# ---- descending viewport ranges ---------------------------------------------


class TestDescendingViewportRanges:
    """A reversed plotly axis reports its viewport high-to-low, verbatim.

    The engine normalizes at ingestion (`_normalize_axis_ranges`): without it,
    `is_between` masks return nothing (blank chart) and the sorted-x
    `search_sorted` slice returns rows from the wrong end. A descending range
    must therefore produce exactly what its ascending twin produces — for
    every downsampler and on both the mask and slice paths.
    """

    N = 5_000

    @classmethod
    def _df(cls) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "ts": list(range(cls.N)),
                "val": [float((i * 37) % 101) for i in range(cls.N)],
            }
        )

    @classmethod
    def _line_updates(
        cls,
        x_range,
        downsample: str = "minmax",
        sorted_x: bool = False,
        via_state: bool = False,
    ) -> dict:
        lf = LFQueryBuilder(cls._df())
        if sorted_x:
            lf.assume_sorted("ts")
        line = LinePlot(x="ts", y="val", n_points=100, downsample=downsample)
        engine = FlexEngine(backend_lf=lf, scalable_traces={line.uid: line})
        infos = [
            TraceInfo(uid=line.uid, axes=("x", "y"), trace_type="line", figure_uid="f1")
        ]
        if via_state:
            # Stored per-figure viewports (state.viewport) take the same
            # normalization as viewport events.
            event = InteractionEvent(type="init", force_update=True)
            deltas = engine.process(
                event, infos, viewports_by_figure={"f1": {"x": x_range}}
            )
        else:
            event = InteractionEvent(type="viewport", axis_ranges={"x": x_range})
            deltas = engine.process(event, infos)
        assert len(deltas) == 1
        return {k: list(v) for k, v in deltas[0].updates.items()}

    @pytest.mark.parametrize("downsample", ["minmax", "fpcs", "nth"])
    @pytest.mark.parametrize("sorted_x", [False, True])
    def test_descending_equals_ascending(self, downsample, sorted_x):
        asc = self._line_updates((100, 900), downsample, sorted_x)
        desc = self._line_updates((900, 100), downsample, sorted_x)
        assert len(asc["x"]) > 0, "ascending baseline must not be empty"
        assert desc == asc

    def test_stored_viewport_descending(self):
        asc = self._line_updates((100, 900), sorted_x=True, via_state=True)
        desc = self._line_updates((900, 100), sorted_x=True, via_state=True)
        assert len(asc["x"]) > 0
        assert desc == asc

    def test_histogram_descending_equals_ascending(self):
        def updates(x_range):
            lf = LFQueryBuilder(self._df())
            hist = Histogram(x="val", bins=20)
            engine = FlexEngine(backend_lf=lf, scalable_traces={hist.uid: hist})
            infos = [TraceInfo(uid=hist.uid, axes=("x", "y"), trace_type="histogram")]
            event = InteractionEvent(type="viewport", axis_ranges={"x": x_range})
            deltas = engine.process(event, infos)
            assert len(deltas) == 1
            return {k: list(v) for k, v in deltas[0].updates.items()}

        asc = updates((20.0, 80.0))
        desc = updates((80.0, 20.0))
        assert len(asc["x"]) > 0
        assert desc == asc

    def test_normalize_helper_shapes(self):
        from flexviz.engine import _normalize_axis_ranges

        got = _normalize_axis_ranges(
            {
                "a": (9, 1),
                "b": [3.5, 1.5],
                "c": (1, 9),
                "d": None,
                "e": ("2024-03-01", "2024-01-01"),  # ISO strings order like dates
                "f": [[1.0, 2.0], [3.0, 4.0]],  # map coordinate points
                "g": ("a", 1),  # incomparable — left as sent
                "h": (None, 5),
            }
        )
        assert got["a"] == (1, 9)
        assert got["b"] == (1.5, 3.5)
        assert got["c"] == (1, 9)
        assert got["d"] is None
        assert got["e"] == ("2024-01-01", "2024-03-01")
        assert got["f"] == [[1.0, 2.0], [3.0, 4.0]]
        assert got["g"] == ("a", 1)
        assert got["h"] == (None, 5)

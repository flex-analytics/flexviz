"""Engine benchmarks on a resident 1M-row frame."""

from __future__ import annotations

import polars as pl
import pytest
from conftest import N, has_data, init_call
from ooc_child import TRACES

from flexviz.engine import FlexEngine, TraceInfo
from flexviz.events import ActiveSource, InteractionEvent
from flexviz.LF import LFQueryBuilder
from flexviz.spec import ClauseFilter, SelectionPredicate, SelectionState
from flexviz.trace.bar import BarPlot
from flexviz.trace.hist import Histogram

# Middle 1% of x, the zoom depth where downsampling has the least to cut.
ZOOM = [N * 0.495, N * 0.505]


@pytest.mark.parametrize("name", list(TRACES))
def test_init(benchmark, frame: pl.DataFrame, name: str) -> None:
    engine, event, infos = init_call(LFQueryBuilder(frame), TRACES[name]())
    deltas = benchmark(engine.process, event, infos)
    assert has_data(deltas)


@pytest.mark.parametrize(
    ("name", "axis_range"),
    [
        ("line-minmax", ZOOM),
        ("line-grouped", ZOOM),
        # The histogram bins ``y`` (standard normal), so its x axis is in sigma.
        ("hist", [-0.5, 0.5]),
        ("hist2d", ZOOM),
    ],
)
def test_viewport(
    benchmark, frame: pl.DataFrame, name: str, axis_range: list[float]
) -> None:
    engine, _, infos = init_call(LFQueryBuilder(frame), TRACES[name]())
    event = InteractionEvent(type="viewport", axis_ranges={"x": axis_range})
    deltas = benchmark(engine.process, event, infos)
    assert has_data(deltas)


def test_selection_cross_filter(benchmark, frame: pl.DataFrame) -> None:
    """Brush on a line figure, re-aggregate the histogram in another figure."""
    line = TRACES["line-minmax"]()
    hist = Histogram(x="y", bins=200)
    engine = FlexEngine(
        backend_lf=LFQueryBuilder(frame),
        scalable_traces={line.uid: line, hist.uid: hist},
    )
    infos = [
        TraceInfo(line.uid, line._axes, line.trace_type, figure_uid="fig_a"),
        TraceInfo(hist.uid, hist._axes, hist.trace_type, figure_uid="fig_b"),
    ]
    event = InteractionEvent(
        type="selection",
        force_update=True,
        selections=[
            SelectionState(
                source_figure_uid="fig_a",
                predicates=[
                    SelectionPredicate(
                        clauses=[ClauseFilter(column="x", range=(N // 3, 2 * N // 3))]
                    )
                ],
            )
        ],
    )
    deltas = benchmark(engine.process, event, infos)
    assert has_data(deltas)
    assert {d.uid for d in deltas} == {hist.uid}


def test_build_cubes(benchmark, frame: pl.DataFrame) -> None:
    """Categorical bar source, histogram target."""
    bar = BarPlot(labels="g", values="y", agg="mean")
    hist = Histogram(x="y", bins=200)
    engine = FlexEngine(
        backend_lf=LFQueryBuilder(frame),
        scalable_traces={bar.uid: bar, hist.uid: hist},
        source_name="bench",
    )
    infos = [
        TraceInfo(bar.uid, bar._axes, bar.trace_type, figure_uid="fig_a"),
        TraceInfo(hist.uid, hist._axes, hist.trace_type, figure_uid="fig_b"),
    ]
    active = ActiveSource(figure_uid="fig_a", column="g", trace_uid=bar.uid)
    cubes, trace_cubes = benchmark(engine.build_cubes, infos, {}, [], active)
    assert cubes and trace_cubes == {hist.uid: 0}

"""Engine event benchmarks: the interaction loop a dashboard actually runs.

Each benchmark measures one ``FlexEngine.process`` call for a realistic
multi-figure dashboard — the path taken by every zoom, brush, and reset the
browser sends.
"""

from __future__ import annotations

import polars as pl

from flexviz.cache import InMemoryLRUCache
from flexviz.engine import FlexEngine, TraceInfo
from flexviz.events import InteractionEvent
from flexviz.LF import LFQueryBuilder
from flexviz.spec import ClauseFilter, SelectionPredicate, SelectionState
from flexviz.trace.bar import BarPlot
from flexviz.trace.hist import Histogram
from flexviz.trace.line import LinePlot

FIG_A = "fig_a"
FIG_B = "fig_b"


def _dashboard(df: pl.DataFrame) -> tuple[FlexEngine, list[TraceInfo]]:
    """Two linked figures: a line + histogram on one, a line + bar on the other."""
    traces = [
        LinePlot(x="ts", y="val", n_points=1_000),
        Histogram(x="val", bins=50),
        LinePlot(x="ts", y="val2", n_points=1_000),
        BarPlot(labels="cat", values="val", agg="mean"),
    ]
    registry = {t.uid: t for t in traces}
    engine = FlexEngine(backend_lf=LFQueryBuilder(df), scalable_traces=registry)
    figures = (FIG_A, FIG_A, FIG_B, FIG_B)
    infos = [
        TraceInfo(
            uid=t.uid,
            axes=tuple(t._axes) if t._axes else None,
            trace_type=t.trace_type,
            figure_uid=fig,
        )
        for t, fig in zip(traces, figures)
    ]
    return engine, infos


def test_init(benchmark, numeric_df: pl.DataFrame) -> None:
    """Cold load: every trace on the dashboard aggregates its full column."""
    engine, infos = _dashboard(numeric_df)
    deltas = benchmark(
        engine.process, InteractionEvent(type="init", force_update=True), infos
    )
    assert len(deltas) == 4


def test_viewport(benchmark, numeric_df: pl.DataFrame) -> None:
    """Zoom on one figure: only the x-bound traces re-aggregate."""
    engine, infos = _dashboard(numeric_df)
    window = [50_000, 120_000]
    event = InteractionEvent(
        type="viewport", axis_ranges={"x": window}, figure_uid=FIG_A
    )
    deltas = benchmark(engine.process, event, infos, {FIG_A: {"x": window}})
    assert len(deltas) > 0


def test_selection_crossfilter(benchmark, numeric_df: pl.DataFrame) -> None:
    """Brush on figure A: figure B re-aggregates behind the resulting filter."""
    engine, infos = _dashboard(numeric_df)
    event = InteractionEvent(
        type="selection",
        force_update=True,
        figure_uid=FIG_A,
        selections=[
            SelectionState(
                source_figure_uid=FIG_A,
                predicates=[
                    SelectionPredicate(
                        clauses=[ClauseFilter(column="ts", range=(40_000, 120_000))]
                    )
                ],
            )
        ],
    )
    deltas = benchmark(engine.process, event, infos)
    assert len(deltas) == 2


def test_selection_multi_clause(benchmark, numeric_df: pl.DataFrame) -> None:
    """A compound brush: a range clause ANDed with a categorical value set."""
    engine, infos = _dashboard(numeric_df)
    event = InteractionEvent(
        type="selection",
        force_update=True,
        figure_uid=FIG_A,
        selections=[
            SelectionState(
                source_figure_uid=FIG_A,
                predicates=[
                    SelectionPredicate(
                        clauses=[
                            ClauseFilter(column="ts", range=(40_000, 160_000)),
                            ClauseFilter(
                                column="cat", values=["cat_0", "cat_3", "cat_7"]
                            ),
                        ]
                    )
                ],
            )
        ],
    )
    deltas = benchmark(engine.process, event, infos)
    assert len(deltas) == 2


def test_init_cached(benchmark, numeric_df: pl.DataFrame) -> None:
    """A warm cache hit: the engine must serve the unfiltered base from memory."""
    traces = [LinePlot(x="ts", y="val", n_points=1_000), Histogram(x="val", bins=50)]
    registry = {t.uid: t for t in traces}
    engine = FlexEngine(
        backend_lf=LFQueryBuilder(numeric_df),
        scalable_traces=registry,
        cache_backend=InMemoryLRUCache(max_entries=64),
        source_name="bench_source",
    )
    infos = [
        TraceInfo(uid=t.uid, axes=("x", "y"), trace_type=t.trace_type, figure_uid=FIG_A)
        for t in traces
    ]
    event = InteractionEvent(type="init", force_update=True)
    engine.process(event, infos)  # prime the cache
    deltas = benchmark(engine.process, event, infos)
    assert len(deltas) == 2

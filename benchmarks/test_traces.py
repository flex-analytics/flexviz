"""Per-trace aggregation benchmarks.

One benchmark per trace type, each measuring a single ``FlexEngine.process``
call on an unfiltered frame — the work a client pays for when a figure first
appears. Every trace here is the scalable (server-side aggregated) variant, so
the measurement covers the Polars plan, the Rust kernels, and the Python-side
delta construction.
"""

from __future__ import annotations

import polars as pl
import pytest
from conftest import build_engine

from flexviz.events import InteractionEvent
from flexviz.trace.bar import BarPlot
from flexviz.trace.box import BoxPlot
from flexviz.trace.corr_heatmap import CorrHeatmap
from flexviz.trace.geo_hist2d import GeoHistogram2D
from flexviz.trace.geo_line import GeoLine
from flexviz.trace.hist import Histogram
from flexviz.trace.hist2d import Histogram2D
from flexviz.trace.line import LinePlot
from flexviz.trace.pie import PiePlot
from flexviz.trace.treemap import TreeMap

INIT = InteractionEvent(type="init", force_update=True)


def _run(benchmark, df: pl.DataFrame, trace) -> None:
    engine, infos = build_engine(df, [trace])
    deltas = benchmark(engine.process, INIT, infos)
    assert len(deltas) == 1


# ---- line ------------------------------------------------------------------


@pytest.mark.parametrize("downsample", ["minmax", "lttb", "fpcs", "nth"])
def test_line(benchmark, numeric_df: pl.DataFrame, downsample: str) -> None:
    """Line downsampling: the Rust min/max bucket kernel and its alternatives."""
    _run(
        benchmark,
        numeric_df,
        LinePlot(x="ts", y="val", n_points=1_000, downsample=downsample),
    )


def test_line_grouped(benchmark, numeric_df: pl.DataFrame) -> None:
    """One downsampled series per category, fused into a single grouped plan."""
    _run(
        benchmark,
        numeric_df,
        LinePlot(x="ts", y="val", n_points=500, group_by="cat"),
    )


# ---- distributions ---------------------------------------------------------


@pytest.mark.parametrize("bins", [20, 200])
def test_histogram(benchmark, numeric_df: pl.DataFrame, bins: int) -> None:
    _run(benchmark, numeric_df, Histogram(x="val", bins=bins))


def test_histogram_grouped(benchmark, numeric_df: pl.DataFrame) -> None:
    _run(benchmark, numeric_df, Histogram(x="val", bins=50, group_by="cat"))


def test_histogram2d(benchmark, numeric_df: pl.DataFrame) -> None:
    _run(
        benchmark,
        numeric_df,
        Histogram2D(x="val", y="val2", x_bins=64, y_bins=64),
    )


def test_boxplot(benchmark, numeric_df: pl.DataFrame) -> None:
    """Quantile computation over the full column."""
    _run(benchmark, numeric_df, BoxPlot(y="val"))


# ---- categorical -----------------------------------------------------------


def test_bar(benchmark, numeric_df: pl.DataFrame) -> None:
    _run(benchmark, numeric_df, BarPlot(labels="cat", values="val", agg="mean"))


def test_pie(benchmark, numeric_df: pl.DataFrame) -> None:
    _run(benchmark, numeric_df, PiePlot(labels="cat", values="val", agg="sum"))


def test_treemap(benchmark, numeric_df: pl.DataFrame) -> None:
    """Hierarchical group-by over a two-level path."""
    _run(benchmark, numeric_df, TreeMap(path=["region", "cat"], values="val"))


def test_corr_heatmap(benchmark, numeric_df: pl.DataFrame) -> None:
    _run(benchmark, numeric_df, CorrHeatmap(columns=["val", "val2", "val3"]))


# ---- geo -------------------------------------------------------------------


def test_geo_histogram2d(benchmark, numeric_df: pl.DataFrame) -> None:
    _run(benchmark, numeric_df, GeoHistogram2D(lat="lat", lon="lon"))


def test_geo_line(benchmark, numeric_df: pl.DataFrame) -> None:
    _run(benchmark, numeric_df, GeoLine(lat="lat", lon="lon", n_points=1_000))

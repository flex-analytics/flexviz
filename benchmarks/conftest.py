"""Shared data and engine fixtures for the CodSpeed benchmark suite.

Frames are built once per session: the benchmarks measure FlexViz, not
``numpy`` random generation. 200k rows is deliberately modest — large enough
that the Polars/Rust work dominates Python overhead, small enough that a
CPU-simulated run of the whole suite stays in the low minutes.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from flexviz.engine import FlexEngine, TraceInfo
from flexviz.LF import LFQueryBuilder
from flexviz.trace.base import FlexTrace

# Row counts. The line/histogram paths scan the whole column, so they set the
# cost of every benchmark in this suite.
N_ROWS = 200_000
N_CATEGORIES = 12


@pytest.fixture(scope="session")
def numeric_df() -> pl.DataFrame:
    """A sorted time series with numeric, categorical, and geo columns."""
    rng = np.random.default_rng(0)
    n = N_ROWS
    value = np.sin(np.arange(n) / 5e3) + rng.standard_normal(n) * 0.05
    return pl.DataFrame(
        {
            "ts": np.arange(n, dtype=np.int64),
            "val": value,
            "val2": np.cumsum(rng.standard_normal(n)) / 100.0,
            "val3": rng.standard_normal(n) * 3.0,
            "cat": rng.integers(0, N_CATEGORIES, n),
            "region": rng.integers(0, 4, n),
            "lat": rng.uniform(-60.0, 60.0, n),
            "lon": rng.uniform(-170.0, 170.0, n),
        }
    ).with_columns(
        pl.col("cat")
        .cast(pl.Utf8)
        .replace_strict({str(i): f"cat_{i}" for i in range(N_CATEGORIES)}),
        pl.col("region")
        .cast(pl.Utf8)
        .replace_strict({"0": "north", "1": "south", "2": "east", "3": "west"}),
    )


@pytest.fixture(scope="session")
def numeric_lf(numeric_df: pl.DataFrame) -> pl.LazyFrame:
    return numeric_df.lazy()


def build_engine(
    df: pl.DataFrame, traces: list[FlexTrace], figure_uid: str = "fig"
) -> tuple[FlexEngine, list[TraceInfo]]:
    """Wire ``traces`` onto ``df`` and return the engine and its trace infos."""
    registry = {trace.uid: trace for trace in traces}
    engine = FlexEngine(backend_lf=LFQueryBuilder(df), scalable_traces=registry)
    infos = [
        TraceInfo(
            uid=trace.uid,
            axes=tuple(trace._axes) if trace._axes else None,
            trace_type=trace.trace_type,
            figure_uid=figure_uid,
        )
        for trace in traces
    ]
    return engine, infos

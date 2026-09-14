"""Shared fixtures for the CodSpeed benchmarks (run with ``make bench``).

This directory sits outside ``testpaths``, so ``make test`` never collects it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

# The trace registry lives next to the out-of-core worker, which is not a
# package; put its directory on the path the way pytest does for tests/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from ooc_child import TRACES

from flexviz.engine import FlexEngine, TraceInfo
from flexviz.events import InteractionEvent, TraceDelta
from flexviz.LF import LFQueryBuilder

N = 1_000_000
WARMUP_ROWS = 1_000


def make_frame(n: int) -> pl.DataFrame:
    """The ``ooc_child`` columns: sorted x, normal y/z, lat/lon, 10 string groups."""
    rng = np.random.default_rng(0)
    return pl.DataFrame(
        {
            "x": np.arange(n, dtype=np.int64),
            "y": rng.standard_normal(n),
            "z": rng.standard_normal(n),
            "lat": rng.uniform(-60, 60, n),
            "lon": rng.uniform(-160, 160, n),
            "g": (np.arange(n) % 10).astype(str),
        }
    )


def init_call(lf: LFQueryBuilder, trace) -> tuple[FlexEngine, InteractionEvent, list]:
    """``(engine, event, infos)`` for one trace's init aggregation."""
    engine = FlexEngine(backend_lf=lf, scalable_traces={trace.uid: trace})
    infos = [TraceInfo(uid=trace.uid, axes=trace._axes, trace_type=trace.trace_type)]
    return engine, InteractionEvent(type="init", force_update=True), infos


def has_data(deltas: list[TraceDelta]) -> bool:
    """True when every delta carries axis updates or grouped children."""
    return bool(deltas) and all(d.updates or d.group_results for d in deltas)


@pytest.fixture(scope="session")
def frame() -> pl.DataFrame:
    return make_frame(N)


@pytest.fixture(scope="session", autouse=True)
def _warmup() -> None:
    """One untimed init per trace: imports and first-touch costs must not land
    inside the first timed benchmark."""
    small = make_frame(WARMUP_ROWS)
    for factory in TRACES.values():
        engine, event, infos = init_call(LFQueryBuilder(small), factory())
        engine.process(event, infos)

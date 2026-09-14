"""Init benchmarks on a Parquet scan: the streaming plans and batch folds that
a resident frame never takes."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from conftest import has_data, init_call
from ooc_child import TRACES

from flexviz.LF import LFQueryBuilder

SCAN_TRACES = ["line-minmax", "hist", "hist2d", "line-grouped"]


@pytest.fixture(scope="session")
def scan_path(frame: pl.DataFrame, tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("bench") / "frame.parquet"
    frame.write_parquet(path, compression="zstd")
    return path


@pytest.mark.parametrize("name", SCAN_TRACES)
def test_scan_init(benchmark, scan_path: Path, name: str) -> None:
    lf = LFQueryBuilder(pl.scan_parquet(scan_path))
    assert lf.is_scan
    engine, event, infos = init_call(lf, TRACES[name]())
    deltas = benchmark(engine.process, event, infos)
    assert has_data(deltas)

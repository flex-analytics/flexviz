"""Shared pytest fixtures for flexviz tests."""

from __future__ import annotations

import socket
import threading
import time
from typing import Generator

import polars as pl
import pytest
from fastapi.testclient import TestClient

from flexviz.LF import LFQueryBuilder
from flexviz.server import app, register_source
from flexviz.trace.box import BoxPlot
from flexviz.trace.hist import Histogram
from flexviz.trace.line import LinePlot


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_browser_server(port: int) -> None:
    """Start the FlexViz FastAPI server on *port* in a daemon thread."""
    df = pl.DataFrame({"ts": list(range(500)), "val": [float(i) for i in range(500)]})
    register_source("_browser_test", df)

    import uvicorn

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"Server did not start on port {port}")


@pytest.fixture(scope="module")
def server_port() -> Generator[int, None, None]:
    port = _free_port()
    _start_browser_server(port)
    yield port


# ---- DataFrames -----------------------------------------------------------


@pytest.fixture(scope="session")
def small_df() -> pl.DataFrame:
    """1 000-row DataFrame with sorted ``ts`` and ``val`` columns."""
    n = 1_000
    return pl.DataFrame({"ts": list(range(n)), "val": [float(i) for i in range(n)]})


@pytest.fixture(scope="session")
def large_df() -> pl.DataFrame:
    """100 000-row DataFrame with sorted ``ts`` and ``val`` columns."""
    n = 100_000
    return pl.DataFrame({"ts": list(range(n)), "val": [float(i) for i in range(n)]})


# ---- LFQueryBuilder -------------------------------------------------------


@pytest.fixture()
def backend_lf(small_df: pl.DataFrame) -> LFQueryBuilder:
    return LFQueryBuilder(small_df)


@pytest.fixture(scope="session")
def grouped_df() -> pl.DataFrame:
    """500-row DataFrame with ``ts``, ``val``, and ``cat`` (A/B) columns for GroupBy tests."""
    n = 500
    return pl.DataFrame(
        {
            "ts": list(range(n)),
            "val": [float(i) for i in range(n)],
            "cat": ["A" if i % 2 == 0 else "B" for i in range(n)],
        }
    )


@pytest.fixture()
def grouped_backend_lf(grouped_df: pl.DataFrame) -> LFQueryBuilder:
    return LFQueryBuilder(grouped_df)


# ---- Trace instances -------------------------------------------------------


@pytest.fixture()
def line_trace() -> LinePlot:
    return LinePlot(x="ts", y="val", name="line", n_points=100)


@pytest.fixture()
def hist_trace() -> Histogram:
    return Histogram(x="val", bins=20, histnorm="count", name="hist")


@pytest.fixture()
def box_trace() -> BoxPlot:
    return BoxPlot(y="val", name="box")


# ---- FastAPI TestClient ----------------------------------------------------

_API_SOURCE = "_pytest_source"


@pytest.fixture(scope="session")
def api_client(small_df: pl.DataFrame) -> TestClient:
    register_source(_API_SOURCE, small_df)
    return TestClient(app)


@pytest.fixture(scope="session")
def api_source_name() -> str:
    return _API_SOURCE

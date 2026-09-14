"""``POST /update`` benchmarks: engine time plus request parsing and delta
serialisation."""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest
from conftest import N
from fastapi.testclient import TestClient

from flexviz.figure import Figure
from flexviz.server import app, register_source

SOURCE = "_bench_source"
ZOOM = [N * 0.495, N * 0.505]


@pytest.fixture(scope="session")
def client(frame: pl.DataFrame) -> TestClient:
    register_source(SOURCE, frame)
    return TestClient(app)


@pytest.fixture(scope="session")
def spec(frame: pl.DataFrame) -> dict[str, Any]:
    fig = Figure(frame)
    fig.add_line(x="x", y="y")
    return fig.to_spec(source=SOURCE).model_dump(mode="json")


def _assert_delta(resp) -> None:
    assert resp.status_code == 200
    deltas = resp.json()["deltas"]
    assert deltas and deltas[0]["updates"]["x"]


def test_update_init(benchmark, client: TestClient, spec: dict[str, Any]) -> None:
    body = {"spec": spec, "event": {"type": "init", "force_update": True}}
    _assert_delta(benchmark(client.post, "/update", json=body))


def test_update_viewport(benchmark, client: TestClient, spec: dict[str, Any]) -> None:
    body = {"spec": spec, "event": {"type": "viewport", "axis_ranges": {"x": ZOOM}}}
    _assert_delta(benchmark(client.post, "/update", json=body))

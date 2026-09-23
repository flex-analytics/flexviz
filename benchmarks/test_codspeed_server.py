"""``POST /dashboard/update`` benchmarks: engine time plus request parsing and
delta serialisation."""

from __future__ import annotations

import copy
from typing import Any

import polars as pl
import pytest
from conftest import N
from fastapi.testclient import TestClient

from flexviz.dashboard import Dashboard
from flexviz.server import app, register_source

SOURCE = "_bench_source"
ZOOM = [N * 0.495, N * 0.505]


@pytest.fixture(scope="session")
def client(frame: pl.DataFrame) -> TestClient:
    register_source(SOURCE, frame)
    return TestClient(app)


@pytest.fixture(scope="session")
def spec(frame: pl.DataFrame) -> dict[str, Any]:
    dash = Dashboard(frame)
    dash.add_figure().add_line(x="x", y="y")
    return dash.to_spec(source_name=SOURCE).model_dump(mode="json")


def _assert_delta(resp, spec: dict[str, Any]) -> None:
    assert resp.status_code == 200
    deltas = resp.json()["figure_deltas"][spec["figures"][0]["uid"]]
    assert deltas and deltas[0]["updates"]["x"]


def test_update_init(benchmark, client: TestClient, spec: dict[str, Any]) -> None:
    body = {"spec": spec, "event": {"type": "init", "force_update": True}}
    _assert_delta(benchmark(client.post, "/dashboard/update", json=body), spec)


def test_update_viewport(benchmark, client: TestClient, spec: dict[str, Any]) -> None:
    key = f"{spec['figures'][0]['uid']}/x"
    zoomed = copy.deepcopy(spec)
    zoomed["state"]["viewport"] = {key: {"min": ZOOM[0], "max": ZOOM[1]}}
    body = {"spec": zoomed, "event": {"type": "viewport", "viewport_keys": [key]}}
    _assert_delta(benchmark(client.post, "/dashboard/update", json=body), spec)

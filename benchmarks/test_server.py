"""End-to-end HTTP benchmarks through the FastAPI app.

These go through the whole request path a browser hits: pydantic validation of
the spec, trace reconstruction, aggregation, and JSON serialisation of the
deltas. They are the closest proxy in this suite for perceived interaction
latency.
"""

from __future__ import annotations

import polars as pl
import pytest
from fastapi.testclient import TestClient

from flexviz.dashboard import Dashboard
from flexviz.figure import Figure
from flexviz.server import app, register_source
from flexviz.spec import decode_spec, encode_spec

SOURCE = "bench_source"


@pytest.fixture(scope="session")
def client(numeric_df: pl.DataFrame) -> TestClient:
    register_source(SOURCE, numeric_df)
    return TestClient(app)


@pytest.fixture(scope="session")
def figure_spec(numeric_df: pl.DataFrame):
    fig = Figure(numeric_df)
    fig.add_line(x="ts", y="val", n_points=1_000)
    fig.add_histogram(x="val", bins=50)
    return fig.to_spec(source=SOURCE)


@pytest.fixture(scope="session")
def dashboard_spec(numeric_df: pl.DataFrame):
    dash = Dashboard(numeric_df)
    fig_a = dash.add_figure(title="signal")
    fig_a.add_line(x="ts", y="val", n_points=1_000)
    fig_b = dash.add_figure(title="distribution")
    fig_b.add_histogram(x="val", bins=50)
    fig_c = dash.add_figure(title="categories")
    fig_c.add_bar(labels="cat", values="val", agg="mean")
    return dash.to_spec(source_name=SOURCE)


def test_update_init(benchmark, client: TestClient, figure_spec) -> None:
    """POST /update for a cold single figure with two traces."""
    payload = {
        "spec": figure_spec.model_dump(),
        "event": {
            "type": "init",
            "axis_ranges": {},
            "selections": [],
            "force_update": True,
        },
    }
    response = benchmark(client.post, "/update", json=payload)
    assert response.status_code == 200
    assert len(response.json()["deltas"]) == 2


def test_update_viewport(benchmark, client: TestClient, figure_spec) -> None:
    """POST /update for a zoom: the re-aggregation path on a narrower window."""
    payload = {
        "spec": figure_spec.model_dump(),
        "event": {
            "type": "viewport",
            "axis_ranges": {"x": [50_000, 120_000]},
            "selections": [],
            "force_update": False,
        },
    }
    response = benchmark(client.post, "/update", json=payload)
    assert response.status_code == 200


def test_dashboard_update_selection(
    benchmark, client: TestClient, dashboard_spec
) -> None:
    """POST /dashboard/update for a brush cross-filtering two other figures."""
    source_uid = dashboard_spec.figures[0].uid
    payload = {
        "spec": dashboard_spec.model_dump(),
        "event": {
            "type": "selection",
            "axis_ranges": {},
            "selections": [
                {
                    "source_figure_uid": source_uid,
                    "predicates": [
                        {"clauses": [{"column": "ts", "range": [40_000, 120_000]}]}
                    ],
                }
            ],
            "force_update": True,
            "figure_uid": source_uid,
        },
    }
    response = benchmark(client.post, "/dashboard/update", json=payload)
    assert response.status_code == 200
    assert len(response.json()["figure_deltas"]) == 3


def test_build_dashboard_spec(benchmark, numeric_df: pl.DataFrame) -> None:
    """Python-side spec construction: three figures wired to the same source."""

    def build():
        dash = Dashboard(numeric_df)
        for title, add in (
            ("signal", lambda f: f.add_line(x="ts", y="val", n_points=1_000)),
            ("distribution", lambda f: f.add_histogram(x="val", bins=50)),
            ("categories", lambda f: f.add_bar(labels="cat", values="val")),
        ):
            add(dash.add_figure(title=title))
        return dash.to_spec(source_name=SOURCE)

    spec = benchmark(build)
    assert len(spec.figures) == 3


def test_spec_roundtrip(benchmark, dashboard_spec) -> None:
    """Share-link encoding: the spec is encoded into, and read back from, a URL."""

    def roundtrip():
        return decode_spec(encode_spec(dashboard_spec))

    decoded = benchmark(roundtrip)
    assert decoded is not None

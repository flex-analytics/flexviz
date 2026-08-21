"""Integration tests using FastAPI TestClient.

All tests in this module are marked with ``@pytest.mark.integration``.
"""

from __future__ import annotations

import json

import polars as pl
import pytest
from fastapi.testclient import TestClient

from flexviz.dashboard import Dashboard
from flexviz.figure import Figure
from flexviz.server import app, register_source
from flexviz.spec import (
    DashboardSpec,
    GridItem,
    LayoutSpec,
    SelectionState,
    VisualizationSpec,
    decode_spec,
    encode_spec,
)

pytestmark = pytest.mark.integration


# ---- fixtures local to this module ----------------------------------------

_SRC = "_integ_src"


@pytest.fixture(scope="module")
def integ_df() -> pl.DataFrame:
    n = 5_000
    return pl.DataFrame({"ts": list(range(n)), "val": [float(i) for i in range(n)]})


@pytest.fixture(scope="module")
def client(integ_df: pl.DataFrame) -> TestClient:
    register_source(_SRC, integ_df)
    return TestClient(app)


# ---- helpers ---------------------------------------------------------------


def _make_single_figure_payload(
    df: pl.DataFrame,
    event_type: str = "init",
    force: bool = True,
    axis_ranges: dict | None = None,
    n_points: int = 200,
) -> dict:
    fig = Figure(df)
    fig.add_line(x="ts", y="val", n_points=n_points)
    spec = fig.to_spec(source=_SRC)
    return {
        "spec": spec.model_dump(),
        "event": {
            "type": event_type,
            "axis_ranges": axis_ranges or {},
            "selections": [],
            "force_update": force,
        },
    }


def _make_dashboard_and_spec(
    df: pl.DataFrame,
    n_figures: int = 2,
) -> tuple[Dashboard, DashboardSpec]:
    dash = Dashboard(df)
    for i in range(n_figures):
        fig = dash.add_figure(title=f"Fig{i}")
        fig.add_line(x="ts", y="val", name=f"L{i}", n_points=200)
    return dash, dash.to_spec(source_name=_SRC)


# ===========================================================================
# POST /update
# ===========================================================================


class TestPostUpdate:
    def test_init_returns_xy_data(self, client: TestClient, integ_df: pl.DataFrame):
        payload = _make_single_figure_payload(integ_df)
        resp = client.post("/update", json=payload)
        assert resp.status_code == 200
        deltas = resp.json()["deltas"]
        assert len(deltas) == 1
        assert len(deltas[0]["updates"]["x"]) > 0
        assert len(deltas[0]["updates"]["y"]) > 0

    def test_viewport_returns_fewer_points(
        self, client: TestClient, integ_df: pl.DataFrame
    ):
        init_payload = _make_single_figure_payload(integ_df, n_points=5000)
        init_resp = client.post("/update", json=init_payload)
        init_count = len(init_resp.json()["deltas"][0]["updates"]["x"])

        vp_payload = _make_single_figure_payload(
            integ_df,
            event_type="viewport",
            force=False,
            axis_ranges={"x": [1000, 2000]},
            n_points=5000,
        )
        vp_resp = client.post("/update", json=vp_payload)
        vp_count = len(vp_resp.json()["deltas"][0]["updates"]["x"])

        assert vp_count < init_count

    def test_selection_returns_no_deltas_for_single_figure(
        self, client: TestClient, integ_df: pl.DataFrame
    ):
        fig = Figure(integ_df)
        fig.add_line(x="ts", y="val", n_points=1000)
        spec = fig.to_spec(source=_SRC)
        payload = {
            "spec": spec.model_dump(),
            "event": {
                "type": "selection",
                "axis_ranges": {},
                "selections": [
                    {
                        "source_figure_uid": spec.figure.uid,
                        "predicates": [
                            {"clauses": [{"column": "ts", "range": [100, 300]}]}
                        ],
                    }
                ],
                "force_update": True,
                "figure_uid": spec.figure.uid,
            },
        }
        resp = client.post("/update", json=payload)
        assert resp.status_code == 200
        assert resp.json()["deltas"] == []


class TestViewportStateHelpers:
    def test_figure_viewport_from_state_preserves_geo_coordinates(self):
        from flexviz.server import _figure_viewport_from_state
        from flexviz.spec import InteractionState

        coords = [[-73.5, 40.5], [-72.5, 40.5], [-72.5, 41.5], [-73.5, 41.5]]
        state = InteractionState(
            viewport={
                "fig-a/x": {"min": 100, "max": 300},
                "fig-a/coordinates": coords,
                "fig-b/x": {"min": 10, "max": 20},
            }
        )

        viewport = _figure_viewport_from_state(state, "fig-a")

        assert viewport["x"] == (100, 300)
        assert viewport["coordinates"] == coords
        assert "fig-b/x" not in viewport


# ===========================================================================
# POST /dashboard/update
# ===========================================================================


class TestPostDashboardUpdate:
    def test_two_figures_keyed_by_uid(self, client: TestClient, integ_df: pl.DataFrame):
        _, spec = _make_dashboard_and_spec(integ_df)
        expected_uids = {f.uid for f in spec.figures}

        payload = {
            "spec": spec.model_dump(),
            "event": {
                "type": "init",
                "axis_ranges": {},
                "selections": [],
                "force_update": True,
            },
        }
        resp = client.post("/dashboard/update", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert set(body["figure_deltas"].keys()) == expected_uids
        for deltas in body["figure_deltas"].values():
            assert len(deltas) > 0
            assert "trace_index" not in deltas[0]

    def test_viewport_only_updates_named_figure(
        self, client: TestClient, integ_df: pl.DataFrame
    ):
        _, spec = _make_dashboard_and_spec(integ_df)
        fig_a_uid = spec.figures[0].uid
        fig_b_uid = spec.figures[1].uid

        payload = {
            "spec": spec.model_dump(),
            "event": {
                "type": "viewport",
                "axis_ranges": {"x": [1000, 2000]},
                "selections": [],
                "force_update": False,
                "figure_uid": fig_a_uid,
            },
        }
        resp = client.post("/dashboard/update", json=payload)
        body = resp.json()
        assert len(body["figure_deltas"].get(fig_a_uid, [])) > 0
        assert len(body["figure_deltas"].get(fig_b_uid, [])) == 0

    def test_reset_only_updates_named_figure(
        self, client: TestClient, integ_df: pl.DataFrame
    ):
        _, spec = _make_dashboard_and_spec(integ_df)
        fig_a_uid = spec.figures[0].uid
        fig_b_uid = spec.figures[1].uid

        payload = {
            "spec": spec.model_dump(),
            "event": {
                "type": "reset",
                "axis_ranges": {},
                "selections": [],
                "force_update": True,
                "figure_uid": fig_a_uid,
            },
        }
        resp = client.post("/dashboard/update", json=payload)
        body = resp.json()
        assert len(body["figure_deltas"].get(fig_a_uid, [])) > 0
        assert len(body["figure_deltas"].get(fig_b_uid, [])) == 0

    def test_selection_excludes_source_figure(
        self, client: TestClient, integ_df: pl.DataFrame
    ):
        dash = Dashboard(integ_df)
        for name in ("A", "B", "C"):
            f = dash.add_figure()
            f.add_line(x="ts", y="val", name=name)
        spec = dash.to_spec(source_name=_SRC)
        fig_a_uid, fig_b_uid, fig_c_uid = [f.uid for f in spec.figures]

        payload = {
            "spec": spec.model_dump(),
            "event": {
                "type": "selection",
                "axis_ranges": {},
                "selections": [
                    {
                        "source_figure_uid": fig_b_uid,
                        "predicates": [
                            {"clauses": [{"column": "ts", "range": [1000, 3000]}]}
                        ],
                    }
                ],
                "force_update": True,
                "figure_uid": fig_b_uid,
            },
        }
        resp = client.post("/dashboard/update", json=payload)
        body = resp.json()
        assert len(body["figure_deltas"].get(fig_a_uid, [])) > 0
        assert len(body["figure_deltas"].get(fig_b_uid, [])) == 0
        assert len(body["figure_deltas"].get(fig_c_uid, [])) > 0

    def test_multi_source_selection_excludes_all(
        self, client: TestClient, integ_df: pl.DataFrame
    ):
        dash = Dashboard(integ_df)
        for name in ("A", "B", "C", "D"):
            f = dash.add_figure()
            f.add_line(x="ts", y="val", name=name)
        spec = dash.to_spec(source_name=_SRC)
        uids = [f.uid for f in spec.figures]

        payload = {
            "spec": spec.model_dump(),
            "event": {
                "type": "selection",
                "axis_ranges": {},
                "selections": [
                    {
                        "source_figure_uid": uids[1],
                        "predicates": [
                            {"clauses": [{"column": "ts", "range": [0, 4999]}]}
                        ],
                    },
                    {
                        "source_figure_uid": uids[2],
                        "predicates": [
                            {"clauses": [{"column": "ts", "range": [0, 4999]}]}
                        ],
                    },
                ],
                "force_update": True,
                "figure_uid": None,
            },
        }
        resp = client.post("/dashboard/update", json=payload)
        body = resp.json()
        assert len(body["figure_deltas"].get(uids[0], [])) > 0
        assert len(body["figure_deltas"].get(uids[1], [])) == 0
        assert len(body["figure_deltas"].get(uids[2], [])) == 0
        assert len(body["figure_deltas"].get(uids[3], [])) > 0

    def test_cross_filter_uses_source_column(self, client: TestClient):
        n = 5_000
        df = pl.DataFrame(
            {
                "src_id": list(range(n)),
                "tgt_id": [i * 2 for i in range(n)],
                "val": [float(i) for i in range(n)],
            }
        )
        src_name = "_integ_cross_col"
        register_source(src_name, df)

        dash = Dashboard(df)
        fig_a = dash.add_figure()
        fig_a.add_line(x="src_id", y="val", name="A")
        fig_b = dash.add_figure()
        fig_b.add_line(x="tgt_id", y="val", name="B")
        spec = dash.to_spec(source_name=src_name)
        fig_a_uid, fig_b_uid = spec.figures[0].uid, spec.figures[1].uid

        payload = {
            "spec": spec.model_dump(),
            "event": {
                "type": "selection",
                "axis_ranges": {},
                "selections": [
                    {
                        "source_figure_uid": fig_a_uid,
                        "predicates": [
                            {"clauses": [{"column": "src_id", "range": [1000, 3000]}]}
                        ],
                    }
                ],
                "force_update": True,
                "figure_uid": fig_a_uid,
            },
        }
        resp = client.post("/dashboard/update", json=payload)
        body = resp.json()
        assert len(body["figure_deltas"].get(fig_a_uid, [])) == 0
        b_deltas = body["figure_deltas"].get(fig_b_uid, [])
        assert len(b_deltas) > 0
        x_vals = b_deltas[0]["updates"]["x"]
        assert min(x_vals) >= 2000

    def test_two_step_selection_excludes_both(
        self, client: TestClient, integ_df: pl.DataFrame
    ):
        dash = Dashboard(integ_df)
        for name in ("A", "B", "C", "D"):
            f = dash.add_figure()
            f.add_line(x="ts", y="val", name=name)
        spec = dash.to_spec(source_name=_SRC)
        uids = [f.uid for f in spec.figures]

        b_sel = {
            "source_figure_uid": uids[1],
            "predicates": [{"clauses": [{"column": "ts", "range": [0, 4999]}]}],
        }
        c_sel = {
            "source_figure_uid": uids[2],
            "predicates": [{"clauses": [{"column": "ts", "range": [0, 4999]}]}],
        }

        # Step 2: both B and C selections active
        spec_with_b = spec.model_copy(deep=True)
        spec_with_b.state.selections = [SelectionState(**b_sel)]

        payload = {
            "spec": spec_with_b.model_dump(),
            "event": {
                "type": "selection",
                "axis_ranges": {},
                "selections": [b_sel, c_sel],
                "force_update": True,
                "figure_uid": None,
            },
        }
        resp = client.post("/dashboard/update", json=payload)
        body = resp.json()
        assert len(body["figure_deltas"].get(uids[0], [])) > 0
        assert len(body["figure_deltas"].get(uids[1], [])) == 0
        assert len(body["figure_deltas"].get(uids[2], [])) == 0
        assert len(body["figure_deltas"].get(uids[3], [])) > 0

    def test_deselect_clears_filter(self, client: TestClient, integ_df: pl.DataFrame):
        """After a selection narrows figure B, deselect must restore full data."""
        dash = Dashboard(integ_df)
        for i in range(2):
            f = dash.add_figure()
            f.add_line(x="ts", y="val", n_points=10_000)
        spec = dash.to_spec(source_name=_SRC)
        fig_a_uid = spec.figures[0].uid
        fig_b_uid = spec.figures[1].uid

        # Step 1: init to get the unfiltered count for figure B.
        init_payload = {
            "spec": spec.model_dump(),
            "event": {
                "type": "init",
                "axis_ranges": {},
                "selections": [],
                "force_update": True,
            },
        }
        init_resp = client.post("/dashboard/update", json=init_payload)
        init_b_count = len(
            init_resp.json()["figure_deltas"][fig_b_uid][0]["updates"]["x"]
        )

        # Step 2: selection on A narrows B to a small range.
        sel_payload = {
            "spec": spec.model_dump(),
            "event": {
                "type": "selection",
                "axis_ranges": {},
                "selections": [
                    {
                        "source_figure_uid": fig_a_uid,
                        "predicates": [
                            {"clauses": [{"column": "ts", "range": [1000, 2000]}]}
                        ],
                    }
                ],
                "force_update": True,
                "figure_uid": None,
            },
        }
        sel_resp = client.post("/dashboard/update", json=sel_payload)
        sel_b_count = len(
            sel_resp.json()["figure_deltas"][fig_b_uid][0]["updates"]["x"]
        )
        assert sel_b_count < init_b_count

        # Step 3: deselect must restore B to its unfiltered count.
        desel_payload = {
            "spec": spec.model_dump(),
            "event": {
                "type": "deselect",
                "axis_ranges": {},
                "selections": [],
                "force_update": True,
                "figure_uid": None,
            },
        }
        desel_resp = client.post("/dashboard/update", json=desel_payload)
        desel_b_count = len(
            desel_resp.json()["figure_deltas"][fig_b_uid][0]["updates"]["x"]
        )
        assert desel_b_count == init_b_count

    def test_histogram_source_cross_filters_line(
        self, client: TestClient, integ_df: pl.DataFrame
    ):
        """Selection on a histogram figure must cross-filter a line figure."""
        src = "_integ_hist_cross"
        register_source(src, integ_df)

        dash = Dashboard(integ_df)
        fig_hist = dash.add_figure(title="Hist")
        fig_hist.add_histogram(x="val", bins=20)
        fig_line = dash.add_figure(title="Line")
        fig_line.add_line(x="ts", y="val", n_points=5000)
        spec = dash.to_spec(source_name=src)
        hist_uid = spec.figures[0].uid
        line_uid = spec.figures[1].uid

        payload = {
            "spec": spec.model_dump(),
            "event": {
                "type": "selection",
                "axis_ranges": {},
                "selections": [
                    {
                        "source_figure_uid": hist_uid,
                        "predicates": [
                            {"clauses": [{"column": "ts", "range": [1000, 3000]}]}
                        ],
                    }
                ],
                "force_update": True,
                "figure_uid": None,
            },
        }
        resp = client.post("/dashboard/update", json=payload)
        body = resp.json()
        assert len(body["figure_deltas"].get(hist_uid, [])) == 0
        line_deltas = body["figure_deltas"].get(line_uid, [])
        assert len(line_deltas) > 0
        y_vals = line_deltas[0]["updates"]["y"]
        assert all(1000 <= v <= 3000 for v in y_vals)

    def test_selection_uses_stored_target_viewport(
        self, client: TestClient, integ_df: pl.DataFrame
    ):
        from flexviz.spec import AxisRange

        dash = Dashboard(integ_df)
        for _ in range(2):
            f = dash.add_figure()
            f.add_line(x="ts", y="val", n_points=10_000)
        spec = dash.to_spec(source_name=_SRC)
        uid_a, uid_b = [f.uid for f in spec.figures]
        spec.state.viewport = {f"{uid_b}/x": AxisRange(min=200, max=400)}

        payload = {
            "spec": spec.model_dump(),
            "event": {
                "type": "selection",
                "axis_ranges": {},
                "selections": [
                    {
                        "source_figure_uid": uid_a,
                        "predicates": [
                            {"clauses": [{"column": "ts", "range": [100, 300]}]}
                        ],
                    }
                ],
                "force_update": True,
                "figure_uid": uid_a,
            },
        }
        resp = client.post("/dashboard/update", json=payload)
        assert resp.status_code == 200
        x_vals = resp.json()["figure_deltas"][uid_b][0]["updates"]["x"]
        assert all(200 <= v <= 300 for v in x_vals)

    def test_overlay_selection_returns_fg_only(
        self, client: TestClient, integ_df: pl.DataFrame
    ):
        dash = Dashboard(integ_df)
        for _ in range(2):
            f = dash.add_figure()
            f.add_line(x="ts", y="val", n_points=10_000)
        spec = dash.to_spec(source_name=_SRC)
        uid_a, uid_b = [f.uid for f in spec.figures]
        spec.state.cross_filter_mode = "overlay"

        payload = {
            "spec": spec.model_dump(),
            "event": {
                "type": "selection",
                "axis_ranges": {},
                "selections": [
                    {
                        "source_figure_uid": uid_a,
                        "predicates": [
                            {"clauses": [{"column": "ts", "range": [1000, 2000]}]}
                        ],
                    }
                ],
                "force_update": True,
                "figure_uid": uid_a,
            },
        }
        resp = client.post("/dashboard/update", json=payload)
        assert resp.status_code == 200
        body = resp.json()["figure_deltas"]
        assert body[uid_a] == []
        layers = {d["layer"] for d in body[uid_b]}
        assert layers == {"fg"}
        fg = next(d for d in body[uid_b] if d["layer"] == "fg")
        assert all(1000 <= v <= 2000 for v in fg["updates"]["x"])

    def test_overlay_viewport_with_active_selection_returns_bg_and_fg(
        self, client: TestClient, integ_df: pl.DataFrame
    ):
        dash = Dashboard(integ_df)
        for _ in range(2):
            f = dash.add_figure()
            f.add_line(x="ts", y="val", n_points=10_000)
        spec = dash.to_spec(source_name=_SRC)
        uid_a, uid_b = [f.uid for f in spec.figures]
        spec.state.cross_filter_mode = "overlay"

        payload = {
            "spec": spec.model_dump(),
            "event": {
                "type": "viewport",
                "axis_ranges": {"x": [1000, 3000]},
                "selections": [
                    {
                        "source_figure_uid": uid_a,
                        "predicates": [
                            {"clauses": [{"column": "ts", "range": [1500, 2000]}]}
                        ],
                    }
                ],
                "force_update": True,
                "figure_uid": uid_b,
            },
        }
        resp = client.post("/dashboard/update", json=payload)
        assert resp.status_code == 200
        body = resp.json()["figure_deltas"]
        layers = {d["layer"] for d in body[uid_b]}
        assert layers == {"bg", "fg"}
        bg = next(d for d in body[uid_b] if d["layer"] == "bg")
        fg = next(d for d in body[uid_b] if d["layer"] == "fg")
        assert all(1000 <= v <= 3000 for v in bg["updates"]["x"])
        assert all(1500 <= v <= 2000 for v in fg["updates"]["x"])

    def test_overlay_deselect_returns_bg_only(
        self, client: TestClient, integ_df: pl.DataFrame
    ):
        dash = Dashboard(integ_df)
        for _ in range(2):
            f = dash.add_figure()
            f.add_line(x="ts", y="val", n_points=10_000)
        spec = dash.to_spec(source_name=_SRC)
        spec.state.cross_filter_mode = "overlay"

        payload = {
            "spec": spec.model_dump(),
            "event": {
                "type": "deselect",
                "axis_ranges": {},
                "selections": [],
                "force_update": True,
            },
        }
        resp = client.post("/dashboard/update", json=payload)
        assert resp.status_code == 200
        for deltas in resp.json()["figure_deltas"].values():
            assert {d["layer"] for d in deltas} == {"bg"}

    def test_overlay_reset_returns_bg_only(
        self, client: TestClient, integ_df: pl.DataFrame
    ):
        dash = Dashboard(integ_df)
        for _ in range(2):
            f = dash.add_figure()
            f.add_line(x="ts", y="val", n_points=10_000)
        spec = dash.to_spec(source_name=_SRC)
        spec.state.cross_filter_mode = "overlay"

        payload = {
            "spec": spec.model_dump(),
            "event": {
                "type": "reset",
                "axis_ranges": {},
                "selections": [],
                "force_update": True,
            },
        }
        resp = client.post("/dashboard/update", json=payload)
        assert resp.status_code == 200
        for deltas in resp.json()["figure_deltas"].values():
            assert {d["layer"] for d in deltas} == {"bg"}

    def test_overlay_init_emits_bg_only(
        self, client: TestClient, integ_df: pl.DataFrame
    ):
        dash = Dashboard(integ_df)
        for _ in range(2):
            f = dash.add_figure()
            f.add_line(x="ts", y="val", n_points=10_000)
        spec = dash.to_spec(source_name=_SRC)
        spec.state.cross_filter_mode = "overlay"

        payload = {
            "spec": spec.model_dump(),
            "event": {
                "type": "init",
                "axis_ranges": {},
                "selections": [],
                "force_update": True,
            },
        }
        resp = client.post("/dashboard/update", json=payload)
        assert resp.status_code == 200
        for deltas in resp.json()["figure_deltas"].values():
            assert all(d["layer"] == "bg" for d in deltas)

    def test_overlay_viewport_no_sel_emits_bg_only(
        self, client: TestClient, integ_df: pl.DataFrame
    ):
        dash = Dashboard(integ_df)
        for _ in range(2):
            f = dash.add_figure()
            f.add_line(x="ts", y="val", n_points=10_000)
        spec = dash.to_spec(source_name=_SRC)
        spec.state.cross_filter_mode = "overlay"

        payload = {
            "spec": spec.model_dump(),
            "event": {
                "type": "viewport",
                "axis_ranges": {"x": [1000, 4000]},
                "selections": [],
                "force_update": True,
            },
        }
        resp = client.post("/dashboard/update", json=payload)
        assert resp.status_code == 200
        for deltas in resp.json()["figure_deltas"].values():
            assert all(d["layer"] == "bg" for d in deltas)

    def test_update_mode_no_layer_field(
        self, client: TestClient, integ_df: pl.DataFrame
    ):
        dash = Dashboard(integ_df)
        for _ in range(2):
            f = dash.add_figure()
            f.add_line(x="ts", y="val", n_points=10_000)
        spec = dash.to_spec(source_name=_SRC)
        uid_a, uid_b = [f.uid for f in spec.figures]

        payload = {
            "spec": spec.model_dump(),
            "event": {
                "type": "selection",
                "axis_ranges": {},
                "selections": [
                    {
                        "source_figure_uid": uid_a,
                        "predicates": [
                            {"clauses": [{"column": "ts", "range": [1000, 2000]}]}
                        ],
                    }
                ],
                "force_update": True,
                "figure_uid": uid_a,
            },
        }
        resp = client.post("/dashboard/update", json=payload)
        assert resp.status_code == 200
        body = resp.json()["figure_deltas"]
        for deltas in body.values():
            assert all(d.get("layer") is None for d in deltas)

    def test_box_trace_cross_filter(self, client: TestClient, integ_df: pl.DataFrame):
        from flexviz.spec import DashboardSpec, InteractionState

        fig_line = Figure(integ_df)
        fig_line.add_line(x="ts", y="val", n_points=10_000)

        fig_box = Figure(integ_df)
        fig_box.add_boxplot(y="val")

        line_spec = fig_line.to_spec(source=_SRC)
        box_spec = fig_box.to_spec(source=_SRC)

        dash_spec = DashboardSpec(
            figures=[line_spec.figure, box_spec.figure],
            state=InteractionState(),
        )
        uid_line = line_spec.figure.uid
        uid_box = box_spec.figure.uid

        # Init: get full-data box median
        init_payload = {
            "spec": dash_spec.model_dump(),
            "event": {
                "type": "init",
                "axis_ranges": {},
                "selections": [],
                "force_update": True,
            },
        }
        init_resp = client.post("/dashboard/update", json=init_payload)
        assert init_resp.status_code == 200
        box_init_deltas = init_resp.json()["figure_deltas"][uid_box]
        assert len(box_init_deltas) == 1

        # Selection on line: ts in [0, 999] — should filter box trace
        sel_payload = {
            "spec": dash_spec.model_dump(),
            "event": {
                "type": "selection",
                "axis_ranges": {},
                "force_update": True,
                "figure_uid": uid_line,
                "selections": [
                    {
                        "source_figure_uid": uid_line,
                        "predicates": [
                            {"clauses": [{"column": "ts", "range": [0, 999]}]}
                        ],
                    }
                ],
            },
        }
        sel_resp = client.post("/dashboard/update", json=sel_payload)
        assert sel_resp.status_code == 200
        box_sel_deltas = sel_resp.json()["figure_deltas"][uid_box]
        assert len(box_sel_deltas) == 1
        # Median of [0..999] is ~499.5, median of full [0..4999] is ~2499.5
        assert (
            box_sel_deltas[0]["updates"]["median"]
            < box_init_deltas[0]["updates"]["median"]
        )

    def test_empty_selection_no_filter(
        self, client: TestClient, integ_df: pl.DataFrame
    ):
        dash = Dashboard(integ_df)
        for _ in range(2):
            f = dash.add_figure()
            f.add_line(x="ts", y="val", n_points=10_000)
        spec = dash.to_spec(source_name=_SRC)
        uid_a, uid_b = [f.uid for f in spec.figures]

        # Selection with null ranges — treated as no filter
        payload = {
            "spec": spec.model_dump(),
            "event": {
                "type": "selection",
                "axis_ranges": {},
                "selections": [
                    {
                        "source_figure_uid": uid_a,
                        "predicates": [],
                    }
                ],
                "force_update": True,
                "figure_uid": uid_a,
            },
        }
        resp = client.post("/dashboard/update", json=payload)
        assert resp.status_code == 200
        body = resp.json()["figure_deltas"]
        # target (uid_b) should get full data — same as init
        assert len(body[uid_b]) >= 1

    def test_histogram_overlay_bg_fg_counts(
        self, client: TestClient, integ_df: pl.DataFrame
    ):
        from flexviz.spec import DashboardSpec, InteractionState

        fig_line = Figure(integ_df)
        fig_line.add_line(x="ts", y="val", n_points=10_000)

        fig_hist = Figure(integ_df)
        fig_hist.add_histogram(x="val", bins=20)

        line_spec = fig_line.to_spec(source=_SRC)
        hist_spec = fig_hist.to_spec(source=_SRC)

        dash_spec = DashboardSpec(
            figures=[line_spec.figure, hist_spec.figure],
            state=InteractionState(cross_filter_mode="overlay"),
        )
        uid_line = line_spec.figure.uid
        uid_hist = hist_spec.figure.uid

        payload = {
            "spec": dash_spec.model_dump(),
            "event": {
                "type": "viewport",
                "axis_ranges": {"x": [0, 4999]},
                "selections": [
                    {
                        "source_figure_uid": uid_line,
                        "predicates": [
                            {"clauses": [{"column": "ts", "range": [0, 999]}]}
                        ],
                    }
                ],
                "force_update": True,
                "figure_uid": uid_hist,
            },
        }
        resp = client.post("/dashboard/update", json=payload)
        assert resp.status_code == 200
        hist_deltas = resp.json()["figure_deltas"][uid_hist]
        layers = {d["layer"] for d in hist_deltas}
        assert layers == {"bg", "fg"}
        bg = next(d for d in hist_deltas if d["layer"] == "bg")
        fg = next(d for d in hist_deltas if d["layer"] == "fg")
        bg_total = sum(bg["updates"]["y"])
        fg_total = sum(fg["updates"]["y"])
        assert bg_total > fg_total


# ===========================================================================
# 3-figure cross-filter scenarios
# ===========================================================================


class TestCrossFilterScenarios:
    """Targeted cross-filter scenarios with 3 figures (A, B, C)."""

    @pytest.fixture()
    def three_fig_spec(self, integ_df: pl.DataFrame):
        dash = Dashboard(integ_df)
        for name in ("A", "B", "C"):
            f = dash.add_figure()
            f.add_line(x="ts", y="val", name=name, n_points=10_000)
        return dash.to_spec(source_name=_SRC)

    def test_selection_on_A_updates_B_and_C(
        self, client: TestClient, three_fig_spec: DashboardSpec
    ):
        spec = three_fig_spec
        uid_a, uid_b, uid_c = [f.uid for f in spec.figures]

        payload = {
            "spec": spec.model_dump(),
            "event": {
                "type": "selection",
                "axis_ranges": {},
                "selections": [
                    {
                        "source_figure_uid": uid_a,
                        "predicates": [
                            {"clauses": [{"column": "ts", "range": [1000, 3000]}]}
                        ],
                    }
                ],
                "force_update": True,
                "figure_uid": uid_a,
            },
        }
        resp = client.post("/dashboard/update", json=payload)
        body = resp.json()
        assert len(body["figure_deltas"].get(uid_a, [])) == 0
        assert len(body["figure_deltas"].get(uid_b, [])) > 0
        assert len(body["figure_deltas"].get(uid_c, [])) > 0
        b_x = body["figure_deltas"][uid_b][0]["updates"]["x"]
        c_x = body["figure_deltas"][uid_c][0]["updates"]["x"]
        assert min(b_x) >= 1000 and max(b_x) <= 3000
        assert min(c_x) >= 1000 and max(c_x) <= 3000

    def test_selection_on_A_then_B_only_updates_C(
        self, client: TestClient, three_fig_spec: DashboardSpec
    ):
        spec = three_fig_spec
        uid_a, uid_b, uid_c = [f.uid for f in spec.figures]

        sel_a = {
            "source_figure_uid": uid_a,
            "predicates": [{"clauses": [{"column": "ts", "range": [1000, 3000]}]}],
        }
        sel_b = {
            "source_figure_uid": uid_b,
            "predicates": [{"clauses": [{"column": "ts", "range": [1500, 2500]}]}],
        }
        payload = {
            "spec": spec.model_dump(),
            "event": {
                "type": "selection",
                "axis_ranges": {},
                "selections": [sel_a, sel_b],
                "force_update": True,
                "figure_uid": None,
            },
        }
        resp = client.post("/dashboard/update", json=payload)
        body = resp.json()
        assert len(body["figure_deltas"].get(uid_a, [])) == 0
        assert len(body["figure_deltas"].get(uid_b, [])) == 0
        assert len(body["figure_deltas"].get(uid_c, [])) > 0

    def test_deselect_restores_all_keeps_zoom(
        self, client: TestClient, three_fig_spec: DashboardSpec
    ):
        spec = three_fig_spec
        uid_a, uid_b, uid_c = [f.uid for f in spec.figures]

        # Step 1: init to get the full data count for B.
        init_resp = client.post(
            "/dashboard/update",
            json={
                "spec": spec.model_dump(),
                "event": {
                    "type": "init",
                    "axis_ranges": {},
                    "selections": [],
                    "force_update": True,
                },
            },
        )
        init_b_count = len(init_resp.json()["figure_deltas"][uid_b][0]["updates"]["x"])

        # Step 2: select on A to narrow B and C.
        sel_payload = {
            "spec": spec.model_dump(),
            "event": {
                "type": "selection",
                "axis_ranges": {},
                "selections": [
                    {
                        "source_figure_uid": uid_a,
                        "predicates": [
                            {"clauses": [{"column": "ts", "range": [1000, 2000]}]}
                        ],
                    }
                ],
                "force_update": True,
                "figure_uid": uid_a,
            },
        }
        sel_resp = client.post("/dashboard/update", json=sel_payload)
        sel_b_count = len(sel_resp.json()["figure_deltas"][uid_b][0]["updates"]["x"])
        assert sel_b_count < init_b_count

        # Step 3: deselect must restore B and C to full data.
        desel_resp = client.post(
            "/dashboard/update",
            json={
                "spec": spec.model_dump(),
                "event": {
                    "type": "deselect",
                    "axis_ranges": {},
                    "selections": [],
                    "force_update": True,
                },
            },
        )
        desel_b_count = len(
            desel_resp.json()["figure_deltas"][uid_b][0]["updates"]["x"]
        )
        assert desel_b_count == init_b_count

    def test_reset_clears_zoom_and_filter(
        self, client: TestClient, three_fig_spec: DashboardSpec
    ):
        spec = three_fig_spec
        uid_a, uid_b, uid_c = [f.uid for f in spec.figures]

        # Init first to get full count.
        init_resp = client.post(
            "/dashboard/update",
            json={
                "spec": spec.model_dump(),
                "event": {
                    "type": "init",
                    "axis_ranges": {},
                    "selections": [],
                    "force_update": True,
                },
            },
        )
        init_b_count = len(init_resp.json()["figure_deltas"][uid_b][0]["updates"]["x"])

        # Zoom figure A + cross-filter from A.
        zoomed = {
            "spec": spec.model_dump(),
            "event": {
                "type": "selection",
                "axis_ranges": {"x": [500, 1500]},
                "selections": [
                    {
                        "source_figure_uid": uid_a,
                        "predicates": [
                            {"clauses": [{"column": "ts", "range": [500, 1500]}]}
                        ],
                    }
                ],
                "force_update": True,
                "figure_uid": uid_a,
            },
        }
        client.post("/dashboard/update", json=zoomed)

        # Global reset must return all figures with full data.
        reset_resp = client.post(
            "/dashboard/update",
            json={
                "spec": spec.model_dump(),
                "event": {
                    "type": "reset",
                    "axis_ranges": {},
                    "selections": [],
                    "force_update": True,
                },
            },
        )
        body = reset_resp.json()
        for uid in (uid_a, uid_b, uid_c):
            deltas = body["figure_deltas"].get(uid, [])
            assert len(deltas) > 0, f"Expected deltas for {uid} after reset"
        reset_b_count = len(body["figure_deltas"][uid_b][0]["updates"]["x"])
        assert reset_b_count == init_b_count


# ===========================================================================
# POST /share + GET /view
# ===========================================================================


class TestShareAndView:
    def test_share_returns_url(self, client: TestClient):
        viz = VisualizationSpec()
        spec_dict = json.loads(viz.model_dump_json())
        resp = client.post(
            "/share", json={"spec": spec_dict, "server_url": "http://testserver"}
        )
        assert resp.status_code == 200
        url = resp.json()["url"]
        assert url.startswith("http://testserver/view?spec=")

        encoded = url.split("spec=", 1)[1]
        decoded = decode_spec(encoded)
        assert isinstance(decoded, VisualizationSpec)

    def test_share_with_path_server_url_returns_relative_url(self, client: TestClient):
        """Deployed pages bake in a path-only server_url (e.g. "/demo4");
        /share echoes it and the toolbar absolutizes client-side."""
        viz = VisualizationSpec()
        spec_dict = json.loads(viz.model_dump_json())
        resp = client.post("/share", json={"spec": spec_dict, "server_url": "/demo4"})
        assert resp.status_code == 200
        assert resp.json()["url"].startswith("/demo4/view?spec=")

    def test_view_embeds_page_relative_server_url(self, client: TestClient):
        """Behind a prefix-stripping reverse proxy (demo deployment) the app
        cannot know its external base URL, so /view must embed a page-relative
        SERVER_URL — never one reconstructed from request.base_url."""
        import re

        encoded = encode_spec(VisualizationSpec())
        resp = client.get(f"/view?spec={encoded}")
        assert resp.status_code == 200
        m = re.search(r'const SERVER_URL\s*=\s*("[^"]*")', resp.text)
        assert m is not None, "SERVER_URL constant missing from /view HTML"
        assert m.group(1) == '"."'

    @pytest.mark.parametrize("renderer", ["plotly", "echarts"])
    def test_view_viz_spec(self, client: TestClient, renderer: str):
        from flexviz.spec import FigureSpec, TraceSpec

        ts = TraceSpec(
            uid="t1", trace_type="line", backend_data={"x": "ts", "y": "val"}
        )
        viz = VisualizationSpec(figure=FigureSpec(uid="f1", source="s", traces=[ts]))
        encoded = encode_spec(viz)
        resp = client.get(f"/view?spec={encoded}&renderer={renderer}")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert renderer in resp.text.lower()

    @pytest.mark.parametrize("renderer", ["plotly", "echarts"])
    def test_view_dashboard_spec(self, client: TestClient, renderer: str):
        from flexviz.spec import (
            DashboardSpec,
            FigureSpec,
            InteractionState,
            LayoutSpec,
            TraceSpec,
        )

        ts1 = TraceSpec(
            uid="t1", trace_type="line", backend_data={"x": "ts", "y": "val"}
        )
        ts2 = TraceSpec(uid="t2", trace_type="histogram", backend_data={"x": "val"})
        dash_spec = DashboardSpec(
            figures=[
                FigureSpec(uid="f1", source="s", traces=[ts1]),
                FigureSpec(uid="f2", source="s", traces=[ts2]),
            ],
            state=InteractionState(),
            layout=LayoutSpec(),
        )
        encoded = encode_spec(dash_spec)
        resp = client.get(f"/view?spec={encoded}&renderer={renderer}")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        if dash_spec.layout.draggable:
            assert "grid-stack" in resp.text
        else:
            assert "fv-dashboard" in resp.text

    def test_view_rejects_unsupported_renderer_trace_combo(self, client: TestClient):
        from flexviz.spec import FigureSpec, TraceSpec, VisualizationSpec

        ts = TraceSpec(
            uid="geo-1",
            trace_type="geo_histogram2d",
            backend_data={"lon": "lon", "lat": "lat"},
        )
        viz = VisualizationSpec(figure=FigureSpec(uid="map-1", source="s", traces=[ts]))
        encoded = encode_spec(viz)
        resp = client.get(f"/view?spec={encoded}&renderer=echarts")
        assert resp.status_code == 400
        assert "renderer='echarts'" in resp.json()["detail"]
        assert "geo_histogram2d" in resp.json()["detail"]
        assert "Use 'plotly'" in resp.json()["detail"]


# ===========================================================================
# Share/restore state preservation
# ===========================================================================


class TestShareStatePreservation:
    """Verify that encode/decode round-trips preserve viewport and selections."""

    def test_viewport_preserved(self):
        from flexviz.spec import AxisRange, InteractionState

        state = InteractionState(
            viewport={"f1/x": {"min": 100, "max": 500}},
            selections=[],
        )
        dash = DashboardSpec(
            figures=[],
            state=state,
        )
        encoded = encode_spec(dash)
        decoded = decode_spec(encoded)
        vp = decoded.state.viewport["f1/x"]
        assert vp == AxisRange(min=100, max=500)
        assert decoded.state.selections == []

    def test_selections_preserved(self):
        from flexviz.spec import (
            ClauseFilter,
            InteractionState,
            SelectionPredicate,
        )

        sel = SelectionState(
            source_figure_uid="fig-a",
            predicates=[
                SelectionPredicate(
                    clauses=[ClauseFilter(column="ts", range=(1000, 3000))]
                )
            ],
        )
        state = InteractionState(selections=[sel])
        dash = DashboardSpec(figures=[], state=state)
        encoded = encode_spec(dash)
        decoded = decode_spec(encoded)
        assert len(decoded.state.selections) == 1
        restored = decoded.state.selections[0]
        assert tuple(restored.predicates[0].clauses[0].range) == (1000.0, 3000.0)
        assert restored.source_figure_uid == "fig-a"

    def test_viewport_and_selections_preserved(self):
        from flexviz.spec import (
            AxisRange,
            ClauseFilter,
            InteractionState,
            SelectionPredicate,
        )

        sel = SelectionState(
            source_figure_uid="fig-b",
            predicates=[
                SelectionPredicate(
                    clauses=[ClauseFilter(column="ts", range=(500, 1500))]
                )
            ],
        )
        state = InteractionState(
            viewport={
                "fig-a/x": {"min": 0, "max": 4999},
                "fig-b/x": {"min": 200, "max": 800},
            },
            selections=[sel],
        )
        dash = DashboardSpec(figures=[], state=state)
        encoded = encode_spec(dash)
        decoded = decode_spec(encoded)
        assert decoded.state.viewport["fig-a/x"] == AxisRange(min=0, max=4999)
        assert decoded.state.viewport["fig-b/x"] == AxisRange(min=200, max=800)
        assert len(decoded.state.selections) == 1
        assert decoded.state.selections[0].source_figure_uid == "fig-b"

    def test_cross_filter_effect_preserved_after_share_roundtrip(
        self, client: TestClient, integ_df: pl.DataFrame
    ):
        """Sharing a spec with an active selection must preserve cross-filter behaviour."""
        from flexviz.spec import InteractionState

        dash = Dashboard(integ_df)
        fig_a = dash.add_figure(title="A")
        fig_a.add_line(x="ts", y="val", n_points=10_000)
        fig_b = dash.add_figure(title="B")
        fig_b.add_line(x="ts", y="val", n_points=10_000)
        spec = dash.to_spec(source_name=_SRC)
        uid_a, uid_b = [f.uid for f in spec.figures]

        # Activate a cross-filter selection on figure A in the spec state,
        # mirroring what the browser does before calling /share.
        from flexviz.spec import ClauseFilter, SelectionPredicate

        sel = SelectionState(
            source_figure_uid=uid_a,
            predicates=[
                SelectionPredicate(
                    clauses=[ClauseFilter(column="ts", range=(1000, 2000))]
                )
            ],
        )
        spec.state = InteractionState(selections=[sel])

        # Encode/decode via the share helpers to simulate /share → /view.
        encoded = encode_spec(spec)
        decoded = decode_spec(encoded)
        assert len(decoded.figures) == 2
        assert len(decoded.state.selections) == 1

        # Step 1: init to capture the unfiltered count for figure B.
        init_resp = client.post(
            "/dashboard/update",
            json={
                "spec": decoded.model_dump(),
                "event": {
                    "type": "init",
                    "axis_ranges": {},
                    "selections": [],
                    "force_update": True,
                },
            },
        )
        init_b_count = len(init_resp.json()["figure_deltas"][uid_b][0]["updates"]["x"])

        # Step 2: replay the saved selection state as a selection event,
        # as done by the adapters' initialisation JS.
        selection_payload = {
            "spec": decoded.model_dump(),
            "event": {
                "type": "selection",
                "axis_ranges": {},
                "selections": [
                    {
                        "source_figure_uid": s.source_figure_uid,
                        "predicates": [p.model_dump() for p in s.predicates],
                    }
                    for s in decoded.state.selections
                ],
                "force_update": True,
                "figure_uid": uid_a,
            },
        }
        sel_resp = client.post("/dashboard/update", json=selection_payload)
        sel_b_count = len(sel_resp.json()["figure_deltas"][uid_b][0]["updates"]["x"])

        assert sel_b_count < init_b_count


# ===========================================================================
# GET /sources
# ===========================================================================


class TestGetSources:
    def test_returns_200(self, client: TestClient):
        resp = client.get("/sources")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


# ===========================================================================
# Figure / Dashboard save/load round-trip
# ===========================================================================


class TestFigureSaveLoad:
    def test_save_load_roundtrip(self, tmp_path, integ_df: pl.DataFrame):
        fig = Figure(integ_df)
        fig.add_line(x="ts", y="val", name="Test", n_points=200)
        fig.update_layout(title="My Chart")

        path = str(tmp_path / "chart.json")
        fig.save_spec(path, source="mydata")

        loaded = Figure.load_spec(path)
        assert isinstance(loaded, VisualizationSpec)
        assert loaded.figure.uid == fig._uid
        assert loaded.figure.traces[0].params["n_points"] == 200
        assert loaded.figure.layout["title"] == "My Chart"


class TestDashboardSaveLoad:
    def test_save_load_roundtrip(self, tmp_path, integ_df: pl.DataFrame):
        dash = Dashboard(integ_df)
        fig1 = dash.add_figure(title="F1")
        fig1.add_line(x="ts", y="val")
        fig2 = dash.add_figure(title="F2")
        fig2.add_histogram(x="val")

        layout = LayoutSpec(
            grid_items=[
                GridItem(fig_uid=fig1._uid, x=0, y=0, w=6, h=5),
                GridItem(fig_uid=fig2._uid, x=6, y=0, w=6, h=5),
            ]
        )
        path = str(tmp_path / "dashboard.json")
        dash.save_spec(path, source_name="mydata", layout=layout)

        loaded = Dashboard.load_spec(path)
        assert isinstance(loaded, DashboardSpec)
        assert len(loaded.figures) == 2
        assert loaded.figures[0].uid == fig1._uid
        assert loaded.figures[1].uid == fig2._uid
        assert loaded.layout.grid_items is not None
        assert loaded.layout.grid_items[0].w == 6


# ===========================================================================
# Spec round-trip with state (zoom, cross-filter, combined)
# ===========================================================================


class TestSpecRoundTripWithState:
    """save_spec → load_spec preserves viewport and selection state."""

    def test_dashboard_with_zoom(self, tmp_path, integ_df: pl.DataFrame):
        from flexviz.spec import AxisRange, InteractionState

        dash = Dashboard(integ_df)
        fig1 = dash.add_figure(title="F1")
        fig1.add_line(x="ts", y="val")
        spec = dash.to_spec(source_name="s")
        spec.state = InteractionState(
            viewport={f"{spec.figures[0].uid}/x": {"min": 100, "max": 500}},
        )

        path = str(tmp_path / "zoom.json")
        import pathlib

        pathlib.Path(path).write_text(spec.model_dump_json(indent=2), encoding="utf-8")
        loaded = Dashboard.load_spec(path)
        assert loaded.state.viewport[f"{spec.figures[0].uid}/x"] == AxisRange(
            min=100, max=500
        )

    def test_dashboard_with_cross_filter(self, tmp_path, integ_df: pl.DataFrame):
        from flexviz.spec import (
            ClauseFilter,
            InteractionState,
            SelectionPredicate,
        )

        dash = Dashboard(integ_df)
        fig1 = dash.add_figure(title="F1")
        fig1.add_line(x="ts", y="val")
        fig2 = dash.add_figure(title="F2")
        fig2.add_line(x="ts", y="val")
        spec = dash.to_spec(source_name="s")
        sel = SelectionState(
            source_figure_uid=spec.figures[0].uid,
            predicates=[
                SelectionPredicate(
                    clauses=[ClauseFilter(column="ts", range=(1000, 3000))]
                )
            ],
        )
        spec.state = InteractionState(selections=[sel])

        path = str(tmp_path / "cf.json")
        import pathlib

        pathlib.Path(path).write_text(spec.model_dump_json(indent=2), encoding="utf-8")
        loaded = Dashboard.load_spec(path)
        assert len(loaded.state.selections) == 1
        assert tuple(loaded.state.selections[0].predicates[0].clauses[0].range) == (
            1000.0,
            3000.0,
        )
        assert loaded.state.selections[0].source_figure_uid == spec.figures[0].uid

    def test_dashboard_with_zoom_and_cross_filter(
        self, tmp_path, integ_df: pl.DataFrame
    ):
        from flexviz.spec import (
            AxisRange,
            ClauseFilter,
            InteractionState,
            SelectionPredicate,
        )

        dash = Dashboard(integ_df)
        fig1 = dash.add_figure(title="F1")
        fig1.add_line(x="ts", y="val")
        fig2 = dash.add_figure(title="F2")
        fig2.add_line(x="ts", y="val")
        spec = dash.to_spec(source_name="s")
        sel = SelectionState(
            source_figure_uid=spec.figures[0].uid,
            predicates=[
                SelectionPredicate(
                    clauses=[ClauseFilter(column="ts", range=(500, 1500))]
                )
            ],
        )
        spec.state = InteractionState(
            viewport={
                f"{spec.figures[0].uid}/x": {"min": 0, "max": 4999},
                f"{spec.figures[1].uid}/x": {"min": 200, "max": 800},
            },
            selections=[sel],
        )

        path = str(tmp_path / "both.json")
        import pathlib

        pathlib.Path(path).write_text(spec.model_dump_json(indent=2), encoding="utf-8")
        loaded = Dashboard.load_spec(path)
        assert loaded.state.viewport[f"{spec.figures[0].uid}/x"] == AxisRange(
            min=0, max=4999
        )
        assert loaded.state.viewport[f"{spec.figures[1].uid}/x"] == AxisRange(
            min=200, max=800
        )
        assert len(loaded.state.selections) == 1
        assert tuple(loaded.state.selections[0].predicates[0].clauses[0].range) == (
            500.0,
            1500.0,
        )

    def test_figure_from_spec_preserves_state(self, integ_df: pl.DataFrame):
        original = Figure(integ_df)
        original.add_line(x="ts", y="val", name="Sig", n_points=100)
        spec = original.to_spec(source="s")
        spec.state.viewport = {"x": {"min": 200, "max": 800}}

        restored = Figure.from_spec(spec)
        new_spec = restored.to_spec(source="s")
        assert new_spec.figure.uid == spec.figure.uid


# ===========================================================================
# Figure.from_spec
# ===========================================================================


class TestFigureFromSpec:
    def test_restores_uid_traces_layout(self, integ_df: pl.DataFrame):
        original = Figure(integ_df)
        original.add_line(x="ts", y="val", name="Sig", n_points=100)
        original.update_layout(title="Restored")

        spec = original.to_spec(source="s")
        restored = Figure.from_spec(spec)

        assert restored._uid == original._uid
        assert len(restored._traces) == 1
        assert restored._traces[0].uid == original._traces[0].uid
        assert restored._layout.get("title") == "Restored"
        assert restored._backend_lf is None

        spec2 = restored.to_spec(source="s")
        assert spec2.figure.uid == spec.figure.uid
        assert spec2.figure.traces[0].uid == spec.figure.traces[0].uid


# ===========================================================================
# mount_into
# ===========================================================================


class TestMountInto:
    def test_mount_routes_accessible(self):
        from fastapi import FastAPI as _FastAPI
        from flexviz.server import mount_into

        host = _FastAPI()
        mount_into(host, prefix="/fv")
        host_client = TestClient(host)
        resp = host_client.get("/fv/sources")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_mount_raises_for_non_asgi(self):
        from flexviz.server import mount_into

        with pytest.raises(TypeError, match="does not support .mount"):
            mount_into(object(), "/fv")


# ===========================================================================
# Public API via flexviz.__init__
# ===========================================================================


class TestPublicAPI:
    def test_top_level_imports(self):
        import flexviz

        assert hasattr(flexviz, "Figure")
        assert hasattr(flexviz, "Dashboard")
        assert hasattr(flexviz, "app")
        assert hasattr(flexviz, "register_source")
        assert hasattr(flexviz, "mount_into")

    def test_register_source_via_init(self, client: TestClient, integ_df: pl.DataFrame):
        import flexviz

        flexviz.register_source("_pub_api_test", integ_df)
        resp = client.get("/sources")
        assert "_pub_api_test" in resp.json()

    @pytest.mark.parametrize("renderer", ["plotly", "echarts"])
    def test_is_valid_renderer(self, integ_df: pl.DataFrame, renderer: str):
        from flexviz.figure import Figure

        fig = Figure(integ_df)
        fig.add_line(x="ts", y="val")
        with pytest.raises(ValueError, match=renderer):
            fig.show(renderer="__invalid__")

    def test_unsupported_renderer_trace_combo_fails_fast(self):
        from flexviz.figure import Figure

        df = pl.DataFrame({"lon": [4.9, 5.1], "lat": [52.3, 52.4]})
        fig = Figure(df)
        fig.add_geo_histogram2d(lon="lon", lat="lat")
        with pytest.raises(ValueError, match="renderer='echarts'"):
            fig.show(renderer="echarts")


# ===========================================================================
# _pick_free_port
# ===========================================================================


class TestPickFreePort:
    def test_returns_valid_port(self):
        from flexviz.figure import _pick_free_port

        port = _pick_free_port()
        assert isinstance(port, int)
        assert 1024 <= port <= 65535

    def test_two_calls_return_different_ports(self):
        from flexviz.figure import _pick_free_port

        ports = {_pick_free_port() for _ in range(5)}
        assert len(ports) >= 2, "Expected at least 2 distinct ports from 5 calls"


# ===========================================================================
# Datetime cross-filtering (minimal repro for datetime support)
# ===========================================================================


class TestDatetimeCrossFiltering:
    """Test cross-filtering with datetime columns (issue: float_parsing error).

    The issue occurs when:
    1. The backend data has a datetime column (e.g., "timestamp_utc")
    2. A selection is made with ISO datetime string ranges (e.g., "2026-03-21 17:55:15")
    3. The cross-filtering should filter other figures by the same datetime range

    Previously this would fail with a float_parsing error because SelectionState.x_range
    was strictly typed as Tuple[float, float], rejecting datetime strings.
    """

    @pytest.fixture()
    def datetime_df(self) -> pl.DataFrame:
        """Create a simple DataFrame with datetime and numeric columns."""
        import datetime

        base_dt = datetime.datetime(2026, 3, 21, 17, 0, 0)
        n = 100
        return pl.DataFrame(
            {
                "timestamp_utc": [
                    base_dt + datetime.timedelta(seconds=i) for i in range(n)
                ],
                "metric_1": [float(i * 1.5) for i in range(n)],
                "metric_2": [float(i * 2.0 - 10) for i in range(n)],
            }
        )

    @pytest.fixture()
    def datetime_client(self, datetime_df: pl.DataFrame) -> TestClient:
        """Register the datetime DataFrame and return a TestClient."""
        src_name = "_datetime_test"
        register_source(src_name, datetime_df)
        return TestClient(app)

    def test_cross_filter_datetime_x_axis(
        self, datetime_client: TestClient, datetime_df: pl.DataFrame
    ):
        """Minimal test: 2 figures with datetime x-axis, cross-filter by datetime range.

        Expected behavior:
        1. Figure A has a line trace with (x="timestamp_utc", y="metric_1")
        2. Figure B has a line trace with (x="timestamp_utc", y="metric_2")
        3. Select a datetime range on Figure A (e.g., "2026-03-21 17:00:30" to "2026-03-21 17:01:30")
        4. Figure B should be cross-filtered to show only data within that range

        Before the fix, this would fail during Pydantic validation with:
           float_parsing error: unable to parse string as a number

        After the fix, the selection should succeed and cross-filter correctly.
        """
        src_name = "_datetime_test"
        dash = Dashboard(datetime_df)

        # Create two figures with the same datetime x-axis
        fig_a = dash.add_figure(title="Metric 1")
        fig_a.add_line(x="timestamp_utc", y="metric_1", n_points=100)

        fig_b = dash.add_figure(title="Metric 2")
        fig_b.add_line(x="timestamp_utc", y="metric_2", n_points=100)

        spec = dash.to_spec(source_name=src_name)
        fig_a_uid = spec.figures[0].uid
        fig_b_uid = spec.figures[1].uid

        # Init to get baseline counts
        init_payload = {
            "spec": spec.model_dump(),
            "event": {
                "type": "init",
                "axis_ranges": {},
                "selections": [],
                "force_update": True,
            },
        }
        init_resp = datetime_client.post("/dashboard/update", json=init_payload)
        assert init_resp.status_code == 200
        init_b_count = len(
            init_resp.json()["figure_deltas"][fig_b_uid][0]["updates"]["x"]
        )

        # Perform a datetime selection on Figure A
        # Select from index 20 to 80 of the datetime data
        # timestamp_utc[20] = 2026-03-21 17:00:20
        # timestamp_utc[80] = 2026-03-21 17:01:20

        start_time = "2026-03-21 17:00:20"
        end_time = "2026-03-21 17:01:20"

        selection_payload = {
            "spec": spec.model_dump(),
            "event": {
                "type": "selection",
                "axis_ranges": {},
                "selections": [
                    {
                        "source_figure_uid": fig_a_uid,
                        "predicates": [
                            {
                                "clauses": [
                                    {
                                        "column": "timestamp_utc",
                                        "range": [start_time, end_time],
                                    }
                                ]
                            }
                        ],
                    }
                ],
                "force_update": True,
                "figure_uid": fig_a_uid,
            },
        }

        # This request should succeed (not fail with float_parsing error)
        sel_resp = datetime_client.post("/dashboard/update", json=selection_payload)
        assert (
            sel_resp.status_code == 200
        ), f"Expected 200, got {sel_resp.status_code}. Response: {sel_resp.text}"

        body = sel_resp.json()

        # Figure A should return no deltas (it's the source of the selection)
        assert len(body["figure_deltas"].get(fig_a_uid, [])) == 0
        fig_a_deltas = body["figure_deltas"].get(fig_a_uid, None)
        assert (
            fig_a_deltas is None or len(fig_a_deltas) == 0
        ), f"Expected no deltas for Figure A, got {len(fig_a_deltas)}"

        # Figure B should be cross-filtered and have fewer points than the full dataset
        fig_b_deltas = body["figure_deltas"].get(fig_b_uid, [])
        assert len(fig_b_deltas) > 0, "Expected Figure B to have deltas"

        sel_b_count = len(fig_b_deltas[0]["updates"]["x"])
        assert sel_b_count < init_b_count, (
            f"Expected cross-filtered data to have fewer points "
            f"({sel_b_count} >= {init_b_count})"
        )

        # Verify the filtered data is within the selected range
        # The x values are datetime strings after filtering
        x_vals = fig_b_deltas[0]["updates"]["x"]
        # Polars returns datetime in ISO format with "T" separator, client sent space-separated
        # Normalize both formats for comparison: "2026-03-21 17:00:20" → "2026-03-21T17:00:20"
        start_normalized = start_time.replace(" ", "T")
        end_normalized = end_time.replace(" ", "T")
        assert all(start_normalized <= str(x) <= end_normalized for x in x_vals), (
            f"Expected all x values to be within [{start_normalized}, {end_normalized}], "
            f"got {min(str(x) for x in x_vals)} to {max(str(x) for x in x_vals)}"
        )

    def test_zoom_datetime_x_axis(
        self, datetime_client: TestClient, datetime_df: pl.DataFrame
    ):
        """Viewport update accepts datetime-string ranges and filters line data."""
        src_name = "_datetime_test"
        dash = Dashboard(datetime_df)

        fig = dash.add_figure(title="Zoom Metric")
        fig.add_line(x="timestamp_utc", y="metric_1", n_points=100)

        spec = dash.to_spec(source_name=src_name)
        fig_uid = spec.figures[0].uid

        init_payload = {
            "spec": spec.model_dump(),
            "event": {
                "type": "init",
                "axis_ranges": {},
                "selections": [],
                "force_update": True,
            },
        }
        init_resp = datetime_client.post("/dashboard/update", json=init_payload)
        assert init_resp.status_code == 200
        init_count = len(init_resp.json()["figure_deltas"][fig_uid][0]["updates"]["x"])

        start_time = "2026-03-21 17:00:20"
        end_time = "2026-03-21 17:01:20"
        zoom_payload = {
            "spec": spec.model_dump(),
            "event": {
                "type": "viewport",
                "axis_ranges": {"x": [start_time, end_time]},
                "selections": [],
                "force_update": False,
                "figure_uid": fig_uid,
            },
        }

        zoom_resp = datetime_client.post("/dashboard/update", json=zoom_payload)
        assert (
            zoom_resp.status_code == 200
        ), f"Expected 200, got {zoom_resp.status_code}. Response: {zoom_resp.text}"

        zoom_deltas = zoom_resp.json()["figure_deltas"].get(fig_uid, [])
        assert len(zoom_deltas) > 0, "Expected viewport update for target figure"

        x_vals = zoom_deltas[0]["updates"]["x"]
        assert (
            len(x_vals) < init_count
        ), f"Expected zoomed data to have fewer points ({len(x_vals)} >= {init_count})"

        start_normalized = start_time.replace(" ", "T")
        end_normalized = end_time.replace(" ", "T")
        assert all(start_normalized <= str(x) <= end_normalized for x in x_vals), (
            f"Expected all x values to be within [{start_normalized}, {end_normalized}], "
            f"got {min(str(x) for x in x_vals)} to {max(str(x) for x in x_vals)}"
        )


# ===========================================================================
# Bar trace integration
# ===========================================================================

_BAR_SRC = "_integ_bar_src"


@pytest.fixture(scope="module")
def bar_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts": list(range(200)),
            "cat": ["A" if i % 2 == 0 else "B" for i in range(200)],
            "region": ["N" if i % 4 < 2 else "S" for i in range(200)],
            "val": [float(i) for i in range(200)],
        }
    )


@pytest.fixture(scope="module")
def bar_client(bar_df: pl.DataFrame) -> TestClient:
    register_source(_BAR_SRC, bar_df)
    return TestClient(app)


class TestBarIntegration:
    def test_simple_bar_init(self, bar_df: pl.DataFrame, bar_client: TestClient):
        from flexviz.spec import DashboardSpec, InteractionState

        fig = Figure(bar_df)
        fig.add_bar(labels="cat", values="val", agg="sum")
        viz_spec = fig.to_spec(source=_BAR_SRC)
        dash_spec = DashboardSpec(figures=[viz_spec.figure], state=InteractionState())
        payload = {
            "spec": dash_spec.model_dump(),
            "event": {
                "type": "init",
                "axis_ranges": {},
                "selections": [],
                "force_update": True,
            },
        }
        resp = bar_client.post("/dashboard/update", json=payload)
        assert resp.status_code == 200
        fig_uid = viz_spec.figure.uid
        deltas = resp.json()["figure_deltas"].get(fig_uid, [])
        assert len(deltas) == 1
        d = deltas[0]
        assert set(d["updates"]["x"]) == {"A", "B"}

    def test_grouped_bar_init(self, bar_df: pl.DataFrame, bar_client: TestClient):
        from flexviz.spec import DashboardSpec, InteractionState

        fig = Figure(bar_df)
        fig.add_bar(labels="cat", values="val", agg="sum", group_by="region")
        viz_spec = fig.to_spec(source=_BAR_SRC)
        dash_spec = DashboardSpec(figures=[viz_spec.figure], state=InteractionState())
        payload = {
            "spec": dash_spec.model_dump(),
            "event": {
                "type": "init",
                "axis_ranges": {},
                "selections": [],
                "force_update": True,
            },
        }
        resp = bar_client.post("/dashboard/update", json=payload)
        assert resp.status_code == 200
        fig_uid = viz_spec.figure.uid
        deltas = resp.json()["figure_deltas"].get(fig_uid, [])
        assert len(deltas) == 1
        parent = deltas[0]
        assert parent["uid"] == viz_spec.figure.traces[0].uid
        assert len(parent["group_results"]) == 2
        group_keys = {cr["group_value_key"] for cr in parent["group_results"]}
        assert group_keys == {"N", "S"}

    def test_grouped_bar_cross_filter_reduces_total(
        self, bar_df: pl.DataFrame, bar_client: TestClient
    ):
        """Cross-filter from a line figure reduces bar aggregation totals."""
        # Two-figure dashboard: line (source) + bar (target)
        fig_line = Figure(bar_df)
        fig_line.add_line(x="ts", y="val", n_points=200)

        fig_bar = Figure(bar_df)
        fig_bar.add_bar(labels="cat", values="val", agg="sum")

        line_spec = fig_line.to_spec(source=_BAR_SRC)
        bar_spec = fig_bar.to_spec(source=_BAR_SRC)

        # Build a minimal DashboardSpec
        from flexviz.spec import DashboardSpec, InteractionState

        dash_spec = DashboardSpec(
            figures=[line_spec.figure, bar_spec.figure],
            state=InteractionState(),
        )

        # Init: get the full-data bar total
        init_payload = {
            "spec": dash_spec.model_dump(),
            "event": {
                "type": "init",
                "axis_ranges": {},
                "selections": [],
                "force_update": True,
            },
        }
        init_resp = bar_client.post("/dashboard/update", json=init_payload)
        assert init_resp.status_code == 200
        bar_deltas = init_resp.json()["figure_deltas"].get(bar_spec.figure.uid, [])
        full_total = sum(bar_deltas[0]["updates"]["y"])

        # Selection on line figure: ts in [0, 99]
        sel_payload = {
            "spec": dash_spec.model_dump(),
            "event": {
                "type": "selection",
                "axis_ranges": {},
                "force_update": True,
                "figure_uid": line_spec.figure.uid,
                "selections": [
                    {
                        "source_figure_uid": line_spec.figure.uid,
                        "predicates": [
                            {"clauses": [{"column": "ts", "range": [0, 99]}]}
                        ],
                    }
                ],
            },
        }
        sel_resp = bar_client.post("/dashboard/update", json=sel_payload)
        assert sel_resp.status_code == 200
        bar_sel_deltas = sel_resp.json()["figure_deltas"].get(bar_spec.figure.uid, [])
        filtered_total = sum(bar_sel_deltas[0]["updates"]["y"])
        assert filtered_total < full_total

    def test_grouped_bar_overlay_selection_returns_fg_parent_delta(
        self, bar_df: pl.DataFrame, bar_client: TestClient
    ):
        from flexviz.spec import DashboardSpec, InteractionState

        fig_line = Figure(bar_df)
        fig_line.add_line(x="ts", y="val", n_points=200)

        fig_bar = Figure(bar_df)
        fig_bar.add_bar(labels="cat", values="val", agg="sum", group_by="region")

        line_spec = fig_line.to_spec(source=_BAR_SRC)
        bar_spec = fig_bar.to_spec(source=_BAR_SRC)

        dash_spec = DashboardSpec(
            figures=[line_spec.figure, bar_spec.figure],
            state=InteractionState(cross_filter_mode="overlay"),
        )

        payload = {
            "spec": dash_spec.model_dump(),
            "event": {
                "type": "selection",
                "axis_ranges": {},
                "force_update": True,
                "figure_uid": line_spec.figure.uid,
                "selections": [
                    {
                        "source_figure_uid": line_spec.figure.uid,
                        "predicates": [
                            {"clauses": [{"column": "ts", "range": [0, 99]}]}
                        ],
                    }
                ],
            },
        }
        resp = bar_client.post("/dashboard/update", json=payload)
        assert resp.status_code == 200
        bar_deltas = resp.json()["figure_deltas"].get(bar_spec.figure.uid, [])
        assert {d["layer"] for d in bar_deltas} == {"fg"}
        assert all("group_results" in d for d in bar_deltas)


class TestBarCrossFilterIntegration:
    """End-to-end tests for BarPlot categorical cross-filtering."""

    def test_bar_selection_filters_line(
        self, bar_df: pl.DataFrame, bar_client: TestClient
    ):
        """A bar selection with categories must cross-filter a line on another figure."""
        from flexviz.spec import DashboardSpec, InteractionState

        fig_bar = Figure(bar_df)
        fig_bar.add_bar(labels="cat", values="val", agg="sum")

        fig_line = Figure(bar_df)
        fig_line.add_line(x="ts", y="val", n_points=500)

        bar_spec = fig_bar.to_spec(source=_BAR_SRC)
        line_spec = fig_line.to_spec(source=_BAR_SRC)

        dash_spec = DashboardSpec(
            figures=[bar_spec.figure, line_spec.figure],
            state=InteractionState(),
        )

        payload = {
            "spec": dash_spec.model_dump(),
            "event": {
                "type": "selection",
                "axis_ranges": {},
                "force_update": True,
                "figure_uid": bar_spec.figure.uid,
                "selections": [
                    {
                        "source_figure_uid": bar_spec.figure.uid,
                        "predicates": [
                            {"clauses": [{"column": "cat", "values": ["A"]}]}
                        ],
                    }
                ],
            },
        }
        resp = bar_client.post("/dashboard/update", json=payload)
        assert resp.status_code == 200
        result = resp.json()["figure_deltas"]

        # Source bar figure is NOT re-processed (empty delta list)
        bar_deltas = result.get(bar_spec.figure.uid, [])
        assert bar_deltas == []

        # Line figure IS updated with filtered data
        line_deltas = result.get(line_spec.figure.uid, [])
        assert len(line_deltas) == 1
        y_values = line_deltas[0]["updates"]["y"]
        # bar_df has 200 rows: even indices → cat=A, so all y values must be even
        assert all(v % 2 == 0 for v in y_values), "Only cat=A rows (even ts values)"

    def test_bar_deselect_returns_full_data(
        self, bar_df: pl.DataFrame, bar_client: TestClient
    ):
        """A deselect event must return unfiltered data for all traces."""
        from flexviz.spec import DashboardSpec, InteractionState

        fig_bar = Figure(bar_df)
        fig_bar.add_bar(labels="cat", values="val", agg="sum")
        fig_line = Figure(bar_df)
        fig_line.add_line(x="ts", y="val", n_points=500)

        bar_spec = fig_bar.to_spec(source=_BAR_SRC)
        line_spec = fig_line.to_spec(source=_BAR_SRC)

        dash_spec = DashboardSpec(
            figures=[bar_spec.figure, line_spec.figure],
            state=InteractionState(),
        )

        payload = {
            "spec": dash_spec.model_dump(),
            "event": {
                "type": "deselect",
                "axis_ranges": {},
                "force_update": True,
                "selections": [],
            },
        }
        resp = bar_client.post("/dashboard/update", json=payload)
        assert resp.status_code == 200
        result = resp.json()["figure_deltas"]

        # Both figures get data on deselect
        bar_deltas = result.get(bar_spec.figure.uid, [])
        line_deltas = result.get(line_spec.figure.uid, [])
        assert len(bar_deltas) == 1
        assert len(line_deltas) == 1

        # Bar shows both categories
        assert set(bar_deltas[0]["updates"]["x"]) == {"A", "B"}


# ===========================================================================
# Cross-filter edge cases
# ===========================================================================


class TestCrossFilterEdgeCases:
    """Edge-case cross-filter scenarios: empty categories, self-exclusion, xy+categories."""

    def test_empty_categories_returns_full_data(
        self, bar_df: pl.DataFrame, bar_client: TestClient
    ):
        """categories=[] means no filter — target line must get all rows."""
        from flexviz.spec import DashboardSpec, InteractionState

        fig_bar = Figure(bar_df)
        fig_bar.add_bar(labels="cat", values="val", agg="sum")
        fig_line = Figure(bar_df)
        fig_line.add_line(x="ts", y="val", n_points=500)

        bar_spec = fig_bar.to_spec(source=_BAR_SRC)
        line_spec = fig_line.to_spec(source=_BAR_SRC)
        dash_spec = DashboardSpec(
            figures=[bar_spec.figure, line_spec.figure],
            state=InteractionState(),
        )

        # Unfiltered reference: init
        init_payload = {
            "spec": dash_spec.model_dump(),
            "event": {
                "type": "init",
                "axis_ranges": {},
                "force_update": True,
                "selections": [],
            },
        }
        init_resp = bar_client.post("/dashboard/update", json=init_payload)
        full_count = len(
            init_resp.json()["figure_deltas"][line_spec.figure.uid][0]["updates"]["x"]
        )

        # Selection with empty categories list — must not filter
        sel_payload = {
            "spec": dash_spec.model_dump(),
            "event": {
                "type": "selection",
                "axis_ranges": {},
                "force_update": True,
                "figure_uid": bar_spec.figure.uid,
                "selections": [
                    {
                        "source_figure_uid": bar_spec.figure.uid,
                        "predicates": [],
                    }
                ],
            },
        }
        sel_resp = bar_client.post("/dashboard/update", json=sel_payload)
        assert sel_resp.status_code == 200
        line_deltas = sel_resp.json()["figure_deltas"].get(line_spec.figure.uid, [])
        assert len(line_deltas) == 1
        assert len(line_deltas[0]["updates"]["x"]) == full_count

    def test_cross_filter_source_figure_not_re_processed(
        self, bar_df: pl.DataFrame, bar_client: TestClient
    ):
        """The source figure's traces must not appear in the response on a selection event."""
        from flexviz.spec import DashboardSpec, InteractionState

        fig_bar = Figure(bar_df)
        fig_bar.add_bar(labels="cat", values="val", agg="sum")
        fig_line = Figure(bar_df)
        fig_line.add_line(x="ts", y="val", n_points=500)

        bar_spec = fig_bar.to_spec(source=_BAR_SRC)
        line_spec = fig_line.to_spec(source=_BAR_SRC)
        dash_spec = DashboardSpec(
            figures=[bar_spec.figure, line_spec.figure],
            state=InteractionState(),
        )

        payload = {
            "spec": dash_spec.model_dump(),
            "event": {
                "type": "selection",
                "axis_ranges": {},
                "force_update": True,
                "figure_uid": bar_spec.figure.uid,
                "selections": [
                    {
                        "source_figure_uid": bar_spec.figure.uid,
                        "predicates": [
                            {"clauses": [{"column": "cat", "values": ["A"]}]}
                        ],
                    }
                ],
            },
        }
        resp = bar_client.post("/dashboard/update", json=payload)
        assert resp.status_code == 200
        result = resp.json()["figure_deltas"]
        # Source figure (bar) must produce no deltas on its own selection
        assert result.get(bar_spec.figure.uid, []) == []
        # Target figure (line) must be updated
        assert len(result.get(line_spec.figure.uid, [])) == 1

    def test_xy_range_cross_filter_line_to_histogram(
        self, client: TestClient, integ_df: pl.DataFrame
    ):
        """Numeric x-range selection from a line must cross-filter a histogram on another figure."""
        fig_line = Figure(integ_df)
        fig_line.add_line(x="ts", y="val", n_points=500)
        fig_hist = Figure(integ_df)
        fig_hist.add_histogram(x="val", bins=20)

        line_spec = fig_line.to_spec(source=_SRC)
        hist_spec = fig_hist.to_spec(source=_SRC)

        from flexviz.spec import DashboardSpec, InteractionState

        dash_spec = DashboardSpec(
            figures=[line_spec.figure, hist_spec.figure],
            state=InteractionState(),
        )

        payload = {
            "spec": dash_spec.model_dump(),
            "event": {
                "type": "selection",
                "axis_ranges": {},
                "force_update": True,
                "figure_uid": line_spec.figure.uid,
                "selections": [
                    {
                        "source_figure_uid": line_spec.figure.uid,
                        "predicates": [
                            {"clauses": [{"column": "ts", "range": [1000, 2000]}]}
                        ],
                    }
                ],
            },
        }
        resp = client.post("/dashboard/update", json=payload)
        assert resp.status_code == 200
        hist_deltas = resp.json()["figure_deltas"].get(hist_spec.figure.uid, [])
        assert len(hist_deltas) == 1
        updates = hist_deltas[0]["updates"]
        bin_centers = updates["x"]
        bin_counts = updates["y"]
        # Bins span the global (unfiltered) data range — cross-filter only changes
        # counts.  ts=[1000,2000] selects 1001 rows where val==ts, so total count
        # must equal 1001 and bins outside that val range must have zero counts.
        assert len(bin_centers) == 20
        assert (
            sum(bin_counts) == 1001
        ), f"Expected 1001 total counts, got {sum(bin_counts)}"
        # Bins with centers well below or well above [1000,2000] must be empty.
        for center, count in zip(bin_centers, bin_counts):
            if center < 800 or center > 2400:
                assert (
                    count == 0
                ), f"Expected zero count far outside selection at center {center}"


# ===========================================================================
# Event sequence tests
# ===========================================================================


class TestEventSequences:
    """Verify stateless spec round-trips across init → select → deselect sequences."""

    def test_init_then_selection_then_deselect(
        self, bar_df: pl.DataFrame, bar_client: TestClient
    ):
        """Full sequence: init → selection cross-filter → deselect restores full data."""
        from flexviz.spec import DashboardSpec, InteractionState

        fig_bar = Figure(bar_df)
        fig_bar.add_bar(labels="cat", values="val", agg="sum")
        fig_line = Figure(bar_df)
        fig_line.add_line(x="ts", y="val", n_points=500)

        bar_spec = fig_bar.to_spec(source=_BAR_SRC)
        line_spec = fig_line.to_spec(source=_BAR_SRC)
        dash_spec = DashboardSpec(
            figures=[bar_spec.figure, line_spec.figure],
            state=InteractionState(),
        )

        def _post(event: dict) -> dict:
            r = bar_client.post(
                "/dashboard/update",
                json={"spec": dash_spec.model_dump(), "event": event},
            )
            assert r.status_code == 200
            return r.json()["figure_deltas"]

        # Step 1: init
        init_result = _post(
            {"type": "init", "axis_ranges": {}, "force_update": True, "selections": []}
        )
        full_y_count = len(init_result[line_spec.figure.uid][0]["updates"]["y"])

        # Step 2: selection — filter to cat=A only
        sel_result = _post(
            {
                "type": "selection",
                "axis_ranges": {},
                "force_update": True,
                "figure_uid": bar_spec.figure.uid,
                "selections": [
                    {
                        "source_figure_uid": bar_spec.figure.uid,
                        "predicates": [
                            {"clauses": [{"column": "cat", "values": ["A"]}]}
                        ],
                    }
                ],
            }
        )
        filtered_y_count = len(sel_result[line_spec.figure.uid][0]["updates"]["y"])
        assert filtered_y_count < full_y_count

        # Step 3: deselect — all data restored
        desel_result = _post(
            {
                "type": "deselect",
                "axis_ranges": {},
                "force_update": True,
                "selections": [],
            }
        )
        restored_y_count = len(desel_result[line_spec.figure.uid][0]["updates"]["y"])
        assert restored_y_count == full_y_count

    def test_overlay_selection_then_deselect_layers(
        self, client: TestClient, integ_df: pl.DataFrame
    ):
        """Overlay: selection → fg-only; deselect → bg-only (no fg)."""
        fig_a = Figure(integ_df)
        fig_a.add_line(x="ts", y="val", n_points=500)
        fig_b = Figure(integ_df)
        fig_b.add_line(x="ts", y="val", n_points=500)

        spec_a = fig_a.to_spec(source=_SRC)
        spec_b = fig_b.to_spec(source=_SRC)

        from flexviz.spec import DashboardSpec, InteractionState

        dash_spec = DashboardSpec(
            figures=[spec_a.figure, spec_b.figure],
            state=InteractionState(cross_filter_mode="overlay"),
        )

        def _post(event: dict) -> dict:
            r = client.post(
                "/dashboard/update",
                json={"spec": dash_spec.model_dump(), "event": event},
            )
            assert r.status_code == 200
            return r.json()["figure_deltas"]

        # Selection → target (fig_b) gets fg-only
        sel_result = _post(
            {
                "type": "selection",
                "axis_ranges": {},
                "force_update": True,
                "figure_uid": spec_a.figure.uid,
                "selections": [
                    {
                        "source_figure_uid": spec_a.figure.uid,
                        "predicates": [
                            {"clauses": [{"column": "ts", "range": [500, 1500]}]}
                        ],
                    }
                ],
            }
        )
        b_deltas = sel_result.get(spec_b.figure.uid, [])
        assert b_deltas, "fig_b must have deltas on selection"
        assert all(d["layer"] == "fg" for d in b_deltas), "selection → fg only"

        # Deselect → target gets bg-only (clears fg)
        desel_result = _post(
            {
                "type": "deselect",
                "axis_ranges": {},
                "force_update": True,
                "selections": [],
            }
        )
        b_desel_deltas = desel_result.get(spec_b.figure.uid, [])
        assert b_desel_deltas, "fig_b must have deltas on deselect"
        layers = {d["layer"] for d in b_desel_deltas}
        assert "fg" not in layers, "deselect must not produce fg"
        assert "bg" in layers, "deselect must produce bg to restore display"

    def test_spec_encode_decode_preserves_selection_for_restore(
        self, bar_df: pl.DataFrame, bar_client: TestClient
    ):
        """encode_spec/decode_spec round-trip preserves selections; restored spec produces same cross-filter result."""
        from flexviz.spec import DashboardSpec, InteractionState

        fig_bar = Figure(bar_df)
        fig_bar.add_bar(labels="cat", values="val", agg="sum")
        fig_line = Figure(bar_df)
        fig_line.add_line(x="ts", y="val", n_points=500)

        bar_spec = fig_bar.to_spec(source=_BAR_SRC)
        line_spec = fig_line.to_spec(source=_BAR_SRC)

        # Build a dash spec with an active selection
        from flexviz.spec import ClauseFilter, SelectionPredicate

        active_state = InteractionState(
            selections=[
                SelectionState(
                    source_figure_uid=bar_spec.figure.uid,
                    predicates=[
                        SelectionPredicate(
                            clauses=[ClauseFilter(column="cat", values=["A"])]
                        )
                    ],
                )
            ]
        )
        dash_spec = DashboardSpec(
            figures=[bar_spec.figure, line_spec.figure],
            state=active_state,
        )

        # Encode → decode and verify selection survives
        encoded = encode_spec(dash_spec)
        restored = decode_spec(encoded)
        assert isinstance(restored, DashboardSpec)
        assert len(restored.state.selections) == 1
        assert restored.state.selections[0].predicates[0].clauses[0].values == ["A"]
        assert restored.state.selections[0].source_figure_uid == bar_spec.figure.uid

        # Restored spec produces same cross-filter result as original
        def _post(spec: DashboardSpec) -> dict:
            r = bar_client.post(
                "/dashboard/update",
                json={
                    "spec": spec.model_dump(),
                    "event": {
                        "type": "selection",
                        "axis_ranges": {},
                        "force_update": True,
                        "figure_uid": bar_spec.figure.uid,
                        "selections": [s.model_dump() for s in spec.state.selections],
                    },
                },
            )
            assert r.status_code == 200
            return r.json()["figure_deltas"]

        original_result = _post(dash_spec)
        restored_result = _post(restored)

        orig_y = original_result[line_spec.figure.uid][0]["updates"]["y"]
        rest_y = restored_result[line_spec.figure.uid][0]["updates"]["y"]
        assert (
            orig_y == rest_y
        ), "Restored spec must produce identical cross-filter output"


# ===========================================================================
# client_state passthrough
# ===========================================================================


class TestClientStatePassthrough:
    def test_client_state_survives_dashboard_update(
        self, client: TestClient, integ_df: pl.DataFrame
    ):
        """client_state must be silently ignored by the server on /dashboard/update."""
        _, spec = _make_dashboard_and_spec(integ_df, n_figures=2)
        # Inject a non-default client_state into the spec dict
        spec_dict = json.loads(spec.model_dump_json())
        spec_dict["client_state"] = {"hover_mode": "on"}

        payload = {
            "spec": spec_dict,
            "event": {
                "type": "init",
                "axis_ranges": {},
                "selections": [],
                "force_update": True,
            },
        }
        resp = client.post("/dashboard/update", json=payload)
        assert resp.status_code == 200, f"Unexpected status: {resp.status_code}"
        # Server must not echo back or modify client_state — it just returns deltas
        data = resp.json()
        assert "figure_deltas" in data

    def test_unknown_client_state_field_does_not_break_server(
        self, client: TestClient, integ_df: pl.DataFrame
    ):
        """Extra keys in client_state must not cause 422/500 errors."""
        _, spec = _make_dashboard_and_spec(integ_df, n_figures=2)
        spec_dict = json.loads(spec.model_dump_json())
        spec_dict["client_state"] = {"hover_mode": "off", "future_field": True}

        payload = {
            "spec": spec_dict,
            "event": {
                "type": "init",
                "axis_ranges": {},
                "selections": [],
                "force_update": True,
            },
        }
        resp = client.post("/dashboard/update", json=payload)
        assert resp.status_code == 200

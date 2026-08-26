"""Server/engine ``request_cube`` path tests (cube Phases 1–2).

Covers the ``cube_request`` protocol on ``/update`` and ``/dashboard/update``,
the engine's ``build_cubes`` flow (source lookup, passive guard, target
enumeration, domain resolution, dedup), the byte-bounded cube cache, and
end-to-end slice parity against a direct Polars recompute — for range (hist)
and categorical (bar/pie/treemap) sources, and count/sum/mean measures.
"""

from __future__ import annotations

import base64
import math
import struct

import polars as pl
import pytest
from fastapi.testclient import TestClient

import flexviz_polars  # noqa: F401 — registers pl.Expr.flexviz namespace
from flexviz.cache import get_cache, get_cube_cache
from flexviz.cube import decode_cube_bundle, decode_fvcube_header
from flexviz.dashboard import Dashboard
from flexviz.figure import Figure
from flexviz.server import app, register_source
from flexviz.spec import AxisRange
from flexviz.trace.hist import _HIST_BIN_EPSILON

pytestmark = pytest.mark.integration

_SRC = "_cube_src"
_CAT_SRC = "_cube_cat_src"
_P = 2048


def _cube_body(resp) -> dict:
    """Normalize a ``/update`` or ``/dashboard/update`` response to a dict.

    A ``cube_request`` is answered with a binary cube bundle
    (``application/octet-stream``); decode it into the ``{"cubes": [b64...],
    "trace_cubes": {...}}`` shape the cube assertions expect. Any other
    (JSON) response passes through unchanged (httpx transparently gunzips).
    """
    if resp.headers.get("content-type", "").startswith("application/octet-stream"):
        blobs, trace_cubes = decode_cube_bundle(resp.content)
        return {
            "cubes": [base64.b64encode(b).decode("ascii") for b in blobs],
            "trace_cubes": trace_cubes,
        }
    return resp.json()


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def df() -> pl.DataFrame:
    n = 4_000
    return pl.DataFrame(
        {
            "a": [((i * 37) % 1000) / 10 for i in range(n)],
            "b": [((i * 53) % 500) / 5 for i in range(n)],
            "c": [float(i % 7) for i in range(n)],
            "g": [f"g{i % 3}" for i in range(n)],
        }
    )


@pytest.fixture()
def client(df: pl.DataFrame) -> TestClient:
    register_source(_SRC, df, cache=True)
    get_cache().clear()
    get_cube_cache().clear()
    yield TestClient(app)
    # The caches are process-global: clear on teardown too so entries built
    # here never leak into order-sensitive tests elsewhere (pytest-randomly).
    get_cache().clear()
    get_cube_cache().clear()


@pytest.fixture()
def cat_df() -> pl.DataFrame:
    n = 4_000
    cats = ["alpha", "beta", "gamma", "delta"]
    subs = ["s1", "s2", "s3"]
    return pl.DataFrame(
        {
            "cat": [cats[i % 4] for i in range(n)],
            "sub": [subs[i % 3] for i in range(n)],
            "a": [((i * 37) % 1000) / 10 for i in range(n)],
            "b": [((i * 53) % 500) / 5 for i in range(n)],
            # Integer label column (e.g. hour-of-day): a valid categorical
            # cube SOURCE with typed free categories.
            "hour": [(i * 7) % 24 for i in range(n)],
        }
    )


@pytest.fixture()
def cat_client(cat_df: pl.DataFrame) -> TestClient:
    register_source(_CAT_SRC, cat_df, cache=True)
    get_cache().clear()
    get_cube_cache().clear()
    yield TestClient(app)
    get_cache().clear()
    get_cube_cache().clear()


def _two_hist_dashboard(df: pl.DataFrame, *, title_suffix: str = ""):
    """Dashboard with a hist source figure (col a) and a hist target (col b)."""
    dash = Dashboard(df)
    fig_a = dash.add_figure(title=f"Source{title_suffix}")
    fig_a.add_histogram(x="a", bins=16)
    fig_b = dash.add_figure(title=f"Target{title_suffix}")
    fig_b.add_histogram(x="b", bins=12)
    return dash.to_spec(source_name=_SRC)


def _cube_event(figure_uid: str, selections: list | None = None) -> dict:
    return {
        "type": "cube_request",
        "axis_ranges": {},
        "selections": selections or [],
        "force_update": False,
        "figure_uid": figure_uid,
    }


def _cube_payload(
    spec, source_fig_uid: str, column: str = "a", trace_uid: str | None = None
) -> dict:
    if trace_uid is None:
        figures = getattr(spec, "figures", None) or [spec.figure]
        src_fig = next(f for f in figures if f.uid == source_fig_uid)
        trace_uid = src_fig.traces[0].uid
    return {
        "spec": spec.model_dump(),
        "event": _cube_event(source_fig_uid),
        "request_cube": True,
        "active_source": {
            "figure_uid": source_fig_uid,
            "column": column,
            "trace_uid": trace_uid,
        },
    }


def _init_payload(spec) -> dict:
    return {
        "spec": spec.model_dump(),
        "event": {
            "type": "init",
            "axis_ranges": {},
            "selections": [],
            "force_update": True,
        },
    }


def _init_centers(client, spec, figure_uid: str) -> dict[str, list[float]]:
    """Bin centers the ordinary (non-cube) display path renders per trace."""
    deltas = client.post("/dashboard/update", json=_init_payload(spec)).json()[
        "figure_deltas"
    ][figure_uid]
    return {d["uid"]: d["updates"]["x"] for d in deltas if "x" in d["updates"]}


def _cube_centers(b64_blob: str) -> list[float]:
    """Bin centers implied by a cube's binned target dim.

    A grouped histogram cube also carries categorical group dims; pick the one
    binned dim (all groups share its edges)."""
    dims = decode_fvcube_header(base64.b64decode(b64_blob))["target_dims"]
    (dim,) = [d for d in dims if d.get("domain") is not None]
    lo, hi = dim["domain"]
    width = (hi - lo) / dim["bins"]
    return [lo + (i + 0.5) * width for i in range(dim["bins"])]


def _buffer_section_start(blob: bytes) -> int:
    (header_len,) = struct.unpack_from("<I", blob, 8)
    return 12 + header_len


def _read_u32_col(blob: bytes, header: dict, name: str) -> list[int]:
    col = next(c for c in header["columns"] if c["name"] == name)
    assert col["dtype"] == "u32"
    start = _buffer_section_start(blob) + col["offset"]
    n = col["byte_len"] // 4
    return list(struct.unpack_from(f"<{n}I", blob, start))


def _read_f64_col(blob: bytes, header: dict, name: str) -> list[float]:
    col = next(c for c in header["columns"] if c["name"] == name)
    assert col["dtype"] == "f64"
    start = _buffer_section_start(blob) + col["offset"]
    n = col["byte_len"] // 8
    return list(struct.unpack_from(f"<{n}d", blob, start))


def _direct_hist(
    df: pl.DataFrame, filter_expr: pl.Expr, col: str, lo: float, hi: float, bins: int
) -> list[int]:
    """Filtered counts through the real ``fixed_hist`` kernel (legacy reference)."""
    raw = (
        df.lazy()
        .filter(filter_expr)
        .select(
            pl.col(col)
            .flexviz.fixed_hist(pl.lit(float(lo)), pl.lit(float(hi)), n_bins=bins)
            .implode()
            .alias("h")
        )
        .collect()["h"]
        .item()
    )
    return raw.explode().struct.unnest()["count"].to_list()


def _snap(domain: tuple[float, float], a: float, b: float):
    """Shared P=2048 snap arithmetic (plan doc, copied verbatim)."""
    lo, hi = domain
    s = hi - lo

    def _bin(v: float) -> int:
        return math.floor((v - lo) / s * _P)

    def _edge(bin_idx: int) -> float:
        return lo + bin_idx * s / _P

    lo_bin = max(0, min(_P, _bin(a)))
    hi_bin = max(0, min(_P, _bin(b)))
    if hi_bin < lo_bin:
        lo_bin, hi_bin = hi_bin, lo_bin
    return lo_bin, hi_bin, _edge(lo_bin), _edge(hi_bin + 1)


# ---------------------------------------------------------------------------
# Dashboard cube path
# ---------------------------------------------------------------------------


class TestDashboardCubeRequest:
    def test_returns_one_blob_mapping_target_uid(self, client, df):
        spec = _two_hist_dashboard(df)
        src_fig = spec.figures[0]
        tgt_trace_uid = spec.figures[1].traces[0].uid
        src_trace_uid = src_fig.traces[0].uid

        resp = client.post("/dashboard/update", json=_cube_payload(spec, src_fig.uid))
        assert resp.status_code == 200
        body = _cube_body(resp)
        # No deltas on the cube path — the bundle carries cubes only.
        assert len(body["cubes"]) == 1
        assert body["trace_cubes"] == {tgt_trace_uid: 0}
        assert src_trace_uid not in body["trace_cubes"]

    def test_blob_reslice_matches_direct_recompute(self, client, df):
        """End-to-end §8.2 parity: cube slice == legacy snapped recompute."""
        spec = _two_hist_dashboard(df)
        src_fig = spec.figures[0]

        resp = client.post("/dashboard/update", json=_cube_payload(spec, src_fig.uid))
        blob = base64.b64decode(_cube_body(resp)["cubes"][0])
        header = decode_fvcube_header(blob)

        # Unzoomed: free domain = full data domain of "a" (no epsilon);
        # target domain = full data domain of "b" + epsilon.
        a_lo, a_hi = df["a"].min(), df["a"].max()
        b_lo, b_hi = df["b"].min(), df["b"].max()
        assert header["free"] == {
            "kind": "continuous",
            "p": _P,
            "domain": [a_lo, a_hi],
        }
        (dim,) = header["target_dims"]
        assert dim["name"] == "b"
        assert dim["bins"] == 12
        assert dim["domain"] == [b_lo, b_hi + _HIST_BIN_EPSILON]

        # Slice the blob over a snapped brush.
        lo_bin, hi_bin, edge_lo, edge_hi = _snap((a_lo, a_hi), 12.3, 61.7)
        free_bin = _read_u32_col(blob, header, "free_bin")
        tgt_bin = _read_u32_col(blob, header, "__bin__b")
        count = _read_u32_col(blob, header, "count")
        sliced = [0] * dim["bins"]
        for fb, tb, n in zip(free_bin, tgt_bin, count):
            if lo_bin <= fb <= hi_bin:
                sliced[tb] += n

        # Direct Polars recompute with the equivalent snapped closed="left"
        # predicate over the raw data, through the real fixed_hist kernel.
        from flexviz.predicates import predicates_to_expr
        from flexviz.spec import ClauseFilter, SelectionPredicate

        pred = SelectionPredicate(
            clauses=[ClauseFilter(column="a", range=(edge_lo, edge_hi), closed="left")]
        )
        expr = predicates_to_expr([pred], df.schema)
        raw = (
            df.lazy()
            .filter(expr)
            .select(
                pl.col("b")
                .flexviz.fixed_hist(
                    pl.lit(float(b_lo)),
                    pl.lit(float(b_hi) + _HIST_BIN_EPSILON),
                    n_bins=dim["bins"],
                )
                .implode()
                .alias("h")
            )
            .collect()["h"]
            .item()
        )
        direct = raw.explode().struct.unnest()["count"].to_list()
        assert sum(sliced) > 0  # non-vacuous: the brush selects rows
        assert sum(sliced) < df.height  # ... but not all of them
        assert sliced == direct  # bit-exact counts

    def test_second_identical_request_hits_cache(self, client, df, monkeypatch):
        import flexviz.engine as engine_mod
        from flexviz.cube import build_cube as real_build_cube

        calls = {"n": 0}

        def counting_build_cube(ldf, spec, **kwargs):
            calls["n"] += 1
            return real_build_cube(ldf, spec, **kwargs)

        monkeypatch.setattr(engine_mod, "build_cube", counting_build_cube)

        spec = _two_hist_dashboard(df)
        payload = _cube_payload(spec, spec.figures[0].uid)
        r1 = client.post("/dashboard/update", json=payload)
        r2 = client.post("/dashboard/update", json=payload)
        assert calls["n"] == 1  # second request served from the cube cache
        assert _cube_body(r1)["cubes"] == _cube_body(r2)["cubes"]  # identical bytes

    def test_cross_session_specs_share_cache_entry(self, client, df, monkeypatch):
        """Two structurally different DashboardSpecs (different uids/titles,
        same source + descriptors) collide on cube_content_key."""
        import flexviz.engine as engine_mod
        from flexviz.cube import build_cube as real_build_cube

        calls = {"n": 0}

        def counting_build_cube(ldf, spec, **kwargs):
            calls["n"] += 1
            return real_build_cube(ldf, spec, **kwargs)

        monkeypatch.setattr(engine_mod, "build_cube", counting_build_cube)

        spec1 = _two_hist_dashboard(df, title_suffix="-one")
        spec2 = _two_hist_dashboard(df, title_suffix="-two")
        assert spec1.figures[0].uid != spec2.figures[0].uid

        r1 = client.post(
            "/dashboard/update", json=_cube_payload(spec1, spec1.figures[0].uid)
        )
        r2 = client.post(
            "/dashboard/update", json=_cube_payload(spec2, spec2.figures[0].uid)
        )
        assert calls["n"] == 1
        assert _cube_body(r1)["cubes"] == _cube_body(r2)["cubes"]
        # Same blob index, different trace uid (per-session join key).
        assert list(_cube_body(r2)["trace_cubes"].values()) == [0]

    def test_box_target_gets_no_cube_hist_still_served(self, client, df):
        dash = Dashboard(df)
        fig_a = dash.add_figure(title="Source")
        fig_a.add_histogram(x="a", bins=16)
        fig_b = dash.add_figure(title="HistTarget")
        fig_b.add_histogram(x="b", bins=12)
        fig_c = dash.add_figure(title="BoxTarget")
        fig_c.add_boxplot(y="c")
        spec = dash.to_spec(source_name=_SRC)
        hist_tgt_uid = spec.figures[1].traces[0].uid
        box_tgt_uid = spec.figures[2].traces[0].uid

        resp = client.post(
            "/dashboard/update", json=_cube_payload(spec, spec.figures[0].uid)
        )
        body = _cube_body(resp)
        assert len(body["cubes"]) == 1
        assert body["trace_cubes"] == {hist_tgt_uid: 0}
        assert box_tgt_uid not in body["trace_cubes"]

    def test_uncacheable_source_returns_empty_cubes(self, df):
        register_source(_SRC, df, cache=False)
        get_cube_cache().clear()
        client = TestClient(app)
        spec = _two_hist_dashboard(df)
        resp = client.post(
            "/dashboard/update", json=_cube_payload(spec, spec.figures[0].uid)
        )
        body = _cube_body(resp)
        assert body["cubes"] == []
        assert body["trace_cubes"] == {}

    def test_passive_selection_bakes_and_excludes_owner(self, client, df):
        """Phase 3: a committed selection from a third figure no longer bails
        out — its filter is baked into the built cubes and the owning
        figure's traces are excluded from the targets (mirroring
        ``_should_process_trace``: legacy selection events never update
        selection-owning figures)."""
        dash = Dashboard(df)
        fig_a = dash.add_figure(title="Source")
        fig_a.add_histogram(x="a", bins=16)
        fig_b = dash.add_figure(title="Target")
        fig_b.add_histogram(x="b", bins=12)
        fig_c = dash.add_figure(title="Third")
        fig_c.add_histogram(x="c", bins=7)
        spec = dash.to_spec(source_name=_SRC)
        third_fig_uid = spec.figures[2].uid
        tgt_uid = spec.figures[1].traces[0].uid
        third_uid = spec.figures[2].traces[0].uid

        payload = _cube_payload(spec, spec.figures[0].uid)
        payload["spec"]["state"]["selections"] = [
            {
                "source_figure_uid": third_fig_uid,
                "predicates": [{"clauses": [{"column": "c", "range": [1.0, 3.0]}]}],
            }
        ]
        resp = client.post("/dashboard/update", json=payload)
        body = _cube_body(resp)
        assert len(body["cubes"]) == 1
        assert tgt_uid in body["trace_cubes"]
        assert third_uid not in body["trace_cubes"]

    def test_selection_from_source_figure_is_ignored(self, client, df):
        """A committed selection from the source figure itself (re-brush case)
        does not trip the passive guard."""
        spec = _two_hist_dashboard(df)
        src_fig_uid = spec.figures[0].uid

        payload = _cube_payload(spec, src_fig_uid)
        payload["spec"]["state"]["selections"] = [
            {
                "source_figure_uid": src_fig_uid,
                "predicates": [{"clauses": [{"column": "a", "range": [10.0, 50.0]}]}],
            }
        ]
        resp = client.post("/dashboard/update", json=payload)
        assert len(_cube_body(resp)["cubes"]) == 1

    def test_active_source_column_mismatch_returns_empty(self, client, df):
        spec = _two_hist_dashboard(df)
        resp = client.post(
            "/dashboard/update",
            json=_cube_payload(spec, spec.figures[0].uid, column="not_a_col"),
        )
        body = _cube_body(resp)
        assert body["cubes"] == []
        assert body["trace_cubes"] == {}

    def test_zoomed_viewports_resolve_cube_domains(self, client, df):
        """Source viewport → free domain verbatim (no epsilon); target viewport
        → binned dim domain + uniform _HIST_BIN_EPSILON."""
        spec = _two_hist_dashboard(df)
        src_fig_uid = spec.figures[0].uid
        tgt_fig_uid = spec.figures[1].uid
        spec.state.viewport[f"{src_fig_uid}/x"] = AxisRange(min=10.0, max=80.0)
        spec.state.viewport[f"{tgt_fig_uid}/x"] = AxisRange(min=5.0, max=60.0)

        resp = client.post("/dashboard/update", json=_cube_payload(spec, src_fig_uid))
        blob = base64.b64decode(_cube_body(resp)["cubes"][0])
        header = decode_fvcube_header(blob)
        assert header["free"]["domain"] == [10.0, 80.0]
        (dim,) = header["target_dims"]
        assert dim["domain"] == [5.0, 60.0 + _HIST_BIN_EPSILON]

    def test_sibling_hist_targets_share_the_display_bin_domain(self, client, df):
        """Two histograms on one figure bin over their *union* min/max in the
        legacy delta, so their cubes must resolve the same union — otherwise the
        cube-served overlay fg lands on narrower, offset bins.
        """
        dash = Dashboard(df)
        fig_src = dash.add_figure(title="Source")
        fig_src.add_histogram(x="a", bins=16)
        fig_tgt = dash.add_figure(title="Target")
        fig_tgt.add_histogram(x="b", bins=12).add_histogram(x="c", bins=12)
        spec = dash.to_spec(source_name=_SRC)

        src_fig_spec, tgt_fig_spec = spec.figures
        centers_by_uid = _init_centers(client, spec, tgt_fig_spec.uid)

        body = _cube_body(
            client.post("/dashboard/update", json=_cube_payload(spec, src_fig_spec.uid))
        )
        assert len(body["trace_cubes"]) == 2
        for ts in tgt_fig_spec.traces:
            assert _cube_centers(body["cubes"][body["trace_cubes"][ts.uid]]) == (
                pytest.approx(centers_by_uid[ts.uid])
            )

    def test_non_cube_capable_sibling_widens_instead_of_killing_the_cube(
        self, client, df
    ):
        """A sibling histogram that is itself unservable (numeric ``group_by``
        fails the categorical gate) must still widen the shared bin domain: its
        column is not a target column, so it only resolves if the domain pass
        is told about it. Otherwise the servable sibling is dropped and the
        whole figure loses its live fg layer."""
        dash = Dashboard(df)
        fig_src = dash.add_figure(title="Source")
        fig_src.add_histogram(x="a", bins=16)
        fig_tgt = dash.add_figure(title="Target")
        fig_tgt.add_histogram(x="b", bins=12)
        fig_tgt.add_histogram(x="c", bins=12, group_by="a")  # numeric group: no cube
        spec = dash.to_spec(source_name=_SRC)
        src_fig_spec, tgt_fig_spec = spec.figures

        centers_by_uid = _init_centers(client, spec, tgt_fig_spec.uid)
        body = _cube_body(
            client.post("/dashboard/update", json=_cube_payload(spec, src_fig_spec.uid))
        )
        assert len(body["trace_cubes"]) == 1  # only the ungrouped sibling
        uid, idx = next(iter(body["trace_cubes"].items()))
        assert _cube_centers(body["cubes"][idx]) == pytest.approx(centers_by_uid[uid])

    def test_cube_capable_grouped_sibling_shares_the_bin_domain(self, client, df):
        """A *grouped* histogram is itself cube-servable (string ``group_by``
        passes the categorical gate) and still belongs to the shared domain
        group — both siblings must land on the union edges."""
        dash = Dashboard(df)
        fig_src = dash.add_figure(title="Source")
        fig_src.add_histogram(x="a", bins=16)
        fig_tgt = dash.add_figure(title="Target")
        fig_tgt.add_histogram(x="b", bins=12)
        fig_tgt.add_histogram(x="c", bins=12, group_by="g")
        spec = dash.to_spec(source_name=_SRC)
        src_fig_spec, tgt_fig_spec = spec.figures

        # Committed legacy centers per target: the ungrouped sibling carries
        # ``updates["x"]``; the grouped sibling's children all share one edge
        # set, so any child's ``x`` is the trace's bin centers.
        deltas = client.post("/dashboard/update", json=_init_payload(spec)).json()[
            "figure_deltas"
        ][tgt_fig_spec.uid]
        legacy_centers = {}
        for d in deltas:
            if d.get("group_results"):
                legacy_centers[d["uid"]] = d["group_results"][0]["updates"]["x"]
            else:
                legacy_centers[d["uid"]] = d["updates"]["x"]

        body = _cube_body(
            client.post("/dashboard/update", json=_cube_payload(spec, src_fig_spec.uid))
        )
        assert len(body["trace_cubes"]) == 2
        # Each cube (ungrouped and grouped) must match its own committed legacy
        # centers — which subsumes "both cubes agree with each other".
        for ts in tgt_fig_spec.traces:
            assert _cube_centers(body["cubes"][body["trace_cubes"][ts.uid]]) == (
                pytest.approx(legacy_centers[ts.uid])
            )

    def test_all_null_sibling_demotes_the_cube(self, client, df):
        """An unresolvable sibling column (all-null) demotes the target to the
        server recompute — a *conservative* choice, not full legacy parity: the
        legacy ``min_horizontal``/``max_horizontal`` bounds simply skip the null
        sibling and bin over the sibling that has data, so no live fg is
        strictly required. Demotion is safe (bg never gets a misaligned fg) but
        drops live interaction for the figure; closing that gap is tracked in
        the shared-domain-descriptor work (issue #72)."""
        nulled = df.with_columns(nulls=pl.lit(None, dtype=pl.Float64))
        register_source(_SRC, nulled, cache=True)
        dash = Dashboard(nulled)
        fig_src = dash.add_figure(title="Source")
        fig_src.add_histogram(x="a", bins=16)
        fig_tgt = dash.add_figure(title="Target")
        fig_tgt.add_histogram(x="b", bins=12).add_histogram(x="nulls", bins=12)
        spec = dash.to_spec(source_name=_SRC)

        resp = client.post(
            "/dashboard/update", json=_cube_payload(spec, spec.figures[0].uid)
        )
        assert resp.status_code == 200
        assert not _cube_body(resp).get("trace_cubes")

    def test_warm_cache_does_not_serve_a_sibling_blind_bin_domain(self, client, df):
        """The per-trace delta cache is content-addressed, and a histogram's
        content includes the siblings it shares a bin domain with. Priming each
        column as a solo figure must not let those (own-column-domain) entries
        be replayed for a figure where the columns are siblings."""
        for col in ("a", "b", "c"):
            solo = Dashboard(df)
            solo.add_figure(title=f"solo {col}").add_histogram(x=col, bins=12)
            solo_spec = solo.to_spec(source_name=_SRC)
            client.post("/dashboard/update", json=_init_payload(solo_spec))

        dash = Dashboard(df)
        dash.add_figure(title="Source").add_histogram(x="a", bins=12)
        fig_tgt = dash.add_figure(title="Target")
        fig_tgt.add_histogram(x="b", bins=12).add_histogram(x="c", bins=12)
        spec = dash.to_spec(source_name=_SRC)
        src_fig_spec, tgt_fig_spec = spec.figures

        centers_by_uid = _init_centers(client, spec, tgt_fig_spec.uid)
        # Siblings share edges with each other …
        assert len(centers_by_uid) == 2
        first, second = centers_by_uid.values()
        assert first == pytest.approx(second)
        # … and with the cube fg layer drawn over them.
        body = _cube_body(
            client.post("/dashboard/update", json=_cube_payload(spec, src_fig_spec.uid))
        )
        for uid, idx in body["trace_cubes"].items():
            assert _cube_centers(body["cubes"][idx]) == pytest.approx(
                centers_by_uid[uid]
            )

    def test_request_cube_false_regression(self, client, df):
        """request_cube=False + a normal init event: deltas unchanged, and the
        JSON delta response carries no cube fields (cubes ride the binary
        cube_request path only)."""
        spec = _two_hist_dashboard(df)
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
        body = _cube_body(resp)
        assert "cubes" not in body
        for fig in spec.figures:
            assert len(body["figure_deltas"][fig.uid]) > 0


# ---------------------------------------------------------------------------
# Single-figure /update plumbing
# ---------------------------------------------------------------------------


class TestSingleFigureCubeRequest:
    def test_update_cube_request_returns_empty_cubes(self, client, df):
        """A single figure has no cross-filter targets besides itself, so the
        /update cube path is plumbed but trivially empty."""
        fig = Figure(df)
        fig.add_histogram(x="a", bins=16)
        spec = fig.to_spec(source=_SRC)
        payload = {
            "spec": spec.model_dump(),
            "event": _cube_event(spec.figure.uid),
            "request_cube": True,
            "active_source": {
                "figure_uid": spec.figure.uid,
                "column": "a",
                "trace_uid": spec.figure.traces[0].uid,
            },
        }
        resp = client.post("/update", json=payload)
        assert resp.status_code == 200
        body = _cube_body(resp)
        assert body["cubes"] == []
        assert body["trace_cubes"] == {}

    def test_update_without_request_cube_unchanged(self, client, df):
        fig = Figure(df)
        fig.add_histogram(x="a", bins=16)
        spec = fig.to_spec(source=_SRC)
        payload = {
            "spec": spec.model_dump(),
            "event": {
                "type": "init",
                "axis_ranges": {},
                "selections": [],
                "force_update": True,
            },
        }
        resp = client.post("/update", json=payload)
        body = _cube_body(resp)
        assert len(body["deltas"]) == 1
        assert "cubes" not in body


# ---------------------------------------------------------------------------
# Active-source trace identity (plan step 0b)
# ---------------------------------------------------------------------------


class TestActiveSourceTraceIdentity:
    """``ActiveSource.trace_uid`` resolves the source trace by uid — never by
    first column match — so two source traces sharing a primary column in one
    figure (bar(cat) + treemap(cat, sub)) cannot be confused."""

    def _ambiguous_source_spec(self, cat_df):
        dash = Dashboard(cat_df)
        fig_a = dash.add_figure(title="CatSource")
        fig_a.add_bar(labels="cat")
        fig_a.add_treemap(path=["cat", "sub"])
        fig_b = dash.add_figure(title="HistTarget")
        fig_b.add_histogram(x="b", bins=12)
        return dash.to_spec(source_name=_CAT_SRC)

    def test_trace_uid_selects_among_same_primary_column_traces(
        self, cat_client, cat_df
    ):
        spec = self._ambiguous_source_spec(cat_df)
        src_fig = spec.figures[0]
        bar_uid = src_fig.traces[0].uid
        treemap_uid = src_fig.traces[1].uid

        r_bar = cat_client.post(
            "/dashboard/update",
            json=_cube_payload(spec, src_fig.uid, column="cat", trace_uid=bar_uid),
        )
        h_bar = decode_fvcube_header(base64.b64decode(_cube_body(r_bar)["cubes"][0]))
        assert h_bar["free"]["cols"] == ["cat"]

        r_tm = cat_client.post(
            "/dashboard/update",
            json=_cube_payload(spec, src_fig.uid, column="cat", trace_uid=treemap_uid),
        )
        h_tm = decode_fvcube_header(base64.b64decode(_cube_body(r_tm)["cubes"][0]))
        assert h_tm["free"]["cols"] == ["cat", "sub"]

    def test_trace_uid_column_mismatch_returns_empty(self, cat_client, cat_df):
        # The bar's primary free column is "cat"; claiming "sub" must not
        # silently fall through to another trace.
        spec = self._ambiguous_source_spec(cat_df)
        src_fig = spec.figures[0]
        resp = cat_client.post(
            "/dashboard/update",
            json=_cube_payload(
                spec, src_fig.uid, column="sub", trace_uid=src_fig.traces[0].uid
            ),
        )
        body = _cube_body(resp)
        assert body["cubes"] == []
        assert body["trace_cubes"] == {}

    def test_unknown_trace_uid_returns_empty(self, cat_client, cat_df):
        spec = self._ambiguous_source_spec(cat_df)
        src_fig = spec.figures[0]
        resp = cat_client.post(
            "/dashboard/update",
            json=_cube_payload(
                spec, src_fig.uid, column="cat", trace_uid="no-such-trace"
            ),
        )
        body = _cube_body(resp)
        assert body["cubes"] == []
        assert body["trace_cubes"] == {}

    def test_trace_uid_from_other_figure_returns_empty(self, cat_client, cat_df):
        spec = self._ambiguous_source_spec(cat_df)
        src_fig = spec.figures[0]
        foreign_uid = spec.figures[1].traces[0].uid
        resp = cat_client.post(
            "/dashboard/update",
            json=_cube_payload(spec, src_fig.uid, column="cat", trace_uid=foreign_uid),
        )
        body = _cube_body(resp)
        assert body["cubes"] == []
        assert body["trace_cubes"] == {}

    def test_active_source_without_trace_uid_rejected(self, client, df):
        spec = _two_hist_dashboard(df)
        payload = _cube_payload(spec, spec.figures[0].uid)
        del payload["active_source"]["trace_uid"]
        resp = client.post("/dashboard/update", json=payload)
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Categorical sources + measure plumbing (cube Phase 2, Step 4)
# ---------------------------------------------------------------------------


class TestCategoricalSourceCubeRequest:
    def test_bar_source_serves_hist_and_bar_targets_count_parity(
        self, cat_client, cat_df
    ):
        """Bar-source cube_request: blobs for both a hist and a bar target;
        categorical free block pinned; hist reslice over an OR'd label set is
        bit-exact against the legacy ``is_in`` recompute."""
        dash = Dashboard(cat_df)
        fig_a = dash.add_figure(title="BarSource")
        fig_a.add_bar(labels="cat")
        fig_b = dash.add_figure(title="HistTarget")
        fig_b.add_histogram(x="b", bins=12)
        fig_c = dash.add_figure(title="BarTarget")
        fig_c.add_bar(labels="sub")
        spec = dash.to_spec(source_name=_CAT_SRC)
        hist_uid = spec.figures[1].traces[0].uid
        bar_uid = spec.figures[2].traces[0].uid

        resp = cat_client.post(
            "/dashboard/update",
            json=_cube_payload(spec, spec.figures[0].uid, column="cat"),
        )
        assert resp.status_code == 200
        body = _cube_body(resp)
        assert hist_uid in body["trace_cubes"]
        assert bar_uid in body["trace_cubes"]
        assert len(body["cubes"]) == 2

        blob = base64.b64decode(body["cubes"][body["trace_cubes"][hist_uid]])
        header = decode_fvcube_header(blob)
        assert header["free"] == {
            "kind": "categorical",
            "cols": ["cat"],
            "categories": [["alpha"], ["beta"], ["delta"], ["gamma"]],
        }
        b_lo, b_hi = cat_df["b"].min(), cat_df["b"].max()
        (dim,) = header["target_dims"]
        assert dim["name"] == "b"
        assert dim["bins"] == 12
        assert dim["domain"] == [b_lo, b_hi + _HIST_BIN_EPSILON]

        # Reslice over the codes for {("alpha",), ("gamma",)}.
        cats = [tuple(t) for t in header["free"]["categories"]]
        selected = {cats.index(("alpha",)), cats.index(("gamma",))}
        free_bin = _read_u32_col(blob, header, "free_bin")
        tgt_bin = _read_u32_col(blob, header, "__bin__b")
        count = _read_u32_col(blob, header, "count")
        sliced = [0] * dim["bins"]
        for fb, tb, n in zip(free_bin, tgt_bin, count):
            if fb in selected:
                sliced[tb] += n

        direct = _direct_hist(
            cat_df,
            pl.col("cat").is_in(["alpha", "gamma"]),
            "b",
            dim["domain"][0],
            dim["domain"][1],
            dim["bins"],
        )
        assert sum(sliced) > 0  # non-vacuous: the selection covers rows
        assert sum(sliced) < cat_df.height  # ... but not all of them
        assert sliced == direct  # bit-exact counts

    def test_mean_bar_target_reslice_matches_direct_recompute(self, cat_client, cat_df):
        """Mean measure: blob ships sum (f64) + count (u32) partials; the
        combined Σsum/Σcount per cell matches a direct mean within 1e-9 and
        contributing-row counts are exact."""
        dash = Dashboard(cat_df)
        fig_a = dash.add_figure(title="BarSource")
        fig_a.add_bar(labels="cat")
        fig_b = dash.add_figure(title="MeanBarTarget")
        fig_b.add_bar(labels="sub", values="b", agg="mean")
        spec = dash.to_spec(source_name=_CAT_SRC)
        tgt_uid = spec.figures[1].traces[0].uid

        body = _cube_body(
            cat_client.post(
                "/dashboard/update",
                json=_cube_payload(spec, spec.figures[0].uid, column="cat"),
            )
        )
        blob = base64.b64decode(body["cubes"][body["trace_cubes"][tgt_uid]])
        header = decode_fvcube_header(blob)
        assert header["measure"] == {"agg": "mean", "value_col": "b"}
        (dim,) = header["target_dims"]
        assert dim == {
            "name": "sub",
            "kind": "categorical",
            "categories": ["s1", "s2", "s3"],
        }

        cats = [tuple(t) for t in header["free"]["categories"]]
        beta = cats.index(("beta",))
        free_bin = _read_u32_col(blob, header, "free_bin")
        sub_code = _read_u32_col(blob, header, "sub")
        sums = _read_f64_col(blob, header, "sum")
        counts = _read_u32_col(blob, header, "count")
        acc: dict[int, tuple[float, int]] = {}
        for fb, sc, s, n in zip(free_bin, sub_code, sums, counts):
            if fb == beta:
                tot_s, tot_n = acc.get(sc, (0.0, 0))
                acc[sc] = (tot_s + s, tot_n + n)
        # Σcount == 0 cells are omitted (contract A finalize).
        sliced = {
            dim["categories"][sc]: tot_s / tot_n
            for sc, (tot_s, tot_n) in acc.items()
            if tot_n > 0
        }
        sliced_counts = {dim["categories"][sc]: tot_n for sc, (_, tot_n) in acc.items()}

        direct_df = (
            cat_df.filter(pl.col("cat") == "beta")
            .group_by("sub")
            .agg(pl.col("b").mean().alias("mean"), pl.len().alias("n"))
        )
        direct = {r["sub"]: r["mean"] for r in direct_df.iter_rows(named=True)}
        direct_n = {r["sub"]: r["n"] for r in direct_df.iter_rows(named=True)}
        assert set(sliced) == set(direct)
        for k in direct:
            assert math.isclose(sliced[k], direct[k], rel_tol=1e-9)
        assert sliced_counts == direct_n  # contributing rows exact

    def test_sum_bar_target_from_hist_source_matches_direct_recompute(
        self, cat_client, cat_df
    ):
        """End-to-end sum measure from a continuous (hist) source: f64 sum
        partials resliced over a snapped brush match the legacy snapped
        closed='left' recompute per label within 1e-9."""
        dash = Dashboard(cat_df)
        fig_a = dash.add_figure(title="HistSource")
        fig_a.add_histogram(x="a", bins=16)
        fig_b = dash.add_figure(title="SumBarTarget")
        fig_b.add_bar(labels="sub", values="b", agg="sum")
        spec = dash.to_spec(source_name=_CAT_SRC)
        tgt_uid = spec.figures[1].traces[0].uid

        body = _cube_body(
            cat_client.post(
                "/dashboard/update",
                json=_cube_payload(spec, spec.figures[0].uid, column="a"),
            )
        )
        blob = base64.b64decode(body["cubes"][body["trace_cubes"][tgt_uid]])
        header = decode_fvcube_header(blob)
        a_lo, a_hi = cat_df["a"].min(), cat_df["a"].max()
        assert header["free"] == {
            "kind": "continuous",
            "p": _P,
            "domain": [a_lo, a_hi],
        }
        assert header["measure"] == {"agg": "sum", "value_col": "b"}
        (dim,) = header["target_dims"]

        lo_bin, hi_bin, edge_lo, edge_hi = _snap((a_lo, a_hi), 12.3, 61.7)
        free_bin = _read_u32_col(blob, header, "free_bin")
        sub_code = _read_u32_col(blob, header, "sub")
        sums = _read_f64_col(blob, header, "sum")
        acc: dict[int, float] = {}
        for fb, sc, s in zip(free_bin, sub_code, sums):
            if lo_bin <= fb <= hi_bin:
                acc[sc] = acc.get(sc, 0.0) + s
        sliced = {dim["categories"][sc]: v for sc, v in acc.items()}

        from flexviz.predicates import predicates_to_expr
        from flexviz.spec import ClauseFilter, SelectionPredicate

        pred = SelectionPredicate(
            clauses=[ClauseFilter(column="a", range=(edge_lo, edge_hi), closed="left")]
        )
        expr = predicates_to_expr([pred], cat_df.schema)
        direct = {
            r["sub"]: r["s"]
            for r in cat_df.filter(expr)
            .group_by("sub")
            .agg(pl.col("b").sum().alias("s"))
            .iter_rows(named=True)
        }
        assert set(sliced) == set(direct)
        for k in direct:
            assert math.isclose(sliced[k], direct[k], rel_tol=1e-9, abs_tol=1e-9)

    def test_bar_and_pie_same_descriptor_share_one_blob(self, cat_client, cat_df):
        """bar ≡ pie: identical labels + measure ⇒ identical CubeSpec ⇒ one
        blob, two trace_cubes entries (content-key dedup)."""
        dash = Dashboard(cat_df)
        fig_a = dash.add_figure(title="HistSource")
        fig_a.add_histogram(x="a", bins=16)
        fig_b = dash.add_figure(title="BarTarget")
        fig_b.add_bar(labels="cat")
        fig_c = dash.add_figure(title="PieTarget")
        fig_c.add_pie(labels="cat")
        spec = dash.to_spec(source_name=_CAT_SRC)
        bar_uid = spec.figures[1].traces[0].uid
        pie_uid = spec.figures[2].traces[0].uid

        body = _cube_body(
            cat_client.post(
                "/dashboard/update",
                json=_cube_payload(spec, spec.figures[0].uid, column="a"),
            )
        )
        assert len(body["cubes"]) == 1
        assert body["trace_cubes"] == {bar_uid: 0, pie_uid: 0}

    def test_grouped_hist_target_served_with_per_group_parity(self, cat_client, cat_df):
        """A grouped hist target is now served (Phase-1 returned None); its
        target dims are (binned data col, group col) in pinned order, and each
        per-group reslice is bit-exact against a per-group recompute."""
        dash = Dashboard(cat_df)
        fig_a = dash.add_figure(title="BarSource")
        fig_a.add_bar(labels="cat")
        fig_b = dash.add_figure(title="GroupedHistTarget")
        fig_b.add_histogram(x="b", bins=8, group_by="sub")
        spec = dash.to_spec(source_name=_CAT_SRC)
        tgt_uid = spec.figures[1].traces[0].uid

        body = _cube_body(
            cat_client.post(
                "/dashboard/update",
                json=_cube_payload(spec, spec.figures[0].uid, column="cat"),
            )
        )
        assert tgt_uid in body["trace_cubes"]
        blob = base64.b64decode(body["cubes"][body["trace_cubes"][tgt_uid]])
        header = decode_fvcube_header(blob)
        b_lo, b_hi = cat_df["b"].min(), cat_df["b"].max()
        assert header["target_dims"] == [
            {
                "name": "b",
                "kind": "binned",
                "bins": 8,
                "domain": [b_lo, b_hi + _HIST_BIN_EPSILON],
            },
            {"name": "sub", "kind": "categorical", "categories": ["s1", "s2", "s3"]},
        ]
        bin_dim, group_dim = header["target_dims"]

        cats = [tuple(t) for t in header["free"]["categories"]]
        alpha = cats.index(("alpha",))
        free_bin = _read_u32_col(blob, header, "free_bin")
        tgt_bin = _read_u32_col(blob, header, "__bin__b")
        sub_code = _read_u32_col(blob, header, "sub")
        count = _read_u32_col(blob, header, "count")
        per_group = {sc: [0] * bin_dim["bins"] for sc in range(3)}
        for fb, tb, sc, n in zip(free_bin, tgt_bin, sub_code, count):
            if fb == alpha:
                per_group[sc][tb] += n

        for sc, g in enumerate(group_dim["categories"]):
            direct = _direct_hist(
                cat_df,
                (pl.col("cat") == "alpha") & (pl.col("sub") == g),
                "b",
                bin_dim["domain"][0],
                bin_dim["domain"][1],
                bin_dim["bins"],
            )
            assert sum(per_group[sc]) > 0
            assert per_group[sc] == direct  # bit-exact per group

    def test_median_bar_target_not_served_hist_still_served(self, cat_client, cat_df):
        """``median`` is outside the cube measure algebra: that bar target is
        absent from trace_cubes while the hist target is still served."""
        dash = Dashboard(cat_df)
        fig_a = dash.add_figure(title="HistSource")
        fig_a.add_histogram(x="a", bins=16)
        fig_b = dash.add_figure(title="MedianBarTarget")
        fig_b.add_bar(labels="cat", values="b", agg="median")
        fig_c = dash.add_figure(title="HistTarget")
        fig_c.add_histogram(x="b", bins=12)
        spec = dash.to_spec(source_name=_CAT_SRC)
        median_uid = spec.figures[1].traces[0].uid
        hist_uid = spec.figures[2].traces[0].uid

        body = _cube_body(
            cat_client.post(
                "/dashboard/update",
                json=_cube_payload(spec, spec.figures[0].uid, column="a"),
            )
        )
        assert median_uid not in body["trace_cubes"]
        assert body["trace_cubes"] == {hist_uid: 0}
        assert len(body["cubes"]) == 1

    def test_float_label_bar_source_serves_targets(self, client, df):
        """A FLOAT-label bar is a valid categorical cube source. Integral
        float labels such as 1.0 stay typed in the FVCube header, avoiding the
        old Python/JS string mismatch (``"1.0"`` vs ``"1"``)."""
        dash = Dashboard(df)
        fig_a = dash.add_figure(title="FloatBarSource")
        fig_a.add_bar(labels="c")  # float column
        fig_b = dash.add_figure(title="HistTarget")
        fig_b.add_histogram(x="b", bins=12)
        spec = dash.to_spec(source_name=_SRC)
        hist_uid = spec.figures[1].traces[0].uid

        body = _cube_body(
            client.post(
                "/dashboard/update",
                json=_cube_payload(spec, spec.figures[0].uid, column="c"),
            )
        )
        assert hist_uid in body["trace_cubes"]
        blob = base64.b64decode(body["cubes"][body["trace_cubes"][hist_uid]])
        header = decode_fvcube_header(blob)
        assert header["free"]["kind"] == "categorical"
        assert header["free"]["cols"] == ["c"]
        cats = [tuple(t) for t in header["free"]["categories"]]
        assert (1.0,) in cats and (2.0,) in cats

        selected = {cats.index((1.0,)), cats.index((2.0,))}
        (dim,) = header["target_dims"]
        free_bin = _read_u32_col(blob, header, "free_bin")
        tgt_bin = _read_u32_col(blob, header, "__bin__b")
        count = _read_u32_col(blob, header, "count")
        sliced = [0] * dim["bins"]
        for fb, tb, n in zip(free_bin, tgt_bin, count):
            if fb in selected:
                sliced[tb] += n

        direct = _direct_hist(
            df,
            pl.col("c").is_in([1.0, 2.0]),
            "b",
            dim["domain"][0],
            dim["domain"][1],
            dim["bins"],
        )
        assert sum(sliced) > 0
        assert sum(sliced) < df.height
        assert sliced == direct

    def test_integer_label_bar_source_serves_targets(self, cat_client, cat_df):
        """An INTEGER-label bar (e.g. hour-of-day) is a valid categorical cube
        source: brushing it builds a cube that drives the other panels live.
        The free block is categorical over the integer column (categories
        stringified, like every categorical free key), and an OR'd integer
        label set reslices bit-exact against the legacy ``is_in`` recompute —
        proving the committed predicate round-trips through the integer column.
        """
        dash = Dashboard(cat_df)
        fig_a = dash.add_figure(title="IntBarSource")
        fig_a.add_bar(labels="hour", values="b", agg="mean")
        fig_b = dash.add_figure(title="HistTarget")
        fig_b.add_histogram(x="b", bins=12)
        spec = dash.to_spec(source_name=_CAT_SRC)
        hist_uid = spec.figures[1].traces[0].uid

        body = _cube_body(
            cat_client.post(
                "/dashboard/update",
                json=_cube_payload(spec, spec.figures[0].uid, column="hour"),
            )
        )
        assert hist_uid in body["trace_cubes"]
        blob = base64.b64decode(body["cubes"][body["trace_cubes"][hist_uid]])
        header = decode_fvcube_header(blob)
        assert header["free"]["kind"] == "categorical"
        assert header["free"]["cols"] == ["hour"]
        # Categories stay typed ints; the client matches values, not Python/JS
        # string formatting.
        cats = [tuple(t) for t in header["free"]["categories"]]
        assert (5,) in cats and (6,) in cats

        # Reslice over the codes for the integer label set {5, 6}.
        selected = {cats.index((5,)), cats.index((6,))}
        (dim,) = header["target_dims"]
        free_bin = _read_u32_col(blob, header, "free_bin")
        tgt_bin = _read_u32_col(blob, header, "__bin__b")
        count = _read_u32_col(blob, header, "count")
        sliced = [0] * dim["bins"]
        for fb, tb, n in zip(free_bin, tgt_bin, count):
            if fb in selected:
                sliced[tb] += n

        direct = _direct_hist(
            cat_df,
            pl.col("hour").is_in([5, 6]),
            "b",
            dim["domain"][0],
            dim["domain"][1],
            dim["bins"],
        )
        assert sum(sliced) > 0  # non-vacuous
        assert sum(sliced) < cat_df.height
        assert sliced == direct  # bit-exact counts

    def test_treemap_source_path_free_axis_prefix_reslice(self, cat_client, cat_df):
        """Treemap source: categorical free axis over the full path; a prefix
        selection (depth-1 node) expands to all matching path tuples and
        reslices bit-exact against the direct prefix recompute."""
        dash = Dashboard(cat_df)
        fig_a = dash.add_figure(title="TreemapSource")
        fig_a.add_treemap(path=["cat", "sub"])
        fig_b = dash.add_figure(title="HistTarget")
        fig_b.add_histogram(x="b", bins=12)
        spec = dash.to_spec(source_name=_CAT_SRC)
        tgt_uid = spec.figures[1].traces[0].uid

        body = _cube_body(
            cat_client.post(
                "/dashboard/update",
                json=_cube_payload(spec, spec.figures[0].uid, column="cat"),
            )
        )
        assert tgt_uid in body["trace_cubes"]
        blob = base64.b64decode(body["cubes"][body["trace_cubes"][tgt_uid]])
        header = decode_fvcube_header(blob)
        assert header["free"]["kind"] == "categorical"
        assert header["free"]["cols"] == ["cat", "sub"]
        expected = sorted(
            (c, s)
            for c in ["alpha", "beta", "gamma", "delta"]
            for s in ["s1", "s2", "s3"]
        )
        assert header["free"]["categories"] == [list(t) for t in expected]

        # Prefix selection: every path tuple whose first part is "beta".
        selected = {i for i, t in enumerate(expected) if t[0] == "beta"}
        (dim,) = header["target_dims"]
        free_bin = _read_u32_col(blob, header, "free_bin")
        tgt_bin = _read_u32_col(blob, header, "__bin__b")
        count = _read_u32_col(blob, header, "count")
        sliced = [0] * dim["bins"]
        for fb, tb, n in zip(free_bin, tgt_bin, count):
            if fb in selected:
                sliced[tb] += n

        direct = _direct_hist(
            cat_df,
            pl.col("cat") == "beta",
            "b",
            dim["domain"][0],
            dim["domain"][1],
            dim["bins"],
        )
        assert sum(sliced) > 0
        assert sliced == direct  # bit-exact counts


# ---------------------------------------------------------------------------
# Engine/server passive baking (plan step 3 / contract E)
# ---------------------------------------------------------------------------


def _passive_dashboard_spec(cat_df: pl.DataFrame, *, title_suffix: str = ""):
    """A hist(a) source; B hist(b) count target; D bar(sub, mean(b)) target;
    C bar(cat) — the passive-selection owner."""
    dash = Dashboard(cat_df)
    dash.add_figure(title=f"Source{title_suffix}").add_histogram(x="a", bins=16)
    dash.add_figure(title=f"Hist{title_suffix}").add_histogram(x="b", bins=12)
    dash.add_figure(title=f"MeanBar{title_suffix}").add_bar(
        labels="sub", values="b", agg="mean"
    )
    dash.add_figure(title=f"Owner{title_suffix}").add_bar(labels="cat")
    return dash.to_spec(source_name=_CAT_SRC)


def _cat_passive_selection(owner_uid: str) -> dict:
    return {
        "source_figure_uid": owner_uid,
        "predicates": [
            {"clauses": [{"column": "cat", "values": ["alpha"]}]},
            {"clauses": [{"column": "cat", "values": ["beta"]}]},
        ],
    }


class TestPassiveBaking:
    def test_baked_parity_count_and_mean(self, cat_client, cat_df):
        """(a)+(e): with a categorical (is_in) passive selection committed on
        C, the cubes served for a cube_request on A bake C's filter: a blob
        reslice over a snapped active range equals the legacy recompute with
        BOTH predicates (counts exact, mean <=1e-9). Domain resolution stays
        UNFILTERED (bin edges are filter-stable)."""
        spec = _passive_dashboard_spec(cat_df)
        src_fig = spec.figures[0]
        hist_uid = spec.figures[1].traces[0].uid
        mean_uid = spec.figures[2].traces[0].uid
        owner_uid = spec.figures[3].uid

        payload = _cube_payload(spec, src_fig.uid)
        payload["spec"]["state"]["selections"] = [_cat_passive_selection(owner_uid)]
        resp = cat_client.post("/dashboard/update", json=payload)
        body = _cube_body(resp)
        assert hist_uid in body["trace_cubes"]
        assert mean_uid in body["trace_cubes"]

        passive_expr = pl.col("cat").is_in(["alpha", "beta"])
        a_lo, a_hi = cat_df["a"].min(), cat_df["a"].max()
        lo_bin, hi_bin, edge_lo, edge_hi = _snap((a_lo, a_hi), 21.7, 68.3)
        active_expr = pl.col("a").is_between(edge_lo, edge_hi, closed="left")

        # --- count parity on the hist target ---
        hist_blob = base64.b64decode(body["cubes"][body["trace_cubes"][hist_uid]])
        hist_header = decode_fvcube_header(hist_blob)
        # Unfiltered domain resolution: the binned dim spans the FULL b
        # domain (+ epsilon) even though the passive filter shrinks the data.
        (dim,) = hist_header["target_dims"]
        b_lo, b_hi = cat_df["b"].min(), cat_df["b"].max()
        assert dim["domain"] == [b_lo, b_hi + _HIST_BIN_EPSILON]
        free_bin = _read_u32_col(hist_blob, hist_header, "free_bin")
        tgt_bin = _read_u32_col(hist_blob, hist_header, "__bin__b")
        count = _read_u32_col(hist_blob, hist_header, "count")
        sliced = [0] * dim["bins"]
        for fb, tb, n in zip(free_bin, tgt_bin, count):
            if lo_bin <= fb <= hi_bin:
                sliced[tb] += n
        direct = _direct_hist(
            cat_df,
            passive_expr & active_expr,
            "b",
            b_lo,
            b_hi + _HIST_BIN_EPSILON,
            dim["bins"],
        )
        assert 0 < sum(sliced) < cat_df.height
        assert sliced == direct

        # --- mean parity on the bar target ---
        bar_blob = base64.b64decode(body["cubes"][body["trace_cubes"][mean_uid]])
        bar_header = decode_fvcube_header(bar_blob)
        (bar_dim,) = bar_header["target_dims"]
        assert bar_dim["kind"] == "categorical"
        cats = bar_dim["categories"]
        bfree = _read_u32_col(bar_blob, bar_header, "free_bin")
        bcode = _read_u32_col(bar_blob, bar_header, "sub")
        bsum = _read_f64_col(bar_blob, bar_header, "sum")
        bcount = _read_u32_col(bar_blob, bar_header, "count")
        sums = {}
        counts = {}
        for fb, code, s, n in zip(bfree, bcode, bsum, bcount):
            if lo_bin <= fb <= hi_bin:
                sums[cats[code]] = sums.get(cats[code], 0.0) + s
                counts[cats[code]] = counts.get(cats[code], 0) + n
        got_means = {k: sums[k] / counts[k] for k in sums if counts[k]}
        ref = (
            cat_df.lazy()
            .filter(passive_expr & active_expr)
            .group_by("sub")
            .agg(pl.col("b").mean())
            .collect()
        )
        ref_means = dict(zip(ref["sub"].to_list(), ref["b"].to_list()))
        assert set(got_means) == set(ref_means)
        for k, v in ref_means.items():
            assert abs(got_means[k] - v) <= 1e-9 * max(1.0, abs(v))

    def test_owner_traces_absent_from_trace_cubes(self, cat_client, cat_df):
        """(b): the selection-owning figure's traces are never cube targets."""
        spec = _passive_dashboard_spec(cat_df)
        owner_fig = spec.figures[3]

        payload = _cube_payload(spec, spec.figures[0].uid)
        payload["spec"]["state"]["selections"] = [_cat_passive_selection(owner_fig.uid)]
        resp = cat_client.post("/dashboard/update", json=payload)
        body = _cube_body(resp)
        assert body["trace_cubes"]
        for ts in owner_fig.traces:
            assert ts.uid not in body["trace_cubes"]

    def test_same_passive_set_shares_cache_distinct_does_not(
        self, cat_client, cat_df, monkeypatch
    ):
        """(d): the content key includes the canonical passive key — two
        sessions with the same (snapped) passive predicates share one build;
        a different passive set builds fresh cubes."""
        import flexviz.engine as engine_mod
        from flexviz.cube import build_cube as real_build_cube

        calls = {"n": 0}

        def counting_build_cube(ldf, spec, **kwargs):
            calls["n"] += 1
            return real_build_cube(ldf, spec, **kwargs)

        monkeypatch.setattr(engine_mod, "build_cube", counting_build_cube)

        spec1 = _passive_dashboard_spec(cat_df, title_suffix="-one")
        spec2 = _passive_dashboard_spec(cat_df, title_suffix="-two")

        p1 = _cube_payload(spec1, spec1.figures[0].uid)
        p1["spec"]["state"]["selections"] = [
            _cat_passive_selection(spec1.figures[3].uid)
        ]
        r1 = cat_client.post("/dashboard/update", json=p1)
        n_first = calls["n"]
        assert n_first > 0

        # Same passive semantics, different session (figure uids differ —
        # the canonical key carries no uids).
        p2 = _cube_payload(spec2, spec2.figures[0].uid)
        p2["spec"]["state"]["selections"] = [
            _cat_passive_selection(spec2.figures[3].uid)
        ]
        r2 = cat_client.post("/dashboard/update", json=p2)
        assert calls["n"] == n_first  # served from the cube cache
        assert sorted(_cube_body(r1)["cubes"]) == sorted(_cube_body(r2)["cubes"])

        # Different passive set ⇒ distinct content keys ⇒ fresh builds.
        p3 = _cube_payload(spec1, spec1.figures[0].uid)
        p3["spec"]["state"]["selections"] = [
            {
                "source_figure_uid": spec1.figures[3].uid,
                "predicates": [{"clauses": [{"column": "cat", "values": ["gamma"]}]}],
            }
        ]
        r3 = cat_client.post("/dashboard/update", json=p3)
        assert calls["n"] > n_first
        assert set(_cube_body(r3)["cubes"]).isdisjoint(set(_cube_body(r1)["cubes"]))

    def test_none_uid_selection_excluded_from_passive(self, cat_client, cat_df):
        """(f): a source_figure_uid=None selection never filters in the
        legacy engine — the cube path excludes it too, byte-identically to a
        no-selection request (zero-passive key unchanged)."""
        spec = _passive_dashboard_spec(cat_df)
        src_fig = spec.figures[0]

        base = cat_client.post(
            "/dashboard/update", json=_cube_payload(spec, src_fig.uid)
        )
        payload = _cube_payload(spec, src_fig.uid)
        payload["spec"]["state"]["selections"] = [
            {
                "source_figure_uid": None,
                "predicates": [{"clauses": [{"column": "cat", "values": ["alpha"]}]}],
            }
        ]
        with_anon = cat_client.post("/dashboard/update", json=payload)
        assert sorted(_cube_body(with_anon)["cubes"]) == sorted(
            _cube_body(base)["cubes"]
        )
        assert _cube_body(with_anon)["trace_cubes"] == _cube_body(base)["trace_cubes"]

    def test_rebrush_own_selection_not_baked(self, cat_client, cat_df):
        """(c): the active figure's own committed selection is ignored — the
        served cubes are byte-identical to a no-selection request."""
        spec = _passive_dashboard_spec(cat_df)
        src_fig = spec.figures[0]

        base = cat_client.post(
            "/dashboard/update", json=_cube_payload(spec, src_fig.uid)
        )
        payload = _cube_payload(spec, src_fig.uid)
        payload["spec"]["state"]["selections"] = [
            {
                "source_figure_uid": src_fig.uid,
                "predicates": [
                    {
                        "clauses": [
                            {
                                "column": "a",
                                "range": [10.0, 50.0],
                                "closed": "left",
                            }
                        ]
                    }
                ],
            }
        ]
        rebrush = cat_client.post("/dashboard/update", json=payload)
        assert sorted(_cube_body(rebrush)["cubes"]) == sorted(_cube_body(base)["cubes"])
        assert _cube_body(rebrush)["trace_cubes"] == _cube_body(base)["trace_cubes"]


# ---------------------------------------------------------------------------
# Temporal sources (plan step 6 / contract G)
# ---------------------------------------------------------------------------

_TS_SRC = "_cube_ts_src"


def _temporal_server_df(time_unit: str = "us") -> pl.DataFrame:
    import datetime as dt

    n = 3_000
    base = dt.datetime(2021, 1, 1)
    return pl.DataFrame(
        {
            "t": pl.Series(
                [base + dt.timedelta(minutes=(i * 37) % 50_000) for i in range(n)],
                dtype=pl.Datetime(time_unit),
            ),
            "b": [((i * 53) % 500) / 5 for i in range(n)],
        }
    )


def _temporal_dashboard(df: pl.DataFrame):
    dash = Dashboard(df)
    dash.add_figure(title="TemporalSource").add_histogram(x="t", bins=16)
    dash.add_figure(title="Target").add_histogram(x="b", bins=12)
    return dash.to_spec(source_name=_TS_SRC)


class TestTemporalCubeSource:
    def _client(self, df: pl.DataFrame) -> TestClient:
        register_source(_TS_SRC, df, cache=True)
        get_cache().clear()
        get_cube_cache().clear()
        return TestClient(app)

    def test_us_source_header_has_unit(self):
        df = _temporal_server_df("us")
        client = self._client(df)
        spec = _temporal_dashboard(df)
        resp = client.post(
            "/dashboard/update", json=_cube_payload(spec, spec.figures[0].uid, "t")
        )
        body = _cube_body(resp)
        assert len(body["cubes"]) == 1
        header = decode_fvcube_header(base64.b64decode(body["cubes"][0]))
        assert header["free"]["kind"] == "temporal"
        assert header["free"]["unit"] == "us"
        phys = df["t"].to_physical().cast(pl.Float64)
        assert header["free"]["domain"] == [phys.min(), phys.max()]

    def test_ns_source_returns_empty(self):
        df = _temporal_server_df("ns")
        client = self._client(df)
        spec = _temporal_dashboard(df)
        resp = client.post(
            "/dashboard/update", json=_cube_payload(spec, spec.figures[0].uid, "t")
        )
        body = _cube_body(resp)
        assert body["cubes"] == []
        assert body["trace_cubes"] == {}

    def test_zoomed_temporal_source_resolves_viewport(self):
        """A zoomed temporal source's date-string viewport must resolve to
        the parsed physical range — not silently to the full domain."""
        import datetime as dt

        df = _temporal_server_df("us")
        client = self._client(df)
        spec = _temporal_dashboard(df)
        src_uid = spec.figures[0].uid
        lo_dt = dt.datetime(2021, 1, 5)
        hi_dt = dt.datetime(2021, 1, 20)
        spec.state.viewport[f"{src_uid}/x"] = AxisRange(
            min="2021-01-05 00:00:00", max="2021-01-20 00:00:00"
        )
        resp = client.post("/dashboard/update", json=_cube_payload(spec, src_uid, "t"))
        header = decode_fvcube_header(base64.b64decode(_cube_body(resp)["cubes"][0]))
        epoch = dt.datetime(1970, 1, 1)
        assert header["free"]["domain"] == [
            (lo_dt - epoch) / dt.timedelta(microseconds=1),
            (hi_dt - epoch) / dt.timedelta(microseconds=1),
        ]


# ---------------------------------------------------------------------------
# Zoom-key interplay hardening (plan step 7)
# ---------------------------------------------------------------------------


class TestZoomKeyInterplay:
    def test_zoomed_source_slice_parity(self, client, df):
        """(a): with the source zoomed, the cube's free domain is the
        viewport (finer bins) and a snapped slice equals the legacy
        recompute with the snapped closed='left' predicate on the zoomed
        grid."""
        spec = _two_hist_dashboard(df)
        src_uid = spec.figures[0].uid
        spec.state.viewport[f"{src_uid}/x"] = AxisRange(min=10.0, max=80.0)

        resp = client.post("/dashboard/update", json=_cube_payload(spec, src_uid))
        blob = base64.b64decode(_cube_body(resp)["cubes"][0])
        header = decode_fvcube_header(blob)
        assert header["free"]["domain"] == [10.0, 80.0]
        (dim,) = header["target_dims"]

        # Snap a brush on the ZOOMED grid (finer than the full-domain grid).
        lo_bin, hi_bin, edge_lo, edge_hi = _snap((10.0, 80.0), 23.4, 57.8)
        assert (80.0 - 10.0) / _P < (df["a"].max() - df["a"].min()) / _P
        free_bin = _read_u32_col(blob, header, "free_bin")
        tgt_bin = _read_u32_col(blob, header, "__bin__b")
        count = _read_u32_col(blob, header, "count")
        sliced = [0] * dim["bins"]
        for fb, tb, n in zip(free_bin, tgt_bin, count):
            if lo_bin <= fb <= hi_bin:
                sliced[tb] += n

        direct = _direct_hist(
            df,
            pl.col("a").is_between(edge_lo, edge_hi, closed="left"),
            "b",
            dim["domain"][0],
            dim["domain"][1],
            dim["bins"],
        )
        assert 0 < sum(sliced) < df.height
        assert sliced == direct

    def test_zoomed_source_with_passive_combined(self, client, df):
        """(c): a zoomed source AND a committed foreign selection: the free
        domain is still the viewport and the slice carries the baked passive
        filter."""
        dash = Dashboard(df)
        dash.add_figure(title="Source").add_histogram(x="a", bins=16)
        dash.add_figure(title="Target").add_histogram(x="b", bins=12)
        dash.add_figure(title="Owner").add_histogram(x="c", bins=7)
        spec = dash.to_spec(source_name=_SRC)
        src_uid = spec.figures[0].uid
        tgt_uid = spec.figures[1].traces[0].uid
        owner_fig_uid = spec.figures[2].uid
        spec.state.viewport[f"{src_uid}/x"] = AxisRange(min=10.0, max=80.0)

        payload = _cube_payload(spec, src_uid)
        payload["spec"]["state"]["selections"] = [
            {
                "source_figure_uid": owner_fig_uid,
                "predicates": [{"clauses": [{"column": "c", "range": [1.0, 4.0]}]}],
            }
        ]
        resp = client.post("/dashboard/update", json=payload)
        body = _cube_body(resp)
        assert tgt_uid in body["trace_cubes"]
        blob = base64.b64decode(body["cubes"][body["trace_cubes"][tgt_uid]])
        header = decode_fvcube_header(blob)
        assert header["free"]["domain"] == [10.0, 80.0]
        (dim,) = header["target_dims"]

        lo_bin, hi_bin, edge_lo, edge_hi = _snap((10.0, 80.0), 23.4, 57.8)
        free_bin = _read_u32_col(blob, header, "free_bin")
        tgt_bin = _read_u32_col(blob, header, "__bin__b")
        count = _read_u32_col(blob, header, "count")
        sliced = [0] * dim["bins"]
        for fb, tb, n in zip(free_bin, tgt_bin, count):
            if lo_bin <= fb <= hi_bin:
                sliced[tb] += n

        both = pl.col("c").is_between(1.0, 4.0) & pl.col("a").is_between(
            edge_lo, edge_hi, closed="left"
        )
        direct = _direct_hist(
            df, both, "b", dim["domain"][0], dim["domain"][1], dim["bins"]
        )
        assert 0 < sum(sliced) < df.height
        assert sliced == direct

    def test_unzoomed_domain_null_convention_unchanged(self, client, df, monkeypatch):
        """(d): two identical unzoomed cube_requests share one build/content
        key — the domain=None (full data domain) convention is unchanged."""
        import flexviz.engine as engine_mod
        from flexviz.cube import build_cube as real_build_cube

        calls = {"n": 0}

        def counting(ldf, spec, **kwargs):
            calls["n"] += 1
            return real_build_cube(ldf, spec, **kwargs)

        monkeypatch.setattr(engine_mod, "build_cube", counting)

        spec = _two_hist_dashboard(df)
        payload = _cube_payload(spec, spec.figures[0].uid)
        r1 = client.post("/dashboard/update", json=payload)
        r2 = client.post("/dashboard/update", json=payload)
        assert calls["n"] == 1
        assert _cube_body(r1)["cubes"] == _cube_body(r2)["cubes"]
        header = decode_fvcube_header(base64.b64decode(_cube_body(r1)["cubes"][0]))
        assert header["free"]["domain"] == [df["a"].min(), df["a"].max()]


# ---------------------------------------------------------------------------
# Box + line 1-D range sources (plan step 8)
# ---------------------------------------------------------------------------


def _box_source_dashboard(df: pl.DataFrame, *, orientation: str = "x"):
    """Dashboard with a box source on column ``a`` and a hist target on ``b``."""
    dash = Dashboard(df)
    fig_a = dash.add_figure(title="BoxSource")
    if orientation == "x":
        fig_a.add_boxplot(x="a")
    else:
        fig_a.add_boxplot(y="a")
    dash.add_figure(title="Target").add_histogram(x="b", bins=12)
    return dash.to_spec(source_name=_SRC)


def _line_source_dashboard(df: pl.DataFrame):
    """Dashboard with a line source (x=a) and a hist target on ``b``.

    The cube path computes no deltas for a cube_request, so the line's own
    aggregation never runs — unsorted x is irrelevant here.
    """
    dash = Dashboard(df)
    dash.add_figure(title="LineSource").add_line(x="a", y="c")
    dash.add_figure(title="Target").add_histogram(x="b", bins=12)
    return dash.to_spec(source_name=_SRC)


class TestBoxLineRangeSources:
    def _slice_and_reference(self, df, blob, header, brush_lo, brush_hi):
        """Snapped cube slice vs the legacy snapped closed='left' recompute."""
        (dim,) = header["target_dims"]
        domain = tuple(header["free"]["domain"])
        lo_bin, hi_bin, edge_lo, edge_hi = _snap(domain, brush_lo, brush_hi)
        free_bin = _read_u32_col(blob, header, "free_bin")
        tgt_bin = _read_u32_col(blob, header, "__bin__b")
        count = _read_u32_col(blob, header, "count")
        sliced = [0] * dim["bins"]
        for fb, tb, n in zip(free_bin, tgt_bin, count):
            if lo_bin <= fb <= hi_bin:
                sliced[tb] += n
        direct = _direct_hist(
            df,
            pl.col("a").is_between(edge_lo, edge_hi, closed="left"),
            "b",
            dim["domain"][0],
            dim["domain"][1],
            dim["bins"],
        )
        return sliced, direct

    def test_box_source_serves_hist_target_slice_parity(self, client, df):
        spec = _box_source_dashboard(df)
        src_fig = spec.figures[0]
        tgt_uid = spec.figures[1].traces[0].uid

        resp = client.post("/dashboard/update", json=_cube_payload(spec, src_fig.uid))
        body = _cube_body(resp)
        assert body["trace_cubes"] == {tgt_uid: 0}
        blob = base64.b64decode(body["cubes"][0])
        header = decode_fvcube_header(blob)
        assert header["free"]["kind"] == "continuous"
        assert header["free"]["p"] == _P
        assert header["free"]["domain"] == [df["a"].min(), df["a"].max()]

        sliced, direct = self._slice_and_reference(df, blob, header, 12.3, 61.7)
        assert 0 < sum(sliced) < df.height
        assert sliced == direct

    def test_box_source_zoomed_prop_axis_viewport(self, client, df):
        # Horizontal box (x="a"): the prop (data) axis is x.
        spec = _box_source_dashboard(df, orientation="x")
        src_uid = spec.figures[0].uid
        spec.state.viewport[f"{src_uid}/x"] = AxisRange(min=10.0, max=80.0)

        resp = client.post("/dashboard/update", json=_cube_payload(spec, src_uid))
        blob = base64.b64decode(_cube_body(resp)["cubes"][0])
        header = decode_fvcube_header(blob)
        assert header["free"]["domain"] == [10.0, 80.0]

        sliced, direct = self._slice_and_reference(df, blob, header, 23.4, 57.8)
        assert 0 < sum(sliced) < df.height
        assert sliced == direct

    def test_y_oriented_box_source_uses_y_viewport(self, client, df):
        # Vertical box (y="a"): the prop axis is y — the free domain must come
        # from the figure's y viewport, not x.
        spec = _box_source_dashboard(df, orientation="y")
        src_uid = spec.figures[0].uid
        spec.state.viewport[f"{src_uid}/y"] = AxisRange(min=20.0, max=70.0)
        spec.state.viewport[f"{src_uid}/x"] = AxisRange(min=-1.0, max=1.0)

        resp = client.post("/dashboard/update", json=_cube_payload(spec, src_uid))
        header = decode_fvcube_header(base64.b64decode(_cube_body(resp)["cubes"][0]))
        assert header["free"]["domain"] == [20.0, 70.0]

    def test_line_source_serves_hist_target_slice_parity(self, client, df):
        spec = _line_source_dashboard(df)
        src_fig = spec.figures[0]
        tgt_uid = spec.figures[1].traces[0].uid

        resp = client.post("/dashboard/update", json=_cube_payload(spec, src_fig.uid))
        body = _cube_body(resp)
        assert body["trace_cubes"] == {tgt_uid: 0}
        blob = base64.b64decode(body["cubes"][0])
        header = decode_fvcube_header(blob)
        assert header["free"]["kind"] == "continuous"
        assert header["free"]["p"] == _P
        assert header["free"]["domain"] == [df["a"].min(), df["a"].max()]

        sliced, direct = self._slice_and_reference(df, blob, header, 12.3, 61.7)
        assert 0 < sum(sliced) < df.height
        assert sliced == direct

    def test_line_source_zoomed_x_viewport(self, client, df):
        spec = _line_source_dashboard(df)
        src_uid = spec.figures[0].uid
        spec.state.viewport[f"{src_uid}/x"] = AxisRange(min=10.0, max=80.0)

        resp = client.post("/dashboard/update", json=_cube_payload(spec, src_uid))
        header = decode_fvcube_header(base64.b64decode(_cube_body(resp)["cubes"][0]))
        assert header["free"]["domain"] == [10.0, 80.0]

    def test_line_y_column_mismatch_returns_empty(self, client, df):
        # active_source.column must equal the line's x column (its selection
        # geometry is x-only) — the y column is not a cube source axis.
        spec = _line_source_dashboard(df)
        src_fig = spec.figures[0]
        resp = client.post(
            "/dashboard/update", json=_cube_payload(spec, src_fig.uid, column="c")
        )
        body = _cube_body(resp)
        assert body["cubes"] == []
        assert body["trace_cubes"] == {}

    def _temporal_line_dashboard(self, df: pl.DataFrame):
        dash = Dashboard(df)
        dash.add_figure(title="LineSource").add_line(x="t", y="b")
        dash.add_figure(title="Target").add_histogram(x="b", bins=12)
        return dash.to_spec(source_name=_TS_SRC)

    def _ts_client(self, df: pl.DataFrame) -> TestClient:
        register_source(_TS_SRC, df, cache=True)
        get_cache().clear()
        get_cube_cache().clear()
        return TestClient(app)

    def test_temporal_line_source_header_unit(self):
        df = _temporal_server_df("us")
        client = self._ts_client(df)
        spec = self._temporal_line_dashboard(df)
        resp = client.post(
            "/dashboard/update", json=_cube_payload(spec, spec.figures[0].uid, "t")
        )
        body = _cube_body(resp)
        assert len(body["cubes"]) == 1
        header = decode_fvcube_header(base64.b64decode(body["cubes"][0]))
        assert header["free"]["kind"] == "temporal"
        assert header["free"]["unit"] == "us"
        phys = df["t"].to_physical().cast(pl.Float64)
        assert header["free"]["domain"] == [phys.min(), phys.max()]

    def test_ns_line_source_returns_empty(self):
        df = _temporal_server_df("ns")
        client = self._ts_client(df)
        spec = self._temporal_line_dashboard(df)
        resp = client.post(
            "/dashboard/update", json=_cube_payload(spec, spec.figures[0].uid, "t")
        )
        body = _cube_body(resp)
        assert body["cubes"] == []
        assert body["trace_cubes"] == {}

    def test_ns_box_source_returns_empty(self):
        df = _temporal_server_df("ns")
        client = self._ts_client(df)
        dash = Dashboard(df)
        dash.add_figure(title="BoxSource").add_boxplot(x="t")
        dash.add_figure(title="Target").add_histogram(x="b", bins=12)
        spec = dash.to_spec(source_name=_TS_SRC)
        resp = client.post(
            "/dashboard/update", json=_cube_payload(spec, spec.figures[0].uid, "t")
        )
        body = _cube_body(resp)
        assert body["cubes"] == []
        assert body["trace_cubes"] == {}


# ---------------------------------------------------------------------------
# 2-D box (hist2d) source (plan step 13 / contract H)
# ---------------------------------------------------------------------------


_BOX2D_P = 128
_BOX2D_S = _BOX2D_P + 1


def _hist2d_source_dashboard(df: pl.DataFrame, *, title_suffix: str = ""):
    """A hist2d(x=a, y=b) source figure; a hist(c) count target."""
    dash = Dashboard(df)
    dash.add_figure(title=f"Box2dSource{title_suffix}").add_histogram2d(
        x="a", y="b", x_bins=8, y_bins=6
    )
    dash.add_figure(title=f"Hist{title_suffix}").add_histogram(x="c", bins=7)
    return dash.to_spec(source_name=_SRC)


def _snap_axis(domain: tuple[float, float], a: float, b: float, p: int = _BOX2D_P):
    lo, hi = domain
    s = (hi - lo) or 1.0

    def _bin(v: float) -> int:
        return max(0, min(p, math.floor((v - lo) / s * p)))

    lo_b, hi_b = _bin(a), _bin(b)
    if hi_b < lo_b:
        lo_b, hi_b = hi_b, lo_b
    return lo_b, hi_b, lo + lo_b * s / p, lo + (hi_b + 1) * s / p


class TestBox2dSourceCubeRequest:
    def _box2d_reslice_count(self, blob, header, x_box, y_box, df):
        """Rectangle-accumulate count over the composite free_bin buffer; also
        return the direct 2-clause closed='left' recompute over the snapped
        rectangle edges."""
        dx = tuple(header["free"]["domains"][0])
        dy = tuple(header["free"]["domains"][1])
        lx, hx, ex0, ex1 = _snap_axis(dx, x_box[0], x_box[1])
        ly, hy, ey0, ey1 = _snap_axis(dy, y_box[0], y_box[1])
        s = header["free"]["p"] + 1
        free_bin = _read_u32_col(blob, header, "free_bin")
        tgt_bin = _read_u32_col(blob, header, "__bin__c")
        count = _read_u32_col(blob, header, "count")
        codes = set()
        for by in range(ly, hy + 1):
            row = by * s
            for bx in range(lx, hx + 1):
                codes.add(row + bx)
        (dim,) = header["target_dims"]
        sliced = [0] * dim["bins"]
        for fb, tb, n in zip(free_bin, tgt_bin, count):
            if fb in codes:
                sliced[tb] += n
        direct = _direct_hist(
            df,
            pl.col("a").is_between(ex0, ex1, closed="left")
            & pl.col("b").is_between(ey0, ey1, closed="left"),
            "c",
            dim["domain"][0],
            dim["domain"][1],
            dim["bins"],
        )
        return sliced, direct

    def test_serves_hist_target_with_box2d_header(self, client, df):
        spec = _hist2d_source_dashboard(df)
        src_fig = spec.figures[0]
        src_uid = src_fig.traces[0].uid
        tgt_uid = spec.figures[1].traces[0].uid
        payload = _cube_payload(spec, src_fig.uid, column="a", trace_uid=src_uid)
        resp = client.post("/dashboard/update", json=payload)
        body = _cube_body(resp)
        assert body["trace_cubes"] == {tgt_uid: 0}
        blob = base64.b64decode(body["cubes"][0])
        header = decode_fvcube_header(blob)
        assert header["free"] == {
            "kind": "box2d",
            "cols": ["a", "b"],
            "p": 128,
            "domains": [[df["a"].min(), df["a"].max()], [df["b"].min(), df["b"].max()]],
        }

    def test_blob_reslice_matches_direct_two_clause_recompute(self, client, df):
        spec = _hist2d_source_dashboard(df)
        src_fig = spec.figures[0]
        src_uid = src_fig.traces[0].uid
        payload = _cube_payload(spec, src_fig.uid, column="a", trace_uid=src_uid)
        body = _cube_body(client.post("/dashboard/update", json=payload))
        blob = base64.b64decode(body["cubes"][0])
        header = decode_fvcube_header(blob)
        sliced, direct = self._box2d_reslice_count(
            blob, header, (12.3, 61.7), (8.1, 70.4), df
        )
        assert 0 < sum(sliced) < df.height
        assert sliced == direct

    def test_second_identical_request_hits_cache(self, client, df, monkeypatch):
        import flexviz.engine as engine_mod

        calls = {"n": 0}
        real = engine_mod.build_cube

        def counting(*a, **k):
            calls["n"] += 1
            return real(*a, **k)

        monkeypatch.setattr(engine_mod, "build_cube", counting)

        spec = _hist2d_source_dashboard(df)
        src_fig = spec.figures[0]
        src_uid = src_fig.traces[0].uid
        payload = _cube_payload(spec, src_fig.uid, column="a", trace_uid=src_uid)
        client.post("/dashboard/update", json=payload)
        after_first = calls["n"]
        assert after_first >= 1
        client.post("/dashboard/update", json=payload)
        assert calls["n"] == after_first  # cache hit, no rebuild

    def test_zoomed_viewports_resolve_both_axes(self, client, df):
        spec = _hist2d_source_dashboard(df)
        src_fig = spec.figures[0]
        src_uid_fig = src_fig.uid
        src_uid = src_fig.traces[0].uid
        spec.state.viewport[f"{src_uid_fig}/x"] = AxisRange(min=10.0, max=80.0)
        spec.state.viewport[f"{src_uid_fig}/y"] = AxisRange(min=5.0, max=70.0)
        payload = _cube_payload(spec, src_uid_fig, column="a", trace_uid=src_uid)
        body = _cube_body(client.post("/dashboard/update", json=payload))
        blob = base64.b64decode(body["cubes"][0])
        header = decode_fvcube_header(blob)
        assert header["free"]["domains"] == [[10.0, 80.0], [5.0, 70.0]]
        sliced, direct = self._box2d_reslice_count(
            blob, header, (20.0, 65.0), (15.0, 55.0), df
        )
        assert 0 < sum(sliced) < df.height
        assert sliced == direct

    def test_passive_selection_bakes_into_build(self, client, df):
        """Contract E: a foreign committed selection bakes into the box2d
        build frame; the reslice equals the legacy 3-clause recompute (passive
        + the two snapped box clauses) on the hist target."""
        # Add a third figure (hist on c) that owns a committed selection.
        dash = Dashboard(df)
        dash.add_figure(title="Box2dSource").add_histogram2d(
            x="a", y="b", x_bins=8, y_bins=6
        )
        dash.add_figure(title="Hist").add_histogram(x="c", bins=7)
        dash.add_figure(title="Owner").add_histogram(x="a", bins=10)
        spec = dash.to_spec(source_name=_SRC)
        src_fig = spec.figures[0]
        src_uid = src_fig.traces[0].uid
        tgt_uid = spec.figures[1].traces[0].uid
        owner_uid = spec.figures[2].uid

        payload = _cube_payload(spec, src_fig.uid, column="a", trace_uid=src_uid)
        passive = {
            "source_figure_uid": owner_uid,
            "predicates": [
                {"clauses": [{"column": "a", "range": [20.0, 70.0], "closed": "left"}]}
            ],
        }
        payload["spec"]["state"]["selections"] = [passive]
        body = _cube_body(client.post("/dashboard/update", json=payload))
        assert tgt_uid in body["trace_cubes"]
        blob = base64.b64decode(body["cubes"][body["trace_cubes"][tgt_uid]])
        header = decode_fvcube_header(blob)
        # Domain resolution stays UNFILTERED (full data domains).
        assert header["free"]["domains"] == [
            [df["a"].min(), df["a"].max()],
            [df["b"].min(), df["b"].max()],
        ]
        dx = tuple(header["free"]["domains"][0])
        dy = tuple(header["free"]["domains"][1])
        _lx, _hx, ex0, ex1 = _snap_axis(dx, 30.0, 60.0)
        _ly, _hy, ey0, ey1 = _snap_axis(dy, 10.0, 50.0)
        passive_expr = pl.col("a").is_between(20.0, 70.0, closed="left")
        sliced, _ = self._box2d_reslice_count(
            blob, header, (30.0, 60.0), (10.0, 50.0), df
        )
        # Reslice over the SAME rectangle, but the build frame had passive
        # baked → compare to the 3-clause recompute.
        (dim,) = header["target_dims"]
        direct = _direct_hist(
            df,
            passive_expr
            & pl.col("a").is_between(ex0, ex1, closed="left")
            & pl.col("b").is_between(ey0, ey1, closed="left"),
            "c",
            dim["domain"][0],
            dim["domain"][1],
            dim["bins"],
        )
        assert 0 < sum(sliced) < df.height
        assert sliced == direct


# ---------------------------------------------------------------------------
# Correlation target (plan step 12 / contract I)
# ---------------------------------------------------------------------------


_CORR_SRC = "_cube_corr_src"


@pytest.fixture()
def corr_df() -> pl.DataFrame:
    """A hist-brushable source column ``a`` plus three numeric columns to
    correlate (``p``/``q``/``rr`` — ``q`` and ``rr`` carry a deterministic
    linear dependence on ``p`` so the off-diagonals are non-trivial)."""
    n = 4_000
    return pl.DataFrame(
        {
            "a": [((i * 37) % 1000) / 10 for i in range(n)],
            "p": [float(i % 50) for i in range(n)],
            "q": [float((i * 3) % 50) - float(i % 50) * 0.5 for i in range(n)],
            "rr": [float((i * 7) % 30) for i in range(n)],
        }
    )


@pytest.fixture()
def corr_client(corr_df: pl.DataFrame) -> TestClient:
    register_source(_CORR_SRC, corr_df, cache=True)
    get_cache().clear()
    get_cube_cache().clear()
    yield TestClient(app)
    get_cache().clear()
    get_cube_cache().clear()


def _corr_dashboard(
    corr_df: pl.DataFrame,
    *,
    columns: list[str] | None = None,
    method: str = "pearson",
    triangular: bool = False,
    absolute: bool = False,
):
    """A hist(a) source + a corr-heatmap target over ``columns``."""
    dash = Dashboard(corr_df)
    dash.add_figure(title="Source").add_histogram(x="a", bins=16)
    dash.add_figure(title="Corr").add_corr_heatmap(
        columns=columns,
        method=method,
        triangular=triangular,
        absolute=absolute,
    )
    return dash.to_spec(source_name=_CORR_SRC)


class TestCorrTargetCubeRequest:
    def test_serves_corr_cube_with_empty_target_dims(self, corr_client, corr_df):
        spec = _corr_dashboard(corr_df, columns=["p", "q", "rr"])
        src_fig = spec.figures[0]
        corr_uid = spec.figures[1].traces[0].uid

        resp = corr_client.post(
            "/dashboard/update", json=_cube_payload(spec, src_fig.uid)
        )
        body = _cube_body(resp)
        assert body["trace_cubes"] == {corr_uid: 0}
        blob = base64.b64decode(body["cubes"][0])
        header = decode_fvcube_header(blob)
        assert header["target_dims"] == []
        assert header["measure"]["agg"] == "corr"
        assert header["measure"]["value_col"] is None
        # Column list pinned to param order; pairs are (i, j) i<j over it.
        assert header["measure"]["columns"] == ["p", "q", "rr"]
        assert header["measure"]["pairs"] == [[0, 1], [0, 2], [1, 2]]
        assert set(header["measure"]["means"]) == {"p", "q", "rr"}
        a_lo, a_hi = corr_df["a"].min(), corr_df["a"].max()
        assert header["free"] == {"kind": "continuous", "p": _P, "domain": [a_lo, a_hi]}

    def test_blob_reslice_matches_legacy_recompute(self, corr_client, corr_df):
        """The client-equivalent reslice (cube ``corr_matrix``) over a snapped
        brush equals the legacy ``CorrHeatmap._to_update`` output on the same
        snapped ``closed="left"`` filter — triangular + absolute + reversal
        mirrored (the §8.2 parity property the server delta would produce)."""
        from flexviz.cube import build_cube, CubeSpec, FreeAxisSpec, MeasureSpec
        from flexviz.predicates import predicates_to_expr
        from flexviz.spec import ClauseFilter, SelectionPredicate
        from flexviz.trace.corr_heatmap import CorrHeatmap

        cols = ["p", "q", "rr"]
        for absolute, triangular in ((False, False), (True, True), (False, True)):
            spec = _corr_dashboard(
                corr_df, columns=cols, absolute=absolute, triangular=triangular
            )
            src_fig = spec.figures[0]
            corr_uid = spec.figures[1].traces[0].uid
            resp = corr_client.post(
                "/dashboard/update", json=_cube_payload(spec, src_fig.uid)
            )
            body = _cube_body(resp)
            header = decode_fvcube_header(
                base64.b64decode(body["cubes"][body["trace_cubes"][corr_uid]])
            )
            a_lo, a_hi = header["free"]["domain"]
            lo_bin, hi_bin, edge_lo, edge_hi = _snap((a_lo, a_hi), 21.4, 73.6)

            # Client-equivalent: rebuild over the same (unfiltered) domain and
            # finalize the snapped slice through corr_matrix (== cube.js).
            local = build_cube(
                corr_df.lazy(),
                CubeSpec(
                    source_name=_CORR_SRC,
                    free=FreeAxisSpec(column="a", p=_P, domain=(a_lo, a_hi)),
                    target_dims=(),
                    measure=MeasureSpec(agg="corr", columns=tuple(cols)),
                ),
            )
            got = local.corr_matrix(
                21.4, 73.6, absolute=absolute, triangular=triangular
            )

            # Legacy server recompute: the corr trace's own _to_update on the
            # snapped closed="left" filtered rows.
            pred = SelectionPredicate(
                clauses=[
                    ClauseFilter(column="a", range=(edge_lo, edge_hi), closed="left")
                ]
            )
            sub = corr_df.filter(predicates_to_expr([pred], corr_df.schema))
            assert 0 < sub.height < corr_df.height  # non-vacuous brush
            trace = CorrHeatmap(columns=cols, absolute=absolute, triangular=triangular)
            trace.uid = "u"
            legacy = trace._to_update(
                sub.select(trace.get_aggregation_spec({}).expr)
            ).updates
            assert got["x"] == legacy["x"]
            assert got["y"] == legacy["y"]
            for gr, lr in zip(got["z"], legacy["z"]):
                for gv, lv in zip(gr, lr):
                    if gv is None or lv is None:
                        assert gv is None and lv is None, (gv, lv)
                    else:
                        assert gv == pytest.approx(lv, abs=1e-9)

    def test_second_identical_request_hits_cache(
        self, corr_client, corr_df, monkeypatch
    ):
        import flexviz.engine as engine_mod

        calls = {"n": 0}
        orig = engine_mod.build_cube

        def counting_build_cube(ldf, spec, **kwargs):
            calls["n"] += 1
            return orig(ldf, spec, **kwargs)

        monkeypatch.setattr(engine_mod, "build_cube", counting_build_cube)
        spec = _corr_dashboard(corr_df, columns=["p", "q", "rr"])
        payload = _cube_payload(spec, spec.figures[0].uid)
        corr_client.post("/dashboard/update", json=payload)
        corr_client.post("/dashboard/update", json=payload)
        assert calls["n"] == 1  # second request served from the cube cache

    def test_spearman_corr_not_served(self, corr_client, corr_df):
        """Spearman is rank-based → not decomposable → legacy POST path."""
        spec = _corr_dashboard(corr_df, columns=["p", "q", "rr"], method="spearman")
        resp = corr_client.post(
            "/dashboard/update", json=_cube_payload(spec, spec.figures[0].uid)
        )
        body = _cube_body(resp)
        assert body["cubes"] == []
        assert body["trace_cubes"] == {}

    def test_implicit_columns_corr_not_served(self, corr_client, corr_df):
        """``columns=None`` resolves server-side but the client cannot — so the
        cube path is declined (the gate requires explicit columns)."""
        spec = _corr_dashboard(corr_df, columns=None)
        resp = corr_client.post(
            "/dashboard/update", json=_cube_payload(spec, spec.figures[0].uid)
        )
        body = _cube_body(resp)
        assert body["cubes"] == []
        assert body["trace_cubes"] == {}

    def test_passive_selection_bakes_into_corr_cube(self, corr_client, corr_df):
        """A committed foreign selection pre-filters the corr build frame
        (contract E): the resliced r equals ``pl.corr`` on (passive ∧ active)
        rows. Domain resolution stays unfiltered (free domain = full ``a``)."""
        from flexviz.cube import build_cube, CubeSpec, FreeAxisSpec, MeasureSpec

        dash = Dashboard(corr_df)
        dash.add_figure(title="Source").add_histogram(x="a", bins=16)
        dash.add_figure(title="Corr").add_corr_heatmap(columns=["p", "q", "rr"])
        dash.add_figure(title="Owner").add_histogram(x="rr", bins=8)
        spec = dash.to_spec(source_name=_CORR_SRC)
        src_fig = spec.figures[0]
        corr_uid = spec.figures[1].traces[0].uid
        owner_uid = spec.figures[2].uid

        passive = {
            "source_figure_uid": owner_uid,
            "predicates": [{"clauses": [{"column": "rr", "range": [0.0, 15.0]}]}],
        }
        payload = _cube_payload(spec, src_fig.uid)
        payload["spec"]["state"]["selections"] = [passive]
        body = _cube_body(corr_client.post("/dashboard/update", json=payload))
        assert corr_uid in body["trace_cubes"]
        header = decode_fvcube_header(
            base64.b64decode(body["cubes"][body["trace_cubes"][corr_uid]])
        )
        # Free domain stays the UNFILTERED full ``a`` domain.
        a_lo, a_hi = corr_df["a"].min(), corr_df["a"].max()
        assert header["free"]["domain"] == [a_lo, a_hi]
        lo_bin, hi_bin, edge_lo, edge_hi = _snap((a_lo, a_hi), 18.0, 82.0)

        passive_expr = pl.col("rr").is_between(0.0, 15.0)
        active_expr = pl.col("a").is_between(edge_lo, edge_hi, closed="left")
        sub = corr_df.filter(passive_expr & active_expr)
        assert 0 < sub.height < corr_df.height

        local = build_cube(
            corr_df.lazy().filter(passive_expr),
            CubeSpec(
                source_name=_CORR_SRC,
                free=FreeAxisSpec(column="a", p=_P, domain=(a_lo, a_hi)),
                target_dims=(),
                measure=MeasureSpec(agg="corr", columns=("p", "q", "rr")),
            ),
        )
        per = local.slice_agg(18.0, 82.0)
        cols = ["p", "q", "rr"]
        for i, j in ((0, 1), (0, 2), (1, 2)):
            cube_r = per.filter((pl.col("i") == i) & (pl.col("j") == j))["r"][0]
            ref = sub.select(pl.corr(cols[i], cols[j], method="pearson")).item()
            assert cube_r == pytest.approx(ref, abs=1e-9)


# ---------------------------------------------------------------------------
# hist2d TARGET cube (step 14)
# ---------------------------------------------------------------------------


_HIST2D_SRC = "_cube_hist2d_src"
_H2_NB_X = 8
_H2_NB_Y = 6


@pytest.fixture()
def hist2d_df() -> pl.DataFrame:
    """A hist-brushable source ``a`` plus a 2-D ``(x, y)`` field and a numeric
    ``z`` to reduce. Values include points exactly on the bin edges of both
    axes (exercising the ``fixed_hist2d`` ``+1e-10`` span epsilon)."""
    n = 4_000
    x_lo, x_hi = 0.0, 80.0
    y_lo, y_hi = -30.0, 30.0
    xs = [
        x_lo + ((i * 13) % (_H2_NB_X * 10)) * (x_hi - x_lo) / (_H2_NB_X * 10)
        for i in range(n)
    ]
    ys = [
        y_lo + ((i * 17) % (_H2_NB_Y * 10)) * (y_hi - y_lo) / (_H2_NB_Y * 10)
        for i in range(n)
    ]
    # Force some exactly-on-edge points.
    step_x = (x_hi - x_lo) / _H2_NB_X
    step_y = (y_hi - y_lo) / _H2_NB_Y
    for k in range(_H2_NB_X + 1):
        xs[k] = x_lo + k * step_x
    for k in range(_H2_NB_Y + 1):
        ys[k] = y_lo + k * step_y
    return pl.DataFrame(
        {
            "a": [((i * 37) % 1000) / 10 for i in range(n)],
            "x": xs,
            "y": ys,
            "z": [float((i % 7) + 1) for i in range(n)],
        }
    )


@pytest.fixture()
def hist2d_client(hist2d_df: pl.DataFrame) -> TestClient:
    register_source(_HIST2D_SRC, hist2d_df, cache=True)
    get_cache().clear()
    get_cube_cache().clear()
    yield TestClient(app)
    get_cache().clear()
    get_cube_cache().clear()


def _hist2d_dashboard(
    hist2d_df: pl.DataFrame, *, histfunc: str | None = None, z: str | None = None
):
    """A hist(a) source + a 2-D histogram target over (x, y)."""
    dash = Dashboard(hist2d_df)
    dash.add_figure(title="Source").add_histogram(x="a", bins=16)
    dash.add_figure(title="Hist2D").add_histogram2d(
        x="x", y="y", x_bins=_H2_NB_X, y_bins=_H2_NB_Y, z=z, histfunc=histfunc
    )
    return dash.to_spec(source_name=_HIST2D_SRC)


def _legacy_hist2d_z(
    df: pl.DataFrame,
    filter_expr: pl.Expr,
    *,
    histfunc: str | None,
    z: str | None,
) -> list:
    """The legacy ``Histogram2D._to_update`` z over the filtered rows, binned
    to the FULL-data range (the unzoomed cube target case)."""
    from flexviz.trace.hist2d import Histogram2D

    x_lo, x_hi = float(df["x"].min()), float(df["x"].max())
    y_lo, y_hi = float(df["y"].min()), float(df["y"].max())
    trace = Histogram2D(
        x="x", y="y", x_bins=_H2_NB_X, y_bins=_H2_NB_Y, z=z, histfunc=histfunc
    )
    trace.uid = "u"
    agg = trace.get_aggregation_spec({"x": (x_lo, x_hi), "y": (y_lo, y_hi)}, df.schema)
    sub = df.filter(filter_expr)
    return trace._to_update(sub.select(agg.expr)).updates["z"]


class TestHist2dTargetCubeRequest:
    def test_serves_hist2d_cube_count_header(self, hist2d_client, hist2d_df):
        spec = _hist2d_dashboard(hist2d_df)
        src_fig = spec.figures[0]
        tgt_uid = spec.figures[1].traces[0].uid

        resp = hist2d_client.post(
            "/dashboard/update", json=_cube_payload(spec, src_fig.uid)
        )
        body = _cube_body(resp)
        assert body["trace_cubes"] == {tgt_uid: 0}
        header = decode_fvcube_header(base64.b64decode(body["cubes"][0]))
        assert [d["name"] for d in header["target_dims"]] == ["x", "y"]
        assert [d["kind"] for d in header["target_dims"]] == ["binned", "binned"]
        assert header["target_dims"][0]["bins"] == _H2_NB_X
        assert header["target_dims"][1]["bins"] == _H2_NB_Y
        assert header["measure"]["agg"] == "count"
        # Both axes resolve to the FULL data domain with NO epsilon padding
        # (hist2d's own span eps provides the boundary tolerance).
        x_lo, x_hi = hist2d_df["x"].min(), hist2d_df["x"].max()
        y_lo, y_hi = hist2d_df["y"].min(), hist2d_df["y"].max()
        assert header["target_dims"][0]["domain"] == [x_lo, x_hi]
        assert header["target_dims"][1]["domain"] == [y_lo, y_hi]
        a_lo, a_hi = hist2d_df["a"].min(), hist2d_df["a"].max()
        assert header["free"] == {"kind": "continuous", "p": _P, "domain": [a_lo, a_hi]}

    def test_serves_hist2d_cube_mean_header(self, hist2d_client, hist2d_df):
        spec = _hist2d_dashboard(hist2d_df, histfunc="mean", z="z")
        src_fig = spec.figures[0]
        tgt_uid = spec.figures[1].traces[0].uid
        resp = hist2d_client.post(
            "/dashboard/update", json=_cube_payload(spec, src_fig.uid)
        )
        body = _cube_body(resp)
        assert tgt_uid in body["trace_cubes"]
        header = decode_fvcube_header(base64.b64decode(body["cubes"][0]))
        assert header["measure"]["agg"] == "mean"
        assert header["measure"]["value_col"] == "z"

    def test_second_identical_request_hits_cache(
        self, hist2d_client, hist2d_df, monkeypatch
    ):
        import flexviz.engine as engine_mod

        calls = {"n": 0}
        orig = engine_mod.build_cube

        def counting_build_cube(ldf, spec, **kwargs):
            calls["n"] += 1
            return orig(ldf, spec, **kwargs)

        monkeypatch.setattr(engine_mod, "build_cube", counting_build_cube)
        spec = _hist2d_dashboard(hist2d_df)
        payload = _cube_payload(spec, spec.figures[0].uid)
        hist2d_client.post("/dashboard/update", json=payload)
        hist2d_client.post("/dashboard/update", json=payload)
        assert calls["n"] == 1  # second request served from the cube cache

    @pytest.mark.parametrize("histfunc, z", [(None, None), ("mean", "z")])
    def test_blob_reslice_matches_legacy_recompute(
        self, hist2d_client, hist2d_df, histfunc, z
    ):
        """A snapped brush reslice of the hist2d cube z-matrix equals the legacy
        ``Histogram2D._to_update`` z over the same snapped closed='left'
        filter (§8.2 parity — the server delta a commit would produce)."""
        spec = _hist2d_dashboard(hist2d_df, histfunc=histfunc, z=z)
        src_fig = spec.figures[0]
        tgt_uid = spec.figures[1].traces[0].uid
        resp = hist2d_client.post(
            "/dashboard/update", json=_cube_payload(spec, src_fig.uid)
        )
        body = _cube_body(resp)
        blob = base64.b64decode(body["cubes"][body["trace_cubes"][tgt_uid]])
        header = decode_fvcube_header(blob)
        a_lo, a_hi = header["free"]["domain"]
        lo_bin, hi_bin, edge_lo, edge_hi = _snap((a_lo, a_hi), 21.4, 73.6)

        free_bin = _read_u32_col(blob, header, "free_bin")
        bin_x = _read_u32_col(blob, header, "__bin__x")
        bin_y = _read_u32_col(blob, header, "__bin__y")
        if z is None:
            count = _read_u32_col(blob, header, "count")
            acc = {}
            for fb, bx, by, n in zip(free_bin, bin_x, bin_y, count):
                if lo_bin <= fb <= hi_bin:
                    acc[(bx, by)] = acc.get((bx, by), 0) + n
            zflat = [None] * (_H2_NB_X * _H2_NB_Y)
            for (bx, by), n in acc.items():
                zflat[by * _H2_NB_X + bx] = None if n == 0 else float(n)
        else:
            sum_c = _read_f64_col(blob, header, "sum")
            cnt_c = _read_u32_col(blob, header, "count")
            acc_s = {}
            acc_n = {}
            for fb, bx, by, s, c in zip(free_bin, bin_x, bin_y, sum_c, cnt_c):
                if lo_bin <= fb <= hi_bin:
                    acc_s[(bx, by)] = acc_s.get((bx, by), 0.0) + s
                    acc_n[(bx, by)] = acc_n.get((bx, by), 0) + c
            zflat = [None] * (_H2_NB_X * _H2_NB_Y)
            for cell, n in acc_n.items():
                bx, by = cell
                zflat[by * _H2_NB_X + bx] = acc_s[cell] / n if n > 0 else None
        got = [zflat[j * _H2_NB_X : (j + 1) * _H2_NB_X] for j in range(_H2_NB_Y)]

        want = _legacy_hist2d_z(
            hist2d_df,
            pl.col("a").is_between(edge_lo, edge_hi, closed="left"),
            histfunc=histfunc,
            z=z,
        )
        non_null = sum(1 for r in got for v in r if v is not None)
        assert non_null > 0
        assert len(got) == len(want)
        for grow, wrow in zip(got, want):
            for g, w in zip(grow, wrow):
                if g is None or w is None:
                    assert g is None and w is None, (g, w)
                else:
                    assert g == pytest.approx(w, abs=1e-9), (g, w)

    def test_zoomed_target_x_not_served(self, hist2d_client, hist2d_df):
        """A zoom on the hist2d target's x axis ⇒ the cube target gates to None
        (full-data only); the legacy POST path handles it."""
        spec = _hist2d_dashboard(hist2d_df)
        src_fig = spec.figures[0]
        tgt_fig_uid = spec.figures[1].uid
        spec.state.viewport[f"{tgt_fig_uid}/x"] = AxisRange(min=10.0, max=60.0)
        resp = hist2d_client.post(
            "/dashboard/update", json=_cube_payload(spec, src_fig.uid)
        )
        body = _cube_body(resp)
        # No cube-capable target ⇒ empty response (legacy delta path).
        assert body["cubes"] == []
        assert body["trace_cubes"] == {}

    def test_zoomed_target_y_not_served(self, hist2d_client, hist2d_df):
        """A zoom on the hist2d target's y axis also declines the cube."""
        spec = _hist2d_dashboard(hist2d_df)
        src_fig = spec.figures[0]
        tgt_fig_uid = spec.figures[1].uid
        spec.state.viewport[f"{tgt_fig_uid}/y"] = AxisRange(min=-10.0, max=10.0)
        resp = hist2d_client.post(
            "/dashboard/update", json=_cube_payload(spec, src_fig.uid)
        )
        body = _cube_body(resp)
        assert body["cubes"] == []
        assert body["trace_cubes"] == {}


# ---------------------------------------------------------------------------
# Treemap target cube (Step 15)
# ---------------------------------------------------------------------------

_TM_SRC = "_cube_treemap_src"


@pytest.fixture()
def treemap_df() -> pl.DataFrame:
    """A hist-brushable source ``a`` plus a 2-level treemap field (cat, sub) and
    a numeric ``val``. ``(cat, sub)`` partitions the rows into non-empty leaves
    so a legacy ``_to_grouped_update`` leaf row exists for every cube cell."""
    n = 4_000
    cats = ["alpha", "beta", "gamma"]
    subs = ["s1", "s2", "s3", "s4"]
    return pl.DataFrame(
        {
            "a": [((i * 37) % 1000) / 10 for i in range(n)],
            "cat": [cats[i % 3] for i in range(n)],
            "sub": [subs[(i * 7) % 4] for i in range(n)],
            "val": [float((i * 13) % 50) for i in range(n)],
        }
    )


@pytest.fixture()
def treemap_client(treemap_df: pl.DataFrame) -> TestClient:
    register_source(_TM_SRC, treemap_df, cache=True)
    get_cache().clear()
    get_cube_cache().clear()
    yield TestClient(app)
    get_cache().clear()
    get_cube_cache().clear()


def _treemap_dashboard(
    treemap_df: pl.DataFrame, *, agg: str = "sum", values: str | None = "val"
):
    """A hist(a) source + a treemap target over path=[cat, sub]."""
    dash = Dashboard(treemap_df)
    dash.add_figure(title="Source").add_histogram(x="a", bins=16)
    dash.add_figure(title="Treemap").add_treemap(
        path=["cat", "sub"], values=values, agg=agg
    )
    return dash.to_spec(source_name=_TM_SRC)


def _read_cat_col(blob: bytes, header: dict, name: str) -> list[str]:
    """Read a categorical target column's codes and map them to strings via the
    header's per-dim ``categories`` list."""
    dim = next(d for d in header["target_dims"] if d["name"] == name)
    codes = _read_u32_col(blob, header, name)
    cats = dim["categories"]
    return [cats[c] for c in codes]


def _legacy_treemap_leaf_rows(df: pl.DataFrame, *, agg: str, values: str | None):
    """The deepest-level rows of a legacy ``TreeMap._to_grouped_update`` over the
    full data: {(cat, sub): finalized leaf value}."""
    from flexviz.trace.treemap import TreeMap

    trace = TreeMap(path=["cat", "sub"], values=values, agg=agg)
    trace.uid = "u"
    ag = trace.get_aggregation_spec({}, df.schema)
    leaf = (
        df.lazy()
        .group_by(list(ag.group_cols))
        .agg(*ag.agg_exprs)
        .sort(list(ag.sort_cols))
        .collect()
    )
    return {(r["cat"], r["sub"]): r["u"] for r in leaf.iter_rows(named=True)}


class TestTreeMapTargetCubeRequest:
    def test_serves_treemap_cube_header_well_formed(self, treemap_client, treemap_df):
        spec = _treemap_dashboard(treemap_df)
        src_fig = spec.figures[0]
        tgt_uid = spec.figures[1].traces[0].uid

        resp = treemap_client.post(
            "/dashboard/update", json=_cube_payload(spec, src_fig.uid)
        )
        assert resp.status_code == 200
        body = _cube_body(resp)
        assert tgt_uid in body["trace_cubes"]
        blob = base64.b64decode(body["cubes"][body["trace_cubes"][tgt_uid]])
        header = decode_fvcube_header(blob)
        assert [d["name"] for d in header["target_dims"]] == ["cat", "sub"]
        assert [d["kind"] for d in header["target_dims"]] == [
            "categorical",
            "categorical",
        ]
        assert header["measure"]["agg"] == "sum"
        assert header["measure"]["value_col"] == "val"
        a_lo, a_hi = treemap_df["a"].min(), treemap_df["a"].max()
        assert header["free"] == {"kind": "continuous", "p": _P, "domain": [a_lo, a_hi]}

    def test_second_identical_request_hits_cache(
        self, treemap_client, treemap_df, monkeypatch
    ):
        import flexviz.engine as engine_mod

        calls = {"n": 0}
        orig = engine_mod.build_cube

        def counting_build_cube(ldf, spec, **kwargs):
            calls["n"] += 1
            return orig(ldf, spec, **kwargs)

        monkeypatch.setattr(engine_mod, "build_cube", counting_build_cube)
        spec = _treemap_dashboard(treemap_df)
        payload = _cube_payload(spec, spec.figures[0].uid)
        treemap_client.post("/dashboard/update", json=payload)
        treemap_client.post("/dashboard/update", json=payload)
        assert calls["n"] == 1  # second request served from the cube cache

    @pytest.mark.parametrize("agg, values", [("sum", "val"), ("mean", "val")])
    def test_leaf_cell_reslice_matches_legacy_leaf_rows(
        self, treemap_client, treemap_df, agg, values
    ):
        """Slice the served blob over the FULL free range and finalize per leaf
        cell; assert the finalized (cat, sub)→value leaf cells equal a legacy
        ``_to_grouped_update``'s deepest-level rows on the same data."""
        spec = _treemap_dashboard(treemap_df, agg=agg, values=values)
        src_fig = spec.figures[0]
        tgt_uid = spec.figures[1].traces[0].uid
        body = _cube_body(
            treemap_client.post(
                "/dashboard/update", json=_cube_payload(spec, src_fig.uid)
            )
        )
        blob = base64.b64decode(body["cubes"][body["trace_cubes"][tgt_uid]])
        header = decode_fvcube_header(blob)

        cat = _read_cat_col(blob, header, "cat")
        sub = _read_cat_col(blob, header, "sub")
        if values is None:
            counts = _read_u32_col(blob, header, "count")
            acc: dict[tuple[str, str], float] = {}
            for c, s, n in zip(cat, sub, counts):
                acc[(c, s)] = acc.get((c, s), 0.0) + n
            got = {k: float(v) for k, v in acc.items()}
        elif agg == "sum":
            sums = _read_f64_col(blob, header, "sum")
            acc = {}
            for c, s, v in zip(cat, sub, sums):
                acc[(c, s)] = acc.get((c, s), 0.0) + v
            got = acc
        else:  # mean: Σsum / Σcount
            sums = _read_f64_col(blob, header, "sum")
            counts = _read_u32_col(blob, header, "count")
            acc_s: dict[tuple[str, str], float] = {}
            acc_n: dict[tuple[str, str], int] = {}
            for c, s, v, n in zip(cat, sub, sums, counts):
                acc_s[(c, s)] = acc_s.get((c, s), 0.0) + v
                acc_n[(c, s)] = acc_n.get((c, s), 0) + n
            got = {k: acc_s[k] / acc_n[k] for k in acc_s if acc_n[k] > 0}

        want = _legacy_treemap_leaf_rows(treemap_df, agg=agg, values=values)
        assert set(got) == set(want)
        assert len(got) > 1  # non-vacuous
        for k in want:
            assert math.isclose(got[k], want[k], rel_tol=1e-9, abs_tol=1e-9), (
                k,
                got[k],
                want[k],
            )

    @pytest.mark.parametrize("agg", ["median", "n_unique"])
    def test_median_n_unique_treemap_not_served(self, treemap_client, treemap_df, agg):
        """A median/n_unique treemap is not in the cube measure algebra ⇒ no cube
        is served (legacy delta path handles it)."""
        spec = _treemap_dashboard(treemap_df, agg=agg, values="val")
        src_fig = spec.figures[0]
        body = _cube_body(
            treemap_client.post(
                "/dashboard/update", json=_cube_payload(spec, src_fig.uid)
            )
        )
        assert body["cubes"] == []
        assert body["trace_cubes"] == {}


# ---------------------------------------------------------------------------
# Cached-cube TTFB regression guards
#
# A cube *cache hit* once cost ~145 ms of server time at 10M rows even though
# build_cube never ran. Two causes, both guarded here:
#   * the cube blobs rode the JSON delta response base64-encoded, and the
#     GZipMiddleware re-gzipped that ~5 MB of high-entropy text at a high level
#     on every request (~100+ ms — the dominant TTFB cost). Fix: cubes now ship
#     as a raw binary bundle (application/octet-stream) that self-gzips at a
#     fixed low level (``_CUBE_GZIP_LEVEL``), so no base64 inflation and far
#     less compression CPU.
#   * an O(N) min/max .collect() per request to re-derive the (cache-key-
#     determining) domain (fix C: memoize physical_minmax on the source).
# Both are behavioural, not wall-clock: the blob size (and so the gzip cost) is
# ~P*targets cells, independent of row count, so a small-data timing assertion
# would not reproduce the regression. These lock the *causes* instead.
# ---------------------------------------------------------------------------


class TestCachedCubeTTFB:
    def test_cube_response_is_low_level_gzipped_binary(self, client, df):
        """The cube bundle ships as a raw binary octet-stream that self-gzips
        at a low level (not base64-in-JSON re-gzipped by the middleware at a
        high level — the original TTFB regression)."""
        from flexviz.server import _CUBE_GZIP_LEVEL

        assert _CUBE_GZIP_LEVEL <= 1, "cube bundle gzip level must stay low (TTFB)"

        spec = _two_hist_dashboard(df)
        resp = client.post(
            "/dashboard/update",
            json=_cube_payload(spec, spec.figures[0].uid),
            headers={"Accept-Encoding": "gzip"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/octet-stream")
        assert resp.headers.get("content-encoding") == "gzip"
        assert _cube_body(resp)["cubes"], "expected a served cube"

    def test_json_gzip_compresslevel_reduced(self):
        """The JSON delta path stays at a moderate gzip level — high levels burn
        CPU on the interactive update path for little size gain."""
        from starlette.middleware.gzip import GZipMiddleware

        levels = [
            mw.kwargs.get("compresslevel", 9)
            for mw in app.user_middleware
            if mw.cls is GZipMiddleware
        ]
        assert levels, "GZipMiddleware not installed on the app"
        assert all(lvl <= 6 for lvl in levels), (
            f"GZip compresslevel must stay <= 6 (was {levels}); level 9 is the "
            "cached-cube TTFB regression"
        )

    def test_warm_cube_request_does_no_polars_collects(self, client, df, monkeypatch):
        """A fully warm cube request (cube blobs cached AND domain memoized)
        must do **zero** Polars collects — build is a cache hit and the domain
        min/max is memoized on the source builder. Without fix C the domain
        scan re-runs every request."""
        spec = _two_hist_dashboard(df)
        payload = _cube_payload(spec, spec.figures[0].uid)

        # Warm both caches: blob cache + the per-source physical_minmax memo.
        client.post("/dashboard/update", json=payload)
        client.post("/dashboard/update", json=payload)

        real_collect = pl.LazyFrame.collect
        calls = {"n": 0}

        def counting_collect(self, *args, **kwargs):
            calls["n"] += 1
            return real_collect(self, *args, **kwargs)

        monkeypatch.setattr(pl.LazyFrame, "collect", counting_collect)
        r = client.post("/dashboard/update", json=payload)
        assert r.status_code == 200
        assert _cube_body(r)["cubes"], "expected a served cube"
        assert calls["n"] == 0, (
            f"a warm cube request triggered {calls['n']} Polars collect(s); "
            "domain resolution is not memoized (fix C regressed)"
        )

    def test_free_axis_column_also_target_dim(self, client, df):
        """Regression for the domain-memo refactor: when the brushed free-axis
        column is ALSO a binned target dim (here ``a`` is both), the unified
        per-column min/max lookup must de-dupe — otherwise it builds duplicate
        select aliases and the request 500s."""
        dash = Dashboard(df)
        fig_src = dash.add_figure(title="Source")
        fig_src.add_histogram(x="a", bins=16)
        fig_tgt = dash.add_figure(title="TargetSameCol")
        fig_tgt.add_histogram(x="a", bins=10)  # target binned dim == free col
        spec = dash.to_spec(source_name=_SRC)

        r = client.post(
            "/dashboard/update", json=_cube_payload(spec, spec.figures[0].uid)
        )
        assert r.status_code == 200, r.text
        assert len(_cube_body(r)["cubes"]) == 1
        assert spec.figures[1].traces[0].uid in _cube_body(r)["trace_cubes"]


# ---------------------------------------------------------------------------
# Free-axis / target-measure compatibility gate
# ---------------------------------------------------------------------------


class TestIncompatibleFreeAxisTargets:
    """A ``line_env`` (contract J) or ``corr`` (contract I) cube target bins/
    scans the free axis as a numeric range, so it can only be built against a
    range (continuous/temporal) free axis. A **categorical** (bar/pie/treemap)
    or **box2d** (hist2d) source must not build such targets — the engine skips
    them (they fall back to the per-commit recompute) and still serves every
    range-compatible target, instead of 500-ing the whole live-brush request.

    Regression for ``ValueError: line_env requires a range
    (continuous/temporal) free axis`` raised through ``_run_cube_path``.
    """

    def test_box2d_source_skips_line_target_serves_hist(self, client, df):
        dash = Dashboard(df)
        dash.add_figure(title="Box2dSource").add_histogram2d(
            x="a", y="b", x_bins=8, y_bins=6
        )
        dash.add_figure(title="LineTarget").add_line(x="a", y="b")
        dash.add_figure(title="HistTarget").add_histogram(x="c", bins=7)
        spec = dash.to_spec(source_name=_SRC)
        src_fig = spec.figures[0]
        src_uid = src_fig.traces[0].uid
        line_uid = spec.figures[1].traces[0].uid
        hist_uid = spec.figures[2].traces[0].uid

        resp = client.post(
            "/dashboard/update",
            json=_cube_payload(spec, src_fig.uid, column="a", trace_uid=src_uid),
        )
        assert resp.status_code == 200, resp.text
        body = _cube_body(resp)
        assert line_uid not in body["trace_cubes"]
        assert body["trace_cubes"] == {hist_uid: 0}
        assert len(body["cubes"]) == 1

    def test_box2d_source_skips_corr_target(self, client, df):
        dash = Dashboard(df)
        dash.add_figure(title="Box2dSource").add_histogram2d(
            x="a", y="b", x_bins=8, y_bins=6
        )
        dash.add_figure(title="CorrTarget").add_corr_heatmap(columns=["a", "c"])
        spec = dash.to_spec(source_name=_SRC)
        src_fig = spec.figures[0]
        src_uid = src_fig.traces[0].uid
        corr_uid = spec.figures[1].traces[0].uid

        resp = client.post(
            "/dashboard/update",
            json=_cube_payload(spec, src_fig.uid, column="a", trace_uid=src_uid),
        )
        assert resp.status_code == 200, resp.text
        body = _cube_body(resp)
        assert corr_uid not in body["trace_cubes"]
        assert body["cubes"] == []


class TestCategoricalLineCorrTargets:
    """A categorical source (bar/pie/treemap) now serves line_env + corr targets
    (the categorical-free extension); box2d sources still don't (#47)."""

    def test_bar_source_serves_line_target(self, cat_client, cat_df):
        dash = Dashboard(cat_df)
        dash.add_figure(title="BarSource").add_bar(labels="cat", values="b", agg="mean")
        dash.add_figure(title="LineTarget").add_line(x="a", y="b")
        spec = dash.to_spec(source_name=_CAT_SRC)
        line_uid = spec.figures[1].traces[0].uid
        resp = cat_client.post(
            "/dashboard/update",
            json=_cube_payload(spec, spec.figures[0].uid, column="cat"),
        )
        assert resp.status_code == 200, resp.text
        body = _cube_body(resp)
        assert line_uid in body["trace_cubes"]
        blob = base64.b64decode(body["cubes"][body["trace_cubes"][line_uid]])
        header = decode_fvcube_header(blob)
        assert header["free"]["kind"] == "categorical"
        assert header["measure"]["agg"] == "line_env"

    def test_bar_source_serves_corr_target(self, cat_client, cat_df):
        dash = Dashboard(cat_df)
        dash.add_figure(title="BarSource").add_bar(labels="cat")
        dash.add_figure(title="CorrTarget").add_corr_heatmap(columns=["a", "b"])
        spec = dash.to_spec(source_name=_CAT_SRC)
        corr_uid = spec.figures[1].traces[0].uid
        resp = cat_client.post(
            "/dashboard/update",
            json=_cube_payload(spec, spec.figures[0].uid, column="cat"),
        )
        assert resp.status_code == 200, resp.text
        body = _cube_body(resp)
        assert corr_uid in body["trace_cubes"]
        header = decode_fvcube_header(
            base64.b64decode(body["cubes"][body["trace_cubes"][corr_uid]])
        )
        assert header["free"]["kind"] == "categorical"
        assert header["measure"]["agg"] == "corr"

    def test_pie_source_serves_line_target(self, cat_client, cat_df):
        dash = Dashboard(cat_df)
        dash.add_figure(title="PieSource").add_pie(labels="cat", values="b")
        dash.add_figure(title="LineTarget").add_line(x="a", y="b")
        spec = dash.to_spec(source_name=_CAT_SRC)
        line_uid = spec.figures[1].traces[0].uid
        resp = cat_client.post(
            "/dashboard/update",
            json=_cube_payload(spec, spec.figures[0].uid, column="cat"),
        )
        assert resp.status_code == 200, resp.text
        assert line_uid in _cube_body(resp)["trace_cubes"]

    def test_grouped_line_target_served(self, cat_client, cat_df):
        dash = Dashboard(cat_df)
        dash.add_figure(title="BarSource").add_bar(labels="cat")
        dash.add_figure(title="GroupedLine").add_line(x="a", y="b", group_by="sub")
        spec = dash.to_spec(source_name=_CAT_SRC)
        line_uid = spec.figures[1].traces[0].uid
        resp = cat_client.post(
            "/dashboard/update",
            json=_cube_payload(spec, spec.figures[0].uid, column="cat"),
        )
        assert resp.status_code == 200, resp.text
        assert line_uid in _cube_body(resp)["trace_cubes"]

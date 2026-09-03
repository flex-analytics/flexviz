"""Browser tests for the client cube runtime + live-brush loop (cube Phase 1).

Step-6 core tests:

1. ``test_live_brush_zero_roundtrips`` — a select-mode drag on a hist source
   fires exactly one ``cube_request`` POST at gesture start, zero further
   POSTs through mouseup, and the target hist updates live mid-drag.
2. ``test_commit_counts_match_reference`` — the committed (snapped,
   ``closed="left"``) selection leaves the target rendering counts that are
   bit-exact against a Python-side ``flexviz.cube`` build + slice over the
   same snapped bins (the §8.2 parity property, through the real browser).
3. ``test_live_brush_off_is_legacy`` — ``live_brush="off"`` restores today's
   behavior byte-for-byte: no cube_request, one selection POST on mouseup,
   unsnapped closed-interval predicate, no mid-drag target updates.

Step-7 coverage:

4. ``test_second_brush_is_store_hit`` — a second gesture on the same source
   is a client-store hit: zero POSTs of any kind through its commit.
5. ``test_mixed_dashboard_commit_posts_once`` — a non-cube-capable target
   (box plot) holds pre-drag state during the drag and forces exactly one
   snapped ``closed="left"`` selection POST on commit; both targets are
   correct afterwards (box server-computed, hist self-healed by the deltas).
6. ``test_abandoned_gesture_restores_targets`` — an empty ``plotly_selected``
   mid-gesture restores the targets' pre-drag rendering; nothing committed.
7. ``test_reset_deselect_unchanged`` — deselect/reset after a cube-committed
   selection still run on the response-cache path (no POST, no errors).

Phase-2 coverage (bar/pie/grouped targets, range-source gestures):

8. ``test_grouped_hist_target_live_updates`` — a grouped hist target is now
   cube-capable: its children reconcile live during the drag, the commit is
   local (one cube_request, zero further POSTs), per-child counts match a
   per-group cube reference, and child uids are never minted client-side.
9. ``test_bar_and_pie_targets_live_update`` — bar + pie targets live-update
   during a hist brush with exactly one cube_request and zero further POSTs
   through commit; committed values equal a direct Polars recompute over the
   snapped ``closed="left"`` edges.
10. ``test_mean_bar_target_commit_parity`` — a ``agg="mean"`` bar target's
    committed values are within 1e-9 rel of a direct reference.
11. ``test_median_bar_target_mixed_dashboard`` — a ``agg="median"`` bar
    target is not cube-capable: the capable hist still live-updates, the
    commit POSTs exactly once, and all returned deltas are applied.
12. ``test_float_label_bar_target_live_updates`` — a bar whose labels column
    is float-typed live-updates during a range brush; integral floats such as
    1.0 stay typed through the cube header instead of demoting or matching an
    empty stringified slice.

Phase-2 coverage (categorical sources: bar box-drag, pie/treemap clicks —
``TestCategoricalSourceCube``):

13. ``test_bar_source_drag_live_updates_hist`` — a box-drag over two bars of
    a categorical bar source live-updates a hist target (one cube_request,
    zero further POSTs through commit); the committed selection carries the
    unchanged legacy is_in predicates (no snapping, no ``closed``) and the
    hist equals a direct ``is_in`` recompute.
14. ``test_pie_click_second_click_local`` — first pie-slice click POSTs the
    selection and fires one fire-and-forget cube_request; the second click
    (OR toggle) commits locally with zero further POSTs, matching a direct
    recompute over the two-label union.
15. ``test_treemap_depth2_click_prefix_served`` — depth-2 treemap node
    clicks: the first POSTs + warms the cube; a second click on a different
    depth-2 node is served from the cube (prefix/path predicates, zero
    further POSTs) and matches the recompute for the accumulated predicates.
16. ``test_live_brush_off_categorical_legacy`` — ``live_brush="off"`` gates
    clicks too: bar box-drag and pie click never request cubes, POST exactly
    once per commit, and never live-update mid-drag.
"""

from __future__ import annotations

import json
import math

import polars as pl
import pytest
from playwright.sync_api import Page, Request as PWRequest

import flexviz_polars  # noqa: F401 — registers pl.Expr.flexviz namespace
from flexviz.cube import (
    CubeSpec,
    FreeAxisSpec,
    MeasureSpec,
    TargetDimSpec,
    build_cube,
    encode_cube_bundle,
    encode_fvcube,
)
from flexviz.trace.hist import _HIST_BIN_EPSILON

from tests.test_browser import _wait_for_init

pytestmark = pytest.mark.browser

_P = 2048
_SRC_BINS = 16
_TGT_BINS = 12


def _cube_df() -> pl.DataFrame:
    """Small dataset with a non-uniform a↔b relationship, so restricting the
    source column ``a`` visibly reshapes the target histogram on ``b``."""
    n = 3_000
    return pl.DataFrame(
        {
            "a": [((i * 37) % 1000) / 10 for i in range(n)],
            "b": [((i * 53) % 500) / 5 for i in range(n)],
        }
    )


def _two_hist_dashboard_url(port: int, source_name: str, live_brush: str) -> str:
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import LayoutSpec, encode_spec

    df = _cube_df()
    register_source(source_name, df, cache=True)

    dash = Dashboard(df)
    dash.add_figure(title="Source").add_histogram(x="a", bins=_SRC_BINS)
    dash.add_figure(title="Target").add_histogram(x="b", bins=_TGT_BINS)
    spec = dash.to_spec(source_name=source_name, layout=LayoutSpec(draggable=False))
    spec.client_state.live_brush = live_brush
    return f"http://127.0.0.1:{port}/view?spec={encode_spec(spec)}&renderer=plotly"


def _mixed_dashboard_url(port: int, source_name: str) -> str:
    """Source hist(a) + cube-capable hist(b) target + box(b) target (box is
    not cube-capable → conditional commit must POST)."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import LayoutSpec, encode_spec

    df = _cube_df()
    register_source(source_name, df, cache=True)

    dash = Dashboard(df)
    dash.add_figure(title="Source").add_histogram(x="a", bins=_SRC_BINS)
    dash.add_figure(title="Target").add_histogram(x="b", bins=_TGT_BINS)
    dash.add_figure(title="Box").add_boxplot(y="b")
    spec = dash.to_spec(source_name=source_name, layout=LayoutSpec(draggable=False))
    spec.client_state.live_brush = "auto"
    return f"http://127.0.0.1:{port}/view?spec={encode_spec(spec)}&renderer=plotly"


def _grouped_df() -> pl.DataFrame:
    df = _cube_df()
    return df.with_columns(
        pl.Series("cat", ["A" if i % 2 == 0 else "B" for i in range(df.height)])
    )


def _grouped_dashboard_url(port: int, source_name: str) -> str:
    """Source hist(a) + grouped hist(b, group_by=cat) target — the grouped
    target is cube-capable since Phase 2 (group cols as categorical dims)."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import LayoutSpec, encode_spec

    df = _grouped_df()
    register_source(source_name, df, cache=True)

    dash = Dashboard(df)
    dash.add_figure(title="Source").add_histogram(x="a", bins=_SRC_BINS)
    dash.add_figure(title="Grouped").add_histogram(
        x="b", bins=_TGT_BINS, group_by="cat"
    )
    spec = dash.to_spec(source_name=source_name, layout=LayoutSpec(draggable=False))
    spec.client_state.live_brush = "auto"
    return f"http://127.0.0.1:{port}/view?spec={encode_spec(spec)}&renderer=plotly"


def _cat_target_df() -> pl.DataFrame:
    """`_cube_df` plus a low-cardinality string column ``g`` (a↔g related
    non-uniformly, so restricting ``a`` reshapes the per-label aggregates) and
    a numeric values column ``v`` for mean/median measures. ``v`` tracks ``a``
    (plus per-row noise) so a brush on ``a`` always shifts every per-label
    mean/median — a filtered aggregate can never echo the unfiltered one."""
    df = _cube_df()
    n = df.height
    return df.with_columns(
        pl.Series("g", [f"g{(i * 7) % 4}" for i in range(n)]),
        pl.Series("v", [((i * 37) % 1000) / 10 + (i % 7) for i in range(n)]),
    )


def _bar_pie_dashboard_url(port: int, source_name: str) -> str:
    """Source hist(a) + bar(labels=g) target + pie(labels=g) target — both
    categorical count targets (and a shared cube: same labels + measure)."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import LayoutSpec, encode_spec

    df = _cat_target_df()
    register_source(source_name, df, cache=True)

    dash = Dashboard(df)
    dash.add_figure(title="Source").add_histogram(x="a", bins=_SRC_BINS)
    dash.add_figure(title="Bar").add_bar(labels="g")
    dash.add_figure(title="Pie").add_pie(labels="g")
    spec = dash.to_spec(source_name=source_name, layout=LayoutSpec(draggable=False))
    spec.client_state.live_brush = "auto"
    return f"http://127.0.0.1:{port}/view?spec={encode_spec(spec)}&renderer=plotly"


def _mean_bar_dashboard_url(port: int, source_name: str) -> str:
    """Source hist(a) + bar(labels=g, values=v, agg=mean) target."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import LayoutSpec, encode_spec

    df = _cat_target_df()
    register_source(source_name, df, cache=True)

    dash = Dashboard(df)
    dash.add_figure(title="Source").add_histogram(x="a", bins=_SRC_BINS)
    dash.add_figure(title="MeanBar").add_bar(labels="g", values="v", agg="mean")
    spec = dash.to_spec(source_name=source_name, layout=LayoutSpec(draggable=False))
    spec.client_state.live_brush = "auto"
    return f"http://127.0.0.1:{port}/view?spec={encode_spec(spec)}&renderer=plotly"


def _median_mixed_dashboard_url(port: int, source_name: str) -> str:
    """Source hist(a) + cube-capable hist(b) target + bar(agg=median) target
    (median is outside the cube measure algebra → conditional commit POSTs)."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import LayoutSpec, encode_spec

    df = _cat_target_df()
    register_source(source_name, df, cache=True)

    dash = Dashboard(df)
    dash.add_figure(title="Source").add_histogram(x="a", bins=_SRC_BINS)
    dash.add_figure(title="Target").add_histogram(x="b", bins=_TGT_BINS)
    dash.add_figure(title="MedianBar").add_bar(labels="g", values="v", agg="median")
    spec = dash.to_spec(source_name=source_name, layout=LayoutSpec(draggable=False))
    spec.client_state.live_brush = "auto"
    return f"http://127.0.0.1:{port}/view?spec={encode_spec(spec)}&renderer=plotly"


def _numeric_label_df() -> pl.DataFrame:
    """`_cube_df` plus a low-cardinality float labels column ``nlab``.
    Numeric labels are categorical cube dims: the FVCube header preserves
    numeric category values instead of relying on Python/JS string parity."""
    df = _cube_df()
    return df.with_columns(pl.Series("nlab", [float(i % 4) for i in range(df.height)]))


def _numeric_label_dashboard_url(port: int, source_name: str) -> str:
    """Source hist(a) + bar(labels=nlab[float]) target."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import LayoutSpec, encode_spec

    df = _numeric_label_df()
    register_source(source_name, df, cache=True)

    dash = Dashboard(df)
    dash.add_figure(title="Source").add_histogram(x="a", bins=_SRC_BINS)
    dash.add_figure(title="Target").add_histogram(x="b", bins=_TGT_BINS)
    dash.add_figure(title="NumBar").add_bar(labels="nlab")
    spec = dash.to_spec(source_name=source_name, layout=LayoutSpec(draggable=False))
    spec.client_state.live_brush = "auto"
    return f"http://127.0.0.1:{port}/view?spec={encode_spec(spec)}&renderer=plotly"


def _float_bar_source_df() -> pl.DataFrame:
    """`_cat_target_df` plus a FLOAT label column ``gf`` mirroring ``g``.

    Values are 0.0..3.0, deliberately including integral floats where Python
    ``str(1.0)`` and JS ``String(1)`` used to diverge.
    """
    df = _cat_target_df()
    return df.with_columns(pl.col("g").str.slice(1).cast(pl.Float64).alias("gf"))


def _float_bar_source_dashboard_url(port: int, source_name: str) -> str:
    """Float-label bar source + sum bar/pie targets."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import LayoutSpec, encode_spec

    df = _float_bar_source_df()
    register_source(source_name, df, cache=True)

    dash = Dashboard(df)
    dash.add_figure(title="FloatBarSource").add_bar(labels="gf")
    dash.add_figure(title="SumBar").add_bar(labels="g", values="v", agg="sum")
    dash.add_figure(title="SumPie").add_pie(labels="g", values="v", agg="sum")
    spec = dash.to_spec(source_name=source_name, layout=LayoutSpec(draggable=False))
    spec.client_state.live_brush = "auto"
    return f"http://127.0.0.1:{port}/view?spec={encode_spec(spec)}&renderer=plotly"


_GROUP_COLOR_MAP = {"P": "#e3a24d", "Q": "#5b8db8"}


def _grouped_color_df() -> pl.DataFrame:
    """`_cat_target_df` plus a 2-value string group column ``k`` so a grouped
    bar (labels=g, group_by=k) renders one coloured child per group value —
    mirroring the demo's grouped "Total generation" bars."""
    df = _cat_target_df()
    n = df.height
    return df.with_columns(
        pl.Series("k", ["P" if i % 2 == 0 else "Q" for i in range(n)])
    )


def _grouped_color_bar_dashboard_url(port: int, source_name: str) -> str:
    """Source hist(a) + grouped bar(labels=g, group_by=k) whose ``color_map`` is
    keyed by the GROUP values (not the labels) — exactly the demo's setup."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import LayoutSpec, encode_spec

    df = _grouped_color_df()
    register_source(source_name, df, cache=True)

    dash = Dashboard(df)
    dash.add_figure(title="Source").add_histogram(x="a", bins=_SRC_BINS)
    dash.add_figure(title="GroupedBar").add_bar(
        labels="g", group_by="k", color_map=_GROUP_COLOR_MAP
    )
    spec = dash.to_spec(source_name=source_name, layout=LayoutSpec(draggable=False))
    spec.client_state.live_brush = "auto"
    return f"http://127.0.0.1:{port}/view?spec={encode_spec(spec)}&renderer=plotly"


def _int_label_df() -> pl.DataFrame:
    """`_cube_df` plus a low-cardinality *integer* labels column ``ilab`` whose
    values span 0..11 — so numeric order (0,1,2,…,10,11) differs from the
    lexicographic string order ("0","1","10","11","2",…). Integer label dtypes
    are cube-capable and stay typed through the cube header."""
    df = _cube_df()
    n = df.height
    return df.with_columns(
        pl.Series("ilab", [i % 12 for i in range(n)], dtype=pl.Int64)
    )


def _int_label_dashboard_url(port: int, source_name: str) -> str:
    """Source hist(a) + bar(labels=ilab) where ``ilab`` is integer-typed —
    mirrors the demo's hour_of_day / month bars (Int8 ``dt`` accessors)."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import LayoutSpec, encode_spec

    df = _int_label_df()
    register_source(source_name, df, cache=True)

    dash = Dashboard(df)
    dash.add_figure(title="Source").add_histogram(x="a", bins=_SRC_BINS)
    dash.add_figure(title="IntBar").add_bar(labels="ilab")
    spec = dash.to_spec(source_name=source_name, layout=LayoutSpec(draggable=False))
    spec.client_state.live_brush = "auto"
    return f"http://127.0.0.1:{port}/view?spec={encode_spec(spec)}&renderer=plotly"


def _grouped_int_color_df() -> pl.DataFrame:
    """`_cube_df` plus an *integer* labels column ``ilab`` (0..11) and a 2-value
    string group column ``k`` — the demo's hour_of_day / month bars exactly:
    integer labels, string group, coloured by the group values."""
    df = _cube_df()
    n = df.height
    return df.with_columns(
        pl.Series("ilab", [i % 12 for i in range(n)], dtype=pl.Int64),
        pl.Series("k", ["P" if i % 2 == 0 else "Q" for i in range(n)]),
    )


def _grouped_int_color_dashboard_url(port: int, source_name: str) -> str:
    """Source hist(a) + grouped bar(labels=ilab[int], group_by=k, color_map
    keyed by the group values) — the demo's grouped integer-label bars, where
    Bug 1 (group colours) and Bug 2 (integer labels) co-occur."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import LayoutSpec, encode_spec

    df = _grouped_int_color_df()
    register_source(source_name, df, cache=True)

    dash = Dashboard(df)
    dash.add_figure(title="Source").add_histogram(x="a", bins=_SRC_BINS)
    dash.add_figure(title="GroupedIntBar").add_bar(
        labels="ilab", group_by="k", color_map=_GROUP_COLOR_MAP
    )
    spec = dash.to_spec(source_name=source_name, layout=LayoutSpec(draggable=False))
    spec.client_state.live_brush = "auto"
    return f"http://127.0.0.1:{port}/view?spec={encode_spec(spec)}&renderer=plotly"


def _capture_posts(page: Page) -> list[str]:
    """Record EVERY POST request URL for the page (any endpoint)."""
    posts: list[str] = []

    def capture(req: PWRequest) -> None:
        if req.method == "POST":
            posts.append(req.url)

    page.on("request", capture)
    return posts


def _capture_updates(page: Page) -> list[dict]:
    """Record every /dashboard/update POST body for the page."""
    bodies: list[dict] = []

    def capture(req: PWRequest) -> None:
        if "/dashboard/update" in req.url and req.method == "POST":
            try:
                bodies.append(json.loads(req.post_data or "{}"))
            except Exception:
                pass

    page.on("request", capture)
    return bodies


def _target_y(page: Page) -> list[float]:
    return page.eval_on_selector(
        "#fv-plot-1", "gd => Array.from((gd.data && gd.data[0] && gd.data[0].y) || [])"
    )


def _enter_select_mode(page: Page) -> None:
    page.locator("#fv-bar-0 .fv-mode-btn[data-mode='select']").click()
    page.wait_for_timeout(300)


def _drag_coords(page: Page) -> tuple[float, float, float, float]:
    box = page.locator("#fv-plot-0 .nsewdrag").bounding_box()
    assert box is not None
    x1 = box["x"] + box["width"] * 0.2
    x2 = box["x"] + box["width"] * 0.6
    y = box["y"] + box["height"] * 0.5
    return x1, x2, y, box["width"]


def _drag_with_live_wait(
    page: Page, x1: float, x2: float, y: float, y_baseline: list[float]
) -> None:
    """One full live-brush gesture: drag, wait for the mid-drag local
    re-render of the target hist (vs *y_baseline*), then commit (mouseup)."""
    page.mouse.move(x1, y)
    page.mouse.down()
    page.mouse.move((x1 + x2) / 2, y, steps=8)
    page.wait_for_function(
        """(yBefore) => {
            const gd = document.querySelector('#fv-plot-1');
            const ys = Array.from((gd.data && gd.data[0] && gd.data[0].y) || []);
            return ys.length > 0 && JSON.stringify(ys) !== JSON.stringify(yBefore);
        }""",
        arg=y_baseline,
        timeout=10_000,
    )
    page.mouse.move(x2, y, steps=8)
    page.wait_for_timeout(300)
    page.mouse.up()


def _committed_edges(page: Page, column: str) -> tuple[float, float]:
    """The single committed snapped ``closed="left"`` clause's range."""
    sels = page.evaluate("DASHBOARD_SPEC.state.selections")
    assert len(sels) == 1
    clause = sels[0]["predicates"][0]["clauses"][0]
    assert clause["column"] == column
    assert clause.get("closed") == "left"
    return clause["range"][0], clause["range"][1]


def _reference_slice_counts(
    df: pl.DataFrame, edge_lo: float, edge_hi: float, free_col: str = "a"
) -> list[float]:
    """Python reference for the target hist: ``flexviz.cube`` build + slice
    over the snapped bins recovered from committed ``closed="left"`` edges
    (the §8.2 parity property — identical to the legacy server recompute)."""
    a_lo, a_hi = df[free_col].min(), df[free_col].max()
    span = a_hi - a_lo
    lo_bin = round((edge_lo - a_lo) / span * _P)
    hi_bin = round((edge_hi - a_lo) / span * _P) - 1
    assert 0 <= lo_bin <= hi_bin <= _P
    # Committed edges must lie exactly on the P-grid over the full domain.
    assert edge_lo == pytest.approx(a_lo + lo_bin * span / _P, abs=1e-9)
    assert edge_hi == pytest.approx(a_lo + (hi_bin + 1) * span / _P, abs=1e-9)

    b_lo, b_hi = df["b"].min(), df["b"].max()
    spec = CubeSpec(
        source_name="_reference",
        free=FreeAxisSpec(
            column=free_col, kind="continuous", p=_P, domain=(a_lo, a_hi)
        ),
        target_dims=(
            TargetDimSpec(
                column="b",
                kind="binned",
                bins=_TGT_BINS,
                domain=(b_lo, b_hi + _HIST_BIN_EPSILON),
            ),
        ),
        measure=MeasureSpec(agg="count"),
    )
    result = build_cube(df.lazy(), spec)
    sliced = (
        result.frame.filter(pl.col("free_bin").is_between(lo_bin, hi_bin))
        .group_by("__bin__b")
        .agg(pl.col("count").sum())
    )
    expected = [0.0] * _TGT_BINS
    for tb, n in sliced.iter_rows():
        expected[tb] += n
    return expected


def _reference_grouped_slice_counts(
    df: pl.DataFrame, edge_lo: float, edge_hi: float, group_col: str = "cat"
) -> dict[str, list[float]]:
    """Per-group variant of ``_reference_slice_counts``: a grouped cube
    (binned ``b`` × categorical *group_col*) built + sliced over the snapped
    bins recovered from committed ``closed="left"`` edges."""
    a_lo, a_hi = df["a"].min(), df["a"].max()
    span = a_hi - a_lo
    lo_bin = round((edge_lo - a_lo) / span * _P)
    hi_bin = round((edge_hi - a_lo) / span * _P) - 1
    assert 0 <= lo_bin <= hi_bin <= _P
    assert edge_lo == pytest.approx(a_lo + lo_bin * span / _P, abs=1e-9)
    assert edge_hi == pytest.approx(a_lo + (hi_bin + 1) * span / _P, abs=1e-9)

    b_lo, b_hi = df["b"].min(), df["b"].max()
    spec = CubeSpec(
        source_name="_reference",
        free=FreeAxisSpec(column="a", kind="continuous", p=_P, domain=(a_lo, a_hi)),
        target_dims=(
            TargetDimSpec(
                column="b",
                kind="binned",
                bins=_TGT_BINS,
                domain=(b_lo, b_hi + _HIST_BIN_EPSILON),
            ),
            TargetDimSpec(column=group_col, kind="categorical"),
        ),
        measure=MeasureSpec(agg="count"),
    )
    result = build_cube(df.lazy(), spec)
    sliced = (
        result.frame.filter(pl.col("free_bin").is_between(lo_bin, hi_bin))
        .group_by(group_col, "__bin__b")
        .agg(pl.col("count").sum())
    )
    expected: dict[str, list[float]] = {}
    for gv, tb, n in sliced.iter_rows():
        expected.setdefault(gv, [0.0] * _TGT_BINS)[tb] += n
    return expected


def _bar_xy(page: Page, selector: str) -> dict:
    return page.eval_on_selector(
        selector,
        "gd => ({x: Array.from(gd.data[0].x || []), "
        "y: Array.from(gd.data[0].y || [])})",
    )


def _pie_data(page: Page, selector: str) -> dict:
    return page.eval_on_selector(
        selector,
        "gd => ({labels: Array.from(gd.data[0].labels || []), "
        "values: Array.from(gd.data[0].values || [])})",
    )


class TestLiveBrushCube:
    def test_live_brush_zero_roundtrips(self, page: Page, server_port: int):
        url = _two_hist_dashboard_url(server_port, "_cube_browser_auto_rt", "auto")
        bodies = _capture_updates(page)
        page.goto(url)
        _wait_for_init(page, "plotly")
        _enter_select_mode(page)

        y_before = _target_y(page)
        assert sum(y_before) > 0

        x1, x2, y, _width = _drag_coords(page)
        n_before_drag = len(bodies)

        page.mouse.move(x1, y)
        page.mouse.down()
        page.mouse.move((x1 + x2) / 2, y, steps=8)
        # Mid-drag, still holding the button: the cube must land and the
        # target must re-render from a local slice (no further POSTs).
        page.wait_for_function(
            """(yBefore) => {
                const gd = document.querySelector('#fv-plot-1');
                const ys = Array.from((gd.data && gd.data[0] && gd.data[0].y) || []);
                return ys.length > 0 && JSON.stringify(ys) !== JSON.stringify(yBefore);
            }""",
            arg=y_before,
            timeout=10_000,
        )
        y_mid = _target_y(page)
        assert y_mid != y_before, "target must update live during the drag"
        page.mouse.move(x2, y, steps=8)
        page.wait_for_timeout(300)
        page.mouse.up()
        page.wait_for_timeout(1_500)

        gesture_bodies = bodies[n_before_drag:]
        cube_requests = [
            b
            for b in gesture_bodies
            if b.get("event", {}).get("type") == "cube_request"
        ]
        assert len(cube_requests) == 1, (
            f"expected exactly one cube_request, got "
            f"{[b.get('event', {}).get('type') for b in gesture_bodies]}"
        )
        assert cube_requests[0].get("request_cube") is True
        active = cube_requests[0].get("active_source") or {}
        assert active.get("column") == "a"
        # The conditional commit: every target was cube-served, so the commit
        # is local — zero further POSTs of any type through mouseup.
        others = [b for b in gesture_bodies if b not in cube_requests]
        assert others == [], (
            f"live cube gesture must not POST beyond the single cube_request; "
            f"got {[b.get('event', {}).get('type') for b in others]}"
        )
        # The selection committed client-side.
        sels = page.evaluate("DASHBOARD_SPEC.state.selections")
        assert len(sels) == 1 and sels[0]["predicates"]

    def test_commit_counts_match_reference(self, page: Page, server_port: int):
        df = _cube_df()
        url = _two_hist_dashboard_url(server_port, "_cube_browser_auto_ref", "auto")
        page.goto(url)
        _wait_for_init(page, "plotly")
        _enter_select_mode(page)

        y_before = _target_y(page)
        x1, x2, y, _width = _drag_coords(page)
        page.mouse.move(x1, y)
        page.mouse.down()
        page.mouse.move((x1 + x2) / 2, y, steps=8)
        page.wait_for_function(
            """(yBefore) => {
                const gd = document.querySelector('#fv-plot-1');
                const ys = Array.from((gd.data && gd.data[0] && gd.data[0].y) || []);
                return ys.length > 0 && JSON.stringify(ys) !== JSON.stringify(yBefore);
            }""",
            arg=y_before,
            timeout=10_000,
        )
        page.mouse.move(x2, y, steps=8)
        page.wait_for_timeout(300)
        page.mouse.up()
        page.wait_for_timeout(800)

        # Committed predicate: snapped to the P-grid, closed="left".
        sels = page.evaluate("DASHBOARD_SPEC.state.selections")
        assert len(sels) == 1
        clause = sels[0]["predicates"][0]["clauses"][0]
        assert clause["column"] == "a"
        assert clause.get("closed") == "left"
        edge_lo, edge_hi = clause["range"]

        # Shared arithmetic (plan doc): bin(v) = floor((v-lo)/s*P),
        # edge(b) = lo + b*s/P; committed range = [edge(lo_bin), edge(hi_bin+1)).
        a_lo, a_hi = df["a"].min(), df["a"].max()
        span = a_hi - a_lo
        lo_bin = round((edge_lo - a_lo) / span * _P)
        hi_bin = round((edge_hi - a_lo) / span * _P) - 1
        assert 0 <= lo_bin <= hi_bin <= _P
        # The committed edges must lie exactly on the P-grid.
        assert edge_lo == pytest.approx(a_lo + lo_bin * span / _P, abs=1e-9)
        assert edge_hi == pytest.approx(a_lo + (hi_bin + 1) * span / _P, abs=1e-9)

        # Python reference: flexviz.cube build + slice over the same bins.
        b_lo, b_hi = df["b"].min(), df["b"].max()
        spec = CubeSpec(
            source_name="_cube_browser_auto_ref",
            free=FreeAxisSpec(column="a", kind="continuous", p=_P, domain=(a_lo, a_hi)),
            target_dims=(
                TargetDimSpec(
                    column="b",
                    kind="binned",
                    bins=_TGT_BINS,
                    domain=(b_lo, b_hi + _HIST_BIN_EPSILON),
                ),
            ),
            measure=MeasureSpec(agg="count"),
        )
        result = build_cube(df.lazy(), spec)
        sliced = (
            result.frame.filter(pl.col("free_bin").is_between(lo_bin, hi_bin))
            .group_by("__bin__b")
            .agg(pl.col("count").sum())
        )
        expected = [0.0] * _TGT_BINS
        for tb, n in sliced.iter_rows():
            expected[tb] += n

        rendered = _target_y(page)
        assert sum(expected) > 0  # non-vacuous brush
        assert sum(expected) < df.height  # ... that does not select everything
        assert rendered == expected  # bit-exact counts

    def test_live_brush_off_is_legacy(self, page: Page, server_port: int):
        url = _two_hist_dashboard_url(server_port, "_cube_browser_off", "off")
        bodies = _capture_updates(page)
        page.goto(url)
        _wait_for_init(page, "plotly")
        _enter_select_mode(page)

        y_before = _target_y(page)
        x1, x2, y, _width = _drag_coords(page)
        n_before_drag = len(bodies)

        page.mouse.move(x1, y)
        page.mouse.down()
        page.mouse.move((x1 + x2) / 2, y, steps=8)
        page.wait_for_timeout(800)
        assert _target_y(page) == y_before, "no mid-drag updates with live_brush=off"
        page.mouse.move(x2, y, steps=8)
        page.wait_for_timeout(400)
        assert _target_y(page) == y_before, "no mid-drag updates with live_brush=off"
        page.mouse.up()
        page.wait_for_timeout(1_500)

        gesture_bodies = bodies[n_before_drag:]
        types = [b.get("event", {}).get("type") for b in gesture_bodies]
        assert (
            "cube_request" not in types
        ), f"live_brush=off must never request cubes: {types}"
        assert types == [
            "selection"
        ], f"expected exactly one selection POST, got {types}"

        # Legacy predicate: raw (unsnapped) drag range, closed-interval default.
        sels = page.evaluate("DASHBOARD_SPEC.state.selections")
        assert len(sels) == 1
        clause = sels[0]["predicates"][0]["clauses"][0]
        assert clause["column"] == "a"
        assert clause.get("closed") in (None, "both")
        lo, hi = clause["range"]
        # Unsnapped: an arbitrary pixel-derived range almost surely misses the
        # exact P-grid (probability ~0 of both edges landing on it).
        df = _cube_df()
        a_lo, a_hi = df["a"].min(), df["a"].max()
        span = a_hi - a_lo

        def _on_grid(v: float) -> bool:
            scaled = (v - a_lo) / span * _P
            return math.isclose(scaled, round(scaled), abs_tol=1e-6)

        assert not (
            _on_grid(lo) and _on_grid(hi)
        ), f"predicate looks snapped ({lo}, {hi}) — legacy path must be unsnapped"
        # And the target did update after the server round-trip.
        assert _target_y(page) != y_before

    def test_second_brush_is_store_hit(self, page: Page, server_port: int):
        """After one full gesture populated the client cube store, a second
        gesture on the same source is fully local: zero POSTs of any kind
        through its commit, with correct rendered values for the new range."""
        df = _cube_df()
        url = _two_hist_dashboard_url(server_port, "_cube_browser_store_hit", "auto")
        bodies = _capture_updates(page)
        posts = _capture_posts(page)
        page.goto(url)
        _wait_for_init(page, "plotly")
        _enter_select_mode(page)

        # Gesture 1: populates the store via the one cube_request.
        y_unfiltered = _target_y(page)
        x1, x2, y, width = _drag_coords(page)
        _drag_with_live_wait(page, x1, x2, y, y_unfiltered)
        page.wait_for_timeout(1_000)
        first_edges = _committed_edges(page, "a")

        # Clear the selection (response-cache path; not under test here).
        page.click("#fv-btn-deselect")
        page.wait_for_timeout(800)
        assert page.evaluate("DASHBOARD_SPEC.state.selections") == []
        assert _target_y(page) == y_unfiltered

        # Gesture 2 over a different range: must be a pure store hit.
        n_bodies = len(bodies)
        n_posts = len(posts)
        x1b = x1 + width * 0.1
        x2b = x2 + width * 0.2
        _drag_with_live_wait(page, x1b, x2b, y, y_unfiltered)
        page.wait_for_timeout(1_000)

        assert posts[n_posts:] == [], (
            f"second gesture must be served from the cube store with zero "
            f"POSTs of any kind; got {posts[n_posts:]}"
        )
        assert bodies[n_bodies:] == []

        # The local commit is correct for the *second* range.
        edge_lo, edge_hi = _committed_edges(page, "a")
        assert (edge_lo, edge_hi) != first_edges
        expected = _reference_slice_counts(df, edge_lo, edge_hi)
        assert sum(expected) > 0
        assert sum(expected) < df.height
        assert _target_y(page) == expected

    def test_mixed_dashboard_commit_posts_once(self, page: Page, server_port: int):
        """A box-plot target is not cube-capable: it holds its pre-drag state
        during the drag, and the conditional commit falls back to exactly one
        snapped ``closed="left"`` selection POST. Afterwards the box is
        server-correct and the hist matches the cube slice (self-healing)."""
        df = _cube_df()
        url = _mixed_dashboard_url(server_port, "_cube_browser_mixed")
        bodies = _capture_updates(page)
        page.goto(url)
        _wait_for_init(page, "plotly")
        _enter_select_mode(page)

        box_props = ["lowerfence", "q1", "median", "q3", "upperfence"]
        read_box = (
            "gd => Object.fromEntries("
            f"{box_props!r}.map(k => [k, Array.from(gd.data[0][k] || [])]))"
        )
        box_before = page.eval_on_selector("#fv-plot-2", read_box)
        assert box_before["median"], "box must render stats at init"

        y_unfiltered = _target_y(page)
        x1, x2, y, _width = _drag_coords(page)
        n_before = len(bodies)
        page.mouse.move(x1, y)
        page.mouse.down()
        page.mouse.move((x1 + x2) / 2, y, steps=8)
        page.wait_for_function(
            """(yBefore) => {
                const gd = document.querySelector('#fv-plot-1');
                const ys = Array.from((gd.data && gd.data[0] && gd.data[0].y) || []);
                return ys.length > 0 && JSON.stringify(ys) !== JSON.stringify(yBefore);
            }""",
            arg=y_unfiltered,
            timeout=10_000,
        )
        # The hist updated live; the box must still show its pre-drag stats.
        assert page.eval_on_selector("#fv-plot-2", read_box) == box_before
        page.mouse.move(x2, y, steps=8)
        page.wait_for_timeout(300)
        assert page.eval_on_selector("#fv-plot-2", read_box) == box_before
        page.mouse.up()
        # Commit round-trips: wait until the server deltas update the box.
        page.wait_for_function(
            f"""(before) => {{
                const gd = document.querySelector('#fv-plot-2');
                const now = Object.fromEntries(
                    {box_props!r}.map(k => [k, Array.from(gd.data[0][k] || [])]));
                return JSON.stringify(now) !== JSON.stringify(before);
            }}""",
            arg=box_before,
            timeout=10_000,
        )
        page.wait_for_timeout(500)

        # Exactly one cube_request + one selection POST, nothing else.
        types = [b.get("event", {}).get("type") for b in bodies[n_before:]]
        assert sorted(types) == ["cube_request", "selection"], types
        sel_body = next(
            b
            for b in bodies[n_before:]
            if b.get("event", {}).get("type") == "selection"
        )
        clause = sel_body["event"]["selections"][0]["predicates"][0]["clauses"][0]
        assert clause["column"] == "a"
        assert clause.get("closed") == "left", "mixed commit must POST snapped edges"
        edge_lo, edge_hi = clause["range"]

        # Hist target: self-healed values equal the cube-slice reference.
        expected = _reference_slice_counts(df, edge_lo, edge_hi)
        assert 0 < sum(expected) < df.height
        assert _target_y(page) == expected

        # Box target: server-computed quantiles over the snapped half-open
        # range (mirrors BoxPlot.get_aggregation_spec + _to_update).
        filtered = df.filter((pl.col("a") >= edge_lo) & (pl.col("a") < edge_hi))["b"]
        qs = [filtered.quantile(q) for q in (0.0, 0.25, 0.5, 0.75, 1.0)]
        iqr = qs[3] - qs[1]
        expected_box = {
            "lowerfence": max(qs[1] - 1.5 * iqr, qs[0]),
            "q1": qs[1],
            "median": qs[2],
            "q3": qs[3],
            "upperfence": min(qs[3] + 1.5 * iqr, qs[4]),
        }
        box_after = page.eval_on_selector("#fv-plot-2", read_box)
        for prop, want in expected_box.items():
            assert box_after[prop][0] == pytest.approx(want, rel=1e-12), prop

    def test_abandoned_gesture_restores_targets(self, page: Page, server_port: int):
        """A gesture that ends in an empty ``plotly_selected`` (the payload
        Plotly emits when a selection collapses/clears) commits nothing and
        restores the targets' exact pre-drag rendering.

        No deterministic pure-mouse trigger exists in headless Plotly 3:
        Escape mid-drag re-fires a trailing ranged ``plotly_selected`` that
        re-commits, and a degenerate drag-back-to-origin release still emits
        a tiny non-empty range.  So the empty event is delivered through the
        real Plotly event channel (``gd.emit``) while the drag is held —
        byte-identical to what ``handleSelected`` receives from Plotly."""
        url = _two_hist_dashboard_url(server_port, "_cube_browser_abandon", "auto")
        bodies = _capture_updates(page)
        page.goto(url)
        _wait_for_init(page, "plotly")
        _enter_select_mode(page)

        y_before = _target_y(page)
        x1, x2, y, _width = _drag_coords(page)
        n_before = len(bodies)

        page.mouse.move(x1, y)
        page.mouse.down()
        page.mouse.move((x1 + x2) / 2, y, steps=8)
        # Live slicing must visibly change the target mid-drag first.
        page.wait_for_function(
            """(yBefore) => {
                const gd = document.querySelector('#fv-plot-1');
                const ys = Array.from((gd.data && gd.data[0] && gd.data[0].y) || []);
                return ys.length > 0 && JSON.stringify(ys) !== JSON.stringify(yBefore);
            }""",
            arg=y_before,
            timeout=10_000,
        )
        assert _target_y(page) != y_before
        # Abandon while the drag is still pending (button held, gesture not
        # consumed by a commit): empty plotly_selected → abort + restore.
        page.evaluate("divs[0].emit('plotly_selected', undefined)")
        page.wait_for_timeout(500)
        # Under load a trailing throttled plotly_selecting can arrive AFTER
        # the abort (the button is still held) and start a fresh live
        # gesture that re-slices the target. Abandon that one too — a second
        # empty plotly_selected is a no-op when no gesture exists.
        page.evaluate("divs[0].emit('plotly_selected', undefined)")
        page.wait_for_timeout(500)

        assert (
            page.evaluate("DASHBOARD_SPEC.state.selections") == []
        ), "abandoned gesture must not store a selection"
        assert (
            _target_y(page) == y_before
        ), "abandoned gesture must restore the target's pre-drag rendering"
        # The only traffic allowed is the gesture-start cube_request.
        types = [b.get("event", {}).get("type") for b in bodies[n_before:]]
        assert types in ([], ["cube_request"]), types
        # Deliberately no mouse.up(): releasing would emit a real ranged
        # plotly_selected and start a fresh (legacy) commit — out of scope.

    def test_reset_deselect_unchanged(self, page: Page, server_port: int):
        """With a cube-committed (client-side) selection, the toolbar deselect
        and reset still work and stay on the response-cache path: unfiltered
        values come back with zero POSTs and zero page errors."""
        url = _two_hist_dashboard_url(server_port, "_cube_browser_resets", "auto")
        bodies = _capture_updates(page)
        errors: list[str] = []
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.goto(url)
        _wait_for_init(page, "plotly")
        _enter_select_mode(page)

        y_unfiltered = _target_y(page)
        x1, x2, y, _width = _drag_coords(page)

        # Gesture 1 → cube-committed selection (no selection POST).
        _drag_with_live_wait(page, x1, x2, y, y_unfiltered)
        page.wait_for_timeout(800)
        assert len(page.evaluate("DASHBOARD_SPEC.state.selections")) == 1
        assert _target_y(page) != y_unfiltered

        # Deselect: served from the client response cache — no POST.
        n_before = len(bodies)
        page.click("#fv-btn-deselect")
        page.wait_for_timeout(800)
        assert bodies[n_before:] == [], "unzoomed deselect must hit the response cache"
        assert page.evaluate("DASHBOARD_SPEC.state.selections") == []
        assert _target_y(page) == y_unfiltered

        # Gesture 2 (store hit) → committed again, then global reset.
        _drag_with_live_wait(page, x1, x2, y, y_unfiltered)
        page.wait_for_timeout(800)
        assert len(page.evaluate("DASHBOARD_SPEC.state.selections")) == 1

        n_before = len(bodies)
        page.click("#fv-btn-reset")
        page.wait_for_timeout(800)
        assert bodies[n_before:] == [], "reset on a cached source must not POST"
        assert page.evaluate("DASHBOARD_SPEC.state.selections") == []
        assert _target_y(page) == y_unfiltered
        assert errors == []

    def test_grouped_hist_target_live_updates(self, page: Page, server_port: int):
        """A grouped hist target is cube-capable (Phase 2): the children
        reconcile live during the drag, the commit stays local (exactly one
        cube_request, zero further POSTs), the committed per-child counts
        match a per-group cube reference over the snapped edges, and the
        rendered child uids are unchanged from init — never minted."""
        df = _grouped_df()
        url = _grouped_dashboard_url(server_port, "_cube_browser_grouped")
        bodies = _capture_updates(page)
        page.goto(url)
        _wait_for_init(page, "plotly")
        _enter_select_mode(page)

        read_children = (
            "gd => (gd.data || []).map(t => "
            "({uid: t.uid, name: t.name, y: Array.from(t.y || [])}))"
        )
        children_before = page.eval_on_selector("#fv-plot-1", read_children)
        assert len(children_before) == 2, "grouped hist must render two children"
        uids_before = sorted(t["uid"] for t in children_before)

        x1, x2, y, _width = _drag_coords(page)
        n_before = len(bodies)
        page.mouse.move(x1, y)
        page.mouse.down()
        page.mouse.move((x1 + x2) / 2, y, steps=8)
        # Mid-drag, still holding the button: children reconciled locally.
        page.wait_for_function(
            """(before) => {
                const gd = document.querySelector('#fv-plot-1');
                const now = (gd.data || []).map(t =>
                    ({name: t.name, y: Array.from(t.y || [])}));
                return now.length > 0 && JSON.stringify(now) !== JSON.stringify(before);
            }""",
            arg=[{"name": t["name"], "y": t["y"]} for t in children_before],
            timeout=10_000,
        )
        children_mid = page.eval_on_selector("#fv-plot-1", read_children)
        assert sorted(t["uid"] for t in children_mid) == uids_before
        page.mouse.move(x2, y, steps=8)
        page.wait_for_timeout(300)
        page.mouse.up()
        page.wait_for_timeout(1_500)

        # Conditional commit: the grouped target was cube-served, so the
        # whole gesture is exactly one cube_request and nothing else.
        types = [b.get("event", {}).get("type") for b in bodies[n_before:]]
        assert types == ["cube_request"], types

        edge_lo, edge_hi = _committed_edges(page, "a")
        expected = _reference_grouped_slice_counts(df, edge_lo, edge_hi)
        children_after = page.eval_on_selector("#fv-plot-1", read_children)
        # Child uids reused from the init-rendered set — never minted.
        assert sorted(t["uid"] for t in children_after) == uids_before
        got = {t["name"]: t["y"] for t in children_after}
        assert got == expected  # bit-exact per-child counts
        total = sum(sum(ys) for ys in got.values())
        assert 0 < total < df.height

    def test_bar_and_pie_targets_live_update(self, page: Page, server_port: int):
        """Bar + pie targets live-update during a hist brush with exactly one
        cube_request and zero further POSTs through commit; the committed
        values equal a direct Polars group-by over the snapped edges."""
        df = _cat_target_df()
        url = _bar_pie_dashboard_url(server_port, "_cube_browser_barpie")
        bodies = _capture_updates(page)
        page.goto(url)
        _wait_for_init(page, "plotly")
        _enter_select_mode(page)

        bar_before = _bar_xy(page, "#fv-plot-1")
        pie_before = _pie_data(page, "#fv-plot-2")
        assert sum(bar_before["y"]) == df.height
        assert sum(pie_before["values"]) == df.height

        x1, x2, y, _width = _drag_coords(page)
        n_before = len(bodies)
        page.mouse.move(x1, y)
        page.mouse.down()
        page.mouse.move((x1 + x2) / 2, y, steps=8)
        # Mid-drag: both categorical targets re-render from local slices.
        page.wait_for_function(
            """(before) => {
                const gd = document.querySelector('#fv-plot-1');
                const ys = Array.from((gd.data && gd.data[0] && gd.data[0].y) || []);
                return ys.length > 0 && JSON.stringify(ys) !== JSON.stringify(before);
            }""",
            arg=bar_before["y"],
            timeout=10_000,
        )
        page.wait_for_function(
            """(before) => {
                const gd = document.querySelector('#fv-plot-2');
                const vs = Array.from(
                    (gd.data && gd.data[0] && gd.data[0].values) || []);
                return vs.length > 0 && JSON.stringify(vs) !== JSON.stringify(before);
            }""",
            arg=pie_before["values"],
            timeout=10_000,
        )
        page.mouse.move(x2, y, steps=8)
        page.wait_for_timeout(300)
        page.mouse.up()
        page.wait_for_timeout(1_500)

        # Exactly one cube_request, zero further POSTs through commit.
        gesture_bodies = bodies[n_before:]
        types = [b.get("event", {}).get("type") for b in gesture_bodies]
        assert types == ["cube_request"], types

        # Committed values: direct Polars group-by under the snapped
        # closed="left" edges (identical to the legacy server recompute).
        edge_lo, edge_hi = _committed_edges(page, "a")
        ref = (
            df.filter((pl.col("a") >= edge_lo) & (pl.col("a") < edge_hi))
            .group_by("g")
            .len()
            .sort("g")
        )
        exp_labels = ref["g"].to_list()
        exp_values = [float(v) for v in ref["len"].to_list()]
        assert 0 < sum(exp_values) < df.height

        bar_after = _bar_xy(page, "#fv-plot-1")
        assert bar_after["x"] == exp_labels
        assert bar_after["y"] == exp_values
        pie_after = _pie_data(page, "#fv-plot-2")
        assert pie_after["labels"] == exp_labels
        assert pie_after["values"] == exp_values

    def test_mean_bar_target_commit_parity(self, page: Page, server_port: int):
        """A ``agg="mean"`` bar target: committed values within 1e-9 rel of a
        direct Polars mean over the snapped closed="left" range."""
        df = _cat_target_df()
        url = _mean_bar_dashboard_url(server_port, "_cube_browser_meanbar")
        bodies = _capture_updates(page)
        page.goto(url)
        _wait_for_init(page, "plotly")
        _enter_select_mode(page)

        bar_before = _bar_xy(page, "#fv-plot-1")
        assert bar_before["y"], "mean bar must render values at init"
        x1, x2, y, _width = _drag_coords(page)
        n_before = len(bodies)
        page.mouse.move(x1, y)
        page.mouse.down()
        page.mouse.move((x1 + x2) / 2, y, steps=8)
        page.wait_for_function(
            """(before) => {
                const gd = document.querySelector('#fv-plot-1');
                const ys = Array.from((gd.data && gd.data[0] && gd.data[0].y) || []);
                return ys.length > 0 && JSON.stringify(ys) !== JSON.stringify(before);
            }""",
            arg=bar_before["y"],
            timeout=10_000,
        )
        page.mouse.move(x2, y, steps=8)
        page.wait_for_timeout(300)
        page.mouse.up()
        page.wait_for_timeout(1_500)

        types = [b.get("event", {}).get("type") for b in bodies[n_before:]]
        assert types == ["cube_request"], types

        edge_lo, edge_hi = _committed_edges(page, "a")
        ref = (
            df.filter((pl.col("a") >= edge_lo) & (pl.col("a") < edge_hi))
            .group_by("g")
            .agg(pl.col("v").mean())
            .sort("g")
        )
        exp_labels = ref["g"].to_list()
        exp_values = ref["v"].to_list()
        assert exp_values, "non-vacuous brush"
        bar_after = _bar_xy(page, "#fv-plot-1")
        assert bar_after["x"] == exp_labels
        assert len(bar_after["y"]) == len(exp_values)
        for got, want in zip(bar_after["y"], exp_values):
            assert got == pytest.approx(want, rel=1e-9)

    def test_median_bar_target_mixed_dashboard(self, page: Page, server_port: int):
        """A ``agg="median"`` bar target is not cube-capable (median is
        outside the cube measure algebra): the capable hist live-updates, the
        median bar holds pre-drag state during the drag, the commit POSTs
        exactly once, and all returned deltas are applied (the bar shows the
        server-filtered medians afterwards)."""
        df = _cat_target_df()
        url = _median_mixed_dashboard_url(server_port, "_cube_browser_median")
        bodies = _capture_updates(page)
        page.goto(url)
        _wait_for_init(page, "plotly")
        _enter_select_mode(page)

        y_unfiltered = _target_y(page)
        bar_before = _bar_xy(page, "#fv-plot-2")
        assert bar_before["y"], "median bar must render values at init"

        x1, x2, y, _width = _drag_coords(page)
        n_before = len(bodies)
        page.mouse.move(x1, y)
        page.mouse.down()
        page.mouse.move((x1 + x2) / 2, y, steps=8)
        page.wait_for_function(
            """(yBefore) => {
                const gd = document.querySelector('#fv-plot-1');
                const ys = Array.from((gd.data && gd.data[0] && gd.data[0].y) || []);
                return ys.length > 0 && JSON.stringify(ys) !== JSON.stringify(yBefore);
            }""",
            arg=y_unfiltered,
            timeout=10_000,
        )
        # The hist updated live; the median bar must hold its pre-drag state.
        assert _bar_xy(page, "#fv-plot-2") == bar_before
        page.mouse.move(x2, y, steps=8)
        page.wait_for_timeout(300)
        assert _bar_xy(page, "#fv-plot-2") == bar_before
        page.mouse.up()
        # Commit round-trips: wait until the server deltas update the bar.
        page.wait_for_function(
            """(before) => {
                const gd = document.querySelector('#fv-plot-2');
                const now = {x: Array.from(gd.data[0].x || []),
                             y: Array.from(gd.data[0].y || [])};
                return JSON.stringify(now) !== JSON.stringify(before);
            }""",
            arg=bar_before,
            timeout=10_000,
        )
        page.wait_for_timeout(500)

        # Exactly one cube_request + one selection POST, nothing else.
        types = [b.get("event", {}).get("type") for b in bodies[n_before:]]
        assert sorted(types) == ["cube_request", "selection"], types
        sel_body = next(
            b
            for b in bodies[n_before:]
            if b.get("event", {}).get("type") == "selection"
        )
        clause = sel_body["event"]["selections"][0]["predicates"][0]["clauses"][0]
        assert clause["column"] == "a"
        assert clause.get("closed") == "left", "mixed commit must POST snapped edges"
        edge_lo, edge_hi = clause["range"]

        # Hist target: cube-sliced values equal the cube reference.
        expected = _reference_slice_counts(df, edge_lo, edge_hi)
        assert 0 < sum(expected) < df.height
        assert _target_y(page) == expected

        # Median bar: server-computed medians over the snapped half-open range.
        ref = (
            df.filter((pl.col("a") >= edge_lo) & (pl.col("a") < edge_hi))
            .group_by("g")
            .agg(pl.col("v").median())
            .sort("g")
        )
        bar_after = _bar_xy(page, "#fv-plot-2")
        assert bar_after["x"] == ref["g"].to_list()
        assert len(bar_after["y"]) == len(ref["v"])
        for got, want in zip(bar_after["y"], ref["v"].to_list()):
            assert got == pytest.approx(want, rel=1e-12)

    def test_float_label_bar_target_live_updates(self, page: Page, server_port: int):
        """A bar whose labels column is float-typed live-updates during a
        range brush. Integral floats such as 1.0 must not demote or render as
        an empty cube slice because Python and JS choose different default
        string forms."""
        df = _numeric_label_df()
        url = _numeric_label_dashboard_url(server_port, "_cube_browser_numlab")
        bodies = _capture_updates(page)
        page.goto(url)
        _wait_for_init(page, "plotly")
        _enter_select_mode(page)

        y_unfiltered = _target_y(page)
        bar_before = _bar_xy(page, "#fv-plot-2")
        assert bar_before["y"], "float-label bar must render values at init"

        x1, x2, y, _width = _drag_coords(page)
        n_before = len(bodies)
        page.mouse.move(x1, y)
        page.mouse.down()
        page.mouse.move((x1 + x2) / 2, y, steps=8)
        page.wait_for_function(
            """(yBefore) => {
                const gd = document.querySelector('#fv-plot-1');
                const ys = Array.from((gd.data && gd.data[0] && gd.data[0].y) || []);
                return ys.length > 0 && JSON.stringify(ys) !== JSON.stringify(yBefore);
            }""",
            arg=y_unfiltered,
            timeout=10_000,
        )
        page.wait_for_function(
            """(before) => {
                const gd = document.querySelector('#fv-plot-2');
                const now = {x: Array.from(gd.data[0].x || []),
                             y: Array.from(gd.data[0].y || [])};
                return now.y.length > 0 && JSON.stringify(now) !== JSON.stringify(before);
            }""",
            arg=bar_before,
            timeout=10_000,
        )
        page.mouse.move(x2, y, steps=8)
        page.wait_for_timeout(300)
        page.mouse.up()
        page.wait_for_timeout(1_500)

        types = [b.get("event", {}).get("type") for b in bodies[n_before:]]
        assert types == ["cube_request"], types
        edge_lo, edge_hi = _committed_edges(page, "a")

        expected = _reference_slice_counts(df, edge_lo, edge_hi)
        assert 0 < sum(expected) < df.height
        assert _target_y(page) == expected

        ref = (
            df.filter((pl.col("a") >= edge_lo) & (pl.col("a") < edge_hi))
            .group_by("nlab")
            .len()
            .sort("nlab")
        )
        bar_after = _bar_xy(page, "#fv-plot-2")
        assert [float(v) for v in bar_after["x"]] == ref["nlab"].to_list()
        assert [float(v) for v in bar_after["y"]] == [
            float(v) for v in ref["len"].to_list()
        ]

    def test_grouped_bar_live_brush_preserves_group_colors(
        self, page: Page, server_port: int
    ):
        """A grouped bar coloured by its ``group_by`` values keeps each child's
        group colour DURING a live brush. The server's grouped delta carries no
        ``marker`` (the colour comes from the group_domain at render), so the
        cube path must likewise omit per-label marker colours — not overwrite
        the group colour with a per-label ``[null, …]`` array (which renders
        every bar black until the commit round-trip self-heals it)."""
        url = _grouped_color_bar_dashboard_url(server_port, "_cube_browser_gcolor")
        page.goto(url)
        _wait_for_init(page, "plotly")
        _enter_select_mode(page)

        read_children = (
            "gd => (gd.data || []).map(t => "
            "({name: t.name, y: Array.from(t.y || []), "
            "color: (t.marker || {}).color}))"
        )
        before = page.eval_on_selector("#fv-plot-1", read_children)
        assert len(before) == 2, "grouped bar must render one child per group"
        # Init colours are the scalar group colours from the color_map.
        assert {c["name"]: c["color"] for c in before} == _GROUP_COLOR_MAP

        x1, x2, y, _width = _drag_coords(page)
        page.mouse.move(x1, y)
        page.mouse.down()
        page.mouse.move((x1 + x2) / 2, y, steps=8)
        # Wait for the children to reconcile live from the local cube slice.
        page.wait_for_function(
            """(before) => {
                const gd = document.querySelector('#fv-plot-1');
                const now = (gd.data || []).map(t =>
                    ({name: t.name, y: Array.from(t.y || [])}));
                return now.length > 0 && JSON.stringify(now) !== JSON.stringify(before);
            }""",
            arg=[{"name": c["name"], "y": c["y"]} for c in before],
            timeout=10_000,
        )
        # Mid-drag: each child still wears its scalar group colour, not a
        # per-label null array.
        mid = page.eval_on_selector("#fv-plot-1", read_children)
        assert {c["name"]: c["color"] for c in mid} == _GROUP_COLOR_MAP
        page.mouse.move(x2, y, steps=8)
        page.wait_for_timeout(300)
        page.mouse.up()
        page.wait_for_timeout(1_000)
        # Commit leaves the colours correct too (server delta + group_domain).
        after = page.eval_on_selector("#fv-plot-1", read_children)
        assert {c["name"]: c["color"] for c in after} == _GROUP_COLOR_MAP

    def test_integer_label_bar_live_updates(self, page: Page, server_port: int):
        """A bar whose labels column is integer-typed live-updates during a
        brush (the demo's hour_of_day / month bars). The label dtype is
        invisible client-side; the server's cube target gate must accept
        integers, and the codec must emit the labels as integers in NUMERIC
        order so the mid-drag slice matches the committed server delta."""
        df = _int_label_df()
        url = _int_label_dashboard_url(server_port, "_cube_browser_intlab")
        bodies = _capture_updates(page)
        page.goto(url)
        _wait_for_init(page, "plotly")
        _enter_select_mode(page)

        bar_before = _bar_xy(page, "#fv-plot-1")
        assert bar_before["y"], "integer-label bar must render at init"
        # Init labels are integers in numeric order (server delta).
        assert bar_before["x"] == list(range(12))

        x1, x2, y, _width = _drag_coords(page)
        n_before = len(bodies)
        page.mouse.move(x1, y)
        page.mouse.down()
        page.mouse.move((x1 + x2) / 2, y, steps=8)
        # The bar must reconcile live from the local cube slice (a demoted
        # target would time out here, holding pre-drag state).
        page.wait_for_function(
            """(before) => {
                const gd = document.querySelector('#fv-plot-1');
                const ys = Array.from((gd.data && gd.data[0] && gd.data[0].y) || []);
                return ys.length > 0 && JSON.stringify(ys) !== JSON.stringify(before);
            }""",
            arg=bar_before["y"],
            timeout=10_000,
        )
        # Mid-drag labels stay integers in strictly increasing numeric order —
        # NOT lexicographic strings ("0","1","10","11","2",…).
        bar_mid = _bar_xy(page, "#fv-plot-1")
        assert all(isinstance(v, int) for v in bar_mid["x"]), bar_mid["x"]
        assert bar_mid["x"] == sorted(bar_mid["x"])
        page.mouse.move(x2, y, steps=8)
        page.wait_for_timeout(300)
        page.mouse.up()
        page.wait_for_timeout(1_500)

        # The gesture is fully cube-served: one cube_request, no further POSTs.
        types = [b.get("event", {}).get("type") for b in bodies[n_before:]]
        assert types == ["cube_request"], types

        # Committed values equal a direct Polars group-by over the snapped edges.
        edge_lo, edge_hi = _committed_edges(page, "a")
        ref = (
            df.filter((pl.col("a") >= edge_lo) & (pl.col("a") < edge_hi))
            .group_by("ilab")
            .len()
            .sort("ilab")
        )
        exp_labels = ref["ilab"].to_list()
        exp_values = [float(v) for v in ref["len"].to_list()]
        assert 0 < sum(exp_values) < df.height
        bar_after = _bar_xy(page, "#fv-plot-1")
        assert bar_after["x"] == exp_labels
        assert bar_after["y"] == exp_values

    def test_grouped_integer_label_bar_live_brush_demo_parity(
        self, page: Page, server_port: int
    ):
        """The demo's grouped integer-label bars (hour_of_day / month grouped by
        source, coloured by source): Bug 1 and Bug 2 co-occur. During the live
        brush each child must keep its group colour AND carry integer labels in
        numeric order; the committed per-child values match a per-group cube
        reference over the snapped edges."""
        df = _grouped_int_color_df()
        url = _grouped_int_color_dashboard_url(server_port, "_cube_browser_gintcol")
        bodies = _capture_updates(page)
        page.goto(url)
        _wait_for_init(page, "plotly")
        _enter_select_mode(page)

        read_children = (
            "gd => (gd.data || []).map(t => "
            "({name: t.name, x: Array.from(t.x || []), "
            "y: Array.from(t.y || []), color: (t.marker || {}).color}))"
        )
        before = page.eval_on_selector("#fv-plot-1", read_children)
        assert {c["name"]: c["color"] for c in before} == _GROUP_COLOR_MAP

        x1, x2, y, _width = _drag_coords(page)
        n_before = len(bodies)
        page.mouse.move(x1, y)
        page.mouse.down()
        page.mouse.move((x1 + x2) / 2, y, steps=8)
        page.wait_for_function(
            """(before) => {
                const gd = document.querySelector('#fv-plot-1');
                const now = (gd.data || []).map(t =>
                    ({name: t.name, y: Array.from(t.y || [])}));
                return now.length > 0 && JSON.stringify(now) !== JSON.stringify(before);
            }""",
            arg=[{"name": c["name"], "y": c["y"]} for c in before],
            timeout=10_000,
        )
        mid = page.eval_on_selector("#fv-plot-1", read_children)
        # Bug 1: group colours intact (scalar per child, not a null array).
        assert {c["name"]: c["color"] for c in mid} == _GROUP_COLOR_MAP
        # Bug 2: integer labels in strictly increasing numeric order per child.
        for c in mid:
            assert all(isinstance(v, int) for v in c["x"]), c
            assert c["x"] == sorted(c["x"])
        page.mouse.move(x2, y, steps=8)
        page.wait_for_timeout(300)
        page.mouse.up()
        page.wait_for_timeout(1_500)

        # Fully cube-served: one cube_request, no further POSTs.
        types = [b.get("event", {}).get("type") for b in bodies[n_before:]]
        assert types == ["cube_request"], types

        # Per-child committed values match a per-group cube reference.
        edge_lo, edge_hi = _committed_edges(page, "a")
        ref = (
            df.filter((pl.col("a") >= edge_lo) & (pl.col("a") < edge_hi))
            .group_by("k", "ilab")
            .len()
            .sort("k", "ilab")
        )
        expected: dict[str, dict[int, float]] = {}
        for gv, lab, n in ref.iter_rows():
            expected.setdefault(gv, {})[lab] = float(n)
        after = page.eval_on_selector("#fv-plot-1", read_children)
        assert {c["name"]: c["color"] for c in after} == _GROUP_COLOR_MAP
        got = {c["name"]: dict(zip(c["x"], c["y"])) for c in after}
        assert got == expected

    def test_edit_existing_box_live_brush(self, page: Page, server_port: int):
        """Moving an already-committed selection box is a live-brush gesture
        too. Plotly emits no event while an activated selection outline is
        dragged (only the mouseup round-trips), so the runtime watches the
        outline drag directly and replays it through the same gesture path:
        the target live-updates mid-move, the whole edit is a pure store hit
        (zero POSTs of any kind), and the re-committed predicate is snapped
        ``closed="left"`` at the new edges, matching the cube reference."""
        df = _cube_df()
        url = _two_hist_dashboard_url(server_port, "_cube_browser_editbox", "auto")
        bodies = _capture_updates(page)
        posts = _capture_posts(page)
        page.goto(url)
        _wait_for_init(page, "plotly")
        _enter_select_mode(page)

        # Gesture 1: a fresh draw commits a snapped box and warms the store.
        y_unfiltered = _target_y(page)
        x1, x2, y, width = _drag_coords(page)
        _drag_with_live_wait(page, x1, x2, y, y_unfiltered)
        page.wait_for_timeout(1_000)
        first_edges = _committed_edges(page, "a")
        y_first = _target_y(page)
        assert y_first != y_unfiltered

        # Activate the rendered box (click on it) — Plotly only allows
        # moving/resizing the activated selection.
        box = page.eval_on_selector(
            "#fv-plot-0 .selectionlayer path",
            "n => { const r = n.getBoundingClientRect();"
            " return {x: (r.left + r.right) / 2, y: (r.top + r.bottom) / 2}; }",
        )
        page.mouse.click(box["x"], box["y"])
        page.wait_for_function(
            "() => divs[0]._fullLayout._activeSelectionIndex >= 0", timeout=5_000
        )

        # Gesture 2: drag the activated box sideways. No plotly_selecting
        # fires for this; the live brush must still engage from the store.
        n_bodies = len(bodies)
        n_posts = len(posts)
        page.mouse.move(box["x"], box["y"])
        page.mouse.down()
        page.mouse.move(box["x"] + width * 0.15, box["y"], steps=10)
        # Mid-move, still holding the button: the target re-renders locally.
        page.wait_for_function(
            """(yBefore) => {
                const gd = document.querySelector('#fv-plot-1');
                const ys = Array.from((gd.data && gd.data[0] && gd.data[0].y) || []);
                return ys.length > 0 && JSON.stringify(ys) !== JSON.stringify(yBefore);
            }""",
            arg=y_first,
            timeout=10_000,
        )
        page.wait_for_timeout(300)
        page.mouse.up()
        # Past the safety-abort window: a missed commit would restore the
        # pre-move rendering and fail the asserts below.
        page.wait_for_timeout(2_000)

        assert posts[n_posts:] == [], (
            f"editing a committed box must be served from the cube store with "
            f"zero POSTs of any kind; got {posts[n_posts:]}"
        )
        assert bodies[n_bodies:] == []

        # The re-committed predicate: snapped closed="left" at the new edges.
        edge_lo, edge_hi = _committed_edges(page, "a")
        assert (edge_lo, edge_hi) != first_edges
        expected = _reference_slice_counts(df, edge_lo, edge_hi)
        assert 0 < sum(expected) < df.height
        assert _target_y(page) == expected


# ---------------------------------------------------------------------------
# Categorical sources (Step 6): bar box-drag gesture + pie/treemap clicks
# ---------------------------------------------------------------------------


def _bar_source_dashboard_url(port: int, source_name: str, live_brush: str) -> str:
    """Bar source (labels=g, 4 string labels) + hist(b) target."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import LayoutSpec, encode_spec

    df = _cat_target_df()
    register_source(source_name, df, cache=True)

    dash = Dashboard(df)
    dash.add_figure(title="BarSource").add_bar(labels="g")
    dash.add_figure(title="Target").add_histogram(x="b", bins=_TGT_BINS)
    spec = dash.to_spec(source_name=source_name, layout=LayoutSpec(draggable=False))
    spec.client_state.live_brush = live_brush
    return f"http://127.0.0.1:{port}/view?spec={encode_spec(spec)}&renderer=plotly"


def _bar_source_line_corr_dashboard_url(port: int, source_name: str) -> str:
    """Bar source (labels=g) + line(x=a, y=b) target + corr([a, b, v]) target —
    a categorical source live-driving line_env and corr cubes (#46).

    Sorted on ``a`` for the ungrouped minmax line's ascending-x contract."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import LayoutSpec, encode_spec

    df = _cat_target_df().sort("a")
    register_source(source_name, df, cache=True)

    dash = Dashboard(df)
    dash.add_figure(title="BarSource").add_bar(labels="g")
    dash.add_figure(title="Line").add_line(
        x="a", y="b", n_points=_LINE_N_POINTS, downsample="minmax"
    )
    dash.add_figure(title="Corr").add_corr_heatmap(columns=["a", "b", "v"])
    spec = dash.to_spec(source_name=source_name, layout=LayoutSpec(draggable=False))
    spec.client_state.live_brush = "auto"
    return f"http://127.0.0.1:{port}/view?spec={encode_spec(spec)}&renderer=plotly"


def _int_bar_source_df() -> pl.DataFrame:
    """`_cat_target_df` plus an INTEGER label column ``gi`` mirroring the string
    partition ``g`` (``gi == int(g[1:])``, values 0..3). A box-drag over an
    integer-label bar source is the case fixed by allowing integer categorical
    SOURCES (the user-reported hour-of-day bar driving a live brush)."""
    df = _cat_target_df()
    return df.with_columns(pl.col("g").str.slice(1).cast(pl.Int64).alias("gi"))


def _int_bar_source_dashboard_url(port: int, source_name: str) -> str:
    """Integer-label bar source (labels=gi, 4 integer labels 0..3) + hist(b)
    target — the integer analogue of ``_bar_source_dashboard_url``."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import LayoutSpec, encode_spec

    df = _int_bar_source_df()
    register_source(source_name, df, cache=True)

    dash = Dashboard(df)
    dash.add_figure(title="IntBarSource").add_bar(labels="gi")
    dash.add_figure(title="Target").add_histogram(x="b", bins=_TGT_BINS)
    spec = dash.to_spec(source_name=source_name, layout=LayoutSpec(draggable=False))
    spec.client_state.live_brush = "auto"
    return f"http://127.0.0.1:{port}/view?spec={encode_spec(spec)}&renderer=plotly"


def _int_bar_source_bar_pie_dashboard_url(port: int, source_name: str) -> str:
    """Integer-label bar source + sum bar/pie targets.

    This matches the user-reported hour-of-day source brushing another
    aggregate bar panel such as ``sum(y_pos)``.
    """
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import LayoutSpec, encode_spec

    df = _int_bar_source_df()
    register_source(source_name, df, cache=True)

    dash = Dashboard(df)
    dash.add_figure(title="IntBarSource").add_bar(labels="gi")
    dash.add_figure(title="SumBar").add_bar(labels="g", values="v", agg="sum")
    dash.add_figure(title="SumPie").add_pie(labels="g", values="v", agg="sum")
    spec = dash.to_spec(source_name=source_name, layout=LayoutSpec(draggable=False))
    spec.client_state.live_brush = "auto"
    return f"http://127.0.0.1:{port}/view?spec={encode_spec(spec)}&renderer=plotly"


def _grouped_int_bar_source_df() -> pl.DataFrame:
    """`_int_bar_source_df` plus a group column ``k`` whose two values BOTH span
    every integer label. ``gi`` cycles with period 4 over the row index, so a
    block-of-4 split (``(i // 4) % 2``) puts all of ``gi`` in both ``P`` and
    ``Q``. That collision is essential: with overlapping categories Plotly
    draws the grouped bars at ``category ± offset`` on the LINEAR numeric axis,
    so the brush point's geometric ``x`` (e.g. ``0.8``) differs from the label
    (``1``) — the exact shape of the user-reported hour-of-day bug. A split
    that gave each label to only one group (e.g. ``i % 2``) would draw every
    bar centred on its integer and silently hide the regression."""
    df = _int_bar_source_df()
    return df.with_columns(
        pl.Series("k", ["P" if (i // 4) % 2 == 0 else "Q" for i in range(df.height)])
    )


def _grouped_int_bar_source_dashboard_url(port: int, source_name: str) -> str:
    """Grouped integer-label bar source + grouped sum bar target."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import LayoutSpec, encode_spec

    df = _grouped_int_bar_source_df()
    register_source(source_name, df, cache=True)

    dash = Dashboard(df)
    dash.add_figure(title="GroupedIntBarSource").add_bar(
        labels="gi", values="v", agg="sum", group_by="k", color_map=_GROUP_COLOR_MAP
    )
    dash.add_figure(title="GroupedSumBar").add_bar(
        labels="g", values="v", agg="sum", group_by="k", color_map=_GROUP_COLOR_MAP
    )
    spec = dash.to_spec(source_name=source_name, layout=LayoutSpec(draggable=False))
    spec.client_state.live_brush = "auto"
    return f"http://127.0.0.1:{port}/view?spec={encode_spec(spec)}&renderer=plotly"


def _pie_source_dashboard_url(port: int, source_name: str) -> str:
    """Pie source (labels=g) + hist(b) target."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import LayoutSpec, encode_spec

    df = _cat_target_df()
    register_source(source_name, df, cache=True)

    dash = Dashboard(df)
    dash.add_figure(title="PieSource").add_pie(labels="g")
    dash.add_figure(title="Target").add_histogram(x="b", bins=_TGT_BINS)
    spec = dash.to_spec(source_name=source_name, layout=LayoutSpec(draggable=False))
    spec.client_state.live_brush = "auto"
    return f"http://127.0.0.1:{port}/view?spec={encode_spec(spec)}&renderer=plotly"


def _float_pie_source_dashboard_url(port: int, source_name: str) -> str:
    """Float-label pie source + hist(b) target."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import LayoutSpec, encode_spec

    df = _float_bar_source_df()
    register_source(source_name, df, cache=True)

    dash = Dashboard(df)
    dash.add_figure(title="FloatPieSource").add_pie(labels="gf")
    dash.add_figure(title="Target").add_histogram(x="b", bins=_TGT_BINS)
    spec = dash.to_spec(source_name=source_name, layout=LayoutSpec(draggable=False))
    spec.client_state.live_brush = "auto"
    return f"http://127.0.0.1:{port}/view?spec={encode_spec(spec)}&renderer=plotly"


def _treemap_df() -> pl.DataFrame:
    """`_cat_target_df` plus a second low-cardinality string column ``h`` so a
    treemap ``path=["g", "h"]`` has depth-2 nodes; (g, h) partitions the rows
    by ``i % 12``, so any path predicate visibly reshapes the hist on ``b``."""
    df = _cat_target_df()
    return df.with_columns(
        pl.Series("h", [f"h{(i * 5) % 3}" for i in range(df.height)])
    )


def _treemap_source_dashboard_url(port: int, source_name: str) -> str:
    """Treemap source (path=[g, h]) + hist(b) target."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import LayoutSpec, encode_spec

    df = _treemap_df()
    register_source(source_name, df, cache=True)

    dash = Dashboard(df)
    dash.add_figure(title="TreemapSource").add_treemap(path=["g", "h"])
    dash.add_figure(title="Target").add_histogram(x="b", bins=_TGT_BINS)
    spec = dash.to_spec(source_name=source_name, layout=LayoutSpec(draggable=False))
    spec.client_state.live_brush = "auto"
    return f"http://127.0.0.1:{port}/view?spec={encode_spec(spec)}&renderer=plotly"


def _ambiguous_source_dashboard_url(port: int, source_name: str) -> str:
    """ONE source figure holding bar(labels=g) AND treemap(path=[g, h]) —
    two cube-source traces sharing the primary column ``g`` — plus a hist(b)
    target (plan step 0b: source identity must follow the interacted trace)."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import LayoutSpec, encode_spec

    df = _treemap_df()
    register_source(source_name, df, cache=True)

    dash = Dashboard(df)
    fig = dash.add_figure(title="AmbiguousSource")
    fig.add_bar(labels="g")
    fig.add_treemap(path=["g", "h"])
    dash.add_figure(title="Target").add_histogram(x="b", bins=_TGT_BINS)
    spec = dash.to_spec(source_name=source_name, layout=LayoutSpec(draggable=False))
    spec.client_state.live_brush = "auto"
    return f"http://127.0.0.1:{port}/view?spec={encode_spec(spec)}&renderer=plotly"


def _categorical_legacy_dashboard_url(port: int, source_name: str) -> str:
    """Bar source + pie source + hist target, live_brush='off' (legacy)."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import LayoutSpec, encode_spec

    df = _cat_target_df()
    register_source(source_name, df, cache=True)

    dash = Dashboard(df)
    dash.add_figure(title="BarSource").add_bar(labels="g")
    dash.add_figure(title="PieSource").add_pie(labels="g")
    dash.add_figure(title="Target").add_histogram(x="b", bins=_TGT_BINS)
    spec = dash.to_spec(source_name=source_name, layout=LayoutSpec(draggable=False))
    spec.client_state.live_brush = "off"
    return f"http://127.0.0.1:{port}/view?spec={encode_spec(spec)}&renderer=plotly"


def _reference_categorical_hist_counts(
    df: pl.DataFrame, filter_expr: pl.Expr
) -> list[int]:
    """Categorical sibling of ``_reference_slice_counts``: the hist target on
    ``b`` recomputed under *filter_expr* through the real ``fixed_hist``
    kernel over the full target domain (identical to the legacy server
    recompute a categorical cube slice must equal bit-for-bit)."""
    b_lo, b_hi = df["b"].min(), df["b"].max()
    raw = (
        df.lazy()
        .filter(filter_expr)
        .select(
            pl.col("b")
            .flexviz.fixed_hist(
                pl.lit(float(b_lo)),
                pl.lit(float(b_hi + _HIST_BIN_EPSILON)),
                n_bins=_TGT_BINS,
            )
            .implode()
            .alias("h")
        )
        .collect()["h"]
        .item()
    )
    return raw.explode().struct.unnest()["count"].to_list()


def _bar_drag_coords(page: Page) -> tuple[float, float, float, float]:
    """A full-height box over the first two of the four bars (category axis
    spans [-0.5, 3.5], so label centers sit at 12.5/37.5/62.5/87.5% of the
    plot width; 5%→45% covers exactly g0 and g1). Near-full height because
    Plotly's bar hit-point sits at the bar *tip* (≈95% of the axis with the
    default autorange padding), not the bar body."""
    box = page.locator("#fv-plot-0 .nsewdrag").bounding_box()
    assert box is not None
    x1 = box["x"] + box["width"] * 0.05
    x2 = box["x"] + box["width"] * 0.45
    y1 = box["y"] + box["height"] * 0.01
    y2 = box["y"] + box["height"] * 0.99
    return x1, x2, y1, y2


def _hist_y(page: Page, selector: str) -> list[float]:
    return page.eval_on_selector(
        selector, "gd => Array.from((gd.data && gd.data[0] && gd.data[0].y) || [])"
    )


def _wait_for_hist_equals(page: Page, selector: str, expected: list) -> None:
    page.wait_for_function(
        """([sel, expected]) => {
            const gd = document.querySelector(sel);
            const ys = Array.from((gd.data && gd.data[0] && gd.data[0].y) || []);
            return ys.length > 0 && JSON.stringify(ys) === JSON.stringify(expected);
        }""",
        arg=[selector, expected],
        timeout=10_000,
    )


def _click_pie_slice(page: Page, fig_sel: str, label: object) -> None:
    """Scroll a pie slice into view by its label, then click its centre."""
    target = page.eval_on_selector(
        fig_sel,
        """(gd, label) => {
            const nodes = Array.from(gd.querySelectorAll('g.slice'));
            const node = nodes.find(n => {
                const d = n.__data__ || {};
                const raw = d.label || (d.data && d.data.label);
                return raw === label || String(raw) === String(label);
            });
            if (!node) return null;
            node.scrollIntoView({block: 'center'});
            const r = node.getBoundingClientRect();
            return { x: (r.left + r.right) / 2, y: (r.top + r.bottom) / 2 };
        }""",
        label,
    )
    assert target is not None, f"pie slice {label!r} not found"
    page.mouse.click(target["x"], target["y"])


def _click_treemap_slice(page: Page, fig_sel: str, node_id: str) -> None:
    """Scroll a treemap slice into view by its id, then click its centre."""
    target = page.eval_on_selector(
        fig_sel,
        """(gd, nid) => {
            const nodes = Array.from(gd.querySelectorAll('g.slice'));
            const node = nodes.find(n => {
                const d = n.__data__ || {};
                return d.id === nid
                    || (d.data && d.data.id === nid)
                    || (d.data && d.data.data && d.data.data.id === nid);
            });
            if (!node) return null;
            node.scrollIntoView({block: 'center'});
            const r = node.getBoundingClientRect();
            return { x: (r.left + r.right) / 2, y: (r.top + r.bottom) / 2 };
        }""",
        node_id,
    )
    assert target is not None, f"treemap slice {node_id!r} not found"
    page.mouse.click(target["x"], target["y"])


def _assert_legacy_isin_predicates(
    predicates: list[dict], column: str, expected_labels: list[str]
) -> None:
    """One predicate per label, each a single ``{column, values:[label]}``
    clause — the unchanged legacy is_in shape (no snapping, no ``closed``,
    no ``range``)."""
    labels = []
    for pred in predicates:
        assert len(pred["clauses"]) == 1
        clause = pred["clauses"][0]
        assert clause["column"] == column
        assert "closed" not in clause, f"categorical commit must not snap: {clause}"
        assert clause.get("range") is None, f"unexpected range clause: {clause}"
        assert isinstance(clause["values"], list) and len(clause["values"]) == 1
        labels.append(clause["values"][0])
    assert sorted(labels) == expected_labels


class TestCategoricalSourceCube:
    def test_bar_source_drag_live_updates_hist(self, page: Page, server_port: int):
        """Box-drag over two bars of a categorical bar source: the hist
        target live-updates mid-drag, the whole gesture is one cube_request
        and nothing else (skipPost), the committed predicates are the
        unchanged legacy is_in shape, and the committed hist equals a direct
        ``is_in`` recompute through ``fixed_hist``."""
        df = _cat_target_df()
        url = _bar_source_dashboard_url(server_port, "_cube_browser_barsrc", "auto")
        bodies = _capture_updates(page)
        page.goto(url)
        _wait_for_init(page, "plotly")
        _enter_select_mode(page)

        y_before = _hist_y(page, "#fv-plot-1")
        assert sum(y_before) == df.height

        x1, x2, y1, y2 = _bar_drag_coords(page)
        n_before = len(bodies)
        page.mouse.move(x1, y1)
        page.mouse.down()
        page.mouse.move((x1 + x2) / 2, (y1 + y2) / 2, steps=8)
        page.mouse.move(x2, y2, steps=8)
        # Mid-drag, still holding the button: the cube must land and the
        # hist must re-render from a local slice (no further POSTs).
        page.wait_for_function(
            """(yBefore) => {
                const gd = document.querySelector('#fv-plot-1');
                const ys = Array.from((gd.data && gd.data[0] && gd.data[0].y) || []);
                return ys.length > 0 && JSON.stringify(ys) !== JSON.stringify(yBefore);
            }""",
            arg=y_before,
            timeout=10_000,
        )
        page.mouse.up()
        page.wait_for_timeout(1_500)

        # Exactly one cube_request keyed on the primary label column, and
        # zero further POSTs through commit (the conditional commit skipped).
        gesture_bodies = bodies[n_before:]
        types = [b.get("event", {}).get("type") for b in gesture_bodies]
        assert types == ["cube_request"], types
        active = gesture_bodies[0].get("active_source") or {}
        assert active.get("column") == "g"

        # Committed predicates: byte-identical legacy is_in shape.
        sels = page.evaluate("DASHBOARD_SPEC.state.selections")
        assert len(sels) == 1
        _assert_legacy_isin_predicates(sels[0]["predicates"], "g", ["g0", "g1"])

        # Committed hist equals the direct Polars is_in recompute.
        expected = _reference_categorical_hist_counts(
            df, pl.col("g").is_in(["g0", "g1"])
        )
        assert 0 < sum(expected) < df.height
        assert _hist_y(page, "#fv-plot-1") == expected

    def test_edit_existing_bar_box_live_brush(self, page: Page, server_port: int):
        """Regression: moving an already-committed selection box on a
        CATEGORICAL (bar) source must live-update the target mid-move, exactly
        like a range source does (``test_edit_existing_box_live_brush``).
        Plotly emits no event while an activated outline is dragged, so the
        runtime watches the outline drag and replays it through the gesture
        path. For a categorical source that replay must resolve the COVERED
        BARS from the outline geometry — a bare range carries no points — or
        the categorical gesture never engages and the target only updates on
        mouseup. The whole edit is a pure store hit (zero POSTs)."""
        df = _cat_target_df()
        url = _bar_source_dashboard_url(server_port, "_cube_edit_barsrc", "auto")
        bodies = _capture_updates(page)
        posts = _capture_posts(page)
        page.goto(url)
        _wait_for_init(page, "plotly")
        _enter_select_mode(page)

        # Gesture 1: a fresh draw over g0, g1 commits and warms the store.
        y_unfiltered = _hist_y(page, "#fv-plot-1")
        assert sum(y_unfiltered) == df.height
        x1, x2, y1, y2 = _bar_drag_coords(page)
        page.mouse.move(x1, y1)
        page.mouse.down()
        page.mouse.move((x1 + x2) / 2, (y1 + y2) / 2, steps=8)
        page.mouse.move(x2, y2, steps=8)
        page.wait_for_function(
            """(yBefore) => {
                const gd = document.querySelector('#fv-plot-1');
                const ys = Array.from((gd.data && gd.data[0] && gd.data[0].y) || []);
                return ys.length > 0 && JSON.stringify(ys) !== JSON.stringify(yBefore);
            }""",
            arg=y_unfiltered,
            timeout=10_000,
        )
        page.mouse.up()
        page.wait_for_timeout(1_000)
        y_first = _hist_y(page, "#fv-plot-1")
        expected_first = _reference_categorical_hist_counts(
            df, pl.col("g").is_in(["g0", "g1"])
        )
        assert y_first == expected_first

        # Activate the rendered box (Plotly only moves the active selection).
        box = page.eval_on_selector(
            "#fv-plot-0 .selectionlayer path",
            "n => { const r = n.getBoundingClientRect();"
            " return {x: (r.left + r.right) / 2, y: (r.top + r.bottom) / 2}; }",
        )
        page.mouse.click(box["x"], box["y"])
        page.wait_for_function(
            "() => divs[0]._fullLayout._activeSelectionIndex >= 0", timeout=5_000
        )

        # Gesture 2: drag the activated box to the right so it now covers
        # g2, g3. No plotly_selecting fires — the live brush must engage from
        # the store and re-render the target mid-move.
        plot = page.locator("#fv-plot-0 .nsewdrag").bounding_box()
        shift = plot["width"] * 0.50
        expected_move = _reference_categorical_hist_counts(
            df, pl.col("g").is_in(["g2", "g3"])
        )
        assert expected_move != expected_first
        n_bodies = len(bodies)
        n_posts = len(posts)
        page.mouse.move(box["x"], box["y"])
        page.mouse.down()
        page.mouse.move(box["x"] + shift, box["y"], steps=12)
        # Mid-move, still holding the button: the target re-renders locally.
        page.wait_for_function(
            """(yFirst) => {
                const gd = document.querySelector('#fv-plot-1');
                const ys = Array.from((gd.data && gd.data[0] && gd.data[0].y) || []);
                return ys.length > 0 && JSON.stringify(ys) !== JSON.stringify(yFirst);
            }""",
            arg=y_first,
            timeout=10_000,
        )
        page.wait_for_timeout(300)
        page.mouse.up()
        # Past the safety-abort window: a missed commit would restore the
        # pre-move rendering and fail the asserts below.
        page.wait_for_timeout(2_000)

        assert posts[n_posts:] == [], (
            f"editing a committed bar box must be a pure store hit; "
            f"got {posts[n_posts:]}"
        )
        assert bodies[n_bodies:] == []

        # The re-committed predicate: legacy is_in over the newly covered bars.
        sels = page.evaluate("DASHBOARD_SPEC.state.selections")
        assert len(sels) == 1
        _assert_legacy_isin_predicates(sels[0]["predicates"], "g", ["g2", "g3"])
        assert _hist_y(page, "#fv-plot-1") == expected_move

    def test_integer_bar_source_drag_live_updates_hist(
        self, page: Page, server_port: int
    ):
        """Regression: a box-drag over an INTEGER-label bar source live-updates
        the hist target. Before the fix the integer source failed the
        string-only free-axis gate (while the same column was a valid target),
        so brushing it built no cube and the other panels never moved — the
        user-reported hour-of-day bar. The committed hist must equal a direct
        integer ``is_in`` recompute, proving the typed free key and the
        committed predicate both round-trip through the integer column."""
        df = _int_bar_source_df()
        url = _int_bar_source_dashboard_url(server_port, "_cube_browser_intbarsrc")
        bodies = _capture_updates(page)
        page.goto(url)
        _wait_for_init(page, "plotly")
        _enter_select_mode(page)

        # The integer source bars render in NUMERIC order 0..3 at init.
        assert _bar_xy(page, "#fv-plot-0")["x"] == [0, 1, 2, 3]
        y_before = _hist_y(page, "#fv-plot-1")
        assert sum(y_before) == df.height

        x1, x2, y1, y2 = _bar_drag_coords(page)
        n_before = len(bodies)
        page.mouse.move(x1, y1)
        page.mouse.down()
        page.mouse.move((x1 + x2) / 2, (y1 + y2) / 2, steps=8)
        page.mouse.move(x2, y2, steps=8)
        # Mid-drag, still holding: the cube must land and the hist re-render
        # from a local slice. A demoted (pre-fix) source builds no cube, so the
        # target would hold pre-drag state and this wait would time out.
        page.wait_for_function(
            """(yBefore) => {
                const gd = document.querySelector('#fv-plot-1');
                const ys = Array.from((gd.data && gd.data[0] && gd.data[0].y) || []);
                return ys.length > 0 && JSON.stringify(ys) !== JSON.stringify(yBefore);
            }""",
            arg=y_before,
            timeout=10_000,
        )
        page.mouse.up()
        page.wait_for_timeout(1_500)

        # Exactly one cube_request keyed on the integer label column, nothing
        # else through commit (the conditional commit is fully cube-served).
        gesture_bodies = bodies[n_before:]
        assert [b.get("event", {}).get("type") for b in gesture_bodies] == [
            "cube_request"
        ]
        assert (gesture_bodies[0].get("active_source") or {}).get("column") == "gi"

        # Committed predicates: legacy is_in shape over the covered integers
        # (the value may be a number or a category string — normalize to compare).
        sels = page.evaluate("DASHBOARD_SPEC.state.selections")
        assert len(sels) == 1
        committed = sorted(
            str(p["clauses"][0]["values"][0]) for p in sels[0]["predicates"]
        )
        assert committed == ["0", "1"]

        # Committed hist equals a direct integer is_in recompute.
        expected = _reference_categorical_hist_counts(df, pl.col("gi").is_in([0, 1]))
        assert 0 < sum(expected) < df.height
        assert _hist_y(page, "#fv-plot-1") == expected

    def test_integer_bar_source_drag_live_updates_bar_and_pie_targets(
        self, page: Page, server_port: int
    ):
        """Regression: integer-label bar source must live-update aggregate
        bar/pie targets, not just histogram targets. The user-reported
        ``hour`` → ``sum(y_pos)`` flow exercises this shape."""
        df = _int_bar_source_df()
        url = _int_bar_source_bar_pie_dashboard_url(
            server_port, "_cube_browser_intbarsrc_barpie"
        )
        bodies = _capture_updates(page)
        page.goto(url)
        _wait_for_init(page, "plotly")
        _enter_select_mode(page)

        assert _bar_xy(page, "#fv-plot-0")["x"] == [0, 1, 2, 3]
        bar_before = _bar_xy(page, "#fv-plot-1")
        pie_before = _pie_data(page, "#fv-plot-2")
        assert bar_before["y"]
        assert pie_before["values"]

        x1, x2, y1, y2 = _bar_drag_coords(page)
        n_before = len(bodies)
        page.mouse.move(x1, y1)
        page.mouse.down()
        page.mouse.move((x1 + x2) / 2, (y1 + y2) / 2, steps=8)
        page.mouse.move(x2, y2, steps=8)
        page.wait_for_function(
            """(before) => {
                const gd = document.querySelector('#fv-plot-1');
                const ys = Array.from((gd.data && gd.data[0] && gd.data[0].y) || []);
                return ys.length > 0 && JSON.stringify(ys) !== JSON.stringify(before);
            }""",
            arg=bar_before["y"],
            timeout=10_000,
        )
        page.wait_for_function(
            """(before) => {
                const gd = document.querySelector('#fv-plot-2');
                const vs = Array.from((gd.data && gd.data[0] && gd.data[0].values) || []);
                return vs.length > 0 && JSON.stringify(vs) !== JSON.stringify(before);
            }""",
            arg=pie_before["values"],
            timeout=10_000,
        )
        page.mouse.up()
        page.wait_for_timeout(1_500)

        gesture_bodies = bodies[n_before:]
        assert [b.get("event", {}).get("type") for b in gesture_bodies] == [
            "cube_request"
        ]
        assert (gesture_bodies[0].get("active_source") or {}).get("column") == "gi"

        ref = (
            df.filter(pl.col("gi").is_in([0, 1]))
            .group_by("g")
            .agg(pl.col("v").sum())
            .sort("g")
        )
        exp_labels = ref["g"].to_list()
        exp_values = ref["v"].to_list()
        assert 0 < sum(exp_values) < df["v"].sum()

        bar_after = _bar_xy(page, "#fv-plot-1")
        assert bar_after["x"] == exp_labels
        # Cube sums recombine within ~1e-9, not bit-exactly (Architecture.md
        # §"cube sums"; float addition is non-associative across partials).
        assert bar_after["y"] == pytest.approx(exp_values, rel=1e-9, abs=1e-9)

        pie_after = _pie_data(page, "#fv-plot-2")
        assert pie_after["labels"] == exp_labels
        assert pie_after["values"] == pytest.approx(exp_values, rel=1e-9, abs=1e-9)

    def test_grouped_integer_bar_source_drag_live_updates_grouped_bar_target(
        self, page: Page, server_port: int
    ):
        """Demo-like regression: a grouped integer-label source bar (hour by
        source) must still resolve the logical source trace and live-update a
        grouped aggregate bar target.

        The group column overlaps every integer label (``_grouped_int_bar_source_df``),
        so Plotly draws the bars at ``category ± offset`` on the linear numeric
        axis. The selection must read the bar's typed DATA label (``0``, ``1``),
        not its geometric ``x`` (``-0.2``, ``0.8`` …) — otherwise the committed
        ``is_in`` carries offsets that match no row and every target empties
        (the user-reported hour-of-day → sum bar bug)."""
        df = _grouped_int_bar_source_df()
        url = _grouped_int_bar_source_dashboard_url(
            server_port, "_cube_browser_grouped_intbarsrc"
        )
        bodies = _capture_updates(page)
        page.goto(url)
        _wait_for_init(page, "plotly")
        _enter_select_mode(page)

        read_children = (
            "gd => (gd.data || []).map(t => "
            "({name: t.name, x: Array.from(t.x || []), "
            "y: Array.from(t.y || []), color: (t.marker || {}).color}))"
        )
        before = page.eval_on_selector("#fv-plot-1", read_children)
        assert {c["name"]: c["color"] for c in before} == _GROUP_COLOR_MAP

        x1, x2, y1, y2 = _bar_drag_coords(page)
        n_before = len(bodies)
        page.mouse.move(x1, y1)
        page.mouse.down()
        page.mouse.move((x1 + x2) / 2, (y1 + y2) / 2, steps=8)
        page.mouse.move(x2, y2, steps=8)
        page.wait_for_function(
            """(before) => {
                const gd = document.querySelector('#fv-plot-1');
                const now = (gd.data || []).map(t =>
                    ({name: t.name, y: Array.from(t.y || [])}));
                return now.length > 0 && JSON.stringify(now) !== JSON.stringify(before);
            }""",
            arg=[{"name": c["name"], "y": c["y"]} for c in before],
            timeout=10_000,
        )
        page.mouse.up()
        page.wait_for_timeout(1_500)

        gesture_bodies = bodies[n_before:]
        assert [b.get("event", {}).get("type") for b in gesture_bodies] == [
            "cube_request"
        ]
        assert (gesture_bodies[0].get("active_source") or {}).get("column") == "gi"

        # The committed predicates carry the TRUE integer labels, never the
        # offset geometric positions of the grouped bars.
        sels = page.evaluate("DASHBOARD_SPEC.state.selections")
        committed = sorted(p["clauses"][0]["values"][0] for p in sels[0]["predicates"])
        assert committed == [0, 1]

        ref = (
            df.filter(pl.col("gi").is_in([0, 1]))
            .group_by("k", "g")
            .agg(pl.col("v").sum())
            .sort("k", "g")
        )
        expected: dict[str, dict[str, float]] = {}
        for group, label, value in ref.iter_rows():
            expected.setdefault(group, {})[label] = value

        after = page.eval_on_selector("#fv-plot-1", read_children)
        assert {c["name"]: c["color"] for c in after} == _GROUP_COLOR_MAP
        got = {c["name"]: dict(zip(c["x"], c["y"])) for c in after}
        # The cube combines f64 sum partials in CSR order while the reference
        # sums in one Polars pass — equal up to float summation order.
        assert got.keys() == expected.keys()
        for name, label_vals in expected.items():
            assert got[name].keys() == label_vals.keys()
            for label, want in label_vals.items():
                assert got[name][label] == pytest.approx(want, rel=1e-9)

    def test_float_bar_source_drag_live_updates_bar_and_pie_targets(
        self, page: Page, server_port: int
    ):
        """Float-label bar source must match cube free categories by value,
        not by Python/JS string formatting. A brush covering 0.0 and 1.0
        should update aggregate bar/pie targets locally."""
        df = _float_bar_source_df()
        url = _float_bar_source_dashboard_url(server_port, "_cube_browser_floatbarsrc")
        bodies = _capture_updates(page)
        page.goto(url)
        _wait_for_init(page, "plotly")
        _enter_select_mode(page)

        assert [float(v) for v in _bar_xy(page, "#fv-plot-0")["x"]] == [
            0.0,
            1.0,
            2.0,
            3.0,
        ]
        bar_before = _bar_xy(page, "#fv-plot-1")
        pie_before = _pie_data(page, "#fv-plot-2")
        assert bar_before["y"]
        assert pie_before["values"]

        x1, x2, y1, y2 = _bar_drag_coords(page)
        n_before = len(bodies)
        page.mouse.move(x1, y1)
        page.mouse.down()
        page.mouse.move((x1 + x2) / 2, (y1 + y2) / 2, steps=8)
        page.mouse.move(x2, y2, steps=8)
        page.wait_for_function(
            """(before) => {
                const gd = document.querySelector('#fv-plot-1');
                const ys = Array.from((gd.data && gd.data[0] && gd.data[0].y) || []);
                return ys.length > 0 && JSON.stringify(ys) !== JSON.stringify(before);
            }""",
            arg=bar_before["y"],
            timeout=10_000,
        )
        page.wait_for_function(
            """(before) => {
                const gd = document.querySelector('#fv-plot-2');
                const vs = Array.from((gd.data && gd.data[0] && gd.data[0].values) || []);
                return vs.length > 0 && JSON.stringify(vs) !== JSON.stringify(before);
            }""",
            arg=pie_before["values"],
            timeout=10_000,
        )
        page.mouse.up()
        page.wait_for_timeout(1_500)

        gesture_bodies = bodies[n_before:]
        assert [b.get("event", {}).get("type") for b in gesture_bodies] == [
            "cube_request"
        ]
        assert (gesture_bodies[0].get("active_source") or {}).get("column") == "gf"

        ref = (
            df.filter(pl.col("gf").is_in([0.0, 1.0]))
            .group_by("g")
            .agg(pl.col("v").sum())
            .sort("g")
        )
        exp_labels = ref["g"].to_list()
        exp_values = ref["v"].to_list()
        assert 0 < sum(exp_values) < df["v"].sum()

        bar_after = _bar_xy(page, "#fv-plot-1")
        assert bar_after["x"] == exp_labels
        # Cube sums recombine within ~1e-9, not bit-exactly (Architecture.md
        # §"cube sums"; float addition is non-associative across partials).
        assert bar_after["y"] == pytest.approx(exp_values, rel=1e-9, abs=1e-9)

        pie_after = _pie_data(page, "#fv-plot-2")
        assert pie_after["labels"] == exp_labels
        assert pie_after["values"] == pytest.approx(exp_values, rel=1e-9, abs=1e-9)

    def test_pie_click_second_click_local(self, page: Page, server_port: int):
        """First slice click: one selection POST + one fire-and-forget
        cube_request (order-independent). Second click on another slice (OR
        toggle accumulates): ZERO further POSTs and the hist matches a direct
        recompute over the two-label union."""
        df = _cat_target_df()
        url = _pie_source_dashboard_url(server_port, "_cube_browser_pieclick")
        bodies = _capture_updates(page)
        page.goto(url)
        _wait_for_init(page, "plotly")

        n_before = len(bodies)
        _click_pie_slice(page, "#fv-plot-0", "g0")
        expected_g0 = _reference_categorical_hist_counts(df, pl.col("g").is_in(["g0"]))
        _wait_for_hist_equals(page, "#fv-plot-1", expected_g0)
        # The fire-and-forget cube_request must land in the client store.
        page.wait_for_function("() => _fvCubeStore.size > 0", timeout=10_000)
        page.wait_for_timeout(300)
        types = sorted(b.get("event", {}).get("type") for b in bodies[n_before:])
        assert types == ["cube_request", "selection"], types
        cube_body = next(
            b
            for b in bodies[n_before:]
            if b.get("event", {}).get("type") == "cube_request"
        )
        assert (cube_body.get("active_source") or {}).get("column") == "g"

        # Second click (OR toggle): served from the cube, zero further POSTs.
        n_local = len(bodies)
        _click_pie_slice(page, "#fv-plot-0", "g1")
        expected_union = _reference_categorical_hist_counts(
            df, pl.col("g").is_in(["g0", "g1"])
        )
        assert 0 < sum(expected_union) < df.height
        _wait_for_hist_equals(page, "#fv-plot-1", expected_union)
        page.wait_for_timeout(500)
        assert bodies[n_local:] == [], (
            f"second pie click must be served from the cube store; got "
            f"{[b.get('event', {}).get('type') for b in bodies[n_local:]]}"
        )
        sels = page.evaluate("DASHBOARD_SPEC.state.selections")
        assert len(sels) == 1
        _assert_legacy_isin_predicates(sels[0]["predicates"], "g", ["g0", "g1"])

    def test_float_pie_click_second_click_local(self, page: Page, server_port: int):
        """Float-label pie source follows the same warm-then-local path as
        string labels. Integral floats exercise typed free-category matching."""
        df = _float_bar_source_df()
        url = _float_pie_source_dashboard_url(
            server_port, "_cube_browser_floatpieclick"
        )
        bodies = _capture_updates(page)
        page.goto(url)
        _wait_for_init(page, "plotly")

        n_before = len(bodies)
        _click_pie_slice(page, "#fv-plot-0", 0)
        expected_0 = _reference_categorical_hist_counts(df, pl.col("gf").is_in([0.0]))
        _wait_for_hist_equals(page, "#fv-plot-1", expected_0)
        page.wait_for_function("() => _fvCubeStore.size > 0", timeout=10_000)
        page.wait_for_timeout(300)
        types = sorted(b.get("event", {}).get("type") for b in bodies[n_before:])
        assert types == ["cube_request", "selection"], types
        cube_body = next(
            b
            for b in bodies[n_before:]
            if b.get("event", {}).get("type") == "cube_request"
        )
        assert (cube_body.get("active_source") or {}).get("column") == "gf"

        n_local = len(bodies)
        _click_pie_slice(page, "#fv-plot-0", 1)
        expected_union = _reference_categorical_hist_counts(
            df, pl.col("gf").is_in([0.0, 1.0])
        )
        assert 0 < sum(expected_union) < df.height
        _wait_for_hist_equals(page, "#fv-plot-1", expected_union)
        page.wait_for_timeout(500)
        assert bodies[n_local:] == [], (
            f"second float pie click must be served from the cube store; got "
            f"{[b.get('event', {}).get('type') for b in bodies[n_local:]]}"
        )

    def test_treemap_depth2_click_prefix_served(self, page: Page, server_port: int):
        """Depth-2 treemap node clicks: the first POSTs and warms the cube;
        a second click on a DIFFERENT depth-2 node accumulates (independent
        paths append, OR) and is served from the cube — zero further POSTs,
        hist matching the recompute over the accumulated path predicates."""
        df = _treemap_df()
        url = _treemap_source_dashboard_url(server_port, "_cube_browser_treemapclick")
        bodies = _capture_updates(page)
        page.goto(url)
        _wait_for_init(page, "plotly")
        page.wait_for_function(
            "() => document.querySelector('#fv-plot-0')"
            "?.data?.[0]?.ids?.includes('root/g0/h1')"
        )

        n_before = len(bodies)
        _click_treemap_slice(page, "#fv-plot-0", "root/g0/h1")
        pred_a = (pl.col("g") == "g0") & (pl.col("h") == "h1")
        expected_a = _reference_categorical_hist_counts(df, pred_a)
        _wait_for_hist_equals(page, "#fv-plot-1", expected_a)
        page.wait_for_function("() => _fvCubeStore.size > 0", timeout=10_000)
        page.wait_for_timeout(300)
        types = sorted(b.get("event", {}).get("type") for b in bodies[n_before:])
        assert types == ["cube_request", "selection"], types
        cube_body = next(
            b
            for b in bodies[n_before:]
            if b.get("event", {}).get("type") == "cube_request"
        )
        assert (cube_body.get("active_source") or {}).get("column") == "g"

        # Second click on a different depth-2 node: fvUpsertPathPredicate
        # appends independent paths (OR), and the commit stays local.
        n_local = len(bodies)
        _click_treemap_slice(page, "#fv-plot-0", "root/g1/h0")
        pred_b = (pl.col("g") == "g1") & (pl.col("h") == "h0")
        expected_union = _reference_categorical_hist_counts(df, pred_a | pred_b)
        assert 0 < sum(expected_union) < df.height
        _wait_for_hist_equals(page, "#fv-plot-1", expected_union)
        page.wait_for_timeout(500)
        assert bodies[n_local:] == [], (
            f"second treemap click must be served from the cube store; got "
            f"{[b.get('event', {}).get('type') for b in bodies[n_local:]]}"
        )
        sels = page.evaluate("DASHBOARD_SPEC.state.selections")
        assert len(sels) == 1
        preds = sels[0]["predicates"]
        assert len(preds) == 2
        clause_sets = sorted(
            tuple(sorted((c["column"], c["values"][0]) for c in p["clauses"]))
            for p in preds
        )
        assert clause_sets == [
            (("g", "g0"), ("h", "h1")),
            (("g", "g1"), ("h", "h0")),
        ]
        for p in preds:
            for c in p["clauses"]:
                assert "closed" not in c and c.get("range") is None

    def test_live_brush_off_categorical_legacy(self, page: Page, server_port: int):
        """``live_brush="off"`` gates the whole step, clicks included: a bar
        box-drag and a pie click never request cubes, never live-update
        mid-drag, and POST exactly one selection per commit."""
        df = _cat_target_df()
        url = _categorical_legacy_dashboard_url(server_port, "_cube_browser_catoff")
        bodies = _capture_updates(page)
        page.goto(url)
        _wait_for_init(page, "plotly")
        _enter_select_mode(page)

        y_before = _hist_y(page, "#fv-plot-2")
        assert sum(y_before) == df.height

        # Bar box-drag: no mid-drag updates, one selection POST on commit.
        x1, x2, y1, y2 = _bar_drag_coords(page)
        n_before = len(bodies)
        page.mouse.move(x1, y1)
        page.mouse.down()
        page.mouse.move((x1 + x2) / 2, (y1 + y2) / 2, steps=8)
        page.mouse.move(x2, y2, steps=8)
        page.wait_for_timeout(800)
        assert (
            _hist_y(page, "#fv-plot-2") == y_before
        ), "no mid-drag updates with live_brush=off"
        page.mouse.up()
        page.wait_for_timeout(1_500)

        types = [b.get("event", {}).get("type") for b in bodies[n_before:]]
        assert types == ["selection"], types
        sels = page.evaluate("DASHBOARD_SPEC.state.selections")
        assert len(sels) == 1
        _assert_legacy_isin_predicates(sels[0]["predicates"], "g", ["g0", "g1"])
        assert _hist_y(page, "#fv-plot-2") != y_before

        # Pie click (the pie shows only g0/g1 now — click a surviving slice):
        # exactly one further selection POST, still no cube_request.
        n_click = len(bodies)
        _click_pie_slice(page, "#fv-plot-1", "g0")
        page.wait_for_function(
            "() => (DASHBOARD_SPEC.state.selections || []).length === 2"
        )
        page.wait_for_timeout(1_000)
        types = [b.get("event", {}).get("type") for b in bodies[n_click:]]
        assert types == ["selection"], types
        all_types = [b.get("event", {}).get("type") for b in bodies]
        assert "cube_request" not in all_types, all_types

    def test_bar_source_drag_live_updates_line_and_corr(
        self, page: Page, server_port: int
    ):
        """Box-drag over two bars of a categorical bar source live-updates BOTH
        a line_env target (#fv-plot-1) and a corr target (#fv-plot-2)
        mid-drag, with exactly ONE cube_request for the whole gesture and zero
        further POSTs (the categorical-free line_env/corr extension, #46). The
        commit keeps the unchanged legacy is_in predicate shape."""
        url = _bar_source_line_corr_dashboard_url(
            server_port, "_cube_browser_bar_line_corr"
        )
        bodies = _capture_updates(page)
        page.goto(url)
        _wait_for_init(page, "plotly")
        _enter_select_mode(page)

        line_before = _line_xy(page, "#fv-plot-1")
        z_before = _corr_z(page, "#fv-plot-2")
        assert len(line_before["x"]) > 0
        assert len(z_before) == 3 and len(z_before[0]) == 3

        x1, x2, y1, y2 = _bar_drag_coords(page)
        n_before = len(bodies)
        page.mouse.move(x1, y1)
        page.mouse.down()
        page.mouse.move((x1 + x2) / 2, (y1 + y2) / 2, steps=8)
        # Mid-drag, button still held: both targets re-render from local cube
        # slices (line envelope + corr finalize) with no further POSTs.
        page.wait_for_function(
            """(before) => {
                const gd = document.querySelector('#fv-plot-1');
                const ys = Array.from((gd.data && gd.data[0] && gd.data[0].y) || []);
                return ys.length > 0 && JSON.stringify(ys) !== JSON.stringify(before);
            }""",
            arg=line_before["y"],
            timeout=10_000,
        )
        page.wait_for_function(
            """(before) => {
                const gd = document.querySelector('#fv-plot-2');
                const z = (gd.data[0] && gd.data[0].z)
                    ? gd.data[0].z.map(r => Array.from(r)) : [];
                return z.length > 0 && JSON.stringify(z) !== JSON.stringify(before);
            }""",
            arg=z_before,
            timeout=10_000,
        )
        # During the drag: exactly one cube_request keyed on the label column,
        # nothing else (both targets served from local slices).
        mid_types = [b.get("event", {}).get("type") for b in bodies[n_before:]]
        assert mid_types == ["cube_request"], mid_types
        assert (bodies[n_before].get("active_source") or {}).get("column") == "g"

        page.mouse.move(x2, y2, steps=8)
        page.mouse.up()
        page.wait_for_timeout(1_500)

        # Commit: the presence of a line target forces the conditional commit to
        # POST (the line-envelope caveat — postRequired), so the gesture closes
        # with a single legacy selection replacing the approximate envelope.
        commit_types = [b.get("event", {}).get("type") for b in bodies[n_before + 1 :]]
        assert commit_types == ["selection"], commit_types
        assert _line_xy(page, "#fv-plot-1")["y"] != line_before["y"]
        assert _corr_z(page, "#fv-plot-2") != z_before
        sels = page.evaluate("DASHBOARD_SPEC.state.selections")
        assert len(sels) == 1
        _assert_legacy_isin_predicates(sels[0]["predicates"], "g", ["g0", "g1"])


# ---------------------------------------------------------------------------
# Cube source identity (plan step 0b)
# ---------------------------------------------------------------------------


class TestCubeSourceIdentity:
    def test_treemap_click_in_bar_treemap_figure_uses_clicked_trace(
        self, page: Page, server_port: int
    ):
        """A source figure holding bar(g) + treemap(g, h): a treemap node
        click must request/serve cubes for the TREEMAP's free axis ([g, h] +
        the treemap's trace uid), never the bar's first-match [g]. The second
        depth-2 click is then locally servable; with the pre-0b first-match
        bug the store would hold a [g]-free cube under the [g, h] key and the
        second click would silently skip both the update and the POST."""
        df = _treemap_df()
        url = _ambiguous_source_dashboard_url(server_port, "_cube_browser_ambig")
        bodies = _capture_updates(page)
        page.goto(url)
        _wait_for_init(page, "plotly")
        page.wait_for_function(
            "() => (document.querySelector('#fv-plot-0')?.data || [])"
            ".some(t => (t.ids || []).includes?.('root/g0/h1'))"
        )
        treemap_uid = page.evaluate("DASHBOARD_SPEC.figures[0].traces[1].uid")

        n_before = len(bodies)
        _click_treemap_slice(page, "#fv-plot-0", "root/g0/h1")
        pred_a = (pl.col("g") == "g0") & (pl.col("h") == "h1")
        expected_a = _reference_categorical_hist_counts(df, pred_a)
        _wait_for_hist_equals(page, "#fv-plot-1", expected_a)
        page.wait_for_function("() => _fvCubeStore.size > 0", timeout=10_000)
        page.wait_for_timeout(300)
        types = sorted(b.get("event", {}).get("type") for b in bodies[n_before:])
        assert types == ["cube_request", "selection"], types
        cube_body = next(
            b
            for b in bodies[n_before:]
            if b.get("event", {}).get("type") == "cube_request"
        )
        active = cube_body.get("active_source") or {}
        assert active.get("column") == "g"
        assert active.get("trace_uid") == treemap_uid

        # The stored cube's free axis is the treemap's full path tuple.
        stored_cols = page.evaluate("[..._fvCubeStore.values()][0].header.free.cols")
        assert stored_cols == ["g", "h"]

        # Second depth-2 click: served locally from the treemap cube.
        n_local = len(bodies)
        _click_treemap_slice(page, "#fv-plot-0", "root/g1/h0")
        pred_b = (pl.col("g") == "g1") & (pl.col("h") == "h0")
        expected_union = _reference_categorical_hist_counts(df, pred_a | pred_b)
        assert 0 < sum(expected_union) < df.height
        _wait_for_hist_equals(page, "#fv-plot-1", expected_union)
        page.wait_for_timeout(500)
        assert bodies[n_local:] == [], (
            f"second treemap click must be served from the cube store; got "
            f"{[b.get('event', {}).get('type') for b in bodies[n_local:]]}"
        )

    def test_header_mismatch_demotes_instead_of_storing(
        self, page: Page, server_port: int
    ):
        """A cube_request answered with a blob whose free header does not
        match the gesture's descriptor (here p=1024 instead of 2048) must be
        refused: nothing stored, no live updates, and the now-mixed commit
        POSTs one legacy selection that self-heals the target."""
        df = _cube_df()
        url = _two_hist_dashboard_url(server_port, "_cube_browser_hdrmm", "auto")

        a_lo, a_hi = float(df["a"].min()), float(df["a"].max())
        b_lo, b_hi = float(df["b"].min()), float(df["b"].max())
        wrong_spec = CubeSpec(
            source_name="_wrong",
            free=FreeAxisSpec(
                column="a", kind="continuous", p=1024, domain=(a_lo, a_hi)
            ),
            target_dims=(
                TargetDimSpec(
                    column="b",
                    kind="binned",
                    bins=_TGT_BINS,
                    domain=(b_lo, b_hi + _HIST_BIN_EPSILON),
                ),
            ),
            measure=MeasureSpec(agg="count"),
        )
        wrong_blob = encode_fvcube(build_cube(df.lazy(), wrong_spec), cube_id="wrong")

        def handle(route):
            body = json.loads(route.request.post_data or "{}")
            if body.get("request_cube"):
                target_uid = body["spec"]["figures"][1]["traces"][0]["uid"]
                bundle = encode_cube_bundle([wrong_blob], {target_uid: 0})
                route.fulfill(
                    body=bundle,
                    content_type="application/octet-stream",
                )
            else:
                route.continue_()

        page.route("**/dashboard/update", handle)
        bodies = _capture_updates(page)
        page.goto(url)
        _wait_for_init(page, "plotly")
        _enter_select_mode(page)

        y_before = _target_y(page)
        x1, x2, y, _ = _drag_coords(page)
        n_before = len(bodies)
        page.mouse.move(x1, y)
        page.mouse.down()
        page.mouse.move((x1 + x2) / 2, y, steps=8)
        # Wait for the poisoned cube_request to land...
        for _ in range(100):
            if any(b.get("request_cube") for b in bodies[n_before:]):
                break
            page.wait_for_timeout(100)
        page.wait_for_timeout(500)
        # ...and assert it was refused: nothing stored, no live update.
        assert page.evaluate("_fvCubeStore.size") == 0
        assert _target_y(page) == y_before

        page.mouse.move(x2, y, steps=8)
        page.wait_for_timeout(300)
        page.mouse.up()
        page.wait_for_timeout(1_000)

        # Mixed commit: exactly one cube_request + one selection POST.
        types = sorted(b.get("event", {}).get("type") for b in bodies[n_before:])
        assert types == ["cube_request", "selection"], types

        # The legacy selection self-heals the target.
        sels = page.evaluate("DASHBOARD_SPEC.state.selections")
        assert len(sels) == 1
        clause = sels[0]["predicates"][0]["clauses"][0]
        assert clause["column"] == "a"
        r0, r1 = clause["range"]
        expected = _reference_categorical_hist_counts(
            df, pl.col("a").is_between(r0, r1)
        )
        assert 0 < sum(expected) < df.height
        _wait_for_hist_equals(page, "#fv-plot-1", expected)


# ---------------------------------------------------------------------------
# Client cube store byte bound (plan step 0c)
# ---------------------------------------------------------------------------


class TestCubeStoreByteBound:
    def test_oversized_entry_refused_smaller_admitted(
        self, page: Page, server_port: int
    ):
        """Mirror the server cache rule: an entry larger than the whole
        budget is never stored (and evicts nothing); smaller entries are
        still admitted afterwards and the byte counter stays exact."""
        url = _two_hist_dashboard_url(server_port, "_cube_browser_bytes", "auto")
        page.goto(url)
        _wait_for_init(page, "plotly")

        result = page.evaluate("""() => {
                const prevBudget = fvCubeStoreSetBudget(1000);
                const before = _fvCubeStoreBytes;
                const refused = fvCubeStorePut('k-big', {bytes: 5000});
                const bytesAfterBig = _fvCubeStoreBytes;
                const admitted = fvCubeStorePut('k-small', {bytes: 200});
                const out = {
                    refused,
                    before,
                    bytesAfterBig,
                    hasBig: fvCubeStoreHas('k-big'),
                    admitted,
                    hasSmall: fvCubeStoreHas('k-small'),
                    bytesAfterSmall: _fvCubeStoreBytes,
                };
                fvCubeStoreSetBudget(prevBudget);
                fvCubeStoreReset();
                return out;
            }""")
        assert result["refused"] is False
        assert result["hasBig"] is False
        assert result["bytesAfterBig"] == result["before"]
        assert result["admitted"] is True
        assert result["hasSmall"] is True
        assert result["bytesAfterSmall"] == result["before"] + 200

    def test_oversized_cube_degrades_gesture_cleanly(
        self, page: Page, server_port: int
    ):
        """With the budget below any real decoded cube, the gesture's
        cube_request lands but the entry is refused: no live updates, the
        store stays empty, and the now-mixed commit POSTs one legacy
        selection that self-heals the target."""
        df = _cube_df()
        url = _two_hist_dashboard_url(server_port, "_cube_browser_bigblob", "auto")
        bodies = _capture_updates(page)
        page.goto(url)
        _wait_for_init(page, "plotly")
        _enter_select_mode(page)
        page.evaluate("fvCubeStoreSetBudget(64)")  # below any real cube

        y_before = _target_y(page)
        x1, x2, y, _ = _drag_coords(page)
        n_before = len(bodies)
        page.mouse.move(x1, y)
        page.mouse.down()
        page.mouse.move((x1 + x2) / 2, y, steps=8)
        for _ in range(100):
            if any(b.get("request_cube") for b in bodies[n_before:]):
                break
            page.wait_for_timeout(100)
        page.wait_for_timeout(500)
        assert page.evaluate("_fvCubeStore.size") == 0
        assert page.evaluate("_fvCubeStoreBytes") == 0
        assert _target_y(page) == y_before

        page.mouse.move(x2, y, steps=8)
        page.wait_for_timeout(300)
        page.mouse.up()
        page.wait_for_timeout(1_000)

        types = sorted(b.get("event", {}).get("type") for b in bodies[n_before:])
        assert types == ["cube_request", "selection"], types

        sels = page.evaluate("DASHBOARD_SPEC.state.selections")
        assert len(sels) == 1
        clause = sels[0]["predicates"][0]["clauses"][0]
        assert clause["column"] == "a"
        r0, r1 = clause["range"]
        expected = _reference_categorical_hist_counts(
            df, pl.col("a").is_between(r0, r1)
        )
        assert 0 < sum(expected) < df.height
        _wait_for_hist_equals(page, "#fv-plot-1", expected)


# ---------------------------------------------------------------------------
# Composite label Unicode parity (plan step 0d)
# ---------------------------------------------------------------------------


def _nonascii_composite_color_map() -> dict[str, str]:
    from flexviz.trace.base import _group_value_key

    return {
        _group_value_key(("é", "x")): "#112233",
        _group_value_key(("é", "y")): "#223344",
        _group_value_key(("ü", "x")): "#334455",
        _group_value_key(("ü", "y")): "#445566",
    }


def _nonascii_bar_dashboard_url(port: int, source_name: str) -> str:
    """Source hist(a) + composite-label bar(labels=[g1, g2]) target where g1
    holds non-ASCII labels; color_map keyed by the Python composite keys."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import LayoutSpec, encode_spec

    df = _cube_df()
    n = df.height
    df = df.with_columns(
        pl.Series("g1", ["é" if i % 2 == 0 else "ü" for i in range(n)]),
        pl.Series("g2", ["x" if i % 3 < 2 else "y" for i in range(n)]),
    )
    register_source(source_name, df, cache=True)

    dash = Dashboard(df)
    dash.add_figure(title="Source").add_histogram(x="a", bins=_SRC_BINS)
    dash.add_figure(title="Bar").add_bar(
        labels=["g1", "g2"], color_map=_nonascii_composite_color_map()
    )
    spec = dash.to_spec(source_name=source_name, layout=LayoutSpec(draggable=False))
    spec.client_state.live_brush = "auto"
    return f"http://127.0.0.1:{port}/view?spec={encode_spec(spec)}&renderer=plotly"


class TestCompositeLabelUnicodeParity:
    def test_fv_json_dumps_ascii_matches_python_goldens(
        self, page: Page, server_port: int
    ):
        """fvJsonDumpsAscii must reproduce Python's compact json.dumps
        (ensure_ascii=True) byte-for-byte — including the astral surrogate
        pair and the 0x7F DEL char bare JSON.stringify leaves raw."""
        from flexviz.trace.base import _group_value_key

        url = _two_hist_dashboard_url(server_port, "_cube_browser_jsondumps", "auto")
        page.goto(url)
        _wait_for_init(page, "plotly")

        cases = [("é", "x"), ("🎉", "x"), ("\x07", "x"), ("\x7f", "x"), ("a", "x")]
        for parts in cases:
            expected = _group_value_key(parts)
            got = page.evaluate("parts => fvJsonDumpsAscii(parts)", list(parts))
            assert (
                got == expected
            ), f"fvJsonDumpsAscii({parts!r}) = {got!r} != {expected!r}"

    def test_composite_nonascii_bar_cube_delta_matches_server(
        self, page: Page, server_port: int
    ):
        """A composite-label bar target with non-ASCII parts: the cube-built
        mid-drag delta's labels and color_map lookups must byte-equal the
        server-rendered ones (Python's ensure_ascii composite keys)."""
        from flexviz.trace.base import _group_value_key

        url = _nonascii_bar_dashboard_url(server_port, "_cube_browser_nonascii")
        cmap = _nonascii_composite_color_map()
        expected_labels = [
            _group_value_key(t)
            for t in sorted([("é", "x"), ("é", "y"), ("ü", "x"), ("ü", "y")])
        ]
        expected_colors = [cmap[label] for label in expected_labels]

        page.goto(url)
        _wait_for_init(page, "plotly")
        _enter_select_mode(page)

        bar_init = _bar_xy(page, "#fv-plot-1")
        # Server delta side of the parity: init labels are the Python keys.
        assert bar_init["x"] == expected_labels

        x1, x2, y, _ = _drag_coords(page)
        page.mouse.move(x1, y)
        page.mouse.down()
        page.mouse.move((x1 + x2) / 2, y, steps=8)
        # Mid-drag: the bar re-renders from a local cube slice.
        page.wait_for_function(
            """(yBefore) => {
                const gd = document.querySelector('#fv-plot-1');
                const ys = Array.from((gd.data && gd.data[0] && gd.data[0].y) || []);
                return ys.length > 0 && JSON.stringify(ys) !== JSON.stringify(yBefore);
            }""",
            arg=bar_init["y"],
            timeout=10_000,
        )
        mid = page.eval_on_selector(
            "#fv-plot-1",
            "gd => ({x: Array.from(gd.data[0].x || []), "
            "colors: Array.from((gd.data[0].marker || {}).color || [])})",
        )
        page.mouse.up()
        assert mid["x"] == expected_labels, mid["x"]
        assert mid["colors"] == expected_colors, mid["colors"]


# ---------------------------------------------------------------------------
# Stale-passive regression guard (plan step 1; updated by step 4)
# ---------------------------------------------------------------------------

_C_BINS = 10


def _three_hist_df() -> pl.DataFrame:
    df = _cube_df()
    n = df.height
    return df.with_columns(pl.Series("c", [((i * 29) % 700) / 7 for i in range(n)]))


def _three_hist_dashboard_url(port: int, source_name: str) -> str:
    """hist A(a) + hist B(b) + hist C(c) — all cube-capable sources/targets."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import LayoutSpec, encode_spec

    df = _three_hist_df()
    register_source(source_name, df, cache=True)

    dash = Dashboard(df)
    dash.add_figure(title="A").add_histogram(x="a", bins=_SRC_BINS)
    dash.add_figure(title="B").add_histogram(x="b", bins=_TGT_BINS)
    dash.add_figure(title="C").add_histogram(x="c", bins=_C_BINS)
    spec = dash.to_spec(source_name=source_name, layout=LayoutSpec(draggable=False))
    spec.client_state.live_brush = "auto"
    return f"http://127.0.0.1:{port}/view?spec={encode_spec(spec)}&renderer=plotly"


def _hist_counts_ref(
    df: pl.DataFrame, filter_expr: pl.Expr, col: str, bins: int
) -> list[int]:
    """Server-reference hist on *col* over its full domain under filter_expr."""
    lo, hi = df[col].min(), df[col].max()
    raw = (
        df.lazy()
        .filter(filter_expr)
        .select(
            pl.col(col)
            .flexviz.fixed_hist(
                pl.lit(float(lo)),
                pl.lit(float(hi) + _HIST_BIN_EPSILON),
                n_bins=bins,
            )
            .implode()
            .alias("h")
        )
        .collect()["h"]
        .item()
    )
    return raw.explode().struct.unnest()["count"].to_list()


def _selection_expr(df: pl.DataFrame, sel: dict) -> pl.Expr:
    from flexviz.predicates import predicates_to_expr
    from flexviz.spec import SelectionPredicate

    preds = [SelectionPredicate.model_validate(p) for p in sel["predicates"]]
    return predicates_to_expr(preds, df.schema)


def _fig_select_mode(page: Page, fig_idx: int) -> None:
    page.locator(f"#fv-bar-{fig_idx} .fv-mode-btn[data-mode='select']").click()
    page.wait_for_timeout(300)


def _fig_drag_coords(
    page: Page, fig_idx: int, lo: float = 0.2, hi: float = 0.6
) -> tuple[float, float, float]:
    box = page.locator(f"#fv-plot-{fig_idx} .nsewdrag").bounding_box()
    assert box is not None
    return (
        box["x"] + box["width"] * lo,
        box["x"] + box["width"] * hi,
        box["y"] + box["height"] * 0.5,
    )


def _fig_hist_y(page: Page, fig_idx: int) -> list[float]:
    return _hist_y(page, f"#fv-plot-{fig_idx}")


class TestStalePassiveGuard:
    def test_rebrush_with_foreign_selection_never_serves_stale_cubes(
        self, page: Page, server_port: int
    ):
        """Step-1 regression updated to the Step-4 passive-keyed behavior:
        brush+commit A (warms zero-passive cubes for B and C) → commit on B
        (now itself a lazy-2nd-selection cube gesture) → re-brush A. The
        re-brush must NOT serve the stale zero-passive slices: the new
        passive set {B} is a store miss ⇒ exactly one cube_request, then C
        live-updates from passive-baked cubes, B (owning a selection) is
        never touched, the commit stays local, and final values equal a
        server reference with both filters."""
        df = _three_hist_df()
        url = _three_hist_dashboard_url(server_port, "_cube_browser_stale")
        bodies = _capture_updates(page)
        page.goto(url)
        _wait_for_init(page, "plotly")

        # --- Phase 1: brush + commit on A (zero-passive, live, skipPost) ---
        _fig_select_mode(page, 0)
        n0 = len(bodies)
        _brush_commit(page, 0, 0.2, 0.6, live_fig_idx=1)
        types = [b.get("event", {}).get("type") for b in bodies[n0:]]
        assert types == ["cube_request"], types  # skipPost commit
        sels_phase1 = page.evaluate("DASHBOARD_SPEC.state.selections")
        assert len(sels_phase1) == 1  # A's committed selection

        # --- Phase 2: commit on B (lazy 2nd selection; passive = {A}) ---
        # A owns a committed selection: never live-updated by the B-gesture.
        _fig_select_mode(page, 1)
        y_a_before = _fig_hist_y(page, 0)
        n1 = len(bodies)
        _brush_commit(page, 1, 0.3, 0.7, live_fig_idx=2)
        types = [b.get("event", {}).get("type") for b in bodies[n1:]]
        assert types == ["cube_request"], types
        assert _fig_hist_y(page, 0) == y_a_before
        sels = page.evaluate("DASHBOARD_SPEC.state.selections")
        assert len(sels) == 2  # A's + B's

        # --- Phase 3: re-brush A — the zero-passive store entries for B and
        # C are stale now; the passive-keyed gesture must fetch fresh cubes
        # (one cube_request), live-update C only, and commit locally. ---
        _fig_select_mode(page, 0)
        y_b_before = _fig_hist_y(page, 1)
        n2 = len(bodies)
        _brush_commit(page, 0, 0.35, 0.75, live_fig_idx=2)
        types = [b.get("event", {}).get("type") for b in bodies[n2:]]
        assert types == ["cube_request"], types
        # B owns a committed selection — held through the whole phase.
        assert _fig_hist_y(page, 1) == y_b_before, "B touched by an A-gesture"

        # Final parity. B is untouched since its phase-2 state (it owns a
        # selection; its rendering still shows A's OLD filter from phase 1 —
        # B's own gesture never re-renders B). C carries BOTH new filters.
        sels = page.evaluate("DASHBOARD_SPEC.state.selections")
        fig_uids = page.evaluate("DASHBOARD_SPEC.figures.map(f => f.uid)")
        by_fig = {s["source_figure_uid"]: s for s in sels}
        expr_a_new = _selection_expr(df, by_fig[fig_uids[0]])
        expr_b = _selection_expr(df, by_fig[fig_uids[1]])
        expr_a_old = _selection_expr(df, sels_phase1[0])
        expected_b = _hist_counts_ref(df, expr_a_old, "b", _TGT_BINS)
        expected_c = _hist_counts_ref(df, expr_a_new & expr_b, "c", _C_BINS)
        assert 0 < sum(expected_c) < df.height
        _wait_for_hist_equals(page, "#fv-plot-1", expected_b)
        _wait_for_hist_equals(page, "#fv-plot-2", expected_c)


# ---------------------------------------------------------------------------
# Client passive keying + lazy 2nd selection (plan step 4 / contract E)
# ---------------------------------------------------------------------------


def _pie_with_other_dashboard_url(port: int, source_name: str) -> str:
    """Pie(g) source + hist(b) target + hist(a) 'other' figure whose committed
    selection is the foreign/passive one for pie clicks."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import LayoutSpec, encode_spec

    df = _cat_target_df()
    register_source(source_name, df, cache=True)

    dash = Dashboard(df)
    dash.add_figure(title="PieSource").add_pie(labels="g")
    dash.add_figure(title="Target").add_histogram(x="b", bins=_TGT_BINS)
    dash.add_figure(title="Other").add_histogram(x="a", bins=_SRC_BINS)
    spec = dash.to_spec(source_name=source_name, layout=LayoutSpec(draggable=False))
    spec.client_state.live_brush = "auto"
    return f"http://127.0.0.1:{port}/view?spec={encode_spec(spec)}&renderer=plotly"


def _brush_commit(
    page: Page, fig_idx: int, lo: float, hi: float, live_fig_idx: int | None = None
) -> None:
    """One full brush+commit gesture on figure *fig_idx*; when live_fig_idx
    is given, waits mid-drag until that figure's hist re-renders."""
    x1, x2, y = _fig_drag_coords(page, fig_idx, lo, hi)
    y_live_before = None if live_fig_idx is None else _fig_hist_y(page, live_fig_idx)
    page.mouse.move(x1, y)
    page.mouse.down()
    page.mouse.move((x1 + x2) / 2, y, steps=8)
    if live_fig_idx is not None:
        page.wait_for_function(
            """([idx, yBefore]) => {
                const gd = document.querySelector('#fv-plot-' + idx);
                const ys = Array.from((gd.data && gd.data[0] && gd.data[0].y) || []);
                return ys.length > 0 && JSON.stringify(ys) !== JSON.stringify(yBefore);
            }""",
            arg=[live_fig_idx, y_live_before],
            timeout=10_000,
        )
    page.mouse.move(x2, y, steps=8)
    page.wait_for_timeout(300)
    page.mouse.up()
    page.wait_for_timeout(1_000)


class TestClientPassiveKeying:
    def test_fv_cube_passive_key_canonicalization(self, page: Page, server_port: int):
        """JS mirror of contract E: empty/own/None-uid sets are null; values
        sorted; selection order insensitive; nesting preserved; closed
        included."""
        url = _two_hist_dashboard_url(server_port, "_cube_browser_pkeyjs", "auto")
        page.goto(url)
        _wait_for_init(page, "plotly")
        checks = page.evaluate("""() => {
                const sel = (uid, ...preds) =>
                  ({source_figure_uid: uid, predicates: preds});
                const range = {clauses: [{column: 'a', range: [1, 2], closed: 'left'}]};
                const rangeBoth = {clauses: [{column: 'a', range: [1, 2]}]};
                const v1 = {clauses: [{column: 'g', values: ['b', 'a']}]};
                const v2 = {clauses: [{column: 'g', values: ['a', 'b']}]};
                return {
                  empty: fvCubePassiveKey([], 'A'),
                  own: fvCubePassiveKey([sel('A', range)], 'A'),
                  anon: fvCubePassiveKey([sel(null, range)], 'A'),
                  noPreds: fvCubePassiveKey([sel('B')], 'A'),
                  valsSorted:
                    fvCubePassiveKey([sel('B', v1)], 'A')
                    === fvCubePassiveKey([sel('B', v2)], 'A'),
                  selOrderInsensitive:
                    fvCubePassiveKey([sel('B', range), sel('C', v1)], 'A')
                    === fvCubePassiveKey([sel('C', v1), sel('B', range)], 'A'),
                  predOrderInsensitive:
                    fvCubePassiveKey([sel('B', v1, range)], 'A')
                    === fvCubePassiveKey([sel('B', range, v1)], 'A'),
                  nestingPreserved:
                    fvCubePassiveKey([sel('B', range), sel('C', v1)], 'A')
                    !== fvCubePassiveKey([sel('B', range, v1)], 'A'),
                  closedIncluded:
                    fvCubePassiveKey([sel('B', range)], 'A')
                    !== fvCubePassiveKey([sel('B', rangeBoth)], 'A'),
                };
            }""")
        assert checks["empty"] is None
        assert checks["own"] is None
        assert checks["anon"] is None
        assert checks["noPreds"] is None
        for name in (
            "valsSorted",
            "selOrderInsensitive",
            "predOrderInsensitive",
            "nestingPreserved",
            "closedIncluded",
        ):
            assert checks[name] is True, name

    def test_lazy_second_selection_one_request_then_live(
        self, page: Page, server_port: int
    ):
        """(a): commit on A, then brush B — the new passive set is a store
        miss ⇒ exactly one cube_request, after which C live-updates mid-drag
        (A, owning a selection, holds) and the commit is local (skipPost).
        Final C equals a server reference with both predicates."""
        df = _three_hist_df()
        url = _three_hist_dashboard_url(server_port, "_cube_browser_lazy2")
        bodies = _capture_updates(page)
        page.goto(url)
        _wait_for_init(page, "plotly")

        # Brush+commit A (zero-passive warm, skipPost).
        _fig_select_mode(page, 0)
        n0 = len(bodies)
        _brush_commit(page, 0, 0.2, 0.6, live_fig_idx=1)
        assert [b["event"]["type"] for b in bodies[n0:]] == ["cube_request"]

        # Brush B: lazy 2nd selection.
        _fig_select_mode(page, 1)
        y_a_before = _fig_hist_y(page, 0)
        n1 = len(bodies)
        _brush_commit(page, 1, 0.3, 0.7, live_fig_idx=2)
        # Exactly one cube_request for the new passive set; zero other POSTs
        # (the commit was local — every non-owner target was cube-served).
        assert [b["event"]["type"] for b in bodies[n1:]] == ["cube_request"], [
            b["event"]["type"] for b in bodies[n1:]
        ]
        # A owns a committed selection — never live-updated by a B-gesture.
        assert _fig_hist_y(page, 0) == y_a_before

        # Committed C equals the recompute with BOTH predicates.
        sels = page.evaluate("DASHBOARD_SPEC.state.selections")
        fig_uids = page.evaluate("DASHBOARD_SPEC.figures.map(f => f.uid)")
        by_fig = {s["source_figure_uid"]: s for s in sels}
        expr_a = _selection_expr(df, by_fig[fig_uids[0]])
        expr_b = _selection_expr(df, by_fig[fig_uids[1]])
        expected_c = _hist_counts_ref(df, expr_a & expr_b, "c", _C_BINS)
        assert 0 < sum(expected_c) < df.height
        _wait_for_hist_equals(page, "#fv-plot-2", expected_c)

    def test_deselect_reverts_to_zero_passive_store_hit(
        self, page: Page, server_port: int
    ):
        """(c): after deselecting A, a re-brush on B reverts to the (still
        cached) zero-passive store entries — live with ZERO requests of any
        kind through its local commit."""
        df = _three_hist_df()
        url = _three_hist_dashboard_url(server_port, "_cube_browser_desel")
        bodies = _capture_updates(page)
        page.goto(url)
        _wait_for_init(page, "plotly")

        # 1. Brush+commit B — warms B-gesture zero-passive cubes (A + C).
        _fig_select_mode(page, 1)
        n0 = len(bodies)
        _brush_commit(page, 1, 0.2, 0.6, live_fig_idx=2)
        assert [b["event"]["type"] for b in bodies[n0:]] == ["cube_request"]

        # 2. Brush+commit A — lazy 2nd selection (passive = {B}).
        _fig_select_mode(page, 0)
        n1 = len(bodies)
        _brush_commit(page, 0, 0.3, 0.7, live_fig_idx=2)
        assert [b["event"]["type"] for b in bodies[n1:]] == ["cube_request"]

        # 3. Deselect A (double-click in select mode) — passive set shrinks.
        box = page.locator("#fv-plot-0 .nsewdrag").bounding_box()
        page.mouse.dblclick(
            box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5
        )
        page.wait_for_function(
            "() => DASHBOARD_SPEC.state.selections.length === 1", timeout=10_000
        )
        page.wait_for_timeout(800)

        # 4. Re-brush B — zero-passive keys again ⇒ full store hit: ZERO
        # requests, live updates, local commit.
        n2 = len(bodies)
        _brush_commit(page, 1, 0.25, 0.55, live_fig_idx=2)
        assert bodies[n2:] == [], [b["event"]["type"] for b in bodies[n2:]]

        sels = page.evaluate("DASHBOARD_SPEC.state.selections")
        assert len(sels) == 1
        expr_b = _selection_expr(df, sels[0])
        expected_c = _hist_counts_ref(df, expr_b, "c", _C_BINS)
        assert 0 < sum(expected_c) < df.height
        _wait_for_hist_equals(page, "#fv-plot-2", expected_c)

    def test_pie_click_with_foreign_selection_passive_keyed(
        self, page: Page, server_port: int
    ):
        """(d): pie clicks with a foreign selection committed — the first
        click POSTs (passive-keyed store miss) and warms; the second click
        conditionally commits ONLY after the passive-keyed store hit (zero
        further POSTs), with the hist matching the recompute under both
        filters."""
        df = _cat_target_df()
        url = _pie_with_other_dashboard_url(server_port, "_cube_browser_piefgn")
        bodies = _capture_updates(page)
        page.goto(url)
        _wait_for_init(page, "plotly")

        # Foreign selection: brush+commit on Other (fig 2, hist a).
        _fig_select_mode(page, 2)
        n0 = len(bodies)
        _brush_commit(page, 2, 0.2, 0.6, live_fig_idx=1)
        assert [b["event"]["type"] for b in bodies[n0:]] == ["cube_request"]
        other_sel = page.evaluate("DASHBOARD_SPEC.state.selections")[0]
        expr_other = _selection_expr(df, other_sel)

        # First pie click: passive-keyed miss ⇒ selection POST + one
        # fire-and-forget cube_request.
        store_before = page.evaluate("_fvCubeStore.size")
        n1 = len(bodies)
        _click_pie_slice(page, "#fv-plot-0", "g0")
        expected_g0 = _reference_categorical_hist_counts(
            df, expr_other & pl.col("g").is_in(["g0"])
        )
        _wait_for_hist_equals(page, "#fv-plot-1", expected_g0)
        # The fire-and-forget cube_request must land in the client store
        # (under the passive-aware key) before the second click can be local.
        page.wait_for_function(
            "(n) => _fvCubeStore.size > n", arg=store_before, timeout=10_000
        )
        page.wait_for_timeout(300)
        types = sorted(b["event"]["type"] for b in bodies[n1:])
        assert types == ["cube_request", "selection"], types

        # Second click: passive-keyed store hit ⇒ local conditional commit.
        n2 = len(bodies)
        _click_pie_slice(page, "#fv-plot-0", "g1")
        expected_union = _reference_categorical_hist_counts(
            df, expr_other & pl.col("g").is_in(["g0", "g1"])
        )
        assert 0 < sum(expected_union) < df.height
        _wait_for_hist_equals(page, "#fv-plot-1", expected_union)
        page.wait_for_timeout(500)
        assert bodies[n2:] == [], [b["event"]["type"] for b in bodies[n2:]]


# ---------------------------------------------------------------------------
# Overlay interplay (plan step 5 / contract F)
# ---------------------------------------------------------------------------


def _overlay_dashboard_url(
    port: int, source_name: str, live_brush: str = "auto", with_box: bool = False
) -> str:
    """Two-hist dashboard in cross_filter_mode='overlay' (optionally plus a
    box target — not cube-capable AND filtered_only — for the mixed case)."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import LayoutSpec, encode_spec

    df = _cube_df()
    register_source(source_name, df, cache=True)

    dash = Dashboard(df)
    dash.add_figure(title="Source").add_histogram(x="a", bins=_SRC_BINS)
    dash.add_figure(title="Target").add_histogram(x="b", bins=_TGT_BINS)
    if with_box:
        dash.add_figure(title="Box").add_boxplot(y="b")
    spec = dash.to_spec(source_name=source_name, layout=LayoutSpec(draggable=False))
    spec.state.cross_filter_mode = "overlay"
    spec.client_state.live_brush = live_brush
    return f"http://127.0.0.1:{port}/view?spec={encode_spec(spec)}&renderer=plotly"


def _rendered_layers(page: Page, sel: str) -> list[dict]:
    return page.eval_on_selector(
        sel,
        """gd => (gd.data || []).map(t => ({
            uid: t.uid,
            opacity: t.opacity,
            y: Array.from(t.y || []),
        }))""",
    )


class TestOverlayCube:
    def test_first_brush_bg_ghost_and_live_fg(self, page: Page, server_port: int):
        """(a): the FIRST brush in overlay mode shows the unfiltered bg ghost
        (dimmed) with the fg live-updating over it — one cube_request, zero
        further POSTs through the commit, and the committed fg equal to a
        server overlay reference."""
        df = _cube_df()
        url = _overlay_dashboard_url(server_port, "_cube_browser_ovl")
        bodies = _capture_updates(page)
        page.goto(url)
        _wait_for_init(page, "plotly")
        _enter_select_mode(page)

        unfiltered = [
            float(v) for v in _hist_counts_ref(df, pl.lit(True), "b", _TGT_BINS)
        ]
        layers0 = _rendered_layers(page, "#fv-plot-1")
        assert len(layers0) == 1  # bg only, full opacity, unfiltered
        assert layers0[0]["uid"].endswith("__fv_layer_bg")
        assert layers0[0]["opacity"] == 1
        assert layers0[0]["y"] == unfiltered

        x1, x2, y, _ = _drag_coords(page)
        n0 = len(bodies)
        page.mouse.move(x1, y)
        page.mouse.down()
        page.mouse.move((x1 + x2) / 2, y, steps=8)
        # Mid-drag: an fg trace appears and differs from the unfiltered bg.
        page.wait_for_function(
            """(unfiltered) => {
                const gd = document.querySelector('#fv-plot-1');
                const fg = (gd.data || []).find(t => t.uid.endsWith('__fv_layer_fg'));
                if (!fg) return false;
                const ys = Array.from(fg.y || []);
                return ys.length > 0 && JSON.stringify(ys) !== JSON.stringify(unfiltered);
            }""",
            arg=unfiltered,
            timeout=10_000,
        )
        mid = _rendered_layers(page, "#fv-plot-1")
        bg_mid = next(t for t in mid if t["uid"].endswith("__fv_layer_bg"))
        assert bg_mid["y"] == unfiltered  # the ghost is the unfiltered result
        assert bg_mid["opacity"] < 1  # ...dimmed
        page.mouse.move(x2, y, steps=8)
        page.wait_for_timeout(300)
        page.mouse.up()
        page.wait_for_timeout(1_000)

        types = [b["event"]["type"] for b in bodies[n0:]]
        assert types == ["cube_request"], types  # zero further POSTs

        # Committed fg equals the server overlay reference (filtered counts).
        sels = page.evaluate("DASHBOARD_SPEC.state.selections")
        assert len(sels) == 1
        expr = _selection_expr(df, sels[0])
        expected_fg = [float(v) for v in _hist_counts_ref(df, expr, "b", _TGT_BINS)]
        assert 0 < sum(expected_fg) < df.height
        final = _rendered_layers(page, "#fv-plot-1")
        fg_final = next(t for t in final if t["uid"].endswith("__fv_layer_fg"))
        bg_final = next(t for t in final if t["uid"].endswith("__fv_layer_bg"))
        assert fg_final["y"] == expected_fg
        assert bg_final["y"] == unfiltered

    def test_abandoned_gesture_restores_layers_incl_bg(
        self, page: Page, server_port: int
    ):
        """(b): an abandoned overlay gesture restores the exact pre-gesture
        rendering — fg gone, bg back to full opacity."""
        url = _overlay_dashboard_url(server_port, "_cube_browser_ovlabandon")
        page.goto(url)
        _wait_for_init(page, "plotly")
        _enter_select_mode(page)

        before = _rendered_layers(page, "#fv-plot-1")
        x1, x2, y, _ = _drag_coords(page)
        page.mouse.move(x1, y)
        page.mouse.down()
        page.mouse.move((x1 + x2) / 2, y, steps=8)
        page.wait_for_function(
            """() => {
                const gd = document.querySelector('#fv-plot-1');
                return (gd.data || []).some(t => t.uid.endsWith('__fv_layer_fg'));
            }""",
            timeout=10_000,
        )
        page.evaluate("divs[0].emit('plotly_selected', undefined)")
        page.wait_for_timeout(500)
        page.evaluate("divs[0].emit('plotly_selected', undefined)")
        page.wait_for_timeout(500)

        assert page.evaluate("DASHBOARD_SPEC.state.selections") == []
        after = _rendered_layers(page, "#fv-plot-1")
        assert after == before

    def test_mixed_overlay_dashboard_commit_posts_once(
        self, page: Page, server_port: int
    ):
        """(c): with a non-capable filtered_only box target, the capable hist
        still live-updates its fg mid-drag, the box holds pre-drag state, the
        commit POSTs exactly once, and the returned overlay deltas leave the
        hist fg equal to the server reference."""
        df = _cube_df()
        url = _overlay_dashboard_url(server_port, "_cube_browser_ovlmix", with_box=True)
        bodies = _capture_updates(page)
        page.goto(url)
        _wait_for_init(page, "plotly")
        _enter_select_mode(page)

        unfiltered = [
            float(v) for v in _hist_counts_ref(df, pl.lit(True), "b", _TGT_BINS)
        ]
        box_before = page.eval_on_selector(
            "#fv-plot-2", "gd => JSON.stringify((gd.data || []).map(t => t.y))"
        )
        x1, x2, y, _ = _drag_coords(page)
        n0 = len(bodies)
        page.mouse.move(x1, y)
        page.mouse.down()
        page.mouse.move((x1 + x2) / 2, y, steps=8)
        page.wait_for_function(
            """(unfiltered) => {
                const gd = document.querySelector('#fv-plot-1');
                const fg = (gd.data || []).find(t => t.uid.endsWith('__fv_layer_fg'));
                if (!fg) return false;
                const ys = Array.from(fg.y || []);
                return ys.length > 0 && JSON.stringify(ys) !== JSON.stringify(unfiltered);
            }""",
            arg=unfiltered,
            timeout=10_000,
        )
        # The box (not cube-capable) holds pre-drag state during the drag.
        assert (
            page.eval_on_selector(
                "#fv-plot-2", "gd => JSON.stringify((gd.data || []).map(t => t.y))"
            )
            == box_before
        )
        page.mouse.move(x2, y, steps=8)
        page.wait_for_timeout(300)
        page.mouse.up()
        page.wait_for_timeout(1_500)

        types = [b["event"]["type"] for b in bodies[n0:]]
        assert types.count("selection") == 1, types
        assert types.count("cube_request") == 1, types

        sels = page.evaluate("DASHBOARD_SPEC.state.selections")
        expr = _selection_expr(df, sels[0])
        expected_fg = [float(v) for v in _hist_counts_ref(df, expr, "b", _TGT_BINS)]
        page.wait_for_function(
            """(expected) => {
                const gd = document.querySelector('#fv-plot-1');
                const fg = (gd.data || []).find(t => t.uid.endsWith('__fv_layer_fg'));
                return fg && JSON.stringify(Array.from(fg.y || [])) === JSON.stringify(expected);
            }""",
            arg=expected_fg,
            timeout=10_000,
        )
        # The box's overlay delta was applied (its fg rendering changed).
        assert (
            page.eval_on_selector(
                "#fv-plot-2", "gd => JSON.stringify((gd.data || []).map(t => t.y))"
            )
            != box_before
        )

    def test_live_brush_off_overlay_is_legacy(self, page: Page, server_port: int):
        """(d): live_brush='off' in overlay mode stays bit-for-bit legacy —
        no cube_request, no mid-drag updates, one selection POST whose deltas
        render bg+fg."""
        df = _cube_df()
        url = _overlay_dashboard_url(server_port, "_cube_browser_ovloff", "off")
        bodies = _capture_updates(page)
        page.goto(url)
        _wait_for_init(page, "plotly")
        _enter_select_mode(page)

        before = _rendered_layers(page, "#fv-plot-1")
        x1, x2, y, _ = _drag_coords(page)
        n0 = len(bodies)
        page.mouse.move(x1, y)
        page.mouse.down()
        page.mouse.move((x1 + x2) / 2, y, steps=8)
        page.wait_for_timeout(800)
        assert _rendered_layers(page, "#fv-plot-1") == before  # no live updates
        page.mouse.move(x2, y, steps=8)
        page.wait_for_timeout(300)
        page.mouse.up()
        page.wait_for_timeout(1_500)

        types = [b["event"]["type"] for b in bodies[n0:]]
        assert types == ["selection"], types
        sels = page.evaluate("DASHBOARD_SPEC.state.selections")
        assert len(sels) == 1
        clause = sels[0]["predicates"][0]["clauses"][0]
        assert "closed" not in clause or clause.get("closed") == "both"
        expr = _selection_expr(df, sels[0])
        expected_fg = [float(v) for v in _hist_counts_ref(df, expr, "b", _TGT_BINS)]
        page.wait_for_function(
            """(expected) => {
                const gd = document.querySelector('#fv-plot-1');
                const fg = (gd.data || []).find(t => t.uid.endsWith('__fv_layer_fg'));
                return fg && JSON.stringify(Array.from(fg.y || [])) === JSON.stringify(expected);
            }""",
            arg=expected_fg,
            timeout=10_000,
        )


# ---------------------------------------------------------------------------
# Temporal client source (plan step 6 / contract G)
# ---------------------------------------------------------------------------


def _temporal_browser_df(kind: str) -> pl.DataFrame:
    """kind='us'|'date'. Temporal source column t + numeric b + labels g and
    values v for a mean bar target."""
    import datetime as dt

    n = 3_000
    if kind == "date":
        base = dt.date(2020, 1, 1)
        t = pl.Series("t", [base + dt.timedelta(days=(i * 7) % 900) for i in range(n)])
    else:
        base = dt.datetime(2020, 1, 1)
        t = pl.Series(
            "t",
            [base + dt.timedelta(minutes=(i * 37) % 50_000) for i in range(n)],
            dtype=pl.Datetime("us"),
        )
    return pl.DataFrame(
        {
            "t": t,
            "b": [((i * 53) % 500) / 5 for i in range(n)],
            "g": [f"g{(i * 7) % 4}" for i in range(n)],
            "v": [((i * 37) % 1000) / 10 + (i % 7) for i in range(n)],
        }
    )


def _temporal_dashboard_url(
    port: int, source_name: str, kind: str = "us", zoom: tuple[str, str] | None = None
) -> str:
    """Temporal hist(t) source + hist(b) count target + bar(g, mean v) target."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import AxisRange, LayoutSpec, encode_spec

    df = _temporal_browser_df(kind)
    register_source(source_name, df, cache=True)

    dash = Dashboard(df)
    dash.add_figure(title="TemporalSource").add_histogram(x="t", bins=_SRC_BINS)
    dash.add_figure(title="Target").add_histogram(x="b", bins=_TGT_BINS)
    dash.add_figure(title="MeanBar").add_bar(labels="g", values="v", agg="mean")
    spec = dash.to_spec(source_name=source_name, layout=LayoutSpec(draggable=False))
    spec.client_state.live_brush = "auto"
    if zoom is not None:
        spec.state.viewport[f"{spec.figures[0].uid}/x"] = AxisRange(
            min=zoom[0], max=zoom[1]
        )
    return f"http://127.0.0.1:{port}/view?spec={encode_spec(spec)}&renderer=plotly"


def _mean_bar_expected(df: pl.DataFrame, filter_expr: pl.Expr) -> dict[str, float]:
    out = df.lazy().filter(filter_expr).group_by("g").agg(pl.col("v").mean()).collect()
    return dict(zip(out["g"].to_list(), out["v"].to_list()))


def _assert_temporal_commit_and_parity(page: Page, df: pl.DataFrame) -> dict:
    """Common tail: read the committed snapped temporal predicate, assert
    parity of the hist count target and the mean bar target."""
    sels = page.evaluate("DASHBOARD_SPEC.state.selections")
    assert len(sels) == 1
    clause = sels[0]["predicates"][0]["clauses"][0]
    assert clause["column"] == "t"
    assert clause.get("closed") == "left"
    assert all(isinstance(v, str) for v in clause["range"]), clause["range"]
    expr = _selection_expr(df, sels[0])

    expected_hist = _hist_counts_ref(df, expr, "b", _TGT_BINS)
    assert 0 < sum(expected_hist) < df.height
    _wait_for_hist_equals(page, "#fv-plot-1", expected_hist)

    expected_means = _mean_bar_expected(df, expr)
    bar = _bar_xy(page, "#fv-plot-2")
    assert bar["x"] == sorted(expected_means)
    for label, got in zip(bar["x"], bar["y"]):
        assert got == pytest.approx(expected_means[label], rel=1e-9)
    return clause


class TestTemporalSourceCube:
    def test_datetime_source_live_brush_and_commit_parity(
        self, page: Page, server_port: int
    ):
        """A Datetime(us) hist source live-brushes a count + a mean target:
        one cube_request, zero further POSTs, snapped closed='left' STRING
        predicate, both targets equal to direct recomputes."""
        df = _temporal_browser_df("us")
        url = _temporal_dashboard_url(server_port, "_cube_browser_tsus")
        bodies = _capture_updates(page)
        page.goto(url)
        _wait_for_init(page, "plotly")
        _fig_select_mode(page, 0)

        n0 = len(bodies)
        _brush_commit(page, 0, 0.2, 0.6, live_fig_idx=1)
        types = [b["event"]["type"] for b in bodies[n0:]]
        assert types == ["cube_request"], types
        _assert_temporal_commit_and_parity(page, df)

    def test_zoomed_temporal_source_gesture(self, page: Page, server_port: int):
        """A pre-zoomed temporal source still live-brushes: the snap grid is
        the server-resolved viewport, the committed string edges lie within
        the zoom, and parity holds."""
        df = _temporal_browser_df("us")
        url = _temporal_dashboard_url(
            server_port,
            "_cube_browser_tszoom",
            zoom=("2020-01-05 00:00:00", "2020-01-25 00:00:00"),
        )
        bodies = _capture_updates(page)
        page.goto(url)
        _wait_for_init(page, "plotly")
        _fig_select_mode(page, 0)

        n0 = len(bodies)
        _brush_commit(page, 0, 0.25, 0.65, live_fig_idx=1)
        types = [b["event"]["type"] for b in bodies[n0:]]
        assert types == ["cube_request"], types
        clause = _assert_temporal_commit_and_parity(page, df)
        import datetime as dt

        lo = dt.datetime.fromisoformat(clause["range"][0])
        hi = dt.datetime.fromisoformat(clause["range"][1])
        assert lo >= dt.datetime(2020, 1, 5)
        assert hi <= dt.datetime(2020, 1, 25)

    def test_date_source_commits_integer_day_edges(self, page: Page, server_port: int):
        """A Date-typed source brush commits YYYY-MM-DD edges (the
        integer-day snap grid) with full parity."""
        df = _temporal_browser_df("date")
        url = _temporal_dashboard_url(server_port, "_cube_browser_tsday", kind="date")
        bodies = _capture_updates(page)
        page.goto(url)
        _wait_for_init(page, "plotly")
        _fig_select_mode(page, 0)

        n0 = len(bodies)
        _brush_commit(page, 0, 0.2, 0.6, live_fig_idx=1)
        types = [b["event"]["type"] for b in bodies[n0:]]
        assert types == ["cube_request"], types
        clause = _assert_temporal_commit_and_parity(page, df)
        # Integer-day edges: plain dates, no time component.
        for v in clause["range"]:
            assert len(v) == 10, v  # YYYY-MM-DD


# ---------------------------------------------------------------------------
# Zoom-key interplay hardening (plan step 7)
# ---------------------------------------------------------------------------


def _zoomed_dashboard_url(
    port: int,
    source_name: str,
    src_zoom: tuple[float, float] | None = None,
    tgt_zoom: tuple[float, float] | None = None,
    with_third: bool = False,
) -> str:
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import AxisRange, LayoutSpec, encode_spec

    df = _three_hist_df() if with_third else _cube_df()
    register_source(source_name, df, cache=True)

    dash = Dashboard(df)
    dash.add_figure(title="Source").add_histogram(x="a", bins=_SRC_BINS)
    dash.add_figure(title="Target").add_histogram(x="b", bins=_TGT_BINS)
    if with_third:
        dash.add_figure(title="Third").add_histogram(x="c", bins=_C_BINS)
    spec = dash.to_spec(source_name=source_name, layout=LayoutSpec(draggable=False))
    spec.client_state.live_brush = "auto"
    if src_zoom is not None:
        spec.state.viewport[f"{spec.figures[0].uid}/x"] = AxisRange(
            min=src_zoom[0], max=src_zoom[1]
        )
    if tgt_zoom is not None:
        spec.state.viewport[f"{spec.figures[1].uid}/x"] = AxisRange(
            min=tgt_zoom[0], max=tgt_zoom[1]
        )
    return f"http://127.0.0.1:{port}/view?spec={encode_spec(spec)}&renderer=plotly"


def _hist_counts_ref_domain(
    df: pl.DataFrame, filter_expr: pl.Expr, col: str, bins: int, lo: float, hi: float
) -> list[int]:
    """Zoomed-target hist reference: the legacy path FILTERS rows to the
    viewport before fixed_hist (out-of-domain rows never clip into the edge
    bins) — mirrored here, matching the cube's filter-don't-clip."""
    raw = (
        df.lazy()
        .filter(filter_expr & pl.col(col).is_between(lo, hi))
        .select(
            pl.col(col)
            .flexviz.fixed_hist(
                pl.lit(float(lo)), pl.lit(float(hi) + _HIST_BIN_EPSILON), n_bins=bins
            )
            .implode()
            .alias("h")
        )
        .collect()["h"]
        .item()
    )
    return raw.explode().struct.unnest()["count"].to_list()


class TestZoomKeyInterplayBrowser:
    def test_zoomed_source_brush_snaps_to_viewport_grid(
        self, page: Page, server_port: int
    ):
        """(a): a brush on a pre-zoomed source snaps to the VIEWPORT grid
        (finer bins): one cube_request, live updates, local commit, edges on
        the zoomed P-grid, full parity."""
        df = _cube_df()
        zoom = (10.0, 80.0)
        url = _zoomed_dashboard_url(server_port, "_cube_browser_zsrc", src_zoom=zoom)
        bodies = _capture_updates(page)
        page.goto(url)
        _wait_for_init(page, "plotly")
        _fig_select_mode(page, 0)

        n0 = len(bodies)
        _brush_commit(page, 0, 0.25, 0.65, live_fig_idx=1)
        types = [b["event"]["type"] for b in bodies[n0:]]
        assert types == ["cube_request"], types

        edge_lo, edge_hi = _committed_edges(page, "a")
        span = zoom[1] - zoom[0]
        # Edges lie exactly on the zoomed P-grid.
        lo_bin = round((edge_lo - zoom[0]) / span * _P)
        hi_bin = round((edge_hi - zoom[0]) / span * _P)
        assert edge_lo == pytest.approx(zoom[0] + lo_bin * span / _P, abs=1e-9)
        assert edge_hi == pytest.approx(zoom[0] + hi_bin * span / _P, abs=1e-9)
        assert zoom[0] <= edge_lo < edge_hi <= zoom[1] + span / _P

        sels = page.evaluate("DASHBOARD_SPEC.state.selections")
        expr = _selection_expr(df, sels[0])
        expected = _hist_counts_ref(df, expr, "b", _TGT_BINS)
        assert 0 < sum(expected) < df.height
        _wait_for_hist_equals(page, "#fv-plot-1", expected)

    def test_zoomed_target_new_key_then_reset_serves_old_entry(
        self, page: Page, server_port: int
    ):
        """(b)+(d): zooming a TARGET changes its store key (one new
        cube_request on the next brush); resetting the zoom reverts to the
        still-cached unzoomed entry (zero requests — the domain:null
        convention regression)."""
        df = _cube_df()
        url = _zoomed_dashboard_url(server_port, "_cube_browser_ztgt")
        bodies = _capture_updates(page)
        page.goto(url)
        _wait_for_init(page, "plotly")
        _fig_select_mode(page, 0)

        # Brush 1 (unzoomed): warms the unzoomed-target key.
        n0 = len(bodies)
        _brush_commit(page, 0, 0.2, 0.5, live_fig_idx=1)
        assert [b["event"]["type"] for b in bodies[n0:]] == ["cube_request"]

        # Zoom the target's x axis: its cube key changes.
        tgt_zoom = [20.0, 70.0]
        page.evaluate(
            """async (rng) => {
                await Plotly.relayout(divs[1], {'xaxis.range': rng});
            }""",
            tgt_zoom,
        )
        page.wait_for_timeout(1_000)

        # Brush 2: exactly one NEW cube_request for the zoomed-target key;
        # live updates and a local commit; parity over the zoomed domain.
        n1 = len(bodies)
        _brush_commit(page, 0, 0.3, 0.7, live_fig_idx=1)
        types = [b["event"]["type"] for b in bodies[n1:]]
        assert types == ["cube_request"], types
        sels = page.evaluate("DASHBOARD_SPEC.state.selections")
        expr = _selection_expr(df, sels[0])
        expected_zoomed = _hist_counts_ref_domain(
            df, expr, "b", _TGT_BINS, tgt_zoom[0], tgt_zoom[1]
        )
        assert 0 < sum(expected_zoomed) < df.height
        _wait_for_hist_equals(page, "#fv-plot-1", expected_zoomed)

        # Reset the target zoom via the per-figure reset (the runtime path
        # that clears the stored viewport — a bare autorange relayout also
        # reports concrete ranges, which would persist as a zoom-to-full).
        page.evaluate("() => window.fvOnResetPanel(DASHBOARD_SPEC.figures[1].uid)")
        page.wait_for_timeout(1_000)

        # Brush 3: the unzoomed entry from brush 1 still serves — ZERO
        # requests of any kind through the local commit (store has both).
        n2 = len(bodies)
        _brush_commit(page, 0, 0.25, 0.6, live_fig_idx=1)
        assert bodies[n2:] == [], [b["event"]["type"] for b in bodies[n2:]]
        sels = page.evaluate("DASHBOARD_SPEC.state.selections")
        expr = _selection_expr(df, sels[0])
        expected = _hist_counts_ref(df, expr, "b", _TGT_BINS)
        assert 0 < sum(expected) < df.height
        _wait_for_hist_equals(page, "#fv-plot-1", expected)

    def test_zoomed_source_with_foreign_selection(self, page: Page, server_port: int):
        """(c): zoom + passive combined — a zoomed source brushes with a
        foreign selection committed: one passive-keyed cube_request, live
        updates on the non-owner target, local commit, parity with both
        filters."""
        df = _three_hist_df()
        url = _zoomed_dashboard_url(
            server_port, "_cube_browser_zpass", src_zoom=(10.0, 80.0), with_third=True
        )
        bodies = _capture_updates(page)
        page.goto(url)
        _wait_for_init(page, "plotly")

        # Foreign selection: brush+commit on Third (hist c).
        _fig_select_mode(page, 2)
        n0 = len(bodies)
        _brush_commit(page, 2, 0.3, 0.7, live_fig_idx=1)
        assert [b["event"]["type"] for b in bodies[n0:]] == ["cube_request"]

        # Zoomed source brush with the passive set {Third}.
        _fig_select_mode(page, 0)
        n1 = len(bodies)
        _brush_commit(page, 0, 0.25, 0.65, live_fig_idx=1)
        types = [b["event"]["type"] for b in bodies[n1:]]
        assert types == ["cube_request"], types

        sels = page.evaluate("DASHBOARD_SPEC.state.selections")
        fig_uids = page.evaluate("DASHBOARD_SPEC.figures.map(f => f.uid)")
        by_fig = {s["source_figure_uid"]: s for s in sels}
        expr_src = _selection_expr(df, by_fig[fig_uids[0]])
        expr_third = _selection_expr(df, by_fig[fig_uids[2]])
        clause = by_fig[fig_uids[0]]["predicates"][0]["clauses"][0]
        assert clause["column"] == "a" and clause.get("closed") == "left"
        edge_lo, edge_hi = clause["range"]
        assert 10.0 <= edge_lo < edge_hi <= 80.0 + (80.0 - 10.0) / _P
        expected = _hist_counts_ref(df, expr_src & expr_third, "b", _TGT_BINS)
        assert 0 < sum(expected) < df.height
        _wait_for_hist_equals(page, "#fv-plot-1", expected)


# ---------------------------------------------------------------------------
# Box + line 1-D range sources (plan step 8)
# ---------------------------------------------------------------------------


def _box_source_dashboard_url(port: int, source_name: str) -> str:
    """Horizontal box source on ``a`` (prop axis = x, so the brush is the
    standard horizontal drag) + cube-capable hist(b) target."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import LayoutSpec, encode_spec

    df = _cube_df()
    register_source(source_name, df, cache=True)

    dash = Dashboard(df)
    dash.add_figure(title="BoxSource").add_boxplot(x="a")
    dash.add_figure(title="Target").add_histogram(x="b", bins=_TGT_BINS)
    spec = dash.to_spec(source_name=source_name, layout=LayoutSpec(draggable=False))
    spec.client_state.live_brush = "auto"
    return f"http://127.0.0.1:{port}/view?spec={encode_spec(spec)}&renderer=plotly"


def _line_source_df() -> pl.DataFrame:
    """`_cube_df` plus a sorted x column ``t`` for the line source. A t-brush
    selects a contiguous row range, which visibly reshapes hist(b)."""
    df = _cube_df()
    n = df.height
    return df.with_columns(pl.Series("t", [i / 30 for i in range(n)]))


def _line_source_dashboard_url(port: int, source_name: str) -> str:
    """Line source (x=t, y=a — selection is x-only) + hist(b) target."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import LayoutSpec, encode_spec

    df = _line_source_df()
    register_source(source_name, df, cache=True)

    dash = Dashboard(df)
    dash.add_figure(title="LineSource").add_line(x="t", y="a")
    dash.add_figure(title="Target").add_histogram(x="b", bins=_TGT_BINS)
    spec = dash.to_spec(source_name=source_name, layout=LayoutSpec(draggable=False))
    spec.client_state.live_brush = "auto"
    return f"http://127.0.0.1:{port}/view?spec={encode_spec(spec)}&renderer=plotly"


class TestBoxLineSourceCube:
    def test_box_source_brush_live_updates_hist(self, page: Page, server_port: int):
        """A brush over the box source live-updates the hist target with
        exactly one cube_request during the gesture and zero further POSTs;
        the conditional commit (skipPost) is local and the committed target
        values equal the server reference over the snapped closed="left"
        edges (the §8.2 parity property)."""
        df = _cube_df()
        url = _box_source_dashboard_url(server_port, "_cube_browser_boxsrc")
        bodies = _capture_updates(page)
        page.goto(url)
        _wait_for_init(page, "plotly")

        _fig_select_mode(page, 0)
        n0 = len(bodies)
        _brush_commit(page, 0, 0.25, 0.65, live_fig_idx=1)

        gesture_bodies = bodies[n0:]
        types = [b.get("event", {}).get("type") for b in gesture_bodies]
        assert types == ["cube_request"], types  # one fetch, skipPost commit
        assert gesture_bodies[0].get("request_cube") is True
        active = gesture_bodies[0].get("active_source") or {}
        assert active.get("column") == "a"

        # Committed client-side: one snapped closed="left" clause on "a".
        edge_lo, edge_hi = _committed_edges(page, "a")
        expected = _reference_slice_counts(df, edge_lo, edge_hi, free_col="a")
        assert 0 < sum(expected) < df.height
        _wait_for_hist_equals(page, "#fv-plot-1", expected)

    def test_line_source_brush_x_only_snapped_commit(
        self, page: Page, server_port: int
    ):
        """A brush over the line source live-updates the hist target (one
        cube_request, zero further POSTs) and commits a snapped x-only
        ``closed="left"`` predicate: exactly one clause, on the line's x
        column — the y column never appears (locked v0.2 x-only decision)."""
        df = _line_source_df()
        url = _line_source_dashboard_url(server_port, "_cube_browser_linesrc")
        bodies = _capture_updates(page)
        page.goto(url)
        _wait_for_init(page, "plotly")

        _fig_select_mode(page, 0)
        n0 = len(bodies)
        _brush_commit(page, 0, 0.25, 0.65, live_fig_idx=1)

        gesture_bodies = bodies[n0:]
        types = [b.get("event", {}).get("type") for b in gesture_bodies]
        assert types == ["cube_request"], types  # one fetch, skipPost commit
        active = gesture_bodies[0].get("active_source") or {}
        assert active.get("column") == "t"

        # x-only: exactly one predicate with exactly one clause, on "t".
        sels = page.evaluate("DASHBOARD_SPEC.state.selections")
        assert len(sels) == 1
        (pred,) = sels[0]["predicates"]
        assert len(pred["clauses"]) == 1, pred["clauses"]
        clause = pred["clauses"][0]
        assert clause["column"] == "t"
        assert clause.get("closed") == "left"
        assert not any(
            c["column"] == "a" for p in sels[0]["predicates"] for c in p["clauses"]
        ), "line commit must not carry a y-column clause"

        edge_lo, edge_hi = clause["range"]
        expected = _reference_slice_counts(df, edge_lo, edge_hi, free_col="t")
        assert 0 < sum(expected) < df.height
        _wait_for_hist_equals(page, "#fv-plot-1", expected)


# ---------------------------------------------------------------------------
# Line TARGET — line_env envelope (plan step 11 / contract J)
# ---------------------------------------------------------------------------

_LINE_N_POINTS = 200
_LINE_BUCKETS = _LINE_N_POINTS // 2


def _line_target_dashboard_url(
    port: int, source_name: str, live_brush: str = "auto"
) -> str:
    """Source hist(a) + minmax line target (x=b, y=a). A brush on ``a``
    restricts which rows contribute, reshaping the per-bucket y envelope.

    Sorted on ``b``: an ungrouped minmax line buckets by x width, which the
    engine only accepts on an ascending x column."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import LayoutSpec, encode_spec

    df = _cube_df().sort("b")
    register_source(source_name, df, cache=True)

    dash = Dashboard(df)
    dash.add_figure(title="Source").add_histogram(x="a", bins=_SRC_BINS)
    dash.add_figure(title="Line").add_line(
        x="b", y="a", n_points=_LINE_N_POINTS, downsample="minmax"
    )
    spec = dash.to_spec(source_name=source_name, layout=LayoutSpec(draggable=False))
    spec.client_state.live_brush = live_brush
    return f"http://127.0.0.1:{port}/view?spec={encode_spec(spec)}&renderer=plotly"


def _grouped_line_target_dashboard_url(port: int, source_name: str) -> str:
    """Source hist(a) + grouped minmax line target (x=b, y=a, group_by=cat).
    The grouped line is cube-capable: group cols are extra categorical dims."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import LayoutSpec, encode_spec

    df = _grouped_df()
    register_source(source_name, df, cache=True)

    dash = Dashboard(df)
    dash.add_figure(title="Source").add_histogram(x="a", bins=_SRC_BINS)
    dash.add_figure(title="GroupedLine").add_line(
        x="b", y="a", n_points=_LINE_N_POINTS, downsample="minmax", group_by="cat"
    )
    spec = dash.to_spec(source_name=source_name, layout=LayoutSpec(draggable=False))
    spec.client_state.live_brush = "auto"
    return f"http://127.0.0.1:{port}/view?spec={encode_spec(spec)}&renderer=plotly"


def _reference_line_envelope(
    df: pl.DataFrame,
    edge_lo: float,
    edge_hi: float,
    group_value: str | None = None,
) -> dict:
    """Python reference for the LIVE line envelope: a ``flexviz.cube``
    ``line_env`` build + ``slice_agg`` over the snapped bins recovered from the
    committed ``closed="left"`` edges, then expanded into the same
    ``{x, y}`` two-points-per-bucket shape the client emits."""
    a_lo, a_hi = float(df["a"].min()), float(df["a"].max())
    span = a_hi - a_lo
    lo_bin = round((edge_lo - a_lo) / span * _P)
    hi_bin = round((edge_hi - a_lo) / span * _P) - 1
    assert 0 <= lo_bin <= hi_bin <= _P

    b_lo, b_hi = float(df["b"].min()), float(df["b"].max())
    target_dims = [
        TargetDimSpec(
            column="b",
            kind="binned",
            bins=_LINE_BUCKETS,
            domain=(b_lo, b_hi + _HIST_BIN_EPSILON),
        )
    ]
    build_df = df
    if group_value is not None:
        target_dims.append(TargetDimSpec(column="cat", kind="categorical"))
    spec = CubeSpec(
        source_name="_reference",
        free=FreeAxisSpec(column="a", kind="continuous", p=_P, domain=(a_lo, a_hi)),
        target_dims=tuple(target_dims),
        measure=MeasureSpec(agg="line_env", value_col="a"),
    )
    result = build_cube(build_df.lazy(), spec)
    sliced = result.slice_agg(
        a_lo + lo_bin * span / _P, a_lo + (hi_bin + 1) * span / _P
    )
    if group_value is not None:
        sliced = sliced.filter(pl.col("cat") == group_value)
    sliced = sliced.sort("__bin__b")
    x: list[float] = []
    y: list[float] = []
    for row in sliced.iter_rows(named=True):
        pts = [
            (row["x_at_ymin"], row["y_min"]),
            (row["x_at_ymax"], row["y_max"]),
        ]
        # equal x ⇒ ymin first (stable on the original order).
        pts.sort(key=lambda p: p[0])
        for px, py in pts:
            x.append(px)
            y.append(py)
    return {"x": x, "y": y}


def _line_xy(page: Page, selector: str) -> dict:
    return page.eval_on_selector(
        selector,
        "gd => ({x: Array.from((gd.data[0] && gd.data[0].x) || []), "
        "y: Array.from((gd.data[0] && gd.data[0].y) || [])})",
    )


def _assert_xy_close(got: dict, expected: dict, tol: float = 1e-4) -> None:
    assert len(got["x"]) == len(
        expected["x"]
    ), f"x length {len(got['x'])} != {len(expected['x'])}"
    assert len(got["y"]) == len(expected["y"])
    for g, e in zip(got["x"], expected["x"]):
        assert g == pytest.approx(e, abs=tol), (g, e)
    for g, e in zip(got["y"], expected["y"]):
        assert g == pytest.approx(e, abs=tol), (g, e)


def _temporal_line_target_dashboard_url(port: int, source_name: str) -> str:
    """Source hist(a) + LINE target with a TEMPORAL x (x=ts, y=b, minmax)."""
    import datetime as dt

    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import LayoutSpec, encode_spec

    base = dt.datetime(2020, 1, 1)
    df = _cube_df().with_columns(
        pl.Series(
            "ts",
            [base + dt.timedelta(minutes=i) for i in range(3_000)],
            dtype=pl.Datetime("us"),
        )
    )
    register_source(source_name, df, cache=True)
    dash = Dashboard(df)
    dash.add_figure(title="Source").add_histogram(x="a", bins=_SRC_BINS)
    dash.add_figure(title="TLine").add_line(
        x="ts",
        y="b",
        n_points=_LINE_N_POINTS,
        downsample="minmax",
        assume_sorted_x=True,
    )
    spec = dash.to_spec(source_name=source_name, layout=LayoutSpec(draggable=False))
    spec.client_state.live_brush = "auto"
    return f"http://127.0.0.1:{port}/view?spec={encode_spec(spec)}&renderer=plotly"


def _grouped_temporal_line_target_dashboard_url(port: int, source_name: str) -> str:
    """Source hist(a) + GROUPED LINE target with a TEMPORAL x (the user's case:
    group dims interleave with the binned temporal bucket dim)."""
    import datetime as dt

    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import LayoutSpec, encode_spec

    base = dt.datetime(2020, 1, 1)
    df = _grouped_df().with_columns(
        pl.Series(
            "ts",
            [base + dt.timedelta(minutes=i) for i in range(3_000)],
            dtype=pl.Datetime("us"),
        )
    )
    register_source(source_name, df, cache=True)
    dash = Dashboard(df)
    dash.add_figure(title="Source").add_histogram(x="a", bins=_SRC_BINS)
    dash.add_figure(title="GTLine").add_line(
        x="ts",
        y="b",
        n_points=_LINE_N_POINTS,
        downsample="minmax",
        group_by="cat",
        assume_sorted_x=True,
    )
    spec = dash.to_spec(source_name=source_name, layout=LayoutSpec(draggable=False))
    spec.client_state.live_brush = "auto"
    return f"http://127.0.0.1:{port}/view?spec={encode_spec(spec)}&renderer=plotly"


def _line_gap_df() -> pl.DataFrame:
    """Line-x column ``b`` has an EMPTY band ``[40, 60)``: the minmax envelope
    always skips those buckets, so the downsampled output jumps the band and the
    gap-mask (diff > 4.1 * median diff) fires. ``a`` (the brush column) cycles
    independently of ``b`` so any contiguous a-subrange still keeps rows on both
    sides of the band — the *live* envelope therefore also straddles the gap."""
    n = 3_000
    a = [((i * 37) % 1000) / 10 for i in range(n)]  # 0..99.9, brush column
    b_raw = [((i * 53) % 800) / 10 for i in range(n)]  # 0..79.9
    b = [v if v < 40.0 else v + 20.0 for v in b_raw]  # [0, 40) ∪ [60, 99.9]
    return pl.DataFrame({"a": a, "b": b})


def _line_gap_dashboard_url(port: int, source_name: str, mode: str) -> str:
    """Source hist(a) + minmax line target (x=b, y=a) whose x has an empty band,
    in cross_filter_mode ``mode``. A live brush on ``a`` drives the line_env
    cube; the live envelope must carry the gap (null break) across the band.

    Sorted on ``b`` for the ungrouped minmax line's ascending-x contract."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import LayoutSpec, encode_spec

    df = _line_gap_df().sort("b")
    register_source(source_name, df, cache=True)

    dash = Dashboard(df)
    dash.add_figure(title="Source").add_histogram(x="a", bins=_SRC_BINS)
    dash.add_figure(title="Line").add_line(
        x="b", y="a", n_points=_LINE_N_POINTS, downsample="minmax"
    )
    spec = dash.to_spec(source_name=source_name, layout=LayoutSpec(draggable=False))
    spec.state.cross_filter_mode = mode
    spec.client_state.live_brush = "auto"
    return f"http://127.0.0.1:{port}/view?spec={encode_spec(spec)}&renderer=plotly"


def _live_line_y(page: Page, mode: str) -> list:
    """The y array of the LIVE line layer the cube path writes: the ``fg`` trace
    in overlay mode, ``data[0]`` (the replaced base) in update mode. Nulls are
    preserved (gap breaks) and surface here as ``None``."""
    if mode == "overlay":
        return page.eval_on_selector(
            "#fv-plot-1",
            """gd => {
                const fg = (gd.data || []).find(
                    t => (t.uid || '').endsWith('__fv_layer_fg'));
                return fg ? Array.from(fg.y || []) : [];
            }""",
        )
    return page.eval_on_selector(
        "#fv-plot-1",
        "gd => Array.from((gd.data[0] && gd.data[0].y) || [])",
    )


def _temporal_line_gap_url(port: int, source_name: str, tz: str | None = None) -> str:
    """A single temporal line whose daily date-x has an EMPTY band [day 20, 40):
    the minmax output jumps the band, so the gap mask (diff > 4.1 * median) fires.
    Used to prove the client renders gaps from ISO-string x with a gapless wire.
    ``tz`` makes the column timezone-aware (the server then serialises x with a
    ``Z`` / ``+00:00`` offset suffix, as energy.parquet's UTC column does)."""
    import datetime as _dt

    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import LayoutSpec, encode_spec

    base = _dt.datetime(2020, 1, 1)
    days = [d for d in range(60) if not (20 <= d < 40)]
    ts = [base + _dt.timedelta(days=d) for d in days]
    val = [float(d) for d in days]
    df = pl.DataFrame({"ts": ts, "val": val})
    if tz is not None:
        df = df.with_columns(pl.col("ts").dt.replace_time_zone(tz))
    register_source(source_name, df, cache=True)

    dash = Dashboard(df)
    dash.add_figure(title="Line").add_line(
        x="ts", y="val", n_points=_LINE_N_POINTS, downsample="minmax"
    )
    spec = dash.to_spec(source_name=source_name, layout=LayoutSpec(draggable=False))
    return f"http://127.0.0.1:{port}/view?spec={encode_spec(spec)}&renderer=plotly"


class TestLineGapsClientSide:
    def test_apply_line_gaps_helper_temporal(self, page: Page, server_port: int):
        """fvApplyLineGaps must insert a null break across a large TEMPORAL jump.
        x = three consecutive days then a ~8-month jump: only the last edge
        exceeds 4.1 * median(1 day), so one null lands before the final point."""
        url = _temporal_line_gap_url(server_port, "_line_gap_helper")
        page.goto(url)
        _wait_for_init(page, "plotly")
        out = page.evaluate("""() => window.fvApplyLineGaps(
                ['2020-01-01', '2020-01-02', '2020-01-03', '2020-09-01'],
                [1, 2, 3, 4], true)""")
        assert None in out["y"], "expected a null break across the temporal jump"
        assert out["x"].count(None) == 1
        # Break sits before the last point: [d1, d2, d3, null, d4].
        assert out["x"].index(None) == 3
        assert out["y"][:3] == [1, 2, 3] and out["y"][-1] == 4

    @pytest.mark.parametrize("suffix", ["Z", "+00:00", "+02:00", "-05:30"])
    def test_apply_line_gaps_helper_tz_aware(
        self, page: Page, server_port: int, suffix: str
    ):
        """fvApplyLineGaps must break across a large jump when x is a TIMEZONE-
        AWARE ISO string. Polars Datetime columns with a time_zone (e.g.
        energy.parquet's UTC) serialise x with a trailing 'Z' / '+HH:MM' offset;
        the gap projection must parse it (regression: tz suffix -> NaN -> no gap,
        so the committed line interpolated across the band)."""
        url = _temporal_line_gap_url(server_port, "_line_gap_tz")
        page.goto(url)
        _wait_for_init(page, "plotly")
        out = page.evaluate(
            """(s) => window.fvApplyLineGaps(
                ['2020-01-01T00:00:00'+s, '2020-01-02T00:00:00'+s,
                 '2020-01-03T00:00:00'+s, '2020-09-01T00:00:00'+s],
                [1, 2, 3, 4], true)""",
            suffix,
        )
        assert None in out["y"], f"expected a null break for tz suffix {suffix!r}"
        assert out["x"].count(None) == 1
        assert out["x"].index(None) == 3
        assert out["y"][:3] == [1, 2, 3] and out["y"][-1] == 4

    def test_temporal_to_physical_converts_offset_to_utc_instant(
        self, page: Page, server_port: int
    ):
        """fvTemporalToPhysical must SUBTRACT the timezone offset to recover the
        true UTC instant -- not merely parse the suffix as finite. The parametrized
        gap test above applies the SAME offset to every point, so the offset
        cancels in consecutive diffs and never proves the arithmetic. Here three
        spellings of one instant (2020-01-01T00:00:00Z) -- naive, +02:00 and
        -05:30 -- must collapse to a SINGLE physical value (the µs epoch)."""
        url = _temporal_line_gap_url(server_port, "_tz_phys_arith")
        page.goto(url)
        _wait_for_init(page, "plotly")
        vals = page.evaluate("""() => ({
                z:       window.fvTemporalToPhysical('2020-01-01T00:00:00Z', 'us'),
                naive:   window.fvTemporalToPhysical('2020-01-01T00:00:00', 'us'),
                plus2:   window.fvTemporalToPhysical('2020-01-01T02:00:00+02:00', 'us'),
                minus530:window.fvTemporalToPhysical('2019-12-31T18:30:00-05:30', 'us'),
                plus2hm: window.fvTemporalToPhysical('2020-01-01T02:00:00+0200', 'us'),
            })""")
        # The real UTC epoch of 2020-01-01T00:00:00Z, in microseconds.
        expected_us = 1577836800000000
        assert vals["z"] == expected_us
        # A naive string (no suffix) is read as UTC wall-clock → same instant.
        assert vals["naive"] == expected_us
        # Offsets are converted, not ignored: +02:00 wall-clock is 2h AHEAD of
        # UTC, -05:30 is 5.5h BEHIND — both denote the same 00:00:00Z instant.
        assert vals["plus2"] == expected_us, f"+02:00 not converted: {vals['plus2']}"
        assert (
            vals["minus530"] == expected_us
        ), f"-05:30 not converted: {vals['minus530']}"
        # The compact ±HHMM spelling must convert identically.
        assert vals["plus2hm"] == expected_us

    def test_temporal_init_line_gap_is_client_rendered(
        self, page: Page, server_port: int
    ):
        """A committed temporal line ships x as ISO strings with NO nulls; the
        rendered trace must still break across the empty date band -- proof the
        client (fvApplyLineGaps) inserts the gap from ISO-string x."""
        url = _temporal_line_gap_url(server_port, "_line_temporal_gap")
        page.goto(url)
        _wait_for_init(page, "plotly")
        xy = _line_xy(page, "#fv-plot-0")
        assert len(xy["y"]) > 0
        assert None in xy["y"], "temporal init line must break across the band"

    def test_tz_aware_init_line_gap_is_client_rendered(
        self, page: Page, server_port: int
    ):
        """The user's case: a tz-AWARE temporal line (energy.parquet is
        Datetime('us', 'UTC')). The committed x ships as ISO strings with a
        'Z' / '+00:00' suffix; the rendered line must still break across the
        empty date band -- regression: the tz suffix defeated the gap parser."""
        url = _temporal_line_gap_url(server_port, "_line_tz_gap", tz="UTC")
        page.goto(url)
        _wait_for_init(page, "plotly")
        xy = _line_xy(page, "#fv-plot-0")
        assert len(xy["y"]) > 0
        assert None in xy["y"], "tz-aware temporal init line must break across band"


def _temporal_histogram_url(port: int, source_name: str, tz: str | None = None) -> str:
    """A single temporal histogram: 100 hourly datetime points, 10 bins. The
    server bins on the physical representation and ships datetime bin centers so
    Plotly auto-detects a DATE axis (regression: fixed_hist panicked on the
    temporal column)."""
    import datetime as _dt

    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import LayoutSpec, encode_spec

    base = _dt.datetime(2020, 1, 1)
    ts = [base + _dt.timedelta(hours=i) for i in range(100)]
    df = pl.DataFrame({"t": pl.Series("t", ts, dtype=pl.Datetime("us"))})
    if tz is not None:
        df = df.with_columns(pl.col("t").dt.replace_time_zone(tz))
    register_source(source_name, df, cache=True)

    dash = Dashboard(df)
    dash.add_figure(title="Histogram").add_histogram(x="t", bins=10)
    spec = dash.to_spec(source_name=source_name, layout=LayoutSpec(draggable=False))
    return f"http://127.0.0.1:{port}/view?spec={encode_spec(spec)}&renderer=plotly"


class TestTemporalHistogramRender:
    @pytest.mark.parametrize("tz", [None, "UTC"])
    def test_temporal_histogram_renders_on_date_axis(
        self, page: Page, server_port: int, tz: str | None
    ):
        """A temporal histogram must render 10 bins on a Plotly DATE axis with
        every point counted (regression: fixed_hist panicked → no trace)."""
        url = _temporal_histogram_url(server_port, f"_temp_hist_{tz}", tz=tz)
        page.goto(url)
        _wait_for_init(page, "plotly")
        info = page.eval_on_selector(
            "#fv-plot-0",
            """gd => ({
                n: (gd.data[0] && gd.data[0].x || []).length,
                total: Array.from(gd.data[0].y || []).reduce((a, b) => a + b, 0),
                axisType: gd._fullLayout.xaxis.type,
            })""",
        )
        assert info["n"] == 10, "expected 10 temporal bins"
        assert info["total"] == 100, "every point must be counted"
        assert info["axisType"] == "date", "temporal histogram must use a date axis"


class TestLineTargetCube:
    def test_temporal_line_target_live_updates_on_date_axis(
        self, page: Page, server_port: int
    ):
        """Regression: a LINE target with a TEMPORAL x renders on a Plotly DATE
        axis (the server ships datetime → ISO strings). The cube bins that axis
        on the column's *physical* representation (epoch µs for Datetime("us")),
        so the live envelope must map physical → epoch-ms before restyling —
        otherwise the bare physical numbers (≈1000× an ms) land millennia
        off-axis and the panel renders EMPTY mid-drag. Asserts the mid-drag
        envelope x values are numeric epoch-ms inside the data's date range."""
        import datetime as dt

        url = _temporal_line_target_dashboard_url(server_port, "_cube_tline_tgt")
        page.goto(url)
        _wait_for_init(page, "plotly")
        _enter_select_mode(page)

        # The target axis is a date axis and the server seeds it with ISO
        # strings — the format the cube delta must remain renderable against.
        assert page.evaluate("divs[1]._fullLayout.xaxis.type") == "date"
        init_xy = _line_xy(page, "#fv-plot-1")
        assert len(init_xy["x"]) > 0
        assert all(isinstance(v, str) for v in init_xy["x"])  # server: ISO strings

        x1, x2, y, _w = _drag_coords(page)
        page.mouse.move(x1, y)
        page.mouse.down()
        page.mouse.move((x1 + x2) / 2, y, steps=8)
        # Mid-drag: the line target re-renders from the local envelope slice.
        page.wait_for_function(
            """(before) => {
                const gd = document.querySelector('#fv-plot-1');
                const xs = Array.from((gd.data && gd.data[0] && gd.data[0].x) || []);
                return xs.length > 0 && JSON.stringify(xs) !== JSON.stringify(before);
            }""",
            arg=init_xy["x"],
            timeout=10_000,
        )
        page.mouse.move(x2, y, steps=8)
        page.wait_for_timeout(300)
        drag_xy = _line_xy(page, "#fv-plot-1")
        page.mouse.up()

        # The panel is NOT empty, the axis is still a date axis, and every
        # envelope x is a numeric epoch-ms that decodes to a 2020 date (the data
        # year) — i.e. on-screen. Pre-fix these were epoch-µs (~year 51935).
        assert page.evaluate("divs[1]._fullLayout.xaxis.type") == "date"
        assert len(drag_xy["x"]) > 0 and len(drag_xy["y"]) > 0
        for xv in drag_xy["x"]:
            assert isinstance(
                xv, (int, float)
            ), f"expected numeric epoch-ms, got {xv!r}"
            d = dt.datetime(1970, 1, 1) + dt.timedelta(milliseconds=xv)
            assert (
                dt.datetime(2019, 12, 1) <= d <= dt.datetime(2020, 2, 1)
            ), f"envelope x {xv} decodes to {d}, outside the data's date range"

    def test_grouped_temporal_line_target_live_updates_on_date_axis(
        self, page: Page, server_port: int
    ):
        """The user's exact case: a GROUPED line target with a temporal x. Each
        child line renders on the shared date axis, so every child's mid-drag
        envelope x must be epoch-ms inside the data's date range — the binned
        temporal bucket dim is resolved correctly even with the group
        (categorical) dim interleaved in the cube target dims."""
        import datetime as dt

        url = _grouped_temporal_line_target_dashboard_url(
            server_port, "_cube_gtline_tgt"
        )
        page.goto(url)
        _wait_for_init(page, "plotly")
        _enter_select_mode(page)

        assert page.evaluate("divs[1]._fullLayout.xaxis.type") == "date"

        def child_xs():
            return page.eval_on_selector(
                "#fv-plot-1",
                "gd => (gd.data || []).map(t => Array.from(t.x || []))",
            )

        before = child_xs()
        assert len(before) >= 2 and all(len(xs) > 0 for xs in before)

        x1, x2, y, _w = _drag_coords(page)
        page.mouse.move(x1, y)
        page.mouse.down()
        page.mouse.move((x1 + x2) / 2, y, steps=8)
        page.wait_for_function(
            """(prev) => {
                const gd = document.querySelector('#fv-plot-1');
                const xs = (gd.data || []).map(t => Array.from(t.x || []));
                return JSON.stringify(xs) !== JSON.stringify(prev);
            }""",
            arg=before,
            timeout=10_000,
        )
        page.mouse.move(x2, y, steps=8)
        page.wait_for_timeout(300)
        after = child_xs()
        page.mouse.up()

        assert page.evaluate("divs[1]._fullLayout.xaxis.type") == "date"
        flat = [xv for xs in after for xv in xs]
        assert len(flat) > 0
        for xv in flat:
            assert isinstance(
                xv, (int, float)
            ), f"expected numeric epoch-ms, got {xv!r}"
            d = dt.datetime(1970, 1, 1) + dt.timedelta(milliseconds=xv)
            assert (
                dt.datetime(2019, 12, 1) <= d <= dt.datetime(2020, 2, 1)
            ), f"child envelope x {xv} decodes to {d}, outside the data's range"

    def test_hist_source_live_updates_line_target(self, page: Page, server_port: int):
        """A hist-source brush live-updates a LINE target: the envelope tracks
        the brush with exactly ONE cube_request and ZERO POSTs *during the
        drag*. The mid-drag envelope matches the quantize-then-combine
        reference (contract J)."""
        df = _cube_df()
        url = _line_target_dashboard_url(server_port, "_cube_browser_line_live")
        bodies = _capture_updates(page)
        page.goto(url)
        _wait_for_init(page, "plotly")
        _enter_select_mode(page)

        line_before = _line_xy(page, "#fv-plot-1")
        assert len(line_before["x"]) > 0

        x1, x2, y, _width = _drag_coords(page)
        n_before = len(bodies)
        page.mouse.move(x1, y)
        page.mouse.down()
        page.mouse.move((x1 + x2) / 2, y, steps=8)
        # Mid-drag, button still held: the line target re-renders from a local
        # envelope slice (no POSTs beyond the gesture-start cube_request).
        page.wait_for_function(
            """(before) => {
                const gd = document.querySelector('#fv-plot-1');
                const ys = Array.from((gd.data && gd.data[0] && gd.data[0].y) || []);
                return ys.length > 0 && JSON.stringify(ys) !== JSON.stringify(before);
            }""",
            arg=line_before["y"],
            timeout=10_000,
        )
        # During the drag: exactly one cube_request, nothing else.
        mid_types = [b.get("event", {}).get("type") for b in bodies[n_before:]]
        assert mid_types == ["cube_request"], mid_types

        page.mouse.move(x2, y, steps=8)
        page.wait_for_timeout(300)

        # Read the live envelope + the snapped brush bins BEFORE releasing.
        live_xy = _line_xy(page, "#fv-plot-1")
        snap = page.evaluate("""() => {
                const g = Object.values(_fvCubeGestures)[0];
                return g && g.lastBins ? g.lastBins : null;
            }""")
        assert snap is not None, "expected an active live gesture mid-drag"
        a_lo, a_hi = float(df["a"].min()), float(df["a"].max())
        span = a_hi - a_lo
        edge_lo = a_lo + snap[0] * span / _P
        edge_hi = a_lo + (snap[1] + 1) * span / _P
        expected = _reference_line_envelope(df, edge_lo, edge_hi)
        assert len(expected["x"]) > 0
        _assert_xy_close(live_xy, expected)

        page.mouse.up()
        page.wait_for_timeout(300)

    @pytest.mark.parametrize("mode", ["update", "overlay"])
    def test_live_brush_draws_line_gaps(self, page: Page, server_port: int, mode: str):
        """A line target whose x has an empty band must show the gap (a null
        break) DURING the live auto-brush, not only after the commit /update.

        The live cube path (``lineEnvDeltaFromCells``) historically emitted a
        gapless ``{x, y}`` while the server's ``LinePlot._to_update`` interleaved
        null rows — so the gap appeared only on mouse-up. This drives both
        cross-filter modes (the live delta is written to ``fg`` in overlay,
        ``base`` in update) and asserts the gap is present mid-drag."""
        url = _line_gap_dashboard_url(server_port, f"_cube_line_gap_{mode}", mode)
        page.goto(url)
        _wait_for_init(page, "plotly")
        _enter_select_mode(page)

        # Sanity: the rendered init line breaks across the empty band. The
        # server now ships gapless x/y; the client (fvApplyLineGaps at render)
        # inserts this null.
        init = _line_xy(page, "#fv-plot-1")
        assert len(init["y"]) > 0
        assert None in init["y"], "rendered init line should carry a client gap"

        x1, x2, y, _w = _drag_coords(page)
        page.mouse.move(x1, y)
        page.mouse.down()
        page.mouse.move((x1 + x2) / 2, y, steps=8)
        # Wait for the mid-drag local re-render of the LIVE line layer.
        if mode == "overlay":
            page.wait_for_function(
                """() => {
                    const gd = document.querySelector('#fv-plot-1');
                    const fg = (gd.data || []).find(
                        t => (t.uid || '').endsWith('__fv_layer_fg'));
                    return fg && (fg.y || []).length > 0;
                }""",
                timeout=10_000,
            )
        else:
            page.wait_for_function(
                """(before) => {
                    const gd = document.querySelector('#fv-plot-1');
                    const ys = Array.from((gd.data[0] && gd.data[0].y) || []);
                    return ys.length > 0
                        && JSON.stringify(ys) !== JSON.stringify(before);
                }""",
                arg=init["y"],
                timeout=10_000,
            )
        page.mouse.move(x2, y, steps=8)
        page.wait_for_timeout(300)

        live_y = _live_line_y(page, mode)
        page.mouse.up()

        assert len(live_y) > 0, "expected a live envelope mid-drag"
        assert None in live_y, (
            "live cube envelope must break the line across the empty band "
            "(gap drawn during the drag, not only after commit)"
        )

    def test_line_target_commit_always_posts_and_matches_no_cube(
        self, page: Page, server_port: int
    ):
        """Commit ALWAYS POSTs when a cube-served line target exists
        (postRequired, contract J): the legacy delta replaces the envelope.
        Committed state is bit-equal to a no-cube (live_brush="off") run over
        the SAME drag — §8.2 commit≡restore parity for the line target."""
        # --- cube run: brush + commit, capturing the committed line + edges --
        url = _line_target_dashboard_url(
            server_port, "_cube_browser_line_commit", "auto"
        )
        bodies = _capture_updates(page)
        page.goto(url)
        _wait_for_init(page, "plotly")
        _enter_select_mode(page)

        x1, x2, y, _width = _drag_coords(page)
        n_before = len(bodies)
        _drag_with_live_wait(page, x1, x2, y, _line_xy(page, "#fv-plot-1")["y"])
        page.wait_for_timeout(1_200)

        gesture_types = [b.get("event", {}).get("type") for b in bodies[n_before:]]
        # Live cube_request AND a committing selection POST (never skipPost for
        # a line target — postRequired forces the commit to POST).
        assert "cube_request" in gesture_types, gesture_types
        assert "selection" in gesture_types, gesture_types

        sels = page.evaluate("DASHBOARD_SPEC.state.selections")
        assert len(sels) == 1
        clause = sels[0]["predicates"][0]["clauses"][0]
        assert clause["column"] == "a"
        # Snapped, closed="left" (cube-eligible source geometry always snaps).
        assert clause.get("closed") == "left"
        edge_lo, edge_hi = clause["range"]

        # After the POST settles the line shows the committed server delta, not
        # the cube envelope. Both bucket by x width now, so the two are no
        # longer told apart by x ordering; the exact match against the freshly
        # fetched legacy delta below is what pins the replacement (the cube
        # packs its y partials as f32, so its envelope never matches exactly).
        page.wait_for_timeout(500)
        committed_line = _line_xy(page, "#fv-plot-1")
        assert len(committed_line["x"]) > 0

        # Reference: the EXACT legacy server delta for the committed selection,
        # fetched directly via /dashboard/update (no client cache, no cube
        # extension) — the §8.2 no-cube parity baseline. The line target is in
        # figures[1]; its only delta carries the legacy {x, y}.
        line_uid = page.evaluate("DASHBOARD_SPEC.figures[1].traces[0].uid")
        fig1_uid = page.evaluate("DASHBOARD_SPEC.figures[1].uid")
        ref_line = page.evaluate(
            """async ([lo, hi, lineUid, fig1Uid]) => {
                const spec = JSON.parse(JSON.stringify(DASHBOARD_SPEC));
                const srcUid = spec.figures[0].uid;
                const sel = {
                    source_figure_uid: srcUid,
                    predicates: [{clauses: [
                        {column: 'a', range: [lo, hi], closed: 'left'}]}],
                };
                spec.state.selections = [sel];
                const resp = await fetch(SERVER_URL + '/dashboard/update', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        spec,
                        event: {type: 'selection', selections: [sel],
                                figure_uid: srcUid, force_update: true},
                    }),
                });
                const data = await resp.json();
                const deltas = (data.figure_deltas[fig1Uid] || [])
                    .filter(d => d.uid === lineUid);
                const d = deltas[0] || {updates: {}};
                return {x: d.updates.x || [], y: d.updates.y || []};
            }""",
            [edge_lo, edge_hi, line_uid, fig1_uid],
        )
        # The committed cube-dashboard line equals the legacy server delta
        # exactly — the legacy delta fully replaced the envelope (contract J).
        assert committed_line["x"] == ref_line["x"]
        assert committed_line["y"] == ref_line["y"]

    def test_line_free_dashboard_still_skips_post(self, page: Page, server_port: int):
        """Regression: a dashboard with NO line target still skipPosts. The
        postRequired conjunct must not break the existing hist-only skipPost
        (a single cube_request, zero further POSTs through the commit)."""
        url = _two_hist_dashboard_url(
            server_port, "_cube_browser_line_free_regress", "auto"
        )
        bodies = _capture_updates(page)
        page.goto(url)
        _wait_for_init(page, "plotly")
        _enter_select_mode(page)

        y_before = _target_y(page)
        x1, x2, y, _width = _drag_coords(page)
        n_before = len(bodies)
        _drag_with_live_wait(page, x1, x2, y, y_before)
        page.wait_for_timeout(1_200)

        types = [b.get("event", {}).get("type") for b in bodies[n_before:]]
        assert types == ["cube_request"], types  # skipPost: no selection POST
        sels = page.evaluate("DASHBOARD_SPEC.state.selections")
        assert len(sels) == 1 and sels[0]["predicates"]

    def test_grouped_line_target_live_updates_children(
        self, page: Page, server_port: int
    ):
        """A grouped line target reconciles its per-series envelopes live
        during the drag; child uids are reused from init (never minted) and
        each child's mid-drag envelope matches a per-group reference."""
        df = _grouped_df()
        url = _grouped_line_target_dashboard_url(
            server_port, "_cube_browser_grouped_line"
        )
        bodies = _capture_updates(page)
        page.goto(url)
        _wait_for_init(page, "plotly")
        _enter_select_mode(page)

        read_children = (
            "gd => (gd.data || []).map(t => "
            "({uid: t.uid, name: t.name, "
            "x: Array.from(t.x || []), y: Array.from(t.y || [])}))"
        )
        children_before = page.eval_on_selector("#fv-plot-1", read_children)
        assert len(children_before) == 2, "grouped line must render two children"
        uids_before = sorted(t["uid"] for t in children_before)

        x1, x2, y, _width = _drag_coords(page)
        n_before = len(bodies)
        page.mouse.move(x1, y)
        page.mouse.down()
        page.mouse.move((x1 + x2) / 2, y, steps=8)
        page.wait_for_function(
            """(before) => {
                const gd = document.querySelector('#fv-plot-1');
                const now = (gd.data || []).map(t =>
                    ({name: t.name, y: Array.from(t.y || [])}));
                return now.length > 0 && JSON.stringify(now) !== JSON.stringify(before);
            }""",
            arg=[{"name": t["name"], "y": t["y"]} for t in children_before],
            timeout=10_000,
        )
        # During the drag: exactly one cube_request, nothing else.
        mid_types = [b.get("event", {}).get("type") for b in bodies[n_before:]]
        assert mid_types == ["cube_request"], mid_types

        page.mouse.move(x2, y, steps=8)
        page.wait_for_timeout(300)

        children_mid = page.eval_on_selector("#fv-plot-1", read_children)
        # Child uids reused from the init-rendered set — never minted.
        assert sorted(t["uid"] for t in children_mid) == uids_before

        snap = page.evaluate("""() => {
                const g = Object.values(_fvCubeGestures)[0];
                return g && g.lastBins ? g.lastBins : null;
            }""")
        assert snap is not None
        a_lo, a_hi = float(df["a"].min()), float(df["a"].max())
        span = a_hi - a_lo
        edge_lo = a_lo + snap[0] * span / _P
        edge_hi = a_lo + (snap[1] + 1) * span / _P

        by_name = {t["name"]: t for t in children_mid}
        for gv in ("A", "B"):
            assert gv in by_name, f"missing series {gv}"
            expected = _reference_line_envelope(df, edge_lo, edge_hi, group_value=gv)
            assert len(expected["x"]) > 0
            _assert_xy_close(by_name[gv], expected)

        page.mouse.up()
        page.wait_for_timeout(300)


# ---------------------------------------------------------------------------
# Correlation TARGET — corr cube (plan step 12 / contract I)
# ---------------------------------------------------------------------------


def _corr_df() -> pl.DataFrame:
    """Brush axis ``a`` plus three numeric columns. ``q`` tracks ``p`` in the
    low-``a`` half and anti-tracks it in the high-``a`` half, so corr(p, q)
    over a sub-range of ``a`` differs sharply from the full-range value — a
    brush therefore visibly reshapes the heatmap ``z``."""
    n = 3_000
    a = [((i * 37) % 1000) / 10 for i in range(n)]
    p = [float(i % 50) for i in range(n)]
    q = [(p[i] if a[i] < 50 else (50.0 - p[i])) + float((i * 11) % 7) for i in range(n)]
    rr = [float((i * 7) % 30) for i in range(n)]
    return pl.DataFrame({"a": a, "p": p, "q": q, "rr": rr})


def _corr_target_dashboard_url(
    port: int,
    source_name: str,
    *,
    columns: tuple[str, ...] = ("p", "q", "rr"),
    method: str = "pearson",
    live_brush: str = "auto",
) -> str:
    """Source hist(a) + corr-heatmap target over ``columns``. A brush on ``a``
    restricts which rows contribute, reshaping the correlation matrix."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import LayoutSpec, encode_spec

    df = _corr_df()
    register_source(source_name, df, cache=True)

    dash = Dashboard(df)
    dash.add_figure(title="Source").add_histogram(x="a", bins=_SRC_BINS)
    dash.add_figure(title="Corr").add_corr_heatmap(columns=list(columns), method=method)
    spec = dash.to_spec(source_name=source_name, layout=LayoutSpec(draggable=False))
    spec.client_state.live_brush = live_brush
    return f"http://127.0.0.1:{port}/view?spec={encode_spec(spec)}&renderer=plotly"


def _corr_mixed_dashboard_url(port: int, source_name: str) -> str:
    """hist(a) source + cube-capable hist(p) target + a SPEARMAN corr target.
    Spearman is rank-based (not decomposable) so the corr declines the cube
    path: the hist live-updates, the spearman corr holds, and the conditional
    commit must POST (mixed-commit self-healing)."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import LayoutSpec, encode_spec

    df = _corr_df()
    register_source(source_name, df, cache=True)

    dash = Dashboard(df)
    dash.add_figure(title="Source").add_histogram(x="a", bins=_SRC_BINS)
    dash.add_figure(title="HistP").add_histogram(x="p", bins=_TGT_BINS)
    dash.add_figure(title="Corr").add_corr_heatmap(
        columns=["p", "q", "rr"], method="spearman"
    )
    spec = dash.to_spec(source_name=source_name, layout=LayoutSpec(draggable=False))
    spec.client_state.live_brush = "auto"
    return f"http://127.0.0.1:{port}/view?spec={encode_spec(spec)}&renderer=plotly"


def _reference_corr_matrix(
    df: pl.DataFrame,
    edge_lo: float,
    edge_hi: float,
    columns: tuple[str, ...],
    *,
    absolute: bool = False,
    triangular: bool = False,
) -> dict:
    """Python reference for the LIVE corr matrix: a ``flexviz.cube`` corr build
    + ``corr_matrix`` finalize over the committed ``closed="left"`` brush edges
    (== the client ``corrDeltaFromEntry`` path)."""
    a_lo, a_hi = float(df["a"].min()), float(df["a"].max())
    spec = CubeSpec(
        source_name="_reference",
        free=FreeAxisSpec(column="a", kind="continuous", p=_P, domain=(a_lo, a_hi)),
        target_dims=(),
        measure=MeasureSpec(agg="corr", columns=tuple(columns)),
    )
    result = build_cube(df.lazy(), spec)
    return result.corr_matrix(
        edge_lo, edge_hi, absolute=absolute, triangular=triangular
    )


def _corr_z(page: Page, selector: str = "#fv-plot-1") -> list:
    return page.eval_on_selector(
        selector,
        "gd => (gd.data[0] && gd.data[0].z) "
        "? gd.data[0].z.map(r => Array.from(r)) : []",
    )


def _assert_z_close(got: list, expected: list, tol: float = 1e-6) -> None:
    assert len(got) == len(expected), (len(got), len(expected))
    for grow, erow in zip(got, expected):
        assert len(grow) == len(erow), (len(grow), len(erow))
        for g, e in zip(grow, erow):
            if g is None or e is None:
                assert g is None and e is None, (g, e)
            else:
                assert g == pytest.approx(e, abs=tol), (g, e)


def _snap_edges_from_gesture(page: Page, df: pl.DataFrame) -> tuple[float, float]:
    """Recover the snapped ``closed="left"`` brush edges from the live gesture's
    ``lastBins`` (the same arithmetic the line-target tests use)."""
    snap = page.evaluate("""() => {
            const g = Object.values(_fvCubeGestures)[0];
            return g && g.lastBins ? g.lastBins : null;
        }""")
    assert snap is not None, "expected an active live gesture mid-drag"
    a_lo, a_hi = float(df["a"].min()), float(df["a"].max())
    span = a_hi - a_lo
    return a_lo + snap[0] * span / _P, a_lo + (snap[1] + 1) * span / _P


class TestCorrTargetCube:
    def test_hist_source_live_updates_corr_target(self, page: Page, server_port: int):
        """A hist-source brush live-updates a CORR heatmap target: the ``z``
        matrix is restyled each frame with exactly ONE cube_request and ZERO
        POSTs during the drag. The mid-drag matrix matches the cube-finalize
        reference (contract I)."""
        df = _corr_df()
        url = _corr_target_dashboard_url(server_port, "_cube_browser_corr_live")
        bodies = _capture_updates(page)
        page.goto(url)
        _wait_for_init(page, "plotly")
        _enter_select_mode(page)

        z_before = _corr_z(page)
        assert len(z_before) == 3 and len(z_before[0]) == 3

        x1, x2, y, _width = _drag_coords(page)
        n_before = len(bodies)
        page.mouse.move(x1, y)
        page.mouse.down()
        page.mouse.move((x1 + x2) / 2, y, steps=8)
        # Mid-drag, button still held: the corr matrix re-renders from a local
        # finalize (no POSTs beyond the gesture-start cube_request).
        page.wait_for_function(
            """(before) => {
                const gd = document.querySelector('#fv-plot-1');
                const z = (gd.data[0] && gd.data[0].z)
                    ? gd.data[0].z.map(r => Array.from(r)) : [];
                return z.length > 0 && JSON.stringify(z) !== JSON.stringify(before);
            }""",
            arg=z_before,
            timeout=10_000,
        )
        mid_types = [b.get("event", {}).get("type") for b in bodies[n_before:]]
        assert mid_types == ["cube_request"], mid_types

        page.mouse.move(x2, y, steps=8)
        page.wait_for_timeout(300)

        live_z = _corr_z(page)
        edge_lo, edge_hi = _snap_edges_from_gesture(page, df)
        expected = _reference_corr_matrix(df, edge_lo, edge_hi, ("p", "q", "rr"))
        _assert_z_close(live_z, expected["z"])

        page.mouse.up()
        page.wait_for_timeout(300)

    def test_corr_target_conditional_commit_skips_post(
        self, page: Page, server_port: int
    ):
        """A pearson corr target is conditional-commit-eligible (postRequired
        is False): with corr the only target, every target is cube-served, so
        the commit is LOCAL — one cube_request, zero further POSTs. The
        committed matrix stays the cube-finalized ``z`` (no server delta
        replaces it) and equals the cube-finalize reference (§8.2 parity)."""
        df = _corr_df()
        url = _corr_target_dashboard_url(server_port, "_cube_browser_corr_commit")
        bodies = _capture_updates(page)
        page.goto(url)
        _wait_for_init(page, "plotly")
        _enter_select_mode(page)

        z_before = _corr_z(page)
        x1, x2, y, _width = _drag_coords(page)
        n_before = len(bodies)
        page.mouse.move(x1, y)
        page.mouse.down()
        page.mouse.move((x1 + x2) / 2, y, steps=8)
        page.wait_for_function(
            """(before) => {
                const gd = document.querySelector('#fv-plot-1');
                const z = (gd.data[0] && gd.data[0].z)
                    ? gd.data[0].z.map(r => Array.from(r)) : [];
                return z.length > 0 && JSON.stringify(z) !== JSON.stringify(before);
            }""",
            arg=z_before,
            timeout=10_000,
        )
        page.mouse.move(x2, y, steps=8)
        page.wait_for_timeout(300)
        edge_lo, edge_hi = _snap_edges_from_gesture(page, df)
        page.mouse.up()
        page.wait_for_timeout(1_200)

        gesture_bodies = bodies[n_before:]
        cube_requests = [
            b
            for b in gesture_bodies
            if b.get("event", {}).get("type") == "cube_request"
        ]
        assert len(cube_requests) == 1, [
            b.get("event", {}).get("type") for b in gesture_bodies
        ]
        # Conditional commit: corr is cube-served AND not postRequired, so the
        # commit is local — zero further POSTs of any type through mouseup.
        others = [b for b in gesture_bodies if b not in cube_requests]
        assert others == [], [b.get("event", {}).get("type") for b in others]

        # The selection committed client-side, snapped closed="left".
        sels = page.evaluate("DASHBOARD_SPEC.state.selections")
        assert len(sels) == 1
        clause = sels[0]["predicates"][0]["clauses"][0]
        assert clause["column"] == "a" and clause.get("closed") == "left"

        # The committed matrix is the cube-finalized z (never replaced by a
        # server delta) and equals the reference for the snapped brush.
        expected = _reference_corr_matrix(df, edge_lo, edge_hi, ("p", "q", "rr"))
        _assert_z_close(_corr_z(page), expected["z"])

    def test_spearman_corr_mixed_commit_self_heals(self, page: Page, server_port: int):
        """A SPEARMAN corr target is not cube-decomposable: the cube-capable
        hist target live-updates during the drag while the spearman corr holds
        its pre-drag matrix, and the conditional commit falls back to exactly
        one snapped selection POST. Afterwards the corr is server-recomputed
        (mixed-commit self-healing) and matches ``CorrHeatmap._to_update``."""
        from flexviz.trace.corr_heatmap import CorrHeatmap

        df = _corr_df()
        url = _corr_mixed_dashboard_url(server_port, "_cube_browser_corr_spearman")
        bodies = _capture_updates(page)
        page.goto(url)
        _wait_for_init(page, "plotly")
        _enter_select_mode(page)

        z_corr_before = _corr_z(page, "#fv-plot-2")
        assert len(z_corr_before) == 3
        hist_before = _target_y(page)  # #fv-plot-1 (hist p)

        x1, x2, y, _width = _drag_coords(page)
        n_before = len(bodies)
        page.mouse.move(x1, y)
        page.mouse.down()
        page.mouse.move((x1 + x2) / 2, y, steps=8)
        # The hist target updates live; the spearman corr must hold its matrix.
        page.wait_for_function(
            """(yBefore) => {
                const gd = document.querySelector('#fv-plot-1');
                const ys = Array.from((gd.data && gd.data[0] && gd.data[0].y) || []);
                return ys.length > 0 && JSON.stringify(ys) !== JSON.stringify(yBefore);
            }""",
            arg=hist_before,
            timeout=10_000,
        )
        assert _corr_z(page, "#fv-plot-2") == z_corr_before, "spearman corr must hold"
        page.mouse.move(x2, y, steps=8)
        page.wait_for_timeout(300)
        page.mouse.up()
        # Commit round-trips: wait until the server delta updates the corr.
        page.wait_for_function(
            """(before) => {
                const gd = document.querySelector('#fv-plot-2');
                const z = (gd.data[0] && gd.data[0].z)
                    ? gd.data[0].z.map(r => Array.from(r)) : [];
                return z.length > 0 && JSON.stringify(z) !== JSON.stringify(before);
            }""",
            arg=z_corr_before,
            timeout=10_000,
        )
        page.wait_for_timeout(500)

        # Exactly one cube_request + one selection POST (spearman declines the
        # cube path → not all targets cube-served → conditional commit POSTs).
        types = [b.get("event", {}).get("type") for b in bodies[n_before:]]
        assert sorted(types) == ["cube_request", "selection"], types
        sel_body = next(
            b
            for b in bodies[n_before:]
            if b.get("event", {}).get("type") == "selection"
        )
        clause = sel_body["event"]["selections"][0]["predicates"][0]["clauses"][0]
        assert clause["column"] == "a" and clause.get("closed") == "left"
        edge_lo, edge_hi = clause["range"]

        # The spearman corr self-heals to the server recompute: the exact
        # CorrHeatmap._to_update output on the snapped half-open filtered rows.
        sub = df.filter((pl.col("a") >= edge_lo) & (pl.col("a") < edge_hi))
        assert 0 < sub.height < df.height
        trace = CorrHeatmap(columns=["p", "q", "rr"], method="spearman")
        trace.uid = "u"
        legacy = trace._to_update(
            sub.select(trace.get_aggregation_spec({}).expr)
        ).updates
        _assert_z_close(_corr_z(page, "#fv-plot-2"), legacy["z"])


# ---------------------------------------------------------------------------
# 2-D box (hist2d) source (Step 13 / contract H)
# ---------------------------------------------------------------------------
#
# plotly_selecting investigation (Step 13): a box-select drag over a hist2d
# heatmap with dragmode="select" fires standard plotly_selecting events with
# eventData.range = {x:[...], y:[...]} on every mousemove — exactly like a
# cartesian box select — so the GESTURE PATH IS THE SAME standard
# plotly_selecting path used by 1-D range sources. No capture-phase
# pointer/rAF outline reading (commit d556253) is needed for the FRESH-DRAW
# gesture; the edit-drag of an already-committed box still uses that machinery
# (handleSelectionEditPointerDown), which feeds both x and y back through
# handleSelecting and so works for box2d unchanged. The tests below prove the
# standard path (live updates + one cube_request) and the edit-drag replay.

_BOX2D_P = 128


def _box2d_df() -> pl.DataFrame:
    """a, b numeric source axes; c a numeric target; g a low-cardinality string
    target. (a, b) jointly partition the rows so a 2-D box on (a, b) reshapes
    both the hist(c) and the bar(g) aggregates."""
    n = 3_000
    return pl.DataFrame(
        {
            "a": [((i * 37) % 1000) / 10 for i in range(n)],
            "b": [((i * 53) % 500) / 5 for i in range(n)],
            "c": [((i * 29) % 700) / 7 for i in range(n)],
            "g": [f"g{(i * 7) % 4}" for i in range(n)],
        }
    )


def _box2d_dashboard_url(port: int, source_name: str, live_brush: str = "auto") -> str:
    """Source hist2d(x=a, y=b) + hist(c) target + bar(g) target."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import LayoutSpec, encode_spec

    df = _box2d_df()
    register_source(source_name, df, cache=True)

    dash = Dashboard(df)
    dash.add_figure(title="Box2dSource").add_histogram2d(
        x="a", y="b", x_bins=8, y_bins=6
    )
    dash.add_figure(title="Hist").add_histogram(x="c", bins=_TGT_BINS)
    dash.add_figure(title="Bar").add_bar(labels="g")
    spec = dash.to_spec(source_name=source_name, layout=LayoutSpec(draggable=False))
    spec.client_state.live_brush = live_brush
    return f"http://127.0.0.1:{port}/view?spec={encode_spec(spec)}&renderer=plotly"


def _box2d_drag_coords(page: Page) -> dict:
    """A 2-D rectangle inside the hist2d source plot (#fv-plot-0)."""
    box = page.locator("#fv-plot-0 .nsewdrag").bounding_box()
    assert box is not None
    return {
        "x1": box["x"] + box["width"] * 0.2,
        "x2": box["x"] + box["width"] * 0.7,
        "y1": box["y"] + box["height"] * 0.25,
        "y2": box["y"] + box["height"] * 0.75,
        "width": box["width"],
        "height": box["height"],
    }


def _edges_to_bins(lo: float, hi: float, dlo: float, dhi: float) -> tuple[int, int]:
    """Recover the [lo_bin, hi_bin] inclusive bin range from snapped closed-left
    edges over a P=128 grid (the committed edges are edge(lo_bin) and
    edge(hi_bin+1) — do NOT re-snap them, which would add one bin)."""
    span = dhi - dlo
    lo_bin = round((lo - dlo) / span * _BOX2D_P)
    hi_bin = round((hi - dlo) / span * _BOX2D_P) - 1
    assert 0 <= lo_bin <= hi_bin <= _BOX2D_P
    return lo_bin, hi_bin


def _reference_box2d_hist_counts(
    df: pl.DataFrame, ex: tuple[float, float], ey: tuple[float, float]
) -> list[float]:
    """Python reference for the hist(c) target: build a box2d cube over the
    full (a, b) domains and rectangle-slice the committed bins (recovered from
    the snapped closed-left edges), mirroring the client."""
    a_lo, a_hi = df["a"].min(), df["a"].max()
    b_lo, b_hi = df["b"].min(), df["b"].max()
    c_lo, c_hi = df["c"].min(), df["c"].max()
    spec = CubeSpec(
        source_name="_ref",
        free=FreeAxisSpec(
            column="a",
            kind="box2d",
            p=_BOX2D_P,
            columns=("a", "b"),
            domains=((a_lo, a_hi), (b_lo, b_hi)),
        ),
        target_dims=(
            TargetDimSpec(
                column="c",
                kind="binned",
                bins=_TGT_BINS,
                domain=(c_lo, c_hi + _HIST_BIN_EPSILON),
            ),
        ),
        measure=MeasureSpec(agg="count"),
    )
    result = build_cube(df.lazy(), spec)
    lx, hx = _edges_to_bins(ex[0], ex[1], a_lo, a_hi)
    ly, hy = _edges_to_bins(ey[0], ey[1], b_lo, b_hi)
    s = _BOX2D_P + 1
    codes = []
    for by in range(ly, hy + 1):
        codes.extend(range(by * s + lx, by * s + hx + 1))
    sliced = (
        result.frame.filter(pl.col("free_bin").is_in(codes))
        .group_by("__bin__c")
        .agg(pl.col("count").sum())
    )
    counts = [0.0] * _TGT_BINS
    for cat, val in zip(sliced["__bin__c"].to_list(), sliced["count"].to_list()):
        counts[cat] += val
    return counts


class TestBox2dSourceCube:
    def test_box2d_live_updates_hist_and_bar(self, page: Page, server_port: int):
        """A box-select on the hist2d source live-updates BOTH a hist and a bar
        target with exactly ONE cube_request and zero further POSTs through the
        drag. Proves plotly_selecting fires over the heatmap."""
        url = _box2d_dashboard_url(server_port, "_cube_box2d_live")
        bodies = _capture_updates(page)
        page.goto(url)
        _wait_for_init(page, "plotly")
        _enter_select_mode(page)

        hist_before = _target_y(page)  # hist(c) is #fv-plot-1
        bar_before = _bar_xy(page, "#fv-plot-2")
        assert len(hist_before) > 0
        coords = _box2d_drag_coords(page)
        n_before = len(bodies)

        page.mouse.move(coords["x1"], coords["y1"])
        page.mouse.down()
        page.mouse.move(
            (coords["x1"] + coords["x2"]) / 2,
            (coords["y1"] + coords["y2"]) / 2,
            steps=8,
        )
        # Mid-drag: the hist target re-renders locally from the rectangle slice.
        page.wait_for_function(
            """(yBefore) => {
                const gd = document.querySelector('#fv-plot-1');
                const ys = Array.from((gd.data && gd.data[0] && gd.data[0].y) || []);
                return ys.length > 0 && JSON.stringify(ys) !== JSON.stringify(yBefore);
            }""",
            arg=hist_before,
            timeout=10_000,
        )
        mid_types = [b.get("event", {}).get("type") for b in bodies[n_before:]]
        assert mid_types == ["cube_request"], mid_types
        # The 2-D gesture snaps BOTH axes (lastBins carries x AND y).
        last = page.evaluate("""() => {
            const g = Object.values(_fvCubeGestures)[0];
            return g && g.lastBins ? g.lastBins : null;
        }""")
        assert last is not None and "x" in last and "y" in last

        # The bar target also live-updated (different from its unfiltered shape).
        page.mouse.move(coords["x2"], coords["y2"], steps=8)
        page.wait_for_timeout(300)
        bar_now = _bar_xy(page, "#fv-plot-2")
        assert bar_now["y"] != bar_before["y"]

        page.mouse.up()
        page.wait_for_timeout(300)

    def test_box2d_commit_skips_post_with_two_clauses(
        self, page: Page, server_port: int
    ):
        """The snapped 2-clause commit skips the POST (all targets cube-served)
        and stores ONE predicate with TWO closed='left' clauses (x and y),
        snapped to the cube grid; the hist target matches the cube reference."""
        df = _box2d_df()
        url = _box2d_dashboard_url(server_port, "_cube_box2d_commit")
        bodies = _capture_updates(page)
        page.goto(url)
        _wait_for_init(page, "plotly")
        _enter_select_mode(page)

        hist_before = _target_y(page)
        coords = _box2d_drag_coords(page)
        n_before = len(bodies)

        page.mouse.move(coords["x1"], coords["y1"])
        page.mouse.down()
        page.mouse.move(
            (coords["x1"] + coords["x2"]) / 2,
            (coords["y1"] + coords["y2"]) / 2,
            steps=8,
        )
        page.wait_for_function(
            """(yBefore) => {
                const gd = document.querySelector('#fv-plot-1');
                const ys = Array.from((gd.data && gd.data[0] && gd.data[0].y) || []);
                return ys.length > 0 && JSON.stringify(ys) !== JSON.stringify(yBefore);
            }""",
            arg=hist_before,
            timeout=10_000,
        )
        page.mouse.move(coords["x2"], coords["y2"], steps=8)
        page.wait_for_timeout(300)
        page.mouse.up()
        page.wait_for_timeout(800)

        # Exactly one cube_request, no selection POST (conditional commit).
        types = [b.get("event", {}).get("type") for b in bodies[n_before:]]
        assert types == ["cube_request"], types

        sels = page.evaluate("DASHBOARD_SPEC.state.selections")
        assert len(sels) == 1
        clauses = sels[0]["predicates"][0]["clauses"]
        assert len(clauses) == 2
        by_col = {c["column"]: c for c in clauses}
        assert set(by_col) == {"a", "b"}
        for c in clauses:
            assert c.get("closed") == "left"
        ex = tuple(by_col["a"]["range"])
        ey = tuple(by_col["b"]["range"])

        # The edges lie on the per-axis P=128 grid over the full domains.
        a_lo, a_hi = df["a"].min(), df["a"].max()
        b_lo, b_hi = df["b"].min(), df["b"].max()
        for (lo, hi), (dlo, dhi) in ((ex, (a_lo, a_hi)), (ey, (b_lo, b_hi))):
            span = dhi - dlo
            klo = round((lo - dlo) / span * _BOX2D_P)
            khi = round((hi - dlo) / span * _BOX2D_P)
            assert lo == pytest.approx(dlo + klo * span / _BOX2D_P, abs=1e-9)
            assert hi == pytest.approx(dlo + khi * span / _BOX2D_P, abs=1e-9)

        expected = _reference_box2d_hist_counts(df, ex, ey)
        assert 0 < sum(expected) < df.height
        assert _target_y(page) == expected

    def test_box2d_edit_drag_replays_live(self, page: Page, server_port: int):
        """Editing (moving) an already-committed 2-D box replays live through
        the outline-watch machinery (d556253): the hist target re-renders
        mid-move and the whole edit is a store hit (zero POSTs of any kind),
        re-committing a snapped 2-clause closed='left' predicate."""
        df = _box2d_df()
        url = _box2d_dashboard_url(server_port, "_cube_box2d_edit")
        bodies = _capture_updates(page)
        posts = _capture_posts(page)
        page.goto(url)
        _wait_for_init(page, "plotly")
        _enter_select_mode(page)

        # Gesture 1: a fresh box-select commits + warms the store.
        hist_unfiltered = _target_y(page)
        coords = _box2d_drag_coords(page)
        page.mouse.move(coords["x1"], coords["y1"])
        page.mouse.down()
        page.mouse.move(
            (coords["x1"] + coords["x2"]) / 2,
            (coords["y1"] + coords["y2"]) / 2,
            steps=8,
        )
        page.wait_for_function(
            """(yBefore) => {
                const gd = document.querySelector('#fv-plot-1');
                const ys = Array.from((gd.data && gd.data[0] && gd.data[0].y) || []);
                return ys.length > 0 && JSON.stringify(ys) !== JSON.stringify(yBefore);
            }""",
            arg=hist_unfiltered,
            timeout=10_000,
        )
        page.mouse.move(coords["x2"], coords["y2"], steps=8)
        page.wait_for_timeout(300)
        page.mouse.up()
        page.wait_for_timeout(800)
        first_sel = page.evaluate("DASHBOARD_SPEC.state.selections")
        assert len(first_sel) == 1
        y_first = _target_y(page)
        assert y_first != hist_unfiltered

        # Activate the rendered box (click on it).
        bb = page.eval_on_selector(
            "#fv-plot-0 .selectionlayer path",
            "n => { const r = n.getBoundingClientRect();"
            " return {x: (r.left + r.right) / 2, y: (r.top + r.bottom) / 2}; }",
        )
        page.mouse.click(bb["x"], bb["y"])
        page.wait_for_function(
            "() => divs[0]._fullLayout._activeSelectionIndex >= 0", timeout=5_000
        )

        # Gesture 2: drag the activated box. No plotly_selecting fires; the
        # outline watch must engage the box2d gesture from the store.
        n_bodies = len(bodies)
        n_posts = len(posts)
        page.mouse.move(bb["x"], bb["y"])
        page.mouse.down()
        page.mouse.move(
            bb["x"] + coords["width"] * 0.12,
            bb["y"] + coords["height"] * 0.1,
            steps=10,
        )
        page.wait_for_function(
            """(yBefore) => {
                const gd = document.querySelector('#fv-plot-1');
                const ys = Array.from((gd.data && gd.data[0] && gd.data[0].y) || []);
                return ys.length > 0 && JSON.stringify(ys) !== JSON.stringify(yBefore);
            }""",
            arg=y_first,
            timeout=10_000,
        )
        page.wait_for_timeout(300)
        page.mouse.up()
        page.wait_for_timeout(2_000)  # past the safety-abort window

        assert posts[n_posts:] == [], f"edit must be a store hit; got {posts[n_posts:]}"
        assert bodies[n_bodies:] == []

        # Re-committed: still one predicate, two snapped closed='left' clauses.
        sels = page.evaluate("DASHBOARD_SPEC.state.selections")
        assert len(sels) == 1
        clauses = sels[0]["predicates"][0]["clauses"]
        assert len(clauses) == 2
        by_col = {c["column"]: c for c in clauses}
        assert set(by_col) == {"a", "b"}
        assert all(c.get("closed") == "left" for c in clauses)
        assert sels != first_sel  # the box moved
        ex = tuple(by_col["a"]["range"])
        ey = tuple(by_col["b"]["range"])
        expected = _reference_box2d_hist_counts(df, ex, ey)
        assert 0 < sum(expected) < df.height
        assert _target_y(page) == expected

    def test_box2d_live_brush_off_is_legacy(self, page: Page, server_port: int):
        """live_brush="off": no cube_request, one selection POST on mouseup,
        a legacy two-clause (unsnapped) predicate, no mid-drag updates."""
        url = _box2d_dashboard_url(server_port, "_cube_box2d_off", "off")
        bodies = _capture_updates(page)
        page.goto(url)
        _wait_for_init(page, "plotly")
        _enter_select_mode(page)

        coords = _box2d_drag_coords(page)
        n_before = len(bodies)
        page.mouse.move(coords["x1"], coords["y1"])
        page.mouse.down()
        page.mouse.move(coords["x2"], coords["y2"], steps=10)
        page.wait_for_timeout(400)
        # No mid-drag local update (no gesture at all).
        assert [b.get("event", {}).get("type") for b in bodies[n_before:]] == []
        page.mouse.up()
        page.wait_for_timeout(600)

        types = [b.get("event", {}).get("type") for b in bodies[n_before:]]
        assert types == ["selection"], types
        sels = page.evaluate("DASHBOARD_SPEC.state.selections")
        clauses = sels[0]["predicates"][0]["clauses"]
        assert len(clauses) == 2
        # Legacy clauses carry no snapped closed="left".
        assert all("closed" not in c or c.get("closed") != "left" for c in clauses)


# ---------------------------------------------------------------------------
# hist2d TARGET (step 14 / contract K)
# ---------------------------------------------------------------------------
#
# A 1-D hist source brushing axis `a` live-updates a hist2d HEATMAP target over
# (x, y) by cube slicing (no POST). The hist2d target is filtered_only and
# served only for the full-data (unzoomed) case; its z-matrix is restyled each
# frame from the sliced cube cells, mirroring Histogram2D._to_update.

_H2T_NB_X = 8
_H2T_NB_Y = 6


def _hist2d_target_df() -> pl.DataFrame:
    """Brush axis ``a`` plus a 2-D ``(x, y)`` field and a numeric ``z``. ``x``
    and ``y`` co-vary with ``a`` so a brush on ``a`` visibly reshapes the
    heatmap. Some points sit exactly on the bin edges (span-eps coverage)."""
    n = 3_000
    x_lo, x_hi = 0.0, 80.0
    y_lo, y_hi = -30.0, 30.0
    a = [((i * 37) % 1000) / 10 for i in range(n)]
    x = [x_lo + (a[i] / 100.0) * (x_hi - x_lo) for i in range(n)]
    y = [y_lo + (((i * 17) % 100) / 100.0) * (y_hi - y_lo) for i in range(n)]
    step_x = (x_hi - x_lo) / _H2T_NB_X
    step_y = (y_hi - y_lo) / _H2T_NB_Y
    for k in range(_H2T_NB_X + 1):
        x[k] = x_lo + k * step_x
    for k in range(_H2T_NB_Y + 1):
        y[k] = y_lo + k * step_y
    z = [float((i % 7) + 1) for i in range(n)]
    return pl.DataFrame({"a": a, "x": x, "y": y, "z": z})


def _hist2d_target_dashboard_url(
    port: int,
    source_name: str,
    *,
    histfunc: str | None = None,
    z: str | None = None,
    live_brush: str = "auto",
    cross_filter_mode: str = "update",
    source_trace: str = "histogram",
) -> str:
    """Source hist/line(a) + hist2d(x, y) target."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import LayoutSpec, encode_spec

    df = _hist2d_target_df()
    if source_trace == "line":
        df = df.sort("a")
    register_source(source_name, df, cache=True)

    dash = Dashboard(df)
    source = dash.add_figure(title="Source")
    if source_trace == "line":
        source.add_line(x="a", y="z", n_points=500)
    else:
        source.add_histogram(x="a", bins=_SRC_BINS)
    dash.add_figure(title="Hist2D").add_histogram2d(
        x="x", y="y", x_bins=_H2T_NB_X, y_bins=_H2T_NB_Y, z=z, histfunc=histfunc
    )
    spec = dash.to_spec(source_name=source_name, layout=LayoutSpec(draggable=False))
    spec.client_state.live_brush = live_brush
    spec.state.cross_filter_mode = cross_filter_mode
    return f"http://127.0.0.1:{port}/view?spec={encode_spec(spec)}&renderer=plotly"


def _hist2d_z(page: Page, selector: str = "#fv-plot-1") -> list:
    return page.eval_on_selector(
        selector,
        "gd => (gd.data[0] && gd.data[0].z) "
        "? gd.data[0].z.map(r => Array.from(r)) : []",
    )


def _reference_hist2d_z(
    df: pl.DataFrame,
    edge_lo: float,
    edge_hi: float,
    *,
    histfunc: str | None,
    z: str | None,
) -> list:
    """The legacy ``Histogram2D._to_update`` z over the committed closed='left'
    brush, binned to the full-data range (== the client cube reslice)."""
    from flexviz.trace.hist2d import Histogram2D

    x_lo, x_hi = float(df["x"].min()), float(df["x"].max())
    y_lo, y_hi = float(df["y"].min()), float(df["y"].max())
    trace = Histogram2D(
        x="x", y="y", x_bins=_H2T_NB_X, y_bins=_H2T_NB_Y, z=z, histfunc=histfunc
    )
    trace.uid = "u"
    agg = trace.get_aggregation_spec({"x": (x_lo, x_hi), "y": (y_lo, y_hi)}, df.schema)
    sub = df.filter((pl.col("a") >= edge_lo) & (pl.col("a") < edge_hi))
    return trace._to_update(sub.select(agg.expr)).updates["z"]


class TestHist2dTargetCube:
    def test_overlay_initial_load_renders_unfiltered_hist2d_target(
        self, page: Page, server_port: int
    ):
        """A saved Overlay dashboard still renders filtered-only traces."""
        url = _hist2d_target_dashboard_url(
            server_port,
            "_cube_browser_h2d_overlay_init",
            cross_filter_mode="overlay",
        )
        page.goto(url)
        _wait_for_init(page, "plotly")

        z = _hist2d_z(page)
        assert len(z) == _H2T_NB_Y and len(z[0]) == _H2T_NB_X

    def test_overlay_reset_restores_unfiltered_hist2d_target(
        self, page: Page, server_port: int
    ):
        """Reset restores a filtered-only cube target in overlay mode."""
        url = _hist2d_target_dashboard_url(
            server_port,
            "_cube_browser_h2d_overlay_reset",
            source_trace="line",
        )
        bodies = _capture_updates(page)
        page.goto(url)
        _wait_for_init(page, "plotly")

        z_unfiltered = _hist2d_z(page)
        page.click("#fv-btn-cfmode")
        page.wait_for_timeout(500)
        _enter_select_mode(page)

        x1, x2, y, _width = _drag_coords(page)
        page.mouse.move(x1, y)
        page.mouse.down()
        page.mouse.move((x1 + x2) / 2, y, steps=8)
        page.wait_for_function(
            """(before) => {
                const gd = document.querySelector('#fv-plot-1');
                const fg = (gd.data || []).find(
                    t => t.uid && t.uid.endsWith('__fv_layer_fg')
                );
                return fg && fg.z && JSON.stringify(fg.z) !== JSON.stringify(before);
            }""",
            arg=z_unfiltered,
            timeout=10_000,
        )
        page.mouse.move(x2, y, steps=8)
        page.wait_for_timeout(300)
        page.mouse.up()
        page.wait_for_timeout(800)
        assert len(page.evaluate("DASHBOARD_SPEC.state.selections")) == 1

        n_before = len(bodies)
        page.click("#fv-btn-reset")
        page.wait_for_timeout(800)

        assert bodies[n_before:] == [], "reset on a cached source must not POST"
        assert page.evaluate("DASHBOARD_SPEC.state.selections") == []
        assert _hist2d_z(page) == z_unfiltered

    def test_hist_source_live_updates_hist2d_target(self, page: Page, server_port: int):
        """A hist-source brush live-updates a hist2d HEATMAP target: the ``z``
        matrix is restyled each frame from a local cube slice with exactly ONE
        cube_request and ZERO POSTs during the drag. The mid-drag matrix matches
        the cube-finalize reference (contract K)."""
        df = _hist2d_target_df()
        url = _hist2d_target_dashboard_url(server_port, "_cube_browser_h2d_live")
        bodies = _capture_updates(page)
        page.goto(url)
        _wait_for_init(page, "plotly")
        _enter_select_mode(page)

        z_before = _hist2d_z(page)
        assert len(z_before) == _H2T_NB_Y and len(z_before[0]) == _H2T_NB_X

        x1, x2, y, _width = _drag_coords(page)
        n_before = len(bodies)
        page.mouse.move(x1, y)
        page.mouse.down()
        page.mouse.move((x1 + x2) / 2, y, steps=8)
        page.wait_for_function(
            """(before) => {
                const gd = document.querySelector('#fv-plot-1');
                const z = (gd.data[0] && gd.data[0].z)
                    ? gd.data[0].z.map(r => Array.from(r)) : [];
                return z.length > 0 && JSON.stringify(z) !== JSON.stringify(before);
            }""",
            arg=z_before,
            timeout=10_000,
        )
        mid_types = [b.get("event", {}).get("type") for b in bodies[n_before:]]
        assert mid_types == ["cube_request"], mid_types

        page.mouse.move(x2, y, steps=8)
        page.wait_for_timeout(300)

        live_z = _hist2d_z(page)
        edge_lo, edge_hi = _snap_edges_from_gesture(page, df)
        expected = _reference_hist2d_z(df, edge_lo, edge_hi, histfunc=None, z=None)
        _assert_z_close(live_z, expected)

        page.mouse.up()
        page.wait_for_timeout(300)

    def test_hist_source_live_updates_mean_hist2d_target(
        self, page: Page, server_port: int
    ):
        """The histfunc=mean variant: a reduce hist2d target live-updates from
        the cube (sum+count partials → mean per cell), parity vs the legacy
        ``Histogram2D._to_update`` mean z over the snapped brush."""
        df = _hist2d_target_df()
        url = _hist2d_target_dashboard_url(
            server_port, "_cube_browser_h2d_mean", histfunc="mean", z="z"
        )
        bodies = _capture_updates(page)
        page.goto(url)
        _wait_for_init(page, "plotly")
        _enter_select_mode(page)

        z_before = _hist2d_z(page)
        x1, x2, y, _width = _drag_coords(page)
        n_before = len(bodies)
        page.mouse.move(x1, y)
        page.mouse.down()
        page.mouse.move((x1 + x2) / 2, y, steps=8)
        page.wait_for_function(
            """(before) => {
                const gd = document.querySelector('#fv-plot-1');
                const z = (gd.data[0] && gd.data[0].z)
                    ? gd.data[0].z.map(r => Array.from(r)) : [];
                return z.length > 0 && JSON.stringify(z) !== JSON.stringify(before);
            }""",
            arg=z_before,
            timeout=10_000,
        )
        mid_types = [b.get("event", {}).get("type") for b in bodies[n_before:]]
        assert mid_types == ["cube_request"], mid_types

        page.mouse.move(x2, y, steps=8)
        page.wait_for_timeout(300)
        live_z = _hist2d_z(page)
        edge_lo, edge_hi = _snap_edges_from_gesture(page, df)
        expected = _reference_hist2d_z(df, edge_lo, edge_hi, histfunc="mean", z="z")
        _assert_z_close(live_z, expected)
        page.mouse.up()
        page.wait_for_timeout(300)

    def test_hist2d_target_conditional_commit_parity(
        self, page: Page, server_port: int
    ):
        """hist2d is the only target and is skipPost-eligible: the commit is
        LOCAL (one cube_request, zero further POSTs) and the committed z stays
        the cube-finalized matrix, equal to the reference for the snapped
        brush (§8.2 parity)."""
        df = _hist2d_target_df()
        url = _hist2d_target_dashboard_url(server_port, "_cube_browser_h2d_commit")
        bodies = _capture_updates(page)
        page.goto(url)
        _wait_for_init(page, "plotly")
        _enter_select_mode(page)

        z_before = _hist2d_z(page)
        x1, x2, y, _width = _drag_coords(page)
        n_before = len(bodies)
        page.mouse.move(x1, y)
        page.mouse.down()
        page.mouse.move((x1 + x2) / 2, y, steps=8)
        page.wait_for_function(
            """(before) => {
                const gd = document.querySelector('#fv-plot-1');
                const z = (gd.data[0] && gd.data[0].z)
                    ? gd.data[0].z.map(r => Array.from(r)) : [];
                return z.length > 0 && JSON.stringify(z) !== JSON.stringify(before);
            }""",
            arg=z_before,
            timeout=10_000,
        )
        page.mouse.move(x2, y, steps=8)
        page.wait_for_timeout(300)
        edge_lo, edge_hi = _snap_edges_from_gesture(page, df)
        page.mouse.up()
        page.wait_for_timeout(1_200)

        gesture_bodies = bodies[n_before:]
        cube_requests = [
            b
            for b in gesture_bodies
            if b.get("event", {}).get("type") == "cube_request"
        ]
        assert len(cube_requests) == 1, [
            b.get("event", {}).get("type") for b in gesture_bodies
        ]
        others = [b for b in gesture_bodies if b not in cube_requests]
        assert others == [], [b.get("event", {}).get("type") for b in others]

        sels = page.evaluate("DASHBOARD_SPEC.state.selections")
        assert len(sels) == 1
        clause = sels[0]["predicates"][0]["clauses"][0]
        assert clause["column"] == "a" and clause.get("closed") == "left"

        expected = _reference_hist2d_z(df, edge_lo, edge_hi, histfunc=None, z=None)
        _assert_z_close(_hist2d_z(page), expected)


# ---------------------------------------------------------------------------
# Treemap target (Step 15): finalize-then-sum hierarchy rollup + fvUrlQuote
# ---------------------------------------------------------------------------


def _treemap_target_df() -> pl.DataFrame:
    """Brush axis ``a`` plus a 2-level treemap field (cat, sub) and a numeric
    ``val``. ``cat`` includes a value needing url-quoting (``"a b"``) so the
    rollup ids exercise ``fvUrlQuote``. ``(cat, sub)`` partitions every row into
    a non-empty leaf (no all-null mean leaves — the cube omits count==0 cells
    while a legacy recompute would emit them as null, a pre-existing bar/pie
    shared edge, kept out of scope here). ``cat``/``sub`` co-vary with ``a`` so
    a brush visibly reshapes the node values."""
    n = 3_000
    cats = ["alpha", "a b", "gamma"]
    subs = ["s1", "s2", "s3", "s4"]
    a = [((i * 37) % 1000) / 10 for i in range(n)]
    cat = [cats[i % 3] for i in range(n)]
    sub = [subs[(i * 7) % 4] for i in range(n)]
    val = [float((i * 13) % 50) + 1.0 for i in range(n)]
    return pl.DataFrame({"a": a, "cat": cat, "sub": sub, "val": val})


def _treemap_target_dashboard_url(
    port: int,
    source_name: str,
    *,
    agg: str = "sum",
    values: str | None = "val",
    live_brush: str = "auto",
) -> str:
    """Source hist(a) + treemap(path=[cat, sub]) target."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import LayoutSpec, encode_spec

    df = _treemap_target_df()
    register_source(source_name, df, cache=True)

    dash = Dashboard(df)
    dash.add_figure(title="Source").add_histogram(x="a", bins=_SRC_BINS)
    dash.add_figure(title="Treemap").add_treemap(
        path=["cat", "sub"], values=values, agg=agg
    )
    spec = dash.to_spec(source_name=source_name, layout=LayoutSpec(draggable=False))
    spec.client_state.live_brush = live_brush
    return f"http://127.0.0.1:{port}/view?spec={encode_spec(spec)}&renderer=plotly"


def _treemap_data(page: Page, selector: str = "#fv-plot-1") -> dict:
    """Read the rendered treemap trace's ids/parents/labels/values."""
    return page.eval_on_selector(
        selector,
        "gd => { const d = gd.data[0] || {}; return {"
        "ids: Array.from(d.ids || []),"
        "parents: Array.from(d.parents || []),"
        "labels: Array.from(d.labels || []),"
        "values: Array.from(d.values || []),"
        "}; }",
    )


def _reference_treemap_delta(
    df: pl.DataFrame,
    edge_lo: float,
    edge_hi: float,
    *,
    agg: str,
    values: str | None,
) -> dict:
    """The server ``TreeMap._to_grouped_update`` delta over the committed
    ``closed="left"`` brush — the byte-identical reference for the client cube
    rollup."""
    from flexviz.trace.treemap import TreeMap

    trace = TreeMap(path=["cat", "sub"], values=values, agg=agg)
    trace.uid = "u"
    ag = trace.get_aggregation_spec({}, df.schema)
    sub = (
        df.lazy()
        .filter((pl.col("a") >= edge_lo) & (pl.col("a") < edge_hi))
        .group_by(list(ag.group_cols))
        .agg(*ag.agg_exprs)
        .sort(list(ag.sort_cols))
        .collect()
    )
    return trace._to_grouped_update(sub).updates


class TestTreeMapTargetCube:
    def test_fv_url_quote_matches_python_quote(self, page: Page, server_port: int):
        """fvUrlQuote must reproduce Python urllib.parse.quote(s, safe="")
        byte-for-byte over a tricky char set (incl. the chars encodeURIComponent
        leaves bare: ! * ' ( ), and a unicode char)."""
        import urllib.parse

        url = _treemap_target_dashboard_url(server_port, "_cube_browser_tm_urlquote")
        page.goto(url)
        _wait_for_init(page, "plotly")

        cases = [" ", "/", "!", "*", "'", "(", ")", "~", "%", "é", "&", "=", "a b"]
        for ch in cases:
            expected = urllib.parse.quote(ch, safe="")
            got = page.evaluate("s => fvUrlQuote(s)", ch)
            assert got == expected, f"fvUrlQuote({ch!r}) = {got!r} != {expected!r}"

    def test_byte_identical_rollup_vs_server(self, page: Page, server_port: int):
        """A source brush builds a treemap delta client-side; its
        ids/parents/labels/values must be byte-identical to the server's
        ``_to_grouped_update`` delta over the same snapped brush (the plan's
        unit-style fixture). One ``cat`` value needs url-quoting so fvUrlQuote
        is exercised in the ids."""
        df = _treemap_target_df()
        url = _treemap_target_dashboard_url(server_port, "_cube_browser_tm_rollup")
        page.goto(url)
        _wait_for_init(page, "plotly")
        _enter_select_mode(page)

        before = _treemap_data(page)
        x1, x2, y, _width = _drag_coords(page)
        page.mouse.move(x1, y)
        page.mouse.down()
        page.mouse.move((x1 + x2) / 2, y, steps=8)
        page.wait_for_function(
            """(before) => {
                const gd = document.querySelector('#fv-plot-1');
                const vs = Array.from((gd.data && gd.data[0] && gd.data[0].values) || []);
                return vs.length > 0 && JSON.stringify(vs) !== JSON.stringify(before);
            }""",
            arg=before["values"],
            timeout=10_000,
        )
        page.mouse.move(x2, y, steps=8)
        page.wait_for_timeout(300)

        live = _treemap_data(page)
        edge_lo, edge_hi = _snap_edges_from_gesture(page, df)
        page.mouse.up()
        page.wait_for_timeout(300)

        ref = _reference_treemap_delta(df, edge_lo, edge_hi, agg="sum", values="val")
        # An id needing url-quoting must actually appear (fvUrlQuote exercised).
        assert any("a%20b" in i for i in live["ids"]), live["ids"]
        assert live["ids"] == ref["ids"]
        assert live["parents"] == ref["parents"]
        assert live["labels"] == ref["labels"]
        assert len(live["values"]) == len(ref["values"])
        for got, want in zip(live["values"], ref["values"]):
            assert got == pytest.approx(want, rel=1e-9, abs=1e-9), (got, want)

    def test_source_brush_live_updates_treemap_and_commit_parity(
        self, page: Page, server_port: int
    ):
        """A source brush live-updates the treemap node values (they track the
        brush — shrink as the window narrows), with exactly ONE cube_request and
        ZERO further POSTs; the conditional commit is LOCAL (skipPost) and the
        committed hierarchy stays the cube-finalized one, byte-identical to the
        server delta over the snapped brush (no visual jump on commit)."""
        df = _treemap_target_df()
        url = _treemap_target_dashboard_url(server_port, "_cube_browser_tm_live")
        bodies = _capture_updates(page)
        page.goto(url)
        _wait_for_init(page, "plotly")
        _enter_select_mode(page)

        root_before = _treemap_data(page)["values"][0]
        x1, x2, y, _width = _drag_coords(page)
        n_before = len(bodies)
        page.mouse.move(x1, y)
        page.mouse.down()
        page.mouse.move((x1 + x2) / 2, y, steps=8)
        page.wait_for_function(
            """(before) => {
                const gd = document.querySelector('#fv-plot-1');
                const vs = Array.from((gd.data && gd.data[0] && gd.data[0].values) || []);
                return vs.length > 0 && vs[0] !== before;
            }""",
            arg=root_before,
            timeout=10_000,
        )
        mid_types = [b.get("event", {}).get("type") for b in bodies[n_before:]]
        assert mid_types == ["cube_request"], mid_types

        page.mouse.move(x2, y, steps=8)
        page.wait_for_timeout(300)
        # A sub-range brush selects fewer rows ⇒ root value shrinks.
        root_live = _treemap_data(page)["values"][0]
        assert root_live < root_before

        live = _treemap_data(page)
        edge_lo, edge_hi = _snap_edges_from_gesture(page, df)
        page.mouse.up()
        page.wait_for_timeout(1_200)

        gesture_bodies = bodies[n_before:]
        cube_requests = [
            b
            for b in gesture_bodies
            if b.get("event", {}).get("type") == "cube_request"
        ]
        assert len(cube_requests) == 1, [
            b.get("event", {}).get("type") for b in gesture_bodies
        ]
        others = [b for b in gesture_bodies if b not in cube_requests]
        assert others == [], [b.get("event", {}).get("type") for b in others]

        sels = page.evaluate("DASHBOARD_SPEC.state.selections")
        assert len(sels) == 1
        clause = sels[0]["predicates"][0]["clauses"][0]
        assert clause["column"] == "a" and clause.get("closed") == "left"

        # Commit ≡ live (no visual jump): committed hierarchy matches the live
        # one and the server reference over the snapped brush.
        committed = _treemap_data(page)
        assert committed["ids"] == live["ids"]
        assert committed["values"] == pytest.approx(live["values"], rel=1e-9, abs=1e-9)
        ref = _reference_treemap_delta(df, edge_lo, edge_hi, agg="sum", values="val")
        assert committed["ids"] == ref["ids"]
        assert committed["labels"] == ref["labels"]
        for got, want in zip(committed["values"], ref["values"]):
            assert got == pytest.approx(want, rel=1e-9, abs=1e-9)

    def test_mean_agg_treemap_live_and_commit_parity(
        self, page: Page, server_port: int
    ):
        """A mean-agg treemap variant: the cube ships sum+count partials; the
        client finalizes leaf means then SUMS them up each level (parents sum
        the leaf means, matching _to_grouped_update). Live + committed
        hierarchy byte-identical to the server delta over the snapped brush."""
        df = _treemap_target_df()
        url = _treemap_target_dashboard_url(
            server_port, "_cube_browser_tm_mean", agg="mean", values="val"
        )
        bodies = _capture_updates(page)
        page.goto(url)
        _wait_for_init(page, "plotly")
        _enter_select_mode(page)

        before = _treemap_data(page)["values"]
        x1, x2, y, _width = _drag_coords(page)
        n_before = len(bodies)
        page.mouse.move(x1, y)
        page.mouse.down()
        page.mouse.move((x1 + x2) / 2, y, steps=8)
        page.wait_for_function(
            """(before) => {
                const gd = document.querySelector('#fv-plot-1');
                const vs = Array.from((gd.data && gd.data[0] && gd.data[0].values) || []);
                return vs.length > 0 && JSON.stringify(vs) !== JSON.stringify(before);
            }""",
            arg=before,
            timeout=10_000,
        )
        mid_types = [b.get("event", {}).get("type") for b in bodies[n_before:]]
        assert mid_types == ["cube_request"], mid_types

        page.mouse.move(x2, y, steps=8)
        page.wait_for_timeout(300)
        edge_lo, edge_hi = _snap_edges_from_gesture(page, df)
        page.mouse.up()
        page.wait_for_timeout(300)

        committed = _treemap_data(page)
        ref = _reference_treemap_delta(df, edge_lo, edge_hi, agg="mean", values="val")
        assert committed["ids"] == ref["ids"]
        assert committed["parents"] == ref["parents"]
        assert committed["labels"] == ref["labels"]
        assert len(committed["values"]) == len(ref["values"])
        for got, want in zip(committed["values"], ref["values"]):
            assert got == pytest.approx(want, rel=1e-9, abs=1e-9), (got, want)

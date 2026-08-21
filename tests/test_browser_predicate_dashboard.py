"""Comprehensive browser test exercising the predicate-based selection wire format.

Tests one large dashboard with every meaningful permutation of:
  - single vs multi-column ``labels``
  - single vs multi-column ``group_by``
  - source: treemap path / pie / bar / line range brush
  - target: bar / pie / line / box / hist
  - hidden trace via ``legendonly`` + subsequent cross-filter

Layout (6 figures):
  fig0  line   x=ts, y=val, group_by=["source","country"]   [target + range source]
  fig1  bar    labels=["source","country"], values=val      [target + cat source]
  fig2  bar    labels="country", group_by=["source"]        [target]
  fig3  pie    labels=["source","country"], values=val      [source]
  fig4  treemap path=["source","country"], values=val       [source]
  fig5  hist   x="val", group_by="source"                   [target]
"""

from __future__ import annotations

import polars as pl
import pytest
from playwright.sync_api import Page

from tests.test_browser import _wait_for_init


def _build_predicate_dashboard_url(port: int) -> str:
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import LayoutSpec, encode_spec

    countries = ["NL", "BE", "DE"]
    sources = ["solar", "wind"]
    rows = []
    for ts in range(80):
        for c in countries:
            for s in sources:
                rows.append(
                    {
                        "ts": ts,
                        "val": float((ts + len(c) + len(s)) % 50),
                        "country": c,
                        "source": s,
                    }
                )
    df = pl.DataFrame(rows)
    register_source("_predicate_dashboard", df)

    dash = Dashboard(df)
    dash.add_figure(title="line").add_line(
        x="ts", y="val", group_by=["source", "country"], n_points=80
    )
    dash.add_figure(title="bar-multi-label").add_bar(
        labels=["source", "country"], values="val", agg="sum"
    )
    dash.add_figure(title="bar-multi-group").add_bar(
        labels="country", values="val", agg="sum", group_by=["source"]
    )
    dash.add_figure(title="pie-multi-label").add_pie(
        labels=["source", "country"], values="val", agg="sum"
    )
    dash.add_figure(title="treemap").add_treemap(
        path=["source", "country"], values="val", agg="sum"
    )
    dash.add_figure(title="hist").add_histogram(x="val", bins=10, group_by="source")

    spec = dash.to_spec(
        source_name="_predicate_dashboard",
        layout=LayoutSpec(draggable=False),
    )
    encoded = encode_spec(spec)
    return f"http://127.0.0.1:{port}/view?spec={encoded}&renderer=plotly"


def _wait_for_selection_count(page: Page, expected: int) -> None:
    page.wait_for_function(
        f"() => (DASHBOARD_SPEC.state.selections || []).length === {expected}"
    )


def _selection_predicates(page: Page) -> list[dict]:
    return page.evaluate(
        "DASHBOARD_SPEC.state.selections.map(s => "
        "({uid: s.source_figure_uid, preds: s.predicates}))"
    )


def _trace_data(page: Page, fig_idx: int, key: str) -> list:
    return page.eval_on_selector(
        f"#fv-plot-{fig_idx}",
        "(gd, k) => (gd.data && gd.data[0] && gd.data[0][k]) || []",
        key,
    )


def _click_slice_by_id(page: Page, node_id: str, fig_sel: str = "#fv-plot-4") -> dict:
    """Scroll a treemap/sunburst slice into view by its id, then click its centre.

    The 6-figure dashboard is taller than the 720px viewport, so slices for
    lower figures (e.g. the treemap at fig 4) render below the fold. Clicking
    the raw ``getBoundingClientRect`` coordinates would land off-screen and miss
    the slice entirely, so we scroll it into view and recompute the rect first.
    Re-finds the node on every call so it stays correct after a re-render.
    """
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
            return { x: (r.left+r.right)/2, y: (r.top+r.bottom)/2 };
        }""",
        node_id,
    )
    assert target is not None, f"treemap slice {node_id!r} not found"
    page.mouse.click(target["x"], target["y"])
    return target


@pytest.mark.browser
class TestPredicateDashboard:
    """Single dashboard exercising every predicate cross-filter scenario."""

    def test_init_renders_all_six_figures(self, page: Page, server_port: int):
        page.goto(_build_predicate_dashboard_url(server_port))
        _wait_for_init(page, "plotly")
        for i in range(6):
            page.wait_for_function(
                f"() => document.querySelector('#fv-plot-{i}')?.data?.length > 0"
            )

    def test_treemap_leaf_click_filters_all_targets(self, page: Page, server_port: int):
        """Click leaf 'solar/NL' on treemap (fig 4) → verify line, both bars, pie, hist
        all show only solar+NL data."""
        page.goto(_build_predicate_dashboard_url(server_port))
        _wait_for_init(page, "plotly")
        page.wait_for_function(
            "() => document.querySelector('#fv-plot-4')?.data?.[0]?.ids?.includes('root/solar/NL')"
        )

        _click_slice_by_id(page, "root/solar/NL")
        _wait_for_selection_count(page, 1)
        page.wait_for_timeout(800)

        sels = _selection_predicates(page)
        assert sels[0]["uid"] == page.evaluate("DASHBOARD_SPEC.figures[4].uid")
        clauses = sels[0]["preds"][0]["clauses"]
        assert {c["column"] for c in clauses} == {"source", "country"}
        assert {c["values"][0] for c in clauses} == {"solar", "NL"}

        # bar-multi-label (fig 1) — only one composite bar visible
        bar_x = _trace_data(page, 1, "x")
        assert bar_x == ['["solar","NL"]']

        # bar-multi-group (fig 2) — only one country
        bar2_x = page.eval_on_selector(
            "#fv-plot-2",
            "gd => (gd.data || []).map(t => t.x).flat()",
        )
        assert bar2_x == ["NL"]

        # pie-multi-label (fig 3) — one slice
        pie_labels = _trace_data(page, 3, "labels")
        assert pie_labels == ['["solar","NL"]']

    def test_pie_multi_label_click_filters_targets(self, page: Page, server_port: int):
        page.goto(_build_predicate_dashboard_url(server_port))
        _wait_for_init(page, "plotly")
        page.wait_for_function(
            "() => document.querySelector('#fv-plot-3')?.data?.[0]?.labels?.length > 1"
        )
        target = page.eval_on_selector(
            "#fv-plot-3",
            """(gd, label) => {
                const nodes = Array.from(gd.querySelectorAll('g.slice'));
                const node = nodes.find(n => {
                    const d = n.__data__ || {};
                    return (d.label || (d.data && d.data.label)) === label;
                });
                if (!node) return null;
                const r = node.getBoundingClientRect();
                return { x: (r.left+r.right)/2, y: (r.top+r.bottom)/2 };
            }""",
            '["solar","NL"]',
        )
        assert target is not None
        page.mouse.click(target["x"], target["y"])
        _wait_for_selection_count(page, 1)
        page.wait_for_timeout(800)

        sels = _selection_predicates(page)
        clauses = sels[0]["preds"][0]["clauses"]
        cols = {c["column"]: c["values"][0] for c in clauses}
        assert cols == {"source": "solar", "country": "NL"}

        # bar-multi-group (fig 2) — only NL
        bar2_x = _trace_data(page, 2, "x")
        assert bar2_x == ["NL"]

    def test_bar_single_label_brush_filters_treemap(self, page: Page, server_port: int):
        # Programmatically post a categorical predicate on country=NL.
        page.goto(_build_predicate_dashboard_url(server_port))
        _wait_for_init(page, "plotly")
        page.wait_for_function(
            "() => (document.querySelector('#fv-plot-2')?.data || []).length > 0"
        )
        page.evaluate("""async () => {
                const fig = DASHBOARD_SPEC.figures[2];
                const sel = {
                    source_figure_uid: fig.uid,
                    predicates: [{
                        clauses: [{ column: 'country', values: ['NL'] }]
                    }]
                };
                DASHBOARD_SPEC.state.selections = [sel];
                await postDashboardUpdate({
                    type: 'selection', axis_ranges: {},
                    selections: [sel], force_update: true, figure_uid: fig.uid
                });
            }""")
        page.wait_for_timeout(800)

        leaf_ids = page.eval_on_selector(
            "#fv-plot-4",
            "gd => (gd.data && gd.data[0] && gd.data[0].ids) || []",
        )
        assert "root/solar/NL" in leaf_ids
        assert "root/solar/BE" not in leaf_ids
        assert "root/wind/NL" in leaf_ids

    def test_line_range_brush_filters_targets(self, page: Page, server_port: int):
        page.goto(_build_predicate_dashboard_url(server_port))
        _wait_for_init(page, "plotly")
        page.wait_for_function(
            "() => (document.querySelector('#fv-plot-0')?.data || []).length > 0"
        )
        page.evaluate("""async () => {
                const fig = DASHBOARD_SPEC.figures[0];
                const sel = {
                    source_figure_uid: fig.uid,
                    predicates: [{
                        clauses: [{ column: 'ts', range: [10, 30] }]
                    }]
                };
                DASHBOARD_SPEC.state.selections = [sel];
                await postDashboardUpdate({
                    type: 'selection', axis_ranges: {},
                    selections: [sel], force_update: true, figure_uid: fig.uid
                });
            }""")
        page.wait_for_timeout(800)
        bar_y = _trace_data(page, 1, "y")
        assert all(v >= 0 for v in bar_y)

    def test_line_box_select_emits_x_only_clause(self, page: Page, server_port: int):
        """A 2-D box-select on the line (x=ts, y=val) must yield only the x (ts)
        clause — line selection is x-only, so the y range is never emitted."""
        page.goto(_build_predicate_dashboard_url(server_port))
        _wait_for_init(page, "plotly")
        page.wait_for_function(
            "() => (document.querySelector('#fv-plot-0')?.data || []).length > 0"
        )
        # Emit a Plotly box-select carrying BOTH an x and a y range.
        page.evaluate("""() => {
                const gd = document.querySelector('#fv-plot-0');
                gd.emit('plotly_selected', { range: { x: [10, 30], y: [5, 25] } });
            }""")
        _wait_for_selection_count(page, 1)
        sels = _selection_predicates(page)
        clauses = sels[0]["preds"][0]["clauses"]
        cols = {c["column"] for c in clauses}
        assert cols == {"ts"}, f"line box-select should be x-only, got {cols}"
        assert clauses[0]["range"] == [10, 30]

    def test_line_selection_box_is_full_height_band(self, page: Page, server_port: int):
        """The x-only line's persisted selection box must omit the y axis (so it
        renders as a full-height band), and a y-only edit must be a no-op that
        snaps the band back rather than cropping it."""
        page.goto(_build_predicate_dashboard_url(server_port))
        _wait_for_init(page, "plotly")
        page.wait_for_function(
            "() => (document.querySelector('#fv-plot-0')?.data || []).length > 0"
        )
        page.evaluate("""() => {
                const gd = document.querySelector('#fv-plot-0');
                gd.emit('plotly_selected', { range: { x: [10, 30], y: [5, 25] } });
            }""")
        _wait_for_selection_count(page, 1)
        # Stored box carries only x — the y bound (the drawn 5..25) is dropped.
        box = page.evaluate("DASHBOARD_SPEC.state.selections[0]._plotly_selection_box")
        assert box.get("x0") == 10 and box.get("x1") == 30
        assert box.get("y0") is None and box.get("yref") is None

        # Rendered selection spans the full y range (height >> the drawn 20).
        page.wait_for_function(
            "() => (document.querySelector('#fv-plot-0').layout.selections || []).length === 1"
        )
        height = page.eval_on_selector(
            "#fv-plot-0",
            "gd => { const s = gd.layout.selections[0]; return Math.abs(s.y1 - s.y0); }",
        )
        assert height > 30, f"band should be full-height, got {height}"

        # Simulate Plotly live-cropping the on-screen band in y, then fire the
        # resulting (no-op) selected event. The stored selection stays x-only and
        # the rendered band must snap back to full height (part 2: re-apply).
        page.evaluate("""() => {
                const gd = document.querySelector('#fv-plot-0');
                Plotly.relayout(gd, { selections: [
                    { type: 'rect', x0: 10, x1: 30, xref: 'x',
                      y0: 12, y1: 14, yref: 'y' } ] });
                gd.emit('plotly_selected', { range: { x: [10, 30], y: [12, 14] } });
            }""")
        page.wait_for_timeout(300)
        sels = _selection_predicates(page)
        assert len(sels) == 1
        assert {c["column"] for c in sels[0]["preds"][0]["clauses"]} == {"ts"}
        box2 = page.evaluate("DASHBOARD_SPEC.state.selections[0]._plotly_selection_box")
        assert box2.get("y0") is None
        height2 = page.eval_on_selector(
            "#fv-plot-0",
            "gd => { const s = gd.layout.selections[0]; return Math.abs(s.y1 - s.y0); }",
        )
        assert height2 > 30, f"band should snap back to full-height, got {height2}"

    def test_treemap_double_click_toggles_off(self, page: Page, server_port: int):
        page.goto(_build_predicate_dashboard_url(server_port))
        _wait_for_init(page, "plotly")
        page.wait_for_function(
            "() => document.querySelector('#fv-plot-4')?.data?.[0]?.ids?.includes('root/solar')"
        )

        _click_slice_by_id(page, "root/solar")
        _wait_for_selection_count(page, 1)
        page.wait_for_timeout(500)

        _click_slice_by_id(page, "root/solar")
        _wait_for_selection_count(page, 0)
        page.wait_for_timeout(500)

        leaves = page.eval_on_selector(
            "#fv-plot-4",
            "gd => (gd.data && gd.data[0] && gd.data[0].ids) || []",
        )
        assert "root/solar/NL" in leaves and "root/wind/NL" in leaves

    def test_repeated_select_deselect_cycle(self, page: Page, server_port: int):
        page.goto(_build_predicate_dashboard_url(server_port))
        _wait_for_init(page, "plotly")
        page.wait_for_function(
            "() => (document.querySelector('#fv-plot-2')?.data || []).length > 0"
        )

        for _ in range(3):
            page.evaluate("""async () => {
                    const fig = DASHBOARD_SPEC.figures[2];
                    const selections = [{
                        source_figure_uid: fig.uid,
                        predicates: [{ clauses: [{ column: 'country', values: ['NL'] }] }],
                    }];
                    // Mirror real selection handlers: client state is set before
                    // posting (postDashboardUpdate does not sync state from the event).
                    DASHBOARD_SPEC.state.selections = selections;
                    await postDashboardUpdate({
                        type: 'selection', axis_ranges: {},
                        selections, force_update: true, figure_uid: fig.uid
                    });
                }""")
            page.wait_for_function(
                "() => (DASHBOARD_SPEC.state.selections || []).length === 1"
            )
            page.evaluate("""async () => {
                    const fig = DASHBOARD_SPEC.figures[2];
                    DASHBOARD_SPEC.state.selections = [];
                    await postDashboardUpdate({
                        type: 'deselect', axis_ranges: {},
                        selections: [], force_update: true, figure_uid: fig.uid
                    });
                }""")
            page.wait_for_function(
                "() => (DASHBOARD_SPEC.state.selections || []).length === 0"
            )

        for i in range(6):
            data_len = page.eval_on_selector(
                f"#fv-plot-{i}", "gd => (gd.data || []).length"
            )
            assert data_len > 0, f"fig-{i} lost data after cycle"


def _build_line_over_hist2d_url(port: int) -> str:
    """One figure with a hist2d (x∧y) and a line (x-only) on the same columns."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import LayoutSpec, encode_spec

    rows = [{"ts": t, "val": float(t % 17)} for t in range(200)]
    df = pl.DataFrame(rows)
    register_source("_line_over_hist2d", df)

    dash = Dashboard(df)
    fig = dash.add_figure(title="overlay")
    fig.add_histogram2d(x="ts", y="val", x_bins=10, y_bins=10)
    fig.add_line(x="ts", y="val", n_points=200)

    spec = dash.to_spec(
        source_name="_line_over_hist2d", layout=LayoutSpec(draggable=False)
    )
    return f"http://127.0.0.1:{port}/view?spec={encode_spec(spec)}&renderer=plotly"


@pytest.mark.browser
class TestMixedRangeGeometry:
    """A figure mixing a 1-clause (x-only line) and a 2-clause (x∧y hist2d)
    range trace must not let the looser x-only predicate drop the y bound."""

    def test_box_select_keeps_tightest_xy_predicate(self, page: Page, server_port: int):
        page.goto(_build_line_over_hist2d_url(server_port))
        _wait_for_init(page, "plotly")
        page.wait_for_function(
            "() => (document.querySelector('#fv-plot-0')?.data || []).length > 1"
        )
        page.evaluate("""() => {
                const gd = document.querySelector('#fv-plot-0');
                gd.emit('plotly_selected', { range: { x: [10, 30], y: [2, 8] } });
            }""")
        _wait_for_selection_count(page, 1)
        sels = _selection_predicates(page)
        preds = sels[0]["preds"]
        # Exactly one predicate, constraining BOTH ts and val (not x-only).
        assert len(preds) == 1, f"expected one tightest predicate, got {preds}"
        cols = {c["column"] for c in preds[0]["clauses"]}
        assert cols == {"ts", "val"}, f"y bound dropped — got {cols}"


def _build_same_col_two_axes_url(port: int) -> str:
    """One figure where the same column is selectable on different axes: a line
    (x=ts) and a horizontal histogram (y=ts). Box-select must keep BOTH ranges."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import LayoutSpec, encode_spec

    df = pl.DataFrame(
        {"ts": list(range(200)), "val": [float(t % 13) for t in range(200)]}
    )
    register_source("_same_col_two_axes", df)

    dash = Dashboard(df)
    fig = dash.add_figure(title="overlay")
    fig.add_line(x="ts", y="val", n_points=200)
    fig.add_histogram(y="ts", bins=10)  # horizontal hist → selects on y=ts

    spec = dash.to_spec(
        source_name="_same_col_two_axes", layout=LayoutSpec(draggable=False)
    )
    return f"http://127.0.0.1:{port}/view?spec={encode_spec(spec)}&renderer=plotly"


@pytest.mark.browser
class TestSameColumnDifferentAxes:
    """A column selectable on x by one trace and on y by another must not have
    one range silently dropped by a column-name-only de-dupe."""

    def test_box_select_keeps_both_axis_ranges(self, page: Page, server_port: int):
        page.goto(_build_same_col_two_axes_url(server_port))
        _wait_for_init(page, "plotly")
        page.wait_for_function(
            "() => (document.querySelector('#fv-plot-0')?.data || []).length > 1"
        )
        page.evaluate("""() => {
                const gd = document.querySelector('#fv-plot-0');
                gd.emit('plotly_selected', { range: { x: [10, 30], y: [40, 60] } });
            }""")
        _wait_for_selection_count(page, 1)
        preds = _selection_predicates(page)[0]["preds"]
        # Two predicates: ts bound on x ([10,30]) and ts bound on y ([40,60]).
        ranges = sorted(
            (c["range"] for p in preds for c in p["clauses"] if c["column"] == "ts")
        )
        assert ranges == [[10, 30], [40, 60]], f"a range was dropped: {preds}"

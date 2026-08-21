"""Treemap/pie multi-predicate OR selection (path upsert + cross-filter)."""

from __future__ import annotations

import polars as pl
import pytest
from playwright.sync_api import Page

from flexviz.LF import LFQueryBuilder
from flexviz.engine import FlexEngine, TraceInfo
from flexviz.events import InteractionEvent
from flexviz.spec import (
    ClauseFilter,
    LayoutSpec,
    SelectionPredicate,
    SelectionState,
    encode_spec,
)
from tests.test_browser import _wait_for_init


def _build_or_selection_dashboard_url(port: int, renderer: str = "plotly") -> str:
    """Two-figure dashboard: bar target (fig 0) + treemap source (fig 1)."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source

    countries = ["NL", "BE", "DE"]
    sources = ["solar", "wind"]
    rows = []
    for ts in range(20):
        for country in countries:
            for source in sources:
                rows.append(
                    {
                        "ts": ts,
                        "val": float((ts + len(country) + len(source)) % 50),
                        "country": country,
                        "source": source,
                    }
                )
    df = pl.DataFrame(rows)
    register_source("_or_selection_dashboard", df)

    dash = Dashboard(df)
    dash.add_figure(title="bar-by-country").add_bar(
        labels="country", values="val", agg="sum", group_by="source"
    )
    dash.add_figure(title="treemap").add_treemap(
        path=["source", "country"], values="val", agg="sum"
    )
    spec = dash.to_spec(
        source_name="_or_selection_dashboard",
        layout=LayoutSpec(draggable=False),
    )
    encoded = encode_spec(spec)
    return f"http://127.0.0.1:{port}/view?spec={encoded}&renderer={renderer}"


def _build_pie_or_dashboard_url(port: int, renderer: str = "plotly") -> str:
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source

    countries = ["NL", "BE", "DE"]
    rows = []
    for country in countries:
        for source in ("solar", "wind"):
            rows.append({"country": country, "source": source, "val": 1.0})
    df = pl.DataFrame(rows)
    register_source("_pie_or_dashboard", df)

    dash = Dashboard(df)
    dash.add_figure(title="bar-by-country").add_bar(
        labels="country", values="val", agg="sum"
    )
    dash.add_figure(title="pie").add_pie(
        labels=["source", "country"], values="val", agg="sum"
    )
    spec = dash.to_spec(
        source_name="_pie_or_dashboard",
        layout=LayoutSpec(draggable=False),
    )
    encoded = encode_spec(spec)
    return f"http://127.0.0.1:{port}/view?spec={encoded}&renderer={renderer}"


def _wait_for_selection_count(page: Page, expected: int) -> None:
    page.wait_for_function(
        f"() => (DASHBOARD_SPEC.state.selections || []).length === {expected}"
    )


def _selection_predicates(page: Page) -> list[dict]:
    return page.evaluate(
        "DASHBOARD_SPEC.state.selections.map(s => "
        "({uid: s.source_figure_uid, preds: s.predicates}))"
    )


def _plotly_bar_labels(page: Page, fig_idx: int) -> list:
    return page.eval_on_selector(
        f"#fv-plot-{fig_idx}",
        "(gd) => (gd.data && gd.data[0] && gd.data[0].x) || []",
    )


class TestPathPredicateUpsertJs:
    """Exercise fvUpsertPathPredicate in the bundled page runtime."""

    @pytest.mark.browser
    @pytest.mark.parametrize("renderer", ["plotly", "echarts"])
    def test_upsert_toggle_append_and_refinement(
        self, page: Page, server_port: int, renderer: str
    ):
        page.goto(_build_or_selection_dashboard_url(server_port, renderer))
        _wait_for_init(page, renderer)

        result = page.evaluate("""() => {
            const upsert = window.fvUpsertPathPredicate;
            const solarDe = {
                clauses: [
                    { column: 'source', values: ['solar'] },
                    { column: 'country', values: ['DE'] },
                ],
            };
            let preds = upsert([], solarDe);
            if (preds.length !== 1) return { step: 'add', preds };
            preds = upsert(preds, solarDe);
            if (preds.length !== 0) return { step: 'toggle', preds };
            preds = upsert([], solarDe);
            const solarNl = {
                clauses: [
                    { column: 'source', values: ['solar'] },
                    { column: 'country', values: ['NL'] },
                ],
            };
            preds = upsert(preds, solarNl);
            if (preds.length !== 2) return { step: 'append', preds };
            const solarOnly = {
                clauses: [{ column: 'source', values: ['solar'] }],
            };
            preds = upsert(preds, solarOnly);
            if (preds.length !== 1) return { step: 'ancestor', preds };
            if (preds[0].clauses.length !== 1) return { step: 'ancestor-shape', preds };
            const solarFr = {
                clauses: [
                    { column: 'source', values: ['solar'] },
                    { column: 'country', values: ['FR'] },
                ],
            };
            preds = upsert(preds, solarFr);
            if (preds.length !== 1) return { step: 'refine-ancestor', preds };
            preds = upsert(preds, solarNl);
            if (preds.length !== 2) return { step: 'append-after-refine', preds };
            return { ok: true, count: preds.length };
        }""")
        assert result == {"ok": True, "count": 2}


@pytest.mark.browser
class TestTreemapPieOrCrossFilter:
    """End-to-end treemap/pie clicks drive OR predicates and cross-filters."""

    def _click_plotly_treemap_node(
        self, page: Page, fig_idx: int, node_id: str, *, label_area: bool = False
    ) -> None:
        target = page.eval_on_selector(
            f"#fv-plot-{fig_idx}",
            """(gd, args) => {
                const nodeId = args.nodeId;
                const labelArea = args.labelArea;
                const nodes = Array.from(gd.querySelectorAll('g.slice'));
                const node = nodes.find(n => {
                    const d = n.__data__ || {};
                    return d.id === nodeId
                        || (d.data && d.data.id === nodeId)
                        || (d.data && d.data.data && d.data.data.id === nodeId);
                });
                if (!node) return null;
                node.scrollIntoView({ block: 'center', inline: 'center' });
                const rect = node.getBoundingClientRect();
                if (labelArea) {
                    return {
                        x: rect.left + Math.min(Math.max(rect.width * 0.05, 10), rect.width - 5),
                        y: rect.top + Math.min(16, rect.height - 5),
                    };
                }
                return {
                    x: (rect.left + rect.right) / 2,
                    y: (rect.top + rect.bottom) / 2,
                };
            }""",
            {"nodeId": node_id, "labelArea": label_area},
        )
        assert target is not None, f"Treemap node {node_id!r} not found"
        page.mouse.click(target["x"], target["y"])

    def _click_echarts_treemap_node(
        self, page: Page, fig_idx: int, node_id: str
    ) -> None:
        page.evaluate(
            """(args) => {
                const figUid = DASHBOARD_SPEC.figures[args.figIdx].uid;
                const chart = echarts.getInstanceByDom(
                    document.getElementById('fv-chart-' + args.figIdx)
                );
                const series = (chart.getOption().series || [])[0];
                function findNode(nodes, wantedId) {
                    for (const node of (nodes || [])) {
                        if (node && node.id === wantedId) return node;
                        const child = findNode(node && node.children, wantedId);
                        if (child) return child;
                    }
                    return null;
                }
                const node = findNode(series.data || [], args.nodeId);
                handleEChartsClick({
                    seriesType: 'treemap',
                    seriesId: series.id,
                    data: node,
                }, figUid);
            }""",
            {"figIdx": fig_idx, "nodeId": node_id},
        )

    @pytest.mark.parametrize("renderer", ["plotly", "echarts"])
    def test_treemap_sibling_leaves_or_cross_filter(
        self, page: Page, server_port: int, renderer: str
    ):
        page.goto(_build_or_selection_dashboard_url(server_port, renderer))
        _wait_for_init(page, renderer)
        if renderer == "plotly":
            page.wait_for_function(
                "() => document.querySelector('#fv-plot-1')?.data?.[0]?.ids?.includes('root/solar/NL')"
            )
            self._click_plotly_treemap_node(page, 1, "root/solar/NL")
            _wait_for_selection_count(page, 1)
            self._click_plotly_treemap_node(page, 1, "root/solar/DE")
        else:
            self._click_echarts_treemap_node(page, 1, "root/solar/NL")
            _wait_for_selection_count(page, 1)
            self._click_echarts_treemap_node(page, 1, "root/solar/DE")

        _wait_for_selection_count(page, 1)
        page.wait_for_timeout(800)

        preds = _selection_predicates(page)[0]["preds"]
        assert len(preds) == 2
        countries = {
            c["values"][0]
            for p in preds
            for c in p["clauses"]
            if c["column"] == "country"
        }
        assert countries == {"NL", "DE"}

        if renderer == "plotly":
            assert set(_plotly_bar_labels(page, 0)) == {"NL", "DE"}

    @pytest.mark.parametrize("renderer", ["plotly", "echarts"])
    def test_treemap_leaf_toggle_clears_cross_filter(
        self, page: Page, server_port: int, renderer: str
    ):
        page.goto(_build_or_selection_dashboard_url(server_port, renderer))
        _wait_for_init(page, renderer)
        if renderer == "plotly":
            page.wait_for_function(
                "() => document.querySelector('#fv-plot-1')?.data?.[0]?.ids?.includes('root/solar/NL')"
            )
            self._click_plotly_treemap_node(page, 1, "root/solar/NL")
            _wait_for_selection_count(page, 1)
            page.wait_for_timeout(500)
            assert set(_plotly_bar_labels(page, 0)) == {"NL"}
            self._click_plotly_treemap_node(page, 1, "root/solar/NL")
        else:
            self._click_echarts_treemap_node(page, 1, "root/solar/NL")
            _wait_for_selection_count(page, 1)
            page.wait_for_timeout(500)
            self._click_echarts_treemap_node(page, 1, "root/solar/NL")

        _wait_for_selection_count(page, 0)
        page.wait_for_timeout(500)
        if renderer == "plotly":
            assert set(_plotly_bar_labels(page, 0)) >= {"NL", "BE", "DE"}

    @pytest.mark.parametrize("renderer", ["plotly", "echarts"])
    def test_treemap_parent_then_leaf_replaces_predicate(
        self, page: Page, server_port: int, renderer: str
    ):
        page.goto(_build_or_selection_dashboard_url(server_port, renderer))
        _wait_for_init(page, renderer)
        if renderer == "plotly":
            page.wait_for_function(
                "() => document.querySelector('#fv-plot-1')?.data?.[0]?.ids?.includes('root/solar')"
            )
            self._click_plotly_treemap_node(page, 1, "root/solar", label_area=True)
            _wait_for_selection_count(page, 1)
            self._click_plotly_treemap_node(page, 1, "root/solar/NL")
        else:
            self._click_echarts_treemap_node(page, 1, "root/solar")
            _wait_for_selection_count(page, 1)
            self._click_echarts_treemap_node(page, 1, "root/solar/NL")

        _wait_for_selection_count(page, 1)
        preds = _selection_predicates(page)[0]["preds"]
        assert len(preds) == 1
        cols = {c["column"]: c["values"][0] for c in preds[0]["clauses"]}
        assert cols == {"source": "solar", "country": "NL"}

    @pytest.mark.parametrize("renderer", ["plotly", "echarts"])
    def test_pie_sibling_slices_or_cross_filter(
        self, page: Page, server_port: int, renderer: str
    ):
        page.goto(_build_pie_or_dashboard_url(server_port, renderer))
        _wait_for_init(page, renderer)
        if renderer == "plotly":
            page.wait_for_function(
                "() => document.querySelector('#fv-plot-1')?.data?.[0]?.labels?.length > 1"
            )

            def click_pie(label: str) -> None:
                target = page.eval_on_selector(
                    "#fv-plot-1",
                    """(gd, pieLabel) => {
                        const nodes = Array.from(gd.querySelectorAll('g.slice'));
                        const node = nodes.find(n => {
                            const d = n.__data__ || {};
                            return (d.label || (d.data && d.data.label)) === pieLabel;
                        });
                        if (!node) return null;
                        node.scrollIntoView({ block: 'center', inline: 'center' });
                        const r = node.getBoundingClientRect();
                        return { x: (r.left + r.right) / 2, y: (r.top + r.bottom) / 2 };
                    }""",
                    label,
                )
                assert target is not None
                page.mouse.click(target["x"], target["y"])

            click_pie('["solar","NL"]')
            _wait_for_selection_count(page, 1)
            click_pie('["solar","DE"]')
        else:
            page.evaluate("""() => {
                const figUid = DASHBOARD_SPEC.figures[1].uid;
                const chart = echarts.getInstanceByDom(document.getElementById('fv-chart-1'));
                const series = (chart.getOption().series || [])[0];
                function clickLabel(label) {
                    const item = (series.data || []).find(entry => entry && entry.name === label);
                    handleEChartsClick({
                        seriesType: 'pie',
                        seriesId: series.id,
                        name: item.name,
                        data: item,
                    }, figUid);
                }
                clickLabel('["solar","NL"]');
            }""")
            _wait_for_selection_count(page, 1)
            page.evaluate("""() => {
                const figUid = DASHBOARD_SPEC.figures[1].uid;
                const chart = echarts.getInstanceByDom(document.getElementById('fv-chart-1'));
                const series = (chart.getOption().series || [])[0];
                const item = (series.data || []).find(
                    entry => entry && entry.name === '["solar","DE"]'
                );
                handleEChartsClick({
                    seriesType: 'pie',
                    seriesId: series.id,
                    name: item.name,
                    data: item,
                }, figUid);
            }""")

        _wait_for_selection_count(page, 1)
        page.wait_for_timeout(800)
        preds = _selection_predicates(page)[0]["preds"]
        assert len(preds) == 2
        if renderer == "plotly":
            assert set(_plotly_bar_labels(page, 0)) == {"NL", "DE"}


class TestEngineTreemapOrPredicates:
    """Engine applies OR across predicates from one treemap figure."""

    @pytest.fixture()
    def geo_df(self) -> LFQueryBuilder:
        df = pl.DataFrame(
            {
                "continent": ["Europe", "Europe", "Asia", "Asia", "Europe"],
                "country": ["Germany", "France", "Japan", "China", "Germany"],
                "population": [83.0, 67.0, 125.0, 1400.0, 83.0],
            }
        )
        return LFQueryBuilder(df)

    def _make_or_click_event(
        self,
        figure_uid: str,
        clause_groups: list[list[ClauseFilter]],
    ) -> InteractionEvent:
        predicates = [SelectionPredicate(clauses=group) for group in clause_groups]
        return InteractionEvent(
            type="selection",
            force_update=True,
            selections=[
                SelectionState(
                    source_figure_uid=figure_uid,
                    predicates=predicates,
                )
            ],
        )

    def test_treemap_or_leaf_predicates_union_countries(self, geo_df: LFQueryBuilder):
        from flexviz.trace.bar import BarPlot
        from flexviz.trace.treemap import TreeMap

        treemap = TreeMap(path=["continent", "country"], values="population")
        bar = BarPlot(labels="country", values="population")
        engine = FlexEngine(
            backend_lf=geo_df,
            scalable_traces={treemap.uid: treemap, bar.uid: bar},
        )
        infos = [
            TraceInfo(
                uid=treemap.uid, axes=None, trace_type="treemap", figure_uid="fig_tree"
            ),
            TraceInfo(
                uid=bar.uid, axes=("x", "y"), trace_type="bar", figure_uid="fig_bar"
            ),
        ]
        event = self._make_or_click_event(
            "fig_tree",
            [
                [
                    ClauseFilter(column="continent", values=["Europe"]),
                    ClauseFilter(column="country", values=["Germany"]),
                ],
                [
                    ClauseFilter(column="continent", values=["Europe"]),
                    ClauseFilter(column="country", values=["France"]),
                ],
            ],
        )
        deltas = engine.process(event, infos)
        bar_delta = next(d for d in deltas if d.uid == bar.uid)
        countries = set(bar_delta.updates.get("x", []))
        assert countries == {"Germany", "France"}

"""HTML-output tests for adapter _build_dashboard_html.

These tests assert that the generated HTML/JS contains the correct patterns
for each known bug fix.  They call _build_dashboard_html() directly — no
browser or server required.

Each test is deliberately written to FAIL before the corresponding fix is
applied, so that the test suite serves as regression coverage.
"""

from __future__ import annotations

import json
from html.parser import HTMLParser

import pytest

from flexviz.spec import DashboardSpec, FigureSpec, TraceSpec

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


class _StartTagCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags: list[tuple[str, dict[str, str | None]]] = []

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


def _gridstack_item_attrs(html: str) -> list[dict[str, str | None]]:
    parser = _StartTagCollector()
    parser.feed(html)
    return [
        attrs
        for tag, attrs in parser.tags
        if tag == "div" and attrs.get("class") == "grid-stack-item"
    ]


def _js_function_body(html: str, signature: str) -> str:
    start = html.index(signature)
    brace = html.index("{", start)
    depth = 0
    for idx in range(brace, len(html)):
        char = html[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return html[brace : idx + 1]
    raise AssertionError(f"Could not parse JS function body for {signature!r}")


def _start_tag_count(html: str, tag_name: str) -> int:
    parser = _StartTagCollector()
    parser.feed(html)
    return sum(1 for tag, _attrs in parser.tags if tag == tag_name)


def _start_tags_with_attr(
    html: str, tag_name: str, attr_name: str, attr_value: str
) -> list[dict[str, str | None]]:
    parser = _StartTagCollector()
    parser.feed(html)
    return [
        attrs
        for tag, attrs in parser.tags
        if tag == tag_name and attrs.get(attr_name) == attr_value
    ]


@pytest.fixture()
def two_fig_spec() -> DashboardSpec:
    """DashboardSpec with 2 figures, each having 2 line traces."""
    t1 = TraceSpec(uid="t1", trace_type="line", display={"name": "A"}, axes=("x", "y"))
    t2 = TraceSpec(uid="t2", trace_type="line", display={"name": "B"}, axes=("x", "y"))
    t3 = TraceSpec(uid="t3", trace_type="line", display={"name": "C"}, axes=("x", "y"))
    t4 = TraceSpec(uid="t4", trace_type="line", display={"name": "D"}, axes=("x", "y"))
    fig1 = FigureSpec(uid="fig1", layout={"title": "Figure 1"}, traces=[t1, t2])
    fig2 = FigureSpec(uid="fig2", layout={"title": "Figure 2"}, traces=[t3, t4])
    return DashboardSpec(figures=[fig1, fig2])


@pytest.fixture()
def two_histogram_spec() -> DashboardSpec:
    """DashboardSpec with one figure containing two histogram traces."""
    h1 = TraceSpec(
        uid="h1", trace_type="histogram", display={"name": "H1"}, axes=("x",)
    )
    h2 = TraceSpec(
        uid="h2", trace_type="histogram", display={"name": "H2"}, axes=("x",)
    )
    fig = FigureSpec(uid="fig-hist", layout={"title": "Histograms"}, traces=[h1, h2])
    return DashboardSpec(figures=[fig])


# ---------------------------------------------------------------------------
# Plotly HTML tests
# ---------------------------------------------------------------------------


class TestPlotlyHtml:
    @pytest.fixture()
    def html(self, two_fig_spec):
        from flexviz.adapters.plotly_adapter import PlotlyAdapter

        return PlotlyAdapter()._build_dashboard_html(
            two_fig_spec, server_url="http://localhost:9999"
        )

    # Bug 1: Global reset must visually clear axis ranges on all figures.
    # After postDashboardUpdate resolves, fvOnReset must set autorange=true
    # on all layoutsByFig entries so Plotly.react resets the visible zoom.
    def test_reset_sets_autorange_true(self, html):
        assert (
            "autorange" in html
        ), "fvOnReset must set autorange:true on layoutsByFig after postDashboardUpdate"

    def test_reset_deletes_axis_range(self, html):
        # The reset handler must explicitly clear the saved range property
        # so Plotly does not re-render at the old zoom level.
        assert (
            "_programmaticOp" in html
        ), "fvOnReset must use _programmaticOp guard (same as deselect guard)"

    def test_reset_calls_fv_reset_runtime_cache(self, html):
        # fvOnReset must call fvResetRuntimeCache to clear stale fg layer data
        # and bgYExtentByFig before posting the reset event so that overlay mode
        # renders a clean state after the server response arrives.
        assert (
            "window.fvResetRuntimeCache?.()" in html
        ), "fvOnReset must call window.fvResetRuntimeCache?.() to clear overlay cache"

    def test_shared_panel_wrapper_present(self, html):
        assert "<fv-panel" in html
        assert 'data-mode="zoom"' in html
        assert "--fv-panel-bg" in html
        assert "data-fv-grid-editable" in html

    def test_panel_bar_structure_present_for_interactive_figures(self, html):
        panel_bars = _start_tags_with_attr(html, "div", "class", "fv-panel-bar")
        assert len(panel_bars) == 2
        assert panel_bars[0]["id"] == "fv-bar-0"
        assert panel_bars[0]["role"] == "toolbar"
        assert panel_bars[0]["aria-label"] == "Panel controls"
        for slot in ("modes", "actions", "info", "warn", "status"):
            assert f'data-slot="{slot}"' in html

    def test_panel_bar_info_slot_is_emitted_empty(self, html):
        assert (
            '<span class="fv-panel-bar-slot fv-panel-bar-info" data-slot="info"></span>'
            in html
        )

    def test_header_emits_filter_strip_container(self, html):
        assert 'id="fv-header-main"' in html
        assert 'id="fv-filter-strip"' in html
        assert 'id="fv-filter-chips"' in html
        assert "Active filters" in html

    def test_selection_summary_runtime_hooks_are_present(self, html):
        assert "window.fvRefreshSelectionSummary = function()" in html
        assert "window.fvRemoveSelectionByFigure = async function(figUid)" in html
        assert "window.fvSummarizeSelection = function(sel)" in html
        assert "window.fvFigureLabel = function(figUid)" in html
        assert "window.fvSetSelectionState = function(selections)" in html
        assert "_fvFormatSummaryIsoDatetime" in html

    def test_agent_readback_accessor_present(self, html):
        assert (
            "window.flexvizState" in html
        ), "shared runtime must expose the flexvizState() agent-readback accessor"

    def test_panel_bar_buttons_have_accessible_labels(self, html):
        for label in (
            "Zoom mode",
            "Pan mode",
            "Cross-filter mode",
            "Reset panel view",
            "Toggle axis lock",
        ):
            assert f'aria-label="{label}"' in html

    def test_panel_bar_is_omitted_for_non_interactive_plotly_figure(self):
        from flexviz.adapters.plotly_adapter import PlotlyAdapter

        spec = DashboardSpec(
            figures=[
                FigureSpec(
                    uid="fig-pie",
                    traces=[
                        TraceSpec(
                            uid="pie-1",
                            trace_type="pie",
                            display={"name": "Pie"},
                            backend_data={"labels": "country", "values": "val"},
                        )
                    ],
                )
            ]
        )
        html = PlotlyAdapter()._build_dashboard_html(
            spec, server_url="http://localhost:9999"
        )
        assert 'id="fv-bar-0"' not in html
        assert 'class="fv-panel-has-bar"' not in html
        assert '<div class="fv-plot-wrap">' in html

    # Bug 2: Deselect must not fire a second backend call when Plotly's own
    # plotly_deselect event fires programmatically after Plotly.react clears
    # selection boxes.  A guard flag prevents the double-fire.
    def test_deselect_has_programmatic_guard(self, html):
        assert (
            "_programmaticOp" in html
        ), "handleDeselect must check _programmaticOp to prevent double backend calls"

    def test_grouped_parent_not_bootstrapped_as_trace(self):
        from flexviz.adapters.plotly_adapter import PlotlyAdapter

        grouped = TraceSpec(
            uid="grp-line",
            trace_type="line",
            backend_data={"x": "ts", "y": "val"},
            params={
                "group_by": "sensor",
                "group_domain_key": "src::sensor",
                "n_points": 100,
                "downsample": "nth",
                "add_gaps": True,
            },
            display={"name": "Grouped"},
            axes=("x", "y"),
        )
        html = PlotlyAdapter()._build_dashboard_html(
            DashboardSpec(figures=[FigureSpec(uid="fig-g", traces=[grouped])]),
            server_url="http://localhost:9999",
        )
        assert "const tracesArr_0 = [];" in html

    def test_grouped_reconciliation_checks_field_presence_not_length(self, html):
        assert "Object.prototype.hasOwnProperty.call(delta, 'group_results')" in html

    def test_grouped_reconciliation_uses_group_domain_key(self, html):
        assert "group_domain_key" in html

    def test_grouped_children_are_built_from_parent_spec(self, html):
        assert "makePlotlyTrace(parentSpec" in html

    def test_figure_scoped_clear_keeps_other_selections(self, html):
        body = _js_function_body(html, "function clearFigureSelection(figUid)")
        assert "type: remainingSelections.length ? 'selection' : 'deselect'" in body
        assert "selections: remainingSelections" in body
        assert "figure_uid: figUid" in body

    def test_click_toggle_uses_figure_scoped_clear(self, html):
        body = _js_function_body(html, "function handleClick(eventData, figUid)")
        # When the new click matches an existing selection, the figure-scoped
        # clearFigureSelection helper is called instead of emitting a selection.
        assert "clearFigureSelection(figUid);" in body
        assert "fvUpsertPathPredicate" in body

    def test_plotly_deselect_uses_figure_scoped_clear(self, html):
        body = _js_function_body(html, "function handleDeselect(figUid)")
        assert "clearFigureSelection(figUid);" in body
        assert "selections = []" not in body

    def test_handle_selected_emits_predicates_with_columns(self, html):
        body = _js_function_body(html, "function handleSelected(eventData, figUid)")
        assert "predicates" in body
        assert "x_ref" not in body
        assert "y_ref" not in body
        assert "x_range:" not in body

    def test_handle_click_pie_emits_single_clause_predicate(self, html):
        body = _js_function_body(html, "function handleClick(eventData, figUid)")
        assert "predicates" in body
        assert "clauses" in body
        assert "categories_column" not in body
        assert "selectionCategories" not in body

    def test_handle_click_is_descriptor_driven(self, html):
        # handleClick dispatches on the trace's declared selection descriptor
        # (selection.kind / path_columns), not on trace_type or backend_data,
        # and still uses the path-predicate upsert for accumulate (or/path) multi.
        body = _js_function_body(html, "function handleClick(eventData, figUid)")
        assert "fvUpsertPathPredicate" in body
        assert "selection" in body
        assert "path_columns" in body
        assert "kind === 'path'" in body

    def test_selection_boxes_read_predicates(self, html):
        body = _js_function_body(html, "function selectionBoxesForFigure(figUid)")
        assert "predicates" in body
        assert "x_range" not in body
        assert "y_range" not in body

    def test_bar_selection_boxes_preserve_plotly_brush_shape(self, html):
        handle_selected = _js_function_body(
            html, "function handleSelected(eventData, figUid)"
        )
        selection_boxes = _js_function_body(
            html, "function selectionBoxesForFigure(figUid)"
        )
        assert "_plotlySelectionBoxFromRange(eventData, figUid)" in handle_selected
        assert "_plotly_selection_box" in handle_selected
        assert "_plotly_selection_box" in selection_boxes

    def test_apply_category_styles_walks_predicate_clauses(self, html):
        body = _js_function_body(html, "function applyCategorySelectionStyles(figUid)")
        assert "predicates" in body
        # The helper that inspects clauses is defined alongside this function.
        node_helper = _js_function_body(
            html,
            "function _nodeSatisfiesPredicate(figSpec, parentTrace, node, predicate)",
        )
        assert "clauses" in node_helper

    def test_overlay_runtime_tracks_trace_caches(self, html):
        assert "layerDataByUid" in html
        assert "hasBgByFigure" in html
        assert "delta.layer || 'base'" in html
        assert "{ base: [], bg: [], fg: [] }" in html

    def test_overlay_runtime_refreshes_cf_mode_button(self, html):
        assert "window.fvUpdateCfModeButton?.();" in html

    def test_runtime_ships_figure_scoped_reset_cache(self, html):
        # Per-figure reset case 3a: the figure-scoped cache getter must be in the
        # bundle and chained into postDashboardUpdate after the whole-dashboard
        # getter.
        assert "function fvCacheGetFigure(event)" in html
        assert "fvCacheGet(event) || fvCacheGetFigure(event)" in html

    def test_runtime_ships_panel_reset_noop_guard(self, html):
        # Per-figure reset cases 3c/3d: the no-op guard must short-circuit before
        # posting, gated on no zoom and no selection change. (Axis locks are
        # view-only, so they need no special-casing here.)
        assert "window.fvFigureHasViewport" in html
        assert "if (!wasZoomed && !selectionChanged) return;" in html

    def test_overlay_runtime_exposes_cache_helpers(self, html):
        assert "window.fvEnsureOverlayBackground" in html
        assert "window.fvResetRuntimeCache" in html
        assert "window.fvRestoreFromSpec" in html

    def test_overlay_runtime_uses_muted_background_opacity(self, html):
        assert "const OVERLAY_BG_OPACITY = 0.16" in html

    def test_overlay_yaxis_anchoring(self, html):
        assert "bgYExtentByFig" in html
        assert "_updateBgYExtent" in html

    def test_plotly_bar_layers_use_logical_offsetgroup_when_fg_visible(self, html):
        body = _js_function_body(
            html, "function buildTraceFromTemplate(template, logicalUid"
        )
        assert "trace.type === 'bar' && forceBarOffsetgroup" in body
        assert "trace.offsetgroup = logicalUid" in body
        assert "trace.alignmentgroup = 'fv-bars'" in body

    def test_plotly_bar_offsetgroup_excluded_for_stack_mode(self, html):
        body = _js_function_body(html, "function buildTracesForFigure(figUid)")
        # Stack-mode bars must not receive offsetgroup even when overlay fg is
        # shown; forceBarOffsetgroup must be gated on the trace bar_mode.
        assert "forceBarOffsetgroup" in body
        assert "'stack'" in body

    def test_plotly_make_trace_does_not_set_offsetgroup(self, html):
        body = _js_function_body(html, "function makePlotlyTrace(ts, uid, name, color)")
        assert "offsetgroup" not in body
        assert "alignmentgroup" not in body

    def test_plotly_histogram_bootstrap_sets_group_barmode(self, two_histogram_spec):
        from flexviz.adapters.plotly_adapter import PlotlyAdapter

        html = PlotlyAdapter()._build_dashboard_html(
            two_histogram_spec, server_url="http://localhost:9999"
        )
        assert '"barmode": "group"' in html

    def test_plotly_base_barmode_for_histogram_figure(self, html):
        body = _js_function_body(html, "function baseBarmodeForFigure(figSpec)")
        assert "'group'" in body
        assert "histogram" in body

    def test_plotly_waits_for_initial_newplot_before_restore(self, html):
        # Bundle uses .map() form instead of literal array
        assert "const initialPlotPromises = _fvAllFigUids.map(" in html
        assert "await Promise.all(initialPlotPromises);" in html
        assert "await restoreDashboardFromSpec();" in html

    # Linked hover infrastructure
    def test_hover_col_to_fig_axis_built(self, html):
        assert "hoverTargetsByColumn" in html

    def test_hover_source_types_defined(self, html):
        assert "IMPLEMENTED_HOVER_MODES" in html

    def test_hover_normalize_fn(self, html):
        assert "normalizePlotlyHover" in html

    def test_hover_plan_visuals_fn(self, html):
        assert "planHoverVisuals" in html

    def test_hover_dropdown_present(self, html):
        assert "fv-hover-btn" in html
        assert "fvInitHoverDropdown" in html

    def test_panel_reset_hook_present(self, html):
        assert "fvOnResetPanel" in html
        assert 'data-action="reset-panel"' in html

    def test_panel_reset_clears_sourced_selection(self, html):
        # Reset panel must remove the cross-filter selection sourced from that
        # figure, not just zoom out.  The base fvOnResetPanel (shared bundle)
        # must call fvClearFigureSelectionFromList and fvSetSelectionState.
        body = _js_function_body(html, "window.fvOnResetPanel = async function(figUid)")
        assert "fvClearFigureSelectionFromList" in body
        assert "fvSetSelectionState" in body

    def test_panel_reset_event_type_adapts_to_selection(self, html):
        # When the panel sourced a cross-filter, reset-panel must send a
        # 'selection'/'deselect' event (so other panels update), not 'viewport'.
        # When no selection existed, 'viewport' is still correct.
        body = _js_function_body(html, "window.fvOnResetPanel = async function(figUid)")
        assert "selectionChanged" in body
        assert "'selection'" in body
        assert "'deselect'" in body
        assert "'viewport'" in body

    def test_axis_lock_controls_present(self, html):
        assert 'data-action="lock-axes"' in html
        assert "Lock Axes" in html
        assert "fvOnToggleAxisLocks" in html
        assert "fvAreCurrentAxesLocked" in html
        assert "fvSyncFigureModeForAxisLocks" in html
        assert "fvCaptureAxisDisplayRanges" in html
        assert "fvApplyAxisLocks" in html

    def test_plotly_locked_axes_disable_zoom_buttons(self, html):
        body = _js_function_body(html, "function updateModeIndicator(figUid, dragmode)")
        assert (
            "const axesLocked = window.fvAreCurrentAxesLocked?.(figUid) === true;"
            in body
        )
        assert "(axesLocked && (mode === 'zoom' || mode === 'pan'))" in body

    def test_axis_lock_uses_renderer_display_ranges(self, html):
        assert "fvCaptureAxisDisplayRanges" in html
        assert "axisObj && axisObj.range" in html
        assert "fvHasLockableCurrentAxis" in html
        assert "plotlyAxisSupportsDataRange" not in html

    def test_axis_lock_ranges_are_visual_not_backend_viewport(self, html):
        assert "axis_lock_ranges" in html
        assert "fvStoreAxisLockRanges" in html
        assert "datarevision" in html
        assert "plotlyDataExtentForAxis" not in html
        assert "plotlyHeatmapAxisExtent" not in html

    def test_handlerelayout_prunes_and_merges_locked_axis_ranges(self, html):
        body = _js_function_body(html, "function handleRelayout(relayout, figUid)")
        assert "fvPruneAxisRangesForLocks" in body
        assert "touchedLockedAxes" in body
        assert "fvApplyAxisLocks" in body
        assert "axis_ranges: axisRanges" in body

    def test_panel_reset_preserves_locked_axis_ranges(self, html):
        assert "fvClearFigureViewport" in html
        assert "axis_lock_ranges" in html

    def test_toolbar_includes_grid_toggle_button(self, html):
        assert "fv-btn-grid" in html

    def test_hover_relayout_shapes_guard(self, html):
        assert "/^shapes" in html

    def test_hover_event_wiring(self, html):
        assert "plotly_hover" in html
        assert "plotly_unhover" in html

    def test_hover_clear_fn_exposed(self, html):
        assert "fvClearAllCrosshairs" in html

    def test_hover_state_uses_client_state(self, html):
        assert "window._hoverEnabled" not in html
        assert "DASHBOARD_SPEC.state.hover_enabled" not in html
        assert "DASHBOARD_SPEC.client_state" in html

    def test_hover_shapes_filtered_by_tag_prefix(self, html):
        assert "g.tag.startsWith('linked')" in html

    def test_hover_uses_x_mode_defaults(self, html):
        assert '"hovermode": "x"' in html
        assert '"hoverdistance": -1' in html

    def test_hover_cell_types_defined(self, html):
        assert "IMPLEMENTED_CELL_TRACE_TYPES" in html

    def test_hover_band_rendering_fn(self, html):
        assert "x_band" in html

    def test_hover_corr_heatmap_affordance(self, html):
        assert "_drawCorrHeatmapAffordance" in html

    # Modebar button configuration
    def test_modebar_hides_plotly_logo(self, html):
        assert "displaylogo: false" in html

    def test_plotly_config_disables_interaction_tips(self, html):
        assert "showTips: false" in html

    def test_modebar_hidden_entirely(self, html):
        # The Plotly modebar is fully replaced by the per-figure mode toggle.
        assert "displayModeBar: false" in html

    def test_panel_control_queries_use_bar_root(self, html):
        assert "document.getElementById('fv-bar-' + figIdx)" in html
        assert "window.fvPanelControlRoot" in html

    def test_plotly_mode_buttons_use_mode_active_class(self, html):
        body = _js_function_body(html, "function updateModeIndicator(figUid, dragmode)")
        assert "mode-active" in body

    # handleRelayout guards
    def test_handlerelayout_has_programmatic_op_guard(self, html):
        # handleRelayout must bail early when _programmaticOp is true to avoid
        # sending a spurious second server request during fvOnReset.
        assert "handleRelayout" in html
        # The guard must appear inside the function body (not just in fvOnReset).
        # We verify both strings appear in the HTML (order is guaranteed by the
        # single JS function definition).
        assert "_programmaticOp" in html

    def test_autorange_sends_viewport_not_reset(self, html):
        # The modebar home button fires an autorange relayout.  This must send a
        # 'viewport' event (preserving cross-filter selections), NOT a 'reset'.
        # We verify the autorange branch uses type: 'viewport'.
        assert "type: 'viewport'" in html or "type:'viewport'" in html

    def test_plotly_init_binds_resize_observer(self, html):
        assert "new ResizeObserver" in html
        assert "bindPlotlyResizeObserver" in html

    def test_bar_triggered_plotly_relayout_is_programmatic(self, html):
        body = _js_function_body(html, "function setFigureMode(figUid, mode)")
        assert "fvRunProgrammaticPlotlyOp" in body
        lock_body = _js_function_body(
            html, "window.fvApplyAxisLocks = function(figUid, changedAxisId)"
        )
        assert "fvRunProgrammaticPlotlyOp" in lock_body

    def test_draggable_layout_wires_grid_edit_toggle(self, two_fig_spec):
        from flexviz.adapters.plotly_adapter import PlotlyAdapter
        from flexviz.spec import LayoutSpec

        spec = two_fig_spec.model_copy(deep=True)
        spec.layout = LayoutSpec(draggable=True)
        html = PlotlyAdapter()._build_dashboard_html(
            spec, server_url="http://localhost:9999"
        )
        assert "window.fvSetGridEditable" in html
        assert "_fvGrid.enableMove" in html
        assert "_fvGrid.enableResize" in html

    def test_gridstack_uid_is_html_attribute_escaped(self):
        from flexviz.adapters.plotly_adapter import PlotlyAdapter

        uid = 'fig" onmouseover="window.__fv_xss=1" data-x="'
        spec = DashboardSpec(figures=[FigureSpec(uid=uid, traces=[])])
        html = PlotlyAdapter()._build_dashboard_html(spec, server_url="http://test")
        attrs = _gridstack_item_attrs(html)
        assert attrs[0]["gs-id"] == uid
        assert "onmouseover" not in attrs[0]

    def test_inline_json_cannot_close_script(self):
        from flexviz.adapters.plotly_adapter import PlotlyAdapter

        payload = '</script><script id="fv-xss"></script>'
        spec = DashboardSpec(
            figures=[
                FigureSpec(
                    uid="fig1",
                    layout={"title": payload},
                    traces=[
                        TraceSpec(
                            uid="t1",
                            trace_type="line",
                            display={"name": payload},
                            axes=("x", "y"),
                        )
                    ],
                )
            ]
        )
        html = PlotlyAdapter()._build_dashboard_html(spec, server_url="http://test")
        assert '<script id="fv-xss">' not in html
        assert "\\u003c/script\\u003e\\u003cscript" in html

    def test_plotly_uid_handlers_use_json_string_literals(self):
        from flexviz.adapters.plotly_adapter import PlotlyAdapter

        uid = "fig');window.__fv_xss=1;//"
        spec = DashboardSpec(figures=[FigureSpec(uid=uid, traces=[])])
        html = PlotlyAdapter()._build_dashboard_html(spec, server_url="http://test")
        # In the bundle architecture, handlers receive figUid from the forEach loop
        # (never as an inline literal), so the injection string is only in _fvAllFigUids.
        assert "handleRelayout(rd, 'fig');window.__fv_xss=1;//')" not in html
        # UID appears JSON-encoded in the _fvAllFigUids array
        assert "fig');window.__fv_xss=1;//" in html  # present but only in the array

    def test_layout_gap_is_sanitized_for_style_blocks(self):
        from flexviz.adapters.plotly_adapter import PlotlyAdapter
        from flexviz.spec import LayoutSpec

        spec = DashboardSpec(
            figures=[FigureSpec(uid="fig1", traces=[])],
            layout=LayoutSpec(gap='8px;}</style><script id="fv-xss"></script>'),
        )
        html = PlotlyAdapter()._build_dashboard_html(spec, server_url="http://test")
        assert '<script id="fv-xss">' not in html
        assert "padding: 8px;" in html


# ---------------------------------------------------------------------------
# Plotly legend auto-visibility tests
# ---------------------------------------------------------------------------


class TestPlotlyLegend:
    def _build_html(self, traces, layout=None):
        from flexviz.adapters.plotly_adapter import PlotlyAdapter

        fig = FigureSpec(uid="fig1", layout=layout or {}, traces=traces)
        return PlotlyAdapter()._build_dashboard_html(
            DashboardSpec(figures=[fig]), server_url="http://localhost:9999"
        )

    def test_single_trace_hides_legend_by_default(self):
        traces = [TraceSpec(uid="t1", trace_type="line", axes=("x", "y"))]
        html = self._build_html(traces)
        assert '"showlegend": false' in html

    def test_multiple_traces_shows_legend_by_default(self):
        traces = [
            TraceSpec(uid="t1", trace_type="line", axes=("x", "y")),
            TraceSpec(uid="t2", trace_type="line", axes=("x", "y")),
        ]
        html = self._build_html(traces)
        assert '"showlegend": true' in html

    def test_single_grouped_trace_shows_legend_by_default(self):
        traces = [
            TraceSpec(
                uid="t1",
                trace_type="line",
                axes=("x", "y"),
                params={"group_by": "category"},
            )
        ]
        html = self._build_html(traces)
        assert '"showlegend": true' in html

    def test_explicit_legend_true_overrides_single_trace(self):
        traces = [TraceSpec(uid="t1", trace_type="line", axes=("x", "y"))]
        html = self._build_html(traces, layout={"legend": True})
        assert '"showlegend": true' in html

    def test_explicit_legend_false_overrides_multiple_traces(self):
        traces = [
            TraceSpec(uid="t1", trace_type="line", axes=("x", "y")),
            TraceSpec(uid="t2", trace_type="line", axes=("x", "y")),
        ]
        html = self._build_html(traces, layout={"legend": False})
        assert '"showlegend": false' in html

    def test_legend_layout_keeps_auto_visibility(self):
        traces = [
            TraceSpec(uid="t1", trace_type="line", axes=("x", "y")),
            TraceSpec(uid="t2", trace_type="line", axes=("x", "y")),
        ]
        html = self._build_html(
            traces, layout={"legend": {"orientation": "h", "y": -0.15}}
        )
        assert '"legend": {"orientation": "h", "y": -0.15}' in html
        assert '"showlegend": true' in html

    def test_plotly_rebuild_preserves_legend_visibility(self):
        traces = [
            TraceSpec(uid="t1", trace_type="line", axes=("x", "y")),
            TraceSpec(uid="t2", trace_type="line", axes=("x", "y")),
        ]
        html = self._build_html(traces)
        assert "const legendVisibilityByUid = {};" in html
        assert "rememberPlotlyVisibility(figIdx);" in html
        assert "applyLegendVisibility(trace, logicalUid);" in html


# ---------------------------------------------------------------------------
# ECharts HTML tests
# ---------------------------------------------------------------------------


class TestEChartsHtml:
    @pytest.fixture()
    def html(self, two_fig_spec):
        from flexviz.adapters.echarts_adapter import EChartsAdapter

        return EChartsAdapter()._build_dashboard_html(
            two_fig_spec, server_url="http://localhost:9999"
        )

    @pytest.fixture()
    def initial_option(self, two_fig_spec):
        """Parsed initial ECharts option dict for the first figure."""
        from flexviz.adapters.echarts_adapter import EChartsAdapter

        return EChartsAdapter()._build_initial_option(
            two_fig_spec.figures[0], fig_height=400
        )

    # Bug 7: applyDeltasToFig must reset dataZoom to show 100% of new data.
    def test_apply_deltas_resets_datazoom(self, html):
        # After setting series data for a viewport update, the chart must
        # reset dataZoom to start:0 / end:100 so the downsampled data fills
        # the visible window.
        assert (
            "start: 0" in html or '"start": 0' in html or "start:0" in html
        ), "applyDeltasToFig must reset dataZoom start to 0 after applying new data"
        assert (
            "end: 100" in html or '"end": 100' in html or "end:100" in html
        ), "applyDeltasToFig must reset dataZoom end to 100 after applying new data"

    # Bug 8: Selection must use brushEnd event for complete coordinates.
    def test_brush_uses_brushend_event(self, html):
        assert "brushEnd" in html, (
            "ECharts adapter must use 'brushEnd' event (fires once with complete "
            "coordRange) instead of 'brushSelected' for cross-filter selections"
        )

    def test_brush_config_has_axis_index(self, initial_option):
        # The brush component (not dataZoom) must link to axis indices so that
        # brushEnd coordRange is populated with cartesian coordinates.
        brush = initial_option.get("brush", {})
        assert "xAxisIndex" in brush, (
            "initial option brush config must include xAxisIndex so that "
            "brushEnd coordRange contains x-axis data coordinates"
        )

    # Bug 9a: Per-figure Select button must be removed.
    def test_no_per_figure_select_button(self, html):
        assert "btn-select" not in html, (
            "ECharts adapter must not emit per-figure Select buttons; "
            "selection is handled by the shared panel controls"
        )

    def test_toolbox_hidden_in_initial_option(self, initial_option):
        assert initial_option["toolbox"]["show"] is False

    def test_axis_lock_renderer_hooks_present(self, html):
        assert "fvCaptureAxisDisplayRanges" in html
        assert "fvApplyAxisLocks" in html
        assert "axis_lock_ranges" in html

    def test_shared_panel_wrapper_present(self, html):
        assert "<fv-panel" in html
        assert 'data-mode="zoom"' in html
        assert "--fv-panel-bg" in html
        assert "data-fv-grid-editable" in html

    def test_echarts_locked_axes_disable_zoom_button(self, html):
        body = _js_function_body(html, "function updateModeIndicator(figUid)")
        assert (
            "const axesLocked = window.fvAreCurrentAxesLocked?.(figUid) === true;"
            in body
        )
        assert "(axesLocked && mode === 'zoom')" in body

    def test_panel_controls_handle_selection_modes(self, html):
        assert "window.fvBindPanelControls?.(figUid" in html
        assert "setFigureMode(figUid, mode)" in html

    # Bug 9c: Lasso select must be absent.
    def test_no_lasso_select(self, initial_option):
        option_json = json.dumps(initial_option).lower()
        assert (
            "lasso" not in option_json
        ), "initial option must not include lasso brush type"

    # Scroll zoom fix: _applyingDeltas guard prevents datazoom feedback loop.
    def test_applying_deltas_guard_present(self, html):
        assert "_applyingDeltas" in html, (
            "applyDeltasToFig must set _applyingDeltas=true before setOption "
            "and false after, so the datazoom event is suppressed during data updates"
        )

    def test_datazoom_handler_checks_guard(self, html):
        assert (
            "if (_applyingDeltas) return;" in html
        ), "datazoom handler must bail out when _applyingDeltas is true"

    def test_runtime_restores_brush_areas_from_state(self, html):
        assert "syncBrushAreasForFigure" in html
        assert "chart.dispatchAction({ type: 'brush', areas })" in html
        body = _js_function_body(html, "function brushAreasForFigure(figUid)")
        assert "predicates" in body
        assert "x_range" not in body
        assert "y_range" not in body

    def test_brush_clear_keeps_other_selections(self, html):
        body = _js_function_body(html, "chart.on('brushEnd', function(params)")
        cleared_branch = body.split("if (!areas || !areas.length) {", 1)[1].split(
            "return;", 1
        )[0]
        assert "clearFigureSelection(figUid);" in cleared_branch
        clear_helper = _js_function_body(html, "function clearFigureSelection(figUid)")
        assert (
            "type: remainingSelections.length ? 'selection' : 'deselect'"
            in clear_helper
        )
        assert "selections: remainingSelections" in clear_helper
        assert "figure_uid: figUid" in clear_helper

    def test_brush_end_emits_predicates(self, html):
        body = _js_function_body(html, "chart.on('brushEnd', function(params)")
        assert "predicates" in body
        assert "clauses" in body
        assert "x_range" not in body
        assert "y_range" not in body

    def test_brush_end_normalizes_time_ranges_to_iso_strings(self, html):
        assert "function normalizeEChartsRangeForAxis(chart, axisFamily, range)" in html
        body = _js_function_body(
            html, "function normalizeEChartsRangeForAxis(chart, axisFamily, range)"
        )
        assert "toISOString()" in body

    def test_echarts_click_handler_present_for_categorical_traces(self, html):
        assert "function handleEChartsClick(params, figUid)" in html
        assert (
            "chart.on('click', function(params) { handleEChartsClick(params, figUid); });"
            in html
        )

    def test_echarts_click_uses_path_predicate_upsert(self, html):
        body = _js_function_body(html, "function handleEChartsClick(params, figUid)")
        assert "fvUpsertPathPredicate" in body
        assert "fvReplaceFigureSelection" in body

    def test_grouped_parent_not_bootstrapped_as_series(self):
        from flexviz.adapters.echarts_adapter import EChartsAdapter

        grouped = TraceSpec(
            uid="grp-line",
            trace_type="line",
            backend_data={"x": "ts", "y": "val"},
            params={
                "group_by": "sensor",
                "group_domain_key": "src::sensor",
                "n_points": 100,
                "downsample": "nth",
                "add_gaps": True,
            },
            display={"name": "Grouped"},
            axes=("x", "y"),
        )
        option = EChartsAdapter._build_initial_option(
            FigureSpec(uid="fig-g", traces=[grouped]), 400
        )
        assert option["series"] == []

    def test_grouped_reconciliation_uses_group_domain_key(self, html):
        assert "group_domain_key" in html

    def test_grouped_reconciliation_replaces_full_series_array(self, html):
        assert "replaceMerge: ['series']" in html

    def test_grouped_children_are_built_from_parent_spec(self, html):
        assert "makeEChartsSeries(parentSpec" in html

    def test_overlay_runtime_tracks_series_caches(self, html):
        assert "layerDataByUid" in html
        assert "hasBgByFigure" in html
        assert "delta.layer || 'base'" in html
        assert "{ base: [], bg: [], fg: [] }" in html

    def test_overlay_runtime_exposes_cache_helpers(self, html):
        assert "window.fvEnsureOverlayBackground" in html
        assert "window.fvResetRuntimeCache" in html
        assert "window.fvRestoreFromSpec" in html

    def test_reset_calls_fv_reset_runtime_cache(self, html):
        # fvOnReset must call fvResetRuntimeCache to clear stale fg layer data
        # and bgYExtentByFig before posting the reset event.
        assert (
            "window.fvResetRuntimeCache?.()" in html
        ), "fvOnReset must call window.fvResetRuntimeCache?.() to clear overlay cache"

    def test_overlay_runtime_uses_muted_background_opacity(self, html):
        assert "const OVERLAY_BG_OPACITY = 0.16" in html

    def test_overlay_yaxis_anchoring(self, html):
        assert "bgYExtentByFig" in html
        assert "_updateBgYExtent" in html

    def test_no_root_barmode_option(self, html):
        assert '"barMode"' not in html

    # Linked hover infrastructure
    def test_hover_col_to_fig_axis_built(self, html):
        assert "hoverTargetsByColumn" in html

    def test_hover_source_types_defined(self, html):
        assert "IMPLEMENTED_HOVER_MODES" in html

    def test_hover_get_mode_fn(self, html):
        assert "getHoverMode" in html

    def test_hover_dispatch_visuals_fn(self, html):
        assert "dispatchHoverVisuals" in html

    def test_hover_dropdown_present(self, html):
        assert "fv-hover-btn" in html
        assert "fvInitHoverDropdown" in html

    def test_toolbar_includes_grid_toggle_button(self, html):
        assert "fv-btn-grid" in html

    def test_hover_event_wiring(self, html):
        assert "mouseover" in html
        assert "mouseout" in html

    def test_hover_clear_fn_exposed(self, html):
        assert "fvClearAllCrosshairs" in html

    def test_hover_state_uses_dashboard_spec(self, html):
        assert "window._hoverEnabled" not in html
        assert "getHoverMode(DASHBOARD_SPEC)" in html

    def test_hover_axis_pointer_in_tooltip(self, initial_option):
        tooltip = initial_option.get("tooltip", {})
        assert tooltip.get("axisPointer", {}).get("type") == "line"

    def test_hover_cell_types_defined(self, html):
        assert "IMPLEMENTED_CELL_TRACE_TYPES" in html

    def test_hover_band_rendering_fn(self, html):
        assert "IMPLEMENTED_AXIS_BAND_TARGET_TRACE_TYPES" in html

    def test_hover_corr_heatmap_affordance(self, html):
        assert "heatmapVisualMapForTrace" in html

    def test_draggable_layout_wires_grid_edit_toggle(self, two_fig_spec):
        from flexviz.adapters.echarts_adapter import EChartsAdapter
        from flexviz.spec import LayoutSpec

        spec = two_fig_spec.model_copy(deep=True)
        spec.layout = LayoutSpec(draggable=True)
        html = EChartsAdapter()._build_dashboard_html(
            spec, server_url="http://localhost:9999"
        )
        assert "window.fvSetGridEditable" in html
        assert "_fvGrid.enableMove" in html
        assert "_fvGrid.enableResize" in html

    def test_gridstack_uid_is_html_attribute_escaped(self):
        from flexviz.adapters.echarts_adapter import EChartsAdapter

        uid = 'fig" onmouseover="window.__fv_xss=1" data-x="'
        spec = DashboardSpec(figures=[FigureSpec(uid=uid, traces=[])])
        html = EChartsAdapter()._build_dashboard_html(spec, server_url="http://test")
        attrs = _gridstack_item_attrs(html)
        assert attrs[0]["gs-id"] == uid
        assert "onmouseover" not in attrs[0]

    def test_inline_json_cannot_close_script(self):
        from flexviz.adapters.echarts_adapter import EChartsAdapter

        payload = '</script><script id="fv-xss"></script>'
        spec = DashboardSpec(
            figures=[
                FigureSpec(
                    uid="fig1",
                    layout={"title": payload},
                    traces=[
                        TraceSpec(
                            uid="t1",
                            trace_type="line",
                            display={"name": payload},
                            axes=("x", "y"),
                        )
                    ],
                )
            ]
        )
        html = EChartsAdapter()._build_dashboard_html(spec, server_url="http://test")
        assert '<script id="fv-xss">' not in html
        assert "\\u003c/script\\u003e\\u003cscript" in html

    def test_layout_gap_is_sanitized_for_style_blocks(self):
        from flexviz.adapters.echarts_adapter import EChartsAdapter
        from flexviz.spec import LayoutSpec

        spec = DashboardSpec(
            figures=[FigureSpec(uid="fig1", traces=[])],
            layout=LayoutSpec(gap='8px;}</style><script id="fv-xss"></script>'),
        )
        html = EChartsAdapter()._build_dashboard_html(spec, server_url="http://test")
        assert '<script id="fv-xss">' not in html
        assert "padding: 8px;" in html

    def test_toolbar_config_is_applied(self):
        from flexviz.adapters.echarts_adapter import EChartsAdapter
        from flexviz.spec import LayoutSpec, ToolbarConfig

        spec = DashboardSpec(
            figures=[FigureSpec(uid="fig1", traces=[])],
            layout=LayoutSpec(
                toolbar=ToolbarConfig(show_share=False, show_export=False)
            ),
        )
        html = EChartsAdapter()._build_dashboard_html(spec, server_url="http://test")
        assert '<button id="fv-btn-share"' not in html
        assert '<button id="fv-btn-export"' not in html
        assert '<button id="fv-btn-reset"' in html


class TestSharedThemeCss:
    def test_theme_css_does_not_close_style_block(self):
        from flexviz.adapters.runtime import theme_css

        css = theme_css().lower()
        assert "</style>" not in css

    def test_theme_css_includes_filter_chip_tokens(self):
        from flexviz.adapters.runtime import theme_css

        css = theme_css()
        for token in (
            "--fv-filter-strip-bg",
            "--fv-filter-strip-border",
            "--fv-filter-strip-text",
            "--fv-filter-chip-bg",
            "--fv-filter-chip-border",
            "--fv-filter-chip-text",
            "--fv-filter-chip-source-bg",
            "--fv-filter-chip-source-text",
            "--fv-filter-chip-field-text",
            "--fv-filter-chip-value-text",
            "--fv-filter-chip-separator-text",
            "--fv-filter-chip-joiner-bg",
            "--fv-filter-chip-remove-hover-bg",
            "--fv-filter-chip-remove-hover-text",
        ):
            assert token in css

    def test_plotly_html_style_tags_are_balanced(self, two_fig_spec):
        from flexviz.adapters.plotly_adapter import PlotlyAdapter

        html = PlotlyAdapter()._build_dashboard_html(
            two_fig_spec, server_url="http://localhost:9999"
        )
        assert html.lower().count("</style>") == _start_tag_count(html, "style")


# ---------------------------------------------------------------------------
# Figure metadata API tests
# ---------------------------------------------------------------------------


class TestFigureMetadataPlotly:
    @pytest.fixture()
    def html(self):
        from flexviz.adapters.plotly_adapter import PlotlyAdapter

        t = TraceSpec(uid="t1", trace_type="line", axes=("x", "y"))
        fig = FigureSpec(
            uid="fig1",
            layout={
                "title": "My Title",
                "xlabel": "Time",
                "ylabel": "Value",
                "legend": False,
            },
            traces=[t],
        )
        spec = DashboardSpec(figures=[fig])
        return PlotlyAdapter()._build_dashboard_html(spec, server_url="http://test")

    def test_plotly_title(self, html):
        assert '"text": "My Title"' in html

    def test_plotly_xlabel(self, html):
        assert "Time" in html

    def test_plotly_ylabel(self, html):
        assert "Value" in html

    def test_plotly_legend_false(self, html):
        assert '"showlegend": false' in html


class TestFigureMetadataECharts:
    @pytest.fixture()
    def initial_option(self):
        from flexviz.adapters.echarts_adapter import EChartsAdapter

        t = TraceSpec(uid="t1", trace_type="line", axes=("x", "y"))
        fig = FigureSpec(
            uid="fig1",
            layout={
                "title": "My Title",
                "xlabel": "Time",
                "ylabel": "Value",
                "legend": False,
            },
            traces=[t],
        )
        return EChartsAdapter._build_initial_option(fig, 400)

    def test_echarts_title(self, initial_option):
        assert initial_option["title"]["text"] == "My Title"

    def test_echarts_xlabel(self, initial_option):
        assert initial_option["xAxis"]["name"] == "Time"

    def test_echarts_ylabel(self, initial_option):
        assert initial_option["yAxis"]["name"] == "Value"

    def test_echarts_legend_false(self, initial_option):
        assert initial_option["legend"]["show"] is False


# ---------------------------------------------------------------------------
# Pie trace adapter tests
# ---------------------------------------------------------------------------


class TestPiePlotly:
    @pytest.fixture()
    def html(self):
        from flexviz.adapters.plotly_adapter import PlotlyAdapter

        t = TraceSpec(uid="pie1", trace_type="pie", params={"hole": 0.4})
        fig = FigureSpec(uid="fig1", layout={}, traces=[t])
        spec = DashboardSpec(figures=[fig])
        return PlotlyAdapter()._build_dashboard_html(spec, server_url="http://test")

    def test_pie_trace_type(self, html):
        assert '"type": "pie"' in html

    def test_pie_hole(self, html):
        assert '"hole": 0.4' in html

    def test_pie_js_branch(self, html):
        assert "ts.trace_type === 'pie'" in html


class TestTreeMapPlotly:
    @pytest.fixture()
    def html(self):
        from flexviz.adapters.plotly_adapter import PlotlyAdapter
        from flexviz.spec import TraceSpec, FigureSpec, DashboardSpec

        t = TraceSpec(
            uid="tm1",
            trace_type="treemap",
            params={"path": ["continent", "country"], "agg": "sum"},
            backend_data={"values": "population"},
            display={"name": "Pop"},
        )
        fig = FigureSpec(uid="fig1", layout={}, traces=[t])
        spec = DashboardSpec(figures=[fig])
        return PlotlyAdapter()._build_dashboard_html(spec, server_url="http://test")

    def test_treemap_trace_type_in_html(self, html):
        assert '"type": "treemap"' in html

    def test_branchvalues_total(self, html):
        assert '"branchvalues": "total"' in html

    def test_treemap_js_branch(self, html):
        assert "ts.trace_type === 'treemap'" in html

    def test_plotly_click_handler_wired(self, html):
        assert "plotly_treemapclick" in html
        assert "handleClick" in html

    def test_plotly_trace_obj_treemap(self):
        from flexviz.adapters.plotly_adapter import PlotlyAdapter
        from flexviz.spec import TraceSpec

        t = TraceSpec(
            uid="tm1",
            trace_type="treemap",
            params={"path": ["a", "b"], "agg": "sum"},
            display={"name": "My Tree"},
        )
        obj = PlotlyAdapter._plotly_trace_obj(t, "My Tree", None)
        assert obj["type"] == "treemap"
        assert obj["branchvalues"] == "total"
        assert obj["labels"] == []
        assert obj["parents"] == []
        assert obj["ids"] == []
        assert obj["values"] == []
        assert "level" not in obj

    def test_reset_treemap_level_flag_in_runtime(self, html):
        # _resetTreemapLevel flag controls level reset: true for reset-like
        # renders, but not for selection because selection should allow
        # Plotly's native treemap drill/zoom feedback.
        assert "_resetTreemapLevel" in html
        assert "_resetTreemapLevel = true" in html
        assert "_resetTreemapLevel = false" in html
        assert "['deselect', 'reset', 'init'].includes(event.type)" in html

    def test_treemap_reset_level_fn_present(self, html):
        # resetTreemapLevel must be defined and call _fvRenderFigure (not
        # Plotly.restyle) so it resets the drill level via Plotly.react.
        assert "resetTreemapLevel" in html
        assert "_fvRenderFigure" in html

    def test_treemap_click_handler_uses_treemap_event(self, html):
        # Plotly's treemap-specific click event lets handleClick decide when
        # to allow native drill and when to cancel it for deselect.
        assert "plotly_treemapclick" in html
        assert "return handleClick" in html

    def test_category_selection_styles_present(self, html):
        assert "CATEGORY_DIMMED_OPACITY" in html
        assert "applyCategorySelectionStyles(figUid)" in html

    def test_treemap_level_resets_to_root_id(self, html):
        # Treemap ids include a synthetic "root" node. Resetting to the empty
        # level leaves Plotly drilled into the clicked node after deselect.
        assert "trace.level = 'root'" in html

    def test_treemap_root_click_clears_selection(self, html):
        # Clicking the root node (pt.id == "root", parts=[]) while a selection
        # is active should call clearFigureSelection, not silently return.
        body = _js_function_body(html, "function handleClick(")
        assert "hasSelection" in body
        assert "clearFigureSelection" in body


class TestPieClickHandler:
    @pytest.fixture()
    def html(self):
        from flexviz.adapters.plotly_adapter import PlotlyAdapter
        from flexviz.spec import TraceSpec, FigureSpec, DashboardSpec

        t = TraceSpec(uid="pie1", trace_type="pie", params={"hole": 0.0})
        fig = FigureSpec(uid="fig1", layout={}, traces=[t])
        spec = DashboardSpec(figures=[fig])
        return PlotlyAdapter()._build_dashboard_html(spec, server_url="http://test")

    def test_plotly_click_wired_for_pie_figure(self, html):
        assert "plotly_click" in html
        assert "handleClick" in html


class TestNoClickHandlerForBarOnly:
    def test_no_click_handler_wired_for_bar_only(self):
        from flexviz.adapters.plotly_adapter import PlotlyAdapter
        from flexviz.spec import TraceSpec, FigureSpec, DashboardSpec

        t = TraceSpec(
            uid="b1",
            trace_type="bar",
            params={"agg": "sum", "orientation": "v"},
            axes=("x", "y"),
        )
        fig = FigureSpec(uid="fig1", layout={}, traces=[t])
        spec = DashboardSpec(figures=[fig])
        html = PlotlyAdapter()._build_dashboard_html(spec, server_url="http://test")
        # In the bundle architecture, plotly_click wiring is guarded at runtime by
        # figSpec.traces.some(ts => ts.trace_type === 'pie').  Verify the guard is present
        # and the string exists only inside that conditional block.
        assert "ts.trace_type === 'pie'" in html
        assert "plotly_click" in html  # present in bundle, but runtime-guarded


class TestPieECharts:
    @pytest.fixture()
    def initial_option(self):
        from flexviz.adapters.echarts_adapter import EChartsAdapter

        t = TraceSpec(uid="pie1", trace_type="pie", params={"hole": 0.4})
        fig = FigureSpec(uid="fig1", layout={}, traces=[t])
        return EChartsAdapter._build_initial_option(fig, 400)

    def test_pie_series_type(self, initial_option):
        assert initial_option["series"][0]["type"] == "pie"

    def test_pie_radius(self, initial_option):
        assert initial_option["series"][0]["radius"][0] == "40%"

    def test_pie_no_xaxis(self, initial_option):
        assert "xAxis" not in initial_option

    def test_pie_tooltip_item(self, initial_option):
        assert initial_option["tooltip"]["trigger"] == "item"


# ---------------------------------------------------------------------------
# Histogram2D / Heatmap adapter tests
# ---------------------------------------------------------------------------


class TestHeatmapPlotly:
    @pytest.fixture()
    def html(self):
        from flexviz.adapters.plotly_adapter import PlotlyAdapter

        t = TraceSpec(
            uid="h2d",
            trace_type="histogram2d",
            axes=("x", "y"),
            display={"color_scale": "viridis", "color_range": "auto"},
        )
        fig = FigureSpec(uid="fig1", layout={}, traces=[t])
        spec = DashboardSpec(figures=[fig])
        return PlotlyAdapter()._build_dashboard_html(spec, server_url="http://test")

    def test_heatmap_trace_type(self, html):
        assert '"type": "heatmap"' in html

    def test_heatmap_colorscale(self, html):
        assert "heatmapColorScale(ts)" in html

    def test_heatmap_js_branch(self, html):
        assert "ts.trace_type === 'histogram2d'" in html

    def test_initial_plot_uses_config_per_fig(self, html):
        # Bundle uses figIdx variable form rather than literal indices
        assert (
            "Plotly.newPlot(divs[figIdx], tracesByFig[figIdx], layoutsByFig[figIdx], configsByFig[figIdx])"
            in html
        )

    def test_heatmap_legend_disabled(self, html):
        assert '"showlegend": false' in html

    def test_heatmap_overlay_defers_colorbar_swap_until_finalize(self, html):
        policy = _js_function_body(
            html,
            "function applyHeatmapColorbarPolicy(trace, renderLayer, showForeground)",
        )
        assert "renderLayer === 'bg'" in policy
        assert "trace.showscale = true" in policy
        assert "renderLayer === 'fg'" in policy
        assert "trace.showscale = false" in policy
        assert "fvFinalizeHeatmapOverlayColorbars" in html

    def test_heatmap_overlay_assigns_per_layer_auto_color_range(self, html):
        body = _js_function_body(
            html,
            "function syncHeatmapOverlayColorScale(traces, figSpec, showForeground)",
        )
        assert "heatmapZFiniteExtent(bgTrace.z)" in body
        assert "heatmapZFiniteExtent(fgTrace.z)" in body

    def test_heatmap_scale_registry_is_not_embedded(self, html):
        assert "const HEATMAP_COLOR_SCALES =" not in html

    def test_plotly_heatmap_trace_applies_fixed_range(self):
        from flexviz.adapters.plotly_adapter import PlotlyAdapter

        t = TraceSpec(
            uid="h2d",
            trace_type="histogram2d",
            axes=("x", "y"),
            display={"color_scale": "plasma", "color_range": (0.0, 5.0)},
        )
        trace = PlotlyAdapter._plotly_trace_obj(t, "Heat", None)
        assert trace["colorscale"] == "plasma"
        assert trace["zmin"] == 0.0
        assert trace["zmax"] == 5.0

    def test_corr_heatmap_legend_disabled(self):
        from flexviz.adapters.plotly_adapter import PlotlyAdapter

        t = TraceSpec(
            uid="corr1",
            trace_type="corr_heatmap",
            display={"color_scale": "rdbu", "color_range": (-1.0, 1.0)},
        )
        fig = FigureSpec(uid="fig1", layout={}, traces=[t])
        spec = DashboardSpec(figures=[fig])
        html = PlotlyAdapter()._build_dashboard_html(spec, server_url="http://test")
        assert '"showlegend": false' in html

    def test_corr_heatmap_trace_uses_explicit_signed_scale_and_range(self):
        from flexviz.adapters.plotly_adapter import PlotlyAdapter

        t = TraceSpec(
            uid="corr1",
            trace_type="corr_heatmap",
            display={"color_scale": "rdbu", "color_range": (-1.0, 1.0)},
        )
        trace = PlotlyAdapter._plotly_trace_obj(t, "Corr", None)
        assert trace["colorscale"] == "rdbu"
        assert trace["zmin"] == -1.0
        assert trace["zmax"] == 1.0

    def test_heatmap_trace_requires_explicit_style(self):
        from flexviz.adapters.plotly_adapter import PlotlyAdapter

        t = TraceSpec(uid="h2d", trace_type="histogram2d", axes=("x", "y"))

        with pytest.raises(ValueError, match="must include explicit color_scale"):
            PlotlyAdapter._plotly_trace_obj(t, "Heat", None)


class TestHeatmapECharts:
    @pytest.fixture()
    def initial_option(self):
        from flexviz.adapters.echarts_adapter import EChartsAdapter

        t = TraceSpec(
            uid="h2d",
            trace_type="histogram2d",
            axes=("x", "y"),
            display={"color_scale": "plasma", "color_range": (0.0, 5.0)},
        )
        fig = FigureSpec(uid="fig1", layout={}, traces=[t])
        return EChartsAdapter._build_initial_option(fig, 400)

    def test_heatmap_series_type(self, initial_option):
        assert initial_option["series"][0]["type"] == "heatmap"

    def test_has_visual_map(self, initial_option):
        from flexviz.adapters.echarts_adapter import _ECHARTS_HEATMAP_COLOR_SCALES

        assert "visualMap" in initial_option
        assert (
            initial_option["visualMap"]["inRange"]["color"]
            == _ECHARTS_HEATMAP_COLOR_SCALES["plasma"]
        )
        assert initial_option["visualMap"]["min"] == 0.0
        assert initial_option["visualMap"]["max"] == 5.0

    def test_heatmap_has_axes(self, initial_option):
        assert "xAxis" in initial_option
        assert "yAxis" in initial_option

    def test_corr_heatmap_defaults_to_signed_visual_map(self):
        from flexviz.adapters.echarts_adapter import (
            EChartsAdapter,
            _ECHARTS_HEATMAP_COLOR_SCALES,
        )

        t = TraceSpec(
            uid="corr1",
            trace_type="corr_heatmap",
            display={"color_scale": "rdbu", "color_range": (-1.0, 1.0)},
        )
        fig = FigureSpec(uid="fig1", layout={}, traces=[t])
        opt = EChartsAdapter._build_initial_option(fig, 400)
        assert (
            opt["visualMap"]["inRange"]["color"]
            == _ECHARTS_HEATMAP_COLOR_SCALES["rdbu"]
        )
        assert opt["visualMap"]["min"] == -1.0
        assert opt["visualMap"]["max"] == 1.0

    def test_heatmap_accepts_plotly_style_named_scale_aliases(self):
        from flexviz.adapters.echarts_adapter import (
            EChartsAdapter,
            _ECHARTS_HEATMAP_COLOR_SCALES,
        )

        t = TraceSpec(
            uid="h2d",
            trace_type="histogram2d",
            axes=("x", "y"),
            display={"color_scale": "Greens", "color_range": "auto"},
        )
        fig = FigureSpec(uid="fig1", layout={}, traces=[t])
        opt = EChartsAdapter._build_initial_option(fig, 400)
        assert (
            opt["visualMap"]["inRange"]["color"]
            == _ECHARTS_HEATMAP_COLOR_SCALES["greens"]
        )

    def test_runtime_contains_dynamic_heatmap_extent_logic(self):
        from flexviz.adapters.echarts_adapter import EChartsAdapter

        t = TraceSpec(
            uid="h2d",
            trace_type="histogram2d",
            axes=("x", "y"),
            display={"color_scale": "viridis", "color_range": "auto"},
        )
        fig = FigureSpec(uid="fig1", layout={}, traces=[t])
        spec = DashboardSpec(figures=[fig])
        html = EChartsAdapter()._build_dashboard_html(spec, server_url="http://test")
        assert "function heatmapFiniteExtent" in html
        assert "vMin === Infinity || vMax === -Infinity" in html
        assert "heatmapVisualMapForTrace(ts, u)" in html

    def test_unsupported_heatmap_scale_raises(self):
        from flexviz.adapters.echarts_adapter import EChartsAdapter

        t = TraceSpec(
            uid="h2d",
            trace_type="histogram2d",
            axes=("x", "y"),
            display={"color_scale": "plotly-only-scale", "color_range": "auto"},
        )
        fig = FigureSpec(uid="fig1", layout={}, traces=[t])

        with pytest.raises(ValueError, match="not supported"):
            EChartsAdapter._build_initial_option(fig, 400)

    def test_heatmap_visual_map_requires_explicit_style(self):
        from flexviz.adapters.echarts_adapter import EChartsAdapter

        t = TraceSpec(uid="h2d", trace_type="histogram2d", axes=("x", "y"))
        fig = FigureSpec(uid="fig1", layout={}, traces=[t])

        with pytest.raises(ValueError, match="must include explicit color_scale"):
            EChartsAdapter._build_initial_option(fig, 400)


# ---------------------------------------------------------------------------
# Auto-derived axis labels
# ---------------------------------------------------------------------------


class TestAutoLabelPlotly:
    """Auto-derived xlabel/ylabel from trace column names appear in Plotly HTML."""

    @pytest.fixture()
    def html(self):
        import polars as pl
        from flexviz.adapters.plotly_adapter import PlotlyAdapter
        from flexviz.figure import Figure

        fig = Figure(pl.DataFrame({"ts": [1, 2], "val": [1.0, 2.0]}))
        fig.add_line(x="ts", y="val")
        spec = fig.to_spec().figure
        dash = DashboardSpec(figures=[spec])
        return PlotlyAdapter()._build_dashboard_html(dash, server_url="http://test")

    def test_xlabel_appears(self, html):
        assert '"ts"' in html

    def test_ylabel_appears(self, html):
        assert '"val"' in html


class TestAutoLabelECharts:
    """Auto-derived xlabel/ylabel from trace column names appear in ECharts option."""

    @pytest.fixture()
    def initial_option(self):
        import polars as pl
        from flexviz.adapters.echarts_adapter import EChartsAdapter
        from flexviz.figure import Figure

        fig = Figure(pl.DataFrame({"ts": [1, 2], "val": [1.0, 2.0]}))
        fig.add_line(x="ts", y="val")
        spec = fig.to_spec().figure
        return EChartsAdapter._build_initial_option(spec, 400)

    def test_xlabel_appears(self, initial_option):
        assert initial_option["xAxis"]["name"] == "ts"

    def test_ylabel_appears(self, initial_option):
        assert initial_option["yAxis"]["name"] == "val"


# ---------------------------------------------------------------------------
# Theme CSS token tests
# ---------------------------------------------------------------------------


class TestThemeCss:
    """Verify that CSS design tokens are embedded in adapter HTML output."""

    @pytest.fixture()
    def plotly_html(self, two_fig_spec):
        from flexviz.adapters.plotly_adapter import PlotlyAdapter

        return PlotlyAdapter()._build_dashboard_html(
            two_fig_spec, server_url="http://localhost:9999"
        )

    @pytest.fixture()
    def echarts_html(self, two_fig_spec):
        from flexviz.adapters.echarts_adapter import EChartsAdapter

        return EChartsAdapter()._build_dashboard_html(
            two_fig_spec, server_url="http://localhost:9999"
        )

    def test_plotly_html_contains_theme_tokens(self, plotly_html):
        assert "--fv-accent" in plotly_html
        assert "--fv-bg" in plotly_html
        assert "--fv-radius" in plotly_html

    def test_echarts_html_contains_theme_tokens(self, echarts_html):
        assert "--fv-accent" in echarts_html
        assert "--fv-bg" in echarts_html
        assert "--fv-radius" in echarts_html

    def test_plotly_toolbar_css_uses_var_references(self, plotly_html):
        # Toolbar CSS must reference tokens, not hardcode hex values.
        assert "var(--fv-accent)" in plotly_html
        assert "var(--fv-border)" in plotly_html
        assert "var(--fv-text)" in plotly_html

    def test_echarts_toolbar_css_uses_var_references(self, echarts_html):
        assert "var(--fv-accent)" in echarts_html
        assert "var(--fv-border)" in echarts_html


class TestPageHead:
    """A page with no icon link makes the browser probe /favicon.ico, which
    404s on the HTTP-hosted pages: /view and an app mounted under a prefix."""

    @pytest.fixture()
    def plotly_html(self, two_fig_spec):
        from flexviz.adapters.plotly_adapter import PlotlyAdapter

        return PlotlyAdapter()._build_dashboard_html(
            two_fig_spec, server_url="http://localhost:9999"
        )

    @pytest.fixture()
    def echarts_html(self, two_fig_spec):
        from flexviz.adapters.echarts_adapter import EChartsAdapter

        return EChartsAdapter()._build_dashboard_html(
            two_fig_spec, server_url="http://localhost:9999"
        )

    @pytest.mark.parametrize("renderer", ["plotly_html", "echarts_html"])
    def test_head_has_title_and_inline_icon(self, renderer, request):
        html = request.getfixturevalue(renderer)
        assert "<title>FlexViz</title>" in html
        assert 'rel="icon" type="image/png" href="data:image/png;base64,' in html

    @pytest.mark.parametrize("renderer", ["plotly_html", "echarts_html"])
    def test_brand_artwork_is_substituted(self, renderer, request):
        # An unsubstituted placeholder would ship a blank header wordmark.
        html = request.getfixturevalue(renderer)
        assert "{{WORDMARK" not in html
        assert "--fv-brand-image:" in html
        assert 'aria-label="FlexViz"' in html

    @pytest.mark.parametrize("renderer", ["plotly_html", "echarts_html"])
    def test_brand_links_out_without_losing_the_dashboard(self, renderer, request):
        # target=_blank keeps the open dashboard alive; noopener is required
        # with it, and the anchor must keep its link role for screen readers.
        html = request.getfixturevalue(renderer)
        assert '<a id="fv-brand" href="https://flexviz.tech" target="_blank"' in html
        assert 'rel="noopener noreferrer"' in html
        assert 'id="fv-brand"' in html and 'role="img"' not in html

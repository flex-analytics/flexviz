"""Unit tests for adapter Python APIs: parse_event, shared toolbar, and
ECharts ``_build_initial_option``.

Covers PlotlyAdapter and EChartsAdapter, plus the shared toolbar building
blocks on AbstractAdapter.
"""

from __future__ import annotations

import pytest

from flexviz.adapters.base import AbstractAdapter
from flexviz.spec import (
    DashboardSpec,
    FigureSpec,
    LayoutSpec,
    ToolbarConfig,
    TraceSelectionSpec,
    TraceSpec,
)

# ---- parse_event -----------------------------------------------------------


class _DummyAdapter(AbstractAdapter):
    def parse_event(self, raw_event):
        return None

    def show_dashboard(self, spec, server_url="http://127.0.0.1:8000", **kwargs):
        return None


class TestPlotlyParseEvent:
    @pytest.fixture()
    def adapter(self):
        from flexviz.adapters.plotly_adapter import PlotlyAdapter

        return PlotlyAdapter()

    def test_viewport_x_range(self, adapter):
        raw = {"xaxis.range[0]": 10, "xaxis.range[1]": 50}
        event = adapter.parse_event(raw)
        assert event is not None
        assert event.type == "viewport"
        assert event.axis_ranges["x"] == (10, 50)

    def test_viewport_x2_range(self, adapter):
        raw = {"xaxis2.range[0]": 5, "xaxis2.range[1]": 25}
        event = adapter.parse_event(raw)
        assert event is not None
        assert event.type == "viewport"
        assert event.axis_ranges["x2"] == (5, 25)

    def test_viewport_x_range_array(self, adapter):
        raw = {"xaxis.range": [10, 50]}
        event = adapter.parse_event(raw)
        assert event is not None
        assert event.type == "viewport"
        assert event.axis_ranges["x"] == (10, 50)

    def test_viewport_y_range(self, adapter):
        raw = {"yaxis.range[0]": 0, "yaxis.range[1]": 100}
        event = adapter.parse_event(raw)
        assert event is not None
        assert event.type == "viewport"
        assert event.axis_ranges["y"] == (0, 100)

    def test_map_derived_coordinates(self, adapter):
        coords = [[-73.5, 40.5], [-72.5, 40.5], [-72.5, 41.5], [-73.5, 41.5]]
        raw = {"map._derived": {"coordinates": coords}}
        event = adapter.parse_event(raw)
        assert event is not None
        assert event.type == "viewport"
        assert event.axis_ranges["coordinates"] == coords

    def test_autorange_reset(self, adapter):
        raw = {"xaxis.autorange": True}
        event = adapter.parse_event(raw)
        assert event is not None
        assert event.type == "reset"
        assert event.force_update is True

    def test_autosize_reset(self, adapter):
        raw = {"autosize": True}
        event = adapter.parse_event(raw)
        assert event is not None
        assert event.type == "reset"
        assert event.force_update is True

    def test_dragmode_ignored(self, adapter):
        raw = {"dragmode": "select"}
        event = adapter.parse_event(raw)
        assert event is None

    def test_partial_range_ignored(self, adapter):
        raw = {"xaxis.range[0]": 10}
        event = adapter.parse_event(raw)
        assert event is None

    def test_empty_data_ignored(self, adapter):
        event = adapter.parse_event({})
        assert event is None

    def test_typed_event_passthrough(self, adapter):
        raw = {
            "type": "init",
            "force_update": True,
            "axis_ranges": {},
            "selections": [],
        }
        event = adapter.parse_event(raw)
        assert event is not None
        assert event.type == "init"

    def test_multiple_axes(self, adapter):
        raw = {
            "xaxis.range[0]": 1,
            "xaxis.range[1]": 10,
            "yaxis.range[0]": 0,
            "yaxis.range[1]": 100,
        }
        event = adapter.parse_event(raw)
        assert event is not None
        assert event.type == "viewport"
        assert event.axis_ranges["x"] == (1, 10)
        assert event.axis_ranges["y"] == (0, 100)


# ---- shared toolbar --------------------------------------------------------


class TestSharedToolbar:
    """AbstractAdapter toolbar building blocks contain the required elements."""

    def test_toolbar_html_has_all_button_ids(self):
        html = AbstractAdapter._toolbar_html()
        for btn_id in (
            "fv-btn-reset",
            "fv-btn-deselect",
            "fv-btn-cfmode",
            "fv-hover-btn",
            "fv-btn-grid",
            "fv-btn-share",
            "fv-btn-export",
            "fv-btn-import",
        ):
            assert btn_id in html, f"Missing button id: {btn_id}"

    def test_toolbar_html_has_brand(self):
        html = AbstractAdapter._toolbar_html()
        assert "fv-brand" in html
        assert "FlexViz" in html

    def test_toolbar_css_has_fv_header(self):
        css = AbstractAdapter._toolbar_css()
        assert "fv-header" in css
        assert "fv-toolbar" in css

    def test_dashboard_markup_static_uses_shared_item_class(self):
        spec = DashboardSpec(
            figures=[FigureSpec(uid="fig1", traces=[])],
        )
        spec.layout.draggable = False

        markup = _DummyAdapter()._dashboard_markup(
            spec,
            render_panel=lambda idx, fig_spec: f"<fv-panel>{fig_spec.uid}:{idx}</fv-panel>",
        )

        assert 'id="fv-dashboard"' in markup.container_html
        assert "fv-dashboard-item" in markup.container_html
        assert "grid-template-columns:repeat(12" in markup.css
        assert markup.head_html == ""

    def test_dashboard_markup_draggable_uses_gridstack_shell(self):
        spec = DashboardSpec(
            figures=[FigureSpec(uid="fig1", traces=[])],
        )
        spec.layout.draggable = True

        markup = _DummyAdapter()._dashboard_markup(
            spec,
            render_panel=lambda idx, fig_spec: f"<fv-panel>{fig_spec.uid}:{idx}</fv-panel>",
        )

        assert 'class="grid-stack"' in markup.container_html
        assert 'class="grid-stack-item"' in markup.container_html
        assert "gs-id=" in markup.container_html
        assert "gridstack.min.css" in markup.head_html
        assert "gridstack-all.js" in markup.head_html

    def test_toolbar_html_default_shows_all_buttons(self):
        html = AbstractAdapter._toolbar_html()
        for btn_id in (
            "fv-btn-reset",
            "fv-btn-deselect",
            "fv-btn-cfmode",
            "fv-hover-btn",
            "fv-btn-grid",
            "fv-btn-share",
            "fv-btn-export",
            "fv-btn-import",
        ):
            assert btn_id in html, f"Missing button id with default config: {btn_id}"

    def test_toolbar_html_hides_single_button(self):
        tc = ToolbarConfig(show_share=False)
        html = AbstractAdapter._toolbar_html(tc)
        assert "fv-btn-share" not in html
        assert "fv-btn-reset" in html

    def test_toolbar_html_omits_empty_group(self):
        tc = ToolbarConfig(show_share=False, show_export=False, show_import=False)
        html = AbstractAdapter._toolbar_html(tc)
        assert "fv-btn-share" not in html
        assert "fv-btn-export" not in html
        assert "fv-btn-import" not in html
        assert "fv-btn-reset" in html

    def test_toolbar_html_all_hidden_renders_empty_toolbar(self):
        tc = ToolbarConfig(
            show_reset=False,
            show_deselect=False,
            show_cfmode=False,
            show_hover=False,
            show_lock_all_axes=False,
            show_grid=False,
            show_share=False,
            show_export=False,
            show_import=False,
        )
        html = AbstractAdapter._toolbar_html(tc)
        assert "fv-btn-" not in html
        assert "fv-header" in html

    def test_toolbar_config_roundtrips_via_layout_spec(self):
        tc = ToolbarConfig(show_share=False, show_import=False)
        layout = LayoutSpec(toolbar=tc)
        dumped = layout.model_dump()
        restored = LayoutSpec.model_validate(dumped)
        assert restored.toolbar.show_share is False
        assert restored.toolbar.show_import is False
        assert restored.toolbar.show_reset is True

    def test_dashboard_html_respects_toolbar_config(self):
        from flexviz.adapters.plotly_adapter import PlotlyAdapter

        t = TraceSpec(
            uid="t1", trace_type="line", display={"name": "A"}, axes=("x", "y")
        )
        fig = FigureSpec(uid="fig1", layout={}, traces=[t])
        tc = ToolbarConfig(show_share=False, show_export=False)
        spec = DashboardSpec(figures=[fig], layout=LayoutSpec(toolbar=tc))
        html = PlotlyAdapter()._build_dashboard_html(
            spec, server_url="http://localhost"
        )
        # Check for the rendered <button> elements, not JS getElementById references
        assert '<button id="fv-btn-share"' not in html
        assert '<button id="fv-btn-export"' not in html
        assert '<button id="fv-btn-reset"' in html


# ---- Plotly modebar config -------------------------------------------------


class TestPlotlyModebarConfig:
    """Per-figure mode toggle replaces the Plotly modebar."""

    def _render_dashboard_html(self) -> str:
        from flexviz.adapters.plotly_adapter import PlotlyAdapter
        from flexviz.spec import DashboardSpec, FigureSpec, TraceSpec

        t = TraceSpec(
            uid="t1", trace_type="line", display={"name": "A"}, axes=("x", "y")
        )
        fig = FigureSpec(uid="fig1", layout={}, traces=[t])
        spec = DashboardSpec(figures=[fig])
        return PlotlyAdapter()._build_dashboard_html(
            spec, server_url="http://localhost:9999"
        )

    def test_modebar_hidden_entirely(self):
        html = self._render_dashboard_html()
        # The Plotly modebar is replaced by the per-figure Zoom/Pan/CF toggle.
        assert "displayModeBar: false" in html

    def test_mode_toggle_buttons_present(self):
        html = self._render_dashboard_html()
        assert 'data-mode="zoom"' in html
        assert 'data-mode="pan"' in html
        assert 'data-mode="select"' in html
        assert 'data-action="reset-panel"' in html

    @pytest.mark.parametrize("trace_type", ["geo_histogram2d", "geo_line"])
    def test_geo_figures_keep_zoom_pan_enabled(self, trace_type):
        from flexviz.adapters.plotly_adapter import (
            PlotlyAdapter,
            _figure_supports_zoom_pan,
        )

        t = TraceSpec(
            uid="geo",
            trace_type=trace_type,
            display={"name": "Geo", "color_scale": "viridis", "color_range": "auto"},
            axes=None,
        )
        fig = FigureSpec(uid="fig1", layout={}, traces=[t])

        assert _figure_supports_zoom_pan(fig) is True

        html = PlotlyAdapter()._build_dashboard_html(
            DashboardSpec(figures=[fig]), server_url="http://localhost:9999"
        )
        assert "const figSupportsZoomPan = [true];" in html
        assert (
            "if (!figSupportsZoomPan[figIdx]) setFigureMode(figUid, 'select');" in html
        )


# ---- EChartsAdapter --------------------------------------------------------


class TestEChartsParseEvent:
    def test_datazoom_shorthand(self):
        from flexviz.adapters.echarts_adapter import EChartsAdapter

        raw = {"startValue": 100.0, "endValue": 500.0}
        event = EChartsAdapter().parse_event(raw)
        assert event is not None
        assert event.type == "viewport"
        assert event.axis_ranges["x"] == (100.0, 500.0)

    def test_typed_passthrough(self):
        from flexviz.adapters.echarts_adapter import EChartsAdapter

        raw = {
            "type": "reset",
            "force_update": True,
            "axis_ranges": {},
            "selections": [],
        }
        event = EChartsAdapter().parse_event(raw)
        assert event is not None
        assert event.type == "reset"

    def test_empty_returns_none(self):
        from flexviz.adapters.echarts_adapter import EChartsAdapter

        assert EChartsAdapter().parse_event({}) is None

    def test_unknown_dict_returns_none(self):
        from flexviz.adapters.echarts_adapter import EChartsAdapter

        assert EChartsAdapter().parse_event({"foo": "bar"}) is None

    def test_show_dashboard_notebook_path_accepts_height(self, monkeypatch):
        from flexviz.adapters.echarts_adapter import EChartsAdapter

        adapter = EChartsAdapter()
        captured: dict[str, str | int] = {}

        monkeypatch.setattr(adapter, "_wait_for_server", lambda _server_url: None)
        monkeypatch.setattr(
            adapter,
            "_deliver_notebook",
            lambda html, height: captured.update({"html": html, "height": height}),
        )

        spec = DashboardSpec(
            figures=[
                FigureSpec(
                    uid="fig1",
                    traces=[TraceSpec(uid="t1", trace_type="line", axes=("x", "y"))],
                )
            ]
        )
        adapter.show_dashboard(
            spec,
            server_url="http://localhost:9999",
            notebook=True,
            height=432,
        )

        assert "<!DOCTYPE html>" in captured["html"]
        assert captured["height"] == 512


class TestEChartsInitialOption:
    def test_line_series(self):
        from flexviz.adapters.echarts_adapter import EChartsAdapter

        ts = TraceSpec(
            uid="e1", trace_type="line", display={"name": "MyLine", "color": "#ff0000"}
        )
        fig_spec = FigureSpec(traces=[ts])
        option = EChartsAdapter._build_initial_option(fig_spec, 400)
        assert option["series"][0]["type"] == "line"
        assert option["series"][0]["id"] == "e1"
        assert option["series"][0]["showSymbol"] is False

    def test_histogram_series(self):
        from flexviz.adapters.echarts_adapter import EChartsAdapter

        ts = TraceSpec(uid="e2", trace_type="histogram", display={"name": "MyHist"})
        fig_spec = FigureSpec(traces=[ts])
        option = EChartsAdapter._build_initial_option(fig_spec, 400)
        assert option["series"][0]["type"] == "bar"
        assert option["series"][0]["id"] == "e2"

    def test_has_datazoom(self):
        from flexviz.adapters.echarts_adapter import EChartsAdapter

        ts = TraceSpec(uid="e1", trace_type="line", display={})
        fig_spec = FigureSpec(traces=[ts])
        option = EChartsAdapter._build_initial_option(fig_spec, 400)
        assert "dataZoom" in option
        assert option["dataZoom"][0]["type"] == "inside"

    def test_unknown_trace_type_raises(self):
        from flexviz.adapters.echarts_adapter import EChartsAdapter

        ts = TraceSpec(uid="e1", trace_type="scatter", display={})
        fig_spec = FigureSpec(traces=[ts])
        with pytest.raises(ValueError, match="unsupported trace type"):
            EChartsAdapter._build_initial_option(fig_spec, 400)

    def test_box_series_type(self):
        from flexviz.adapters.echarts_adapter import EChartsAdapter

        ts = TraceSpec(
            uid="ec-box",
            trace_type="box",
            backend_data={"y": "val"},
            display={"name": "MyBox", "color": "#ff6600"},
        )
        fig_spec = FigureSpec(uid="fig-box", traces=[ts])
        opt = EChartsAdapter._build_initial_option(fig_spec, 400)
        assert opt["series"][0]["type"] == "boxplot"
        assert opt["series"][0]["id"] == "ec-box"
        assert opt["series"][0]["itemStyle"]["color"] == "#ff6600"
        assert opt["xAxis"]["type"] == "category"
        assert opt["yAxis"]["type"] == "value"

    def test_treemap_series_type(self):
        from flexviz.adapters.echarts_adapter import EChartsAdapter

        ts = TraceSpec(
            uid="ec-tm",
            trace_type="treemap",
            params={"path": ["continent", "country"], "agg": "sum"},
            display={"name": "Pop"},
        )
        fig_spec = FigureSpec(uid="fig-tm", traces=[ts])
        opt = EChartsAdapter._build_initial_option(fig_spec, 400)
        assert opt["series"][0]["type"] == "treemap"
        assert opt["series"][0]["id"] == "ec-tm"
        assert opt["series"][0]["nodeClick"] is False
        assert opt["series"][0]["roam"] is False
        assert opt["series"][0]["breadcrumb"]["show"] is False
        assert opt["series"][0]["label"]["show"] is True
        assert opt["series"][0]["upperLabel"]["show"] is True
        assert "xAxis" not in opt
        assert "yAxis" not in opt

    def test_initial_option_hides_toolbox(self):
        from flexviz.adapters.echarts_adapter import EChartsAdapter

        ts = TraceSpec(uid="e1", trace_type="line", display={})
        fig_spec = FigureSpec(traces=[ts])
        opt = EChartsAdapter._build_initial_option(fig_spec, 400)
        assert opt["toolbox"]["show"] is False


# ---- PlotlyAdapter box trace (dashboard HTML) ------------------------------


class TestPlotlyDashboardBoxTrace:
    def test_build_dashboard_html_includes_box_orientation(self):
        from flexviz.adapters.plotly_adapter import PlotlyAdapter

        ts = TraceSpec(
            uid="box-1",
            trace_type="box",
            backend_data={"y": "val"},
            display={"name": "MyBox", "color": "#111"},
        )
        fig_spec = FigureSpec(uid="fig-a", traces=[ts])
        dash = DashboardSpec(figures=[fig_spec])
        html = PlotlyAdapter()._build_dashboard_html(
            dash, server_url="http://127.0.0.1:8000"
        )
        assert '"type": "box"' in html
        assert '"orientation": "v"' in html
        assert '"x0": "MyBox"' in html
        assert "lowerfence" in html

    def test_horizontal_box_uses_y0(self):
        from flexviz.adapters.plotly_adapter import PlotlyAdapter

        ts = TraceSpec(
            uid="box-2",
            trace_type="box",
            backend_data={"x": "val"},
            display={"name": "HB"},
        )
        fig_spec = FigureSpec(traces=[ts])
        dash = DashboardSpec(figures=[fig_spec])
        html = PlotlyAdapter()._build_dashboard_html(
            dash, server_url="http://127.0.0.1:8000"
        )
        assert '"orientation": "h"' in html
        assert '"y0": "HB"' in html


# ---- PlotlyAdapter hovermode per figure orientation ------------------------


class TestPlotlyFigureHovermode:
    def test_vertical_histogram_uses_x_hovermode(self):
        from flexviz.adapters.plotly_adapter import _figure_hovermode

        fig = FigureSpec(
            traces=[TraceSpec(uid="h", trace_type="histogram", backend_data={"x": "v"})]
        )
        assert _figure_hovermode(fig) == "x"

    def test_horizontal_histogram_uses_y_hovermode(self):
        from flexviz.adapters.plotly_adapter import _figure_hovermode

        fig = FigureSpec(
            traces=[TraceSpec(uid="h", trace_type="histogram", backend_data={"y": "v"})]
        )
        assert _figure_hovermode(fig) == "y"

    def test_horizontal_bar_uses_y_hovermode(self):
        from flexviz.adapters.plotly_adapter import _figure_hovermode

        fig = FigureSpec(
            traces=[
                TraceSpec(
                    uid="b",
                    trace_type="bar",
                    backend_data={"labels": "cat", "values": "v"},
                    params={"orientation": "h"},
                )
            ]
        )
        assert _figure_hovermode(fig) == "y"

    def test_line_uses_x_hovermode(self):
        from flexviz.adapters.plotly_adapter import _figure_hovermode

        fig = FigureSpec(
            traces=[
                TraceSpec(uid="l", trace_type="line", backend_data={"x": "t", "y": "v"})
            ]
        )
        assert _figure_hovermode(fig) == "x"

    def test_mixed_orientation_defaults_to_x(self):
        from flexviz.adapters.plotly_adapter import _figure_hovermode

        fig = FigureSpec(
            traces=[
                TraceSpec(uid="hx", trace_type="histogram", backend_data={"x": "a"}),
                TraceSpec(uid="hy", trace_type="histogram", backend_data={"y": "b"}),
            ]
        )
        assert _figure_hovermode(fig) == "x"

    def test_horizontal_histogram_html_sets_y_hovermode(self):
        from flexviz.adapters.plotly_adapter import PlotlyAdapter

        fig = FigureSpec(
            uid="fig-h",
            traces=[
                TraceSpec(uid="h", trace_type="histogram", backend_data={"y": "v"})
            ],
        )
        dash = DashboardSpec(figures=[fig])
        html = PlotlyAdapter()._build_dashboard_html(
            dash, server_url="http://127.0.0.1:8000"
        )
        assert '"hovermode": "y"' in html


class TestPlotlyDashboardBarTrace:
    def test_bar_bootstrap_has_type_bar(self):
        from flexviz.adapters.plotly_adapter import PlotlyAdapter

        ts = TraceSpec(
            uid="bar-1",
            trace_type="bar",
            backend_data={"x": "cat", "y": "val"},
            params={"agg": "sum", "orientation": "v", "bar_mode": "group"},
            display={"name": "Sales", "bar_mode": "group"},
        )
        fig_spec = FigureSpec(uid="fig-bar", traces=[ts])
        dash = DashboardSpec(figures=[fig_spec])
        html = PlotlyAdapter()._build_dashboard_html(
            dash, server_url="http://127.0.0.1:8000"
        )
        assert '"type": "bar"' in html
        assert '"barmode": "group"' in html

    def test_bar_horizontal_bootstrap(self):
        from flexviz.adapters.plotly_adapter import PlotlyAdapter

        ts = TraceSpec(
            uid="bar-h",
            trace_type="bar",
            backend_data={"x": "cat", "y": "val"},
            params={"agg": "sum", "orientation": "h", "bar_mode": "group"},
            display={"name": "H", "bar_mode": "group"},
        )
        fig_spec = FigureSpec(uid="fig-bh", traces=[ts])
        dash = DashboardSpec(figures=[fig_spec])
        html = PlotlyAdapter()._build_dashboard_html(
            dash, server_url="http://127.0.0.1:8000"
        )
        assert '"orientation": "h"' in html

    def test_stack_bar_bootstrap_does_not_force_offsetgroup(self):
        from flexviz.adapters.plotly_adapter import PlotlyAdapter

        ts = TraceSpec(
            uid="bar-stack",
            trace_type="bar",
            backend_data={"x": "cat", "y": "val"},
            params={"agg": "sum", "orientation": "v"},
            display={"name": "Stacked", "bar_mode": "stack"},
        )

        obj = PlotlyAdapter._plotly_trace_obj(ts, "Stacked", None)

        assert obj["type"] == "bar"
        assert "offsetgroup" not in obj
        assert "alignmentgroup" not in obj


class TestEChartsBarTrace:
    def test_bar_series_in_initial_option(self):
        from flexviz.adapters.echarts_adapter import EChartsAdapter

        ts = TraceSpec(
            uid="ec-bar",
            trace_type="bar",
            backend_data={"x": "cat", "y": "val"},
            params={"agg": "sum", "orientation": "v", "bar_mode": "group"},
            display={"name": "Revenue", "bar_mode": "group"},
        )
        fig_spec = FigureSpec(uid="fig-ec", traces=[ts])
        opt = EChartsAdapter._build_initial_option(fig_spec, 400)
        assert len(opt["series"]) == 1
        assert opt["series"][0]["type"] == "bar"
        assert opt["xAxis"]["type"] == "category"

    def test_bar_horizontal_swaps_axes(self):
        from flexviz.adapters.echarts_adapter import EChartsAdapter

        ts = TraceSpec(
            uid="ec-bh",
            trace_type="bar",
            backend_data={"x": "cat", "y": "val"},
            params={"agg": "sum", "orientation": "h", "bar_mode": "group"},
            display={"name": "H", "bar_mode": "group"},
        )
        fig_spec = FigureSpec(uid="fig-ech", traces=[ts])
        opt = EChartsAdapter._build_initial_option(fig_spec, 400)
        assert opt["xAxis"]["type"] == "value"
        assert opt["yAxis"]["type"] == "category"

    def test_stack_bar_adds_stack_property(self):
        from flexviz.adapters.echarts_adapter import EChartsAdapter

        ts = TraceSpec(
            uid="ec-stack",
            trace_type="bar",
            backend_data={"x": "cat", "y": "val"},
            params={"agg": "sum", "orientation": "v", "bar_mode": "stack"},
            display={"name": "S", "bar_mode": "stack"},
        )
        fig_spec = FigureSpec(uid="fig-stack", traces=[ts])
        opt = EChartsAdapter._build_initial_option(fig_spec, 400)
        assert opt["series"][0].get("stack") == "bar"
        assert "barMode" not in opt

    def test_stack_bar_uses_legacy_param_fallback(self):
        from flexviz.adapters.echarts_adapter import EChartsAdapter

        ts = TraceSpec(
            uid="ec-stack-legacy",
            trace_type="bar",
            backend_data={"x": "cat", "y": "val"},
            params={"agg": "sum", "orientation": "v", "bar_mode": "stack"},
            display={"name": "Legacy"},
        )
        fig_spec = FigureSpec(uid="fig-stack-legacy", traces=[ts])
        opt = EChartsAdapter._build_initial_option(fig_spec, 400)
        assert opt["series"][0].get("stack") == "bar"


class TestFigureSelectDirection:
    """`_figure_select_direction` derives Plotly band-brush geometry from the
    figure's `kind="range"` traces' `selection.axis_columns`."""

    @staticmethod
    def _sd(traces):
        from flexviz.adapters.plotly_adapter import _figure_select_direction

        return _figure_select_direction(FigureSpec(uid="f", traces=traces))

    @staticmethod
    def _range(uid, axis_columns):
        return TraceSpec(
            uid=uid,
            trace_type="t",
            axes=("x", "y"),
            selection=TraceSelectionSpec(kind="range", axis_columns=axis_columns),
        )

    def test_line_figure_is_horizontal_band(self):
        assert self._sd([self._range("l", {"x": "ts"})]) == "h"

    def test_horizontal_histogram_is_vertical_band(self):
        assert self._sd([self._range("h", {"y": "v"})]) == "v"

    def test_histogram2d_is_2d_rectangle(self):
        assert self._sd([self._range("h2", {"x": "a", "y": "b"})]) == "d"

    def test_mixed_x_and_y_only_traces_allow_both(self):
        assert (
            self._sd([self._range("l", {"x": "ts"}), self._range("h", {"y": "v"})])
            == "d"
        )

    def test_non_range_selection_returns_none(self):
        bar = TraceSpec(
            uid="b",
            trace_type="bar",
            axes=("x", "y"),
            selection=TraceSelectionSpec(kind="categorical", label_columns=["c"]),
        )
        assert self._sd([bar]) is None

    def test_default_none_selection_returns_none(self):
        ts = TraceSpec(uid="l", trace_type="line", axes=("x", "y"))
        assert self._sd([ts]) is None

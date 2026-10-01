"""Unit tests for adapter Python APIs.

Covers PlotlyAdapter, plus the shared toolbar building blocks on
AbstractAdapter.
"""

from __future__ import annotations

import re

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

# ---- shared toolbar --------------------------------------------------------


class _DummyAdapter(AbstractAdapter):
    def show_dashboard(self, spec, server_url="http://127.0.0.1:8000", **kwargs):
        return None


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
            render_panel=lambda idx, fig_spec: (
                f"<fv-panel>{fig_spec.uid}:{idx}</fv-panel>"
            ),
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
            render_panel=lambda idx, fig_spec: (
                f"<fv-panel>{fig_spec.uid}:{idx}</fv-panel>"
            ),
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

    def test_toolbar_html_all_hidden_keeps_only_the_mode_button(self):
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
        # The light/dark mode is a viewer preference, not a dashboard option.
        assert re.findall(r'id="(fv-btn-[\w-]+)"', html) == ["fv-btn-mode"]
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


class TestNotebookDelivery:
    def test_notebook_iframe_loads_the_page_from_the_server(
        self, server_port, monkeypatch
    ):
        """The page runs on the server's own origin, so it needs no CORS."""
        import sys
        import types

        import polars as pl

        from flexviz.adapters import build_adapter
        from flexviz.dashboard import Dashboard

        shown = []
        display_module = types.ModuleType("IPython.display")
        display_module.IFrame = lambda src, width, height: {
            "src": src,
            "height": height,
        }
        display_module.display = shown.append
        monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
        monkeypatch.setitem(sys.modules, "IPython.display", display_module)

        dash = Dashboard(pl.DataFrame({"ts": [0, 1], "val": [0.0, 1.0]}))
        dash.add_figure().add_line(x="ts", y="val")
        spec = dash.to_spec(source_name="_browser_test")
        server_url = f"http://127.0.0.1:{server_port}"
        build_adapter("plotly").show_dashboard(
            spec, server_url=server_url, notebook=True, height=432
        )

        (iframe,) = shown
        assert iframe["src"].startswith(f"{server_url}/view?spec=")
        assert iframe["src"].endswith("&renderer=plotly")
        assert iframe["height"] == 432


class TestBrowserDelivery:
    @pytest.mark.parametrize("in_notebook", [True, False])
    def test_block_waits_only_outside_a_notebook(self, monkeypatch, in_notebook):
        """A kernel keeps the server alive, so a notebook cell must not hang."""
        import asyncio
        import threading
        import webbrowser

        opened, waited = [], []
        monkeypatch.setattr(webbrowser, "open", opened.append)
        monkeypatch.setattr(threading.Event, "wait", lambda self: waited.append(1))
        url = "http://127.0.0.1:8000/view?spec=x"

        async def in_kernel():
            AbstractAdapter._deliver_browser(url, block=True)

        if in_notebook:
            asyncio.run(in_kernel())
        else:
            AbstractAdapter._deliver_browser(url, block=True)

        assert opened == [url]
        assert waited == ([] if in_notebook else [1])


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


class TestShowBlock:
    """``show()`` waits for Ctrl-C outside a notebook (``block=True``)."""

    @staticmethod
    def _patch_browser(monkeypatch):
        import threading
        import webbrowser

        import requests

        class _Resp:
            @staticmethod
            def raise_for_status() -> None:
                pass

            @staticmethod
            def json() -> dict[str, str]:
                return {"url": "http://127.0.0.1:9999/view?spec=x"}

        opened: list[str] = []
        waited: list[bool] = []

        monkeypatch.setattr(requests, "post", lambda *a, **kw: _Resp())
        monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url))

        def _wait(self, timeout=None):
            waited.append(True)
            raise KeyboardInterrupt

        monkeypatch.setattr(threading.Event, "wait", _wait)
        return opened, waited

    @staticmethod
    def _adapter(monkeypatch):
        from flexviz.adapters.plotly_adapter import PlotlyAdapter

        adapter = PlotlyAdapter()
        monkeypatch.setattr(adapter, "_wait_for_server", lambda _server_url: None)
        return adapter

    def test_block_true_returns_after_keyboard_interrupt(self, monkeypatch, capsys):
        opened, waited = self._patch_browser(monkeypatch)
        adapter = self._adapter(monkeypatch)

        adapter.show_dashboard(
            DashboardSpec(figures=[FigureSpec(uid="f1")]),
            server_url="http://127.0.0.1:9999",
            notebook=False,
            block=True,
        )

        assert opened and waited == [True]
        assert "FlexViz server stopped." in capsys.readouterr().out

    def test_block_false_returns_at_once(self, monkeypatch):
        opened, waited = self._patch_browser(monkeypatch)
        adapter = self._adapter(monkeypatch)

        adapter.show_dashboard(
            DashboardSpec(figures=[FigureSpec(uid="f1")]),
            server_url="http://127.0.0.1:9999",
            notebook=False,
            block=False,
        )

        assert opened and waited == []


class TestPlotlyLayoutOverrides:
    """update_layout() refines the derived layout instead of replacing it."""

    def _layout(self, fig) -> dict:
        import json
        import re

        import polars as pl

        from flexviz.adapters.plotly_adapter import PlotlyAdapter
        from flexviz.dashboard import Dashboard

        dash = Dashboard(pl.LazyFrame({"a": [1.0, 2.0], "b": [1.0, 2.0]}))
        dash._figures.append(fig)
        spec = dash._finalized_spec(
            "data",
            rows=None,
            cols=None,
            draggable=None,
            effective_cache=False,
            live_brush=None,
            layout=None,
        )
        html = PlotlyAdapter()._build_dashboard_html(spec, server_url=".")
        return json.loads(re.search(r"const layoutArr_0 = (\{.*?\});\n", html).group(1))

    def _figure(self):
        import polars as pl

        from flexviz.figure import Figure

        fig = Figure(pl.LazyFrame({"a": [1.0, 2.0], "b": [1.0, 2.0]}))
        fig.add_line(x="a", y="b")
        return fig

    def test_axis_override_keeps_the_axis_title(self):
        fig = self._figure().ylabel("Y label").update_layout(yaxis={"type": "log"})
        yaxis = self._layout(fig)["yaxis"]
        assert yaxis["type"] == "log"
        assert yaxis["title"]["text"] == "Y label"

    def test_nested_axis_override_keeps_the_axis_title(self):
        fig = self._figure().xlabel("X label")
        fig.update_layout(xaxis={"title": {"font": {"size": 20}}})
        title = self._layout(fig)["xaxis"]["title"]
        assert title["text"] == "X label"
        assert title["font"] == {"size": 20}

    def test_explicit_axis_title_wins_over_xlabel(self):
        fig = self._figure().xlabel("X label")
        fig.update_layout(xaxis={"title": {"text": "Explicit"}})
        assert self._layout(fig)["xaxis"]["title"]["text"] == "Explicit"

    def test_string_axis_title_is_left_alone(self):
        """Plotly accepts a bare string title; do not index into it."""
        fig = self._figure().xlabel("X label")
        fig.update_layout(xaxis={"title": "Explicit"})
        assert self._layout(fig)["xaxis"]["title"] == "Explicit"

    def test_legend_dict_shows_the_legend(self):
        fig = self._figure().legend(True).update_layout(legend={"orientation": "h"})
        layout = self._layout(fig)
        assert layout["showlegend"] is True
        assert layout["legend"]["orientation"] == "h"

    def test_legend_config_survives_either_call_order(self):
        fig = self._figure().update_layout(legend={"orientation": "h"}).legend(True)
        layout = self._layout(fig)
        assert layout["showlegend"] is True
        assert layout["legend"]["orientation"] == "h"

    def test_hiding_the_legend_keeps_its_placement(self):
        fig = self._figure().update_layout(legend={"orientation": "h"}).legend(False)
        layout = self._layout(fig)
        assert layout["showlegend"] is False
        assert layout["legend"]["orientation"] == "h"


class TestShowKwargValidation:
    def test_unknown_show_kwarg_raises(self):
        """`**kwargs` used to swallow these, so typos were silent no-ops."""
        from flexviz.adapters.plotly_adapter import PlotlyAdapter
        from flexviz.spec import DashboardSpec

        with pytest.raises(TypeError, match="draggable"):
            PlotlyAdapter().show_dashboard(DashboardSpec(), draggable=False)


class TestRendererRegistry:
    def test_plotly_supports_every_registered_trace(self):
        """Plotly is the primary renderer, so a trace it misses fails at /view."""
        from flexviz.adapters.registry import PLOTLY_TRACE_TYPES
        from flexviz.trace import _REGISTRY

        assert PLOTLY_TRACE_TYPES == set(_REGISTRY)

    def test_unsupported_trace_type_names_the_figure(self):
        from flexviz.adapters import validate_dashboard_renderer

        trace = TraceSpec(uid="t-1", trace_type="not_in_plotly")
        figure = FigureSpec(uid="fig-1", layout={"title": "Map"}, traces=[trace])
        with pytest.raises(
            ValueError,
            match="'plotly' does not support trace type 'not_in_plotly' in figure 'Map'",
        ):
            validate_dashboard_renderer("plotly", DashboardSpec(figures=[figure]))

    def test_renderer_name_must_match_exactly(self):
        from flexviz.adapters import build_adapter

        with pytest.raises(ValueError, match="Unknown renderer 'Plotly'"):
            build_adapter("Plotly")

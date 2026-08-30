"""ECharts adapter for flexviz.

Renders flexviz figures using Apache ECharts 5 (loaded from CDN) as
self-contained HTML pages.

Interaction loop
----------------
1. ``show_dashboard`` builds a self-contained HTML page with ECharts 5.
2. The page fetches initial data from ``/dashboard/update`` on load.
3. ``datazoom`` events are mapped to viewport ``InteractionEvent``s.
4. ``brushSelected`` events are mapped to selection ``InteractionEvent``s.
5. Returned ``TraceDelta``s are applied via ``chart.setOption`` — ECharts
   matches series by the ``id`` field (set to the trace uid).

Trace types
-----------
- ``"line"``      → ``{type: 'line', showSymbol: false}``
- ``"histogram"`` → ``{type: 'bar', barMaxWidth: 40}`` with value x-axis

Selection UX
------------
A per-figure "Select" toolbar button activates ECharts brush mode via
``chart.dispatchAction({type: 'takeGlobalCursor', key: 'brush', ...})``.
Brush is deactivated when the button is toggled off.
The global toolbar Deselect button clears all brush selections and posts
a ``type="deselect"`` event.
"""

from __future__ import annotations

import math
from typing import Any, Dict

from .base import (
    AbstractAdapter,
    _in_async_context,
    _json_for_inline_script,
)
from .runtime import (
    echarts_bundle_js,
    gridstack_bridge_js,
    page_head_html,
    shared_runtime_js,
    theme_css,
)
from ..events import InteractionEvent
from ..spec import DashboardSpec

_ECHARTS_JS = "https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"
_HEATMAP_STYLE_INVARIANT_ERROR = "Generated heatmap specs must include explicit color_scale and color_range defaults."
_ECHARTS_HEATMAP_COLOR_SCALES: dict[str, list[str]] = {
    "viridis": [
        "#440154",
        "#482878",
        "#3e4989",
        "#31688e",
        "#26828e",
        "#1f9e89",
        "#35b779",
        "#6ece58",
        "#b5de2b",
        "#fde725",
    ],
    "plasma": [
        "#0d0887",
        "#41049d",
        "#6a00a8",
        "#8f0da4",
        "#b12a90",
        "#cc4778",
        "#e16462",
        "#f2844b",
        "#fca636",
        "#f0f921",
    ],
    "magma": [
        "#000004",
        "#180f3d",
        "#440f76",
        "#721f81",
        "#9e2f7f",
        "#cd4071",
        "#f1605d",
        "#fd9668",
        "#feca8d",
        "#fcfdbf",
    ],
    "inferno": [
        "#000004",
        "#1b0c41",
        "#4a0c6b",
        "#781c6d",
        "#a52c60",
        "#cf4446",
        "#ed6925",
        "#fb9b06",
        "#f7d13d",
        "#fcffa4",
    ],
    "cividis": [
        "#00224e",
        "#123570",
        "#3b496c",
        "#575d6d",
        "#707173",
        "#8a8678",
        "#a59c74",
        "#c3b369",
        "#e1cc55",
        "#fee838",
    ],
    "blues": [
        "#f7fbff",
        "#deebf7",
        "#c6dbef",
        "#9ecae1",
        "#6baed6",
        "#4292c6",
        "#2171b5",
        "#08519c",
        "#08306b",
    ],
    "greens": [
        "#f7fcf5",
        "#e5f5e0",
        "#c7e9c0",
        "#a1d99b",
        "#74c476",
        "#41ab5d",
        "#238b45",
        "#006d2c",
        "#00441b",
    ],
    "reds": [
        "#fff5f0",
        "#fee0d2",
        "#fcbba1",
        "#fc9272",
        "#fb6a4a",
        "#ef3b2c",
        "#cb181d",
        "#a50f15",
        "#67000d",
    ],
    "rdbu": [
        "#67001f",
        "#b2182b",
        "#d6604d",
        "#f4a582",
        "#fddbc7",
        "#f7f7f7",
        "#d1e5f0",
        "#92c5de",
        "#4393c3",
        "#2166ac",
        "#053061",
    ],
}


def _echarts_figure_is_cartesian(fig_spec: DashboardSpec | Any) -> bool:
    """Return True when the figure has at least one cartesian trace."""
    return any(ts.axes and len(ts.axes) > 0 for ts in fig_spec.traces)


def _echarts_lockable_axis_families(fig_spec: DashboardSpec | Any) -> list[str]:
    """Return axis families supported by the current ECharts interaction model."""
    return ["x"] if _echarts_figure_is_cartesian(fig_spec) else []


def _echarts_canonical_heatmap_color_scale(color_scale: str) -> str:
    return color_scale.strip().casefold()


def _echarts_heatmap_color_scale(ts: Any) -> str:
    if "color_scale" not in ts.display:
        raise ValueError(_HEATMAP_STYLE_INVARIANT_ERROR)
    color_scale = ts.display["color_scale"]
    if not isinstance(color_scale, str) or not color_scale:
        raise TypeError("heatmap color_scale must be a non-empty string")
    return _echarts_canonical_heatmap_color_scale(color_scale)


def _echarts_heatmap_colors(ts: Any) -> list[str]:
    color_scale = _echarts_heatmap_color_scale(ts)
    colors = _ECHARTS_HEATMAP_COLOR_SCALES.get(color_scale)
    if colors is None:
        supported = ", ".join(sorted(_ECHARTS_HEATMAP_COLOR_SCALES))
        raise ValueError(
            f"ECharts heatmap color_scale {color_scale!r} is not supported. "
            f"Supported names: {supported}"
        )
    return list(colors)


def _echarts_heatmap_color_range(ts: Any) -> tuple[float, float] | str:
    if "color_range" not in ts.display:
        raise ValueError(_HEATMAP_STYLE_INVARIANT_ERROR)
    color_range = ts.display["color_range"]
    if color_range == "auto":
        return "auto"
    if not (isinstance(color_range, (list, tuple)) and len(color_range) == 2):
        raise TypeError(
            "heatmap color_range must be 'auto' or a (min, max) numeric tuple"
        )
    lo = float(color_range[0])
    hi = float(color_range[1])
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError("heatmap color_range values must be finite numbers")
    if lo >= hi:
        raise ValueError("heatmap color_range must satisfy min < max")
    return (lo, hi)


def _echarts_heatmap_visual_map(
    ts: Any,
    *,
    fallback_range: tuple[float, float] = (0.0, 1.0),
) -> dict:
    color_range = _echarts_heatmap_color_range(ts)
    if color_range == "auto":
        lo, hi = fallback_range
    else:
        lo, hi = color_range
    return {
        "min": lo,
        "max": hi,
        "calculable": True,
        "inRange": {"color": _echarts_heatmap_colors(ts)},
    }


class EChartsAdapter(AbstractAdapter):
    """Adapter that renders flexviz figures with Apache ECharts 5."""

    # ------------------------------------------------------------------
    # parse_event
    # ------------------------------------------------------------------

    def parse_event(self, raw_event: Dict[str, Any]) -> InteractionEvent | None:
        """Convert a raw event dict into an ``InteractionEvent``.

        Accepts:
        - ``{"type": ..., ...}`` passthrough — used by programmatic callers.
        - ``{"startValue": v, "endValue": v}`` — ECharts datazoom shorthand.
        """
        if not raw_event:
            return None
        if "type" in raw_event:
            try:
                return InteractionEvent(**raw_event)
            except Exception:
                return None
        if "startValue" in raw_event and "endValue" in raw_event:
            return InteractionEvent(
                type="viewport",
                axis_ranges={"x": (raw_event["startValue"], raw_event["endValue"])},
            )
        return None

    # ------------------------------------------------------------------
    # show_dashboard
    # ------------------------------------------------------------------

    def show_dashboard(
        self,
        spec: DashboardSpec,
        server_url: str = "http://127.0.0.1:8000",
        notebook: bool | None = None,
        height: int = 400,
        **kwargs: Any,
    ) -> None:
        """Render a dashboard as a self-contained ECharts HTML page.

        Parameters
        ----------
        spec:
            The ``DashboardSpec`` describing all figures and shared state.
        server_url:
            URL of the running flexviz FastAPI server.
        notebook:
            ``True``  — display inline via ``IPython.display.IFrame``.
            ``False`` — open in the default browser.
            ``None``  — auto-detect.
        height:
            Per-figure chart height in pixels.  IFrame height =
            ``height * n_figures + 80``.
        """
        if notebook is None:
            notebook = _in_async_context()

        self._wait_for_server(server_url)

        if notebook:
            n_figs = len(spec.figures)
            html = self._build_dashboard_html(
                spec, server_url=server_url, fig_height=height
            )
            self._deliver_notebook(html, height * n_figs + 80)
        else:
            self._deliver_browser_shared(spec, server_url, "echarts")

    # ------------------------------------------------------------------
    # _build_initial_option
    # ------------------------------------------------------------------

    @staticmethod
    def _build_initial_option(fig_spec: Any, fig_height: int) -> dict:
        """Build the initial ECharts option dict for one figure.

        Series are initialized with empty ``data`` arrays.  The trace ``uid``
        is stored as the series ``id`` so that ``chart.setOption`` can match
        and update series by uid without rebuilding the full option.
        """
        heatmap_spec = next(
            (
                ts
                for ts in fig_spec.traces
                if ts.trace_type in ("histogram2d", "corr_heatmap")
            ),
            None,
        )
        series: list[dict] = []
        for ts in fig_spec.traces:
            if ts.params.get("group_by") and not ts.params.get("group_value"):
                continue
            color = ts.display.get("color")
            name = ts.display.get("name", ts.uid)
            if ts.trace_type == "line":
                s: dict = {
                    "id": ts.uid,
                    "type": "line",
                    "name": name,
                    "data": [],
                    "showSymbol": False,
                    "smooth": False,
                }
                if color:
                    s["lineStyle"] = {"color": color}
                    s["itemStyle"] = {"color": color}
            elif ts.trace_type == "histogram":
                s = {
                    "id": ts.uid,
                    "type": "bar",
                    "name": name,
                    "data": [],
                    "barMaxWidth": 40,
                }
                if color:
                    s["itemStyle"] = {"color": color}
            elif ts.trace_type == "bar":
                bar_mode = ts.display.get(
                    "bar_mode", ts.params.get("bar_mode", "group")
                )
                s = {
                    "id": ts.uid,
                    "type": "bar",
                    "name": name,
                    "data": [],
                }
                if bar_mode == "stack":
                    s["stack"] = "bar"
                if color:
                    s["itemStyle"] = {"color": color}
            elif ts.trace_type == "pie":
                hole = ts.params.get("hole", 0)
                inner = f"{int(hole * 100)}%" if hole else "0%"
                s = {
                    "id": ts.uid,
                    "type": "pie",
                    "name": name,
                    "data": [],
                    "radius": [inner, "75%"],
                }
            elif ts.trace_type in ("histogram2d", "corr_heatmap"):
                s = {
                    "id": ts.uid,
                    "type": "heatmap",
                    "name": name,
                    "data": [],
                }
            elif ts.trace_type == "box":
                s = {
                    "id": ts.uid,
                    "type": "boxplot",
                    "name": name,
                    "data": [],
                }
                if color:
                    s["itemStyle"] = {"color": color}
            elif ts.trace_type == "treemap":
                s = {
                    "id": ts.uid,
                    "type": "treemap",
                    "name": name,
                    "data": [],
                    "roam": False,
                    "nodeClick": False,
                    "breadcrumb": {"show": False},
                    "label": {"show": True, "formatter": "{b}"},
                    "upperLabel": {"show": True},
                    "itemStyle": {
                        "borderColor": "#fafaf8",
                        "borderWidth": 2,
                        "gapWidth": 1,
                    },
                    "levels": [
                        {
                            "itemStyle": {
                                "borderColor": "#fafaf8",
                                "borderWidth": 2,
                                "gapWidth": 1,
                            }
                        },
                        {
                            "upperLabel": {"show": True, "height": 24},
                            "itemStyle": {
                                "borderColor": "#fafaf8",
                                "borderWidth": 3,
                                "gapWidth": 6,
                            },
                        },
                        {
                            "itemStyle": {
                                "borderColor": "#fafaf8",
                                "borderWidth": 2,
                                "gapWidth": 2,
                            }
                        },
                    ],
                }
            else:
                raise ValueError(
                    f"EChartsAdapter: unsupported trace type {ts.trace_type!r}"
                )
            series.append(s)

        # Determine axis types based on trace types
        _NON_CARTESIAN = {"pie", "treemap"}
        has_heatmap = any(
            ts.trace_type in ("histogram2d", "corr_heatmap") for ts in fig_spec.traces
        )
        all_non_cartesian = all(
            ts.trace_type in _NON_CARTESIAN for ts in fig_spec.traces
        )
        has_horizontal_bar = any(
            ts.trace_type == "bar" and ts.params.get("orientation") == "h"
            for ts in fig_spec.traces
        )
        has_bar = any(ts.trace_type == "bar" for ts in fig_spec.traces)
        has_horizontal_box = any(
            ts.trace_type == "box" and "x" in (ts.backend_data or {})
            for ts in fig_spec.traces
        )
        has_vertical_box = any(ts.trace_type == "box" for ts in fig_spec.traces)

        if has_horizontal_bar or has_horizontal_box:
            x_axis: dict = {"type": "value"}
            y_axis: dict = {"type": "category"}
        elif has_bar or has_vertical_box:
            x_axis = {"type": "category"}
            y_axis = {"type": "value"}
        else:
            x_axis = {"type": "value", "min": "dataMin", "max": "dataMax"}
            y_axis = {"type": "value"}

        fv_title = fig_spec.layout.get("title") or ""
        fv_xlabel = fig_spec.layout.get("xlabel")
        fv_ylabel = fig_spec.layout.get("ylabel")
        fv_legend = fig_spec.layout.get("legend")
        if fv_xlabel:
            x_axis["name"] = fv_xlabel
        if fv_ylabel:
            y_axis["name"] = fv_ylabel
        opt: dict = {
            "title": {"text": fv_title, "textStyle": {"fontSize": 13}},
            "tooltip": {
                "trigger": "item" if all_non_cartesian else "axis",
                "axisPointer": {"type": "line"},
            },
            "legend": {"show": fv_legend if fv_legend is not None else True},
            "toolbox": {"show": False},
            "series": series,
            "animation": False,
        }
        if not all_non_cartesian:
            opt["grid"] = {
                "left": "3%",
                "right": "4%",
                "bottom": "3%",
                "containLabel": True,
            }
            opt["xAxis"] = x_axis
            opt["yAxis"] = y_axis
            opt["dataZoom"] = [{"type": "inside", "xAxisIndex": 0}]
            opt["brush"] = {"xAxisIndex": 0, "yAxisIndex": 0}
        if has_heatmap:
            assert heatmap_spec is not None
            opt["visualMap"] = _echarts_heatmap_visual_map(heatmap_spec)
        return opt

    # ------------------------------------------------------------------
    # _build_dashboard_html
    # ------------------------------------------------------------------

    def _build_dashboard_html(
        self,
        spec: DashboardSpec,
        *,
        server_url: str = "http://127.0.0.1:8000",
        fig_height: int = 400,
    ) -> str:
        """Build a self-contained ECharts dashboard HTML page.

        Layout is resolved from ``spec.layout.grid_items`` (or auto-generated
        defaults when ``grid_items`` is ``None``).

        Each figure section has a small per-figure toolbar with a "Select"
        button that activates ECharts brush mode.  The global toolbar (shared
        header) provides Reset, Deselect, Share, Export, and Import actions.

        Data updates use ``chart.setOption({series: [{id, data}]})`` which
        ECharts merges by series ``id`` — only the changed series are updated.
        """
        dash_json = _json_for_inline_script(spec.model_dump())
        server_url_js = _json_for_inline_script(server_url)
        layout = spec.layout
        from ..spec import _GRIDSTACK_CELL_HEIGHT_PX

        dashboard = self._dashboard_markup(
            spec,
            render_panel=lambda idx, fig_spec: self._panel_html(
                idx,
                f'<div class="fv-chart fv-renderer-mount" id="fv-chart-{idx}"></div>',
                _echarts_lockable_axis_families(fig_spec),
                supports_zoom_pan=_echarts_figure_is_cartesian(fig_spec),
            ),
        )

        # -- Per-figure initial options (JSON-serializable) ----------------
        fig_uids = [fig_spec.uid for fig_spec in spec.figures]
        initial_options: dict[str, dict] = {
            fig_spec.uid: self._build_initial_option(fig_spec, fig_height)
            for fig_spec in spec.figures
        }
        fig_uid_map = {fig_spec.uid: i for i, fig_spec in enumerate(spec.figures)}
        fig_uid_to_idx_js = (
            f"const figUidToIdx = {_json_for_inline_script(fig_uid_map)};"
        )
        fig_supports_zoom_pan_js = (
            "const figSupportsZoomPan = "
            + _json_for_inline_script(
                [_echarts_figure_is_cartesian(fig_spec) for fig_spec in spec.figures]
            )
            + ";"
        )
        fig_lockable_axes_js = (
            "const figLockableAxes = "
            + _json_for_inline_script(
                [_echarts_lockable_axis_families(fig_spec) for fig_spec in spec.figures]
            )
            + ";"
        )

        fig_uids_json = _json_for_inline_script(fig_uids)
        initial_options_json = _json_for_inline_script(initial_options)
        heatmap_color_scales_json = _json_for_inline_script(
            _ECHARTS_HEATMAP_COLOR_SCALES
        )

        toolbar_html = self._toolbar_html(spec.layout.toolbar)
        toolbar_css = self._toolbar_css()

        return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  {page_head_html()}
  <script src="{_ECHARTS_JS}"></script>
  {dashboard.head_html}
  <style>
{theme_css()}
    body {{
      margin: 0;
      padding: 0;
      box-sizing: border-box;
      font-family: var(--fv-font);
    }}
{toolbar_css}
    {dashboard.css}
  </style>
</head>
<body>
{toolbar_html}
  {dashboard.container_html}
  {self._filter_strip_html()}
  <script>
    const DASHBOARD_SPEC    = {dash_json};
    const SERVER_URL        = {server_url_js};
    const FIG_UIDS          = {fig_uids_json};
    const INITIAL_OPTIONS   = {initial_options_json};
    const _fvAllFigUids     = FIG_UIDS;
    {fig_uid_to_idx_js}
    {fig_supports_zoom_pan_js}
    {fig_lockable_axes_js}
    const ECHARTS_HEATMAP_COLOR_SCALES = {heatmap_color_scales_json};
    const _FV_CELL_HEIGHT   = {_GRIDSTACK_CELL_HEIGHT_PX};

    // --- Shared runtime + ECharts-specific bundle ---
    {shared_runtime_js()}
    {gridstack_bridge_js() if layout.draggable else ""}
    {echarts_bundle_js()}

  </script>
</body>
</html>"""

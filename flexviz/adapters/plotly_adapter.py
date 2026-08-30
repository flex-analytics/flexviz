"""Plotly adapter for flexviz.

Renders flexviz figures using Plotly.js (loaded from CDN) as self-contained
HTML pages.  No Dash or Jupyter widget extensions required.

Interaction loop
----------------
1. ``show_dashboard`` builds a self-contained HTML page with Plotly.js 3.
2. The page fetches initial data from ``/dashboard/update`` on load and
   calls ``Plotly.react``.
3. Every zoom/pan (``relayoutData``) is parsed into an ``InteractionEvent``
   and posted to ``/dashboard/update``; returned deltas are applied via
   ``Plotly.react``.

"""

from __future__ import annotations

import math
import re
from typing import Any, Dict


from .base import (
    AbstractAdapter,
    _in_async_context,
    _json_for_inline_script,
)
from .runtime import (
    gridstack_bridge_js,
    page_head_html,
    plotly_bundle_js,
    shared_runtime_js,
    theme_css,
)
from ..events import InteractionEvent
from ..spec import DashboardSpec, FigureSpec

_HEATMAP_STYLE_INVARIANT_ERROR = "Generated heatmap specs must include explicit color_scale and color_range defaults."
_PLOTLY_RANGE_INDEX_RE = re.compile(r"(x|y)axis(\d*)\.range\[([01])\]$")
_PLOTLY_RANGE_ARRAY_RE = re.compile(r"(x|y)axis(\d*)\.range$")
_PLOTLY_AUTORANGE_RE = re.compile(r"(x|y)axis(\d*)\.autorange$")
_PLOTLY_MAP_TRACE_TYPES = {"geo_histogram2d", "geo_line"}


def _figure_has_multi_y(fig_spec: FigureSpec) -> bool:
    """Return True when the figure has ≥2 cartesian traces with distinct (x,y) column pairs.

    Used to decide whether to show the OR-filter warning in Select mode.
    Mirrors the JS _colsForTrace helper.
    """
    cartesian = [
        ts for ts in fig_spec.traces if ts.axes and len(ts.axes) > 0 and ts.backend_data
    ]
    if len(cartesian) < 2:
        return False
    seen: set = set()
    for ts in cartesian:
        bd = ts.backend_data
        is_h = ts.params.get("orientation") == "h"
        if is_h:
            x_col = bd.get("values") or bd.get("x")
            lbl = bd.get("labels")
            if not lbl:
                y_col = bd.get("y")
            elif isinstance(lbl, list):
                y_col = lbl[0] if len(lbl) == 1 else None
            else:
                y_col = lbl
        else:
            lbl = bd.get("labels")
            if not lbl:
                x_col = bd.get("x")
            elif isinstance(lbl, list):
                x_col = lbl[0] if len(lbl) == 1 else None
            else:
                x_col = lbl
            y_col = bd.get("values") or bd.get("y")
        seen.add((x_col, y_col))
    return len(seen) > 1


def _figure_is_cartesian(fig_spec: FigureSpec) -> bool:
    """Return True when the figure has at least one cartesian trace (axes defined)."""
    return any(ts.axes and len(ts.axes) > 0 for ts in fig_spec.traces)


def _figure_select_direction(fig_spec: FigureSpec) -> str | None:
    """Return the Plotly ``selectdirection`` matching the figure's brush geometry.

    Derived from the ``selection.axis_columns`` of the figure's ``kind="range"``
    traces (line / histogram / box / histogram2d).  An x-only figure (e.g. a
    line) brushes as a full-height vertical band (``"h"``); a y-only figure as a
    horizontal band (``"v"``); a figure that selects on both keeps the default
    2-D rectangle (``"d"``).  Returns ``None`` when no range-selectable trace is
    present, leaving Plotly's default untouched.
    """
    selects_x = selects_y = False
    for ts in fig_spec.traces:
        sel = ts.selection
        if sel is None or sel.kind != "range":
            continue
        axes = ts.axes or ()
        if len(axes) < 2:
            continue
        anchors = set(sel.axis_columns)
        if axes[0] in anchors:
            selects_x = True
        if axes[1] in anchors:
            selects_y = True
    if selects_x and selects_y:
        return "d"
    if selects_x:
        return "h"
    if selects_y:
        return "v"
    return None


def _figure_supports_zoom_pan(fig_spec: FigureSpec) -> bool:
    """Return True when Plotly drag Zoom/Pan should remain available.

    Map traces have no cartesian axes, but Plotly still uses dragmode for map
    navigation. Keep them out of cartesian selection logic while allowing their
    native pan/zoom relayouts to drive geo viewport updates.
    """
    return _figure_is_cartesian(fig_spec) or any(
        ts.trace_type in _PLOTLY_MAP_TRACE_TYPES for ts in fig_spec.traces
    )


def _figure_hovermode(fig_spec: FigureSpec) -> str:
    """Return the Plotly ``hovermode`` matching the figure's data orientation.

    Plotly's ``hovermode`` decides which axis the cursor position is matched
    against to pick a point.  ``"x"`` is correct for vertically-laid-out traces
    (a vertical histogram, a line), but a *horizontal* histogram or bar lays its
    bars out along ``y`` — there ``"x"`` matches the count axis and per-bar hover
    becomes unreliable.  Use ``"y"`` when the figure's data traces are all
    horizontal, otherwise keep ``"x"``.
    """
    has_horizontal = False
    has_vertical = False
    for ts in fig_spec.traces:
        if ts.trace_type == "histogram":
            if "y" in ts.backend_data and "x" not in ts.backend_data:
                has_horizontal = True
            else:
                has_vertical = True
        elif ts.trace_type == "bar":
            if ts.params.get("orientation", "v") == "h":
                has_horizontal = True
            else:
                has_vertical = True
    if has_horizontal and not has_vertical:
        return "y"
    return "x"


def _figure_lockable_axis_families(fig_spec: FigureSpec) -> list[str]:
    """Return renderer-agnostic axis families that can be range-locked."""
    families: set[str] = set()
    for ts in fig_spec.traces:
        for axis_id in ts.axes or ():
            axis_family = str(axis_id)[:1]
            if axis_family in {"x", "y"}:
                families.add(axis_family)
    return [axis for axis in ("x", "y") if axis in families]


def _auto_show_legend(traces: list) -> bool:
    if len(traces) > 1:
        return True
    return bool(traces and traces[0].params.get("group_by"))


def _plotly_heatmap_color_scale(ts: Any) -> str:
    if "color_scale" not in ts.display:
        raise ValueError(_HEATMAP_STYLE_INVARIANT_ERROR)
    color_scale = ts.display["color_scale"]
    if not isinstance(color_scale, str) or not color_scale:
        raise TypeError("heatmap color_scale must be a non-empty string")
    return color_scale


def _plotly_heatmap_color_range(ts: Any) -> tuple[float, float] | str:
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


def _plotly_heatmap_trace_obj(ts: Any, name: str) -> dict:
    trace = {
        "uid": ts.uid,
        "type": "heatmap",
        "name": name,
        "x": [],
        "y": [],
        "z": [],
        "colorscale": _plotly_heatmap_color_scale(ts),
        "showlegend": False,
    }
    color_range = _plotly_heatmap_color_range(ts)
    if color_range != "auto":
        trace["zmin"], trace["zmax"] = color_range
    return trace


def _plotly_choropleth_trace_obj(ts: Any, name: str) -> dict:
    trace: dict = {
        "uid": ts.uid,
        "type": "choroplethmap",
        "name": name,
        "geojson": {"type": "FeatureCollection", "features": []},
        "locations": [],
        "z": [],
        "featureidkey": "id",
        "colorscale": _plotly_heatmap_color_scale(ts),
        "showlegend": False,
        "showscale": True,
        "marker": {"line": {"width": 0}},
    }
    color_range = _plotly_heatmap_color_range(ts)
    if color_range != "auto":
        trace["zmin"], trace["zmax"] = color_range
    return trace


def _plotly_geo_line_trace_obj(ts: Any, name: str) -> dict:
    obj: dict = {
        "uid": ts.uid,
        "type": "scattermap",
        "name": name,
        "lat": [],
        "lon": [],
        "mode": "lines",
    }
    color = ts.display.get("color")
    if color:
        obj["line"] = {"color": color}
    return obj


def _plotly_axis_ranges(
    relayout_data: Dict[str, Any],
) -> tuple[Dict[str, tuple[Any, Any]], bool]:
    """Collect complete Plotly axis ranges from a ``relayoutData`` dict."""
    axis_ranges: Dict[str, list[Any | None]] = {}
    has_autorange = False

    for key, value in relayout_data.items():
        indexed = _PLOTLY_RANGE_INDEX_RE.match(key)
        if indexed:
            axis_id = f"{indexed.group(1)}{indexed.group(2)}"
            axis_ranges.setdefault(axis_id, [None, None])[int(indexed.group(3))] = value
            continue

        array = _PLOTLY_RANGE_ARRAY_RE.match(key)
        if array:
            if isinstance(value, (list, tuple)) and len(value) == 2:
                axis_id = f"{array.group(1)}{array.group(2)}"
                axis_ranges[axis_id] = [value[0], value[1]]
            continue

        if _PLOTLY_AUTORANGE_RE.match(key):
            has_autorange = True

    complete = {
        axis_id: (range_values[0], range_values[1])
        for axis_id, range_values in axis_ranges.items()
        if range_values[0] is not None and range_values[1] is not None
    }
    return complete, has_autorange


class PlotlyAdapter(AbstractAdapter):
    """Adapter that renders flexviz figures with Plotly.js."""

    # ------------------------------------------------------------------
    # parse_event
    # ------------------------------------------------------------------

    def parse_event(self, relayout_data: Dict[str, Any]) -> InteractionEvent | None:
        """Parse Plotly ``relayoutData`` into an ``InteractionEvent``.

        Handles:
        - ``xaxis.range`` array patterns → ``"viewport"``
        - ``xaxis.range[0]`` / ``xaxis.range[1]`` patterns → ``"viewport"``
        - ``xaxis.autorange`` / ``autosize`` / ``dragmode`` → ``"reset"``
          or ignored respectively.
        """
        if not relayout_data:
            return None

        # If the caller already built a typed event dict (e.g. from a Dash
        # callback or programmatic use), construct directly without parsing.
        if "type" in relayout_data:
            try:
                return InteractionEvent(**relayout_data)
            except Exception:
                pass

        # Ignore drag-mode changes (no data update needed)
        if list(relayout_data.keys()) == ["dragmode"]:
            return None

        # Full reset (autorange button)
        if relayout_data.get("autosize") is True:
            return InteractionEvent(type="reset", force_update=True)

        for key, value in relayout_data.items():
            if re.match(r"map\d*\._derived$", key):
                coordinates = (
                    value.get("coordinates") if isinstance(value, dict) else None
                )
                if coordinates:
                    return InteractionEvent(
                        type="viewport",
                        axis_ranges={"coordinates": coordinates},
                    )

        complete, has_autorange = _plotly_axis_ranges(relayout_data)
        if has_autorange and not complete:
            return InteractionEvent(type="reset", force_update=True)

        if not complete:
            return None

        return InteractionEvent(type="viewport", axis_ranges=complete)

    # ------------------------------------------------------------------
    # show_dashboard
    # ------------------------------------------------------------------

    def show_dashboard(
        self,
        spec: DashboardSpec,
        server_url: str = "http://127.0.0.1:8000",
        notebook: bool | None = None,
        height: int = 500,
        **kwargs: Any,
    ) -> None:
        """Render a dashboard as a self-contained Plotly.js HTML page.

        Parameters
        ----------
        spec:
            The ``DashboardSpec`` describing all figures + shared state.
        server_url:
            URL of the running flexviz FastAPI server.
        notebook:
            ``True``  — display inline via ``IPython.display.IFrame``.
            ``False`` — open in the default browser.
            ``None``  — auto-detect: inline when an event loop is already
            running (Jupyter / VS Code notebook), browser otherwise.
        height:
            IFrame height in pixels (notebook mode only).
        """
        if notebook is None:
            notebook = _in_async_context()

        self._wait_for_server(server_url)

        if notebook:
            html = self._build_dashboard_html(spec, server_url=server_url)
            self._deliver_notebook(html, height)
        else:
            self._deliver_browser_shared(spec, server_url, "plotly")

    def _build_dashboard_html(
        self,
        spec: DashboardSpec,
        *,
        server_url: str = "http://127.0.0.1:8000",
    ) -> str:
        """Build a self-contained multi-figure Plotly.js HTML page."""
        dash_json = _json_for_inline_script(spec.model_dump())
        server_url_js = _json_for_inline_script(server_url)
        from ..cache import is_source_cacheable

        cacheable_sources_js = _json_for_inline_script(
            sorted(
                {
                    f.source
                    for f in spec.figures
                    if f.source is not None and is_source_cacheable(f.source)
                }
            )
        )
        n_figs = len(spec.figures)
        layout = spec.layout
        from ..spec import _GRIDSTACK_CELL_HEIGHT_PX

        dashboard = self._dashboard_markup(
            spec,
            render_panel=lambda idx, fig_spec: self._panel_html(
                idx,
                f'<div id="fv-plot-{idx}" class="fv-renderer-mount"></div>',
                _figure_lockable_axis_families(fig_spec),
                supports_zoom_pan=_figure_supports_zoom_pan(fig_spec),
            ),
        )
        gridstack_js = ""
        if layout.draggable:
            gridstack_js = (
                "    window._fvResizeChart = function(el) {\n"
                + "      var d = el.querySelector('[id^=\"fv-plot-\"]');\n"
                + "      if (d) Plotly.Plots.resize(d);\n"
                + "    };\n"
            )

        traces_js_lines: list[str] = []
        for i, fig_spec in enumerate(spec.figures):
            traces_arr: list[dict] = []
            for ts in fig_spec.traces:
                if ts.params.get("group_by") and not ts.params.get("group_value"):
                    continue
                color = ts.display.get("color")
                name = ts.display.get("name", ts.uid)
                obj = self._plotly_trace_obj(ts, name, color)
                traces_arr.append(obj)
            layout_obj: dict = {
                "autosize": True,
                "margin": {"t": 40, "b": 40, "l": 60, "r": 20},
                "hovermode": _figure_hovermode(fig_spec),
                "hoverdistance": -1,
            }
            select_direction = _figure_select_direction(fig_spec)
            if select_direction is not None:
                layout_obj["selectdirection"] = select_direction
            bar_ts = next(
                (ts for ts in fig_spec.traces if ts.trace_type == "bar"), None
            )
            if bar_ts is not None:
                layout_obj["barmode"] = bar_ts.display.get(
                    "bar_mode", bar_ts.params.get("bar_mode", "group")
                )
            elif any(ts.trace_type == "histogram" for ts in fig_spec.traces):
                # Plotly's default barmode is 'stack', which is wrong for
                # independent-variable histograms.  Mirror in JS baseBarmodeForFigure.
                layout_obj["barmode"] = "group"
            has_geo = any(
                ts.trace_type in ("geo_histogram2d", "geo_line")
                for ts in fig_spec.traces
            )
            if has_geo:
                layout_obj["map"] = {
                    "style": "open-street-map",
                    "center": {"lat": 0, "lon": 0},
                    "zoom": 1,
                }
            raw_layout = dict(fig_spec.layout or {})
            fv_title = raw_layout.pop("title", None)
            fv_xlabel = raw_layout.pop("xlabel", None)
            fv_ylabel = raw_layout.pop("ylabel", None)
            fv_legend = raw_layout.pop("legend", None)
            if isinstance(fv_legend, dict):
                layout_obj["legend"] = fv_legend
            if fv_title is not None:
                layout_obj.setdefault("title", {})["text"] = fv_title
            if fv_xlabel is not None:
                layout_obj.setdefault("xaxis", {}).setdefault("title", {})[
                    "text"
                ] = fv_xlabel
            if fv_ylabel is not None:
                layout_obj.setdefault("yaxis", {}).setdefault("title", {})[
                    "text"
                ] = fv_ylabel
            layout_obj["showlegend"] = (
                fv_legend
                if isinstance(fv_legend, bool)
                else _auto_show_legend(fig_spec.traces)
            )
            layout_obj.update(raw_layout)
            traces_js_lines.append(
                f"const tracesArr_{i} = {_json_for_inline_script(traces_arr)};"
            )
            traces_js_lines.append(
                f"const layoutArr_{i} = {_json_for_inline_script(layout_obj)};"
            )

        traces_js = "\n    ".join(traces_js_lines)
        traces_by_fig = (
            "const tracesByFig = ["
            + ", ".join(f"tracesArr_{i}" for i in range(n_figs))
            + "];"
        )
        layouts_by_fig = (
            "const layoutsByFig = ["
            + ", ".join(f"layoutArr_{i}" for i in range(n_figs))
            + "];"
        )
        divs_js = (
            "const divs = ["
            + ", ".join(
                f"document.getElementById('fv-plot-{i}')" for i in range(n_figs)
            )
            + "];"
        )
        fig_uid_map = {fig_spec.uid: i for i, fig_spec in enumerate(spec.figures)}
        fig_uid_to_idx_js = (
            f"const figUidToIdx = {_json_for_inline_script(fig_uid_map)};"
        )
        fig_uids_js = _json_for_inline_script(list(fig_uid_map.keys()))
        fig_has_multi_y_js = f"const figHasMultiY = {_json_for_inline_script([_figure_has_multi_y(f) for f in spec.figures])};"
        fig_supports_zoom_pan_js = f"const figSupportsZoomPan = {_json_for_inline_script([_figure_supports_zoom_pan(f) for f in spec.figures])};"
        fig_lockable_axes_js = f"const figLockableAxes = {_json_for_inline_script([_figure_lockable_axis_families(f) for f in spec.figures])};"

        return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  {page_head_html()}
  <script src="https://cdn.plot.ly/plotly-3.0.0.min.js"></script>
  {dashboard.head_html}
  <style>
{theme_css()}
    body {{ margin: 0; padding: 0; box-sizing: border-box; font-family: var(--fv-font); background-color: var(--fv-bg); }}
    {dashboard.css}
  {self._toolbar_css()}
  </style>
</head>
<body>
  {self._toolbar_html(spec.layout.toolbar)}
  {dashboard.container_html}
  {self._filter_strip_html()}
  <script>
    const DASHBOARD_SPEC = {dash_json};
    const SERVER_URL     = {server_url_js};
    const _fvAllFigUids  = {fig_uids_js};
    const FV_CACHEABLE_SOURCES = {cacheable_sources_js};
    {fig_has_multi_y_js}
    {fig_supports_zoom_pan_js}
    {fig_lockable_axes_js}

    {traces_js}
    {traces_by_fig}
    {layouts_by_fig}
    {divs_js}
    {fig_uid_to_idx_js}
    const _FV_CELL_HEIGHT = {_GRIDSTACK_CELL_HEIGHT_PX};
    {gridstack_js}

    // --- Shared runtime + Plotly-specific bundle ---
    {shared_runtime_js()}
    {gridstack_bridge_js() if layout.draggable else ""}
    {plotly_bundle_js()}
  </script>
</body>
</html>"""

    @staticmethod
    def _plotly_trace_obj(ts: Any, name: str, color: str | None) -> dict:
        """Build an initial Plotly trace object from a ``TraceSpec``."""
        if ts.trace_type == "histogram":
            return {
                "uid": ts.uid,
                "type": "bar",
                "name": name,
                "x": [],
                "y": [],
                "offsetgroup": ts.uid,
                "alignmentgroup": "fv-bars",
                "marker": {"color": color},
            }
        if ts.trace_type == "line":
            return {
                "uid": ts.uid,
                "mode": "lines",  # +markers",
                "type": "scatter",  # TODO do we use scattergl
                "name": name,
                "x": [],
                "y": [],
                "line": {"color": color},
                "marker": {"opacity": 0},
            }
        if ts.trace_type == "box":
            obj: dict = {
                "uid": ts.uid,
                "type": "box",
                "name": name,
                "lowerfence": [],
                "q1": [],
                "median": [],
                "q3": [],
                "upperfence": [],
                "marker": {"color": color},
            }
            if "y" in ts.backend_data:
                obj["orientation"] = "v"
                obj["x0"] = name
            else:
                obj["orientation"] = "h"
                obj["y0"] = name
            return obj
        if ts.trace_type == "bar":
            obj = {"uid": ts.uid, "type": "bar", "name": name, "x": [], "y": []}
            if ts.params.get("orientation") == "h":
                obj["orientation"] = "h"
            bar_mode = ts.display.get("bar_mode", ts.params.get("bar_mode", "group"))
            if bar_mode != "stack":
                obj["offsetgroup"] = ts.uid
                obj["alignmentgroup"] = "fv-bars"
            if color:
                obj["marker"] = {"color": color}
            return obj
        if ts.trace_type == "pie":
            obj = {
                "uid": ts.uid,
                "type": "pie",
                "name": name,
                "labels": [],
                "values": [],
                "hole": ts.params.get("hole", 0),
            }
            return obj
        if ts.trace_type in ("histogram2d", "corr_heatmap"):
            return _plotly_heatmap_trace_obj(ts, name)
        if ts.trace_type == "geo_histogram2d":
            return _plotly_choropleth_trace_obj(ts, name)
        if ts.trace_type == "geo_line":
            return _plotly_geo_line_trace_obj(ts, name)
        if ts.trace_type == "treemap":
            return {
                "uid": ts.uid,
                "type": "treemap",
                "name": name,
                "labels": [],
                "parents": [],
                "ids": [],
                "values": [],
                "branchvalues": "total",
            }
        raise ValueError(f"Unsupported trace type {ts.trace_type!r}")

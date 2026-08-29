"""Abstract adapter interface.

An adapter is the *only* renderer-coupled component in flexviz.
It translates between the renderer-agnostic core (VisualizationSpec,
TraceDelta, InteractionEvent) and a specific frontend library.

Responsibilities
----------------
parse_event(raw_event)
    Convert a renderer-specific interaction payload (e.g. Plotly
    ``relayoutData``) into an ``InteractionEvent`` that the backend
    understands.  Return ``None`` if the event should be ignored.

show_dashboard(spec, server_url, **kwargs)
    High-level convenience: build the dashboard HTML, deliver it to the
    user (notebook IFrame or browser), and start an interactive session.

show(spec, server_url, **kwargs)
    Wraps a ``VisualizationSpec`` into a 1-figure ``DashboardSpec`` and
    calls ``show_dashboard``.  Concrete — adapters do not override this.

Shared frontend
---------------
Adapters share the header HTML, panel wrapper HTML, CSS tokens, and the
bundled JS runtime under ``flexviz/adapters/js``.  This module only owns
the HTML/CSS fragments and delivery helpers used by both adapters.
"""

from __future__ import annotations

import html
import json
import re
import time
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, NamedTuple

from ..events import InteractionEvent
from ..spec import DashboardSpec, VisualizationSpec
from .runtime import toolbar_css

_CSS_LENGTH_TOKEN_RE = re.compile(
    r"^(?:0|(?:\d+(?:\.\d+)?|\.\d+)(?:px|em|rem|%|vh|vw|vmin|vmax|ch|ex|cm|mm|in|pt|pc))$"
)


def _html_attr(value: Any) -> str:
    """Return ``value`` encoded for a quoted HTML attribute."""
    return html.escape(str(value), quote=True)


def _json_for_inline_script(value: Any) -> str:
    """Return JSON safe to embed directly in a ``<script>`` block."""
    return (
        json.dumps(value)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _safe_css_gap(value: Any, *, default: str = "8px") -> str:
    """Return a conservative CSS gap/padding value for generated style blocks."""
    tokens = str(value).strip().split()
    if not tokens:
        return default
    if tokens == ["normal"]:
        return "normal"
    if len(tokens) <= 2 and all(_CSS_LENGTH_TOKEN_RE.fullmatch(t) for t in tokens):
        return " ".join(tokens)
    return default


def _in_async_context() -> bool:
    """Return ``True`` when called from inside a running async event loop.

    Jupyter kernels permanently run a Tornado/asyncio event loop, so
    ``asyncio.get_running_loop()`` succeeds there.  In a plain Python
    script there is no running loop and it raises ``RuntimeError``.
    """
    import asyncio

    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


class _DashboardMarkup(NamedTuple):
    """Shared dashboard shell returned by ``AbstractAdapter._dashboard_markup``."""

    container_html: str
    css: str
    head_html: str


# ``noreferrer`` strips the origin, so these tags are the only attribution
# signal. Umami buckets as Referral only on medium referral/app/link.
_BRAND_URL = "https://flexviz.tech/?utm_source=flexviz_dashboard&amp;utm_medium=app"


class AbstractAdapter(ABC):
    """Base class for all flexviz rendering adapters."""

    # ------------------------------------------------------------------
    # Abstract interface — adapters must implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def parse_event(self, raw_event: Any) -> InteractionEvent | None:
        """Convert a renderer-specific event into an ``InteractionEvent``.

        Parameters
        ----------
        raw_event:
            Renderer-native event data.

        Returns
        -------
        InteractionEvent or None
            ``None`` if the event should not trigger a backend call.
        """

    @abstractmethod
    def show_dashboard(
        self,
        spec: DashboardSpec,
        server_url: str = "http://127.0.0.1:8000",
        **kwargs: Any,
    ) -> None:
        """Build and display a dashboard for *spec*.

        Parameters
        ----------
        spec:
            Full ``DashboardSpec`` (N figures + shared interaction state).
        server_url:
            Base URL of the running flexviz FastAPI server.
        **kwargs:
            Adapter-specific keyword arguments.
        """

    # ------------------------------------------------------------------
    # Concrete — shared across all adapters
    # ------------------------------------------------------------------

    def show(
        self,
        spec: VisualizationSpec,
        server_url: str = "http://127.0.0.1:8000",
        **kwargs: Any,
    ) -> None:
        """Wrap *spec* as a 1-figure dashboard and call ``show_dashboard``.

        This is the canonical single-figure entry point.  It is concrete
        and should not be overridden — all rendering goes through the
        dashboard path so that the shared toolbar appears in every view.
        """
        dash_spec = DashboardSpec(
            figures=[spec.figure],
            state=spec.state,
        )
        self.show_dashboard(dash_spec, server_url=server_url, **kwargs)

    # ------------------------------------------------------------------
    # Shared HTML / CSS / JS building blocks
    # ------------------------------------------------------------------

    @staticmethod
    def _panel_html(
        idx: int,
        mount_html: str,
        lockable_axes: list[str] | None = None,
        supports_zoom_pan: bool = True,
    ) -> str:
        """Return a shared panel wrapper around a renderer mount element."""
        lockable_attr = _html_attr(" ".join(lockable_axes or []))
        has_panel_bar = supports_zoom_pan or bool(lockable_axes)
        panel_class_attr = ' class="fv-panel-has-bar"' if has_panel_bar else ""
        return (
            f'<fv-panel fig-idx="{idx}" lockable="{lockable_attr}"{panel_class_attr}>'
            f'<div class="fv-plot-wrap">{mount_html}</div>'
            f"{AbstractAdapter._panel_bar_html(idx, lockable_axes, supports_zoom_pan) if has_panel_bar else ''}"
            f"</fv-panel>"
        )

    @staticmethod
    def _panel_bar_html(
        idx: int,
        lockable_axes: list[str] | None = None,
        supports_zoom_pan: bool = True,
    ) -> str:
        """Return the per-figure bottom control bar HTML fragment.

        Parameters
        ----------
        idx:
            Zero-based figure index used for ``id="fv-bar-{idx}"``.
        lockable_axes:
            List of axis families (``"x"`` and/or ``"y"``) that should have
            lock buttons.  Pass ``None`` or empty list to omit lock buttons.
        supports_zoom_pan:
            Whether the figure should expose the Zoom / Pan / CF mode buttons
            and the Reset action button.
        """
        mode_group = ""
        if supports_zoom_pan:
            mode_group = (
                '<div class="fv-mode-toggle">'
                '<button type="button" class="fv-mode-btn" data-mode="zoom" aria-label="Zoom mode">Zoom</button>'
                '<button type="button" class="fv-mode-btn" data-mode="pan" aria-label="Pan mode">Pan</button>'
                '<button type="button" class="fv-mode-btn" data-mode="select" aria-label="Cross-filter mode">CF</button>'
                "</div>"
            )
        action_btns = ""
        if supports_zoom_pan:
            action_btns += (
                '<button type="button" class="fv-mode-action-btn"'
                ' data-action="reset-panel" aria-label="Reset panel view">Reset</button>'
            )
        if lockable_axes:
            lockable_attr = _html_attr(" ".join(lockable_axes))
            action_btns += (
                f'<button type="button" class="fv-mode-action-btn" data-action="lock-axes"'
                f' data-lockable="{lockable_attr}" aria-label="Toggle axis lock" aria-pressed="false">Lock Axes</button>'
            )
        return (
            f'<div id="fv-bar-{idx}" class="fv-panel-bar" role="toolbar" aria-label="Panel controls">'
            f'<div class="fv-panel-bar-slot" data-slot="modes">{mode_group}</div>'
            f'<div class="fv-panel-bar-slot" data-slot="actions">{action_btns}</div>'
            f'<span class="fv-panel-bar-slot fv-panel-bar-info" data-slot="info"></span>'
            f'<span class="fv-panel-bar-slot" data-slot="warn"><span class="fv-mode-warn"></span></span>'
            f'<span class="fv-panel-bar-slot" data-slot="status"></span>'
            f"</div>"
        )

    def _dashboard_markup(
        self,
        spec: DashboardSpec,
        *,
        render_panel: Callable[[int, Any], str],
    ) -> _DashboardMarkup:
        """Return the shared dashboard container markup for either adapter."""
        layout = spec.layout
        layout_gap = _safe_css_gap(layout.gap)
        from ..spec import GridItem, _GRIDSTACK_CELL_HEIGHT_PX, _auto_grid_items

        grid_items = layout.grid_items or _auto_grid_items(spec.figures)
        item_map = {gi.fig_uid: gi for gi in grid_items}

        def _grid_item_for(idx: int, fig_spec: Any) -> Any:
            item = item_map.get(fig_spec.uid)
            if item is not None:
                return item
            return GridItem(fig_uid=fig_spec.uid, x=0, y=idx * 5, w=12, h=5)

        if layout.draggable:
            items = []
            for idx, fig_spec in enumerate(spec.figures):
                item = _grid_item_for(idx, fig_spec)
                fig_uid_attr = _html_attr(fig_spec.uid)
                items.append(
                    f'  <div class="grid-stack-item"'
                    f' gs-id="{fig_uid_attr}"'
                    f' gs-x="{item.x}" gs-y="{item.y}" gs-w="{item.w}" gs-h="{item.h}">\n'
                    f'    <div class="grid-stack-item-content">{render_panel(idx, fig_spec)}</div>\n'
                    f"  </div>"
                )
            container_html = (
                '<div class="grid-stack">\n' + "\n".join(items) + "\n</div>"
            )
            return _DashboardMarkup(
                container_html=container_html,
                css=f".grid-stack {{ width: 100%; padding: {layout_gap}; box-sizing: border-box; }}",
                head_html=self._gridstack_css() + "\n  " + self._gridstack_script_tag(),
            )

        items = []
        for idx, fig_spec in enumerate(spec.figures):
            item = _grid_item_for(idx, fig_spec)
            items.append(
                f'  <div class="fv-dashboard-item" '
                f'style="grid-column:{item.x + 1} / span {item.w}; '
                f'grid-row:{item.y + 1} / span {item.h};">\n'
                f"    {render_panel(idx, fig_spec)}\n"
                f"  </div>"
            )
        container_html = '<div id="fv-dashboard">\n' + "\n".join(items) + "\n</div>"
        return _DashboardMarkup(
            container_html=container_html,
            css=(
                f"#fv-dashboard {{ width: 100%; display:grid; "
                f"grid-template-columns:repeat(12, minmax(0, 1fr)); "
                f"grid-auto-rows:{_GRIDSTACK_CELL_HEIGHT_PX}px; gap:{layout_gap}; "
                f"padding: 8px; box-sizing: border-box; }}\n"
                "#fv-dashboard .fv-dashboard-item { min-width: 0; min-height: 0; }"
            ),
            head_html="",
        )

    @staticmethod
    def _toolbar_html(toolbar: "Any | None" = None) -> str:
        """Return the shared FlexViz header HTML fragment.

        Includes the brand label and toolbar buttons.  Pass a ``ToolbarConfig``
        to hide individual buttons; empty groups are omitted automatically.
        Embed directly in the adapter's ``<body>`` before the chart container(s).
        """
        from ..spec import ToolbarConfig as _TC

        tc = toolbar if toolbar is not None else _TC()

        def _group(btns: list) -> str:
            if not btns:
                return ""
            return (
                '    <div class="fv-btn-group">\n' + "\n".join(btns) + "\n    </div>\n"
            )

        g1 = []
        if tc.show_reset:
            g1.append('      <button id="fv-btn-reset">Reset</button>')
        if tc.show_deselect:
            g1.append('      <button id="fv-btn-deselect">Deselect</button>')

        g2 = []
        if tc.show_cfmode:
            g2.append('      <button id="fv-btn-cfmode">CF: Update</button>')
        if tc.show_lock_all_axes:
            g2.append(
                '      <button id="fv-btn-lock-all" aria-pressed="false">Lock All Axes</button>'
            )
        if tc.show_hover:
            g2.append(
                '      <div id="fv-hover-dropdown">\n'
                '        <button id="fv-hover-btn" aria-pressed="false">Hover: Off</button>\n'
                "      </div>"
            )

        # Grid/layout control sits in its own group so a divider separates it
        # from the cross-filter / lock / hover controls to its left.
        g_grid = []
        if tc.show_grid:
            g_grid.append('      <button id="fv-btn-grid">Layout: Edit</button>')

        g3 = []
        if tc.show_share:
            g3.append('      <button id="fv-btn-share">Share</button>')
        if tc.show_export:
            g3.append('      <button id="fv-btn-export">Export</button>')
        if tc.show_import:
            g3.append(
                '      <input type="file" id="fv-import-input" style="display:none" accept=".json">'
            )
            g3.append('      <button id="fv-btn-import">Import</button>')

        inner = _group(g1) + _group(g2) + _group(g_grid) + _group(g3)
        return (
            '<header id="fv-header">\n'
            '  <div id="fv-header-main">\n'
            f'    <a id="fv-brand" href="{_BRAND_URL}" target="_blank"'
            ' rel="noopener noreferrer" aria-label="FlexViz"></a>\n'
            '    <div id="fv-toolbar">\n' + inner + "    </div>\n"
            "  </div>\n"
            "</header>"
        )

    @staticmethod
    def _filter_strip_html() -> str:
        """Return the floating active-filters bar HTML.

        Rendered as a ``position:fixed`` bottom bar outside the dashboard
        container so it overlays content without shifting panel layout.
        """
        return (
            '<div id="fv-filter-strip" hidden>\n'
            '  <span id="fv-filter-strip-label">Active filters:</span>\n'
            '  <div id="fv-filter-chips"></div>\n'
            "</div>\n"
        )

    @staticmethod
    def _toolbar_css() -> str:
        """Return shared CSS for the FlexViz toolbar header.

        Sourced from ``js/toolbar.css``; embed inside a ``<style>`` block after
        ``theme_css()``, which defines the ``--fv-*`` custom properties it uses.
        """
        return toolbar_css()

    # ------------------------------------------------------------------
    # Gridstack.js helpers (draggable layout)
    # ------------------------------------------------------------------

    @staticmethod
    def _gridstack_css() -> str:
        """Return the Gridstack CDN ``<link>`` tag plus minimal override CSS."""
        return (
            '<link rel="stylesheet"'
            ' href="https://cdn.jsdelivr.net/npm/gridstack@12.6.0/dist/gridstack.min.css">\n'
            "<style>\n"
            "  .grid-stack-item-content { overflow: hidden; }\n"
            "  .grid-stack-item-content > div { width: 100%; height: 100%; }\n"
            "</style>"
        )

    @staticmethod
    def _gridstack_script_tag() -> str:
        """Return the Gridstack CDN ``<script>`` tag."""
        return (
            '<script src="https://cdn.jsdelivr.net/npm/gridstack@12.6.0/dist/gridstack-all.js">'
            "</script>"
        )

    # ------------------------------------------------------------------
    # HTML delivery helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _deliver_notebook(html: str, height: int) -> None:
        """Display *html* inline as an IFrame in a Jupyter / VS Code notebook."""
        import base64

        from IPython.display import IFrame, display

        encoded = base64.b64encode(html.encode()).decode()
        data_uri = f"data:text/html;base64,{encoded}"
        display(IFrame(src=data_uri, width="100%", height=height))

    # @staticmethod
    # def _deliver_browser_local(html: str) -> None:
    #     """Write *html* to a temp file and open it in the default browser."""
    #     import tempfile
    #     import webbrowser

    #     with tempfile.NamedTemporaryFile(
    #         mode="w", suffix=".html", delete=False, encoding="utf-8"
    #     ) as f:
    #         f.write(html)
    #         path = f.name
    #     webbrowser.open(f"file://{path}")

    @staticmethod
    def _deliver_browser_shared(
        spec: DashboardSpec, server_url: str, renderer: str
    ) -> None:
        """POST *spec* to ``/share`` and open the returned URL in the browser."""
        import requests
        import webbrowser

        resp = requests.post(
            f"{server_url}/share",
            json={"spec": spec.model_dump(), "server_url": server_url},
            timeout=5,
        )
        resp.raise_for_status()
        url = resp.json()["url"]
        url += f"&renderer={renderer}"
        webbrowser.open(url)

    # ------------------------------------------------------------------
    # Server polling
    # ------------------------------------------------------------------

    @staticmethod
    def _wait_for_server(
        url: str, timeout: float = 10.0, interval: float = 0.1
    ) -> None:
        """Poll *url*/sources until the server responds or *timeout* elapses."""
        import requests

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                requests.get(f"{url}/sources", timeout=1)
                return
            except Exception:
                time.sleep(interval)
        warnings.warn(f"flexviz: server at {url} did not respond within {timeout}s")

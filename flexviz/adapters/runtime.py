"""Shared FlexViz client-side JS runtime.

This module provides the single source of truth for all renderer-agnostic
JS logic used by both the Plotly and ECharts adapters.  The runtime
handles delta application, overlay caching, grouped-child reconciliation,
linked-hover dispatch, toolbar hooks, and state management.

The JS source lives in ``js/`` as small modules.  This module concatenates
them into the renderer bundles **once at import time**, in pure Python, so no
Node and no separate build step are required — editing a file under ``js/`` is
picked up on the next import.  The concatenated strings are cached in module
globals; the public ``*_js()`` / ``theme_css()`` accessors just return them.

Each adapter must define the following globals **before** the shared
runtime is included:

- ``_fvAllFigUids``    — ``string[]`` of all figure UIDs
- ``_fvRenderFigure(figUid)`` — re-render one figure via the renderer
- ``_fvClearAllCrosshairs()`` — clear hover guides from all figures
- ``_fvShowCrosshair(figUid, axis, value)`` — draw a crosshair line
"""

from __future__ import annotations

from pathlib import Path

_JS_DIR = Path(__file__).parent / "js"

# Concatenation order for each renderer bundle.  Single source of truth for
# which js/ sources go into each assembled bundle.
_SHARED_SOURCES: list[str] = [
    "panel.js",
    "runtime/state.js",
    "runtime/cache.js",
    "runtime/cube.js",
    "runtime/selections.js",
    "runtime/delta.js",
    "runtime/selection-summary.js",
    "runtime/overlay.js",
    "runtime/hover.js",
    "toolbar.js",
]

_PLOTLY_SOURCES: list[str] = [
    "plotly/traces.js",
    "plotly/render.js",
    "plotly/events.js",
    "plotly/hover.js",
    "plotly/init.js",
]

_ECHARTS_SOURCES: list[str] = [
    "echarts/series.js",
    "echarts/render.js",
    "echarts/init.js",
]


def _read(rel: str) -> str:
    return (_JS_DIR / rel).read_text(encoding="utf-8")


def _concat(sources: list[str]) -> str:
    parts: list[str] = []
    for rel in sources:
        parts.append(f"// --- {rel} ---")
        parts.append(_read(rel))
    return "\n".join(parts)


# Assembled once at import; the accessors below hand back these cached strings.
_THEME_CSS = _read("theme.css")
_TOOLBAR_CSS = _read("toolbar.css")
_GRIDSTACK_BRIDGE_JS = _read("gridstack-bridge.js")
_SHARED_RUNTIME_JS = _concat(_SHARED_SOURCES)
_PLOTLY_BUNDLE_JS = _concat(_PLOTLY_SOURCES)
_ECHARTS_BUNDLE_JS = _concat(_ECHARTS_SOURCES)


def theme_css() -> str:
    """Return the FlexViz CSS design tokens (custom properties).

    Embed this string inside a ``<style>`` block.  Token values can be
    overridden by appending a ``<style>:root { --fv-accent: ...; }</style>``
    block after this one.
    """
    return _THEME_CSS


def toolbar_css() -> str:
    """Return the FlexViz toolbar/header CSS.

    Embed inside a ``<style>`` block after :func:`theme_css`, which defines the
    ``--fv-*`` custom properties this stylesheet references.
    """
    return _TOOLBAR_CSS


def shared_runtime_js() -> str:
    """Return the shared FlexViz JS runtime bundle.

    Embed this string inside a ``<script>`` block **after** the adapter
    has defined the four required globals listed in the module docstring.
    """
    return _SHARED_RUNTIME_JS


def gridstack_bridge_js() -> str:
    """Return the GridStack bridge bundle used by draggable dashboards."""
    return _GRIDSTACK_BRIDGE_JS


def plotly_bundle_js() -> str:
    """Return the Plotly-specific JS bundle."""
    return _PLOTLY_BUNDLE_JS


def echarts_bundle_js() -> str:
    """Return the ECharts-specific JS bundle."""
    return _ECHARTS_BUNDLE_JS

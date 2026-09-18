"""Markdown reports whose figures are live, embedded FlexViz dashboards.

An `fv:N` line in a findings document is not a picture of a chart: it names
a share URL recorded in ``flexviz.history``. ``expand`` turns such a line
into an iframe pointing at the real, still-interactive dashboard, or into
the bare URL for a plain-markdown copy. ``to_html`` wraps the expanded
markdown in a small page that renders it in the browser.
"""

from __future__ import annotations

import re

from . import history
from .adapters.base import (
    _STATIC_GRID_PADDING_PX,
    _html_attr,
    _json_for_inline_script,
)
from .spec import (
    _GRIDSTACK_CELL_HEIGHT_PX,
    DashboardSpec,
    VisualizationSpec,
    _auto_grid_items,
    decode_spec,
    encoded_spec_from_url,
)

# flexviz/adapters/js/theme.css --fv-toolbar-height (44px), plus the 1px
# border-bottom on #fv-header in flexviz/adapters/js/toolbar.css.
_HEADER_HEIGHT_PX = 45

# Pinned the way Gridstack is pinned in flexviz/adapters/base.py. The UMD
# builds are the ones that define the globals ``marked`` and ``DOMPurify``.
_MARKED_URL = "https://cdn.jsdelivr.net/npm/marked@18.0.13/lib/marked.umd.min.js"
_DOMPURIFY_URL = "https://cdn.jsdelivr.net/npm/dompurify@3.4.15/dist/purify.min.js"

_FV_LINE_RE = re.compile(r"fv:(\d+)")
# A fence opener: up to 3 spaces of indent, then 3 or more backticks or tildes.
_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")


def _embed_url(line: str, entries: list[dict]) -> str | None:
    """Return the share URL a report line should embed, or None for plain text."""
    stripped = line.strip()
    match = _FV_LINE_RE.fullmatch(stripped)
    if match:
        n = int(match.group(1))
        for entry in entries:
            if entry["n"] == n:
                return entry["url"]
        raise ValueError(f"no history entry {n}")
    # A URL has no spaces; this excludes prose that merely mentions one and,
    # importantly, an already-rendered <iframe ...> line (which does).
    if "/view?spec=" in stripped and " " not in stripped:
        return stripped
    return None


def _iframe_height(url: str) -> int:
    """Return the pixel height that fits a dashboard's grid with no scrollbar.

    Both layout paths render a panel at ``h * _GRIDSTACK_CELL_HEIGHT_PX`` and
    neither adds ``LayoutSpec.gap`` to the page height: GridStack carries the
    gutter in the panel margin and the static grid in the item padding. Only
    the grid container's own padding differs. Measured against real ``/view``
    pages by ``TestLockedLayoutBrowser`` in tests/test_browser.py.
    """
    spec = decode_spec(encoded_spec_from_url(url))
    if isinstance(spec, VisualizationSpec):
        # /view wraps a single-figure spec in a default 1-figure dashboard.
        spec = DashboardSpec(figures=[spec.figure])
    grid_items = spec.layout.grid_items or _auto_grid_items(spec.figures)
    rows = max((item.y + item.h for item in grid_items), default=0)
    height = rows * _GRIDSTACK_CELL_HEIGHT_PX + _HEADER_HEIGHT_PX
    if spec.layout.draggable:
        return height
    # Top plus bottom padding of the static grid container.
    return height + 2 * _STATIC_GRID_PADDING_PX


def expand(md: str, *, as_html: bool) -> str:
    """Replace each ``fv:N`` or raw share-URL line with its embed.

    ``as_html=False`` swaps in the bare URL, for a copy that degrades to a
    plain link on GitHub or in chat. ``as_html=True`` swaps in an iframe
    wrapped in blank lines, so a markdown renderer passes the HTML block
    through untouched instead of trying to parse it as prose.

    Lines inside a fenced code block are left alone: a report that documents
    the ``fv:N`` syntax writes it in a fence.
    """
    entries = history.entries()
    out = []
    fence_char = ""
    for line in md.splitlines():
        fence = _FENCE_RE.match(line)
        # A fence closes only on its own character, so a ``~~~`` line inside a
        # backtick fence stays content.
        if fence and fence_char in ("", fence.group(1)[0]):
            fence_char = "" if fence_char else fence.group(1)[0]
        url = None if fence_char else _embed_url(line, entries)
        if url is None:
            out.append(line)
        elif as_html:
            height = _iframe_height(url)
            out.append("")
            out.append(
                f'<iframe loading="lazy" src="{_html_attr(url)}" '
                f'style="width:100%;height:{height}px;border:0"></iframe>'
            )
            out.append("")
        else:
            out.append(url)
    return "\n".join(out)


def to_html(md: str) -> str:
    """Wrap expanded markdown in a minimal report page.

    The page needs the network: it loads ``marked`` and ``DOMPurify`` from a
    CDN, and its embedded dashboards only render while the servers they point
    at run.
    """
    expanded = expand(md, as_html=True)
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>FlexViz report</title>
  <script src="{_MARKED_URL}"></script>
  <script src="{_DOMPURIFY_URL}"></script>
  <style>
    body {{
      max-width: 900px; margin: 2rem auto; padding: 0 1rem;
      font-family: system-ui, sans-serif; line-height: 1.5;
    }}
    iframe {{ display: block; width: 100%; margin: 1rem 0; }}
    footer {{ margin-top: 2rem; color: #666; font-size: 0.85rem; }}
  </style>
</head>
<body>
  <div id="fv-report"></div>
  <footer>
    Embedded dashboards render only while the FlexViz server named in their
    URLs is running.
  </footer>
  <script id="fv-md" type="text/markdown">{_json_for_inline_script(expanded)}</script>
  <script>
    document.getElementById("fv-report").innerHTML = DOMPurify.sanitize(
      marked.parse(JSON.parse(document.getElementById("fv-md").textContent)),
      {{ADD_TAGS: ["iframe"], ADD_ATTR: ["loading"]}});
  </script>
</body>
</html>"""

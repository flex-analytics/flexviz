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
from .adapters.base import _json_for_inline_script
from .cli import _encoded_from
from .spec import _GRIDSTACK_CELL_HEIGHT_PX, _auto_grid_items, decode_spec

# flexviz/adapters/js/theme.css --fv-toolbar-height
_TOOLBAR_HEIGHT_PX = 44
# grid container's top + bottom padding (flexviz/adapters/base.py _dashboard_markup)
_GRID_PADDING_PX = 16

# Pinned the way Gridstack is pinned in flexviz/adapters/base.py.
_MARKED_VERSION = "12.0.2"

_FV_LINE_RE = re.compile(r"fv:(\d+)")


def _embed_url(line: str, entries: list[dict]) -> str | None:
    """Return the share URL a report line should embed, or None for plain text."""
    stripped = line.strip()
    match = _FV_LINE_RE.fullmatch(stripped)
    if match:
        n = int(match.group(1))
        for entry in entries:
            if entry["n"] == n:
                return entry["url"]
        raise SystemExit(f"no history entry {n}")
    # A URL has no spaces; this excludes prose that merely mentions one and,
    # importantly, an already-rendered <iframe ...> line (which does).
    if "/view?spec=" in stripped and " " not in stripped:
        return stripped
    return None


def _gap_px(gap: str) -> int:
    """Parse a plain CSS px gap like "8px"; non-px gaps fall back to the spec default."""
    match = re.fullmatch(r"(\d+(?:\.\d+)?)px", gap.strip())
    return round(float(match.group(1))) if match else 8


def _iframe_height(url: str) -> int:
    """Return the pixel height that fits a dashboard's grid with no scrollbar.

    Mirrors the two layout formulas in docs/guides/customizing.md
    "Exact panel positions": GridStack (``draggable=True``) sizes a panel at
    ``h * 80``; the static grid (``draggable=False``) stretches a panel
    across the row gaps it spans (issue #51), so its rows also add
    ``(rows - 1) * gap``.
    """
    spec = decode_spec(_encoded_from(url))
    grid_items = spec.layout.grid_items or _auto_grid_items(spec.figures)
    rows = max(item.y + item.h for item in grid_items)
    grid_height = rows * _GRIDSTACK_CELL_HEIGHT_PX
    if not spec.layout.draggable:
        grid_height += (rows - 1) * _gap_px(spec.layout.gap)
    return grid_height + _TOOLBAR_HEIGHT_PX + _GRID_PADDING_PX


def expand(md: str, *, as_html: bool) -> str:
    """Replace each ``fv:N`` or raw share-URL line with its embed.

    ``as_html=False`` swaps in the bare URL, for a copy that degrades to a
    plain link on GitHub or in chat. ``as_html=True`` swaps in an iframe
    wrapped in blank lines, so a markdown renderer passes the HTML block
    through untouched instead of trying to parse it as prose.
    """
    entries = history.entries()
    out = []
    for line in md.splitlines():
        url = _embed_url(line, entries)
        if url is None:
            out.append(line)
        elif as_html:
            height = _iframe_height(url)
            out.append("")
            out.append(
                f'<iframe loading="lazy" src="{url}" '
                f'style="width:100%;height:{height}px;border:0"></iframe>'
            )
            out.append("")
        else:
            out.append(url)
    return "\n".join(out)


def to_html(md: str) -> str:
    """Wrap expanded markdown in a minimal, self-contained report page."""
    expanded = expand(md, as_html=True)
    # ponytail: the report is generated locally and read by its own author,
    # so marked's default HTML pass-through (needed for the embedded
    # iframes) is accepted as-is instead of sanitized.
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>FlexViz report</title>
  <script src="https://cdn.jsdelivr.net/npm/marked@{_MARKED_VERSION}/marked.min.js"></script>
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
    document.getElementById("fv-report").innerHTML =
      marked.parse(JSON.parse(document.getElementById("fv-md").textContent));
  </script>
</body>
</html>"""

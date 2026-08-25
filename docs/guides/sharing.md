# Sharing views

Every FlexViz view is fully described by its spec: figures, traces, viewport,
selections, cross-filter mode, hover mode, and dashboard layout. Because the
server stores no interaction state, sharing a view means sharing the spec.

## Share URLs

The toolbar's **share** button encodes the complete current spec
(gzip + base64url) into a single URL. Anyone who opens it against the same
server sees the exact live view: same zoom, same selections, same panel
arrangement, still fully interactive. The server stores nothing per share;
the URL *is* the state.

Rearranging panels in the dashboard grid travels the same way: drag, resize,
lock the layout, then share.

## Export and import

The toolbar can also **export** the spec as a JSON file and **import** one,
which is the same round-trip through a file instead of a URL.

Programmatically, specs save and load the same way:

```python
dash.save_spec("dashboard.json", source_name="readings")

from flexviz import Dashboard
spec = Dashboard.load_spec("dashboard.json")
```

`Figure.save_spec` / `Figure.load_spec` / `Figure.from_spec` do the same for
a single figure. A loaded spec keeps its uids, so restoring it reconnects to
the same traces.

!!! warning "Specs are versioned, not stable"
    Pre-1.0, shared URLs and exported specs are only guaranteed to round-trip
    within the same minor version of FlexViz. Treat them as a way to share a
    view, not as a long-term storage format.

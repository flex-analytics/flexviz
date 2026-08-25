# Cross-filtering

Figures that share a data source cross-filter each other: select something in
one figure and every other figure re-aggregates against the matching rows.
This is automatic for all figures in a `Dashboard` and for traces within one
`Figure`. No wiring code is needed.

```python
import polars as pl
from flexviz import Dashboard

dash = Dashboard(pl.scan_parquet("trips.parquet"))
dash.add_figure().add_line(x="pickup_time", y="fare")
dash.add_figure().add_histogram(x="distance", bins=40)
dash.add_figure().add_bar(labels="vendor")
dash.show()
```

Brush a time range on the line: the histogram and bar chart re-aggregate over
that window. Click a vendor bar: the line and histogram re-aggregate over that
vendor. Selections on different figures combine with AND.

## What each trace selects on

How a figure emits a selection depends on its trace type:

| Trace | Selects on |
|---|---|
| `line` | x-axis band brush (a vertical band across all series) |
| `histogram`, `box` | range brush on the data axis (the count axis is ignored) |
| `histogram2d` | 2-D box select on both axes |
| `bar`, `pie` | category click on a bar or slice |
| `treemap` | hierarchical path click (a node selects its subtree) |
| `geo_histogram2d` | map box select (longitude and latitude bounds) |
| `corr_heatmap`, `geo_line` | not a cross-filter source |

Range selections become typed `is_between` filters and category clicks become
`is_in` filters, compiled into Polars expressions and applied lazily before
any aggregation. A figure is never filtered by its own selection, only by the
selections of other figures.

## Update vs. overlay mode

The toolbar's cross-filter mode toggle switches how filtered results render:

- **Update** (default): each filtered figure replaces its data with the
  filtered aggregate.
- **Overlay**: each filtered figure keeps the unfiltered aggregate as a muted
  background layer and draws the filtered aggregate on top, so you see the
  selection in context.

The mode travels with the spec (`cross_filter_mode`), so it persists through
[shared URLs](sharing.md).

## Clearing selections

- The toolbar's **deselect** button (or a Plotly double-click deselect)
  clears selections but keeps zoom.
- The toolbar's **reset** button clears selections and viewport both.

## Zoom is not a filter

Zooming a figure re-aggregates that figure to its viewport (a zoomed line
re-downsamples, a zoomed histogram re-bins), but it does not filter the other
figures. Only selections cross-filter. Categorical traces (bar, pie, treemap)
never re-aggregate on zoom.

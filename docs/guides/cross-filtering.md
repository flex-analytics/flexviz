# Cross-filtering

Figures that share a data source cross-filter each other: select something in
one figure and every other figure re-aggregates against the matching rows.
This is automatic for all figures in a `Dashboard`. Traces inside one `Figure`
do not filter each other. No wiring code is needed.

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
- The toolbar's **reset** button clears selections and viewport both. A
  locked axis keeps its range.

## Zoom is not a filter

Zooming a figure re-aggregates that figure to its viewport (a zoomed line
re-downsamples, a zoomed histogram re-bins), but it does not filter the other
figures. Only selections cross-filter. Categorical traces (bar, pie, treemap)
never re-aggregate on zoom. To make other figures follow a zoom, link their
axes.

## Linked axes

Linked axes always show the same range. Zoom, pan, double-click or reset one of
them, and the others follow. Each figure re-aggregates at the new range, all in
one request: a line re-downsamples, a histogram re-bins.

```python
dash = Dashboard(pl.scan_parquet("sensors.parquet"))
temp = dash.add_figure()
temp.add_line(x="timestamp", y="temperature")
hum = dash.add_figure()
hum.add_line(x="timestamp", y="humidity")
dash.add_figure().add_histogram(x="timestamp", bins=100)
dash.link_axes(on="timestamp")
dash.show()
```

`link_axes` has three forms:

| Call | Links |
|---|---|
| `link_axes(on="timestamp")` | the axis showing `timestamp` in every figure that shows it |
| `link_axes(temp, hum, on="timestamp")` | that axis in the given figures only |
| `link_axes(temp, hum, axis="y")` | the same axis of each figure |
| `link_axes((temp, "x"), (hist, "y"))` | the named axes, e.g. a line's x and the y of a horizontal histogram `hist` |

Calls that share an axis merge into one group. Links are resolved when the
dashboard is built, so figures added after the call count too.

Rules and limits:

- Only x and y axes that show a numeric, date or datetime column can be
  linked, one axis per figure in a group. A histogram's count axis, a bar, a
  map, a log axis and a time-of-day or duration axis cannot. Numeric and
  temporal axes do not mix; date and datetime axes do.
- Locking the axes of one figure locks every axis linked to them.
- After a double-click autorange, each figure autoranges to its own data. On
  one shared column the ranges usually match; on a filtered figure or a
  histogram they can differ slightly until the next zoom.
- A linked y of a line is display-only: the line does not re-aggregate on y.
- The links live in `client_state.axis_links`, so they survive
  [shared URLs](sharing.md), export and import. A spec whose linked axes hold
  different ranges is rejected, so an agent that patches the viewport with
  `flexvizApply` must set every key of a group.
- The deprecated ECharts renderer does not support linked axes.

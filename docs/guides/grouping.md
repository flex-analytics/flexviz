# Grouping

`group_by` splits a trace into one child series per distinct value of a
category column, computed in a single grouped Polars query.

```python
import polars as pl
from flexviz import Dashboard

dash = Dashboard(pl.scan_parquet("sensors.parquet"))
dash.add_figure().add_line(x="timestamp", y="value", group_by="sensor_id")
dash.add_figure().add_histogram(x="value", bins=30, group_by="sensor_id")
dash.show()
```

`group_by` is supported on `add_line`, `add_histogram`, and `add_boxplot`.
`add_bar` uses it as a second grouping dimension next to `labels` (a hue
split), rendered side by side or stacked via `bar_mode="group" | "stack"`.

## Composite groups

`group_by` accepts a list of columns; each distinct combination becomes one
series:

```python
fig.add_line(x="timestamp", y="value", group_by=["site", "sensor_id"])
```

The same applies to `labels` on `add_bar` and `add_pie`. Selecting a
composite category cross-filters on every source column of the combination.

## Colors

Group colors are assigned from the renderer palette in first-seen order and
stay stable while you interact: a group keeps its color across zooms,
filters, and shared URLs. To pin specific colors, pass a `color_map`:

```python
fig.add_line(
    x="timestamp", y="value", group_by="sensor_id",
    color_map={"s1": "#345d8f", "s2": "#d19a32"},
)
```

Grouped histograms share one set of bin edges across all groups (computed
from the pre-filter global min and max), so bars of different groups line up
and stay comparable.

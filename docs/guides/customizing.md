# Customizing

FlexViz splits appearance into two layers.

- **Figure appearance** is per chart: title, axis labels, legend, colors.
  It travels in `FigureSpec.layout`.
- **Dashboard layout** is per page: panel positions, gap, toolbar buttons.
  It travels in `LayoutSpec`.

The two never overlap. A figure cannot resize its panel, and a dashboard
cannot move a legend.

## Figure appearance

### Titles and axis labels

```python
fig = dash.add_figure()
fig.add_line(x="timestamp", y="power")
fig.title("Power draw").xlabel("Time").ylabel("Watt")
```

FlexViz derives both axis labels from the trace columns when you set none.

### Show or hide the legend

```python
fig.legend(False)
```

With no call, the legend appears when the figure has more than one trace or a
grouped trace.

### Put the legend below the chart

Placement is renderer-specific, so it goes through `update_layout()`. Every
key that is not `title`, `xlabel`, `ylabel` or `legend` passes straight to the
renderer. With Plotly, that is any Plotly layout option.

```python
fig.legend(True).update_layout(
    legend={"orientation": "h", "y": -0.25, "yanchor": "top",
            "x": 0.5, "xanchor": "center"},
    margin={"t": 40, "b": 80, "l": 60, "r": 20},
)
```

A legend below the chart sits outside the plotting area. Increase the bottom
margin, or the legend overlaps the x-axis labels.

Passing a `legend` dict also makes the legend visible, so `legend(True)` is
optional here.

### Other renderer options

```python
fig.ylabel("Power").update_layout(
    yaxis={"type": "log"},
    colorway=["#345d8f", "#d19a32"],
    font={"family": "DM Sans", "size": 12},
)
```

Dict values merge one level deep. The example keeps the y-axis label and adds
a log scale. A replacing merge would erase the label.

!!! warning "Plotly only"
    ECharts is deprecated and reads `title`, `xlabel`, `ylabel` and legend
    visibility only. Every other `update_layout()` key is ignored there.

### Colors

Pass `color` for a single trace and `color_map` for a grouped one. See
[Grouping](grouping.md) for how group colors stay stable across interactions.

## Dashboard layout

### Rows and columns

The grid is 12 columns wide. By default figures are placed two per row, so a
single figure fills half the width.

```python
dash.show(cols=1)   # one full-width column
dash.show(rows=2)   # two rows, columns derived from the figure count
```

`rows` and `cols` are mutually exclusive.

### Lock the layout

```python
dash.show(draggable=False)
```

This disables drag and resize, and hides the toolbar's layout button.

`LayoutSpec.grid_editable` is not a lock. It sets the starting mode only, and
the toolbar button can still turn editing on. Use `draggable=False` when the
layout must stay fixed.

### Exact panel positions

`GridItem` places one figure on the 12-column grid. One row unit is 80 px, and
a panel spanning `h` rows also spans the `h - 1` gaps between them. On a locked
grid the panel is `h * 80 + (h - 1) * gap` pixels tall, so the default `h=5`
with the default 8 px gap gives 432 px.

```python
from flexviz.spec import GridItem, LayoutSpec

uids = [f.uid for f in dash.to_spec().figures]
dash.show(
    layout=LayoutSpec(
        gap="16px",
        grid_items=[
            GridItem(fig_uid=uids[0], x=0, y=0, w=12, h=4),   # full width, 368 px at gap 16
            GridItem(fig_uid=uids[1], x=0, y=4, w=6, h=8),    # half width, 752 px at gap 16
            GridItem(fig_uid=uids[2], x=6, y=4, w=6, h=8),
        ],
    )
)
```

`grid_items` cannot be combined with `rows` or `cols`. Passing both raises a
`ValueError`, because the two describe the same thing.

### Toolbar buttons

`ToolbarConfig` hides buttons in the top toolbar. Every button shows by
default.

```python
from flexviz.spec import LayoutSpec, ToolbarConfig

dash.show(
    draggable=False,
    layout=LayoutSpec(toolbar=ToolbarConfig(show_export=False, show_import=False)),
)
```

The fields are `show_reset`, `show_deselect`, `show_cfmode`, `show_hover`,
`show_lock_all_axes`, `show_grid`, `show_share`, `show_export` and
`show_import`. An empty button group disappears with its divider.

### Toolbar versus panel controls

The top toolbar holds dashboard-wide state: reset, deselect, cross-filter
mode, hover, axis lock, layout, share, export and import. `ToolbarConfig`
controls these.

Each panel also has its own bottom bar with Zoom, Pan, CF, Reset and axis
lock. These follow what the traces in that panel support, and they are not
configurable today.

## Precedence

`layout` owns every field it sets. `rows`, `cols` and `draggable` are
convenience overrides that apply on top of it.

| You pass | Result |
|---|---|
| nothing | Two columns, default `h=5` panels, drag enabled |
| `cols=1` | One full-width column |
| `layout.grid_items` | Your positions, untouched |
| `layout.grid_items` and `cols` | `ValueError` |
| `draggable=False` | Overrides `layout.draggable` |
| `layout.draggable=False` only | Honored, because `draggable` defaults to `None` |

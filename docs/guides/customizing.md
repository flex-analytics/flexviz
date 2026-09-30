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

Plotly owns both dicts. Its
[legend guide](https://plotly.com/python/legend/) shows the positioning
patterns, and the reference lists every key of
[`layout.legend`](https://plotly.com/python/reference/layout/#layout-legend)
and [`layout.margin`](https://plotly.com/python/reference/layout/#layout-margin).

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

Every key comes from the
[Plotly layout reference](https://plotly.com/python/reference/layout/).

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

This renders a static CSS grid. Panels cannot move, and the layout button
disappears.

Three separate fields decide how rearranging works. Each answers a different
question.

| Field | Question it answers | Fixed when the page renders |
|---|---|---|
| `LayoutSpec.draggable` | Which layout engine renders the page: GridStack, or a static CSS grid | Yes |
| `LayoutSpec.grid_editable` | Whether panels can be moved and resized right now | No, this is live state |
| `ToolbarConfig.show_grid` | Whether the toolbar offers the built-in lock and unlock button | Yes |

`draggable` is a rendering choice, `grid_editable` is live state, and
`show_grid` is an affordance. They combine into five results.

| `draggable` | `grid_editable` | `show_grid` | Result |
|---|---|---|---|
| `False` | ignored | ignored | Static grid, no GridStack assets. The layout button is hidden either way. |
| `True` | `False` | `True` | GridStack, locked. The button reads **Layout: Locked** and unlocks. |
| `True` | `True` | `True` | GridStack, editable. The button reads **Layout: Edit** and locks. |
| `True` | `False` | `False` | GridStack, locked, with no built-in layout button. |
| `True` | `True` | `False` | GridStack, editable, with no built-in layout button to lock it. |

Read the last two rows carefully. `show_grid=False` hides the built-in
control. It does not remove the capability: a layout that starts editable
stays editable, and your own JavaScript can call `fvSetGridEditable`. With
`show_import=True`, an imported spec can also set `grid_editable` again. Set
both `show_grid=False` and `show_import=False` to remove the built-in routes
to that state. Use `draggable=False` when the layout must not move at all.

!!! note "GridStack is a CDN dependency"
    `draggable=True` loads GridStack's stylesheet and script from a CDN.
    `draggable=False` loads neither. Prefer it for a locked embed, and in any
    page that must not reach a CDN.

`grid_editable` keeps its value when `draggable=False`. It is inactive, but it
is not cleared, so specs round-trip unchanged.

### Exact panel positions

`GridItem` places one figure on the 12-column grid. One row unit is 80 px, so
a panel of `h` rows is `h * 80` pixels tall. The default `h=5` is 400 px.

Both layout engines give the same height, because the space between panels sits
inside the `h * 80` box. On the static grid (`draggable=False`) that space is
`gap`. GridStack keeps its own panel margin, so there `gap` only pads the outer
edge of the grid.

```python
from flexviz import GridItem, LayoutSpec

uids = [f.uid for f in dash.to_spec().figures]
dash.show(
    layout=LayoutSpec(
        gap="16px",
        draggable=False,
        grid_items=[
            GridItem(fig_uid=uids[0], x=0, y=0, w=12, h=4),   # full width, 320 px
            GridItem(fig_uid=uids[1], x=0, y=4, w=6, h=8),    # half width, 640 px
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
from flexviz import LayoutSpec, ToolbarConfig

dash.show(
    draggable=False,
    layout=LayoutSpec(toolbar=ToolbarConfig(show_export=False, show_import=False)),
)
```

The fields are `show_reset`, `show_deselect`, `show_cfmode`, `show_hover`,
`show_lock_all_axes`, `show_grid`, `show_share`, `show_export` and
`show_import`. An empty button group disappears with its divider. The
**Mode** button always shows: see [Light and dark mode](#light-and-dark-mode).

Most of these hide a button whose state you can reach another way.
`show_grid` is different: it hides the only built-in button for changing
`grid_editable`. `show_import` is a separate route for restoring that state.
See [Lock the layout](#lock-the-layout) for what these flags do and do not
change.

### Toolbar versus panel controls

The top toolbar holds dashboard-wide state: reset, deselect, cross-filter
mode, hover, axis lock, layout, share, export and import. `ToolbarConfig`
controls these.

Each panel also has its own bottom bar with Zoom, Pan, CF, Reset and axis
lock. These follow what the traces in that panel support, and they are not
configurable today.

## Light and dark mode

The page has a light and a dark mode. The **Mode** button in the toolbar
cycles **Auto**, **Light** and **Dark**. Auto follows the system setting and
changes with it. The browser remembers a Light or Dark choice for the server
address.

The mode is a viewer preference. It is not part of the spec, so a share URL
or an exported spec does not carry it. A switch sends no request to the
server and keeps your zoom and selections.

Both modes use the Okabe-Ito series colors, which stay apart for the common
types of color blindness. The colors do not change with the mode, so a group
keeps its color after a switch. In dark mode, maps darken the OpenStreetMap
tiles, and the `"Viridis"` scale starts at a lighter blue, so sparse cells
stay visible. Other color scales do not change.

### Override the theme

`update_layout()` wins over the theme in both modes:

```python
fig.update_layout(plot_bgcolor="#f5f5f5")        # a cartesian figure
map_fig.update_layout(map={"style": "white-bg"})  # a map figure
```

The page look comes from CSS custom properties (`--fv-*`). FlexViz has no
Python option for them. If you serve the `/view` page through your own
proxy, add a style rule after the FlexViz styles:

```css
/* Both modes */
:root, :root[data-fv-mode="dark"] { --fv-accent: #0d9488; }
/* Dark mode only */
:root[data-fv-mode="dark"] { --fv-accent: #2dd4bf; }
```

The dark block outranks a plain `:root` rule. To change a token in both
modes, name both selectors. Keep `--fv-series` the same in both modes,
because the spec stores each group's color as a hex value. The full token
list is in `flexviz/adapters/js/theme.css`.

## Precedence

`layout` owns every field it sets. `rows`, `cols` and `draggable` are
convenience overrides that apply on top of it.

| You pass | Result |
|---|---|
| nothing | Two columns, default `h=5` panels, GridStack locked initially |
| `cols=1` | One full-width column |
| `layout.grid_items` | Your positions, untouched |
| `layout.grid_items` and `cols` | `ValueError` |
| `draggable=False` | Overrides `layout.draggable` |
| `layout.draggable=False` only | Honored, because `draggable` defaults to `None` |

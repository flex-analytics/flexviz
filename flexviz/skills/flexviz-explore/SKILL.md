---
name: flexviz-explore
description: >
  Serve a LOCAL Parquet/CSV file as a live, interactive FlexViz dashboard, read
  back the zooms and selections a human makes in it, change it from your side,
  and report the findings. Use when the user wants to explore or visualize a
  large local dataset (too big to plot inline), asks for an interactive or
  cross-filter dashboard, or hands you a flexviz /view URL. Prerequisites: the
  data file is on this machine, flexviz is installed in the project environment,
  and the human's browser can reach this machine. Do NOT use for small datasets
  that fit an inline plot, or in remote sessions where the human cannot reach
  the served port.
---

# Explore large data with FlexViz

FlexViz serves interactive cross-filter dashboards from lazy Polars queries. The
data stays server-side; the browser gets bounded aggregates and small samples of
real values. Parquet larger than RAM streams, so peak memory stays flat as rows
grow (box plots excepted).

Every dashboard view is a URL that carries the complete spec: 1 to 4 KB, and a
browser tool echoes the page URL in every snapshot. So you never hold one.
Record each URL under a number with `flexviz history`, open it at `/h/N`, and
call it `fv:N`.

Privacy: rows stay in the lazy engine unless you collect them. The schema, your
samples, and the ranges the human selects do enter your context. A share URL
embeds column names and filters, so add `.flexviz/` to `.gitignore`.

## The loop

### 1. Inspect the schema

```bash
flexviz schema data.parquet
```

Not on PATH? Use the project venv (`.venv/bin/flexviz` or `uv run flexviz`) and
keep that form below. If the package is absent, ask the human before installing.

The output names the file, the source, and every column with its dtype. Pick an
x column (usually time) and the columns worth plotting. A `.head(5).collect()`
peek is fine. Never collect the full frame.

### 2. Serve the file (background process)

```bash
flexviz serve data.parquet --cache --port 8077
```

Run the server, and every command below, from the project directory: the history
file (`.flexviz/history.jsonl`) and `/h/N` both resolve against a working
directory.

- Each file becomes a source named by its stem (`data.parquet` -> `"data"`).
- `--cache` enables cross-filter cubes and live brushing. Use it whenever
  the file will not change while serving.
- Wait for readiness: poll until the answer names YOUR source, for example
  `until curl -s http://127.0.0.1:8077/sources | grep -q '"data"'; do sleep 1; done`.
  A bare "it answered" check is not enough, because another server can own
  the port and answer with its own source list. On a busy port, `flexviz
  serve` exits with `cannot bind ...`. Read the serve log, then retry.
- Keep the process running while the human explores, tell them the port and
  PID, and stop it at the end. An orphan server otherwise holds the port.
- To add files later, restart with the FULL file list (old + new), same port.
- Serve on loopback. Another interface exposes unauthenticated endpoints, so
  only do it if the human explicitly accepts that.

`flexviz serve` scans the file as stored and cannot cast a column, so a
timestamp held as `String` gives a string x axis with no warning. If a dtype
needs a cast, register the source yourself and build step 3 on that LazyFrame:

```python
import polars as pl, uvicorn
from flexviz.server import app, register_source

lf = pl.scan_parquet("data.parquet").with_columns(pl.col("timestamp").str.to_datetime())
register_source("data", lf, cache=True)   # call this before uvicorn.run
uvicorn.run(app, host="127.0.0.1", port=8077)
```

### 3. Build the dashboard and record it

Build and record in ONE script, and print only the number:

```python
import polars as pl
from flexviz import Dashboard, history

dash = Dashboard(pl.scan_parquet("data.parquet"), cache=True)
dash.add_figure().add_line(x="timestamp", y="value", group_by="sensor_id")
dash.add_figure().add_histogram(x="value", bins=50)
url = dash.share_url(server_url="http://127.0.0.1:8077", source_name="data")
print(history.add(url, note="line + histogram, initial view", actor="agent"))
```

- This script only builds a spec: cheap, lazy, and it exits at once. The serve
  process answers every interaction.
- `source_name` must match the served stem, and `cache=True` must match
  `--cache`.
- No categorical column to `group_by`? Split metrics across figures instead,
  and use `add_corr_heatmap` or `add_histogram2d` to relate numeric columns.

### 4. Open the entry

Open `http://127.0.0.1:8077/h/N` for the number step 3 printed, and give the
human the same address. Tell them: drag on one chart to cross-filter the
others, zoom to re-aggregate at higher detail, double-click to reset.

### 5. Read the state back

All interaction state lives in ONE browser tab, so which tab the human uses
decides what you can read. Ask once: "do you use the window I open, or your own
browser?" Assume separate browsers until they confirm otherwise.

**Shared tab.** Your browser tool drives a window on this machine that the human
can also use: a headed Playwright session (never `--headless`), or an extension
attached to the human's own browser. One tab holds the state, so you read it:

```js
window.flexvizState({compact: true})   // via your browser evaluate tool
```

It returns `{version, state, client_state, revision}`. A rising `revision`
means the human moved. Example, `fig1` zoomed on x with one brush selection on `value`:

<!-- compact-state -->
```json
{"version": "0.6", "revision": 3, "state": {
  "viewport": {"fig1/x": {"min": 100, "max": 300}},
  "selections": [{"source_figure_uid": "fig1",
    "predicates": [{"clauses": [{"column": "value", "range": [10.0, 25.0]}]}]}],
  "group_domains": {}, "cross_filter_mode": "update"},
 "client_state": {"hover_mode": "off", "live_brush": "auto",
  "axis_locks": {}, "axis_lock_ranges": {}}}
```

- `version`/`revision`: the spec version, and a counter that bumps on any change.
- `state`: server-visible — viewport (`fig_uid/axis`), selections, group_domains, cross_filter_mode.
- `client_state`: client-only display state; the server ignores it.

**Separate browsers.** A headless session, or a human at another machine. Your
tab and their tab hold independent state, so polling yours tells you nothing
about their zooms and selections. They hand the state over: ask them to click
**Share** in the toolbar (it copies a URL that captures their current view) and
to run this in the project directory:

```bash
flexviz history add "<paste it here>" --actor human --note "what I was looking at"
```

They tell you only the number it prints; you read it with `flexviz history show
N`. If they will not run a command, let them paste the URL and run it
yourself. That is the one place a URL enters your context, and it enters once.
The address bar does NOT track interactions; only Share captures them.

### 6. Record the human's state before you act

With separate browsers the human's `history add` already made this entry, so
skip the step. In the shared tab the page cannot write the history file, so you
record what you read. `record_state` takes the figures from the entry they
opened and the state you just read:

```python
from flexviz import history
print(history.record_state(3, compact["state"], compact["client_state"],
                           note="human brushed 09:00-11:00 on sensor 12"))
```

It rewrites only the `spec=` value of the recorded URL, so the host and port
come from the entry and the URL never reaches your context. Do it before you
change anything: this entry is what the report cites later.

### 7. Change the dashboard

A state change (viewport, selections, hover mode, axis locks) goes through the
write half, in the tab you control. Smallest payloads, figure `F` (a datetime axis
stores the date string Plotly reports, e.g. `"2024-01-01 00:00:00"`, not ISO-8601):

```js
await window.flexvizApply({state: {viewport: {"F/x": {min: a, max: b}}}})
await window.flexvizApply({state: {selections: []}})   // clear all selections
```

Top-level keys replace. `state` and `client_state` merge one level deep, so a
partial `state` keeps its sibling keys (`viewport`, `group_domains`,
`cross_filter_mode`). It re-renders and resolves with the compact state. With
separate browsers the human does not see it, so record the state with
`record_state` and give them the new `/h/N` instead.

A structure change (add or remove a figure, change traces, change layout) needs
a new spec, because panels are built server-side. Rebuild in Python as in step
3, `history.add(...)`, then open the new `/h/N`. Note your own entries too, so
the history reads as a sequence of who did what.

### 8. Report the findings

Write plain markdown. A line that holds nothing but `fv:5` becomes the live
dashboard of entry 5. Put your prose around such lines, then run:

```bash
flexviz report findings.md -o findings.html --md findings.expanded.md
```

`findings.html` embeds each entry as a real, zoomable dashboard, for as long as
the server runs. `findings.expanded.md` holds bare links instead, for GitHub or
chat. Filter the LazyFrame to the brushed range to put numbers next to a figure.

## API cheat-sheet

Defaults shown. The `#` notes mark the arguments whose meaning the signature
does not give: those are where a call runs cleanly and charts the wrong thing.
For the rest, read `flexviz/figure.py` or https://docs.flexviz.tech.

```python
Dashboard(data, cache=False)     # data: pl.LazyFrame/DataFrame, pandas, pyarrow
                                 # cache=True enables cross-filter cubes (live brushing)
dash.add_figure(title=...)       # -> Figure; chainable builders below
dash.share_url(server_url, source_name, rows=None, cols=None,
               draggable=None,   # GridStack, locked initially; False = static/read-only
               cache=None, live_brush=None, layout=None)
LayoutSpec(gap="8px", draggable=True, grid_editable=False,
           grid_items=None,      # [GridItem(...)]; None auto-places the figures
           toolbar=ToolbarConfig())   # show_reset/deselect/cfmode/hover/
                                      #   lock_all_axes/grid/share/export/import
GridItem(fig_uid, x=0, y=0, w=6, h=5)  # 12-column grid; height is h * 80 px
history.add(url, note="", actor="agent")   # -> int; the only place a URL belongs

fig.add_line(x, y, name=None, color=None, n_points=1000,
             downsample="minmax",          # or "lttb" | "fpcs" | "nth"
             assume_sorted_x=False, group_by=None)
fig.add_histogram(x=None, y=None, bins=20, histnorm="count",
                  name=None, group_by=None)
fig.add_bar(labels, values=None,           # values=None counts rows per label
            agg="sum",                     # "mean"|"median"|"min"|"max"|"n_unique";
                                           #   ignored while values is None
            orientation="v", bar_mode="group", group_by=None)
fig.add_boxplot(x=None, y=None, name=None, group_by=None)
fig.add_pie(labels, values=None, agg="sum", hole=0.0)  # values=None counts rows
fig.add_treemap(path=[...], values=None, agg="sum")   # path: hierarchy columns;
                                                      #   values=None counts rows
fig.add_histogram2d(x, y, x_bins=20, y_bins=20,
                    z=None, histfunc=None)   # z + histfunc are a pair: both or
                                             #   neither, either alone raises;
                                             #   no z counts rows.
fig.add_corr_heatmap(columns=None, method="pearson", triangular=False)
fig.add_geo_histogram2d(lat, lon, lat_bins=64, lon_bins=64,
                        z=None, histfunc=None)  # same pair rule as above
# The bin grid spans the full lat/lon extent, so a few far-out points push every
#   real point into one cell. Zoom the map: the grid then spans the map bounds.
fig.add_geo_line(lat, lon, n_points=1000)

fig.title(text); fig.xlabel(text); fig.ylabel(text); fig.legend(show=True)
fig.update_layout(**plotly_layout)   # any Plotly layout key passes through;
                                     #   dicts merge one level deep
```

Legend placement, margins, log axes and fonts are renderer options, so they go
through `update_layout`. A legend below the chart needs `margin={"b": 80}` too.
Panel size is dashboard layout: `cols=1` gives a full-width panel and
`GridItem.h` sets the height. `group_by="col"` splits a trace into one child
per group value, with stable colors.

## Rules

Token discipline:

- Never print or restate a share URL. Refer to entries as `fv:N`. A handover
  can bring one URL in; nothing may take it out again.
- Run `flexviz history show N --url` only when a human explicitly asks for
  the link.
- Never read `.flexviz/history.jsonl` or `findings*.html` directly: both hold
  full share URLs. Use `flexviz history list` and `show N` instead. If
  `flexviz` is not on PATH, use the venv form from step 1, not the raw files.
- Poll `flexvizState({compact: true})`, never the full accessor. The full one
  repeats every figure and trace, which you already know.
- Do not screenshot the dashboard to "see" it. The state is exact, pixels are
  not, and a full-page screenshot costs more than the state it replaces.

Other rules:

- Never collect the full dataset into your context. The data stays in the
  lazy engine.
- One serve process per port. Pick an uncommon free one (8077 for example).
  Your own earlier sessions are the likeliest occupant of a busy port.
- The history file belongs to one working directory and grows across sessions.
  Numbers never restart, and an old entry still opens at `/h/N` after a server
  restart, if the same file is served under the same source name. Two agents in
  one directory can take the same number, so give each session its own.

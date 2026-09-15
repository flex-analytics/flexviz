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

FlexViz serves interactive cross-filter dashboards from lazy Polars queries.
The full dataset stays server-side. The browser receives bounded aggregates and
small samples of real values. Parquet sources larger than RAM stream through
Polars, so peak memory stays flat as rows grow (box plots are the exception).

Every dashboard view is a URL that carries the complete spec. That URL is about
4 KB, near 1.3k tokens, and a browser tool echoes the page URL in every
snapshot. So you never hold one. You record each URL under a number with
`flexviz history`, open it at `/h/N`, and refer to it as `fv:N`.

Privacy: rows stay in the lazy engine unless you collect them. The schema, your
samples, and the ranges the human selects do enter your context. A share URL
embeds column names and filters, so add `.flexviz/` to `.gitignore`.

## The loop

### 1. Inspect the schema

```bash
flexviz schema data.parquet
```

If `flexviz` is not on PATH, look for a project venv. Run `.venv/bin/flexviz`
or `uv run flexviz`, and keep that form in every command that follows. If the
package is absent, ask the human before you install it.

The output names the file, the source, and every column with its dtype. Pick an
x column (usually time) and the columns worth plotting. A `.head(5).collect()`
peek in Polars is fine. Never collect the full frame.

### 2. Serve the file (background process)

```bash
flexviz serve data.parquet --cache --port 8077
```

Run the server from the project directory, and run every Python and CLI command
below from that same directory. The history file (`.flexviz/history.jsonl`) and
the `/h/N` route each resolve it against their own working directory.

- Each file becomes a source named by its stem (`data.parquet` -> `"data"`).
- `--cache` enables cross-filter cubes and live brushing. Use it whenever
  the file will not change while serving.
- Wait for readiness: poll until the answer names YOUR source, for example
  `until curl -s http://127.0.0.1:8077/sources | grep -q '"data"'; do sleep 1; done`.
  A bare "it answered" check is not enough. Another server can already own
  the port and answer with its own source list. On a busy port, `flexviz
  serve` exits with `cannot bind ...`. Read the serve log, then retry.
- Keep the process running while the human explores. Tell them the port and
  the PID, and stop the server when the session ends. A one-shot run
  otherwise leaves an orphan server that holds the port.
- To add files later, restart with the FULL file list (old + new), same port.
  Entries stay openable only while a server at the same address serves the
  same source names.
- Serve on loopback. Another interface exposes unauthenticated endpoints, so
  only do it if the human explicitly accepts that.

`flexviz serve` scans the file as stored and cannot cast a column, so a
timestamp held as `String` gives a string x axis and nothing warns you. If the
dtypes from step 1 need a cast, register the source yourself and build step 3
on the same cast LazyFrame:

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

- This script only builds a spec. It is cheap, lazy, and exits at once. The
  serve process answers all interactions.
- `source_name` must match the served stem. `cache=True` must match `--cache`.
  The number `history.add` returns is what you and the human exchange.
- No categorical column to `group_by`? Split related metrics across figures
  instead (one figure per metric family), and consider `add_corr_heatmap` or
  `add_histogram2d` to relate the numeric columns.

### 4. Open the entry

Open `http://127.0.0.1:8077/h/N` for the number step 3 printed. The page
address stays that short, so later browser snapshots never echo a 4 KB URL.

If you have browser tooling (Playwright MCP, a Chrome extension), that tab lives
in YOUR browser profile. It is not the human's window, and the human never sits
in front of it. Use it for your own checks and readbacks, and give the human the
same `/h/N` address for their own browser. Both tabs show the same dashboard,
each with its own state.

Tell the human: drag on one chart to cross-filter the others, zoom to
re-aggregate at higher detail, double-click to reset.

### 5. Read the state back

```js
window.flexvizState({compact: true})   // via your browser evaluate tool
```

It returns `{version, state, client_state, revision}`: the brushed ranges in
`state.selections`, the zoom in `state.viewport`, and a `revision` that goes up
whenever the state differs from your previous read. Poll it and compare
`revision` to see whether the human moved. Never read the full
`flexvizState()`. It repeats every figure and trace, which you already know.

Without browser tooling, the human clicks **Share** in the toolbar (it copies
the current-state URL) and pastes it to you. Then:

```bash
flexviz history add "<url>" --actor human --note "what they were looking at"
flexviz history show N --state
```

This is the only step where a URL enters your context, and it enters once. Work
from the number afterwards. The address bar does NOT track interactions; only
the Share button captures them.

### 6. Record the human's state before you act

The page cannot write the history file, so you record the state you read. Take
the entry the human opened, replace its state with the compact read, and add
the result under `actor="human"`:

```python
from flexviz import history
from flexviz.spec import (ClientState, InteractionState, decode_spec,
                          encode_spec, encoded_spec_from_url)

compact = {...}      # paste the state and client_state you just read
spec = decode_spec(encoded_spec_from_url(history.entry(3)["url"]))  # entry they opened
spec.state = InteractionState.model_validate(compact["state"])
spec.client_state = ClientState.model_validate(compact["client_state"])
url = f"http://127.0.0.1:8077/view?spec={encode_spec(spec)}"
print(history.add(url, note="human brushed 09:00-11:00 on sensor 12", actor="human"))
```

This reuses the recorded figures, so you never rebuild the builder and the URL
never reaches your context. Do it before you change anything: this entry is
what the report cites later.

### 7. Change the dashboard

A state change (viewport, selections, hover mode, axis locks) goes through the
write half, in the tab you control:

```js
await window.flexvizApply({state: {selections: [...]}})   // via browser evaluate
```

Top-level keys replace. `state` and `client_state` merge one level deep, so a
partial `state` keeps its sibling keys (`viewport`, `group_domains`,
`cross_filter_mode`). It re-renders and resolves with the compact state.

A structure change (add or remove a figure, change traces, change layout) needs
a new spec, because panels are built server-side. Rebuild in Python as in step
3, `history.add(...)`, then navigate to the new `/h/N`. Give your own entries a
note, so the history reads as a sequence of who did what.

### 8. Report the findings

Write plain markdown. A line that holds nothing but `fv:5` becomes the live
dashboard of entry 5. Put the prose around such lines, then run:

```bash
flexviz report findings.md -o findings.html --md findings.expanded.md
```

`findings.html` embeds each entry as a real, zoomable dashboard, for as long as
the server runs. `findings.expanded.md` holds bare links instead, for GitHub or
chat. Filter the LazyFrame to the brushed range to put numbers next to each
figure.

## API cheat-sheet

Defaults shown. Full reference: https://docs.flexviz.tech. The `#` notes mark
the arguments whose meaning you cannot read off the signature: those are where a
call runs cleanly and charts the wrong thing. For anything not here, the
docstrings in `flexviz/figure.py` are the source of truth.

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
                    z=None, histfunc=None)   # z and histfunc are a pair: pass
                                             #   both or neither; either alone
                                             #   raises. No z counts rows.
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
Panel size is dashboard layout, not figure layout: `cols=1` gives a full-width
panel, and `GridItem.h` sets the height. Every builder returns the `Figure`, so
calls chain. `group_by="col"` splits a trace into one child per group value
with stable colors.

## Rules

Token discipline:

- Never print, paste, or restate a share URL. Refer to entries as `fv:N`.
- Run `flexviz history show N` (no `--state`) only when a human explicitly
  asks for the link.
- Poll `flexvizState({compact: true})`, never the full accessor.
- Do not screenshot the dashboard to "see" it. The state is exact, pixels are
  not, and a full-page screenshot costs more than the state it replaces.

Other rules:

- Never collect the full dataset into your context. The data stays in the
  lazy engine.
- One serve process per port. Pick an uncommon port (8077 for example) and make
  sure it is free. Your own earlier sessions are the likeliest occupant.
- The history file belongs to one working directory and grows across sessions.
  Numbers never restart, and old entries stay openable at `/h/N` after a server
  restart, as long as the same file is served under the same source name.
- Two agents in the same directory at the same time can take the same number.
  Use one working directory per agent session.

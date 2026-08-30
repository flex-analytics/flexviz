---
name: flexviz-explore
description: >
  Serve a LOCAL Parquet/CSV file as a live, interactive FlexViz dashboard the
  human explores in their browser, while you read back their zooms and
  selections. Use when the user wants to explore or visualize a large local
  dataset (too big to plot inline), asks for an interactive/cross-filter
  dashboard, or hands you a flexviz /view URL. Prerequisites: the data file is
  on this machine, flexviz is installed in the project environment, and the
  human's browser can reach this machine. Do NOT use for small datasets that
  fit an ordinary inline plot, or in cloud/remote sessions with no way for the
  human to reach the served port.
---

# Explore large data with FlexViz

FlexViz serves interactive cross-filter dashboards from lazy Polars queries.
The full dataset stays server-side; the browser receives bounded aggregates
and small representative samples (min/max line points are real data values).
Interaction speed is benchmarked for the built-in traces (see 
flexviz.tech/benchmarks.html); larger-than-RAM Parquet streaming is work in
progress. Every dashboard view is a URL carrying the complete spec — opening
one needs a running server with the same source registered. 
The loop: you mint a URL, the human explores, you read back exactly what they
zoomed and selected.

## Privacy: know what leaves the data

- The full dataset stays in the server's lazy query engine. The browser
  receives bounded aggregates plus small representative samples of real
  values (downsampled line points, bin edges). Rows enter YOUR context only
  if you sample them yourself, so don't collect the frame.
- `flexvizState()` exposes the spec and interaction state, not rendered
  trace payloads.
- Schema, any samples you take, and the human's selected ranges or category
  values DO enter your context. That is usually fine; be deliberate.
- A share URL embeds the complete spec: column names, filters, selections.
  Treat it as sensitive as the filters it contains, and warn the human
  before a URL travels beyond the two of you.

## The loop

### 1. Inspect the schema

```bash
flexviz schema data.parquet
```

If `flexviz` is not on PATH, look for a project venv. Run `.venv/bin/flexviz`
or `uv run flexviz`, and keep that form in every command that follows.

Stable JSON: file, source name, columns with dtypes. Pick an x column
(usually time) and the numeric/categorical columns worth plotting. A
`.head(5).collect()` peek in Polars is fine; never collect the full frame.

### 2. Serve the file (background process)

```bash
flexviz serve data.parquet --cache --port 8077
```

- Each file becomes a source named by its stem (`data.parquet` -> `"data"`).
- `--cache` enables cross-filter cubes and live brushing. Use it whenever
  the file will not change while serving.
- Wait for readiness: poll until the answer names YOUR source, for example
  `until curl -s http://127.0.0.1:8077/sources | grep -q '"data"'; do sleep 1; done`.
  A bare "it answered" check is not enough. Another server can already own
  the port and answer with its own source list.
- If the port is busy, `flexviz serve` exits with `cannot bind ...`. Read the
  serve log, then retry on a free port.
- Keep the process running while the human explores. Tell them the port and
  the PID. Stop the server when the session is done. A one-shot run
  otherwise leaves an orphan server that holds the port.
- To add files later, restart with the FULL file list (old + new), same
  port. URLs stay valid only while a server at the same address serves the
  same source names.
- Serve on loopback. Binding another interface exposes unauthenticated
  endpoints; only do it if the human explicitly accepts that.

`flexviz serve` scans the file as stored and cannot cast a column. Read the
dtypes from step 1. A timestamp held as `String` gives a string x axis, not a
time axis, and nothing warns you. Register the source yourself instead:

```python
import polars as pl, uvicorn
from flexviz.server import app, register_source

lf = pl.scan_parquet("data.parquet").with_columns(
    pl.col("timestamp").str.to_datetime()
)
register_source("data", lf, cache=True)   # call this before uvicorn.run
uvicorn.run(app, host="127.0.0.1", port=8077)
```

Then build step 3 on that same cast LazyFrame. If the spec and the registered
source disagree about dtypes, the figures are wrong.

### 3. Build the dashboard and mint the URL

```python
import polars as pl
from flexviz import Dashboard

dash = Dashboard(pl.scan_parquet("data.parquet"), cache=True)
dash.add_figure().add_line(x="timestamp", y="value", group_by="sensor_id")
dash.add_figure().add_histogram(x="value", bins=50)
print(dash.share_url(server_url="http://127.0.0.1:8077", source_name="data"))
```

- This script only builds a spec; it is cheap, lazy, and exits immediately.
  The serve process answers all interactions.
- `source_name` must match the served stem; `cache=True` must match
  `--cache`.
- Give the human the complete URL in your reply. Never truncate it and
  never write "see above". The URL is the only thing they can click.
- No categorical column to `group_by`? Split related metrics across
  figures instead (one figure per metric family), and consider
  `add_corr_heatmap` or `add_histogram2d` to surface relationships
  between the numeric columns.

API cheat-sheet (defaults shown; full reference: https://docs.flexviz.tech).
The `#` notes mark the arguments whose meaning you cannot read off the
signature. Those are where a call runs cleanly and charts the wrong thing, so
read them before you write the call. For a detail that is not here, the
docstrings in `flexviz/figure.py` are the source of truth:

```python
Dashboard(data, cache=False)     # data: pl.LazyFrame/DataFrame, pandas, pyarrow
                                 # cache=True enables cross-filter cubes (live brushing)
dash.add_figure(title=...)       # -> Figure; chainable builders below
dash.share_url(server_url, source_name, rows=None, cols=None, cache=None)

fig.add_line(x, y, name=None, color=None, n_points=1000,
             downsample="minmax",          # or "fpcs" | "nth"
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
                                             #   both or neither. Either one on
                                             #   its own raises. No z counts
                                             #   rows per cell.
fig.add_corr_heatmap(columns=None, method="pearson", triangular=False)
fig.add_geo_histogram2d(lat, lon, lat_bins=64, lon_bins=64,
                        z=None, histfunc=None,  # same pair rule as above
                        bin_boundaries="data")
# bin_boundaries picks what the bin grid is anchored to, and the default is
#   often wrong. "data" spans the full data extent, so a handful of far-out
#   points push every real point into one cell. "viewport" anchors the grid to
#   the map bounds instead. Read the lat/lon min and max first, and pass
#   "viewport" when the extent is much wider than the region you care about.
fig.add_geo_line(lat, lon, n_points=1000)

fig.title(text); fig.xlabel(text); fig.ylabel(text); fig.legend(show=True)
```

Every builder returns the `Figure`, so calls chain. `group_by="col"`
splits a trace into one child per group value with stable colors.

### 4. Watch along (preferred when you control a browser)

If you have browser tooling (Playwright MCP, a Chrome extension), open the
minted URL there and let the human explore in that window. Then read their
current state at any moment:

```js
window.flexvizState()   // via your browser evaluate tool
```

It returns the complete current spec: `state.selections` (brushed ranges or
categories) and `state.viewport` (where they zoomed). Poll it when you need
to know what they are looking at; that is the watching-together loop. Tell
the human: drag on one chart to cross-filter the others, zoom to
re-aggregate at higher detail, double-click to reset.

### 5. Fallback: Share button + decode

Without browser tooling, the human clicks **Share** in the dashboard
toolbar (it copies the current-state URL to their clipboard) and pastes it
to you. Then:

```bash
flexviz decode "<url>"
```

The address bar does NOT track their interactions; only the Share button
captures the current state.

### 6. Continue the analysis

Filter the LazyFrame to the brushed range or selected categories and
compute statistics on exactly what the human pointed at.

## Rules

- Never collect the full dataset into your context; the point is that data
  stays in the lazy engine.
- Read state through `flexvizState()` or `flexviz decode`; do not
  screenshot the dashboard to "see" it. The spec is exact, pixels are not.
- One serve process per port. Pick an uncommon port (for example 8077).
  Make sure that the port is free: your own earlier flexviz sessions are
  the most likely occupant.

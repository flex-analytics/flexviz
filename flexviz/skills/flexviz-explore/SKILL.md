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
- Wait for readiness: poll `curl -s http://127.0.0.1:8077/sources` until it
  answers with the source list.
- Keep the process running while the human explores; stop it when the
  session is done.
- To add files later, restart with the FULL file list (old + new), same
  port. URLs stay valid only while a server at the same address serves the
  same source names.
- Serve on loopback. Binding another interface exposes unauthenticated
  endpoints; only do it if the human explicitly accepts that.

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
- No categorical column to `group_by`? Split related metrics across
  figures instead (one figure per metric family), and consider
  `add_corr_heatmap` or `add_histogram2d` to surface relationships
  between the numeric columns.

API cheat-sheet — write calls from this, do not read the source
(defaults shown; full reference: https://docs.flexviz.tech):

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
fig.add_bar(labels, values=None,
            agg="sum",                     # "mean"|"median"|"min"|"max"|"n_unique"
            orientation="v", bar_mode="group", group_by=None)
fig.add_boxplot(x=None, y=None, name=None, group_by=None)
fig.add_pie(labels, values=None, agg="sum", hole=0.0)
fig.add_treemap(path=[...], values=None, agg="sum")   # path: hierarchy columns
fig.add_histogram2d(x, y, x_bins=20, y_bins=20, z=None, histfunc=None)
fig.add_corr_heatmap(columns=None, method="pearson", triangular=False)
fig.add_geo_histogram2d(lat, lon, lat_bins=64, lon_bins=64, z=None)
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
- One serve process per port. Pick an uncommon port (e.g. 8077) to avoid
  colliding with the user's own servers.

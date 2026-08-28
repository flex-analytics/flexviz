---
name: flexviz-explore
description: >
  Give the human a live, interactive dashboard for a large tabular dataset
  (Parquet/CSV, millions to billions of rows) instead of a static plot, and
  read back what they brushed or zoomed. Use when the user wants to explore
  or visualize a dataset too big to plot inline, asks for an interactive
  dashboard, or hands back a flexviz /view URL to continue analysis from.
---

# Explore large data with FlexViz

FlexViz serves interactive cross-filter dashboards from lazy Polars queries.
Only small aggregates reach the browser, so 100M+ rows stay instant. Every
dashboard view is a self-contained URL: you mint one for the human, and the
URL they hand back encodes exactly what they zoomed and selected. Data never
enters your context.

## The loop

### 1. Inspect the schema yourself

You have Polars. Do not guess columns:

```bash
python -c "import polars as pl; print(pl.scan_parquet('data.parquet').collect_schema())"
```

Pick an x column (usually time) and the numeric/categorical columns worth
plotting. `.head(5).collect()` is fine for a peek; never collect the full
frame.

### 2. Serve the file (background process)

```bash
flexviz serve data.parquet --cache --port 8077
```

- Each file becomes a source named by its stem (`data.parquet` -> `"data"`).
- `--cache` enables cross-filter cubes and live brushing. Use it whenever the
  file will not change while serving.
- Keep this process running while the human explores. To add more files,
  restart it with the extra files listed: the server is stateless, so all
  URLs minted earlier keep working.

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
  The serve process from step 2 answers all interactions.
- `source_name` must match the served stem. `cache=True` must match
  `--cache`.
- Trace builders: `add_line`, `add_histogram`, `add_bar`, `add_boxplot`,
  `add_pie`, `add_treemap`, `add_histogram2d`, `add_corr_heatmap`,
  `add_geo_histogram2d`, `add_geo_line`. Grouped traces take
  `group_by="col"`. Full API: https://docs.flexviz.tech

### 4. Hand the human the URL

Tell them: drag on one chart to cross-filter the others, zoom to
re-aggregate at higher detail, double-click to reset. The address bar always
encodes the current view, so they can send you (or anyone) the URL at any
moment.

### 5. Read back what they did

When the human pastes a URL:

```bash
flexviz decode "<url>"
```

This prints the full JSON spec, including `state.selections` (the ranges or
categories they brushed) and per-figure viewports (where they zoomed).
Continue the analysis from that state, e.g. filter the LazyFrame to the
brushed range and compute statistics on exactly what they pointed at.

## Rules

- Never load the dataset into your own context or collect it eagerly; the
  whole point is that the data stays in the server's lazy query engine.
- Do not screenshot the dashboard to "see" it; decode the URL instead. The
  spec is exact, pixels are not.
- One serve process per port. Pick an uncommon port (e.g. 8077) to avoid
  colliding with the user's own servers.

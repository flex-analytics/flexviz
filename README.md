<p align="center">
  <img src="docs/logo.svg" width="110" alt="FlexViz logo">
</p>

<h1 align="center">FlexViz</h1>

<p align="center">
  <b>Interactive cross-filter dashboards on 100M+ rows, in pure Python.</b><br>
  Polars-native &nbsp;·&nbsp; stateless server &nbsp;·&nbsp; Rust-accelerated
</p>

---

FlexViz is a visualization library for exploring datasets that are far too big
for conventional Python dashboarding tools. Charts stay interactive (zoom,
brush, cross-filter) at 100M+ rows, because every interaction is answered by
lazy Polars aggregations and Rust kernels instead of by shipping raw data to
the browser.

## Why it's fast

- **Polars-native.** Data stays a lazy `LazyFrame` until the last moment;
  in-memory frames and parquet-backed sources both work, and larger-than-RAM
  sources stream through Polars' streaming engine.
- **Rust kernels.** Min/max line downsampling and fixed-bin histogram/heatmap
  binning run as parallel Polars expression plugins
  (`flexviz_polars`), at memory-bandwidth speed.
- **Cube live-brushing.** Dragging a brush is served client-side from a small
  pre-aggregated cube — zero server round-trips during the drag.
- **Stateless server.** The client owns all interaction state and every request
  carries the complete dashboard spec. No sessions, no server affinity — and
  shareable dashboard URLs fall out for free.

## Features

- 10 trace types: line, histogram, box, bar, pie, treemap, 2D histogram,
  correlation heatmap, geo 2D histogram, geo line.
- Native cross-filtering between figures (update or overlay mode), grouped traces,
  linked hover, drag-and-drop dashboard layout.
- Accepts Polars DataFrames/LazyFrames, pandas DataFrames, and PyArrow tables.
- Renders with Plotly.js; the renderer sits behind a clean adapter boundary.
- Embeds into an existing FastAPI app via `mount_into()`.

## Quickstart

```python
import polars as pl
from flexviz import Dashboard

lf = pl.scan_parquet("readings.parquet")  # 100M rows — stays lazy

dash = Dashboard(lf)
dash.add_figure().add_line(x="timestamp", y="value")
dash.add_figure().add_histogram(x="value", bins=50)
dash.show()  # brush one chart to cross-filter the other
```

## Install

FlexViz is not on PyPI yet (the `flexviz` name currently holds a placeholder;
the first release lands soon). 

From source:

```bash
git clone https://github.com/flex-analytics/flexviz
cd flexviz
uv sync              # installs deps and builds the Rust plugin
make test
```

The Rust plugin builds automatically; the toolchain is pinned in
`rust-toolchain.toml`.

## Benchmarks — coming soon

A cleaned-up benchmark suite is on its way: time-to-first-render and
interaction latency against other large-data visualization tools at up to
hundreds of millions of rows, plus a memory-bounded out-of-core suite. The
numbers will live here and in the accompanying launch write-up.

## Status

Pre-1.0. The Python API and the spec wire format may change between minor
versions — see the compatibility policy in [AGENTS.md](AGENTS.md).
[Architecture.md](Architecture.md) is the design source of truth.

## License

[TO DECIDE](LICENSE) © 2026 [Flex Analytics BV](https://flexviz.tech)

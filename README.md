<h1 align="center">
  <a href="https://flexviz.tech/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="docs/assets/flexviz-wordmark-dark.png">
      <img src="docs/assets/flexviz-wordmark-light.png" width="212" alt="FlexViz">
    </picture>
  </a>
</h1>

<p align="center">
  <b>Interactive visualization at scale.</b><br>
  Python &nbsp;·&nbsp; Polars-native &nbsp;·&nbsp; stateless server &nbsp;·&nbsp; Rust-accelerated
</p>

> [!WARNING]
> FlexViz is pre-1.0 and under active development. APIs, defaults, and the spec
> format may change between minor releases, and rough edges remain. Bug reports
> are very welcome.

---

FlexViz is a visualization library for exploring datasets that are far too big
for conventional Python dashboarding tools. Charts stay interactive (zoom,
brush, cross-filter) at 100M+ rows, because every interaction is answered by
lazy Polars aggregations and Rust kernels instead of by shipping raw data to
the browser.

<p align="center">
  <a href="https://flexviz.tech/demo.html">
    <img src="docs/crossfilter.gif" width="850" alt="Cross-filter demo: brushing a range on a 100M-point line chart re-aggregates the linked histogram">
  </a>
</p>

<p align="center">
  Brush one chart and every linked chart re-aggregates against the filtered
  set. Try it yourself on 100M rows in the
  <a href="https://flexviz.tech/demo.html">live demo</a>.
</p>

## Why it's fast

- **Polars-native.** Data stays a lazy `LazyFrame` until the last moment;
  in-memory frames and parquet-backed sources both work, and larger-than-RAM
  sources stream through Polars' streaming engine.
- **Rust kernels.** Min/max line downsampling and fixed-bin histogram/heatmap
  binning run as parallel Polars expression plugins
  (`flexviz_polars`), at memory-bandwidth speed.
- **Cube live-brushing.** Dragging a brush is served client-side from a small
  pre-aggregated cube: zero server round-trips during the drag.
- **Stateless server.** The client owns all interaction state and every request
  carries the complete dashboard spec. No sessions, no server affinity, and
  shareable dashboard URLs fall out for free.

## Features

- **10 trace types.** Line, histogram, box, bar, pie, treemap, 2D histogram,
  correlation heatmap, geo 2D histogram, geo line.
- **Native cross-filtering.** Brush one figure to filter the others (update or
  overlay mode), with grouped traces and linked hover.
- **Shareable URLs.** Every view — viewport, selections, cross-filter mode, and
  layout — encodes into a single URL. Send the link and a teammate opens the
  exact live view; the server stores nothing.
- **Drag-and-drop dashboard grid.** Rearrange and resize panels in the browser
  and lock the layout when it's done; the arrangement also travels with the URL.
- **Bring any DataFrame.** Polars DataFrames/LazyFrames, pandas DataFrames, and
  PyArrow tables.
- **Plotly.js rendering.** The renderer sits behind a clean adapter boundary.
- **Embeddable.** Mounts into an existing FastAPI app via `mount_into()`.

## Quickstart

```python
import polars as pl
from flexviz import Dashboard

lf = pl.scan_parquet("readings.parquet")  # 100M rows, stays lazy

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

## Benchmarks

Time to render 1 billion points per chart: 5 traces × 200M rows from an
in-memory frame, clocked browser-side from the request to painted pixels
(median of 5 warm repeats). Each engine renders its own native chart: the line
runs against Datashader and Mosaic, the histogram against Vaex and Mosaic,
because neither of those tools has the other chart.

<p align="center">
  <a href="https://flexviz.tech/benchmarks.html">
    <img src="docs/benchmark-ttfr.png" width="850" alt="Time to first render at 200M rows and 5 traces. Line: flexviz 344 ms, datashader 8,670 ms, mosaic 10,033 ms. Histogram: flexviz 617 ms, vaex 1,304 ms, mosaic 3,009 ms.">
  </a>
</p>

Peak backend memory stays at ~25 MB from 1M to 200M rows: FlexViz aggregates
the caller's frame zero-copy. Interactive charts for the full matrix (1M–200M
rows, 1/2/5 traces, in-memory and Parquet-backed) are at
[flexviz.tech/benchmarks](https://flexviz.tech/benchmarks.html); the harness,
correctness gates, per-trial results, and caveats live in
[flexviz-benchmarks](https://github.com/flex-analytics/flexviz-benchmarks).

## Status

**Pre-1.0**.
The Python API and the spec wire format may change between minor
versions; see the compatibility policy in [AGENTS.md](AGENTS.md).
[Architecture.md](Architecture.md) is the design source of truth.

## License

[Apache-2.0](LICENSE) © 2026 [Flex Analytics BV](https://flexviz.tech)

<h1 align="center">
  <a href="https://flexviz.tech/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/flex-analytics/flexviz/main/docs/assets/flexviz-wordmark-dark.png">
      <img src="https://raw.githubusercontent.com/flex-analytics/flexviz/main/docs/assets/flexviz-wordmark-light.png" width="212" alt="FlexViz">
    </picture>
  </a>
</h1>

<p align="center">
  <b>Interactive visualization at scale.</b><br>
  Python &nbsp;·&nbsp; Polars-native &nbsp;·&nbsp; stateless server &nbsp;·&nbsp; Rust-accelerated &nbsp;·&nbsp; agent-ready
</p>

<p align="center">
  <a href="https://github.com/flex-analytics/flexviz/actions/workflows/ci.yml"><img src="https://github.com/flex-analytics/flexviz/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI status"></a>
  <a href="https://pypi.org/project/flexviz/"><img src="https://img.shields.io/pypi/v/flexviz" alt="PyPI version"></a>
  <a href="https://pypi.org/project/flexviz/"><img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue" alt="Supported Python versions"></a>
  <!-- <a href="https://pypi.org/project/flexviz/"><img src="https://img.shields.io/pypi/dm/flexviz" alt="PyPI downloads per month"></a> -->
  <a href="https://github.com/flex-analytics/flexviz/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="Apache-2.0 license"></a>
</p>

<p align="center">
  <a href="https://docs.flexviz.tech">Documentation</a> ·
  <a href="https://flexviz.tech/demo.html">Live demo</a> ·
  <a href="https://flexviz.tech/benchmarks.html">Benchmarks</a> ·
  <a href="https://docs.flexviz.tech/guides/ai-agents/">Agents</a>
</p>

---

FlexViz is a visualization library for exploring datasets that are far too big
for conventional Python dashboarding tools.  
Charts stay interactive (zoom, pan, cross-filter) at 100M+ rows because every
interaction is answered by lazy [Polars](https://github.com/pola-rs/polars)
aggregations and Rust kernels instead of by shipping raw data to the browser.

<p align="center">
  <a href="https://flexviz.tech/demo.html">
    <img src="https://raw.githubusercontent.com/flex-analytics/flexviz/main/docs/crossfilter.gif" width="850" alt="Cross-filter demo: brushing a range on a 100M-point line chart re-aggregates the linked histogram">
  </a>
</p>

<p align="center">
  Brush one chart and every linked chart re-aggregates against the filtered
  set. Try it yourself on 2 x 100M rows in the
  <a href="https://flexviz.tech/demo.html">live demo</a>.
</p>

> [!WARNING]
> FlexViz is pre-1.0 and under active development. APIs, defaults, and the spec
> format may change between minor releases, and rough edges remain. Bug reports
> are very welcome.

## Install

```bash
pip install flexviz
```

The Rust kernels arrive as a prebuilt wheel (`flexviz-polars`) on Linux
(x86_64, aarch64), macOS (Intel and Apple silicon), and Windows (x64). Any
other platform builds them from source and needs a Rust toolchain.

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

Guides and the full API reference live at
[docs.flexviz.tech](https://docs.flexviz.tech).

## Agents

Give a coding agent a data file and it hands back a live dashboard, not a
static plot. The wheel ships an [Agent Skill](https://agentskills.io) that
teaches the workflow:

```bash
flexviz skill install   # writes SKILL.md into .agents/skills/ and .claude/skills/
```

Then ask your agent to explore `readings.parquet`. It reads the schema, serves
the file, and gives you a dashboard URL. You zoom and brush in the browser. The
agent reads your viewport and selections back through `window.flexvizState()`,
or from the URL that the **Share** button copies, and continues from that
state. The rows stay in the query engine. Only specs and URLs travel.

See the [agent guide](https://docs.flexviz.tech/guides/ai-agents/).

## Features

FlexViz is a library, not a (cloud) service. 
The dashboard server **runs where your data lives**, on your laptop or in your 
own infra, and rows never leave it.

- **10 trace types**: line, histogram, box, bar, pie, treemap, 2D histogram,
  correlation heatmap, geo 2D histogram, geo line.
- **Bring any DataFrame**: Polars DataFrames/LazyFrames, pandas DataFrames,
  and PyArrow tables.
- **Interactivity**:
  - **Zoom re-aggregation**: zooming recomputes a figure for its viewport, so
    a line re-downsamples and a histogram re-bins; detail appears as you dive.
  - **Native cross-filtering**: brush or click one figure to filter the
    others, either replacing their view (update mode) or drawing the filtered
    aggregate on top of the totals (overlay mode).
  - **Linked hover**: hovering one figure highlights the matching position in
    every figure that shares its columns, fully client-side.
- **Grouped traces**: `group_by` splits a trace into per-category series with
  stable colors, computed in a single grouped Polars query.
- **Shareable URLs**: every view (viewport, selections, cross-filter mode, and
  layout) encodes into a single URL. Send the link and a teammate opens the
  exact live view; the server stores nothing.
- **Draggable dashboard grid**: rearrange and resize panels in the browser
  and lock the layout when it's done; the arrangement also travels with the URL.
- **Embeddable**: mounts into an existing 
  [FastAPI](https://github.com/fastapi/fastapi) app via `mount_into()`.


## Why it scales

- **Polars-native.** Data stays a lazy `LazyFrame` until the last moment;
  in-memory frames and parquet-backed sources both work, and larger-than-RAM
  sources stream through Polars' streaming engine (*WIP*).
- **Rust kernels.** Min/max line downsampling and fixed-bin histogram/heatmap
  binning run as parallel Polars expression plugins
  (`flexviz_polars`), at memory-bandwidth speed.
- **Aggregates over the wire.** The browser receives a few thousand points per
  trace, never the raw rows
- **Cube live-brushing.** Dragging a brush is served client-side from a small
  pre-aggregated cube: zero server round-trips during the drag.
- **Stateless server.** The client owns all interaction state and every request
  carries the complete dashboard spec. No sessions, no server affinity, and
  shareable dashboard URLs are a free feature.
- **Renderer-agnostic core.** Specs, traces, and the engine know nothing about
  the renderer; a thin adapter maps updates onto Plotly.js, the default
  renderer.

## Benchmarks

Time to render 1 billion points per chart: 5 traces × 200M rows from an
in-memory frame, clocked browser-side from the request to painted pixels
(median of 5 warm repeats). Each engine renders its own native chart: the line
runs against Datashader and Mosaic, the histogram against Vaex and Mosaic,
because neither of those tools has the other chart.

<p align="center">
  <a href="https://flexviz.tech/benchmarks.html">
    <img src="https://raw.githubusercontent.com/flex-analytics/flexviz/main/docs/benchmark-ttfr.png" width="850" alt="Time to first render at 200M rows and 5 traces: FlexViz against Datashader, Vaex, and Mosaic, fastest in both the line and histogram panels. Latest numbers at flexviz.tech/benchmarks.">
  </a>
</p>

Peak backend memory stays at ~25 MB from 1M to 200M rows: FlexViz aggregates
the caller's frame zero-copy.

Charts for the full matrix (1M–200M rows, 1/2/5 traces, in-memory
and Parquet-backed) are at
[flexviz.tech/benchmarks](https://flexviz.tech/benchmarks.html); the harness,
correctness gates, per-trial results, and caveats live in
[flexviz-benchmarks](https://github.com/flex-analytics/flexviz-benchmarks).

## Development

```bash
git clone https://github.com/flex-analytics/flexviz
cd flexviz
uv sync              # installs deps and builds the Rust plugin
make test
```

The Rust plugin builds automatically; the toolchain is pinned in
`rust-toolchain.toml`.

See the compatibility policy in the
[changelog](https://github.com/flex-analytics/flexviz/blob/main/CHANGELOG.md).  
[Architecture.md](https://github.com/flex-analytics/flexviz/blob/main/Architecture.md) 
is the design source of truth.

## License

[Apache-2.0](https://github.com/flex-analytics/flexviz/blob/main/LICENSE) © 2026 [Flex Analytics BV](https://flexviz.tech)

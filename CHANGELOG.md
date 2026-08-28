# Changelog

All notable changes to FlexViz are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Compatibility policy

FlexViz is pre-1.0. Until 1.0, **minor versions (0.x to 0.y) may break
anything**: the Python API, the spec wire format, shared-URL encoding, and
storage or cache formats. Breaking spec changes bump the spec version.

`decode_spec`, shared URLs, and imported specs are only guaranteed to
round-trip specs produced by the same minor version. There are no dual-read
shims or version-gated branches: when a field changes shape, its producers and
consumers change with it and the old path is deleted. Breaking changes are
documented in the release notes below instead.

`flexviz` and `flexviz-polars` are released together but versioned
independently. `flexviz` pins a compatible `flexviz-polars` range.

## [0.1.0b2] - 2026-08-28

### Added

- Agent interface. A coding agent can serve a dataset, mint a dashboard URL,
  and read back the viewport and selections a person leaves behind.
  - `flexviz` command line: `serve` registers files as named sources and runs
    the server, `schema` prints columns and dtypes as JSON, `decode` turns a
    `/view` URL back into its spec, and `skill install` copies the packaged
    agent skill into a project.
  - `Dashboard.share_url()` builds a `/view` URL from a spec without opening a
    browser.
  - `window.flexvizState()` returns a detached snapshot of the current spec,
    including viewport and selections.
  - The wheel ships the `flexviz-explore` Agent Skill. `flexviz skill install`
    writes it into `.agents/skills/` and `.claude/skills/`.
- Agent guide at docs.flexviz.tech, and a documented security model.

### Changed

- Line downsampling builds the out-of-core min/max envelope in one streaming
  collect instead of two passes.
- The polars floor is now 1.44.1.

### Fixed

- Histogram bounds are aggregated after the horizontal reduction, so grouped
  histograms bin against the correct range.
- Line bucket width divides on the column dtype instead of the range dtype,
  which keeps float x values at full envelope resolution.
- `flexviz skill install` does not overwrite an unrelated file, and CSV sources
  parse dates instead of reading timestamps as strings.

## [0.1.0b1] - 2026-08-25

### Added

- First public release of FlexViz: a renderer-agnostic, lazily evaluated,
  stateless visualization library built on Polars and FastAPI, for exploring
  datasets of 100M+ rows.
- Ten trace types: line, histogram, box, bar, pie, treemap, 2D histogram,
  correlation heatmap, geo 2D histogram, and geo line.
- Native cross-filtering in update or overlay mode, with grouped traces and
  linked hover.
- Client-side cube live-brushing, so dragging a brush costs no server
  round-trips.
- Shareable URLs that encode viewport, selections, cross-filter mode, and
  dashboard layout, with no server-side state.
- Drag-and-drop dashboard grid, with the arrangement carried in the URL.
- Plotly.js rendering behind an adapter boundary.
- `mount_into()` for embedding into an existing FastAPI application.
- `flexviz-polars`, the Rust Polars expression kernels behind the line
  downsampling and fixed-bin histogram and heatmap paths.

### Changed

- The default line downsampling path (`downsample="minmax"`) now runs as one
  fused `minmax_line` kernel call per trace instead of an `arg_min_max` index
  expression feeding two gathers. Polars does not common-subexpression-eliminate
  plugin expressions, so the two-gather form scanned every column twice; the
  fused call halves kernel work and makes frame times markedly steadier.
  Output is bit-identical (pinned by a differential test against the two-gather
  form, which remains available as `pl.Expr.flexviz.arg_min_max`).
- The `arg_min_max` window scan is parallel across windows on the plugin's
  kernel thread pool. This trades cross-trace overlap for per-scan speed:
  measured at 100M rows it is worth ~1.3-1.6x on a single trace and costs at
  most ~9% when 3-5 traces share a bandwidth-saturated host, fading by 20
  traces. `flexviz` requires a `flexviz-polars` build that ships `minmax_line`;
  the two are released together.

[0.1.0b2]: https://github.com/flex-analytics/flexviz/releases/tag/v0.1.0b2
[0.1.0b1]: https://github.com/flex-analytics/flexviz/releases/tag/v0.1.0b1

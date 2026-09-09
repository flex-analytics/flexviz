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

## [Unreleased]

## [0.1.0b3] - 2026-09-09

### Added

- `downsample="lttb"` on `add_line`. MinMaxLTTB thins the min-max pass down to
  `n_points` points, which reads smoother than an envelope on noisy data.
- Equal x-width buckets for every line except `"nth"`, grouped lines included.
- An x contract for those lines: x must be sorted, null-free and NaN-free on a
  resident frame, and `n_points` must be between 2 and 25000.
- Out-of-core histograms. 1-D, 2-D and geographic histograms run on a Parquet
  scan at memory that does not grow with the row count.
- Out-of-core `"nth"` lines. Ungrouped and grouped nth traces stream on a scan
  source via an ordered `collect_batches` fold.
- A Parquet footer probe. On a single-file local scan, column bounds come from
  the footer statistics instead of a column decode.
- Branded page title, favicon, and wordmark. The header links to the project
  website.
- User-scope skill install (`flexviz skill install --user`) and distribution
  through the Codex marketplace.

### Changed

- A zoomed histogram grid snaps to a fixed lattice, so bars keep their place
  while you pan. A zoomed axis can show one bin more than configured.
- Delta wire format: `hover_bounds` is replaced by an `x_edges` / `y_edges` /
  `lat_edges` / `lon_edges` triple per binned axis. The client derives every
  bin bound from the triple, and builds the geographic rectangles itself.
- A figure is drawn once on load instead of twice.
- Domain bounds are memoized on a static source: a resident frame, or a scan
  registered with `cache=True`. A `cache=False` scan resolves them again, so an
  uncached reset sees changed data on disk.
- Spec version 0.5.
- `pyarrow` is now a required dependency. The Parquet footer probe reads
  row-group statistics through it.

### Removed

- `bin_boundaries` on `add_geo_histogram2d`.
- `row_index_col` on `Figure` and `Dashboard`.
- The `arg_min_max`, `fpcs`, `minmax_line` and `fpcs_line` plugin kernels.
  The default line kernel is now a fused pairs envelope
  (`minmax_pairs_line`).
- `LFQueryBuilder.check_sorted`, replaced by `check_line_x`.
- `cache_schema` on `LFQueryBuilder`, replaced by `cache`.

### Fixed

- `register_source(cache=True)` silently ignored the `cache` flag.
- Re-registering the same source name raised an error instead of warning.
- 2-D histogram emitted wrong cells when a reduction produced NaN.

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

[0.1.0b3]: https://github.com/flex-analytics/flexviz/releases/tag/v0.1.0b3
[0.1.0b2]: https://github.com/flex-analytics/flexviz/releases/tag/v0.1.0b2
[0.1.0b1]: https://github.com/flex-analytics/flexviz/releases/tag/v0.1.0b1

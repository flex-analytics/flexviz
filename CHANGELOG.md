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

[Unreleased]: https://github.com/flex-analytics/flexviz/commits/main

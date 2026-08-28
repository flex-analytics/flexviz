# Architecture

**flexviz** — renderer-agnostic, scalable, linked visualizations for large datasets.

**Spec version:** `0.3` — predicate-based `SelectionState`.

## Core Properties

- **Lazy execution** — data lives in a Polars `LazyFrame`; only what the viewport requests is materialized
- **Stateless backend** — the server never stores interaction state; the client is authoritative
- **Renderer-agnostic** — adapters bridge the core to concrete JS charting libraries; zero Python renderer dependencies
- **Cross-filtering** — selections (range brushes or categorical clicks) from one figure filter aggregations in linked figures, as a backend Polars primitive
- **Cube live brushing** — Mosaic-style fixed-P pre-aggregations, built and cached once server-side, shipped as binary FVCube blobs, and sliced client-side per drag frame — zero roundtrips during (and, when fully cube-served, after) a brush gesture
- **Linked hover** — hover events from one figure are propagated to linked figures (fully client-side)
- **Low-code API** — `Figure` / `Dashboard` with `add_*` builder methods
- **Shareable views** — full state encoded as gzip + base64url in a single URL parameter

=> SPEC is key to all of this

- **Rust plugin** — `flexviz_polars` crate exposes `every_nth`, `arg_min_max`, `minmax_line`, `fixed_hist`, `fixed_hist2d`, `fixed_hist2d_reduce`, and `fpcs` / `_fpcs_line` kernels via `pl.Expr.flexviz` namespace; single-pass, no `len()` expression dependency, enabling N grouped sub-traces to parallelize on one scan. The hist and argmin/argmax kernels are rayon-parallel on a dedicated pool sized from `POLARS_MAX_THREADS`, with serial fallbacks that produce identical output
- **Grouped traces** — grouped parents stay logical-only, `LFQueryBuilder` batches shared grouped queries, and group→color mapping persists in client-owned spec state
- **Multi-column labels and group_by** — `BarPlot.labels`, `PiePlot.labels`, and any trace's `group_by` accept either a string or a list of strings.  Multi-column values are stored as JSON-encoded composite strings (e.g. `'["Europe","Germany"]'`) for legend keys, color-domain mapping, and child-uid stability.  Aggregation uses multi-column `group_by(...)` in Polars; cross-filter emission decomposes the composite back into per-column clauses so the wire format remains a list of `ClauseFilter` objects, never a JSON-encoded string.
- **Overlay mode** — adapters own cached unfiltered backgrounds per figure, `TraceDelta.layer` carries `bg` / `fg` on the wire, per-trace `overlay_style` controls which layers are emitted, and share/import/export preserve declarative state only
- **Linked hover** — fully client-side; a single on/off toggle (`hover_mode`), with the runtime auto-selecting the projection from the hovered source trace: lines and 1D histograms project a point onto shared axes (guides + bin-bands), while 2D cell sources (histogram2d, geo) project a cell. Column-to-figure-axis mapping computed at page load; persists in spec state through share/restore
- **Draggable dashboard grid** — optional GridStack layout (`layout.draggable=True`) with client-side drag/resize, persisted `grid_items`, and a toolbar lock button that toggles editability (`layout.grid_editable`) without backend round-trips
- **Pre-group global statistics** — aggregation specs can list columns in `global_stats_cols`; `LFQueryBuilder` injects per-column `with_columns(col.min(), col.max())` before filtering/grouping, exposing global bounds as constant columns that expressions can reference with `.first()`.  `Histogram` uses these stats for stable no-viewport bin edges, and the engine supplies same-figure histogram domain columns so sibling histogram traces can share one lazy min/max domain without an extra collect.

Implemented trace types: **LinePlot**, **Histogram**, **BoxPlot**, **BarPlot**, **PiePlot**, **TreeMap**, **Histogram2D**, **GeoHistogram2D**, **CorrHeatmap**, **GeoLine**
Implemented renderers: **PlotlyAdapter** (line, histogram, box, bar, pie, treemap, heatmap, choroplethmap, scattermap) · **EChartsAdapter** (line, histogram, bar, pie, heatmap)
=> EchartsAdapter is currently deprecated. Very distant future we will update this.

Python ≥ 3.10 · Polars · FastAPI · Uvicorn · Pydantic · flexviz_polars (Rust + maturin)

---

## Layer Overview

```
┌────────────────────────────────────────────────────────────────┐
│  USER LAYER                                                    │
│  fig = Figure(df)                                              │
│  fig.add_line(x="ts", y="val", n_points=1000, add_gaps=True)   │
│  fig.add_histogram(x="val", bins=20)                           │
│  fig.add_boxplot(y="val")                                      │
│  fig.add_bar(x="category", y="val", agg="sum")                 │
│  fig.add_bar(x="category", y="val", agg="sum", group_by="region") │
│  fig.add_pie(labels="category", values="val")                  │
│  fig.add_treemap(path=["region","category"], values="val")      │
│  fig.add_histogram2d(x="x", y="y", x_bins=20, y_bins=20)       │
│  fig.add_geo_histogram2d(lat="lat", lon="lon", lat_bins=64)     │
│  fig.add_geo_line(lat="lat", lon="lon", n_points=2000)          │
│  fig.add_corr_heatmap(columns=["a","b","c"])                    │
│  fig.title("My Title").xlabel("Time").ylabel("Value")           │
│  fig.show(renderer="plotly")                                   │
│                                                                │
│  dash = Dashboard(df)                                          │
│  fig1 = dash.add_figure()                                      │
│  fig1.add_line(x="ts", y="val")                                │
│  fig2 = dash.add_figure()                                      │
│  fig2.add_histogram(x="val")                                   │
│  dash.show(renderer="plotly", layout="rows")                   │
└─────────────────────┬──────────────────────────────────────────┘
                      │ VisualizationSpec / DashboardSpec
                      ▼
┌────────────────────────────────────────────────────────────────┐
│  SPEC / EVENT LAYER  (Pydantic, fully JSON-serializable)       │
│  VisualizationSpec · DashboardSpec · FigureSpec · TraceSpec    │
│  InteractionState · InteractionEvent · TraceDelta              │
│  encode_spec / decode_spec  (gzip + base64url)                 │
└─────────────────────┬──────────────────────────────────────────┘
                      │
                      ▼
┌────────────────────────────────────────────────────────────────┐
│  SERVER LAYER  (FastAPI, stateless)                            │
│  POST /update            — single-figure interaction           │
│  POST /dashboard/update  — dashboard interaction               │
│  POST /share             — encode spec → shareable URL         │
│  GET  /view              — render shared spec as HTML          │
│  GET  /sources           — health / introspection              │
│  _sources: name → LFQueryBuilder  (registered once at show())  │
└─────────────────────┬──────────────────────────────────────────┘
                      │
                      ▼
┌────────────────────────────────────────────────────────────────┐
│  ENGINE LAYER  (FlexEngine, stateless, per-request)            │
│  process(event, trace_infos, viewports_by_figure, mode)        │
│      → List[TraceDelta]                                        │
│    1. Derive active selections from event                      │
│    2. Build cross-filter Polars exprs from source figures      │
│    3. Collect regular + grouped specs per active trace         │
│    4. Execute only the required overlay layers for the event   │
│    5. Shape regular + grouped child deltas                     │
│  build_cubes(event=cube_request) → encoded FVCube blobs        │
└─────────────────────┬──────────────────────────────────────────┘
                      │
                      ▼
┌────────────────────────────────────────────────────────────────┐
│  DATA LAYER                                                    │
│  LFQueryBuilder — wraps pl.LazyFrame                           │
│    select(*regular_exprs) + fused group_by(...).agg(...).sort()│
│  AggregationSpec — one regular pl.Expr                         │
│  GroupedAggregationSpec — grouped query batch descriptor       │
└─────────────────────┬──────────────────────────────────────────┘
                      │ TraceDelta list (JSON, uid-keyed)
                      ▼
┌────────────────────────────────────────────────────────────────┐
│  ADAPTER LAYER  (renderer-specific)                            │
│  PlotlyAdapter  — HTML + Plotly.js 3 CDN (line/hist/box/bar/pie/heatmap/scattermap) │
│  EChartsAdapter — HTML + ECharts 5 CDN  (line/hist/bar/pie/heatmap) │
│  AbstractAdapter — shared toolbar, delivery helpers            │
└────────────────────────────────────────────────────────────────┘
```

---

## Spec and Event Model

All client ↔ server traffic uses Pydantic models from `spec.py` and `events.py`. The server never stores interaction state; every request carries the full spec and the triggering event.

```
VisualizationSpec
├── version: str
├── figure: FigureSpec
│   ├── uid: str                          ← stable; assigned by Figure.__init__
│   ├── source: str | None                ← named data source registered on server
│   ├── layout: Dict[str, Any]            ← renderer hints (title, height, ...)
│   └── traces: List[TraceSpec]
│       ├── uid: str
│       ├── trace_type: str               ← "line" | "histogram" | "box" | "bar" | "pie" | "treemap" | "histogram2d" | "geo_histogram2d" | "corr_heatmap" | "geo_line"
│       ├── axes: Tuple[str, ...] | None
│       ├── backend_data: Dict[str, str | list[str]]  ← {"x": "ts"} or {"labels": ["continent", "country"]}
│       ├── params: Dict[str, Any]        ← backend-only params
│       │                                   grouped parents include:
│       │                                   group_by, group_domain_key
│       ├── display: Dict[str, Any]       ← renderer hints; heatmap traces
│       │                                   materialize color_scale/color_range here
│       └── recompute_axes: Tuple[str, ...] | None
│           ← current specs must emit concrete anchors; () = frozen.
│             None is only an omitted-field default: the browser treats it as
│             empty/no fetch trigger, while server reconstruction may derive.
└── state: InteractionState
    ├── viewport: Dict[str, AxisRange | GeoViewportCoordinates | None]
    │   ├── cartesian keys store AxisRange
    │   └── `"{figure_uid}/coordinates"` stores raw map corner points
    │       as `[[lon, lat], ...]`
    ├── selections: List[SelectionState]
    │   ├── source_figure_uid: str | None      ← uid of figure where the selection was created
    │   └── predicates: list[SelectionPredicate]
    │       └── SelectionPredicate
    │           └── clauses: list[ClauseFilter]   ← AND-combined within a predicate
    │               └── ClauseFilter
    │                   ├── column: str          ← single column from the shared LazyFrame
    │                   ├── range: (lo, hi) | None    ← continuous filter (numeric, temporal)
    │                   ├── values: list[Any] | None  ← categorical is_in filter (exactly one of range/values)
    │                   └── closed: "both" | "left"   ← range endpoint inclusivity; "left" = [lo, hi),
    │                                                    emitted by cube-snapped commits (default "both")
    ├── group_domains: Dict[str, GroupDomainState]
    │   ├── key: "{source_or_figure_uid}::{group_by_col}"
    │   └── GroupDomainState
    │       ├── mapping: Dict[str, str]   ← group_value_key → CSS color
    │       └── next_color_index: int
    ├── cross_filter_mode: "update" | "overlay"
    └── hover_mode: "off" | "on"          ← single linked-hover toggle (in
                                            ClientState); persists through
                                            share/restore.

DashboardSpec
├── version: str
├── figures: List[FigureSpec]
├── state: InteractionState               ← shared across all figures
└── layout: LayoutSpec
    ├── gap: str
    ├── draggable: bool                   ← enables GridStack layout rendering
    ├── grid_editable: bool               ← initial lock/edit mode when draggable
    ├── grid_items: List[GridItem] | None ← client-updated positions/sizes
    └── toolbar: ToolbarConfig            ← controls which toolbar buttons are rendered
        └── show_reset / show_deselect / show_cfmode / show_hover
            / show_grid / show_share / show_export / show_import: bool (all True by default)

InteractionEvent
├── type: "init" | "viewport" | "selection" | "deselect" | "reset" | "cube_request"
├── axis_ranges: Dict[str, Any]
├── selections: List[SelectionState]
├── force_update: bool
└── figure_uid: str | None               ← scopes viewport/reset to one figure

"cube_request" events carry no active range and produce no deltas; they ride with
`request_cube: true` + `active_source: {figure_uid, column, trace_uid}` on the request and return
FVCube blobs (see "Cube Pre-Aggregation & Live Brushing"). `ClientState.live_brush`
("auto" | "off", client-only) gates the live-brush loop and persists through
share/restore like `hover_mode`.

TraceDelta  (server → client)
├── uid: str
├── updates: Dict[str, Any]              ← {"x": [...], "y": [...]}
├── layer: "bg" | "fg" | None
└── group_results: List[GroupedChildDelta] | None
    ├── None  → ordinary trace delta
    ├── []    → grouped parent now has zero visible children
    └── item  → {uid, parent_uid, group_value_key, updates}
```

Selection predicate mental model:

```text
ClauseFilter        = one column test
SelectionPredicate  = AND(clauses)
SelectionState      = OR(predicates)
Engine filter       = AND(selection_states from other figures)
```

`encode_spec` / `decode_spec` in `spec.py` provide gzip + base64url round-trip. `decode_spec` auto-detects `VisualizationSpec` vs `DashboardSpec`.

Overlay caches are runtime-only adapter state:

- immutable trace / series templates keyed by logical uid
- cached `base`, `bg`, and `fg` data payloads
- `hasBgByFigure` validity flags used when entering overlay mode

Those caches are reset on page load, import, shared-URL open, and toolbar reset. Shared URLs and exported specs preserve only declarative state: viewport, selections, `cross_filter_mode`, `group_domains`, `hover_mode`, and layout data (`gap`, `draggable`, `grid_editable`, `grid_items`).

---

## Figure and Dashboard

User-facing Python API in `figure.py` and `dashboard.py`.

### Figure

```
Figure
├── _uid: str                             ← stable; round-trips through every to_spec()
├── _backend_lf: LFQueryBuilder | None
├── _traces: List[FlexTrace]
├── _layout: Dict[str, Any]
│
├── add_line(x, y, ..., color_map=None, group_by=None)
├── add_histogram(x|y, ..., color_map=None, group_by=None)
├── add_boxplot(x|y, color=None, color_map=None, group_by=None)
├── add_bar(labels, values=None, agg="sum", ..., color_map=None, group_by=None)
├── add_pie(labels, values, agg, hole, color_map)
├── add_treemap(path, values=None, agg="sum", name, color_map)
├── add_histogram2d(x, y, x_bins, y_bins, name, color_scale, color_range, axes)
├── add_geo_histogram2d(lat, lon, lat_bins, lon_bins, …, bin_boundaries)
├── add_geo_line(lat, lon, n_points, name, color, marker_size)
├── add_corr_heatmap(columns, method, absolute, name, color_scale, color_range)
├── title(text) / xlabel(text) / ylabel(text) / legend(show)
├── update_layout(**kw)
├── to_spec(source=None)    → VisualizationSpec
├── save_spec(path)
├── load_spec(path)         [staticmethod]
├── from_spec(spec, backend_lf)  [classmethod]
└── show(renderer, port, notebook)
      registers source, starts FastAPI thread, renders via adapter
```

`to_spec()` auto-derives layout labels when the user has not set them explicitly:

- **`xlabel` / `ylabel`** — `_derive_axis_labels()` scans cartesian traces; if all agree on the same column for an axis, that column name becomes the label.
- **`title`** — `_derive_auto_title()` scans aggregated traces (`bar`, `pie`, `histogram`); if all produce the same candidate string (e.g. `"sum of revenue"`), it becomes the title.  Non-aggregated traces (line, box, hist2d, geo, corr_heatmap) are ignored.  Any of the three labels can be overridden via the explicit setter before or after trace construction.

### Dashboard

```
Dashboard
├── _backend_lf: LFQueryBuilder | None    ← shared across all figures
├── _figures: List[Figure]
│
├── add_figure(**layout_kw) → Figure      ← points new Figure at shared _backend_lf
├── to_spec(source=None)    → DashboardSpec
├── save_spec(path) / load_spec(path)
├── share_url(server_url, source_name, …) → str   ← /view URL for a running server
└── show(renderer, port, layout, notebook)
```

`Dashboard` currently assumes one shared data source for all figures. The server supports multiple sources per `DashboardSpec`, but the `Dashboard` builder does not expose this.

`share_url()` encodes a finalized spec into a `/view` URL for an
already-running server: no server start, no source registration, no browser.
`show()` and `share_url()` share one spec-finalization path
(`_finalized_spec`), so a shared URL and a local `show()` produce identical
specs.

---

## Agent interface

Coding agents drive FlexViz through the same stateless surface humans use.

- **CLI** (`flexviz/cli.py`): `serve` registers Parquet/CSV files as sources
  named by file stem and runs the server. It binds loopback by default;
  non-loopback binds print a warning because the endpoints are
  unauthenticated and CORS is open. `schema` prints columns and dtypes as
  JSON. `decode` turns a `/view` URL back into its spec. `skill install`
  copies the packaged skill (`flexviz/skills/flexviz-explore/SKILL.md`) into
  a project's `.agents/skills/` and `.claude/skills/`.
- **Readback contract**: the shared runtime exposes `window.flexvizState()`,
  which returns the live `DashboardSpec` (persistent serialized state only,
  no transient hover visuals). An agent with browser tooling opens a share
  URL, lets the human explore, and reads viewport and selections through
  this accessor at any time. Without browser tooling, the human clicks
  **Share** and the agent decodes the copied URL. The address bar does not
  track interactions.

Watch-along stays client-side by design: the server keeps no interaction
state, so "what is the human looking at" lives only in the browser tab.
Server-side snapshot mailboxes and agent-side listeners were considered and
rejected. A hosted watch broker and MCP Apps `updateModelContext`
publication are possible future phases; neither changes the aggregation
server's statelessness.

---

## Trace Layer

`FlexTrace` in `trace/base.py` is a renderer-agnostic ABC. Traces carry no data — they describe how to aggregate and filter. `trace/__init__.py` holds the registry and factory:

```python
_REGISTRY = {
    "line": LinePlot, "histogram": Histogram, "box": BoxPlot, "bar": BarPlot,
    "pie": PiePlot, "treemap": TreeMap, "histogram2d": Histogram2D,
    "geo_histogram2d": GeoHistogram2D, "corr_heatmap": CorrHeatmap, "geo_line": GeoLine,
}

def build_trace_from_spec(spec: TraceSpec) -> FlexTrace: ...
```

### FlexTrace (abstract base)

```
GroupedChildResult
├── child_uid: str
├── group_value_key: str
└── updates: Dict[str, pl.Series | list | Any]

TraceResult                              ← return type of _to_update / _to_grouped_update
├── updates: Dict[str, pl.Series | list | Any]
└── group_results: list[GroupedChildResult] | None
      None  → regular trace result
      []    → grouped parent with zero visible children

FlexTrace (ABC)
├── uid: str
├── trace_type: str
├── overlay_style: ClassVar[str]          ← "full" | "filtered_only"
│                                           suppresses duplicate bg aggregation
│                                           while a filtered fg is active
├── _backend_data: Dict[str, str | list[str]]
├── _display: Dict[str, Any]
├── _params: Dict[str, Any]
├── _recompute_axes: Tuple[str, ...]     ← data-binding axes (anchor space); the
│                                           engine gates per-trace recompute on these
├── update_on_zoom: bool [property]      ← derived: bool(recompute_axes)
├── _default_recompute_axes() → tuple    ← per-trace policy (see table below)
├── _axes: Tuple[str, ...] | None        ← render anchors; decoupled from recompute
│
├── group_by: str | list[str] | None     ← raw value from _params["group_by"]
├── group_by_cols: tuple[str, ...] | None← normalized tuple form of group_by
│
├── get_aggregation_spec(update_range, schema)
│     → AggregationSpec | GroupedAggregationSpec                  [abstract]
├── _to_update(df_agg) → TraceResult                              [abstract]
├── _to_grouped_update(df_grouped) → TraceResult                  [grouped parents]
├── _range_filter_exprs(col, range_, schema) → List[pl.Expr]      [convenience]
└── to_trace_spec(domain_source=None) / from_trace_spec()
      domain_source: computes group_domain_key internally
```

Cross-filter compilation is no longer per-trace.  ``flexviz/predicates.py::predicates_to_expr``
translates each ``SelectionState.predicates`` into a Polars expression directly from
column names, then the engine ANDs the expressions from non-source selections before
aggregation.  Click/brush handlers in adapters emit ``SelectionPredicate`` clauses by
reading the source trace's ``backend_data``.

### Zoom re-aggregation policy

Each trace declares the anchors whose viewport *range* parameterizes its
aggregation, via ``_default_recompute_axes()`` (overridable per trace; the
escape hatch on line/geo_line freezes a trace to ``()``).  A viewport change
re-aggregates a trace only when it moves one of these axes — so a line ignores
y-only zoom, a vertical histogram ignores count-axis zoom, and categorical
traces ignore zoom entirely.  The same concrete ``recompute_axes`` set is
serialized into `TraceSpec`, read by the client to suppress no-op `/update` POSTs
and by the engine (`_should_process_trace`) to gate per-trace recompute.  This is
a producer contract: current specs must carry the concrete tuple/list emitted by
``to_trace_spec()``.  Omitted or null values are not repaired in the browser (the
JS no-op guard treats them as empty), though server-side trace reconstruction may
still derive a default policy for hand-built specs.

The table below is generated from the trace registry by
`flexviz.trace.recompute_policy_table()` (a test keeps it in sync — do not edit
by hand):

<!-- recompute-policy:start -->
| `trace_type` | re-aggregates on zoom of |
|---|---|
| `bar` | none — never re-aggregates on zoom |
| `box` | none — never re-aggregates on zoom |
| `corr_heatmap` | none — never re-aggregates on zoom |
| `geo_histogram2d` | map coordinates — re-bins on viewport change |
| `geo_line` | map coordinates (frozen if update_on_zoom=False) |
| `histogram` | binned axis (x or y by orientation) — re-bins to viewport |
| `histogram2d` | both axes (x, y) — re-bins to viewport |
| `line` | x anchor — downsample window (frozen if update_on_zoom=False) |
| `pie` | none — never re-aggregates on zoom |
| `treemap` | none — never re-aggregates on zoom |
<!-- recompute-policy:end -->

### Selection geometry

Each trace declares the anchors whose box-select range becomes a selection
clause, via ``_default_select_axes()`` (anchor space, a subset of ``axes``).  A
line is **x-only** so a brush cross-filters every series sharing the x axis; a
1-D histogram/box selects only on its data (prop) axis (the orthogonal range is
dropped, not silently mis-applied); a 2-D histogram selects on both.  The
``select_axes`` set is serialized into `TraceSpec`, read by the client to
constrain the box-select (and to derive the Plotly `selectdirection` band-brush
geometry per figure).  This is also the *source geometry* of a cross-filter
brush that the cube pre-aggregation indexes on.

The table below is generated from the trace registry by
`flexviz.trace.select_policy_table()` (a test keeps it in sync — do not edit by
hand):

<!-- select-policy:start -->
| `trace_type` | selects on |
|---|---|
| `bar` | categorical — label click (not a box-range) |
| `box` | data (prop) axis only — orthogonal range dropped |
| `corr_heatmap` | none — not a cross-filter source |
| `geo_histogram2d` | map box — (lon, lat) bounds |
| `geo_line` | none — scattermap point selection not supported |
| `histogram` | data (prop) axis only — orthogonal range dropped |
| `histogram2d` | both axes (x, y) — 2-D box select |
| `line` | x anchor only — vertical band across all series |
| `pie` | categorical — slice click |
| `treemap` | categorical — hierarchical path click |
<!-- select-policy:end -->

### LinePlot

```python
fig.add_line(x="timestamp", y="value", name="Sensor A", n_points=1000, add_gaps=True)
```

- `trace_type = "line"`
- Three downsampling strategies, selected via `downsample` param:
  - **`"nth"`** — uniform stride: uses the `every_nth` Rust kernel (`pl.col(...).filter(vp).flexviz.every_nth(n_points)`) — stride computed inside the kernel, no `len()` expression dependency, enabling N grouped sub-traces to parallelize in one `select()`.
  - **`"minmax"` (default)** — min-max envelope: splits into `n_points // 2` buckets, selects argmin + argmax of y per bucket and gathers x and y at those indices. On a resident frame this uses the single `minmax_line` Rust kernel call (one kernel call on purpose: Polars does not CSE opaque plugin expressions, so the earlier two-gather form ran the whole scan twice per trace). On a scan source the kernel would materialise the whole column, so the engine swaps in `_streaming_envelope_plan`: a single streaming `group_by` with equal-width buckets in x and `min_by`/`max_by` for the x at each extremum. Preserves extrema and spikes on both paths.
  - **`"fpcs"`** — Feature-Preserving Compensated Sampling: runs the same MinMax bucket pass as `minmax`, then applies a compensation algorithm to carry forward "deferred" extrema across windows. Uses the `_fpcs_line` combined kernel (index selection + gather in one call). `n_points` is a target, not a hard cap; output can reach up to roughly `2 * n_points`.
- Viewport restriction, ungrouped lines: when the x column has been asserted sorted (`check_sorted` / `assume_sorted`, surfaced via `LFQueryBuilder.is_sorted`, threaded by the engine as `x_sorted`), the viewport becomes a binary-searched, zero-copy `slice(search_sorted(lo), search_sorted(hi) - start)`; otherwise a dtype-aware `is_between` mask. Performance-only choice — `tests/test_trace_line.py::TestSortedViewportSlice` asserts the slice returns exactly what the mask returns. Grouped lines always mask (the filter runs frame-level, before `group_by`).
- The engine normalizes descending viewport ranges (reversed plotly axes report high-to-low) to `lo <= hi` at ingestion — `_normalize_axis_ranges` in `engine.py` — so neither formulation ever sees a reversed pair.
- Grouped line traces use a true grouped query: viewport filtering is applied before
  `group_by`, and downsampling happens inside the grouped aggregation expression.
- `_to_update`: unpacks struct → `{"x": series, "y": series}`, gapless. Gap (`null`) breaks across large x jumps are a client-side display concern, inserted at render time by `fvApplyLineGaps` (`adapters/js/plotly/traces.js`) for every line trace — init, commit, and live cube.
- Range-brush selections emit a `ClauseFilter(range=...)` per axis; the predicate compiler applies typed `is_between` filters.

### Histogram

```python
fig.add_histogram(x="value", bins=20, histnorm="count")
```

- `trace_type = "histogram"`
- Supports `x` or `y`, not both simultaneously.
- Uses the `fixed_hist` Rust plugin with explicit bin bounds.
- **Temporal data axis**: `fixed_hist` is numeric-only, so a temporal column
  (`Date` / `Datetime`, any time zone) is binned on its `to_physical()`
  representation (µs / days). Viewport bounds and the injected global min/max
  stats are converted to the same physical unit (`LFQueryBuilder.aggregate`
  emits physical stats for temporal columns). `_to_update` casts the physical
  bin centers back to the temporal dtype so the delta carries datetimes (the
  renderer auto-detects a date axis, like the line trace) and emits hover-band
  bounds in epoch-ms. Mirrors the cube's `_typed_temporal_lit(...).to_physical()`
  idiom and `temporal_unit` (contract G).
- Optional viewport range filter applied before binning; without a viewport,
  per-column global stats provide stable edges across cross-filter updates.
- Multiple active histogram traces on the same figure, axes, data axis, and
  coordinate unit share one no-viewport min/max domain before calling
  `fixed_hist`. Numeric and differing temporal physical units remain separate.
- Grouped histogram traces use a fused grouped query with the visible-range filter
  applied before `group_by`.
- `_to_update` normalizes counts per `histnorm`; returns `{"x": centers, "y": counts}` (vertical) or `{"x": counts, "y": centers, "orientation": "h"}` (horizontal).

### BoxPlot

```python
fig.add_boxplot(y="value")
```

- `trace_type = "box"`
- Computes quantiles `[0, 0.25, 0.5, 0.75, 1]` via Polars, aliased to trace uid.
- Grouped box traces compute per-group quantiles inside one grouped aggregation and
  emit one logical child result per visible group.
- `_to_update` returns Plotly-style box arrays with `x0`/`y0` and orientation.
- `recompute_axes = ()` — box stats are not recomputed on viewport changes.

### BarPlot

```python
fig.add_bar(labels="category", values="revenue", agg="sum")
fig.add_bar(labels="category")                              # implicit count (values=None)
fig.add_bar(labels="label", values="revenue", agg="mean", group_by="region", orientation="h")
```

- `trace_type = "bar"` · `axes = ("x", "y")` · `recompute_axes = ()`
- `labels` column goes on the category axis; `values` is optional — when `None`, rows are counted (implicit count, no phantom column).
- `agg` ∈ `{"sum", "mean", "min", "max"}` — only meaningful when `values` is given; defaults to `"sum"`.
- Grouped bars (`group_by` given): one child series per distinct group value; `overlay_style = "filtered_only"`.
- Non-grouped bars: `overlay_style = "full"` — receives a background layer in overlay mode.
- `orientation = "v"` (default) or `"h"` — flips which axis holds labels vs values.
- `bar_mode` (`"group"` or `"stack"`) is stored in `display`; all bars in a figure must share the same mode.
- Click / box-select on a bar emits one `SelectionPredicate` per clicked category, with one `ClauseFilter` per `labels` column (composite labels decompose into a list of column-value clauses). Numeric labels are still categorical values here; their predicate values cast back through the backend schema.
- **Selection reads the bar's typed data label, not its axis coordinate.** String labels render on a Plotly *category* axis (the coordinate *is* the label), but numeric labels render on a *linear* axis where `barmode="group"` draws each group's bar at `category ± offset`. The adapter therefore takes the covered label from the trace's data array at the selected point index (`pt.data[catKey][pt.pointNumber]`), never the geometric `pt.x`/`pt.y` — so the predicate value, the cube free categories, and the committed `is_in` all carry the true typed label (`1`, not the offset `0.8`/`1.2`). This is what lets a grouped numeric source (e.g. `hour_of_day` by `source`) drive a live brush.
- `color_map` stored in `display["color_map"]`; applied to `marker.color` array in `_to_grouped_update`.

### PiePlot

```python
fig.add_pie(labels="category", values="revenue", agg="sum", hole=0.4)
fig.add_pie(labels="category")                              # implicit count (values=None)
```

- `trace_type = "pie"` · `axes = None` · `recompute_axes = ()` · `overlay_style = "filtered_only"`
- Non-cartesian — does not participate in zoom or linked hover; cross-filtering is click-based.
- `values` is optional — when `None`, rows are counted per label (implicit count, no phantom column).
- `agg` ∈ `{"sum", "mean", "min", "max"}` — only meaningful when `values` is given.
- Aggregation uses `GroupedAggregationSpec` with `group_cols=(labels_col,)`.
- `_to_update` returns `{"labels": [...], "values": [...]}`.
- Click on a slice emits a `SelectionPredicate` whose clauses cover every column in `labels` (single-column → one `ClauseFilter(values=[label])`; multi-column → one clause per column, decoded from the JSON-encoded composite slice label).
- Donut variant: `hole > 0` cuts the center.
- `color_map` stored in `display["color_map"]`; applied to marker colors in adapter rendering.

### TreeMap

```python
fig.add_treemap(path=["continent", "country"], values="revenue", agg="sum")
fig.add_treemap(path=["region", "category"])               # implicit count (values=None)
fig.add_treemap(path=["country", "source"], values="generation_mw", color_map={...})
```

- `trace_type = "treemap"` · `axes = None` · `recompute_axes = ()` · `overlay_style = "filtered_only"`
- Non-cartesian — does not participate in zoom or linked hover; cross-filtering is click-based.
- `path` is a list of column names defining the hierarchy from root to leaf (e.g. `["continent", "country"]`).
- `values` is optional — when `None`, rows are counted per node (implicit count).
- `agg` ∈ `{"sum", "mean", "min", "max"}` — only meaningful when `values` is given; defaults to `"sum"`.
- Aggregation uses `GroupedAggregationSpec` with `group_cols = tuple(path)` and `sort_cols = tuple(path)`.
- `_to_grouped_update` builds a hierarchical node tree: a root node plus one node per distinct combination of path values. Returns `{"labels": [...], "parents": [...], "ids": [...], "values": [...]}` for Plotly treemap rendering.
- Path-click emits one `SelectionPredicate` whose clauses are `[ClauseFilter(column=path[i], values=[part_i]), ...]` from the leaf node up to the clicked depth — every parent column on the chosen branch is constrained, so cross-filtering follows the path naturally.
- `color_map` stored in `display["color_map"]`; applied to `marker.colors` array mapping label strings to CSS colors.

### Histogram2D

```python
fig.add_histogram2d(x="x", y="y", x_bins=20, y_bins=20, color_scale="plasma")
fig.add_histogram2d(x="x", y="y", histfunc="sum", z="weight", histnorm="percent")
```

- `trace_type = "histogram2d"` · `axes = ("x", "y")` · `recompute_axes = ("x", "y")` · `overlay_style = "filtered_only"`
- `z` is optional — when `None`, rows are counted per bin (implicit count). When given, `histfunc` is required.
- Count-only `Histogram2D` uses the `flexviz_polars` `fixed_hist2d` Rust kernel when available.
- z-column `Histogram2D` supports `histfunc in {"sum", "mean", "min", "max"}` and uses `fixed_hist2d_reduce` when available.
- A temporal x and/or y axis is binned on its `to_physical()` representation (the kernel is numeric-only) with physical bin edges; `_to_update` restores datetime centers on that axis (date axis) and emits epoch-ms hover bounds — same scheme as the 1-D `Histogram`.
- `median` and `n_unique` are intentionally not supported for cartesian `Histogram2D` in this fast-path stage; they can be added back as separate reducers if needed.
- The current viewport path still prefilters x/y/z before calling the Rust kernel. A later viewport-aware kernel can fuse range rejection into the Rust loop.
- `histnorm` controls post-aggregation normalization: `None` (no normalization, default), `"percent"`, `"probability"`, `"density"`, `"probability density"`.
- Bin edges span the viewport range (or data range on init).
- Empty bins are emitted as `None`; empty viewports / all-null inputs produce an all-null grid so renderers show gaps instead of zero-count cells.
- Public style API: `color_scale` and `color_range`; trace-owned defaults are `"viridis"` and `"auto"`.
- `_to_update` returns `{"x": [...centers], "y": [...centers], "z": [[values]]}`.
- Box-select inside the heatmap emits a `SelectionPredicate` with two `ClauseFilter(range=...)` clauses (x and y columns).

### GeoHistogram2D

```python
fig.add_geo_histogram2d(lat="latitude", lon="longitude", lat_bins=64, lon_bins=64)
fig.add_geo_histogram2d(lat="lat", lon="lon", histfunc="mean", z="temperature")
fig.add_geo_histogram2d(lat="lat", lon="lon", bin_boundaries="viewport")
```

- `trace_type = "geo_histogram2d"` · `axes = None` · `recompute_axes = ("coordinates",)` · `overlay_style = "filtered_only"`
- `GeoHistogram2D` uses the `flexviz_polars` Rust kernel (same as `Histogram2D`): count via `fixed_hist2d`, z-reduction via `fixed_hist2d_reduce`. lat maps to the kernel x (inner) axis and lon to the y (outer) axis so `z_flat` is already in the lon-major order the GeoJSON builder needs.
- z-column support is `histfunc in {"sum", "mean", "min", "max"}`. `median` and `n_unique` are **not** supported on this fast path (mirrors `Histogram2D`); see the Roadmap below. `from_trace_spec` raises on legacy `median`/`n_unique` specs.
- Rows are clipped to the visible map bounds before binning (a `.filter(mask)` inside the kernel expression). Viewport is passed as `update_range["coordinates"]` — a list of `[lon, lat]` corner points (from `PlotlyAdapter` / map `relayout`).
- **`bin_boundaries`** (`params`, default `"data"`): `"data"` passes the min/max of the (viewport- and cross-filter-filtered) points as the kernel `lo`/`hi` bounds (bins shift when panning/zooming). `"viewport"` passes the viewport rectangle bounds (stable grid on pan/zoom, analogous to plotly-flex `bin_boundaries="viewport"`).
- `histnorm` is applied in `_to_update` (post-kernel) over the flat z grid, with `bin_area = lat_step * lon_step`; bin centers/edges are derived from the kernel-echoed `lo`/`hi` and `nb`.
- `_to_update` returns `{"geojson": {...}, "locations": [...], "z": [...]}` — a GeoJSON FeatureCollection of rectangular polygons for choropleth rendering.
- Cross-filtering: Plotly geo selections are derived from the selected choropleth bin ids (`locations`) and collapsed to one lon/lat bounding box, then emitted as a `SelectionPredicate` with two `ClauseFilter(range=...)` clauses on the trace's `lon` and `lat` columns.
- Public style API: `color_scale` and `color_range`; defaults are `"viridis"` and `"auto"`.
- PlotlyAdapter renders as a `choroplethmap` trace with OpenStreetMap base tiles.
- ECharts geo rendering is not yet supported.

### CorrHeatmap

```python
fig.add_corr_heatmap(
    columns=["a", "b", "c"],
    method="pearson",
    absolute=False,
    color_range="auto",
)
```

- `trace_type = "corr_heatmap"` · `axes = None` · `recompute_axes = ()` · `overlay_style = "filtered_only"`
- Computes pairwise correlation matrix via Polars `pl.corr(...)` expressions packed into a symmetric matrix; supports `"pearson"` and `"spearman"` methods.
- When `columns=None`, auto-detects all numeric columns from the schema.
- `absolute=True` applies `abs(corr)` inside the aggregation; `triangular=True` masks the mirrored upper triangle in `_to_update`.
- Public style API: `color_scale` and `color_range`; defaults are semantic: signed = `"rdbu"` / `(-1.0, 1.0)`, absolute = `"viridis"` / `(0.0, 1.0)`, with explicit `"auto"` allowed.
- `_to_update` returns `{"x": [...col_names], "y": [...col_names], "z": [[corr_values]]}`.
- CorrHeatmap is not a cross-filter source: it neither emits selection predicates nor responds to incoming filters.
- Reuses the same heatmap rendering code in adapters as Histogram2D.

### Heatmap Styling Contract

```python
fig.add_histogram2d(x="x", y="y", color_scale="plasma", color_range=(0.0, 5.0))
fig.add_corr_heatmap(columns=["a", "b"], color_scale="rdbu", color_range="auto")
```

- `color_scale` and `color_range` live in `TraceSpec.display`, not in `params`, because they are renderer-facing style hints.
- Defaults are owned by the trace classes: `Histogram2D` materializes `"viridis"` / `"auto"`; `CorrHeatmap` materializes signed vs absolute defaults based on `absolute`.
- Generated specs must include explicit `display.color_scale` and `display.color_range`; `Figure` and adapters validate this invariant instead of re-deriving heatmap defaults.
- `from_trace_spec()` on the heatmap traces remains the single backward-compat normalization point for older specs missing those style keys.
- `Figure.to_spec()` validates that all heatmap-like traces in one figure share the same effective style, because renderers treat the heatmap color control as figure-level.
- Renderer split:
  - Plotly consumes the raw `color_scale` string directly and applies `zmin` / `zmax` only when `color_range` is fixed.
  - ECharts maps a supported set of heatmap scale names (`viridis`, `plasma`, `magma`, `inferno`, `cividis`, `blues`, `reds`, `rdbu`) to local color arrays for one per-figure `visualMap`.

### Dtype-Aware Filtering

`base.py` provides shared helpers used by all trace subclasses:

```python
_dtype_for_col(schema, col_name)   → pl.DataType | None
_typed_temporal_lit(value, dtype)  → pl.Expr
_typed_range_bounds(lo, hi, dtype) → Tuple[pl.Expr, pl.Expr]
_range_filter_expr(col, lo, hi, schema) → pl.Expr  # is_between with typed literals
```

All traces call these for consistent datetime, integer, and float casting in `is_between` expressions.

---

## Plugin Layer

`flexviz_polars` is a required Rust/Polars plugin crate in `flexviz_polars/`. It exposes performance-critical downsampling kernels as a `.flexviz` Polars expression namespace.

**Build and install** (requires Rust + maturin in dev dependencies):
```bash
make build-plugin          # debug build
make build-plugin-release  # release build (use for benchmarks)
```

**Public API** (activated by `import flexviz_polars`):
```
FlexvizExprNamespace  — registered as pl.Expr.flexviz via @pl.api.register_expr_namespace
├── every_nth(n_points: int) → pl.Expr
│     Stride-based downsampling. Stride computed inside Rust kernel from series length.
│     No Polars len() expression dependency → N grouped sub-traces parallelize in one select().
│     Returns at most n_points elements of the same dtype.
│
├── arg_min_max(n_points: int) → pl.Expr
│     Min-max envelope. Splits into n_points//2 buckets; returns sorted, deduplicated
│     UInt32 indices of argmin + argmax within each bucket. Pass to .gather() to
│     build a downsampled series that preserves extrema and spikes.
│
├── fpcs(n_points: int) → pl.Expr
│     Feature-Preserving Compensated Sampling (indices only). Runs the same MinMax
│     bucket pass as arg_min_max, then applies a compensation pass that carries
│     "deferred" extrema across window boundaries. Returns sorted, deduplicated
│     UInt32 indices. n_points is a target, not a hard cap; output may reach ~2×.
│     Prefer _fpcs_line() for the combined index+gather path used by LinePlot.
│
├── fixed_hist(lo_expr, hi_expr, n_bins: int) → pl.Expr
│     O(n) fixed-bin 1D histogram. lo_expr / hi_expr evaluate to scalar (length-1)
│     Series; n_bins is the number of bins. Uses direct floor-division indexing
│     instead of binary search. Returns Struct{breakpoint: Float64, count: UInt32}
│     of length n_bins — identical output shape to polars.hist(include_breakpoint=True).
│     Used by Histogram trace (both ungrouped and grouped paths) to guarantee
│     O(n) bin assignment with stable, pre-specified edges.
│
├── fixed_hist2d(y_expr, x_lo, x_hi, y_lo, y_hi, nb_x, nb_y) → pl.Expr
│     O(n) fixed-bin 2D count histogram. Uses typed dispatch (avoids f64 cast for
│     same-type pairs) and a contiguous-slice fast path for null-free single-chunk
│     inputs. Returns Struct{z_flat: List(UInt32), x_lo, x_hi, y_lo, y_hi} of
│     length 1; z_flat is row-major (yi * nb_x + xi). Empty bins have count 0.
│     Used by Histogram2D (count-only fast path) and GeoHistogram2D (count;
│     lat→x, lon→y). lo/hi may be literals or scalar expressions (e.g. a
│     filtered min/max), and are echoed back unchanged for edge derivation.
│
└── fixed_hist2d_reduce(y_expr, z_expr, x_lo, x_hi, y_lo, y_hi, nb_x, nb_y, histfunc) → pl.Expr
      O(n) fixed-bin 2D reducer for histfunc ∈ {"sum", "mean", "min", "max"}.
      Same typed-dispatch / cont_slice optimizations as fixed_hist2d. Returns
      Struct{z_flat: List(Float64), x_lo, x_hi, y_lo, y_hi} of length 1; empty
      bins are null. Used by Histogram2D and GeoHistogram2D when z is given.
      `median` / `n_unique` are not implemented (see Roadmap).

flexviz_polars._fpcs_line(x_expr, y_expr, n_points, x_name=None, y_name=None) → pl.Expr
      Combined FPCS index-selection + gather kernel. Takes x and y expressions,
      runs FPCS, gathers both x and y at the selected indices in a single pass.
      Returns Struct{x_name: x_dtype, y_name: y_dtype}. Used by LinePlot with
      downsample="fpcs" to avoid a separate gather step.

flexviz_polars._minmax_line(x_expr, y_expr, n_points, x_name=None, y_name=None) → pl.Expr
      Combined min-max index-selection + gather kernel, same shape as _fpcs_line.
      Runs the arg_min_max bucket pass on y, gathers both x and y at the selected
      indices in one call. Returns Struct{x_name: x_dtype, y_name: y_dtype}. Used
      by LinePlot with downsample="minmax" (the default). Exists because Polars
      does not CSE opaque plugin expressions: the two-gather form ran the scan
      twice per trace. tests/test_trace_line.py pins the one-call plan shape, and
      TestMinmaxLineDifferential pins bit-identity against the two-gather form.
```

**Rust internals** (`src/expressions.rs`):
- `every_nth` — computes `stride = max(1, len / n_points)`, builds capped gather indices, calls `Series::take()`.
- `arg_min_max` / `minmax_line` — reuse internal helpers: `uniform_offsets`, `simd_argminmax` (SIMD fast path for contiguous numeric types), and `fallback_window_argminmax` (general Series API for non-SIMD/other dtypes). Flatten min+max indices, sort, deduplicate; `minmax_line` then `take`s x and y at those indices inside the same call. The window scan is rayon-parallel via `par_by_window` on the kernel pool — split by whole windows, so output is bit-identical to the serial path by construction. The split is a trade against the memory-bandwidth ceiling (see its doc comment for the measured numbers): concurrent traces queue through the shared pool, worth ~1.3-1.6x at one trace on a bandwidth-saturated host and bounded at ~-9% for 3-5 concurrent traces, fading by 20.
- `fpcs` / `_fpcs_line` — shares the `arg_min_max` MinMax bucket pass via `arg_min_max_pairs()`; then `fpcs_compensate()` dispatches per dtype, calling `fpcs_compensate_with_values()` which carries deferred extrema across windows with a dedup guard on every push (prevents duplicate index 0 when the first bucket's argmin equals the start point, and prevents endpoint duplication).
- `fixed_hist` — dispatches on native dtype via `FixedHistValue` trait (avoids full-column cast); maps each value to a bin with `((v - lo) * scale).floor()` (+ round eps), clamps boundaries. **Rayon-parallel**: null-free contiguous runs are split into work units folded into private per-group count tables and merged by add. Falls back to the scalar single-table loop for chunks with nulls, undispatched dtypes, input below `MIN_PAR` (2^17 rows), or when two private tables exceed the byte budget — counts are identical either way (asserted against an independent Python reference in `test_plugin_functions.py`).
- `fixed_hist2d` — O(n) 2D binning, same parallel/fallback structure as `fixed_hist` (x/y chunks aligned via `align_chunks_binary`); counts (UInt32) stored row-major. `fixed_hist2d_reduce` (Float64 reductions) remains single-threaded.
- **Private-table budget**: both parallel kernels bound live scratch by `MAX_PRIVATE_BYTES` (32 MiB). Work units are folded in at most `n_chunks` groups — one table per group — so fragmented (many-chunk) input cannot multiply tables past the budget; when even two tables do not fit (≳4M bins), the kernel stays scalar.
- **Kernel thread pool**: a cdylib statically links its own polars-core, so it cannot join Polars' rayon pool (pola-rs/polars#19650). The kernels run on a dedicated `OnceLock` pool sized `POLARS_MAX_THREADS` → `RAYON_NUM_THREADS` → `available_parallelism()`, so a container that limits Polars' threads limits the kernels too.

**Integration with LinePlot**: `line.py` imports `flexviz_polars` at module level. `LinePlot.get_aggregation_spec()` dispatches on `self.downsample` via `_plugin_line_agg_expr()` to select `_plugin_nth_agg_expr`, `_plugin_minmax_agg_expr`, or `_plugin_fpcs_agg_expr`.

**Integration with Histogram**: `hist.py` imports `flexviz_polars` at module level. Both the ungrouped and grouped paths in `Histogram.get_aggregation_spec()` call `.flexviz.fixed_hist(lo_expr, hi_expr, n_bins=self.bins)` instead of `polars.hist(bins=...)`, providing O(n) stable-edge binning.

**Integration with Histogram2D / GeoHistogram2D**: `hist2d.py` and `geo_hist2d.py` both import `flexviz_polars` at module level (hard import — raises `ImportError` without the plugin). The count path calls `.flexviz.fixed_hist2d(...)`; the reduce path calls `.flexviz.fixed_hist2d_reduce(...)`. `GeoHistogram2D` maps lat→x and lon→y and passes filtered-data min/max (or viewport bounds) as the kernel `lo`/`hi`.

---

## Engine Layer

`FlexEngine` in `engine.py` is stateless and reconstructed per request.

```
FlexEngine
├── _backend_lf: LFQueryBuilder
├── _scalable_traces: Dict[str, FlexTrace]   ← keyed by trace uid
│
└── process(event, trace_infos, viewports_by_figure, cross_filter_mode) → List[TraceDelta]
      0. Normalize axis ranges at ingestion: descending (hi, lo) viewport pairs
         (reversed plotly axes report high-to-low) are swapped so every
         consumer downstream holds the lo <= hi invariant
      1. Derive active selections from event.type + event.selections
      2. Identify source figures; build cross-filter Polars exprs by passing each
         non-source selection's predicates through `predicates_to_expr` (one expression
         per selection — predicates ORed inside, expressions ANDed across selections)
      3. Decide which traces need re-aggregation (one uniform rule for cartesian
         and map traces):
         force_update=True  OR  (trace.recompute_axes ∩ changed event axes ≠ ∅)
         where changed axes are cartesian keys ("x"/"y2"/…) or "coordinates" for maps
      4. Collect AggregationSpec / GroupedAggregationSpec per active trace;
         for histogram traces without a viewport range, group coordinate-compatible
         same-figure siblings by (figure_uid, axes, data_axis, coordinate_unit) via
         `_histogram_domain_cols_by_uid` so they share one unfiltered global min/max
         domain (no extra collect)
      5. In update mode: aggregate once with selection filters
      6. In overlay mode: filter agg specs per layer via overlay_style
         (with an active selection, traces with "filtered_only" reuse their
         cached unfiltered layer instead of recomputing bg; init/deselect/reset
         still emit that sole unfiltered layer for every trace);
         execute only the layers required by the event
         (`selection` → fg, `init`/`deselect`/`reset` → bg,
         `viewport` → bg or bg+fg depending on active selections)
      7. Route regular results to _to_update(df_agg)
      8. Route grouped results to _to_grouped_update(df_grouped)
      9. Normalize Series → list once; emit TraceDelta / GroupedChildDelta
```

`TraceInfo` (dataclass) carries `uid`, `axes`, `trace_type`, `figure_uid` — the minimal metadata the engine needs without holding trace instances directly.

**Cross-filtering:** Selections live in `state.selections`.  Each `SelectionState` carries one or more `SelectionPredicate` objects whose clauses translate directly to Polars expressions via `flexviz/predicates.py::predicates_to_expr` and are applied lazily to the shared LazyFrame *before* aggregation.  Predicates from one selection are ORed; predicates from different source figures are ANDed by `LazyFrame.filter(*exprs)`.  When a figure is the source of a selection its traces are excluded from re-aggregation during `selection` events.  Trace classes no longer participate in filter compilation — every selection is interpreted column-by-column at the engine boundary.

**Per-figure reset (`fvOnResetPanel`):** each figure's panel has its own Reset button that resets **only that figure**, with semantics that depend on the *direction* of any cross-filter relative to the figure:

- *Source* (the figure holds a selection filtering others): the selection is cleared, so other figures un-filter (keeping their own zoom). The figure's own viewport is also cleared.
- *Target* (other figures' selections filter this figure): only this figure's viewport is reset; the incoming selections are **kept and stay applied**, so the figure re-renders filtered-by-others at autorange and the source figures are untouched.
- *Sole source* (the figure's selection is the only one in the dashboard): clearing it returns the whole dashboard to unfiltered.

Because the engine keys its recompute/scoping on event type, `fvOnResetPanel` derives the emitted type from the resulting state: `selection` when filters remain, `deselect` when none remain, `viewport` (scoped to the figure) when only the viewport changed. The global toolbar **Reset** (`fvOnReset`) instead clears all viewports + selections and emits `init`; the global **Deselect** clears selections only and **keeps zoom** (emitting `deselect`).

**Treemap / pie multi-click:** successive clicks on the same figure append OR predicates via `fvUpsertPathPredicate` in the shared runtime (Plotly and ECharts).  Re-clicking the same node toggles that predicate off; refining along one branch (parent → child or child → parent) replaces the broader/narrower predicate instead of accumulating redundant filters.  *UX note:* this follows common additive-filter BI patterns; we should periodically reassess whether modifier keys or explicit multi-select mode would better match natural visual exploration for hierarchical charts.

**Grouped architecture:** the engine no longer discovers groups or fabricates child traces. Group membership is decided in the grouped Polars query, and traces own the conversion from grouped result frames to child payloads.

---

## Data Layer

`LFQueryBuilder` in `LF.py` wraps a `pl.LazyFrame`.

```
LFQueryBuilder
├── _ldf: pl.LazyFrame
├── _row_index_col: str | None  ← optional; not auto-added anymore
├── _sorted_cols: Set[str]
│
├── schema                      ← cached property
├── check_sorted(col)           ← verifies with collect() — O(n)
├── assume_sorted(col)          ← skips verification; caller guarantees order
└── aggregate(filter_exprs, agg_specs) → tuple[pl.DataFrame, dict[str, pl.DataFrame]]
      base_ldf = _ldf (+ with_columns(col.min(), col.max()) for any global_stats_cols)
      filtered_ldf = base_ldf if not filter_exprs else base_ldf.filter(*filter_exprs)
      regular specs  → one batched select(...).collect()
      grouped specs  → fused group_by(...).agg(...).sort(...).collect() per batch
```

`AggregationSpec` (dataclass):

```
AggregationSpec
├── expr: pl.Expr    ← evaluated in the shared filtered LazyFrame context;
│                       output column aliased to trace uid; yields a Struct series
├── uid: str = ""    ← trace uid; used by engine for overlay_style dispatch
└── global_stats_cols: Tuple[str, ...] = ()
                     ← columns for which __hist_lo_<col>__ / __hist_hi_<col>__
                        are added on the unfiltered base frame before filtering
```

`GroupedAggregationSpec` (dataclass):

```
GroupedAggregationSpec
├── uid: str
├── group_cols: Tuple[str, ...]
├── sort_cols: Tuple[str, ...]
├── agg_exprs: Tuple[pl.Expr, ...]       ← already aliased to logical parent uids
├── pre_group_filters: Tuple[pl.Expr, ...]
├── pre_group_filter_key: Any            ← semantic key for grouped fusion safety
├── batch_key: Tuple[Any, ...]
└── global_stats_cols: Tuple[str, ...] = ()
                     ← same semantics as AggregationSpec.global_stats_cols
```

Grouped specs with identical `(group_cols, sort_cols, batch_key)` are fused into one
Polars grouped query. If `pre_group_filters` are present, every fused spec must also
provide the same `pre_group_filter_key`; otherwise fusion is rejected. The collected
grouped `DataFrame` is then shared back to each logical parent uid in that batch.

`_to_update` receives the result as `df_agg[uid][0]` — a Python dict of `{field: value}` where array fields are Python lists (after `implode()` inside the trace expression).

---

## Server Layer

FastAPI app in `server.py`. The only persistent server state is:

```python
_sources: Dict[str, LFQueryBuilder]
```

Populated via `register_source(name, data, cache=False)` at `show()` time. Everything else is request-scoped.

### Caching carve-out to the stateless invariant

The "stateless server" invariant forbids *authoritative interaction state* (viewport, selection, overlay, hover, grid) — it does **not** forbid a **content-addressed memoization cache**. `flexviz/cache.py` may hold a reconstructable cache derived solely from `_sources` + request content, provided it is:

- **session-invariant** — keyed only on request content (source + trace identity); zero session/connection/client bytes, so behaviour is identical for 1 or 10,000 viewers and for a viewer's 1st or Nth request;
- **never authoritative** — a miss recomputes a byte-identical result, so correctness never depends on a hit (any replica may serve any request);
- **droppable** — eviction is global (LRU/size), never per-session.

Phase 1 (issue #26) caches only the **unfiltered *and* viewport-free** computation, gated on a per-source `cache=True` flag that **asserts the source data is static for the process lifetime** (no data-change invalidation yet — issue #27). The content key is viewport-blind, so a trace is cached **only when its resolved `update_range` is empty** — a trace that is zoomed/panned is neither stored nor served and always recomputes (otherwise a zoomed result would alias the full-range entry). This makes the eligible events `init`, `reset`, and *unzoomed* `deselect`:

- `init` and `reset` are viewport-free by construction (`reset` forces an empty `update_range`);
- `deselect` clears selections but **preserves zoom**, so a deselect issued while zoomed is viewport-dependent and bypasses the cache.

In the engine the short-circuit only fires when *every* delta-producing trace is a viewport-free cache hit; if any deliverable trace is zoomed, the request falls through to a normal recompute of all traces (the viewport-free ones are still stored for a future fully-unzoomed request). The cache is engine-hosted (injected `CacheBackend`), in-process (issue #28 adds Redis/disk), and mirrored client-side as a whole-response `Map` (`runtime/cache.js`); the client cache is additionally gated on **no figure being zoomed** (a single zoomed figure disqualifies the whole-dashboard entry). Re-registering an existing source name clears the cache wholesale (its data may have changed); registering a new name leaves other sources' entries intact. The server never tracks client cache state; the set of cacheable sources is embedded into the bootstrap (`FV_CACHEABLE_SOURCES`). The same carve-out and invalidation hook cover the second, byte-bounded **cube-blob cache** (see "Cube Pre-Aggregation & Live Brushing" below) — re-registering a source clears both.

### Routes

| Method | Path                | Purpose                                           |
|--------|---------------------|---------------------------------------------------|
| `POST` | `/update`           | Single-figure interaction; returns `List[TraceDelta]` |
| `POST` | `/dashboard/update` | Dashboard interaction; returns per-figure deltas  |
| `POST` | `/share`            | Encode spec → shareable URL                       |
| `GET`  | `/view`             | Render shared spec (`?renderer=plotly\|echarts`)  |
| `GET`  | `/sources`          | List registered source names (health check)       |
| `GET`  | `/cache/stats`      | Cache hits/misses/entries + cacheable sources     |

### Request Flows

**Single figure (`POST /update`):**
1. Resolve `figure.source` → `LFQueryBuilder`
2. `build_trace_from_spec()` per `TraceSpec`
3. Construct `FlexEngine`; run `engine.process()` in threadpool
4. Serialize deltas to JSON; grouped parent deltas preserve `group_results=[]`

**Dashboard (`POST /dashboard/update`):**
1. Collect distinct sources across all figures; resolve each once
2. Reconstruct all traces; group by source
3. `"cube_request"` events short-circuit here: run `FlexEngine.build_cubes` for the
   `active_source` figure's source only (gated on `request_cube=True` and a `cache=True`
   source) and return a **binary cube bundle** instead of JSON deltas
4. Run a separate `FlexEngine` per source
5. Scope viewport/reset events to `event.figure_uid` when set
6. Partition `TraceDelta` list by figure uid

`UpdateRequest` / `DashboardRequest` carry the additive cube fields
(`request_cube: bool = False`, `active_source: ActiveSource | None`). A `cube_request` is
answered out-of-band (not via the `UpdateResponse` / `DashboardResponse` JSON models) with an
`application/octet-stream` **cube bundle** — `encode_cube_bundle` packs the FVCube blobs raw
(no base64) plus the `trace_cubes` map (target trace uid → blob index) behind a thin binary
envelope, and the cube path gzips it itself at a fixed low level (`_cube_response`) so the
`GZipMiddleware` leaves it untouched. Shipping raw binary instead of base64-in-JSON avoids 33%
inflation and the high-CPU gzip-of-text that dominated cached-cube TTFB. The single-figure
`/update` cube path is plumbing only — one figure has no cross-filter targets, so it always
returns an empty bundle.

**Share / restore:**
- `POST /share` → `/view?spec=<encoded>&renderer=...` — the URL is built from the client-sent
  `server_url` and may be relative (deployed pages bake in a path like `/demo`); the toolbar
  absolutizes it against `window.location` before copying.
- `GET /view` decodes spec; wraps single-figure specs into a one-figure `DashboardSpec`; returns
  adapter HTML with a page-relative `SERVER_URL` (`"."`) — the server cannot know its external
  base URL behind a prefix-stripping reverse proxy, so API calls resolve in the browser as
  siblings of `/view`.

`mount_into(host_app, prefix="/flexviz")` mounts the flexviz ASGI app into an existing Starlette/FastAPI application.

---

## Cube Pre-Aggregation & Live Brushing

Every cross-filter interaction above recomputes the target aggregations over the (filtered) raw
data — O(n) in rows, once per committed selection. The **cube** path ports Mosaic's data-cube
index so an *active brush* becomes a slice over a tiny pre-aggregation — O(cells), flat in
dataset size — and adds drag-time updates that flexviz previously did not have at all (only
`plotly_selected`/mouseup was bound). This section is the normative description of what is
implemented.

A cube is one target trace's grouping × the brushed (free) axis, holding decomposable partial
measures. A **range** free axis (hist / box / line source) is binned to a **fixed resolution
P = 2048** over the source figure's viewport domain; a **box2d** free axis (a 2-D box-select on a
hist2d source) is two range axes binned at **P₂D = 128** each and packed into one composite
`free_bin`; a **categorical** free axis (bar/pie/treemap source) is the exact tuple of
label/path column values — no binning, no domain, dictionary-encoded in sorted order. Fixed P —
rather than Mosaic's pixel resolution — makes the cube width-independent,
content-addressable, and shareable across sessions; shipping the whole cube to the browser
(rather than slicing it server-side per frame) makes every drag step a local computation. Rows
with an out-of-domain or null free/target value are **filtered, not clipped** during the build;
a range value exactly at the domain max lands in the degenerate top bin `P`.

#### Measures (partial algebra)

`MeasureSpec(agg, value_col)` — `value_col` required for every agg except `count` (and forbidden
for it). Each cube cell stores *partials* that combine associatively across cells, so any slice
finalizes locally:

| agg | partials stored | combine | finalize |
|---|---|---|---|
| `count` | `count` (u32) | Σ | Σ |
| `sum` | `sum` (f64) | Σ | Σ |
| `mean` | `sum` (f64) + non-null `count` (u32) | Σ, Σ | `Σsum / Σcount`; 0-count cells omitted (≡ legacy absent label) |
| `min` / `max` | `min`/`max` (f64) | min/max skipping NaN | all-NaN cells omitted |
| `corr` | per explicit pair: `n`, `Σx`, `Σy`, `Σxy`, `Σxx`, `Σyy` (mean-centered, f64) | Σ each | `r = cov/√(varₓ·var_y)`; non-finite ⇒ 0.0 |
| `line_env` | per x-bucket: packed `(min_y, max_y)` as **f32** + the argmin/argmax x as **u16** offsets | min/max over y, keep extremal x | the row-bucket min-max envelope |

f64 partials encode null as NaN. Counts and min/max reslice bit-exactly; sums/means/corr match a
direct aggregation within ~1e-9 (combine-order float caveat — tested at that tolerance).
`median`/`n_unique` are not decomposable ⇒ those traces are not cube targets. The `line_env`
and `corr` builds bin/scan the free axis as a numeric range OR partition it by category, so they
require a range (continuous/temporal) **or categorical** free axis (`cube_target_buildable`); only a
**box2d** (hist2d) source is rejected for them — it falls back to the per-commit recompute (the
cell-count wall, #47), exactly like the `box`/`median` targets above. A categorical source partitions
the build by the free category (`__free__` columns) and runs the `fixed_line_envelope2d` kernel with
a degenerate 1-bin free axis (line_env) or `group_by(__free__cols)` (corr).
`corr` is **pearson-only** (spearman is rank-based, not decomposable) and needs ≥2 explicit numeric
columns. The `line_env` measure is the only one shipped in a **packed f32/u16** layout (not f64):
it is an approximate live-drag envelope, so its commit always POSTs (see "the line envelope
caveat" below).

```
flexviz/cube.py                      ← trace-agnostic core: CubeSpec descriptor,
                                        build_cube (pure Polars, streaming collect),
                                        cube_content_key, FVCube v1 binary codec
flexviz/trace/base.py                ← get_cube_source_spec / get_cube_target_spec
                                        (return None = not cube-capable; hist overrides)
flexviz/engine.py                    ← FlexEngine.build_cubes: assemble CubeSpecs,
                                        resolve None domains, dedupe by content key,
                                        build/fetch + encode
flexviz/cache.py                     ← second, byte-bounded cube-blob cache
flexviz/adapters/js/runtime/cube.js  ← client store, FVCube decode, CSR slice
                                        (generalized partials → cells), hist/bar/pie
                                        delta synthesis, grouped-child reconciliation,
                                        snap arithmetic, predicate→category matching
flexviz/adapters/js/plotly/events.js ← gesture machine: plotly_selecting binding
                                        (range + categorical), selection-edit drag
                                        watcher (move/resize an existing box), rAF
                                        throttle, snapped commit, click conditional
                                        commit, capability demotion, conditional POST
```

The cube core never branches on `trace_type`: traces emit descriptors (`FreeAxisSpec` /
`CubeTargetSpec`), and the engine/builder consume them. The server stays stateless — a
`cube_request` is an ordinary request carrying the full spec; the caches below are
content-addressed memoization under the same carve-out as the delta cache, never authoritative.

### Request flow (`live_brush: "auto"`, the default)

`ClientState.live_brush` (`"auto" | "off"`, client-only state) gates the whole system. `"off"`
never binds `plotly_selecting` and skips the click conditional commit — today's behavior
bit-for-bit, zero cube cost. With `"auto"`, `init.js` binds `plotly_selecting` on figures with
range **or categorical (bar)** selection geometry:

1. **First `plotly_selecting` of a drag = gesture start.** The client resolves the source —
   a range constraint (exactly one (column, axis-role) pair across the figure's range traces;
   2-D / multi-column range geometry is not a source) or, failing that, a categorical source
   trace (free descriptor = the ordered label columns, `p: 0`, no domain) — computes the
   canonical descriptor key per target trace, and checks the client cube store. All cube-capable
   targets present ⇒ the gesture is live immediately. Any miss ⇒ **one range-less `cube_request`
   POST** (`request_cube: true` + `active_source: {figure_uid, column, trace_uid}` — `column` is
   the primary label column for categorical sources, and `trace_uid` names the brushed trace so the
   engine resolves the source trace by uid when two traces in a figure share a column; the
   in-progress range never touches the server). The server
   computes **no deltas** for it; the response is a binary cube bundle (`decodeCubeBundle`) of
   the FVCube blobs (one per distinct cube) plus `trace_cubes` (target trace uid → blob index —
   **the trace uid is the client/server join key**, no cross-language hash). **Capability self-healing:** any
   optimistically-capable target absent from `trace_cubes` is demoted (the server's verdict —
   e.g. a numeric label dtype the client cannot see), so one incapable target never pins the
   gesture to mouseup-only. A failed/timed-out request degrades the gesture to mouseup-only.
2. **Each further `plotly_selecting`** (rAF-throttled, superseded frames dropped) re-slices
   locally. A range source snaps the in-progress range outward to the P-grid
   (`lo_bin = floor((a-lo)/span·P)` clamped to `[0, P]`); a categorical source matches the
   covered labels to category codes (deduped per frame on the sorted label set). The slice
   accumulates partials per composite target key over CSR row ranges and finalizes per the
   measure table; per-trace delta synthesis mirrors the Python `_to_update`/`_to_grouped_update`
   builders exactly — ungrouped hist (centers, histnorm, hover_bounds, orientation), bar
   (sorted labels, orientation, color_map), pie (`marker.colors`), line (the packed-envelope
   `(x, y)` polyline), corr (finalize `r` per pair → z-matrix), hist2d (densified nb_y×nb_x z,
   centers, per-slice `histnorm`), treemap (finalize leaf cells → level-summed `ids`/`parents`/
   `labels`/`values` rollup, ids url-quoted to match Python `quote`), and grouped targets (cells
   split by group dims; child updates reconciled through `setGroupedLayerData`, **child uids
   always reused from the recorded grouped layer data, never minted** — they embed a SHA-1).
   Everything routes through the existing layer-data + render path. Non-cube targets simply
   hold their pre-drag state. In overlay mode the slice drives the `fg` layer.
3. **`plotly_selected` (mouseup) = conditional commit.** A range commit replaces the predicate
   with the snapped bin edges as `ClauseFilter(range=(edge(lo_bin), edge(hi_bin+1)),
   closed="left")` — half-open so cube slice ≡ legacy `is_between(closed="left")` recompute ≡
   share/restore, bit-exact for counts; the stored selection box uses the same snapped edges.
   A categorical commit keeps the legacy `is_in` predicates **byte-for-byte** (no snapping, no
   `closed`) — only the POST is conditional. If **every** trace in **every** other figure was
   cube-served this gesture, the commit is local: selection state updates client-side and **no
   request is sent** (zero roundtrips for the whole gesture on a store hit). Otherwise the
   normal `selection` event POSTs and the client applies **all** returned deltas — including
   over cube-rendered traces, so any residual drift self-heals at commit.
4. **Abandoned gesture** (empty `plotly_selected` / `plotly_deselect` mid-drag): the saved
   pre-drag layer data of every touched target is restored (grouped parents restore their
   child-result lists); nothing is committed.

**Editing an existing box** (moving/resizing an activated selection) emits **no Plotly event
until mouseup** — the outline controllers rewrite the SVG silently, and only the drag-end
relayout re-emits `plotly_selected`. So `init.js` also binds a capture-phase `pointerdown`
(same live-brush gate): a drag starting on the selection outline or its edge handles arms a
rAF loop that converts the outline path's bbox back to data coordinates and replays each frame
through `handleSelecting` — the same gesture machine, so liveness gating, snapping, and the
conditional commit (via the re-emitted `plotly_selected`) apply unchanged, and without cube
coverage the edit stays mouseup-only. A safety timer aborts the gesture if the commit never
arrives. A **categorical (bar) source** has no range gesture — `handleSelecting` matches the
covered bars from `eventData.points`, which the bare outline range lacks — so the replay
resolves the covered bars itself (each rendered bar whose `axis.d2c(label)` falls inside the
outline span on the category axis) and feeds them as synthetic points; without this the
categorical gesture never engages during an edit-move and the target updates only on the
mouseup commit.

**Click commits** (pie slice, treemap node) run the same conditional commit without a gesture:
after the toggled predicate set is built, the client checks that the predicates are *servable*
(every clause an `is_in` test on the clicked trace's label/path columns; treemap path predicates
constrain a prefix of the columns and deeper levels match anything) and that every target is
capable and store-served — then slices locally, renders, and **skips the POST**. Any miss POSTs
exactly as today **and** fires one fire-and-forget `cube_request` (idempotent, content-cached)
so subsequent clicks are local. Toggle-to-empty always takes the legacy deselect path.

An unzoomed axis is keyed as `domain: null`; the snap domain then comes from the decoded cube
header's server-resolved free domain (remembered per (source, column, P) for later gestures).
When no domain is known at commit time (cold gesture whose request failed, or a dashboard with no
cube-capable target at all), the commit degrades to the legacy unsnapped closed-interval
predicate.

### Passive baking, target exclusion, overlay, temporal units (Phase 3)

**Passive baking.** A gesture's *passive set* is every committed `SelectionState` with a
non-`None` `source_figure_uid` different from the active figure and non-empty predicates
(`None`-uid selections never filter in the legacy engine; the active figure's own selection is
the re-brush case — both excluded). The server pre-filters the build frame with the compiled
passive predicates before `build_cube` and keys the blob with the **canonical passive key**
(`canonical_passive_key` in `predicates.py`; JS mirror `fvCubePassiveKey`). The two sides never
need to produce the same string — keying is asymmetric (the join key is the trace uid in
`trace_cubes`); each side only needs in-language determinism. The canonical form sorts clauses,
predicates and selections, keeps the nesting (AND across selections of OR-within-selection), and
carries no figure uids — so two sessions with the same snapped predicates share one cached
build. **Domain resolution stays on the unfiltered frame**: unzoomed domains are unfiltered
min/max, so bin edges are filter-stable (≡ the legacy `__hist_lo/hi__` semantics). That
equivalence includes the **sibling union**: coordinate-compatible same-figure histograms grouped
by `_histogram_domain_cols_by_uid` bin over the union of their columns' min/max in the legacy
path, so a cube target widens its binned dim the same way — otherwise the cube-served fg layer
renders on narrower, offset bins over the bg layer it overlays. Numeric and temporal columns,
and temporal columns with differing physical units, form separate groups. A sibling that is not
itself a cube target still contributes its min/max, so the domain pass resolves the group's
columns and not just the target columns; only a genuinely unresolvable column (empty/all-null)
makes the target unservable (fall back to recompute) rather than silently misaligned. Because
the shared domain is a property of the figure's histogram *set* and not of the trace alone, the
set is also part of the per-trace delta **cache key** (`content_key(domain_cols=...)`): without
it a solo-figure entry would be replayed for a sibling figure and re-introduce the misalignment.

**Passive is one global set per gesture** (verified legacy semantics — the engine applies one
filter list to every recomputed trace, with no per-target scoping), so all targets of a gesture
share one passive key and bar≡pie content-key sharing is unaffected. **Target exclusion:** a
trace is a cube target only if its figure is not the active figure *and owns no committed
selection* — legacy selection events never update selection-owning figures, and the cube path
mirrors that on both sides (`build_cubes` and `_fvCubeEnumerateTargets`).

The client computes the passive key **once per gesture** (selections cannot change mid-drag)
and threads it through every target key. The *lazy second selection* falls out: a new passive
set is a store miss ⇒ one `cube_request` (which already carries `state.selections` for the
bake) ⇒ live. Deselect/reset shrink the passive set ⇒ keys revert to the still-cached earlier
entries with zero requests. Pie/treemap click commits use the same passive-aware keys.

**Overlay interplay.** In `cross_filter_mode="overlay"` the live slice drives the `fg` layer;
the bg ghost is the *unfiltered* result. At gesture start, any cube-served target figure missing
its bg gets it materialized **locally** from the client response cache (the unfiltered init
payload applied to the bg layers only — zero round-trips; a cold cache degrades cleanly).
While a filtered foreground is active, `filtered_only` traces need no newly computed bg;
`overlay_style` is not serialized, so the client derives it from `trace_type` (+ `group_by`
for bar). Unfiltered init/reset/deselect responses still include every trace so a cleared
runtime cache and a dashboard opened directly in Overlay mode have a complete sole layer. The
conditional commit gains a bg conjunct: every cube-served target's figure must have its bg
established (or the trace be `filtered_only`), else the commit POSTs and the server's bg+fg
deltas self-heal. An abandoned gesture rolls back any bg state it created. A live gesture also
flips the renderer into the ghost+fg presentation before any selection exists
(`fvCubeOverlayFgActive`). `live_brush="off"` overlay stays bit-for-bit legacy.

**Temporal sources.** Temporal free axes (and temporal binned target dims) carry a physical
`unit` in the FVCube header — `us`/`ms` for `Datetime`, `day` for `Date`, derived from the
schema dtype by the engine; `Datetime("ns")` and `Time` gate to no cube (the string round-trip
is µs-precision). `unit:"day"` switches to an **integer-day snap grid** (`day_grid`: width
`w = max(1, ceil(span/2048))` whole days, `P' = ceil(span/w)` bins, header carries `w`/`p_eff`)
so `YYYY-MM-DD` edges round-trip bit-exactly; `us`/`ms` keep P=2048. Client side: temporal
viewports key as self-consistent epoch-ms tokens (never sent to the server — the server parses
the original date strings via the schema dtype in `_cube_axis_range`); the snap grid is adopted
from the decoded header; drag ranges convert through `fvTemporalToPhysical` (manual UTC parse,
never bare `Date.parse`); commits emit snapped `closed="left"` **string** ranges rendered by
`fvPhysicalToTemporal` — the edge is ceil-ed to an integral count of the unit, which preserves
integer membership of the half-open range and makes the string parse back exactly through
`_typed_range_bounds`. The `plotly_selected` echo guard converts both sides to physical before
its half-bin comparison.

**Temporal binned *target* dims.** A binned target dim over a temporal column is built on the
column's physical representation (epoch µs/ms, day index) and the header ships its `unit`. Most
temporal targets (hist, hist2d) render that axis as a Plotly **linear** axis — the server delta
*also* emits physical numbers, so the cube delta matches and no conversion is needed. A **line**
target is the exception: the server seeds its x as datetime → ISO strings, so Plotly makes it a
**date** axis (which reads bare numbers as epoch-ms). The cube line-envelope therefore maps each
bucket x from physical → epoch-ms via `fvPhysicalToEpochMs` (`fvLineEnvCells`); without it the raw
physical value — epoch µs is ~1000× an ms — lands millennia off-axis and the panel renders empty
mid-drag. No quantize is needed: the line is `postRequired`, so the commit POST's legacy delta
replaces the approximate envelope.

### The two cube caches (and the existing response cache)

| Cache | Where | Holds | Keyed by | Cleared by |
|---|---|---|---|---|
| cube-blob cache | server (`cache.py`, second `CacheBackend`, byte-bounded LRU, 512 MB default) | *encoded* FVCube blobs (re-encoding, not slicing, is the expensive part) | `cube_content_key(CubeSpec)` — content-addressed, cross-session shared | same invalidation hook as the delta cache: re-registering the source clears both |
| cube store | client (`runtime/cube.js`, byte-bounded LRU, 256 MB default) | *decoded* entries (TypedArray views + CSR offsets) | client-canonical descriptor JSON (pinned property order; unzoomed domains as `d: null` — a token, never a client-computed float; `p` slot = the gesture's canonical passive key, `null` for zero-passive) | full restore/import (`fvCacheReset` is wrapped); reset/deselect deliberately leave it untouched |

Both are **orthogonal** to the existing response cache: the response cache serves *unfiltered*
init/reset/deselect output; the cube store serves *filtered* slices during active brushing.
Reset and deselect stay 100% on the response-cache path, and zero-passive cubes stay valid across
them.

### Cube eligibility (Phases 1–4) and transport

- **Current support status**:

  | trace type | cube source? | cube target? | notes |
  |---|---:|---:|---|
  | `histogram` | yes | yes | Source is a 1-D range axis at `P=2048`. Target is a binned count; grouped histograms add categorical group dims. |
  | `box` | yes | no | Source is a 1-D range over the box data axis. Box is not a target because quantiles are not decomposable. |
  | `line` | yes | yes, limited | Source is an x-only range at `P=2048`. Target requires `downsample="minmax"` and numeric `y`; the target is a live-only `line_env`, and commit always POSTs. |
  | `histogram2d` | yes | yes, limited | Source is a `box2d` free axis at `128 x 128`. Target is full-data-only 2-D binned count/reduce; zoomed target axes fall back to the normal POST path. |
  | `bar` | yes | yes | Source is categorical labels. Target dims are label columns plus optional string `group_by`; supports `count`/`sum`/`mean`/`min`/`max`. |
  | `pie` | yes | yes | Source is categorical labels. Target is equivalent to an ungrouped bar with the same labels and measure. |
  | `treemap` | yes | yes | Source is the categorical full `path`. Target stores leaf path cells, and the client rolls them up into the hierarchy. |
  | `corr_heatmap` | no | yes, limited | Target only for Pearson correlation with `columns` explicitly passed. Omitting `columns` currently does **not** cube-accelerate, even though the legacy server path can infer columns from the schema. |
  | `geo_histogram2d` | no | no | Normal geo selection/update exists, but no cube source/target descriptors are exposed. |
  | `geo_line` | no | no | No cube source/target descriptors. |

- **Sources** (`get_cube_source_spec(axis_range, schema)`; the engine verifies the returned
  free axis matches `active_source.column`, mismatch ⇒ silent empty response):

  | trace | free axis |
  |---|---|
  | hist | 1-D range, P=2048 over the viewport (or server-resolved full) domain; continuous + temporal (`us`/`ms`/`day` physical units — see “Temporal sources” below; `Datetime("ns")`/`Time` gate to no cube) |
  | box | 1-D range over the `data_col` (same shape/gates as hist) |
  | line | 1-D range over the **x** column only (line selection is x-only — see below); P=2048; source geometry independent of `downsample` |
  | hist2d | **box2d**: two range axes (x, y) at P₂D=128 each, packed into one composite `free_bin` (`bin_y·(P₂D+1) + bin_x`); per-axis domains resolved by the engine; a rectangle brush slices a 2-D sub-grid |
  | bar / pie | categorical over the ordered label columns (`axis_range` ignored — label geometry is viewport-independent) |
  | treemap | categorical over the full `path` |

- **Targets** (`get_cube_target_spec(axis_range, schema)`; dim order is part of cube identity):

  | trace | target dims | measure |
  |---|---|---|
  | hist (ungrouped) | (binned data col) — display bin edges, `domain_hi + _HIST_BIN_EPSILON`, so a slice reproduces the Rust `fixed_hist` membership bit-exactly | count |
  | hist (grouped) | (binned data col, *group cols as categorical) | count |
  | bar | (*label cols, *group cols — all categorical) | `count`/`sum`/`mean`/`min`/`max` over `values` |
  | pie | (*label cols as categorical) | same |
  | line | (binned x col @ `n_points/2` buckets, *group cols as categorical) — **minmax-only** | `line_env` over `y` |
  | corr_heatmap | `()` — no grouping dims; the matrix cells are the explicit `columns` pairs; `columns` must be passed explicitly for cube support | `corr` (Pearson only) |
  | hist2d | (binned x col, binned y col) — both `bin_variant="hist2d"`, **bit-equal to the `fixed_hist2d` kernel** (the `+1e-10` span eps, not `fixed_hist`); **full-data only** (declines when either axis is zoomed) | count, or `histfunc` over `z` |
  | treemap | (*path cols as categorical) — the **leaf** level; the client finalizes leaf cells then **sums** them up every path level (parents = Σ of child finalized values, mirroring `_to_grouped_update`) | `count`/`sum`/`mean`/`min`/`max` over `values` |

  bar ≡ pie descriptor sharing falls out of content-key dedup: same labels + same measure ⇒ one
  blob, two `trace_cubes` entries. **Box is not a target** (quantiles are not decomposable) and
  CorrHeatmap is not a *source* (it emits no selection); it is a target only for explicit-column
  Pearson heatmaps (`columns=[...]`, not schema-inferred columns). The
  `line` (`line_env`) and `corr_heatmap` (`corr`) targets build from a **range
  (hist/box/line) OR categorical (bar/pie/treemap) source** — only a **box2d**
  (hist2d) source is excluded (the cell-count wall, #47): a categorical source
  partitions the build by the free category (`__free__` columns),
  reusing the `fixed_line_envelope2d` kernel with a degenerate 1-bin free axis (line) or
  `group_by(__free__cols)` (corr). The
  hist2d `histnorm` and the heatmap color scale are **client-side display** transforms applied
  per-slice — not part of cube identity, so two hist2ds differing only in `histnorm` share one
  cube. **The line envelope caveat:** the cube ships the *minmax-bucket* envelope so the drag is
  live every frame, but a line target's commit **always POSTs** (`postRequired`) so the legacy
  row-bucket delta replaces the approximate envelope — keeping commit ≡ share/restore bit-exact.
  This is the one target type where the cube slice and the committed delta deliberately differ.
- **Dtype/name gates** (descriptor methods return `None`; a schema is required for categorical
  capability): every categorical dim column must be `String`/`Categorical`/`Enum`, with one
  exception — a **bar/pie label** column may also be **integer- or float-typed**, on **both** the
  cube target *and* its matching categorical free axis (source). The codec preserves numeric
  categories as typed JSON numbers in numeric order, and categorical free keys keep the typed
  source values in the `__free__` columns instead of string-casting them. This avoids Python/JS
  formatting drift for floats such as `1.0` vs `1`, and committed `is_in` values cast back through
  `predicates._values_to_typed_series`. A trace's source and target gates therefore stay symmetric
  — bar/pie allow integer and float labels both ways (the demo's `hour_of_day` / `month` bars drive
  *and* receive a live brush), while treemap path columns stay string-only on both sides. **`group_by`**
  cols remain string-only because grouped-child identity is `json.dumps`-stringified server-side
  while the client reconciles children by renderer category value. A non-count measure's `value_col`
  must be numeric; a dim column named `count`, `sum`, `min`, `max` or `free_bin` would collide with
  the partial columns in the long-format frame and is refused. The client cannot see dtypes, so it
  gates only on what it can (agg, group shape) and relies on the demotion path above for the rest.
- **Gating:** cubes are only built/cached/served for `cache=True` sources (same "data is static
  for the process" contract as the delta cache). Committed selections from *other* figures are
  **baked in** as passive filters (Phase 3, below).
- **Transport:** FVCube v1 is a transport-independent binary codec — magic + version + JSON
  header (free block: range `{kind, p, domain}` or categorical `{kind, cols, categories}`;
  target-dim metadata with per-dim `categories` for categorical dims; measure; column manifest)
  + 8-byte-aligned little-endian typed-array buffers sorted by `free_bin` (u32 bins/codes/counts,
  f64 measure partials with null as NaN — except the `line_env` measure, which packs `(min_y,
  max_y)` as f32 and the extremal-x offsets as u16; categorical columns are dictionary-encoded
  codes into the header's sorted category lists). It rides raw (no base64) inside a thin binary
  **cube bundle** envelope (`encode_cube_bundle` / `decodeCubeBundle`) as the
  `application/octet-stream` body of the `/dashboard/update` cube response; the same bytes can
  later move to WebSocket binary frames. The cube path **gzip-compresses** the bundle itself at a
  fixed low level (binary numeric arrays barely compress, and high levels dominated cached-cube
  TTFB) and the client decompresses transparently. The codec stays version 1 across these
  extensions: blobs never outlive a process and client and server ship in lockstep.

### Invariants

- The server stores nothing per session; `request_cube` responses are reconstructable from
  `_sources` + request content.
- Traces stay renderer-agnostic and cube-agnostic beyond the two descriptor methods; no
  `trace_type` reaches `cube.py`.
- **`engine.build_cubes` has one trace-type-aware seam** (issue #72): it calls
  `_histogram_domain_cols_by_uid` to group shared bin domains, and that helper reads
  `trace_type` / `prop_key` / `data_col`. This is a **known, deliberate relaxation**, not an
  accident: the same grouping decides the legacy aggregation domains, and the two paths must
  agree exactly or a cube-served fg layer lands on different bin edges than the bg layer it
  overlays. Duplicating the rule per path is how they diverged in the first place, so it lives
  in one shared helper until shared-domain identity is expressible as a generic descriptor.
  Adding a trace type does **not** require touching it — only a trace with a *figure-shared*
  bin domain would. The helper gates shared domains by coordinate unit; its remaining known
  limitation is the latent secondary-axis anchor mismatch (#73), which matters only if
  secondary axes become renderer-supported.
- Runtime cube state (store, gesture machine, remembered free domains) is client-only and never
  serialized into the spec; the committed *selection* (snapped, `closed="left"`) is ordinary
  declarative state and round-trips through share/restore like any other predicate.

---

## Adapter Layer

`AbstractAdapter` in `adapters/base.py`. Shared client-side JS runtime in `adapters/runtime.py`.

```
AbstractAdapter (ABC)
├── parse_event(raw_event) → InteractionEvent | None    [abstract]
├── show_dashboard(spec, server_url, **kw) → None       [abstract]
│
├── show(spec, server_url, **kw)
│     wraps VisualizationSpec → one-figure DashboardSpec; calls show_dashboard()
│
├── _dashboard_markup(spec, render_panel)
│     shared Python-side dashboard shell builder for static + GridStack layouts
├── _toolbar_html(toolbar: ToolbarConfig | None) / _toolbar_css
│     shared header HTML/CSS; accepts optional ToolbarConfig to hide buttons
├── _mode_indicator_html(idx, lockable_axes)
│     renders per-figure Zoom/Pan/CF toggle + Reset + Lock Axes button
├── _deliver_notebook(html, height)
│     base64 iframe delivery for Jupyter
└── _deliver_browser_shared(spec, server_url, renderer)
      POST /share → open /view?spec=...&renderer=...
```

### JS Build Pipeline

JS source lives under `adapters/js/` as small modules.  `adapters/runtime.py` concatenates them
into the renderer bundles **at import time**, in pure Python — no Node and no build step.  The
bundle composition (which sources go into `shared` / `plotly` / `echarts`) is declared in
`runtime.py`; editing any file under `adapters/js/` takes effect on the next import.

```
adapters/js/
├── theme.css                 ← CSS custom properties (design tokens)
├── toolbar.js                ← shared toolbar hooks + state helpers
├── gridstack-bridge.js       ← GridStack.init() IIFE + change/resizestop handlers
├── panel.js                  ← <fv-panel> wrapper + shared panel-control binding
├── runtime/
│   ├── state.js              ← shared state init (layerDataByUid, etc.)
│   ├── cache.js              ← client response cache (init/reset/deselect)
│   ├── cube.js               ← cube store + FVCube decode + CSR slice + snap
│   ├── delta.js              ← postDashboardUpdate, applyDeltas
│   ├── overlay.js            ← overlay cache helpers
│   ├── selections.js         ← selection list/predicate helpers
│   ├── selection-summary.js  ← per-figure selection summary UI
│   └── hover.js              ← linked-hover dispatch
├── plotly/
│   ├── traces.js             ← configsByFig, trace template builders
│   ├── render.js             ← _fvRenderFigure, axis helpers, lock/capture
│   ├── events.js             ← relayout/selected/deselect/click handlers
│   ├── hover.js              ← Plotly crosshair helpers
│   └── init.js               ← Plotly.newPlot calls, event wiring, startup IIFE
└── echarts/
    ├── series.js             ← chartsByFig, series template builders
    ├── render.js             ← _fvRenderFigure, brush/zoom helpers
    └── init.js               ← echarts.init calls, event wiring, window._fvResizeChart
```

`runtime.py` assembles three bundles from these sources at import: `shared` (panel.js +
runtime/*.js + toolbar.js), `plotly` (plotly/*.js), and `echarts` (echarts/*.js); `theme.css`
and `gridstack-bridge.js` are served verbatim.

Each adapter's `<style>` block starts with `theme_css()` so all CSS custom properties
are available.  `_toolbar_css()` uses `var(--fv-*)` references throughout — no hardcoded
color or spacing values remain.

### Shared Runtime (`adapters/runtime.py`)

Both adapters embed `shared_runtime_js()` which provides renderer-agnostic logic.
`runtime.py` also exposes `theme_css()`, renderer-specific bundles, and
`gridstack_bridge_js()` for draggable dashboards.

Shared runtime responsibilities:

- **State init** — `layerDataByUid`, `groupedDataByParent`, `hasBgByFigure`, `bgYExtentByFig`, palette, linked-hover lookup tables (`colToFigAxis`, `traceSpecByUid`, `traceTypeByUid`)
- **Utilities** — `cloneObj`, `isGroupedParent`, `ensureLayerData`, `selectionSourceFigureUids`, `figureHasSelectionSource`, `rendererUid`, `figureViewportRanges` (`rendererUid` now emits CSS-safe `__fv_layer_bg` / `__fv_layer_fg`)
- **Layer data** — `ensureGroupColor`, `setLayerData`, `setGroupedLayerData`
- **Delta application** — `postDashboardUpdate` separates request, delta-apply, and render error handling; updates layer caches, tracks bg y-extent for axis anchoring, and renders each dirty figure with per-figure guards
- **Overlay restore** — `fvEnsureOverlayBackground`, `fvResetRuntimeCache`, `fvRestoreFromSpec`
- **Linked hover** — `stripLayerSuffix` (handles new suffixes and legacy `::bg` / `::fg`), `getHoverPayload`, `_fvIsHoverEnabled`, `dispatchCrosshairs`

Each adapter defines four renderer-specific hooks:
- `_fvRenderFigure(figUid)` — re-render one figure
- `_fvShowCrosshair(figUid, axis, value)` — draw a crosshair line
- `_fvClearAllCrosshairs()` — clear hover guides from all figures
- `_fvResetRendererCache()` — reset adapter-specific caches

### GridStack dashboard layout

When `DashboardSpec.layout.draggable=True`, both adapters render figures as GridStack items.

- `GridStack.init(...)` runs from `gridstack-bridge.js`, which adapters include only when `layout.draggable=True`; it executes before the renderer-specific bundle so each chart measures a non-zero container size.
- Initial `layout.grid_items` is either provided by the spec or auto-generated from `Dashboard.show(rows=..., cols=...)` seed inputs (default: auto 2-column) before render.
- Frontend `change` events update `DASHBOARD_SPEC.layout.grid_items` after each drag/resize without calling `/dashboard/update`.
- Toolbar button `#fv-btn-grid` toggles `DASHBOARD_SPEC.layout.grid_editable` at runtime and calls `window.fvSetGridEditable(...)` to enable/disable GridStack move+resize in place.
- `layout.draggable` remains the declarative switch for GridStack rendering; `layout.grid_editable` controls only editability of that grid.

### PlotlyAdapter

- Self-contained HTML + Plotly.js 3 from CDN; no Python Plotly dependency.
- One `<div>` per figure; grouped parents are logical-only and are not bootstrapped
  as renderer traces.
- Keeps per-figure base traces plus `groupedChildrenByParent`; rebuilds the figure's
  trace array in backend order before each `Plotly.react(...)`.
- Loads initial data via `POST /dashboard/update` with `init` event.
- Updates via `Plotly.react(...)`.
- Supported trace types: **line**, **histogram**, **box**, **bar**, **pie**, **treemap**, **heatmap** (histogram2d / corr_heatmap).
- **Per-figure mode toggle** (Zoom | Pan | CF) rendered at top-right of each figure; Plotly modebar hidden entirely (`displayModeBar: false`). Zoom and Pan buttons stay enabled for cartesian and Plotly map figures (geo histogram / geo line) because map `dragmode` drives viewport relayouts; they are disabled for non-navigable figures such as pie, treemap, and corr_heatmap, where CF is active by default. CF sets `dragmode: 'select'` for range-based cross-filter drag. Configurable per-figure default mode is a future TODO.
- Events: `plotly_relayout` → viewport (including figure-scoped viewport reset that preserves selections) · `plotly_selected` → selection · `plotly_deselect` → deselect · `plotly_click` → categorical cross-filter (pie and treemap).
- **Click-based cross-filtering** (`plotly_click`): for pie traces, the clicked slice label is decoded against the source trace's `labels` columns and emitted as one `ClauseFilter` per label column.  For treemap, the clicked node's path becomes one `ClauseFilter` per `path` column from leaf depth to root.  Clicking an already-selected node (matched structurally by `_selectionMatches`) deselects.  TreeMap drill-in is suppressed by resetting `trace.level` on each click.
- **Predicate-based selection wire format**: `handleClick` and `handleSelected` build `SelectionPredicate` objects directly from each clicked node, brush rectangle, or geo polygon, reading the source trace's `backend_data` to populate column names.  The shared runtime (`runtime.py`) helpers (`selectionSourceFigureUids`, `figureHasSelectionSource`) inspect predicates rather than legacy field names.
- Geo viewport relayout uses Plotly map `._derived.coordinates` when available and stores it in shared spec state as `state.viewport["{figure_uid}/coordinates"] = [[lon, lat], ...]`.
- Geo selection on `choroplethmap` traces is not cartesian `eventData.range`; the adapter derives one lon/lat bounding box from the selected polygon ids and emits an ordinary rectangular `SelectionState`.
- `_programmaticOp` guard prevents duplicate deselect/relayout events triggered by programmatic `Plotly.react` calls (toolbar reset/deselect). `handleSelected` additionally checks `_selectionMatches` + `_selectionBoxMatches` to suppress the spurious `plotly_selected` re-fire that Plotly emits after a `Plotly.react` re-render.
- `plotly_deselect` calls `clearFigureSelection(figUid)`, which removes only that figure's entry from `DASHBOARD_SPEC.state.selections` rather than wiping all selections across every figure.
- Range-brush selections on figures with multiple traces that map to different (x-col, y-col) pairs emit one `SelectionPredicate` per unique pair (OR-combined), reflecting that the engine ANDs predicates across figures but ORs them within a single `SelectionState`.
- Toolbar reset override clears viewport/selections, resets runtime overlay cache, and posts an `init` event.
- Heatmap traces consume explicit `display.color_scale` / `display.color_range`; the adapter runtime validates those fields instead of inferring defaults client-side.
- Group colors are assigned from `display.color_map` first, then from
  `state.group_domains[group_domain_key]`.
- Overlay mode: switches `barmode` to `"overlay"` when fg is visible; pairs bg/fg bar layers with logical `offsetgroup` values only in that state so base stacked bars keep Plotly's default stacking behavior; pins y-axis to bg extent via `bgYExtentByFig`.
- Linked hover: shapes tagged with `_fvHover` prefix; `renderHoverGuides` replaces shapes by tag prefix to prevent accumulation.

### EChartsAdapter

DEPRECATED => currently not maintained anymore.

- Self-contained HTML + ECharts 5 from jsDelivr CDN.
- One chart instance per figure; grouped parents are logical-only and are not
  bootstrapped as renderer series.
- Keeps per-figure base series plus `groupedChildrenByParent`; rebuilds the figure's
  series array and applies it with `replaceMerge: ['series']`.
- Updates via `chart.setOption(...)`.
- Supported trace types: **line**, **histogram**, **bar**, **pie**, **heatmap** (histogram2d / corr_heatmap).
- Events: `datazoom` → viewport · `brushEnd` → selection · empty brush → deselect.
- `_applyingDeltas` guard prevents `datazoom` feedback loops from programmatic updates.
- Toolbar reset override clears viewport/selections, resets runtime overlay cache, and posts an `init` event.
- Heatmap traces consume explicit `display.color_scale` / `display.color_range`; the adapter maps supported scale names to local colors and uses one figure-level `visualMap` for heatmap-like traces.
- Group colors are assigned from `display.color_map` first, then from
  `state.group_domains[group_domain_key]`.
- Overlay mode: pins y-axis to bg extent via `bgYExtentByFig`.
- Linked hover: dispatches `updateAxisPointer` actions to linked chart instances.

---

## Directory Structure

```
flexviz/
├── flexviz/
│   ├── __init__.py          ← public API: Figure, Dashboard, app,
│   │                           register_source, mount_into
│   ├── spec.py              ← VisualizationSpec, DashboardSpec, FigureSpec,
│   │                           TraceSpec, LayoutSpec, ToolbarConfig,
│   │                           InteractionState, SelectionState,
│   │                           encode_spec, decode_spec,
│   │                           TraceDisplay, TraceParams (TypedDict hints)
│   ├── events.py            ← InteractionEvent, ActiveSource, TraceDelta,
│   │                           GroupedChildDelta
│   ├── engine.py            ← FlexEngine, TraceInfo
│   ├── predicates.py        ← predicates_to_expr (selection → Polars filter)
│   ├── cube.py              ← CubeSpec, build_cube, cube_content_key,
│   │                           FVCube v1 codec
│   ├── cache.py             ← CacheBackend, delta cache + cube-blob cache
│   ├── LF.py                ← LFQueryBuilder, AggregationSpec,
│   │                           GroupedAggregationSpec
│   ├── server.py            ← FastAPI app, register_source, mount_into
│   ├── figure.py            ← Figure
│   ├── dashboard.py         ← Dashboard
│   ├── trace/
│   │   ├── __init__.py      ← build_trace_from_spec(), _REGISTRY
│   │   ├── base.py          ← FlexTrace ABC + dtype filter helpers
│   │   ├── line.py          ← LinePlot
│   │   ├── hist.py          ← Histogram
│   │   ├── box.py           ← BoxPlot
│   │   ├── bar.py           ← BarPlot
│   │   ├── pie.py           ← PiePlot
│   │   ├── treemap.py       ← TreeMap
│   │   ├── hist2d.py        ← Histogram2D
│   │   ├── geo_hist2d.py    ← GeoHistogram2D
│   │   ├── _hist_helpers.py   ← shared 2D binning helpers (Histogram2D + GeoHistogram2D)
│   │   └── corr_heatmap.py  ← CorrHeatmap
│   └── adapters/
│       ├── __init__.py
│       ├── base.py          ← AbstractAdapter, shared toolbar + delivery
│       ├── runtime.py       ← shared_runtime_js() — renderer-agnostic JS
│       ├── plotly_adapter.py
│       └── echarts_adapter.py
├── flexviz_polars/          ← Rust/Polars plugin (build with make build-plugin)
│   ├── src/
│   │   ├── lib.rs           ← PyO3 module entry, global Polars allocator
│   │   └── expressions.rs   ← every_nth, arg_min_max kernels + shared helpers
│   ├── flexviz_polars/
│   │   ├── __init__.py      ← FlexvizExprNamespace (@pl.api.register_expr_namespace)
│   │   ├── _internal.pyi    ← type stub for compiled extension (__version__)
│   │   └── typing.py        ← IntoExprColumn type alias
│   ├── tests/
│   │   └── test_plugin_functions.py ← unit tests for every_nth and arg_min_max
│   ├── Cargo.toml
│   └── pyproject.toml
└── tests/
    ├── conftest.py          ← shared fixtures (DataFrames, traces, API client)
    ├── test_spec.py
    ├── test_lf.py
    ├── test_engine.py
    ├── test_trace_base.py
    ├── test_trace_line.py
    ├── test_trace_hist.py
    ├── test_trace_box.py
    ├── test_trace_pie.py
    ├── test_trace_treemap.py
    ├── test_trace_hist2d.py
    ├── test_trace_geo_hist2d.py
    ├── test_trace_corr_heatmap.py
    ├── test_adapters.py
    ├── test_html_adapters.py
    ├── test_integration.py
    ├── test_predicates.py
    ├── test_cube.py         ← cube build / codec / descriptor unit tests
    ├── test_cube_server.py  ← request_cube server/engine path
    ├── test_cache.py
    ├── test_browser.py      ← Playwright (make test-browser)
    └── test_browser_cube.py ← Playwright live-brush / cube gesture tests
```

---

## Design Notes

**uid-based trace identity**
All logical parent traces are uid-based end-to-end. Grouped child traces derive stable
child uids from `(parent_uid, group_value)` and include a short hash suffix so renderer
reconciliation does not collide on sanitized labels.

**Renderer-agnostic trace definitions**
`FlexTrace` subclasses contain zero renderer imports. Renderer-specific display hints live in `_display`; backend-only config in `_params`. The adapter interprets `display` during bootstrap; `params` is used server-side only.

**`TraceResult` as the engine handoff**
`_to_update` and `_to_grouped_update` return `TraceResult` (defined in `trace/base.py`)
instead of raw dicts. The engine owns the single `pl.Series → list` normalization step.
`TraceResult.updates` may hold either `pl.Series` or Python lists; traces keep Series
where natural and do not call `.to_list()` themselves. Grouped parents use
`TraceResult.group_results`, which is translated to `GroupedChildDelta` on the wire.

**`axes` are used by the engine, not by traces**
Every trace stores `_axes` (e.g. `("x", "y")`) or `None` for non-cartesian traces. `_axes` is a *render-anchor* set and is decoupled from re-aggregation: the engine routes cross-filter selections and matches viewport events using `TraceInfo.axes`, but gates per-trace recompute on the narrower `recompute_axes` (the data-binding subset, or `("coordinates",)` for maps). Cross-filter selections for `axes is None` traces are keyed by `figure_uid` only. `_axes` must stay for secondary-axis support on cartesian traces.

**Stateless engine**  
`FlexEngine` is constructed fresh per request; it stores only references to `_backend_lf` and `_scalable_traces`. Thread-safe; trivially horizontally scalable.

**Cross-filtering mechanics**  
Filter expressions from source traces are applied lazily to target traces before aggregation. Schema is passed through so selection bounds are typed to the backend column dtype — critical for temporal axes. No data is copied.

**Client-authoritative state**  
The client sends the full `VisualizationSpec` / `DashboardSpec` with every request. Dashboard viewport keys are stored client-side as `"figure_uid/axis_id"`; geo map viewports use `"figure_uid/coordinates"` and store raw corner points. The server never holds viewport or selection state between requests. Because of that, an engine-only change was insufficient for GeoHistogram2D: Plotly geo zoom and selection only worked once the browser emitted spec-valid viewport / selection payloads.

**Grouped parents are logical-only**
Grouped parents exist in the spec and engine, but adapters do not materialize them as
renderer traces. Only current child series are rendered, in the exact order supplied by
the backend.

---

## Known Issues / Technical Debt

EChartsAdapter is currently deprecated. No goal to support this in the near future.

| Location | Description |
|----------|-------------|
| `LF.py` `check_sorted` | `collect()` to verify sort order — O(n); `assume_sorted` available as opt-in |
| `row_index_col` | Designed for linked-hover via `customdata`, which is not implemented. Automatic `with_row_index()` was removed; `row_index_col` remains in `LFQueryBuilder` for future use but is not consumed by traces or the engine. |
| `adapters/base.py` | `_deliver_browser` always routes through `POST /share` and opens an encoded URL, even for an ordinary local `fig.show()` call |
| `EChartsAdapter` | Does not support `BoxPlot` or `TreeMap`; supports line, histogram, bar, pie, and heatmap |
| `trace/` interface | `_backend_data`, `_display`, `_params` dicts have `TypedDict` hints (`TraceDisplay`, `TraceParams`); `backend_data` values are `str | list[str]` |
| `LinePlot`, `Histogram`, `Histogram2D`, `GeoHistogram2D` | All require `flexviz_polars` plugin; raise `ImportError` at import time without it (no pure-Python fallback). |
| Cube line-envelope kernel (issue #36) | The future line-as-target cube shape needs exact argmin/argmax-by-y partials; pure Polars pays ~4–5× for the exact args. A one-pass Rust `fixed_line_envelope2d` kernel is planned before line targets land (Phase 4). |
| Cube client temporal snapping | The server builds temporal-free-axis cubes (physical epoch cast), but the client gesture machine bails on non-numeric viewport ranges, so temporal source axes currently degrade to mouseup-only behavior. |

### Roadmap

- **`median` / `n_unique` reducers for `Histogram`, `Histogram2D`, and `GeoHistogram2D`.** When these traces moved to the `flexviz_polars` Rust kernel they dropped `median` and `n_unique`, which the kernel does not implement (count + `sum`/`mean`/`min`/`max` only). Both break the kernel's fixed-memory, single-pass design — `median` needs per-bin value retention, `n_unique` needs a per-bin set. The intended path is to add `FixedHist2DReducer::Median` / `NUnique` (and a 1D equivalent for `Histogram`) so all three traces regain uniform, fast support; the prior pure-Polars `bin_2d` pipeline in `trace/_hist_helpers.py` is the reference implementation for the expected semantics. Until then, requesting `histfunc="median"` or `"n_unique"` on these traces raises `ValueError`.

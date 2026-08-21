"""Renderer-agnostic visualization spec.

The VisualizationSpec is a pure-data tree that is fully JSON-serializable.
It captures the *declarative* state of a figure (layout, trace config, viewport,
selections) without holding any runtime objects (LazyFrames, Polars expressions,
callables).  Runtime bindings live separately in the engine layer.

Pydantic dataclasses are used so that the same types serve both as the
internal engine model *and* as validated FastAPI request/response bodies —
no duplication, no translation layer.
"""

from __future__ import annotations

import base64
import gzip
import json as _json
from typing import Any, Dict, List, Literal, Tuple, TypedDict, TypeAlias, Union
from uuid import uuid4

from pydantic import (
    BaseModel,
    Field,
    field_serializer,
    model_validator,
)

_SPEC_VERSION = "0.4"

# Per-trace hover *capabilities* (what a trace can emit/receive as a source or
# target). These are declared by each trace and consumed by the client runtime,
# which auto-selects the projection from the hovered source.
HoverMode = Literal["off", "axis", "cell"]

# Client-facing hover toggle. Linked hover is a single on/off control; the
# runtime decides the projection per hovered trace (point/axis vs cell).
HoverToggle = Literal["off", "on"]


# ---------------------------------------------------------------------------
# TypedDict hints for TraceSpec.display and TraceSpec.params.
#
# These are documentation-only at present: the wire format remains
# Dict[str, Any] so that existing specs round-trip without validation errors.
# Over time, trace implementations can narrow their annotations.
# ---------------------------------------------------------------------------


class TraceDisplay(TypedDict, total=False):
    name: str
    color: str
    bar_mode: Literal["group", "stack"]
    color_map: Dict[str, str]
    color_scale: str
    color_range: Tuple[float, float] | Literal["auto"]


class TraceParams(TypedDict, total=False):
    agg: str
    orientation: str
    group_by: str | list[str]
    group_domain_key: str
    group_value: str
    bins: int
    histnorm: str
    n_points: int
    downsample: str
    add_gaps: bool
    hole: float
    x_bins: int
    y_bins: int
    method: str
    columns: list[str]
    absolute: bool
    triangular: bool


BackendDataValue: TypeAlias = str | list[str]


class AxisRange(BaseModel):
    """A min/max range for a single axis."""

    min: float | str
    max: float | str

    def as_tuple(self) -> Tuple[float, float] | Tuple[str, str]:
        return (self.min, self.max)


GeoViewportCoordinates: TypeAlias = List[Tuple[float, float]]
ViewportStateValue: TypeAlias = AxisRange | GeoViewportCoordinates | None


class ClauseFilter(BaseModel):
    """One column filter. Exactly one of range/values is set.

    - ``range``: continuous `is_between(lo, hi)` filter (numeric, temporal).
    - ``values``: categorical `is_in([...])` filter.
    - ``closed``: range endpoint inclusivity. ``"left"`` is the half-open
      interval `[lo, hi)` emitted by cube-snapped commits; ``"both"``
      (default) is the closed interval. Only valid with ``range``.
    """

    column: str
    range: Tuple[Any, Any] | None = None
    values: List[Any] | None = None
    closed: Literal["both", "left"] = "both"

    @model_validator(mode="after")
    def _exactly_one_field(self) -> "ClauseFilter":
        has_range = self.range is not None
        has_values = self.values is not None
        if has_range == has_values:
            raise ValueError("ClauseFilter requires exactly one of `range` or `values`")
        if has_values and self.closed != "both":
            raise ValueError("ClauseFilter `closed` is only valid with `range`")
        return self


class SelectionPredicate(BaseModel):
    """One OR-disjunct. Clauses inside are ANDed together."""

    clauses: List[ClauseFilter] = Field(default_factory=list)


class SelectionState(BaseModel):
    """A per-figure cross-filter selection expressed as a logical predicate.

    Semantics:
        ClauseFilter        = one column test
        SelectionPredicate  = AND(clauses)
        SelectionState      = OR(predicates)
        Engine filter       = AND(selection_states from other figures)
    """

    source_figure_uid: str | None = None
    predicates: List[SelectionPredicate] = Field(default_factory=list)


class TraceHoverSpec(BaseModel):
    """Hover interaction capabilities for a single trace.

    ``source_modes`` lists the hover modes this trace can emit as a source.
    ``target_modes`` lists the hover modes this trace can receive as a target.
    Modes declared here may exceed what is currently implemented; the JS
    runtime intersects with ``IMPLEMENTED_HOVER_MODES`` gates before dispatch.
    """

    source_modes: List[HoverMode] = Field(default_factory=list)
    target_modes: List[HoverMode] = Field(default_factory=list)


SelectionKind = Literal["range", "categorical", "path", "geo_box", "none"]


class TraceSelectionSpec(BaseModel):
    """Declarative cross-filter selection geometry for a single trace.

    Mirrors ``TraceHoverSpec``: the trace declares *how* a gesture becomes
    ``SelectionPredicate`` clauses, and a generic client runtime interprets it —
    so the selection logic never branches on ``trace_type``.  This is also the
    source geometry the cube pre-aggregation indexes on.

    ``kind``
        - ``"range"``       — a box-select on cartesian axes (line/hist/box/hist2d).
                              ``axis_columns`` maps each *selectable* anchor to its
                              column; a clause is a range on that column.
        - ``"categorical"`` — label equality from a click or a box-drag over
                              categories (bar/pie).  ``label_columns`` are the
                              source columns; a composite (multi-column) label is
                              the JSON-encoded tuple, decomposed one clause per
                              column.
        - ``"path"``        — a hierarchical drill-down click (treemap).  The hit
                              id is split on ``path_separator`` into one clause
                              per ``path_columns`` level.
        - ``"geo_box"``     — a map box/lasso; the hit feature's bounding box
                              becomes ``lon``/``lat`` range clauses.
        - ``"none"``        — not a cross-filter source (e.g. corr_heatmap).

    ``multi`` is the client-side accumulation policy across successive gestures:
    ``"replace"`` (overwrite), ``"or"`` (toggle-able OR of predicates), or
    ``"path"`` (OR with ancestor/descendant replacement along a hierarchy).
    """

    kind: SelectionKind = "none"
    axis_columns: Dict[str, str] = Field(default_factory=dict)
    label_columns: List[str] = Field(default_factory=list)
    path_columns: List[str] = Field(default_factory=list)
    path_separator: str = "/"
    lon_column: str | None = None
    lat_column: str | None = None
    multi: Literal["replace", "or", "path"] = "replace"


class TraceSpec(BaseModel):
    """Serializable description of a single trace.

    ``backend_data`` maps trace-property names to column-name strings or lists
    of column-name strings when the trace references the figure's shared
    backend LazyFrame, or is empty / absent for in-memory (per-trace Series)
    data.

    ``params`` holds backend-only configuration that is not a column mapping
    and not a renderer hint — e.g. ``n_points`` for ``LinePlot``.

    ``display`` holds renderer hints interpreted by adapters — e.g. ``name``,
    ``color``, ``line_width``.
    """

    uid: str
    trace_type: str
    # (x_anchor, y_anchor) for cartesian axes;
    # empty tuple or None for traces with no axis binding.
    axes: Tuple[str, ...] | None = None
    backend_data: Dict[str, BackendDataValue] = Field(default_factory=dict)
    params: Dict[str, Any] = Field(default_factory=dict)
    display: Dict[str, Any] = Field(default_factory=dict)
    # Anchor ids whose viewport range parameterizes this trace's aggregation
    # (e.g. ``("x",)`` for a line, ``("coordinates",)`` for a map trace).
    # Read by both the client (to suppress no-op /update POSTs) and the engine
    # (to gate per-trace recompute).  Current producers must emit a concrete
    # tuple: populated means "these viewport anchors re-aggregate the trace",
    # while an explicit empty tuple means deliberately frozen.  ``None`` is only
    # the Pydantic default for hand-built/imported specs that omit the field:
    # server-side trace reconstruction may derive the default policy from it,
    # but the browser does not repair it and treats null/omitted as no
    # recompute axes.
    recompute_axes: Tuple[str, ...] | None = None
    # Declarative cross-filter selection geometry — how a gesture on this trace
    # becomes predicate clauses.  Read by the generic client runtime (no
    # per-``trace_type`` branching) and the source of the figure's Plotly
    # ``selectdirection``.  Replaces the removed ``TraceSpec.select_axes`` *wire
    # field*: a range trace's selectable anchors are ``selection.axis_columns``
    # keys.  (This is distinct from the ``FlexTrace.select_axes`` runtime
    # property, which the engine reads off the live trace; both that property and
    # this field are derived views of the same ``_default_select_axes()`` source,
    # so they cannot disagree.)
    selection: TraceSelectionSpec = Field(default_factory=TraceSelectionSpec)
    hover: TraceHoverSpec = Field(default_factory=TraceHoverSpec)


class FigureSpec(BaseModel):
    """Static figure configuration (layout + traces).

    ``source`` is the name of a data source registered on the server via
    ``register_source()``.  It is ``None`` for figures whose traces are
    entirely in-memory (no shared backend LazyFrame).

    ``uid`` is a stable identifier assigned by ``Figure.__init__`` and
    carried through every ``to_spec()`` call. Used by the dashboard layer
    to route trace deltas back to the correct renderer div.
    """

    uid: str = Field(default_factory=lambda: str(uuid4()))
    source: str | None = None
    layout: Dict[str, Any] = Field(default_factory=dict)
    traces: List[TraceSpec] = Field(default_factory=list)


class GroupDomainState(BaseModel):
    """Client-owned group→color mapping for one group_by domain.

    The domain key is ``"{source_or_figure_uid}::{group_by_col}"``.
    ``mapping`` maps each group value (as a string) to a CSS colour string.
    ``next_color_index`` tracks which palette index to assign next.
    """

    mapping: Dict[str, str] = Field(default_factory=dict)
    next_color_index: int = 0


class InteractionState(BaseModel):
    """Mutable interaction state (viewport ranges + selections).

    The client owns this state and echoes it back with every request so that
    the server remains fully stateless.
    """

    viewport: Dict[str, ViewportStateValue] = Field(default_factory=dict)
    selections: List[SelectionState] = Field(default_factory=list)
    group_domains: Dict[str, GroupDomainState] = Field(default_factory=dict)
    cross_filter_mode: Literal["update", "overlay"] = "update"

    @field_serializer("viewport")
    def _serialize_viewport(
        self, viewport: Dict[str, ViewportStateValue]
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, value in viewport.items():
            if value is None or isinstance(value, AxisRange):
                out[key] = value
            else:
                out[key] = [[point[0], point[1]] for point in value]
        return out


class ClientState(BaseModel):
    """Client-only persistent state not read by the server engine.

    Included in every POST body to ``/dashboard/update`` and in
    share/export/import serialisation, but the server silently ignores it.

    This is the designated home for client-only-but-persistent state:
    hover mode, annotation visibility, panel collapse state, axis locks, etc.

    ``axis_locks`` maps ``"{figure_uid}/{axis_family}"`` → locked flag and
    ``axis_lock_ranges`` maps the same keys → the pinned ``AxisRange``. Both are
    applied entirely client-side (JS pins the viewport); the server never reads
    them.

    ``live_brush`` gates the cube live-brush loop (spec §2.5): ``"auto"``
    (default) binds ``plotly_selecting`` on range-geometry figures and slices
    client-side cubes during the drag; ``"off"`` never binds it — today's
    mouseup-only behavior bit-for-bit. The server never reads this field.
    """

    hover_mode: HoverToggle = "off"
    live_brush: Literal["auto", "off"] = "auto"
    axis_locks: Dict[str, bool] = Field(default_factory=dict)
    axis_lock_ranges: Dict[str, AxisRange] = Field(default_factory=dict)


class VisualizationSpec(BaseModel):
    """Root spec object — fully JSON-serializable snapshot of a visualisation.

    Sent from client → server on every interaction request.  The server never
    stores it; all mutable state lives in ``state`` and is round-tripped by
    the client.
    """

    version: str = _SPEC_VERSION
    figure: FigureSpec = Field(default_factory=FigureSpec)
    state: InteractionState = Field(default_factory=InteractionState)


_GRIDSTACK_CELL_HEIGHT_PX: int = 80
"""Pixels per Gridstack row unit.  h=5 → 400 px (matches the adapter default)."""


class GridItem(BaseModel):
    """Gridstack position and size for one figure in a draggable layout.

    Coordinates use the Gridstack 12-column grid system.
    ``fig_uid`` must match a ``FigureSpec.uid`` in the same ``DashboardSpec``.
    """

    fig_uid: str
    x: int = 0  # column (0–11)
    y: int = 0  # row (0-based Gridstack row units)
    w: int = 6  # width in columns (1–12)
    h: int = 5  # height in row units (h × _GRIDSTACK_CELL_HEIGHT_PX = pixel height)


def _auto_grid_items(
    figures: "List[FigureSpec]",
    rows: int | None = None,
    cols: int | None = None,
) -> "list[GridItem]":
    """Generate ``GridItem`` positions from row/column seed hints.

    Used when ``LayoutSpec.grid_items`` is ``None``.
    """
    import math

    if rows is not None and cols is not None:
        raise ValueError("rows and cols are mutually exclusive")
    if rows is not None and rows < 1:
        raise ValueError("rows must be >= 1")
    if cols is not None and cols < 1:
        raise ValueError("cols must be >= 1")

    n = len(figures)
    if n == 0:
        return []

    resolved_cols = cols
    if resolved_cols is None:
        if rows is None:
            resolved_cols = 2
        else:
            resolved_cols = max(1, math.ceil(n / rows))

    if resolved_cols > 12:
        raise ValueError("cols must be <= 12")

    base = 12 // resolved_cols
    rem = 12 % resolved_cols
    widths = [base + (1 if i < rem else 0) for i in range(resolved_cols)]
    x_offsets: list[int] = []
    x = 0
    for w in widths:
        x_offsets.append(x)
        x += w

    item_h = 5  # 400 px / 80 px = 5 units
    items: list[GridItem] = []
    for i, fig in enumerate(figures):
        col_idx = i % resolved_cols
        row_idx = i // resolved_cols
        items.append(
            GridItem(
                fig_uid=fig.uid,
                x=x_offsets[col_idx],
                y=row_idx * item_h,
                w=widths[col_idx],
                h=item_h,
            )
        )
    return items


class ToolbarConfig(BaseModel):
    """Controls which buttons appear in the shared toolbar.

    All buttons are shown by default. Set a field to ``False`` to hide it.
    Empty button groups are omitted automatically.
    """

    show_reset: bool = True
    show_deselect: bool = True
    show_cfmode: bool = True
    show_hover: bool = True
    show_lock_all_axes: bool = True
    show_grid: bool = True
    show_share: bool = True
    show_export: bool = True
    show_import: bool = True


class LayoutSpec(BaseModel):
    """HTML/CSS layout hints for a multi-figure dashboard.

    ``gap``
        CSS gap between figures (default ``"8px"``).

    ``draggable``
        When ``True``, the dashboard renders using Gridstack.js.
        Drag/resize interactions are enabled by ``grid_editable``.
        Position changes update ``grid_items`` in client-side state only —
        no backend request is fired.

    ``grid_editable``
        Initial Gridstack edit mode when ``draggable=True``.
        ``True`` enables drag/resize handles; ``False`` locks the layout.
        The toolbar lock button updates this value client-side.

    ``grid_items``
        Per-figure Gridstack positions.  ``None`` causes positions to be
        auto-generated at render time.
        Updated in-place by the frontend after each drag/resize.

    ``toolbar``
        Controls which toolbar buttons are rendered.
    """

    gap: str = "8px"
    draggable: bool = True
    grid_editable: bool = False
    grid_items: List[GridItem] | None = None
    toolbar: ToolbarConfig = Field(default_factory=ToolbarConfig)


class DashboardSpec(BaseModel):
    """Top-level spec for a multi-figure dashboard.

    Peer of ``VisualizationSpec``: an entire dashboard round-trips as one
    ``DashboardSpec`` so the server remains stateless even when figures share
    interaction state.

    ``figures``
        Ordered list of ``FigureSpec`` objects, one per chart.  Each carries
        its own stable ``uid``, data source, and trace list.

    ``state``
        Shared ``InteractionState`` — single source of truth for viewport and
        selections across all figures.  The client echoes it back on every
        ``/dashboard/update`` request.

    ``layout``
        Layout hints (gap, draggable/grid editability, and ``grid_items``).
    """

    version: str = _SPEC_VERSION
    figures: List[FigureSpec] = Field(default_factory=list)
    state: InteractionState = Field(default_factory=InteractionState)
    layout: LayoutSpec = Field(default_factory=LayoutSpec)
    client_state: ClientState = Field(default_factory=ClientState)


# ---------------------------------------------------------------------------
# Spec encoding helpers (for shareable URLs)
# ---------------------------------------------------------------------------


def encode_spec(spec: "Union[VisualizationSpec, DashboardSpec, dict]") -> str:
    """Encode a spec to a compact, URL-safe string.

    The spec is JSON-serialised, gzip-compressed (level 9), and
    base64url-encoded without ``=`` padding so the result is safe in a URL
    query parameter.

    Parameters
    ----------
    spec:
        A ``VisualizationSpec``, ``DashboardSpec``, or any JSON-serialisable
        ``dict``.

    Returns
    -------
    str
        URL-safe encoded string.
    """
    if isinstance(spec, (VisualizationSpec, DashboardSpec)):
        raw: bytes = spec.model_dump_json().encode()
    elif isinstance(spec, dict):
        raw = _json.dumps(spec).encode()
    else:
        raise TypeError(f"encode_spec: unsupported type {type(spec)!r}")
    compressed = gzip.compress(raw, compresslevel=9)
    return base64.urlsafe_b64encode(compressed).rstrip(b"=").decode()


def decode_spec(encoded: str) -> "Union[VisualizationSpec, DashboardSpec]":
    """Decode a string produced by :func:`encode_spec` into a spec model.

    The spec type is detected from the decoded JSON: a dict containing the
    key ``"figures"`` is treated as a ``DashboardSpec``; otherwise a
    ``VisualizationSpec`` is returned.

    Parameters
    ----------
    encoded:
        URL-safe base64-encoded gzip-compressed JSON string.

    Returns
    -------
    VisualizationSpec | DashboardSpec
    """
    # Re-add stripped padding so base64 decoding succeeds.
    padding = 4 - len(encoded) % 4
    if padding != 4:
        encoded += "=" * padding
    compressed = base64.urlsafe_b64decode(encoded)
    raw = gzip.decompress(compressed)
    data: dict = _json.loads(raw)
    if "figures" in data:
        return DashboardSpec.model_validate(data)
    return VisualizationSpec.model_validate(data)

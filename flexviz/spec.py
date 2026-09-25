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
from typing import Annotated, Any, Literal, TypeAlias, TypedDict
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

from pydantic import (
    AfterValidator,
    BaseModel,
    Field,
    field_serializer,
    model_validator,
)

_SPEC_VERSION = "0.6"


def _check_spec_version(version: str) -> str:
    # Specs round-trip only within one spec version (pre-1.0 policy).
    if version != _SPEC_VERSION:
        raise ValueError(
            f"spec version {version!r} is not supported; this FlexViz reads "
            f"spec version {_SPEC_VERSION!r}"
        )
    return version


SpecVersion: TypeAlias = Annotated[str, AfterValidator(_check_spec_version)]

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
    color_map: dict[str, str]
    color_scale: str
    color_range: tuple[float, float] | Literal["auto"]
    color_norm: Literal["linear", "log"]


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

    def as_tuple(self) -> tuple[float, float] | tuple[str, str]:
        return (self.min, self.max)


GeoViewportCoordinates: TypeAlias = list[tuple[float, float]]
ViewportStateValue: TypeAlias = AxisRange | GeoViewportCoordinates | None


class ClauseFilter(BaseModel):
    """One column filter. Exactly one of range/values is set.

    - ``range``: continuous `is_between(lo, hi)` filter (numeric, temporal).
    - ``values``: categorical membership filter (see ``predicates.py`` for the
      compiled form).
    - ``closed``: range endpoint inclusivity. ``"left"`` is the half-open
      interval `[lo, hi)` emitted by cube-snapped commits; ``"both"``
      (default) is the closed interval. Only valid with ``range``.
    """

    column: str
    range: tuple[Any, Any] | None = None
    values: list[Any] | None = None
    closed: Literal["both", "left"] = "both"

    @model_validator(mode="after")
    def _exactly_one_field(self) -> ClauseFilter:
        has_range = self.range is not None
        has_values = self.values is not None
        if has_range == has_values:
            raise ValueError("ClauseFilter requires exactly one of `range` or `values`")
        if has_values and self.closed != "both":
            raise ValueError("ClauseFilter `closed` is only valid with `range`")
        return self


class SelectionPredicate(BaseModel):
    """One OR-disjunct. Clauses inside are ANDed together."""

    clauses: list[ClauseFilter] = Field(default_factory=list)


class SelectionState(BaseModel):
    """A per-figure cross-filter selection expressed as a logical predicate.

    Semantics:
        ClauseFilter        = one column test
        SelectionPredicate  = AND(clauses)
        SelectionState      = OR(predicates)
        Engine filter       = AND(selection_states from other figures)
    """

    source_figure_uid: str | None = None
    predicates: list[SelectionPredicate] = Field(default_factory=list)


class TraceHoverSpec(BaseModel):
    """Hover interaction capabilities for a single trace.

    ``source_modes`` lists the hover modes this trace can emit as a source.
    ``target_modes`` lists the hover modes this trace can receive as a target.
    Modes declared here may exceed what is currently implemented; the JS
    runtime intersects with ``IMPLEMENTED_HOVER_MODES`` gates before dispatch.
    """

    source_modes: list[HoverMode] = Field(default_factory=list)
    target_modes: list[HoverMode] = Field(default_factory=list)


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
    axis_columns: dict[str, str] = Field(default_factory=dict)
    label_columns: list[str] = Field(default_factory=list)
    path_columns: list[str] = Field(default_factory=list)
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
    axes: tuple[str, ...] | None = None
    backend_data: dict[str, BackendDataValue] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    display: dict[str, Any] = Field(default_factory=dict)
    # Anchor ids whose viewport range parameterizes this trace's aggregation
    # (e.g. ``("x",)`` for a line, ``("coordinates",)`` for a map trace).
    # Read by both the client (to suppress no-op viewport POSTs) and the engine
    # (to gate per-trace recompute).  Current producers must emit a concrete
    # tuple: populated means "these viewport anchors re-aggregate the trace",
    # while an explicit empty tuple means deliberately frozen.  ``None`` is only
    # the Pydantic default for hand-built/imported specs that omit the field:
    # server-side trace reconstruction may derive the default policy from it,
    # but the browser does not repair it and treats null/omitted as no
    # recompute axes.
    recompute_axes: tuple[str, ...] | None = None
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

    # No "/": viewport keys are "<figure_uid>/<axis_id>".
    uid: str = Field(default_factory=lambda: str(uuid4()), pattern=r"^[^/]+$")
    source: str | None = None
    layout: dict[str, Any] = Field(default_factory=dict)
    traces: list[TraceSpec] = Field(default_factory=list)


def figure_axis_columns(figure: FigureSpec) -> dict[str, set[str]]:
    """Map each cartesian axis id of ``figure`` to the columns its traces show.

    Roles follow the trace convention: ``axes[0]`` shows ``backend_data["x"]``
    and ``axes[1]`` shows ``backend_data["y"]``. An axis without a column (a
    histogram's count axis, a bar's value axis) is absent.
    """
    columns: dict[str, set[str]] = {}
    for trace in figure.traces:
        for axis_id, role in zip(trace.axes or (), ("x", "y")):
            col = trace.backend_data.get(role)
            cols = [col] if isinstance(col, str) else list(col or [])
            if cols:
                columns.setdefault(axis_id, set()).update(cols)
    return columns


class GroupDomainState(BaseModel):
    """Client-owned group→color mapping for one group_by domain.

    The domain key is ``"{source_or_figure_uid}::{group_by_col}"``.
    ``mapping`` maps each group value (as a string) to a CSS colour string.
    ``next_color_index`` tracks which palette index to assign next.
    """

    mapping: dict[str, str] = Field(default_factory=dict)
    next_color_index: int = 0


class InteractionState(BaseModel):
    """Mutable interaction state (viewport ranges + selections).

    The client owns this state and echoes it back with every request so that
    the server remains fully stateless.
    """

    viewport: dict[str, ViewportStateValue] = Field(default_factory=dict)
    selections: list[SelectionState] = Field(default_factory=list)
    group_domains: dict[str, GroupDomainState] = Field(default_factory=dict)
    cross_filter_mode: Literal["update", "overlay"] = "update"

    @field_serializer("viewport")
    def _serialize_viewport(
        self, viewport: dict[str, ViewportStateValue]
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in viewport.items():
            if value is None or isinstance(value, AxisRange):
                out[key] = value
            else:
                out[key] = [[point[0], point[1]] for point in value]
        return out


class ClientState(BaseModel):
    """Client-only persistent state not read by the server engine.

    Included in every POST body to ``/dashboard/update`` and in
    share/export/import serialisation. The engine ignores it. The
    ``DashboardSpec`` validator reads ``axis_links``, ``axis_locks`` and
    ``axis_lock_ranges`` to check the links.

    This is the designated home for client-only-but-persistent state:
    hover mode, annotation visibility, panel collapse state, axis locks, etc.

    ``axis_locks`` maps ``"{figure_uid}/{axis_family}"`` → locked flag, where
    the family ``x`` also covers ``x2``. ``axis_lock_ranges`` maps
    ``"{figure_uid}/{axis_id}"`` → the pinned ``AxisRange`` of each locked
    axis. Both are applied entirely client-side (JS pins the viewport); only
    the link validator reads them.

    ``live_brush`` gates the cube live-brush loop (spec §2.5): ``"auto"``
    (default) binds ``plotly_selecting`` on range-geometry figures and slices
    client-side cubes during the drag; ``"off"`` never binds it — today's
    mouseup-only behavior bit-for-bit. The server never reads this field.

    ``axis_links`` holds groups of linked axes as viewport keys
    (``"{figure_uid}/{axis_id}"``). The client writes one range into every key
    of a group, so linked axes zoom, pan, autorange and reset together. The
    engine never reads it; ``DashboardSpec`` validates it (see
    ``_check_axis_links``) so no spec can carry a link the client cannot keep.
    """

    hover_mode: HoverToggle = "off"
    live_brush: Literal["auto", "off"] = "auto"
    axis_locks: dict[str, bool] = Field(default_factory=dict)
    axis_lock_ranges: dict[str, AxisRange] = Field(default_factory=dict)
    axis_links: list[list[str]] = Field(default_factory=list)


class VisualizationSpec(BaseModel):
    """Root spec object — fully JSON-serializable snapshot of a visualisation.

    Sent from client → server on every interaction request.  The server never
    stores it; all mutable state lives in ``state`` and is round-tripped by
    the client.
    """

    version: SpecVersion = _SPEC_VERSION
    figure: FigureSpec = Field(default_factory=FigureSpec)
    state: InteractionState = Field(default_factory=InteractionState)

    @model_validator(mode="after")
    def _check_viewport_figures(self) -> VisualizationSpec:
        _check_viewport_figures(self.state.viewport, {self.figure.uid})
        return self


def _check_viewport_figures(
    viewport: dict[str, ViewportStateValue], fig_uids: set[str]
) -> None:
    # A figure uid holds no "/", so a key splits at its one slash.
    for key in viewport:
        fig_uid, _, axis_id = key.partition("/")
        if not fig_uid or not axis_id or "/" in axis_id:
            raise ValueError(
                f"viewport key {key!r} must have the form '<figure_uid>/<axis_id>'"
            )
        if fig_uid not in fig_uids:
            raise ValueError(f"viewport key {key!r} names no figure in this spec")


_GRIDSTACK_CELL_HEIGHT_PX: int = 80
"""Pixels per grid row unit.  h=5 → 400 px on either layout path."""


class GridItem(BaseModel):
    """Position and size for one figure in a dashboard layout.

    Coordinates use the 12-column grid system shared by GridStack and CSS grid.
    ``fig_uid`` must match a ``FigureSpec.uid`` in the same ``DashboardSpec``.
    """

    fig_uid: str
    x: int = 0  # column (0–11)
    y: int = 0  # row (0-based row units)
    w: int = 6  # width in columns (1–12)
    h: int = 5  # height in row units (h × _GRIDSTACK_CELL_HEIGHT_PX = pixel height)


def _auto_grid_items(
    figures: list[FigureSpec],
    rows: int | None = None,
    cols: int | None = None,
) -> list[GridItem]:
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

    ``show_grid`` is the one field that hides more than a button: it is the
    only built-in button that toggles ``LayoutSpec.grid_editable``. Hiding it
    leaves the current mode in place rather than locking the layout. Use
    ``LayoutSpec.draggable=False`` for a layout that cannot move. ``show_import``
    can still restore ``grid_editable`` from an imported spec.
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
        CSS space between figures on the static grid (default ``"8px"``). With
        ``draggable=True`` GridStack keeps its own panel margin, so ``gap``
        only pads the outer edge of the grid.

    ``draggable``
        Selects the layout implementation for the rendered page. It is not a
        runtime edit-mode switch. ``True`` renders with GridStack.js and loads
        its CDN stylesheet and script; ``False`` renders a static CSS grid and
        loads neither.
        Position changes update ``grid_items`` in client-side state only —
        no backend request is fired.

    ``grid_editable``
        When ``draggable=True``, whether panels can currently be moved and
        resized. This is live client-side state: the layout button, an
        imported spec, and custom JavaScript can update it. It has no effect
        while ``draggable`` is ``False``, but keeps its value so specs
        round-trip unchanged.

    ``grid_items``
        Per-figure positions in the 12-column grid.  ``None`` causes positions
        to be auto-generated at render time.
        Updated in-place by the frontend after each drag/resize.

    ``toolbar``
        Controls which toolbar buttons are rendered.
    """

    gap: str = "8px"
    draggable: bool = True
    grid_editable: bool = False
    grid_items: list[GridItem] | None = None
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

    version: SpecVersion = _SPEC_VERSION
    figures: list[FigureSpec] = Field(default_factory=list)
    state: InteractionState = Field(default_factory=InteractionState)
    layout: LayoutSpec = Field(default_factory=LayoutSpec)
    client_state: ClientState = Field(default_factory=ClientState)

    @model_validator(mode="after")
    def _check_viewport_figures(self) -> DashboardSpec:
        _check_viewport_figures(self.state.viewport, {fig.uid for fig in self.figures})
        return self

    @model_validator(mode="after")
    def _check_axis_links(self) -> DashboardSpec:
        """Reject links the client cannot keep equal or the engine cannot use.

        Every spec entry point (builder, share URL, import, each request) runs
        this, so a hand-built or patched spec gets the same rules as
        ``Dashboard.link_axes``. The rules that need the data schema are in
        ``check_axis_link_types``.
        """
        figures = {fig.uid: fig for fig in self.figures}
        viewport = self.state.viewport
        # Links are x or y only, so a linked axis id is also its lock family,
        # the key form of axis_locks.
        locks = self.client_state.axis_locks
        lock_ranges = self.client_state.axis_lock_ranges
        seen: set[str] = set()
        for group in self.client_state.axis_links:
            fig_uids = {key.partition("/")[0] for key in group}
            if len(group) < 2 or len(fig_uids) != len(group):
                raise ValueError(
                    f"axis link group {group} needs axes of two or more figures, "
                    "one axis each; it cannot hold two axes of one figure"
                )
            if shared := seen.intersection(group):
                raise ValueError(f"axes {sorted(shared)} are in two link groups")
            seen.update(group)
            is_reversed: dict[str, bool] = {}
            for key in group:
                fig_uid, _, axis_id = key.partition("/")
                figure = figures.get(fig_uid)
                if figure is None:
                    raise ValueError(
                        f"linked axis {key!r} names no figure in this spec"
                    )
                if axis_id not in ("x", "y"):
                    raise ValueError(f"only x and y axes can be linked, not {key!r}")
                if axis_id not in figure_axis_columns(figure):
                    raise ValueError(
                        f"linked axis {key!r} shows no data column (count axes, "
                        "categorical traces and maps cannot be linked)"
                    )
                axis = _layout_axis(figure, axis_id)
                if axis.get("type") not in _LINKABLE_AXIS_TYPES:
                    raise ValueError(
                        f"{axis.get('type')} axis {key!r} cannot be linked: its "
                        "range is not in data units"
                    )
                is_reversed[key] = _axis_reversed(axis)
            for rule, value_of in (
                ("all be reversed or all be normal", is_reversed.get),
                ("hold equal ranges", viewport.get),
                ("be locked together", lambda key: locks.get(key, False)),
                ("be locked at one range", lock_ranges.get),
            ):
                values = [value_of(key) for key in group]
                if any(value != values[0] for value in values):
                    raise ValueError(f"linked axes {group} must {rule}")
        return self


# Plotly axis types whose range is in data units. A log range is in log10
# units and a category range in positions, so copying one would be wrong.
_LINKABLE_AXIS_TYPES = (None, "-", "linear", "date")


def _layout_axis(figure: FigureSpec, axis_id: str) -> dict[str, Any]:
    axis = figure.layout.get(f"{axis_id}axis")
    return axis if isinstance(axis, dict) else {}


def _axis_reversed(axis: dict[str, Any]) -> bool:
    """Whether Plotly draws the axis high to low: a "reversed" autorange
    variant, or a fixed range given high to low."""
    autorange = axis.get("autorange")
    if isinstance(autorange, str) and "reversed" in autorange:
        return True
    rng = axis.get("range")
    if isinstance(rng, (list, tuple)) and len(rng) == 2 and None not in rng:
        try:
            return rng[0] > rng[1]
        except TypeError:
            return False
    return False


def check_axis_link_types(spec: DashboardSpec, schemas: dict[str | None, Any]) -> None:
    """Check linked axes against the data types, which the spec cannot see.

    ``schemas`` maps each figure source name to its Polars schema; a source
    without one is skipped. The builder and the server both run this. Raises
    ``ValueError``.

    Only numeric, ``Date`` and ``Datetime`` columns can be linked: ``Time`` and
    ``Duration`` render as category axes, whose ranges are positions. A numeric
    column on a ``date`` axis is refused: its zoom sends dates the column
    cannot compare. The axes of a group must then agree on three things, as the
    client copies one range to all of them:

    - numeric or temporal, because a range of one does not parse as the other;
    - the time zone (``Date`` and a naive ``Datetime`` have none), because the
      range is wall-clock text, so one window would be two instants;
    - the Plotly axis type, because a ``date`` axis reports date strings and a
      ``linear`` one numbers.
    """
    import polars as pl

    figures = {fig.uid: fig for fig in spec.figures}
    for group in spec.client_state.axis_links:
        kinds: set[str] = set()
        for key in group:
            fig_uid, _, axis_id = key.partition("/")
            figure = figures[fig_uid]
            schema = schemas.get(figure.source) or {}
            axis_type = _layout_axis(figure, axis_id).get("type")
            for col in figure_axis_columns(figure)[axis_id]:
                dtype = schema.get(col)
                if dtype is None:
                    continue
                if dtype.is_numeric() and axis_type != "date":
                    kinds.add("numeric on a linear axis")
                elif isinstance(dtype, (pl.Date, pl.Datetime)):
                    zone = getattr(dtype, "time_zone", None) or "naive"
                    shown = "linear" if axis_type == "linear" else "date"
                    kinds.add(f"{zone} time on a {shown} axis")
                else:
                    raise ValueError(
                        f"linked axis {key!r} shows {col!r} of type {dtype} on a "
                        f"{axis_type or 'default'} axis; only numeric, Date and "
                        "Datetime columns can be linked, and a numeric one not "
                        "on a date axis"
                    )
        if len(kinds) > 1:
            raise ValueError(f"linked axes {group} mix {' and '.join(sorted(kinds))}")


# ---------------------------------------------------------------------------
# Spec encoding helpers (for shareable URLs)
# ---------------------------------------------------------------------------


def encode_spec(spec: VisualizationSpec | DashboardSpec) -> str:
    """Encode a spec to a compact, URL-safe string.

    The spec is JSON-serialised, gzip-compressed (level 9), and
    base64url-encoded without ``=`` padding so the result is safe in a URL
    query parameter.
    """
    compressed = gzip.compress(spec.model_dump_json().encode(), compresslevel=9)
    return base64.urlsafe_b64encode(compressed).rstrip(b"=").decode()


def decode_spec(encoded: str) -> VisualizationSpec | DashboardSpec:
    """Decode a string produced by :func:`encode_spec` into a spec model.

    See :func:`parse_spec` for the model choice.
    """
    padded = encoded + "=" * (-len(encoded) % 4)
    raw = gzip.decompress(base64.urlsafe_b64decode(padded))
    return parse_spec(_json.loads(raw))


def parse_spec(data: dict[str, Any]) -> VisualizationSpec | DashboardSpec:
    """Validate a spec dict: a ``DashboardSpec`` when it has ``"figures"``,
    else a ``VisualizationSpec``.

    Both models refuse a spec of another version. Raises ``ValueError``.
    """
    if "figures" in data:
        return DashboardSpec.model_validate(data)
    return VisualizationSpec.model_validate(data)


def encoded_spec_from_url(url: str) -> str:
    """Pull the ``spec=`` query value out of a share URL.

    A bare encoded spec (no ``://`` or ``?``) is returned unchanged, so the
    same helper accepts both a full URL and the raw value. Raises
    ``ValueError`` when a URL carries no ``spec=`` value.
    """
    if "://" in url or "?" in url:
        values = parse_qs(urlsplit(url).query).get("spec")
        if not values:
            raise ValueError("no spec= query parameter in URL")
        return values[0]
    return url

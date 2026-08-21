"""Interaction event model and trace-delta output.

These types decouple the engine from any specific transport (Dash relayoutData,
FastAPI JSON payloads, WebSocket messages, ...).

Pydantic dataclasses are used so that the same types serve both as the
internal engine model *and* as validated FastAPI request/response bodies —
no duplication, no translation layer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field

from .spec import SelectionState


class InteractionEvent(BaseModel):
    """A single renderer-agnostic interaction event.

    ``type``
        - ``"init"``      — initial data load (no viewport, no selections)
        - ``"viewport"``  — zoom / pan (axis ranges changed)
        - ``"selection"`` — one or more rectangle selections changed
        - ``"deselect"``  — all selections cleared
        - ``"reset"``     — full reset (autorange + clear selections)
        - ``"cube_request"`` — brush-start cube materialization hint.  Carries
          the usual ``axis_ranges`` + ``selections`` (the committed ones — the
          future passive set) and **no active range**; the server computes no
          deltas for it.  Only meaningful with ``request_cube=True`` on the
          request body (see ``server.UpdateRequest``).

    ``axis_ranges``
        Mapping of trace-anchor axis id (e.g. ``"x"``, ``"y2"``, ``"map"``)
        to range values.  For cartesian axes this is a ``(start, stop)`` tuple;
        for maps it may be a dict of centre/zoom data; ``None`` means
        "autorange / reset".

    ``selections``
        Full list of current rectangular selections.  Populated for
        ``"selection"`` events; empty list for ``"deselect"``.

    ``force_update``
        When ``True`` every scalable trace is recomputed regardless of whether
        its axes appear in ``axis_ranges``.  Useful for the ``"init"`` event.

    ``figure_uid``
        For dashboard ``"viewport"`` events, the uid of the figure that
        triggered the interaction.  The server restricts re-aggregation to
        traces belonging to that figure only, leaving all other figures
        unchanged.  ``None`` means *all* figures are affected (used for
        ``"init"``, ``"reset"``, ``"selection"``, and ``"deselect"`` events).
    """

    type: Literal["init", "viewport", "selection", "deselect", "reset", "cube_request"]
    axis_ranges: Dict[str, Any] = Field(default_factory=dict)
    selections: List[SelectionState] = Field(default_factory=list)
    force_update: bool = False
    figure_uid: str | None = None


class ActiveSource(BaseModel):
    """The brushed figure/trace/column triple that defines a cube's free axis.

    Sent alongside a ``"cube_request"`` event (``request_cube=True``): the
    engine takes ``column`` as the free axis, binned to P over ``figure_uid``'s
    viewport.  The active brush *range* never touches the server — slicing is
    client-side.

    ``trace_uid`` names the trace actually interacted with; the engine
    resolves the source trace by uid (validating ``column`` is its primary
    free column) so two source traces sharing a primary column in one figure
    (e.g. bar(cat) + treemap(cat, sub)) can never be confused.
    """

    figure_uid: str
    column: str
    trace_uid: str


class GroupedChildDelta(BaseModel):
    """One child update inside a grouped parent delta."""

    uid: str
    updates: Dict[str, Any] = Field(default_factory=dict)
    parent_uid: str
    group_value_key: str


class TraceDelta(BaseModel):
    """Canonical trace update: semantic data keys (x, y, customdata, etc.).

    Adapters translate ``updates`` to renderer-specific format.
    ``updates`` values are plain Python lists (numpy arrays are converted
    before serialisation).

    ``uid`` is the stable trace identity assigned by ``Figure._add_trace``.
    Adapters must look up traces by uid rather than by position so that
    multi-figure dashboards do not accidentally patch the wrong trace.

    For grouped parent traces (``group_by`` set), ``group_results`` carries
    one child ``GroupedChildDelta`` per currently visible group.  The adapter
    reconciles child series against its previous set for this parent —
    absent children are removed.  ``updates`` is empty for grouped parents.
    """

    uid: str
    updates: Dict[str, Any] = Field(default_factory=dict)
    group_results: List[GroupedChildDelta] | None = None
    layer: Literal["bg", "fg"] | None = None

    @property
    def data(self) -> Dict[str, Any]:
        """Canonical data dict (alias for updates)."""
        return self.updates

"""Interaction event model and trace-delta output.

These types decouple the engine from any specific transport (Dash relayoutData,
FastAPI JSON payloads, WebSocket messages, ...).

Pydantic dataclasses are used so that the same types serve both as the
internal engine model *and* as validated FastAPI request/response bodies —
no duplication, no translation layer.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .spec import SelectionState


class InteractionEvent(BaseModel):
    """A single renderer-agnostic interaction event.

    ``type``
        - ``"init"``      — initial data load (no viewport, no selections)
        - ``"viewport"``  — zoom / pan / autorange (viewport state changed)
        - ``"selection"`` — one or more rectangle selections changed
        - ``"deselect"``  — all selections cleared
        - ``"cube_request"`` — brush-start cube materialization hint.  Carries
          the ``selections`` (the committed ones — the future passive set)
          and **no active range**; the server computes no deltas for it.
          Only meaningful with ``request_cube=True`` on the request body.

    ``viewport_keys``
        The ``state.viewport`` keys (``"<figure_uid>/<axis_id>"``) this event
        changed.  The ranges themselves are read from ``state.viewport``; a
        listed key that is absent there was autoranged.  A trace re-aggregates
        when one of its ``recompute_axes`` is listed for its figure, so one
        event can cover several figures (linked axes).

    ``selections``
        Full list of current rectangular selections.  Populated for
        ``"selection"`` events; empty list for ``"deselect"``.

    ``force_update``
        When ``True`` every scalable trace is recomputed, except the traces of
        a figure that sources a selection in a ``"selection"`` event.
    """

    type: Literal["init", "viewport", "selection", "deselect", "cube_request"]
    viewport_keys: list[str] = Field(default_factory=list)
    selections: list[SelectionState] = Field(default_factory=list)
    force_update: bool = False


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
    updates: dict[str, Any] = Field(default_factory=dict)
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
    updates: dict[str, Any] = Field(default_factory=dict)
    group_results: list[GroupedChildDelta] | None = None
    layer: Literal["bg", "fg"] | None = None

    @property
    def data(self) -> dict[str, Any]:
        """Canonical data dict (alias for updates)."""
        return self.updates

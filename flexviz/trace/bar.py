"""BarTrace — renderer-agnostic bar / grouped-bar trace.

A bar trace **always aggregates**: ``labels`` is the category column (used as
the primary ``group_by`` key).  When ``values`` is omitted the trace counts
rows per label; when ``values`` is given the trace applies ``agg`` to that
column.

Orientation
-----------
``orientation="v"`` (default): label axis = x, value axis = y.
``orientation="h"``: label axis = y, value axis = x.
The backend aggregation is identical regardless of orientation.

Grouping
--------
Without ``group_by``: one bar per unique label value.
With ``group_by="col"``: a second colour dimension — one sub-series per
unique ``group_by`` value; ``_to_update`` returns ``group_results``, one
child ``TraceResult`` per group.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import polars as pl

from ..cube import CubeTargetSpec, FreeAxisSpec, TargetDimSpec
from ..LF import GroupedAggregationSpec
from ..spec import TraceSelectionSpec, TraceSpec
from ._hist_helpers import _AGG_FUNCTIONS
from .base import (
    FlexTrace,
    GroupedChildResult,
    TraceResult,
    _categorical_dims_ok,
    _child_uid_for_group,
    _composite_label,
    _cube_measure_spec,
    _group_value_key,
    _group_values_from_frame,
    _to_col_tuple,
)


class BarPlot(FlexTrace):
    """Scalable bar trace backed by a Polars LazyFrame.

    Parameters
    ----------
    labels:
        Column name for the label / category axis (also the primary group key).
    values:
        Column name for the values to aggregate.  When ``None`` (default) the
        trace counts rows per label — ``agg`` is irrelevant in that case.
    agg:
        Aggregation function applied to ``values``: ``"sum"`` (default),
        ``"mean"``, ``"median"``, ``"min"``, ``"max"``, or ``"n_unique"``.
        Ignored when ``values`` is ``None``.
    name:
        Legend / series name.
    color:
        CSS colour string passed to the renderer.
    orientation:
        ``"v"`` (default) for vertical bars (labels on x, values on y);
        ``"h"`` for horizontal bars (labels on y, values on x).
    bar_mode:
        ``"group"`` (default, side-by-side) or ``"stack"``; only meaningful
        when ``group_by`` is set.  Passed as a display hint to the adapter.
    group_by:
        Optional second grouping column that produces one coloured sub-series
        per unique value.
    axes:
        Axis anchor tuple, e.g. ``("x", "y")``.
    """

    trace_type: str = "bar"
    select_policy_doc: str = "categorical — label click (not a box-range)"

    def __init__(
        self,
        labels: str | Sequence[str],
        values: str | None = None,
        agg: Literal["sum", "mean", "median", "min", "max", "n_unique"] = "sum",
        name: str | None = None,
        color: str | None = None,
        orientation: Literal["v", "h"] = "v",
        bar_mode: Literal["group", "stack"] = "group",
        group_by: str | Sequence[str] | None = None,
        color_map: dict | None = None,
        axes: tuple[str, ...] = ("x", "y"),
    ) -> None:
        if values is not None and agg not in _AGG_FUNCTIONS:
            raise ValueError(f"agg must be one of {list(_AGG_FUNCTIONS)}, got {agg!r}")
        label_cols = _to_col_tuple(labels, "labels")
        group_cols = (
            _to_col_tuple(group_by, "group_by") if group_by is not None else None
        )
        backend_data: dict[str, str | list[str]] = {"labels": list(label_cols)}
        if values is not None:
            backend_data["values"] = values
        # Store "count" internally when values is None; agg param is irrelevant then.
        stored_agg = "count" if values is None else agg
        super().__init__(
            backend_data=backend_data,
            display={
                "name": name
                or (values if values is not None else _composite_label(label_cols)),
                **({"color": color} if color is not None else {}),
                "bar_mode": bar_mode,
                **({"color_map": color_map} if color_map is not None else {}),
            },
            params={
                "agg": stored_agg,
                "orientation": orientation,
                **({"group_by": list(group_cols)} if group_cols is not None else {}),
            },
            axes=axes,
        )
        if group_cols is not None:
            self.overlay_style = "filtered_only"

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def label_cols(self) -> tuple[str, ...]:
        labels = self._backend_data["labels"]
        if isinstance(labels, str):
            return (labels,)
        return tuple(labels)

    @property
    def values_col(self) -> str | None:
        return self._backend_data.get("values")

    @property
    def agg(self) -> str:
        return self._params["agg"]

    @property
    def orientation(self) -> str:
        return self._params.get("orientation", "v")

    @property
    def bar_mode(self) -> str:
        return self._display.get("bar_mode", "group")

    def _make_selection_spec(self) -> TraceSelectionSpec:
        # Categorical: a box-drag over bars selects the covered labels (is_in) as
        # one fresh set each drag, so the accumulation policy is "replace".
        return TraceSelectionSpec(
            kind="categorical", label_columns=list(self.label_cols), multi="replace"
        )

    # ------------------------------------------------------------------
    # Cube descriptors (cross-filter pre-aggregation)
    # ------------------------------------------------------------------

    def get_cube_source_spec(
        self,
        axis_range: tuple[float, float] | None,
        schema: pl.Schema | None = None,
    ) -> FreeAxisSpec | None:
        """A box-drag over bars defines a categorical free axis on the label
        columns: the free key is the label tuple, ``columns[0]`` the primary
        join column.

        ``axis_range`` is ignored — categorical selection geometry is
        viewport-independent: a drag selects the *labels* it covers, and the
        same label set means the same cube slice no matter which portion of
        the label axis is in view (no domain, no binning). Grouped bars are
        still valid sources: the brush selects labels, independent of the
        grouping. Gates (contract B): a schema is required and every label
        column must be string-, integer-, or float-dtyped, else ``None`` — the
        same gate as ``get_cube_target_spec`` (``allow_numeric=True``).
        Numeric labels round-trip because the cube free categories stay typed
        in the FVCube header and committed ``is_in`` values cast back to the
        source column through ``predicates._values_to_typed_series``; the
        demo's hour-of-day / month bars rely on this to drive a live brush.
        """
        if not _categorical_dims_ok(schema, self.label_cols, allow_numeric=True):
            return None
        return FreeAxisSpec(
            column=self.label_cols[0],
            columns=self.label_cols,
            kind="categorical",
            p=0,
            domain=None,
        )

    def get_cube_target_spec(
        self,
        axis_range: tuple[float, float] | None,
        schema: pl.Schema | None = None,
    ) -> CubeTargetSpec | None:
        """A bar is a categorical cube target: one dim per label column, then
        one per ``group_by`` column (pinned order — contract C), all
        categorical.

        ``axis_range`` is ignored — bar aggregation never depends on the
        viewport. Gates: the agg must be in the cube measure algebra
        (``median``/``n_unique`` ⇒ ``None``), a non-count agg's value column
        must be numeric, and every label/group column must pass the
        categorical-dim gate (contracts A/B). **Label** columns additionally
        accept integer and float dtypes (the codec preserves numeric
        categories in numeric order, e.g. the demo's hour_of_day / month
        bars); **group_by** columns stay string-only because the grouped child's identity
        (``_group_value_key``) is ``json.dumps``-stringified server-side
        and the client cube path keys children by renderer category value, so
        numeric groups would not reconcile reliably.
        """
        if not _categorical_dims_ok(schema, self.label_cols, allow_numeric=True):
            return None
        if self.group_by_cols and not _categorical_dims_ok(schema, self.group_by_cols):
            return None
        measure = _cube_measure_spec(schema, self.agg, self.values_col)
        if measure is None:
            return None
        dim_cols = (*self.label_cols, *(self.group_by_cols or ()))
        return CubeTargetSpec(
            target_dims=tuple(
                TargetDimSpec(column=c, kind="categorical") for c in dim_cols
            ),
            measure=measure,
        )

    # ------------------------------------------------------------------
    # FlexTrace interface
    # ------------------------------------------------------------------

    def get_aggregation_spec(
        self,
        update_range: dict[str, Any],
        schema: pl.Schema | None = None,
    ) -> GroupedAggregationSpec:
        """Return a grouped aggregation spec for ``group_by().agg().sort()``."""
        if self.values_col is None:
            agg_expr = pl.len().alias(self.uid)
        else:
            agg_fn = _AGG_FUNCTIONS[self.agg]
            agg_expr = agg_fn(self.values_col).alias(self.uid)

        label_cols = self.label_cols
        group_by_cols = self.group_by_cols
        if group_by_cols is not None:
            group_cols = (*label_cols, *group_by_cols)
            sort_cols = (*group_by_cols, *label_cols)
        else:
            group_cols = label_cols
            sort_cols = label_cols

        return GroupedAggregationSpec(
            uid=self.uid,
            group_cols=group_cols,
            agg_exprs=(agg_expr,),
            sort_cols=sort_cols,
        )

    def _to_update(self, df: pl.DataFrame) -> TraceResult:
        """Backward-compatible alias for grouped query output parsing."""
        return self._to_grouped_update(df)

    def _to_grouped_update(self, df: pl.DataFrame) -> TraceResult:
        """Shape grouped query output into ``TraceResult``."""
        orientation = self.orientation

        def _make_updates(labels: list, values: list) -> dict:
            # TODO: Plotly bars could eventually render multi-column labels as a
            # multicategory axis; for now they are flat composite strings.
            if orientation == "h":
                return {"x": values, "y": labels, "orientation": "h"}
            return {"x": labels, "y": values}

        label_cols = self.label_cols
        group_by_cols = self.group_by_cols
        if group_by_cols is None:
            raw_labels = _group_values_from_frame(df, label_cols)
            labels = (
                raw_labels
                if len(label_cols) == 1
                else [_group_value_key(v) for v in raw_labels]
            )
            values = df[self.uid].to_list()
            updates = _make_updates(labels, values)
            color_map = self._display.get("color_map")
            if color_map is not None:
                updates["marker"] = {
                    "color": [color_map.get(str(label)) for label in labels]
                }
            return TraceResult(updates=updates)

        # Grouped bar — one child TraceResult per group_by value
        group_results: list[GroupedChildResult] = []
        group_vals = _group_values_from_frame(df, group_by_cols)
        start = 0
        for i in range(1, len(group_vals) + 1):
            if i < len(group_vals) and group_vals[i] == group_vals[start]:
                continue
            group_df = df.slice(start, i - start)
            gv = group_vals[start]
            gvk = _group_value_key(gv)
            child_uid = _child_uid_for_group(self.uid, gv)
            raw_labels = _group_values_from_frame(group_df, label_cols)
            labels = (
                raw_labels
                if len(label_cols) == 1
                else [_group_value_key(v) for v in raw_labels]
            )
            values = group_df[self.uid].to_list()
            group_results.append(
                GroupedChildResult(
                    child_uid=child_uid,
                    group_value_key=gvk,
                    updates=_make_updates(labels, values),
                )
            )
            start = i
        return TraceResult(group_results=group_results)

    # ------------------------------------------------------------------
    # Spec reconstruction (server-side)
    # ------------------------------------------------------------------

    @classmethod
    def from_trace_spec(cls, spec: TraceSpec) -> "BarPlot":
        bar_mode = spec.display.get("bar_mode", "group")
        if "x" in spec.backend_data:
            # Backward compat: old specs stored {"x": labels_col, "y": values_col}
            labels = spec.backend_data["x"]
            old_y = spec.backend_data["y"]
            old_agg = spec.params.get("agg", "sum")
            values = None if old_agg == "count" else old_y
            agg = "sum" if old_agg == "count" else old_agg
        else:
            labels = spec.backend_data["labels"]
            values = spec.backend_data.get("values")
            agg = spec.params.get("agg", "sum")
        trace = cls(
            labels=labels,
            values=values,
            agg=agg,
            name=spec.display.get("name"),
            color=spec.display.get("color"),
            orientation=spec.params.get("orientation", "v"),
            bar_mode=bar_mode,
            group_by=spec.params.get("group_by"),
            color_map=spec.display.get("color_map"),
            axes=spec.axes or ("x", "y"),
        )
        if "group_domain_key" in spec.params:
            trace._params["group_domain_key"] = spec.params["group_domain_key"]
        trace.uid = spec.uid
        return trace

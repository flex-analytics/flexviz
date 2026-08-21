"""PiePlot — renderer-agnostic pie / donut trace.

A pie trace **always aggregates**: ``labels`` is the category column (used as
the ``group_by`` key), ``values`` is the column to aggregate.  Setting
``hole > 0`` turns it into a donut chart.

Non-cartesian
-------------
Pie traces have no cartesian axes — they do not participate in viewport/zoom
events, brush selection, or linked hover.
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
    TraceResult,
    _categorical_dims_ok,
    _composite_label,
    _cube_measure_spec,
    _group_value_key,
    _group_values_from_frame,
    _to_col_tuple,
)


class PiePlot(FlexTrace):
    """Scalable pie / donut trace backed by a Polars LazyFrame.

    Parameters
    ----------
    labels:
        Column name for the category / slice labels.
    values:
        Column name for the values to aggregate per slice.  When ``None``
        (default) the trace counts rows per label — ``agg`` is irrelevant
        in that case.
    agg:
        Aggregation function applied to ``values``: ``"sum"`` (default),
        ``"mean"``, ``"median"``, ``"min"``, ``"max"``, or ``"n_unique"``.
        Ignored when ``values`` is ``None``.
    name:
        Legend / series name.
    hole:
        Fraction of the radius to cut out (0 = pie, 0.4–0.6 = donut).
    color_map:
        Optional ``{label: css_color}`` dict.
    """

    trace_type: str = "pie"
    select_policy_doc: str = "categorical — slice click"
    overlay_style: str = "filtered_only"

    def __init__(
        self,
        labels: str | Sequence[str],
        values: str | None = None,
        agg: Literal["sum", "mean", "median", "min", "max", "n_unique"] = "sum",
        name: str | None = None,
        hole: float = 0.0,
        color_map: dict | None = None,
    ) -> None:
        if values is not None and agg not in _AGG_FUNCTIONS:
            raise ValueError(f"agg must be one of {list(_AGG_FUNCTIONS)}, got {agg!r}")
        label_cols = _to_col_tuple(labels, "labels")
        backend_data: dict[str, Any] = {"labels": list(label_cols)}
        if values is not None:
            backend_data["values"] = values
        stored_agg = "count" if values is None else agg
        super().__init__(
            backend_data=backend_data,
            display={
                "name": name
                or (values if values is not None else _composite_label(label_cols)),
                **({"color_map": color_map} if color_map is not None else {}),
            },
            params={
                "agg": stored_agg,
                "hole": hole,
            },
            axes=None,
        )

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
    def hole(self) -> float:
        return self._params.get("hole", 0.0)

    def _make_selection_spec(self) -> TraceSelectionSpec:
        # Categorical: a slice click selects that label; multi-click ORs.
        return TraceSelectionSpec(
            kind="categorical", label_columns=list(self.label_cols), multi="or"
        )

    # ------------------------------------------------------------------
    # Cube descriptors (cross-filter pre-aggregation)
    # ------------------------------------------------------------------

    def get_cube_source_spec(
        self,
        axis_range: tuple[float, float] | None,
        schema: pl.Schema | None = None,
    ) -> FreeAxisSpec | None:
        """A slice click defines a categorical free axis on the label columns
        (OR'd multi-click = a union of category keys at slice time).

        ``axis_range`` is ignored — categorical selection geometry is
        viewport-independent (a pie has no cartesian viewport at all): a click
        selects a *label*, never a range, so the cube needs no domain and no
        binning. Gates (contract B): a schema is required and every label
        column must be string-, integer-, or float-dtyped, else ``None`` — the
        same gate as ``get_cube_target_spec`` (``allow_numeric=True``), so a
        numeric-label pie is symmetric (it can both drive and receive a live
        brush).
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
        """A pie is a categorical cube target: one dim per label column in
        order (contract C) — identical to an ungrouped bar with the same
        labels and measure, so the two share one cube blob.

        ``axis_range`` is ignored (pies have no viewport). Gates: agg in the
        cube measure algebra (``median``/``n_unique`` ⇒ ``None``), numeric
        value column for non-count aggs, and the categorical-dim gate on the
        label columns (contracts A/B). Numeric labels are accepted (matching
        an ungrouped bar with the same labels — the codec preserves numeric
        categories in numeric order so the two still share one cube blob).
        """
        if not _categorical_dims_ok(schema, self.label_cols, allow_numeric=True):
            return None
        measure = _cube_measure_spec(schema, self.agg, self.values_col)
        if measure is None:
            return None
        return CubeTargetSpec(
            target_dims=tuple(
                TargetDimSpec(column=c, kind="categorical") for c in self.label_cols
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
        if self.values_col is None:
            agg_expr = pl.len().alias(self.uid)
        else:
            agg_fn = _AGG_FUNCTIONS[self.agg]
            agg_expr = agg_fn(self.values_col).alias(self.uid)

        return GroupedAggregationSpec(
            uid=self.uid,
            group_cols=self.label_cols,
            agg_exprs=(agg_expr,),
            sort_cols=self.label_cols,
        )

    def _to_update(self, df: pl.DataFrame) -> TraceResult:
        return self._to_grouped_update(df)

    def _to_grouped_update(self, df: pl.DataFrame) -> TraceResult:
        label_cols = self.label_cols
        raw_labels = _group_values_from_frame(df, label_cols)
        labels = (
            raw_labels
            if len(label_cols) == 1
            else [_group_value_key(v) for v in raw_labels]
        )
        values = df[self.uid].to_list()
        updates: dict = {"labels": labels, "values": values}
        color_map = self._display.get("color_map")
        if color_map is not None:
            updates["marker"] = {
                "colors": [color_map.get(str(label)) for label in labels]
            }
        return TraceResult(updates=updates)

    # ------------------------------------------------------------------
    # Spec reconstruction
    # ------------------------------------------------------------------

    @classmethod
    def from_trace_spec(cls, spec: TraceSpec) -> "PiePlot":
        old_agg = spec.params.get("agg", "sum")
        stored_values = spec.backend_data.get("values")
        # Backward compat: old specs always had "values"; if agg was "count" the
        # values column was ignored — reconstruct as values=None.
        values = None if old_agg == "count" else stored_values
        agg = "sum" if old_agg == "count" else old_agg
        trace = cls(
            labels=spec.backend_data["labels"],
            values=values,
            agg=agg,
            name=spec.display.get("name"),
            hole=spec.params.get("hole", 0.0),
            color_map=spec.display.get("color_map"),
        )
        trace.uid = spec.uid
        return trace

"""Box plot — renderer-agnostic 1-D box trace.

Either ``x`` or ``y`` must be provided (not both): the chosen column holds the
numeric sample.  The engine aggregates ``min``, quartiles, and ``max``, then
``_to_update`` emits Plotly-style precomputed box arrays (single-element lists)
plus ``orientation`` and ``x0`` / ``y0`` for the orthogonal category label.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict

import polars as pl

from ..cube import FreeAxisSpec, temporal_unit
from ..LF import AggregationSpec, GroupedAggregationSpec
from ..spec import TraceSpec
from .base import (
    FlexTrace,
    GroupedChildResult,
    TraceResult,
    _child_uid_for_group,
    _dtype_for_col,
    _group_value_key,
    _group_values_from_frame,
    _to_col_tuple,
)


def _range_cube_source_spec(
    column: str,
    axis_range: tuple[float, float] | None,
    schema: pl.Schema | None,
) -> FreeAxisSpec | None:
    """Shared 1-D range cube-source descriptor (hist-shaped; cube plan step 8).

    Continuous/temporal kind from the schema dtype; temporal axes carry their
    physical ``unit`` (contract G) and unsupported temporal dtypes
    (``Datetime("ns")``, ``Time``) gate to ``None``, as do non-numeric dtypes.
    Without a schema the kind defaults to ``"continuous"``. ``domain`` is the
    viewport range verbatim (``None`` = unzoomed; engine-resolved).

    A twin lives in ``line.py`` — keep the two in sync (a shared home in
    ``base.py`` is deliberately out of this step's file scope).
    """
    dtype = _dtype_for_col(schema, column)
    if dtype is not None:
        if dtype.is_temporal():
            unit = temporal_unit(dtype)
            if unit is None:
                return None
            return FreeAxisSpec(
                column=column, kind="temporal", p=2048, domain=axis_range, unit=unit
            )
        if not dtype.is_numeric():
            return None
    return FreeAxisSpec(column=column, kind="continuous", p=2048, domain=axis_range)


class BoxPlot(FlexTrace):
    """Scalable box plot trace backed by a Polars LazyFrame.

    Parameters
    ----------
    x:
        Column name for the sample when the box is horizontal (values on x).
        Provide either ``x`` or ``y``, not both.
    y:
        Column name for the sample when the box is vertical (values on y).
    name:
        Legend / trace name; also used as ``x0`` / ``y0`` for the dummy category.
    color:
        Colour hint (CSS string), passed to the renderer.
    axes:
        Axis anchors, e.g. ``("x", "y")``.
    """

    trace_type: str = "box"
    select_policy_doc: str = "data (prop) axis only — orthogonal range dropped"
    overlay_style: str = "filtered_only"

    def __init__(
        self,
        x: str | None = None,
        y: str | None = None,
        name: str | None = None,
        color: str | None = None,
        color_map: dict | None = None,
        axes: tuple[str, ...] = ("x", "y"),
        group_by: str | Sequence[str] | None = None,
    ) -> None:
        if (x is None) == (y is None):
            raise ValueError("Provide either x or y, not both (or neither).")
        group_cols = (
            _to_col_tuple(group_by, "group_by") if group_by is not None else None
        )

        col = x if x is not None else y
        prop_key = "x" if x is not None else "y"

        super().__init__(
            backend_data={prop_key: col},
            display={
                "name": name or col,
                **({"color": color} if color is not None else {}),
                **({"color_map": color_map} if color_map is not None else {}),
            },
            params={
                **({"group_by": list(group_cols)} if group_cols is not None else {}),
            },
            axes=axes,
        )

    # ------------------------------------------------------------------
    # Properties (convenience access)
    # ------------------------------------------------------------------

    @property
    def prop_key(self) -> str:
        assert len(self._backend_data) == 1, "BoxPlot must have exactly one data column"
        return next(iter(self._backend_data))

    @property
    def data_col(self) -> str:
        return self._backend_data[self.prop_key]

    def _default_select_axes(self) -> tuple[str, ...]:
        # Select only on the data (prop) axis; the orthogonal axis is decorative.
        anchor = self._axes[0] if self.prop_key == "x" else self._axes[1]
        return (anchor,)

    def _make_selection_spec(self):
        return self._range_selection_spec()

    # ------------------------------------------------------------------
    # Cube descriptors (cross-filter pre-aggregation)
    # ------------------------------------------------------------------

    def get_cube_source_spec(
        self,
        axis_range: tuple[float, float] | None,
        schema: pl.Schema | None = None,
    ) -> FreeAxisSpec | None:
        """A brush on a box plot defines a 1-D free axis on its data column.

        Identical shape to the histogram's source descriptor: the brush is on
        the data (prop) axis — x or y by orientation — and grouping does not
        affect source-ability. ``domain`` is the prop-axis viewport range
        verbatim (``None`` = unzoomed; the engine resolves it to the full
        data domain). Temporal dtypes carry their physical ``unit``
        (contract G); ``Datetime("ns")``/``Time`` have no exact string
        round-trip and gate to ``None``, as do non-numeric dtypes. Without a
        schema the kind defaults to ``"continuous"`` (the engine re-derives
        it from the source schema before building).
        """
        return _range_cube_source_spec(self.data_col, axis_range, schema)

    # ------------------------------------------------------------------
    # FlexTrace interface
    # ------------------------------------------------------------------

    def get_aggregation_spec(
        self,
        update_range: Dict[str, Any],
        schema: pl.Schema | None = None,
    ) -> AggregationSpec | GroupedAggregationSpec:
        """Return either a regular or grouped box aggregation spec."""
        expr = pl.col(self.data_col)

        group_by_cols = self.group_by_cols
        if group_by_cols is not None:
            grouped_expr = pl.struct(
                q0=pl.col(self.data_col).quantile(0.0),
                q1=pl.col(self.data_col).quantile(0.25),
                q2=pl.col(self.data_col).quantile(0.5),
                q3=pl.col(self.data_col).quantile(0.75),
                q4=pl.col(self.data_col).quantile(1.0),
            ).alias(self.uid)
            return GroupedAggregationSpec(
                uid=self.uid,
                group_cols=group_by_cols,
                sort_cols=group_by_cols,
                agg_exprs=(grouped_expr,),
            )

        expr = expr.quantile([0.0, 0.25, 0.5, 0.75, 1.0]).alias(self.uid)

        return AggregationSpec(expr=expr, uid=self.uid)

    def _to_update(
        self,
        df_agg: pl.DataFrame,
    ) -> TraceResult:
        """Unpack struct → precomputed box stats (single category)."""
        # TODO: TraceResult.updates -> current: puts values in a 1-element list
        # => explore the use of a scalar value instead
        raw = df_agg.select(pl.col(self.uid)).item()
        if isinstance(raw, dict):
            quantiles = [raw["q0"], raw["q1"], raw["q2"], raw["q3"], raw["q4"]]
            has_nulls = any(v is None for v in quantiles)
        else:
            quantiles = raw.to_list() if isinstance(raw, pl.Series) else list(raw)
            has_nulls = raw.is_null().any()
        assert len(quantiles) == 5, "Box plot should have 5 quantiles"
        if has_nulls:
            return TraceResult(
                updates={
                    "lowerfence": [],
                    "q1": [],
                    "median": [],
                    "q3": [],
                    "upperfence": [],
                }
            )

        iqr = quantiles[3] - quantiles[1]
        lower = max(quantiles[1] - 1.5 * iqr, quantiles[0])
        upper = min(quantiles[3] + 1.5 * iqr, quantiles[4])

        label = self._display.get("name", self.data_col)
        updates: Dict[str, Any] = {
            "lowerfence": [lower],
            "q1": [quantiles[1]],
            "median": [quantiles[2]],
            "q3": [quantiles[3]],
            "upperfence": [upper],
        }

        if self.prop_key == "x":
            updates["orientation"] = "h"
            updates["y0"] = label
        else:
            updates["orientation"] = "v"
            updates["x0"] = label

        return TraceResult(updates=updates)

    def _to_grouped_update(self, df_grouped: pl.DataFrame) -> TraceResult:
        """Unpack grouped box output into one child result per group."""
        group_by_cols = self.group_by_cols
        assert group_by_cols is not None, "Grouped box requires group_by"
        group_results: list[GroupedChildResult] = []
        for i, gv in enumerate(_group_values_from_frame(df_grouped, group_by_cols)):
            child_df = df_grouped.select(pl.col(self.uid).slice(i, 1))
            child_result = self._to_update(child_df)
            updates = dict(child_result.updates)
            group_value_key = _group_value_key(gv)
            if self.prop_key == "x":
                updates["y0"] = group_value_key
            else:
                updates["x0"] = group_value_key
            group_results.append(
                GroupedChildResult(
                    child_uid=_child_uid_for_group(self.uid, gv),
                    group_value_key=group_value_key,
                    updates=updates,
                )
            )
        return TraceResult(group_results=group_results)

    # ------------------------------------------------------------------
    # Spec reconstruction (server-side)
    # ------------------------------------------------------------------

    @classmethod
    def from_trace_spec(cls, spec: TraceSpec) -> "BoxPlot":
        prop_key = next(iter(spec.backend_data))
        col = spec.backend_data[prop_key]
        trace = cls(
            **{prop_key: col},
            name=spec.display.get("name"),
            color=spec.display.get("color"),
            color_map=spec.display.get("color_map"),
            axes=spec.axes or ("x", "y"),
            group_by=spec.params.get("group_by"),
        )
        if "group_domain_key" in spec.params:
            trace._params["group_domain_key"] = spec.params["group_domain_key"]
        trace.uid = spec.uid
        return trace

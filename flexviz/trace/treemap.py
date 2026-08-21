"""TreeMap trace — renderer-agnostic hierarchy trace backed by Polars LazyFrame."""

from __future__ import annotations

from typing import Any, Dict, Literal
from urllib.parse import quote as _url_quote

import polars as pl

from ..cube import CubeTargetSpec, FreeAxisSpec, TargetDimSpec
from ..LF import GroupedAggregationSpec
from ..spec import TraceSelectionSpec, TraceSpec
from ._hist_helpers import _AGG_FUNCTIONS
from .base import FlexTrace, TraceResult, _categorical_dims_ok, _cube_measure_spec


class TreeMap(FlexTrace):
    """Scalable treemap trace backed by a Polars LazyFrame.

    Parameters
    ----------
    path:
        Ordered list of column names from root to leaf (e.g. ["continent", "country"]).
    values:
        Column to aggregate per leaf node. When None, counts rows.
    agg:
        Aggregation function: "sum", "mean", "median", "min", "max", "n_unique".
        Ignored when values is None.
    name:
        Legend / series name.
    color_map:
        Optional {label: css_color} dict.
    """

    trace_type: str = "treemap"
    select_policy_doc: str = "categorical — hierarchical path click"
    overlay_style: str = "filtered_only"

    def __init__(
        self,
        path: list[str],
        values: str | None = None,
        agg: Literal["sum", "mean", "median", "min", "max", "n_unique"] = "sum",
        name: str | None = None,
        color_map: dict | None = None,
    ) -> None:
        if values is not None and agg not in _AGG_FUNCTIONS:
            raise ValueError(f"agg must be one of {list(_AGG_FUNCTIONS)}, got {agg!r}")
        backend_data: dict[str, str] = {}
        if values is not None:
            backend_data["values"] = values
        stored_agg = "count" if values is None else agg
        super().__init__(
            backend_data=backend_data,
            display={
                "name": name or "treemap",
                **({"color_map": color_map} if color_map is not None else {}),
            },
            params={"path": path, "agg": stored_agg},
            axes=None,
        )

    def _make_selection_spec(self) -> TraceSelectionSpec:
        # Hierarchical path: a node click drills down one clause per path level.
        return TraceSelectionSpec(
            kind="path", path_columns=list(self._params["path"]), multi="path"
        )

    # ------------------------------------------------------------------
    # Cube descriptors (cross-filter pre-aggregation)
    # ------------------------------------------------------------------

    def get_cube_source_spec(
        self,
        axis_range: tuple[float, float] | None,
        schema: pl.Schema | None = None,
    ) -> FreeAxisSpec | None:
        """A node click defines a categorical free axis on the full ``path``:
        the free key is the root-to-leaf tuple; a depth-``k`` click is a
        prefix predicate over the first ``k`` columns, expanded to category
        keys at slice time.

        ``axis_range`` is ignored — categorical selection geometry is
        viewport-independent (a treemap has no cartesian viewport): a click
        selects a path node, never a range, so the cube needs no domain and
        no binning. Gates (contract B): a schema is required and every path
        column must be string-dtyped, else ``None``.
        """
        path = tuple(self._params["path"])
        if not _categorical_dims_ok(schema, path):
            return None
        return FreeAxisSpec(
            column=path[0],
            columns=path,
            kind="categorical",
            p=0,
            domain=None,
        )

    def get_cube_target_spec(
        self,
        axis_range: tuple[float, float] | None,
        schema: pl.Schema | None = None,
    ) -> "CubeTargetSpec | None":
        """A treemap is a categorical cube target on its **leaf** path: one
        categorical dim per path column (pinned root-to-leaf order). The cube
        ships the finalized *leaf* aggregate per full-path cell; the client
        rebuilds the hierarchy by finalizing each leaf then **summing** those
        finalized leaf values up every path level (parents = Σ of child
        finalized values, mirroring ``_to_grouped_update`` — parents sum the
        leaf means too, never re-finalizing at parent levels).

        ``axis_range`` is ignored — a treemap has no cartesian viewport, so
        categorical selection geometry is viewport-independent. Gates
        (contracts A/B): the agg must be in the cube measure algebra
        (``median``/``n_unique`` ⇒ ``None``), a non-count agg's value column
        must be numeric, and every path column must pass the string-dtype +
        reserved-name gate (a schema is therefore required; ``schema=None`` ⇒
        ``None``).
        """
        path = tuple(self._params["path"])
        if not _categorical_dims_ok(schema, path):
            return None
        measure = _cube_measure_spec(
            schema, self._params["agg"], self._backend_data.get("values")
        )
        if measure is None:
            return None
        return CubeTargetSpec(
            target_dims=tuple(
                TargetDimSpec(column=c, kind="categorical") for c in path
            ),
            measure=measure,
        )

    def get_aggregation_spec(
        self,
        update_range: Dict[str, Any],
        schema: pl.Schema | None = None,
    ) -> GroupedAggregationSpec:
        path = self._params["path"]
        values_col = self._backend_data.get("values")
        if values_col is None:
            agg_expr = pl.len().alias(self.uid)
        else:
            agg_fn = _AGG_FUNCTIONS[self._params["agg"]]
            agg_expr = agg_fn(values_col).alias(self.uid)
        return GroupedAggregationSpec(
            uid=self.uid,
            group_cols=tuple(path),
            agg_exprs=(agg_expr,),
            sort_cols=tuple(path),
        )

    def _to_update(self, df: pl.DataFrame) -> TraceResult:
        return self._to_grouped_update(df)

    def _to_grouped_update(self, df: pl.DataFrame) -> TraceResult:
        path = self._params["path"]
        uid = self.uid
        ids: list = []
        labels: list = []
        parents: list = []
        values: list = []

        # Root node
        ids.append("root")
        labels.append("")
        parents.append("")
        values.append(df[uid].sum())

        for level in range(len(path)):
            level_cols = path[: level + 1]
            level_df = df.group_by(level_cols).agg(pl.col(uid).sum()).sort(level_cols)
            for row in level_df.iter_rows(named=True):
                parts = [str(row[col]) for col in level_cols]
                encoded = [_url_quote(p, safe="") for p in parts]
                node_id = "root/" + "/".join(encoded)
                parent_id = "root" if level == 0 else "root/" + "/".join(encoded[:-1])
                ids.append(node_id)
                labels.append(parts[-1])
                parents.append(parent_id)
                values.append(row[uid])

        updates: dict = {
            "labels": labels,
            "parents": parents,
            "ids": ids,
            "values": values,
        }
        color_map = self._display.get("color_map")
        if color_map is not None:
            updates["marker"] = {
                "colors": [color_map.get(str(label)) for label in labels]
            }
        return TraceResult(updates=updates)

    @classmethod
    def from_trace_spec(cls, spec: TraceSpec) -> "TreeMap":
        agg = spec.params.get("agg", "sum")
        values = spec.backend_data.get("values")
        values = None if agg == "count" else values
        agg = "sum" if agg == "count" else agg
        trace = cls(
            path=spec.params["path"],
            values=values,
            agg=agg,
            name=spec.display.get("name"),
            color_map=spec.display.get("color_map"),
        )
        trace.uid = spec.uid
        return trace

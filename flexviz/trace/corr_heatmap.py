"""CorrHeatmap — renderer-agnostic correlation heatmap trace.

Computes a pairwise Pearson (or Spearman) correlation matrix for a set
of numeric columns and returns it in the heatmap data format
``{x: [...cols], y: [...cols], z: [[corr_values]]}``.

Non-zoom / non-viewport
-----------------------
``recompute_axes = ()``    — correlation is a global statistic.
``_axes = None``           — no cartesian viewport semantics.
``overlay_style = "filtered_only"`` — shows filtered correlation when
cross-filtered.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Literal

import polars as pl

from ..cube import CubeTargetSpec, MeasureSpec
from ..LF import AggregationSpec
from ..spec import TraceSpec
from .base import (
    FlexTrace,
    TraceResult,
    _CUBE_RESERVED_COLS,
    _dtype_for_col,
)
from ._hist_helpers import (
    HeatmapColorRange,
    normalize_heatmap_color_scale,
    normalize_heatmap_color_range,
)


class CorrHeatmap(FlexTrace):
    """Scalable correlation heatmap trace.

    Parameters
    ----------
    columns:
        Subset of column names to correlate.  When ``None``, all numeric
        columns in the source are used.
    method:
        ``"pearson"`` (default) or ``"spearman"``.
    triangular:
        If ``True``, keep only one triangle (excl. diagonal).
    absolute:
        If ``True``, report ``abs(corr)`` (default ``False``).
    name:
        Legend / series name.
    """

    trace_type: str = "corr_heatmap"
    select_policy_doc: str = "none — not a cross-filter source"
    overlay_style: str = "filtered_only"
    DEFAULT_COLOR_SCALE_BY_ABSOLUTE: ClassVar[dict[bool, str]] = {
        False: "rdbu",
        True: "viridis",
    }
    DEFAULT_COLOR_RANGE_BY_ABSOLUTE: ClassVar[dict[bool, tuple[float, float]]] = {
        False: (-1.0, 1.0),
        True: (0.0, 1.0),
    }

    @classmethod
    def default_color_scale(cls, *, absolute: bool) -> str:
        return cls.DEFAULT_COLOR_SCALE_BY_ABSOLUTE[bool(absolute)]

    @classmethod
    def default_color_range(cls, *, absolute: bool) -> HeatmapColorRange:
        return cls.DEFAULT_COLOR_RANGE_BY_ABSOLUTE[bool(absolute)]

    @classmethod
    def _normalize_color_scale(cls, color_scale: str | None, *, absolute: bool) -> str:
        return normalize_heatmap_color_scale(
            color_scale,
            cls.DEFAULT_COLOR_SCALE_BY_ABSOLUTE[bool(absolute)],
            trace_name="CorrHeatmap",
        )

    @classmethod
    def _normalize_color_range(
        cls, color_range: Any, *, absolute: bool
    ) -> HeatmapColorRange:
        return normalize_heatmap_color_range(
            color_range,
            cls.DEFAULT_COLOR_RANGE_BY_ABSOLUTE[bool(absolute)],
            trace_name="CorrHeatmap",
        )

    def __init__(
        self,
        columns: list[str] | None = None,
        method: Literal["pearson", "spearman"] = "pearson",
        triangular: bool = False,
        absolute: bool = False,
        name: str | None = None,
        color_scale: str | None = None,
        color_range: tuple[float, float] | Literal["auto"] | None = None,
    ) -> None:
        super().__init__(
            backend_data={},
            display={
                "name": name or "Correlation",
                "color_scale": type(self)._normalize_color_scale(
                    color_scale, absolute=absolute
                ),
                "color_range": type(self)._normalize_color_range(
                    color_range, absolute=absolute
                ),
            },
            params={
                "method": method,
                "triangular": triangular,
                "absolute": absolute,
                **({"columns": columns} if columns is not None else {}),
            },
            axes=None,
        )
        self._columns: list[str] | None = columns

    @property
    def method(self) -> str:
        return self._params["method"]

    @property
    def triangular(self) -> bool:
        return self._params["triangular"]

    @property
    def absolute(self) -> bool:
        return self._params["absolute"]

    @property
    def color_scale(self) -> str:
        return self._display["color_scale"]

    @property
    def color_range(self) -> HeatmapColorRange:
        return self._display["color_range"]

    # ------------------------------------------------------------------
    # FlexTrace interface
    # ------------------------------------------------------------------

    def get_aggregation_spec(
        self,
        update_range: Dict[str, Any],
        schema: pl.Schema | None = None,
    ) -> AggregationSpec:
        cols = self._columns
        if cols is None and schema is not None:
            cols = [c for c, dt in schema.items() if dt.is_numeric()]
        if not cols or len(cols) < 2:
            raise ValueError(
                "CorrHeatmap requires at least 2 numeric columns; " f"got {cols!r}"
            )
        expr = _corr_expr(cols, self.method, self.absolute, self.uid)
        return AggregationSpec(expr=expr, uid=self.uid)

    def _to_update(self, df: pl.DataFrame) -> TraceResult:
        raw = df[self.uid][0]
        cols = list(raw["columns"])
        z_flat = list(raw["z_flat"])
        n = len(cols)
        z = [z_flat[j * n : (j + 1) * n] for j in range(n)]

        # Optional triangular mode: keep lower triangle (incl. diagonal)
        # in source-column order, hide mirrored upper-half cells.
        if self.triangular:
            for j in range(n):
                for i in range(j, n):
                    z[j][i] = None

        # Reverse rows so visual top→bottom order matches x-column order
        # on renderers where the first y-category appears at the bottom.
        y = list(reversed(cols))
        z = list(reversed(z))
        return TraceResult(updates={"x": cols, "y": y, "z": z})

    # ------------------------------------------------------------------
    # Cube descriptors (cross-filter pre-aggregation)
    # ------------------------------------------------------------------

    def get_cube_target_spec(
        self,
        axis_range: tuple[float, float] | None,
        schema: pl.Schema | None = None,
    ) -> "CubeTargetSpec | None":
        """A Pearson correlation heatmap is a ``corr`` cube target (contract I).

        The cube stores decomposable per-pair partials (mean-centered sums) over
        the brushed source's free axis with **empty target dims**; the client
        reslices and finalizes a Pearson ``r`` per pair, assembling the same
        matrix ``CorrHeatmap._to_update`` produces. ``triangular``/``absolute``
        are display params applied at finalize, NOT part of the cube — two
        heatmaps differing only there share one cube.

        Gates (any failure ⇒ ``None`` ⇒ legacy server recompute):

        * ``method == "pearson"`` only — Spearman is rank-based and not
          decomposable (spec §4).
        * ``columns`` must be passed **explicitly** with ≥2 entries. When
          ``columns is None`` the server resolves them from the schema, but the
          client (which never sees the schema) cannot reproduce that list to
          build its store key — so **pass ``columns`` explicitly to
          cube-accelerate corr**; otherwise this falls back to the legacy POST
          path.
        * Every column must be numeric (a ``schema`` is therefore required) and
          must not collide with a reserved cube partial-column name.
        """
        if self.method != "pearson":
            return None
        cols = self._columns
        if cols is None or len(cols) < 2:
            return None
        if schema is None:
            return None
        for c in cols:
            if c in _CUBE_RESERVED_COLS:
                return None
            dtype = _dtype_for_col(schema, c)
            if dtype is None or not dtype.is_numeric():
                return None
        return CubeTargetSpec(
            target_dims=(),
            measure=MeasureSpec(agg="corr", columns=tuple(cols)),
        )

    # ------------------------------------------------------------------
    # Spec reconstruction
    # ------------------------------------------------------------------

    @classmethod
    def from_trace_spec(cls, spec: TraceSpec) -> "CorrHeatmap":
        trace = cls(
            columns=spec.params.get("columns"),
            method=spec.params["method"],
            triangular=spec.params["triangular"],
            absolute=spec.params["absolute"],
            name=spec.display.get("name"),
            color_scale=spec.display.get("color_scale"),
            color_range=spec.display.get("color_range"),
        )
        trace.uid = spec.uid
        return trace


def _corr_expr(
    cols: list[str],
    method: str,
    absolute: bool,
    uid: str,
) -> pl.Expr:
    """Build a Polars expression that computes a correlation matrix.

    Uses ``pl.corr(a, b, method=...)`` for each pair and packs results
    into a struct with keys ``columns`` and ``z_flat`` (row-major
    symmetric matrix).
    """
    from itertools import combinations

    n = len(cols)
    pair_exprs: list[pl.Expr] = []
    for c1, c2 in combinations(cols, 2):
        e = pl.corr(c1, c2, method=method)
        if absolute:
            e = e.abs()
        pair_exprs.append(e.alias(f"_fv_{c1}__vs__{c2}"))

    def _pack(s: pl.Series) -> pl.Series:
        """Unpack pair correlations into a full symmetric matrix."""
        df = s.struct.unnest()
        matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            matrix[i][i] = 1.0
        pair_idx = 0
        for i, j in combinations(range(n), 2):
            val = df.row(0)[pair_idx]
            if val is None:
                val = 0.0
            matrix[i][j] = val
            matrix[j][i] = val
            pair_idx += 1

        z_flat: list[float] = []
        for row in matrix:
            z_flat.extend(row)

        return pl.DataFrame({"columns": [cols], "z_flat": [z_flat]}).to_struct("corr")

    return_dtype = pl.Struct(
        {
            "columns": pl.List(pl.String),
            "z_flat": pl.List(pl.Float64),
        }
    )
    return (
        pl.struct(*pair_exprs).map_batches(_pack, return_dtype=return_dtype).alias(uid)
    )

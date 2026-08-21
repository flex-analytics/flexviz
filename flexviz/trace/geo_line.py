"""GeoLine — renderer-agnostic geo line trace.

Downsamples a lat/lon path to at most ``n_points`` visible in the current
map viewport using the **every-nth** strategy, then optionally inserts
``None`` gap markers where consecutive points are unusually far apart.

Aggregation strategy
--------------------
Only ``"nth"`` (uniform-stride gather) is supported: every
``max(1, n // n_points)``-th row is kept.  Stride is computed inside the
``flexviz_polars`` Rust kernel — no ``len()`` expression dependency, enabling
full parallelism within a single ``select()`` call.

Map viewport
------------
When a map viewport is present in ``update_range["coordinates"]`` (a list of
``[lon, lat]`` corner points), points outside the bounding box are filtered
out *before* stride selection, so the returned points are all within view.

Gap detection
-------------
When ``add_gaps=True`` (default), consecutive points whose Euclidean
degree-space distance exceeds ``4.1 × median_distance`` are separated by a
``(None, None)`` lat/lon pair, breaking the rendered line at discontinuities
(same strategy as ``LinePlot``).

Cross-filter convention
-----------------------
``filter_selection`` maps ``sel_dict["x"]`` → longitude column and
``sel_dict["y"]`` → latitude column, following the Plotly convention where
the map x-axis is longitude and the y-axis is latitude.

Assumption
----------
We assume the dataframe is already in the desired sort order (same as
``LinePlot``).
"""

from __future__ import annotations

from typing import Any, Dict

import polars as pl

from ..LF import AggregationSpec
from ..spec import TraceSpec
from .base import FlexTrace, TraceResult, _range_filter_expr

import flexviz_polars as _fvp  # noqa: F401 — registers pl.Expr.flexviz namespace

# ---------------------------------------------------------------------------
# Aggregation expression builder
# ---------------------------------------------------------------------------


def _geo_line_nth_agg_expr(
    lat_col: str,
    lon_col: str,
    vp_expr: pl.Expr | None,
    n_points: int,
    uid: str,
) -> pl.Expr:
    """Single-pass every-nth downsampling using the flexviz_polars Rust kernel.

    Stride is computed inside the kernel — no ``len()`` expression dependency,
    so the aggregation can parallelize with other traces in the same
    ``select()`` call.

    Parameters
    ----------
    lat_col:
        Column name for latitude.
    lon_col:
        Column name for longitude.
    vp_expr:
        Optional combined boolean filter expression (lat AND lon bounding
        box).  Applied via ``.filter()`` before stride selection.
    n_points:
        Maximum number of points to return.
    uid:
        Alias for the output column (trace uid).
    """
    lat = pl.col(lat_col)
    lon = pl.col(lon_col)
    if vp_expr is not None:
        lat = lat.filter(vp_expr)
        lon = lon.filter(vp_expr)
    return (
        pl.struct(
            **{
                lat_col: lat.flexviz.every_nth(n_points),
                lon_col: lon.flexviz.every_nth(n_points),
            }
        )
        .implode()
        .alias(uid)
    )


# ---------------------------------------------------------------------------
# Viewport helper
# ---------------------------------------------------------------------------


def _extract_lat_lon_range(
    update_range: Dict[str, Any],
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """Extract a lat/lon bounding box from a map viewport.

    The viewport is expected as ``update_range["coordinates"]``: a list of
    ``[lon, lat]`` corner points (Plotly / Leaflet format).

    Returns
    -------
    (lat_range, lon_range)
        Each is a ``(min, max)`` float tuple, or ``None`` when no viewport
        coordinates are present.
    """
    coordinates = update_range.get("coordinates")
    if not coordinates:
        return None, None
    lons = [c[0] for c in coordinates]
    lats = [c[1] for c in coordinates]
    return (min(lats), max(lats)), (min(lons), max(lons))


# ---------------------------------------------------------------------------
# GeoLine trace
# ---------------------------------------------------------------------------


class GeoLine(FlexTrace):
    """Scalable geo line trace backed by a Polars LazyFrame.

    Downsamples a lat/lon path to ``n_points`` using uniform-stride
    (every-nth) selection, optionally restricted to the current map viewport.
    Consecutive points that are far apart (relative to the median step size)
    are separated by ``None`` gap markers so the rendered line breaks at
    discontinuities.

    Parameters
    ----------
    lat:
        Column name for latitude.
    lon:
        Column name for longitude.
    n_points:
        Maximum number of points returned per viewport update.
    name:
        Legend / series name passed to the renderer.
    color:
        Line colour hint (CSS string), passed to the renderer.
    add_gaps:
        When ``True`` (default), inserts ``None`` separators between
        consecutive points whose degree-space distance exceeds
        ``4.1 × median_distance``.
    update_on_zoom:
        When ``True`` (default) the engine re-aggregates this trace whenever
        the map viewport changes.
    """

    trace_type: str = "geo_line"
    select_policy_doc: str = "none — scattermap point selection not supported"
    recompute_policy_doc: str = "map coordinates (frozen if update_on_zoom=False)"

    def __init__(
        self,
        lat: str,
        lon: str,
        n_points: int = 1000,
        name: str | None = None,
        color: str | None = None,
        add_gaps: bool = True,
        update_on_zoom: bool = True,
    ) -> None:
        super().__init__(
            backend_data={"lat": lat, "lon": lon},
            display={
                "name": name or f"{lat}/{lon}",
                **({"color": color} if color is not None else {}),
            },
            params={
                "n_points": n_points,
                "add_gaps": add_gaps,
            },
            axes=None,  # non-cartesian, like GeoHistogram2D
            recompute_axes=None if update_on_zoom else (),
        )

    def _default_recompute_axes(self) -> tuple[str, ...]:
        return ("coordinates",)  # re-downsamples on each map viewport change

    # Selection geometry: inherits the base ``kind="none"``. A scattermap line
    # exposes points (coordinates), not GeoJSON features with location ids, so
    # the client's feature-bbox path cannot derive a selection — geo_line is not
    # a cross-filter source until scattermap point-bounds selection is wired.

    # ------------------------------------------------------------------
    # Properties (convenience access)
    # ------------------------------------------------------------------

    @property
    def lat_col(self) -> str:
        return self._backend_data["lat"]

    @property
    def lon_col(self) -> str:
        return self._backend_data["lon"]

    @property
    def n_points(self) -> int:
        return self._params["n_points"]

    @property
    def add_gaps(self) -> bool:
        return self._params["add_gaps"]

    # ------------------------------------------------------------------
    # FlexTrace interface
    # ------------------------------------------------------------------

    def get_aggregation_spec(
        self,
        update_range: Dict[str, Any],
        schema: pl.Schema | None = None,
    ) -> AggregationSpec:
        """Return an every-nth aggregation spec, optionally viewport-filtered."""
        lat_range, lon_range = _extract_lat_lon_range(update_range)

        vp_expr: pl.Expr | None = None
        if lat_range is not None and lon_range is not None:
            lat_f = _range_filter_expr(self.lat_col, lat_range, schema)
            lon_f = _range_filter_expr(self.lon_col, lon_range, schema)
            if lat_f is not None and lon_f is not None:
                vp_expr = lat_f & lon_f

        expr = _geo_line_nth_agg_expr(
            self.lat_col, self.lon_col, vp_expr, self.n_points, self.uid
        )
        return AggregationSpec(expr=expr, uid=self.uid)

    def _get_gap_mask(self, lat: pl.Series, lon: pl.Series) -> pl.Series | None:
        """Return a boolean mask where gaps should be inserted.

        A gap is detected where the Euclidean degree-space distance between
        consecutive points exceeds ``4.1 × median_distance`` — the same
        threshold used by ``LinePlot``.
        """
        if not self.add_gaps:
            return None
        if lat.len() < 2:
            return None

        lat_diff = lat.diff()
        lon_diff = lon.diff()

        if lat_diff.null_count() == lat_diff.len():
            return None

        dist = (lat_diff**2 + lon_diff**2).sqrt()
        med_dist = dist.drop_nulls().median()
        if med_dist is None or med_dist == 0:
            return None

        return dist > 4.1 * med_dist

    def _to_update(self, df_agg: pl.DataFrame) -> TraceResult:
        """Unpack the aggregated imploded struct column → ``{"lat": series, "lon": series}``."""
        raw: pl.Series = df_agg[self.uid].item()
        df_points = raw.explode().struct.unnest()
        lat = df_points[self.lat_col]
        lon = df_points[self.lon_col]

        gap_mask = self._get_gap_mask(lat, lon)
        if gap_mask is not None and gap_mask.any():
            gap_positions = gap_mask.fill_null(False).arg_true()
            gap_rows = pl.DataFrame(
                {
                    self.lat_col: pl.Series(
                        [None] * len(gap_positions), dtype=lat.dtype
                    ),
                    self.lon_col: pl.Series(
                        [None] * len(gap_positions), dtype=lon.dtype
                    ),
                    "_sk": gap_positions.cast(pl.Float64) - 0.5,
                }
            )
            df_combined = (
                df_points.with_columns(
                    pl.int_range(0, len(df_points), eager=True)
                    .cast(pl.Float64)
                    .alias("_sk")
                )
                .vstack(gap_rows)
                .sort("_sk")
                .drop("_sk")
            )
            return TraceResult(
                updates={
                    "lat": df_combined[self.lat_col],
                    "lon": df_combined[self.lon_col],
                }
            )

        return TraceResult(updates={"lat": lat, "lon": lon})

    # ------------------------------------------------------------------
    # Spec reconstruction (server-side)
    # ------------------------------------------------------------------

    @classmethod
    def from_trace_spec(cls, spec: TraceSpec) -> "GeoLine":
        trace = cls(
            lat=spec.backend_data.get("lat", ""),
            lon=spec.backend_data.get("lon", ""),
            n_points=spec.params.get("n_points", 1000),
            name=spec.display.get("name"),
            color=spec.display.get("color"),
            add_gaps=spec.params.get("add_gaps", True),
            # Frozen only when the spec carries an explicit empty tuple.
            update_on_zoom=spec.recompute_axes != (),
        )
        if spec.recompute_axes is not None:
            trace._recompute_axes = tuple(spec.recompute_axes)
        trace.uid = spec.uid
        return trace

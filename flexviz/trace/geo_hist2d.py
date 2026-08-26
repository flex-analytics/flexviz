"""GeoHistogram2D — renderer-agnostic geospatial 2D histogram trace.

Bins latitude/longitude data into a 2D grid and computes per-bin aggregates.
Returns GeoJSON FeatureCollection rectangles with associated values, suitable
for rendering as a ``choroplethmap`` (Plotly) or similar geo heatmap.

Supported *histfunc* values: ``"count"`` (implicit when *z* is omitted),
``"sum"``, ``"mean"``, ``"min"``, ``"max"`` — all computed by the
``flexviz_polars`` Rust kernel.  ``"median"`` and ``"n_unique"`` are not
supported on this fast path (see Architecture.md roadmap).

Supported *histnorm* values: ``None`` (no normalization, default), ``"percent"``,
``"probability"``, ``"density"``, ``"probability density"``.

Viewport filtering
------------------
``recompute_axes = ("coordinates",)`` — recomputed on each map viewport change.
The viewport is expected as ``update_range["coordinates"]``, a list of
``[lon, lat]`` corner points describing the visible bounding box.

Cross-filter convention
-----------------------
``filter_selection`` maps ``sel_dict["x"]`` → longitude column and
``sel_dict["y"]`` → latitude column, following the Plotly convention where
the map x-axis is longitude and the y-axis is latitude.  This convention is
enforced by the Plotly adapter when building ``SelectionState``.
"""

from __future__ import annotations

from typing import Any, Dict, Literal

import polars as pl

from ..cube import _fixed_hist2d_bin_expr
from ..LF import AggregationSpec
from ..spec import TraceHoverSpec, TraceSelectionSpec, TraceSpec
from .base import FlexTrace, TraceResult, _typed_range_bounds
from ._hist_helpers import (
    HeatmapColorRange,
    _HISTNORM_OPTIONS,
    apply_histnorm,
    normalize_heatmap_color_scale,
    normalize_heatmap_color_range,
)

import flexviz_polars  # noqa: F401 — registers pl.Expr.flexviz namespace

_DEFAULT_COLOR_SCALE = "viridis"
_DEFAULT_COLOR_RANGE: HeatmapColorRange = "auto"

_BIN_BOUNDARIES_OPTIONS = ("data", "viewport")
# Reducers backed by the flexviz_polars Rust kernel (fixed_hist2d_reduce).
# `median` and `n_unique` are intentionally unsupported on this fast path —
# mirrors Histogram2D. See the roadmap note in Architecture.md.
_GEO_HIST2D_HISTFUNC_OPTIONS = ("sum", "mean", "min", "max")


class GeoHistogram2D(FlexTrace):
    """Scalable geospatial 2D histogram trace.

    Parameters
    ----------
    lat:
        Column name for latitude.
    lon:
        Column name for longitude.
    lat_bins:
        Number of bins along latitude (default 64).
    lon_bins:
        Number of bins along longitude (default 64).
    z:
        Column name for the value to aggregate per bin.  When ``None``
        (default) the trace counts rows per bin.
    histfunc:
        Aggregation function applied to ``z``.  Required when ``z`` is
        given; must be one of ``"sum"``, ``"mean"``, ``"min"``, ``"max"``.
        Forbidden when ``z`` is ``None``.
    histnorm:
        Normalization applied after aggregation.
    name:
        Legend / series name.
    bin_boundaries:
        How bin edges are chosen when a map viewport is present:

        - ``"data"`` (default): after filtering to the viewport, bin edges span
          the min/max of the visible points (bins shift when panning/zooming).
        - ``"viewport"``: bin edges span the viewport bounds (stable grid on
          pan/zoom, like plotly-flex ``bin_boundaries="viewport"``).
    """

    trace_type: str = "geo_histogram2d"
    select_policy_doc: str = "map box — (lon, lat) bounds"
    recompute_policy_doc: str = "map coordinates — re-bins on viewport change"
    overlay_style: str = "filtered_only"

    def __init__(
        self,
        lat: str,
        lon: str,
        lat_bins: int = 64,
        lon_bins: int = 64,
        z: str | None = None,
        histfunc: str | None = None,
        histnorm: str | None = None,
        name: str | None = None,
        color_scale: str | None = None,
        color_range: tuple[float, float] | str | None = None,
        bin_boundaries: Literal["data", "viewport"] = "data",
    ) -> None:
        if z is None and histfunc is not None:
            raise ValueError("histfunc is only meaningful when z is given.")
        if z is not None and histfunc is None:
            raise ValueError("histfunc is required when z is given.")
        if z is not None and histfunc not in _GEO_HIST2D_HISTFUNC_OPTIONS:
            raise ValueError(f"histfunc must be one of {_GEO_HIST2D_HISTFUNC_OPTIONS}.")
        if histnorm not in _HISTNORM_OPTIONS:
            raise ValueError(f"histnorm must be one of {_HISTNORM_OPTIONS}.")
        if bin_boundaries not in _BIN_BOUNDARIES_OPTIONS:
            raise ValueError(
                f"bin_boundaries must be one of {_BIN_BOUNDARIES_OPTIONS}, "
                f"got {bin_boundaries!r}"
            )

        backend_data: Dict[str, str] = {"lat": lat, "lon": lon}
        if z is not None:
            backend_data["z"] = z

        super().__init__(
            backend_data=backend_data,
            display={
                "name": name or f"geo {lat} x {lon}",
                "color_scale": normalize_heatmap_color_scale(
                    color_scale, _DEFAULT_COLOR_SCALE, trace_name="GeoHistogram2D"
                ),
                "color_range": normalize_heatmap_color_range(
                    color_range, _DEFAULT_COLOR_RANGE, trace_name="GeoHistogram2D"
                ),
            },
            params={
                "lat_bins": lat_bins,
                "lon_bins": lon_bins,
                "histfunc": histfunc,
                "histnorm": histnorm,
                "bin_boundaries": bin_boundaries,
            },
            axes=None,
        )

    def _default_recompute_axes(self) -> tuple[str, ...]:
        return ("coordinates",)  # re-bins on each map viewport change

    def _make_selection_spec(self) -> TraceSelectionSpec:
        # Map box → lon/lat range clauses from the hit feature's bounding box.
        return TraceSelectionSpec(
            kind="geo_box",
            lon_column=self._backend_data["lon"],
            lat_column=self._backend_data["lat"],
        )

    def _make_hover_spec(self) -> "TraceHoverSpec":
        return TraceHoverSpec(
            source_modes=["cell"],
            target_modes=["cell"],
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def lat_col(self) -> str:
        return self._backend_data["lat"]

    @property
    def lon_col(self) -> str:
        return self._backend_data["lon"]

    @property
    def z_col(self) -> str | None:
        return self._backend_data.get("z")

    @property
    def lat_bins(self) -> int:
        return self._params["lat_bins"]

    @property
    def lon_bins(self) -> int:
        return self._params["lon_bins"]

    @property
    def histfunc(self) -> str | None:
        return self._params["histfunc"]

    @property
    def histnorm(self) -> str | None:
        return self._params["histnorm"]

    @property
    def bin_boundaries(self) -> Literal["data", "viewport"]:
        return self._params["bin_boundaries"]

    @property
    def color_scale(self) -> str:
        return self._display["color_scale"]

    @property
    def color_range(self) -> HeatmapColorRange:
        return self._display["color_range"]

    # ------------------------------------------------------------------
    # Viewport helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_lat_lon_range(
        update_range: Dict[str, Any],
    ) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
        """Extract lat/lon bounding box from a map viewport.

        The viewport is expected as ``update_range["coordinates"]``: a list
        of ``[lon, lat]`` corner points.  Returns ``(lat_range, lon_range)``
        or ``(None, None)`` when no viewport is available.
        """
        coordinates = update_range.get("coordinates")
        if not coordinates:
            return None, None
        lons = [c[0] for c in coordinates]
        lats = [c[1] for c in coordinates]
        return (min(lats), max(lats)), (min(lons), max(lons))

    # ------------------------------------------------------------------
    # FlexTrace interface
    # ------------------------------------------------------------------

    def get_aggregation_spec(
        self,
        update_range: Dict[str, Any],
        schema: pl.Schema | None = None,
        *,
        scan_source: bool = False,
    ) -> AggregationSpec:
        lat_range, lon_range = self._extract_lat_lon_range(update_range)
        if scan_source:
            # The kernels materialize both columns; a scan swaps in a streaming
            # plan computing the same cells (see _native_geo_hist2d_plan).
            return AggregationSpec(
                expr=pl.lit(None).alias(self.uid),
                uid=self.uid,
                plan=_native_geo_hist2d_plan(
                    self.lat_col,
                    self.lon_col,
                    self.lat_bins,
                    self.lon_bins,
                    lat_range,
                    lon_range,
                    self.uid,
                    self.histfunc,
                    self.z_col,
                    self.bin_boundaries,
                    schema,
                ),
            )
        expr = _geo_hist2d_expr(
            self.lat_col,
            self.lon_col,
            self.lat_bins,
            self.lon_bins,
            lat_range,
            lon_range,
            self.uid,
            histfunc=self.histfunc,
            z_col=self.z_col,
            bin_boundaries=self.bin_boundaries,
            schema=schema,
        )
        return AggregationSpec(expr=expr, uid=self.uid)

    def _to_update(self, df: pl.DataFrame) -> TraceResult:
        raw = df[self.uid][0]
        nb_lat = self.lat_bins
        nb_lon = self.lon_bins

        # Rust kernel output: Struct{z_flat, x_lo, x_hi, y_lo, y_hi}.
        # We map lat -> kernel x (inner axis), lon -> kernel y (outer axis), so
        # z_flat is laid out as z_flat[lon_idx * nb_lat + lat_idx] — exactly the
        # row-major (lon-major) order _build_geojson_rectangles expects.
        z_flat_raw: list = raw["z_flat"]
        lat_lo: float = raw["x_lo"]
        lat_hi: float = raw["x_hi"]
        lon_lo: float = raw["y_lo"]
        lon_hi: float = raw["y_hi"]

        if self.z_col is None:
            # Count kernel returns UInt32; 0 marks an empty bin → emit None so
            # _build_geojson_rectangles skips it (no rectangle drawn).
            z_flat = [None if v == 0 else float(v) for v in z_flat_raw]
        else:
            # Reducer kernel returns nullable Float64 values directly.
            z_flat = [None if v is None else float(v) for v in z_flat_raw]

        lat_step = (lat_hi - lat_lo) / nb_lat
        lon_step = (lon_hi - lon_lo) / nb_lon

        if self.histnorm is not None:
            z_series = pl.Series("value", z_flat, dtype=pl.Float64)
            z_df = apply_histnorm(
                pl.DataFrame({"value": z_series}),
                "value",
                self.histnorm,
                lat_step * lon_step,
            )
            z_flat = z_df["value"].to_list()

        lat_centers = [lat_lo + (i + 0.5) * lat_step for i in range(nb_lat)]
        lon_centers = [lon_lo + (j + 0.5) * lon_step for j in range(nb_lon)]
        lat_edges = [lat_lo + i * lat_step for i in range(nb_lat + 1)]
        lon_edges = [lon_lo + j * lon_step for j in range(nb_lon + 1)]

        geojson, locations, values = _build_geojson_rectangles(
            lat_centers,
            lon_centers,
            lat_edges,
            lon_edges,
            z_flat,
        )

        return TraceResult(
            updates={
                "geojson": geojson,
                "locations": locations,
                "z": values,
            }
        )

    # ------------------------------------------------------------------
    # Spec reconstruction
    # ------------------------------------------------------------------

    @classmethod
    def from_trace_spec(cls, spec: TraceSpec) -> "GeoHistogram2D":
        z = spec.backend_data.get("z")
        raw_histfunc = spec.params.get("histfunc")
        # Backward compat: old specs stored histfunc="count" when z was None.
        if raw_histfunc == "count" or raw_histfunc is None:
            histfunc = None
        elif raw_histfunc in ("median", "n_unique"):
            raise ValueError(
                f"histfunc={raw_histfunc!r} is no longer supported by "
                f"GeoHistogram2D (removed in favour of the Rust kernel). "
                f"Use one of: {_GEO_HIST2D_HISTFUNC_OPTIONS}."
            )
        else:
            histfunc = raw_histfunc
        trace = cls(
            lat=spec.backend_data["lat"],
            lon=spec.backend_data["lon"],
            lat_bins=spec.params.get("lat_bins", 64),
            lon_bins=spec.params.get("lon_bins", 64),
            z=z,
            histfunc=histfunc if z is not None else None,
            histnorm=spec.params.get("histnorm"),
            name=spec.display.get("name"),
            color_scale=spec.display.get("color_scale"),
            color_range=spec.display.get("color_range"),
            bin_boundaries=spec.params.get("bin_boundaries", "data"),
        )
        trace.uid = spec.uid
        return trace


# ---------------------------------------------------------------------------
# Pure-Polars expression builder
# ---------------------------------------------------------------------------


def _geo_hist2d_expr(
    lat_col: str,
    lon_col: str,
    nb_lat: int,
    nb_lon: int,
    lat_range: tuple[float, float] | None,
    lon_range: tuple[float, float] | None,
    uid: str,
    histfunc: str | None = None,
    z_col: str | None = None,
    bin_boundaries: Literal["data", "viewport"] = "data",
    schema: pl.Schema | None = None,
) -> pl.Expr:
    """Build a geo 2D histogram expression backed by the Rust kernel.

    lat is mapped to the kernel's x (inner) axis and lon to its y (outer) axis,
    so the kernel's row-major ``z_flat[yi * nb_x + xi]`` becomes
    ``z_flat[lon_idx * nb_lat + lat_idx]`` — the layout the GeoJSON builder
    expects.  The expression aliases to *uid* and yields a length-1 struct
    ``{z_flat, x_lo, x_hi, y_lo, y_hi}`` (x = lat bounds, y = lon bounds).

    Bin edges follow ``bin_boundaries``:

    - ``"viewport"`` (with a viewport): edges span the viewport rectangle, so
      the grid is stable on pan/zoom.
    - ``"data"`` (default): edges span the min/max of the visible (viewport- and
      cross-filter-filtered) points, matching the prior pure-Polars behaviour.
    """
    has_viewport = lat_range is not None and lon_range is not None

    if has_viewport:
        # Cast the viewport bounds to each column's dtype so the is_between
        # comparison runs in the column's native type. With raw f64 Python
        # floats, Polars widens every f32/integer element to f64 for the
        # comparison (no f32 SIMD), which is ~4-9x slower on the hot pan/zoom
        # path. Mirrors Histogram2D's _hist2d_bounds. The integer ceil/floor
        # logic in _typed_range_bounds also keeps is_between semantics correct.
        lat_bounds = _typed_range_bounds(lat_col, (lat_range[0], lat_range[1]), schema)
        lon_bounds = _typed_range_bounds(lon_col, (lon_range[0], lon_range[1]), schema)
        mask = pl.col(lat_col).is_between(*lat_bounds) & pl.col(lon_col).is_between(
            *lon_bounds
        )
        lat_expr = pl.col(lat_col).filter(mask)
        lon_expr = pl.col(lon_col).filter(mask)
    else:
        lat_expr = pl.col(lat_col)
        lon_expr = pl.col(lon_col)

    if bin_boundaries == "viewport" and has_viewport:
        lat_lo: pl.Expr = pl.lit(float(lat_range[0]))
        lat_hi: pl.Expr = pl.lit(float(lat_range[1]))
        lon_lo: pl.Expr = pl.lit(float(lon_range[0]))
        lon_hi: pl.Expr = pl.lit(float(lon_range[1]))
    else:
        # "data" mode: bins span the (filtered) data extent. fill_null keeps the
        # kernel's lo <= hi contract satisfied when no rows survive filtering.
        lat_lo = lat_expr.min().fill_null(0.0)
        lat_hi = lat_expr.max().fill_null(1.0)
        lon_lo = lon_expr.min().fill_null(0.0)
        lon_hi = lon_expr.max().fill_null(1.0)

    if z_col is None:
        expr = lat_expr.flexviz.fixed_hist2d(
            lon_expr, lat_lo, lat_hi, lon_lo, lon_hi, nb_lat, nb_lon
        )
    else:
        assert histfunc is not None
        z_expr = pl.col(z_col).filter(mask) if has_viewport else pl.col(z_col)
        expr = lat_expr.flexviz.fixed_hist2d_reduce(
            lon_expr, z_expr, lat_lo, lat_hi, lon_lo, lon_hi, nb_lat, nb_lon, histfunc
        )
    return expr.alias(uid)


def _native_geo_hist2d_plan(
    lat_col: str,
    lon_col: str,
    nb_lat: int,
    nb_lon: int,
    lat_range: tuple | None,
    lon_range: tuple | None,
    uid: str,
    histfunc: str | None,
    z_col: str | None,
    bin_boundaries: str,
    schema: pl.Schema | None,
):
    """Streaming replacement of ``_geo_hist2d_expr`` for scan sources.

    Same cells as the kernels via a bounded streaming ``group_by`` over the
    two bin indices (see ``_native_hist2d_plan`` in ``hist2d.py`` for the
    equivalence argument; lat maps to kernel x, lon to kernel y, so
    ``z_flat[lon_bin * nb_lat + lat_bin]`` keeps the lon-major order).

    ``"data"`` bin boundaries need the filtered lat/lon extent before binning,
    which on a scan is one extra bounded pass; the bounds come from the same
    ``min``/``max`` expressions the kernel path evaluates, computed *before*
    the NaN drop so both paths see identical extents.
    """

    def run(
        filtered_ldf: pl.LazyFrame, stats_row: pl.DataFrame | None = None
    ) -> pl.DataFrame:
        has_viewport = lat_range is not None and lon_range is not None
        src = filtered_ldf
        if has_viewport:
            lat_bounds = _typed_range_bounds(
                lat_col, (lat_range[0], lat_range[1]), schema
            )
            lon_bounds = _typed_range_bounds(
                lon_col, (lon_range[0], lon_range[1]), schema
            )
            src = src.filter(
                pl.col(lat_col).is_between(*lat_bounds)
                & pl.col(lon_col).is_between(*lon_bounds)
            )
        proj = [
            pl.col(lat_col).cast(pl.Float64).alias("__lat"),
            pl.col(lon_col).cast(pl.Float64).alias("__lon"),
        ]
        keep = pl.col("__lat").is_not_nan() & pl.col("__lon").is_not_nan()
        if z_col is None:
            agg = pl.len().alias("__agg")
        else:
            assert histfunc is not None
            proj.append(pl.col(z_col).cast(pl.Float64).alias("__z"))
            keep = keep & pl.col("__z").is_not_nan()
            agg = getattr(pl.col("__z"), histfunc)().alias("__agg")
        base = src.select(*proj)

        if bin_boundaries == "viewport" and has_viewport:
            lat_lo, lat_hi = float(lat_range[0]), float(lat_range[1])
            lon_lo, lon_hi = float(lon_range[0]), float(lon_range[1])
        else:
            bounds = base.select(
                pl.col("__lat").min().alias("lat_lo"),
                pl.col("__lat").max().alias("lat_hi"),
                pl.col("__lon").min().alias("lon_lo"),
                pl.col("__lon").max().alias("lon_hi"),
            ).collect(engine="streaming")

            def _b(name: str, default: float) -> float:
                val = bounds[name][0]
                return default if val is None else float(val)

            lat_lo, lat_hi = _b("lat_lo", 0.0), _b("lat_hi", 1.0)
            lon_lo, lon_hi = _b("lon_lo", 0.0), _b("lon_hi", 1.0)

        rows = (
            base.filter(keep)
            .group_by(
                _fixed_hist2d_bin_expr(pl.col("__lat"), lat_lo, lat_hi, nb_lat, "__ba"),
                _fixed_hist2d_bin_expr(pl.col("__lon"), lon_lo, lon_hi, nb_lon, "__bo"),
            )
            .agg(agg)
            .collect(engine="streaming")
        )
        n_cells = nb_lat * nb_lon
        z_flat: list = [0] * n_cells if z_col is None else [None] * n_cells
        for ba, bo, val in rows.iter_rows():
            z_flat[bo * nb_lat + ba] = val
        out_schema = pl.Struct(
            {
                "z_flat": pl.List(pl.UInt32 if z_col is None else pl.Float64),
                "x_lo": pl.Float64,
                "x_hi": pl.Float64,
                "y_lo": pl.Float64,
                "y_hi": pl.Float64,
            }
        )
        return pl.DataFrame(
            {
                uid: [
                    {
                        "z_flat": z_flat,
                        "x_lo": lat_lo,
                        "x_hi": lat_hi,
                        "y_lo": lon_lo,
                        "y_hi": lon_hi,
                    }
                ]
            },
            schema={uid: out_schema},
        )

    return run


# ---------------------------------------------------------------------------
# GeoJSON rectangle builder
# ---------------------------------------------------------------------------


def _build_geojson_rectangles(
    lat_centers: list[float],
    lon_centers: list[float],
    lat_edges: list[float],
    lon_edges: list[float],
    z_flat: list,
) -> tuple[dict, list, list]:
    """Build a GeoJSON FeatureCollection of rectangle polygons.

    *z_flat* is in row-major (lon-major) order: ``z_flat[j * nb_lat + i]``
    corresponds to bin ``(lat_i, lon_j)``.

    Returns ``(geojson, locations, values)`` where *locations* are feature IDs
    and *values* are the filtered (non-null) z values.
    """
    nb_lat = len(lat_centers)
    nb_lon = len(lon_centers)
    features: list[dict] = []
    locations: list[str] = []
    values: list[float] = []

    for j in range(nb_lon):
        lon_left = lon_edges[j]
        lon_right = lon_edges[j + 1]
        for i in range(nb_lat):
            idx = j * nb_lat + i
            val = z_flat[idx]
            if val is None:
                continue

            lat_bottom = lat_edges[i]
            lat_top = lat_edges[i + 1]
            bin_id = f"r{i}_c{j}"
            locations.append(bin_id)
            values.append(val)

            coordinates = [
                [
                    [lon_left, lat_bottom],
                    [lon_right, lat_bottom],
                    [lon_right, lat_top],
                    [lon_left, lat_top],
                    [lon_left, lat_bottom],
                ]
            ]
            features.append(
                {
                    "type": "Feature",
                    "id": bin_id,
                    "geometry": {"type": "Polygon", "coordinates": coordinates},
                }
            )

    geojson = {"type": "FeatureCollection", "features": features}
    return geojson, locations, values

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
``[lon, lat]`` corner points describing the visible bounding box.  Unzoomed,
the bin edges span the engine-resolved data range.  Zoomed, they span the
viewport snapped outward to a fixed lattice, so the grid stands still while the
user pans.

Cross-filter convention
-----------------------
``filter_selection`` maps ``sel_dict["x"]`` → longitude column and
``sel_dict["y"]`` → latitude column, following the Plotly convention where
the map x-axis is longitude and the y-axis is latitude.  This convention is
enforced by the Plotly adapter when building ``SelectionState``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict

import polars as pl

from ..LF import AggregationSpec
from ..spec import TraceHoverSpec, TraceSelectionSpec, TraceSpec
from .base import FlexTrace, TraceResult
from ._hist_helpers import (
    HeatmapColorRange,
    _HISTNORM_OPTIONS,
    normalize_heatmap_color_scale,
    normalize_heatmap_color_range,
)
from .hist2d import hist2d_agg_spec, unpack_hist2d_grid

_DEFAULT_COLOR_SCALE = "viridis"
_DEFAULT_COLOR_RANGE: HeatmapColorRange = "auto"

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
        Number of bins along latitude (default 64). A zoomed axis can show one more,
        because the grid snaps to a fixed lattice.
    lon_bins:
        Number of bins along longitude (default 64). A zoomed axis can show one more,
        because the grid snaps to a fixed lattice.
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
    ) -> None:
        if z is None and histfunc is not None:
            raise ValueError("histfunc is only meaningful when z is given.")
        if z is not None and histfunc is None:
            raise ValueError("histfunc is required when z is given.")
        if z is not None and histfunc not in _GEO_HIST2D_HISTFUNC_OPTIONS:
            raise ValueError(f"histfunc must be one of {_GEO_HIST2D_HISTFUNC_OPTIONS}.")
        if histnorm not in _HISTNORM_OPTIONS:
            raise ValueError(f"histnorm must be one of {_HISTNORM_OPTIONS}.")

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
            },
            axes=None,
        )
        # The grid the last request actually binned on: a zoomed request snaps
        # its edges to a lattice, which can add one bin per axis. _to_update
        # unpacks z_flat with this, not with the configured bin counts.
        self._grid: tuple[int, int] = (lat_bins, lon_bins)

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

    def domain_cols(self, update_range: Dict[str, Any]) -> tuple[str, ...]:
        # A map viewport supplies both bounds at once, so it is all or nothing.
        if update_range.get("coordinates"):
            return ()
        return (self.lat_col, self.lon_col)

    def get_aggregation_spec(
        self,
        update_range: Dict[str, Any],
        schema: pl.Schema | None = None,
        *,
        domains: Mapping[str, tuple[Any, Any]] | None = None,
        scan_source: bool = False,
        **_: Any,
    ) -> AggregationSpec:
        """Return the geo 2-D histogram aggregation spec.

        lat maps to the kernel x (inner) axis and lon to its y (outer) axis, so
        ``z_flat`` comes back in the lon-major order the GeoJSON builder wants.
        Binning itself is the shared path (see ``hist2d_agg_spec``).
        """
        lat_range, lon_range = self._extract_lat_lon_range(update_range)
        spec, self._grid = hist2d_agg_spec(
            self.lat_col,
            self.lon_col,
            self.z_col,
            self.histfunc,
            lat_range,
            lon_range,
            self.lat_bins,
            self.lon_bins,
            self.uid,
            domains,
            schema,
            scan_source,
        )
        return spec

    def _to_update(self, df: pl.DataFrame) -> TraceResult:
        nb_lat, nb_lon = self._grid
        # lat maps to the kernel x (inner) axis and lon to its y (outer) axis,
        # so z_flat is laid out as z_flat[lon_idx * nb_lat + lat_idx] — exactly
        # the row-major (lon-major) order the client's rectangle builder wants.
        z_flat, lat_lo, _, lon_lo, _, lat_step, lon_step = unpack_hist2d_grid(
            df[self.uid][0],
            self._grid,
            counts=self.z_col is None,
            histnorm=self.histnorm,
        )

        # The client builds one GeoJSON rectangle per non-empty cell from these
        # triples and the flat z; the rectangles are most of a geo response.
        return TraceResult(
            updates={
                "lat_edges": [lat_lo, lat_step, nb_lat],
                "lon_edges": [lon_lo, lon_step, nb_lon],
                "z": z_flat,
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
        )
        trace.uid = spec.uid
        return trace

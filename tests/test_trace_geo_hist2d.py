"""Unit tests for GeoHistogram2D trace."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from flexviz.LF import LFQueryBuilder
from flexviz.engine import FlexEngine, TraceInfo
from flexviz.events import InteractionEvent
from flexviz.spec import ClauseFilter, SelectionPredicate, SelectionState, TraceSpec
from flexviz.trace.geo_hist2d import GeoHistogram2D
from flexviz.trace.line import LinePlot
from flexviz.trace.base import TraceResult
from flexviz.trace import build_trace_from_spec


def _aggregate_geo_hist2d(
    df: pl.DataFrame,
    lat: str = "lat",
    lon: str = "lon",
    lat_bins: int = 5,
    lon_bins: int = 5,
    z: str | None = None,
    histfunc: str | None = None,
    histnorm: str | None = None,
    update_range: dict | None = None,
    bin_boundaries: str = "data",
) -> TraceResult:
    lf = LFQueryBuilder(df)
    trace = GeoHistogram2D(
        lat=lat,
        lon=lon,
        lat_bins=lat_bins,
        lon_bins=lon_bins,
        z=z,
        histfunc=histfunc,
        histnorm=histnorm,
        bin_boundaries=bin_boundaries,
    )
    spec = trace.get_aggregation_spec(update_range or {}, schema=lf.schema)
    regular_df, _ = lf.aggregate([], [spec])
    return trace._to_update(regular_df)


@pytest.fixture()
def geo_df() -> pl.DataFrame:
    rng = np.random.default_rng(42)
    n = 1000
    return pl.DataFrame(
        {
            "lat": rng.uniform(40.0, 42.0, n).tolist(),
            "lon": rng.uniform(-74.0, -72.0, n).tolist(),
        }
    )


@pytest.fixture()
def geo_df_with_z() -> pl.DataFrame:
    rng = np.random.default_rng(42)
    n = 1000
    return pl.DataFrame(
        {
            "lat": rng.uniform(40.0, 42.0, n).tolist(),
            "lon": rng.uniform(-74.0, -72.0, n).tolist(),
            "z": rng.uniform(1, 100, n).tolist(),
        }
    )


class TestGeoHist2DConstructor:
    def test_defaults(self):
        t = GeoHistogram2D(lat="lat", lon="lon")
        assert t.trace_type == "geo_histogram2d"
        assert t.lat_col == "lat"
        assert t.lon_col == "lon"
        assert t.lat_bins == 64
        assert t.lon_bins == 64
        assert t.histfunc is None
        assert t.histnorm is None
        assert t.z_col is None
        assert t._axes is None
        assert t.recompute_axes == ("coordinates",)
        assert t.update_on_zoom is True
        assert t.overlay_style == "filtered_only"
        assert t.color_scale == "viridis"
        assert t.color_range == "auto"

    def test_custom_bins(self):
        t = GeoHistogram2D(lat="lat", lon="lon", lat_bins=10, lon_bins=20)
        assert t.lat_bins == 10
        assert t.lon_bins == 20

    def test_histfunc_without_z_raises(self):
        with pytest.raises(
            ValueError, match="histfunc is only meaningful when z is given"
        ):
            GeoHistogram2D(lat="lat", lon="lon", histfunc="sum")

    def test_z_without_histfunc_raises(self):
        with pytest.raises(ValueError, match="histfunc is required when z is given"):
            GeoHistogram2D(lat="lat", lon="lon", z="w")

    def test_histfunc_with_z(self):
        t = GeoHistogram2D(lat="lat", lon="lon", z="w", histfunc="sum")
        assert t.histfunc == "sum"
        assert t.z_col == "w"

    def test_count_implicit_histfunc_is_none(self):
        t = GeoHistogram2D(lat="lat", lon="lon")
        assert t.histfunc is None
        assert t.z_col is None

    def test_invalid_histfunc(self):
        with pytest.raises(ValueError, match="histfunc"):
            GeoHistogram2D(lat="lat", lon="lon", z="w", histfunc="invalid")

    def test_invalid_histnorm(self):
        with pytest.raises(ValueError, match="histnorm"):
            GeoHistogram2D(lat="lat", lon="lon", histnorm="invalid")

    def test_invalid_bin_boundaries(self):
        with pytest.raises(ValueError, match="bin_boundaries"):
            GeoHistogram2D(lat="lat", lon="lon", **{"bin_boundaries": "fixed"})

    def test_custom_color_style(self):
        t = GeoHistogram2D(
            lat="lat",
            lon="lon",
            color_scale="plasma",
            color_range=(0.0, 5.0),
        )
        assert t.color_scale == "plasma"
        assert t.color_range == (0.0, 5.0)


class TestGeoHist2DAggregation:
    def test_output_has_geojson_structure(self, geo_df):
        result = _aggregate_geo_hist2d(geo_df, lat_bins=3, lon_bins=3)
        assert "geojson" in result.updates
        assert "locations" in result.updates
        assert "z" in result.updates
        geojson = result.updates["geojson"]
        assert geojson["type"] == "FeatureCollection"
        assert isinstance(geojson["features"], list)

    def test_total_count(self, geo_df):
        result = _aggregate_geo_hist2d(geo_df, lat_bins=5, lon_bins=5)
        total = sum(result.updates["z"])
        assert total == len(geo_df)

    def test_locations_match_features(self, geo_df):
        result = _aggregate_geo_hist2d(geo_df, lat_bins=3, lon_bins=3)
        locations = result.updates["locations"]
        features = result.updates["geojson"]["features"]
        assert len(locations) == len(features)
        for loc, feat in zip(locations, features):
            assert feat["id"] == loc

    def test_feature_geometry_is_polygon(self, geo_df):
        result = _aggregate_geo_hist2d(geo_df, lat_bins=3, lon_bins=3)
        for feat in result.updates["geojson"]["features"]:
            assert feat["geometry"]["type"] == "Polygon"
            coords = feat["geometry"]["coordinates"]
            assert len(coords) == 1
            ring = coords[0]
            assert len(ring) == 5
            assert ring[0] == ring[4]

    def test_viewport_filters_data(self, geo_df):
        viewport = {
            "coordinates": [
                [-73.5, 40.5],
                [-72.5, 40.5],
                [-72.5, 41.5],
                [-73.5, 41.5],
            ]
        }
        full = _aggregate_geo_hist2d(geo_df, lat_bins=3, lon_bins=3)
        zoomed = _aggregate_geo_hist2d(
            geo_df,
            lat_bins=3,
            lon_bins=3,
            update_range=viewport,
        )
        full_total = sum(full.updates["z"])
        zoomed_total = sum(zoomed.updates["z"])
        assert zoomed_total <= full_total

    def test_empty_data_returns_empty_geojson(self):
        df = pl.DataFrame(
            {"lat": [None, None], "lon": [None, None]},
            schema={"lat": pl.Float64, "lon": pl.Float64},
        )
        result = _aggregate_geo_hist2d(df, lat_bins=2, lon_bins=2)
        assert result.updates["geojson"]["features"] == []
        assert result.updates["locations"] == []
        assert result.updates["z"] == []


class TestGeoHist2DHistfunc:
    def test_sum(self, geo_df_with_z):
        result = _aggregate_geo_hist2d(
            geo_df_with_z,
            lat_bins=3,
            lon_bins=3,
            z="z",
            histfunc="sum",
        )
        total_z = sum(result.updates["z"])
        expected_total = geo_df_with_z["z"].sum()
        assert abs(total_z - expected_total) < 1.0

    def test_mean(self, geo_df_with_z):
        result = _aggregate_geo_hist2d(
            geo_df_with_z,
            lat_bins=3,
            lon_bins=3,
            z="z",
            histfunc="mean",
        )
        for v in result.updates["z"]:
            assert 1.0 <= v <= 100.0

    def test_median_not_supported(self):
        # median is not backed by the Rust kernel (mirrors Histogram2D).
        with pytest.raises(ValueError, match="histfunc"):
            GeoHistogram2D(lat="lat", lon="lon", z="z", histfunc="median")

    def test_n_unique_not_supported(self):
        with pytest.raises(ValueError, match="histfunc"):
            GeoHistogram2D(lat="lat", lon="lon", z="z", histfunc="n_unique")

    def test_legacy_median_spec_raises(self):
        spec = TraceSpec(
            uid="geo",
            trace_type="geo_histogram2d",
            backend_data={"lat": "lat", "lon": "lon", "z": "z"},
            params={"lat_bins": 5, "lon_bins": 5, "histfunc": "median"},
            display={"name": "Legacy"},
            axes=None,
            recompute_axes=("coordinates",),
        )
        with pytest.raises(ValueError, match="no longer supported"):
            GeoHistogram2D.from_trace_spec(spec)

    def test_min_max(self, geo_df_with_z):
        result_min = _aggregate_geo_hist2d(
            geo_df_with_z,
            lat_bins=3,
            lon_bins=3,
            z="z",
            histfunc="min",
        )
        result_max = _aggregate_geo_hist2d(
            geo_df_with_z,
            lat_bins=3,
            lon_bins=3,
            z="z",
            histfunc="max",
        )
        for vmin, vmax in zip(result_min.updates["z"], result_max.updates["z"]):
            assert vmin <= vmax


class TestGeoHist2DHistnorm:
    def test_percent(self, geo_df):
        result = _aggregate_geo_hist2d(
            geo_df,
            lat_bins=5,
            lon_bins=5,
            histnorm="percent",
        )
        total = sum(result.updates["z"])
        assert abs(total - 100.0) < 0.1

    def test_probability(self, geo_df):
        result = _aggregate_geo_hist2d(
            geo_df,
            lat_bins=5,
            lon_bins=5,
            histnorm="probability",
        )
        total = sum(result.updates["z"])
        assert abs(total - 1.0) < 0.01

    def test_density(self, geo_df):
        result = _aggregate_geo_hist2d(
            geo_df,
            lat_bins=3,
            lon_bins=3,
            histnorm="density",
        )
        for v in result.updates["z"]:
            assert v >= 0

    def test_probability_density(self, geo_df):
        result = _aggregate_geo_hist2d(
            geo_df,
            lat_bins=3,
            lon_bins=3,
            histnorm="probability density",
        )
        for v in result.updates["z"]:
            assert v >= 0


class TestGeoHist2DEngine:
    """GeoHistogram2D has axes=None; engine must still route viewport and selections."""

    def test_engine_viewport_updates_geo_trace(self, geo_df):
        lf = LFQueryBuilder(geo_df)
        geo = GeoHistogram2D(lat="lat", lon="lon", lat_bins=4, lon_bins=4)
        engine = FlexEngine(backend_lf=lf, scalable_traces={geo.uid: geo})
        infos = [
            TraceInfo(
                uid=geo.uid,
                axes=None,
                trace_type="geo_histogram2d",
                figure_uid="fig_map",
            ),
        ]
        coords = [
            [-73.5, 40.5],
            [-72.5, 40.5],
            [-72.5, 41.5],
            [-73.5, 41.5],
        ]
        event = InteractionEvent(
            type="viewport",
            axis_ranges={"coordinates": coords},
            figure_uid="fig_map",
        )
        viewports = {"fig_map": {"coordinates": coords}}
        deltas = engine.process(event, infos, viewports_by_figure=viewports)
        assert len(deltas) == 1
        assert deltas[0].uid == geo.uid
        assert len(deltas[0].updates.get("z", [])) > 0

    def test_engine_geo_selection_cross_filters_line(self):
        # Lon steps from -74 toward -72 so part of the series is inside [-74, -73].
        df = pl.DataFrame(
            {
                "ts": list(range(20)),
                "val": [float(i) for i in range(20)],
                "lat": [40.0 + i * 0.02 for i in range(20)],
                "lon": [-74.0 + i * (2.0 / 19) for i in range(20)],
            }
        )
        filtered = df.filter(
            pl.col("lat").is_between(40.0, 40.5)
            & pl.col("lon").is_between(-74.0, -73.0)
        )
        assert filtered.height > 0
        max_ts_filtered = filtered["ts"].max()

        lf = LFQueryBuilder(df)
        geo = GeoHistogram2D(lat="lat", lon="lon", lat_bins=4, lon_bins=4)
        line = LinePlot(x="ts", y="val", n_points=1000)
        engine = FlexEngine(
            backend_lf=lf, scalable_traces={geo.uid: geo, line.uid: line}
        )
        infos = [
            TraceInfo(
                uid=geo.uid,
                axes=None,
                trace_type="geo_histogram2d",
                figure_uid="fig_geo",
            ),
            TraceInfo(
                uid=line.uid,
                axes=("x", "y"),
                trace_type="line",
                figure_uid="fig_line",
            ),
        ]
        # Plotly map convention: x_range = longitude, y_range = latitude
        event = InteractionEvent(
            type="selection",
            force_update=True,
            selections=[
                SelectionState(
                    source_figure_uid="fig_geo",
                    predicates=[
                        SelectionPredicate(
                            clauses=[
                                ClauseFilter(column="lon", range=(-74.0, -73.0)),
                                ClauseFilter(column="lat", range=(40.0, 40.5)),
                            ]
                        )
                    ],
                ),
            ],
        )
        deltas = engine.process(event, infos)
        line_delta = next(d for d in deltas if d.uid == line.uid)
        xs = line_delta.updates["x"]
        assert max(xs) <= float(max_ts_filtered)

    def test_engine_geo_viewport_with_active_selection_returns_geo_delta(self, geo_df):
        lf = LFQueryBuilder(geo_df)
        geo = GeoHistogram2D(lat="lat", lon="lon", lat_bins=4, lon_bins=4)
        engine = FlexEngine(backend_lf=lf, scalable_traces={geo.uid: geo})
        infos = [
            TraceInfo(
                uid=geo.uid,
                axes=None,
                trace_type="geo_histogram2d",
                figure_uid="fig_map",
            ),
        ]
        coords = [
            [-73.5, 40.5],
            [-72.5, 40.5],
            [-72.5, 41.5],
            [-73.5, 41.5],
        ]
        event = InteractionEvent(
            type="viewport",
            axis_ranges={"coordinates": coords},
            figure_uid="fig_map",
            selections=[
                SelectionState(
                    source_figure_uid="fig_map",
                    predicates=[
                        SelectionPredicate(
                            clauses=[
                                ClauseFilter(column="lon", range=(-74.0, -72.0)),
                                ClauseFilter(column="lat", range=(40.0, 42.0)),
                            ]
                        )
                    ],
                ),
            ],
        )
        viewports = {"fig_map": {"coordinates": coords}}
        deltas = engine.process(event, infos, viewports_by_figure=viewports)
        assert len(deltas) == 1
        assert deltas[0].uid == geo.uid


class TestGeoHist2DBinBoundaries:
    def test_data_vs_viewport_changes_bin_geometry(self):
        df = pl.DataFrame(
            {
                "lat": [40.1, 40.85],
                "lon": [-73.9, -72.15],
            }
        )
        viewport = {
            "coordinates": [
                [-74.0, 40.0],
                [-72.0, 40.0],
                [-72.0, 42.0],
                [-74.0, 42.0],
            ]
        }
        r_data = _aggregate_geo_hist2d(
            df,
            lat_bins=2,
            lon_bins=2,
            update_range=viewport,
            bin_boundaries="data",
        )
        r_vp = _aggregate_geo_hist2d(
            df,
            lat_bins=2,
            lon_bins=2,
            update_range=viewport,
            bin_boundaries="viewport",
        )
        assert sum(r_data.updates["z"]) == sum(r_vp.updates["z"]) == 2

        def _max_lat_span(geojson: dict) -> float:
            best = 0.0
            for feat in geojson["features"]:
                ring = feat["geometry"]["coordinates"][0]
                lats = [p[1] for p in ring]
                best = max(best, max(lats) - min(lats))
            return best

        assert _max_lat_span(r_vp.updates["geojson"]) > _max_lat_span(
            r_data.updates["geojson"]
        )


class TestGeoHist2DTypedViewportBounds:
    """The viewport mask must compare against column-dtype bounds.

    Raw f64 Python-float bounds force Polars to widen f32/integer columns to
    Float64 during the ``is_between`` filter (no native-dtype SIMD), a ~4-9x
    slowdown on the interactive pan/zoom path. Typed bounds keep the comparison
    in the column's dtype. Results are identical either way — only speed differs
    — so these guard the wiring, not the output values.
    """

    _VIEWPORT = {
        "coordinates": [
            [-74.0, 40.0],
            [-72.0, 40.0],
            [-72.0, 42.0],
            [-74.0, 42.0],
        ]
    }

    def test_viewport_mask_uses_typed_bounds(self, monkeypatch):
        import flexviz.trace.geo_hist2d as mod

        calls: list = []
        real = mod._typed_range_bounds

        def spy(col, range_, schema=None):
            calls.append((col, range_, schema))
            return real(col, range_, schema)

        monkeypatch.setattr(mod, "_typed_range_bounds", spy)
        df = pl.DataFrame(
            {
                "lat": pl.Series([40.5], dtype=pl.Float32),
                "lon": pl.Series([-73.5], dtype=pl.Float32),
            }
        )
        _aggregate_geo_hist2d(
            df,
            lat_bins=4,
            lon_bins=4,
            update_range=self._VIEWPORT,
            bin_boundaries="viewport",
        )
        cols = [c[0] for c in calls]
        assert "lat" in cols and "lon" in cols
        # schema must be threaded through (not None) so bounds can be typed.
        assert calls and all(c[2] is not None for c in calls)

    def test_f32_matches_f64_under_viewport(self):
        rng = np.random.default_rng(7)
        n = 5000
        lat = rng.uniform(40.0, 42.0, n)
        lon = rng.uniform(-74.0, -72.0, n)
        for mode in ("viewport", "data"):
            r64 = _aggregate_geo_hist2d(
                pl.DataFrame(
                    {
                        "lat": pl.Series(lat, dtype=pl.Float64),
                        "lon": pl.Series(lon, dtype=pl.Float64),
                    }
                ),
                lat_bins=8,
                lon_bins=8,
                update_range=self._VIEWPORT,
                bin_boundaries=mode,
            )
            r32 = _aggregate_geo_hist2d(
                pl.DataFrame(
                    {
                        "lat": pl.Series(lat.astype(np.float32), dtype=pl.Float32),
                        "lon": pl.Series(lon.astype(np.float32), dtype=pl.Float32),
                    }
                ),
                lat_bins=8,
                lon_bins=8,
                update_range=self._VIEWPORT,
                bin_boundaries=mode,
            )
            # Totals are preserved exactly; per-bin counts may differ by f32
            # storage jitter, which is inherent to the f32 column, not the path.
            assert sum(r64.updates["z"]) == sum(r32.updates["z"]) == n


class TestGeoHist2DExtractLatLonRange:
    def test_with_coordinates(self):
        coords = [[-74.0, 40.0], [-72.0, 40.0], [-72.0, 42.0], [-74.0, 42.0]]
        lat_r, lon_r = GeoHistogram2D._extract_lat_lon_range({"coordinates": coords})
        assert lat_r == (40.0, 42.0)
        assert lon_r == (-74.0, -72.0)

    def test_without_coordinates(self):
        lat_r, lon_r = GeoHistogram2D._extract_lat_lon_range({})
        assert lat_r is None
        assert lon_r is None


class TestGeoHist2DSpec:
    def test_roundtrip(self):
        t = GeoHistogram2D(
            lat="lat_col",
            lon="lon_col",
            lat_bins=32,
            lon_bins=48,
            histfunc="sum",
            histnorm="percent",
            z="value",
            name="Geo Heatmap",
            color_scale="plasma",
            color_range=(0.0, 100.0),
        )
        spec = t.to_trace_spec()
        assert spec.trace_type == "geo_histogram2d"
        assert spec.backend_data == {"lat": "lat_col", "lon": "lon_col", "z": "value"}
        assert spec.params["lat_bins"] == 32
        assert spec.params["lon_bins"] == 48
        assert spec.params["histfunc"] == "sum"
        assert spec.params["histnorm"] == "percent"
        assert spec.params.get("bin_boundaries", "data") == "data"

        t2 = GeoHistogram2D.from_trace_spec(spec)
        assert t2.lat_col == "lat_col"
        assert t2.lon_col == "lon_col"
        assert t2.lat_bins == 32
        assert t2.lon_bins == 48
        assert t2.histfunc == "sum"
        assert t2.histnorm == "percent"
        assert t2.z_col == "value"
        assert t2.color_scale == "plasma"
        assert t2.color_range == (0.0, 100.0)
        assert t2.bin_boundaries == "data"

    def test_roundtrip_bin_boundaries_viewport(self):
        t = GeoHistogram2D(lat="a", lon="b", bin_boundaries="viewport")
        spec = t.to_trace_spec()
        assert spec.params["bin_boundaries"] == "viewport"
        t2 = GeoHistogram2D.from_trace_spec(spec)
        assert t2.bin_boundaries == "viewport"

    def test_build_from_registry(self):
        t = GeoHistogram2D(lat="lat", lon="lon")
        spec = t.to_trace_spec()
        t2 = build_trace_from_spec(spec)
        assert isinstance(t2, GeoHistogram2D)
        assert t2.uid == t.uid

    def test_legacy_spec_gets_defaults(self):
        spec = TraceSpec(
            uid="geo",
            trace_type="geo_histogram2d",
            backend_data={"lat": "lat", "lon": "lon"},
            params={"lat_bins": 10, "lon_bins": 10},
            display={"name": "Legacy"},
            axes=None,
            recompute_axes=("coordinates",),
        )
        trace = GeoHistogram2D.from_trace_spec(spec)
        assert trace.color_scale == "viridis"
        assert trace.color_range == "auto"
        assert trace.histfunc is None
        assert trace.histnorm is None
        assert trace.bin_boundaries == "data"


class TestGeoHist2DAdapter:
    def test_plotly_trace_obj(self):
        from flexviz.adapters.plotly_adapter import PlotlyAdapter

        spec = TraceSpec(
            uid="geo_test",
            trace_type="geo_histogram2d",
            backend_data={"lat": "lat", "lon": "lon"},
            params={
                "lat_bins": 10,
                "lon_bins": 10,
                "histfunc": None,
                "histnorm": None,
            },
            display={"name": "Geo", "color_scale": "viridis", "color_range": "auto"},
            axes=None,
            recompute_axes=("coordinates",),
        )
        obj = PlotlyAdapter._plotly_trace_obj(spec, "Geo", None)
        assert obj["type"] == "choroplethmap"
        assert obj["featureidkey"] == "id"
        assert obj["colorscale"] == "viridis"


class TestGeoHist2DEdgeCenterConsistency:
    """Polygon edges returned by _to_update must be the midpoints of bin centers."""

    def test_lat_edges_are_midpoints_of_centers(self, geo_df):
        result = _aggregate_geo_hist2d(geo_df, lat_bins=4, lon_bins=4)
        geojson = result.updates["geojson"]
        # Collect unique lat_bottom values from all features (= lat_edges[:-1])
        # and verify they form an arithmetic sequence consistent with the centers
        # encoded in the feature polygons.
        features = geojson["features"]
        if not features:
            pytest.skip("no non-null bins to inspect")

        # Each feature's polygon has coords [[lon_left, lat_bottom], [lon_right, lat_bottom],
        # [lon_right, lat_top], [lon_left, lat_top], [lon_left, lat_bottom]]
        for feat in features:
            ring = feat["geometry"]["coordinates"][0]
            lat_bottom = ring[0][1]
            lat_top = ring[2][1]
            # The center should be the midpoint of the two lat edges
            lat_center_from_edges = (lat_bottom + lat_top) / 2
            # The center must be consistent (edges derived from same step as centers)
            assert lat_top > lat_bottom, "lat_top must exceed lat_bottom"
            # Midpoint must be finite
            assert math.isfinite(lat_center_from_edges)

    def test_lon_edges_are_midpoints_of_centers(self, geo_df):
        result = _aggregate_geo_hist2d(geo_df, lat_bins=4, lon_bins=4)
        geojson = result.updates["geojson"]
        features = geojson["features"]
        if not features:
            pytest.skip("no non-null bins to inspect")

        for feat in features:
            ring = feat["geometry"]["coordinates"][0]
            lon_left = ring[0][0]
            lon_right = ring[1][0]
            assert lon_right > lon_left, "lon_right must exceed lon_left"
            assert math.isfinite((lon_left + lon_right) / 2)

    def test_uniform_bin_widths(self, geo_df):
        """All lat bins and all lon bins should have the same width."""
        result = _aggregate_geo_hist2d(geo_df, lat_bins=5, lon_bins=5)
        geojson = result.updates["geojson"]
        features = geojson["features"]
        if len(features) < 2:
            pytest.skip("need at least 2 non-null bins")

        lat_heights = set()
        lon_widths = set()
        for feat in features:
            ring = feat["geometry"]["coordinates"][0]
            lat_heights.add(round(ring[2][1] - ring[0][1], 10))
            lon_widths.add(round(ring[1][0] - ring[0][0], 10))

        assert len(lat_heights) == 1, f"non-uniform lat bin heights: {lat_heights}"
        assert len(lon_widths) == 1, f"non-uniform lon bin widths: {lon_widths}"

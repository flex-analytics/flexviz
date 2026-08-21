"""Unit tests for GeoLine trace."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from flexviz.LF import LFQueryBuilder
from flexviz.engine import FlexEngine, TraceInfo
from flexviz.events import InteractionEvent
from flexviz.spec import TraceSpec
from flexviz.trace.geo_line import GeoLine, _extract_lat_lon_range
from flexviz.trace.base import TraceResult
from flexviz.trace import build_trace_from_spec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _aggregate_geo_line(
    df: pl.DataFrame,
    lat: str = "lat",
    lon: str = "lon",
    n_points: int = 1000,
    add_gaps: bool = True,
    update_range: dict | None = None,
) -> TraceResult:
    """Run the full aggregation pipeline for GeoLine."""
    lf = LFQueryBuilder(df)
    trace = GeoLine(lat=lat, lon=lon, n_points=n_points, add_gaps=add_gaps)
    spec = trace.get_aggregation_spec(update_range or {}, schema=lf.schema)
    regular_df, _ = lf.aggregate([], [spec])
    return trace._to_update(regular_df)


@pytest.fixture()
def geo_df() -> pl.DataFrame:
    rng = np.random.default_rng(42)
    n = 10_000
    return pl.DataFrame(
        {
            "lat": rng.uniform(40.0, 42.0, n).tolist(),
            "lon": rng.uniform(-74.0, -72.0, n).tolist(),
        }
    )


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestGeoLineConstructor:
    def test_defaults(self):
        t = GeoLine(lat="lat", lon="lon")
        assert t.trace_type == "geo_line"
        assert t.lat_col == "lat"
        assert t.lon_col == "lon"
        assert t.n_points == 1000
        assert t.add_gaps is True
        assert t._axes is None
        assert t.recompute_axes == ("coordinates",)
        assert t.update_on_zoom is True

    def test_custom_params(self):
        t = GeoLine(
            lat="latitude",
            lon="longitude",
            n_points=500,
            name="My Path",
            color="#ff0000",
            add_gaps=False,
        )
        assert t.lat_col == "latitude"
        assert t.lon_col == "longitude"
        assert t.n_points == 500
        assert t._display["name"] == "My Path"
        assert t._display["color"] == "#ff0000"
        assert t.add_gaps is False

    def test_default_name(self):
        t = GeoLine(lat="latitude", lon="longitude")
        assert t._display["name"] == "latitude/longitude"

    def test_axes_is_none(self):
        t = GeoLine(lat="lat", lon="lon")
        assert t._axes is None

    def test_update_on_zoom_default(self):
        t = GeoLine(lat="lat", lon="lon")
        assert t.recompute_axes == ("coordinates",)
        assert t.update_on_zoom is True

    def test_update_on_zoom_override(self):
        t = GeoLine(lat="lat", lon="lon", update_on_zoom=False)
        assert t.recompute_axes == ()
        assert t.update_on_zoom is False


# ---------------------------------------------------------------------------
# Row count (downsampling)
# ---------------------------------------------------------------------------


class TestGeoLineRowCount:
    @pytest.mark.parametrize(
        "n_rows,n_points",
        [
            (100, 1000),  # fewer rows than n_points → all rows returned
            (1_000, 1000),
            (5_000, 1000),
            (10_000, 500),
            (100_000, 2000),
        ],
    )
    def test_output_lte_n_points(self, n_rows, n_points):
        rng = np.random.default_rng(0)
        df = pl.DataFrame(
            {
                "lat": rng.uniform(40.0, 42.0, n_rows).tolist(),
                "lon": rng.uniform(-74.0, -72.0, n_rows).tolist(),
            }
        )
        result = _aggregate_geo_line(df, n_points=n_points, add_gaps=False)
        assert len(result.updates["lat"]) <= n_points
        assert len(result.updates["lon"]) <= n_points

    def test_fewer_rows_than_n_points(self):
        """When fewer rows than n_points, all rows are returned."""
        df = pl.DataFrame(
            {
                "lat": [40.0, 41.0, 42.0],
                "lon": [-74.0, -73.0, -72.0],
            }
        )
        result = _aggregate_geo_line(df, n_points=1000, add_gaps=False)
        assert len(result.updates["lat"]) == 3
        assert len(result.updates["lon"]) == 3


# ---------------------------------------------------------------------------
# Viewport filtering
# ---------------------------------------------------------------------------


class TestGeoLineViewport:
    def test_coordinates_filter_reduces_count(self, geo_df):
        # Use n_points larger than the dataset so both raw and filtered return their
        # true row counts (not capped at n_points), making the comparison meaningful.
        n_points = len(geo_df) + 1

        result_full = _aggregate_geo_line(geo_df, n_points=n_points, add_gaps=False)
        n_full = len(result_full.updates["lat"])
        assert n_full == len(geo_df)

        # Narrow viewport: ~25% of the lat/lon range
        # geo_df has lat in [40, 42] and lon in [-74, -72], so quarter = [40,41] x [-74,-73]
        viewport = {
            "coordinates": [
                [-74.0, 40.0],  # SW
                [-73.0, 40.0],  # SE
                [-73.0, 41.0],  # NE
                [-74.0, 41.0],  # NW
            ]
        }
        result_vp = _aggregate_geo_line(
            geo_df, n_points=n_points, add_gaps=False, update_range=viewport
        )
        assert len(result_vp.updates["lat"]) < n_full

    def test_coordinates_filter_keeps_points_in_range(self, geo_df):
        lat_min, lat_max = 40.5, 41.0
        lon_min, lon_max = -73.5, -73.0
        viewport = {
            "coordinates": [
                [lon_min, lat_min],
                [lon_max, lat_min],
                [lon_max, lat_max],
                [lon_min, lat_max],
            ]
        }
        result = _aggregate_geo_line(
            geo_df, n_points=2000, add_gaps=False, update_range=viewport
        )
        lats = result.updates["lat"].to_list()
        lons = result.updates["lon"].to_list()
        assert all(lat_min <= v <= lat_max for v in lats)
        assert all(lon_min <= v <= lon_max for v in lons)

    def test_no_coordinates_returns_data(self, geo_df):
        result = _aggregate_geo_line(geo_df, n_points=500, update_range={})
        assert len(result.updates["lat"]) > 0

    def test_empty_coordinates_returns_data(self, geo_df):
        result = _aggregate_geo_line(
            geo_df, n_points=500, update_range={"coordinates": []}
        )
        assert len(result.updates["lat"]) > 0


# ---------------------------------------------------------------------------
# _to_update output
# ---------------------------------------------------------------------------


class TestGeoLineToUpdate:
    def test_returns_lat_lon_series(self, geo_df):
        result = _aggregate_geo_line(geo_df)
        assert "lat" in result.updates
        assert "lon" in result.updates
        assert isinstance(result.updates["lat"], pl.Series)
        assert isinstance(result.updates["lon"], pl.Series)

    def test_lat_lon_equal_length(self, geo_df):
        result = _aggregate_geo_line(geo_df, n_points=300)
        assert len(result.updates["lat"]) == len(result.updates["lon"])

    def test_no_group_results(self, geo_df):
        result = _aggregate_geo_line(geo_df)
        assert result.group_results is None


# ---------------------------------------------------------------------------
# Gap detection
# ---------------------------------------------------------------------------


class TestGeoLineGaps:
    def test_gap_inserts_none(self):
        """A large jump between two segments should produce None separators."""
        # Two clusters far apart: lat jump of 10 degrees is >> 4.1 * median step
        lats = [40.0, 40.1, 40.2, 50.0, 50.1, 50.2]
        lons = [-74.0, -74.0, -74.0, -74.0, -74.0, -74.0]
        df = pl.DataFrame({"lat": lats, "lon": lons})
        result = _aggregate_geo_line(df, n_points=len(df) + 1, add_gaps=True)
        lat_out = result.updates["lat"].to_list()
        lon_out = result.updates["lon"].to_list()
        # At least one None should be present where the gap was inserted
        assert None in lat_out
        assert None in lon_out
        # lat and lon remain the same length
        assert len(lat_out) == len(lon_out)

    def test_gap_none_count_matches_gap_count(self):
        """Exactly one None per gap."""
        # One big gap between two segments
        lats = [40.0, 40.1, 40.2, 50.0, 50.1, 50.2]
        lons = [-74.0, -74.0, -74.0, -74.0, -74.0, -74.0]
        df = pl.DataFrame({"lat": lats, "lon": lons})
        result = _aggregate_geo_line(df, n_points=len(df) + 1, add_gaps=True)
        lat_out = result.updates["lat"].to_list()
        assert lat_out.count(None) == 1

    def test_no_gaps_when_add_gaps_false(self):
        """add_gaps=False never inserts None values."""
        lats = [40.0, 40.1, 40.2, 50.0, 50.1, 50.2]
        lons = [-74.0, -74.0, -74.0, -74.0, -74.0, -74.0]
        df = pl.DataFrame({"lat": lats, "lon": lons})
        result = _aggregate_geo_line(df, n_points=len(df) + 1, add_gaps=False)
        lat_out = result.updates["lat"].to_list()
        assert None not in lat_out

    def test_uniform_data_no_gaps(self):
        """Uniformly spaced data should produce no gaps."""
        lats = [40.0 + i * 0.01 for i in range(100)]
        lons = [-74.0 + i * 0.01 for i in range(100)]
        df = pl.DataFrame({"lat": lats, "lon": lons})
        result = _aggregate_geo_line(df, n_points=len(df) + 1, add_gaps=True)
        lat_out = result.updates["lat"].to_list()
        assert None not in lat_out

    def test_gap_preserves_segment_order(self):
        """Points on each side of a gap remain in their original order."""
        lats = [40.0, 40.1, 40.2, 50.0, 50.1, 50.2]
        lons = [-74.0, -74.0, -74.0, -74.0, -74.0, -74.0]
        df = pl.DataFrame({"lat": lats, "lon": lons})
        result = _aggregate_geo_line(df, n_points=len(df) + 1, add_gaps=True)
        lat_out = result.updates["lat"].to_list()
        non_null = [v for v in lat_out if v is not None]
        # Original order must be preserved
        assert non_null == lats

    def test_single_point_no_gap(self):
        df = pl.DataFrame({"lat": [40.0], "lon": [-74.0]})
        result = _aggregate_geo_line(df, n_points=10, add_gaps=True)
        assert None not in result.updates["lat"].to_list()

    def test_two_points_no_gap(self):
        """Two points — even a large jump — should not crash."""
        df = pl.DataFrame({"lat": [40.0, 80.0], "lon": [-74.0, 100.0]})
        result = _aggregate_geo_line(df, n_points=10, add_gaps=True)
        # No error; result has two original points (possibly a gap between them)
        assert len(result.updates["lat"]) >= 2


# ---------------------------------------------------------------------------
# _extract_lat_lon_range helper
# ---------------------------------------------------------------------------


class TestExtractLatLonRange:
    def test_extracts_bounds(self):
        coordinates = [[-74.0, 40.0], [-72.0, 40.0], [-72.0, 42.0], [-74.0, 42.0]]
        lat_range, lon_range = _extract_lat_lon_range({"coordinates": coordinates})
        assert lat_range == (40.0, 42.0)
        assert lon_range == (-74.0, -72.0)

    def test_no_coordinates(self):
        lat_range, lon_range = _extract_lat_lon_range({})
        assert lat_range is None
        assert lon_range is None

    def test_empty_coordinates(self):
        lat_range, lon_range = _extract_lat_lon_range({"coordinates": []})
        assert lat_range is None
        assert lon_range is None


# ---------------------------------------------------------------------------
# Spec round-trip
# ---------------------------------------------------------------------------


class TestGeoLineSpec:
    def test_to_trace_spec_fields(self):
        t = GeoLine(
            lat="latitude",
            lon="longitude",
            n_points=500,
            name="Path",
            color="#ff0000",
            add_gaps=False,
        )
        spec = t.to_trace_spec()
        assert spec.trace_type == "geo_line"
        assert spec.backend_data == {"lat": "latitude", "lon": "longitude"}
        assert spec.params["n_points"] == 500
        assert spec.params["add_gaps"] is False
        assert spec.display["name"] == "Path"
        assert spec.display["color"] == "#ff0000"
        assert spec.axes is None
        assert spec.recompute_axes == ("coordinates",)

    def test_from_trace_spec_round_trip(self):
        t = GeoLine(
            lat="lat",
            lon="lon",
            n_points=750,
            name="Test",
            color="#0000ff",
            add_gaps=False,
        )
        spec = t.to_trace_spec()
        t2 = GeoLine.from_trace_spec(spec)
        assert t2.uid == t.uid
        assert t2.lat_col == "lat"
        assert t2.lon_col == "lon"
        assert t2.n_points == 750
        assert t2._display["color"] == "#0000ff"
        assert t2.add_gaps is False

    def test_registry_round_trip(self):
        t = GeoLine(lat="lat", lon="lon")
        spec = t.to_trace_spec()
        t2 = build_trace_from_spec(spec)
        assert isinstance(t2, GeoLine)
        assert t2.uid == t.uid

    def test_from_trace_spec_defaults(self):
        spec = TraceSpec(
            uid="test-uid",
            trace_type="geo_line",
            backend_data={"lat": "lat", "lon": "lon"},
            params={"n_points": 1000, "add_gaps": True},
            display={"name": "lat/lon"},
        )
        t = GeoLine.from_trace_spec(spec)
        assert t.uid == "test-uid"
        assert t.lat_col == "lat"
        assert t.lon_col == "lon"
        assert t.n_points == 1000
        assert t.add_gaps is True

    def test_from_trace_spec_preserves_explicit_recompute_axes(self):
        spec = TraceSpec(
            uid="geo-custom-recompute",
            trace_type="geo_line",
            backend_data={"lat": "lat", "lon": "lon"},
            params={"n_points": 1000, "add_gaps": True},
            recompute_axes=("coordinates", "debug"),
        )

        t = GeoLine.from_trace_spec(spec)

        assert t.recompute_axes == ("coordinates", "debug")


# ---------------------------------------------------------------------------
# Engine integration
# ---------------------------------------------------------------------------


class TestGeoLineEngine:
    def test_init_event_returns_delta(self, geo_df):
        lf = LFQueryBuilder(geo_df)
        trace = GeoLine(lat="lat", lon="lon", n_points=500, add_gaps=False)
        engine = FlexEngine(backend_lf=lf, scalable_traces={trace.uid: trace})
        infos = [
            TraceInfo(
                uid=trace.uid, axes=None, trace_type="geo_line", figure_uid="fig1"
            ),
        ]
        event = InteractionEvent(type="init", force_update=True, figure_uid="fig1")
        deltas = engine.process(event, infos)
        assert len(deltas) == 1
        delta = deltas[0]
        assert delta.uid == trace.uid
        assert "lat" in delta.updates
        assert "lon" in delta.updates
        assert len(delta.updates["lat"]) <= 500
        assert len(delta.updates["lat"]) == len(delta.updates["lon"])

    def test_viewport_event_filters_data(self, geo_df):
        lf = LFQueryBuilder(geo_df)
        # n_points > dataset → no cap, so viewport filter is visible in count
        trace = GeoLine(lat="lat", lon="lon", n_points=len(geo_df) + 1, add_gaps=False)
        engine = FlexEngine(backend_lf=lf, scalable_traces={trace.uid: trace})
        infos = [
            TraceInfo(
                uid=trace.uid, axes=None, trace_type="geo_line", figure_uid="fig_map"
            ),
        ]

        init_event = InteractionEvent(
            type="init", force_update=True, figure_uid="fig_map"
        )
        init_deltas = engine.process(init_event, infos)
        n_init = len(init_deltas[0].updates["lat"])
        assert n_init == len(geo_df)

        coords = [[-74.0, 40.0], [-73.0, 40.0], [-73.0, 41.0], [-74.0, 41.0]]
        viewport_event = InteractionEvent(
            type="viewport",
            axis_ranges={"coordinates": coords},
            figure_uid="fig_map",
        )
        viewports = {"fig_map": {"coordinates": coords}}
        vp_deltas = engine.process(viewport_event, infos, viewports_by_figure=viewports)
        assert len(vp_deltas[0].updates["lat"]) < n_init

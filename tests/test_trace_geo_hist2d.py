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
from flexviz.trace import _hist_helpers as helpers_mod
from flexviz.trace.geo_hist2d import GeoHistogram2D
from flexviz.trace.line import LinePlot
from flexviz.trace.base import TraceResult
from flexviz.trace import build_trace_from_spec


def _cells(updates: dict) -> list[float]:
    """The non-empty cells: the values the client draws a rectangle for."""
    return [v for v in updates["z"] if v is not None]


def _aggregate_geo_hist2d(
    df: pl.DataFrame | pl.LazyFrame,
    lat: str = "lat",
    lon: str = "lon",
    lat_bins: int = 5,
    lon_bins: int = 5,
    z: str | None = None,
    histfunc: str | None = None,
    histnorm: str | None = None,
    update_range: dict | None = None,
    filter_exprs: list | None = None,
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
    )
    update_range = update_range or {}
    cols = trace.domain_cols(update_range)
    spec = trace.get_aggregation_spec(
        update_range,
        schema=lf.schema,
        domains=(
            lf.physical_minmax(list(cols), lf.schema, memoize=False) if cols else {}
        ),
        scan_source=lf.is_scan,
    )
    regular_df, _ = lf.aggregate(filter_exprs or [], [spec])
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
    def test_output_has_edge_triples_and_a_flat_z(self, geo_df):
        result = _aggregate_geo_hist2d(geo_df, lat_bins=3, lon_bins=3)
        lat_lo, lat_step, nb_lat = result.updates["lat_edges"]
        lon_lo, lon_step, nb_lon = result.updates["lon_edges"]
        assert (nb_lat, nb_lon) == (3, 3), "the triples carry the grid"
        assert lat_step > 0 and lon_step > 0
        assert math.isfinite(lat_lo) and math.isfinite(lon_lo)
        # One entry per cell, empty cells included: the client skips the nulls.
        assert len(result.updates["z"]) == nb_lat * nb_lon

    def test_total_count(self, geo_df):
        result = _aggregate_geo_hist2d(geo_df, lat_bins=5, lon_bins=5)
        total = sum(_cells(result.updates))
        assert total == len(geo_df)

    def test_z_is_lon_major(self, geo_df):
        """z[j * nb_lat + i] is cell (lat i, lon j): the order the client's
        rectangle builder walks."""
        df = pl.DataFrame({"lat": [40.25, 40.75], "lon": [-74.75, -74.25]})
        result = _aggregate_geo_hist2d(df, lat_bins=2, lon_bins=2)
        # Two rows on the diagonal: the lowest lat with the lowest lon, and the
        # highest lat with the highest lon.
        assert result.updates["z"] == [1.0, None, None, 1.0]

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
        full_total = sum(_cells(full.updates))
        zoomed_total = sum(_cells(zoomed.updates))
        assert zoomed_total <= full_total

    def test_empty_data_returns_all_empty_cells(self):
        df = pl.DataFrame(
            {"lat": [None, None], "lon": [None, None]},
            schema={"lat": pl.Float64, "lon": pl.Float64},
        )
        result = _aggregate_geo_hist2d(df, lat_bins=2, lon_bins=2)
        assert result.updates["z"] == [None] * 4


class TestGeoHist2DHistfunc:
    def test_sum(self, geo_df_with_z):
        result = _aggregate_geo_hist2d(
            geo_df_with_z,
            lat_bins=3,
            lon_bins=3,
            z="z",
            histfunc="sum",
        )
        total_z = sum(_cells(result.updates))
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
        for v in _cells(result.updates):
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
        for vmin, vmax in zip(_cells(result_min.updates), _cells(result_max.updates)):
            assert vmin <= vmax


class TestGeoHist2DHistnorm:
    def test_percent(self, geo_df):
        result = _aggregate_geo_hist2d(
            geo_df,
            lat_bins=5,
            lon_bins=5,
            histnorm="percent",
        )
        total = sum(_cells(result.updates))
        assert abs(total - 100.0) < 0.1

    def test_probability(self, geo_df):
        result = _aggregate_geo_hist2d(
            geo_df,
            lat_bins=5,
            lon_bins=5,
            histnorm="probability",
        )
        total = sum(_cells(result.updates))
        assert abs(total - 1.0) < 0.01

    def test_density(self, geo_df):
        result = _aggregate_geo_hist2d(
            geo_df,
            lat_bins=3,
            lon_bins=3,
            histnorm="density",
        )
        for v in _cells(result.updates):
            assert v >= 0

    def test_probability_density(self, geo_df):
        result = _aggregate_geo_hist2d(
            geo_df,
            lat_bins=3,
            lon_bins=3,
            histnorm="probability density",
        )
        for v in _cells(result.updates):
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
        import flexviz.trace._hist_helpers as mod

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
        for update_range in (self._VIEWPORT, {}):
            r64 = _aggregate_geo_hist2d(
                pl.DataFrame(
                    {
                        "lat": pl.Series(lat, dtype=pl.Float64),
                        "lon": pl.Series(lon, dtype=pl.Float64),
                    }
                ),
                lat_bins=8,
                lon_bins=8,
                update_range=update_range,
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
                update_range=update_range,
            )
            # Totals are preserved exactly; per-bin counts may differ by f32
            # storage jitter, which is inherent to the f32 column, not the path.
            assert sum(_cells(r64.updates)) == sum(_cells(r32.updates)) == n


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
        assert "bin_boundaries" not in spec.params

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


class TestGeoHist2DEdgeTriples:
    """The triples must span the grid the kernel binned on, so the client's
    ``lo + i * step`` rectangles land on the server's cells."""

    def test_the_triples_span_the_binned_grid(self, geo_df):
        result = _aggregate_geo_hist2d(geo_df, lat_bins=4, lon_bins=3)
        lat_lo, lat_step, nb_lat = result.updates["lat_edges"]
        lon_lo, lon_step, nb_lon = result.updates["lon_edges"]
        assert (nb_lat, nb_lon) == (4, 3)
        assert lat_lo + nb_lat * lat_step == pytest.approx(geo_df["lat"].max())
        assert lon_lo + nb_lon * lon_step == pytest.approx(geo_df["lon"].max())
        assert lat_lo == pytest.approx(geo_df["lat"].min())
        assert lon_lo == pytest.approx(geo_df["lon"].min())


# ---------------------------------------------------------------------------
# Domains and the scan batch fold
# ---------------------------------------------------------------------------


class TestGeoHist2DDomainCols:
    def test_unzoomed_asks_for_both_columns(self):
        t = GeoHistogram2D(lat="lat", lon="lon")
        assert t.domain_cols({}) == ("lat", "lon")

    def test_a_map_viewport_supplies_the_bounds(self):
        t = GeoHistogram2D(lat="lat", lon="lon")
        coords = [[-74.0, 40.0], [-72.0, 40.0], [-72.0, 42.0], [-74.0, 42.0]]
        assert t.domain_cols({"coordinates": coords}) == ()

    def test_a_cross_filter_does_not_move_the_bin_edges(self):
        """Unzoomed edges come from the unfiltered domain, so a selection
        recolors the grid instead of re-binning it."""
        df = pl.DataFrame(
            {
                "lat": [40.0, 41.0, 42.0, 43.0],
                "lon": [-74.0, -73.0, -72.0, -71.0],
            }
        )
        full = _aggregate_geo_hist2d(df, lat_bins=2, lon_bins=2)
        filtered = _aggregate_geo_hist2d(
            df,
            lat_bins=2,
            lon_bins=2,
            filter_exprs=[pl.col("lat").is_between(40.0, 41.0)],
        )

        for axis in ("lat_edges", "lon_edges"):
            assert full.updates[axis] == filtered.updates[axis], axis
        # The selection empties cells, it does not move them.
        assert len(_cells(filtered.updates)) < len(_cells(full.updates))


_NAN = float("nan")
_LAT = [40.0, 40.5, 41.0, 41.25, 41.5, 42.0, 42.5, 43.0]
_LON = [-74.0, -73.5, -73.0, -72.0, -72.5, -71.5, -71.0, -70.0]
_Z = [1.5, -2.0, 3.25, 0.0, 7.5, -1.25, 4.0, 9.0]

_MAP_VIEWPORT = {
    "coordinates": [
        [-73.5, 40.5],
        [-71.5, 40.5],
        [-71.5, 42.5],
        [-73.5, 42.5],
    ]
}
_EMPTY_VIEWPORT = {
    "coordinates": [[10.0, 10.0], [11.0, 10.0], [11.0, 11.0], [10.0, 11.0]]
}


def _geo_df(lat=None, lon=None, z=None) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "lat": pl.Series("lat", _LAT) if lat is None else lat,
            "lon": pl.Series("lon", _LON) if lon is None else lon,
            "z": pl.Series("z", _Z) if z is None else z,
        }
    )


def _geo_updates(df: pl.DataFrame, **kwargs) -> tuple[dict, dict]:
    """The ``_to_update`` output of the resident kernel and of the scan fold."""
    lf = LFQueryBuilder(df)
    trace = GeoHistogram2D(
        lat="lat",
        lon="lon",
        lat_bins=kwargs.pop("lat_bins", 4),
        lon_bins=kwargs.pop("lon_bins", 3),
        z=kwargs.pop("z", None),
        histfunc=kwargs.pop("histfunc", None),
        histnorm=kwargs.pop("histnorm", None),
    )
    update_range = kwargs.pop("update_range", None) or {}
    assert not kwargs, kwargs
    cols = trace.domain_cols(update_range)
    domains = lf.physical_minmax(list(cols), lf.schema, memoize=False) if cols else {}
    out = []
    for scan_source in (False, True):
        spec = trace.get_aggregation_spec(
            update_range,
            schema=lf.schema,
            domains=domains,
            scan_source=scan_source,
        )
        assert (spec.plan is not None) is scan_source
        agg, _ = lf.aggregate([], [spec])
        out.append(trace._to_update(agg).updates)
    return out[0], out[1]


def _assert_geo_grids_match(resident: dict, scanned: dict, *, exact: bool) -> None:
    assert resident["lat_edges"] == scanned["lat_edges"]
    assert resident["lon_edges"] == scanned["lon_edges"]
    if exact:
        assert resident["z"] == scanned["z"]
        return
    # A null cell draws no rectangle, so an identical null pattern is required.
    for a, b in zip(resident["z"], scanned["z"]):
        assert (a is None) == (b is None)
        if a is not None:
            assert math.isclose(a, b, rel_tol=1e-9)


class TestGeoHist2DScanFoldEquivalence:
    """A scan source folds the kernel over batches; the grid must match.

    ``scan_source`` picks a formulation, never a result. Count, min and max are
    exact; sum and mean fold in a different order, so they are compared with a
    relative tolerance and an identical null pattern.
    """

    @pytest.mark.parametrize(
        "name,df,kwargs",
        [
            ("clean", _geo_df(), {}),
            (
                "f32_axes",
                _geo_df(
                    lat=pl.Series("lat", _LAT, dtype=pl.Float32),
                    lon=pl.Series("lon", _LON, dtype=pl.Float32),
                ),
                {},
            ),
            ("nan_lat", _geo_df(lat=[_NAN] + _LAT[1:]), {}),
            ("null_lat", _geo_df(lat=[None] + _LAT[1:]), {}),
            ("nan_lon", _geo_df(lon=[_NAN] + _LON[1:]), {}),
            ("null_lon", _geo_df(lon=[None] + _LON[1:]), {}),
            ("map_viewport", _geo_df(), {"update_range": _MAP_VIEWPORT}),
            ("empty_viewport", _geo_df(), {"update_range": _EMPTY_VIEWPORT}),
            ("one_lat_bin", _geo_df(), {"lat_bins": 1}),
            (
                "empty",
                pl.DataFrame(
                    schema={"lat": pl.Float64, "lon": pl.Float64, "z": pl.Float64}
                ),
                {},
            ),
            ("histnorm", _geo_df(), {"histnorm": "probability density"}),
        ],
    )
    def test_count_matches_kernel(self, name, df, kwargs):
        resident, scanned = _geo_updates(df, **kwargs)
        # A count grid, and any normalization of it, is exact on both paths.
        _assert_geo_grids_match(resident, scanned, exact=True)

    @pytest.mark.parametrize("histfunc", ["sum", "mean", "min", "max"])
    @pytest.mark.parametrize(
        "name,df,kwargs",
        [
            ("clean", _geo_df(), {}),
            ("nan_z", _geo_df(z=[_NAN] + _Z[1:]), {}),
            ("null_z", _geo_df(z=[None] + _Z[1:]), {}),
            ("all_null_z", _geo_df(z=[None] * 8), {}),
            ("nan_lat", _geo_df(lat=[_NAN] + _LAT[1:]), {}),
            ("null_lon", _geo_df(lon=[None] + _LON[1:]), {}),
            ("map_viewport", _geo_df(), {"update_range": _MAP_VIEWPORT}),
            ("empty_viewport", _geo_df(), {"update_range": _EMPTY_VIEWPORT}),
            (
                "empty",
                pl.DataFrame(
                    schema={"lat": pl.Float64, "lon": pl.Float64, "z": pl.Float64}
                ),
                {},
            ),
        ],
    )
    def test_reducer_matches_kernel(self, name, df, kwargs, histfunc):
        resident, scanned = _geo_updates(df, z="z", histfunc=histfunc, **kwargs)
        _assert_geo_grids_match(resident, scanned, exact=histfunc in ("min", "max"))

    @pytest.mark.parametrize("z_col", ["lat", "lon"])
    def test_z_may_be_an_axis_column(self, z_col):
        """The fold selects the columns it bins, so a z that is also an axis
        column must be selected once."""
        resident, scanned = _geo_updates(_geo_df(), z=z_col, histfunc="sum")
        _assert_geo_grids_match(resident, scanned, exact=False)

    @pytest.mark.parametrize("histfunc", [None, "sum", "mean", "min", "max"])
    def test_fold_merges_across_batches(self, monkeypatch, histfunc):
        """A frame larger than one chunk must fold to the single-batch grid."""
        monkeypatch.setattr(helpers_mod, "_FOLD_CHUNK_ROWS", 7)
        seen: list[tuple[int, int]] = []
        original = pl.LazyFrame.collect_batches

        def spy(self, *args, **kwargs):
            batches = list(original(self, *args, **kwargs))
            seen.append((kwargs["chunk_size"], len(batches)))
            return batches

        monkeypatch.setattr(pl.LazyFrame, "collect_batches", spy)

        n = 50
        df = pl.DataFrame(
            {
                "lat": [40.0 + (i % 11) * 0.2 for i in range(n)],
                "lon": [-74.0 + (i % 7) * 0.3 for i in range(n)],
                "z": [float((i * 13) % 17) - 8.0 for i in range(n)],
            }
        )
        resident, scanned = _geo_updates(
            df, z=None if histfunc is None else "z", histfunc=histfunc
        )
        _assert_geo_grids_match(
            resident, scanned, exact=histfunc in (None, "min", "max")
        )
        assert seen == [(7, 8)], seen


class TestGeoHist2DResidencySeam:
    """A scan geo histogram folds the kernel over batches, through the engine."""

    @staticmethod
    def _geo_delta(src, event, coords):
        lf = LFQueryBuilder(src)
        geo = GeoHistogram2D(lat="lat", lon="lon", lat_bins=6, lon_bins=5)
        engine = FlexEngine(backend_lf=lf, scalable_traces={geo.uid: geo})
        infos = [
            TraceInfo(
                uid=geo.uid,
                axes=None,
                trace_type="geo_histogram2d",
                figure_uid="fig_map",
            )
        ]
        viewports = {"fig_map": {"coordinates": coords}} if coords else {}
        deltas = engine.process(event, infos, viewports_by_figure=viewports)
        return deltas[0].updates, lf.is_scan

    @pytest.mark.parametrize("zoomed", [False, True], ids=["init", "viewport"])
    def test_scan_matches_resident(self, tmp_path, zoomed):
        n = 20_000
        df = pl.DataFrame(
            {
                "lat": [40.0 + ((i * 7919) % 9973) / 4986.5 for i in range(n)],
                "lon": [-74.0 + ((i * 31) % 97) / 48.5 for i in range(n)],
            }
        )
        path = tmp_path / "geo.parquet"
        df.write_parquet(path)

        coords = [[-73.5, 40.5], [-72.5, 40.5], [-72.5, 41.5], [-73.5, 41.5]]
        if zoomed:
            event = InteractionEvent(
                type="viewport",
                axis_ranges={"coordinates": coords},
                figure_uid="fig_map",
            )
        else:
            event = InteractionEvent(type="init", force_update=True)
            coords = None

        resident, resident_is_scan = self._geo_delta(df, event, coords)
        scanned, scan_is_scan = self._geo_delta(pl.scan_parquet(path), event, coords)
        assert resident_is_scan is False and scan_is_scan is True
        assert resident["lat_edges"] == scanned["lat_edges"]
        assert resident["lon_edges"] == scanned["lon_edges"]
        assert resident["z"] == scanned["z"]

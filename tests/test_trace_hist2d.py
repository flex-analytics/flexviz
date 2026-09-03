"""Unit tests for Histogram2D trace."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from flexviz.LF import LFQueryBuilder
from flexviz.spec import TraceSpec
from flexviz.trace.hist2d import Histogram2D
from flexviz.trace.base import TraceResult


def _aggregate_hist2d(
    df: pl.DataFrame,
    x: str = "x",
    y: str = "y",
    x_bins: int = 5,
    y_bins: int = 5,
    z: str | None = None,
    histfunc: str | None = None,
    histnorm: str | None = None,
    update_range: dict | None = None,
) -> TraceResult:
    lf = LFQueryBuilder(df)
    trace = Histogram2D(
        x=x,
        y=y,
        x_bins=x_bins,
        y_bins=y_bins,
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
    )
    regular_df, _ = lf.aggregate([], [spec])
    return trace._to_update(regular_df)


@pytest.fixture()
def grid_df() -> pl.DataFrame:
    rng = np.random.default_rng(42)
    n = 1000
    return pl.DataFrame(
        {
            "x": rng.uniform(0, 10, n).tolist(),
            "y": rng.uniform(0, 10, n).tolist(),
        }
    )


@pytest.fixture()
def grid_df_with_z() -> pl.DataFrame:
    rng = np.random.default_rng(42)
    n = 1000
    return pl.DataFrame(
        {
            "x": rng.uniform(0, 10, n).tolist(),
            "y": rng.uniform(0, 10, n).tolist(),
            "z": rng.uniform(1, 100, n).tolist(),
        }
    )


class TestHist2DConstructor:
    def test_defaults(self):
        t = Histogram2D(x="x", y="y")
        assert t.trace_type == "histogram2d"
        assert t.x_col == "x"
        assert t.y_col == "y"
        assert t.x_bins == 20
        assert t.y_bins == 20
        assert t.histfunc is None
        assert t.histnorm is None
        assert t.z_col is None
        assert t._axes == ("x", "y")
        assert t.recompute_axes == ("x", "y")
        assert t.update_on_zoom is True
        assert t.overlay_style == "filtered_only"
        assert t.color_scale == "viridis"
        assert t.color_range == "auto"

    def test_custom_bins(self):
        t = Histogram2D(x="a", y="b", x_bins=10, y_bins=15)
        assert t.x_bins == 10
        assert t.y_bins == 15

    def test_custom_heatmap_style(self):
        t = Histogram2D(
            x="a",
            y="b",
            color_scale="plasma",
            color_range=(0.0, 5.0),
        )
        assert t.color_scale == "plasma"
        assert t.color_range == (0.0, 5.0)

    def test_histfunc_without_z_raises(self):
        with pytest.raises(
            ValueError, match="histfunc is only meaningful when z is given"
        ):
            Histogram2D(x="x", y="y", histfunc="sum")

    def test_z_without_histfunc_raises(self):
        with pytest.raises(ValueError, match="histfunc is required when z is given"):
            Histogram2D(x="x", y="y", z="w")

    def test_histfunc_with_z(self):
        t = Histogram2D(x="x", y="y", z="w", histfunc="sum")
        assert t.histfunc == "sum"
        assert t.z_col == "w"

    def test_invalid_histfunc(self):
        with pytest.raises(ValueError, match="histfunc"):
            Histogram2D(x="x", y="y", z="w", histfunc="invalid")

    def test_invalid_histnorm(self):
        with pytest.raises(ValueError, match="histnorm"):
            Histogram2D(x="x", y="y", histnorm="invalid")

    def test_histnorm_density(self):
        t = Histogram2D(x="x", y="y", histnorm="density")
        assert t.histnorm == "density"

    def test_supported_histfunc_options(self):
        for hf in ("sum", "mean", "min", "max"):
            t = Histogram2D(x="x", y="y", z="w", histfunc=hf)
            assert t.histfunc == hf

    @pytest.mark.parametrize("histfunc", ["median", "n_unique"])
    def test_removed_histfunc_options_raise(self, histfunc):
        with pytest.raises(ValueError, match="sum.*mean.*min.*max"):
            Histogram2D(x="x", y="y", z="w", histfunc=histfunc)

    def test_count_implicit_histfunc_is_none(self):
        t = Histogram2D(x="x", y="y")
        assert t.histfunc is None
        assert t.z_col is None


class TestHist2DAggregation:
    def test_output_shape(self, grid_df):
        result = _aggregate_hist2d(grid_df, x_bins=5, y_bins=4)
        assert len(result.updates["x"]) == 5
        assert len(result.updates["y"]) == 4
        assert len(result.updates["z"]) == 4
        assert len(result.updates["z"][0]) == 5

    def test_total_count(self, grid_df):
        result = _aggregate_hist2d(grid_df, x_bins=5, y_bins=5)
        total = sum(sum(v or 0 for v in row) for row in result.updates["z"])
        assert total == len(grid_df)

    def test_viewport_range(self, grid_df):
        result = _aggregate_hist2d(
            grid_df,
            x_bins=3,
            y_bins=3,
            update_range={"x": (2.0, 8.0), "y": (2.0, 8.0)},
        )
        assert len(result.updates["x"]) == 3
        assert len(result.updates["y"]) == 3
        x_centers = result.updates["x"]
        assert x_centers[0] > 2.0
        assert x_centers[-1] < 8.0

    def test_empty_bins_become_null(self):
        df = pl.DataFrame({"x": [0.25, 1.75], "y": [0.25, 1.75]})
        result = _aggregate_hist2d(
            df,
            x_bins=2,
            y_bins=2,
            update_range={"x": (0.0, 2.0), "y": (0.0, 2.0)},
        )
        assert result.updates["z"] == [[1, None], [None, 1]]

    def test_empty_viewport_returns_all_null_bins(self, grid_df):
        result = _aggregate_hist2d(
            grid_df,
            x_bins=3,
            y_bins=3,
            update_range={"x": (20.0, 30.0), "y": (20.0, 30.0)},
        )
        assert all(v is None for row in result.updates["z"] for v in row)

    def test_all_null_input_returns_all_null_bins(self):
        df = pl.DataFrame(
            {"x": [None, None], "y": [None, None]},
            schema={"x": pl.Float64, "y": pl.Float64},
        )
        result = _aggregate_hist2d(df, x_bins=2, y_bins=2)
        assert all(v is None for row in result.updates["z"] for v in row)


class TestHist2DHistfunc:
    @pytest.fixture()
    def exact_z_df(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "x": [0.25, 0.75, 1.25, 1.75, 1.25],
                "y": [0.25, 0.25, 0.25, 1.25, 1.75],
                "z": [2.0, 3.0, 10.0, 20.0, 40.0],
            }
        )

    @pytest.mark.parametrize(
        ("histfunc", "expected"),
        [
            ("sum", [[5.0, 10.0], [None, 60.0]]),
            ("mean", [[2.5, 10.0], [None, 30.0]]),
            ("min", [[2.0, 10.0], [None, 20.0]]),
            ("max", [[3.0, 10.0], [None, 40.0]]),
        ],
    )
    def test_exact_z_reducer_grid(self, exact_z_df, histfunc, expected):
        result = _aggregate_hist2d(
            exact_z_df,
            x_bins=2,
            y_bins=2,
            z="z",
            histfunc=histfunc,
            update_range={"x": (0.0, 2.0), "y": (0.0, 2.0)},
        )
        assert result.updates["z"] == expected

    def test_sum_preserves_zero_value_bins(self):
        df = pl.DataFrame(
            {
                "x": [0.25, 1.25],
                "y": [0.25, 1.25],
                "z": [0.0, 2.0],
            }
        )
        result = _aggregate_hist2d(
            df,
            x_bins=2,
            y_bins=2,
            z="z",
            histfunc="sum",
            update_range={"x": (0.0, 2.0), "y": (0.0, 2.0)},
        )
        assert result.updates["z"] == [[0.0, None], [None, 2.0]]

    def test_sum(self, grid_df_with_z):
        result = _aggregate_hist2d(
            grid_df_with_z, x_bins=3, y_bins=3, z="z", histfunc="sum"
        )
        assert len(result.updates["z"]) == 3
        total_z = sum(v or 0 for row in result.updates["z"] for v in row)
        expected_total = grid_df_with_z["z"].sum()
        assert abs(total_z - expected_total) < 1.0

    def test_mean(self, grid_df_with_z):
        result = _aggregate_hist2d(
            grid_df_with_z, x_bins=3, y_bins=3, z="z", histfunc="mean"
        )
        for row in result.updates["z"]:
            for v in row:
                if v is not None:
                    assert 1.0 <= v <= 100.0

    def test_min(self, grid_df_with_z):
        result = _aggregate_hist2d(
            grid_df_with_z, x_bins=3, y_bins=3, z="z", histfunc="min"
        )
        for row in result.updates["z"]:
            for v in row:
                if v is not None:
                    assert v >= 1.0

    def test_max(self, grid_df_with_z):
        result = _aggregate_hist2d(
            grid_df_with_z, x_bins=3, y_bins=3, z="z", histfunc="max"
        )
        for row in result.updates["z"]:
            for v in row:
                if v is not None:
                    assert v <= 100.0


class TestHist2DHistnorm:
    def test_percent(self, grid_df):
        result = _aggregate_hist2d(grid_df, x_bins=5, y_bins=5, histnorm="percent")
        total = sum(v or 0 for row in result.updates["z"] for v in row)
        assert abs(total - 100.0) < 0.1

    def test_probability(self, grid_df):
        result = _aggregate_hist2d(grid_df, x_bins=5, y_bins=5, histnorm="probability")
        total = sum(v or 0 for row in result.updates["z"] for v in row)
        assert abs(total - 1.0) < 0.01

    def test_density(self, grid_df):
        result = _aggregate_hist2d(grid_df, x_bins=5, y_bins=5, histnorm="density")
        for row in result.updates["z"]:
            for v in row:
                if v is not None:
                    assert v >= 0

    def test_probability_density(self, grid_df):
        result = _aggregate_hist2d(
            grid_df, x_bins=5, y_bins=5, histnorm="probability density"
        )
        for row in result.updates["z"]:
            for v in row:
                if v is not None:
                    assert v >= 0

    def test_histfunc_sum_with_histnorm_percent(self, grid_df_with_z):
        result = _aggregate_hist2d(
            grid_df_with_z,
            x_bins=3,
            y_bins=3,
            z="z",
            histfunc="sum",
            histnorm="percent",
        )
        total = sum(v or 0 for row in result.updates["z"] for v in row)
        assert abs(total - 100.0) < 0.1

    def test_histfunc_sum_with_histnorm_density(self, grid_df_with_z):
        result = _aggregate_hist2d(
            grid_df_with_z,
            x_bins=3,
            y_bins=3,
            z="z",
            histfunc="sum",
            histnorm="density",
        )
        for row in result.updates["z"]:
            for v in row:
                if v is not None:
                    assert v >= 0

    def test_histfunc_sum_with_histnorm_probability_density(self, grid_df_with_z):
        result = _aggregate_hist2d(
            grid_df_with_z,
            x_bins=3,
            y_bins=3,
            z="z",
            histfunc="sum",
            histnorm="probability density",
        )
        for row in result.updates["z"]:
            for v in row:
                if v is not None:
                    assert v >= 0

    def test_histnorm_empty_bins_stay_null_with_z_reducer(self):
        """Empty bins stay None through normalization; zero-sum bins stay 0.0."""
        df = pl.DataFrame(
            {
                "x": [0.25, 1.25],
                "y": [0.25, 1.25],
                "z": [0.0, 5.0],
            }
        )
        result = _aggregate_hist2d(
            df,
            x_bins=2,
            y_bins=2,
            z="z",
            histfunc="sum",
            histnorm="density",
            update_range={"x": (0.0, 2.0), "y": (0.0, 2.0)},
        )
        # Off-diagonal bins are empty → must stay None (gap rendering)
        assert result.updates["z"][0][1] is None
        assert result.updates["z"][1][0] is None
        # Diagonal bins have data; zero-sum bin stays 0.0 (not None)
        assert result.updates["z"][0][0] == 0.0
        assert result.updates["z"][1][1] is not None and result.updates["z"][1][1] > 0


class TestHist2DSpec:
    def test_roundtrip(self):
        t = Histogram2D(
            x="a",
            y="b",
            x_bins=10,
            y_bins=15,
            name="My Heatmap",
            color_scale="plasma",
            color_range=(0.0, 5.0),
        )
        spec = t.to_trace_spec()
        assert spec.trace_type == "histogram2d"
        assert spec.backend_data == {"x": "a", "y": "b"}
        assert spec.params["x_bins"] == 10
        assert spec.params["y_bins"] == 15
        assert spec.display["color_scale"] == "plasma"
        assert spec.display["color_range"] == (0.0, 5.0)

        t2 = Histogram2D.from_trace_spec(spec)
        assert t2.x_col == "a"
        assert t2.y_col == "b"
        assert t2.x_bins == 10
        assert t2.y_bins == 15
        assert t2.color_scale == "plasma"
        assert t2.color_range == (0.0, 5.0)

    def test_roundtrip_with_histfunc(self):
        t = Histogram2D(
            x="a",
            y="b",
            z="w",
            histfunc="sum",
            histnorm="percent",
        )
        spec = t.to_trace_spec()
        assert spec.params["histfunc"] == "sum"
        assert spec.params["histnorm"] == "percent"
        assert spec.backend_data["z"] == "w"

        t2 = Histogram2D.from_trace_spec(spec)
        assert t2.histfunc == "sum"
        assert t2.histnorm == "percent"
        assert t2.z_col == "w"

    def test_legacy_spec_without_display_gets_defaults(self):
        spec = TraceSpec(
            uid="hist2d",
            trace_type="histogram2d",
            backend_data={"x": "x", "y": "y"},
            params={"x_bins": 10, "y_bins": 10},
            display={"name": "Legacy"},
            axes=("x", "y"),
            recompute_axes=("x", "y"),
        )

        trace = Histogram2D.from_trace_spec(spec)
        assert trace.color_scale == "viridis"
        assert trace.color_range == "auto"
        assert trace.histfunc is None
        assert trace.histnorm is None

    @pytest.mark.parametrize("histfunc", ["median", "n_unique"])
    def test_legacy_spec_with_removed_histfunc_raises(self, histfunc):
        spec = TraceSpec(
            uid="hist2d",
            trace_type="histogram2d",
            backend_data={"x": "x", "y": "y", "z": "w"},
            params={"x_bins": 10, "y_bins": 10, "histfunc": histfunc},
            display={"name": "Legacy"},
            axes=("x", "y"),
            recompute_axes=("x", "y"),
        )
        with pytest.raises(ValueError, match="no longer supported"):
            Histogram2D.from_trace_spec(spec)


class TestHist2DZSliceDimensions:
    """z[j] must have exactly x_bins elements and len(z) == y_bins."""

    def test_z_row_width_equals_x_bins(self, grid_df):
        result = _aggregate_hist2d(grid_df, x_bins=7, y_bins=5)
        assert all(len(row) == 7 for row in result.updates["z"])

    def test_z_height_equals_y_bins(self, grid_df):
        result = _aggregate_hist2d(grid_df, x_bins=7, y_bins=5)
        assert len(result.updates["z"]) == 5

    def test_z_flat_total_matches_row_count(self, grid_df):
        result = _aggregate_hist2d(grid_df, x_bins=4, y_bins=4)
        total = sum(v for row in result.updates["z"] for v in row if v is not None)
        assert int(total) == len(grid_df)


class TestHistogram2DHoverBounds:
    def test_hist2d_has_hover_bounds(self):
        """_to_update must include hover_bounds as a 2D list matching z shape."""
        df = pl.DataFrame(
            {
                "a": [float(i % 10) for i in range(40)],
                "b": [float(i // 10) for i in range(40)],
            }
        )
        tr = _aggregate_hist2d(df, x="a", y="b", x_bins=4, y_bins=3)
        assert "hover_bounds" in tr.updates, "hover_bounds must be in updates"
        bounds = tr.updates["hover_bounds"]
        assert isinstance(bounds, list), "hover_bounds must be a list"
        assert len(bounds) == 3, "outer dimension must equal y_bins"
        assert len(bounds[0]) == 4, "inner dimension must equal x_bins"

    def test_hist2d_hover_bounds_cell_shape(self):
        """Each cell must have x0, x1, y0, y1."""
        df = pl.DataFrame(
            {
                "a": [float(i) for i in range(30)],
                "b": [float(i % 6) for i in range(30)],
            }
        )
        tr = _aggregate_hist2d(df, x="a", y="b", x_bins=3, y_bins=2)
        bounds = tr.updates["hover_bounds"]
        for row in bounds:
            for cell in row:
                for key in ("x0", "x1", "y0", "y1"):
                    assert key in cell, f"cell bounds must have {key}"
                assert cell["x1"] > cell["x0"]
                assert cell["y1"] > cell["y0"]

    def test_hist2d_hover_bounds_not_in_z(self):
        """hover_bounds must not be placed in the z array."""
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [1.0, 2.0, 3.0, 4.0]})
        tr = _aggregate_hist2d(df, x="a", y="b", x_bins=2, y_bins=2)
        assert "z" in tr.updates
        # z must be a 2D array of numbers/None, not dicts
        for row in tr.updates["z"]:
            for v in row:
                assert v is None or isinstance(
                    v, (int, float)
                ), "z values must be numbers, not dicts"


class TestHistogram2DHoverSpec:
    def test_hist2d_has_cell_source_mode(self):
        from flexviz.trace.hist2d import Histogram2D

        t = Histogram2D(x="a", y="b")
        spec = t.to_trace_spec()
        assert "cell" in spec.hover.source_modes
        assert "axis" not in spec.hover.source_modes

    def test_hist2d_has_cell_and_axis_target_modes(self):
        from flexviz.trace.hist2d import Histogram2D

        t = Histogram2D(x="a", y="b")
        spec = t.to_trace_spec()
        assert "cell" in spec.hover.target_modes
        assert "axis" in spec.hover.target_modes


class TestHistogram2DTemporal:
    """A temporal x/y axis must bin via its physical representation.

    Regression: ``Histogram2D(x="t", ...)`` handed the temporal column (and
    temporal min/max stats) straight to the numeric ``fixed_hist2d`` kernel,
    which panicked (``not implemented``). The fix bins in physical space and
    returns datetime bin centers so the renderer draws a date axis.
    """

    @staticmethod
    def _temporal_df(tz: str | None = None) -> pl.DataFrame:
        import datetime as dt

        base = dt.datetime(2020, 1, 1)
        ts = [base + dt.timedelta(hours=i) for i in range(50)]
        s = pl.Series("t", ts, dtype=pl.Datetime("us"))
        if tz is not None:
            s = s.dt.replace_time_zone(tz)
        return pl.DataFrame({"t": s, "v": [float(i % 5) for i in range(50)]})

    def test_temporal_x_counts_all(self):
        res = _aggregate_hist2d(self._temporal_df(), x="t", y="v", x_bins=5, y_bins=5)
        total = sum(c for row in res.updates["z"] for c in row if c is not None)
        assert total == 50

    def test_temporal_x_utc_counts_all(self):
        res = _aggregate_hist2d(
            self._temporal_df(tz="UTC"), x="t", y="v", x_bins=5, y_bins=5
        )
        total = sum(c for row in res.updates["z"] for c in row if c is not None)
        assert total == 50

    def test_temporal_x_centers_are_datetime(self):
        import datetime as dt

        res = _aggregate_hist2d(self._temporal_df(), x="t", y="v", x_bins=5, y_bins=5)
        assert all(isinstance(c, dt.datetime) for c in res.updates["x"])
        # y stays numeric
        assert all(isinstance(c, (int, float)) for c in res.updates["y"])

    def test_temporal_x_with_z_reduce(self):
        df = self._temporal_df().with_columns(
            pl.Series("w", [float(i) for i in range(50)])
        )
        res = _aggregate_hist2d(
            df, x="t", y="v", x_bins=5, y_bins=5, z="w", histfunc="mean"
        )
        import datetime as dt

        assert all(isinstance(c, dt.datetime) for c in res.updates["x"])
        # at least one populated cell
        assert any(c is not None for row in res.updates["z"] for c in row)

    def test_temporal_x_viewport_zoom(self):
        # Zoom x to the first 24 hours via a Plotly date-axis range string; the
        # physical bin edges must follow the viewport, not the full domain.
        import datetime as dt

        res = _aggregate_hist2d(
            self._temporal_df(),
            x="t",
            y="v",
            x_bins=5,
            y_bins=5,
            update_range={
                "x": ("2020-01-01 00:00:00", "2020-01-01 23:00:00"),
                "y": (0.0, 5.0),
            },
        )
        total = sum(c for row in res.updates["z"] for c in row if c is not None)
        assert total == 24
        assert all(isinstance(c, dt.datetime) for c in res.updates["x"])

    def test_temporal_x_viewport_tz_offset_bound(self):
        """A tz-aware (offset) viewport bound against a UTC temporal x must bin
        the zoomed window — not raise in the viewport mask path (_hist2d_bounds
        → _typed_range_bounds → _typed_temporal_lit)."""
        res = _aggregate_hist2d(
            self._temporal_df(tz="UTC"),
            x="t",
            y="v",
            x_bins=5,
            y_bins=5,
            update_range={
                "x": ("2020-01-01T02:00:00+02:00", "2020-01-02T01:00:00+02:00"),
                "y": (0.0, 5.0),
            },
        )
        total = sum(c for row in res.updates["z"] for c in row if c is not None)
        assert total == 24

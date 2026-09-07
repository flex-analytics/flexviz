"""Unit tests for Histogram2D trace."""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import polars as pl
import pytest

from flexviz.trace import _hist_helpers as helpers_mod

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


class TestHist2DViewportSnap:
    """A zoomed grid snaps to the lattice of its own bin width, so a pan moves
    the data under a grid that stands still."""

    @staticmethod
    def _edges(result: TraceResult) -> tuple[float, float, int]:
        lo, step, n = result.updates["x_edges"]
        return (lo, lo + n * step, n)

    def test_lattice_aligned_viewport_is_unchanged(self, grid_df):
        # width 1.0, and both bounds are multiples of it.
        result = _aggregate_hist2d(
            grid_df,
            x_bins=4,
            y_bins=5,
            update_range={"x": (2.0, 6.0), "y": (0.0, 10.0)},
        )
        assert self._edges(result) == (2.0, 6.0, 4)

    def test_offset_viewport_snaps_outward_and_gains_a_bin(self, grid_df):
        # width 0.8, so the lattice is ..., 1.6, 2.4, ...: [2.0, 6.0] snaps to
        # [1.6, 6.4] and holds 6 bins instead of 5.
        offset = _aggregate_hist2d(
            grid_df,
            x_bins=5,
            y_bins=5,
            update_range={"x": (2.0, 6.0), "y": (0.0, 10.0)},
        )
        lo, hi, n = self._edges(offset)
        assert n == 6
        assert math.isclose(lo, 1.6) and math.isclose(hi, 6.4)

    def test_a_pan_keeps_every_shared_cell_in_place(self, grid_df):
        """Panning by a fraction of a bin must not move the cell boundaries."""

        def _bounds(x_lo, x_hi):
            res = _aggregate_hist2d(
                grid_df,
                x_bins=5,
                y_bins=5,
                update_range={"x": (x_lo, x_hi), "y": (0.0, 10.0)},
            )
            lo, step, n = res.updates["x_edges"]
            return [lo + i * step for i in range(n)]

        before = _bounds(2.0, 6.0)
        after = _bounds(2.3, 6.3)
        shared = [x for x in after if any(math.isclose(x, b) for b in before)]
        assert len(shared) >= 4

    def test_the_snapped_rectangle_is_fully_counted(self):
        """The mask must follow the snapped edges, so an edge cell is complete."""
        df = pl.DataFrame({"x": [1.7, 2.5, 6.2], "y": [1.0, 1.0, 1.0]})
        result = _aggregate_hist2d(
            df, x_bins=5, y_bins=1, update_range={"x": (2.0, 6.0), "y": (0.0, 2.0)}
        )
        # 1.7 and 6.2 sit outside the viewport but inside the snapped [1.6, 6.4].
        total = sum(v for row in result.updates["z"] for v in row if v is not None)
        assert total == 3


class TestHist2DPartialViewport:
    """The client sends only the axes a zoom moved, so each axis re-bins on its
    own: a zoom on x alone must not drag y back to the full domain."""

    X_RANGE = (2.0, 6.0)
    Y_RANGE = (3.0, 7.0)
    # x_bins=5 over a span of 4.0 gives width 0.8, so [2.0, 6.0] snaps outward
    # to [1.6, 6.4] over 6 bins. y_bins=4 over 4.0 gives width 1.0, a lattice
    # multiple, so [3.0, 7.0] stays put.
    SNAPPED = {"x": (1.6, 6.4, 6), "y": (3.0, 7.0, 4)}

    @staticmethod
    def _axis_edges(result: TraceResult, axis: str) -> tuple[float, float, int]:
        lo, step, n = result.updates[f"{axis}_edges"]
        return (lo, lo + n * step, n)

    @pytest.mark.parametrize("zoomed", [(), ("x",), ("y",), ("x", "y")])
    def test_each_axis_resolves_on_its_own(self, grid_df, zoomed):
        ranges = {"x": self.X_RANGE, "y": self.Y_RANGE}
        update_range = {ax: ranges[ax] for ax in zoomed}
        trace = Histogram2D(x="x", y="y", x_bins=5, y_bins=4)
        assert trace.domain_cols(update_range) == tuple(
            ax for ax in ("x", "y") if ax not in zoomed
        )

        result = _aggregate_hist2d(
            grid_df, x_bins=5, y_bins=4, update_range=update_range
        )
        for axis in ("x", "y"):
            lo, hi, n = self._axis_edges(result, axis)
            if axis in zoomed:
                assert (lo, hi, n) == pytest.approx(self.SNAPPED[axis])
            else:
                assert (lo, hi) == pytest.approx(
                    (grid_df[axis].min(), grid_df[axis].max())
                )

    def test_an_x_only_zoom_masks_x_alone(self, grid_df):
        """Every row inside the snapped x band is counted, whatever its y."""
        result = _aggregate_hist2d(
            grid_df, x_bins=5, y_bins=4, update_range={"x": self.X_RANGE}
        )
        x_lo, x_hi, _ = self.SNAPPED["x"]
        expected = grid_df.filter(pl.col("x").is_between(x_lo, x_hi)).height
        total = sum(v for row in result.updates["z"] for v in row if v is not None)
        assert total == expected


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


class TestHistogram2DBinEdges:
    """The wire format sends one ``[lo, step, n]`` triple per axis; the client
    derives the per-cell hover bounds from them."""

    def test_hist2d_has_both_edge_triples(self):
        df = pl.DataFrame(
            {
                "a": [float(i % 10) for i in range(40)],
                "b": [float(i // 10) for i in range(40)],
            }
        )
        tr = _aggregate_hist2d(df, x="a", y="b", x_bins=4, y_bins=3)
        x_lo, x_step, nb_x = tr.updates["x_edges"]
        y_lo, y_step, nb_y = tr.updates["y_edges"]
        assert (nb_x, nb_y) == (4, 3), "the triples carry the grid"
        assert x_step > 0 and y_step > 0
        # The triples span the same grid the centers sit on.
        assert tr.updates["x"] == pytest.approx(
            [x_lo + (i + 0.5) * x_step for i in range(nb_x)]
        )
        assert tr.updates["y"] == pytest.approx(
            [y_lo + (j + 0.5) * y_step for j in range(nb_y)]
        )
        assert len(tr.updates["z"]) == nb_y
        assert len(tr.updates["z"][0]) == nb_x

    def test_hist2d_edges_are_not_in_z(self):
        """The z array stays numbers only."""
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [1.0, 2.0, 3.0, 4.0]})
        tr = _aggregate_hist2d(df, x="a", y="b", x_bins=2, y_bins=2)
        assert "z" in tr.updates
        for row in tr.updates["z"]:
            for v in row:
                assert v is None or isinstance(v, (int, float))


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
        → _hist2d_bound_lits → _typed_temporal_lit)."""
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


def _hist2d_updates(
    df: pl.DataFrame,
    *,
    x: str = "x",
    y: str = "y",
    z: str | None = None,
    histfunc: str | None = None,
    histnorm: str | None = None,
    x_bins: int = 5,
    y_bins: int = 4,
    update_range: dict | None = None,
) -> tuple[dict, dict]:
    """The ``_to_update`` output of the resident kernel and of the scan fold."""
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


def _assert_grids_match(resident: dict, scanned: dict, *, exact: bool) -> float:
    """Compare the two grids and return the largest relative deviation seen."""
    assert resident["x"] == scanned["x"]
    assert resident["y"] == scanned["y"]
    assert resident["x_edges"] == scanned["x_edges"]
    assert resident["y_edges"] == scanned["y_edges"]
    if exact:
        assert resident["z"] == scanned["z"]
        return 0.0
    worst = 0.0
    for row_a, row_b in zip(resident["z"], scanned["z"]):
        assert len(row_a) == len(row_b)
        for a, b in zip(row_a, row_b):
            assert (a is None) == (b is None)
            if a is None:
                continue
            assert math.isclose(a, b, rel_tol=1e-9)
            if a != b:
                worst = max(worst, abs(a - b) / max(abs(a), 1e-300))
    return worst


_NAN = float("nan")
_X = [0.0, 1.0, 2.5, 3.0, 4.0, 5.5, 7.9, 8.0]
_Y = [0.0, 2.0, 4.0, 6.0, 1.0, 3.0, 5.0, 7.0]
_Z = [1.5, -2.0, 3.25, 0.0, 7.5, -1.25, 4.0, 9.0]


def _xyz_df(
    x: list | pl.Series | None = None,
    y: list | pl.Series | None = None,
    z: list | pl.Series | None = None,
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "x": pl.Series("x", _X) if x is None else x,
            "y": pl.Series("y", _Y) if y is None else y,
            "z": pl.Series("z", _Z) if z is None else z,
        }
    )


class TestHist2DScanFoldEquivalence:
    """A scan source folds the kernel over batches; the grid must match.

    ``scan_source`` picks a formulation, never a result. Count, min and max are
    exact; sum and mean fold in a different order, so they are compared with a
    relative tolerance and an identical null pattern.
    """

    @pytest.mark.parametrize(
        "name,df,kwargs",
        [
            ("f64", _xyz_df(), {}),
            (
                "f32_axes",
                _xyz_df(
                    x=pl.Series("x", _X, dtype=pl.Float32),
                    y=pl.Series("y", _Y, dtype=pl.Float32),
                ),
                {},
            ),
            (
                "int_x",
                _xyz_df(x=pl.Series("x", [0, 1, 2, 3, 4, 5, 7, 8], dtype=pl.Int64)),
                {},
            ),
            ("nan_x", _xyz_df(x=[_NAN] + _X[1:]), {}),
            ("null_x", _xyz_df(x=[None] + _X[1:]), {}),
            ("nan_y", _xyz_df(y=[_NAN] + _Y[1:]), {}),
            ("null_y", _xyz_df(y=[None] + _Y[1:]), {}),
            (
                "viewport",
                _xyz_df(),
                {"update_range": {"x": (2.0, 6.0), "y": (1.0, 6.0)}},
            ),
            ("one_x_bin", _xyz_df(), {"x_bins": 1}),
            (
                "empty",
                pl.DataFrame(
                    schema={"x": pl.Float64, "y": pl.Float64, "z": pl.Float64}
                ),
                {},
            ),
            ("zero_span_x", _xyz_df(x=[3.0] * 8), {}),
            ("histnorm", _xyz_df(), {"histnorm": "probability density"}),
        ],
    )
    def test_count_matches_kernel(self, name, df, kwargs):
        resident, scanned = _hist2d_updates(df, **kwargs)
        # A count grid, and any normalization of it, is exact on both paths.
        _assert_grids_match(resident, scanned, exact=True)

    def test_count_matches_kernel_on_a_temporal_x(self):
        ts = pl.datetime_range(
            dt.datetime(2020, 1, 1),
            dt.datetime(2020, 1, 8),
            interval="1d",
            eager=True,
        ).rename("x")
        resident, scanned = _hist2d_updates(_xyz_df(x=ts))
        _assert_grids_match(resident, scanned, exact=True)

    @pytest.mark.parametrize("histfunc", ["sum", "mean", "min", "max"])
    @pytest.mark.parametrize(
        "name,df",
        [
            ("clean", _xyz_df()),
            ("nan_z", _xyz_df(z=[_NAN] + _Z[1:])),
            ("null_z", _xyz_df(z=[None] + _Z[1:])),
            ("all_null_z", _xyz_df(z=[None] * 8)),
            ("nan_x", _xyz_df(x=[_NAN] + _X[1:])),
            ("null_y", _xyz_df(y=[None] + _Y[1:])),
            (
                "empty",
                pl.DataFrame(
                    schema={"x": pl.Float64, "y": pl.Float64, "z": pl.Float64}
                ),
            ),
        ],
    )
    def test_reducer_matches_kernel(self, name, df, histfunc):
        resident, scanned = _hist2d_updates(df, z="z", histfunc=histfunc)
        _assert_grids_match(resident, scanned, exact=histfunc in ("min", "max"))

    @pytest.mark.parametrize("z_col", ["x", "y"])
    def test_z_may_be_an_axis_column(self, z_col):
        """The fold selects the columns it bins, so a z that is also an axis
        column must be selected once."""
        resident, scanned = _hist2d_updates(_xyz_df(), z=z_col, histfunc="sum")
        _assert_grids_match(resident, scanned, exact=False)

    @pytest.mark.parametrize("histfunc", ["sum", "mean", "min", "max"])
    def test_reducer_matches_kernel_in_a_viewport(self, histfunc):
        resident, scanned = _hist2d_updates(
            _xyz_df(),
            z="z",
            histfunc=histfunc,
            update_range={"x": (2.0, 6.0), "y": (1.0, 6.0)},
        )
        _assert_grids_match(resident, scanned, exact=histfunc in ("min", "max"))

    def test_empty_input_folds_to_the_kernel_shape(self):
        """No batches must leave the grid the kernel returns for empty input:
        zero counts, null reducer cells, bounds echoed."""
        empty = pl.DataFrame(schema={"x": pl.Float64, "y": pl.Float64, "z": pl.Float64})
        resident, scanned = _hist2d_updates(empty)
        # _to_update maps a zero count to None (the gap-rendering contract).
        assert scanned["z"] == resident["z"]
        assert all(v is None for row in scanned["z"] for v in row)
        resident, scanned = _hist2d_updates(empty, z="z", histfunc="sum")
        assert scanned["z"] == resident["z"]
        assert all(v is None for row in scanned["z"] for v in row)

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
                "x": [float(i % 11) for i in range(n)],
                "y": [float(i % 7) for i in range(n)],
                "z": [float((i * 13) % 17) - 8.0 for i in range(n)],
            }
        )
        resident, scanned = _hist2d_updates(
            df, z=None if histfunc is None else "z", histfunc=histfunc
        )
        _assert_grids_match(resident, scanned, exact=histfunc in (None, "min", "max"))
        assert seen == [(7, 8)], seen


def test_collect_batches_chunking_semantics():
    """Pin what the fold relies on: fixed-size batches covering every row.

    ``collect_batches`` is marked unstable in Polars. If the API or the
    chunking changes, this fails before the fold's own tests do.
    """
    n, chunk = 50, 7
    values = list(range(n))
    batches = list(
        pl.DataFrame({"a": values})
        .lazy()
        .collect_batches(chunk_size=chunk, maintain_order=False, engine="streaming")
    )
    assert len(batches) == math.ceil(n / chunk)
    assert sum(b.height for b in batches) == n
    assert sorted(v for b in batches for v in b["a"].to_list()) == values

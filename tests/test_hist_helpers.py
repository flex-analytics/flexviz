"""Unit tests for shared _hist_helpers utilities.

Covers normalize_heatmap_color_scale, normalize_heatmap_color_range, and the
updated bin_2d signature (returns a_step, b_step instead of bin_area).
"""

from __future__ import annotations

import polars as pl
import pytest

from flexviz.trace._hist_helpers import (
    _EPSILON,
    bin_2d,
    normalize_heatmap_color_scale,
    normalize_heatmap_color_range,
)

# ---------------------------------------------------------------------------
# normalize_heatmap_color_scale
# ---------------------------------------------------------------------------


class TestNormalizeHeatmapColorScale:
    def test_none_returns_default(self):
        assert (
            normalize_heatmap_color_scale(None, "viridis", trace_name="T") == "viridis"
        )

    def test_valid_string_passed_through(self):
        assert (
            normalize_heatmap_color_scale("plasma", "viridis", trace_name="T")
            == "plasma"
        )

    def test_empty_string_raises(self):
        with pytest.raises(TypeError, match="non-empty string"):
            normalize_heatmap_color_scale("", "viridis", trace_name="T")

    def test_non_string_raises(self):
        with pytest.raises(TypeError, match="non-empty string"):
            normalize_heatmap_color_scale(42, "viridis", trace_name="T")

    def test_trace_name_in_error(self):
        with pytest.raises(TypeError, match="MyTrace"):
            normalize_heatmap_color_scale(123, "viridis", trace_name="MyTrace")


# ---------------------------------------------------------------------------
# normalize_heatmap_color_range
# ---------------------------------------------------------------------------


class TestNormalizeHeatmapColorRange:
    def test_none_returns_default_auto(self):
        assert normalize_heatmap_color_range(None, "auto", trace_name="T") == "auto"

    def test_none_returns_default_tuple(self):
        assert normalize_heatmap_color_range(None, (-1.0, 1.0), trace_name="T") == (
            -1.0,
            1.0,
        )

    def test_auto_string(self):
        assert normalize_heatmap_color_range("auto", "auto", trace_name="T") == "auto"

    def test_valid_tuple(self):
        result = normalize_heatmap_color_range((0.0, 10.0), "auto", trace_name="T")
        assert result == (0.0, 10.0)

    def test_list_accepted(self):
        result = normalize_heatmap_color_range([0.0, 10.0], "auto", trace_name="T")
        assert result == (0.0, 10.0)

    def test_non_finite_lo_raises(self):
        with pytest.raises(ValueError, match="finite"):
            normalize_heatmap_color_range((float("inf"), 10.0), "auto", trace_name="T")

    def test_non_finite_hi_raises(self):
        with pytest.raises(ValueError, match="finite"):
            normalize_heatmap_color_range((0.0, float("nan")), "auto", trace_name="T")

    def test_inverted_range_raises(self):
        with pytest.raises(ValueError, match="min < max"):
            normalize_heatmap_color_range((10.0, 0.0), "auto", trace_name="T")

    def test_equal_bounds_raises(self):
        with pytest.raises(ValueError, match="min < max"):
            normalize_heatmap_color_range((5.0, 5.0), "auto", trace_name="T")

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError, match="'auto' or a"):
            normalize_heatmap_color_range("bad", "auto", trace_name="T")

    def test_trace_name_in_error(self):
        with pytest.raises(TypeError, match="MyTrace"):
            normalize_heatmap_color_range("nope", "auto", trace_name="MyTrace")


# ---------------------------------------------------------------------------
# bin_2d return signature: (a_centers, b_centers, z_flat, a_step, b_step)
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_df() -> pl.DataFrame:
    import numpy as np

    rng = np.random.default_rng(0)
    n = 500
    return pl.DataFrame(
        {
            "a": rng.uniform(0.0, 10.0, n).tolist(),
            "b": rng.uniform(0.0, 5.0, n).tolist(),
        }
    )


class TestBin2DReturnSteps:
    def test_returns_five_values(self, simple_df):
        result = bin_2d(simple_df, "a", "b", 4, 3)
        assert len(result) == 5

    def test_a_step_matches_center_spacing(self, simple_df):
        a_centers, _, _, a_step, _ = bin_2d(simple_df, "a", "b", 4, 3)
        # Centers are evenly spaced; each gap should equal a_step
        for i in range(1, len(a_centers)):
            assert abs((a_centers[i] - a_centers[i - 1]) - a_step) < 1e-8

    def test_b_step_matches_center_spacing(self, simple_df):
        _, b_centers, _, _, b_step = bin_2d(simple_df, "a", "b", 4, 3)
        for i in range(1, len(b_centers)):
            assert abs((b_centers[i] - b_centers[i - 1]) - b_step) < 1e-8

    def test_a_step_formula(self, simple_df):
        """a_step == (_EPSILON + a_max - a_min) / nb_a"""
        a_centers, _, _, a_step, _ = bin_2d(simple_df, "a", "b", 10, 5)
        a_min = simple_df["a"].min()
        a_max = simple_df["a"].max()
        expected = (_EPSILON + (a_max - a_min)) / 10
        assert abs(a_step - expected) < 1e-10

    def test_edges_from_centers_match(self, simple_df):
        """Edges derived from centers and step must satisfy center[i] == (edge[i]+edge[i+1])/2."""
        a_centers, b_centers, _, a_step, b_step = bin_2d(simple_df, "a", "b", 6, 4)

        a_min = a_centers[0] - 0.5 * a_step
        a_edges = [a_min + i * a_step for i in range(len(a_centers) + 1)]
        for i, c in enumerate(a_centers):
            midpoint = (a_edges[i] + a_edges[i + 1]) / 2
            assert abs(c - midpoint) < 1e-10, f"a_center[{i}] mismatch"

        b_min = b_centers[0] - 0.5 * b_step
        b_edges = [b_min + j * b_step for j in range(len(b_centers) + 1)]
        for j, c in enumerate(b_centers):
            midpoint = (b_edges[j] + b_edges[j + 1]) / 2
            assert abs(c - midpoint) < 1e-10, f"b_center[{j}] mismatch"

    def test_empty_df_returns_range_step(self):
        """For empty df the step comes from the supplied a_range / b_range."""
        empty = pl.DataFrame(
            {"a": pl.Series([], dtype=pl.Float64), "b": pl.Series([], dtype=pl.Float64)}
        )
        a_centers, b_centers, z_flat, a_step, b_step = bin_2d(
            empty,
            "a",
            "b",
            5,
            4,
            a_range=(0.0, 10.0),
            b_range=(0.0, 8.0),
        )
        assert abs(a_step - 10.0 / 5) < 1e-10
        assert abs(b_step - 8.0 / 4) < 1e-10
        assert len(a_centers) == 5
        assert len(b_centers) == 4
        assert all(v is None for v in z_flat)

    def test_empty_df_no_range_uses_unit_step(self):
        empty = pl.DataFrame(
            {"a": pl.Series([], dtype=pl.Float64), "b": pl.Series([], dtype=pl.Float64)}
        )
        _, _, _, a_step, b_step = bin_2d(empty, "a", "b", 3, 3)
        # default a_range=[0,1], so step = 1/3
        assert abs(a_step - 1.0 / 3) < 1e-10

    def test_z_flat_length(self, simple_df):
        _, _, z_flat, _, _ = bin_2d(simple_df, "a", "b", 6, 4)
        assert len(z_flat) == 24  # nb_a * nb_b

    def test_count_total_equals_row_count(self, simple_df):
        _, _, z_flat, _, _ = bin_2d(simple_df, "a", "b", 5, 5)
        total = sum(v for v in z_flat if v is not None)
        assert int(total) == len(simple_df)

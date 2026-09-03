"""Unit tests for shared _hist_helpers utilities."""

from __future__ import annotations

import pytest

from flexviz.trace._hist_helpers import (
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

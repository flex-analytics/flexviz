"""Unit tests for shared _hist_helpers utilities."""

from __future__ import annotations

import math

import pytest

from flexviz.trace._hist_helpers import (
    _snap_range,
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
# _snap_range
# ---------------------------------------------------------------------------


def _edges(lo: float, hi: float, n: int) -> list[float]:
    step = (hi - lo) / n
    return [lo + i * step for i in range(n + 1)]


class TestSnapRange:
    """A zoomed 2-D histogram snaps its viewport to the lattice of its own bin
    width, so a pan keeps every cell where it was."""

    def test_exact_multiples_stay_put(self):
        assert _snap_range(0.0, 10.0, 5) == (0.0, 10.0, 5)
        assert _snap_range(2.0, 8.0, 3) == (2.0, 8.0, 3)

    def test_offset_viewport_grows_by_one_bin(self):
        lo, hi, n = _snap_range(0.5, 10.5, 5)
        assert (lo, hi, n) == (0.0, 12.0, 6)

    @pytest.mark.parametrize("shift", [0.0, 0.5, 1.0, 1.7, -3.3])
    def test_count_is_n_or_n_plus_one(self, shift):
        _, _, n = _snap_range(4.0 + shift, 9.0 + shift, 5)
        assert n in (5, 6)

    def test_a_pan_keeps_the_lattice(self):
        # Same span, moved by half a bin: both grids sit on multiples of the
        # same width, so their shared cells coincide exactly.
        width = 2.0
        a_lo, a_hi, a_n = _snap_range(0.0, 10.0, 5)
        b_lo, b_hi, b_n = _snap_range(1.0, 11.0, 5)
        for lo, hi, n in ((a_lo, a_hi, a_n), (b_lo, b_hi, b_n)):
            for edge in _edges(lo, hi, n):
                assert math.isclose(edge / width, round(edge / width), abs_tol=1e-9)
        shared = set(_edges(a_lo, a_hi, a_n)) & set(_edges(b_lo, b_hi, b_n))
        assert len(shared) > 1

    def test_negative_coordinates(self):
        assert _snap_range(-10.0, -5.0, 5) == (-10.0, -5.0, 5)
        lo, hi, n = _snap_range(-9.5, -4.5, 5)
        assert (lo, hi, n) == (-10.0, -4.0, 6)

    def test_degenerate_span_is_returned_unchanged(self):
        assert _snap_range(3.0, 3.0, 4) == (3.0, 3.0, 4)
        assert _snap_range(5.0, 1.0, 4) == (5.0, 1.0, 4)

    def test_float_error_does_not_buy_a_bin(self):
        # 0.1 * 3 != 0.3 in binary floating point; the epsilon must absorb it.
        lo, hi, n = _snap_range(0.1, 0.1 + 0.3, 3)
        assert n == 3

    def test_snapped_range_covers_the_viewport(self):
        lo, hi, n = _snap_range(1.3, 7.9, 4)
        assert lo <= 1.3 and hi >= 7.9
        assert n == 5

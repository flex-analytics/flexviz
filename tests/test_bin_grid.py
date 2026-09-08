"""Unit tests for the shared bin grid."""

from __future__ import annotations

import math

import pytest

from flexviz.trace.bin_grid import snap_range


def _edges(lo: float, hi: float, n: int) -> list[float]:
    step = (hi - lo) / n
    return [lo + i * step for i in range(n + 1)]


class TestSnapRange:
    """A zoomed 2-D histogram snaps its viewport to the lattice of its own bin
    width, so a pan keeps every cell where it was."""

    def test_exact_multiples_stay_put(self):
        assert snap_range(0.0, 10.0, 5) == (0.0, 10.0, 5)
        assert snap_range(2.0, 8.0, 3) == (2.0, 8.0, 3)

    def test_offset_viewport_grows_by_one_bin(self):
        lo, hi, n = snap_range(0.5, 10.5, 5)
        assert (lo, hi, n) == (0.0, 12.0, 6)

    @pytest.mark.parametrize("shift", [0.0, 0.5, 1.0, 1.7, -3.3])
    def test_count_is_n_or_n_plus_one(self, shift):
        _, _, n = snap_range(4.0 + shift, 9.0 + shift, 5)
        assert n in (5, 6)

    def test_a_pan_keeps_the_lattice(self):
        # Same span, moved by half a bin: both grids sit on multiples of the
        # same width, so their shared cells coincide exactly.
        width = 2.0
        a_lo, a_hi, a_n = snap_range(0.0, 10.0, 5)
        b_lo, b_hi, b_n = snap_range(1.0, 11.0, 5)
        for lo, hi, n in ((a_lo, a_hi, a_n), (b_lo, b_hi, b_n)):
            for edge in _edges(lo, hi, n):
                assert math.isclose(edge / width, round(edge / width), abs_tol=1e-9)
        shared = set(_edges(a_lo, a_hi, a_n)) & set(_edges(b_lo, b_hi, b_n))
        assert len(shared) > 1

    def test_negative_coordinates(self):
        assert snap_range(-10.0, -5.0, 5) == (-10.0, -5.0, 5)
        lo, hi, n = snap_range(-9.5, -4.5, 5)
        assert (lo, hi, n) == (-10.0, -4.0, 6)

    def test_degenerate_span_is_returned_unchanged(self):
        assert snap_range(3.0, 3.0, 4) == (3.0, 3.0, 4)
        assert snap_range(5.0, 1.0, 4) == (5.0, 1.0, 4)

    def test_float_error_does_not_buy_a_bin(self):
        # 0.1 * 3 != 0.3 in binary floating point; the epsilon must absorb it.
        lo, hi, n = snap_range(0.1, 0.1 + 0.3, 3)
        assert n == 3

    def test_snapped_range_covers_the_viewport(self):
        lo, hi, n = snap_range(1.3, 7.9, 4)
        assert lo <= 1.3 and hi >= 7.9
        assert n == 5

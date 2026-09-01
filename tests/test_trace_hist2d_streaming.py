"""Equivalence tests for the streaming (scan-source) 2D histogram.

Same bar as the 1D streaming plan: the scan path and the resident path answer
the same question, so the output must be identical -- counts *and* the bin
bounds packed alongside them, because ``_to_update`` derives cell centres and
hover edges from those bounds.

Two things make it non-trivial, and both cost a bug when they were missed.

The kernels skip null and NaN rows. A plan that bins them anyway does not just
disagree, it raises: a strict cast of NaN to ``Int32`` fails. So the tests here
assert the *answer* (the total equals the clean row count) rather than only
that the two paths agree -- two paths that both bin NaN into cell 0 would agree
and both be wrong.

The 2D kernels widen the span before computing the scale, but return the
*raw* bounds they were handed (``expressions.rs``, ``fixed_hist2d_impl``).
Packing the widened bound instead shifts every cell centre.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from flexviz.LF import LFQueryBuilder
from flexviz.trace.hist2d import Histogram2D


def _run(df: pl.DataFrame, *, scan_source: bool, x_bins=4, y_bins=4, **kw) -> dict:
    """Aggregate one Histogram2D and return the kernel-shaped struct."""
    trace = Histogram2D(x="x", y="y", x_bins=x_bins, y_bins=y_bins)
    spec = trace.get_aggregation_spec(
        kw.pop("update_range", {}), schema=df.schema, scan_source=scan_source, **kw
    )
    regular, _ = LFQueryBuilder(df.lazy()).aggregate([], [spec])
    return regular[trace.uid][0]


def _both(df: pl.DataFrame, **kw) -> tuple[dict, dict]:
    return _run(df, scan_source=False, **kw), _run(df, scan_source=True, **kw)


def _same(kernel: dict, streaming: dict) -> None:
    assert streaming["z_flat"] == kernel["z_flat"], "cell counts differ"
    for edge in ("x_lo", "x_hi", "y_lo", "y_hi"):
        assert streaming[edge] == kernel[edge], f"{edge} differs"


def _grid(n: int) -> pl.DataFrame:
    """A deterministic spread of points over both axes."""
    return pl.DataFrame(
        {
            "x": [((i * 7919) % 1000) / 7.0 for i in range(n)],
            "y": [((i * 104729) % 997) / 3.0 for i in range(n)],
        }
    )


# ---- the seam --------------------------------------------------------------


class TestScanSelectsTheStreamingPlan:
    def test_scan_brings_a_plan_and_resident_does_not(self):
        h = Histogram2D(x="x", y="y")
        assert h.get_aggregation_spec({}, scan_source=False).plan is None
        assert h.get_aggregation_spec({}, scan_source=True).plan is not None

    def test_z_reduce_keeps_the_kernel_even_on_a_scan(self):
        """The streaming aggregate cannot reproduce the per-cell reducer."""
        h = Histogram2D(x="x", y="y", z="z", histfunc="mean")
        assert h.get_aggregation_spec({}, scan_source=True).plan is None


# ---- equivalence -----------------------------------------------------------


class TestStreamingMatchesKernel:
    @pytest.mark.parametrize(
        "name,df",
        [
            ("f64", _grid(5_000)),
            ("f32", _grid(5_000).cast({"x": pl.Float32, "y": pl.Float32})),
            ("i64", _grid(5_000).cast({"x": pl.Int64, "y": pl.Int64})),
            ("mixed_dtypes", _grid(5_000).cast({"x": pl.Int64})),
            (
                "constant_x",
                pl.DataFrame(
                    {"x": [3.0] * 500, "y": [float(i % 17) for i in range(500)]}
                ),
            ),
            ("single_point", pl.DataFrame({"x": [1.0], "y": [2.0]})),
            (
                "empty",
                pl.DataFrame(
                    {"x": [], "y": []}, schema={"x": pl.Float64, "y": pl.Float64}
                ),
            ),
        ],
    )
    def test_identical_across_dtypes_and_shapes(self, name, df):
        _same(*_both(df))

    @pytest.mark.parametrize("bins", [(1, 1), (1, 7), (7, 1), (3, 5), (64, 64)])
    def test_identical_across_bin_counts(self, bins):
        nx, ny = bins
        _same(*_both(_grid(5_000), x_bins=nx, y_bins=ny))

    @pytest.mark.parametrize(
        "x_range,y_range",
        [
            ((10.0, 90.0), (10.0, 90.0)),
            ((0.0, 50.0), (100.0, 300.0)),
            ((-20.0, 500.0), (-5.0, 500.0)),
        ],
    )
    def test_identical_under_a_viewport(self, x_range, y_range):
        update_range = {"x": list(x_range), "y": list(y_range)}
        _same(*_both(_grid(5_000), update_range=update_range))

    def test_identical_on_temporal_axes(self):
        stamps = pl.datetime_range(
            dt.datetime(2020, 1, 1), dt.datetime(2020, 1, 20), interval="1h", eager=True
        )
        df = pl.DataFrame(
            {"x": stamps, "y": [float(i % 53) for i in range(len(stamps))]}
        )
        _same(*_both(df, x_bins=12, y_bins=8))

    def test_float32_bins_at_float64_precision(self):
        """Polars keeps ``Float32 - <python float>`` in Float32, but the
        kernels widen to f64 before any arithmetic. Without the cast a Float32
        column bins at Float32 precision and drifts: this value sits one bin
        high without it. Found by sweeping random (lo, span, bins) triples,
        which disagreed on about 2 percent of them.
        """
        lo, span, bins = 96.82396809638065, 114.71186180643278, 13
        df = pl.DataFrame(
            {"x": [185.06386], "y": [185.06386]},
            schema={"x": pl.Float32, "y": pl.Float32},
        )
        update_range = {"x": [lo, lo + span], "y": [lo, lo + span]}
        kernel, streaming = _both(
            df, x_bins=bins, y_bins=bins, update_range=update_range
        )
        _same(kernel, streaming)
        assert streaming["z_flat"].index(1) == 9 * bins + 9, "bin 9 on both axes"


# ---- null and NaN ----------------------------------------------------------


class TestNullAndNanAreSkipped:
    """The kernels drop these rows. Binning them raises on the strict cast."""

    @pytest.mark.parametrize("bad", [None, float("nan")])
    @pytest.mark.parametrize("axis", ["x", "y", "both"])
    def test_skipped_not_binned(self, bad, axis):
        clean = {"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]}
        dirty = {"x": list(clean["x"]), "y": list(clean["y"])}
        dirty["x"].append(bad if axis in ("x", "both") else 1.0)
        dirty["y"].append(bad if axis in ("y", "both") else 1.0)

        kernel, streaming = _both(pl.DataFrame(dirty))
        _same(kernel, streaming)
        # Assert the answer, not only the agreement: the bad row must vanish,
        # not land in cell 0 on both paths.
        assert sum(streaming["z_flat"]) == 3

    def test_a_nan_column_does_not_raise(self):
        """Regression: the plan used a strict cast, so one NaN was a 500."""
        df = pl.DataFrame({"x": [float("nan")] * 10, "y": [1.0] * 10})
        assert sum(_run(df, scan_source=True)["z_flat"]) == 0

    def test_an_all_null_frame_is_all_zeros(self):
        df = pl.DataFrame(
            {"x": [None] * 10, "y": [None] * 10},
            schema={"x": pl.Float64, "y": pl.Float64},
        )
        _same(*_both(df))
        assert sum(_run(df, scan_source=True)["z_flat"]) == 0


# ---- bounds ----------------------------------------------------------------


class TestBounds:
    def test_bounds_are_the_raw_bounds_not_the_widened_span(self):
        """Regression: the plan packed ``lo + span``, which carries the
        kernel's 1e-10 span pad and shifts every cell centre."""
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
        kernel, streaming = _both(df)
        assert streaming["x_hi"] == 3.0
        assert streaming["y_hi"] == 3.0
        _same(kernel, streaming)

    def test_a_value_exactly_at_hi_lands_in_the_top_cell(self):
        """What the span pad exists for; both paths must agree on it."""
        df = pl.DataFrame({"x": [0.0, 4.0], "y": [0.0, 4.0]})
        kernel, streaming = _both(df, x_bins=4, y_bins=4)
        _same(kernel, streaming)
        assert streaming["z_flat"][-1] == 1, "the max value belongs in the top cell"

    def test_inverted_bounds_raise(self):
        """The kernel has ``polars_ensure!(x_hi >= x_lo)``; the plan must not
        quietly return zeros instead."""
        df = pl.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]})
        with pytest.raises(ValueError, match="inverted"):
            _run(df, scan_source=True, update_range={"x": [5.0, 1.0], "y": [0.0, 3.0]})


# ---- totals ----------------------------------------------------------------


class TestEveryRowIsCountedOnce:
    def test_no_row_is_dropped_or_double_counted(self):
        df = _grid(20_000)
        assert sum(_run(df, scan_source=True)["z_flat"]) == df.height

    def test_out_of_viewport_rows_are_excluded_on_both_paths(self):
        df = _grid(20_000)
        kernel, streaming = _both(
            df, update_range={"x": [10.0, 90.0], "y": [10.0, 90.0]}
        )
        _same(kernel, streaming)
        assert sum(streaming["z_flat"]) < df.height, "the viewport must exclude rows"

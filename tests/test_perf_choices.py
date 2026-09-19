"""Performance-decision checks: one test per decision.

Each test names the decision it checks, the evidence that motivated it (with
numbers), and what the assertion verifies. Run with ``make test-perf``;
excluded from ``make test`` (marker ``benchmark``) because timings need a
quiet machine and at least 4 Polars threads.
"""

from __future__ import annotations

import statistics
import time

import numpy as np
import polars as pl
import pytest
from ooc_child import PeakSampler

from flexviz.LF import LFQueryBuilder

pytestmark = pytest.mark.benchmark

# 80 MB of Float64: past the streaming engine's fixed overhead (it already wins
# at 10M rows) and still ~2 ms per probe, so the whole check costs under a second.
PROBE_ROWS = 10_000_000
# Measured on this data at 4 threads, the CI runner's core count: 2.47 ms
# in-memory against 1.53 ms streaming, a ratio of 0.62. 0.85 leaves headroom
# for a shared runner. Below 4 threads the streaming engine has no win, hence
# the skip.
PROBE_TIME_RATIO = 0.85
MIN_PROBE_THREADS = 4
# The streaming engine reduces morsel by morsel, so it must never copy the
# column. Measured 0.2 MB at 200M rows; 64 MB is well under the 80 MB column.
PROBE_PEAK_MB = 64


def _median_ms(call) -> float:
    call()  # warm-up: first touch of the column
    times = []
    for _ in range(5):
        start = time.perf_counter()
        call()
        times.append(time.perf_counter() - start)
    return statistics.median(times) * 1e3


@pytest.mark.skipif(
    pl.thread_pool_size() < MIN_PROBE_THREADS,
    reason=f"the streaming min/max needs >= {MIN_PROBE_THREADS} threads to win",
)
def test_domain_probe_streams_on_resident_frames() -> None:
    """Decision: `LFQueryBuilder.physical_minmax` collects its min/max select
    with the streaming engine even on a resident frame, although every other
    resident collect is pinned to the in-memory engine.

    Evidence: the in-memory engine reads the column twice (one pass for min,
    one for max) while the streaming engine folds both into one morsel pass;
    measured on an M5 at 10M rows 1.87 vs 3.51 ms (0.53x), at 200M rows
    34.7 vs 63.4 ms, and 0.62x at 4 threads; below 4 threads the streaming
    engine has no win, hence the skip.

    Check: the probe takes at most PROBE_TIME_RATIO of the in-memory select,
    and its peak memory stays under PROBE_PEAK_MB (the engine must never copy
    the column).
    """
    rng = np.random.default_rng(0)
    df = pl.DataFrame({"v": rng.standard_normal(PROBE_ROWS)})
    exprs = [pl.col("v").min().alias("__min_v__"), pl.col("v").max().alias("__max_v__")]

    # One sampler over both timings: it costs each engine the same, and a 1 ms
    # poll is needed because a probe lasts ~2 ms.
    with PeakSampler(interval=0.001) as sampler:
        # The old code path, for reference.
        reference_ms = _median_ms(
            lambda: df.lazy().select(exprs).collect(engine="in-memory")
        )
        # A fresh builder per call: the min/max memo is per builder.
        probe_ms = _median_ms(lambda: LFQueryBuilder(df.lazy()).physical_minmax(["v"]))

    assert probe_ms <= PROBE_TIME_RATIO * reference_ms, (
        f"probe took {probe_ms:.2f} ms, over {PROBE_TIME_RATIO} x the "
        f"{reference_ms:.2f} ms in-memory select"
    )
    assert sampler.peak_mb <= PROBE_PEAK_MB, (
        f"the min/max select peaked {sampler.peak_mb:.1f} MB over its baseline, "
        f"above the {PROBE_PEAK_MB} MB cap"
    )


# 20M rows in ~125k-row chunks: a multi-chunk resident frame, the shape a
# concatenated source has, and small enough to keep the whole check under 3 s.
FILTERED_PROBE_ROWS = 20_000_000
FILTERED_PROBE_CHUNK = 125_000
# Observed ratios on this shape were 0.2 to 1.2, so 2.0 catches a switch back
# to the streaming engine without flagging timing noise.
FILTERED_PROBE_RATIO = 2.0


@pytest.mark.skipif(
    pl.thread_pool_size() < MIN_PROBE_THREADS,
    reason=f"the engine comparison needs >= {MIN_PROBE_THREADS} threads",
)
def test_filtered_domain_probe_stays_in_memory() -> None:
    """Decision: a filtered `LFQueryBuilder.physical_minmax` on a resident
    frame collects with the in-memory engine (the builder's `collect_engine`),
    not with the streaming engine the unfiltered probe uses.

    Evidence: on the real request path at 100M rows (M5, 2026-09-19) the
    in-memory engine is no worse than the streaming one on either frame shape,
    and 5x faster on a resident frame read from Parquet: 40 ms against 215 ms
    with 12.5M surviving rows. At this test's scale the gap is only 1.1x to 3x
    and unstable, so the check is a guard against a switch back to streaming,
    not a proof of the 5x.

    Check: on a multi-chunk resident frame the filtered probe takes at most
    FILTERED_PROBE_RATIO of the same select on the streaming engine, and it
    returns the bounds of the surviving rows, so it cannot pass on a no-op.
    """
    width = FILTERED_PROBE_ROWS // 8
    df = pl.concat(
        [
            pl.DataFrame(
                {"x": np.arange(start, start + FILTERED_PROBE_CHUNK, dtype=np.int64)}
            ).with_columns((pl.col("x") // width).cast(pl.Int32).alias("grp"))
            for start in range(0, FILTERED_PROBE_ROWS, FILTERED_PROBE_CHUNK)
        ],
        rechunk=False,
    )
    pred = pl.col("grp").is_in([3])
    minmax = [
        pl.col("x").min().alias("__min_x__"),
        pl.col("x").max().alias("__max_x__"),
    ]

    def probe():
        # A fresh builder per call, as a request gets.
        return LFQueryBuilder(df).physical_minmax(["x"], filter_exprs=[pred])

    def reference():
        return df.lazy().filter(pred).select(minmax).collect(engine="streaming")

    probe_ms: list[float] = []
    reference_ms: list[float] = []
    # Alternate the two, so a warming cache or a thermal drift costs both.
    for run in range(6):  # one warm-up pass, then five timed ones
        for call, times in ((probe, probe_ms), (reference, reference_ms)):
            start = time.perf_counter()
            call()
            if run:
                times.append((time.perf_counter() - start) * 1e3)

    assert probe() == {"x": (3 * width, 4 * width - 1)}
    median_probe = statistics.median(probe_ms)
    median_reference = statistics.median(reference_ms)
    assert median_probe <= FILTERED_PROBE_RATIO * median_reference, (
        f"the filtered probe took {median_probe:.2f} ms, over "
        f"{FILTERED_PROBE_RATIO} x the {median_reference:.2f} ms streaming select"
    )

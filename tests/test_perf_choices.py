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


# The streaming engine reduces over the filtered morsels, so a filtered probe
# must not materialize the surviving rows either. Measured 0.1 to 2.3 MB on a
# warm allocator against 52 to 62 MB for the same select on the in-memory
# engine; 8 MB sits between the two.
FILTERED_PEAK_MB = 8


def test_filtered_domain_probe_streams_too() -> None:
    """Decision: a filtered `LFQueryBuilder.physical_minmax` streams like the
    unfiltered one, on a resident frame as well as on a scan.

    Evidence: only the predicate kind flips the engine winner, not the filter.
    On `is_between` and on `==` the streaming engine wins on resident frames
    (M5, 100M rows, y band keeping 10M: 34.0 ms / 0.0 MB streaming against
    53.5 ms / 235 MB in-memory; 50M with 90 % kept: 12.5 against 29.3 ms and
    9.5 against 118 MB). The deleted `test_filtered_domain_probe_stays_in_memory`
    pinned the in-memory engine from an `is_in` fixture, the one predicate
    shape where in-memory wins, and values clauses no longer compile to
    `is_in` below the equality-chain cutoff.

    Check: the probe's collect names the streaming engine, its peak stays
    under FILTERED_PEAK_MB, and it returns the surviving rows' bounds, so it
    cannot pass on a no-op. No wall assertion: at this scale the gap is 1.1x
    to 3x and unstable.
    """
    rng = np.random.default_rng(0)
    df = pl.DataFrame(
        {
            "x": np.arange(PROBE_ROWS, dtype=np.int64),
            "y": rng.standard_normal(PROBE_ROWS),
        }
    )
    pred = pl.col("y").is_between(-0.5, 0.5)

    engines: list[str | None] = []
    original_collect = pl.LazyFrame.collect

    def tracked_collect(self, *args, **kwargs):
        engines.append(kwargs.get("engine"))
        return original_collect(self, *args, **kwargs)

    # Warm-up: the first probe also pays the allocator's first touch (12 MB).
    LFQueryBuilder(df).physical_minmax(["x"], filter_exprs=[pred])

    pl.LazyFrame.collect = tracked_collect
    try:
        with PeakSampler(interval=0.001) as sampler:
            bounds = LFQueryBuilder(df).physical_minmax(["x"], filter_exprs=[pred])
    finally:
        pl.LazyFrame.collect = original_collect

    assert engines == ["streaming"]
    assert sampler.peak_mb <= FILTERED_PEAK_MB, (
        f"the filtered probe peaked {sampler.peak_mb:.1f} MB over its baseline, "
        f"above the {FILTERED_PEAK_MB} MB cap"
    )
    kept = df.filter(pred)
    assert bounds == {"x": (kept["x"].min(), kept["x"].max())}


# 20 categories over PROBE_ROWS rows, the shape of a category cross-filter.
CHAIN_CATEGORIES = 20
CHAIN_VALUES = 5
# The chain streams: it never holds more than a morsel (measured 0.0 to
# 0.3 MB).
CHAIN_PEAK_MB = 8


def test_small_values_clause_compiles_to_an_equality_chain() -> None:
    """Decision: on a resident source a values clause of at most
    `_EQUALITY_CHAIN_MAX_VALUES` members compiles to
    `any_horizontal(col == v, ...)` instead of `is_in`. A scan source always
    compiles `is_in`. The source kind is the discriminator, not k.

    Evidence: M5, 100M rows, a String column, both forms on the streaming
    engine (the engine the product uses), probe over the filtered frame.
    Resident, k=1/5/10/14/20: the chain 18/54/102/132/172 ms against
    280/581/947/961/634 ms for `is_in`, 7 to 15x. The chain costs about 9 ms
    per extra value while `is_in` stays flat at 500 to 1000 ms, so the two
    cross around k 80 to 100 (200 categories: k=60 545 against 798 ms, k=80
    746 against 784, k=100 950 against 922). 64 keeps a margin.

    On a Parquet scan both forms are pushed into the scan node (identical
    plans) and the ranking flips: `is_in` costs about 2.4 ms per extra value
    on a 90 ms base, the chain about 12.7 ms, so the chain loses from k=2 on
    (k=5 164 against 126 ms, k=20 332 against 144). End to end through the
    engine on a scan, k=14: 773 against 521 ms.

    Memory is NOT the argument. Both forms stream in a few MB on the
    streaming engine (0.2 to 5 MB). The large `is_in` peaks reported earlier
    belong to the in-memory engine, which no product path takes here.

    Check: a k=5 resident clause compiles without `is_in`, a k=5 scan clause
    keeps `is_in`, both forms select the same rows, and the chain's streaming
    select stays under CHAIN_PEAK_MB.
    """
    from flexviz.predicates import predicates_to_expr
    from flexviz.spec import ClauseFilter, SelectionPredicate

    rng = np.random.default_rng(0)
    df = pl.DataFrame(
        {
            "g": "g"
            + pl.Series(rng.integers(0, CHAIN_CATEGORIES, PROBE_ROWS)).cast(pl.String),
            "x": np.arange(PROBE_ROWS, dtype=np.int64),
        }
    )
    values = [f"g{i}" for i in range(CHAIN_VALUES)]
    chain_expr = predicates_to_expr(
        [SelectionPredicate(clauses=[ClauseFilter(column="g", values=values)])],
        df.schema,
    )
    assert "is_in" not in str(chain_expr)
    # The scan rule, checked without timing: the same clause on a scan source
    # keeps `is_in` whatever k is.
    scan_expr = predicates_to_expr(
        [SelectionPredicate(clauses=[ClauseFilter(column="g", values=values)])],
        df.schema,
        is_scan=True,
    )
    assert "is_in" in str(scan_expr)
    is_in_expr = pl.col("g").is_in(pl.Series(values, dtype=pl.String).implode())
    minmax = [pl.col("x").min().alias("lo"), pl.col("x").max().alias("hi")]

    # Warm-up: the first select also pays the allocator's first touch, which
    # is not what is measured here.
    df.lazy().filter(chain_expr).select(minmax).collect(engine="streaming")

    with PeakSampler(interval=0.001) as chain_sampler:
        chain = df.lazy().filter(chain_expr).select(minmax).collect(engine="streaming")
    # The other form on the same engine the product runs: the comparison is
    # chain against `is_in`, not streaming against in-memory.
    reference = df.lazy().filter(is_in_expr).select(minmax).collect(engine="streaming")

    assert chain.equals(reference)
    assert chain_sampler.peak_mb <= CHAIN_PEAK_MB, (
        f"the chain peaked {chain_sampler.peak_mb:.1f} MB over its baseline, "
        f"above the {CHAIN_PEAK_MB} MB cap"
    )

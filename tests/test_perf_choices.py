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

from flexviz import cube
from flexviz.cube import (
    CubeSpec,
    FreeAxisSpec,
    MeasureSpec,
    TargetDimSpec,
    build_cube,
    encode_fvcube,
)
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
# Observed ratios on this shape were 0.2 to 1.2, so 2.0 allows timing noise
# while still catching a large slowdown.
FILTERED_PROBE_RATIO = 2.0


@pytest.mark.skipif(
    pl.thread_pool_size() < MIN_PROBE_THREADS,
    reason=f"the engine comparison needs >= {MIN_PROBE_THREADS} threads",
)
def test_filtered_domain_probe_stays_in_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Decision: a filtered `LFQueryBuilder.physical_minmax` on a resident
    frame collects with the in-memory engine (the builder's `collect_engine`),
    not with the streaming engine the unfiltered probe uses.

    Evidence: on the real request path at 100M rows (M5, 2026-09-19) the
    in-memory engine is no worse than the streaming one on either frame shape,
    and 5x faster on a resident frame read from Parquet: 40 ms against 215 ms
    with 12.5M surviving rows. At this test's scale the gap is only 1.1x to 3x
    and unstable, so the timing check bounds overhead rather than proving the
    5x result.

    Check: the probe passes ``"in-memory"`` to the shared collector, takes at
    most FILTERED_PROBE_RATIO of the same select on the streaming engine, and
    returns the surviving rows' bounds, so it cannot pass on a no-op.
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
    collect_engines: list[str] = []
    original_collect = LFQueryBuilder._minmax_collect

    def tracked_collect(self, ldf, columns, schema, engine):
        collect_engines.append(engine)
        return original_collect(self, ldf, columns, schema, engine)

    monkeypatch.setattr(LFQueryBuilder, "_minmax_collect", tracked_collect)

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
    assert set(collect_engines) == {"in-memory"}
    median_probe = statistics.median(probe_ms)
    median_reference = statistics.median(reference_ms)
    assert median_probe <= FILTERED_PROBE_RATIO * median_reference, (
        f"the filtered probe took {median_probe:.2f} ms, over "
        f"{FILTERED_PROBE_RATIO} x the {median_reference:.2f} ms streaming select"
    )


# 8M rows in 100k-row chunks: 80 chunks, the shape a Parquet collect gives
# (one chunk per row group), at a size where the kernel runs tens of ms.
ENVELOPE_ROWS = 8_000_000
ENVELOPE_CHUNK = 100_000
# The per-chunk loop costs one extra iterator setup per chunk and nothing else,
# so the multi-chunk frame must stay close to the rechunked one. Measured 1.0x
# to 1.1x on an M5; 1.3 leaves headroom for a shared runner.
ENVELOPE_TIME_RATIO = 1.3


def test_envelope_kernel_scans_chunks_directly() -> None:
    """Decision: `envelope_scan` takes its dense fast path per chunk, on any
    chunk layout that is null-free and aligned, instead of only on a single
    contiguous chunk.

    Evidence: a Parquet collect has one chunk per row group, 10 at 10M rows and
    407 at 50M, so a real cube build never reached the old single-chunk fast
    path and ran the `Option` iterator instead. Lane C2 (2026-09-20) measured
    1.8x to 2.1x on kernel time from an explicit `.rechunk()`, which buys that
    speed by copying the whole projection (240 MB at 10M, 1.2 GB at 50M).

    Check: the multi-chunk frame gives the same envelope as the rechunked one
    and takes at most ENVELOPE_TIME_RATIO of its wall time, so the fast path is
    reached without the copy.
    """
    import flexviz_polars  # noqa: F401 — registers pl.Expr.flexviz namespace

    rng = np.random.default_rng(0)
    chunks = [
        pl.DataFrame(
            {
                "x": rng.uniform(0.0, 100.0, ENVELOPE_CHUNK),
                "y": rng.standard_normal(ENVELOPE_CHUNK),
                "f": rng.uniform(0.0, 10.0, ENVELOPE_CHUNK),
            }
        )
        for _ in range(ENVELOPE_ROWS // ENVELOPE_CHUNK)
    ]
    chunked = pl.concat(chunks, rechunk=False)
    assert chunked["x"].n_chunks() == ENVELOPE_ROWS // ENVELOPE_CHUNK
    single = chunked.rechunk()
    assert single["x"].n_chunks() == 1

    def envelope(df: pl.DataFrame) -> pl.DataFrame:
        return df.select(
            pl.col("x").flexviz.fixed_line_envelope2d(
                pl.col("y"),
                pl.col("f"),
                pl.lit(0.0),
                pl.lit(100.0),
                pl.lit(0.0),
                pl.lit(10.0),
                512,
                64,
            )
        )

    assert envelope(chunked).equals(envelope(single))

    # The kernel is serial, so this ratio does not depend on the thread pool,
    # hence no thread-count skip. Alternate the two frames, so a warming cache
    # or a thermal drift costs both.
    chunked_ms: list[float] = []
    single_ms: list[float] = []
    for run in range(6):  # one warm-up pass, then five timed ones
        for df, times in ((chunked, chunked_ms), (single, single_ms)):
            start = time.perf_counter()
            envelope(df)
            if run:
                times.append((time.perf_counter() - start) * 1e3)

    median_chunked = statistics.median(chunked_ms)
    median_single = statistics.median(single_ms)
    assert median_chunked <= ENVELOPE_TIME_RATIO * median_single, (
        f"the {len(chunks)}-chunk frame took {median_chunked:.1f} ms, over "
        f"{ENVELOPE_TIME_RATIO} x the {median_single:.1f} ms single-chunk frame"
    )


# ~1M cells over 50k distinct labels: a realistic large bar/treemap cube, and
# the shape where the per-row dict lookup cost is visible without the test
# taking seconds. The frame must come from `build_cube`: a hand-gathered
# string column is a scattered Utf8 view, on which Polars `unique`/`cast` cost
# 5x more than on the compact column a group_by emits.
ENCODER_ROWS = 1_000_000
ENCODER_LABELS = 50_000
# Lane Cd measured 3.0x (10M rows, K=1000: 256 → 84 ms), 3.2x (K=100k:
# 1683 → 523 ms) and 6.8x (K=1M: 8824 → 1294 ms). 2.5x leaves headroom.
ENCODER_SPEEDUP = 2.5


def _legacy_dim_dictionary(s: pl.Series) -> tuple[list, pl.Series]:
    """The pre-columnar encoder path, for reference: materialize the column as
    Python objects and look every row up in a dict."""
    categories = sorted(str(v) for v in s.unique().to_list())
    code_of = {v: i for i, v in enumerate(categories)}
    codes = np.fromiter(
        (code_of[str(v)] for v in s.to_list()), dtype="<u4", count=len(s)
    )
    return categories, pl.Series(codes)


def test_cube_encoder_codes_columnar_not_per_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Decision: `encode_fvcube` dictionary-encodes categorical dims and
    categorical free keys with Polars expressions (unique/sort, Enum cast or
    search_sorted, a join for the free keys), not with `to_list()` plus a
    per-row Python dict lookup.

    Evidence: lane Cd, 10M rows on an M5 — a range free axis with 1000 labels
    encoded in 84 ms against 256 ms, 100k labels in 523 ms against 1683 ms,
    1M labels in 1294 ms against 8824 ms, and a categorical free axis with
    100k labels in 590 ms against 3339 ms, with 1.25x to 2.8x less peak
    memory. Re-measured on this test's cube (995k cells, 50k labels): the dim
    step 24.5 ms against 103.4 ms (4.2x) and the whole encode 51.8 ms against
    133.5 ms (2.6x — the frame sort is shared); at 3.9M cells 184 ms against
    805 ms. The blobs are byte-identical on null-free data.

    Check: on a real `build_cube` frame the dim step is at least
    ENCODER_SPEEDUP faster than the per-row path, and the whole encode emits
    the same bytes either way. The timing covers the dim step alone because
    that is what the decision changed — the shared sort dilutes an end-to-end
    ratio without being part of it.
    """
    rng = np.random.default_rng(0)
    df = pl.DataFrame(
        {
            "active": rng.random(ENCODER_ROWS),
            "cat": pl.Series(
                "cat", [f"label-{i:06d}" for i in range(ENCODER_LABELS)]
            ).gather(rng.integers(0, ENCODER_LABELS, ENCODER_ROWS)),
        }
    )
    result = build_cube(
        df.lazy(),
        CubeSpec(
            source_name="s",
            free=FreeAxisSpec(column="active", p=2048, domain=(0.0, 1.0)),
            target_dims=(TargetDimSpec(column="cat", kind="categorical"),),
            measure=MeasureSpec(agg="count"),
        ),
    )

    def encode() -> bytes:
        return encode_fvcube(result, cube_id="perf")

    shipped_blob = encode()
    monkeypatch.setattr(cube, "_dim_dictionary", _legacy_dim_dictionary)
    assert encode() == shipped_blob
    monkeypatch.undo()

    # The encoder's own input: the frame after its total-order sort.
    col = result.frame.sort(["free_bin", "cat"])["cat"]
    shipped_ms = _median_ms(lambda: cube._dim_dictionary(col))
    legacy_ms = _median_ms(lambda: _legacy_dim_dictionary(col))

    assert shipped_ms * ENCODER_SPEEDUP <= legacy_ms, (
        f"the columnar dim dictionary took {shipped_ms:.1f} ms against "
        f"{legacy_ms:.1f} ms per-row, under {ENCODER_SPEEDUP}x"
    )

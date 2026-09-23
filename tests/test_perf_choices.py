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


def test_domain_probe_streams_on_resident_frames() -> None:
    """Decision: `LFQueryBuilder.physical_minmax` collects its min/max select
    with the streaming engine even on a resident frame, although every other
    resident collect is pinned to the in-memory engine.

    Evidence: the in-memory engine reads the column twice (one pass for min,
    one for max) while the streaming engine folds both into one morsel pass;
    measured on an M5 at 10M rows 1.87 vs 3.51 ms (0.53x), at 200M rows
    34.7 vs 63.4 ms, and 0.62x at 4 threads. At this test's ~3 ms scale a
    wall-time ratio is noise, not signal: on the 4-thread CI runner it once
    measured 2.81 ms against a 2.59 ms cutoff and failed for no behavioral
    reason.

    Check: the probe's collect names the streaming engine and its peak stays
    under PROBE_PEAK_MB; no wall assertion, because at this scale the gap is
    within runner noise.
    """
    rng = np.random.default_rng(0)
    df = pl.DataFrame({"v": rng.standard_normal(PROBE_ROWS)})

    engines: list[str | None] = []
    original_collect = pl.LazyFrame.collect

    def tracked_collect(self, *args, **kwargs):
        engines.append(kwargs.get("engine"))
        return original_collect(self, *args, **kwargs)

    # Warm-up: the first probe also pays the allocator's first touch (12 MB).
    LFQueryBuilder(df.lazy()).physical_minmax(["v"])

    pl.LazyFrame.collect = tracked_collect
    try:
        with PeakSampler(interval=0.001) as sampler:
            bounds = LFQueryBuilder(df.lazy()).physical_minmax(["v"])
    finally:
        pl.LazyFrame.collect = original_collect

    assert engines == ["streaming"]
    assert sampler.peak_mb <= PROBE_PEAK_MB, (
        f"the min/max select peaked {sampler.peak_mb:.1f} MB over its baseline, "
        f"above the {PROBE_PEAK_MB} MB cap"
    )
    assert bounds == {"v": (df["v"].min(), df["v"].max())}


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


def test_overlay_backgrounds_share_one_aggregate_call() -> None:
    """Decision: in overlay mode the engine runs the unfiltered background of
    every partition (each selection-owning figure is its own partition) in one
    `LFQueryBuilder.aggregate` call, instead of one call per partition.

    Evidence: a linked zoom over three figures, two of them owning a
    selection, on a resident 20M-row frame (M5, release plugin, machine under
    load, old and new engine interleaved in separate processes, median of 9):
    three histograms 75-80 ms -> 59 ms, three hist2d 65-79 -> 55-58 ms, three
    sorted lines 52-62 -> 49-51 ms. Specs that share the source's select read
    it once; a spec with its own plan (scan sources) still scans on its own,
    so scans gain nothing but lose nothing.

    Check: the background runs as one unfiltered call carrying every trace;
    no wall assertion, because a ~20% gap at ~50 ms is within runner noise.
    """
    from flexviz.engine import FlexEngine, TraceInfo
    from flexviz.events import InteractionEvent
    from flexviz.spec import ClauseFilter, SelectionPredicate, SelectionState
    from flexviz.trace.hist import Histogram

    rng = np.random.default_rng(0)
    df = pl.DataFrame({"ts": np.arange(100_000), "v": rng.standard_normal(100_000)})
    hists = {fig: Histogram(x="ts", bins=50) for fig in "abc"}
    engine = FlexEngine(LFQueryBuilder(df), {h.uid: h for h in hists.values()})
    infos = [
        TraceInfo(uid=h.uid, axes=("x", "y"), trace_type="histogram", figure_uid=f)
        for f, h in hists.items()
    ]

    def owned(fig: str) -> SelectionState:
        clause = ClauseFilter(column="v", range=(-0.5, 0.5))
        return SelectionState(
            source_figure_uid=fig,
            predicates=[SelectionPredicate(clauses=[clause])],
        )

    event = InteractionEvent(
        type="viewport",
        viewport_keys=["a/x", "b/x", "c/x"],
        selections=[owned("b"), owned("c")],
    )
    calls: list[tuple[int, int]] = []
    original = LFQueryBuilder.aggregate

    def tracked(self, filter_exprs, specs):
        calls.append((len(filter_exprs), len(specs)))
        return original(self, filter_exprs, specs)

    LFQueryBuilder.aggregate = tracked
    try:
        engine.process(event, infos, {}, cross_filter_mode="overlay")
    finally:
        LFQueryBuilder.aggregate = original

    assert [c for c in calls if c[0] == 0] == [(0, 3)]

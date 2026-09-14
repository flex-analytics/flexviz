"""Query-layer benchmarks: the primitives every event path is built from.

``LFQueryBuilder`` resolves domains and runs the batched aggregation;
``predicates_to_expr`` compiles a brush into the Polars filter that precedes
it. Both run once per request, so their fixed cost matters as much as the scan
itself.
"""

from __future__ import annotations

import polars as pl

from flexviz.LF import AggregationSpec, LFQueryBuilder
from flexviz.predicates import canonical_passive_key, predicates_to_expr
from flexviz.spec import ClauseFilter, SelectionPredicate, SelectionState

_SELECTIONS = [
    SelectionState(
        source_figure_uid="fig_a",
        predicates=[
            SelectionPredicate(
                clauses=[
                    ClauseFilter(column="ts", range=(40_000, 120_000)),
                    ClauseFilter(column="cat", values=[f"cat_{i}" for i in range(6)]),
                ]
            ),
            SelectionPredicate(clauses=[ClauseFilter(column="val", range=(-0.5, 0.5))]),
        ],
    ),
    SelectionState(
        source_figure_uid="fig_b",
        predicates=[
            SelectionPredicate(
                clauses=[ClauseFilter(column="region", values=["north", "east"])]
            )
        ],
    ),
]


def test_physical_minmax(benchmark, numeric_df: pl.DataFrame) -> None:
    """Domain resolution: a streaming min/max pass over three columns."""
    lf = LFQueryBuilder(numeric_df)
    # A non-static builder re-resolves on every call, so the memo never hides
    # the work being measured.
    bounds = benchmark(lf.physical_minmax, ["ts", "val", "val2"])
    assert len(bounds) == 3


def test_aggregate_batched(benchmark, numeric_df: pl.DataFrame) -> None:
    """Several aggregations fused into one filtered ``select``."""
    lf = LFQueryBuilder(numeric_df)
    specs = [
        AggregationSpec(uid="a", expr=pl.col("val").mean().alias("a")),
        AggregationSpec(uid="b", expr=pl.col("val2").std().alias("b")),
        AggregationSpec(uid="c", expr=pl.col("ts").max().alias("c")),
    ]
    filters = [pl.col("ts").is_between(20_000, 180_000)]

    def run():
        return lf.aggregate(filters, specs)

    regular_df, _ = benchmark(run)
    assert regular_df.width == 3


def test_predicates_to_expr(benchmark, numeric_df: pl.DataFrame) -> None:
    """Brush compilation: predicates to a Polars filter expression."""
    schema = LFQueryBuilder(numeric_df).schema
    predicates = _SELECTIONS[0].predicates

    def run():
        return predicates_to_expr(predicates, schema)

    expr = benchmark(run)
    assert expr is not None


def test_canonical_passive_key(benchmark) -> None:
    """Cube cache keying over the passive selection set."""
    key = benchmark(canonical_passive_key, _SELECTIONS, "fig_a")
    assert key is not None

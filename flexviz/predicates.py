"""Predicate → Polars expression compiler.

This is the single primitive used by the engine to translate a list of
``SelectionPredicate`` objects into a Polars filter expression that is
applied to the shared LazyFrame before any aggregation runs.

`SelectionPredicate.clauses` are ANDed together; multiple predicates in
the same `SelectionState` are ORed.  Predicates from different
``SelectionState`` objects (i.e. from different source figures) are ANDed
by the caller, not here — the caller passes one selection's predicates
at a time.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

import polars as pl

from .spec import ClauseFilter, SelectionPredicate, SelectionState
from .trace.base import _dtype_for_col, _typed_range_bounds


def _values_to_typed_series(
    column: str, values: Iterable[Any], schema: pl.Schema | None
) -> pl.Series:
    """Coerce a python list of values to the column's dtype where possible."""
    dtype = _dtype_for_col(schema, column)
    coerced = list(values)
    if dtype == pl.Boolean:
        _bool_map = {"true": True, "false": False, None: None}
        try:
            coerced = [_bool_map[v] for v in coerced]
        except KeyError as e:
            raise ValueError(
                f"Boolean predicate value must be 'true' or 'false', got {e.args[0]!r}"
            ) from e
    series = pl.Series(coerced)
    if dtype is not None:
        series = series.cast(dtype, strict=False)
    return series


# Up to this many values an OR of equality tests beats `is_in` on a resident
# frame. Measured on 100M rows, a String column, both forms on the streaming
# engine: k=1/5/10/14/20 the chain runs 18/54/102/132/172 ms against
# 280/581/947/961/634 ms for `is_in`, 7 to 15x. The chain costs about 9 ms per
# extra value while `is_in` stays flat at 500 to 1000 ms, so the two cross
# around k 80 to 100 (200 categories: k=60 545 against 798 ms, k=80 746
# against 784, k=100 950 against 922). 64 keeps a margin under that crossover.
_EQUALITY_CHAIN_MAX_VALUES = 64


def _clause_to_expr(
    clause: ClauseFilter, schema: pl.Schema | None, *, is_scan: bool = False
) -> pl.Expr:
    if clause.values is not None:
        series = _values_to_typed_series(clause.column, clause.values, schema)
        # A null never matches, so it is dropped. An empty remainder still
        # goes through `is_in`, which selects nothing but keeps naming the
        # column, so a missing column still raises at collect time.
        # `any_horizontal` needs at least one test, hence the lower bound.
        typed = series.drop_nulls()
        # The source kind decides the form, not k. A scan pushes either form
        # into the reader, where `is_in` costs one pass over the column and
        # the chain costs k passes. A resident frame is the other way round:
        # `is_in` on a String column is 5 to 15x slower than the chain up to
        # the cutoff.
        if not is_scan and 0 < len(typed) <= _EQUALITY_CHAIN_MAX_VALUES:
            # Literals carry the cast series' dtype, so the comparison never
            # widens the column (a Float32 column against a Float64 literal
            # would copy it).
            return pl.any_horizontal(
                *[pl.col(clause.column) == pl.lit(v, dtype=typed.dtype) for v in typed]
            )
        return pl.col(clause.column).is_in(typed.implode())

    bounds = _typed_range_bounds(clause.column, clause.range, schema, clause.closed)
    if bounds is None:
        # range was None — should never happen because the model_validator
        # rejects it, but treat as a no-op for safety.
        return pl.lit(True)
    lo, hi = bounds
    return pl.col(clause.column).is_between(lo, hi, closed=clause.closed)


def predicate_to_expr(
    predicate: SelectionPredicate, schema: pl.Schema | None, *, is_scan: bool = False
) -> pl.Expr:
    """Convert one predicate (AND of clauses) to a Polars expression."""
    if not predicate.clauses:
        return pl.lit(True)
    exprs = [_clause_to_expr(c, schema, is_scan=is_scan) for c in predicate.clauses]
    return pl.all_horizontal(*exprs)


def _canonical_clause(clause: ClauseFilter) -> dict:
    """Pinned canonical clause shape (contract E): key insertion order is the
    wire format — ``{"c", "r", "cl"}`` for ranges, ``{"c", "v"}`` for value
    sets (values sorted by their string form, the pinned comparator)."""
    if clause.values is not None:
        return {"c": clause.column, "v": sorted(clause.values, key=str)}
    lo, hi = clause.range  # type: ignore[misc]  # model_validator guarantees
    return {"c": clause.column, "r": [lo, hi], "cl": clause.closed}


def _canonical_predicate(predicate: SelectionPredicate) -> str:
    """Compact JSON of the clause list, clauses sorted by (column, kind)."""
    clauses = sorted(
        predicate.clauses,
        key=lambda c: (c.column, "v" if c.values is not None else "r"),
    )
    return json.dumps(
        [_canonical_clause(c) for c in clauses], separators=(",", ":"), default=str
    )


def canonical_passive_key(
    selections: list[SelectionState], active_figure_uid: str | None
) -> str | None:
    """The canonical passive key for a cube gesture (contract E).

    The passive set is every ``SelectionState`` with a non-``None``
    ``source_figure_uid`` different from the active figure and non-empty
    predicates (``None``-uid selections never filter in the legacy engine;
    the active figure's own selection is the re-brush case). Returns ``None``
    for an empty passive set so zero-passive cube keys stay byte-identical
    to Phases 1–2.

    Canonical form preserves nesting (AND across selections of
    OR-within-selection — flattening would change semantics): each selection
    becomes the compact JSON array of its predicates' canonical strings
    (sorted), and the key is the compact JSON array of those selection
    strings (sorted). The JS mirror is ``fvCubePassiveKey`` — the two never
    need to match each other (asymmetric keying, spec §3); each only needs
    to be deterministic in its own language.
    """
    passive = [
        s
        for s in selections
        if s.source_figure_uid is not None
        and s.source_figure_uid != active_figure_uid
        and s.predicates
    ]
    if not passive:
        return None
    selection_strs = sorted(
        json.dumps(
            sorted(_canonical_predicate(p) for p in sel.predicates),
            separators=(",", ":"),
        )
        for sel in passive
    )
    return json.dumps(selection_strs, separators=(",", ":"))


def predicates_to_expr(
    predicates: list[SelectionPredicate],
    schema: pl.Schema | None,
    *,
    is_scan: bool = False,
) -> pl.Expr:
    """Convert a list of predicates (OR of ANDs) to one Polars expression.

    An empty list produces ``pl.lit(True)`` so the caller can pass it
    through ``filter()`` unconditionally without branching.

    ``is_scan`` is the source's ``LFQueryBuilder.is_scan``: it selects the
    compiled form of a values clause, the same residency signal that already
    picks kernel against plan formulations. The default suits a resident
    frame.
    """
    if not predicates:
        return pl.lit(True)
    disjuncts = [predicate_to_expr(p, schema, is_scan=is_scan) for p in predicates]
    return pl.any_horizontal(*disjuncts)

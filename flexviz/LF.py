from __future__ import annotations

import polars as pl

from typing import Any, Callable, List, Set, Tuple
from dataclasses import dataclass

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import pyarrow as pa
except ImportError:
    pa = None


def polars_lf_from(data) -> pl.LazyFrame:
    if isinstance(data, (pl.DataFrame, pl.LazyFrame)):
        return data.lazy()
    elif pd is not None and isinstance(data, pd.DataFrame):
        return pl.from_pandas(data).lazy()
    elif pa is not None and isinstance(data, pa.Table):
        return pl.from_arrow(data).lazy()
    # elif hasattr("__dataframe__", data): ??? # TODO?
    #     return pl.from_dataframe(data)
    raise ValueError(f"Unsupported data type: {type(data)}")


def polars_col_from(col: str | pl.Expr) -> pl.Expr:
    if isinstance(col, str):
        col = pl.col(col)
    return col


def get_col_name(col: str | pl.Expr) -> str:
    if isinstance(col, str):
        return col
    assert col.meta.is_column()
    return col.meta.output_name()


@dataclass(frozen=True)
class AggregationSpec:
    """Specification for one aggregation output.

    Executed as ``filtered_ldf.select(expr)`` (batched with other specs),
    unless ``plan`` is set — see below.
    """

    expr: pl.Expr
    uid: str = ""
    #: Optional escape hatch for an aggregation that cannot be a select
    #: expression. Called as ``plan(filtered_ldf)`` and must return a one-row
    #: DataFrame whose single column is named ``uid``, i.e. exactly the column
    #: the batched ``select`` would have produced. Set only when a spec needs
    #: its own plan — the out-of-core line envelope uses a streaming group_by
    #: that cannot ride the shared select.
    plan: "Callable[[pl.LazyFrame], pl.DataFrame] | None" = None


@dataclass(frozen=True)
class GroupedAggregationSpec:
    """Specification for one grouped aggregation query.

    ``agg_exprs`` must already be aliased to their logical parent trace uids.
    ``pre_group_filters`` are applied before ``group_by`` so group membership is
    derived from the visible/filtered rows, not the full source frame.
    ``batch_key`` allows callers to prevent unsafe fusion when grouped traces
    have different pre-group semantics (for example, different viewport ranges).
    """

    uid: str
    group_cols: Tuple[str, ...]
    sort_cols: Tuple[str, ...]
    agg_exprs: Tuple[pl.Expr, ...]
    pre_group_filters: Tuple[pl.Expr, ...] = ()
    pre_group_filter_key: Any = None
    batch_key: Tuple[Any, ...] = ()


class LFQueryBuilder:
    """
    Query builder for LazyFrame.

    You can pass both a Polars DataFrame or LazyFrame to initialize the LFQueryBuilder.
    If a DataFrame is passed, it will be converted to a LazyFrame.
    """

    def __init__(
        self,
        ldf: pl.DataFrame | pl.LazyFrame,
        row_index_col: str | pl.Expr | None = None,
        cache_schema: bool = True,
    ):
        ldf = polars_lf_from(ldf)
        assert isinstance(ldf, pl.LazyFrame)
        if row_index_col is not None:
            row_index_col = polars_col_from(row_index_col)
            explicit_row_index_col = True
        else:
            explicit_row_index_col = False

        # Store the lazy frame and the row index column
        self._ldf: pl.LazyFrame = ldf
        self._row_index_col: pl.Expr | None = row_index_col
        self._explicit_row_index_col: bool = explicit_row_index_col
        self._cache_schema: bool = cache_schema  # whether to cache the ldf its schema
        self._sorted_cols: Set[str] = set()  # columns that are sorted
        self._minmax_memo: dict[str, Tuple[Any, Any]] = {}

    @property
    def is_scan(self) -> bool:
        """Whether this source reads from storage rather than a resident frame.

        The residency signal behind both the kernel-vs-native trace choice and
        ``collect_engine``. A resident frame's unoptimized plan roots at
        ``DF [...]``; a file source roots at ``<Format> SCAN [...]``. Computed
        once — ``explain`` walks the plan, and this is asked per request.

        Not a correctness switch: both paths must produce identical output, and
        there is a test that asserts it. It only selects which formulation runs.
        """
        if not hasattr(self, "_cached_is_scan"):
            try:
                self._cached_is_scan = "SCAN [" in self._ldf.explain(optimized=False)
            except Exception:
                # An un-explainable plan is treated as resident: that is the
                # path that works for every source, just not bounded.
                self._cached_is_scan = False
        return self._cached_is_scan

    @property
    def collect_engine(self) -> str:
        """The Polars engine every collect on this source uses.

        Fixed per source kind rather than left to ``"auto"``: a file scan must
        stream, a resident frame must not pay the streaming machinery.
        """
        return "streaming" if self.is_scan else "in-memory"

    @property
    def row_index_col(self) -> str | None:
        if self._row_index_col is None:
            return None
        return self._row_index_col.meta.output_name()

    @property
    def explicit_row_index_col(self) -> str | None:
        return self.row_index_col if self._explicit_row_index_col else None

    # Is ~ 40x faster than LazyFrame.collect_schema() when the LazyFrame is in memory
    @property
    def schema(self):
        if self._cache_schema and hasattr(self, "_cached_schema"):
            return self._cached_schema

        schema = self._ldf.collect_schema()

        if self._cache_schema:
            self._cached_schema = schema

        return schema

    def physical_minmax(
        self,
        columns: List[str],
        schema: pl.Schema | None = None,
        *,
        memoize: bool,
    ) -> dict[str, Tuple[Any, Any]]:
        """``(min, max)`` of each column in its physical representation.

        Temporal columns reduce on ``to_physical()``; every other column on its
        raw value. Nothing is cast to Float64, so large integer bounds stay
        exact. An empty or all-null column yields ``(None, None)``.

        ``memoize`` keeps the result for the builder's lifetime. Only a
        ``cache=True`` source may set it — the same static-data contract that
        governs schema caching — because otherwise a reset must be able to see
        changed source data. Re-registering a source replaces the builder and
        drops the memo.
        """
        memo = self._minmax_memo if memoize else {}
        sch = schema if schema is not None else self.schema
        # De-dupe: the same column can be requested in several roles at once
        # (e.g. the free axis is also a binned target dim), and a column may
        # already be memoized. ``dict.fromkeys`` preserves first-seen order.
        missing = list(dict.fromkeys(c for c in columns if c not in memo))
        if missing:
            exprs: List[pl.Expr] = []
            for c in missing:
                val = pl.col(c)
                dtype = sch.get(c) if hasattr(sch, "get") else None
                if dtype is not None and dtype.is_temporal():
                    val = val.to_physical()
                exprs.append(val.min().alias(f"__min_{c}__"))
                exprs.append(val.max().alias(f"__max_{c}__"))
            stats = self._ldf.select(exprs).collect(engine=self.collect_engine)
            for c in missing:
                memo[c] = (stats[f"__min_{c}__"].item(), stats[f"__max_{c}__"].item())
        return {c: memo[c] for c in columns}

    # --------------- Handling flags ---------------

    def check_sorted(self, col: str | pl.Expr):
        """
        Check if a column is sorted and set it as sorted, if not already set.

        This uses under the hood ._sorted_cols to keep track of the columns that are
        sorted. This is useful as one cannot query the flags of a LazyFrame.

        Parameters
        ----------
        col : str | pl.Expr
            Column name or expression.

        Raises
        ------
        AssertionError
            If the column is not in the schema or if the column is not sorted.
        """
        col: str = get_col_name(col)
        assert col in self.schema, f"Column '{col}' not in schema"
        if col not in self._sorted_cols:
            # TODO: how expensive is this?
            s: pl.Series = (
                self._ldf.select(col).collect(engine=self.collect_engine).to_series()
            )
            assert s.is_sorted(), f"Column '{col}' not sorted"
            self._ldf = self._ldf.set_sorted(col)
            self._sorted_cols.add(col)

    def is_sorted(self, col: str | pl.Expr) -> bool:
        """Whether ``col`` was asserted sorted via ``assume_sorted``/``check_sorted``.

        A guarantee, not a check — this never collects. Consumers use it only to
        pick a faster equivalent formulation, never to change results.
        """
        return get_col_name(col) in self._sorted_cols

    def assume_sorted(self, col: str | pl.Expr) -> None:
        """Mark a column as sorted without verifying (no collect).

        This sets the sorted flag on the underlying LazyFrame, enabling optimizations
        that rely on sortedness. Use only when you *guarantee* the column is sorted
        ascending; otherwise results may be incorrect.
        """
        col_name: str = get_col_name(col)
        assert col_name in self.schema, f"Column '{col_name}' not in schema"
        if col_name in self._sorted_cols:
            return
        self._ldf = self._ldf.set_sorted(col_name)
        self._sorted_cols.add(col_name)

    # --------------- Aggregation ---------------

    def aggregate(
        self,
        filter_exprs: List[pl.Expr],
        agg_specs: "List[AggregationSpec | GroupedAggregationSpec]",
    ) -> "Tuple[pl.DataFrame, dict[str, pl.DataFrame]]":
        """Aggregate the data using the provided specifications.

        Parameters
        ----------
        filter_exprs : List[pl.Expr]
            Filter expressions from cross-filter selections.
        agg_specs : List[AggregationSpec | GroupedAggregationSpec]
            Mixed list of regular and grouped aggregation specs.

        Returns
        -------
        tuple[pl.DataFrame, dict[str, pl.DataFrame]]
            ``(regular_df, grouped_dfs)`` where ``regular_df`` is the result
            of a batched ``select()`` for all ``AggregationSpec``s, and
            ``grouped_dfs`` maps each grouped parent uid to its fused
            ``group_by().agg().sort()`` result DataFrame.
        """
        filtered_ldf = (
            self._ldf if not filter_exprs else self._ldf.filter(*filter_exprs)
        )

        regular_specs = [s for s in agg_specs if isinstance(s, AggregationSpec)]
        grouped_specs = [s for s in agg_specs if isinstance(s, GroupedAggregationSpec)]

        # Specs carrying their own plan cannot join the shared select; run each
        # and hstack it back so callers see one flat regular_df either way.
        expr_specs = [s for s in regular_specs if s.plan is None]
        plan_specs = [s for s in regular_specs if s.plan is not None]

        if expr_specs:
            regular_df = filtered_ldf.select(*[s.expr for s in expr_specs]).collect(
                engine=self.collect_engine
            )
        else:
            regular_df = pl.DataFrame()
        for spec in plan_specs:
            planned = spec.plan(filtered_ldf)
            if planned.width != 1 or planned.columns[0] != spec.uid:
                raise ValueError(
                    f"AggregationSpec.plan for {spec.uid!r} must return exactly "
                    f"one column named {spec.uid!r}, got {planned.columns!r}"
                )
            regular_df = (
                planned if regular_df.is_empty() else regular_df.hstack(planned)
            )

        grouped_dfs: dict[str, pl.DataFrame] = {}
        grouped_batches: dict[tuple, list[GroupedAggregationSpec]] = {}
        for spec in grouped_specs:
            batch_id = (spec.group_cols, spec.sort_cols, spec.batch_key)
            grouped_batches.setdefault(batch_id, []).append(spec)

        for batch_specs in grouped_batches.values():
            first = batch_specs[0]
            if first.pre_group_filters and first.pre_group_filter_key is None:
                raise ValueError(
                    "GroupedAggregationSpec with pre_group_filters must provide "
                    "pre_group_filter_key so grouped batch fusion can validate "
                    "semantic filter equality."
                )
            for spec in batch_specs[1:]:
                if spec.pre_group_filters and spec.pre_group_filter_key is None:
                    raise ValueError(
                        "GroupedAggregationSpec with pre_group_filters must provide "
                        "pre_group_filter_key so grouped batch fusion can validate "
                        "semantic filter equality."
                    )
                if spec.pre_group_filter_key != first.pre_group_filter_key:
                    raise ValueError(
                        "Unsafe grouped batch fusion: specs with batch key "
                        f"{(first.group_cols, first.sort_cols, first.batch_key)!r} "
                        "have different pre_group_filter_key values. Extend "
                        "batch_key or normalize the grouped filter semantics."
                    )
            batch_ldf = (
                filtered_ldf
                if not first.pre_group_filters
                else filtered_ldf.filter(*first.pre_group_filters)
            )
            agg_exprs: list[pl.Expr] = []
            for spec in batch_specs:
                agg_exprs.extend(spec.agg_exprs)

            batch_df = (
                batch_ldf.group_by(list(first.group_cols))
                .agg(*agg_exprs)
                .sort(list(first.sort_cols))
                .collect(engine=self.collect_engine)
            )
            for spec in batch_specs:
                grouped_dfs[spec.uid] = batch_df

        return regular_df, grouped_dfs

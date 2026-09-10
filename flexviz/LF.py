from __future__ import annotations

import os
import re
import struct
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import polars as pl

try:
    import pandas as pd
except ImportError:
    pd = None

import pyarrow as pa


def polars_lf_from(data) -> pl.LazyFrame:
    if isinstance(data, (pl.DataFrame, pl.LazyFrame)):
        return data.lazy()
    elif pd is not None and isinstance(data, pd.DataFrame):
        return pl.from_pandas(data).lazy()
    elif isinstance(data, pa.Table):
        return pl.from_arrow(data).lazy()
    # elif hasattr("__dataframe__", data): ??? # TODO?
    #     return pl.from_dataframe(data)
    raise ValueError(f"Unsupported data type: {type(data)}")


def get_col_name(col: str | pl.Expr) -> str:
    if isinstance(col, str):
        return col
    assert col.meta.is_column()
    return col.meta.output_name()


def _parquet_footer_minmax(
    path: str, columns: list[str], sch: pl.Schema
) -> dict[str, tuple[Any, Any]]:
    """``(min, max)`` per column from the Parquet footer, in physical form.

    Returns only the columns the footer can answer. The footer holds per
    row-group statistics, so this reads a few kilobytes instead of decoding the
    column. Values match what ``pl.col(c).min()`` (or ``.to_physical().min()``
    for a temporal column) would collect.
    """
    import pyarrow.parquet as pq

    meta = pq.read_metadata(path)
    arrow_schema = meta.schema.to_arrow_schema()
    # A nested column contributes several leaves, so an arrow field index stops
    # being a leaf index and the statistics would be read off the wrong column.
    if len(arrow_schema) != meta.num_columns:
        return {}

    out: dict[str, tuple[Any, Any]] = {}
    for c in columns:
        dtype = sch.get(c)
        if dtype is None:
            continue
        temporal = dtype.is_temporal()
        # Only these reduce to a scalar that is comparable across row groups and
        # convertible back to the physical value the collect path returns.
        if not (temporal or dtype.is_integer() or dtype.is_float()):
            continue
        half = dtype == pl.Float16
        idx = arrow_schema.get_field_index(c)
        if idx < 0:
            continue

        lo = hi = None
        for rg in range(meta.num_row_groups):
            stats = meta.row_group(rg).column(idx).statistics
            if stats is None or not stats.has_min_max:
                lo = None  # no statistics for this column, leave it to the collect
                break
            # Unsigned ints need the decoded value: the raw one is the signed
            # reading of the same bytes. Temporal columns need the raw one: the
            # decoded one is a Python datetime and cannot hold nanoseconds.
            mn, mx = (
                (stats.min_raw, stats.max_raw) if temporal else (stats.min, stats.max)
            )
            if half:
                # A Float16 statistic comes back as its raw 2-byte half, and
                # comparing raw bytes orders negative values wrong, so it is
                # decoded before the fold.
                mn, mx = struct.unpack("<e", mn)[0], struct.unpack("<e", mx)[0]
            lo = mn if lo is None else min(lo, mn)
            hi = mx if hi is None else max(hi, mx)
        if lo is None:
            continue

        if temporal:
            # Read the raw ints back through the file's own arrow type, so a unit
            # the Polars dtype does not share (a millisecond time, say) converts.
            raw = pa.array([lo, hi], type=arrow_schema.field(idx).type)
            lo, hi = pl.from_arrow(raw).cast(dtype).to_physical()
        out[c] = (lo, hi)
    return out


@dataclass(frozen=True)
class AggregationSpec:
    """Specification for one aggregation output.

    Executed as ``filtered_ldf.select(expr)`` (batched with other specs),
    unless ``plan`` is set — see below. ``expr`` is None only when ``plan``
    carries the aggregation instead.
    """

    expr: pl.Expr | None = None
    uid: str = ""
    #: Optional escape hatch for an aggregation that cannot be a select
    #: expression. Called as ``plan(filtered_ldf)`` and must return a one-row
    #: DataFrame whose single column is aliased to ``uid``, i.e. exactly the
    #: column the batched ``select`` would have produced. Set only when a spec
    #: needs a streaming plan or a batch fold that cannot ride the shared
    #: select.
    plan: Callable[[pl.LazyFrame], pl.DataFrame] | None = None

    def __post_init__(self) -> None:
        if self.expr is None and self.plan is None:
            raise ValueError("AggregationSpec needs either an expr or a plan")


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
    group_cols: tuple[str, ...]
    sort_cols: tuple[str, ...]
    agg_exprs: tuple[pl.Expr, ...]
    pre_group_filters: tuple[pl.Expr, ...] = ()
    pre_group_filter_key: Any = None
    batch_key: tuple[Any, ...] = ()
    #: Optional escape hatch for a grouped aggregation that is a whole plan
    #: instead of an expression list. Called as ``plan(batch_ldf)``, where
    #: ``batch_ldf`` already carries the cross-filter and this spec's
    #: ``pre_group_filters``. It must return a frame with ``group_cols`` plus
    #: one column named ``uid``, the shape the fused grouped query returns.
    #: ``agg_exprs`` is ignored when ``plan`` is set, and a plan spec never
    #: fuses with other grouped specs.
    plan: Callable[[pl.LazyFrame], pl.DataFrame] | None = None


class LFQueryBuilder:
    """
    Query builder for LazyFrame.

    You can pass both a Polars DataFrame or LazyFrame to initialize the LFQueryBuilder.
    If a DataFrame is passed, it will be converted to a LazyFrame.
    """

    def __init__(
        self,
        ldf: pl.DataFrame | pl.LazyFrame,
        cache: bool = False,
    ):
        ldf = polars_lf_from(ldf)
        assert isinstance(ldf, pl.LazyFrame)
        self._ldf: pl.LazyFrame = ldf
        self.cache: bool = cache  # the registrar's static-data assertion
        self._sorted_cols: set[str] = set()  # columns that are sorted
        self._minmax_memo: dict[str, tuple[Any, Any]] = {}

    @property
    def static(self) -> bool:
        """Whether the data cannot change under this builder.

        A resident frame is a snapshot, and a ``cache=True`` scan is declared
        static by the cache contract. Everything the builder keeps across
        requests (resolved bounds, the sorted flag) rests on this.
        """
        return self.cache or not self.is_scan

    @cached_property
    def is_scan(self) -> bool:
        """Whether this source reads from storage rather than a resident frame.

        The residency signal behind both the kernel-vs-native trace choice and
        ``collect_engine``. A resident frame's unoptimized plan roots at
        ``DF [...]``; a file source roots at ``<Format> SCAN [...]``. Computed
        once — ``explain`` walks the plan, and this is asked per request.

        Not a correctness switch: both paths must produce identical output, and
        there is a test that asserts it. It only selects which formulation runs.
        """
        try:
            return "SCAN [" in self._ldf.explain(optimized=False)
        except Exception:
            # An un-explainable plan is treated as resident: that is the path
            # that works for every source, just not bounded.
            return False

    @cached_property
    def _parquet_path(self) -> str | None:
        """The file behind a bare single-file local Parquet scan, else ``None``.

        Only that shape lets the footer speak for the whole source. A multi-file
        or hive scan, a cloud URL, or any node above the scan changes which rows
        the statistics describe. Computed once, like ``is_scan``.
        """
        try:
            lines = [
                ln.strip() for ln in self._ldf.explain(optimized=False).split("\n")
            ]
        except Exception:
            return None
        # A sorted hint (``set_sorted``, so ``assume_sorted`` too) keeps every
        # row the footer describes, so it is the only node the probe looks
        # through. Each hint indents the plan below it one more level.
        while lines and lines[0].startswith("hint.sorted("):
            lines.pop(0)
        scan = re.fullmatch(r"Parquet SCAN \[(.+)\]", lines[0]) if lines else None
        if scan is None:
            return None
        path = scan[1]
        # A comma means several files; ``isfile`` rules out cloud URLs. A slice
        # (``scan_parquet(n_rows=...)``) sits inside the scan node and reads
        # fewer rows than the footer describes.
        if (
            "," in path
            or any(ln.startswith("SLICE") for ln in lines[1:])
            or not os.path.isfile(path)
        ):
            return None
        # Polars prints forward slashes on every OS; normalize so the path
        # compares equal to what the caller passed in.
        return os.path.normpath(path)

    @property
    def collect_engine(self) -> str:
        """The Polars engine the builder's own collects use.

        Fixed per source kind rather than left to ``"auto"``: a file scan must
        stream, a resident frame must not pay the streaming machinery. The line
        bucket plan, the grouped histogram plan and the domain probe are the
        exceptions: all stream on both source kinds.
        """
        return "streaming" if self.is_scan else "in-memory"

    # Is ~ 40x faster than LazyFrame.collect_schema() when the LazyFrame is in memory
    @cached_property
    def schema(self):
        return self._ldf.collect_schema()

    def physical_minmax(
        self,
        columns: list[str],
        schema: pl.Schema | None = None,
    ) -> dict[str, tuple[Any, Any]]:
        """``(min, max)`` of each column in its physical representation.

        Temporal columns reduce on ``to_physical()``; every other column on its
        raw value. Nothing is cast to Float64, so large integer bounds stay
        exact. An empty or all-null column yields ``(None, None)``.

        On a bare single-file Parquet scan the Parquet footer answers what it
        can, which reads a few kilobytes instead of decoding the column.
        Whatever the footer cannot answer is collected as before. The footer is
        an optimization only: any problem with it falls back to the collect.

        A ``static`` source keeps the result for the builder's lifetime. An
        uncached scan may change on disk between requests, so it resolves again
        and an uncached reset sees the changed data. Re-registering a source
        with raw data or a new builder replaces the builder and drops the memo.
        Re-registering the same builder object keeps it, and the server warns.
        """
        memo = self._minmax_memo if self.static else {}
        sch = schema if schema is not None else self.schema
        # De-dupe: the same column can be requested in several roles at once
        # (e.g. the free axis is also a binned target dim), and a column may
        # already be memoized. ``dict.fromkeys`` preserves first-seen order.
        missing = list(dict.fromkeys(c for c in columns if c not in memo))
        path = self._parquet_path if missing else None
        if path is not None:
            try:
                found = _parquet_footer_minmax(path, missing, sch)
            except Exception:
                found = {}
            memo.update(found)
            missing = [c for c in missing if c not in found]
        if missing:
            exprs: list[pl.Expr] = []
            for c in missing:
                val = pl.col(c)
                dtype = sch.get(c) if hasattr(sch, "get") else None
                if dtype is not None and dtype.is_temporal():
                    val = val.to_physical()
                exprs.append(val.min().alias(f"__min_{c}__"))
                exprs.append(val.max().alias(f"__max_{c}__"))
            # Always streaming: the min/max select is ~2x faster on the
            # streaming engine than on the in-memory one, on both source kinds.
            stats = self._ldf.select(exprs).collect(engine="streaming")
            for c in missing:
                memo[c] = (stats[f"__min_{c}__"].item(), stats[f"__max_{c}__"].item())
        return {c: memo[c] for c in columns}

    # --------------- Handling flags ---------------

    def check_line_x(self, col: str | pl.Expr) -> None:
        """Verify the data of an x column against the resident-frame line contract.

        The x-width kernel needs x sorted ascending and free of nulls and NaN.
        Verifying that costs one collect: an O(1) null check, one pass over the
        order, and on a float dtype an O(1) read of the last element. NaN sorts
        last, so on a sorted null-free column every NaN is a suffix.

        The dtype half of the contract lives on the trace
        (``LinePlot.check_schema``), and ``LinePlot.check_source`` calls this
        only where the data matters: an ungrouped x-width line on a resident
        frame. A grouped plan and a scan plan read x in no order.

        A ``static`` source flags a passing column sorted in ``._sorted_cols``,
        which skips the collect on later requests, and sets the Polars sorted
        flag, which turns the column's min/max into an O(1) read. The flags of a
        LazyFrame cannot be queried, hence the set.

        Raises
        ------
        ValueError
            If the column breaks the contract.
        """
        col_name: str = get_col_name(col)
        if col_name in self._sorted_cols:
            return
        is_float = self.schema[col_name].is_float()
        x = pl.col(col_name)
        stats = self._ldf.select(
            x.has_nulls().alias("__has_nulls"),
            x.is_sorted().alias("__sorted"),
            *([x.last().is_nan().alias("__last_nan")] if is_float else []),
        ).collect(engine=self.collect_engine)
        if stats["__has_nulls"].item():
            raise ValueError(
                f"x column '{col_name}' has null values. A minmax line needs "
                f"a null-free x. Drop the null rows first."
            )
        if not stats["__sorted"].item():
            # A NaN in the middle also lands here: it sorts last, so it breaks
            # the order too.
            raise ValueError(
                f"Column '{col_name}' is not sorted ascending. Sort the frame by "
                f"'{col_name}', or pass assume_sorted_x=True if you guarantee it."
            )
        if is_float and stats["__last_nan"].item():
            raise ValueError(
                f"x column '{col_name}' has NaN values. A minmax line needs "
                f"a NaN-free x. Drop the NaN rows first."
            )
        if self.static:
            self._ldf = self._ldf.set_sorted(col_name)
            self._sorted_cols.add(col_name)

    @property
    def sorted_cols(self) -> frozenset[str]:
        """The columns asserted sorted via ``assume_sorted`` / ``check_line_x``.

        A guarantee, not a check — this never collects. Consumers use it only to
        pick a faster equivalent formulation, never to change results.
        """
        return frozenset(self._sorted_cols)

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
        filter_exprs: list[pl.Expr],
        agg_specs: list[AggregationSpec | GroupedAggregationSpec],
    ) -> tuple[pl.DataFrame, dict[str, pl.DataFrame]]:
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
            ``grouped_dfs`` maps each grouped parent uid to its result
            DataFrame, from the fused ``group_by().agg().sort()`` or from the
            spec's own plan.
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

        # A grouped spec with its own plan cannot fuse; run it alone on the same
        # rows the batched path would have given it.
        grouped_plan_specs = [s for s in grouped_specs if s.plan is not None]
        grouped_expr_specs = [s for s in grouped_specs if s.plan is None]

        grouped_dfs: dict[str, pl.DataFrame] = {}
        for spec in grouped_plan_specs:
            batch_ldf = (
                filtered_ldf
                if not spec.pre_group_filters
                else filtered_ldf.filter(*spec.pre_group_filters)
            )
            grouped_dfs[spec.uid] = spec.plan(batch_ldf)

        grouped_batches: dict[tuple, list[GroupedAggregationSpec]] = {}
        for spec in grouped_expr_specs:
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

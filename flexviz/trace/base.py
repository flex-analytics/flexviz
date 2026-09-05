"""Abstract base class for all flexviz traces.

A FlexTrace is a pure-Python, renderer-agnostic object. It owns:
- the mapping from semantic data roles to LazyFrame column names
  (``_backend_data``)
- display hints that adapters interpret (``_display``)
- the aggregation and filtering logic that the engine executes

FlexTrace deliberately contains zero renderer imports.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
import datetime
from typing import Any, get_args
from uuid import uuid4
import hashlib
import json
import math
import re

import polars as pl

from ..cube import CubeTargetSpec, FreeAxisSpec, MeasureAgg, MeasureSpec
from ..LF import AggregationSpec, GroupedAggregationSpec
from ..spec import BackendDataValue, TraceHoverSpec, TraceSelectionSpec, TraceSpec


@dataclass
class GroupedChildResult:
    """Renderer-agnostic payload for one grouped child trace."""

    child_uid: str
    group_value_key: str
    updates: dict[str, pl.Series | list | Any] = field(default_factory=dict)


@dataclass
class TraceResult:
    """Return value of ``FlexTrace._to_update`` / ``_to_grouped_update``.

    ``updates`` maps semantic keys (``"x"``, ``"y"``, …) to ``pl.Series``,
    Python lists, or scalar values.  The engine normalizes all ``pl.Series``
    values to Python lists before building ``TraceDelta``; keeping Series here
    avoids premature ``.to_list()`` calls inside trace implementations.

    For grouped parents ``group_results`` carries one ``GroupedChildResult``
    per visible group; ``updates`` is empty in that case.

    Extension points:
    - Phase 2 (group_by): ``group_results``
    - Phase 3 (overlay):  ``layer: Literal["fg", "bg"]``
    """

    updates: dict[str, pl.Series | list | Any] = field(default_factory=dict)
    group_results: list[GroupedChildResult] | None = None


class FlexTrace(ABC):
    """Renderer-agnostic base class for a single data trace.

    Subclasses implement two abstract methods:

    ``get_aggregation_spec``
        Return an ``AggregationSpec`` that the engine will execute against
        the shared ``LFQueryBuilder``.  The result column **must** be aliased
        as ``self.uid``.

    ``_to_update``
        Convert the aggregated DataFrame column (named ``self.uid``) into a
        ``TraceDelta.updates`` dict using **semantic keys** (``"x"``,
        ``"y"``, …) that are renderer-agnostic.

    Cross-filtering is no longer the trace's responsibility — the engine
    compiles ``SelectionState.predicates`` directly into Polars expressions
    via ``flexviz.predicates.predicates_to_expr``.
    """

    # Subclasses override these at class level.
    trace_type: str = ""
    overlay_style: str = "full"  # "full" | "filtered_only"
    # One-line human description of the zoom re-aggregation policy, surfaced in
    # the generated Architecture.md table. Override alongside
    # ``_default_recompute_axes``.
    recompute_policy_doc: str = "none — never re-aggregates on zoom"
    # One-line human description of the selection geometry, surfaced in the
    # generated Architecture.md table. Override alongside ``_default_select_axes``.
    select_policy_doc: str = "all axes — full 2-D rectangular selection"

    def __init__(
        self,
        backend_data: dict[str, BackendDataValue],
        display: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        recompute_axes: tuple[str, ...] | None = None,
        axes: tuple[str, ...] | None = ("x", "y"),
    ) -> None:
        self.uid: str = str(uuid4())
        self._backend_data: dict[str, BackendDataValue] = backend_data
        self._display: dict[str, Any] = display or {}
        self._params: dict[str, Any] = params or {}
        self._axes: tuple[str, ...] | None = axes
        # Source of truth for zoom-driven re-aggregation. ``None`` means
        # "derive from the trace's policy" (``_default_recompute_axes``); an
        # explicit tuple — including ``()`` for a frozen line — overrides it.
        self._recompute_axes: tuple[str, ...] = (
            tuple(recompute_axes)
            if recompute_axes is not None
            else self._default_recompute_axes()
        )

    # ------------------------------------------------------------------
    # Zoom-driven re-aggregation policy
    # ------------------------------------------------------------------

    def _default_recompute_axes(self) -> tuple[str, ...]:
        """Anchor ids whose viewport *range* parameterizes this aggregation.

        Expressed in anchor space (a subset of ``self._axes``, or
        ``("coordinates",)`` for map traces).  The base default is empty: a
        trace is never re-aggregated on zoom/pan unless it opts in.  Subclasses
        override to declare their data-binding axes.
        """
        return ()

    @property
    def recompute_axes(self) -> tuple[str, ...]:
        """The effective set of axes whose range change triggers re-aggregation."""
        return self._recompute_axes

    @property
    def update_on_zoom(self) -> bool:
        """Interpretable derived flag: does *any* viewport change re-aggregate
        this trace?  Equivalent to ``bool(self.recompute_axes)``."""
        return bool(self._recompute_axes)

    # ------------------------------------------------------------------
    # Selection geometry (which axes a box-select brushes on)
    # ------------------------------------------------------------------

    def _default_select_axes(self) -> tuple[str, ...]:
        """Anchor ids whose box-select range becomes a selection clause.

        Anchor space (a subset of ``self._axes``).  The base default selects on
        *all* of a trace's axes (a full 2-D rectangular selection), preserving
        prior behaviour.  Subclasses override to restrict the geometry — e.g. a
        line is x-only so a brush works across multiple series sharing the x
        axis, and a 1-D histogram/box selects only on its data (prop) axis.
        """
        return tuple(self._axes) if self._axes else ()

    @property
    def select_axes(self) -> tuple[str, ...]:
        """The anchor ids this trace emits selection clauses on.

        Computed (not cached) from ``_default_select_axes()``; the cube engine
        (``FlexEngine._locate_free_axis`` / ``_locate_box2d_axis``) reads it to
        map a brushed source trace onto its free-axis column(s)."""
        return self._default_select_axes()

    def _make_selection_spec(self) -> "TraceSelectionSpec":
        """Return the cross-filter selection geometry for this trace.

        The default is ``kind="none"`` (not a cross-filter source).  Override in
        concrete subclasses — range traces typically just delegate to
        ``_range_selection_spec()``; categorical/path/geo traces build their own.
        Mirrors ``_make_hover_spec``; consumed by the generic client runtime so
        selection never branches on ``trace_type``.
        """
        return TraceSelectionSpec()

    def _range_selection_spec(self, multi: str = "replace") -> "TraceSelectionSpec":
        """Build a ``kind="range"`` selection spec from this trace's policy.

        Maps each *selectable* anchor (``_default_select_axes()``) to its column by
        the standard role convention — ``axes[0]`` is the x-role
        (``backend_data["x"]``) and ``axes[1]`` the y-role
        (``backend_data["y"]``); a 1-D histogram/box stores its single column
        under the matching ``"x"``/``"y"`` key.  Works for line, histogram, box,
        and histogram2d without per-trace column wiring.
        """
        axis_columns: dict[str, str] = {}
        axes = self._axes or ()
        role_for = {axes[0]: "x", axes[1]: "y"} if len(axes) >= 2 else {}
        for anchor in self._default_select_axes():
            col = self._backend_data.get(role_for.get(anchor, ""))
            if isinstance(col, str):
                axis_columns[anchor] = col
        return TraceSelectionSpec(kind="range", axis_columns=axis_columns, multi=multi)

    # ------------------------------------------------------------------
    # Cube descriptors (cross-filter pre-aggregation)
    # ------------------------------------------------------------------

    def get_cube_source_spec(
        self,
        axis_range: tuple[float, float] | None,
        schema: pl.Schema | None = None,
    ) -> "FreeAxisSpec | None":
        """The free axis a brush on this trace defines, or None (not a cube source).

        axis_range — the source figure's viewport on this trace's selectable axis
        (None = unzoomed; domain resolution happens in the engine).
        """
        return None

    def get_cube_target_spec(
        self,
        axis_range: tuple[float, float] | None,
        schema: pl.Schema | None = None,
    ) -> "CubeTargetSpec | None":
        """This trace's grouping+measure as a cube target, or None (fall back)."""
        return None

    def domain_cols(
        self,
        update_range: dict[str, Any],
        *,
        scan_source: bool = False,
    ) -> tuple[str, ...]:
        """Columns whose **unfiltered** ``(min, max)`` this trace's spec needs.

        Empty when the viewport already supplies the bounds, or when the trace
        needs none. The engine unions these over the request and resolves them
        in one min/max collect, so bin edges stay put under cross-filtering.

        ``scan_source`` says the rows come from storage rather than a resident
        frame; only a trace that swaps formulation on residency reads it.
        """
        return ()

    # ------------------------------------------------------------------
    # Abstract interface (implemented by concrete trace classes)
    # ------------------------------------------------------------------

    @abstractmethod
    def get_aggregation_spec(
        self,
        update_range: dict[str, Any],
        schema: pl.Schema | None = None,
    ) -> AggregationSpec | GroupedAggregationSpec:
        """Return an aggregation spec for the engine to execute.

        Parameters
        ----------
        update_range:
            Viewport axis ranges.  Keys are single-letter axis names
            (``"x"``, ``"y"``) with ``(min, max)`` tuple values.
            May be empty (e.g. on init or force-update).
        Returns
        -------
        AggregationSpec | GroupedAggregationSpec
            Result expressions **must** be aliased as ``self.uid`` when they
            belong to this logical parent trace.
        """

    @abstractmethod
    def _to_update(
        self,
        df_agg: pl.DataFrame,
    ) -> TraceResult:
        """Extract a ``TraceResult`` from the aggregated DataFrame.

        Parameters
        ----------
        df_agg:
            The aggregated result from ``LFQueryBuilder.aggregate``.
            Contains a column named ``self.uid``.

        Returns
        -------
        TraceResult
            ``updates`` maps semantic keys (``"x"``, ``"y"``, …) to
            ``pl.Series``, Python lists, or scalar values.  The engine
            normalizes ``pl.Series`` to lists; subclasses do not need to
            call ``.to_list()`` themselves.
        """

    # ------------------------------------------------------------------
    # Grouped-trace helpers
    # ------------------------------------------------------------------

    @property
    def group_by(self) -> str | list[str] | None:
        return self._params.get("group_by")

    @property
    def group_by_cols(self) -> tuple[str, ...] | None:
        gb = self._params.get("group_by")
        return None if gb is None else _to_col_tuple(gb, "group_by")

    @property
    def group_domain_key(self) -> str | None:
        return self._params.get("group_domain_key")

    def is_grouped_parent(self) -> bool:
        return self.group_by_cols is not None

    def _to_grouped_update(self, df_grouped: pl.DataFrame) -> TraceResult:
        """Convert a grouped result frame into a parent-scoped ``TraceResult``.

        Subclasses that support grouped execution must override this.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement grouped updates"
        )

    # ------------------------------------------------------------------
    # Filtering helpers
    # ------------------------------------------------------------------

    def _range_filter_exprs(
        self,
        col_name: str,
        range_: tuple | None,
        schema: pl.Schema | None = None,
    ) -> list[pl.Expr]:
        """Return a typed ``is_between`` filter expression as a single-element list.

        Wraps the module-level ``_range_filter_expr`` helper and returns a
        list so callers can use ``exprs += self._range_filter_exprs(...)``
        without a None-check.  Returns ``[]`` when ``range_`` is ``None``.
        """
        expr = _range_filter_expr(col_name, range_, schema)
        return [expr] if expr is not None else []

    # ------------------------------------------------------------------
    # Spec serialisation / deserialisation
    # ------------------------------------------------------------------

    def _make_hover_spec(self) -> "TraceHoverSpec":
        """Return the hover capability spec for this trace.

        Override in concrete subclasses to declare source/target modes.
        The default returns empty capabilities (no linked hover).
        """
        return TraceHoverSpec()

    def to_trace_spec(self, domain_source: str | None = None) -> TraceSpec:
        """Serialise this trace to a ``TraceSpec`` (wire format).

        Parameters
        ----------
        domain_source:
            When provided and the trace has ``group_by`` set,
            ``group_domain_key`` is computed as
            ``"{domain_source}::{group_by}"`` and included in ``params``.
        """
        params = dict(self._params)
        group_cols = self.group_by_cols
        if (
            domain_source is not None
            and group_cols is not None
            and "group_domain_key" not in params
        ):
            if len(group_cols) == 1:
                suffix = group_cols[0]
            else:
                suffix = json.dumps(list(group_cols), separators=(",", ":"))
            params["group_domain_key"] = f"{domain_source}::{suffix}"
        return TraceSpec(
            uid=self.uid,
            trace_type=self.trace_type,
            axes=self._axes,
            backend_data=self._backend_data,
            display=dict(self._display),
            params=params,
            recompute_axes=self._recompute_axes,
            selection=self._make_selection_spec(),
            hover=self._make_hover_spec(),
        )

    @classmethod
    def from_trace_spec(cls, spec: TraceSpec) -> "FlexTrace":
        """Reconstruct a trace from a ``TraceSpec`` (server-side factory).

        The default implementation calls ``cls.__init__`` with the fields
        from ``spec`` and restores ``uid``.  Subclasses may override if
        they need special reconstruction logic.
        """
        trace = cls(
            backend_data=spec.backend_data,
            display=spec.display,
            params=spec.params,
            recompute_axes=spec.recompute_axes,
            axes=spec.axes,
        )
        trace.uid = spec.uid
        return trace

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"uid={self.uid!r}, "
            f"backend_data={self._backend_data!r})"
        )


def _to_col_tuple(columns: str | Sequence[str], param_name: str) -> tuple[str, ...]:
    """Coerce a string or sequence of strings to a non-empty tuple of column names."""
    if isinstance(columns, str):
        return (columns,)
    if not isinstance(columns, Sequence):
        raise TypeError(f"{param_name} must be a string or a list/tuple of strings")
    cols = tuple(columns)
    if not cols:
        raise ValueError(f"{param_name} must contain at least one column")
    if not all(isinstance(c, str) for c in cols):
        raise TypeError(f"{param_name} must be a string or a list/tuple of strings")
    return cols


def _composite_label(cols: tuple[str, ...]) -> str:
    """Compact human-readable label for one or more column names."""
    if len(cols) == 1:
        return cols[0]
    return "[" + ", ".join(cols) + "]"


def _group_values_from_frame(
    df: pl.DataFrame,
    group_cols: tuple[str, ...],
) -> list[Any]:
    """Return scalar group values or composite tuples from grouped output rows."""
    if len(group_cols) == 1:
        return df[group_cols[0]].to_list()
    return list(df.select(list(group_cols)).iter_rows())


def _sanitize_group_value(value: Any) -> str:
    """Return a filesystem/uid-safe string for a group value."""
    text = value if isinstance(value, str) else _group_value_key(value)
    label = re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")
    return label or "group"


def _group_value_key(value: Any) -> str:
    """Return a stable string key for a grouped value."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    except TypeError:
        return str(value)


def child_uid_from_group_key(parent_uid: str, group_value_key: str) -> str:
    """Return the deterministic child uid for a parent uid + group value key.

    ``group_value_key`` is the stable string produced by ``_group_value_key``
    (the same value carried on ``GroupedChildResult.group_value_key``).  This
    lets callers re-derive a child uid from a cached, uid-agnostic payload —
    the cache stores ``group_value_key`` rather than the resolved child uid so
    the same grouped result can be re-stamped for any requesting parent.
    """
    parent_label = _sanitize_group_value(parent_uid)
    label = _sanitize_group_value(group_value_key)[:32]
    digest = hashlib.sha1(group_value_key.encode("utf-8")).hexdigest()[:12]
    return f"fv_{parent_label}_{label}_{digest}"


def _child_uid_for_group(parent_uid: str, group_value: Any) -> str:
    """Return a deterministic child uid without relying on lossy sanitization."""
    return child_uid_from_group_key(parent_uid, _group_value_key(group_value))


def _dtype_for_col(schema: pl.Schema | None, col_name: str) -> pl.DataType | None:
    """Return the dtype for ``col_name`` from a Polars schema (or ``None``)."""
    if schema is None:
        return None
    try:
        return schema[col_name]
    except Exception:
        return None


#: The cube measure algebra (cube contract A). ``median``/``n_unique`` have no
#: decomposable partials, so traces using them are not cube targets.
_CUBE_MEASURE_AGGS: frozenset[str] = frozenset(
    get_args(MeasureAgg)  # ("count", "sum", "mean", "min", "max")
)

#: Column names reserved by the cube's long-format frame: the measure partial
#: columns plus the free-axis key (cube contract A). A categorical dim using
#: one of these names would collide, so descriptor methods return ``None``.
_CUBE_RESERVED_COLS: frozenset[str] = frozenset(
    {"count", "sum", "min", "max", "free_bin"}
)


def _categorical_dims_ok(
    schema: pl.Schema | None,
    cols: Sequence[str],
    *,
    allow_numeric: bool = False,
) -> bool:
    """Shared dtype gate for cube categorical dim columns (contract B).

    Every categorical dim column — free-axis cols, bar/pie label cols, treemap
    path cols, ``group_by`` cols used as extra target dims — must be
    ``String``, ``Categorical``, or ``Enum`` by default. A schema is therefore
    *required* for categorical capability: ``schema=None`` (or a column missing
    from it) ⇒ not cube-capable. Reserved partial-column names are rejected too
    (``_CUBE_RESERVED_COLS``).

    ``allow_numeric`` additionally accepts integer and float dtypes for
    bar/pie **label** dims and their matching categorical **free axis**. The
    cube codec keeps those categories typed in the FVCube header, so the
    browser compares values rather than relying on Python and JavaScript
    string formatting to agree (notably ``1.0`` vs ``1``). Treemap paths and
    bar ``group_by`` dims keep the default string-only gate because their
    identity is still renderer string-keyed.
    """
    if schema is None:
        return False
    for col in cols:
        if col in _CUBE_RESERVED_COLS:
            return False
        dtype = _dtype_for_col(schema, col)
        if dtype is None:
            return False
        if isinstance(dtype, (pl.String, pl.Categorical, pl.Enum)):
            continue
        if allow_numeric and (dtype.is_integer() or dtype.is_float()):
            continue
        return False
    return True


def _cube_measure_spec(
    schema: pl.Schema | None, agg: str, value_col: str | None
) -> MeasureSpec | None:
    """Build a cube ``MeasureSpec`` from a trace's stored agg, or ``None``.

    Gates (cube contracts A/B): the agg must be in the cube measure algebra
    (``median``/``n_unique`` ⇒ ``None``), and for every agg except ``count``
    the value column must have a numeric dtype in the schema —
    temporal/string values ⇒ ``None``. Traces store ``agg == "count"`` (with
    no value column) when ``values`` is omitted, matching ``MeasureSpec``'s
    requirement that ``count`` takes no ``value_col``.
    """
    if agg not in _CUBE_MEASURE_AGGS:
        return None
    if agg == "count":
        return MeasureSpec(agg="count")
    dtype = _dtype_for_col(schema, value_col) if value_col is not None else None
    if dtype is None or not dtype.is_numeric():
        return None
    return MeasureSpec(agg=agg, value_col=value_col)


def _typed_temporal_lit(value: Any, dtype: pl.DataType | None) -> pl.Expr:
    assert dtype.is_temporal(), "dtype must be temporal"
    if isinstance(value, str):
        stripped = value.strip()
        if len(stripped) == 10 and stripped[4] == "-" and stripped[7] == "-":
            parsed_date = datetime.date.fromisoformat(stripped)
            if dtype == pl.Date:
                return pl.lit(parsed_date, dtype=dtype)
            parsed_dt = datetime.datetime.combine(parsed_date, datetime.time())
            return pl.lit(parsed_dt, dtype=dtype)

        iso_candidate = stripped
        if iso_candidate.endswith("Z"):
            iso_candidate = iso_candidate[:-1] + "+00:00"
        try:
            parsed_dt = datetime.datetime.fromisoformat(iso_candidate)
        except ValueError:
            parsed_dt = None
        if parsed_dt is not None:
            col_tz = getattr(dtype, "time_zone", None)
            if parsed_dt.tzinfo is not None and col_tz is None:
                # A naive target column cannot be compared against a tz-aware
                # literal (Polars raises SchemaError). Keep the wall-clock
                # components and drop the tz so the comparison stays valid.
                parsed_dt = parsed_dt.replace(tzinfo=None)
            elif parsed_dt.tzinfo is not None and col_tz is not None:
                # tz-aware column + tz-aware value: pl.lit(dt, dtype=tz) raises
                # when the value's offset differs from the column tz (e.g.
                # "...+02:00" against a UTC column). The offset already pins the
                # absolute instant, so normalise to UTC and cast into the column
                # dtype (instant-preserving) rather than matching tz strings.
                utc_dt = parsed_dt.astimezone(datetime.timezone.utc)
                return pl.lit(utc_dt).cast(dtype)
            return pl.lit(parsed_dt, dtype=dtype)

        # Common formats produced by browsers / Plotly for datetime axes:
        # - "YYYY-MM-DD HH:MM:SS"
        # - "YYYY-MM-DD HH:MM:SS.ffffff"
        # - same with "T" separator
        sep = "T" if "T" in value else " "
        if "." in value:
            fmt = f"%Y-%m-%d{sep}%H:%M:%S%.f"
        else:
            fmt = f"%Y-%m-%d{sep}%H:%M:%S"

        return pl.lit(value).str.strptime(dtype, fmt, strict=False)

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        timestamp_ms = float(value)
        if math.isfinite(timestamp_ms):
            dt_value = datetime.datetime.fromtimestamp(
                timestamp_ms / 1000.0, tz=datetime.timezone.utc
            )
            if dtype == pl.Date:
                return pl.lit(dt_value.date(), dtype=dtype)
            # Naive target column: keep the UTC wall-clock components but drop
            # the tz so the literal can be compared against a naive column.
            if getattr(dtype, "time_zone", None) is None:
                dt_value = dt_value.replace(tzinfo=None)
            return pl.lit(dt_value, dtype=dtype)

    # For python datetime/date objects, let Polars cast directly.
    return pl.lit(value, dtype=dtype)


# ---------------------------------------------------------------------------
# Temporal ↔ physical helpers for the numeric histogram kernels.
#
# ``fixed_hist`` / ``fixed_hist2d`` are numeric-only Rust kernels: a temporal
# dtype (or a temporal bound literal) makes them panic. Temporal columns are
# therefore binned on their *physical* representation (µs for ``Datetime("us")``,
# days for ``Date`` — i.e. ``Series.to_physical()``), mirroring the cube's
# ``_typed_temporal_lit(...).to_physical()`` idiom (``engine._cube_axis_range``).
# Bin centers are cast back to the temporal dtype so the delta carries datetimes
# and the renderer auto-detects a date axis, exactly like the line trace.
# ---------------------------------------------------------------------------


def _temporal_dtype_for_col(
    col_name: str, schema: pl.Schema | None
) -> pl.DataType | None:
    """The temporal dtype of ``col_name``, or ``None`` when it is non-temporal
    or the schema is unavailable."""
    dtype = _dtype_for_col(schema, col_name)
    return dtype if (dtype is not None and dtype.is_temporal()) else None


def _physical_bound_expr(value: Any, dtype: pl.DataType) -> pl.Expr:
    """A temporal range bound (epoch-ms number or date string) → a physical-unit
    literal expression in ``dtype``'s physical representation."""
    return _typed_temporal_lit(value, dtype).to_physical()


def _phys_epoch_ms_factor(dtype: pl.DataType) -> float:
    """Multiplier turning a physical temporal value into epoch-milliseconds
    (Plotly's numeric date coordinate, used for hover-band bounds)."""
    if isinstance(dtype, pl.Datetime):
        return {"ms": 1.0, "us": 1e-3, "ns": 1e-6}.get(dtype.time_unit, 1.0)
    if dtype == pl.Date:
        return 86_400_000.0  # days → ms
    return 1.0


def _physical_to_temporal_series(
    values: Sequence[float] | pl.Series, dtype: pl.DataType, name: str = ""
) -> pl.Series:
    """Cast physical bin centers (floats in ``dtype``'s physical unit) back to
    the temporal ``dtype`` (rounded to the nearest physical unit)."""
    s = values if isinstance(values, pl.Series) else pl.Series(name, list(values))
    return s.round().cast(pl.Int64).cast(dtype)


def _typed_range_bounds(
    col_name: str,
    range_: tuple[Any, Any] | None,
    schema: pl.Schema | None = None,
    closed: str = "both",
) -> tuple[pl.Expr, pl.Expr] | None:
    """Return typed lower/upper literal expressions for a column range.

    ``closed`` must match the ``is_between`` closed-ness the caller will
    apply to the returned bounds; it only affects integer-column rounding.
    """
    if range_ is None:
        return None

    lo, hi = range_
    dtype = _dtype_for_col(schema, col_name)
    if dtype is None:
        return (pl.lit(lo), pl.lit(hi))
    if dtype.is_temporal():
        return (_typed_temporal_lit(lo, dtype), _typed_temporal_lit(hi, dtype))
    if dtype.is_integer():
        # Convert float bounds to integers preserving real-valued membership
        # over the integers, per side of the interval:
        #   closed bound → round toward the interval interior:
        #     lo=0.25, x ≥ 0.25 ⟺ x ≥ ceil(0.25) = 1
        #     hi=86.5, x ≤ 86.5 ⟺ x ≤ floor(86.5) = 86
        #   open bound → round away from the interior:
        #     lo=0.2,  x > 0.2  ⟺ x > floor(0.2) = 0
        #     hi=1.2,  x < 1.2  ⟺ x < ceil(1.2)  = 2
        # Exact-integer float bounds are unchanged by either rounding.
        # trunc (Polars' default cast) is wrong for positive fractional lo and
        # negative fractional hi (it rounds toward zero instead of inward).
        # Casting literals to the column dtype also avoids Polars upcasting the
        # entire column to f64 (measured ~10x speedup).
        lo_closed = closed in ("both", "left")
        hi_closed = closed in ("both", "right")
        if isinstance(lo, float):
            lo = math.ceil(lo) if lo_closed else math.floor(lo)
        if isinstance(hi, float):
            hi = math.floor(hi) if hi_closed else math.ceil(hi)
        return (
            pl.lit(lo).cast(dtype, strict=False),
            pl.lit(hi).cast(dtype, strict=False),
        )
    # Cast to match column dtype (e.g. f32) to avoid implicit column upcast
    # to f64 (measured ~5x slower at 5M rows).
    return (pl.lit(lo).cast(dtype, strict=False), pl.lit(hi).cast(dtype, strict=False))


def _range_filter_expr(
    col_name: str,
    range_: tuple[Any, Any] | None,
    schema: pl.Schema | None = None,
) -> pl.Expr | None:
    """Build a typed ``is_between`` expression for ``col_name`` and ``range_``."""
    bounds = _typed_range_bounds(col_name, range_, schema)
    if bounds is None:
        return None
    return pl.col(col_name).is_between(*bounds)

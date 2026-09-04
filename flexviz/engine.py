"""Core aggregation engine, decoupled from any rendering or transport layer."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Literal

import polars as pl

from .cache import CacheBackend, content_key
from .cube import (
    CubeSpec,
    CubeTargetSpec,
    FreeAxisSpec,
    build_cube,
    cube_content_key,
    cube_target_buildable,
    encode_fvcube,
    temporal_unit,
)
from .events import ActiveSource, GroupedChildDelta, InteractionEvent, TraceDelta
from .LF import (
    LFQueryBuilder,
    AggregationSpec,
    GroupedAggregationSpec,
    check_line_x_dtype,
)
from .spec import SelectionState
from .trace.base import (
    FlexTrace,
    _dtype_for_col,
    _typed_temporal_lit,
    child_uid_from_group_key,
)
from .trace.hist import _HIST_BIN_EPSILON
from .predicates import canonical_passive_key, predicates_to_expr

# Event types whose computation is unfiltered and viewport-free. These are the
# only events cached in Phase 1 (#26): the engine forces them to drop all
# selections (`_active_selections`), so the result is the unfiltered base — a
# float-free, content-addressable computation shared across reloads/viewers.
_CACHEABLE_EVENT_TYPES = frozenset({"init", "reset", "deselect"})

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TraceInfo:
    """Lightweight description of a trace's identity and axes."""

    uid: str
    axes: tuple | None  # (x_anchor, y_anchor) for cartesian
    trace_type: str
    figure_uid: str | None = None  # dashboard figure uid; None for single-figure use


@dataclass(frozen=True)
class _AggregationTrace:
    info: TraceInfo
    trace: FlexTrace
    update_range: Dict[str, Any]


def _union_shared_domains(
    domains: Dict[str, tuple[float, float]],
    sibling_cols: tuple[str, ...] | None,
) -> Dict[str, tuple[float, float]] | None:
    """Widen every ``sibling_cols`` domain to the union over the whole group.

    The target's own binned dim is one of ``sibling_cols``, so widening the
    group covers it without the caller having to know which column that is.
    Returns ``domains`` unchanged when there is nothing to widen, or ``None``
    when a sibling's domain is unresolvable (all-null column) — the caller then
    skips the target rather than serving bins the bg layer will not match.
    """
    if not sibling_cols or len(sibling_cols) < 2:
        return domains
    bounds = [domains[c] for c in sibling_cols if c in domains]
    if len(bounds) != len(sibling_cols):
        return None
    union = (min(b[0] for b in bounds), max(b[1] for b in bounds))
    return {**domains, **{c: union for c in sibling_cols}}


def _normalize_axis_ranges(ranges: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of ``ranges`` with every descending ``(lo, hi)`` swapped.

    A reversed plotly axis reports its viewport high-to-low, and clients
    forward it verbatim. Every consumer downstream assumes ``lo <= hi`` —
    ``is_between`` masks return nothing and the sorted-x ``search_sorted``
    slice returns the wrong rows — so order is normalized once, here at
    ingestion. Non-range values (``None``, map coordinate point lists) pass
    through untouched, as does any pair that does not compare.
    """
    out: Dict[str, Any] = {}
    for axis, rng in ranges.items():
        if (
            isinstance(rng, (list, tuple))
            and len(rng) == 2
            and not isinstance(rng[0], (list, tuple))
        ):
            lo, hi = rng
            try:
                if lo is not None and hi is not None and hi < lo:
                    rng = (hi, lo)
            except TypeError:
                pass  # incomparable pair — leave as sent
        out[axis] = rng
    return out


class FlexEngine:
    """Renderer-agnostic, fully stateless aggregation engine.

    Holds references to the shared ``LFQueryBuilder`` and the scalable-trace
    registry.  Owns no mutable state — all interaction state is passed in on
    every ``process`` call and never stored.  This makes the engine safe to
    construct per-request and trivially thread-safe.
    """

    def __init__(
        self,
        backend_lf: LFQueryBuilder | None,
        scalable_traces: Dict[str, FlexTrace],
        cache_backend: CacheBackend | None = None,
        source_name: str | None = None,
    ):
        self._backend_lf = backend_lf
        self._scalable_traces = scalable_traces
        # Caching is active only when a backend supplies a cache and the source
        # opted in (server passes both). Keyed by source_name so entries from
        # different sources never collide.
        self._cache = cache_backend
        self._source_name = source_name

    # -- main entry-point ------------------------------------------------------

    def process(
        self,
        event: InteractionEvent,
        trace_infos: List[TraceInfo],
        viewports_by_figure: Dict[str, Dict[str, Any]] | None = None,
        cross_filter_mode: Literal["update", "overlay"] = "update",
    ) -> List[TraceDelta]:
        """Process an ``InteractionEvent`` and return per-trace deltas.

        Fully stateless: all interaction state is supplied by the caller on
        every call.

        Parameters
        ----------
        event:
            The interaction event to process.
        trace_infos:
            Ordered list of all trace descriptors for the active source(s).
            Each carries a ``figure_uid`` so the engine can internally
            separate filter-expression sources from aggregation targets.
        """
        logger.info(f"FlexEngine.process: {event.type}")
        t_start_0 = time.perf_counter()
        if viewports_by_figure is None:
            viewports_by_figure = {}
        else:
            # Normalize once at ingestion; every downstream consumer
            # (trace update ranges, cube/hist domain resolution) then holds
            # the lo <= hi invariant.
            viewports_by_figure = {
                fig: _normalize_axis_ranges(vp)
                for fig, vp in viewports_by_figure.items()
            }

        backend_schema = (
            self._backend_lf.schema if self._backend_lf is not None else None
        )
        active_selections = self._active_selections(event)
        selection_fig_uids = {
            sel.source_figure_uid
            for sel in active_selections
            if sel.source_figure_uid is not None
        }

        filter_exprs = self._selection_filter_exprs(
            active_selections=active_selections,
            backend_schema=backend_schema,
            event=event,
        )

        aggregation_traces = self._aggregation_traces(
            event=event,
            trace_infos=trace_infos,
            selection_fig_uids=selection_fig_uids,
            viewports_by_figure=viewports_by_figure,
        )

        # Sibling histograms on one figure bin over a shared domain, so this
        # grouping feeds both the aggregation specs and the cache key (an
        # entry computed for a different sibling set is a different result).
        # ``data_axis_zoomed`` is "has a real viewport range", not merely
        # "key present": an autorange/reset posts ``{axis: null}`` (unzoomed),
        # which must NOT drop the histogram from its shared-domain group — else
        # siblings fall back to per-column domains and misalign. Mirrors the
        # cube caller's ``anchor_range is not None`` and ``get_aggregation_spec``
        # itself, which treat ``None`` as unzoomed.
        histogram_domains = self._histogram_domain_cols_by_uid(
            (
                (
                    item.info,
                    item.trace,
                    item.update_range.get(getattr(item.trace, "prop_key", None))
                    is not None,
                )
                for item in aggregation_traces
            ),
            schema=backend_schema,
        )

        # -- cache fast-path (Phase 1: unfiltered, viewport-free only) ---------
        cache_layer = "bg" if cross_filter_mode == "overlay" else None
        cache_items = self._cacheable_items(
            event, aggregation_traces, histogram_domains
        )
        # Short-circuit only when every delta-producing trace is a viewport-free
        # cache hit. If any deliverable trace is viewport-dependent it is absent
        # from cache_items, so the length check fails and we fall through to a
        # normal recompute of all traces (the viewport-free ones still get
        # stored below for a future fully-unzoomed request).
        if cache_items and len(cache_items) == len(aggregation_traces):
            cached = [(it, k, self._cache.get(k)) for it, k in cache_items]
            if all(payload is not None for _, _, payload in cached):
                logger.debug("Cache hit: %d trace(s) served from cache", len(cached))
                return [
                    self._delta_from_cached(it, payload, cache_layer)
                    for it, _, payload in cached
                ]
            logger.debug(
                "Cache miss: %d/%d cacheable trace(s) absent; recomputing",
                sum(1 for _, _, payload in cached if payload is None),
                len(cached),
            )

        # Before the domain scan: the check leaves the x column flagged sorted,
        # which makes the min/max collect O(1) on this very first request.
        line_x_cols = self._check_line_x(aggregation_traces)

        # Resolved only here, past the fast-path: a fully cached request needs
        # no min/max scan at all. Both overlay layers reuse these specs, so the
        # shared unfiltered domain is resolved once per request.
        domain_cols, domains = self._resolve_domains(
            aggregation_traces, histogram_domains, backend_schema, line_x_cols
        )

        t_agg_start = time.perf_counter()
        agg_specs = self._collect_aggregation_specs(
            aggregation_traces=aggregation_traces,
            backend_schema=backend_schema,
            domain_cols=domain_cols,
            domains=domains,
        )
        t_agg_end = time.perf_counter()
        logger.info(f"Total get agg_spec time: {t_agg_end - t_agg_start:.4f}s")

        if not agg_specs:
            return []

        t_start = time.perf_counter()
        if cross_filter_mode == "overlay":
            deltas = self._process_overlay_mode(
                agg_specs=agg_specs,
                filter_exprs=filter_exprs,
                has_active_selections=bool(active_selections),
                trace_infos=trace_infos,
                event=event,
                selection_fig_uids=selection_fig_uids,
                viewports_by_figure=viewports_by_figure,
            )
        else:
            deltas = self._process_update_mode(
                agg_specs=agg_specs,
                filter_exprs=filter_exprs,
                trace_infos=trace_infos,
                event=event,
                selection_fig_uids=selection_fig_uids,
                viewports_by_figure=viewports_by_figure,
            )

        if cache_items:
            self._store_in_cache(cache_items, deltas)

        t_end = time.perf_counter()
        logger.info(f"Process mode time: {t_end - t_start:.4f}s")
        logger.info(f"Total wall time: {t_end - t_start_0:.4f}s")

        return deltas

    # -- cube path (range + categorical sources; count/sum/mean/min/max) --------

    def build_cubes(
        self,
        trace_infos: List[TraceInfo],
        viewports_by_figure: Dict[str, Dict[str, Any]],
        selections: List[SelectionState],
        active_source: ActiveSource,
        cube_cache: CacheBackend | None = None,
    ) -> tuple[List[bytes], Dict[str, int]]:
        """Build (or fetch) the cubes serving a ``cube_request`` event.

        Fully stateless, like ``process``: the caller supplies the trace set,
        per-figure viewports (``state.viewport``), committed selections
        (``state.selections``) and the active source on every call.

        Returns ``(cubes, trace_cubes)`` where ``cubes`` holds one encoded
        FVCube blob per *distinct* ``cube_content_key`` and ``trace_cubes``
        maps every served target trace uid to its blob index.  Returns empty
        when the source figure has no matching cube-source trace or when no
        target trace is cube-capable.

        Passive baking (contract E): every committed selection from another
        figure is pre-applied to the build frame as a filter and joins the
        content address via the canonical passive key. The source figure's
        own selection is ignored (re-brush case), as are ``None``-uid
        selections (they never filter in the legacy engine). Domain
        resolution stays on the UNFILTERED frame — unzoomed domains are
        unfiltered min/max, so bin edges are filter-stable, exactly as in the
        aggregation path.
        """
        if self._backend_lf is None or self._source_name is None:
            return [], {}

        schema = self._backend_lf.schema

        free_spec = self._locate_free_axis(
            trace_infos, viewports_by_figure, active_source, schema
        )
        if free_spec is None:
            return [], {}

        passive = [
            sel
            for sel in selections
            if sel.source_figure_uid is not None
            and sel.source_figure_uid != active_source.figure_uid
            and sel.predicates
        ]
        passive_key = canonical_passive_key(selections, active_source.figure_uid)
        build_ldf = self._backend_lf._ldf
        if passive:
            build_ldf = build_ldf.filter(
                *[predicates_to_expr(sel.predicates, schema) for sel in passive]
            )
        owning_figures = {sel.source_figure_uid for sel in passive}

        # Targets: every cube-capable trace in every OTHER figure that owns
        # no committed selection (mirrors ``_should_process_trace``: legacy
        # selection events never update selection-owning figures).
        targets: list[tuple[TraceInfo, FlexTrace, CubeTargetSpec]] = []
        # Every eligible trace, cube-capable or not: a shared bin domain is a
        # property of the *figure's* trace set, so a sibling that is not itself
        # a target still widens the domain. The grouping (and the only
        # trace-type knowledge) lives in ``_histogram_domain_cols_by_uid``,
        # shared with the aggregation path.
        domain_group_candidates: list[tuple[TraceInfo, FlexTrace, bool]] = []
        for ti in trace_infos:
            if ti.figure_uid == active_source.figure_uid:
                continue
            if ti.figure_uid in owning_figures:
                continue
            trace = self._scalable_traces.get(ti.uid)
            if trace is None:
                continue
            anchor = trace.recompute_axes[0] if trace.recompute_axes else None
            # The bound column per anchor (for temporal viewport parsing):
            # backend_data maps prop keys (x/y) to columns for binned targets.
            backend = getattr(trace, "_backend_data", None) or {}

            def _axis_range(axis: str | None) -> tuple[float, float] | None:
                col = backend.get(axis)
                return self._cube_axis_range(
                    viewports_by_figure,
                    ti.figure_uid,
                    axis,
                    schema=schema,
                    column=col if isinstance(col, str) else None,
                )

            # A multi-axis target (e.g. hist2d, recompute_axes=(x, y)) is keyed
            # by the engine only on its anchor; a zoom on any OTHER recompute
            # axis is invisible to ``get_cube_target_spec``, so gate it here. A
            # cube can only serve such a target when every recompute axis is at
            # full data range (the anchor's own range gates inside the trace).
            if any(_axis_range(ax) is not None for ax in trace.recompute_axes[1:]):
                continue
            anchor_range = _axis_range(anchor)
            domain_group_candidates.append((ti, trace, anchor_range is not None))
            target = trace.get_cube_target_spec(anchor_range, schema=schema)
            if target is not None:
                targets.append((ti, trace, target))
        if not targets:
            return [], {}

        # Sibling histograms on one figure share a bin domain in the legacy
        # aggregation path; the cube fg layer must use the exact same union or
        # it lands on different bin edges than the bg layer it overlays. A
        # sibling that is not itself a target still contributes its min/max, so
        # its column has to be resolved alongside the target columns.
        shared_domain_cols = self._histogram_domain_cols_by_uid(
            domain_group_candidates, schema=schema
        )

        # Only groups containing an actual target need resolving: a figure of
        # histograms none of which is cube-servable would otherwise pull its
        # columns through ``physical_minmax`` for nothing.
        free, domains = self._resolve_cube_domains(
            free_spec,
            [t for _, _, t in targets],
            schema,
            extra_cols={
                col
                for ti, _, _ in targets
                for col in shared_domain_cols.get(ti.uid, ())
            },
        )
        if free is None:
            return [], {}

        # Dedup by content key; build/fetch each distinct cube exactly once.
        cubes: List[bytes] = []
        index_by_key: Dict[str, int] = {}
        trace_cubes: Dict[str, int] = {}
        for ti, trace, target in targets:
            uid = ti.uid
            # A line_env/corr target builds against a range OR categorical free
            # axis; only a box2d (hist2d) source skips it (it falls back to the
            # per-commit recompute, #47) rather than 500-ing the whole
            # live-brush request.
            if not cube_target_buildable(free, target.measure):
                continue
            target_domains = _union_shared_domains(domains, shared_domain_cols.get(uid))
            if target_domains is None:
                continue  # sibling domain unresolvable — don't serve misaligned bins
            dims = self._resolved_target_dims(
                target.target_dims, target_domains, schema
            )
            if dims is None:
                continue
            spec = CubeSpec(
                source_name=self._source_name,
                free=free,
                target_dims=dims,
                measure=target.measure,
                passive_key=passive_key,
            )
            key = cube_content_key(spec)
            idx = index_by_key.get(key)
            if idx is None:
                blob = cube_cache.get(key) if cube_cache is not None else None
                if blob is None:
                    t_start = time.perf_counter()
                    result = build_cube(build_ldf, spec)
                    blob = encode_fvcube(result, cube_id=key)
                    if cube_cache is not None:
                        cube_cache.set(key, blob)
                    logger.info(
                        "Cube build %s: %d cells, %.4fs",
                        key[:8],
                        result.n_cells,
                        time.perf_counter() - t_start,
                    )
                idx = len(cubes)
                cubes.append(blob)
                index_by_key[key] = idx
            trace_cubes[uid] = idx
        return cubes, trace_cubes

    def _locate_free_axis(
        self,
        trace_infos: List[TraceInfo],
        viewports_by_figure: Dict[str, Dict[str, Any]],
        active_source: ActiveSource,
        schema: pl.Schema | None,
    ) -> FreeAxisSpec | None:
        """The source trace named by ``active_source.trace_uid``, validated:
        it must belong to the active figure and its cube-source primary free
        column must equal ``active_source.column``; ``None`` on any mismatch
        (silent empty response). Resolving by uid removes first-match
        ambiguity when two source traces in one figure share a primary column
        (e.g. bar(cat) + treemap(cat, sub)). For categorical sources the
        primary column is ``columns[0]`` — bar/pie's first label col,
        treemap's ``path[0]``."""
        for ti in trace_infos:
            if ti.uid != active_source.trace_uid:
                continue
            if ti.figure_uid != active_source.figure_uid:
                return None
            trace = self._scalable_traces.get(ti.uid)
            if trace is None:
                return None
            anchor = trace.select_axes[0] if trace.select_axes else None
            candidate = trace.get_cube_source_spec(
                self._cube_axis_range(
                    viewports_by_figure,
                    ti.figure_uid,
                    anchor,
                    schema=schema,
                    column=active_source.column,
                ),
                schema=schema,
            )
            if candidate is not None and candidate.column == active_source.column:
                if candidate.kind == "temporal":
                    # Contract G: the unit comes from the schema dtype;
                    # Datetime("ns")/Time are unsupported (string round-trip
                    # is µs-precision) — no cube at all.
                    unit = temporal_unit(_dtype_for_col(schema, candidate.column))
                    if unit is None:
                        return None
                    candidate = replace(candidate, unit=unit)
                elif candidate.kind == "box2d":
                    candidate = self._locate_box2d_axis(
                        candidate, ti, viewports_by_figure, schema
                    )
                return candidate
            return None
        return None

    def _locate_box2d_axis(
        self,
        candidate: FreeAxisSpec,
        ti: TraceInfo,
        viewports_by_figure: Dict[str, Dict[str, Any]],
        schema: pl.Schema | None,
    ) -> FreeAxisSpec | None:
        """Resolve a box2d (hist2d) free axis's per-axis units and viewports
        (contract H). ``active_source.column`` is the x column (validated in
        ``_locate_free_axis``); the trace's two select anchors map to its two
        columns. Each temporal axis takes its physical unit from the schema
        dtype (Datetime("ns")/Time gate to no cube); each axis's viewport range
        is resolved independently (``None`` = unzoomed → engine-resolved full
        domain). Returns ``None`` to gate the whole cube."""
        trace = self._scalable_traces.get(ti.uid)
        if trace is None or len(candidate.columns or ()) != 2:
            return None
        cx, cy = candidate.columns  # type: ignore[misc]
        # The two select anchors, in (x, y) column order. select_axes is
        # (x_anchor, y_anchor) for hist2d; axes[0] is x, axes[1] is y.
        anchors = trace.select_axes
        if len(anchors) < 2:
            return None
        anchor_x, anchor_y = anchors[0], anchors[1]

        units: list[str | None] = []
        for col in (cx, cy):
            dtype = _dtype_for_col(schema, col)
            if dtype is not None and dtype.is_temporal():
                unit = temporal_unit(dtype)
                if unit is None:
                    return None  # ns/Time gate
                units.append(unit)
            else:
                units.append(None)

        dom_x = self._cube_axis_range(
            viewports_by_figure, ti.figure_uid, anchor_x, schema=schema, column=cx
        )
        dom_y = self._cube_axis_range(
            viewports_by_figure, ti.figure_uid, anchor_y, schema=schema, column=cy
        )
        # Per-axis None viewports are resolved to the full data domain in
        # _resolve_cube_domains; carry a partial (x-zoom only / y-zoom only) as
        # a domains tuple with one resolved + one None axis.
        domains: tuple | None
        if dom_x is None and dom_y is None:
            domains = None
        else:
            domains = (dom_x, dom_y)
        new_unit = (units[0], units[1]) if any(units) else None
        return replace(candidate, unit=new_unit, domains=domains)

    @staticmethod
    def _cube_axis_range(
        viewports_by_figure: Dict[str, Dict[str, Any]],
        figure_uid: str | None,
        anchor: str | None,
        schema: pl.Schema | None = None,
        column: str | None = None,
    ) -> tuple[float, float] | None:
        """A figure's viewport range on one anchor as physical floats, or
        None (unzoomed).

        Reads the same per-figure viewport mapping that ``process`` consumes
        for ``update_range``. Temporal viewports arrive as date strings —
        with ``schema`` and ``column`` given they are parsed into the
        column's physical representation (contract G; previously they
        silently resolved to the full domain — wrong for a zoomed temporal
        source). Other non-numeric values (map coordinates, autorange
        ``None``) yield ``None`` — i.e. the full data domain.
        """
        if figure_uid is None or anchor is None:
            return None
        rng = (viewports_by_figure.get(figure_uid) or {}).get(anchor)
        if not (isinstance(rng, (tuple, list)) and len(rng) == 2):
            return None
        if all(isinstance(v, (int, float)) for v in rng):
            return (float(rng[0]), float(rng[1]))
        if column is not None and all(isinstance(v, str) for v in rng):
            dtype = _dtype_for_col(schema, column)
            if dtype is not None and dtype.is_temporal():
                try:
                    lo, hi = (
                        pl.select(
                            _typed_temporal_lit(v, dtype).to_physical().cast(pl.Float64)
                        ).item()
                        for v in rng
                    )
                except Exception:
                    return None
                if lo is not None and hi is not None:
                    return (float(lo), float(hi))
        return None

    def _resolve_cube_domains(
        self,
        free_spec: FreeAxisSpec,
        target_specs: list[CubeTargetSpec],
        schema: pl.Schema | None,
        extra_cols: set[str] | None = None,
    ) -> tuple[FreeAxisSpec | None, Dict[str, tuple[float, float]]]:
        """Resolve ``domain=None`` (= full data domain) to concrete floats.

        One batched min/max ``select`` over the **unfiltered** LazyFrame covers
        the free axis and every unresolved binned target column (the same
        unfiltered-domain rule the aggregation path uses; the builder itself is
        never mutated). The free axis gets the min/max verbatim — the binned-dim
        epsilon is applied later, uniformly, in ``_resolved_target_dims``.

        A **categorical** free axis (bar/pie/treemap source) is not binned and
        takes no domain: free-domain resolution is skipped entirely (``domain``
        stays ``None``, per ``FreeAxisSpec``'s contract); binned target dims
        still resolve exactly as for range sources.

        The min/max lookups go through ``LFQueryBuilder.physical_minmax``, which
        **memoizes** each column's physical ``(min, max)`` for the source's
        lifetime — so a cube *cache hit* no longer re-scans the full data just
        to re-derive the (cache-key-determining) domain.
        """
        is_box2d = free_spec.kind == "box2d"
        resolve_free = (
            free_spec.kind not in ("categorical", "box2d") and free_spec.domain is None
        )

        # box2d (contract H): each axis's viewport is either already resolved or
        # None (full data domain). Collect every None axis's column.
        box2d_axes: list[tuple[int, str]] = []
        if is_box2d:
            cols2 = free_spec.columns or ()
            cur = free_spec.domains or (None, None)
            for axis, (col, dom) in enumerate(zip(cols2, cur)):
                if dom is None:
                    box2d_axes.append((axis, col))

        # ``extra_cols`` are columns no target binned dim asks for but a shared
        # domain group still unions over (a sibling that is not itself a cube
        # target); without them the group is unresolvable and the target gets
        # dropped instead of widened.
        unresolved_cols = sorted(
            {
                d.column
                for target in target_specs
                for d in target.target_dims
                if d.kind == "binned" and d.domain is None
            }
            | (extra_cols or set())
        )

        # One memoized batched min/max over every column we still need. The
        # physical (temporal→to_physical) reduction is identical across the
        # free axis, box2d axes, and binned target dims, so a single per-column
        # (min, max) serves all three roles.
        needed: list[str] = []
        if resolve_free:
            needed.append(free_spec.column)
        needed += [col for _, col in box2d_axes]
        needed += unresolved_cols
        # Cubes are built only for ``cache=True`` sources, so the static-data
        # contract that ``memoize`` requires already holds.
        minmax = (
            self._backend_lf.physical_minmax(needed, schema, memoize=True)
            if needed
            else {}
        )

        free = free_spec
        if resolve_free:
            lo, hi = minmax[free_spec.column]
            if lo is None or hi is None:  # empty / all-null free column
                return None, {}
            free = replace(free, domain=(float(lo), float(hi)))

        if is_box2d:
            cur = list(free_spec.domains or (None, None))
            for axis, col in box2d_axes:
                lo, hi = minmax[col]
                if lo is None or hi is None:  # empty / all-null axis column
                    return None, {}
                cur[axis] = (float(lo), float(hi))
            if any(d is None for d in cur):  # both axes must be resolved now
                return None, {}
            free = replace(free, domains=(cur[0], cur[1]))

        domains: Dict[str, tuple[float, float]] = {}
        for col in unresolved_cols:
            lo, hi = minmax[col]
            if lo is not None and hi is not None:
                domains[col] = (float(lo), float(hi))
        return free, domains

    def _resolved_target_dims(
        self,
        target_dims: tuple,
        domains: Dict[str, tuple[float, float]],
        schema: pl.Schema | None = None,
    ) -> tuple | None:
        """Resolve binned target dims and epsilon-pad their upper bounds.

        ``_HIST_BIN_EPSILON`` is added **uniformly** to every ``hist1d``-variant
        binned dim's resolved-or-zoomed domain — mirroring
        ``_histogram_bounds_exprs``, which pads in both the unzoomed and zoomed
        cases — so cube bins align with display bins. A ``hist2d``-variant dim
        (contract K) is NOT padded: the ``fixed_hist2d`` kernel's own ``+1e-10``
        span epsilon already provides the boundary tolerance, and the cube
        ``_fixed_hist2d_bin_expr`` reproduces it; padding here too would shift
        bins and break the bit-equality with the server delta. Returns ``None``
        when a dim's domain cannot be resolved (all-null column): that target is
        simply not served.

        Temporal binned dims gain their physical ``unit`` from the schema
        dtype (contract G); an unsupported temporal dtype (``Datetime("ns")``,
        ``Time``) means the target is not served.
        """
        dims = []
        for d in target_dims:
            if d.kind != "binned":
                dims.append(d)
                continue
            domain = d.domain if d.domain is not None else domains.get(d.column)
            if domain is None:
                return None
            unit = d.unit
            dtype = _dtype_for_col(schema, d.column)
            if dtype is not None and dtype.is_temporal():
                unit = temporal_unit(dtype)
                if unit is None:
                    return None  # ns/Time gate
            pad = 0.0 if d.bin_variant == "hist2d" else _HIST_BIN_EPSILON
            dims.append(replace(d, domain=(domain[0], domain[1] + pad), unit=unit))
        return tuple(dims)

    def _active_selections(self, event: InteractionEvent) -> List[Any]:
        if event.type in ("deselect", "reset", "init"):
            return []
        return list(event.selections)

    def _selection_filter_exprs(
        self,
        active_selections: List[Any],
        backend_schema: pl.Schema | None,
        event: InteractionEvent,
    ) -> List[pl.Expr]:
        """Build filter expressions from non-self selections' predicates.

        Each non-self selection contributes one Polars expression (its
        predicates ORed together).  All such expressions are returned
        in a list — the caller passes them as positional args to
        ``LazyFrame.filter(*exprs)`` which ANDs them.
        """
        if not active_selections or self._backend_lf is None:
            return []

        ignore_source_uid = event.figure_uid if event.type == "viewport" else None

        filter_exprs: List[pl.Expr] = []
        for sel in active_selections:
            if sel.source_figure_uid is None:
                continue
            if sel.source_figure_uid == ignore_source_uid:
                continue
            if not sel.predicates:
                continue
            filter_exprs.append(predicates_to_expr(sel.predicates, backend_schema))
        return filter_exprs

    def _should_process_trace(
        self,
        event: InteractionEvent,
        trace_info: TraceInfo,
        selection_fig_uids: set[str | None],
        viewports_by_figure: Dict[str, Dict[str, Any]],
    ) -> bool:
        if selection_fig_uids and trace_info.figure_uid in selection_fig_uids:
            if event.type == "selection" or trace_info.figure_uid != event.figure_uid:
                return False

        scalable = self._scalable_traces[trace_info.uid]
        changed_axes = self._trace_event_ranges(
            event=event,
            figure_uid=trace_info.figure_uid,
            viewports_by_figure=viewports_by_figure,
        )
        # A viewport change only re-aggregates this trace if it moved one of the
        # trace's data-binding axes. ``recompute_axes`` is anchor-space and
        # unifies cartesian (``x``/``y2``/…) and map (``coordinates``) traces.
        needs_zoom = bool(set(scalable.recompute_axes) & set(changed_axes))
        return bool(event.force_update or needs_zoom)

    def _trace_event_ranges(
        self,
        event: InteractionEvent,
        figure_uid: str | None,
        viewports_by_figure: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        if event.type == "viewport":
            return _normalize_axis_ranges(event.axis_ranges)
        if figure_uid is None:
            return {}
        return dict(viewports_by_figure.get(figure_uid, {}))

    def _trace_update_range(
        self,
        event: InteractionEvent,
        trace_info: TraceInfo,
        viewports_by_figure: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        if event.type == "reset":
            return {}
        update_range = self._trace_event_ranges(
            event=event,
            figure_uid=trace_info.figure_uid,
            viewports_by_figure=viewports_by_figure,
        )
        # Hand the trace only the ranges for axes it actually aggregates on, so
        # a non-binding axis (e.g. a line's y) is never fed into its agg spec.
        binding = self._scalable_traces[trace_info.uid].recompute_axes
        return {ax: update_range[ax] for ax in binding if ax in update_range}

    def _resolve_domains(
        self,
        aggregation_traces: list[_AggregationTrace],
        histogram_domains: Dict[str, tuple[str, ...]],
        schema: pl.Schema | None,
        line_x_cols: set[str],
    ) -> tuple[Dict[str, tuple[str, ...]], Dict[str, tuple[Any, Any]]]:
        """Resolve every unfiltered ``(min, max)`` this request needs, in one collect.

        Bin edges must not move when a cross-filter narrows the data, so the
        bounds come from the unfiltered frame. Each trace states its own columns
        (``FlexTrace.domain_cols``); an unzoomed histogram instead takes the
        union its same-figure siblings share, so their bars line up.

        Memoized only for a cached source, whose data the cache contract already
        treats as static — an uncached reset must be able to see changed data.

        ``line_x_cols`` carries the scan-source x columns whose null/NaN
        contract this same collect checks (see ``_check_line_x``).
        """
        if self._backend_lf is None:
            return {}, {}
        domain_cols = {
            item.info.uid: histogram_domains.get(item.info.uid)
            or item.trace.domain_cols(
                item.update_range, scan_source=self._backend_lf.is_scan
            )
            for item in aggregation_traces
        }
        needed = sorted({col for cols in domain_cols.values() for col in cols})
        if not needed:
            return domain_cols, {}
        return domain_cols, self._backend_lf.physical_minmax(
            needed, schema, memoize=self._cache is not None, contract_cols=line_x_cols
        )

    def _check_line_x(self, aggregation_traces: list[_AggregationTrace]) -> set[str]:
        """Validate the x column of every ungrouped minmax line.

        Those buckets are equal in x width, so a well-formed x is a contract,
        not a hint. A resident source pays the full check once per source and
        column (``assume_sorted_x=True`` skips it). A scan runs an
        order-independent plan, so its order is never checked; its dtype is
        gated here (free, off the schema) and its null/NaN contract rides the
        domain probe, which is why those columns are returned. A frame sorted by
        group then x is not globally sorted, so a grouped line is not checked.

        Known limit: a zoomed request resolves no domain, so on a scan it never
        reaches the null/NaN check. The first request of a figure is unzoomed,
        so a dirty x is still caught once.
        """
        lf = self._backend_lf
        scan_x_cols: set[str] = set()
        if lf is None:
            return scan_x_cols
        for item in aggregation_traces:
            trace = item.trace
            if not (
                trace.trace_type == "line"
                and trace.group_by_cols is None
                and trace.downsample == "minmax"
            ):
                continue
            if not lf.is_scan:
                lf.check_line_x(trace.x_col)
                continue
            if (dtype := lf.schema.get(trace.x_col)) is not None:
                check_line_x_dtype(trace.x_col, dtype)
                scan_x_cols.add(trace.x_col)
        return scan_x_cols

    def _collect_aggregation_specs(
        self,
        aggregation_traces: list[_AggregationTrace],
        backend_schema: pl.Schema | None,
        domain_cols: Dict[str, tuple[str, ...]],
        domains: Dict[str, tuple[Any, Any]],
    ) -> List[AggregationSpec | GroupedAggregationSpec]:
        agg_specs: List[AggregationSpec | GroupedAggregationSpec] = []

        for item in aggregation_traces:
            ti = item.info
            trace = item.trace
            t_s = time.perf_counter()
            trace_domains = {c: domains[c] for c in domain_cols.get(ti.uid, ())}

            if trace.trace_type in ("histogram", "histogram2d"):
                agg_specs.append(
                    trace.get_aggregation_spec(
                        update_range=item.update_range,
                        schema=backend_schema,
                        domains=trace_domains,
                    )
                )
            elif trace.trace_type == "line":
                # TODO: make sorted the default behavior for line traces
                #       -> I do not like the specific branch here for line traces
                lf = self._backend_lf
                is_scan = lf is not None and lf.is_scan
                # Sorted x turns the viewport into a contiguous row range, which
                # the line trace can slice instead of mask.
                agg_specs.append(
                    trace.get_aggregation_spec(
                        update_range=item.update_range,
                        schema=backend_schema,
                        x_sorted=(lf is not None and lf.is_sorted(trace.x_col)),
                        scan_source=is_scan,
                        domains=trace_domains,
                    )
                )
            else:
                agg_specs.append(
                    trace.get_aggregation_spec(
                        update_range=item.update_range,
                        schema=backend_schema,
                    )
                )

            t_e = time.perf_counter()
            logger.info(
                f"[{ti.uid[:8]}] {ti.trace_type:10s} - agg_spec: {t_e - t_s:.4f}s"
            )
        return agg_specs

    # -- cache (Phase 1) -------------------------------------------------------

    def _cache_active(self, event: InteractionEvent) -> bool:
        """Whether the per-trace cache applies to this request.

        Only the unfiltered ``init``/``reset``/``deselect`` computation is
        cached in Phase 1, and only when the engine was given a cache backend
        for a source that opted in.
        """
        return (
            self._cache is not None
            and self._source_name is not None
            and event.type in _CACHEABLE_EVENT_TYPES
        )

    def _content_key(
        self, item: _AggregationTrace, domain_cols: tuple[str, ...] | None
    ) -> str:
        trace = item.trace
        return content_key(
            source_name=self._source_name,
            trace_type=trace.trace_type,
            axes=item.info.axes,
            backend_data=trace._backend_data,
            params=trace._params,
            domain_cols=domain_cols,
        )

    def _cacheable_items(
        self,
        event: InteractionEvent,
        aggregation_traces: list[_AggregationTrace],
        histogram_domains: Dict[str, tuple[str, ...]],
    ) -> list[tuple[_AggregationTrace, str]]:
        """Return ``(item, key)`` pairs for traces whose delta is cacheable.

        A trace is cacheable only when it is **viewport-free** (empty
        ``update_range``): the content key is viewport-blind, so a zoomed trace
        must never be stored or served (it would alias the full-range entry).

        Because viewport-dependent traces are dropped here, the returned list
        can be a strict subset of the delta-producing set; the caller compares
        its length against the full trace count before short-circuiting, so a
        zoomed trace forces a normal recompute instead of being lost.
        """
        if not self._cache_active(event):
            return []
        items: list[tuple[_AggregationTrace, str]] = []
        for item in aggregation_traces:
            if item.update_range:  # viewport-dependent — not content-addressable
                continue
            items.append(
                (item, self._content_key(item, histogram_domains.get(item.info.uid)))
            )
        return items

    def _cache_payload(self, delta: TraceDelta) -> Dict[str, Any]:
        """Reduce a ``TraceDelta`` to a uid-agnostic, JSON-safe payload.

        Grouped payloads keep ``group_value_key`` (not the resolved child uid)
        so a cache hit can re-stamp child uids for the requesting parent.
        """
        if delta.group_results is not None:
            return {
                "group_results": [
                    {
                        "group_value_key": cr.group_value_key,
                        "updates": cr.updates,
                    }
                    for cr in delta.group_results
                ]
            }
        return {"updates": delta.updates}

    def _delta_from_cached(
        self,
        item: _AggregationTrace,
        payload: Dict[str, Any],
        layer: Literal["bg", "fg"] | None,
    ) -> TraceDelta:
        uid = item.info.uid
        if "group_results" in payload:
            child_deltas = [
                GroupedChildDelta(
                    uid=child_uid_from_group_key(uid, entry["group_value_key"]),
                    updates=entry["updates"],
                    parent_uid=uid,
                    group_value_key=entry["group_value_key"],
                )
                for entry in payload["group_results"]
            ]
            return TraceDelta(
                uid=uid, updates={}, group_results=child_deltas, layer=layer
            )
        return TraceDelta(uid=uid, updates=payload["updates"], layer=layer)

    def _store_in_cache(
        self,
        cache_items: list[tuple[_AggregationTrace, str]],
        deltas: List[TraceDelta],
    ) -> None:
        key_by_uid = {item.info.uid: key for item, key in cache_items}
        for delta in deltas:
            key = key_by_uid.get(delta.uid)
            if key is not None:
                self._cache.set(key, self._cache_payload(delta))

    def _aggregation_traces(
        self,
        event: InteractionEvent,
        trace_infos: List[TraceInfo],
        selection_fig_uids: set[str | None],
        viewports_by_figure: Dict[str, Dict[str, Any]],
    ) -> list[_AggregationTrace]:
        aggregation_traces: list[_AggregationTrace] = []
        for ti in trace_infos:
            if not self._should_process_trace(
                event=event,
                trace_info=ti,
                selection_fig_uids=selection_fig_uids,
                viewports_by_figure=viewports_by_figure,
            ):
                continue

            aggregation_traces.append(
                _AggregationTrace(
                    info=ti,
                    trace=self._scalable_traces[ti.uid],
                    update_range=self._trace_update_range(
                        event=event,
                        trace_info=ti,
                        viewports_by_figure=viewports_by_figure,
                    ),
                )
            )
        return aggregation_traces

    def _histogram_domain_cols_by_uid(
        self,
        items: Iterable[tuple[TraceInfo, FlexTrace, bool]],
        schema: pl.Schema | None,
    ) -> Dict[str, tuple[str, ...]]:
        """Map each unzoomed histogram uid → the sibling columns it bins over.

        Same figure + same axes + same data axis + compatible coordinate unit ⇒
        one shared min/max domain, so sibling histograms' bars line up. Both the
        aggregation and cube paths use this grouping; native temporal physical
        units (days/ms/us/ns) must never be unioned with each other or numerics.
        """
        _GroupKey = tuple[str | None, tuple[str, ...] | None, str, str | None]
        groups: Dict[_GroupKey, list[str]] = {}
        uid_to_group: Dict[str, _GroupKey] = {}

        for ti, trace, data_axis_zoomed in items:
            if trace.trace_type != "histogram" or data_axis_zoomed:
                continue

            data_axis = trace.prop_key
            data_col = trace.data_col
            dtype = _dtype_for_col(schema, data_col)
            coordinate_unit = None
            if dtype is not None and dtype.is_temporal():
                coordinate_unit = temporal_unit(dtype) or str(dtype)

            axes_key = tuple(ti.axes) if ti.axes else None
            group_key = (ti.figure_uid, axes_key, data_axis, coordinate_unit)
            domain_cols = groups.setdefault(group_key, [])
            if data_col not in domain_cols:
                domain_cols.append(data_col)
            uid_to_group[ti.uid] = group_key

        return {
            uid: tuple(groups[group_key]) for uid, group_key in uid_to_group.items()
        }

    def _normalise_updates(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        return {
            k: v.to_list() if isinstance(v, pl.Series) else v
            for k, v in updates.items()
        }

    def _to_trace_delta(
        self,
        uid: str,
        result: Any,
        layer: Literal["bg", "fg"] | None = None,
    ) -> TraceDelta:
        if result.group_results is not None:
            child_deltas = [
                GroupedChildDelta(
                    uid=cr.child_uid,
                    updates=self._normalise_updates(cr.updates),
                    parent_uid=uid,
                    group_value_key=cr.group_value_key,
                )
                for cr in result.group_results
            ]
            return TraceDelta(
                uid=uid,
                updates={},
                group_results=child_deltas,
                layer=layer,
            )
        return TraceDelta(
            uid=uid,
            updates=self._normalise_updates(result.updates),
            layer=layer,
        )

    def _process_update_mode(
        self,
        agg_specs: List[AggregationSpec | GroupedAggregationSpec],
        filter_exprs: List[pl.Expr],
        trace_infos: List[TraceInfo],
        event: InteractionEvent,
        selection_fig_uids: set[str | None],
        viewports_by_figure: Dict[str, Dict[str, Any]],
    ) -> List[TraceDelta]:
        t_start = time.perf_counter()
        df_agg, grouped_dfs = self._backend_lf.aggregate(filter_exprs, agg_specs)
        t_end = time.perf_counter()
        logger.info(f"Aggregate time: {t_end - t_start:.4f}s")
        t_build_start = time.perf_counter()
        deltas = self._build_deltas(
            event=event,
            trace_infos=trace_infos,
            selection_fig_uids=selection_fig_uids,
            viewports_by_figure=viewports_by_figure,
            df_agg=df_agg,
            grouped_dfs=grouped_dfs,
        )
        t_build_end = time.perf_counter()
        logger.info(f"Build deltas time: {t_build_end - t_build_start:.4f}s")
        return deltas

    def _overlay_layers_for_event(
        self,
        event: InteractionEvent,
        has_active_selections: bool,
    ) -> tuple[Literal["bg", "fg"], ...]:
        if event.type == "selection":
            return ("fg",)
        if event.type in ("init", "deselect", "reset"):
            return ("bg",)
        if event.type == "viewport":
            return ("bg", "fg") if has_active_selections else ("bg",)
        raise ValueError(f"Unsupported interaction event type: {event.type!r}")

    def _specs_for_layer(
        self,
        specs: List[AggregationSpec | GroupedAggregationSpec],
        layer: Literal["bg", "fg"],
        has_active_selections: bool,
    ) -> List[AggregationSpec | GroupedAggregationSpec]:
        """Filter aggregation specs based on per-trace overlay policy.

        Every trace needs the sole unfiltered layer used by init/reset/deselect.
        With an active filtered foreground, ``filtered_only`` traces reuse that
        cached unfiltered data instead of recomputing a duplicate background.
        """
        if layer == "fg" or not has_active_selections:
            return specs
        return [
            s
            for s in specs
            if getattr(
                self._scalable_traces.get(s.uid, None),
                "overlay_style",
                "full",
            )
            != "filtered_only"
        ]

    def _process_overlay_mode(
        self,
        agg_specs: List[AggregationSpec | GroupedAggregationSpec],
        filter_exprs: List[pl.Expr],
        has_active_selections: bool,
        trace_infos: List[TraceInfo],
        event: InteractionEvent,
        selection_fig_uids: set[str | None],
        viewports_by_figure: Dict[str, Dict[str, Any]],
    ) -> List[TraceDelta]:
        requested_layers = self._overlay_layers_for_event(
            event=event,
            has_active_selections=has_active_selections,
        )
        deltas: List[TraceDelta] = []
        for layer in requested_layers:
            if layer == "fg" and not filter_exprs:
                continue
            layer_specs = self._specs_for_layer(agg_specs, layer, has_active_selections)
            if not layer_specs:
                continue
            t_start = time.perf_counter()
            layer_filters = [] if layer == "bg" else filter_exprs
            df_agg, grouped_dfs = self._backend_lf.aggregate(layer_filters, layer_specs)
            t_end = time.perf_counter()
            logger.info("Overlay aggregate time (%s): %.4fs", layer, t_end - t_start)
            t_build_start = time.perf_counter()
            layer_deltas = self._build_deltas(
                event=event,
                trace_infos=trace_infos,
                selection_fig_uids=selection_fig_uids,
                viewports_by_figure=viewports_by_figure,
                df_agg=df_agg,
                grouped_dfs=grouped_dfs,
                layer=layer,
            )
            t_build_end = time.perf_counter()
            logger.info(
                "Overlay build deltas time (%s): %.4fs",
                layer,
                t_build_end - t_build_start,
            )
            deltas.extend(layer_deltas)
        return deltas

    def _build_deltas(
        self,
        event: InteractionEvent,
        trace_infos: List[TraceInfo],
        selection_fig_uids: set[str | None],
        viewports_by_figure: Dict[str, Dict[str, Any]],
        df_agg: pl.DataFrame,
        grouped_dfs: Dict[str, pl.DataFrame],
        layer: Literal["bg", "fg"] | None = None,
    ) -> List[TraceDelta]:
        deltas: List[TraceDelta] = []
        for ti in trace_infos:
            if not self._should_process_trace(
                event=event,
                trace_info=ti,
                selection_fig_uids=selection_fig_uids,
                viewports_by_figure=viewports_by_figure,
            ):
                continue

            trace = self._scalable_traces[ti.uid]
            if ti.uid in grouped_dfs:
                deltas.append(
                    self._to_trace_delta(
                        ti.uid,
                        trace._to_grouped_update(grouped_dfs[ti.uid]),
                        layer=layer,
                    )
                )
            elif ti.uid in df_agg.columns:
                deltas.append(
                    self._to_trace_delta(ti.uid, trace._to_update(df_agg), layer=layer)
                )
        return deltas

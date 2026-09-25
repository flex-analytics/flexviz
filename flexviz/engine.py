"""Core aggregation engine, decoupled from any rendering or transport layer."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any, Literal

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
from .LF import AggregationSpec, GroupedAggregationSpec, LFQueryBuilder
from .predicates import canonical_passive_key, predicates_to_expr
from .spec import SelectionState
from .trace.base import (
    FlexTrace,
    _dtype_for_col,
    _physical_bound_expr,
    child_uid_from_group_key,
)
from .trace.hist import _HIST_BIN_EPSILON

# Event types whose computation is unfiltered and viewport-free. These are the
# only events cached in Phase 1: the engine forces them to drop all
# selections (`_active_selections`), so the result is the unfiltered base — a
# float-free, content-addressable computation shared across reloads/viewers.
_CACHEABLE_EVENT_TYPES = frozenset({"init", "deselect"})

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TraceInfo:
    """Lightweight description of a trace's identity and axes."""

    uid: str
    axes: tuple | None  # (x_anchor, y_anchor) for cartesian
    trace_type: str
    figure_uid: str  # dashboard figure uid


@dataclass(frozen=True)
class _AggregationTrace:
    info: TraceInfo
    trace: FlexTrace
    update_range: dict[str, Any]


def _union_shared_domains(
    domains: dict[str, tuple[float, float]],
    sibling_cols: tuple[str, ...] | None,
) -> dict[str, tuple[float, float]] | None:
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


def _changed_axes_by_figure(viewport_keys: list[str]) -> dict[str, set[str]]:
    """Split ``"<figure_uid>/<axis_id>"`` state keys into axis ids per figure."""
    changed: dict[str, set[str]] = {}
    for key in viewport_keys:
        figure_uid, _, axis_id = key.partition("/")
        changed.setdefault(figure_uid, set()).add(axis_id)
    return changed


@dataclass(frozen=True)
class _Partition:
    """Traces that share one selection filter set.

    ``owner`` is the figure whose own selection is left out of ``filter_exprs``
    (a figure is never filtered by its own selection), or ``None`` for traces
    of figures that own no active selection.
    """

    owner: str | None
    items: list[_AggregationTrace]
    filter_exprs: list[pl.Expr]


def _normalize_axis_ranges(ranges: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``ranges`` with every descending ``(lo, hi)`` swapped.

    A reversed plotly axis reports its viewport high-to-low, and clients
    forward it verbatim. Every consumer downstream assumes ``lo <= hi`` —
    ``is_between`` masks return nothing and the sorted-x ``search_sorted``
    slice returns the wrong rows — so order is normalized once, here at
    ingestion. Non-range values (``None``, map coordinate point lists) pass
    through untouched, as does any pair that does not compare.
    """
    out: dict[str, Any] = {}
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
        scalable_traces: dict[str, FlexTrace],
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
        trace_infos: list[TraceInfo],
        viewports_by_figure: dict[str, dict[str, Any]] | None = None,
        cross_filter_mode: Literal["update", "overlay"] = "update",
    ) -> list[TraceDelta]:
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
        changed_axes = _changed_axes_by_figure(event.viewport_keys)
        active_selections = self._active_selections(event)
        selection_fig_uids = {
            sel.source_figure_uid
            for sel in active_selections
            if sel.source_figure_uid is not None
        }

        aggregation_traces = self._aggregation_traces(
            event=event,
            trace_infos=trace_infos,
            selection_fig_uids=selection_fig_uids,
            changed_axes=changed_axes,
            viewports_by_figure=viewports_by_figure,
            cross_filter_mode=cross_filter_mode,
        )

        # Sibling histograms on one figure bin over a shared domain, so this
        # grouping feeds both the aggregation specs and the cache key (an
        # entry computed for a different sibling set is a different result).
        # ``data_axis_zoomed`` is "has a real viewport range", not merely
        # "key present": a ``None`` state value is unzoomed and must NOT drop
        # the histogram from its shared-domain group — else siblings fall back
        # to per-column domains and misalign. Mirrors the cube caller's
        # ``anchor_range is not None`` and ``get_aggregation_spec`` itself,
        # which treat ``None`` as unzoomed.
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

        # Before the domain scan: on a static source the line check leaves the
        # x column flagged sorted, which makes the min/max collect O(1).
        if self._backend_lf is not None:
            for item in aggregation_traces:
                item.trace.check_source(self._backend_lf)

        partitions = self._partitions(
            aggregation_traces, active_selections, selection_fig_uids, backend_schema
        )

        # Resolved only here, past the fast-path: a fully cached request needs
        # no min/max scan at all. Both overlay layers reuse these specs, so the
        # mode decides the scope: in overlay mode every column resolves
        # unfiltered, because the background layer pins the axis.
        domains_by_uid = self._resolve_domains(
            partitions,
            histogram_domains,
            backend_schema,
            cross_filter_mode=cross_filter_mode,
        )

        if not partitions:
            return []

        t_agg_start = time.perf_counter()
        specs_by_partition = [
            (
                partition,
                self._collect_aggregation_specs(
                    aggregation_traces=partition.items,
                    backend_schema=backend_schema,
                    domains_by_uid=domains_by_uid,
                ),
            )
            for partition in partitions
        ]
        t_agg_end = time.perf_counter()
        logger.info(f"Total get agg_spec time: {t_agg_end - t_agg_start:.4f}s")

        t_start = time.perf_counter()
        if cross_filter_mode == "overlay":
            deltas = self._process_overlay_mode(
                specs_by_partition,
                has_active_selections=bool(active_selections),
                event=event,
                changed_axes=changed_axes,
            )
        else:
            deltas = [
                delta
                for partition, part_specs in specs_by_partition
                for delta in self._aggregate_layer(
                    partition.filter_exprs, part_specs, partition.items
                )
            ]

        if cache_items:
            self._store_in_cache(cache_items, deltas)

        t_end = time.perf_counter()
        logger.info(f"Process mode time: {t_end - t_start:.4f}s")
        logger.info(f"Total wall time: {t_end - t_start_0:.4f}s")

        return deltas

    # -- cube path (range + categorical sources; count/sum/mean/min/max) --------

    def build_cubes(
        self,
        trace_infos: list[TraceInfo],
        viewports_by_figure: dict[str, dict[str, Any]],
        selections: list[SelectionState],
        active_source: ActiveSource,
        cube_cache: CacheBackend | None = None,
    ) -> tuple[list[bytes], dict[str, int]]:
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
        aggregation path. A ``line_env`` target dim keeps the unfiltered domain
        too, because the brush is unknown at build time: the live envelope is
        an approximate preview that the committed selection POST replaces.
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
                *[
                    predicates_to_expr(
                        sel.predicates, schema, is_scan=self._backend_lf.is_scan
                    )
                    for sel in passive
                ]
            )
        owning_figures = {sel.source_figure_uid for sel in passive}

        # Targets: every cube-capable trace in every OTHER figure that owns
        # no committed selection. A figure with a selection is filtered by the
        # passive set minus its own selection, which this cube does not hold,
        # so in update mode the commit request refreshes it.
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
        cubes: list[bytes] = []
        index_by_key: dict[str, int] = {}
        trace_cubes: dict[str, int] = {}
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
        trace_infos: list[TraceInfo],
        viewports_by_figure: dict[str, dict[str, Any]],
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
        viewports_by_figure: dict[str, dict[str, Any]],
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
        viewports_by_figure: dict[str, dict[str, Any]],
        figure_uid: str | None,
        anchor: str | None,
        schema: pl.Schema | None = None,
        column: str | None = None,
    ) -> tuple[float, float] | None:
        """A figure's viewport range on one anchor as physical floats, or
        None (unzoomed).

        Reads the same per-figure viewport mapping that ``process`` consumes
        for ``update_range``. A temporal column's bounds go through
        ``_physical_bound_expr``, the one conversion the display path uses
        (contract G): a date string and an epoch-ms number both land in the
        column's physical unit, so a cube grid cannot miss the display grid by
        the unit factor. Unparsable bounds and other non-numeric values (map
        coordinates, autorange ``None``) yield ``None`` — i.e. the full data
        domain.
        """
        if figure_uid is None or anchor is None:
            return None
        rng = (viewports_by_figure.get(figure_uid) or {}).get(anchor)
        if not (isinstance(rng, (tuple, list)) and len(rng) == 2):
            return None
        dtype = _dtype_for_col(schema, column) if column is not None else None
        if dtype is not None and dtype.is_temporal():
            try:
                lo, hi = (
                    pl.select(_physical_bound_expr(v, dtype).cast(pl.Float64)).item()
                    for v in rng
                )
            except Exception:
                return None
            if lo is None or hi is None:
                return None
            return (float(lo), float(hi))
        if all(isinstance(v, (int, float)) for v in rng):
            return (float(rng[0]), float(rng[1]))
        return None

    def _resolve_cube_domains(
        self,
        free_spec: FreeAxisSpec,
        target_specs: list[CubeTargetSpec],
        schema: pl.Schema | None,
        extra_cols: set[str] | None = None,
    ) -> tuple[FreeAxisSpec | None, dict[str, tuple[float, float]]]:
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
        lifetime (cubes are built only for static sources) — so a cube *cache
        hit* no longer re-scans the full data just to re-derive the
        (cache-key-determining) domain.
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
        minmax = self._backend_lf.physical_minmax(needed, schema) if needed else {}

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

        domains: dict[str, tuple[float, float]] = {}
        for col in unresolved_cols:
            lo, hi = minmax[col]
            if lo is not None and hi is not None:
                domains[col] = (float(lo), float(hi))
        return free, domains

    def _resolved_target_dims(
        self,
        target_dims: tuple,
        domains: dict[str, tuple[float, float]],
        schema: pl.Schema | None = None,
    ) -> tuple | None:
        """Resolve binned target dims and epsilon-pad their upper bounds.

        ``_HIST_BIN_EPSILON`` is added to every ``hist1d``-variant binned dim's
        resolved-or-zoomed domain. This mirrors ``_histogram_bounds_exprs``,
        which pads in both the unzoomed and zoomed cases, so cube bins align
        with display bins. A ``hist2d``-variant dim (contract K) gets no pad:
        the ``fixed_hist2d`` kernel and ``_fixed_hist_bin_expr`` both fold a
        value at ``hi`` into the top bin through the top clamp. A pad here
        would shift bins and break bit-equality with the server delta.
        Returns ``None`` when a dim's domain cannot be resolved (an all-null
        column). That target is not served.

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

    def _active_selections(self, event: InteractionEvent) -> list[Any]:
        if event.type in ("deselect", "init"):
            return []
        return list(event.selections)

    def _selection_filter_exprs(
        self,
        active_selections: list[Any],
        backend_schema: pl.Schema | None,
        owner: str | None,
    ) -> list[pl.Expr]:
        """Build filter expressions from every selection except ``owner``'s.

        Each selection contributes one Polars expression (its predicates ORed
        together).  All such expressions are returned in a list — the caller
        passes them as positional args to ``LazyFrame.filter(*exprs)`` which
        ANDs them.  A figure is never filtered by its own selection.
        """
        if not active_selections or self._backend_lf is None:
            return []

        filter_exprs: list[pl.Expr] = []
        for sel in active_selections:
            if sel.source_figure_uid is None:
                continue
            if sel.source_figure_uid == owner:
                continue
            if not sel.predicates:
                continue
            filter_exprs.append(
                predicates_to_expr(
                    sel.predicates, backend_schema, is_scan=self._backend_lf.is_scan
                )
            )
        return filter_exprs

    def _partitions(
        self,
        aggregation_traces: list[_AggregationTrace],
        active_selections: list[Any],
        selection_fig_uids: set[str | None],
        backend_schema: pl.Schema | None,
    ) -> list[_Partition]:
        """Group traces by the selection they must ignore: their own figure's.

        One partition, one pass, unless a re-aggregated figure owns
        an active selection, e.g. a linked zoom that reaches an owner figure.
        """
        items_by_owner: dict[str | None, list[_AggregationTrace]] = {}
        for item in aggregation_traces:
            fig_uid = item.info.figure_uid
            owner = fig_uid if fig_uid in selection_fig_uids else None
            items_by_owner.setdefault(owner, []).append(item)
        return [
            _Partition(
                owner=owner,
                items=items,
                filter_exprs=self._selection_filter_exprs(
                    active_selections, backend_schema, owner
                ),
            )
            for owner, items in items_by_owner.items()
        ]

    def _viewport_changed(
        self, trace_info: TraceInfo, changed_axes: dict[str, set[str]]
    ) -> bool:
        """Whether the event moved one of the trace's data-binding axes.

        ``recompute_axes`` is anchor-space and unifies cartesian
        (``x``/``y2``/…) and map (``coordinates``) traces.
        """
        changed = changed_axes.get(trace_info.figure_uid, set())
        return bool(
            changed.intersection(self._scalable_traces[trace_info.uid].recompute_axes)
        )

    def _should_process_trace(
        self,
        event: InteractionEvent,
        trace_info: TraceInfo,
        selection_fig_uids: set[str | None],
        changed_axes: dict[str, set[str]],
        cross_filter_mode: str,
    ) -> bool:
        if self._viewport_changed(trace_info, changed_axes):
            return True
        if not event.force_update:
            return False
        if event.type != "selection":
            return True
        if cross_filter_mode == "overlay":
            # An owner shows only its unfiltered background, which no
            # selection changes.
            return trace_info.figure_uid not in selection_fig_uids
        # A figure is filtered by every selection but its own, so only the
        # figure whose selection changed keeps its data.
        return trace_info.figure_uid != event.selection_figure_uid

    def _trace_update_range(
        self,
        trace_info: TraceInfo,
        viewports_by_figure: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        viewport = viewports_by_figure.get(trace_info.figure_uid, {})
        # Hand the trace only the ranges for axes it actually aggregates on, so
        # a non-binding axis (e.g. a line's y) is never fed into its agg spec.
        binding = self._scalable_traces[trace_info.uid].recompute_axes
        return {ax: viewport[ax] for ax in binding if ax in viewport}

    def _resolve_domains(
        self,
        partitions: list[_Partition],
        histogram_domains: dict[str, tuple[str, ...]],
        schema: pl.Schema | None,
        *,
        cross_filter_mode: str,
    ) -> dict[str, dict[str, tuple[Any, Any]]]:
        """Resolve the ``(min, max)`` bounds of this request, per trace.

        Each trace states its own columns (``FlexTrace.domain_cols``); an
        unzoomed histogram instead takes the union its same-figure siblings
        share, so their bars line up.

        Histogram bin edges must not move when a cross-filter narrows the data,
        so they take the unfiltered frame: one collect for every partition. A
        trace with ``domain_follows_filter`` instead takes its partition's
        filtered rows' extent (one collect per partition that has any), but
        only in update mode: in overlay mode the unfiltered background layer
        pins the axis, and both layers share one spec list.

        The builder memoizes the unfiltered bounds when it is ``static``. An
        uncached scan resolves again, so a reset can see changed data on disk.
        The filtered bounds are request-local and never memoized.
        """
        if self._backend_lf is None:
            return {}
        unfiltered_cols: set[str] = set()
        filtered_scopes: list[
            tuple[list[pl.Expr], set[str], dict[str, tuple[str, ...]]]
        ] = []
        unfiltered_uids: dict[str, tuple[str, ...]] = {}
        for partition in partitions:
            follow = bool(partition.filter_exprs) and cross_filter_mode == "update"
            filtered_cols: set[str] = set()
            filtered_uids: dict[str, tuple[str, ...]] = {}
            for item in partition.items:
                cols = histogram_domains.get(item.info.uid) or item.trace.domain_cols(
                    item.update_range
                )
                if follow and item.trace.domain_follows_filter:
                    filtered_cols.update(cols)
                    filtered_uids[item.info.uid] = cols
                else:
                    unfiltered_cols.update(cols)
                    unfiltered_uids[item.info.uid] = cols
            if filtered_cols:
                filtered_scopes.append(
                    (partition.filter_exprs, filtered_cols, filtered_uids)
                )

        unfiltered = (
            self._backend_lf.physical_minmax(sorted(unfiltered_cols), schema)
            if unfiltered_cols
            else {}
        )
        domains = {
            uid: {c: unfiltered[c] for c in cols}
            for uid, cols in unfiltered_uids.items()
        }
        for filter_exprs, cols, uids in filtered_scopes:
            filtered = self._backend_lf.physical_minmax(
                sorted(cols), schema, filter_exprs=filter_exprs
            )
            domains.update(
                {
                    uid: {c: filtered[c] for c in uid_cols}
                    for uid, uid_cols in uids.items()
                }
            )
        return domains

    def _collect_aggregation_specs(
        self,
        aggregation_traces: list[_AggregationTrace],
        backend_schema: pl.Schema | None,
        domains_by_uid: dict[str, dict[str, tuple[Any, Any]]],
    ) -> list[AggregationSpec | GroupedAggregationSpec]:
        agg_specs: list[AggregationSpec | GroupedAggregationSpec] = []
        lf = self._backend_lf

        for item in aggregation_traces:
            ti = item.info
            trace = item.trace
            t_s = time.perf_counter()
            trace_domains = domains_by_uid.get(ti.uid, {})

            agg_specs.append(
                trace.get_aggregation_spec(
                    update_range=item.update_range,
                    schema=backend_schema,
                    domains=trace_domains,
                    scan_source=lf is not None and lf.is_scan,
                    sorted_cols=lf.sorted_cols if lf is not None else frozenset(),
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

        Only the unfiltered ``init``/``deselect`` computation is
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
        histogram_domains: dict[str, tuple[str, ...]],
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

    def _cache_payload(self, delta: TraceDelta) -> dict[str, Any]:
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
        payload: dict[str, Any],
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
        deltas: list[TraceDelta],
    ) -> None:
        key_by_uid = {item.info.uid: key for item, key in cache_items}
        for delta in deltas:
            key = key_by_uid.get(delta.uid)
            if key is not None:
                self._cache.set(key, self._cache_payload(delta))

    def _aggregation_traces(
        self,
        event: InteractionEvent,
        trace_infos: list[TraceInfo],
        selection_fig_uids: set[str | None],
        changed_axes: dict[str, set[str]],
        viewports_by_figure: dict[str, dict[str, Any]],
        cross_filter_mode: str,
    ) -> list[_AggregationTrace]:
        return [
            _AggregationTrace(
                info=ti,
                trace=self._scalable_traces[ti.uid],
                update_range=self._trace_update_range(ti, viewports_by_figure),
            )
            for ti in trace_infos
            if self._should_process_trace(
                event=event,
                trace_info=ti,
                selection_fig_uids=selection_fig_uids,
                changed_axes=changed_axes,
                cross_filter_mode=cross_filter_mode,
            )
        ]

    def _histogram_domain_cols_by_uid(
        self,
        items: Iterable[tuple[TraceInfo, FlexTrace, bool]],
        schema: pl.Schema | None,
    ) -> dict[str, tuple[str, ...]]:
        """Map each unzoomed histogram uid → the sibling columns it bins over.

        Same figure + same axes + same data axis + compatible coordinate unit ⇒
        one shared min/max domain, so sibling histograms' bars line up. Both the
        aggregation and cube paths use this grouping; native temporal physical
        units (days/ms/us/ns) must never be unioned with each other or numerics.
        """
        _GroupKey = tuple[str | None, tuple[str, ...] | None, str, str | None]
        groups: dict[_GroupKey, list[str]] = {}
        uid_to_group: dict[str, _GroupKey] = {}

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

    def _normalise_updates(self, updates: dict[str, Any]) -> dict[str, Any]:
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

    def _aggregate_layer(
        self,
        filter_exprs: list[pl.Expr],
        specs: list[AggregationSpec | GroupedAggregationSpec],
        items: list[_AggregationTrace],
        layer: Literal["bg", "fg"] | None = None,
    ) -> list[TraceDelta]:
        t_start = time.perf_counter()
        df_agg, grouped_dfs = self._backend_lf.aggregate(filter_exprs, specs)
        t_end = time.perf_counter()
        logger.info("Aggregate time (%s): %.4fs", layer or "update", t_end - t_start)
        return self._build_deltas(items, df_agg, grouped_dfs, layer=layer)

    def _overlay_layers(
        self,
        event: InteractionEvent,
        partition: _Partition,
        has_active_selections: bool,
        changed_axes: dict[str, set[str]],
    ) -> tuple[Literal["bg", "fg"], ...]:
        if partition.owner is not None:
            # A selection's source figure draws only the unfiltered background
            # in overlay mode (the renderer shows no fg for it).
            return ("bg",)
        if event.type == "selection":
            viewport_changed = any(
                self._viewport_changed(item.info, changed_axes)
                for item in partition.items
            )
            # ponytail: bg recomputed for the whole partition when any key
            # changed in a selection event; split if measured costly.
            return ("bg", "fg") if viewport_changed else ("fg",)
        if event.type in ("init", "deselect"):
            return ("bg",)
        if event.type == "viewport":
            return ("bg", "fg") if has_active_selections else ("bg",)
        raise ValueError(f"Unsupported interaction event type: {event.type!r}")

    def _background_specs(
        self,
        specs: list[AggregationSpec | GroupedAggregationSpec],
        foreground_shown: bool,
    ) -> list[AggregationSpec | GroupedAggregationSpec]:
        """Filter background specs based on per-trace overlay policy.

        Every trace needs the sole unfiltered layer used by init/deselect.
        While a filtered foreground is shown, ``filtered_only`` traces reuse
        that cached unfiltered data instead of recomputing a duplicate
        background.
        """
        if not foreground_shown:
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
        specs_by_partition: list[
            tuple[_Partition, list[AggregationSpec | GroupedAggregationSpec]]
        ],
        has_active_selections: bool,
        event: InteractionEvent,
        changed_axes: dict[str, set[str]],
    ) -> list[TraceDelta]:
        bg_specs: list[AggregationSpec | GroupedAggregationSpec] = []
        bg_items: list[_AggregationTrace] = []
        fg_jobs = []
        for partition, specs in specs_by_partition:
            layers = self._overlay_layers(
                event=event,
                partition=partition,
                has_active_selections=has_active_selections,
                changed_axes=changed_axes,
            )
            if "bg" in layers:
                # Only a partition with filters gets a foreground, and an owner
                # figure never shows one: otherwise the background must carry
                # every trace, filtered_only ones included.
                foreground_shown = partition.owner is None and bool(
                    partition.filter_exprs
                )
                bg_specs += self._background_specs(specs, foreground_shown)
                bg_items += partition.items
            if "fg" in layers and partition.filter_exprs:
                fg_jobs.append((partition, specs))
        # The background is unfiltered in every partition, so one aggregate call
        # serves all of them: specs that share the source's select read it once
        # (a spec with its own plan still scans on its own).
        deltas = self._aggregate_layer([], bg_specs, bg_items, "bg") if bg_specs else []
        for partition, specs in fg_jobs:
            deltas += self._aggregate_layer(
                partition.filter_exprs, specs, partition.items, "fg"
            )
        return deltas

    def _build_deltas(
        self,
        items: list[_AggregationTrace],
        df_agg: pl.DataFrame,
        grouped_dfs: dict[str, pl.DataFrame],
        layer: Literal["bg", "fg"] | None = None,
    ) -> list[TraceDelta]:
        deltas: list[TraceDelta] = []
        for item in items:
            uid = item.info.uid
            trace = item.trace
            if uid in grouped_dfs:
                deltas.append(
                    self._to_trace_delta(
                        uid, trace._to_grouped_update(grouped_dfs[uid]), layer=layer
                    )
                )
            elif uid in df_agg.columns:
                deltas.append(
                    self._to_trace_delta(uid, trace._to_update(df_agg), layer=layer)
                )
        return deltas

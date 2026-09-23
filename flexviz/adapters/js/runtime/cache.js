// === FlexViz shared runtime — client-side init cache (Phase 1, issue #26) ===
// Requires: state.js loaded first (DASHBOARD_SPEC, cloneObj).
//
// Caches the *unfiltered, viewport-free* init/reset/deselect response so repeat
// occurrences (reset, deselect-to-empty, re-init) need zero server round-trips.
// Design points:
//
//   * Gated on the source(s) opting into caching (FV_CACHEABLE_SOURCES, set by
//     the server at bootstrap from its per-source registry) AND on *every*
//     figure being cacheable — because /dashboard/update is a single
//     whole-dashboard request, so one non-cacheable figure means the request
//     must still go out and nothing is saved.
//   * Gated on no figure being zoomed/panned (see _fvAnyViewportSet): the
//     cached payload is viewport-free, so a deselect while zoomed must go to
//     the server for the viewport-correct result.
//   * It caches the whole figure_deltas payload keyed by (event type, cross-
//     filter mode). For static data the unfiltered response is identical every
//     time, so replaying it reproduces identical client state. Per-trace /
//     content-addressed client caching only pays off once partial requests
//     exist (Phase 2 cubing); it is intentionally deferred.
//
// The server keeps its own independent content-addressed cache; the client
// never reports its cache to the server (statelessness preserved).

const _fvCacheableSources = new Set(
  typeof FV_CACHEABLE_SOURCES !== 'undefined' ? FV_CACHEABLE_SOURCES : []
);
const _FV_CACHE_EVENT_TYPES = new Set(['init', 'deselect']);
const _fvResponseCache = new Map(); // `${type}|${mode}` -> figure_deltas

function _fvCrossFilterMode() {
  return (DASHBOARD_SPEC.state && DASHBOARD_SPEC.state.cross_filter_mode) || 'update';
}

function _fvAllFiguresCacheable() {
  const figs = DASHBOARD_SPEC.figures || [];
  return (
    figs.length > 0 &&
    figs.every(f => f.source != null && _fvCacheableSources.has(f.source))
  );
}

// True when any figure is zoomed/panned. The cached payload is the unfiltered,
// *viewport-free* response, so a deselect issued while zoomed would otherwise
// replay full-range data inside a zoomed axis. The cache is whole-dashboard, so
// a single zoomed figure disqualifies the whole entry and we fall back to a
// full /dashboard/update (the server still serves any unzoomed traces from its
// own per-trace cache).
function _fvAnyViewportSet() {
  const vp = (DASHBOARD_SPEC.state && DASHBOARD_SPEC.state.viewport) || {};
  return Object.values(vp).some(v => v != null);
}

function fvCacheActive(event) {
  return (
    _fvCacheableSources.size > 0 &&
    _FV_CACHE_EVENT_TYPES.has(event.type) &&
    !_fvAnyViewportSet() &&
    _fvAllFiguresCacheable()
  );
}

function _fvCacheKey(event) {
  // init and deselect resolve to the same unfiltered output on the server, so
  // they share one entry per cross-filter mode — the first global reset (an
  // init) or deselect after load already hits the entry populated by init.
  return 'unfiltered|' + _fvCrossFilterMode();
}

// Return a deep clone of the cached figure_deltas for this event, or null.
function fvCacheGet(event) {
  if (!fvCacheActive(event)) return null;
  const hit = _fvResponseCache.get(_fvCacheKey(event));
  return hit ? cloneObj(hit) : null;
}

function fvCachePut(event, figureDeltas) {
  if (!fvCacheActive(event) || !figureDeltas) return;
  _fvResponseCache.set(_fvCacheKey(event), cloneObj(figureDeltas));
}

// === Figure-scoped reset cache (per-figure reset or autorange, case 3a) ===
// A viewport event that returns figures to full autorange (a per-figure reset,
// or a double-click that clears the last zoomed axis) names keys whose figures
// now hold no viewport at all. When *no figure* cross-filters (event.selections
// empty), each such figure's unfiltered slice is exactly the slice already held
// in the whole-dashboard unfiltered blob — so serve those slices and leave the
// other figures untouched.
//
// Gated on event shape, not on which control fired it. No per-figure store is
// needed: the whole-dashboard blob is only ever written when every figure is
// cacheable (see fvCacheActive), so a present blob already contains each
// figure's exact unfiltered slice.
function _fvFigureCacheEligible(event) {
  const figUids = fvFiguresOfKeys(event.viewport_keys || []);
  return (
    _fvCacheableSources.size > 0 &&
    event.type === 'viewport' &&
    figUids.length > 0 &&
    (event.selections || []).length === 0 &&
    figUids.every(figUid => Object.keys(figureViewportRanges(figUid)).length === 0)
  );
}

// Return a deep clone of those figures' unfiltered slices as a figure_deltas
// payload, or null.
function fvCacheGetFigure(event) {
  if (!_fvFigureCacheEligible(event)) return null;
  const blob = _fvResponseCache.get(_fvCacheKey(event));
  if (!blob) return null;
  const out = {};
  for (const figUid of fvFiguresOfKeys(event.viewport_keys || [])) {
    if (!blob[figUid]) return null;
    out[figUid] = cloneObj(blob[figUid]);
  }
  return out;
}

// Cleared on full restore/import (a different spec invalidates the payloads).
function fvCacheReset() {
  _fvResponseCache.clear();
}

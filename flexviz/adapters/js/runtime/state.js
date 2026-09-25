// === FlexViz shared runtime — state initialisation ===
// Requires: DASHBOARD_SPEC (set by Python init block)

// Agent contract: external automation (Playwright, browser extensions) reads
// the live spec through `flexvizState` and writes it back through
// `flexvizApply`, never through the raw DASHBOARD_SPEC binding.
// Reads return a detached snapshot (persistent serialized state only; no
// transient hover/cursor visuals) so callers cannot mutate the authoritative
// client state through it. `flexvizState({compact: true})` returns only the
// interaction state plus a revision, which is what a polling agent needs.
// `flexvizApply(obj)` merges the `state`, `client_state` and `layout` keys
// (`state` and `client_state` one level deeper) and ignores every other key with
// a warning. It re-renders and resolves with the compact state once the
// re-request has completed. It rejects when that re-request fails. The merge
// happens first, so the spec can then be ahead of the page.
window.flexvizState = (opts) =>
  opts && opts.compact ? _fvCompactState() : structuredClone(DASHBOARD_SPEC);

window.flexvizApply = async function(obj) {
  // Clone so the caller keeps no live reference into the authoritative spec,
  // mirroring the detached snapshot the read half returns.
  const patch = structuredClone(obj);
  // `figures` would half-apply: figSpecByUid is built once at load and the panels
  // are server-rendered, so a structure change needs a new share URL.
  const known = ['state', 'client_state', 'layout'];
  for (const key of Object.keys(patch)) {
    if (known.includes(key)) continue;
    console.warn(`flexviz: flexvizApply ignores the key '${key}'`);
    delete patch[key];
  }
  // A rejected patch must leave the page as it was. The restore clears the
  // runtime caches before it requests, so rolling back re-restores the old spec.
  const snapshot = structuredClone({
    state: DASHBOARD_SPEC.state,
    client_state: DASHBOARD_SPEC.client_state,
    layout: DASHBOARD_SPEC.layout,
  });
  // `state` and `client_state` merge one level deep: a patch that carries only
  // `selections` must keep `viewport` and `group_domains`, which several readers
  // dereference without a guard (delta.js ensureGroupColor, plotly relayout).
  Object.assign(DASHBOARD_SPEC, patch, {
    state: { ...DASHBOARD_SPEC.state, ...patch.state },
    client_state: { ...DASHBOARD_SPEC.client_state, ...patch.client_state },
  });
  if (!(await _fvApplySpecToPage())) {
    const reason = _fvLastUpdateError;
    Object.assign(DASHBOARD_SPEC, snapshot);
    const restored = await _fvApplySpecToPage();
    throw new Error(
      `flexviz: dashboard update failed${reason ? ` (${reason})` : ''}`
      + (restored ? '' : '; restoring the previous state failed too, reload the page')
    );
  }
  return _fvCompactState();
};

async function _fvApplySpecToPage() {
  // Grid layout first: the panels must be sized before the re-render.
  window._fvRestoreGridLayout?.();
  window.fvSetGridEditable?.((DASHBOARD_SPEC.layout && DASHBOARD_SPEC.layout.grid_editable) === true);
  window.fvUpdateGridButton?.();
  // fvRestoreFromSpec owns the rest: runtime cache, hover lookups, cross-filter
  // button and selection summary, all before it re-requests.
  return window.fvRestoreFromSpec();
}

let _fvRevision = 0;
let _fvRevisionKey = null;
// ponytail: the revision is diffed here at read time instead of bumped at each
// of the nine mutation sites. Ceiling: several mutations between two reads
// count as one bump. Per-mutation counting needs every site to call in.
function _fvCompactState() {
  const state = DASHBOARD_SPEC.state || {};
  const client_state = DASHBOARD_SPEC.client_state || {};
  const key = JSON.stringify({ state, client_state });
  if (key !== _fvRevisionKey) {
    _fvRevisionKey = key;
    _fvRevision++;
  }
  return {
    version: DASHBOARD_SPEC.version,
    state: structuredClone(state),
    client_state: structuredClone(client_state),
    revision: _fvRevision,
  };
}

const _fvPalette = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'];
if (!DASHBOARD_SPEC.state) DASHBOARD_SPEC.state = {};
if (!DASHBOARD_SPEC.state.group_domains) DASHBOARD_SPEC.state.group_domains = {};
const figSpecByUid = Object.fromEntries(
  DASHBOARD_SPEC.figures.map(f => [f.uid, f])
);
const layerDataByUid = {};
const groupedDataByParent = Object.fromEntries(
  DASHBOARD_SPEC.figures.map(f => [f.uid, Object.fromEntries(
    f.traces
      .filter(ts => ts.params && ts.params.group_by && !ts.params.group_value)
      .map(ts => [ts.uid, { base: [], bg: [], fg: [] }])
  )])
);
const hasBgByFigure = Object.fromEntries(
  DASHBOARD_SPEC.figures.map(f => [f.uid, false])
);
const bgYExtentByFig = Object.fromEntries(
  DASHBOARD_SPEC.figures.map(f => [f.uid, null])
);
const OVERLAY_BG_OPACITY = 0.16;
window.__fvLayerDataByUid = layerDataByUid;
window.__fvGroupedDataByParent = groupedDataByParent;
window.__fvHasBgByFigure = hasBgByFigure;

// Linked-hover lookup tables (Phases 1+2: off, axis; Phase 3: cell)
const IMPLEMENTED_HOVER_MODES = new Set(['off', 'axis', 'cell']);
const IMPLEMENTED_CELL_TRACE_TYPES = new Set(['histogram', 'histogram2d']);
const IMPLEMENTED_AXIS_BAND_TARGET_TRACE_TYPES = new Set(['histogram', 'histogram2d']);

const traceSpecByUid = {};
const traceTypeByUid = {};
const childUidToParentUid = {};

// Map from column name to list of potential hover targets (axis-linked figures)
// { colName -> [{figUid, traceUid, axis, targetModes}] }
const hoverTargetsByColumn = {};

// Map from traceUid to source hover capability
// { traceUid -> {sourceModes, columns, figUid} }
const hoverSourceByTrace = {};

// Registry rebuilt by Plotly adapter after each render; cleared on re-render
// { traceUid -> [{bounds, pointIndex, rowIndex?, colIndex?, coordSpace}] }
const hoverCellsByTraceUid = {};

function _fvClearObject(obj) {
  for (const key of Object.keys(obj)) delete obj[key];
}

function fvRebuildHoverLookups() {
  _fvClearObject(traceSpecByUid);
  _fvClearObject(traceTypeByUid);
  _fvClearObject(hoverTargetsByColumn);
  _fvClearObject(hoverSourceByTrace);

  for (const fig of (DASHBOARD_SPEC.figures || [])) {
    for (const ts of (fig.traces || [])) {
      const uid = ts.uid;
      traceSpecByUid[uid] = ts;
      traceTypeByUid[uid] = ts.trace_type;
      if (ts.hover && ts.hover.source_modes && ts.hover.source_modes.length) {
        hoverSourceByTrace[uid] = {
          sourceModes: ts.hover.source_modes,
          columns: ts.backend_data || {},
          figUid: fig.uid,
        };
      }
      if (ts.hover && ts.hover.target_modes && ts.hover.target_modes.length) {
        for (const [axis, colName] of Object.entries(ts.backend_data || {})) {
          if (!['x', 'y'].includes(axis)) continue;
          if (!hoverTargetsByColumn[colName]) hoverTargetsByColumn[colName] = [];
          hoverTargetsByColumn[colName].push({
            figUid: fig.uid,
            traceUid: uid,
            axis,
            targetModes: ts.hover.target_modes,
          });
        }
      }
    }
  }
}
window.fvRebuildHoverLookups = fvRebuildHoverLookups;
fvRebuildHoverLookups();

// Ensure DASHBOARD_SPEC.client_state exists with a hover_mode
if (!DASHBOARD_SPEC.client_state) DASHBOARD_SPEC.client_state = {};
if (typeof DASHBOARD_SPEC.client_state.hover_mode !== 'string') {
  DASHBOARD_SPEC.client_state.hover_mode = 'off';
}

// --- Utility helpers ---

const RENDER_LAYER_SUFFIX = {
  bg: '__fv_layer_bg',
  fg: '__fv_layer_fg',
};
function stripLayerSuffix(uid) {
  if (!uid) return uid;
  if (uid.endsWith(RENDER_LAYER_SUFFIX.bg))
    return uid.slice(0, -RENDER_LAYER_SUFFIX.bg.length);
  if (uid.endsWith(RENDER_LAYER_SUFFIX.fg))
    return uid.slice(0, -RENDER_LAYER_SUFFIX.fg.length);
  // Backward compatibility for existing rendered ids.
  return uid.replace(/::(bg|fg)$/, '');
}
function cloneObj(obj) {
  return JSON.parse(JSON.stringify(obj));
}
function isGroupedParent(ts) {
  return !!(ts && ts.params && ts.params.group_by && !ts.params.group_value);
}
function ensureLayerData(uid) {
  if (!layerDataByUid[uid]) {
    layerDataByUid[uid] = { base: {}, bg: {}, fg: {} };
  }
  return layerDataByUid[uid];
}
function selectionSourceFigureUids(selections) {
  const out = new Set();
  for (const sel of (selections || [])) {
    if (!sel || !sel.source_figure_uid) continue;
    if (!(sel.predicates || []).length) continue;
    out.add(sel.source_figure_uid);
  }
  return out;
}
function figureHasSelectionSource(figUid, selections) {
  return selectionSourceFigureUids(selections).has(figUid);
}
// A figure is filtered by every selection but its own, so its base data is
// unfiltered only when no other figure has a selection.
function isUnfilteredBaseForFigure(event, figUid) {
  if (['init', 'deselect'].includes(event.type)) return true;
  return !(event.selections || []).some(
    s => s && s.source_figure_uid != null && s.source_figure_uid !== figUid
      && (s.predicates || []).length > 0
  );
}
function backgroundDataLayerForFigure(figUid) {
  return hasBgByFigure[figUid] ? 'bg' : 'base';
}
function rendererUid(uid, layer) {
  if (layer === 'base') return uid;
  const suffix = RENDER_LAYER_SUFFIX[layer];
  return suffix ? (uid + suffix) : uid;
}
function _updateBgYExtent(figUid, yArr) {
  if (!yArr || !yArr.length) return;
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < yArr.length; i++) {
    const v = yArr[i];
    if (v != null && isFinite(v)) {
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
  }
  if (!isFinite(mn)) return;
  const prev = bgYExtentByFig[figUid];
  bgYExtentByFig[figUid] = prev
    ? [Math.min(prev[0], mn), Math.max(prev[1], mx)]
    : [mn, mx];
}
// The viewport keys linked with `key` (ClientState.axis_links), itself included.
function fvLinkedKeys(key) {
  const groups = (DASHBOARD_SPEC.client_state && DASHBOARD_SPEC.client_state.axis_links) || [];
  return groups.find(group => group.includes(key)) || [key];
}

// The one writer of state.viewport: linked axes hold equal ranges by
// construction (the server rejects a spec where they differ). `value` is an
// axis range or map coordinates; null deletes the keys (autorange). Returns
// every key written.
function fvWriteViewport(key, value) {
  const viewport = DASHBOARD_SPEC.state.viewport;
  const keys = fvLinkedKeys(key);
  for (const k of keys) {
    if (value == null) delete viewport[k];
    else viewport[k] = cloneObj(value);
  }
  return keys;
}

// Figure uids of viewport keys, deduplicated in order.
function fvFiguresOfKeys(keys) {
  return [...new Set(keys.map(key => key.split('/')[0]))];
}

function figureViewportRanges(figUid) {
  const viewport = (DASHBOARD_SPEC.state && DASHBOARD_SPEC.state.viewport) || {};
  const ranges = {};
  for (const [key, value] of Object.entries(viewport)) {
    const [keyFigUid, axisId] = key.split('/');
    if (keyFigUid !== figUid || !axisId || !value) continue;
    if (axisId === 'coordinates' && Array.isArray(value)) {
      ranges.coordinates = cloneObj(value);
      continue;
    }
    if (typeof value === 'object' && value !== null
        && Object.prototype.hasOwnProperty.call(value, 'min')
        && Object.prototype.hasOwnProperty.call(value, 'max')) {
      ranges[axisId] = [value.min, value.max];
    }
  }
  return ranges;
}

// Union of the data-binding anchors across a figure's traces. A viewport
// change only needs a backend round-trip if it moves one of these axes — every
// other axis (a line's y, a bar's category axis, …) leaves the data unchanged.
// Mirrors the server's per-trace `recompute_axes` gate in `_should_process_trace`.
function fvFigureRecomputeAxes(figUid) {
  const traces = (figSpecByUid[figUid] && figSpecByUid[figUid].traces) || [];
  const axes = new Set();
  for (const ts of traces) {
    for (const ax of (ts.recompute_axes || [])) axes.add(ax);
  }
  return axes;
}

// True when at least one changed axis re-aggregates a trace in the figure.
function fvNeedsFetch(figUid, changedAxisIds) {
  const binding = fvFigureRecomputeAxes(figUid);
  return (changedAxisIds || []).some(ax => binding.has(ax));
}

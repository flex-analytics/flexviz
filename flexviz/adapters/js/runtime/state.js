// === FlexViz shared runtime — state initialisation ===
// Requires: DASHBOARD_SPEC (set by Python init block)

// Agent-readback contract: external automation (Playwright, browser
// extensions) reads the live spec through this stable accessor rather than
// the raw DASHBOARD_SPEC binding. Returns a detached snapshot (persistent
// serialized state only; no transient hover/cursor visuals) so callers
// cannot mutate the authoritative client state through it.
window.flexvizState = () => structuredClone(DASHBOARD_SPEC);

const _fvPalette = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'];
let _resetTreemapLevel = false;
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
function requestHasActiveSelections(event) {
  return !['init', 'deselect', 'reset'].includes(event.type)
    && !!(event.selections && event.selections.length);
}
function isUnfilteredBaseForFigure(event, figUid) {
  if (['init', 'deselect', 'reset'].includes(event.type)) return true;
  if (!requestHasActiveSelections(event)) return true;
  return figureHasSelectionSource(figUid, event.selections || []);
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
function figureViewportRanges(figUid) {
  const viewport = (DASHBOARD_SPEC.state && DASHBOARD_SPEC.state.viewport) || {};
  const ranges = {};
  for (const [key, value] of Object.entries(viewport)) {
    const slashIdx = key.indexOf('/');
    if (slashIdx === -1) continue;
    const currentFigUid = key.slice(0, slashIdx);
    if (currentFigUid !== figUid || !value) continue;
    const axisId = key.slice(slashIdx + 1);
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
window.figureViewportRanges = figureViewportRanges;

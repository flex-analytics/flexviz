// === Plotly adapter — event handlers ===
// Requires: render.js, state.js

let _programmaticOp = false;

function clearFigureSelection(figUid) {
  const remainingSelections = window.fvClearFigureSelectionFromList?.(
    figUid,
    DASHBOARD_SPEC.state.selections || []
  ) || [];
  window.fvSetSelectionState?.(remainingSelections);
  postDashboardUpdate({
    type: remainingSelections.length ? 'selection' : 'deselect',
    axis_ranges: {},
    selections: remainingSelections,
    force_update: true,
    figure_uid: figUid,
  });
}

function resetTreemapLevel(figUid) {
  _resetTreemapLevel = true;
  _programmaticOp = true;
  _fvRenderFigure(figUid);
  _programmaticOp = false;
  _resetTreemapLevel = false;
}

function _selectionBoxMatches(left, right) {
  if (!left && !right) return true;
  if (!left || !right) return false;
  const keys = ['x0', 'x1', 'xref', 'y0', 'y1', 'yref'];
  return keys.every(k => JSON.stringify(left[k]) === JSON.stringify(right[k]));
}

// Decompose a clicked categorical label into one clause per source column.
// A composite (multi-column) label is the JSON-encoded tuple of its parts.
function _categoricalClausesFromLabel(clicked, labelCols) {
  if (clicked == null || !Array.isArray(labelCols) || !labelCols.length) return null;
  let parts;
  if (labelCols.length === 1) {
    parts = [clicked];
  } else {
    try { parts = JSON.parse(String(clicked)); }
    catch (e) { return null; }
    if (!Array.isArray(parts) || parts.length !== labelCols.length) return null;
  }
  return labelCols.map((col, i) => ({ column: col, values: [parts[i]] }));
}

// Click selection (pie slice / treemap node) — dispatched on the trace's
// declared selection.kind, never on its renderer trace type.
function handleClick(eventData, figUid) {
  if (_programmaticOp) return false;
  const pt = eventData && eventData.points && eventData.points[0];
  if (!pt) return false;
  const figSpec = figSpecByUid[figUid];
  if (!figSpec) return false;
  const logicalUid = stripLayerSuffix((pt.data && pt.data.uid) || '');
  const tsSpec = figSpec.traces.find(ts => ts.uid === logicalUid);
  const sel = tsSpec && tsSpec.selection;
  if (!sel) return false;

  let clauses = null;
  if (sel.kind === 'categorical') {
    clauses = _categoricalClausesFromLabel(pt.label, sel.label_columns);
  } else if (sel.kind === 'path') {
    const sep = sel.path_separator || '/';
    const parts = ((pt.id || '').split(sep).slice(1)).map(v => decodeURIComponent(v));
    if (!parts.length) {
      const hasSelection = (DASHBOARD_SPEC.state.selections || []).some(
        s => s.source_figure_uid === figUid
      );
      if (hasSelection) clearFigureSelection(figUid);
      return false;
    }
    clauses = parts.map((value, i) => ({
      column: (sel.path_columns || [])[i],
      values: [value],
    }));
  }
  if (!clauses) return false;

  const predicate = { clauses };
  const existing = window.fvFigureSelection?.(figUid, DASHBOARD_SPEC.state.selections || []);
  // multi policy: "replace" overwrites; "or"/"path" accumulate (toggle / ancestor-replace).
  const newPredicates = sel.multi === 'replace'
    ? [predicate]
    : (window.fvUpsertPathPredicate?.(
        existing ? (existing.predicates || []) : [],
        predicate
      ) || []);
  if (!newPredicates.length) {
    if (existing) {
      if (sel.kind === 'path') resetTreemapLevel(figUid);
      clearFigureSelection(figUid);
    }
    return false;
  }
  const nextSelections = window.fvReplaceFigureSelection?.(
    figUid,
    { predicates: newPredicates },
    DASHBOARD_SPEC.state.selections || []
  ) || [];
  window.fvSetSelectionState?.(nextSelections);
  // Conditional cube commit — live_brush === "off" gates clicks too
  // (bit-for-bit legacy). Served entirely from the client store ⇒ no POST;
  // any miss POSTs as today and warms the store for the next click.
  if (_fvLiveBrushEnabled() && _fvCubeClickCommit(figUid, tsSpec, newPredicates)) {
    return false;
  }
  postDashboardUpdate({
    type: 'selection', axis_ranges: {},
    selections: nextSelections,
    force_update: true, figure_uid: figUid,
  });
  return false;
}

function extractMapBounds(figUid) {
  const figIdx = figUidToIdx[figUid];
  if (figIdx === undefined) return null;
  const gd = divs[figIdx];
  if (!gd || !gd._fullLayout || !gd._fullLayout.map) return null;
  const mapLayout = gd._fullLayout.map;
  if (!mapLayout._subplot) return null;
  try {
    const mapObj = mapLayout._subplot.getMap();
    if (!mapObj) return null;
    const bounds = mapObj.getBounds();
    if (!bounds) return null;
    const sw = bounds.getSouthWest();
    const ne = bounds.getNorthEast();
    return [
      [sw.lng, sw.lat],
      [ne.lng, sw.lat],
      [ne.lng, ne.lat],
      [sw.lng, ne.lat],
    ];
  } catch (e) {
    return null;
  }
}

function mapCoordinatesFromRelayout(relayout) {
  if (!relayout) return null;
  for (const [key, value] of Object.entries(relayout)) {
    if (!/^map\d*\._derived$/.test(key)) continue;
    if (value && Array.isArray(value.coordinates) && value.coordinates.length) {
      return value.coordinates;
    }
  }
  return null;
}

function geoSelectionFromPoints(eventData, figUid) {
  const points = (eventData && eventData.points) || [];
  if (!points.length) return null;
  const figIdx = figUidToIdx[figUid];
  if (figIdx === undefined) return null;
  const renderedTraces = (divs[figIdx] && divs[figIdx].data) || tracesByFig[figIdx] || [];
  let minLon = Infinity, maxLon = -Infinity, minLat = Infinity, maxLat = -Infinity;

  for (const point of points) {
    const logicalUid = stripLayerSuffix(
      (point && point.data && point.data.uid)
      || (point && point.fullData && point.fullData.uid)
      || ''
    );
    if (!logicalUid) continue;
    const renderedTrace = renderedTraces.find(trace =>
      stripLayerSuffix((trace && trace.uid) || '') === logicalUid
    );
    if (!renderedTrace) continue;
    const location = (point && point.location) ?? (point && point.id);
    if (location == null) continue;
    const feature = ((renderedTrace.geojson || {}).features || []).find(
      item => item && item.id === location
    );
    const ring = (((feature || {}).geometry || {}).coordinates || [])[0] || [];
    for (const coord of ring) {
      minLon = Math.min(minLon, coord[0]);
      maxLon = Math.max(maxLon, coord[0]);
      minLat = Math.min(minLat, coord[1]);
      maxLat = Math.max(maxLat, coord[1]);
    }
  }

  if (![minLon, maxLon, minLat, maxLat].every(Number.isFinite)) return null;
  const geo = _figureSourceTrace(figUid, 'geo_box');
  if (!geo) return null;
  const lonCol = geo.selection && geo.selection.lon_column;
  const latCol = geo.selection && geo.selection.lat_column;
  if (!lonCol || !latCol) return null;
  return [{
    clauses: [
      { column: lonCol, range: [minLon, maxLon] },
      { column: latCol, range: [minLat, maxLat] },
    ],
  }];
}

// First trace in a figure whose declared selection geometry is `kind`.
function _figureSourceTrace(figUid, kind) {
  const figSpec = figSpecByUid[figUid];
  if (!figSpec) return null;
  return (figSpec.traces || []).find(
    ts => ts.selection && ts.selection.kind === kind
  ) || null;
}

function handleRelayout(relayout, figUid) {
  if (!relayout) return;
  if (_programmaticOp) return;
  const keys = Object.keys(relayout);
  if (keys.length === 1 && keys[0] === 'dragmode') {
    updateModeIndicator(figUid, relayout.dragmode);
    return;
  }
  if (keys.every(k => /^shapes(\[|$)/.test(k))) return;
  if (relayout.autosize === true) {
    return;
  }

  const isMapEvent = keys.some(k => k.startsWith('map.') || k.startsWith('map2.'));
  if (isMapEvent) {
    const coords = mapCoordinatesFromRelayout(relayout) || extractMapBounds(figUid);
    if (coords) {
      DASHBOARD_SPEC.state.viewport[figUid + '/coordinates'] = coords;  // persist always
      if (fvNeedsFetch(figUid, ['coordinates'])) {
        postDashboardUpdate({
          type: 'viewport', axis_ranges: { coordinates: coords },
          selections: DASHBOARD_SPEC.state.selections,
          force_update: false, figure_uid: figUid
        });
      }
    }
    return;
  }

  const rangeIndexRe = /^(x|y)axis(\d*)\.range\[([01])\]$/;
  const rangeArrayRe = /^(x|y)axis(\d*)\.range$/;
  const autoRe = /^(x|y)axis(\d*)\.autorange$/;
  const ranges = {};
  let hasAuto = false;
  const autoAxes = [];
  for (const [key, val] of Object.entries(relayout)) {
    const indexed = rangeIndexRe.exec(key);
    if (indexed) {
      const axId = indexed[1] + indexed[2];
      if (!ranges[axId]) ranges[axId] = [null, null];
      ranges[axId][parseInt(indexed[3], 10)] = val;
      continue;
    }
    const array = rangeArrayRe.exec(key);
    if (array) {
      if (Array.isArray(val) && val.length === 2) {
        const axId = array[1] + array[2];
        ranges[axId] = [val[0], val[1]];
      }
      continue;
    }
    const auto = autoRe.exec(key);
    if (auto) {
      hasAuto = true;
      autoAxes.push(auto[1] + auto[2]);
    }
  }
  const complete = {};
  for (const [k, v] of Object.entries(ranges)) {
    if (v[0] != null && v[1] != null) complete[k] = v;
  }
  if (hasAuto && Object.keys(complete).length === 0) {
    // Per-axis autorange (double-click): clear only the autoranged axes locally
    // so a still-zoomed sibling axis is untouched (persist always).
    for (const ax of autoAxes) window.fvClearFigureAxisViewport?.(figUid, ax);
    // Only round-trip if an autoranged axis re-aggregates a trace.
    if (!fvNeedsFetch(figUid, autoAxes)) return;
    // null marks "reset to full range" so the engine recomputes full-range for
    // binding traces; remaining axes keep their stored zoom.
    const axisRanges = figureViewportRanges(figUid);
    for (const ax of autoAxes) axisRanges[ax] = null;
    postDashboardUpdate({
      type: 'viewport',
      axis_ranges: axisRanges,
      force_update: false,
      selections: DASHBOARD_SPEC.state.selections || [],
      figure_uid: figUid,
    });
    return;
  }
  if (Object.keys(complete).length === 0) return;
  const unlockedComplete = window.fvPruneAxisRangesForLocks?.(figUid, complete) || complete;
  const touchedLockedAxes = Object.keys(complete).some(k => window.fvIsAxisLocked?.(figUid, k));
  for (const [k, v] of Object.entries(unlockedComplete)) {
    DASHBOARD_SPEC.state.viewport[figUid + '/' + k] = { min: v[0], max: v[1] };
  }
  const axisRanges = window.fvPruneAxisRangesForLocks?.(figUid, figureViewportRanges(figUid)) || figureViewportRanges(figUid);
  if (Object.keys(unlockedComplete).length === 0) {
    window.fvApplyAxisLocks?.(figUid);
    return;
  }
  if (touchedLockedAxes) window.fvApplyAxisLocks?.(figUid);
  // Viewport is persisted above regardless; only round-trip when a changed axis
  // re-aggregates a trace (e.g. a line's x, not its y). Mirrors the server gate.
  if (!fvNeedsFetch(figUid, Object.keys(unlockedComplete))) return;
  postDashboardUpdate({ type: 'viewport', axis_ranges: axisRanges, selections: DASHBOARD_SPEC.state.selections, force_update: false, figure_uid: figUid });
}

// The TRUE category value of a selected bar point: the trace's underlying
// data value at the point index, NEVER the geometric axis coordinate.
// String labels render on a Plotly *category* axis, so pt.x/pt.y is the label
// already; but NUMERIC labels render on a *linear* axis where grouped bars are
// drawn at category ± offset (barmode="group"), so pt.x is the offset position
// (e.g. 1.2) rather than the label (1) — those offsets match no real column
// value, leaving every cross-filter target empty. Reading
// pt.data[catKey][pt.pointNumber] returns the label exactly as the server
// emitted it: typed (number for numeric labels, JSON-tuple string for
// composite labels) and offset-free, so the predicate, the cube free
// categories, and the committed is_in all agree. Falls back to the geometric
// coordinate when the point carries no data array (defensive).
function _categoricalPointLabel(pt, catKey) {
  if (!pt) return null;
  const idx = pt.pointNumber != null ? pt.pointNumber : pt.pointIndex;
  const arr = pt.data && pt.data[catKey];
  if (arr && idx != null && idx >= 0 && idx < arr.length && arr[idx] != null) {
    return arr[idx];
  }
  return pt[catKey];
}

// Box-drag over categories (bar): the covered labels become is_in clauses.
// Label columns come from the source trace's declared selection descriptor;
// orientation (which point coord holds the category) from its params.
function _categoricalSelectionPredicates(eventData, figUid) {
  const source = _figureSourceTrace(figUid, 'categorical');
  if (!source) return null;
  const labelCols = (source.selection && source.selection.label_columns) || [];
  if (!labelCols.length) return null;
  const isHorizontal = source.params && source.params.orientation === 'h';
  const catKey = isHorizontal ? 'y' : 'x';
  const pts = (eventData && eventData.points) || [];
  if (!pts.length) return null;

  const seen = new Set();
  const predicates = [];
  for (const pt of pts) {
    const raw = _categoricalPointLabel(pt, catKey);
    if (raw == null) continue;
    const key = String(raw);
    if (seen.has(key)) continue;
    seen.add(key);
    const clauses = _categoricalClausesFromLabel(raw, labelCols);
    if (clauses) predicates.push({ clauses });
  }
  return predicates.length ? predicates : null;
}

// Which cartesian axes the figure's range traces actually select on. A box on a
// non-selectable axis carries no filter, so the stored selection box omits it
// (and it auto-completes to the full axis range, rendering as a band). Falls
// back to both axes when no range trace declares geometry (categorical/bar).
function _figureSelectableAxes(figUid) {
  const figSpec = figSpecByUid[figUid];
  let x = false, y = false;
  for (const ts of ((figSpec && figSpec.traces) || [])) {
    const sel = ts.selection;
    if (!sel || sel.kind !== 'range') continue;
    const axes = ts.axes || [];
    for (const anchor of Object.keys(sel.axis_columns || {})) {
      if (anchor === axes[0]) x = true;
      else if (anchor === axes[1]) y = true;
    }
  }
  return (x || y) ? { x, y } : { x: true, y: true };
}

function _plotlySelectionBoxFromRange(eventData, figUid) {
  const range = eventData && eventData.range;
  if (!range) return null;
  const selectable = _figureSelectableAxes(figUid);
  let shape = null;
  if (selectable.x && Array.isArray(range.x)) {
    shape = { x0: range.x[0], x1: range.x[1], xref: 'x' };
  }
  if (selectable.y && Array.isArray(range.y)) {
    shape = shape || {};
    shape.y0 = range.y[0];
    shape.y1 = range.y[1];
    shape.yref = 'y';
  }
  return shape;
}

// Box-select on cartesian axes: each `kind="range"` trace's selection
// descriptor maps its selectable anchor(s) to columns; an anchor matching the
// x-role (axes[0]) takes range.x, the y-role (axes[1]) takes range.y.
function _rangeSelectionPredicates(eventData, figUid) {
  const range = eventData && eventData.range;
  if (!range) return null;
  const figSpec = figSpecByUid[figUid];
  if (!figSpec) return null;

  // One candidate per distinct constraint set, where a constraint is identified
  // by (column, axis-role). The same column bound to x is a *different*
  // constraint than bound to y, so keying on column names alone would wrongly
  // collapse e.g. a line(x=t) beside a horizontal hist(y=t), or hist2d(x=a,y=b)
  // beside hist2d(x=b,y=a) — silently dropping a range.
  const byKey = new Map(); // sorted "col:role" JSON -> { sig, clauses }
  for (const ts of (figSpec.traces || [])) {
    const sel = ts.selection;
    if (!sel || sel.kind !== 'range') continue;
    const axes = ts.axes || [];
    const cols = sel.axis_columns || {};
    const clauses = [];
    const constraints = [];
    for (const anchor of Object.keys(cols)) {
      const col = cols[anchor];
      if (typeof col !== 'string') continue;
      if (anchor === axes[0] && Array.isArray(range.x)) {
        clauses.push({ column: col, range: range.x });
        constraints.push(col + ':x');
      } else if (anchor === axes[1] && Array.isArray(range.y)) {
        clauses.push({ column: col, range: range.y });
        constraints.push(col + ':y');
      }
    }
    if (!clauses.length) continue;
    const sig = constraints.slice().sort();
    byKey.set(JSON.stringify(sig), { sig: new Set(sig), clauses });
  }

  // Drop a candidate whose constraints are a strict subset of another's. Under
  // one box-select all candidates share the same x/y ranges, so a subset is a
  // strictly looser constraint; since figure predicates are OR'd, keeping it
  // would loosen the whole selection (e.g. an x-only line beside an x∧y
  // histogram2d would drop the y bound). Non-subset sets are both kept,
  // preserving the intended any-trace OR.
  const candidates = [...byKey.values()];
  const predicates = candidates
    .filter(cand => !candidates.some(other =>
      other !== cand
      && other.sig.size > cand.sig.size
      && [...cand.sig].every(s => other.sig.has(s))
    ))
    .map(cand => ({ clauses: cand.clauses }));
  return predicates.length ? predicates : null;
}

// === Cube live brush (range + categorical sources → hist/bar/pie + grouped targets) ===
// A drag in select mode becomes a *gesture*: the first plotly_selecting event
// resolves the cube descriptor set and checks the client store (one
// cube_request POST on a miss); each further selecting event re-slices
// locally, rAF-throttled — a range source snaps the in-progress range to the
// P-grid, a categorical (bar) source matches the covered labels to category
// codes. Commit (plotly_selected) snaps range predicates (closed="left");
// categorical predicates stay the legacy is_in shape byte-for-byte. Either
// way the server round-trip is skipped entirely when every cross-filter
// target was cube-served. Pie / treemap click commits run the same
// conditional-commit check without a gesture (_fvCubeClickCommit).

const _fvCubeGestures = {}; // figUid -> gesture state (one drag at a time)

function _fvLiveBrushEnabled() {
  return ((DASHBOARD_SPEC.client_state || {}).live_brush || 'auto') !== 'off';
}

function _fvCubeOverlayMode() {
  return (DASHBOARD_SPEC.state && DASHBOARD_SPEC.state.cross_filter_mode) === 'overlay';
}

// True while a live overlay-mode cube gesture that has applied at least one
// frame is driving this figure's fg layer. Consulted by the renderer
// (buildTracesForFigure / _fvRenderFigure) so the ghost+fg presentation
// applies mid-drag — before any selection is committed, state.selections is
// still empty and the committed-selection check alone would hide the fg.
function fvCubeOverlayFgActive(figUid) {
  for (const gesture of Object.values(_fvCubeGestures)) {
    if (!gesture || gesture.inert || !gesture.live) continue;
    if (!(gesture.lastBins || gesture.lastKey)) continue; // no frame applied
    if ((gesture.targets || []).some(t => t.capable && t.figUid === figUid)) {
      return true;
    }
  }
  return false;
}

// Contract F: in overlay mode every cube-served target figure needs its bg
// ghost (the unfiltered result) before the live fg can be drawn over it.
// The unfiltered data is normally already mirrored into the bg layers
// (init/deselect responses mirror base → bg in postDashboardUpdate); when a
// figure lost it, re-materialize it LOCALLY from the response cache —
// fvCacheGet of the unfiltered init payload, applied to the bg layers only.
// filtered_only targets do not require gesture-created bg state. Zero
// round-trips by construction: a cold cache means no bg this
// gesture — live fg still runs, the skipPost bg conjunct forces the commit
// POST, and the server's bg+fg deltas self-heal. Everything created here is
// recorded on the gesture so an abandoned gesture rolls it back.
function _fvCubeEnsureOverlayBg(gesture) {
  if (!_fvCubeOverlayMode() || gesture.inert) return;
  const needed = new Set();
  for (const t of gesture.targets) {
    if (!t.capable || hasBgByFigure[t.figUid]) continue;
    if (fvCubeTraceFilteredOnly(traceSpecByUid[t.uid])) continue;
    needed.add(t.figUid);
  }
  if (!needed.size) return;
  const blob = fvCacheGet({
    type: 'init', axis_ranges: {}, selections: [], force_update: true,
  });
  if (!blob) return; // cold cache — degrade (skipPost bg conjunct)
  for (const figUid of needed) {
    const deltas = blob[figUid] || [];
    if (!deltas.length) continue;
    const restores = [];
    const prevYExtent = bgYExtentByFig[figUid];
    bgYExtentByFig[figUid] = null;
    for (const delta of deltas) {
      const ts = traceSpecByUid[delta.uid];
      if (fvCubeTraceFilteredOnly(ts)) continue;
      if (delta.group_results) {
        restores.push({
          uid: delta.uid,
          grouped: true,
          data: cloneObj((groupedDataByParent[figUid][delta.uid] || {}).bg || []),
        });
        for (const cr of delta.group_results) childUidToParentUid[cr.uid] = delta.uid;
        setGroupedLayerData(figUid, delta.uid, 'bg', delta.group_results);
        for (const cr of delta.group_results) {
          _updateBgYExtent(figUid, (cr.updates || {}).y);
        }
      } else {
        restores.push({
          uid: delta.uid,
          grouped: false,
          data: cloneObj(ensureLayerData(delta.uid).bg || {}),
        });
        setLayerData(delta.uid, 'bg', delta.updates || {});
        _updateBgYExtent(figUid, (delta.updates || {}).y);
      }
    }
    hasBgByFigure[figUid] = true;
    (gesture.createdBg = gesture.createdBg || []).push({
      figUid, restores, prevYExtent,
    });
  }
}

// skipPost bg conjunct (contract F): in overlay mode a cube-served target
// only supports a local commit when its figure's bg ghost is established —
// otherwise the committed state would render a fg over a missing/stale
// ghost. filtered_only targets do not require gesture-created bg state, so
// they pass unconditionally; their normal unfiltered init layer may remain.
function _fvCubeTargetBgOk(target) {
  if (!_fvCubeOverlayMode()) return true;
  return hasBgByFigure[target.figUid]
    || fvCubeTraceFilteredOnly(traceSpecByUid[target.uid]);
}

// The figure's single cube-eligible source constraint. Two recognized shapes:
//   * 1-D: exactly one (column, axis-role) pair across the figure's range
//     traces ⇒ {column, role, anchor, uid} (Phase-1 shape, unchanged).
//   * box2d (contract H): exactly two pairs (one x AND one y) on a SINGLE
//     trace (a hist2d) and no other pairs ⇒
//     {kind:'box2d', x:{column,anchor}, y:{column,anchor}, uid}.
// Any other count (two pairs on two different traces, three+ pairs, mixed) ⇒
// null — today's "not a cube source" behavior.
function _fvCubeSourceConstraint(figUid) {
  const figSpec = figSpecByUid[figUid];
  if (!figSpec) return null;
  const found = new Map(); // "col:role" -> {column, role, anchor, uid}
  for (const ts of (figSpec.traces || [])) {
    const sel = ts.selection;
    if (!sel || sel.kind !== 'range') continue;
    const axes = ts.axes || [];
    for (const [anchor, col] of Object.entries(sel.axis_columns || {})) {
      if (typeof col !== 'string') continue;
      const role = anchor === axes[0] ? 'x' : (anchor === axes[1] ? 'y' : null);
      if (!role) continue;
      found.set(col + ':' + role, { column: col, role, anchor, uid: ts.uid });
    }
  }
  if (found.size === 1) return found.values().next().value;
  if (found.size === 2) {
    const pairs = [...found.values()];
    const [a, b] = pairs;
    // Both pairs on one trace, one x and one y ⇒ a box2d (hist2d) source.
    if (a.uid === b.uid && a.role !== b.role) {
      const x = a.role === 'x' ? a : b;
      const y = a.role === 'y' ? a : b;
      return {
        kind: 'box2d',
        x: { column: x.column, anchor: x.anchor },
        y: { column: y.column, anchor: y.anchor },
        uid: a.uid,
      };
    }
  }
  return null;
}

// A figure axis's viewport as a cube-key domain: numeric tuple when zoomed,
// null when unzoomed, undefined for non-numeric values the cube path cannot
// key (e.g. map coordinates). Temporal viewports (date-string ranges)
// convert to epoch-ms floats — a SELF-CONSISTENT key token, client-local
// and never sent to the server (the server parses the original strings via
// the schema dtype, contract G). The token is NOT a snap domain: temporal
// snapping always uses the decoded header's physical domain.
function _fvCubeViewportDomain(figUid, anchor) {
  if (!anchor) return null;
  const rng = figureViewportRanges(figUid)[anchor];
  if (rng === undefined) return null;
  if (Array.isArray(rng) && rng.length === 2) {
    if (rng.every(Number.isFinite)) return [rng[0], rng[1]];
    const ms = rng.map(v =>
      typeof v === 'string' ? fvTemporalToPhysical(v, 'ms') : NaN
    );
    if (ms.every(Number.isFinite)) return ms;
  }
  return undefined;
}

// True when the stored viewport for this axis is a temporal (date-string)
// range — the gesture must then defer its snap grid to the decoded header.
function _fvCubeViewportIsTemporal(figUid, anchor) {
  if (!anchor) return false;
  const rng = figureViewportRanges(figUid)[anchor];
  return Array.isArray(rng) && rng.some(v => typeof v === 'string');
}

const _FV_CUBE_AGGS = ['count', 'sum', 'mean', 'min', 'max'];

// Canonical store key for one target trace, or null when the trace is not
// cube-capable. Capability is data, not trace logic: the dispatch mirrors the
// server's capability matrix (the trace get_cube_target_spec gates) minus the
// dtype gates the client cannot see — the server's verdict self-heals those
// (demotion in _fvCubeFetchForGesture). Target-dim order is pinned per
// contract C: hist = (binned data col, *group cols); bar = (*label cols,
// *group cols); pie = (*label cols) — all group/label dims categorical.
// `passiveKey` is the gesture's canonical passive key (contract E) — one
// global set per gesture, shared by every target key.
//
// Returns {key, postRequired}: `key` is null when the trace is not
// cube-capable; `postRequired` is true ONLY for live-only targets that must
// never satisfy skipPost (the line envelope — contract J: the approximate
// minmax-bucket envelope live-updates every drag frame, but every commit
// POSTs so the legacy row-bucket delta replaces it, keeping commit≡restore
// bit-exact). All other target types are skipPost-eligible (postRequired:
// false).
function _fvCubeTargetKey(sourceName, freeDesc, passiveKey, fig, ts) {
  if (!fig.source || fig.source !== sourceName) return { key: null, postRequired: false };
  const params = ts.params || {};
  const bd = ts.backend_data || {};
  if (ts.trace_type === 'histogram') {
    const col = bd[Object.keys(bd)[0]];
    if (typeof col !== 'string') return { key: null, postRequired: false };
    const domain = _fvCubeViewportDomain(fig.uid, (ts.recompute_axes || [])[0] || null);
    if (domain === undefined) return { key: null, postRequired: false };
    const groupCols = params.group_by || [];
    return {
      key: fvCubeKey({
        s: sourceName,
        free: freeDesc,
        t: [
          { c: col, k: 'binned', b: params.bins, d: domain },
          ...groupCols.map(c => ({ c, k: 'categorical' })),
        ],
        m: { a: 'count', v: null },
        p: passiveKey,
      }),
      postRequired: false,
    };
  }
  if (ts.trace_type === 'bar' || ts.trace_type === 'pie') {
    if (!_FV_CUBE_AGGS.includes(params.agg)) return { key: null, postRequired: false }; // median/n_unique
    const labelCols = Array.isArray(bd.labels) ? bd.labels : [bd.labels];
    const groupCols = ts.trace_type === 'bar' ? (params.group_by || []) : [];
    const dimCols = [...labelCols, ...groupCols];
    if (!dimCols.length || dimCols.some(c => typeof c !== 'string')) {
      return { key: null, postRequired: false };
    }
    return {
      key: fvCubeKey({
        s: sourceName,
        free: freeDesc,
        t: dimCols.map(c => ({ c, k: 'categorical' })),
        m: { a: params.agg, v: bd.values ?? null },
        p: passiveKey,
      }),
      postRequired: false,
    };
  }
  if (ts.trace_type === 'treemap') {
    // A treemap is a categorical cube target on its leaf path (contract K):
    // one categorical dim per path column. The agg-algebra gate
    // (median/n_unique ⇒ legacy) is here; the numeric-value-col gate lives
    // server-side and self-heals via demotion. skipPost-eligible
    // (postRequired:false): the leaf cells finalize then sum up every level,
    // so the cube delta byte-equals the server _to_grouped_update delta.
    if (!_FV_CUBE_AGGS.includes(params.agg)) return { key: null, postRequired: false };
    const pathCols = params.path || [];
    if (!pathCols.length || pathCols.some(c => typeof c !== 'string')) {
      return { key: null, postRequired: false };
    }
    return {
      key: fvCubeKey({
        s: sourceName,
        free: freeDesc,
        t: pathCols.map(c => ({ c, k: 'categorical' })),
        m: { a: params.agg, v: bd.values ?? null },
        p: passiveKey,
      }),
      postRequired: false,
    };
  }
  if (ts.trace_type === 'line') {
    // minmax-only gate (contract J); nth/fpcs are not decomposable. The y
    // numeric / x temporal-unit gates live server-side (the client cannot see
    // the schema) and self-heal via demotion in _fvCubeFetchAndStore.
    if (params.downsample !== 'minmax') return { key: null, postRequired: false };
    const xCol = bd.x;
    const yCol = bd.y;
    if (typeof xCol !== 'string' || typeof yCol !== 'string') {
      return { key: null, postRequired: false };
    }
    const nPoints = params.n_points;
    if (!Number.isFinite(nPoints)) return { key: null, postRequired: false };
    const nBuckets = Math.max(1, Math.floor(nPoints / 2));
    const domain = _fvCubeViewportDomain(fig.uid, (ts.recompute_axes || [])[0] || null);
    if (domain === undefined) return { key: null, postRequired: false };
    const groupCols = params.group_by || [];
    return {
      key: fvCubeKey({
        s: sourceName,
        free: freeDesc,
        t: [
          { c: xCol, k: 'binned', b: nBuckets, d: domain },
          ...groupCols.map(c => ({ c, k: 'categorical' })),
        ],
        m: { a: 'line_env', v: yCol },
        p: passiveKey,
      }),
      postRequired: true, // live-only — never skipPost (contract J)
    };
  }
  if (ts.trace_type === 'histogram2d') {
    // hist2d is a count/reduce cube target (contract K) served ONLY for the
    // full-data (unzoomed) case: both x and y must be at full range so the
    // resolved full-data domains give bit-equal binning. A zoom on either
    // axis ⇒ legacy POST (key:null). The numeric-z gate lives server-side
    // (the client cannot see the schema) and self-heals via demotion.
    const xCol = bd.x;
    const yCol = bd.y;
    const zCol = bd.z;
    if (typeof xCol !== 'string' || typeof yCol !== 'string') {
      return { key: null, postRequired: false };
    }
    const axes = ts.axes || ['x', 'y'];
    const xDomain = _fvCubeViewportDomain(fig.uid, axes[0] || 'x');
    const yDomain = _fvCubeViewportDomain(fig.uid, axes[1] || 'y');
    // undefined ⇒ malformed viewport (decline); a non-null domain ⇒ zoomed
    // (decline — full-data only). Only both-null (unzoomed) is cube-served.
    if (xDomain !== null || yDomain !== null) {
      return { key: null, postRequired: false };
    }
    const params2 = ts.params || {};
    const measure = (zCol == null)
      ? { a: 'count', v: null }
      : { a: params2.histfunc, v: zCol };
    return {
      key: fvCubeKey({
        s: sourceName,
        free: freeDesc,
        t: [
          { c: xCol, k: 'binned', b: params2.x_bins, d: null },
          { c: yCol, k: 'binned', b: params2.y_bins, d: null },
        ],
        m: measure,
        p: passiveKey,
      }),
      postRequired: false,
    };
  }
  if (ts.trace_type === 'corr_heatmap') {
    // pearson-only gate (contract I); spearman is rank-based, not decomposable.
    // The numeric-column / reserved-name gates live server-side and self-heal
    // via demotion. EXPLICIT columns are required: implicit (columns omitted)
    // ⇒ the server resolves them from the schema but the client cannot, so the
    // legacy POST self-heals (key:null, postRequired:false).
    if (params.method !== 'pearson') return { key: null, postRequired: false };
    if (!Array.isArray(params.columns) || params.columns.length < 2) {
      return { key: null, postRequired: false };
    }
    // corr has EMPTY target dims (t:[]); the measure block carries the column
    // list (cc) in param order. absolute/triangular are display params, NOT
    // cube-determining — never in the key (two heatmaps differing only there
    // share one cube). corr is conditional-commit-eligible (postRequired:false).
    return {
      key: fvCubeKey({
        s: sourceName,
        free: freeDesc,
        t: [],
        m: { a: 'corr', v: null, cc: params.columns },
        p: passiveKey,
      }),
      postRequired: false,
    };
  }
  return { key: null, postRequired: false };
}

// Cube-capable targets for one source descriptor: every trace in every
// figure other than the source figure THAT OWNS NO COMMITTED SELECTION
// (mirrors the engine's target enumeration / _should_process_trace: legacy
// selection events never update selection-owning figures, so the cube path
// must not either), keyed (or marked incapable) by _fvCubeTargetKey.
function _fvCubeEnumerateTargets(figUid, sourceName, freeDesc, passiveKey) {
  const owning = new Set(
    (DASHBOARD_SPEC.state.selections || [])
      .filter(s => s && s.source_figure_uid != null && (s.predicates || []).length > 0)
      .map(s => s.source_figure_uid)
  );
  const targets = [];
  for (const fig of (DASHBOARD_SPEC.figures || [])) {
    if (fig.uid === figUid || owning.has(fig.uid)) continue;
    for (const ts of (fig.traces || [])) {
      const { key, postRequired } =
        _fvCubeTargetKey(sourceName, freeDesc, passiveKey, fig, ts);
      targets.push({
        figUid: fig.uid,
        uid: ts.uid,
        capable: !!key,
        key,
        postRequired: !!postRequired,
      });
    }
  }
  return targets;
}

// The interacted trace's spec resolved from the event's points (selection
// kind `kind`), falling back to the figure's first matching trace when the
// event carries no usable point. The cube's free axis must be bound to the
// trace actually brushed (step 0b): in a bar(cat) + treemap(cat, sub) figure
// the first-match trace can differ from the interacted one, storing a wrong
// cube under the gesture's key.
function _fvEventSourceTrace(figUid, eventData, kind) {
  const figSpec = figSpecByUid[figUid];
  if (!figSpec) return null;
  for (const pt of ((eventData && eventData.points) || [])) {
    const uid = stripLayerSuffix(
      (pt && pt.data && pt.data.uid) || (pt && pt.fullData && pt.fullData.uid) || ''
    );
    if (!uid) continue;
    const ts = (figSpec.traces || []).find(t => t.uid === uid);
    if (ts && ts.selection && ts.selection.kind === kind) return ts;
  }
  return _figureSourceTrace(figUid, kind);
}

// Gesture start = first plotly_selecting of a drag: dispatch on the figure's
// selection geometry (a range constraint, else a categorical bar source),
// enumerate targets, and check the store. All cube-capable targets covered ⇒
// live immediately; any miss ⇒ one async cube_request (the gesture stays
// mouseup-only until it lands). The canonical passive key is computed ONCE
// per gesture (selections cannot change mid-drag) and threads through every
// target key — a new passive set is simply a store miss, so the lazy 2nd
// selection falls out: one cube_request (which already carries
// state.selections for server-side baking) and the gesture goes live.
// Deselect/reset shrink the passive set ⇒ keys revert to the still-cached
// earlier entries.
function _fvCubeGestureStart(figUid, eventData) {
  const figSpec = figSpecByUid[figUid];
  if (!figSpec || !figSpec.source) return { inert: true };
  const constraint = _fvCubeSourceConstraint(figUid);
  let gesture = { inert: true };
  if (constraint && constraint.kind === 'box2d') {
    gesture = _fvCubeBox2dGestureStart(figUid, figSpec, constraint);
  } else if (constraint) {
    gesture = _fvCubeRangeGestureStart(figUid, figSpec, constraint);
  } else {
    const catSource = _fvEventSourceTrace(figUid, eventData, 'categorical');
    if (catSource) {
      gesture = _fvCubeCategoricalGestureStart(figUid, figSpec, catSource);
    }
  }
  _fvCubeEnsureOverlayBg(gesture);
  return gesture;
}

// Adopt a decoded range free header (or a remembered free record) as the
// gesture's snap grid: the physical domain, the temporal unit, and the
// integer-day grid for unit:"day" (contract G). Idempotent — first adoption
// wins; numeric zoomed gestures already hold their viewport snapDomain.
function _fvCubeAdoptFreeGrid(gesture, free) {
  if (!free || free.kind === 'categorical' || !Array.isArray(free.domain)) return;
  if (!gesture.snapDomain) gesture.snapDomain = free.domain.slice();
  if (free.unit && !gesture.unit) gesture.unit = free.unit;
  if (free.unit === 'day' && !gesture.dayGrid) {
    gesture.dayGrid = { w: free.w, pEff: free.p_eff };
  }
}

// Snap a (physical) range on the gesture's grid: the integer-day grid for a
// unit:"day" source, the shared P-grid arithmetic otherwise — the latter
// byte-identical to Phases 1–2 (the edge arithmetic is parity-pinned).
function _fvCubeGestureSnap(gesture, a, b) {
  if (gesture.dayGrid) {
    return fvCubeSnapDay(
      gesture.snapDomain[0], gesture.dayGrid.w, gesture.dayGrid.pEff, a, b
    );
  }
  return fvCubeSnap(gesture.snapDomain, _FV_CUBE_P, a, b);
}

function _fvCubeRangeGestureStart(figUid, figSpec, constraint) {
  const sourceDomain = _fvCubeViewportDomain(figUid, constraint.anchor);
  if (sourceDomain === undefined) return { inert: true };
  const temporalViewport = _fvCubeViewportIsTemporal(figUid, constraint.anchor);
  const freeDesc = {
    c: constraint.column, k: 'continuous', p: _FV_CUBE_P, d: sourceDomain,
  };
  const passiveKey = fvCubePassiveKey(DASHBOARD_SPEC.state.selections, figUid);
  const gesture = {
    kind: 'range',
    figUid,
    column: constraint.column,
    traceUid: constraint.uid,
    role: constraint.role,
    sourceName: figSpec.source,
    sourceZoomed: sourceDomain != null,
    // Snapping needs a resolved PHYSICAL domain: the zoomed numeric
    // viewport, or (temporal/unzoomed) the server-resolved domain learned
    // from a decoded cube header — a temporal viewport's ms token is a key,
    // never a snap domain (the column's physical unit may differ).
    snapDomain: (!temporalViewport && sourceDomain) || null,
    unit: null,
    dayGrid: null,
    targets: _fvCubeEnumerateTargets(figUid, figSpec.source, freeDesc, passiveKey),
    live: false,
    pendingRange: null,
    rafId: 0,
    lastBins: null,
    savedLayers: null,
  };
  if (!gesture.sourceZoomed) {
    _fvCubeAdoptFreeGrid(
      gesture, fvCubeFreeDomain(figSpec.source, constraint.column, _FV_CUBE_P)
    );
  }
  const capable = gesture.targets.filter(t => t.capable);
  if (capable.length && capable.every(t => fvCubeStoreHas(t.key))) {
    gesture.live = true;
    _fvCubeAdoptFreeGrid(gesture, fvCubeStoreGet(capable[0].key).header.free);
  } else if (capable.length) {
    _fvCubeFetchForGesture(gesture);
  }
  return gesture;
}

// Adopt a decoded box2d free header (contract H) as the gesture's per-axis
// snap grid: each axis's physical domain, temporal unit, and integer-day grid
// (from the header's "units"/"grids" 2-element lists). Idempotent — first
// adoption per axis wins; zoomed numeric axes already hold their viewport
// snapDomain.
function _fvCubeAdoptBox2dGrid(gesture, free) {
  if (!free || free.kind !== 'box2d' || !Array.isArray(free.domains)) return;
  const units = free.units || [null, null];
  const grids = free.grids || [null, null];
  for (let a = 0; a < 2; a++) {
    const ax = gesture.axes[a];
    const dom = free.domains[a];
    if (!Array.isArray(dom)) continue;
    if (!ax.snapDomain) ax.snapDomain = dom.slice();
    if (units[a] && !ax.unit) ax.unit = units[a];
    if (units[a] === 'day' && !ax.dayGrid && grids[a]) {
      ax.dayGrid = { w: grids[a].w, pEff: grids[a].p_eff };
    }
  }
}

// Snap one box2d axis on its grid (integer-day grid for a unit:"day" axis,
// the shared P=128 arithmetic otherwise). Returns the fvCubeSnap result.
function _fvCubeBox2dSnapAxis(ax, a, b) {
  if (ax.dayGrid) {
    return fvCubeSnapDay(ax.snapDomain[0], ax.dayGrid.w, ax.dayGrid.pEff, a, b);
  }
  return fvCubeSnap(ax.snapDomain, _FV_CUBE_BOX2D_P, a, b);
}

// box2d (hist2d) source gesture (contract H): the free axis is a 2-D box over
// (x_col, y_col) at P₂D=128 per axis. Snaps BOTH axes each frame; the
// per-frame payload is the rectangle of composite free bins. Mirrors
// _fvCubeRangeGestureStart but with a per-axis structure.
function _fvCubeBox2dGestureStart(figUid, figSpec, constraint) {
  const domX = _fvCubeViewportDomain(figUid, constraint.x.anchor);
  const domY = _fvCubeViewportDomain(figUid, constraint.y.anchor);
  if (domX === undefined || domY === undefined) return { inert: true };
  const tempX = _fvCubeViewportIsTemporal(figUid, constraint.x.anchor);
  const tempY = _fvCubeViewportIsTemporal(figUid, constraint.y.anchor);
  const freeDesc = {
    c: [constraint.x.column, constraint.y.column],
    k: 'box2d',
    p: _FV_CUBE_BOX2D_P,
    d: [domX, domY],
  };
  const passiveKey = fvCubePassiveKey(DASHBOARD_SPEC.state.selections, figUid);
  const gesture = {
    kind: 'box2d',
    figUid,
    // active_source join key is x (the primary column); the trace uid is the
    // interacted hist2d (step 0b).
    column: constraint.x.column,
    traceUid: constraint.uid,
    sourceName: figSpec.source,
    cols: [constraint.x.column, constraint.y.column],
    sourceZoomed: domX != null || domY != null,
    // Per-axis snap grids. A zoomed NUMERIC viewport is the snap domain
    // directly; a temporal/unzoomed axis defers to the decoded header.
    axes: [
      { snapDomain: (!tempX && domX) || null, unit: null, dayGrid: null },
      { snapDomain: (!tempY && domY) || null, unit: null, dayGrid: null },
    ],
    targets: _fvCubeEnumerateTargets(figUid, figSpec.source, freeDesc, passiveKey),
    live: false,
    pendingRange: null,
    rafId: 0,
    lastBins: null, // {x:[lo,hi], y:[lo,hi]}
    savedLayers: null,
  };
  if (!gesture.sourceZoomed) {
    _fvCubeAdoptBox2dGrid(
      gesture, _fvCubeRememberedBox2dGrid(figSpec.source, gesture.cols)
    );
  }
  const capable = gesture.targets.filter(t => t.capable);
  if (capable.length && capable.every(t => fvCubeStoreHas(t.key))) {
    gesture.live = true;
    _fvCubeAdoptBox2dGrid(gesture, fvCubeStoreGet(capable[0].key).header.free);
  } else if (capable.length) {
    _fvCubeFetchForGesture(gesture);
  }
  return gesture;
}

// Categorical (bar) source gesture: the free axis is the label-column tuple
// (contract B) — no snap domain, no viewport coupling; the per-frame payload
// is the covered-label predicate set instead of a range.
function _fvCubeCategoricalGestureStart(figUid, figSpec, source) {
  const labelCols = (source.selection && source.selection.label_columns) || [];
  if (!labelCols.length || labelCols.some(c => typeof c !== 'string')) {
    return { inert: true };
  }
  const freeDesc = { c: labelCols, k: 'categorical', p: 0, d: null };
  const passiveKey = fvCubePassiveKey(DASHBOARD_SPEC.state.selections, figUid);
  const gesture = {
    kind: 'categorical',
    figUid,
    column: labelCols[0], // active_source join key (the primary label column)
    traceUid: source.uid,
    sourceName: figSpec.source,
    targets: _fvCubeEnumerateTargets(figUid, figSpec.source, freeDesc, passiveKey),
    live: false,
    pendingPredicates: null,
    rafId: 0,
    lastKey: null,
    savedLayers: null,
  };
  const capable = gesture.targets.filter(t => t.capable);
  if (capable.length && capable.every(t => fvCubeStoreHas(t.key))) {
    gesture.live = true;
  } else if (capable.length) {
    _fvCubeFetchForGesture(gesture);
  }
  return gesture;
}

// One cube_request POST: decode + store every served blob, then demote any
// optimistically-capable target absent from trace_cubes — capability
// self-healing (contract D): the server's verdict wins (e.g. a numeric label
// dtype the client cannot see). Liveness and the commit-time skipPost
// predicates then evaluate over the demoted set, so one server-incapable
// target no longer pins a gesture to mouseup-only (the now-mixed dashboard
// still POSTs on commit — correct and expected). `source` is the interacted
// trace: {column, traceUid} (the active_source join keys, step 0b). A blob
// whose decoded free header does not match the target key's free descriptor
// is never stored — the target is demoted instead (fvCubeHeaderMatchesKey).
// Mutates `targets` in place; `onServed` (optional) sees each stored
// (target, entry) pair. Returns the served uid→index map, or null on
// transport/decode failure (callers degrade: gestures stay mouseup-only,
// clicks stay on the POST path).
async function _fvCubeFetchAndStore(figUid, source, targets, onServed) {
  const data = await fvCubeRequest(
    {
      type: 'cube_request',
      axis_ranges: figureViewportRanges(figUid),
      selections: DASHBOARD_SPEC.state.selections || [],
      force_update: false,
      figure_uid: figUid,
    },
    { figure_uid: figUid, column: source.column, trace_uid: source.traceUid }
  );
  if (!data || !Array.isArray(data.blobs)) return null;
  let decoded;
  try {
    decoded = data.blobs.map(decodeFVCube); // decode each blob once
  } catch (e) {
    console.warn('flexviz cube decode failed', e);
    return null;
  }
  const served = data.trace_cubes || {};
  for (const [uid, idx] of Object.entries(served)) {
    const target = targets.find(t => t.uid === uid && t.key);
    const entry = decoded[idx];
    if (!target || !entry) continue;
    if (!fvCubeHeaderMatchesKey(target.key, entry.header)) {
      console.warn('flexviz cube header mismatch — target demoted', uid);
      target.capable = false;
      target.key = null;
      continue;
    }
    if (!fvCubeStorePut(target.key, entry)) {
      // Refused by the byte budget (entry alone exceeds it) — not served.
      target.capable = false;
      target.key = null;
      continue;
    }
    if (onServed) onServed(target, entry);
  }
  for (const target of targets) {
    if (target.capable && !(target.uid in served)) {
      target.capable = false;
      target.key = null;
    }
  }
  return served;
}

async function _fvCubeFetchForGesture(gesture) {
  const served = await _fvCubeFetchAndStore(
    gesture.figUid,
    { column: gesture.column, traceUid: gesture.traceUid },
    gesture.targets,
    (target, entry) => {
      // Snap-grid learning only applies to range/box2d gestures: a categorical
      // free header carries categories, not a numeric domain.
      if (gesture.kind === 'categorical') return;
      if (gesture.kind === 'box2d') {
        if (!gesture.sourceZoomed) {
          _fvCubeRememberBox2dGrid(
            gesture.sourceName, gesture.cols, entry.header.free
          );
        }
        _fvCubeAdoptBox2dGrid(gesture, entry.header.free);
        return;
      }
      if (!gesture.sourceZoomed) {
        fvCubeRememberFreeDomain(
          gesture.sourceName, gesture.column, _FV_CUBE_P, entry.header.free
        );
      }
      _fvCubeAdoptFreeGrid(gesture, entry.header.free);
    }
  );
  if (served === null) return; // degrade: mouseup-only
  // The gesture may have ended meanwhile — the entries stay cached for the
  // next one; liveness only flips for the gesture still in flight.
  if (_fvCubeGestures[gesture.figUid] !== gesture) return;
  const stillCapable = gesture.targets.filter(t => t.capable);
  gesture.live = stillCapable.length > 0
    && stillCapable.every(t => fvCubeStoreHas(t.key));
  if (gesture.live && (gesture.pendingRange || gesture.pendingPredicates)) {
    _fvCubeScheduleFrame(gesture);
  }
}

function _fvCubeScheduleFrame(gesture) {
  if (gesture.rafId) return; // superseded frames are dropped
  gesture.rafId = requestAnimationFrame(() => {
    gesture.rafId = 0;
    if (!gesture.live) return;
    if (gesture.kind === 'categorical') {
      if (gesture.pendingPredicates) {
        _fvCubeApplyCategorical(gesture, gesture.pendingPredicates);
      }
    } else if (gesture.kind === 'box2d') {
      if (
        gesture.axes[0].snapDomain
        && gesture.axes[1].snapDomain
        && gesture.pendingRange
      ) {
        _fvCubeApplyBox2d(gesture, gesture.pendingRange);
      }
    } else if (gesture.snapDomain && gesture.pendingRange) {
      _fvCubeApplyRange(gesture, gesture.pendingRange);
    }
  });
}

function _fvCubeRenderFigures(figUids) {
  for (const figUid of figUids) {
    try {
      _fvRenderFigure(figUid);
    } catch (e) {
      console.warn(`flexviz render failed for figure ${figUid}`, e);
    }
  }
}

// Slice every cube-served target over the free-bin ranges `rangesForEntry`
// yields for its decoded entry (inclusive [lo, hi] pairs — one pair for a
// snapped range gesture, one degenerate pair per selected category code for
// a categorical selection; null skips the target) and route each result
// through the existing delta path (setLayerData / setGroupedLayerData).
// When `gesture` is given, the pre-drag layer of every touched target is
// snapshotted once so an abandoned gesture can restore it (click commits
// pass null — committed state needs no restore). Non-cube targets simply
// hold their pre-drag state. Returns the set of dirty figure uids.
function _fvCubeApplyTargetSlices(targets, rangesForEntry, gesture) {
  const overlayMode =
    (DASHBOARD_SPEC.state && DASHBOARD_SPEC.state.cross_filter_mode) === 'overlay';
  const layerKey = overlayMode ? 'fg' : 'base';
  const dirty = new Set();
  for (const target of targets) {
    if (!target.capable) continue;
    const entry = fvCubeStoreGet(target.key);
    if (!entry) continue;
    const binRanges = rangesForEntry(entry);
    if (!binRanges) continue;
    const ts = traceSpecByUid[target.uid];
    const grouped = isGroupedParent(ts);
    if (gesture) {
      if (!gesture.savedLayers) gesture.savedLayers = {};
      if (!(target.uid in gesture.savedLayers)) {
        // Snapshot the pre-drag layer (grouped parents: their child-result
        // list) so an abandoned gesture can restore it.
        gesture.savedLayers[target.uid] = grouped
          ? {
              figUid: target.figUid,
              layerKey,
              grouped: true,
              data: cloneObj(groupedDataByParent[target.figUid][target.uid][layerKey] || []),
            }
          : {
              figUid: target.figUid,
              layerKey,
              data: cloneObj(ensureLayerData(target.uid)[layerKey] || {}),
            };
      }
    }
    // Dispatch per target shape: grouped parents reconcile child results
    // (line_env envelopes for grouped lines, count cells otherwise); ungrouped
    // hists keep the Phase-1 dense path; ungrouped lines build the envelope
    // delta; bar/pie go cells → delta.
    if (grouped) {
      const groupResults = ts.trace_type === 'line'
        ? fvLineEnvGroupedResults(
            target.figUid, ts, entry.header, fvLineEnvCells(entry, binRanges))
        : fvGroupedResultsFromCells(
            target.figUid, ts, entry.header, fvCubeSliceCells(entry, binRanges));
      for (const cr of groupResults) childUidToParentUid[cr.uid] = target.uid;
      setGroupedLayerData(target.figUid, target.uid, layerKey, groupResults);
    } else if (ts.trace_type === 'histogram') {
      const counts = sliceHistCounts(entry, binRanges);
      const delta = histDeltaFromCounts(ts, entry.header, counts);
      setLayerData(delta.uid, layerKey, delta.updates);
    } else if (ts.trace_type === 'line') {
      const cells = fvLineEnvCells(entry, binRanges);
      const delta = lineEnvDeltaFromCells(ts, cells);
      setLayerData(delta.uid, layerKey, delta.updates);
    } else if (ts.trace_type === 'corr_heatmap') {
      // corr restyles the full z matrix each frame (cells = len(cols)^2, tiny).
      // Live filtered data goes to 'fg' in overlay, otherwise 'base'.
      const delta = fvCorrDeltaFromEntry(ts, entry, binRanges);
      setLayerData(delta.uid, layerKey, delta.updates);
    } else if (ts.trace_type === 'histogram2d') {
      // hist2d restyles the full z matrix each frame from the sliced cells.
      // Live filtered data goes to 'fg' in overlay, otherwise 'base'.
      const delta = fvHist2dDeltaFromEntry(ts, entry, binRanges);
      setLayerData(delta.uid, layerKey, delta.updates);
    } else if (ts.trace_type === 'treemap') {
      // treemap rebuilds its full hierarchy (ids/parents/labels/values) each
      // frame: leaf cells finalize then sum up every path level. Live filtered
      // data goes to 'fg' in overlay, otherwise 'base'.
      const delta = fvTreemapDeltaFromEntry(ts, entry, binRanges);
      setLayerData(delta.uid, layerKey, delta.updates);
    } else {
      const cells = fvCubeSliceCells(entry, binRanges);
      const delta = ts.trace_type === 'pie'
        ? pieDeltaFromCells(ts, cells)
        : barDeltaFromCells(ts, cells);
      setLayerData(delta.uid, layerKey, delta.updates);
    }
    dirty.add(target.figUid);
  }
  return dirty;
}

// Snap the drag range and slice every cube-served target (range gestures);
// de-dupes on unchanged snapped bins. Temporal sources convert the drag's
// date strings to the cube's physical unit first (contract G).
function _fvCubeApplyRange(gesture, range) {
  let r = gesture.role === 'x' ? range.x : range.y;
  if (!Array.isArray(r) || r.length !== 2) return;
  if (gesture.unit) r = r.map(v => fvTemporalToPhysical(v, gesture.unit));
  if (!r.every(Number.isFinite)) return;
  const snap = _fvCubeGestureSnap(gesture, r[0], r[1]);
  if (
    gesture.lastBins
    && gesture.lastBins[0] === snap.loBin
    && gesture.lastBins[1] === snap.hiBin
  ) return;
  gesture.lastBins = [snap.loBin, snap.hiBin];
  const dirty = _fvCubeApplyTargetSlices(
    gesture.targets, () => [[snap.loBin, snap.hiBin]], gesture
  );
  _fvCubeRenderFigures(dirty);
}

// Snap a box2d gesture's drag box on BOTH axes (contract H). Temporal axes
// convert their date-string range to the axis's physical unit first. Returns
// the fvCubeSnap2d-shaped result, or null when a range is malformed.
function _fvCubeBox2dSnap(gesture, range) {
  if (!Array.isArray(range.x) || range.x.length !== 2) return null;
  if (!Array.isArray(range.y) || range.y.length !== 2) return null;
  const conv = (ax, r) =>
    ax.unit ? r.map(v => fvTemporalToPhysical(v, ax.unit)) : r;
  const rx = conv(gesture.axes[0], range.x);
  const ry = conv(gesture.axes[1], range.y);
  if (!rx.every(Number.isFinite) || !ry.every(Number.isFinite)) return null;
  return {
    x: _fvCubeBox2dSnapAxis(gesture.axes[0], rx[0], rx[1]),
    y: _fvCubeBox2dSnapAxis(gesture.axes[1], ry[0], ry[1]),
  };
}

// Snap both axes and slice the rectangle for every cube-served target (box2d
// gestures); de-dupes on the unchanged 2-D snapped bins. The rectangle's
// per-row composite free-bin ranges are built per entry (its p sets S).
function _fvCubeApplyBox2d(gesture, range) {
  const snap = _fvCubeBox2dSnap(gesture, range);
  if (!snap) return;
  const lb = gesture.lastBins;
  if (
    lb
    && lb.x[0] === snap.x.loBin && lb.x[1] === snap.x.hiBin
    && lb.y[0] === snap.y.loBin && lb.y[1] === snap.y.hiBin
  ) return;
  gesture.lastBins = {
    x: [snap.x.loBin, snap.x.hiBin],
    y: [snap.y.loBin, snap.y.hiBin],
  };
  const dirty = _fvCubeApplyTargetSlices(
    gesture.targets, entry => fvCubeRectRanges(entry.header, snap), gesture
  );
  _fvCubeRenderFigures(dirty);
}

// Per-frame dedupe key for a categorical gesture: the SORTED covered-label
// predicate set (the categorical analogue of the range path's lastBins).
function _fvCubePredicateSetKey(predicates) {
  return JSON.stringify(predicates.map(p => JSON.stringify(p.clauses)).sort());
}

// Slice every cube-served target to the covered-label predicates
// (categorical gesture frame / commit); category codes are matched per entry
// header (contract B). De-dupes on the sorted label set.
function _fvCubeApplyCategorical(gesture, predicates) {
  const key = _fvCubePredicateSetKey(predicates);
  if (gesture.lastKey === key) return;
  gesture.lastKey = key;
  const dirty = _fvCubeApplyTargetSlices(
    gesture.targets,
    entry => {
      const codes = fvCubeMatchCategoryCodes(entry.header.free, predicates);
      return codes && codes.map(c => [c, c]);
    },
    gesture
  );
  _fvCubeRenderFigures(dirty);
}

// Detach and return the gesture (commit consumed it; committed state is
// canonical, so no restore).
function _fvCubeGestureTake(figUid) {
  const gesture = _fvCubeGestures[figUid];
  if (!gesture) return null;
  delete _fvCubeGestures[figUid];
  if (gesture.rafId) {
    cancelAnimationFrame(gesture.rafId);
    gesture.rafId = 0;
  }
  return gesture;
}

// Abandoned gesture (empty select / deselect mid-drag): restore the pre-drag
// layer data of every target the live loop touched (grouped parents restore
// their child-result list — the recorded child uids are unchanged, so
// childUidToParentUid needs no rollback), plus any bg layer state the
// gesture itself created (_fvCubeEnsureOverlayBg — contract F).
function _fvCubeGestureAbort(figUid) {
  const gesture = _fvCubeGestureTake(figUid);
  if (!gesture || (!gesture.savedLayers && !gesture.createdBg)) return;
  const dirty = new Set();
  for (const [uid, saved] of Object.entries(gesture.savedLayers || {})) {
    if (saved.grouped) {
      setGroupedLayerData(saved.figUid, uid, saved.layerKey, saved.data);
    } else {
      setLayerData(uid, saved.layerKey, saved.data);
    }
    dirty.add(saved.figUid);
  }
  for (const created of (gesture.createdBg || [])) {
    hasBgByFigure[created.figUid] = false;
    bgYExtentByFig[created.figUid] = created.prevYExtent;
    for (const r of created.restores) {
      if (r.grouped) {
        setGroupedLayerData(created.figUid, r.uid, 'bg', r.data);
      } else {
        setLayerData(r.uid, 'bg', r.data);
      }
    }
    dirty.add(created.figUid);
  }
  _fvCubeRenderFigures(dirty);
}

// Bound (init.js) only when live_brush !== "off" and the figure has range
// or categorical (bar) selection geometry.
function handleSelecting(eventData, figUid) {
  if (_programmaticOp) return;
  if (!_fvLiveBrushEnabled()) return;
  if (!eventData || (!eventData.range && !(eventData.points || []).length)) return;
  let gesture = _fvCubeGestures[figUid];
  if (!gesture) {
    gesture = _fvCubeGestureStart(figUid, eventData);
    _fvCubeGestures[figUid] = gesture;
  }
  if (gesture.inert) return;
  if (gesture.kind === 'categorical') {
    // Covered labels exactly like the commit path; a frame covering no bars
    // yet keeps the last pending set rather than clearing the targets.
    const predicates = _categoricalSelectionPredicates(eventData, figUid);
    if (!predicates) return;
    gesture.pendingPredicates = predicates;
  } else {
    if (!eventData.range) return;
    gesture.pendingRange = eventData.range;
  }
  _fvCubeScheduleFrame(gesture);
}

// === Live brush while editing an existing selection box ===
// Plotly emits no event while an activated selection outline is moved or
// resized: the outline controllers rewrite the SVG path silently each frame,
// and only the mouseup doneFn round-trips through relayout, re-emitting
// plotly_selected. So the drag is watched directly: pointerdown on the
// outline (group move) or its edge handles (resize) arms a rAF loop that
// converts the outline path's bbox back to data coordinates and replays it
// through handleSelecting — the same gesture path as a fresh draw, so cube
// liveness gating, P-grid snapping, and the conditional commit apply
// unchanged, and without cube coverage the gesture stays mouseup-only
// exactly as today. A categorical (bar) source has no range gesture: its
// replay resolves the covered bars from the outline span and feeds them as
// synthetic points (the shape handleSelecting expects). The re-emitted
// plotly_selected consumes the gesture; a
// safety timer aborts a leftover one (restoring pre-drag layers) so a commit
// that never arrives cannot poison the echo guard or the next gesture.

const _FV_SELECTION_EDIT_ABORT_MS = 1500;

function _fvAxisFromRef(gd, ref) {
  if (typeof ref !== 'string' || !/^[xy]\d*$/.test(ref)) return null;
  return gd._fullLayout[ref.charAt(0) + 'axis' + ref.slice(1)] || null;
}

// The outline path's bbox (paper pixels) as an eventData.range-shaped object
// ({x: [lo, hi], y: [lo, hi]}, ascending) — mirrors Plotly's
// makeFillRangeItems (p2r over the pixel extremes, sorted per axis).
function _fvOutlineDataRange(node, xa, ya) {
  let bb;
  try { bb = node.getBBox(); } catch (e) { return null; }
  if (!bb || (bb.width === 0 && bb.height === 0)) return null;
  const asc = (a, b) => (a < b ? -1 : a > b ? 1 : 0);
  const range = {};
  if (xa) {
    range.x = [xa.p2r(bb.x - xa._offset), xa.p2r(bb.x + bb.width - xa._offset)].sort(asc);
  }
  if (ya) {
    range.y = [ya.p2r(bb.y - ya._offset), ya.p2r(bb.y + bb.height - ya._offset)].sort(asc);
  }
  return range;
}

// Shape the edit-drag replay event for handleSelecting. Range / box2d sources
// consume only the geometric range (handleSelecting reads eventData.range); a
// categorical (bar) source instead matches the COVERED bars from eventData
// .points — a bare range carries none, so without synthesizing them the
// categorical gesture never engages during an edit-move and the target would
// update only on the mouseup commit.
function _fvEditReplayEventData(figUid, range, gd) {
  if (_fvCubeSourceConstraint(figUid)) return { range };
  const source = _figureSourceTrace(figUid, 'categorical');
  if (!source) return { range };
  return { range, points: _fvCoveredBarPoints(figUid, range, gd, source) };
}

// Synthesize plotly_selecting-shaped points for the bars of `source` whose
// category position falls inside the outline's span on the category axis.
// Vertical bars carry the category on x, horizontal on y. The axis's d2c maps
// each rendered label to the SAME coordinate space as the outline range (built
// from p2r in _fvOutlineDataRange), so membership is uniform across a category
// axis (string labels) and a linear axis (numeric labels — drawn at their
// value, the cross-filter target). Each point mirrors the shape
// _categoricalSelectionPredicates consumes (pt.data[catKey][pt.pointNumber]).
function _fvCoveredBarPoints(figUid, range, gd, source) {
  const isHorizontal = source.params && source.params.orientation === 'h';
  const catKey = isHorizontal ? 'y' : 'x';
  const span = range && range[catKey];
  if (!Array.isArray(span) || span.length !== 2) return [];
  const axis = gd._fullLayout[catKey + 'axis'];
  if (!axis || typeof axis.d2c !== 'function') return [];
  const lo = Math.min(span[0], span[1]);
  const hi = Math.max(span[0], span[1]);
  const figIdx = figUidToIdx[figUid];
  const rendered = (figIdx !== undefined && divs[figIdx] && divs[figIdx].data) || [];
  const points = [];
  for (const trace of rendered) {
    const stripped = stripLayerSuffix((trace && trace.uid) || '');
    // The source bar trace itself, or (grouped source) one of its children.
    if (stripped !== source.uid && childUidToParentUid[stripped] !== source.uid) {
      continue;
    }
    const arr = trace[catKey];
    if (!Array.isArray(arr)) continue;
    for (let i = 0; i < arr.length; i++) {
      const c = axis.d2c(arr[i]);
      if (Number.isFinite(c) && c >= lo && c <= hi) {
        points.push({ data: trace, pointNumber: i, [catKey]: arr[i] });
      }
    }
  }
  return points;
}

// Bound (init.js, capture phase) alongside plotly_selecting, under the same
// live_brush / selection-geometry gate.
function handleSelectionEditPointerDown(evt, figUid) {
  if (_programmaticOp || !_fvLiveBrushEnabled()) return;
  if (evt.button !== 0 || _fvCubeGestures[figUid]) return;
  const el = evt.target;
  if (!el || !el.closest) return;
  // Group moves start on the selection outline (selectionlayer paths);
  // resizes start on its controller handles (outline-controllers group).
  if (!el.closest('.selectionlayer') && !el.closest('.outline-controllers')) return;
  const figIdx = figUidToIdx[figUid];
  const gd = figIdx === undefined ? null : divs[figIdx];
  if (!gd || !gd._fullLayout) return;
  // Only the activated selection is editable. The click that activates one
  // arrives while no index is set yet, and a plain (no-move) click on an
  // active box never rewrites the outline — both stay inert below.
  const idx = gd._fullLayout._activeSelectionIndex;
  if (!(idx >= 0)) return;
  const selLayout = (gd._fullLayout.selections || [])[idx];
  // Plotly renders two paths per selection (visible outline + fat invisible
  // hit area); an edit drag rewrites only the one the gesture is bound to,
  // so watch both and follow whichever moves.
  const nodes = Array.from(
    gd.querySelectorAll('.selectionlayer path[data-index="' + idx + '"]')
  );
  if (!selLayout || !nodes.length) return;
  const xa = _fvAxisFromRef(gd, selLayout.xref);
  const ya = _fvAxisFromRef(gd, selLayout.yref);
  if (!xa && !ya) return;
  let rafId = 0;
  const lastDs = nodes.map(n => n.getAttribute('d'));
  const frame = () => {
    rafId = requestAnimationFrame(frame);
    let moved = null;
    for (let i = 0; i < nodes.length; i++) {
      const d = nodes[i].getAttribute('d');
      if (d !== lastDs[i]) {
        lastDs[i] = d;
        moved = nodes[i];
      }
    }
    if (!moved) return;
    const range = _fvOutlineDataRange(moved, xa, ya);
    if (range) handleSelecting(_fvEditReplayEventData(figUid, range, gd), figUid);
  };
  const finish = () => {
    cancelAnimationFrame(rafId);
    window.removeEventListener('pointerup', finish, true);
    window.removeEventListener('pointercancel', finish, true);
    const gesture = _fvCubeGestures[figUid];
    if (!gesture) return;
    setTimeout(() => {
      // Same-instance check: never touch a newer gesture started after this
      // drag's commit consumed (or replaced) ours.
      if (_fvCubeGestures[figUid] === gesture) _fvCubeGestureAbort(figUid);
    }, _FV_SELECTION_EDIT_ABORT_MS);
  };
  window.addEventListener('pointerup', finish, true);
  window.addEventListener('pointercancel', finish, true);
  rafId = requestAnimationFrame(frame);
}

// On commit, replace the raw range predicate with the snapped closed="left"
// one (feature-level snap policy, spec §8.2 — applies to every gesture on
// cube-eligible source geometry while live_brush is "auto", live or not) and
// report whether the commit can stay local. Returns null when no gesture ran
// (programmatic selections stay byte-for-byte legacy) or no snap domain is
// known (cold failed request on an unzoomed axis degrades to unsnapped).
function _fvCubeCommitOverride(figUid, range, box) {
  const gesture = _fvCubeGestureTake(figUid);
  if (!gesture || gesture.inert) return null;
  if (gesture.kind === 'box2d') return _fvCubeBox2dCommitOverride(gesture, range, box);
  if (gesture.kind !== 'range' || !gesture.snapDomain) return null;
  let r = gesture.role === 'x' ? range.x : range.y;
  if (!Array.isArray(r) || r.length !== 2) return null;
  // Temporal sources: the drag range arrives as date strings — convert to
  // the cube's physical unit, snap, then render the snapped edges back as
  // strings (contract G; the server's _typed_range_bounds parses them back
  // to the exact integral-unit bounds).
  if (gesture.unit) r = r.map(v => fvTemporalToPhysical(v, gesture.unit));
  if (!r.every(Number.isFinite)) return null;
  const snap = _fvCubeGestureSnap(gesture, r[0], r[1]);
  const edgeLoOut = gesture.unit
    ? fvPhysicalToTemporal(snap.edgeLo, gesture.unit)
    : snap.edgeLo;
  const edgeHiOut = gesture.unit
    ? fvPhysicalToTemporal(snap.edgeHi, gesture.unit)
    : snap.edgeHi;
  const predicates = [{
    clauses: [{
      column: gesture.column,
      range: [edgeLoOut, edgeHiOut],
      closed: 'left',
    }],
  }];
  // The stored/rendered selection shape uses the same snapped edges.
  const snappedBox = { ...(box || {}) };
  if (gesture.role === 'x') {
    snappedBox.x0 = edgeLoOut;
    snappedBox.x1 = edgeHiOut;
    snappedBox.xref = snappedBox.xref || 'x';
  } else {
    snappedBox.y0 = edgeLoOut;
    snappedBox.y1 = edgeHiOut;
    snappedBox.yref = snappedBox.yref || 'y';
  }
  if (gesture.live) {
    // Render the exact committed bins (the rAF may have dropped the last
    // frame; _fvCubeApplyRange de-dupes when the bins are unchanged).
    _fvCubeApplyRange(gesture, range);
  }
  return { predicates, box: snappedBox, skipPost: _fvCubeSkipPost(gesture) };
}

// Conditional commit predicate (shared by every range/box2d gesture): local
// only when EVERY trace in EVERY other figure was cube-served this gesture.
// Any non-capable target (box plot, median bar, a server-demoted target)
// makes the dashboard mixed and forces the normal selection POST. In overlay
// mode each served target additionally needs its bg ghost established (or to
// be filtered_only) — contract F. A cube-served LINE target is live-only
// (postRequired, contract J): the commit must still POST so the legacy delta
// replaces the approximate envelope (commit≡restore parity, spec §8.2).
function _fvCubeSkipPost(gesture) {
  return gesture.live
    && gesture.targets.length > 0
    && gesture.targets.every(
      t => t.capable && fvCubeStoreHas(t.key) && _fvCubeTargetBgOk(t)
    )
    && !gesture.targets.some(t => t.capable && t.postRequired);
}

// Commit a box2d gesture (contract H): ONE predicate with TWO snapped
// closed="left" clauses (x and y) — today's two-clause hist2d selection
// shape, snapped to the cube grid. The stored/rendered box carries the same
// snapped edges on both axes. Conditional-commit skipPost is shared
// (_fvCubeSkipPost). Returns null if either axis has no snap domain (cold
// degrade) or the range is malformed.
function _fvCubeBox2dCommitOverride(gesture, range, box) {
  if (!gesture.axes[0].snapDomain || !gesture.axes[1].snapDomain) return null;
  const snap = _fvCubeBox2dSnap(gesture, range);
  if (!snap) return null;
  const edge = (ax, e) => (ax.unit ? fvPhysicalToTemporal(e, ax.unit) : e);
  const exLo = edge(gesture.axes[0], snap.x.edgeLo);
  const exHi = edge(gesture.axes[0], snap.x.edgeHi);
  const eyLo = edge(gesture.axes[1], snap.y.edgeLo);
  const eyHi = edge(gesture.axes[1], snap.y.edgeHi);
  const predicates = [{
    clauses: [
      { column: gesture.cols[0], range: [exLo, exHi], closed: 'left' },
      { column: gesture.cols[1], range: [eyLo, eyHi], closed: 'left' },
    ],
  }];
  const snappedBox = {
    ...(box || {}),
    x0: exLo, x1: exHi, xref: (box && box.xref) || 'x',
    y0: eyLo, y1: eyHi, yref: (box && box.yref) || 'y',
  };
  if (gesture.live) _fvCubeApplyBox2d(gesture, range);
  return { predicates, box: snappedBox, skipPost: _fvCubeSkipPost(gesture) };
}

// Commit-time decision for a categorical (bar) gesture: the committed
// predicates stay the legacy is_in shape byte-for-byte (no snapping, no
// closed) — only the POST is conditional. Local iff the gesture went live
// and EVERY trace in EVERY other figure is capable and store-served
// (mirrors _fvCubeCommitOverride's skipPost predicate). Always consumes the
// gesture; a live gesture first renders the exact committed label set (the
// rAF may have dropped the last frame — the apply de-dupes on the sorted
// label set). Returns false when no gesture ran (programmatic selections
// and live_brush="off" stay byte-for-byte legacy).
function _fvCubeCategoricalCommitSkip(figUid, predicates) {
  const gesture = _fvCubeGestureTake(figUid);
  if (!gesture || gesture.inert || gesture.kind !== 'categorical') return false;
  if (gesture.live) _fvCubeApplyCategorical(gesture, predicates);
  return gesture.live
    && gesture.targets.length > 0
    && gesture.targets.every(
      t => t.capable && fvCubeStoreHas(t.key) && _fvCubeTargetBgOk(t)
    )
    && !gesture.targets.some(t => t.capable && t.postRequired);
}

// Conditional click commit (pie slice / treemap node): with the committed
// predicates servable against the clicked trace's categorical free axis
// (label / path columns) and EVERY trace in EVERY other figure capable +
// store-served, apply the slices locally, render every figure, and skip the
// POST (returns true). Any miss returns false — the caller POSTs exactly as
// today — and fires one fire-and-forget cube_request (idempotent,
// content-cached server-side) so subsequent clicks are local. `tsSpec` is
// the CLICKED trace spec — its uid is the active_source trace_uid (step 0b),
// so the server resolves the same trace the free descriptor came from.
function _fvCubeClickCommit(figUid, tsSpec, predicates) {
  const figSpec = figSpecByUid[figUid];
  if (!figSpec || !figSpec.source) return false;
  const sel = tsSpec.selection;
  const cols = (sel.kind === 'path' ? sel.path_columns : sel.label_columns) || [];
  if (!cols.length || cols.some(c => typeof c !== 'string')) return false;
  const freeDesc = { c: cols, k: 'categorical', p: 0, d: null };
  // Passive-aware keys (contract E): the clicked figure's own (just-set)
  // selection is excluded; foreign selections key the store entries so a
  // conditional commit only ever serves slices with those filters baked.
  const passiveKey = fvCubePassiveKey(DASHBOARD_SPEC.state.selections, figUid);
  const targets = _fvCubeEnumerateTargets(figUid, figSpec.source, freeDesc, passiveKey);
  const served = targets.length > 0
    && targets.every(
      t => t.capable && fvCubeStoreHas(t.key) && _fvCubeTargetBgOk(t)
    )
    && !targets.some(t => t.capable && t.postRequired)
    && fvCubePredicatesServable(cols, predicates);
  if (!served) {
    if (targets.some(t => t.capable)) {
      // Fire-and-forget warm-up — never awaited, never blocks the POST.
      _fvCubeFetchAndStore(
        figUid, { column: cols[0], traceUid: tsSpec.uid }, targets
      );
    }
    return false;
  }
  _fvCubeApplyTargetSlices(
    targets,
    entry => {
      const codes = fvCubeMatchCategoryCodes(entry.header.free, predicates);
      return codes && codes.map(c => [c, c]);
    },
    null
  );
  // Mirror the post-selection render pass: every figure redraws (the source
  // picks up its committed selection; summaries were refreshed by the
  // caller's fvSetSelectionState).
  _fvCubeRenderFigures(_fvAllFigUids);
  return true;
}

// Re-applying layout.selections during a post-commit render makes Plotly
// re-emit plotly_selected (asynchronously) with a pixel-roundtripped range.
// With a snapped cube commit the stored box is no longer pixel-aligned, so
// the echoed range drifts off the stored edges and would re-enter the legacy
// path as a "new" unsnapped selection (replacing the committed closed="left"
// predicate and double-POSTing). Discriminator: no live-brush gesture ran
// (programmatic re-emission never fires plotly_selecting) AND the range
// matches the stored selection box within half a snap bin — below snap
// resolution a genuinely new brush is indistinguishable from the old one, so
// dropping it is semantically a no-op. Legacy (unsnapped) boxes are
// pixel-aligned and echo back exactly, so tol=0 keeps them covered too.
function _fvCubeEchoOfStoredSelection(figUid, range) {
  const existing = window.fvFigureSelection?.(figUid, DASHBOARD_SPEC.state.selections || []);
  const box = existing && existing._plotly_selection_box;
  if (!box) return false;
  const constraint = _fvLiveBrushEnabled() ? _fvCubeSourceConstraint(figUid) : null;
  const figSpec = figSpecByUid[figUid];
  // Per-axis {column, anchor, role} list of the cube-eligible source geometry
  // (one entry for a 1-D range source; both x and y for a box2d source). Each
  // contributes a half-bin tolerance + a physical unit for its axis.
  const axisConstraints = [];
  if (constraint && constraint.kind === 'box2d') {
    axisConstraints.push({ role: 'x', ...constraint.x });
    axisConstraints.push({ role: 'y', ...constraint.y });
  } else if (constraint) {
    axisConstraints.push({
      role: constraint.role, column: constraint.column, anchor: constraint.anchor,
    });
  }
  // Per-axis tolerance + unit, keyed by role.
  const tolByRole = { x: 0, y: 0 };
  const unitByRole = { x: null, y: null };
  // A box2d source's per-axis snap grid is remembered as a box2d free header
  // (units/grids/domains lists), not a per-column free record; resolve once.
  const box2dGrid = constraint && constraint.kind === 'box2d' && figSpec && figSpec.source
    ? _fvCubeRememberedBox2dGrid(figSpec.source, [constraint.x.column, constraint.y.column])
    : null;
  for (let ci = 0; ci < axisConstraints.length; ci++) {
    const ac = axisConstraints[ci];
    const isBox2d = constraint.kind === 'box2d';
    const p = isBox2d ? _FV_CUBE_BOX2D_P : _FV_CUBE_P;
    // 1-D sources remember a per-(source, column) free record; box2d axes read
    // their {domain, unit, grid} from the remembered box2d header by index.
    let recDomain = null;
    let unit = null;
    let dayW = null;
    if (isBox2d && box2dGrid) {
      recDomain = (box2dGrid.domains || [])[ci] || null;
      unit = (box2dGrid.units || [])[ci] || null;
      const grid = (box2dGrid.grids || [])[ci];
      dayW = grid && grid.w;
    } else if (!isBox2d) {
      const rec = figSpec && figSpec.source
        ? fvCubeFreeDomain(figSpec.source, ac.column, p)
        : null;
      recDomain = rec && rec.domain;
      unit = (rec && rec.unit) || null;
      dayW = rec && rec.w;
    }
    unitByRole[ac.role] = unit;
    let domain = _fvCubeViewportDomain(figUid, ac.anchor);
    if (unit && Array.isArray(domain)) {
      domain = domain.map(v =>
        unit === 'us' ? v * 1000 : unit === 'day' ? v / 86400000 : v
      );
    }
    if (!Array.isArray(domain) && Array.isArray(recDomain)) {
      domain = recDomain;
    }
    if (Array.isArray(domain)) {
      tolByRole[ac.role] = unit === 'day' && dayW
        ? dayW / 2
        : (domain[1] - domain[0]) / p / 2;
    }
  }
  // The echoed values are date strings on temporal axes — convert both
  // sides to physical before the half-bin comparison (contract G).
  const phys = (v, role) =>
    typeof v === 'string' ? fvTemporalToPhysical(v, unitByRole[role] || 'ms') : v;
  let checked = false;
  for (const [axis, k0, k1] of [['x', 'x0', 'x1'], ['y', 'y0', 'y1']]) {
    const r = range && range[axis];
    const hasBoxAxis = box[k0] != null && box[k1] != null;
    // A box axis absent means "unconstrained" (e.g. a hist's count axis) —
    // the echoed range still carries that axis; ignore it rather than
    // treating it as a mismatch.
    if (!hasBoxAxis) continue;
    if (!Array.isArray(r) || r.length !== 2) return false;
    const axisTol = tolByRole[axis] || 0;
    const lo = Math.min(phys(box[k0], axis), phys(box[k1], axis));
    const hi = Math.max(phys(box[k0], axis), phys(box[k1], axis));
    const rlo = Math.min(phys(r[0], axis), phys(r[1], axis));
    const rhi = Math.max(phys(r[0], axis), phys(r[1], axis));
    if (!(Math.abs(lo - rlo) <= axisTol && Math.abs(hi - rhi) <= axisTol)) return false;
    checked = true;
  }
  return checked;
}

// Re-apply the stored canonical selection boxes (snap a live-edited or
// pixel-drifted rendered box back to the committed edges) without a server
// round-trip. Programmatic relayout does not re-enter handleSelected.
function _reapplyCanonicalSelectionBoxes(figUid) {
  const figIdx = figUidToIdx[figUid];
  if (figIdx === undefined || !divs[figIdx]) return;
  fvRunProgrammaticPlotlyOp(() =>
    Plotly.relayout(divs[figIdx], { selections: selectionBoxesForFigure(figUid) })
  );
}

function handleSelected(eventData, figUid) {
  if (!eventData) {
    _fvCubeGestureAbort(figUid);
    return;
  }
  let predicates = geoSelectionFromPoints(eventData, figUid);
  let plotlySelectionBox = null;
  let cubeSkipPost = false;
  if (!predicates && eventData.range) {
    const categorical = _categoricalSelectionPredicates(eventData, figUid);
    if (categorical) {
      predicates = categorical;
      plotlySelectionBox = _plotlySelectionBoxFromRange(eventData, figUid);
      // The predicates stay legacy is_in — only the POST is conditional.
      cubeSkipPost = _fvCubeCategoricalCommitSkip(figUid, predicates);
    } else {
      if (
        !_fvCubeGestures[figUid]
        && _fvCubeEchoOfStoredSelection(figUid, eventData.range)
      ) {
        _reapplyCanonicalSelectionBoxes(figUid);
        return;
      }
      predicates = _rangeSelectionPredicates(eventData, figUid);
      plotlySelectionBox = _plotlySelectionBoxFromRange(eventData, figUid);
      if (predicates) {
        const cubeCommit = _fvCubeCommitOverride(figUid, eventData.range, plotlySelectionBox);
        if (cubeCommit) {
          predicates = cubeCommit.predicates;
          plotlySelectionBox = cubeCommit.box;
          cubeSkipPost = cubeCommit.skipPost;
        }
      }
    }
  }
  if (!predicates) {
    _fvCubeGestureAbort(figUid);
    return;
  }
  const sel = { predicates };
  if (plotlySelectionBox) sel._plotly_selection_box = plotlySelectionBox;
  const existing = window.fvFigureSelection?.(figUid, DASHBOARD_SPEC.state.selections || []);
  if (
    existing
    && window.fvSelectionMatches?.(existing.predicates || [], predicates)
    && _selectionBoxMatches(existing._plotly_selection_box, plotlySelectionBox)
  ) {
    // No semantic change. Plotly edits selections live, so the user may have
    // dragged/cropped the rendered box on a non-selectable axis; re-apply the
    // canonical boxes to snap it back. No server round-trip.
    _reapplyCanonicalSelectionBoxes(figUid);
    return;
  }
  const nextSelections = window.fvReplaceFigureSelection?.(
    figUid,
    sel,
    DASHBOARD_SPEC.state.selections || []
  ) || [];
  window.fvSetSelectionState?.(nextSelections);
  if (cubeSkipPost) {
    // Every cross-filter target rendered the exact committed slice already —
    // the client is authoritative, no round-trip. Mirror the post-selection
    // render pass: every figure redraws (the source picks up its committed
    // selection box; summaries refreshed by fvSetSelectionState above).
    _fvCubeRenderFigures(_fvAllFigUids);
    return;
  }
  postDashboardUpdate({
    type: 'selection', axis_ranges: {},
    selections: nextSelections,
    force_update: true, figure_uid: figUid,
  });
}

function handleDeselect(figUid) {
  if (_programmaticOp) return;
  _fvCubeGestureAbort(figUid);
  clearFigureSelection(figUid);
}

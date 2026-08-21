// === Plotly adapter — linked-hover rendering ===
// Requires: render.js, state.js, runtime/hover.js

let _hoverSuspendedForDrag = false;
const hoverGuidesByFig = {};
window.__fvHoverGuidesByFig = hoverGuidesByFig;

// Teal color system — matches spec CSS tokens
const LINKED_HOVER_LINE_STYLE = { color: 'rgba(13, 148, 136, 0.85)', width: 2, dash: 'solid' };

// ── Overlay DOM helpers (unchanged from previous version) ──────────────────

function hoverGuideShape(axis, value, style, tag) {
  return { tag, axis, value, style };
}

function ensureHoverOverlay(figUid) {
  const figIdx = figUidToIdx[figUid];
  if (figIdx === undefined) return null;
  const div = divs[figIdx];
  if (!div) return null;
  if (window.getComputedStyle(div).position === 'static') {
    div.style.position = 'relative';
  }
  let overlay = div.querySelector('.fv-hover-overlay');
  if (!overlay) {
    overlay = document.createElement('div');
    overlay.className = 'fv-hover-overlay';
    Object.assign(overlay.style, {
      position: 'absolute',
      inset: '0',
      pointerEvents: 'none',
      overflow: 'hidden',
      zIndex: '30',
    });
    div.appendChild(overlay);
  }
  return overlay;
}

function hoverGuideBorderStyle(style) {
  const dash = (style && style.dash) || 'solid';
  if (dash === 'dot') return 'dotted';
  if (dash === 'dash') return 'dashed';
  return 'solid';
}

function renderHoverOverlay(figUid) {
  const figIdx = figUidToIdx[figUid];
  if (figIdx === undefined) return;
  const overlay = ensureHoverOverlay(figUid);
  if (!overlay) return;
  overlay.replaceChildren();

  const gd = divs[figIdx];
  const fullLayout = gd && gd._fullLayout;
  const xAxis = fullLayout && fullLayout.xaxis;
  const yAxis = fullLayout && fullLayout.yaxis;
  if (!xAxis || !yAxis) return;

  for (const guide of (hoverGuidesByFig[figUid] || [])) {

    // --- x_guide / y_guide (solid lines) ---
    if (guide.axis === 'x' || guide.axis === 'y') {
      const axisObj = guide.axis === 'x' ? xAxis : yAxis;
      if (!axisObj || typeof axisObj.l2p !== 'function') continue;
      const pixel = axisObj._offset + axisObj.l2p(guide.value);
      if (!Number.isFinite(pixel)) continue;

      const line = document.createElement('div');
      line.className = 'fv-hover-guide';
      line.dataset.fvHover = guide.tag;
      line.dataset.fvHoverAxis = guide.axis;
      line.style.position = 'absolute';
      line.style.pointerEvents = 'none';

      const width = (((guide.style && guide.style.width) || 1)) + 'px';
      const borderStyle = hoverGuideBorderStyle(guide.style || {});
      const color = (guide.style && guide.style.color) || '#000';

      if (guide.axis === 'x') {
        line.style.left = pixel + 'px';
        line.style.top = yAxis._offset + 'px';
        line.style.height = yAxis._length + 'px';
        line.style.borderLeft = width + ' ' + borderStyle + ' ' + color;
        line.style.transform = 'translateX(-0.5px)';
      } else {
        line.style.left = xAxis._offset + 'px';
        line.style.top = pixel + 'px';
        line.style.width = xAxis._length + 'px';
        line.style.borderTop = width + ' ' + borderStyle + ' ' + color;
        line.style.transform = 'translateY(-0.5px)';
      }
      overlay.appendChild(line);

    // --- x_band (vertical teal band) ---
    } else if (guide.tag && guide.tag.startsWith('linked:x_band')) {
      if (typeof xAxis.l2p !== 'function') continue;
      const px0 = xAxis._offset + xAxis.l2p(guide.x0);
      const px1 = xAxis._offset + xAxis.l2p(guide.x1);
      if (!Number.isFinite(px0) || !Number.isFinite(px1)) continue;
      const left = Math.min(px0, px1);
      const bandWidth = Math.max(1, Math.abs(px1 - px0));

      const band = document.createElement('div');
      band.className = 'fv-hover-guide';
      band.dataset.fvHover = guide.tag;
      band.style.position = 'absolute';
      band.style.pointerEvents = 'none';
      band.style.left = left + 'px';
      band.style.top = yAxis._offset + 'px';
      band.style.width = bandWidth + 'px';
      band.style.height = yAxis._length + 'px';
      band.style.background = 'var(--fv-hover-interval-fill, rgba(13,148,136,0.10))';
      band.style.borderLeft = '1.5px dashed var(--fv-hover-interval-edge, rgba(13,148,136,0.60))';
      band.style.borderRight = '1.5px dashed var(--fv-hover-interval-edge, rgba(13,148,136,0.60))';
      overlay.appendChild(band);

    // --- y_band (horizontal teal band) ---
    } else if (guide.tag && guide.tag.startsWith('linked:y_band')) {
      if (typeof yAxis.l2p !== 'function') continue;
      const py0 = yAxis._offset + yAxis.l2p(guide.y0);
      const py1 = yAxis._offset + yAxis.l2p(guide.y1);
      if (!Number.isFinite(py0) || !Number.isFinite(py1)) continue;
      const top = Math.min(py0, py1);
      const bandHeight = Math.max(1, Math.abs(py1 - py0));

      const band = document.createElement('div');
      band.className = 'fv-hover-guide';
      band.dataset.fvHover = guide.tag;
      band.style.position = 'absolute';
      band.style.pointerEvents = 'none';
      band.style.left = xAxis._offset + 'px';
      band.style.top = top + 'px';
      band.style.width = xAxis._length + 'px';
      band.style.height = bandHeight + 'px';
      band.style.background = 'var(--fv-hover-interval-fill, rgba(13,148,136,0.10))';
      band.style.borderTop = '1.5px dashed var(--fv-hover-interval-edge, rgba(13,148,136,0.60))';
      band.style.borderBottom = '1.5px dashed var(--fv-hover-interval-edge, rgba(13,148,136,0.60))';
      overlay.appendChild(band);

    // --- rect (cell outline) ---
    } else if (guide.tag && guide.tag.startsWith('linked:rect')) {
      const b = guide.bounds;
      if (!b) continue;
      if (typeof xAxis.l2p !== 'function' || typeof yAxis.l2p !== 'function') continue;
      const rx0 = xAxis._offset + xAxis.l2p(b.x0);
      const rx1 = xAxis._offset + xAxis.l2p(b.x1);
      const ry0 = yAxis._offset + yAxis.l2p(b.y0);
      const ry1 = yAxis._offset + yAxis.l2p(b.y1);
      if (!Number.isFinite(rx0) || !Number.isFinite(rx1) || !Number.isFinite(ry0) || !Number.isFinite(ry1)) continue;
      const left = Math.min(rx0, rx1);
      const top = Math.min(ry0, ry1);
      const rectW = Math.max(1, Math.abs(rx1 - rx0));
      const rectH = Math.max(1, Math.abs(ry1 - ry0));

      const rect = document.createElement('div');
      rect.className = 'fv-hover-guide';
      rect.dataset.fvHover = guide.tag;
      rect.style.position = 'absolute';
      rect.style.pointerEvents = 'none';
      rect.style.left = left + 'px';
      rect.style.top = top + 'px';
      rect.style.width = rectW + 'px';
      rect.style.height = rectH + 'px';
      rect.style.border = '2px solid var(--fv-hover-cell-border, rgba(13,148,136,0.80))';
      rect.style.background = 'transparent';
      overlay.appendChild(rect);
    }
  }
}

// ── Clear / suspend / resume ───────────────────────────────────────────────

function clearAllPlotlyCrosshairs() {
  for (const figUid of Object.keys(figUidToIdx)) {
    if (!hoverGuidesByFig[figUid] || hoverGuidesByFig[figUid].length === 0) continue;
    hoverGuidesByFig[figUid] = [];
    renderHoverOverlay(figUid);
  }
}
window.fvClearAllCrosshairs = clearAllPlotlyCrosshairs;
window.fvClearAllHoverVisuals = clearAllPlotlyCrosshairs;

function suspendHoverForDrag() {
  if (_hoverSuspendedForDrag) return;
  _hoverSuspendedForDrag = true;
  clearAllPlotlyCrosshairs();
}

function resumeHoverAfterDrag() {
  if (!_hoverSuspendedForDrag) return;
  window.setTimeout(() => { _hoverSuspendedForDrag = false; }, 0);
}

// ── Plotly → NormalizedHoverEvent ─────────────────────────────────────────

/**
 * Translate a native Plotly plotly_hover event into a NormalizedHoverEvent.
 * Returns null if the event is ineligible (no source trace, no columns).
 * The caller is responsible for setting sourceFigUid on the result.
 *
 * @param {object} eventData - Plotly plotly_hover eventData
 * @param {object} traceSpecByUid - map from traceUid to TraceSpec
 * @returns {object|null} NormalizedHoverEvent | null
 */
function normalizePlotlyHover(eventData, traceSpecByUid, mode) {
  if (!eventData || !eventData.points || !eventData.points.length) return null;
  const pt = eventData.points[0];
  const rawUid = (pt.data && pt.data.uid) || '';
  const logicalUid = stripLayerSuffix(rawUid);
  const resolvedUid = childUidToParentUid[logicalUid] || logicalUid;
  const ts = traceSpecByUid[resolvedUid];
  if (!ts) return null;
  if (!ts.hover || !ts.hover.source_modes || !ts.hover.source_modes.length) return null;

  // Cell event: only in cell mode. A binned trace (histogram/histogram2d)
  // carries customdata bin bounds, but in axis mode it acts as a normal
  // point source (e.g. a histogram bar projects an x-guide at its bin centre),
  // so the cell branch must not shadow the point branch outside cell mode.
  // customdata is the raw hover_bounds entry: {x0,x1} or {x0,x1,y0,y1}.
  if (mode === 'cell'
      && pt.customdata && typeof pt.customdata === 'object' && !Array.isArray(pt.customdata)
      && ('x0' in pt.customdata || 'y0' in pt.customdata)) {
    const bounds = pt.customdata;
    const columns = {};
    if (ts.backend_data && ts.backend_data.x) columns.x = ts.backend_data.x;
    if (ts.backend_data && ts.backend_data.y) columns.y = ts.backend_data.y;
    if (!Object.keys(columns).length) return null;
    return {
      sourceFigUid: null,      // filled by handlePlotlyHover
      sourceTraceUid: rawUid,
      kind: 'cell',
      values: {},
      columns,
      bounds,
      coordSpace: 'cartesian',
      key: null,
    };
  }

  const values = {};
  const columns = {};

  if (pt.x !== undefined && pt.x !== null) {
    values.x = pt.x;
    if (ts.backend_data && ts.backend_data.x) columns.x = ts.backend_data.x;
  }
  if (pt.y !== undefined && pt.y !== null) {
    values.y = pt.y;
    if (ts.backend_data && ts.backend_data.y) columns.y = ts.backend_data.y;
  }

  if (!Object.keys(columns).length) return null;

  return {
    sourceFigUid: null,  // filled by handlePlotlyHover
    sourceTraceUid: rawUid,
    kind: 'point',
    values,
    columns,
    bounds: {},
    coordSpace: 'cartesian',
    key: null,
  };
}

// ── applyHoverVisuals (Plotly adapter implementation) ─────────────────────

/**
 * Render a list of visual instructions onto a target figure's overlay.
 * Called via window.__fvApplyHoverVisuals by the shared runtime.
 *
 * Supported: x_guide, y_guide, x_band, y_band, rect.
 * Future phases add: marker, key.
 *
 * @param {string} figUid
 * @param {Array<{type: string, value: *, role: string}>} visuals
 */
function applyHoverVisuals(figUid, visuals) {
  // Retain only non-linked guides (source affordances / future types)
  hoverGuidesByFig[figUid] = (hoverGuidesByFig[figUid] || []).filter(
    g => !g.tag || !g.tag.startsWith('linked')
  );

  for (const v of (visuals || [])) {
    if (v.type === 'x_guide') {
      hoverGuidesByFig[figUid] = hoverGuidesByFig[figUid] || [];
      hoverGuidesByFig[figUid].push(
        hoverGuideShape('x', v.value, LINKED_HOVER_LINE_STYLE, 'linked:x')
      );
    } else if (v.type === 'y_guide') {
      hoverGuidesByFig[figUid] = hoverGuidesByFig[figUid] || [];
      hoverGuidesByFig[figUid].push(
        hoverGuideShape('y', v.value, LINKED_HOVER_LINE_STYLE, 'linked:y')
      );
    } else if (v.type === 'x_band') {
      hoverGuidesByFig[figUid].push({ tag: 'linked:x_band', x0: v.x0, x1: v.x1 });
    } else if (v.type === 'y_band') {
      hoverGuidesByFig[figUid].push({ tag: 'linked:y_band', y0: v.y0, y1: v.y1 });
    } else if (v.type === 'rect') {
      hoverGuidesByFig[figUid].push({ tag: 'linked:rect', bounds: v.bounds, coordSpace: v.coordSpace });
    }
  }

  renderHoverOverlay(figUid);
}

// Register the Plotly adapter implementation with the shared runtime hook.
window.__fvApplyHoverVisuals = function(figUid, visuals) {
  applyHoverVisuals(figUid, visuals);
};

// ── Plotly event handlers ──────────────────────────────────────────────────

/**
 * Draw a teal cell rect around the hovered corr_heatmap cell.
 * Source affordance only — does not dispatch a linked event to other figures.
 * Uses Plotly axis pixel mapping for the categorical axes.
 */
function _drawCorrHeatmapAffordance(figUid, pt) {
  const figIdx = figUidToIdx[figUid];
  if (figIdx === undefined) return;
  const gd = divs[figIdx];
  const fullLayout = gd && gd._fullLayout;
  const xAxis = fullLayout && fullLayout.xaxis;
  const yAxis = fullLayout && fullLayout.yaxis;
  if (!xAxis || !yAxis || typeof xAxis.l2p !== 'function' || typeof yAxis.l2p !== 'function') return;

  // For categorical heatmap, pointIndex is [rowIdx, colIdx]
  const pi = pt.pointIndex;
  if (!Array.isArray(pi) || pi.length < 2) return;
  const [rowIdx, colIdx] = pi;

  // Categorical axis: category center is at integer index; cell spans ±0.5
  const cx0 = xAxis._offset + xAxis.l2p(colIdx - 0.5);
  const cx1 = xAxis._offset + xAxis.l2p(colIdx + 0.5);
  const cy0 = yAxis._offset + yAxis.l2p(rowIdx - 0.5);
  const cy1 = yAxis._offset + yAxis.l2p(rowIdx + 0.5);

  if (!Number.isFinite(cx0) || !Number.isFinite(cx1) || !Number.isFinite(cy0) || !Number.isFinite(cy1)) return;

  clearAllPlotlyCrosshairs();

  const left = Math.min(cx0, cx1);
  const top = Math.min(cy0, cy1);
  const w = Math.max(1, Math.abs(cx1 - cx0));
  const h = Math.max(1, Math.abs(cy1 - cy0));

  const overlay = ensureHoverOverlay(figUid);
  if (!overlay) return;
  overlay.replaceChildren();
  const rect = document.createElement('div');
  rect.className = 'fv-hover-guide';
  rect.dataset.fvHover = 'source:corr_rect';
  rect.style.position = 'absolute';
  rect.style.pointerEvents = 'none';
  rect.style.left = left + 'px';
  rect.style.top = top + 'px';
  rect.style.width = w + 'px';
  rect.style.height = h + 'px';
  rect.style.border = '2px solid var(--fv-hover-cell-border, rgba(13,148,136,0.80))';
  rect.style.background = 'transparent';
  overlay.appendChild(rect);
}

function handlePlotlyHover(eventData, sourceFigUid) {
  if (_hoverSuspendedForDrag) return;
  if (!eventData || !eventData.points || !eventData.points.length) return;

  const pt0 = eventData.points[0];
  const rawUid = (pt0.data && pt0.data.uid) || '';
  const resolvedUid = childUidToParentUid[stripLayerSuffix(rawUid)]
    || stripLayerSuffix(rawUid);

  // corr_heatmap: source affordance only — draw local rect, no linked dispatch
  if (traceTypeByUid[resolvedUid] === 'corr_heatmap') {
    _drawCorrHeatmapAffordance(sourceFigUid, pt0);
    return;
  }

  // Single on/off control: any non-'off' value enables linked hover.
  if (getHoverMode(DASHBOARD_SPEC) === 'off') return;

  // Auto-select the projection from the hovered source trace: 2D cell sources
  // (heatmaps, geo) project a cell; lines and 1D histograms project a point.
  const effectiveMode = effectiveHoverModeForSource(resolvedUid);
  if (!effectiveMode) return;

  const event = normalizePlotlyHover(eventData, traceSpecByUid, effectiveMode);
  if (!event) return;
  event.sourceFigUid = sourceFigUid;

  // Clear all existing linked visuals before dispatching new ones.
  clearAllPlotlyCrosshairs();

  // Plan and dispatch visual instructions to each target figure.
  const visualsByFig = planHoverVisuals(
    event, effectiveMode, hoverTargetsByColumn, hoverSourceByTrace, hoverCellsByTraceUid,
    { implementedAxisBandTargetTraceTypes: IMPLEMENTED_AXIS_BAND_TARGET_TRACE_TYPES }
  );
  for (const [figUid, visuals] of visualsByFig) {
    dispatchHoverVisuals(figUid, visuals);
  }
}

function handlePlotlyUnhover() {
  if (_hoverSuspendedForDrag) return;
  clearAllPlotlyCrosshairs();
}

// Legacy alias used by some test helpers
function _fvClearAllCrosshairs() { clearAllPlotlyCrosshairs(); }

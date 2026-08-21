// === Plotly adapter — trace construction ===
// Requires: state.js (figSpecByUid, layerDataByUid, groupedDataByParent, etc.)
// Requires: tracesByFig, divs set by Python init

const traceTemplateByUid = {};
const legendVisibilityByUid = {};
window.__fvLegendVisibilityByUid = legendVisibilityByUid;

const configsByFig = DASHBOARD_SPEC.figures.map(() => ({
  responsive: true, displaylogo: false, showTips: false, displayModeBar: false,
}));

function heatmapColorScale(ts) {
  const display = (ts && ts.display) || {};
  if (!Object.prototype.hasOwnProperty.call(display, 'color_scale')) {
    throw new Error('Generated heatmap specs must include explicit color_scale and color_range defaults.');
  }
  const colorScale = display.color_scale;
  if (typeof colorScale !== 'string' || !colorScale) {
    throw new Error('heatmap color_scale must be a non-empty string');
  }
  return colorScale;
}

function heatmapColorRange(ts) {
  const display = (ts && ts.display) || {};
  if (!Object.prototype.hasOwnProperty.call(display, 'color_range')) {
    throw new Error('Generated heatmap specs must include explicit color_scale and color_range defaults.');
  }
  return display.color_range;
}

function applyPlotlyColor(trace, color) {
  if (!trace || !color) return trace;
  const next = { ...trace };
  if (trace.line) next.line = { ...trace.line, color };
  if (trace.marker) next.marker = { ...trace.marker, color };
  if (trace.type === 'bar' || trace.type === 'histogram') {
    next.marker = { ...(trace.marker || {}), color };
  }
  if (trace.type === 'box') {
    next.marker = { ...(trace.marker || {}), color };
    next.line = { ...(trace.line || {}), color };
  }
  return next;
}

// Initialise trace templates from the Python-generated tracesByFig array
tracesByFig.forEach((ts, figIdx) => {
  let colorIndex = 0;
  ts.forEach(t => {
    const figSpec = DASHBOARD_SPEC.figures[figIdx];
    const tsSpec = figSpec.traces.find(specTrace => specTrace.uid === t.uid);
    const color = (tsSpec && tsSpec.display && tsSpec.display.color)
      || _fvPalette[colorIndex % _fvPalette.length];
    traceTemplateByUid[t.uid] = applyPlotlyColor(cloneObj(t), color);
    ensureLayerData(t.uid);
    colorIndex += 1;
  });
});

function makePlotlyTrace(ts, uid, name, color) {
  const traceUid = uid || ts.uid;
  const traceName = name || ((ts.display && ts.display.name) || ts.uid);
  if (ts.trace_type === 'histogram') {
    return { uid: traceUid, type: 'bar', name: traceName, x: [], y: [],
              marker: color ? { color } : {} };
  }
  if (ts.trace_type === 'line') {
    return { uid: traceUid, mode: 'lines', name: traceName, x: [], y: [],
              line: color ? { color } : {}, marker: { opacity: 0 } };
  }
  if (ts.trace_type === 'box') {
    const obj = { uid: traceUid, type: 'box', name: traceName,
                   lowerfence: [], q1: [], median: [], q3: [], upperfence: [],
                   marker: color ? { color } : {} };
    if (ts.backend_data && ts.backend_data.y) { obj.orientation = 'v'; obj.x0 = traceName; }
    else { obj.orientation = 'h'; obj.y0 = traceName; }
    return obj;
  }
  if (ts.trace_type === 'bar') {
    const obj = { uid: traceUid, type: 'bar', name: traceName, x: [], y: [] };
    if (ts.params && ts.params.orientation === 'h') obj.orientation = 'h';
    if (color) obj.marker = { color };
    return obj;
  }
  if (ts.trace_type === 'pie') {
    return { uid: traceUid, type: 'pie', name: traceName,
              labels: [], values: [],
              hole: (ts.params && ts.params.hole) || 0 };
  }
  if (ts.trace_type === 'treemap') {
    return { uid: traceUid, type: 'treemap', name: traceName,
              labels: [], parents: [], ids: [], values: [],
              branchvalues: 'total' };
  }
  if (ts.trace_type === 'histogram2d' || ts.trace_type === 'corr_heatmap') {
    const obj = { uid: traceUid, type: 'heatmap', name: traceName,
              x: [], y: [], z: [], colorscale: heatmapColorScale(ts), showlegend: false };
    const colorRange = heatmapColorRange(ts);
    if (colorRange !== 'auto') {
      obj.zmin = colorRange[0];
      obj.zmax = colorRange[1];
    }
    return obj;
  }
  if (ts.trace_type === 'geo_histogram2d') {
    const obj = { uid: traceUid, type: 'choroplethmap', name: traceName,
              geojson: { type: 'FeatureCollection', features: [] },
              locations: [], z: [], featureidkey: 'id',
              colorscale: heatmapColorScale(ts), showlegend: false, showscale: true,
              marker: { line: { width: 0 } } };
    const colorRange = heatmapColorRange(ts);
    if (colorRange !== 'auto') {
      obj.zmin = colorRange[0];
      obj.zmax = colorRange[1];
    }
    return obj;
  }
  if (ts.trace_type === 'geo_line') {
    const obj = { uid: traceUid, type: 'scattermap', name: traceName,
              lat: [], lon: [], mode: 'lines' };
    if (ts.display && ts.display.color) { obj.line = { color: ts.display.color }; }
    return obj;
  }
  throw new Error('Unsupported trace type ' + ts.trace_type);
}

const HEATMAP_TRACE_TYPES = new Set(['histogram2d', 'corr_heatmap', 'geo_histogram2d']);

function isHeatmapScaledTrace(trace) {
  return trace && (trace.type === 'heatmap' || trace.type === 'choroplethmap');
}

function heatmapZFiniteExtent(z) {
  let vMin = Infinity;
  let vMax = -Infinity;
  for (const row of (z || [])) {
    for (const value of (row || [])) {
      if (value == null || !Number.isFinite(value)) continue;
      if (value < vMin) vMin = value;
      if (value > vMax) vMax = value;
    }
  }
  if (!Number.isFinite(vMin)) return null;
  if (vMin === vMax) {
    const pad = Math.abs(vMin) * 0.01 || 1;
    return [vMin - pad, vMax + pad];
  }
  return [vMin, vMax];
}

function heatmapHasRenderableCells(trace) {
  return heatmapZFiniteExtent(trace && trace.z) !== null;
}

function applyHeatmapColorbarPolicy(trace, renderLayer, showForeground) {
  if (!isHeatmapScaledTrace(trace)) return;
  if (!showForeground) {
    trace.showscale = true;
    return;
  }
  // During Plotly.react keep the bg colorbar; swap to fg-only after render
  // completes (see fvFinalizeHeatmapOverlayColorbars in render.js).
  if (renderLayer === 'bg') {
    trace.showscale = true;
  } else if (renderLayer === 'fg') {
    trace.showscale = false;
  }
}

window.fvFinalizeHeatmapOverlayColorbars = function(figUid) {
  const figIdx = figUidToIdx[figUid];
  if (figIdx === undefined) return Promise.resolve();
  const overlayMode = (DASHBOARD_SPEC.state && DASHBOARD_SPEC.state.cross_filter_mode) === 'overlay';
  const selections = (DASHBOARD_SPEC.state && DASHBOARD_SPEC.state.selections) || [];
  if (!overlayMode || !selections.length || figureHasSelectionSource(figUid, selections)) {
    return Promise.resolve();
  }
  const gd = divs[figIdx];
  if (!gd || !gd.data) return Promise.resolve();

  const bgIndices = [];
  const fgIndices = [];
  const fgZmin = [];
  const fgZmax = [];
  for (let i = 0; i < gd.data.length; i++) {
    const trace = gd.data[i];
    if (!isHeatmapScaledTrace(trace)) continue;
    if (trace.uid.endsWith(RENDER_LAYER_SUFFIX.bg)) {
      bgIndices.push(i);
    } else if (trace.uid.endsWith(RENDER_LAYER_SUFFIX.fg)) {
      if (!heatmapHasRenderableCells(trace)) continue;
      fgIndices.push(i);
      fgZmin.push(trace.zmin);
      fgZmax.push(trace.zmax);
    }
  }
  if (!fgIndices.length) return Promise.resolve();
  const restyle = () => {
    const steps = [];
    if (bgIndices.length) {
      steps.push(Plotly.restyle(gd, { showscale: false }, bgIndices));
    }
    steps.push(
      Plotly.restyle(
        gd,
        { showscale: true, zmin: fgZmin, zmax: fgZmax },
        fgIndices
      )
    );
    return Promise.all(steps);
  };
  return typeof fvRunProgrammaticPlotlyOp === 'function'
    ? fvRunProgrammaticPlotlyOp(restyle)
    : Promise.resolve(restyle());
};

function applyHeatmapZRange(trace, extent) {
  if (!trace || !extent) return;
  trace.zmin = extent[0];
  trace.zmax = extent[1];
}

function syncHeatmapOverlayColorScale(traces, figSpec, showForeground) {
  if (!showForeground || !figSpec) return;
  for (const ts of figSpec.traces) {
    if (!HEATMAP_TRACE_TYPES.has(ts.trace_type)) continue;
    const bgTrace = traces.find(
      t => stripLayerSuffix(t.uid) === ts.uid && t.uid.endsWith(RENDER_LAYER_SUFFIX.bg)
    );
    const fgTrace = traces.find(
      t => stripLayerSuffix(t.uid) === ts.uid && t.uid.endsWith(RENDER_LAYER_SUFFIX.fg)
    );
    if (!bgTrace || !fgTrace) continue;
    const colorRange = heatmapColorRange(ts);
    if (colorRange !== 'auto') {
      applyHeatmapZRange(bgTrace, colorRange);
      applyHeatmapZRange(fgTrace, colorRange);
      continue;
    }
    // Auto range: bg uses full cached data; fg uses filtered data so the
    // post-finalize colorbar reflects the selection, not the original scale.
    applyHeatmapZRange(bgTrace, heatmapZFiniteExtent(bgTrace.z));
    applyHeatmapZRange(fgTrace, heatmapZFiniteExtent(fgTrace.z));
  }
}

/**
 * Flatten hover_bounds into hoverCellsByTraceUid for cell matching.
 * hoverBounds can be:
 *   - 1D array: [{x0,x1}, ...] or [{y0,y1}, ...] (histogram)
 *   - 2D array: [[{x0,x1,y0,y1}, ...], ...] (histogram2d, outer=rows/y, inner=cols/x)
 * @param {string} logicalUid
 * @param {Array} hoverBounds
 */
function _rebuildHoverCells(logicalUid, hoverBounds) {
  // Hover-cell lookups (planHoverVisuals, axis-band targets) key on the logical
  // *parent* uid taken from the trace spec, but grouped children render under
  // their own child uid. Mirror the cells under the parent so grouped histograms
  // resolve as hover targets — every child shares the same bin edges, so the
  // last-written child's bounds are representative.
  const parentUid = childUidToParentUid[logicalUid] || logicalUid;
  if (!hoverBounds || !hoverBounds.length) {
    delete hoverCellsByTraceUid[logicalUid];
    if (parentUid !== logicalUid) delete hoverCellsByTraceUid[parentUid];
    return;
  }
  const cells = [];
  if (Array.isArray(hoverBounds[0])) {
    // 2D case: histogram2d. Outer = row (y), inner = col (x).
    for (let r = 0; r < hoverBounds.length; r++) {
      for (let c = 0; c < hoverBounds[r].length; c++) {
        cells.push({
          bounds: hoverBounds[r][c],
          pointIndex: r * hoverBounds[r].length + c,
          rowIndex: r,
          colIndex: c,
          coordSpace: 'cartesian',
        });
      }
    }
  } else {
    // 1D case: histogram.
    for (let i = 0; i < hoverBounds.length; i++) {
      cells.push({
        bounds: hoverBounds[i],
        pointIndex: i,
        coordSpace: 'cartesian',
      });
    }
  }
  hoverCellsByTraceUid[logicalUid] = cells;
  if (parentUid !== logicalUid) hoverCellsByTraceUid[parentUid] = cells;
}

function buildTraceFromTemplate(template, logicalUid, renderLayer, updates, opacity, showlegend, forceBarOffsetgroup = false, showForeground = false, applyLineGaps = false) {
  if (!template) return null;
  const effectiveShowlegend = (template.showlegend !== undefined)
    ? template.showlegend
    : showlegend;
  const trace = { ...template, uid: rendererUid(logicalUid, renderLayer),
                   legendgroup: logicalUid, opacity, showlegend: effectiveShowlegend };
  if (template.line) trace.line = { ...template.line };
  if (template.marker) trace.marker = { ...template.marker };
  for (const [k, v] of Object.entries(updates || {})) {
    if (k === 'hover_bounds') continue;  // handled below as customdata
    trace[k] = v;
  }
  if (updates && updates.hover_bounds) {
    trace.customdata = updates.hover_bounds;
    _rebuildHoverCells(logicalUid, updates.hover_bounds);
  }
  if (trace.type === 'bar' && forceBarOffsetgroup) {
    trace.offsetgroup = logicalUid;
    trace.alignmentgroup = 'fv-bars';
  }
  if (_resetTreemapLevel && template.type === 'treemap') { trace.level = 'root'; }
  if (applyLineGaps && Array.isArray(trace.x) && Array.isArray(trace.y)) {
    const gapped = fvApplyLineGaps(trace.x, trace.y, true);
    trace.x = gapped.x;
    trace.y = gapped.y;
  }
  applyHeatmapColorbarPolicy(trace, renderLayer, showForeground);
  applyLegendVisibility(trace, logicalUid);
  return trace;
}

function rememberPlotlyVisibility(figIdx) {
  const gd = divs[figIdx];
  for (const trace of ((gd && gd.data) || [])) {
    const logicalUid = stripLayerSuffix((trace && trace.uid) || '');
    if (!logicalUid) continue;
    legendVisibilityByUid[logicalUid] =
      trace.visible === 'legendonly' ? 'legendonly' : true;
  }
}

function applyLegendVisibility(trace, logicalUid) {
  if (!Object.prototype.hasOwnProperty.call(legendVisibilityByUid, logicalUid)) return;
  if (legendVisibilityByUid[logicalUid] === 'legendonly') {
    trace.visible = 'legendonly';
  } else {
    delete trace.visible;
  }
}

function buildGroupedPlotlyChildren(figUid, parentUid, childResults, renderLayer, opacity, showlegend, forceBarOffsetgroup = false, showForeground = false) {
  const parentSpec = figSpecByUid[figUid] && figSpecByUid[figUid].traces.find(ts => ts.uid === parentUid);
  if (!parentSpec) return [];
  const applyLineGaps = parentSpec.trace_type === 'line'
    && !(parentSpec.params && parentSpec.params.add_gaps === false);
  return childResults.map(cr => {
    const color = ensureGroupColor(parentSpec, cr.group_value_key);
    const template = makePlotlyTrace(parentSpec, cr.uid, cr.group_value_key, color);
    return buildTraceFromTemplate(template, cr.uid, renderLayer, cr.updates || {}, opacity, showlegend, forceBarOffsetgroup, showForeground, applyLineGaps);
  }).filter(Boolean);
}

function buildTracesForFigure(figUid) {
  const figSpec = figSpecByUid[figUid];
  const overlayMode = (DASHBOARD_SPEC.state && DASHBOARD_SPEC.state.cross_filter_mode) === 'overlay';
  const selections = (DASHBOARD_SPEC.state && DASHBOARD_SPEC.state.selections) || [];
  const hasSelections = selections.length > 0;
  const sourceFigure = figureHasSelectionSource(figUid, selections);
  // A live cube gesture drives the fg layer before any selection is
  // committed (contract F) — fvCubeOverlayFgActive covers that window.
  const showForeground = overlayMode && !sourceFigure
    && (hasSelections
        || (typeof fvCubeOverlayFgActive === 'function' && fvCubeOverlayFgActive(figUid)));
  const backgroundOpacity = showForeground ? OVERLAY_BG_OPACITY : 1;
  const bgDataLayer = backgroundDataLayerForFigure(figUid);
  const traces = [];
  for (const ts of figSpec.traces) {
    const barMode = (ts.display && ts.display.bar_mode) || (ts.params && ts.params.bar_mode) || 'group';
    const forceBarOffsetgroup = showForeground && barMode !== 'stack';
    if (isGroupedParent(ts)) {
      const childLayers = groupedDataByParent[figUid][ts.uid] || { base: [], bg: [], fg: [] };
      if (overlayMode) {
        traces.push(...buildGroupedPlotlyChildren(figUid, ts.uid, childLayers[bgDataLayer] || [], 'bg', backgroundOpacity, true, forceBarOffsetgroup, showForeground));
        if (showForeground) {
          traces.push(...buildGroupedPlotlyChildren(figUid, ts.uid, childLayers.fg || [], 'fg', 1, false, forceBarOffsetgroup, showForeground));
        }
      } else {
        traces.push(...buildGroupedPlotlyChildren(figUid, ts.uid, childLayers.base || [], 'base', 1, true));
      }
    } else {
      const template = traceTemplateByUid[ts.uid];
      const layers = ensureLayerData(ts.uid);
      const applyLineGaps = ts.trace_type === 'line'
        && !(ts.params && ts.params.add_gaps === false);
      if (overlayMode) {
        const bgTrace = buildTraceFromTemplate(template, ts.uid, 'bg', layers[bgDataLayer] || {}, backgroundOpacity, true, forceBarOffsetgroup, showForeground, applyLineGaps);
        if (bgTrace) traces.push(bgTrace);
        if (showForeground) {
          const fgTrace = buildTraceFromTemplate(template, ts.uid, 'fg', layers.fg || {}, 1, false, forceBarOffsetgroup, showForeground, applyLineGaps);
          if (fgTrace) traces.push(fgTrace);
        }
      } else {
        const baseTrace = buildTraceFromTemplate(template, ts.uid, 'base', layers.base || {}, 1, true, false, false, applyLineGaps);
        if (baseTrace) traces.push(baseTrace);
      }
    }
  }
  syncHeatmapOverlayColorScale(traces, figSpec, showForeground);
  return traces;
}

function baseBarmodeForFigure(figSpec) {
  if (!figSpec) return null;
  const barTs = figSpec.traces.find(ts => ts.trace_type === 'bar');
  if (barTs) {
    return (barTs.display && barTs.display.bar_mode)
      || (barTs.params && barTs.params.bar_mode)
      || 'group';
  }
  return figSpec.traces.some(ts => ts.trace_type === 'histogram') ? 'group' : null;
}

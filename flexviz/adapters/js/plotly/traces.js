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

// Only Histogram2D and GeoHistogram2D carry color_norm; a missing one is linear.
function heatmapColorNorm(ts) {
  return (ts && ts.display && ts.display.color_norm) || 'linear';
}

function heatmapColorRange(ts) {
  const display = (ts && ts.display) || {};
  if (!Object.prototype.hasOwnProperty.call(display, 'color_range')) {
    throw new Error('Generated heatmap specs must include explicit color_scale and color_range defaults.');
  }
  const range = display.color_range;
  // A fixed range is in data units; a log norm colors in log10 space.
  if (range !== 'auto' && heatmapColorNorm(ts) === 'log') {
    return [Math.log10(range[0]), Math.log10(range[1])];
  }
  return range;
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

// Colorbar ticks for a log10 color axis, labelled in data units (1, 10, 1K).
function logColorbarTicks(lo, hi) {
  const ticks = [];
  for (let e = Math.floor(lo); e <= Math.ceil(hi); e++) {
    for (const m of [1, 2, 5]) {
      const v = e + Math.log10(m);
      if (v >= lo - 1e-9 && v <= hi + 1e-9) ticks.push({ v, decade: m === 1 });
    }
  }
  // Decades alone once there are enough of them; 1-2-5 steps fill narrow ranges,
  // and a range too narrow for two of those gets its ends labelled.
  const decades = ticks.filter(t => t.decade);
  let chosen = decades.length >= 3 ? decades : ticks;
  if (chosen.length < 2) chosen = [{ v: lo }, ...chosen, { v: hi }];
  const format = new Intl.NumberFormat('en', { notation: 'compact', maximumSignificantDigits: 2 });
  const tickvals = [];
  const ticktext = [];
  for (const t of chosen) {
    const text = format.format(10 ** t.v);
    if (text === ticktext[ticktext.length - 1]) continue;
    tickvals.push(t.v);
    ticktext.push(text);
  }
  return { tickvals, ticktext };
}

// Log color norm: color by log10(value), keep the raw value for hover, and
// label the colorbar in data units. A value <= 0 has no log and is not drawn.
function applyLogColorNorm(trace) {
  const toLog = v => (v != null && v > 0 ? Math.log10(v) : null);
  const raw = trace.z || [];
  trace.text = raw;
  trace.z = raw.map(v => (Array.isArray(v) ? v.map(toLog) : toLog(v)));
  trace.hovertemplate = trace.type === 'choroplethmap'
    ? '%{location}<br>%{text:.10~r}<extra></extra>'
    : 'x: %{x}<br>y: %{y}<br>z: %{text:.10~r}<extra></extra>';
  // A fixed range is already on the template. An auto range is pinned to the
  // drawn cells, so the colorbar ticks match the colors Plotly draws.
  // z is 2-D for a heatmap and flat for a choropleth.
  if (trace.zmin == null) {
    const extent = heatmapZFiniteExtent([trace.z.flat()]);
    if (!extent) return;
    [trace.zmin, trace.zmax] = extent;
  }
  trace.colorbar = { ...trace.colorbar, ...logColorbarTicks(trace.zmin, trace.zmax) };
}

function applyHeatmapColorbarPolicy(trace, renderLayer, showForeground) {
  if (!isHeatmapScaledTrace(trace)) return;
  if (!showForeground) {
    trace.showscale = true;
    return;
  }
  // During Plotly.react keep the bg colorbar; swap to fg-only after render
  // completes (see fvFinalizeHeatmapOverlayColorbars).
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
  return fvRunProgrammaticPlotlyOp(figUid, restyle);
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
 * Bin bounds for hover, derived from the per-axis [lo, step, n] triples the
 * server sends (one triple per axis instead of one object per bin/cell).
 * Every edge is lo + i * step, the arithmetic the server bins with.
 * @param {object} updates
 * @returns {Array|null} 1D [{x0,x1}|{y0,y1}, ...] or 2D [[{x0,x1,y0,y1}, ...], ...]
 */
function _hoverBoundsFromEdges(updates) {
  const xe = updates && updates.x_edges;
  const ye = updates && updates.y_edges;
  if (!xe && !ye) return null;
  if (xe && ye) {
    // 2D: outer = row (y bin), inner = col (x bin), matching z.
    const rows = [];
    for (let r = 0; r < ye[2]; r++) {
      const row = [];
      for (let c = 0; c < xe[2]; c++) {
        row.push({
          x0: xe[0] + c * xe[1], x1: xe[0] + (c + 1) * xe[1],
          y0: ye[0] + r * ye[1], y1: ye[0] + (r + 1) * ye[1],
        });
      }
      rows.push(row);
    }
    return rows;
  }
  // 1D: x= histogram bins on x, y= (horizontal) histogram bins on y.
  const e = xe || ye;
  const lo = xe ? 'x0' : 'y0';
  const hi = xe ? 'x1' : 'y1';
  const out = [];
  for (let i = 0; i < e[2]; i++) {
    out.push({ [lo]: e[0] + i * e[1], [hi]: e[0] + (i + 1) * e[1] });
  }
  return out;
}

/**
 * One GeoJSON rectangle per non-empty cell of a geo histogram, from the two
 * [lo, step, n] triples and the flat z the server sends (z[j * nbLat + i],
 * null = empty cell). The map selection reads these rings back to build its
 * lon/lat box, so the ids and the ring order are part of the contract.
 * @param {object} updates
 * @returns {{geojson: object, locations: Array<string>, z: Array<number>}}
 */
function _geoRectanglesFromEdges(updates) {
  const [latLo, latStep, nbLat] = updates.lat_edges;
  const [lonLo, lonStep, nbLon] = updates.lon_edges;
  const zFlat = updates.z || [];
  const features = [];
  const locations = [];
  const z = [];
  for (let j = 0; j < nbLon; j++) {
    const lonLeft = lonLo + j * lonStep;
    const lonRight = lonLo + (j + 1) * lonStep;
    for (let i = 0; i < nbLat; i++) {
      const value = zFlat[j * nbLat + i];
      if (value === null || value === undefined) continue;
      const latBottom = latLo + i * latStep;
      const latTop = latLo + (i + 1) * latStep;
      const id = 'r' + i + '_c' + j;
      locations.push(id);
      z.push(value);
      features.push({
        type: 'Feature',
        id,
        geometry: {
          type: 'Polygon',
          coordinates: [[
            [lonLeft, latBottom],
            [lonRight, latBottom],
            [lonRight, latTop],
            [lonLeft, latTop],
            [lonLeft, latBottom],
          ]],
        },
      });
    }
  }
  return { geojson: { type: 'FeatureCollection', features }, locations, z };
}

/**
 * Flatten hover bounds into hoverCellsByTraceUid for cell matching.
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
    // Bin-edge triples are expanded below, not copied onto the trace.
    if (k === 'x_edges' || k === 'y_edges') continue;
    if (k === 'lat_edges' || k === 'lon_edges') continue;
    trace[k] = v;
  }
  if (updates && updates.lat_edges && updates.lon_edges) {
    Object.assign(trace, _geoRectanglesFromEdges(updates));
  }
  const hoverBounds = _hoverBoundsFromEdges(updates);
  if (hoverBounds) {
    trace.customdata = hoverBounds;
    _rebuildHoverCells(logicalUid, hoverBounds);
  }
  if (trace.type === 'bar' && forceBarOffsetgroup) {
    trace.offsetgroup = logicalUid;
    trace.alignmentgroup = 'fv-bars';
  }
  if (applyLineGaps && Array.isArray(trace.x) && Array.isArray(trace.y)) {
    const gapped = fvApplyLineGaps(trace.x, trace.y, true);
    trace.x = gapped.x;
    trace.y = gapped.y;
  }
  if (heatmapColorNorm(traceSpecByUid[logicalUid]) === 'log') applyLogColorNorm(trace);
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

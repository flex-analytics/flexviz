// === ECharts adapter — rendering and viewport sync ===
// Requires: series.js, state.js

function heatmapFiniteExtent(z) {
  let vMin = Infinity;
  let vMax = -Infinity;
  for (const row of (z || [])) {
    for (const value of (row || [])) {
      if (typeof value !== 'number' || !Number.isFinite(value)) continue;
      if (value < vMin) vMin = value;
      if (value > vMax) vMax = value;
    }
  }
  if (vMin === Infinity || vMax === -Infinity) return [0, 1];
  if (vMin === vMax) {
    const pad = Math.abs(vMin) * 0.01 || 1;
    return [vMin - pad, vMax + pad];
  }
  return [vMin, vMax];
}

function heatmapVisualMapForTrace(ts, updates) {
  const colorRange = heatmapColorRange(ts);
  const extent = colorRange === 'auto'
    ? heatmapFiniteExtent((updates && updates.z) || [])
    : colorRange;
  return {
    min: extent[0],
    max: extent[1],
    calculable: true,
    inRange: { color: heatmapColors(ts) },
  };
}

function boxAxisPatchForFigure(figUid) {
  const figSpec = figSpecByUid[figUid];
  if (!figSpec) return null;
  const boxTraces = (figSpec.traces || []).filter(ts => ts.trace_type === 'box');
  if (!boxTraces.length) return null;

  const categories = [];
  const seen = new Set();
  let orientation = null;

  function rememberLabel(updates) {
    if (!updates) return;
    if (typeof updates.orientation === 'string' && !orientation) {
      orientation = updates.orientation;
    }
    const label = updates.x0 ?? updates.y0;
    if (label == null) return;
    const key = String(label);
    if (seen.has(key)) return;
    seen.add(key);
    categories.push(key);
  }

  for (const ts of boxTraces) {
    if (isGroupedParent(ts)) {
      const grouped = groupedDataByParent[figUid][ts.uid] || { base: [], bg: [], fg: [] };
      for (const layer of ['base', 'bg', 'fg']) {
        for (const child of (grouped[layer] || [])) {
          rememberLabel(child && child.updates);
        }
      }
    } else {
      const layers = ensureLayerData(ts.uid);
      for (const layer of ['base', 'bg', 'fg']) {
        rememberLabel(layers[layer]);
      }
    }
  }

  if (!categories.length) return null;
  if (orientation === 'h') {
    return { yAxis: { type: 'category', data: categories } };
  }
  return { xAxis: { type: 'category', data: categories } };
}

function isTemporalAxisValue(value) {
  if (value instanceof Date) return Number.isFinite(value.valueOf());
  if (typeof value !== 'string') return false;
  if (!/[A-Za-zT:\-\/]/.test(value)) return false;
  return Number.isFinite(Date.parse(value));
}

function timeAxisPatchForSeries(series) {
  for (const item of (series || [])) {
    const data = (item && item.data) || [];
    for (const point of data) {
      const xValue = Array.isArray(point)
        ? point[0]
        : (point && Array.isArray(point.value) ? point.value[0] : null);
      if (isTemporalAxisValue(xValue)) {
        return { xAxis: { type: 'time', min: 'dataMin', max: 'dataMax' } };
      }
      if (xValue != null) return null;
    }
  }
  return null;
}

function brushAreasForFigure(figUid) {
  const figSpec = figSpecByUid[figUid];
  const sel = window.fvFigureSelection?.(
    figUid,
    (DASHBOARD_SPEC.state && DASHBOARD_SPEC.state.selections) || []
  );
  const areas = [];
  if (!sel || !sel.predicates || !figSpec) return areas;
  const cartesian = (figSpec.traces || []).find(
    ts => Array.isArray(ts.axes) && ts.axes.length > 0 && ts.backend_data
  );
  if (!cartesian) return areas;
  const { xCol, yCol } = window.fvColsForTrace?.(cartesian) || {};
  for (const pred of (sel.predicates || [])) {
    const xClause = (pred.clauses || []).find(clause => clause.column === xCol && clause.range);
    const yClause = (pred.clauses || []).find(clause => clause.column === yCol && clause.range);
    if (!xClause && !yClause) continue;
    areas.push({
      brushType: 'rect',
      coordRange: yClause ? [xClause ? xClause.range : null, yClause.range].filter(Boolean) : [xClause.range],
      xAxisIndex: 0,
      yAxisIndex: 0,
    });
  }
  return areas;
}

function syncBrushAreasForFigure(figUid) {
  const chart = chartsByFig[figUid];
  if (!chart) return;
  const areas = brushAreasForFigure(figUid);
  _brushAreasByFig[figUid] = areas;
  try {
    chart.dispatchAction({ type: 'brush', areas });
  } catch (e) {
    // ECharts may throw when clearing brush on certain chart states (e.g. empty areas
    // after a brush-select was active). Swallow to keep the render pipeline alive.
  }
}

// Guard flag: set while calling setOption so the datazoom event fired by
// the dataZoom reset inside setOption is ignored (prevents feedback loop).
let _applyingDeltas = false;

function _fvRenderFigure(figUid) {
  const chart = chartsByFig[figUid];
  if (!chart) return;
  _applyingDeltas = true;
  const series = buildSeriesForFigure(figUid);
  const axisPatch = {};
  const opt = chart.getOption();
  const xAxis = Array.isArray(opt.xAxis) ? opt.xAxis[0] : opt.xAxis;
  if (xAxis && xAxis.type === 'value') {
    axisPatch.xAxis = { min: 'dataMin', max: 'dataMax' };
  }
  // Heatmap axes: update category data from first heatmap trace's updates
  const figSpec = figSpecByUid[figUid];
  if (figSpec) {
    for (const ts of figSpec.traces) {
      if (ts.trace_type === 'histogram2d' || ts.trace_type === 'corr_heatmap') {
        const layers = layerDataByUid[ts.uid];
        const u = (layers && (layers.base || layers.fg || layers.bg)) || {};
        if (u.x && u.y) {
          axisPatch.xAxis = { type: 'category', data: u.x.map(String) };
          axisPatch.yAxis = { type: 'category', data: u.y.map(String) };
          axisPatch.visualMap = heatmapVisualMapForTrace(ts, u);
        }
        break;
      }
    }
    const boxAxisPatch = boxAxisPatchForFigure(figUid);
    if (boxAxisPatch) {
      Object.assign(axisPatch, boxAxisPatch);
    }
    const timeAxisPatch = timeAxisPatchForSeries(series);
    if (timeAxisPatch) {
      Object.assign(axisPatch, timeAxisPatch);
    }
  }

  // Y-axis anchoring: pin to bg extent when overlay fg is visible
  const overlayMode = (DASHBOARD_SPEC.state && DASHBOARD_SPEC.state.cross_filter_mode) === 'overlay';
  const selections = (DASHBOARD_SPEC.state && DASHBOARD_SPEC.state.selections) || [];
  const sourceFigure = figureHasSelectionSource(figUid, selections);
  const hasFg = overlayMode && selections.length > 0 && !sourceFigure;
  if (hasFg && bgYExtentByFig[figUid]) {
    const ext = bgYExtentByFig[figUid];
    axisPatch.yAxis = { min: ext[0], max: ext[1] };
  }

  const displayRanges = {
    ...(window.fvAxisLockRangesForFigure?.(figUid) || {}),
    ...figureViewportRanges(figUid),
  };
  const xLocked = window.fvIsAxisLocked?.(figUid, 'x') === true;
  const dataZoom = displayRanges.x
    ? [{
        startValue: displayRanges.x[0],
        endValue: displayRanges.x[1],
        disabled: xLocked,
        zoomOnMouseWheel: !xLocked,
        moveOnMouseWheel: !xLocked,
        moveOnMouseMove: !xLocked,
      }]
    : [{
        start: 0,
        end: 100,
        disabled: xLocked,
        zoomOnMouseWheel: !xLocked,
        moveOnMouseWheel: !xLocked,
        moveOnMouseMove: !xLocked,
      }];

  chart.setOption(
    { series, dataZoom, ...axisPatch },
    { replaceMerge: ['series'] }
  );
  _applyingDeltas = false;
  syncBrushAreasForFigure(figUid);
  window.fvUpdateAxisLockButtons?.(figUid);
}

function _fvResetRendererCache() {
  for (const figUid of FIG_UIDS) {
    _brushAreasByFig[figUid] = [];
  }
}

window.fvCaptureAxisDisplayRanges = function(figUid, axisFamily) {
  if (String(axisFamily || '').charAt(0) !== 'x') return {};
  const chart = chartsByFig[figUid];
  if (!chart) return {};
  const opt = chart.getOption();
  const dz = opt.dataZoom && opt.dataZoom[0];
  if (!dz) return {};
  if (dz.startValue !== undefined && dz.endValue !== undefined) {
    return { x: [dz.startValue, dz.endValue] };
  }
  return {};
};

window.fvHasLockableCurrentAxis = function(figUid, axisFamily) {
  return Object.keys(window.fvCaptureAxisDisplayRanges?.(figUid, axisFamily) || {}).length > 0;
};

window.fvApplyAxisLocks = function(figUid) {
  const chart = chartsByFig[figUid];
  if (!chart) return;
  const ranges = {
    ...(window.fvAxisLockRangesForFigure?.(figUid) || {}),
    ...figureViewportRanges(figUid),
  };
  const xLocked = window.fvIsAxisLocked?.(figUid, 'x') === true;
  if (!ranges.x) {
    chart.setOption({
      dataZoom: [{
        start: 0,
        end: 100,
        disabled: xLocked,
        zoomOnMouseWheel: !xLocked,
        moveOnMouseWheel: !xLocked,
        moveOnMouseMove: !xLocked,
      }],
    });
    return;
  }
  chart.setOption({
    dataZoom: [{
      startValue: ranges.x[0],
      endValue: ranges.x[1],
      disabled: xLocked,
      zoomOnMouseWheel: !xLocked,
      moveOnMouseWheel: !xLocked,
      moveOnMouseMove: !xLocked,
    }],
  });
};

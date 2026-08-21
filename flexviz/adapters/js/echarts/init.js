// === ECharts adapter — startup and event wiring ===
// Requires: series.js, render.js

// Renderer-specific resize callback used by gridstack-bridge.js on resizestop.
window._fvResizeChart = function(el) {
  var c = el.querySelector('[id^="fv-chart-"]');
  if (!c) return;
  var idx = parseInt(c.id.replace('fv-chart-', ''));
  var uid = FIG_UIDS[idx];
  if (uid && chartsByFig[uid]) chartsByFig[uid].resize();
};

const figureModesByUid = Object.fromEntries(
  FIG_UIDS.map((figUid, idx) => [figUid, figSupportsZoomPan[idx] ? 'zoom' : 'select'])
);

function axisOptionForFamily(chart, axisFamily) {
  const option = chart && chart.getOption && chart.getOption();
  if (!option) return null;
  const key = axisFamily === 'y' ? 'yAxis' : 'xAxis';
  const axis = option[key];
  return Array.isArray(axis) ? (axis[0] || null) : (axis || null);
}

function normalizeEChartsRangeForAxis(chart, axisFamily, range) {
  if (!Array.isArray(range)) return range;
  const axis = axisOptionForFamily(chart, axisFamily);
  if (!axis || axis.type !== 'time') return range;
  return range.map(value => {
    const num = Number(value);
    if (!Number.isFinite(num)) return value;
    return new Date(num).toISOString();
  });
}

function clearFigureSelection(figUid) {
  const remainingSelections = window.fvClearFigureSelectionFromList?.(
    figUid,
    DASHBOARD_SPEC.state.selections || []
  ) || [];
  window.fvSetSelectionState?.(remainingSelections);
  _brushAreasByFig[figUid] = [];
  postDashboardUpdate({
    type: remainingSelections.length ? 'selection' : 'deselect',
    axis_ranges: {},
    selections: remainingSelections,
    force_update: true,
    figure_uid: figUid,
  });
}

function handleEChartsClick(params, figUid) {
  const figSpec = figSpecByUid[figUid];
  if (!figSpec) return;
  const logicalUid = stripLayerSuffix((params && params.seriesId) || '');
  const tsSpec = (figSpec.traces || []).find(ts => ts.uid === logicalUid);
  if (!tsSpec) return;

  let clauses = null;
  if (params.seriesType === 'pie') {
    const labels = tsSpec.backend_data && tsSpec.backend_data.labels;
    if (!labels) return;
    const labelCols = Array.isArray(labels) ? labels : [labels];
    const clicked = params.name;
    if (clicked == null) return;
    let parts;
    if (labelCols.length === 1) {
      parts = [String(clicked)];
    } else {
      try { parts = JSON.parse(String(clicked)); }
      catch (e) { return; }
      if (!Array.isArray(parts) || parts.length !== labelCols.length) return;
    }
    clauses = labelCols.map((column, idx) => ({ column, values: [parts[idx]] }));
  } else if (params.seriesType === 'treemap') {
    const path = (tsSpec.params && tsSpec.params.path) || [];
    const nodeId = String((params.data && params.data.id) || '');
    const parts = nodeId.split('/').slice(1).map(value => decodeURIComponent(String(value)));
    if (!parts.length) {
      if (window.fvFigureSelection?.(figUid, DASHBOARD_SPEC.state.selections || [])) {
        clearFigureSelection(figUid);
      }
      return;
    }
    clauses = parts.map((value, idx) => ({ column: path[idx], values: [value] }));
  }

  if (!clauses) return;
  const predicate = { clauses };
  const existing = window.fvFigureSelection?.(figUid, DASHBOARD_SPEC.state.selections || []);
  const newPredicates = window.fvUpsertPathPredicate?.(
    existing ? (existing.predicates || []) : [],
    predicate
  ) || [];
  if (!newPredicates.length) {
    if (existing) clearFigureSelection(figUid);
    return;
  }
  const nextSelection = { predicates: newPredicates };
  const nextSelections = window.fvReplaceFigureSelection?.(
    figUid,
    nextSelection,
    DASHBOARD_SPEC.state.selections || []
  ) || [];
  window.fvSetSelectionState?.(nextSelections);
  postDashboardUpdate({
    type: 'selection',
    axis_ranges: {},
    selections: nextSelections,
    force_update: true,
    figure_uid: figUid,
  });
}

function updateModeIndicator(figUid) {
  const figIdx = figUidToIdx[figUid];
  if (figIdx === undefined) return;
  const controls = window.fvPanelControlRoot?.(figIdx);
  if (!controls) return;
  const currentMode = figureModesByUid[figUid] || 'zoom';
  const supportsZoomPan = figSupportsZoomPan[figIdx];
  const axesLocked = window.fvAreCurrentAxesLocked?.(figUid) === true;
  for (const btn of controls.querySelectorAll('.fv-mode-btn')) {
    const mode = btn.dataset.mode;
    btn.classList.toggle('mode-active', mode === currentMode);
    btn.disabled = mode === 'pan'
      || (!supportsZoomPan && mode !== 'select')
      || (axesLocked && mode === 'zoom');
  }
  for (const btn of controls.querySelectorAll('.fv-mode-action-btn[data-action="reset-panel"]')) {
    btn.disabled = !supportsZoomPan;
  }
  window.fvUpdateAxisLockButtons?.(figUid);
  const warn = controls.querySelector('.fv-mode-warn');
  if (warn) warn.textContent = axesLocked ? 'axes locked' : '';
}

function setFigureMode(figUid, mode) {
  const figIdx = figUidToIdx[figUid];
  if (figIdx === undefined || mode === 'pan') return;
  if (window.fvAreCurrentAxesLocked?.(figUid) && mode === 'zoom') return;
  const chart = chartsByFig[figUid];
  if (!chart) return;
  figureModesByUid[figUid] = mode;
  if (mode === 'select') {
    chart.dispatchAction({
      type: 'takeGlobalCursor',
      key: 'brush',
      brushOption: { brushType: 'rect', brushMode: 'single' },
    });
  } else {
    chart.dispatchAction({
      type: 'takeGlobalCursor',
      key: 'brush',
      brushOption: { brushType: false },
    });
  }
  updateModeIndicator(figUid);
}

window.fvSyncFigureModeForAxisLocks = function(figUid) {
  if (window.fvAreCurrentAxesLocked?.(figUid) && (figureModesByUid[figUid] || 'zoom') === 'zoom') {
    setFigureMode(figUid, 'select');
    return;
  }
  updateModeIndicator(figUid);
};

// ---- initialise one ECharts instance per figure -----------------
FIG_UIDS.forEach((figUid, fi) => {
  const container = document.getElementById('fv-chart-' + fi);
  const chart = echarts.init(container);
  chart.setOption(INITIAL_OPTIONS[figUid]);
  chartsByFig[figUid] = chart;

  // Resize when container changes.
  const ro = new ResizeObserver(() => chart.resize());
  ro.observe(container);

  // ---- datazoom: map to viewport event -------------------------
  // Use a debounce to avoid flooding the backend during scroll zoom.
  // _applyingDeltas guards against re-entrancy: applyDeltasToFig calls
  // setOption (which resets dataZoom), which would otherwise fire another
  // datazoom event and undo the zoom.
  let dzTimer = null;
  chart.on('datazoom', function(params) {
    if (_applyingDeltas) return;
    clearTimeout(dzTimer);
    dzTimer = setTimeout(() => {
      let startVal, endVal;
      if (params.startValue !== undefined) {
        startVal = params.startValue;
        endVal   = params.endValue;
      } else if (params.batch && params.batch.length) {
        startVal = params.batch[0].startValue;
        endVal   = params.batch[0].endValue;
      }
      if (startVal === undefined || endVal === undefined) {
        const opt = chart.getOption();
        const dz = opt.dataZoom && opt.dataZoom[0];
        if (dz) { startVal = dz.startValue; endVal = dz.endValue; }
      }
      if (startVal === undefined || endVal === undefined) return;
      if (window.fvIsAxisLocked?.(figUid, 'x')) {
        window.fvApplyAxisLocks?.(figUid);
        return;
      }
      if (!DASHBOARD_SPEC.state) DASHBOARD_SPEC.state = {};
      if (!DASHBOARD_SPEC.state.viewport) DASHBOARD_SPEC.state.viewport = {};
      // Write viewport state before posting.
      DASHBOARD_SPEC.state.viewport[figUid + '/x'] = { min: startVal, max: endVal };
      window.fvUpdateAxisLockButtons?.(figUid);
      postDashboardUpdate({
        type: 'viewport',
        axis_ranges: { x: [startVal, endVal] },
        selections: DASHBOARD_SPEC.state.selections,
        force_update: false,
        figure_uid: figUid,
      });
    }, 150);
  });

  // ---- brushEnd: map to selection event -------------------------
  // brushEnd fires once when the user finishes drawing a brush area,
  // with complete coordRange for the selected cartesian region.
  chart.on('brushEnd', function(params) {
    const areas = params.areas;
    if (!areas || !areas.length) {
      clearFigureSelection(figUid);
      return;
    }
    const coordRange = areas[0].coordRange;
    if (!coordRange || !coordRange[0]) return;
    const xRange = normalizeEChartsRangeForAxis(chart, 'x', coordRange[0]);
    const yRange = (coordRange.length > 1 && coordRange[1])
      ? normalizeEChartsRangeForAxis(chart, 'y', coordRange[1])
      : null;
    const figSpec = figSpecByUid[figUid];
    const cartesian = (figSpec && figSpec.traces || []).filter(
      ts => Array.isArray(ts.axes) && ts.axes.length > 0 && ts.backend_data
    );
    const seen = new Set();
    const predicates = [];
    for (const ts of cartesian) {
      const { xCol, yCol } = window.fvColsForTrace?.(ts) || {};
      const key = JSON.stringify([xCol, yCol]);
      if (seen.has(key)) continue;
      seen.add(key);
      const clauses = [];
      if (xCol && xRange) clauses.push({ column: xCol, range: xRange });
      if (yCol && yRange) clauses.push({ column: yCol, range: yRange });
      if (clauses.length) predicates.push({ clauses });
    }
    if (!predicates.length) return;

    const nextSelections = window.fvReplaceFigureSelection?.(
      figUid,
      { predicates },
      DASHBOARD_SPEC.state.selections || []
    ) || [];
    window.fvSetSelectionState?.(nextSelections);
    _brushAreasByFig[figUid] = areas;
    postDashboardUpdate({
      type: 'selection',
      axis_ranges: {},
      selections: nextSelections,
      force_update: true,
      figure_uid: figUid,
    });
  });

  if ((figSpecByUid[figUid].traces || []).some(ts => ts.trace_type === 'pie' || ts.trace_type === 'treemap')) {
    chart.on('click', function(params) { handleEChartsClick(params, figUid); });
  }

  chart.on('mouseover', function(params) { handleEChartsHover(params, figUid); });
  chart.on('mouseout',  function()       { handleEChartsUnhover(); });
});

function showEChartsCrosshair(figUid, axis, value) {
  const chart = chartsByFig[figUid];
  if (!chart) return;
  chart.dispatchAction({
    type: 'updateAxisPointer',
    currTrigger: 'mousemove',
    dataByCoordSys: [{
      dataByAxis: [{ axisDim: axis, axisIndex: 0, value }],
    }],
  });
}

function clearAllEChartsCrosshairs() {
  for (const figUid of FIG_UIDS) {
    const chart = chartsByFig[figUid];
    if (chart) {
      chart.dispatchAction({ type: 'hideTip' });
      chart.dispatchAction({ type: 'updateAxisPointer', currTrigger: 'leave' });
    }
  }
}
window.fvClearAllCrosshairs = clearAllEChartsCrosshairs;
window.fvClearAllHoverVisuals = clearAllEChartsCrosshairs;

function _fvShowCrosshair(figUid, axis, value) {
  showEChartsCrosshair(figUid, axis, value);
}
function _fvClearAllCrosshairs() { clearAllEChartsCrosshairs(); }

window.__fvApplyHoverVisuals = function(figUid, visuals) {
  for (const visual of (visuals || [])) {
    if (visual.type === 'x_guide') {
      showEChartsCrosshair(figUid, 'x', visual.value);
    } else if (visual.type === 'y_guide') {
      showEChartsCrosshair(figUid, 'y', visual.value);
    }
  }
};

function handleEChartsHover(params, sourceFigUid) {
  const mode = getHoverMode(DASHBOARD_SPEC);
  if (mode === 'off') return;
  const rawUid = params.seriesId || '';
  const logicalUid = stripLayerSuffix(rawUid);
  const resolvedUid = childUidToParentUid[logicalUid] || logicalUid;
  const ts = traceSpecByUid[resolvedUid];
  if (!ts || !ts.hover || !ts.hover.source_modes || !ts.hover.source_modes.length) return;

  const values = {};
  const columns = {};
  if (Array.isArray(params.value)) {
    if (params.value[0] !== undefined && params.value[0] !== null && ts.backend_data && ts.backend_data.x) {
      values.x = params.value[0];
      columns.x = ts.backend_data.x;
    }
    if (params.value[1] !== undefined && params.value[1] !== null && ts.backend_data && ts.backend_data.y) {
      values.y = params.value[1];
      columns.y = ts.backend_data.y;
    }
  }
  if (!Object.keys(columns).length) return;

  clearAllEChartsCrosshairs();
  const event = {
    sourceFigUid,
    sourceTraceUid: rawUid,
    kind: 'point',
    values,
    columns,
    bounds: {},
    coordSpace: 'cartesian',
    key: null,
  };
  const visualsByFig = planHoverVisuals(
    event, mode, hoverTargetsByColumn, hoverSourceByTrace, hoverCellsByTraceUid,
    { implementedAxisBandTargetTraceTypes: IMPLEMENTED_AXIS_BAND_TARGET_TRACE_TYPES }
  );
  for (const [figUid, visuals] of visualsByFig) {
    dispatchHoverVisuals(figUid, visuals);
  }
}

function handleEChartsUnhover() {
  clearAllEChartsCrosshairs();
}

// ECharts-specific hook overrides (placed after shared hooks from toolbar.js)
window.fvOnReset = async function() {
  window.fvClearUnlockedViewports?.();
  window.fvSetSelectionState?.([]);
  window.fvResetRuntimeCache?.();
  await postDashboardUpdate({type: 'init', axis_ranges: {}, selections: [], force_update: true});
};
window.fvOnDeselect = async function() {
  window.fvSetSelectionState?.([]);
  await postDashboardUpdate({type: 'deselect', axis_ranges: {}, selections: [], force_update: true});
};

(async () => {
  for (const figUid of FIG_UIDS) {
    const figIdx = figUidToIdx[figUid];
    if (figIdx === undefined) continue;
    window.fvBindPanelControls?.(figUid, {
      setMode(mode) { setFigureMode(figUid, mode); },
      resetPanel() { window.fvOnResetPanel?.(figUid); },
      toggleAxisLocks() { window.fvOnToggleAxisLocks?.(figUid); },
    });
    updateModeIndicator(figUid);
  }

  await restoreDashboardFromSpec();

  for (const figUid of FIG_UIDS) {
    updateModeIndicator(figUid);
  }
})();

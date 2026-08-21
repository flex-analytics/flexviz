// === Plotly adapter — rendering and viewport sync ===
// Requires: traces.js, state.js

function plotlyAxisKey(axId) {
  return axId.replace(/^(x|y)(\d*)$/, '$1axis$2');
}

function plotlyAxisId(layoutKey) {
  return layoutKey.replace(/^(x|y)axis(\d*)$/, '$1$2');
}

function fvRunProgrammaticPlotlyOp(operation) {
  _programmaticOp = true;
  const result = operation?.();
  return Promise.resolve(result).finally(() => {
    _programmaticOp = false;
  });
}

window.fvCaptureAxisDisplayRanges = function(figUid, axisFamily) {
  const figIdx = figUidToIdx[figUid];
  if (figIdx === undefined) return {};
  const family = String(axisFamily || '').charAt(0);
  if (family !== 'x' && family !== 'y') return {};
  const gd = divs[figIdx];
  const fullLayout = (gd && gd._fullLayout) || {};
  const out = {};
  for (const [layoutKey, axisObj] of Object.entries(fullLayout)) {
    const match = /^(x|y)axis(\d*)$/.exec(layoutKey);
    if (!match || match[1] !== family) continue;
    const axId = plotlyAxisId(layoutKey);
    const range = axisObj && axisObj.range;
    if (Array.isArray(range) && range.length === 2) {
      out[axId] = [range[0], range[1]];
    }
  }
  return out;
};

window.fvHasLockableCurrentAxis = function(figUid, axisFamily) {
  return Object.keys(window.fvCaptureAxisDisplayRanges?.(figUid, axisFamily) || {}).length > 0;
};

window.fvApplyAxisLocks = function(figUid, changedAxisId) {
  const figIdx = figUidToIdx[figUid];
  if (figIdx === undefined) return;
  const ranges = { ...(window.fvAxisLockRangesForFigure?.(figUid) || {}), ...figureViewportRanges(figUid) };
  const gd = divs[figIdx];
  const fullLayout = (gd && gd._fullLayout) || {};
  const update = {};
  for (const [axId, range] of Object.entries(ranges)) {
    if (!/^(x|y)\d*$/.test(axId)) continue;
    const key = plotlyAxisKey(axId);
    update[key + '.range'] = range;
    update[key + '.autorange'] = false;
  }
  const changedFamily = String(changedAxisId || '').charAt(0);
  if ((changedFamily === 'x' || changedFamily === 'y') && !window.fvIsAxisLocked?.(figUid, changedAxisId)) {
    for (const layoutKey of Object.keys(fullLayout)) {
      const match = /^(x|y)axis(\d*)$/.exec(layoutKey);
      if (!match || match[1] !== changedFamily) continue;
      const axId = plotlyAxisId(layoutKey);
      if (Object.prototype.hasOwnProperty.call(ranges, axId)) continue;
      update[layoutKey + '.autorange'] = true;
    }
  }
  if (!Object.keys(update).length) return;
  return fvRunProgrammaticPlotlyOp(() => Plotly.relayout(divs[figIdx], update));
};

function _axisRangeForSelectionBox(figUid, axisProp) {
  const figIdx = figUidToIdx[figUid];
  if (figIdx === undefined) return null;
  const axisKey = axisProp === 'x' ? 'xaxis' : 'yaxis';
  const fullRange = divs[figIdx]
    && divs[figIdx]._fullLayout
    && divs[figIdx]._fullLayout[axisKey]
    && divs[figIdx]._fullLayout[axisKey].range;
  if (Array.isArray(fullRange) && fullRange.length === 2) return fullRange;
  const layoutRange = layoutsByFig[figIdx]
    && layoutsByFig[figIdx][axisKey]
    && layoutsByFig[figIdx][axisKey].range;
  if (Array.isArray(layoutRange) && layoutRange.length === 2) return layoutRange;
  return null;
}

function _completeSelectionBox(figUid, shape) {
  if (!shape) return null;
  const out = { type: 'rect', ...shape };
  if (out.x0 == null || out.x1 == null) {
    const xRange = _axisRangeForSelectionBox(figUid, 'x');
    if (!xRange) return null;
    out.x0 = xRange[0];
    out.x1 = xRange[1];
    out.xref = out.xref || 'x';
  }
  if (out.y0 == null || out.y1 == null) {
    const yRange = _axisRangeForSelectionBox(figUid, 'y');
    if (!yRange) return null;
    out.y0 = yRange[0];
    out.y1 = yRange[1];
    out.yref = out.yref || 'y';
  }
  return out;
}

function _columnFor(figSpec, axisProp) {
  const cartesian = (figSpec.traces || []).find(
    ts => Array.isArray(ts.axes) && ts.axes.length > 0 && ts.backend_data
  );
  if (!cartesian) return null;
  const bd = cartesian.backend_data || {};
  const isHoriz = cartesian.params && cartesian.params.orientation === 'h';
  if (axisProp === 'x') {
    if (isHoriz) return bd.values || bd.x || null;
    const labels = bd.labels;
    if (labels) return Array.isArray(labels) ? (labels.length === 1 ? labels[0] : null) : labels;
    return bd.x || null;
  }
  // axisProp === 'y'
  if (isHoriz) {
    const labels = bd.labels;
    if (labels) return Array.isArray(labels) ? (labels.length === 1 ? labels[0] : null) : labels;
    return bd.y || null;
  }
  return bd.values || bd.y || null;
}

function updateModeIndicator(figUid, dragmode) {
  const figIdx = figUidToIdx[figUid];
  if (figIdx === undefined) return;
  const controls = window.fvPanelControlRoot?.(figIdx);
  if (!controls) return;
  const supportsZoomPan = figSupportsZoomPan[figIdx];
  const axesLocked = window.fvAreCurrentAxesLocked?.(figUid) === true;
  for (const btn of controls.querySelectorAll('.fv-mode-btn')) {
    const mode = btn.dataset.mode;
    btn.classList.toggle('mode-active', mode === dragmode);
    btn.disabled = (!supportsZoomPan && (mode === 'zoom' || mode === 'pan'))
      || (axesLocked && (mode === 'zoom' || mode === 'pan'));
  }
  for (const btn of controls.querySelectorAll('.fv-mode-action-btn[data-action="reset-panel"]')) {
    btn.disabled = !supportsZoomPan;
  }
  window.fvUpdateAxisLockButtons?.(figUid);
  const warn = controls.querySelector('.fv-mode-warn');
  if (warn) {
    if (axesLocked) warn.textContent = 'axes locked';
    else warn.textContent = (dragmode === 'select' && figHasMultiY[figIdx]) ? 'any-trace OR filter' : '';
  }
}

function setFigureMode(figUid, mode) {
  const figIdx = figUidToIdx[figUid];
  if (figIdx === undefined) return;
  if (window.fvAreCurrentAxesLocked?.(figUid) && (mode === 'zoom' || mode === 'pan')) return;
  return Promise.resolve(
    fvRunProgrammaticPlotlyOp(() => Plotly.relayout(divs[figIdx], { dragmode: mode }))
  ).then(() => {
    updateModeIndicator(figUid, mode);
  });
}

window.fvSyncFigureModeForAxisLocks = function(figUid) {
  const figIdx = figUidToIdx[figUid];
  if (figIdx === undefined) return;
  const dragmode = divs[figIdx]?._fullLayout?.dragmode || 'zoom';
  if (window.fvAreCurrentAxesLocked?.(figUid) && (dragmode === 'zoom' || dragmode === 'pan')) {
    setFigureMode(figUid, 'select');
    return;
  }
  updateModeIndicator(figUid, dragmode);
};

function selectionBoxesForFigure(figUid) {
  const figSpec = figSpecByUid[figUid];
  const hasCartesian = !!(figSpec && (figSpec.traces || []).some(
    ts => Array.isArray(ts.axes) && ts.axes.length > 0
  ));
  if (!hasCartesian) return [];
  const selections = (DASHBOARD_SPEC.state && DASHBOARD_SPEC.state.selections) || [];
  const xCol = _columnFor(figSpec, 'x');
  const yCol = _columnFor(figSpec, 'y');
  const shapes = [];
  for (const sel of selections) {
    if (!sel || sel.source_figure_uid !== figUid) continue;
    if (sel._plotly_selection_box) {
      const box = _completeSelectionBox(figUid, sel._plotly_selection_box);
      if (box) shapes.push(box);
      continue;
    }
    for (const pred of (sel.predicates || [])) {
      const xClause = (pred.clauses || []).find(c => c.column === xCol && c.range);
      const yClause = (pred.clauses || []).find(c => c.column === yCol && c.range);
      if (!xClause && !yClause) continue;
      const shape = {};
      if (xClause) {
        shape.x0 = xClause.range[0];
        shape.x1 = xClause.range[1];
        shape.xref = 'x';
      }
      if (yClause) {
        shape.y0 = yClause.range[0];
        shape.y1 = yClause.range[1];
        shape.yref = 'y';
      }
      const box = _completeSelectionBox(figUid, shape);
      if (box) shapes.push(box);
    }
  }
  return shapes;
}

const CATEGORY_DIMMED_OPACITY = 0.28;

function _nodeSatisfiesPredicate(figSpec, parentTrace, node, predicate) {
  const data = (node && node.__data__) || {};
  const traceType = parentTrace && parentTrace.trace_type;
  if (traceType === 'pie') {
    const labels = parentTrace.backend_data && parentTrace.backend_data.labels;
    const labelCols = Array.isArray(labels) ? labels : [labels];
    const sliceLabel = data.label
      ?? (data.data && data.data.label)
      ?? (data.data && data.data.data && data.data.data.label);
    if (sliceLabel == null) return false;
    let parts;
    if (labelCols.length === 1) {
      parts = [String(sliceLabel)];
    } else {
      try { parts = JSON.parse(String(sliceLabel)); }
      catch (e) { return false; }
      if (!Array.isArray(parts) || parts.length !== labelCols.length) return false;
    }
    return (predicate.clauses || []).every(c => {
      const idx = labelCols.indexOf(c.column);
      if (idx < 0) return true;
      return (c.values || []).map(String).includes(String(parts[idx]));
    });
  }
  if (traceType === 'treemap') {
    const path = (parentTrace.params && parentTrace.params.path) || [];
    const id = data.id
      || (data.data && data.data.id)
      || (data.data && data.data.data && data.data.data.id);
    if (!id || !id.startsWith('root/')) return false;
    const idParts = id.split('/').slice(1).map(v => decodeURIComponent(String(v)));
    return (predicate.clauses || []).every(c => {
      const idx = path.indexOf(c.column);
      if (idx < 0) return true;
      if (idx >= idParts.length) return false;
      return (c.values || []).map(String).includes(idParts[idx]);
    });
  }
  return false;
}

function applyCategorySelectionStyles(figUid) {
  const figIdx = figUidToIdx[figUid];
  const gd = figIdx === undefined ? null : divs[figIdx];
  if (!gd) return;
  const slices = Array.from(gd.querySelectorAll('g.slice'));
  for (const node of slices) node.style.opacity = '';

  const predicates = window.fvPredicatesForFigure?.(
    figUid,
    (DASHBOARD_SPEC.state && DASHBOARD_SPEC.state.selections) || []
  ) || [];
  if (!predicates.length) return;

  const figSpec = figSpecByUid[figUid];
  const parentTrace = (figSpec && figSpec.traces || []).find(
    ts => ts.trace_type === 'pie' || ts.trace_type === 'treemap'
  );
  if (!parentTrace) return;

  for (const node of slices) {
    const matchesAny = predicates.some(
      p => _nodeSatisfiesPredicate(figSpec, parentTrace, node, p)
    );
    node.style.opacity = matchesAny ? '1' : String(CATEGORY_DIMMED_OPACITY);
  }
}

function syncLayoutViewport(figUid) {
  const figIdx = figUidToIdx[figUid];
  if (figIdx === undefined) return;
  const layout = layoutsByFig[figIdx];
  const lockRanges = window.fvAxisLockRangesForFigure?.(figUid) || {};
  const ranges = { ...lockRanges, ...figureViewportRanges(figUid) };
  const cartesianRanges = Object.fromEntries(
    Object.entries(ranges).filter(([axId]) => /^(x|y)\d*$/.test(axId))
  );
  const cartesianKeys = Object.fromEntries(
    Object.keys(cartesianRanges).map(axId => [plotlyAxisKey(axId), true])
  );
  for (const key of Object.keys(layout)) {
    if (/^[xy]axis\d*$/.test(key) && !Object.prototype.hasOwnProperty.call(cartesianKeys, key)) {
      if (layout[key]) { delete layout[key].range; layout[key].autorange = true; }
    }
  }
  for (const [axId, range] of Object.entries(cartesianRanges)) {
    const key = plotlyAxisKey(axId);
    if (!layout[key]) layout[key] = {};
    layout[key].range = range;
    layout[key].autorange = false;
  }
}

function _fvRenderFigure(figUid) {
  const figIdx = figUidToIdx[figUid];
  if (figIdx === undefined) return;
  rememberPlotlyVisibility(figIdx);
  tracesByFig[figIdx] = buildTracesForFigure(figUid);
  syncLayoutViewport(figUid);
  layoutsByFig[figIdx].datarevision = (layoutsByFig[figIdx].datarevision || 0) + 1;

  const overlayMode = (DASHBOARD_SPEC.state && DASHBOARD_SPEC.state.cross_filter_mode) === 'overlay';
  const selections = (DASHBOARD_SPEC.state && DASHBOARD_SPEC.state.selections) || [];
  const sourceFigure = figureHasSelectionSource(figUid, selections);
  // Mirror buildTracesForFigure's showForeground: a live cube gesture also
  // drives the fg layer before any selection is committed (contract F).
  const hasFg = overlayMode && !sourceFigure
    && (selections.length > 0
        || (typeof fvCubeOverlayFgActive === 'function' && fvCubeOverlayFgActive(figUid)));
  const figSpec = figSpecByUid[figUid];
  const hasBarOrHist = tracesByFig[figIdx].some(t => t.type === 'bar');
  if (hasBarOrHist && hasFg) {
    layoutsByFig[figIdx].barmode = 'overlay';
  } else if (hasBarOrHist) {
    const baseBarmode = baseBarmodeForFigure(figSpec);
    if (baseBarmode) layoutsByFig[figIdx].barmode = baseBarmode;
  } else {
    delete layoutsByFig[figIdx].barmode;
  }

  layoutsByFig[figIdx].selections = selectionBoxesForFigure(figUid);
  const renderPromise = Plotly.react(divs[figIdx], tracesByFig[figIdx], layoutsByFig[figIdx], configsByFig[figIdx]);
  Promise.resolve(renderPromise).then(() => {
    applyCategorySelectionStyles(figUid);
    renderHoverOverlay(figUid);
    return window.fvFinalizeHeatmapOverlayColorbars?.(figUid);
  });
}

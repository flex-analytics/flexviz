// === Plotly adapter — startup, event wiring, and Plotly-specific hook overrides ===
// This block runs after all function definitions above are in scope.
// Requires: traces.js, render.js, events.js, hover.js, and the shared bundle.

// Launch all initial Plotly renders; the startup IIFE awaits these.
const initialPlotPromises = _fvAllFigUids.map((_, figIdx) =>
  Plotly.newPlot(divs[figIdx], tracesByFig[figIdx], layoutsByFig[figIdx], configsByFig[figIdx])
);

function bindPlotlyResizeObserver(gd) {
  if (!gd || gd._fvResizeObserverBound === true || typeof ResizeObserver === 'undefined') return;
  gd._fvResizeObserverBound = true;

  let lastWidth = -1;
  let lastHeight = -1;
  let rafId = 0;
  const resizeNow = function() {
    rafId = 0;
    const width = gd.clientWidth;
    const height = gd.clientHeight;
    if (width <= 0 || height <= 0) return;
    if (width === lastWidth && height === lastHeight) return;
    lastWidth = width;
    lastHeight = height;
    Plotly.Plots.resize(gd);
  };
  const observer = new ResizeObserver(() => {
    if (rafId) cancelAnimationFrame(rafId);
    rafId = requestAnimationFrame(resizeNow);
  });
  observer.observe(gd);
  gd._fvResizeObserver = observer;
}

// Plotly-specific override: wrap with _programmaticOp guard
const _fvBaseOnResetPanel = window.fvOnResetPanel;
window.fvOnResetPanel = async function(figUid) {
  await fvRunProgrammaticPlotlyOp(() => _fvBaseOnResetPanel?.(figUid));
};
window.fvOnReset = async function() {
  await fvRunProgrammaticPlotlyOp(async function() {
    window.fvClearUnlockedViewports?.();
    window.fvSetSelectionState?.([]);
    window.fvResetRuntimeCache?.();
    await postDashboardUpdate({type: 'init', axis_ranges: {}, selections: [], force_update: true});
  });
};
window.fvOnDeselect = async function() {
  await fvRunProgrammaticPlotlyOp(async function() {
    window.fvSetSelectionState?.([]);
    await postDashboardUpdate({type: 'deselect', axis_ranges: {}, selections: [], force_update: true});
  });
};

// Wire all per-figure Plotly event handlers using the globally available arrays.
_fvAllFigUids.forEach((figUid, figIdx) => {
  divs[figIdx].on('plotly_relayout', function(rd) { handleRelayout(rd, figUid); });
  divs[figIdx].on('plotly_selected', function(ed) { handleSelected(ed, figUid); });
  // Cube live brush: drag-time selecting events, for figures with range or
  // categorical (bar) selection geometry, and only when live_brush !== "off"
  // ("off" = today's behavior bit-for-bit, the cube system stays idle).
  if (
    _fvLiveBrushEnabled()
    && (_figureSourceTrace(figUid, 'range') || _figureSourceTrace(figUid, 'categorical'))
  ) {
    divs[figIdx].on('plotly_selecting', function(ed) { handleSelecting(ed, figUid); });
    // Editing an existing selection box emits no Plotly event until mouseup;
    // watch the outline drag directly and replay it through handleSelecting
    // (capture phase, so Plotly's drag machinery cannot swallow it).
    divs[figIdx].addEventListener(
      'pointerdown',
      function(evt) { handleSelectionEditPointerDown(evt, figUid); },
      true
    );
  }
  divs[figIdx].on('plotly_deselect', function() { handleDeselect(figUid); });
  divs[figIdx].on('plotly_hover',    function(ed) { handlePlotlyHover(ed, figUid); });
  divs[figIdx].on('plotly_unhover',  function()   { handlePlotlyUnhover(); });
  divs[figIdx].addEventListener('pointerdown', suspendHoverForDrag);
  divs[figIdx].addEventListener('pointerleave', resumeHoverAfterDrag);

  // Conditional click handlers for pie / treemap figures
  const figSpec = figSpecByUid[figUid];
  if (figSpec && figSpec.traces.some(ts => ts.trace_type === 'pie')) {
    divs[figIdx].on('plotly_click', function(ed) { handleClick(ed, figUid); });
  }
  if (figSpec && figSpec.traces.some(ts => ts.trace_type === 'treemap')) {
    divs[figIdx].on('plotly_treemapclick', function(ed) { return handleClick(ed, figUid); });
  }
});

window.addEventListener('pointerup', resumeHoverAfterDrag);
window.addEventListener('resize', function() {
  for (const figUid of Object.keys(figUidToIdx)) renderHoverOverlay(figUid);
});

// Startup: wait for Plotly to render all figures, then wire mode indicator
// buttons and restore any saved viewport / selection state.
(async function _fvInitPlotly() {
  await Promise.all(initialPlotPromises);

  for (const gd of divs) {
    bindPlotlyResizeObserver(gd);
  }

  for (const figUid of _fvAllFigUids) {
    const figIdx = figUidToIdx[figUid];
    if (figIdx === undefined) continue;
    window.fvBindPanelControls?.(figUid, {
      setMode(mode) { setFigureMode(figUid, mode); },
      resetPanel() { window.fvOnResetPanel?.(figUid); },
      toggleAxisLocks() { window.fvOnToggleAxisLocks?.(figUid); },
    });
    if (!figSupportsZoomPan[figIdx]) setFigureMode(figUid, 'select');
  }

  await restoreDashboardFromSpec();

  for (const figUid of _fvAllFigUids) {
    const idx = figUidToIdx[figUid];
    const dragmode = idx !== undefined && divs[idx]?._fullLayout?.dragmode;
    updateModeIndicator(figUid, dragmode || 'zoom');
  }
})();

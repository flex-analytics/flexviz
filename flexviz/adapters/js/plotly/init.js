// === Plotly adapter — startup, event wiring, and Plotly-specific hook overrides ===
// This block runs after all function definitions above are in scope.
// Requires: traces.js, render.js, events.js, hover.js, and the shared bundle.

function bindPlotlyResizeObserver(gd) {
  if (typeof ResizeObserver === 'undefined') return;

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

// Wire one figure's Plotly event handlers, resize observer and panel controls.
// Called once per figure, after its first render resolved: only then is the div
// a Plotly graph div (gd.on exists), and no event can reach a handler before
// the figure is drawn.
function bindFigure(figUid) {
  const figIdx = figUidToIdx[figUid];
  if (figIdx === undefined) return;
  const gd = divs[figIdx];
  if (!gd || gd._fvBound === true) return;
  gd._fvBound = true;

  gd.on('plotly_relayout', function(rd) { handleRelayout(rd, figUid); });
  gd.on('plotly_selected', function(ed) { handleSelected(ed, figUid); });
  // Cube live brush: drag-time selecting events, for figures with range or
  // categorical (bar) selection geometry, and only when live_brush !== "off"
  // ("off" = today's behavior bit-for-bit, the cube system stays idle).
  if (
    _fvLiveBrushEnabled()
    && (_figureSourceTrace(figUid, 'range') || _figureSourceTrace(figUid, 'categorical'))
  ) {
    gd.on('plotly_selecting', function(ed) { handleSelecting(ed, figUid); });
    // Editing an existing selection box emits no Plotly event until mouseup;
    // watch the outline drag directly and replay it through handleSelecting
    // (capture phase, so Plotly's drag machinery cannot swallow it).
    gd.addEventListener(
      'pointerdown',
      function(evt) { handleSelectionEditPointerDown(evt, figUid); },
      true
    );
  }
  gd.on('plotly_deselect', function() { handleDeselect(figUid); });
  gd.on('plotly_hover',    function(ed) { handlePlotlyHover(ed, figUid); });
  gd.on('plotly_unhover',  function()   { handlePlotlyUnhover(); });
  gd.addEventListener('pointerdown', suspendHoverForDrag);
  gd.addEventListener('pointerleave', resumeHoverAfterDrag);

  // Conditional click handlers for pie / treemap figures
  const figSpec = figSpecByUid[figUid];
  if (figSpec && figSpec.traces.some(ts => ts.trace_type === 'pie')) {
    gd.on('plotly_click', function(ed) { handleClick(ed, figUid); });
  }
  if (figSpec && figSpec.traces.some(ts => ts.trace_type === 'treemap')) {
    gd.on('plotly_treemapclick', function(ed) { return handleClick(ed, figUid); });
  }

  bindPlotlyResizeObserver(gd);
  window.fvBindPanelControls?.(figUid, {
    setMode(mode) { setFigureMode(figUid, mode); },
    resetPanel() { window.fvOnResetPanel?.(figUid); },
    toggleAxisLocks() { window.fvOnToggleAxisLocks?.(figUid); },
  });
  if (!figSupportsZoomPan[figIdx]) setFigureMode(figUid, 'select');
  updateModeIndicator(figUid, gd._fullLayout?.dragmode || 'zoom');
}

window.addEventListener('pointerup', resumeHoverAfterDrag);
window.addEventListener('resize', function() {
  for (const figUid of Object.keys(figUidToIdx)) renderHoverOverlay(figUid);
});

// Startup: ask for the initial data straight away. Plotly.react on an unplotted
// div plots it, so the response draws every figure once and each first render
// binds its figure. A figure the response never drew (a failed request) still
// needs a graph div for its events and for later updates to react into.
(async function _fvInitPlotly() {
  await restoreDashboardFromSpec();

  for (const figUid of _fvAllFigUids) {
    const figIdx = figUidToIdx[figUid];
    if (figIdx === undefined) continue;
    const gd = divs[figIdx];
    if (!gd || gd._fullLayout) continue;
    await Plotly.newPlot(gd, tracesByFig[figIdx], layoutsByFig[figIdx], configsByFig[figIdx]);
    bindFigure(figUid);
  }
})();

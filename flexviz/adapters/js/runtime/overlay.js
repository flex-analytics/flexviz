// === FlexViz shared runtime — overlay cache restore ===
// Requires: state.js, delta.js loaded first

window.fvEnsureOverlayBackground = async function(selections) {
  if (!(selections && selections.length)) return;
  const missingBg = DASHBOARD_SPEC.figures.some(fig => !hasBgByFigure[fig.uid]);
  if (!missingBg) return;
  await postDashboardUpdate({
    type: 'init', axis_ranges: {}, selections, force_update: true,
  });
};
window.fvResetRuntimeCache = function() {
  for (const uid of Object.keys(layerDataByUid)) {
    layerDataByUid[uid] = { base: {}, bg: {}, fg: {} };
  }
  for (const fig of DASHBOARD_SPEC.figures) {
    hasBgByFigure[fig.uid] = false;
    bgYExtentByFig[fig.uid] = null;
    const grouped = groupedDataByParent[fig.uid] || {};
    for (const parentUid of Object.keys(grouped)) {
      grouped[parentUid] = { base: [], bg: [], fg: [] };
    }
  }
  if (typeof _fvResetRendererCache === 'function') _fvResetRendererCache();
};
async function restoreDashboardFromSpec() {
  window.fvResetRuntimeCache?.();
  fvCacheReset();
  window.fvRebuildHoverLookups?.();
  window.fvUpdateCfModeButton?.();
  window.fvSyncHoverDropdown?.();
  window.fvRefreshSelectionSummary?.();
  const savedSelections = (DASHBOARD_SPEC.state && DASHBOARD_SPEC.state.selections) || [];
  const figViewports = Object.fromEntries(
    DASHBOARD_SPEC.figures.map(fig => [fig.uid, figureViewportRanges(fig.uid)])
  );
  await postDashboardUpdate({
    type: 'init', axis_ranges: {}, selections: savedSelections, force_update: true,
  });
  if (savedSelections.length) {
    await postDashboardUpdate({
      type: 'selection', axis_ranges: {}, selections: savedSelections, force_update: true,
    });
  }
  for (const [figUid, axRanges] of Object.entries(figViewports)) {
    if (!Object.keys(axRanges).length) continue;
    await postDashboardUpdate({
      type: 'viewport', axis_ranges: axRanges,
      selections: savedSelections, force_update: true, figure_uid: figUid,
    });
  }
  _fvAllFigUids.forEach(figUid => _fvRenderFigure(figUid));
  window.fvRefreshSelectionSummary?.();
}
window.fvRestoreFromSpec = async function() {
  await restoreDashboardFromSpec();
};

// === FlexViz shared runtime — overlay cache restore ===
// Requires: state.js, delta.js loaded first

window.fvEnsureOverlayBackground = async function(selections) {
  if (!(selections && selections.length)) return;
  const missingBg = DASHBOARD_SPEC.figures.some(fig => !hasBgByFigure[fig.uid]);
  if (!missingBg) return;
  await postDashboardUpdate({
    type: 'init', selections, force_update: true,
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
  // One init also restores the viewport. The request carries the complete spec,
  // so the server aggregates within state.viewport; init marks every figure
  // dirty, so each figure's render applies that viewport to its layout.
  let ok = await postDashboardUpdate({
    type: 'init', selections: savedSelections, force_update: true,
  });
  // Stop here: a second request would clear the error reason of the first.
  if (!ok) return false;
  if (savedSelections.length) {
    ok = await postDashboardUpdate({
      type: 'selection', selections: savedSelections, force_update: true,
    });
  }
  window.fvRefreshSelectionSummary?.();
  // The spec can carry other axis locks (an apply, a rollback, a shared URL).
  for (const figUid of _fvAllFigUids) {
    window.fvUpdateAxisLockButtons?.(figUid);
    window.fvSyncFigureModeForAxisLocks?.(figUid);
  }
  return ok;
}
window.fvRestoreFromSpec = restoreDashboardFromSpec;

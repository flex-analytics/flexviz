// === FlexViz toolbar — window hooks + button wiring ===
// Requires: DASHBOARD_SPEC, SERVER_URL, postDashboardUpdate defined before this block

window.fvUpdateCfModeButton = function() {
  const btn = document.getElementById('fv-btn-cfmode');
  if (!btn || !DASHBOARD_SPEC || !DASHBOARD_SPEC.state) return;
  const mode = DASHBOARD_SPEC.state.cross_filter_mode || 'update';
  btn.textContent = mode === 'overlay' ? 'CF: Overlay' : 'CF: Update';
  btn.classList.toggle('active', mode === 'overlay');
};
function fvEnsureState() {
  if (!DASHBOARD_SPEC.state) DASHBOARD_SPEC.state = {};
  if (!DASHBOARD_SPEC.state.viewport) DASHBOARD_SPEC.state.viewport = {};
  if (!DASHBOARD_SPEC.state.group_domains) DASHBOARD_SPEC.state.group_domains = {};
  if (!DASHBOARD_SPEC.state.selections) DASHBOARD_SPEC.state.selections = [];
  return DASHBOARD_SPEC.state;
}
// Axis locks are client-only state: they live on client_state (never read by
// the server engine), not on the server-facing state.
function fvEnsureClientState() {
  if (!DASHBOARD_SPEC.client_state) DASHBOARD_SPEC.client_state = {};
  if (!DASHBOARD_SPEC.client_state.axis_locks) DASHBOARD_SPEC.client_state.axis_locks = {};
  if (!DASHBOARD_SPEC.client_state.axis_lock_ranges) DASHBOARD_SPEC.client_state.axis_lock_ranges = {};
  return DASHBOARD_SPEC.client_state;
}
function fvAxisFamily(axisId) {
  const value = String(axisId || '');
  const family = value.charAt(0);
  return (family === 'x' || family === 'y') ? family : value;
}
function fvAxisLockKey(figUid, axisId) {
  return figUid + '/' + fvAxisFamily(axisId);
}
function fvLockableAxesForFigure(figUid) {
  if (!figUid || typeof figUidToIdx === 'undefined' || typeof figLockableAxes === 'undefined') {
    return [];
  }
  const figIdx = figUidToIdx[figUid];
  if (figIdx === undefined) return [];
  return Array.isArray(figLockableAxes[figIdx]) ? figLockableAxes[figIdx] : [];
}
function fvCurrentLockableAxes(figUid) {
  return fvLockableAxesForFigure(figUid)
    .filter(axis => window.fvHasLockableCurrentAxis?.(figUid, axis));
}
function fvClearAxisLockRanges(figUid, axisId) {
  const cstate = fvEnsureClientState();
  const family = fvAxisFamily(axisId);
  for (const key of Object.keys(cstate.axis_lock_ranges || {})) {
    if (!key.startsWith(figUid + '/')) continue;
    const storedAxisId = key.slice(figUid.length + 1);
    if (fvAxisFamily(storedAxisId) === family) delete cstate.axis_lock_ranges[key];
  }
}
window.fvIsAxisLocked = function(figUid, axisId) {
  if (!figUid || !axisId) return false;
  const cstate = fvEnsureClientState();
  return cstate.axis_locks[fvAxisLockKey(figUid, axisId)] === true;
};
window.fvSetAxisLocked = function(figUid, axisId, locked) {
  if (!figUid || !axisId) return;
  const cstate = fvEnsureClientState();
  const key = fvAxisLockKey(figUid, axisId);
  if (locked) cstate.axis_locks[key] = true;
  else {
    delete cstate.axis_locks[key];
    fvClearAxisLockRanges(figUid, axisId);
  }
};
window.fvAreCurrentAxesLocked = function(figUid) {
  const availableAxes = fvCurrentLockableAxes(figUid);
  return availableAxes.length > 0
    && availableAxes.every(axis => window.fvIsAxisLocked(figUid, axis));
};
window.fvUpdateAxisLockButtons = function(figUid) {
  if (!figUid || typeof figUidToIdx === 'undefined') return;
  const figIdx = figUidToIdx[figUid];
  if (figIdx === undefined) return;
  const controls = window.fvPanelControlRoot?.(figIdx);
  if (!controls) return;
  const availableAxes = fvCurrentLockableAxes(figUid);
  const fullyLocked = window.fvAreCurrentAxesLocked(figUid);
  for (const btn of controls.querySelectorAll('.fv-mode-action-btn[data-action="lock-axes"]')) {
    btn.disabled = availableAxes.length === 0;
    btn.classList.toggle('active', fullyLocked);
    btn.setAttribute('aria-pressed', fullyLocked ? 'true' : 'false');
    btn.textContent = fullyLocked ? 'Unlock Axes' : 'Lock Axes';
  }
  // Keep the global "Lock All Axes" toolbar button in sync whenever a per-figure
  // lock state changes.
  window.fvUpdateLockAllAxesButton?.();
};
window.fvPruneAxisRangesForLocks = function(figUid, ranges) {
  const out = {};
  for (const [axisId, range] of Object.entries(ranges || {})) {
    if (!window.fvIsAxisLocked(figUid, axisId)) out[axisId] = range;
  }
  return out;
};
window.fvStoreAxisLockRanges = function(figUid, ranges) {
  if (!figUid) return;
  const cstate = fvEnsureClientState();
  for (const [axisId, range] of Object.entries(ranges || {})) {
    if (!Array.isArray(range) || range.length !== 2) continue;
    cstate.axis_lock_ranges[figUid + '/' + axisId] = { min: range[0], max: range[1] };
  }
};
window.fvAxisLockRangesForFigure = function(figUid) {
  const cstate = fvEnsureClientState();
  const out = {};
  for (const [key, value] of Object.entries(cstate.axis_lock_ranges || {})) {
    if (!key.startsWith(figUid + '/')) continue;
    const axisId = key.slice(figUid.length + 1);
    if (!window.fvIsAxisLocked(figUid, axisId)) continue;
    if (value && value.min !== undefined && value.max !== undefined) {
      out[axisId] = [value.min, value.max];
    }
  }
  return out;
};
window.fvClearFigureViewport = function(figUid) {
  if (!figUid) return;
  const state = fvEnsureState();
  const viewport = state.viewport || {};
  for (const key of Object.keys(viewport)) {
    if (!key.startsWith(figUid + '/')) continue;
    delete viewport[key];
  }
  state.viewport = viewport;
};
// True when this figure has any stored viewport (zoom/pan) range. Used by the
// per-figure reset no-op guard, which must sample zoom state *before* clearing.
window.fvFigureHasViewport = function(figUid) {
  if (!figUid) return false;
  const state = fvEnsureState();
  const viewport = state.viewport || {};
  return Object.keys(viewport).some(key => key.startsWith(figUid + '/'));
};
// Clear a single axis' stored range (used by per-axis autorange / double-click),
// leaving any other still-zoomed axis on the same figure untouched.
window.fvClearFigureAxisViewport = function(figUid, axisId) {
  if (!figUid || !axisId) return;
  const state = fvEnsureState();
  const viewport = state.viewport || {};
  delete viewport[figUid + '/' + axisId];
  state.viewport = viewport;
};
window.fvClearUnlockedViewports = function() {
  const state = fvEnsureState();
  state.viewport = {};
};
function fvTryLockAxis(figUid, axisId) {
  const ranges = window.fvCaptureAxisDisplayRanges?.(figUid, axisId) || {};
  if (!Object.keys(ranges).length) return false;
  window.fvSetAxisLocked(figUid, axisId, true);
  window.fvStoreAxisLockRanges(figUid, ranges);
  return true;
}
window.fvOnToggleAxisLocks = async function(figUid) {
  if (!figUid) return;
  const availableAxes = fvCurrentLockableAxes(figUid);
  if (!availableAxes.length) {
    window.fvUpdateAxisLockButtons?.(figUid);
    return;
  }
  const shouldLock = availableAxes.some(axis => !window.fvIsAxisLocked(figUid, axis));
  if (shouldLock) {
    for (const axisId of availableAxes) {
      fvTryLockAxis(figUid, axisId);
    }
    window.fvUpdateAxisLockButtons?.(figUid);
    window.fvSyncFigureModeForAxisLocks?.(figUid);
    return;
  }
  for (const axisId of availableAxes) {
    window.fvSetAxisLocked(figUid, axisId, false);
  }
  window.fvUpdateAxisLockButtons?.(figUid);
  window.fvSyncFigureModeForAxisLocks?.(figUid);
  await Promise.resolve(window.fvApplyAxisLocks?.(figUid));
};
// ── Global "Lock All Axes" toolbar button ──────────────────────────────────
// Returns true only when every figure with lockable axes is fully locked.
window.fvAreAllFiguresLocked = function() {
  if (typeof figUidToIdx === 'undefined') return false;
  const uids = Object.keys(figUidToIdx);
  if (!uids.length) return false;
  // A figure with no lockable axes is treated as "not relevant" — skip it so
  // an empty/geo figure doesn't prevent the button from showing "locked".
  const relevant = uids.filter(uid => fvCurrentLockableAxes(uid).length > 0);
  if (!relevant.length) return false;
  return relevant.every(uid => window.fvAreCurrentAxesLocked(uid));
};

// Sync the global toolbar button label + active state.
window.fvUpdateLockAllAxesButton = function() {
  const btn = document.getElementById('fv-btn-lock-all');
  if (!btn) return;
  const allLocked = window.fvAreAllFiguresLocked();
  btn.textContent = allLocked ? 'Unlock All Axes' : 'Lock All Axes';
  btn.classList.toggle('active', allLocked);
  btn.setAttribute('aria-pressed', allLocked ? 'true' : 'false');
};

// Toggle: if any figure has unlocked axes → lock all; if all locked → unlock all.
window.fvOnLockAllAxes = async function() {
  if (typeof figUidToIdx === 'undefined') return;
  const uids = Object.keys(figUidToIdx);
  if (!uids.length) return;
  const shouldLock = !window.fvAreAllFiguresLocked();
  const unlockPromises = [];
  for (const figUid of uids) {
    const availableAxes = fvCurrentLockableAxes(figUid);
    if (!availableAxes.length) continue;
    if (shouldLock) {
      for (const axisId of availableAxes) {
        fvTryLockAxis(figUid, axisId);
      }
      window.fvUpdateAxisLockButtons?.(figUid);
      window.fvSyncFigureModeForAxisLocks?.(figUid);
    } else {
      for (const axisId of availableAxes) {
        window.fvSetAxisLocked(figUid, axisId, false);
      }
      window.fvUpdateAxisLockButtons?.(figUid);
      window.fvSyncFigureModeForAxisLocks?.(figUid);
      unlockPromises.push(Promise.resolve(window.fvApplyAxisLocks?.(figUid)));
    }
  }
  await Promise.all(unlockPromises);
  window.fvUpdateLockAllAxesButton?.();
};

// Per-figure reset. Resets ONLY this figure: clears its viewport (zoom) and
// removes any selection *sourced by* this figure (which was cross-filtering
// other figures). Selections sourced by OTHER figures are kept and stay applied
// to this figure — so resetting a figure that is merely a cross-filter *target*
// re-renders it filtered-by-others at autorange, without disturbing the others.
//
// The emitted event type is derived from the resulting state, because the
// engine keys its recompute/scoping on event type:
//   * selection removed, others remain  -> 'selection' (re-apply remaining filters)
//   * selection removed, none remain     -> 'deselect'  (whole dashboard unfiltered)
//   * no selection changed (viewport-only reset) -> 'viewport' (scoped to this figure)
window.fvOnResetPanel = async function(figUid) {
  if (!figUid) return;
  // Sample zoom state before clearing — the no-op guard below needs to know
  // whether the reset actually removes anything.
  const wasZoomed = window.fvFigureHasViewport?.(figUid) || false;
  window.fvClearFigureViewport?.(figUid);
  const before = window.fvSelectionState?.() || [];
  const remaining = window.fvClearFigureSelectionFromList?.(figUid, before) || before;
  const selectionChanged = remaining.length !== before.length;
  // No-op: nothing this figure shows would change. It was not zoomed and sourced
  // no selection, so it is already at full autorange, filtered only by *other*
  // figures (which this reset does not touch). Skip the round-trip and the
  // redundant re-render entirely. Covers the case with no other cross-filters
  // and the case where other figures keep cross-filtering F. Axis locks are
  // view-only (pruned from server requests, re-applied on render), so a locked
  // figure is no different here — its lock is already holding, and a reset that
  // clears nothing leaves it untouched.
  if (!wasZoomed && !selectionChanged) return;
  window.fvSetSelectionState?.(remaining);
  const axisRanges = window.figureViewportRanges?.(figUid) || {};
  await postDashboardUpdate({
    type: selectionChanged ? (remaining.length ? 'selection' : 'deselect') : 'viewport',
    axis_ranges: axisRanges,
    selections: remaining,
    force_update: true,
    figure_uid: figUid,
  });
};
window.fvOnReset = async function() {
  window.fvClearUnlockedViewports?.();
  window.fvSetSelectionState?.([]);
  await postDashboardUpdate({type: 'init', axis_ranges: {}, selections: [], force_update: true});
};
window.fvOnDeselect = async function() {
  window.fvSetSelectionState?.([]);
  await postDashboardUpdate({type: 'deselect', axis_ranges: {}, selections: [], force_update: true});
};
window.fvOnCfMode = async function() {
  if (!DASHBOARD_SPEC.state) DASHBOARD_SPEC.state = {};
  const mode = DASHBOARD_SPEC.state.cross_filter_mode || 'update';
  const nextMode = mode === 'overlay' ? 'update' : 'overlay';
  DASHBOARD_SPEC.state.cross_filter_mode = nextMode;
  window.fvUpdateCfModeButton?.();
  const selections = DASHBOARD_SPEC.state.selections || [];
  if (nextMode === 'overlay') {
    await window.fvEnsureOverlayBackground?.(selections);
  }
  await postDashboardUpdate({
    type: selections.length ? 'selection' : 'deselect',
    axis_ranges: {},
    selections,
    force_update: true,
  });
};
window.fvUpdateGridButton = function() {
  const layout = DASHBOARD_SPEC.layout || {};
  const btn = document.getElementById('fv-btn-grid');
  // The grid button lives in its own toolbar group (so a divider separates it
  // from the controls to its left); hide/show the whole group to avoid leaving
  // an empty group rendering a stray divider.
  const group = btn ? btn.closest('.fv-btn-group') : null;
  if (!layout.draggable) {
    document.body.dataset.fvGridEditable = 'false';
    if (!btn) return;
    if (group) group.style.display = 'none';
    return;
  }
  if (!btn) {
    document.body.dataset.fvGridEditable = (layout.grid_editable === true) ? 'true' : 'false';
    return;
  }
  if (group) group.style.display = '';
  const editable = layout.grid_editable === true;
  document.body.dataset.fvGridEditable = editable ? 'true' : 'false';
  btn.textContent = editable ? 'Layout: Edit' : 'Layout: Locked';
  btn.classList.toggle('active', editable);
};
window.fvOnGridToggle = function() {
  const layout = DASHBOARD_SPEC.layout || {};
  if (!layout.draggable) return;
  const editable = layout.grid_editable === true;
  window.fvSetGridEditable?.(!editable);
  window.fvUpdateGridButton?.();
};
// ── Hover on/off toggle ─────────────────────────────────────────────────────
// Linked hover is a single on/off control. When on, the runtime auto-selects
// the projection from the hovered source trace (point/axis vs cell), so there
// is no per-mode dropdown to choose.

function _fvUpdateHoverToggleLabel() {
  const btn = document.getElementById('fv-hover-btn');
  if (!btn) return;
  const on = (DASHBOARD_SPEC.client_state && DASHBOARD_SPEC.client_state.hover_mode) === 'on';
  btn.textContent = 'Hover: ' + (on ? 'On' : 'Off');
  btn.setAttribute('aria-pressed', on ? 'true' : 'false');
  btn.classList.toggle('active', on);
}

function _fvSetHoverMode(mode) {
  if (!DASHBOARD_SPEC.client_state) DASHBOARD_SPEC.client_state = {};
  DASHBOARD_SPEC.client_state.hover_mode = mode;
  _fvUpdateHoverToggleLabel();
  window.fvClearAllHoverVisuals?.();
}

function fvInitHoverDropdown() {
  const btn = document.getElementById('fv-hover-btn');
  if (!btn) return;

  // Ensure client_state exists; coerce any legacy per-mode value onto on/off.
  if (!DASHBOARD_SPEC.client_state) DASHBOARD_SPEC.client_state = {};
  const cur = DASHBOARD_SPEC.client_state.hover_mode;
  if (cur !== 'on' && cur !== 'off') {
    DASHBOARD_SPEC.client_state.hover_mode = cur ? 'on' : 'off';
  }

  // Hover is offered only when at least one linkable source→target pair exists.
  const gates = {
    implementedModes: IMPLEMENTED_HOVER_MODES,
    implementedCellTraceTypes: IMPLEMENTED_CELL_TRACE_TYPES,
    implementedAxisBandTargetTraceTypes: IMPLEMENTED_AXIS_BAND_TARGET_TRACE_TYPES,
  };
  const modes = (typeof availableHoverModes === 'function')
    ? availableHoverModes(DASHBOARD_SPEC, gates)
    : new Set(['off']);
  const hoverAvailable = [...modes].some(m => m !== 'off');

  const wrapper = btn.closest('#fv-hover-dropdown') || btn.parentElement;
  if (wrapper) wrapper.style.display = hoverAvailable ? '' : 'none';
  if (!hoverAvailable) {
    // Nothing to link: keep hover off.
    DASHBOARD_SPEC.client_state.hover_mode = 'off';
    return;
  }

  btn.onclick = function(e) {
    e.stopPropagation();
    const on = DASHBOARD_SPEC.client_state.hover_mode === 'on';
    _fvSetHoverMode(on ? 'off' : 'on');
  };

  _fvUpdateHoverToggleLabel();
}

window.fvSyncHoverDropdown = function() {
  fvInitHoverDropdown();
};

window.fvOnShare = async function() {
  const btn = document.getElementById('fv-btn-share');
  try {
    const resp = await fetch(SERVER_URL + '/share', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({spec: DASHBOARD_SPEC, server_url: SERVER_URL}),
    });
    if (!resp.ok) { btn.textContent = 'Error'; setTimeout(() => { btn.textContent = 'Share'; }, 2000); return; }
    const { url } = await resp.json();
    // SERVER_URL may be a path ("/demo") or page-relative ("."), so the
    // returned url can be relative — absolutize it before handing it out.
    const absUrl = new URL(url, window.location.href).href;
    try {
      await navigator.clipboard.writeText(absUrl);
      btn.textContent = 'Copied!';
    } catch (_) {
      window.open(absUrl, '_blank');
      btn.textContent = 'Opened!';
    }
    setTimeout(() => { btn.textContent = 'Share'; }, 2000);
  } catch (e) {
    console.warn('flexviz /share failed', e);
    btn.textContent = 'Error';
    setTimeout(() => { btn.textContent = 'Share'; }, 2000);
  }
};
window.fvOnExport = function() {
  const ts = new Date().toISOString().replace(/[:.]/g, '-');
  const blob = new Blob([JSON.stringify(DASHBOARD_SPEC, null, 2)], {type: 'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'spec_' + ts + '.json';
  a.click();
};
window.fvOnImport = function(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = async function(e) {
    try {
      const loaded = JSON.parse(e.target.result);
      Object.assign(DASHBOARD_SPEC, loaded);
      window.fvRebuildHoverLookups?.();
      window.fvRefreshSelectionSummary?.();
      window._fvRestoreGridLayout?.();
      window.fvSetGridEditable?.((DASHBOARD_SPEC.layout && DASHBOARD_SPEC.layout.grid_editable) === true);
      window.fvResetRuntimeCache?.();
      window.fvUpdateCfModeButton?.();
      window.fvUpdateGridButton?.();
      if (window.fvRestoreFromSpec) {
        await window.fvRestoreFromSpec();
      } else {
        await postDashboardUpdate({
          type: 'init',
          axis_ranges: {},
          selections: (DASHBOARD_SPEC.state && DASHBOARD_SPEC.state.selections) || [],
          force_update: true,
        });
      }
    } catch (err) {
      console.warn('flexviz import failed', err);
    }
  };
  reader.readAsText(file);
};

// Wire toolbar buttons
(function _fvWireToolbar() {
  const _resetBtn = document.getElementById('fv-btn-reset');
  if (_resetBtn) _resetBtn.onclick = () => window.fvOnReset?.();
  const _deselectBtn = document.getElementById('fv-btn-deselect');
  if (_deselectBtn) _deselectBtn.onclick = () => window.fvOnDeselect?.();
  const _cfBtn = document.getElementById('fv-btn-cfmode');
  if (_cfBtn) _cfBtn.onclick = () => window.fvOnCfMode?.();
  const _lockAllBtn = document.getElementById('fv-btn-lock-all');
  if (_lockAllBtn) _lockAllBtn.onclick = () => window.fvOnLockAllAxes?.();
  const _gridBtn = document.getElementById('fv-btn-grid');
  if (_gridBtn) _gridBtn.onclick = () => window.fvOnGridToggle?.();
  // Hover dropdown is initialized and wired by fvSyncHoverDropdown (called after
  // page load once Plotly has rendered all figures and hoverSourceByTrace is built).
  window.fvSyncHoverDropdown?.();
  const _shareBtn = document.getElementById('fv-btn-share');
  if (_shareBtn) _shareBtn.onclick = () => window.fvOnShare?.();
  const _exportBtn = document.getElementById('fv-btn-export');
  if (_exportBtn) _exportBtn.onclick = () => window.fvOnExport?.();
  const _importBtn = document.getElementById('fv-btn-import');
  if (_importBtn) _importBtn.onclick = () => document.getElementById('fv-import-input').click();
  const _importInput = document.getElementById('fv-import-input');
  if (_importInput) _importInput.onchange = e => window.fvOnImport?.(e.target.files[0]);
  window.fvUpdateCfModeButton?.();
  window.fvUpdateGridButton?.();
  window.fvUpdateLockAllAxesButton?.();
  window.fvRefreshSelectionSummary?.();
})();

// === FlexViz shared runtime — delta application ===
// Requires: state.js loaded first

function ensureGroupColor(parentSpec, groupValueKey) {
  if (!parentSpec) return null;
  const colorMap = parentSpec.display && parentSpec.display.color_map;
  if (colorMap && colorMap[groupValueKey]) return colorMap[groupValueKey];
  const domainKey = parentSpec.params && parentSpec.params.group_domain_key;
  if (!domainKey) return null;
  if (!DASHBOARD_SPEC.state.group_domains[domainKey]) {
    DASHBOARD_SPEC.state.group_domains[domainKey] = { mapping: {}, next_color_index: 0 };
  }
  const domain = DASHBOARD_SPEC.state.group_domains[domainKey];
  if (!domain.mapping[groupValueKey]) {
    domain.mapping[groupValueKey] = _fvPalette[domain.next_color_index % _fvPalette.length];
    domain.next_color_index++;
  }
  return domain.mapping[groupValueKey];
}

// Every write of figure data carries a number from one counter: a request's
// number, taken when it is sent, or a new number for a client-side write. Each
// slot (a trace layer, a figure's background flag) keeps the number of the
// data it holds and refuses an older write. So responses can arrive in any
// order: the newest data wins, and a late response still fills the slots
// that newer requests did not re-aggregate.
let _fvWriteSeq = 0;
const _fvSlotSeq = {};

function fvNextWriteSeq() {
  return ++_fvWriteSeq;
}

function _fvLayerSlot(uid, layerKey) {
  return uid + '|' + layerKey;
}
function _fvBackgroundSlot(figUid) {
  return figUid + '|hasBg';
}

function _fvClaimSlot(slot, seq) {
  if ((_fvSlotSeq[slot] || 0) > seq) return false;
  _fvSlotSeq[slot] = seq;
  return true;
}

function setLayerData(uid, layerKey, updates, seq = fvNextWriteSeq()) {
  if (!_fvClaimSlot(_fvLayerSlot(uid, layerKey), seq)) return;
  ensureLayerData(uid)[layerKey] = cloneObj(updates || {});
}
function setGroupedLayerData(figUid, parentUid, layerKey, groupResults, seq = fvNextWriteSeq()) {
  if (!_fvClaimSlot(_fvLayerSlot(parentUid, layerKey), seq)) return;
  groupedDataByParent[figUid][parentUid][layerKey] = cloneObj(groupResults || []);
}
function setHasBackground(figUid, hasBackground, seq = fvNextWriteSeq()) {
  if (!_fvClaimSlot(_fvBackgroundSlot(figUid), seq)) return;
  hasBgByFigure[figUid] = hasBackground;
}

// A live-brush gesture writes its previews with new numbers, so an older
// response cannot overwrite them, and a commit keeps them. An abandoned
// gesture puts back each slot it changed together with the slot's earlier
// number. A response to a request sent before the gesture then still applies
// when it lands after the abort.
// ponytail: a response that lands during the gesture is lost when the gesture
// is abandoned. Keep that data with the saved slot if this case shows up.

// Returns a function that puts back the layer's current data and number.
function fvSaveLayerData(figUid, uid, layerKey) {
  const grouped = isGroupedParent(traceSpecByUid[uid]);
  const layers = grouped ? groupedDataByParent[figUid][uid] : ensureLayerData(uid);
  const data = cloneObj(layers[layerKey] || (grouped ? [] : {}));
  const slot = _fvLayerSlot(uid, layerKey);
  const seq = _fvSlotSeq[slot] || 0;
  return () => {
    layers[layerKey] = data;
    _fvSlotSeq[slot] = seq;
  };
}
// Returns a function that puts back the figure's current background flag and
// number.
function fvSaveHasBackground(figUid) {
  const hasBackground = hasBgByFigure[figUid];
  const slot = _fvBackgroundSlot(figUid);
  const seq = _fvSlotSeq[slot] || 0;
  return () => {
    hasBgByFigure[figUid] = hasBackground;
    _fvSlotSeq[slot] = seq;
  };
}

// Why the last failed request failed, for callers that report it
// (flexvizApply). Only the latest failure is kept.
let _fvLastUpdateError = null;

function _fvErrorDetail(body) {
  const detail = body && body.detail;
  if (Array.isArray(detail)) return detail.map(d => (d && d.msg) || String(d)).join('; ');
  return detail ? String(detail) : '';
}

// After a gesture on `sourceFigUid` wrote `keys` (linked keys included): redraw
// the other figures it moved, since rendering follows state and the gesture's
// own figure already shows its range (pass null to redraw them all). Then
// re-aggregate, in one request, every figure where a changed axis binds a trace
// (a line's x, not its y). Mirrors the server gate. The redraws post nothing:
// the relayout handler ignores what Plotly.react emits. Returns the request's
// promise, or undefined when nothing needs a fetch.
function fvCommitViewportChange(sourceFigUid, keys) {
  if (!keys.length) return;
  const figUids = fvFiguresOfKeys(keys);
  for (const figUid of figUids) {
    if (figUid !== sourceFigUid) _fvRenderFigure(figUid);
  }
  const needsFetch = figUids.some(figUid => fvNeedsFetch(
    figUid,
    keys.filter(k => k.startsWith(figUid + '/')).map(k => k.slice(figUid.length + 1))
  ));
  if (!needsFetch) return;
  return postDashboardUpdate({
    type: 'viewport',
    viewport_keys: keys,
    selections: DASHBOARD_SPEC.state.selections || [],
    force_update: false,
  });
}

async function postDashboardUpdate(event) {
  _fvLastUpdateError = null;
  // Taken in the same synchronous step that serializes the spec, so the
  // number orders the requests by the state they carry.
  const seq = fvNextWriteSeq();
  const cacheKey = fvCacheKeyFor(event);
  let data;
  // Client-side init cache: replay the unfiltered response without a fetch.
  // Whole-dashboard blob first (init / deselect); then the figure-scoped
  // slice (a per-figure reset or autorange to full range with no other filters).
  const cachedFigureDeltas = fvCacheGet(cacheKey) || fvCacheGetFigure(event);
  if (cachedFigureDeltas) {
    data = { figure_deltas: cachedFigureDeltas };
  } else {
    try {
      const resp = await fetch(SERVER_URL + '/dashboard/update', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ spec: DASHBOARD_SPEC, event }),
      });
      if (!resp.ok) {
        const detail = _fvErrorDetail(await resp.json().catch(() => null));
        _fvLastUpdateError = `status ${resp.status}${detail ? ': ' + detail : ''}`;
        console.warn('flexviz /dashboard/update returned', _fvLastUpdateError);
        return false;
      }
      data = await resp.json();
    } catch (e) {
      _fvLastUpdateError = String(e);
      console.warn('flexviz /dashboard/update request failed', e);
      return false;
    }
    fvCachePut(cacheKey, data.figure_deltas);
  }

  const dirtyFigUids = new Set();
  try {
    for (const [figUid, deltas] of Object.entries(data.figure_deltas)) {
      let sawBackground = false;
      if (['deselect', 'init'].includes(event.type)) {
        bgYExtentByFig[figUid] = null;
      }
      for (const delta of deltas) {
        const layerKey = delta.layer || 'base';
        if (Object.prototype.hasOwnProperty.call(delta, 'group_results')) {
          for (const cr of delta.group_results || []) {
            childUidToParentUid[cr.uid] = delta.uid;
          }
          setGroupedLayerData(figUid, delta.uid, layerKey, delta.group_results || [], seq);
          if (layerKey === 'bg') {
            setGroupedLayerData(figUid, delta.uid, 'base', delta.group_results || [], seq);
            sawBackground = true;
            for (const cr of delta.group_results || []) {
              _updateBgYExtent(figUid, (cr.updates || {}).y);
            }
          } else if (layerKey === 'base' && isUnfilteredBaseForFigure(event, figUid)) {
            setGroupedLayerData(figUid, delta.uid, 'bg', delta.group_results || [], seq);
            sawBackground = true;
            for (const cr of delta.group_results || []) {
              _updateBgYExtent(figUid, (cr.updates || {}).y);
            }
          }
        } else {
          setLayerData(delta.uid, layerKey, delta.updates || {}, seq);
          if (layerKey === 'bg') {
            setLayerData(delta.uid, 'base', delta.updates || {}, seq);
            sawBackground = true;
            _updateBgYExtent(figUid, (delta.updates || {}).y);
          } else if (layerKey === 'base' && isUnfilteredBaseForFigure(event, figUid)) {
            setLayerData(delta.uid, 'bg', delta.updates || {}, seq);
            sawBackground = true;
            _updateBgYExtent(figUid, (delta.updates || {}).y);
          }
        }
      }
      if (sawBackground) {
        setHasBackground(figUid, true, seq);
      } else if (event.type === 'viewport' && !isUnfilteredBaseForFigure(event, figUid)) {
        // A zoom under a cross-filter: the background still holds the old range.
        setHasBackground(figUid, false, seq);
      }
      if (deltas.length) dirtyFigUids.add(figUid);
    }
    if (['selection', 'deselect', 'init'].includes(event.type)) {
      _fvAllFigUids.forEach(figUid => dirtyFigUids.add(figUid));
    } else {
      selectionSourceFigureUids(DASHBOARD_SPEC.state.selections || [])
        .forEach(figUid => dirtyFigUids.add(figUid));
    }
  } catch (e) {
    _fvLastUpdateError = `applying the response failed: ${e}`;
    console.warn('flexviz /dashboard/update delta apply failed', e);
    return false;
  }

  for (const figUid of dirtyFigUids) {
    try {
      _fvRenderFigure(figUid);
    } catch (e) {
      console.warn(`flexviz render failed for figure ${figUid}`, e);
    }
  }
  window.fvRefreshSelectionSummary?.();
  // A render error is not an apply failure: the page state still matches the spec.
  return true;
}

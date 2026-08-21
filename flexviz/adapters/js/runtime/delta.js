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

function setLayerData(uid, layerKey, updates) {
  const layers = ensureLayerData(uid);
  layers[layerKey] = cloneObj(updates || {});
}
function setGroupedLayerData(figUid, parentUid, layerKey, groupResults) {
  groupedDataByParent[figUid][parentUid][layerKey] = cloneObj(groupResults || []);
}

async function postDashboardUpdate(event) {
  let data;
  // Client-side init cache: replay the unfiltered response without a fetch.
  // Whole-dashboard blob first (init / reset / deselect); then the figure-scoped
  // slice (a per-figure reset or autorange to full range with no other filters).
  const cachedFigureDeltas = fvCacheGet(event) || fvCacheGetFigure(event);
  if (cachedFigureDeltas) {
    data = { figure_deltas: cachedFigureDeltas };
  } else {
    try {
      const resp = await fetch(SERVER_URL + '/dashboard/update', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ spec: DASHBOARD_SPEC, event }),
      });
      if (!resp.ok) return;
      data = await resp.json();
    } catch (e) {
      console.warn('flexviz /dashboard/update request failed', e);
      return;
    }
    fvCachePut(event, data.figure_deltas);
  }

  const dirtyFigUids = new Set();
  try {
    for (const [figUid, deltas] of Object.entries(data.figure_deltas)) {
      const sourceFigure = figureHasSelectionSource(figUid, event.selections || []);
      let sawBackground = false;
      if (['deselect', 'reset', 'init'].includes(event.type)) {
        bgYExtentByFig[figUid] = null;
      }
      for (const delta of deltas) {
        const layerKey = delta.layer || 'base';
        if (Object.prototype.hasOwnProperty.call(delta, 'group_results')) {
          for (const cr of delta.group_results || []) {
            childUidToParentUid[cr.uid] = delta.uid;
          }
          setGroupedLayerData(figUid, delta.uid, layerKey, delta.group_results || []);
          if (layerKey === 'bg') {
            setGroupedLayerData(figUid, delta.uid, 'base', delta.group_results || []);
            sawBackground = true;
            for (const cr of delta.group_results || []) {
              _updateBgYExtent(figUid, (cr.updates || {}).y);
            }
          } else if (layerKey === 'base' && isUnfilteredBaseForFigure(event, figUid)) {
            setGroupedLayerData(figUid, delta.uid, 'bg', delta.group_results || []);
            sawBackground = true;
            for (const cr of delta.group_results || []) {
              _updateBgYExtent(figUid, (cr.updates || {}).y);
            }
          }
        } else {
          setLayerData(delta.uid, layerKey, delta.updates || {});
          if (layerKey === 'bg') {
            setLayerData(delta.uid, 'base', delta.updates || {});
            sawBackground = true;
            _updateBgYExtent(figUid, (delta.updates || {}).y);
          } else if (layerKey === 'base' && isUnfilteredBaseForFigure(event, figUid)) {
            setLayerData(delta.uid, 'bg', delta.updates || {});
            sawBackground = true;
            _updateBgYExtent(figUid, (delta.updates || {}).y);
          }
        }
      }
      if (sawBackground) {
        hasBgByFigure[figUid] = true;
      } else if (
        event.type === 'viewport'
        && requestHasActiveSelections(event)
        && !sourceFigure
      ) {
        hasBgByFigure[figUid] = false;
      }
      if (deltas.length) dirtyFigUids.add(figUid);
    }
    if (['selection', 'deselect', 'reset', 'init'].includes(event.type)) {
      _fvAllFigUids.forEach(figUid => dirtyFigUids.add(figUid));
    } else {
      selectionSourceFigureUids(DASHBOARD_SPEC.state.selections || [])
        .forEach(figUid => dirtyFigUids.add(figUid));
    }
  } catch (e) {
    console.warn('flexviz /dashboard/update delta apply failed', e);
    return;
  }

  if (['deselect', 'reset', 'init'].includes(event.type)) {
    _resetTreemapLevel = true;
  }
  for (const figUid of dirtyFigUids) {
    try {
      _fvRenderFigure(figUid);
    } catch (e) {
      console.warn(`flexviz render failed for figure ${figUid}`, e);
    }
  }
  _resetTreemapLevel = false;
  window.fvRefreshSelectionSummary?.();
}

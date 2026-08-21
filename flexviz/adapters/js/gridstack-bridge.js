// === FlexViz GridStack bridge ===
// Only included when layout.draggable is true.
// Requires: _FV_CELL_HEIGHT constant set by Python init block before this runs.

(function _fvInitGridStack() {
  if (typeof GridStack === 'undefined') return;

  const _fvGrid = GridStack.init({
    column: 12,
    cellHeight: typeof _FV_CELL_HEIGHT !== 'undefined' ? _FV_CELL_HEIGHT : 80,
    animate: true,
    float: false,
    cancel: '.fv-panel-bar, .fv-panel-bar *',
  });
  if (!DASHBOARD_SPEC.layout) DASHBOARD_SPEC.layout = {};
  if (typeof DASHBOARD_SPEC.layout.grid_editable !== 'boolean') {
    DASHBOARD_SPEC.layout.grid_editable = false;
  }
  window.fvSetGridEditable = function(editable) {
    const nextEditable = !!editable;
    DASHBOARD_SPEC.layout.grid_editable = nextEditable;
    document.body.dataset.fvGridEditable = nextEditable ? 'true' : 'false';
    if (typeof _fvGrid.enableMove === 'function') _fvGrid.enableMove(nextEditable);
    if (typeof _fvGrid.enableResize === 'function') _fvGrid.enableResize(nextEditable);
  };
  window.fvSetGridEditable(DASHBOARD_SPEC.layout.grid_editable === true);
  window.fvUpdateGridButton?.();
  // Sync initial Gridstack positions into DASHBOARD_SPEC immediately after init.
  DASHBOARD_SPEC.layout.grid_items = _fvGrid.save(false).map(function(item) {
    return { fig_uid: item.id, x: item.x, y: item.y, w: item.w, h: item.h };
  });

  // Sync drag/resize positions into DASHBOARD_SPEC — NO backend call.
  _fvGrid.on('change', function(event, changedItems) {
    if (!DASHBOARD_SPEC.layout) DASHBOARD_SPEC.layout = {};
    DASHBOARD_SPEC.layout.grid_items = _fvGrid.save(false).map(function(item) {
      return { fig_uid: item.id, x: item.x, y: item.y, w: item.w, h: item.h };
    });
  });

  // After resize, trigger chart resize (renderer-specific).
  _fvGrid.on('resizestop', function(event, el) {
    window._fvResizeChart && window._fvResizeChart(el);
  });

  // Called by fvOnImport to restore grid positions from a loaded spec.
  window._fvRestoreGridLayout = function() {
    var _items = (DASHBOARD_SPEC.layout && DASHBOARD_SPEC.layout.grid_items) || [];
    if (!_items.length) return;
    _fvGrid.load(_items.map(function(gi) {
      return { id: gi.fig_uid, x: gi.x, y: gi.y, w: gi.w, h: gi.h };
    }));
  };

  window._fvGrid = _fvGrid;
})();

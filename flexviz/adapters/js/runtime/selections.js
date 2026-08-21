// === FlexViz shared runtime — selection semantics ===
// Requires: state.js loaded first
//
// Path-based categorical selection (treemap / pie clicks) uses
// fvUpsertPathPredicate to OR sibling predicates while refining or
// replacing along the same hierarchy branch. See Architecture.md.
//
// NOTE (UX): Multi-click OR on treemap/pie follows common BI “additive
// filter” patterns, but we should periodically reassess whether this
// matches natural interactive visual data exploration — e.g. vs
// modifier keys, explicit multi-select mode, or single-resolution
// (replace-only) semantics for hierarchical charts.

function fvSelectionState() {
  if (!DASHBOARD_SPEC.state) DASHBOARD_SPEC.state = {};
  if (!Array.isArray(DASHBOARD_SPEC.state.selections)) DASHBOARD_SPEC.state.selections = [];
  return DASHBOARD_SPEC.state.selections;
}

function figureSelection(figUid, selections) {
  return (selections || fvSelectionState()).find(
    sel => sel && sel.source_figure_uid === figUid
  ) || null;
}

function predicatesForFigure(figUid, selections) {
  const sel = figureSelection(figUid, selections);
  return sel ? (sel.predicates || []) : [];
}

function replaceFigureSelection(figUid, selection, selections) {
  const remaining = (selections || fvSelectionState())
    .filter(sel => sel && sel.source_figure_uid !== figUid);
  if (!selection) return remaining;
  return remaining.concat([{ ...selection, source_figure_uid: figUid }]);
}

function clearFigureSelectionFromList(figUid, selections) {
  return replaceFigureSelection(figUid, null, selections);
}

function selectionMatches(left, right) {
  if (!left || !right) return false;
  if (left.length !== right.length) return false;
  for (let i = 0; i < left.length; i++) {
    const lc = left[i].clauses || [];
    const rc = right[i].clauses || [];
    if (lc.length !== rc.length) return false;
    for (let j = 0; j < lc.length; j++) {
      if (lc[j].column !== rc[j].column) return false;
      if ((lc[j].closed || 'both') !== (rc[j].closed || 'both')) return false;
      const lv = JSON.stringify(lc[j].values || lc[j].range);
      const rv = JSON.stringify(rc[j].values || rc[j].range);
      if (lv !== rv) return false;
    }
  }
  return true;
}

function _fvPredicateClauseValues(predicate) {
  const out = [];
  for (const clause of (predicate && predicate.clauses) || []) {
    if (!clause || typeof clause.column !== 'string') continue;
    const values = clause.values;
    if (!Array.isArray(values) || values.length !== 1) continue;
    out.push({ column: clause.column, value: String(values[0]) });
  }
  return out;
}

function _fvPredicatesEqual(left, right) {
  const lc = _fvPredicateClauseValues(left);
  const rc = _fvPredicateClauseValues(right);
  if (lc.length !== rc.length) return false;
  for (let i = 0; i < lc.length; i++) {
    if (lc[i].column !== rc[i].column || lc[i].value !== rc[i].value) return false;
  }
  return true;
}

/** True when ``ancestor`` is a strict prefix path of ``descendant``. */
function _fvIsPathAncestor(ancestor, descendant) {
  const ac = _fvPredicateClauseValues(ancestor);
  const dc = _fvPredicateClauseValues(descendant);
  if (!ac.length || ac.length >= dc.length) return false;
  for (let i = 0; i < ac.length; i++) {
    if (ac[i].column !== dc[i].column || ac[i].value !== dc[i].value) return false;
  }
  return true;
}

/**
 * Update a figure's categorical path predicates for treemap/pie clicks.
 *
 * - Exact re-click removes that predicate (toggle off).
 * - Strict ancestor/descendant on the same branch replaces the other.
 * - Independent paths append (OR semantics via multiple predicates).
 */
function upsertPathPredicate(existingPredicates, newPredicate) {
  const predicates = Array.isArray(existingPredicates)
    ? existingPredicates.slice()
    : [];
  const incoming = newPredicate || { clauses: [] };
  if (!Array.isArray(incoming.clauses) || incoming.clauses.length === 0) {
    return predicates;
  }

  const exactIdx = predicates.findIndex(p => _fvPredicatesEqual(p, incoming));
  if (exactIdx >= 0) {
    predicates.splice(exactIdx, 1);
    return predicates;
  }

  const remaining = predicates.filter(p => {
    return !_fvIsPathAncestor(incoming, p) && !_fvIsPathAncestor(p, incoming);
  });
  remaining.push(incoming);
  return remaining;
}

function colsForTrace(ts) {
  const bd = (ts && ts.backend_data) || {};
  const isHoriz = ts && ts.params && ts.params.orientation === 'h';
  let xCol, yCol;
  if (isHoriz) {
    xCol = bd.values || bd.x || null;
    const lbl = bd.labels;
    yCol = lbl ? (Array.isArray(lbl) ? (lbl.length === 1 ? lbl[0] : null) : lbl) : (bd.y || null);
  } else {
    const lbl = bd.labels;
    xCol = lbl ? (Array.isArray(lbl) ? (lbl.length === 1 ? lbl[0] : null) : lbl) : (bd.x || null);
    yCol = bd.values || bd.y || null;
  }
  return { xCol, yCol };
}

window.fvSelectionState = fvSelectionState;
window.fvFigureSelection = figureSelection;
window.fvPredicatesForFigure = predicatesForFigure;
window.fvReplaceFigureSelection = replaceFigureSelection;
window.fvClearFigureSelectionFromList = clearFigureSelectionFromList;
window.fvSelectionMatches = selectionMatches;
window.fvUpsertPathPredicate = upsertPathPredicate;
window.fvColsForTrace = colsForTrace;

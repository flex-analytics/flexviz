// === FlexViz shared runtime — cross-filter selection summaries ===
// Requires: panel.js, state.js, selections.js, delta.js loaded first

function _fvPanelInfoSlot(figUid) {
  if (!figUid || typeof figUidToIdx === 'undefined') return null;
  const figIdx = figUidToIdx[figUid];
  if (figIdx === undefined) return null;
  const controls = window.fvPanelControlRoot?.(figIdx);
  return controls ? controls.querySelector('[data-slot="info"]') : null;
}

function _fvClearPanelSelectionEchoes() {
  for (const fig of (DASHBOARD_SPEC.figures || [])) {
    const slot = _fvPanelInfoSlot(fig && fig.uid);
    if (slot) {
      slot.replaceChildren();
      slot.title = '';
    }
  }
}

function _fvFormatSummaryNumber(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return null;
  const text = num.toFixed(3).replace(/\.?0+$/, '');
  return text === '-0' ? '0' : text;
}

function _fvFormatSummaryIsoDatetime(value) {
  if (typeof value !== 'string') return null;
  if (!/^\d{4}-\d{2}-\d{2}T/.test(value)) return null;
  const parsed = new Date(value);
  if (!Number.isFinite(parsed.valueOf())) return null;
  const iso = parsed.toISOString();
  return iso.replace('T', ' ').replace('.000Z', ' UTC').replace('Z', ' UTC');
}

function _fvFormatSummaryValue(value) {
  if (typeof value === 'number') return _fvFormatSummaryNumber(value);
  if (typeof value === 'string') return _fvFormatSummaryIsoDatetime(value) || value;
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  return null;
}

function _fvClauseSummaryData(clause) {
  if (!clause || typeof clause.column !== 'string' || !clause.column) return null;
  if (Array.isArray(clause.range) && clause.range.length === 2) {
    const lo = _fvFormatSummaryValue(clause.range[0]);
    const hi = _fvFormatSummaryValue(clause.range[1]);
    if (lo == null || hi == null) return null;
    return {
      column: clause.column,
      valueText: lo + ' to ' + hi,
      text: clause.column + ' ' + lo + ' to ' + hi,
    };
  }
  if (Array.isArray(clause.values) && clause.values.length > 0) {
    const values = clause.values.map(_fvFormatSummaryValue);
    if (values.some(v => v == null)) return null;
    return {
      column: clause.column,
      valueText: values.length === 1 ? ('= ' + values[0]) : ('in [' + values.join(', ') + ']'),
      text: values.length === 1
        ? (clause.column + ' = ' + values[0])
        : (clause.column + ' in [' + values.join(', ') + ']'),
    };
  }
  return null;
}

function _fvPredicateSummaryData(predicate) {
  if (!predicate || !Array.isArray(predicate.clauses) || predicate.clauses.length === 0) {
    return null;
  }
  const clauses = predicate.clauses.map(_fvClauseSummaryData);
  if (clauses.some(item => item == null)) return null;
  return {
    clauses,
    text: clauses.map(item => item.text).join(' | '),
  };
}

function _fvSelectionSummaryData(sel) {
  if (!sel || !Array.isArray(sel.predicates) || sel.predicates.length === 0) return null;
  const predicates = sel.predicates.map(_fvPredicateSummaryData);
  if (predicates.some(item => item == null)) return null;
  return {
    predicates,
    text: predicates.map(item => item.text).join(' OR '),
  };
}

window.fvSummarizeSelection = function(sel) {
  const summary = _fvSelectionSummaryData(sel);
  return summary ? summary.text : null;
};

window.fvFigureLabel = function(figUid) {
  if (!figUid) return null;
  const figIdx = (typeof figUidToIdx !== 'undefined') ? figUidToIdx[figUid] : undefined;
  const figSpec = figSpecByUid[figUid]
    || ((figIdx !== undefined && DASHBOARD_SPEC.figures) ? DASHBOARD_SPEC.figures[figIdx] : null);
  const title = figSpec && figSpec.layout && figSpec.layout.title;
  if (typeof title === 'string' && title.trim()) return title.trim();
  if (title && typeof title === 'object' && typeof title.text === 'string' && title.text.trim()) {
    return title.text.trim();
  }
  return figIdx !== undefined ? ('Figure ' + (figIdx + 1)) : 'Figure';
};

window.fvSetSelectionState = function(selections) {
  if (!DASHBOARD_SPEC.state) DASHBOARD_SPEC.state = {};
  DASHBOARD_SPEC.state.selections = Array.isArray(selections) ? selections : [];
  window.fvRefreshSelectionSummary?.();
  return DASHBOARD_SPEC.state.selections;
};

window.fvRemoveSelectionByFigure = async function(figUid) {
  const remaining = window.fvClearFigureSelectionFromList?.(figUid, window.fvSelectionState?.())
    || [];
  window.fvSetSelectionState?.(remaining);
  await postDashboardUpdate({
    type: remaining.length ? 'selection' : 'deselect',
    axis_ranges: {},
    selections: remaining,
    force_update: true,
    figure_uid: figUid,
  });
};

function _fvAppendSelectionSummary(container, summaryData) {
  if (!container || !summaryData) return;
  const fragment = document.createDocumentFragment();
  summaryData.predicates.forEach((predicate, predicateIdx) => {
    if (predicateIdx > 0) {
      const orSep = document.createElement('span');
      orSep.className = 'fv-filter-or';
      orSep.textContent = ' OR ';
      fragment.appendChild(orSep);
    }
    const predicateNode = document.createElement('span');
    predicateNode.className = 'fv-filter-predicate';
    predicate.clauses.forEach((clause, clauseIdx) => {
      if (clauseIdx > 0) {
        const sep = document.createElement('span');
        sep.className = 'fv-filter-sep';
        sep.textContent = ' | ';
        predicateNode.appendChild(sep);
      }
      const clauseNode = document.createElement('span');
      clauseNode.className = 'fv-filter-clause';

      const field = document.createElement('span');
      field.className = 'fv-filter-field';
      field.textContent = clause.column;

      const value = document.createElement('span');
      value.className = 'fv-filter-value';
      value.textContent = clause.valueText;

      clauseNode.appendChild(field);
      clauseNode.appendChild(document.createTextNode(' '));
      clauseNode.appendChild(value);
      predicateNode.appendChild(clauseNode);
    });
    fragment.appendChild(predicateNode);
  });
  container.appendChild(fragment);
}

window.fvRefreshSelectionSummary = function() {
  const strip = document.getElementById('fv-filter-strip');
  const chipsRoot = document.getElementById('fv-filter-chips');
  _fvClearPanelSelectionEchoes();
  if (!strip || !chipsRoot) return;

  const summaries = [];
  for (const sel of (window.fvSelectionState?.() || [])) {
    const figUid = sel && sel.source_figure_uid;
    const summaryData = _fvSelectionSummaryData(sel);
    if (!figUid || !summaryData) continue;
    summaries.push({
      figUid,
      figureLabel: window.fvFigureLabel?.(figUid) || 'Figure',
      summary: summaryData.text,
      summaryData,
    });
    const slot = _fvPanelInfoSlot(figUid);
    if (slot) {
      slot.replaceChildren();
      _fvAppendSelectionSummary(slot, summaryData);
      slot.title = summaryData.text;
    }
  }

  chipsRoot.replaceChildren();
  strip.hidden = summaries.length === 0;
  for (const item of summaries) {
    const chip = document.createElement('div');
    chip.className = 'fv-filter-chip';
    chip.dataset.figureUid = item.figUid;

    const source = document.createElement('span');
    source.className = 'fv-filter-chip-source';
    source.textContent = item.figureLabel;

    const text = document.createElement('span');
    text.className = 'fv-filter-chip-text';
    _fvAppendSelectionSummary(text, item.summaryData);
    text.title = item.summary;

    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'fv-filter-chip-remove';
    remove.textContent = '×';
    remove.setAttribute('aria-label', 'Clear filter from ' + item.figureLabel);
    remove.onclick = () => window.fvRemoveSelectionByFigure?.(item.figUid);

    chip.appendChild(source);
    chip.appendChild(text);
    chip.appendChild(remove);
    chipsRoot.appendChild(chip);
  }
  window.fvSyncFilterStripInset?.();
};

// The filter strip is a position:fixed bottom bar that slides via opacity /
// transform (it keeps display:flex even when hidden), so it never shifts the
// dashboard layout. On short viewports that means it can cover the bottom of
// the last figure. Reserve matching bottom padding on <body> while it is
// visible so the figure can always be scrolled clear of it. Its height is
// dynamic (chips wrap), so keep the inset in sync via a ResizeObserver too.
window.fvSyncFilterStripInset = function() {
  const strip = document.getElementById('fv-filter-strip');
  if (!strip || !document.body) return;
  document.body.style.paddingBottom = strip.hidden ? '' : (strip.offsetHeight + 'px');
};
(function _fvObserveFilterStrip() {
  const strip = document.getElementById('fv-filter-strip');
  if (!strip || typeof ResizeObserver === 'undefined') return;
  new ResizeObserver(() => window.fvSyncFilterStripInset?.()).observe(strip);
})();

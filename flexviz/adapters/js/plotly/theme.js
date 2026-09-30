// === Plotly adapter — theme template ===
// Requires: state.js (fvThemeToken, _fvPalette), traces.js
// The Plotly layout template is built from the CSS tokens, so theme.css drives
// the chrome and the plots. A template only fills what the figure layout
// leaves unset: update_layout(...) values still win.

// A figure layout that brings its own template keeps it.
const _fvFigureHasOwnTemplate = layoutsByFig.map(layout => layout.template !== undefined);

function fvPlotlyTemplate() {
  const text = fvThemeToken('--fv-plot-text');
  const tick = fvThemeToken('--fv-plot-tick');
  const bg = fvThemeToken('--fv-plot-bg');
  const family = fvThemeToken('--fv-plot-font');
  // Tick labels stay at 12px: a smaller size failed the readability check.
  const tickfont = { family: fvThemeToken('--fv-plot-tick-font'), size: 12, color: tick };
  const colorbar = { outlinewidth: 0, thickness: 12, tickfont };
  const axis = {
    gridcolor: fvThemeToken('--fv-plot-grid'),
    linecolor: fvThemeToken('--fv-plot-axis'),
    tickcolor: fvThemeToken('--fv-plot-axis'),
    zeroline: false,
    tickfont,
    title: { font: { size: 12, color: tick } },
  };
  return {
    layout: {
      font: { family, size: 12, color: text },
      paper_bgcolor: bg,
      plot_bgcolor: bg,
      colorway: _fvPalette,
      title: { font: { size: 14, weight: 600, color: text }, x: 0.015, xanchor: 'left' },
      xaxis: { ...axis, showline: true },
      yaxis: axis,
      legend: { font: { size: 12, color: text }, bgcolor: 'rgba(0,0,0,0)' },
      newselection: { line: { color: fvThemeToken('--fv-plot-select'), width: 1.5 } },
    },
    data: {
      heatmap: [{ colorbar }],
      choroplethmap: [{ colorbar }],
      pie: [{ marker: { line: { color: bg, width: 1 } } }],
      treemap: [{ marker: { line: { color: bg, width: 1 } } }],
      box: [{ line: { width: 1.5 } }],
    },
  };
}

// Read the tokens again and redraw every drawn figure. Runs at load (no
// figure is drawn yet, so it only sets the templates) and on a mode change.
// The redraw is a Plotly.react from state, guarded so no Plotly event from it
// reaches the viewport or selection handlers.
window.fvApplyTheme = function() {
  const template = fvPlotlyTemplate();
  layoutsByFig.forEach((layout, figIdx) => {
    if (!_fvFigureHasOwnTemplate[figIdx]) layout.template = template;
  });
  for (const figUid of _fvAllFigUids) {
    if (divs[figUidToIdx[figUid]]?._fullLayout) {
      fvRunProgrammaticPlotlyOp(figUid, () => _fvRenderFigure(figUid));
    }
  }
};
window.fvApplyTheme();

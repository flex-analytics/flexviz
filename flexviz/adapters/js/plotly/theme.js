// === Plotly adapter — theme template ===
// Requires: state.js (fvThemeToken, _fvPalette), traces.js
// The Plotly layout template is built from the CSS tokens, so theme.css drives
// the chrome and the plots. A template only fills what the figure layout
// leaves unset: update_layout(...) values still win.

// A figure layout that brings its own template keeps it.
const _fvFigureHasOwnTemplate = layoutsByFig.map(layout => layout.template !== undefined);

function fvIsDarkMode() {
  return document.documentElement.dataset.fvMode === 'dark';
}

// The OpenStreetMap tiles of Plotly's 'open-street-map' style, darkened by
// MapLibre raster paint. The id differs from the light style, so Plotly sets
// the style again on a mode switch.
const _FV_DARK_MAP_STYLE = {
  id: 'fv-osm-dark',
  version: 8,
  sources: {
    'plotly-osm-tiles': {
      type: 'raster',
      attribution: '© <a target="_blank" href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
      tileSize: 256,
    },
  },
  layers: [{
    id: 'plotly-osm-tiles',
    type: 'raster',
    source: 'plotly-osm-tiles',
    minzoom: 0,
    maxzoom: 22,
    // Swapping the brightness bounds inverts the tiles; the hue turn keeps
    // water blue, and less saturation keeps the land grey under the cells.
    paint: {
      'raster-brightness-min': 0.9,
      'raster-brightness-max': 0.1,
      'raster-hue-rotate': 180,
      'raster-saturation': -0.9,
    },
  }],
  glyphs: 'https://fonts.openmaptiles.org/{fontstack}/{range}.pbf',
};

// Viridis without its three darkest stops, so it starts at #424086: the
// darkest stops almost vanish on a dark plot. The spec stores only the scale
// name, so this also applies to a Viridis that the user picked by name.
const _FV_DARK_VIRIDIS = [
  '#424086', '#3b528b', '#33638d', '#2c728e', '#26828e', '#21918c', '#1fa088',
  '#28ae80', '#3fbc73', '#5ec962', '#84d44b', '#addc30', '#d8e219', '#fde725',
].map((color, i, stops) => [i / (stops.length - 1), color]);

// The template's neutral hover label, framed in the series color: text on
// the series color itself reads at about 3:1. A trace value beats the layout,
// so a border from the figure layout or its own template skips this.
function fvApplySeriesHoverBorders(traces, figUid) {
  const figIdx = figUidToIdx[figUid];
  if (_fvFigureHasOwnTemplate[figIdx] || layoutsByFig[figIdx].hoverlabel?.bordercolor !== undefined) return;
  for (const trace of traces) {
    const seriesColor = !isHeatmapScaledTrace(trace) && (trace.line?.color || trace.marker?.color);
    if (seriesColor) trace.hoverlabel = { ...trace.hoverlabel, bordercolor: seriesColor };
  }
}

function fvThemeColorScale(colorscale) {
  return colorscale === 'Viridis' && fvIsDarkMode() ? _FV_DARK_VIRIDIS : colorscale;
}

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
      // Long category labels widen the left margin instead of being cut off.
      // A margin only grows past the figure's own margin.l, so labels that
      // fit keep the plot area where it is.
      yaxis: { ...axis, automargin: true },
      legend: { font: { size: 12, color: text }, bgcolor: 'rgba(0,0,0,0)' },
      // One neutral label for every trace. fvApplySeriesHoverBorders frames
      // the label of a series in its color.
      hoverlabel: {
        bgcolor: fvThemeToken('--fv-plot-tooltip-bg'),
        bordercolor: fvThemeToken('--fv-plot-tooltip-border'),
        font: { family, size: 12, color: fvThemeToken('--fv-plot-tooltip-text') },
      },
      newselection: { line: { color: fvThemeToken('--fv-plot-select'), width: 1.5 } },
      map: { style: fvIsDarkMode() ? _FV_DARK_MAP_STYLE : 'open-street-map' },
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
    if (!_fvFigureHasOwnTemplate[figIdx]) {
      layout.template = template;
    } else if (layout.map && !layout.map.style && !layout.template.layout?.map?.style) {
      // Plotly's default map style loads CARTO tiles.
      layout.map.style = 'open-street-map';
    }
  });
  for (const figUid of _fvAllFigUids) {
    if (divs[figUidToIdx[figUid]]?._fullLayout) {
      fvRunProgrammaticPlotlyOp(figUid, () => _fvRenderFigure(figUid));
    }
  }
};
window.fvApplyTheme();

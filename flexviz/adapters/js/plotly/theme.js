// === Plotly adapter — theme template ===
// Requires: state.js (fvThemeToken, _fvPalette), traces.js
// The Plotly layout template is built from the CSS tokens, so theme.css drives
// the chrome and the plots. A template only fills what the figure layout
// leaves unset: update_layout(...) values still win.

// A figure layout that brings its own template keeps it. Python None arrives
// as null, which Plotly reads as unset, so the theme reads it that way too.
const _fvFigureHasOwnTemplate = layoutsByFig.map(layout => layout.template != null);

function fvIsDarkMode() {
  return document.documentElement.dataset.fvMode === 'dark';
}

const _fvColorCanvas = document.createElement('canvas').getContext('2d', { willReadFrequently: true });

// 'light' or 'dark': the mode of a figure's own plot background. A see-through
// plot background shows the paper, and a see-through paper shows the page.
function fvPlotSurfaceMode(fullLayout) {
  for (const color of [fullLayout.plot_bgcolor, fullLayout.paper_bgcolor]) {
    _fvColorCanvas.clearRect(0, 0, 1, 1);
    _fvColorCanvas.fillStyle = color;
    _fvColorCanvas.fillRect(0, 0, 1, 1);
    const [r, g, b, a] = _fvColorCanvas.getImageData(0, 0, 1, 1).data;
    if (a < 128) continue;
    // Below a relative luminance of 0.18, light text has the higher contrast.
    const linear = c => ((c /= 255) <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
    return 0.2126 * linear(r) + 0.7152 * linear(g) + 0.0722 * linear(b) < 0.18 ? 'dark' : 'light';
  }
  return fvIsDarkMode() ? 'dark' : 'light';
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

// The theme parts that a template cannot hold, because they are per trace.
// A series hover label keeps the neutral template colors, framed in the series
// color: text on the series color itself reads at about 3:1. A trace value
// beats the layout, so a border from the figure layout skips this.
function fvApplyThemeToTraces(traces, figUid) {
  const figIdx = figUidToIdx[figUid];
  if (_fvFigureHasOwnTemplate[figIdx]) return;
  const seriesBorder = layoutsByFig[figIdx].hoverlabel?.bordercolor == null;
  for (const trace of traces) {
    if (trace.colorscale === 'Viridis' && fvIsDarkMode()) trace.colorscale = _FV_DARK_VIRIDIS;
    const seriesColor = seriesBorder && !isHeatmapScaledTrace(trace) && (trace.line?.color || trace.marker?.color);
    if (seriesColor) trace.hoverlabel = { ...trace.hoverlabel, bordercolor: seriesColor };
  }
}

// A key that the figure's own font sets reaches every text, as it does
// without a template, so the template drops that key from its text fonts.
// The hover label keeps its font: it has its own background.
function fvPlotlyTemplate(ownFont) {
  const ownKeys = Object.keys(ownFont ?? {});
  const font = f => Object.fromEntries(Object.entries(f).filter(([key]) => !ownKeys.includes(key)));
  const text = fvThemeToken('--fv-plot-text');
  const tick = fvThemeToken('--fv-plot-tick');
  const bg = fvThemeToken('--fv-plot-bg');
  const family = fvThemeToken('--fv-plot-font');
  // Tick labels stay at 12px: a smaller size failed the readability check.
  const tickfont = font({ family: fvThemeToken('--fv-plot-tick-font'), size: 12, color: tick });
  const colorbar = { outlinewidth: 0, thickness: 12, tickfont };
  const axis = {
    gridcolor: fvThemeToken('--fv-plot-grid'),
    linecolor: fvThemeToken('--fv-plot-axis'),
    tickcolor: fvThemeToken('--fv-plot-axis'),
    zeroline: false,
    tickfont,
    title: { font: font({ size: 12, color: tick }) },
  };
  return {
    layout: {
      font: { family, size: 12, color: text },
      paper_bgcolor: bg,
      plot_bgcolor: bg,
      colorway: _fvPalette,
      title: { font: font({ size: 14, weight: 600, color: text }), x: 0.015, xanchor: 'left' },
      xaxis: { ...axis, showline: true },
      // Long category labels widen the left margin instead of being cut off.
      // A margin only grows past the figure's own margin.l, so labels that
      // fit keep the plot area where it is.
      yaxis: { ...axis, automargin: true },
      legend: { font: font({ size: 12, color: text }), bgcolor: 'rgba(0,0,0,0)' },
      // One neutral label for every trace. fvApplyThemeToTraces frames the
      // label of a series in its color.
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
  layoutsByFig.forEach((layout, figIdx) => {
    if (!_fvFigureHasOwnTemplate[figIdx]) {
      layout.template = fvPlotlyTemplate(layout.font);
    } else if (layout.map && !layout.map.style && !layout.template.layout?.map?.style) {
      // Plotly's default map style loads CARTO tiles.
      layout.map.style = 'open-street-map';
    }
  });
  for (const figUid of _fvAllFigUids) {
    const figIdx = figUidToIdx[figUid];
    // No part of a figure with its own template follows the mode.
    if (!_fvFigureHasOwnTemplate[figIdx] && divs[figIdx]?._fullLayout) {
      fvRunProgrammaticPlotlyOp(figUid, () => _fvRenderFigure(figUid));
    }
  }
};
window.fvApplyTheme();

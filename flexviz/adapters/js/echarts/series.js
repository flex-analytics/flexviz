// === ECharts adapter — series construction ===
// Requires: state.js (figSpecByUid, layerDataByUid, groupedDataByParent, etc.)
// Requires: INITIAL_OPTIONS, ECHARTS_HEATMAP_COLOR_SCALES set by Python init

const chartsByFig = {};
const _brushAreasByFig = {};
window.__fvBrushAreasByFig = _brushAreasByFig;
const seriesTemplateByUid = {};
const CATEGORY_DIMMED_OPACITY = 0.28;

function heatmapColorScale(ts) {
  const display = (ts && ts.display) || {};
  if (!Object.prototype.hasOwnProperty.call(display, 'color_scale')) {
    throw new Error('Generated heatmap specs must include explicit color_scale and color_range defaults.');
  }
  const colorScale = display.color_scale;
  if (typeof colorScale !== 'string' || !colorScale) {
    throw new Error('heatmap color_scale must be a non-empty string');
  }
  return colorScale.trim().toLowerCase();
}

function heatmapColorRange(ts) {
  const display = (ts && ts.display) || {};
  if (!Object.prototype.hasOwnProperty.call(display, 'color_range')) {
    throw new Error('Generated heatmap specs must include explicit color_scale and color_range defaults.');
  }
  return display.color_range;
}

function heatmapColors(ts) {
  const colorScale = heatmapColorScale(ts);
  const colors = ECHARTS_HEATMAP_COLOR_SCALES[colorScale];
  if (!colors) {
    throw new Error('Unsupported ECharts heatmap color_scale: ' + colorScale);
  }
  return colors;
}

function applySeriesColor(series, color) {
  if (!series || !color) return series;
  const next = { ...series };
  next.lineStyle = { ...(series.lineStyle || {}), color };
  next.itemStyle = { ...(series.itemStyle || {}), color };
  return next;
}

function buildEChartsTreemapData(updates) {
  const labels = updates.labels || [];
  const parents = updates.parents || [];
  const ids = updates.ids || [];
  const values = updates.values || [];
  const marker = updates.marker || {};
  const colors = marker.colors || [];
  const root = { id: 'root', name: '', children: [] };
  const nodeMap = { root };

  for (let i = 0; i < labels.length; i++) {
    const id = String(ids[i] || labels[i] || '');
    if (id === 'root') continue;
    const node = {
      id,
      name: String(labels[i] ?? ''),
      value: values[i] || 0,
      children: [],
    };
    if (colors[i]) {
      node.itemStyle = { color: colors[i] };
    }
    nodeMap[id] = node;
  }
  for (let i = 0; i < labels.length; i++) {
    const id = String(ids[i] || labels[i] || '');
    if (id === 'root') continue;
    const parentId = String(parents[i] || '');
    const parent = nodeMap[parentId] || root;
    const node = nodeMap[id];
    const parentColor = parent && parent.itemStyle && parent.itemStyle.color;
    if (parentColor) {
      node.itemStyle = {
        ...(node.itemStyle || {}),
        borderColor: parentColor,
        borderWidth: 3,
      };
    }
    parent.children.push(node);
  }
  return root.children;
}

function echartsItemSatisfiesPredicate(ts, item, predicate) {
  if (!ts || !item || !predicate) return false;
  if (ts.trace_type === 'pie') {
    const labels = ts.backend_data && ts.backend_data.labels;
    if (!labels) return false;
    const labelCols = Array.isArray(labels) ? labels : [labels];
    const rawName = item.name;
    if (rawName == null) return false;
    let parts;
    if (labelCols.length === 1) {
      parts = [String(rawName)];
    } else {
      try { parts = JSON.parse(String(rawName)); }
      catch (e) { return false; }
      if (!Array.isArray(parts) || parts.length !== labelCols.length) return false;
    }
    return (predicate.clauses || []).every(clause => {
      const idx = labelCols.indexOf(clause.column);
      if (idx < 0) return true;
      return (clause.values || []).map(String).includes(String(parts[idx]));
    });
  }
  if (ts.trace_type === 'treemap') {
    const path = (ts.params && ts.params.path) || [];
    const itemId = String((item && item.id) || '');
    if (!itemId.startsWith('root/')) return false;
    const idParts = itemId.split('/').slice(1).map(v => decodeURIComponent(String(v)));
    return (predicate.clauses || []).every(clause => {
      const idx = path.indexOf(clause.column);
      if (idx < 0) return true;
      if (idx >= idParts.length) return false;
      return (clause.values || []).map(String).includes(idParts[idx]);
    });
  }
  return false;
}

function applyCategorySelectionToData(figUid, ts, data) {
  if (ts.trace_type !== 'pie' && ts.trace_type !== 'treemap') return data;
  const predicates = window.fvPredicatesForFigure?.(
    figUid,
    (DASHBOARD_SPEC.state && DASHBOARD_SPEC.state.selections) || []
  ) || [];
  return data.map(item => {
    const selected = !predicates.length || predicates.some(
      predicate => echartsItemSatisfiesPredicate(ts, item, predicate)
    );
    const next = {
      ...item,
      itemStyle: {
        ...((item && item.itemStyle) || {}),
        opacity: selected ? 1 : CATEGORY_DIMMED_OPACITY,
      },
    };
    if (ts.trace_type === 'treemap' && Array.isArray(item.children)) {
      next.children = applyCategorySelectionToData(figUid, ts, item.children);
    }
    return next;
  });
}

// Initialise series templates from the Python-generated INITIAL_OPTIONS
FIG_UIDS.forEach(figUid => {
  const figSpec = figSpecByUid[figUid];
  let colorIndex = 0;
  (INITIAL_OPTIONS[figUid].series || []).forEach(series => {
    const tsSpec = figSpec.traces.find(ts => ts.uid === series.id);
    const color = (tsSpec && tsSpec.display && tsSpec.display.color)
      || _fvPalette[colorIndex % _fvPalette.length];
    seriesTemplateByUid[series.id] = applySeriesColor(cloneObj(series), color);
    ensureLayerData(series.id);
    colorIndex += 1;
  });
});

function makeEChartsSeries(ts, uid, name, color) {
  const seriesId = uid || ts.uid;
  const seriesName = name || ((ts.display && ts.display.name) || ts.uid);
  if (ts.trace_type === 'line') {
    const s = {
      id: seriesId,
      type: 'line',
      name: seriesName,
      data: [],
      showSymbol: false,
      smooth: false,
    };
    if (color) {
      s.lineStyle = { color };
      s.itemStyle = { color };
    }
    return s;
  }
  if (ts.trace_type === 'histogram' || ts.trace_type === 'bar') {
    const s = {
      id: seriesId,
      type: 'bar',
      name: seriesName,
      data: [],
    };
    if (ts.trace_type === 'histogram') s.barMaxWidth = 40;
    const barMode = (ts.display && ts.display.bar_mode) || (ts.params && ts.params.bar_mode) || 'group';
    if (ts.trace_type === 'bar' && barMode === 'stack') {
      s.stack = 'bar';
    }
    if (color) s.itemStyle = { color };
    return s;
  }
  if (ts.trace_type === 'pie') {
    const hole = (ts.params && ts.params.hole) || 0;
    const inner = hole ? (hole * 100) + '%' : '0%';
    return {
      id: seriesId,
      type: 'pie',
      name: seriesName,
      data: [],
      radius: [inner, '75%'],
    };
  }
  if (ts.trace_type === 'box') {
    const s = {
      id: seriesId,
      type: 'boxplot',
      name: seriesName,
      data: [],
    };
    if (color) s.itemStyle = { color };
    return s;
  }
  if (ts.trace_type === 'treemap') {
    return {
      id: seriesId,
      type: 'treemap',
      name: seriesName,
      data: [],
      roam: false,
      nodeClick: false,
      breadcrumb: { show: false },
      label: { show: true, formatter: '{b}' },
      upperLabel: { show: true },
      itemStyle: {
        borderColor: '#fafaf8',
        borderWidth: 2,
        gapWidth: 1,
      },
      levels: [
        {
          itemStyle: {
            borderColor: '#fafaf8',
            borderWidth: 2,
            gapWidth: 1,
          },
        },
        {
          upperLabel: { show: true, height: 24 },
          itemStyle: {
            borderColor: '#fafaf8',
            borderWidth: 3,
            gapWidth: 6,
          },
        },
        {
          itemStyle: {
            borderColor: '#fafaf8',
            borderWidth: 2,
            gapWidth: 2,
          },
        },
      ],
    };
  }
  if (ts.trace_type === 'histogram2d' || ts.trace_type === 'corr_heatmap') {
    return {
      id: seriesId,
      type: 'heatmap',
      name: seriesName,
      data: [],
    };
  }
  throw new Error('Unsupported trace type ' + ts.trace_type);
}

function seriesDataFromUpdates(updates, template) {
  if (template && template.type === 'pie') {
    const labels = updates.labels || [];
    const values = updates.values || [];
    const marker = updates.marker || {};
    const colors = marker.colors || [];
    return labels.map((label, i) => {
      const item = { name: label, value: values[i] };
      if (colors[i]) item.itemStyle = { color: colors[i] };
      return item;
    });
  }
  if (template && template.type === 'heatmap') {
    const xs = updates.x || [];
    const ys = updates.y || [];
    const z = updates.z || [];
    const data = [];
    for (let j = 0; j < ys.length; j++) {
      for (let i = 0; i < xs.length; i++) {
        const row = z[j] || [];
        const value = row[i];
        data.push([i, j, value === undefined ? null : value]);
      }
    }
    return data;
  }
  if (template && template.type === 'boxplot') {
    const lower = updates.lowerfence || [];
    const q1 = updates.q1 || [];
    const median = updates.median || [];
    const q3 = updates.q3 || [];
    const upper = updates.upperfence || [];
    const label = updates.x0 ?? updates.y0 ?? template.name;
    return lower.map((_, i) => ({
      name: String(label),
      value: [lower[i], q1[i], median[i], q3[i], upper[i]],
    }));
  }
  if (template && template.type === 'treemap') {
    return buildEChartsTreemapData(updates);
  }
  const xs = updates.x || [];
  const ys = updates.y || [];
  return xs.map((x, i) => [x, ys[i]]);
}

function buildSeriesFromTemplate(figUid, ts, template, logicalUid, renderLayer, updates, opacity, zValue) {
  if (!template) return null;
  let data = seriesDataFromUpdates(updates || {}, template);
  data = applyCategorySelectionToData(figUid, ts, data);
  const series = {
    ...template,
    id: rendererUid(logicalUid, renderLayer),
    data,
    opacity,
    z: zValue,
  };
  series.lineStyle = { ...(template.lineStyle || {}), opacity };
  series.itemStyle = { ...(template.itemStyle || {}), opacity };
  return series;
}

function buildGroupedEChartsChildren(figUid, parentUid, childResults, renderLayer, opacity, zValue) {
  const figSpec = figSpecByUid[figUid];
  const parentSpec = figSpec && figSpec.traces.find(ts => ts.uid === parentUid);
  if (!parentSpec) return [];
  return childResults.map(cr => {
    const color = ensureGroupColor(parentSpec, cr.group_value_key);
    const template = makeEChartsSeries(parentSpec, cr.uid, cr.group_value_key, color);
    return buildSeriesFromTemplate(
      figUid,
      parentSpec,
      template,
      cr.uid,
      renderLayer,
      cr.updates || {},
      opacity,
      zValue
    );
  }).filter(Boolean);
}

function buildSeriesForFigure(figUid) {
  const figSpec = figSpecByUid[figUid];
  const overlayMode = (DASHBOARD_SPEC.state && DASHBOARD_SPEC.state.cross_filter_mode) === 'overlay';
  const selections = (DASHBOARD_SPEC.state && DASHBOARD_SPEC.state.selections) || [];
  const hasSelections = selections.length > 0;
  const sourceFigure = figureHasSelectionSource(figUid, selections);
  const showForeground = overlayMode && hasSelections && !sourceFigure;
  const backgroundOpacity = overlayMode && hasSelections && !sourceFigure ? OVERLAY_BG_OPACITY : 1;
  const bgDataLayer = backgroundDataLayerForFigure(figUid);
  const series = [];
  for (const ts of figSpec.traces) {
    if (isGroupedParent(ts)) {
      const childLayers = groupedDataByParent[figUid][ts.uid] || { base: [], bg: [], fg: [] };
      if (overlayMode) {
        series.push(
          ...buildGroupedEChartsChildren(
            figUid, ts.uid, childLayers[bgDataLayer] || [], 'bg',
            backgroundOpacity, hasSelections && !sourceFigure ? 1 : 2
          )
        );
        if (showForeground) {
          series.push(
            ...buildGroupedEChartsChildren(figUid, ts.uid, childLayers.fg || [], 'fg', 1, 3)
          );
        }
      } else {
        series.push(
          ...buildGroupedEChartsChildren(figUid, ts.uid, childLayers.base || [], 'base', 1, 2)
        );
      }
    } else {
      const template = seriesTemplateByUid[ts.uid];
      const layers = ensureLayerData(ts.uid);
      if (overlayMode) {
        const bgSeries = buildSeriesFromTemplate(
          figUid, ts, template, ts.uid, 'bg', layers[bgDataLayer] || {},
          backgroundOpacity, hasSelections && !sourceFigure ? 1 : 2
        );
        if (bgSeries) series.push(bgSeries);
        if (showForeground) {
          const fgSeries = buildSeriesFromTemplate(
            figUid,
            ts,
            template,
            ts.uid,
            'fg',
            layers.fg || {},
            1,
            3
          );
          if (fgSeries) series.push(fgSeries);
        }
      } else {
        const baseSeries = buildSeriesFromTemplate(
          figUid,
          ts,
          template,
          ts.uid,
          'base',
          layers.base || {},
          1,
          2
        );
        if (baseSeries) series.push(baseSeries);
      }
    }
  }
  return series;
}

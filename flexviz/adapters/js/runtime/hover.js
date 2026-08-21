// === FlexViz shared runtime — linked-hover dispatch (renderer-agnostic) ===
// Requires: state.js loaded first (provides stripLayerSuffix, hoverTargetsByColumn,
//           hoverSourceByTrace, hoverCellsByTraceUid, DASHBOARD_SPEC)

/**
 * Return the active hover mode from the dashboard spec.
 * @param {object} dashboardSpec
 * @returns {string} HoverMode
 */
function getHoverMode(dashboardSpec) {
  return (dashboardSpec && dashboardSpec.client_state && dashboardSpec.client_state.hover_mode) || 'off';
}

/**
 * Compute the set of hover modes available for the current dashboard.
 *
 * A non-off mode is included only when:
 *   - it is in implementedModes,
 *   - at least one source trace declares the mode in source_modes,
 *   - at least one target trace in a DIFFERENT figure declares the mode in
 *     target_modes and shares a backend column with the source.
 *
 * @param {object} dashboardSpec
 * @param {{implementedModes: Set<string>, implementedCellTraceTypes: Set<string>, implementedAxisBandTargetTraceTypes: Set<string>}} implementationGates
 * @returns {Set<string>}
 */
function availableHoverModes(dashboardSpec, implementationGates) {
  const modes = new Set(['off']);
  const {
    implementedModes,
    implementedCellTraceTypes = new Set(),
    implementedAxisBandTargetTraceTypes = new Set(),
  } = implementationGates;

  if (implementedModes.has('axis')) {
    let found = false;
    for (const [srcUid, src] of Object.entries(hoverSourceByTrace)) {
      if (!src.sourceModes.includes('axis')) continue;
      for (const [axisRole, colName] of Object.entries(src.columns || {})) {
        if (!['x', 'y'].includes(axisRole)) continue;
        const targets = (hoverTargetsByColumn[colName] || []).filter(t => {
          if (t.figUid === src.figUid || !t.targetModes.includes('axis')) return false;
          const targetTrace = traceSpecByUid[t.traceUid];
          if (isBinnedAxisTarget(targetTrace)) {
            return implementedAxisBandTargetTraceTypes.has(targetTrace.trace_type);
          }
          return true;
        });
        if (targets.length > 0) { found = true; break; }
      }
      if (found) break;
    }
    if (found) modes.add('axis');
  }

  if (implementedModes.has('cell')) {
    let found = false;
    for (const [srcUid, src] of Object.entries(hoverSourceByTrace)) {
      if (!src.sourceModes.includes('cell')) continue;
      const srcTrace = traceSpecByUid[srcUid];
      if (!srcTrace || !implementedCellTraceTypes.has(srcTrace.trace_type)) continue;
      for (const [, colName] of Object.entries(src.columns || {})) {
        const targets = (hoverTargetsByColumn[colName] || []).filter(
          t => {
            if (t.figUid === src.figUid) return false;
            const targetTrace = traceSpecByUid[t.traceUid];
            if (!targetTrace) return false;
            if (t.targetModes.includes('cell')
                && implementedCellTraceTypes.has(targetTrace.trace_type)) {
              return true;
            }
            if (t.targetModes.includes('axis')) return true;
            return false;
          }
        );
        if (targets.length > 0) { found = true; break; }
      }
      if (found) break;
    }
    if (found) modes.add('cell');
  }

  return modes;
}

/**
 * Pick the projection a hovered source trace should use in the unified "on"
 * hover state. 2D cell sources (heatmaps, geo) project a region ('cell');
 * everything else (lines, 1D histograms) projects a point onto the shared axis
 * ('axis'). Returns null when the trace declares no source capability.
 *
 * @param {string} resolvedUid logical (parent) trace uid
 * @returns {('axis'|'cell'|null)}
 */
function effectiveHoverModeForSource(resolvedUid) {
  const ts = traceSpecByUid[resolvedUid];
  const modes = ts && ts.hover && ts.hover.source_modes;
  if (!modes || !modes.length) return null;
  if (modes.includes('cell') && !modes.includes('axis')) return 'cell';
  return 'axis';
}

/** Return the visual list for figUid, creating it on first use. */
function ensureFigVisuals(result, figUid) {
  if (!result.has(figUid)) result.set(figUid, []);
  return result.get(figUid);
}

/**
 * Push an axis band ('x_band' / 'y_band') for figUid, unless a band of that
 * type already exists for the figure. No-op when either bound is undefined.
 */
function pushBandOnce(result, figUid, axisRole, lo, hi) {
  if (lo === undefined || hi === undefined) return;
  const type = axisRole === 'x' ? 'x_band' : 'y_band';
  const visuals = ensureFigVisuals(result, figUid);
  if (visuals.some(v => v.type === type)) return;
  visuals.push(axisRole === 'x'
    ? { type: 'x_band', x0: lo, x1: hi, role: 'linked' }
    : { type: 'y_band', y0: lo, y1: hi, role: 'linked' });
}

/**
 * Apply mode policy and produce per-figure visual instruction lists.
 * This is a pure function: no DOM access, no Plotly calls.
 *
 * @param {object} event - NormalizedHoverEvent (or null)
 * @param {string} mode  - Active HoverMode
 * @param {object} hoverTargetsByColumn
 * @param {object} hoverSourceByTrace
 * @param {object} hoverCellsByTraceUid
 * @returns {Map<string, Array>} figUid -> Visual[]
 */
function planHoverVisuals(event, mode, hoverTargetsByColumn, hoverSourceByTrace, hoverCellsByTraceUid, implementationGates) {
  const gates = implementationGates || {
    implementedAxisBandTargetTraceTypes: IMPLEMENTED_AXIS_BAND_TARGET_TRACE_TYPES,
  };
  const implementedAxisBandTargetTraceTypes =
    gates.implementedAxisBandTargetTraceTypes || new Set();
  const result = new Map();
  if (mode === 'off' || !event) return result;

  if (mode === 'axis' && event.kind === 'point' && event.coordSpace === 'cartesian') {
    const { sourceFigUid, sourceTraceUid } = event;

    // Resolve logical UID (strip render-layer suffix, resolve grouped child)
    const logicalUid = stripLayerSuffix(sourceTraceUid || '');
    const resolvedUid = childUidToParentUid[logicalUid] || logicalUid;

    // Verify source declares axis mode
    const srcInfo = hoverSourceByTrace[resolvedUid];
    if (!srcInfo || !srcInfo.sourceModes.includes('axis')) return result;

    const requiredMode = 'axis';
    const axisRoles = ['x', 'y'];
    for (const axisRole of axisRoles) {
      const colName = event.columns && event.columns[axisRole];
      if (!colName) continue;
      const val = event.values && event.values[axisRole];
      if (val === undefined || val === null) continue;

      const targets = (hoverTargetsByColumn[colName] || []).filter(
        t => t.figUid !== sourceFigUid && t.targetModes.includes(requiredMode)
      );

      for (const target of targets) {
        const targetTrace = traceSpecByUid[target.traceUid];
        if (isBinnedAxisTarget(targetTrace)) {
          if (!implementedAxisBandTargetTraceTypes.has(targetTrace.trace_type)) continue;
          // Emit x_band or y_band to show which bin the hovered value falls in.
          // The bin bounds live on the axis where the TARGET plots colName
          // (target.axis), which may differ from the axis role the source used
          // (e.g. a line projects 'sin' on y, but an x= histogram bins it on x).
          const cells = hoverCellsByTraceUid[target.traceUid] || [];
          const tAxis = target.axis;
          for (const cell of cells) {
            const b = cell.bounds;
            if (tAxis === 'x' && b.x0 !== undefined && b.x0 <= val && val < b.x1) {
              pushBandOnce(result, target.figUid, 'x', b.x0, b.x1);
              break;
            } else if (tAxis === 'y' && b.y0 !== undefined && b.y0 <= val && val < b.y1) {
              pushBandOnce(result, target.figUid, 'y', b.y0, b.y1);
              break;
            }
          }
          continue;
        }
        // Draw the guide on the axis where the TARGET plots colName
        // (target.axis), not the axis role the source used. A vertical (x=)
        // histogram bins 'sin' on x, but a line plotting 'sin' on y must show a
        // horizontal y-guide at the projected value, not a vertical x-guide.
        const instrType = target.axis === 'x' ? 'x_guide' : 'y_guide';
        const visuals = ensureFigVisuals(result, target.figUid);
        // Deduplicate by type + value
        if (!visuals.some(v => v.type === instrType && v.value === val)) {
          visuals.push({ type: instrType, value: val, role: 'linked' });
        }
      }
    }
  }

  if (mode === 'cell' && event.kind === 'cell' && event.coordSpace === 'cartesian') {
    const { sourceFigUid, sourceTraceUid, bounds, columns } = event;

    // Resolve logical UID
    const logicalUid = stripLayerSuffix(sourceTraceUid || '');
    const resolvedUid = childUidToParentUid[logicalUid] || logicalUid;

    const srcInfo = hoverSourceByTrace[resolvedUid];
    if (!srcInfo || !srcInfo.sourceModes.includes('cell')) return result;

    // Source center for cell matching
    const cx = (bounds.x0 !== undefined && bounds.x1 !== undefined)
      ? (bounds.x0 + bounds.x1) / 2 : null;
    const cy = (bounds.y0 !== undefined && bounds.y1 !== undefined)
      ? (bounds.y0 + bounds.y1) / 2 : null;

    for (const [axisRole, colName] of Object.entries(columns || {})) {
      if (!['x', 'y'].includes(axisRole)) continue;

      const allTargets = (hoverTargetsByColumn[colName] || []).filter(
        t => t.figUid !== sourceFigUid
      );

      for (const target of allTargets) {
        const targetTs = traceSpecByUid[target.traceUid];
        if (!targetTs) continue;
        const targetAxis = target.axis;
        const interval0 = axisRole === 'x' ? bounds.x0 : bounds.y0;
        const interval1 = axisRole === 'x' ? bounds.x1 : bounds.y1;

        // Cell-capable target: a single-cell rect is only meaningful when the
        // source constrains BOTH the target's data axes (i.e. the source is a
        // 2D cell sharing both columns). A 1D histogram source constrains only
        // one axis, so highlight the bin strip (band) on the shared-column axis
        // instead of a spurious single cell.
        if (target.targetModes.includes('cell') && IMPLEMENTED_CELL_TRACE_TYPES.has(targetTs.trace_type)) {
          const sourceIs2D = cx !== null && cy !== null;
          if (!sourceIs2D) {
            pushBandOnce(result, target.figUid, targetAxis, interval0, interval1);
            continue;
          }
          const cells = hoverCellsByTraceUid[target.traceUid] || [];
          let matchedCell = null;
          for (const cell of cells) {
            const b = cell.bounds;
            const xIn = (b.x0 !== undefined && b.x0 <= cx && cx < b.x1);
            const yIn = (b.y0 !== undefined && b.y0 <= cy && cy < b.y1);
            if (xIn && yIn) { matchedCell = cell; break; }
          }
          if (matchedCell) {
            const visuals = ensureFigVisuals(result, target.figUid);
            if (!visuals.some(v => v.type === 'rect')) {
              visuals.push({ type: 'rect', bounds: matchedCell.bounds, coordSpace: 'cartesian', role: 'linked' });
            }
          } else {
            // Fallback: band on matching axis
            pushBandOnce(result, target.figUid, targetAxis, interval0, interval1);
          }
          continue;
        }

        // Binned axis-capable target: emit band
        if (target.targetModes.includes('axis') && IMPLEMENTED_AXIS_BAND_TARGET_TRACE_TYPES.has(targetTs.trace_type)) {
          pushBandOnce(result, target.figUid, targetAxis, interval0, interval1);
          continue;
        }

        // Point-like axis-capable target (line trace, etc.): emit band for interval semantics
        if (target.targetModes.includes('axis')) {
          pushBandOnce(result, target.figUid, targetAxis, interval0, interval1);
        }
      }
    }
  }

  return result;
}

function isBinnedAxisTarget(traceSpec) {
  return !!traceSpec && (traceSpec.trace_type === 'histogram' || traceSpec.trace_type === 'histogram2d');
}

/**
 * Dispatch visual instruction list to a target figure.
 * Calls window.__fvApplyHoverVisuals if registered by the active adapter.
 * @param {string} figUid
 * @param {Array} visuals
 */
function dispatchHoverVisuals(figUid, visuals) {
  if (typeof window.__fvApplyHoverVisuals === 'function') {
    window.__fvApplyHoverVisuals(figUid, visuals);
  }
}

// Initialize the hook slot; Plotly adapter registers its implementation at init time.
window.__fvApplyHoverVisuals = null;

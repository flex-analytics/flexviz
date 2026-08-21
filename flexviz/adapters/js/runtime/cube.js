// === FlexViz shared runtime — client cube store, FVCube decoder, slicing ===
// Requires: state.js, cache.js loaded first (DASHBOARD_SPEC, SERVER_URL, fvCacheReset).
//
// Cube live-brush path (see Architecture.md, "Cube Pre-Aggregation & Live Brushing"):
// a cube is a pre-aggregation of one target trace's dims × the active brush
// column binned to P=2048 (or its category tuples for a categorical source).
// The server builds + encodes it once (FVCube v1); this module decodes the
// blob, stores it under a client-local canonical descriptor key, and
// re-slices it per drag frame — turning every mousemove of a live brush into
// an O(cells) local computation with zero round-trips. Phase 2 adds the
// count/sum/mean/min/max measure algebra, categorical target dims (bar/pie),
// and grouped targets (group_by cols as extra categorical dims).
//
// Design points:
//   * The store key is CLIENT-LOCAL: the server↔client join key is the trace
//     uid (response `trace_cubes`); the key here only needs to be
//     deterministic client-side. Domains the client cannot reproduce
//     bit-exactly (unzoomed = server-resolved) are keyed as `d: null` —
//     a token, never a float.
//   * The store is a byte-bounded LRU (256 MB default) of *decoded* entries
//     (TypedArray views + CSR offsets), cleared with the response cache on
//     restore/import (a new spec invalidates the descriptors).
//   * Slicing/normalization mirrors `Histogram._to_update` so a cube-rendered
//     delta is shaped exactly like a server `TraceDelta` and flows through
//     the existing delta-application path (`setLayerData` + `_fvRenderFigure`).

const _FV_CUBE_P = 2048;
const _FV_CUBE_BOX2D_P = 128; // P₂D — the per-axis resolution of a box2d source
let _FV_CUBE_STORE_BUDGET_BYTES = 256 * 1024 * 1024;

// Test hook: override the store byte budget; returns the previous value.
function fvCubeStoreSetBudget(bytes) {
  const prev = _FV_CUBE_STORE_BUDGET_BYTES;
  _FV_CUBE_STORE_BUDGET_BYTES = bytes;
  return prev;
}

// ---------------------------------------------------------------------------
// Canonical client descriptor key
// ---------------------------------------------------------------------------

// Pinned property-insertion order: {s, free:{c,k,p,d}, t:[{c,k,b,d}|{c,k}],
// m:{a,v}, p, v:1}. `d` is null when the axis is unzoomed; zoomed domains
// use the exact numbers stored in DASHBOARD_SPEC.state.viewport. For a box2d
// source (contract H) `c` is the [cx, cy] column tuple and `d` is the
// per-axis token pair [dx|null, dy|null] — both pass through the generic
// {c,k,p,d} shape. `p` is the gesture's canonical passive key
// (fvCubePassiveKey; null for a zero-passive gesture — byte-identical to the
// Phase-1/2 keys).
function fvCubeKey(desc) {
  return JSON.stringify({
    s: desc.s,
    free: {
      c: desc.free.c,
      k: desc.free.k,
      p: desc.free.p,
      d: desc.free.d ?? null,
    },
    t: (desc.t || []).map(dim => (
      dim.k === 'binned'
        ? { c: dim.c, k: dim.k, b: dim.b, d: dim.d ?? null }
        : { c: dim.c, k: dim.k }
    )),
    // The measure block is {a, v}; corr adds cc (the column list in param
    // order). The client key is client-local (asymmetric keying), so it only
    // needs to be self-consistent — cc is omitted for non-corr measures so
    // their keys stay stable.
    m: desc.m.cc
      ? { a: desc.m.a, v: desc.m.v ?? null, cc: desc.m.cc }
      : { a: desc.m.a, v: desc.m.v ?? null },
    p: desc.p ?? null,
    v: 1,
  });
}

// JS mirror of Python's canonical_passive_key (contract E). The two sides
// NEVER need to match each other (asymmetric keying, spec §3 — the
// server↔client join key is the trace uid in `trace_cubes`); each only
// needs to be deterministic in its own language. Canonical clause shapes
// {"c","r","cl"} / {"c","v"} (values sorted with the default JS comparator —
// the pinned choice), clauses sorted by (column, kind) within a predicate,
// predicate strings sorted within a selection, selection strings sorted in
// the key. Nesting is kept: AND across selections of OR-within-selection.
// Returns null for an empty passive set (the active figure's own selection
// and source_figure_uid=null selections are excluded — re-brush case and
// never-filtering selections respectively).
function fvCubePassiveKey(selections, activeFigUid) {
  const passive = (selections || []).filter(s =>
    s && s.source_figure_uid != null
    && s.source_figure_uid !== activeFigUid
    && (s.predicates || []).length > 0
  );
  if (!passive.length) return null;
  const canonClause = c => (
    Array.isArray(c.values)
      ? { c: c.column, v: c.values.slice().sort() }
      : { c: c.column, r: [c.range[0], c.range[1]], cl: c.closed || 'both' }
  );
  const canonPredicate = p => {
    const clauses = (p.clauses || []).slice().sort((a, b) => {
      if (a.column !== b.column) return a.column < b.column ? -1 : 1;
      const ka = Array.isArray(a.values) ? 'v' : 'r';
      const kb = Array.isArray(b.values) ? 'v' : 'r';
      return ka < kb ? -1 : ka > kb ? 1 : 0;
    });
    return JSON.stringify(clauses.map(canonClause));
  };
  const selectionStrs = passive.map(sel =>
    JSON.stringify((sel.predicates || []).map(canonPredicate).sort())
  ).sort();
  return JSON.stringify(selectionStrs);
}

// ---------------------------------------------------------------------------
// Temporal units (contract G)
// ---------------------------------------------------------------------------

// Parse a Plotly date string ("YYYY-MM-DD[ HH:MM[:SS[.ffffff]]][Z|±HH:MM]", T
// or space separator) AS UTC BY HAND — bare Date.parse is local-time ambiguous
// for the space-separated form — and scale to the column's physical unit
// ("us" | "ms" | "day"). A trailing timezone (Z or ±HH:MM / ±HHMM) is honoured:
// tz-aware Polars columns (Datetime with a time_zone) serialise x WITH an
// offset, so the offset is subtracted to recover the true UTC instant (matching
// the numeric epoch the live cube path uses). Naive strings (no suffix) are
// unchanged. Numbers pass through as epoch-ms (Plotly's numeric date
// representation), rescaled to the unit. Returns NaN when unparseable.
function fvTemporalToPhysical(value, unit) {
  if (typeof value === 'number') {
    if (unit === 'us') return value * 1000;
    if (unit === 'day') return value / 86400000;
    return value;
  }
  const m = /^(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2})(?::(\d{2})(?:\.(\d+))?)?)?(Z|[+-]\d{2}:?\d{2})?$/
    .exec(String(value).trim());
  if (!m) return NaN;
  const frac = ((m[7] || '') + '000000').slice(0, 6); // µs, padded/truncated
  let baseMs = Date.UTC(
    +m[1], +m[2] - 1, +m[3], +(m[4] || 0), +(m[5] || 0), +(m[6] || 0)
  );
  const tz = m[8];
  if (tz && tz !== 'Z') {
    // wall-clock at +HH:MM is (HH:MM) ahead of UTC → subtract to get UTC.
    const offMin = (+tz.slice(1, 3)) * 60 + (+tz.slice(-2));
    baseMs -= (tz[0] === '-' ? -1 : 1) * offMin * 60000;
  }
  const us = baseMs * 1000 + Number(frac);
  if (unit === 'us') return us;
  if (unit === 'day') return us / 86400000000;
  return us / 1000;
}

// Render a snapped physical edge back to the committed UTC string. The value
// is first ceil-ed to an INTEGRAL count of the unit: column values are
// integral in their unit, so membership of a closed-left [lo, hi) range is
// unchanged (lo: v >= e ⟺ v >= ceil(e); hi open: v < e ⟺ v < ceil(e)) and
// the string then represents the bound exactly — the server's
// _typed_range_bounds parses it back bit-exactly (§8.2 round-trip). unit
// "day" renders YYYY-MM-DD (edges are integer days by the day_grid
// construction); us/ms render µs-precision datetimes.
function fvPhysicalToTemporal(value, unit) {
  if (unit === 'day') {
    return new Date(Math.ceil(value) * 86400000).toISOString().slice(0, 10);
  }
  const us = unit === 'us' ? Math.ceil(value) : Math.ceil(value) * 1000;
  const usRem = ((us % 1000) + 1000) % 1000;
  const d = new Date((us - usRem) / 1000);
  const pad = (n, w) => String(n).padStart(w, '0');
  return d.getUTCFullYear()
    + '-' + pad(d.getUTCMonth() + 1, 2)
    + '-' + pad(d.getUTCDate(), 2)
    + ' ' + pad(d.getUTCHours(), 2)
    + ':' + pad(d.getUTCMinutes(), 2)
    + ':' + pad(d.getUTCSeconds(), 2)
    + '.' + pad(d.getUTCMilliseconds(), 3) + pad(usRem, 3);
}

// A physical temporal value (epoch µs/ms, or integer day index) → epoch
// milliseconds — the numeric form a Plotly DATE axis renders directly (it reads
// bare numbers as ms-since-epoch). Inverse of fvTemporalToPhysical's number
// branch; used for a line target's binned bucket axis (contract J + G), whose
// cube is built on the column's physical representation but must render on the
// line's date axis. No ceil/quantize: a line is postRequired (the commit POST
// replaces this approximate envelope), so faithful sub-unit x is preferred.
function fvPhysicalToEpochMs(value, unit) {
  if (unit === 'us') return value / 1000;
  if (unit === 'day') return value * 86400000;
  return value; // 'ms'
}

// JS mirror of Python's day_grid (contract G): integer-day snap grid for
// unit:"day" free axes — w whole days per bin, P' = ceil(span/w) bins.
function fvCubeDayGrid(lo, hi, p) {
  const span = hi - lo;
  const w = Math.max(1, Math.ceil(span / p));
  return { w, pEff: Math.max(0, Math.ceil(span / w)) };
}

// Day-grid snap: same shape as fvCubeSnap but over the integer-day grid
// (width w, count pEff) — all arithmetic exact over integer days.
function fvCubeSnapDay(lo, w, pEff, a, b) {
  let loBin = Math.max(0, Math.min(pEff, Math.floor((a - lo) / w)));
  let hiBin = Math.max(0, Math.min(pEff, Math.floor((b - lo) / w)));
  if (hiBin < loBin) { const t = loBin; loBin = hiBin; hiBin = t; }
  return {
    loBin,
    hiBin,
    edgeLo: lo + loBin * w,
    edgeHi: lo + (hiBin + 1) * w,
  };
}

// Mirror of the trace classes' overlay_style for the engine's active-selection
// rule (filtered_only traces do not recompute bg while fg is active). overlay_style is
// not serialized in TraceSpec — it is a pure function of trace_type (plus
// group_by for bar, see BarPlot.__init__), so derive it rather than invent a
// new wire flag.
const _FV_FILTERED_ONLY_TRACE_TYPES = new Set([
  'box', 'pie', 'treemap', 'histogram2d', 'geo_histogram2d', 'corr_heatmap',
]);
function fvCubeTraceFilteredOnly(ts) {
  if (!ts) return false;
  if (ts.trace_type === 'bar') return !!(ts.params && ts.params.group_by);
  return _FV_FILTERED_ONLY_TRACE_TYPES.has(ts.trace_type);
}

// Structural guard before every store put (plan step 0b): the decoded blob's
// free header must match the key's free descriptor — kind, column tuple,
// p, and domain when the client knows it (`d` null is a token for a
// server-resolved domain, not a constraint). A mismatched blob means the
// server resolved a different source trace (or the protocol drifted): the
// entry must NOT be stored — callers demote the target for the gesture.
function fvCubeHeaderMatchesKey(key, header) {
  let keyFree;
  try { keyFree = (JSON.parse(key) || {}).free; } catch (e) { return false; }
  if (!keyFree) return false;
  const free = (header && header.free) || {};
  // The client cannot see dtypes, so a temporal column is keyed as
  // k:'continuous' (a client-local token); the server header says
  // kind:'temporal' — compatible, not a mismatch.
  const kindOk = keyFree.k === free.kind
    || (keyFree.k === 'continuous' && free.kind === 'temporal');
  if (!kindOk) return false;
  if (keyFree.k === 'categorical') {
    const keyCols = Array.isArray(keyFree.c) ? keyFree.c : [keyFree.c];
    const cols = free.cols || [];
    return keyCols.length === cols.length
      && keyCols.every((c, i) => c === cols[i]);
  }
  if (keyFree.k === 'box2d') {
    // The (x, y) column tuple and p must match; per-axis domains are checked
    // only on axes the client knows (a null token is "server-resolved").
    const keyCols = Array.isArray(keyFree.c) ? keyFree.c : [keyFree.c];
    const cols = free.cols || [];
    if (keyCols.length !== 2 || cols.length !== 2) return false;
    if (!keyCols.every((c, i) => c === cols[i])) return false;
    if (keyFree.p !== free.p) return false;
    const doms = free.domains || [];
    const units = free.units || [null, null];
    const keyD = Array.isArray(keyFree.d) ? keyFree.d : [null, null];
    for (let a = 0; a < 2; a++) {
      // Temporal axis domains are epoch-ms tokens vs the header's physical
      // unit — only comparable for unit:"ms"; skip the check otherwise.
      if (keyD[a] != null && !units[a]) {
        const d = doms[a] || [];
        if (keyD[a][0] !== d[0] || keyD[a][1] !== d[1]) return false;
      }
    }
    return true;
  }
  if (keyFree.p !== free.p) return false;
  // Temporal key domains are epoch-ms tokens while the header domain is in
  // the column's physical unit — only comparable for unit:"ms", so the
  // domain check is skipped for temporal headers.
  if (keyFree.d != null && free.kind !== 'temporal') {
    const dom = free.domain || [];
    if (keyFree.d[0] !== dom[0] || keyFree.d[1] !== dom[1]) return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Byte-bounded LRU store (decoded entries)
// ---------------------------------------------------------------------------

const _fvCubeStore = new Map(); // key -> decoded entry; Map order = LRU order
let _fvCubeStoreBytes = 0;

function fvCubeStoreHas(key) {
  return _fvCubeStore.has(key);
}

function fvCubeStoreGet(key) {
  const entry = _fvCubeStore.get(key);
  if (!entry) return null;
  _fvCubeStore.delete(key); // touch: move to most-recent position
  _fvCubeStore.set(key, entry);
  return entry;
}

// Returns true when the entry was stored. Mirrors the server cache rule
// (cache.py): an entry larger than the whole budget is refused outright —
// the eviction loop skips the entry just added, so storing it would evict
// everything else and still exceed the budget permanently. Callers treat a
// refusal as not-served (the target is demoted for the gesture).
function fvCubeStorePut(key, entry) {
  if (entry.bytes > _FV_CUBE_STORE_BUDGET_BYTES) return false;
  const prev = _fvCubeStore.get(key);
  if (prev) {
    _fvCubeStoreBytes -= prev.bytes;
    _fvCubeStore.delete(key);
  }
  _fvCubeStore.set(key, entry);
  _fvCubeStoreBytes += entry.bytes;
  for (const [k, e] of _fvCubeStore) {
    if (_fvCubeStoreBytes <= _FV_CUBE_STORE_BUDGET_BYTES) break;
    if (k === key) continue; // never evict the entry just added
    _fvCubeStore.delete(k);
    _fvCubeStoreBytes -= e.bytes;
  }
  return true;
}

function fvCubeStoreReset() {
  _fvCubeStore.clear();
  _fvCubeStoreBytes = 0;
  _fvCubeFreeDomains.clear();
  _fvCubeBox2dGrids.clear();
}

// Restore/import must drop decoded cubes alongside the response cache — a new
// spec invalidates the descriptors. fvCacheReset() is exactly the
// restore/import hook (reset/deselect deliberately leave cubes untouched), so
// wrap it rather than duplicating the call site.
const _fvCacheResetBase = fvCacheReset;
fvCacheReset = function () {
  _fvCacheResetBase();
  fvCubeStoreReset();
};

// Server-resolved full-data free blocks seen in decoded headers, keyed by
// (source, column, p). Lets a later gesture on the same unzoomed axis snap
// even before its own cube response lands. Each record carries the
// {domain, unit, w, p_eff} subset of the header free block (unit/w/p_eff
// null/undefined for plain continuous axes).
const _fvCubeFreeDomains = new Map();

function fvCubeRememberFreeDomain(sourceName, column, p, free) {
  if (!free || !Array.isArray(free.domain) || free.domain.length !== 2) return;
  _fvCubeFreeDomains.set(JSON.stringify([sourceName, column, p]), {
    domain: [free.domain[0], free.domain[1]],
    unit: free.unit || null,
    w: free.w,
    p_eff: free.p_eff,
  });
}

function fvCubeFreeDomain(sourceName, column, p) {
  return _fvCubeFreeDomains.get(JSON.stringify([sourceName, column, p])) || null;
}

// Server-resolved box2d free headers seen in decoded headers (contract H),
// keyed by (source, [cx, cy]). Lets a later unzoomed box2d gesture on the
// same source adopt its per-axis snap grid before its own response lands.
const _fvCubeBox2dGrids = new Map();

function _fvCubeRememberBox2dGrid(sourceName, cols, free) {
  if (!free || free.kind !== 'box2d' || !Array.isArray(free.domains)) return;
  _fvCubeBox2dGrids.set(JSON.stringify([sourceName, cols]), free);
}

function _fvCubeRememberedBox2dGrid(sourceName, cols) {
  return _fvCubeBox2dGrids.get(JSON.stringify([sourceName, cols])) || null;
}

// ---------------------------------------------------------------------------
// FVCube v1 decoder
// ---------------------------------------------------------------------------

// Layout (little-endian): magic "FVCUBE" | u8 version | u8 reserved |
// u32 header_len | JSON header (padded so the buffer section is 8-byte
// aligned) | buffer section (each buffer 8-byte aligned; header offsets are
// relative to the section start). Dtypes: u32 (bins, codes, counts) and
// f64 (sum/min/max measure partials; null encoded as NaN).
//
// `bytes` is a Uint8Array whose `.buffer` starts at the blob's first byte
// (byteOffset 0) — decodeCubeBundle slices each blob into its own buffer, so
// the column offsets stay 8-byte aligned for the zero-copy TypedArray views.
function decodeFVCube(bytes) {
  if (bytes.length < 12 || String.fromCharCode(...bytes.subarray(0, 6)) !== 'FVCUBE') {
    throw new Error('bad FVCube magic');
  }
  if (bytes[6] !== 1) throw new Error('unsupported FVCube version ' + bytes[6]);
  const headerLen = new DataView(bytes.buffer, 8, 4).getUint32(0, true);
  if (bytes.length < 12 + headerLen) throw new Error('FVCube blob truncated inside the header');
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(12, 12 + headerLen)));
  const sectionStart = 12 + headerLen;

  const cols = {};
  let bytesTotal = 0;
  for (const col of header.columns) {
    if (!['u32', 'f64', 'f32', 'u16'].includes(col.dtype)) {
      throw new Error('unsupported FVCube dtype ' + col.dtype);
    }
    const abs = sectionStart + col.offset;
    // 8-byte aligned by construction (padded header + padded buffers); every
    // dtype (f64/u32/f32/u16) is therefore safely aligned for its TypedArray.
    if (abs % 8 !== 0) throw new Error('misaligned FVCube buffer ' + col.name);
    if (abs + col.byte_len > bytes.length) {
      throw new Error('FVCube blob truncated: column ' + col.name);
    }
    if (col.dtype === 'f64') {
      cols[col.name] = new Float64Array(bytes.buffer, abs, col.byte_len / 8);
    } else if (col.dtype === 'u32') {
      cols[col.name] = new Uint32Array(bytes.buffer, abs, col.byte_len / 4);
    } else if (col.dtype === 'f32') {
      // line_env y partials (contract J) — already Math.fround-quantized.
      cols[col.name] = new Float32Array(bytes.buffer, abs, col.byte_len / 4);
    } else { // 'u16' — line_env in-bucket x offsets (contract J)
      cols[col.name] = new Uint16Array(bytes.buffer, abs, col.byte_len / 2);
    }
    bytesTotal += col.byte_len;
  }

  // CSR offsets over the free_bin-sorted rows: rows of bin b are
  // binStart[b] .. binStart[b+1]-1. Range kinds span b in 0..P inclusive
  // (degenerate top bin P included), hence P+2 slots — P being the
  // EFFECTIVE bin count (p_eff for an integer-day grid, p otherwise); a
  // categorical free axis has exactly one bin per category tuple (no
  // degenerate bin). A box2d free axis (contract H) has composite bins
  // by*S + bx with S = p+1, so every index 0..S²-1 is reachable ⇒ S² + 1
  // slots (the prefix-sum entry indexes every composite bin).
  let slots;
  if (header.free.kind === 'categorical') {
    slots = header.free.categories.length + 1;
  } else if (header.free.kind === 'box2d') {
    const s = header.free.p + 1;
    slots = s * s + 1;
  } else {
    slots = (header.free.p_eff ?? header.free.p) + 2;
  }
  const freeBin = cols.free_bin;
  const binStart = new Uint32Array(slots);
  for (let i = 0; i < freeBin.length; i++) binStart[freeBin[i] + 1]++;
  for (let b = 1; b < binStart.length; b++) binStart[b] += binStart[b - 1];
  bytesTotal += binStart.byteLength;

  // Target-dim accessors: each dim's code/bin buffer plus its header
  // metadata, in the header's pinned dim order (contract C).
  const targetDims = header.target_dims.map(d => ({
    name: d.name,
    kind: d.kind,
    bins: d.bins,
    domain: d.domain,
    // Physical unit of a temporal binned dim (contract G); null for numeric
    // dims. A line target renders this axis on a Plotly DATE axis, so the
    // physical bucket value must be mapped back to epoch-ms (fvLineEnvCells).
    unit: d.unit ?? null,
    categories: d.categories,
    col: d.kind === 'binned' ? cols['__bin__' + d.name] : cols[d.name],
  }));

  return { header, cols, binStart, targetDims, bytes: bytesTotal };
}

// Cube bundle envelope (mirrors flexviz.cube.encode_cube_bundle). Layout
// (little-endian): magic "FVCBNDL\0" (8) | u8 version | u8 reserved |
// u16 reserved | u32 meta_len | meta (UTF-8 JSON, space-padded to an 8-byte
// boundary) | the FVCube blobs concatenated in index order. Returns
// { trace_cubes, blobs } where each blob is a fresh Uint8Array (its own buffer,
// byteOffset 0) so decodeFVCube's TypedArray views stay 8-byte aligned.
function decodeCubeBundle(arrayBuffer) {
  const bytes = new Uint8Array(arrayBuffer);
  if (bytes.length < 16 ||
      String.fromCharCode(...bytes.subarray(0, 8)) !== 'FVCBNDL\0') {
    throw new Error('bad cube bundle magic');
  }
  if (bytes[8] !== 1) throw new Error('unsupported cube bundle version ' + bytes[8]);
  const metaLen = new DataView(bytes.buffer, bytes.byteOffset + 12, 4).getUint32(0, true);
  if (bytes.length < 16 + metaLen) throw new Error('cube bundle truncated inside meta');
  const meta = JSON.parse(
    new TextDecoder().decode(bytes.subarray(16, 16 + metaLen))
  );
  const blobs = [];
  let off = 16 + metaLen;
  for (const len of meta.lengths) {
    if (off + len > bytes.length) throw new Error('cube bundle truncated inside blobs');
    blobs.push(bytes.slice(off, off + len)); // copy → own 8-byte-aligned buffer
    off += len;
  }
  return { trace_cubes: meta.trace_cubes || {}, blobs };
}

// ---------------------------------------------------------------------------
// Snap + slice + delta (Shared arithmetic; mirrors Histogram._to_update)
// ---------------------------------------------------------------------------

// Natural-floor bin indices of the brush endpoints, clamped to [0, P] — P
// included so a brush reaching the domain max selects the degenerate top bin.
// edge(b) = lo + b*span/P; committed range = [edge(loBin), edge(hiBin+1)).
function fvCubeSnap(domain, p, a, b) {
  const lo = domain[0];
  const span = (domain[1] - lo) || 1.0;
  let loBin = Math.max(0, Math.min(p, Math.floor((a - lo) / span * p)));
  let hiBin = Math.max(0, Math.min(p, Math.floor((b - lo) / span * p)));
  if (hiBin < loBin) { const t = loBin; loBin = hiBin; hiBin = t; }
  return {
    loBin,
    hiBin,
    edgeLo: lo + loBin * span / p,
    edgeHi: lo + (hiBin + 1) * span / p,
  };
}

// Snap a 2-D box on a box2d free axis (contract H): snap each axis
// independently with the same per-axis arithmetic as fvCubeSnap (or the
// integer-day grid for a unit:"day" axis), against that axis's decoded
// physical domain / day-grid. `axes` is [{domain, unit, dayGrid}, ...] for x
// then y. Returns {x:{loBin,hiBin,edgeLo,edgeHi}, y:{...}}.
function fvCubeSnap2d(axes, p, box) {
  const snapAxis = (ax, a, b) => {
    if (ax && ax.dayGrid) {
      return fvCubeSnapDay(ax.domain[0], ax.dayGrid.w, ax.dayGrid.pEff, a, b);
    }
    return fvCubeSnap(ax.domain, p, a, b);
  };
  return {
    x: snapAxis(axes[0], box.x[0], box.x[1]),
    y: snapAxis(axes[1], box.y[0], box.y[1]),
  };
}

// The highest valid free-bin index for an entry: the last category for a
// categorical axis, the composite top S²-1 for box2d (contract H), the
// effective top bin otherwise. Used to clamp slice ranges.
function _fvCubeMaxBin(header) {
  if (header.free.kind === 'categorical') return header.free.categories.length - 1;
  if (header.free.kind === 'box2d') {
    const s = header.free.p + 1;
    return s * s - 1;
  }
  return header.free.p_eff ?? header.free.p;
}

// Build the per-row composite free-bin ranges of a snapped 2-D box (contract
// H): for each by in [ly..hy], one inclusive range [by*S+lx, by*S+hx]. The
// rows of one by form a contiguous CSR block, so the generalized slice walks
// binStart[by*S+lx] .. binStart[by*S+hx+1] exactly. S = header.free.p + 1.
function fvCubeRectRanges(header, snap2d) {
  const s = header.free.p + 1;
  const lx = snap2d.x.loBin, hx = snap2d.x.hiBin;
  const ly = snap2d.y.loBin, hy = snap2d.y.hiBin;
  const ranges = [];
  for (let by = ly; by <= hy; by++) {
    ranges.push([by * s + lx, by * s + hx]);
  }
  return ranges;
}

// Generalized slice (shared contract A): accumulate the measure partials per
// composite target key over the CSR row ranges of the given free-bin ranges
// (inclusive [lo, hi] pairs; a range source passes one), then finalize:
//   count/sum ⇒ Σ; mean ⇒ Σsum/Σcount (Σcount === 0 cells omitted);
//   min/max ⇒ NaN (null) partials skipped, all-NaN cells omitted.
// Output cells: {dims: [...], value} — a binned dim contributes its bin
// index (number), a categorical dim its typed category value.
function fvCubeSliceCells(entry, binRanges) {
  const header = entry.header;
  const dims = entry.targetDims;
  const agg = header.measure.agg;
  const maxBin = _fvCubeMaxBin(header);
  const cnt = entry.cols.count; // count / mean
  const sum = entry.cols.sum;   // sum / mean
  const ext = entry.cols[agg];  // min / max
  const acc = new Map(); // composite code key -> accumulator cell
  for (const range of binRanges) {
    const lo = Math.max(0, range[0]);
    const hi = Math.min(maxBin, range[1]);
    if (hi < lo) continue;
    for (let r = entry.binStart[lo]; r < entry.binStart[hi + 1]; r++) {
      let key = '';
      for (let i = 0; i < dims.length; i++) key += dims[i].col[r] + ':';
      let cell = acc.get(key);
      if (!cell) {
        const codes = new Array(dims.length);
        for (let i = 0; i < dims.length; i++) codes[i] = dims[i].col[r];
        cell = { codes, count: 0, sum: 0, ext: NaN };
        acc.set(key, cell);
      }
      if (agg === 'count' || agg === 'mean') cell.count += cnt[r];
      if (agg === 'sum' || agg === 'mean') cell.sum += sum[r];
      if (agg === 'min') {
        const v = ext[r];
        if (!Number.isNaN(v)) cell.ext = Number.isNaN(cell.ext) ? v : Math.min(cell.ext, v);
      } else if (agg === 'max') {
        const v = ext[r];
        if (!Number.isNaN(v)) cell.ext = Number.isNaN(cell.ext) ? v : Math.max(cell.ext, v);
      }
    }
  }
  const cells = [];
  for (const cell of acc.values()) {
    let value;
    if (agg === 'count') value = cell.count;
    else if (agg === 'sum') value = cell.sum;
    else if (agg === 'mean') {
      if (cell.count === 0) continue; // null cell — omitted like the legacy output
      value = cell.sum / cell.count;
    } else {
      value = cell.ext;
      if (Number.isNaN(value)) continue; // all-null cell — omitted
    }
    cells.push({
      dims: cell.codes.map((code, i) =>
        dims[i].kind === 'binned' ? code : dims[i].categories[code]),
      value,
    });
  }
  return cells;
}

// Sum counts per target bin over the given free-bin ranges (inclusive
// [lo, hi] pairs — the dense ungrouped-hist view over the generalized slice).
function sliceHistCounts(entry, binRanges) {
  const dim = entry.header.target_dims[0];
  const counts = new Float64Array(dim.bins);
  for (const cell of fvCubeSliceCells(entry, binRanges)) {
    counts[cell.dims[0]] += cell.value;
  }
  return counts;
}

// ---------------------------------------------------------------------------
// Categorical predicate → category-code matching (shared contract B)
// ---------------------------------------------------------------------------

// A categorical selection is servable from a cube iff every clause is an
// is_in test ({column, values}) on one of the free-axis columns. A range
// clause, an empty clause set, or a column outside `cols` means the cube
// cannot answer it — the caller must fall back to the legacy POST path.
function fvCubePredicatesServable(cols, predicates) {
  if (!Array.isArray(predicates) || !predicates.length) return false;
  for (const pred of predicates) {
    const clauses = (pred && pred.clauses) || [];
    if (!clauses.length) return false;
    for (const clause of clauses) {
      if (!clause || !cols.includes(clause.column)) return false;
      if (!Array.isArray(clause.values) || !clause.values.length) return false;
    }
  }
  return true;
}

// Selected category codes for a categorical free axis: a category tuple
// (parallel to `freeHeader.cols`) matches a predicate iff every clause's
// column is a free col AND the tuple's part for it is string-equal to a
// member of `values` — OR across predicates, AND across clauses. Treemap
// path predicates constrain a *prefix* of cols; unconstrained deeper levels
// match anything. Returns null when the selection is not servable.
function fvCubeMatchCategoryCodes(freeHeader, predicates) {
  const cols = freeHeader.cols || [];
  if (!fvCubePredicatesServable(cols, predicates)) return null;
  const codes = [];
  (freeHeader.categories || []).forEach((tuple, code) => {
    const match = predicates.some(pred => pred.clauses.every(clause => {
      const part = String(tuple[cols.indexOf(clause.column)]);
      return clause.values.some(v => String(v) === part);
    }));
    if (match) codes.push(code);
  });
  return codes;
}

// Reproduce Histogram._to_update for the ungrouped case: centers at
// lo + (i+0.5)*width over the header's target domain (the server's
// breakpoint-minus-half-width arithmetic), histnorm normalization,
// hover_bounds, horizontal orientation when the trace's prop key is y.
// The result is shaped exactly like a server TraceDelta entry.
function histDeltaFromCounts(traceSpec, header, counts) {
  const dim = header.target_dims[0];
  const lo = dim.domain[0];
  const width = (dim.domain[1] - lo) / dim.bins;
  const centers = new Array(dim.bins);
  for (let i = 0; i < dim.bins; i++) centers[i] = lo + (i + 0.5) * width;

  let values = Array.from(counts);
  const histnorm = (traceSpec.params && traceSpec.params.histnorm) || 'count';
  if (histnorm !== 'count') {
    let total = 0;
    for (const v of values) total += v;
    const w = width || 1.0; // degenerate-bin guard, mirrors _to_update
    if (histnorm === 'percent') {
      values = values.map(v => (total ? v / total * 100 : 0));
    } else if (histnorm === 'probability') {
      values = values.map(v => (total ? v / total : 0));
    } else if (histnorm === 'density') {
      values = values.map(v => v / w);
    } else { // 'probability density'
      values = values.map(v => (total ? v / (total * w) : 0));
    }
  }

  const halfW = width / 2;
  // Prop key = the single backend_data key, exactly how the adapter
  // (Histogram.from_trace_spec) determines orientation.
  const propKey = Object.keys(traceSpec.backend_data || {})[0] || 'x';
  if (propKey === 'x') {
    return {
      uid: traceSpec.uid,
      updates: {
        x: centers,
        y: values,
        hover_bounds: centers.map(c => ({ x0: c - halfW, x1: c + halfW })),
      },
    };
  }
  return {
    uid: traceSpec.uid,
    updates: {
      x: values,
      y: centers,
      orientation: 'h',
      hover_bounds: centers.map(c => ({ y0: c - halfW, y1: c + halfW })),
    },
  };
}

// ---------------------------------------------------------------------------
// Categorical-target deltas (contract D; mirror BarPlot/PiePlot._to_grouped_update)
// ---------------------------------------------------------------------------

// Reproduce Python's compact ``json.dumps(..., ensure_ascii=True)`` for the
// composite-label tuples the cube path serializes (plan step 0d):
// JSON.stringify, then escape every UTF-16 code unit > 0x7E as lowercase
// \uXXXX. Python escapes astral chars as surrogate pairs — matched for free
// by per-code-unit escaping — and 0x7F (DEL), which bare JSON.stringify
// leaves raw. Pinned by golden-string tests against _group_value_key.
function fvJsonDumpsAscii(value) {
  return JSON.stringify(value).replace(/[\u007f-\uffff]/g, ch =>
    '\\u' + ch.charCodeAt(0).toString(16).padStart(4, '0')
  );
}

// Label per contract D: the raw string for a single label column, else the
// JSON-encoded tuple of parts (≡ _group_value_key's compact json.dumps,
// byte-for-byte — non-ASCII parts must escape identically or color_map
// keying and server↔cube delta parity break).
function _fvCubeCellLabel(parts) {
  return parts.length === 1 ? parts[0] : fvJsonDumpsAscii(parts);
}

// Ascending label-tuple order (per-column string <), mirroring the server's
// sort over the label columns. Byte-order vs UTF-16 divergence on astral
// chars is a documented non-goal (contract D).
function _fvCubeSortCells(cells) {
  return cells.slice().sort((a, b) => {
    for (let i = 0; i < a.dims.length; i++) {
      if (a.dims[i] < b.dims[i]) return -1;
      if (a.dims[i] > b.dims[i]) return 1;
    }
    return 0;
  });
}

// Reproduce BarPlot._to_grouped_update for the ungrouped case: sorted labels,
// orientation from params, color_map → per-label marker colors. The result is
// shaped exactly like a server TraceDelta entry.
//
// ``includeColor`` is false for GROUPED children: their colour is the single
// group colour resolved at render time (ensureGroupColor / group_domain),
// exactly like the server's grouped delta which carries no marker. The
// color_map of a grouped bar is keyed by the GROUP values, not the labels, so
// a per-label lookup here would yield an all-null marker that overwrites the
// group colour (every bar renders black) — see _to_grouped_update (bar.py).
function barDeltaFromCells(traceSpec, cells, includeColor = true) {
  const sorted = _fvCubeSortCells(cells);
  const labels = sorted.map(c => _fvCubeCellLabel(c.dims));
  const values = sorted.map(c => c.value);
  const params = traceSpec.params || {};
  const updates = params.orientation === 'h'
    ? { x: values, y: labels, orientation: 'h' }
    : { x: labels, y: values };
  const colorMap = includeColor && traceSpec.display && traceSpec.display.color_map;
  if (colorMap) {
    updates.marker = { color: labels.map(l => colorMap[String(l)] ?? null) };
  }
  return { uid: traceSpec.uid, updates };
}

// Reproduce PiePlot._to_grouped_update: {labels, values}; color_map →
// marker.colors (plural — pie's key, unlike bar's marker.color).
function pieDeltaFromCells(traceSpec, cells) {
  const sorted = _fvCubeSortCells(cells);
  const labels = sorted.map(c => _fvCubeCellLabel(c.dims));
  const updates = { labels, values: sorted.map(c => c.value) };
  const colorMap = traceSpec.display && traceSpec.display.color_map;
  if (colorMap) {
    updates.marker = { colors: labels.map(l => colorMap[String(l)] ?? null) };
  }
  return { uid: traceSpec.uid, updates };
}

// ---------------------------------------------------------------------------
// Line-envelope target (contract J; mirror CubeResult._combine_line_env)
// ---------------------------------------------------------------------------

// The single binned target dim of a line_env cube = the line's x bucket axis.
// All other target dims are categorical group dims (grouped lines).
function _fvLineEnvBucketDim(entry) {
  const dims = entry.targetDims;
  for (let i = 0; i < dims.length; i++) {
    if (dims[i].kind === 'binned') return { dim: dims[i], idx: i };
  }
  return null;
}

// Combine the decoded line_env partials per target cell (group dims × bucket)
// over the CSR row ranges of the given free-bin ranges (contract J). Per
// bucket, y_min = min over the QUANTIZED f32 y partials (NaN-skipped),
// tracking x@ymin dequantized from the winning row's u16 x_argmin offset;
// likewise y_max/x_argmax. Rows are walked in CSR order (free_bin ascending,
// as fvCubeSliceCells does) and strict </> keep the FIRST occurrence on a tie
// — bit-for-bit equal to the Polars reference (quantize-then-combine on both
// sides). Each cell carries its full dim-code tuple (bucket bin index +
// categorical codes) so grouped lines can split on the group dims. Returns
// cells [{codes, bucketIdx, yMin, xAtYmin, yMax, xAtYmax}].
function fvLineEnvCells(entry, binRanges) {
  const header = entry.header;
  const dims = entry.targetDims;
  const bucketInfo = _fvLineEnvBucketDim(entry);
  if (!bucketInfo) return [];
  const bdim = bucketInfo.dim;
  const lo = bdim.domain[0];
  const width = (bdim.domain[1] - lo) / bdim.bins;
  // A temporal bucket axis is binned on the column's physical representation
  // (epoch µs/ms, day index); the line renders on a Plotly DATE axis, which
  // reads bare numbers as epoch-ms — so map physical → epoch-ms (contract G).
  // Without this the raw physical value (e.g. epoch µs, ~1000× an ms) lands
  // millennia off-axis and the panel renders empty.
  const unit = bdim.unit || null;
  const yMinCol = entry.cols.y_min;
  const yMaxCol = entry.cols.y_max;
  const xArgMin = entry.cols.x_argmin;
  const xArgMax = entry.cols.x_argmax;
  const maxBin = _fvCubeMaxBin(header);

  const dequant = (bin, off) => {
    // bucket_lo = lo + bin*width; x = bucket_lo + (off/65535)*width.
    const bucketLo = lo + bin * width;
    const phys = bucketLo + (off / 65535) * width;
    return unit ? fvPhysicalToEpochMs(phys, unit) : phys;
  };

  const acc = new Map(); // composite code key -> cell
  for (const range of binRanges) {
    const rLo = Math.max(0, range[0]);
    const rHi = Math.min(maxBin, range[1]);
    if (rHi < rLo) continue;
    for (let r = entry.binStart[rLo]; r < entry.binStart[rHi + 1]; r++) {
      let key = '';
      for (let i = 0; i < dims.length; i++) key += dims[i].col[r] + ':';
      let cell = acc.get(key);
      if (!cell) {
        const codes = new Array(dims.length);
        for (let i = 0; i < dims.length; i++) codes[i] = dims[i].col[r];
        cell = {
          codes,
          bucketIdx: codes[bucketInfo.idx],
          yMin: NaN, xAtYmin: NaN, yMax: NaN, xAtYmax: NaN,
        };
        acc.set(key, cell);
      }
      const ymin = yMinCol[r];
      const ymax = yMaxCol[r];
      if (!Number.isNaN(ymin) && (Number.isNaN(cell.yMin) || ymin < cell.yMin)) {
        cell.yMin = ymin;
        cell.xAtYmin = dequant(cell.bucketIdx, xArgMin[r]);
      }
      if (!Number.isNaN(ymax) && (Number.isNaN(cell.yMax) || ymax > cell.yMax)) {
        cell.yMax = ymax;
        cell.xAtYmax = dequant(cell.bucketIdx, xArgMax[r]);
      }
    }
  }
  const cells = [];
  for (const cell of acc.values()) {
    // A bucket with no surviving rows (both extrema NaN) is dropped, matching
    // the Polars reference's is_not_nan filter.
    if (Number.isNaN(cell.yMin) || Number.isNaN(cell.yMax)) continue;
    cells.push(cell);
  }
  return cells;
}

// Project a line-x value to a number for gap diffing. Numbers pass through;
// ISO / Plotly date strings parse via fvTemporalToPhysical. The UNIT is
// irrelevant — gap detection is a ratio (diff > 4.1 * median), so any
// consistent monotonic projection yields the same gaps; 'ms' is arbitrary.
function fvLineGapXToNumber(v) {
  if (typeof v === 'number') return v;
  if (v == null) return NaN;
  return fvTemporalToPhysical(v, 'ms');
}

// Insert a null (x, y) break wherever the gap between consecutive x exceeds
// 4.1 * the median diff — a pure DISPLAY nicety applied at render time to every
// line trace (buildTraceFromTemplate), mirroring the heuristic the server used
// to bake in. x may be numeric (live envelope) or ISO strings (committed
// temporal line); fvLineGapXToNumber projects to numbers for diffing while the
// ORIGINAL x values are emitted. add_gaps defaults true at the call site. A
// non-positive median (degenerate / all-equal x) yields no gaps; a null x
// projects to NaN, so its diffs are non-finite and skipped (idempotent over
// already-gapped data).
function fvApplyLineGaps(x, y, addGaps) {
  if (!addGaps || x.length < 2) return { x, y };
  const xn = x.map(fvLineGapXToNumber);
  const diffs = [];
  for (let i = 1; i < xn.length; i++) {
    const d = xn[i] - xn[i - 1];
    if (Number.isFinite(d)) diffs.push(d);
  }
  if (!diffs.length) return { x, y };
  diffs.sort((a, b) => a - b);
  const mid = diffs.length >> 1;
  const med = diffs.length % 2 ? diffs[mid] : (diffs[mid - 1] + diffs[mid]) / 2;
  if (!(med > 0)) return { x, y };
  const threshold = 4.1 * med;
  const outX = [x[0]];
  const outY = [y[0]];
  for (let i = 1; i < x.length; i++) {
    if (xn[i] - xn[i - 1] > threshold) { outX.push(null); outY.push(null); }
    outX.push(x[i]);
    outY.push(y[i]);
  }
  return { x: outX, y: outY };
}

// Build a line-envelope delta from combined cells (contract J): per non-empty
// bucket emit the two points (x@ymin, y_min) and (x@ymax, y_max) sorted by x
// within the bucket (equal x ⇒ ymin first), buckets concatenated by ascending
// bucket index → {x:[...], y:[...]}. y values are the decoded f32-quantized
// partials. Gaps are applied later at render time (fvApplyLineGaps in
// buildTraceFromTemplate); this envelope, like the server delta, is gapless.
function lineEnvDeltaFromCells(traceSpec, cells) {
  const sorted = cells.slice().sort((a, b) => a.bucketIdx - b.bucketIdx);
  const x = [];
  const y = [];
  for (const c of sorted) {
    const ptMin = { x: c.xAtYmin, y: c.yMin };
    const ptMax = { x: c.xAtYmax, y: c.yMax };
    // Sort the two points by x within the bucket; equal x ⇒ ymin first.
    const first = ptMax.x < ptMin.x ? ptMax : ptMin;
    const second = first === ptMin ? ptMax : ptMin;
    x.push(first.x, second.x);
    y.push(first.y, second.y);
  }
  // Gaps are inserted client-side at RENDER time (buildTraceFromTemplate →
  // fvApplyLineGaps); the live envelope ships gapless, exactly like the server
  // delta now does. The committed /update delta still replaces this envelope.
  return { uid: traceSpec.uid, updates: { x, y } };
}

// Grouped line target (contract D + J): split the combined envelope cells by
// the parent's categorical group dims into per-series envelopes and build each
// child's line delta with the ungrouped builder. Child uids are NEVER minted —
// each group reuses the uid recorded in the existing grouped layer data by
// group_value_key (mirrors fvGroupedResultsFromCells); a group with no
// recorded uid is skipped.
function fvLineEnvGroupedResults(figUid, traceSpec, header, cells) {
  const groupBy = (traceSpec.params && traceSpec.params.group_by) || [];
  const groupIdx = [];
  header.target_dims.forEach((d, i) => {
    if (groupBy.includes(d.name)) groupIdx.push(i);
  });
  // Map each group dim's index to a code→category resolver from the header.
  const byGroup = new Map(); // group_value_key -> {parts, cells}
  for (const cell of cells) {
    const parts = groupIdx.map(i => header.target_dims[i].categories[cell.codes[i]]);
    const gvk = _fvCubeCellLabel(parts);
    let group = byGroup.get(gvk);
    if (!group) {
      group = { parts, cells: [] };
      byGroup.set(gvk, group);
    }
    group.cells.push(cell);
  }

  const recorded = (groupedDataByParent[figUid] || {})[traceSpec.uid] || {};
  const uidForGroup = gvk => {
    for (const layer of ['base', 'bg', 'fg']) {
      for (const cr of (recorded[layer] || [])) {
        if (cr.group_value_key === gvk) return cr.uid;
      }
    }
    return null;
  };

  // Group order mirrors the server's sort over the group columns.
  const ordered = [...byGroup.entries()].sort((a, b) => {
    const pa = a[1].parts, pb = b[1].parts;
    for (let i = 0; i < pa.length; i++) {
      if (pa[i] < pb[i]) return -1;
      if (pa[i] > pb[i]) return 1;
    }
    return 0;
  });
  const groupResults = [];
  for (const [gvk, group] of ordered) {
    const uid = uidForGroup(gvk);
    if (!uid) continue;
    const updates = lineEnvDeltaFromCells(traceSpec, group.cells).updates;
    groupResults.push({ uid, group_value_key: gvk, updates });
  }
  return groupResults;
}

// Grouped targets (grouped hist / grouped bar): split the slice cells by the
// parent's group dims (the dims whose column names appear in params.group_by)
// and build each group's child updates with the ungrouped builder logic over
// that group's cells. Groups absent from the slice are dropped from the list,
// exactly like a server grouped delta. The child uid is NEVER minted
// client-side (it embeds a SHA-1): each group reuses the uid recorded in the
// existing grouped layer data by matching group_value_key; a group with no
// recorded uid is skipped — filtering can only shrink the unfiltered group
// set rendered at init, so a slice can never surface an unseen group.
function fvGroupedResultsFromCells(figUid, traceSpec, header, cells) {
  const groupBy = (traceSpec.params && traceSpec.params.group_by) || [];
  const groupIdx = [];
  const cellIdx = [];
  header.target_dims.forEach((d, i) => {
    (groupBy.includes(d.name) ? groupIdx : cellIdx).push(i);
  });
  const byGroup = new Map(); // group_value_key -> {parts, cells}
  for (const cell of cells) {
    const parts = groupIdx.map(i => cell.dims[i]);
    const gvk = _fvCubeCellLabel(parts);
    let group = byGroup.get(gvk);
    if (!group) {
      group = { parts, cells: [] };
      byGroup.set(gvk, group);
    }
    group.cells.push({ dims: cellIdx.map(i => cell.dims[i]), value: cell.value });
  }

  const recorded = (groupedDataByParent[figUid] || {})[traceSpec.uid] || {};
  const uidForGroup = gvk => {
    for (const layer of ['base', 'bg', 'fg']) {
      for (const cr of (recorded[layer] || [])) {
        if (cr.group_value_key === gvk) return cr.uid;
      }
    }
    return null;
  };

  // Group order mirrors the server's sort over the group columns.
  const ordered = [...byGroup.entries()].sort((a, b) => {
    const pa = a[1].parts, pb = b[1].parts;
    for (let i = 0; i < pa.length; i++) {
      if (pa[i] < pb[i]) return -1;
      if (pa[i] > pb[i]) return 1;
    }
    return 0;
  });
  const groupResults = [];
  for (const [gvk, group] of ordered) {
    const uid = uidForGroup(gvk);
    if (!uid) continue;
    let updates;
    if (traceSpec.trace_type === 'histogram') {
      // Densify to the binned dim, then normalize per child — total = this
      // child's slice total, mirroring Histogram._to_update per child frame.
      const dim = header.target_dims[0];
      const counts = new Float64Array(dim.bins);
      for (const c of group.cells) counts[c.dims[0]] += c.value;
      updates = histDeltaFromCounts(traceSpec, header, counts).updates;
    } else {
      // Grouped child: no per-label marker — the group colour is applied at
      // render (mirrors the server's marker-less grouped delta).
      updates = barDeltaFromCells(traceSpec, group.cells, false).updates;
    }
    groupResults.push({ uid, group_value_key: gvk, updates });
  }
  return groupResults;
}

// ---------------------------------------------------------------------------
// Correlation target (contract I; mirror CubeResult._combine_corr + _to_update)
// ---------------------------------------------------------------------------

// Finalize Pearson r from the combined centered partials (contract I):
//   m_x = sx/n, cov = sxy/n - m_x*m_y, var_x = sxx/n - m_x^2 (likewise y),
//   r = cov / sqrt(var_x*var_y). Non-finite (n < 2, zero variance) => 0.0 —
//   mirroring _corr_expr._pack's None -> 0.0.
function fvCorrFinalize(n, sx, sy, sxy, sxx, syy) {
  if (n < 2) return 0.0;
  const mx = sx / n;
  const my = sy / n;
  const cov = sxy / n - mx * my;
  const varX = sxx / n - mx * mx;
  const varY = syy / n - my * my;
  if (varX <= 0.0 || varY <= 0.0) return 0.0;
  const r = cov / Math.sqrt(varX * varY);
  return Number.isFinite(r) ? r : 0.0;
}

// Combine the decoded corr partials over the free-bin ranges (contract I):
// one GLOBAL accumulator per pair (corr has no target dims), Sum each partial
// across the sliced rows, finalize per pair. Returns a Map "i_j" -> r (raw,
// pre-absolute). header.measure.pairs lists the [i, j] pairs (i < j, column
// order); the per-pair partial columns are __corr__{i}_{j}__{stat}.
function fvCorrCombine(entry, binRanges) {
  const header = entry.header;
  const pairs = (header.measure && header.measure.pairs) || [];
  const freeBin = entry.cols.free_bin;
  const maxBin = _fvCubeMaxBin(header);
  // Pre-resolve the six partial TypedArrays per pair.
  const acc = pairs.map(([i, j]) => ({
    i, j,
    n: entry.cols['__corr__' + i + '_' + j + '__n'],
    sx: entry.cols['__corr__' + i + '_' + j + '__sx'],
    sy: entry.cols['__corr__' + i + '_' + j + '__sy'],
    sxy: entry.cols['__corr__' + i + '_' + j + '__sxy'],
    sxx: entry.cols['__corr__' + i + '_' + j + '__sxx'],
    syy: entry.cols['__corr__' + i + '_' + j + '__syy'],
    sums: { n: 0, sx: 0, sy: 0, sxy: 0, sxx: 0, syy: 0 },
  }));
  for (const range of binRanges) {
    const lo = Math.max(0, range[0]);
    const hi = Math.min(maxBin, range[1]);
    if (hi < lo) continue;
    for (let r = entry.binStart[lo]; r < entry.binStart[hi + 1]; r++) {
      for (const p of acc) {
        p.sums.n += p.n[r];
        p.sums.sx += p.sx[r];
        p.sums.sy += p.sy[r];
        p.sums.sxy += p.sxy[r];
        p.sums.sxx += p.sxx[r];
        p.sums.syy += p.syy[r];
      }
    }
  }
  const out = new Map();
  for (const p of acc) {
    const s = p.sums;
    out.set(p.i + '_' + p.j,
      fvCorrFinalize(s.n, s.sx, s.sy, s.sxy, s.sxx, s.syy));
  }
  return out;
}

// Build a corr-heatmap delta from the decoded cube (contract I): combine the
// per-pair partials over the free-bin ranges, finalize r, then assemble the
// n x n z matrix EXACTLY like CorrHeatmap._to_update — diag 1.0, symmetric
// fill from header.measure.pairs/columns, absolute => |r| per cell, triangular
// nulls the upper half (z[j][i]=null for i in [j..n)), rows + y labels
// reversed. absolute/triangular come from the TRACE SPEC's params (display
// params, NOT the cube header). Shaped exactly like a server corr TraceDelta.
function corrDeltaFromEntry(traceSpec, entry, binRanges) {
  const header = entry.header;
  const cols = (header.measure && header.measure.columns) || [];
  const pairs = (header.measure && header.measure.pairs) || [];
  const n = cols.length;
  const params = traceSpec.params || {};
  const absolute = !!params.absolute;
  const triangular = !!params.triangular;

  const rByPair = fvCorrCombine(entry, binRanges);
  const matrix = [];
  for (let j = 0; j < n; j++) {
    const row = new Array(n).fill(0.0);
    row[j] = 1.0;
    matrix.push(row);
  }
  for (const [i, j] of pairs) {
    let r = rByPair.get(i + '_' + j) || 0.0;
    if (absolute) r = Math.abs(r);
    matrix[i][j] = r;
    matrix[j][i] = r;
  }
  if (triangular) {
    for (let j = 0; j < n; j++) {
      for (let i = j; i < n; i++) matrix[j][i] = null;
    }
  }
  const y = cols.slice().reverse();
  const z = matrix.slice().reverse();
  return { uid: traceSpec.uid, updates: { x: cols.slice(), y, z } };
}

// Build a hist2d-heatmap delta from the decoded cube (contract K): slice the
// two binned target dims (x, y) over the free-bin ranges, densify into a flat
// measure array indexed j*nb_x + i (j = y bin, i = x bin), apply histnorm to
// the flat z BEFORE reshaping, then reshape to z[j] = zflat[j*nb_x .. ].
// Mirrors Histogram2D._to_update EXACTLY: count empty cell -> null, reduce
// missing cell -> null; centers at lo + (i+0.5)*step over each header dim
// domain; histnorm per apply_histnorm with bin_area = x_step * y_step.
// hist2d is filtered_only (no background layer); shaped like a server delta.
function hist2dDeltaFromEntry(traceSpec, entry, binRanges) {
  const header = entry.header;
  const dimX = header.target_dims[0];
  const dimY = header.target_dims[1];
  const nbX = dimX.bins;
  const nbY = dimY.bins;
  const xLo = dimX.domain[0], xHi = dimX.domain[1];
  const yLo = dimY.domain[0], yHi = dimY.domain[1];

  // fvCubeSliceCells returns {dims: [binX, binY], value}; missing cells are
  // simply absent (the cube is sparse), so initialise the flat z to null.
  const zflat = new Array(nbX * nbY).fill(null);
  for (const cell of fvCubeSliceCells(entry, binRanges)) {
    const i = cell.dims[0]; // x bin
    const j = cell.dims[1]; // y bin
    zflat[j * nbX + i] = cell.value;
  }

  const histnorm = (traceSpec.params && traceSpec.params.histnorm) || null;
  let values = zflat;
  if (histnorm) {
    const xStep = (xHi - xLo) / nbX;
    const yStep = (yHi - yLo) / nbY;
    const binArea = xStep * yStep;
    let total = 0;
    for (const v of zflat) if (v !== null) total += v;
    // Unguarded division to stay bit-equal to apply_histnorm (Polars): a zero
    // total yields NaN/Inf there too, so guarding to 0 here would diverge from
    // a committed server delta. null stays null (matches Polars null arithmetic).
    if (histnorm === 'percent') {
      values = zflat.map(v => (v === null ? null : v / total * 100));
    } else if (histnorm === 'probability') {
      values = zflat.map(v => (v === null ? null : v / total));
    } else if (histnorm === 'density') {
      values = zflat.map(v => (v === null ? null : v / binArea));
    } else { // 'probability density'
      values = zflat.map(v => (v === null ? null : v / (total * binArea)));
    }
  }

  const xCenters = _fvCenters(xLo, xHi, nbX);
  const yCenters = _fvCenters(yLo, yHi, nbY);
  const z = [];
  for (let j = 0; j < nbY; j++) z.push(values.slice(j * nbX, (j + 1) * nbX));
  return { uid: traceSpec.uid, updates: { x: xCenters, y: yCenters, z } };
}

// Cell centers: lo + (i + 0.5) * (hi - lo) / n — mirrors hist2d.py _centers.
function _fvCenters(lo, hi, n) {
  const step = (hi - lo) / n;
  const out = new Array(n);
  for (let i = 0; i < n; i++) out[i] = lo + (i + 0.5) * step;
  return out;
}

// ---------------------------------------------------------------------------
// Treemap target (contract K; mirror TreeMap._to_grouped_update)
// ---------------------------------------------------------------------------

// Byte-identical to Python urllib.parse.quote(s, safe=""): encodeURIComponent
// leaves `! * ' ( )` unescaped, but Python's quote(safe="") escapes them, so
// re-escape those five. Both already leave `A-Za-z0-9 -_.~` unescaped and
// encode space as %20. (Astral/code-point > U+FFFF edge: JS surrogate halves
// percent-encode to the same UTF-8 bytes as Python, so they agree there too.)
function fvUrlQuote(s) {
  return encodeURIComponent(String(s)).replace(
    /[!*'()]/g,
    c => '%' + c.charCodeAt(0).toString(16).toUpperCase()
  );
}

// Build a treemap delta from the decoded cube (contract K). fvCubeSliceCells
// returns the FINALIZED leaf cells {dims:[part0, part1, ...], value}; this then
// reproduces TreeMap._to_grouped_update EXACTLY: a "root" node summing all leaf
// values, then for each path level a node per distinct prefix whose value is the
// SUM of the finalized leaf values under it (parents sum the leaf means too —
// never re-finalized at parent levels). Nodes within a level are sorted by their
// full prefix tuple (lexicographic </> — matches Polars Utf8 sort for ASCII;
// astral / code-point > U+FFFF byte-order is a documented edge, like contract
// D). ids/parents are url-quoted per part; the label is the RAW part. color_map
// → marker.colors (PLURAL, like pie) over the FULL labels array (root="" too).
function treemapDeltaFromEntry(traceSpec, entry, binRanges) {
  const cells = fvCubeSliceCells(entry, binRanges);
  const path = (traceSpec.params && traceSpec.params.path) || [];

  const ids = ['root'];
  const labels = [''];
  const parents = [''];
  let rootValue = 0;
  for (const c of cells) rootValue += c.value;
  const values = [rootValue];

  for (let level = 0; level < path.length; level++) {
    // Aggregate leaf cells by their first (level+1) parts, summing value.
    const agg = new Map(); // prefix-key -> {parts, value}
    for (const cell of cells) {
      const parts = cell.dims.slice(0, level + 1);
      const key = _fvCubeCellLabel(parts);
      let node = agg.get(key);
      if (!node) {
        node = { parts, value: 0 };
        agg.set(key, node);
      }
      node.value += cell.value;
    }
    // Sort nodes by the full prefix tuple (lexicographic </>).
    const nodes = Array.from(agg.values()).sort((a, b) => {
      for (let i = 0; i < a.parts.length; i++) {
        if (a.parts[i] < b.parts[i]) return -1;
        if (a.parts[i] > b.parts[i]) return 1;
      }
      return 0;
    });
    for (const node of nodes) {
      const encoded = node.parts.map(fvUrlQuote);
      ids.push('root/' + encoded.join('/'));
      parents.push(level === 0 ? 'root' : 'root/' + encoded.slice(0, -1).join('/'));
      labels.push(node.parts[level]);
      values.push(node.value);
    }
  }

  const updates = { labels, parents, ids, values };
  const colorMap = traceSpec.display && traceSpec.display.color_map;
  if (colorMap) {
    updates.marker = { colors: labels.map(l => colorMap[String(l)] ?? null) };
  }
  return { uid: traceSpec.uid, updates };
}

// ---------------------------------------------------------------------------
// cube_request transport
// ---------------------------------------------------------------------------

// One range-less POST per gesture-with-store-miss. Goes straight to
// /dashboard/update with the cube extension fields; the response is a binary
// cube bundle (application/octet-stream), not JSON deltas. Returns the decoded
// { trace_cubes, blobs } bundle, or null on transport/decode failure.
async function fvCubeRequest(event, activeSource) {
  try {
    const resp = await fetch(SERVER_URL + '/dashboard/update', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({
        spec: DASHBOARD_SPEC,
        event,
        request_cube: true,
        active_source: activeSource,
      }),
    });
    if (!resp.ok) return null;
    return decodeCubeBundle(await resp.arrayBuffer());
  } catch (e) {
    console.warn('flexviz cube request failed', e);
    return null;
  }
}

window.fvCubeKey = fvCubeKey;
window.fvCubePassiveKey = fvCubePassiveKey;
window.fvCubeHeaderMatchesKey = fvCubeHeaderMatchesKey;
window.fvTemporalToPhysical = fvTemporalToPhysical;
window.fvPhysicalToTemporal = fvPhysicalToTemporal;
window.fvPhysicalToEpochMs = fvPhysicalToEpochMs;
window.fvCubeDayGrid = fvCubeDayGrid;
window.fvCubeStoreReset = fvCubeStoreReset;
window.fvDecodeFVCube = decodeFVCube;
window.fvDecodeCubeBundle = decodeCubeBundle;
window.fvCubeSnap2d = fvCubeSnap2d;
window.fvCubeRectRanges = fvCubeRectRanges;
window.fvLineEnvCells = fvLineEnvCells;
window.fvApplyLineGaps = fvApplyLineGaps;
window.fvLineEnvDeltaFromCells = lineEnvDeltaFromCells;
window.fvCorrCombine = fvCorrCombine;
window.fvCorrFinalize = fvCorrFinalize;
window.fvCorrDeltaFromEntry = corrDeltaFromEntry;
window.fvHist2dDeltaFromEntry = hist2dDeltaFromEntry;
window.fvUrlQuote = fvUrlQuote;
window.fvTreemapDeltaFromEntry = treemapDeltaFromEntry;

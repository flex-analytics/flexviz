# Cube cross-filter support — current state

**Source of truth:** `Architecture.md` (§ "Cube Pre-Aggregation & Live Brushing" and
"Cube eligibility and transport").
This file is a navigational summary — if it disagrees with `Architecture.md`, that wins.

The cube path lets a **brush/selection on a source figure live-update target figures
client-side**, with zero server round-trips after the one `cube_request` that fetches the cube
blob. A cube works whenever the measure is **decomposable** over a partition of the source axis:
`agg(A ∪ B) = combine(agg(A), agg(B))`. The free axis is just the partition key — numeric bins,
2-D composite bins, or category codes all work the same.

---

## 1. Sources (what can emit a brush → a "free axis")

A source trace implements `get_cube_source_spec()` and produces a **free axis** of one kind:

| Source trace | Free-axis kind | Geometry |
|---|---|---|
| `histogram` | **range** (continuous / temporal) | 1-D, P=2048 over the viewport (or full-data) domain |
| `box` | **range** | 1-D over the `data_col` (same gates as hist) |
| `line` | **range** | 1-D over the **x** column only (line selection is x-only); P=2048 |
| `histogram2d` | **box2d** | two range axes (x, y) at P₂D=128 each, packed into one composite `free_bin` |
| `bar` | **categorical** | over the ordered label column(s) |
| `pie` | **categorical** | over the label column(s) |
| `treemap` | **categorical** | over the full `path` (prefix selects a subtree) |

**Not sources:** `corr_heatmap` (emits no selection), `geo_histogram2d`, `geo_line`.

Temporal range sources support `us` / `ms` / `day` physical units; `Datetime("ns")` / `Time`
gate to no cube.

---

## 2. Targets (what can be live-updated from a cube)

A target trace implements `get_cube_target_spec()` → target dims + a decomposable measure:

| Target trace | Target dims | Measure(s) |
|---|---|---|
| `histogram` (ungrouped) | (binned data col) | `count` |
| `histogram` (grouped) | (binned data col, *group cols) | `count` |
| `bar` | (*label cols, *group cols) | `count` / `sum` / `mean` / `min` / `max` over `values` |
| `pie` | (*label cols) | same as bar |
| `line` | (binned x @ `n_points/2` buckets, *group cols) | `line_env` over `y` — **minmax-only** |
| `corr_heatmap` | () — cells are the explicit column pairs | `corr` (**pearson only**, ≥2 numeric cols) |
| `histogram2d` | (binned x, binned y) | `count` or `histfunc` over `z` — **full-data only** (declines when zoomed) |
| `treemap` | (*path cols, leaf level) | `count` / `sum` / `mean` / `min` / `max` over `values` |

**Not targets:**
- `box` — quantiles (median/quartiles) are **not decomposable**.
- Any `median` / `n_unique` measure — not decomposable.
- `geo_histogram2d`, `geo_line`.

`bar` ≡ `pie` with the same labels + measure **share one blob** (content-key dedup).

---

## 3. Which source × target combinations are buildable

The gate is `cube_target_buildable(free, measure)` in `flexviz/cube.py`. Read it as: **which
source free-axis kinds can feed each target measure.**

| Target measure | range (hist/box/line) | categorical (bar/pie/treemap) | box2d (hist2d) |
|---|:---:|:---:|:---:|
| `count` | ✅ | ✅ | ✅ |
| `sum` / `mean` / `min` / `max` | ✅ | ✅ | ✅ |
| `line_env` (line target) | ✅ | ✅ | ❌ #47 |
| `corr` (corr target) | ✅ | ✅ | ❌ #47 |
| `median` / `n_unique` | ❌ (not decomposable) | ❌ | ❌ |
| box quantiles (box target) | ❌ (not decomposable) | ❌ | ❌ |

Notes on the two `❌` columns:
- **`box2d × {line_env, corr}` (#47):** a feasibility-free but real **cell-count / wire-size
  wall** — a box2d `line_env` cube is `n_x_buckets × (P₂D+1)²` ≈ `500 × 129²` ≈ 8.3M cells per
  line. Gated off; those targets fall back to the per-commit recompute. Tracked in #47.
- **`median` / `n_unique` / box quantiles:** mathematically not decomposable over a partition —
  never cube targets, by any source.

When a target is **not** buildable for the active source, the engine **silently skips it** (it
falls back to the per-commit `POST /update`) instead of failing the whole live-brush request.
Buildable targets in the same dashboard are still served.

---

## 4. How categorical `line_env` / `corr` work (the 2026-06-22 addition)

Both measures are decomposable over a categorical partition, and the binary codec + the entire JS
client were already free-kind-agnostic, so this was a **server-only** change in `flexviz/cube.py`:

- **`corr`** — `group_by(*__free__cols).agg(*corr_partials)`; per-pair mean-centered sums
  (`n, Σx̃, Σỹ, Σx̃ỹ, Σx̃², Σỹ²`) combine by summing across the OR'd category set.
- **`line_env`** — partition the build by `free ∪ group` columns and run the
  `fixed_line_envelope2d` kernel with a **degenerate 1-bin free axis** (approach A). Per cell ships
  `(y_min, x@ymin, y_max, x@ymax)`; combine = min/max per x-bucket.
- The categorical free key is the tuple of typed `__free__{col}` values; the encoder
  dictionary-encodes them into the u32 `free_bin` code column and preserves numeric labels as
  JSON numbers so float labels such as `1.0` match browser values such as `1`.
- **Numeric bar sources read the typed data label, not the axis coordinate.** A numeric bar axis
  is *linear* (only string labels get a *category* axis), so grouped bars draw at `category ±
  offset`. The Plotly adapter takes the brushed label from the trace's data array
  (`pt.data[catKey][pt.pointNumber]`), giving the true typed value (`1`) rather than the offset
  coordinate (`0.8`/`1.2`). Without this the committed `is_in` carries offsets that match no row
  and every cube/target empties — the original `hour_of_day` grouped-bar regression.

**Multi-select = OR** over selected categories.

**The line-envelope caveat (unchanged):** a line target's commit **always POSTs** so the exact
legacy row-bucket delta replaces the approximate live envelope — keeping commit ≡ share/restore
bit-exact. This holds for categorical sources too.

---

## 5. Known limitations / open issues

| Issue | Status |
|---|---|
| #46 | categorical `line_env`/`corr` — **done** (this work) |
| #47 | `box2d` (hist2d source) × {line_env, corr} — gated off (cell-count wall) |
| #48 | categorical `line_env` build (approach A) is **slower** than pure-Polars B — flip the categorical branch to B; parity already proven |

**Cardinality safety net:** categorical `line_env` cells = `n_buckets × n_categories`. bar/pie are
tiny; treemap leaves are the only high-cardinality risk. No server-side cap — the existing
**client byte-budget demotion** (an over-budget cube is refused → the target self-heals to the
commit POST) is the guard.

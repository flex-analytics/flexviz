# Line downsampling

A line trace never sends raw rows to the browser. On every viewport change it
selects roughly `n_points` representative points from the rows inside the
visible x-range, using a Rust kernel that runs as a parallel Polars expression
plugin. Zooming in progressively reveals detail; at full zoom-out you still
see the shape of the whole series, including spikes.

```python
fig.add_line(
    x="timestamp", y="value",
    n_points=1000,          # target points per viewport (2 to 25000)
    downsample="minmax",    # "minmax" | "lttb" | "fpcs" | "nth"
    assume_sorted_x=True,   # skip the x check (see below)
)
```

## Algorithms

- **`"minmax"`** (default): splits the x range into `n_points // 2` buckets
  and keeps the y-minimum and y-maximum of each. The buckets are equal in x
  width, grouped or not. Extremes and spikes always survive, which makes it the
  right default for monitoring-style data.
- **`"lttb"`**: MinMaxLTTB. Runs the min-max pass with four times the budget,
  then keeps the point with the largest triangle area in each of `n_points`
  buckets. The line looks smoother than a min-max envelope on noisy data. 
  It is not a cross-filter cube target.  
  MinMaxLTTB paper: https://arxiv.org/pdf/2305.00332
- **`"fpcs"`**: Feature-Preserving Compensated Sampling. Runs the same min-max
  pass, then carries deferred extrema forward across buckets to reduce visual
  artifacts on oscillating signals. It buckets by x width.  
  FPCS paper: https://ieeevis.b-cdn.net/vis_2024/pdfs/v-full-1363.pdf 
- **`"nth"`**: uniform stride, keeping every n-th row. Cheapest, but a spike
  between kept points disappears. Use it when the data is smooth or when you
  want deterministic spacing.

## Point counts

`n_points` is a target, not a guarantee. `minmax`, `lttb`, and `fpcs` bucket the
visible x range, and a sparse viewport does not fill every bucket. An empty
bucket gives no point. A bucket whose minimum and maximum are the same row gives
one point, not two. FlexViz drops a row with a null or NaN y before this.

Only the ceiling differs per strategy:

| Strategy | Points per viewport |
| --- | --- |
| `minmax` | At most `n_points`: two per bucket, duplicates removed. |
| `lttb` | Exactly `n_points` when the prefetch holds more. |
| `fpcs` | Up to about `2 * n_points`. |
| `nth` | `n_points`, or every row when the viewport holds fewer. Gaps do not lower it. |

## Grouped lines

A grouped line puts every series on one grid: the same buckets an ungrouped line
over the same x column and `n_points` builds. A group that covers a tenth of the
x domain therefore gets about a tenth of the points, not a full budget of its
own. That split is provisional, and issue #16 tracks it. `"nth"` is the
exception: it keeps a stride per group, so every series gets `n_points` points.

## The x contract

An ungrouped `"minmax"`, `"lttb"` or `"fpcs"` line buckets by equal x width and
binary-searches the bucket edges. Its x column must be a 64-bit-or-smaller numeric, or a temporal,
and must not be infinite. Wider numerics (`Int128`, `Decimal`) have no edge type
in the kernel and are rejected. On a resident frame x must also be sorted
ascending and free of nulls and NaN. The engine verifies this before it
aggregates and raises `ValueError` when the column breaks the contract.
`Figure.add_line` itself checks nothing.

A `UInt64` x whose values go above `i64::MAX` fails on a resident frame,
because the kernel reads its bounds as signed 64-bit integers. Cast the column
to `Int64` or `Float64` first.

A file source runs an order-independent plan that drops null and NaN x, so only
its dtype is gated.

The y column is gated on its dtype as well. An x-width line rejects `Decimal`,
`Int128`, `Categorical` and `Enum`: the kernel cannot compare them, while the
file-source plan can, so without the gate the same line would work on one source
kind and fail on the other. An `"lttb"` line asks for more, a numeric, temporal
or Boolean y, because the triangle rule does arithmetic on it.

- The order, null, and NaN check costs one pass over x, and only an ungrouped
  x-width line on a resident frame runs it. A resident frame is a snapshot, so
  the check runs once per source and column.
- `add_line(..., assume_sorted_x=True)` skips the check. Only pass it when you
  can guarantee the column. A column that breaks the contract then produces
  wrong output.
- Sorted x also makes a viewport zoom a zero-copy binary-searched slice of the
  frame instead of a row-by-row range filter, which matters at 100M+ rows.
- A grouped line is checked on its dtype only. Its buckets are arithmetic on x,
  which needs no order. An `"nth"` line is not checked at all: a stride needs no
  grid. A grouped line always masks the viewport, and an `"nth"` line masks it
  when x is not declared sorted, which is always correct but slower on very
  large frames.
- `n_points` must be between 2 and 25000. The client posts the trace spec on
  every update, so the bound is enforced wherever a line is built.

### Equal-row-count buckets

`add_line` requires `x`, and x width is the only bucket rule FlexViz offers. An
x-width line spends its budget on x width, so a dense burst in a narrow x span
gets few points. To spend the budget on row count instead, plot against a row
index. There is no separate entry point:

```python
df = df.with_row_index("i")
fig.add_line(x="i", y="value")   # a uniform x makes every bucket hold equal rows
```

### Nulls, NaN, and infinities

| Column | What happens |
| --- | --- |
| x | An infinite value raises `ValueError`. A null or NaN raises on a resident frame. A file source and a grouped line drop the row. |
| y | `nth` is a stride. It keeps every nth row, null or NaN y included, so the renderer draws a gap at the true position. `minmax`, `lttb`, and `fpcs` drop a row with a null or NaN y. |

An infinite y is a value: the x-width strategies keep it as an extremum, and
only null and NaN are dropped.

An infinite bound has no finite bucket width, so the grid cannot be built. Drop
the rows first:

```python
df = df.drop_nulls("x")                     # nulls
df = df.filter(pl.col("x").is_finite())     # NaN and infinities
```

## Gap handling

Real-world series have holes (sensor dropouts, nights, maintenance windows).
With `add_gaps=True` (default), FlexViz inserts breaks where consecutive x
values are unusually far apart, so the renderer draws a broken line instead
of bridging the gap with a misleading straight segment. Pass
`add_gaps=False` to always connect.

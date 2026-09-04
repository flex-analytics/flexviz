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
  and keeps the y-minimum and y-maximum of each. An ungrouped line makes the
  buckets equal in x width. A grouped one makes them equal in row count.
  Extremes and spikes always survive, which makes it the right default for
  monitoring-style data.
- **`"lttb"`**: MinMaxLTTB. Runs the min-max pass with four times the budget,
  then keeps the point with the largest triangle area in each of `n_points`
  buckets. The output holds exactly `n_points` points when the prefetch holds
  more, and fewer when x gaps leave buckets empty. The line looks smoother than
  a min-max envelope on noisy data. Ungrouped lines only, and it is not a
  cross-filter cube target.
- **`"fpcs"`**: Feature-Preserving Compensated Sampling. Runs the same min-max
  pass, then carries deferred extrema forward across buckets to reduce visual
  artifacts on oscillating signals. An ungrouped line buckets by x width, a
  grouped one by row count. `n_points` is a target, not a cap: output can reach
  roughly `2 * n_points`, and holds fewer points when the x gaps leave buckets
  empty.
- **`"nth"`**: uniform stride, keeping every n-th row. Cheapest, but a spike
  between kept points disappears. Use it when the data is smooth or when you
  want deterministic spacing.

## The x contract

An ungrouped `"minmax"`, `"lttb"` or `"fpcs"` line buckets by equal x width and
binary-searches the bucket edges. Its x column must be a 64-bit-or-smaller numeric, or a temporal,
and must not be infinite. Wider numerics (`Int128`, `Decimal`) have no edge type
in the kernel and are rejected. On a resident frame x must also be sorted
ascending and free of nulls and NaN. The engine verifies this on the first
request and raises `ValueError` when the column breaks the contract.
`Figure.add_line` itself checks nothing.

A file source runs an order-independent plan that drops null x and tolerates
NaN, so only its dtype is gated.

- The order, null, and NaN check costs one pass over x. A `cache=True` source
  pays it once per source and column. A `cache=False` source may have changed
  since the last request, so it pays it on every unzoomed request.
- `add_line(..., assume_sorted_x=True)` skips the check. Only pass it when you
  can guarantee the column. A column that breaks the contract then produces
  wrong output.
- Sorted x also makes a viewport zoom a zero-copy binary-searched slice of the
  frame instead of a row-by-row range filter, which matters at 100M+ rows.
- Grouped lines and `"nth"` lines are not checked. Their buckets hold equal row
  counts, and they mask the viewport when x is not declared sorted, which is
  always correct but slower on very large frames.
- `n_points` must be between 2 and 25000. The client posts the trace spec on
  every update, so the bound is enforced wherever a line is built.

### Equal-row-count buckets

An ungrouped x-width line spends its budget on x width, so a dense burst in a
narrow x span gets few points. To spend the budget on row count instead, plot against a
row index:

```python
df = df.with_row_index("i")
fig.add_line(x="i", y="value")   # a uniform x makes every bucket hold equal rows
```

### Nulls, NaN, and infinities

| Column | What happens |
| --- | --- |
| x | An infinite value raises `ValueError`. A null or NaN raises on a resident frame. A file source drops it. |
| y | Skipped, on every downsampling path. |

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

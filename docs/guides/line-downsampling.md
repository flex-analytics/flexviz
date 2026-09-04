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
    downsample="minmax",    # "minmax" | "fpcs" | "nth"
    assume_sorted_x=True,   # skip the x check (see below)
)
```

## Algorithms

- **`"minmax"`** (default): splits the viewport into `n_points // 2` buckets
  and keeps the y-minimum and y-maximum of each. Extremes and spikes always
  survive, which makes it the right default for monitoring-style data.
- **`"fpcs"`**: Feature-Preserving Compensated Sampling. Runs the same
  min-max pass, then carries deferred extrema forward across windows to
  reduce visual artifacts on oscillating signals. `n_points` is a target, not
  a cap; output can reach roughly `2 * n_points`.
- **`"nth"`**: uniform stride, keeping every n-th row. Cheapest, but a spike
  between kept points disappears. Use it when the data is smooth or when you
  want deterministic spacing.

## The x contract

An ungrouped `"minmax"` line buckets by equal x width and binary-searches the
bucket edges, so its x column must be numeric or temporal, hold no nulls and no
NaN, and be sorted ascending. The engine verifies this once per registered
source and column, with one pass over x, and raises `ValueError` when the
column breaks the contract. A file source runs an order-independent plan, so it
is not order-checked: its dtype is read off the schema, and its null and NaN
counts ride the domain probe of the first unzoomed request.

- `add_line(..., assume_sorted_x=True)` skips the check. Only pass it when you
  can guarantee the column; a column that breaks the contract then produces
  wrong output.
- Sorted x also makes a viewport zoom a zero-copy binary-searched slice of the
  frame instead of a row-by-row range filter, which matters at 100M+ rows.
- Grouped, `"nth"` and `"fpcs"` lines are not checked. Their buckets hold equal
  row counts, and they mask the viewport when x is not declared sorted, which
  is always correct but slower on very large frames.
- `n_points` must be between 2 and 25000. The client posts the trace spec on
  every update, so the bound is enforced wherever a line is built.

### Equal-row-count buckets

An ungrouped line spends its budget on x width, so a dense burst in a narrow x
span gets few points. To spend the budget on row count instead, plot against a
row index:

```python
df = df.with_row_index("i")
fig.add_line(x="i", y="value")   # a uniform x makes every bucket hold equal rows
```

### Nulls and NaN

| Column | What happens |
| --- | --- |
| x | Raises `ValueError` with the number of null or NaN values. |
| y | Skipped, on every downsampling path. |

A null or NaN has no position on the axis, and the bucket edges are found by
binary search, so a dirty x has no bucket to fall in. Drop the rows first:

```python
df = df.drop_nulls("ts")                    # nulls
df = df.filter(pl.col("x").is_not_nan())    # NaN
```

## Gap handling

Real-world series have holes (sensor dropouts, nights, maintenance windows).
With `add_gaps=True` (default), FlexViz inserts breaks where consecutive x
values are unusually far apart, so the renderer draws a broken line instead
of bridging the gap with a misleading straight segment. Pass
`add_gaps=False` to always connect.

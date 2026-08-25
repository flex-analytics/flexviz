# Line downsampling

A line trace never sends raw rows to the browser. On every viewport change it
selects roughly `n_points` representative points from the rows inside the
visible x-range, using a Rust kernel that runs as a parallel Polars expression
plugin. Zooming in progressively reveals detail; at full zoom-out you still
see the shape of the whole series, including spikes.

```python
fig.add_line(
    x="timestamp", y="value",
    n_points=1000,          # target points per viewport
    downsample="minmax",    # "minmax" | "fpcs" | "nth"
    assume_sorted_x=True,   # skip the sortedness check (see below)
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

## Sorted x and performance

The downsamplers need x sorted ascending. When FlexViz knows the x column is
sorted, a viewport zoom becomes a zero-copy binary-searched slice of the
frame instead of a row-by-row range filter, which matters at 100M+ rows.

- `add_line(..., assume_sorted_x=True)` declares the column sorted without
  verification. Only pass it when you can guarantee the order; a wrongly
  declared sort produces wrong output.
- Without it, FlexViz falls back to a range-filter mask, which is always
  correct but slower on very large frames.

## Gap handling

Real-world series have holes (sensor dropouts, nights, maintenance windows).
With `add_gaps=True` (default), FlexViz inserts breaks where consecutive x
values are unusually far apart, so the renderer draws a broken line instead
of bridging the gap with a misleading straight segment. Pass
`add_gaps=False` to always connect.

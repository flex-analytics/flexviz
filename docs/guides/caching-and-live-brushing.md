# Caching and live brushing

Both features are opt-in through one flag, `cache=True`, because both rely on
the same contract: **the source data does not change for the lifetime of the
process**. There is no data-change invalidation yet, so only opt in for
static data (re-registering a source under the same name does clear its
caches).

```python
dash = Dashboard(lf, cache=True)
dash.show()  # live_brush="auto" engages automatically
```

`cache` can be set on the `Figure` or `Dashboard` constructor, overridden per
`show()` call, or passed to `register_source()` when
[embedding](embedding.md).

## Init-load caching

With `cache=True`, the expensive first aggregation of each trace (the
unfiltered, unzoomed view) is memoized on the server and mirrored in the
client, so opening the dashboard again, resetting it, or clearing a selection
at full zoom-out is served without recomputing. Zoomed or filtered requests
always recompute; the cache is never authoritative, a miss simply recomputes
the identical result.

## Live brushing

Normally a brush is resolved when you release the mouse: the selection goes
to the server, aggregates come back. With live brushing, FlexViz
pre-aggregates a small data cube per source figure and answers **every
intermediate frame of the drag client-side**, so linked figures update
continuously while you brush, with zero server round-trips during the drag.

`show(live_brush=...)` controls it:

- `"auto"` (default): live-brush wherever a cube is available, fall back to
  mouseup-only elsewhere.
- `"off"`: mouseup-only selection everywhere.

Live brushing requires `cache=True`, since cubes are only built for sources
that assert static data. With caching off, `live_brush="auto"` is forced to
`"off"` (with a warning if you asked for it explicitly).

Cubes exist for brushes originating from line, histogram, box, 2-D histogram,
bar, pie, and treemap figures, and accelerate target aggregations that can be
combined from per-bin partials (counts, sums, means, min/max). Anything not
cube-eligible silently stays on the normal server path; the result is always
the same, only the latency differs.

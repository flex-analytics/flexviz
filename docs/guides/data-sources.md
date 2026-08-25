# Data sources

`Figure(data)`, `Dashboard(data)`, and `register_source(name, data)` accept:

- a Polars `LazyFrame` (preferred) or `DataFrame`
- a pandas `DataFrame`
- a PyArrow `Table`

Everything is normalized to a lazy Polars frame internally. Non-Polars inputs
are converted once at construction time; from then on all work is lazy.

## Stay lazy

FlexViz never collects your frame up front. Each interaction turns into one
batched Polars query that filters, aggregates, and collects only what the
charts need. That means the best input is a `LazyFrame` that scans its
storage directly:

```python
lf = pl.scan_parquet("readings.parquet")   # nothing is read yet
Dashboard(lf).show()
```

- **In-memory frames** aggregate zero-copy: backend memory stays flat no
  matter the row count.
- **Parquet-backed frames** (`scan_parquet`) let Polars push filters into the
  scan and stream batches, so larger-than-RAM datasets work.
- Any transformation you apply before handing the frame over
  (`lf.filter(...).with_columns(...)`) stays lazy and is fused into every
  FlexViz query.

## Sorted time axes

Line traces are fastest when their x column is known to be sorted: viewport
zooms then become binary-searched slices instead of scans. If your data comes
out of storage already ordered by time, declare it:

```python
fig.add_line(x="timestamp", y="value", assume_sorted_x=True)
```

See [Line downsampling](line-downsampling.md) for the details and the
correctness caveat.

## Named sources

A `Figure` or `Dashboard` registers its frame under a generated name
automatically at `show()` time. When you run your own server or share one
dataset across multiple entry points, register it yourself with
`register_source(name, data)` and reference the name; see
[Embedding](embedding.md).

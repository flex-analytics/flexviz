# Data sources

`Figure(data)`, `Dashboard(data)`, and `register_source(name, data)` accept:

- a Polars `LazyFrame` (preferred: a scan stays on disk) or `DataFrame`
- a pandas `DataFrame`
- a PyArrow `Table`

Everything is normalized to a lazy Polars frame internally. Non-Polars inputs
are converted once at construction time; from then on all work is lazy.

## Stay lazy

FlexViz does not collect your frame when you build a figure. Each interaction
becomes one batched Polars query that filters, aggregates, and collects only
what the charts need.

Two source kinds behave differently. FlexViz reads the query plan once and
takes the scan path when the plan roots at a scan node:

```python
lf = pl.scan_parquet("readings.parquet")   # nothing is read yet
Dashboard(lf).show()
```

- **Resident frames** are a Polars `DataFrame`, a pandas frame, a PyArrow table,
or a `LazyFrame` built over one of them. The rows are already in memory, so
FlexViz aggregates them in place and never copies the frame.
- **File scans** are `pl.scan_parquet(...)`, `pl.scan_csv(...)`,
`pl.scan_ipc(...)`, and `pl.scan_ndjson(...)`. Polars pushes the column
selection and the viewport filter into the scan. Line, histogram, and 2-D
histogram traces then run a streaming formulation over batches instead of the
in-memory kernel. Peak memory tracks the batch and the output, not the row
count. Both paths give the same result. Only the formulation differs.

Two things still grow with the data, on both source kinds:
- A grouped trace keeps state per group.
- A dense 2-D histogram keeps a grid proportional to its cell count (issue #19).

For a Parquet fold, peak memory is the row-group prefetch window of the
reader. Set `POLARS_ROW_GROUP_PREFETCH_SIZE` to bound it.

A multi-file, hive-partitioned, or cloud scan runs the same plans. Its memory
use is not characterized here.

### Column bounds

An axis without a zoom range needs the minimum and the maximum of its column.
On a bare single-file local Parquet scan, FlexViz reads both from the footer
statistics through pyarrow, which reads a few kilobytes. Every other source
decodes the column instead. On a CSV scan, that pass reads the whole file.

An uncached scan resolves the bounds again on each request, because the file
can change between requests. `cache=True` declares the data static and keeps
the resolved bounds for the life of the process. See
[Caching and live brushing](caching-and-live-brushing.md).

If you explore the same CSV file more than once, write it to Parquet first.
The scan then reads only the columns the charts use, and the bounds come from
the footer.

### Transformations

Any transformation that you apply before you hand the frame over
(`lf.filter(...).with_columns(...)`) stays lazy and fuses into every FlexViz
query. A transformation above a scan node does not remove the scan path, but a
`collect()` does.

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

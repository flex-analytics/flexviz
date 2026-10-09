# Live data

`register_stream` registers a source that grows while the page is open. You
add rows with `Stream.append`. The page asks the server for new rows once a
second and then refreshes every chart.

```python
import threading
import time
from datetime import datetime, timedelta

import polars as pl

from flexviz import Dashboard, register_stream

df = pl.DataFrame({"timestamp": [datetime.now()], "value": [0.0]})
stream = register_stream(
    "sensor", df, order_by="timestamp", window=timedelta(minutes=1)
)


def produce() -> None:
    value = 0.0
    while True:
        time.sleep(1)
        value += 1.0
        stream.append(
            pl.DataFrame({"timestamp": [datetime.now()], "value": [value % 10]})
        )


threading.Thread(target=produce, daemon=True).start()

dash = Dashboard()  # no data: the dashboard reads the stream by name
dash.add_figure(title="Value").add_line(x="timestamp", y="value")
dash.add_figure(title="Distribution").add_histogram(x="value", bins=10)
dash.show(source_name="sensor")
```

## Show a stream

Build the `Dashboard` or `Figure` without data. Then pass the stream name to
`show(source_name=...)`. A dashboard with data of its own cannot use the name
of a stream: `show()` raises a `ValueError`, because the data would replace
the stream.

## Rows

The rows must be sorted by `order_by`. The `order_by` column must not have
nulls or NaN. Each append must start at or after the last row of the
stream. An append with rows out of order raises a `ValueError`, and the
stream does not change. The columns of an append must have the same names,
order and dtypes as the stream. You can append from more than one thread.

`Stream.version` goes up by one on each append that adds rows. The page
reads this number to know when to refresh.

## Zoom and selections

- An axis at autorange follows the data. Each refresh fits it to the new rows.
- A zoomed axis keeps its range. Only its data refreshes.
- A double-click returns the axis to autorange, so it follows the data again.
- A selection stays active. Each refresh applies it to the new rows too.
- While you hold a mouse button down, for example to drag a zoom or brush,
  the page does not refresh.

## Window

With `window`, an x axis over `order_by` that is not zoomed shows only the
last part of the data: `[last - window, last]`. The line then slides as rows
arrive. Use a positive `timedelta` for a `Datetime` `order_by` and a
positive number for a numeric one. A zoom or pan shows any range of the
history. Charts of other columns, such as the histogram in the example,
still use all rows.

A locked x axis gets no window. Its line uses all rows, and the lock keeps
the range that you locked.

## Limits

A stream keeps every row in memory, and each refresh recomputes the charts
on the server. These features are not available:

- A limit on rows or memory (old rows are never removed)
- Live or locked modes, and window preset buttons
- Following new data while an axis is zoomed or locked
- Push updates (WebSocket or SSE). The page polls once a second.
- Incremental updates and cube deltas
- Caching (a stream is never cached, and live brushing is off)
- A shrinking y axis in overlay mode (the y range only grows)
- Rows out of order, updates and deletes of rows
- A stream over a file scan (the first data is collected into memory)
- More than one server process

## A resident frame is a snapshot

`register_source(name, df)` and `Dashboard(df)` read `df` once. Rows that
you add to `df` in place later are not seen, also with `cache=False`. For
data that grows, use `register_stream`.

# Databases

FlexViz can read a database table or query in two ways:

- **Live**: a `SQLSource` sends each chart aggregation to the database as one
  SQL query. Only the aggregated result travels to FlexViz.
- **Extract**: you load the rows into a Polars frame once, and FlexViz works on
  the frame.

Which way is faster depends on the database engine and on the network. The
next section gives measured numbers.

## Live or extract

The measurements below use one dashboard with four charts. Two charts show one
line per process run, one bar chart counts the runs per recipe, and one
histogram shows the pressure distribution. The data is 100 runs of 50,000
rows (5M rows), selected from a 100M-row detail table with a join. Times are
medians in seconds. "Ready" is the time until the first chart request can
start.

| Path | Network | Ready | First render | Zoom | Brush |
|---|---|---|---|---|---|
| Live ClickHouse | same machine | 0.6 | 1.0 | 0.55 | 1.1 |
| Live ClickHouse | 20 Mbit/s, 30 ms | 0.7 | 2.0 | 1.3 | 1.5 |
| Live Postgres | same machine | 0.1 | 8.8 | 0.9 | 7.2 |
| Live Postgres | 20 Mbit/s, 30 ms | 0.8 | 15.2 | 3.9 | 9.5 |
| Extract, ADBC | same machine | 2.0 | 0.5 | 0.35 | 0.4 |
| Extract, ADBC | 20 Mbit/s, 30 ms | 105 | 0.4 | 0.3 | 0.3 |
| Extract, SQLAlchemy | same machine | 9.7 | 0.5 | 0.35 | 0.4 |

On the full 100M-row table, live ClickHouse renders in 1.5 s and live
Postgres in 45 s on the same machine.

Use these rules:

- **A columnar engine** (ClickHouse, DuckDB, and warehouses such as Snowflake,
  BigQuery or Trino): use a live `SQLSource`. The engine aggregates millions
  of rows in milliseconds, so every zoom and selection stays interactive.
- **A row store** (Postgres) on a fast network, with rows that fit in memory:
  extract the rows once with an Arrow driver. Postgres aggregates 10 to 30
  times more slowly than a columnar engine, and every live interaction waits
  for it.
- **A row store over a slow network, or more rows than fit in memory**: use a
  live `SQLSource`. Only the aggregates cross the network.

## Live: `SQLSource`

```python
import flexviz as fv

src = fv.SQLSource(
    "postgresql://user:password@host:5432/dbname",
    query="""
        SELECT h.id AS run_id, h.recipe, d.elapsed_s, d.chamber_pressure
        FROM log_detail AS d JOIN log_header AS h ON d.header_id = h.id
        WHERE h.tool_id = 'ETCH-07'
    """,
)
dash = fv.Dashboard(src)
dash.add_figure().add_line(x="elapsed_s", y="chamber_pressure", group_by="run_id")
dash.add_figure().add_histogram(x="chamber_pressure", bins=60)
dash.show()
```

Install the extra first: `pip install "flexviz[sql]"`. For a Postgres URI,
also install a driver: `pip install adbc-driver-postgresql` (preferred, it
returns Arrow) or `pip install "psycopg[binary]"`.

Give either `table=` or `query=`:

- `table="log_detail"` reads a table or a view. Write the name as you write it
  in SQL, with a schema prefix if necessary.
- `query="SELECT ..."` reads the result of a query, for example a join.
  FlexViz puts the query in a `WITH` clause unchanged, and the database
  optimizes the two together.

`connection` can be:

| Value | Example | Dialect |
|---|---|---|
| A Postgres URI | `"postgresql://user:pw@host/db"` | found |
| A SQLAlchemy engine | `sqlalchemy.create_engine(uri)` | found |
| A DuckDB connection | `duckdb.connect("data.duckdb", read_only=True)` | found |
| A function that opens a DB-API connection | `lambda: clickhouse_connect.dbapi.connect(...)` | give `dialect=` |

FlexViz opens up to `max_connections` connections (4 by default) and runs the
queries of one request in parallel. It does not share one connection between
threads, so it needs a function that opens a connection, not a connection
object. A DuckDB connection is the exception: FlexViz opens a cursor per
thread.

ClickHouse example:

```python
from clickhouse_connect import dbapi

src = fv.SQLSource(
    lambda: dbapi.connect(host="ch.example", username="reader", password=pw),
    table="log_detail",
    dialect="clickhouse",
)
```

### What runs in the database

Each chart is one `GROUP BY` query. A line becomes one row per x bucket (per
series) with the y minimum, the y maximum and the x of each. A histogram
becomes one count per bin. A bar, pie or treemap becomes one aggregate per
category. FlexViz first asks the database for the minimum and maximum of the
binned columns, so that the bins line up across charts.

The results are equal to the results on a Polars frame with the same rows.
The test suite checks this for every supported chart.

### Supported dialects and charts

Dialects: Postgres, DuckDB and ClickHouse. Other dialects give an error at
construction.

| Chart | Live on a `SQLSource` |
|---|---|
| `add_line` with `downsample` `minmax`, `lttb` or `fpcs`, grouped or not | yes |
| `add_line` with `downsample="nth"` | no: a table has no row order to stride |
| `add_histogram`, grouped or not | yes |
| `add_bar`, `add_pie`, `add_treemap` | yes |
| `add_histogram2d`, `add_geo_histogram2d` | yes |
| `add_corr_heatmap` | Pearson only |
| `add_boxplot`, `add_geo_line` | no |

A chart that a SQL source cannot run, or a column that the source does not
have, gives an error when you add the chart, in your own code.

### Limits

- Live brushing (the cube) is off for a SQL source. A brush updates the other
  charts when you release the mouse, with one request.
- Every interaction is a database query. Zoom and selection speed follow the
  database. Add an index on the columns that you filter and zoom on, for
  example `(run_id, elapsed_s)`.
- On Postgres, a larger `work_mem` for the FlexViz user lets the grouping stay
  in memory instead of sorting on disk. A line over 5M rows took 3.1 s with
  64 MB and 1.9 s with 1 GB:
  `ALTER ROLE flexviz_reader SET work_mem = '1GB'`.
- `Dashboard(src, cache=True)` declares that the data does not change while the
  server runs. FlexViz then keeps the column bounds and the first render. Do
  not set it on a table that receives new rows.

### Security

- Column names come from the dashboard spec, which the browser sends. FlexViz
  puts only columns of the source schema into the SQL, quoted, and renders
  every value as a typed literal, so a spec cannot inject SQL. A column name
  that is not in the schema is refused before any query runs.
- FlexViz runs its connections in autocommit mode, so an idle connection holds
  no lock on your tables.
- The database credentials stay in the Python process. The browser receives
  only aggregates.
- The FlexViz server has no authentication. Whoever can reach its port can
  request aggregates of the source. Connect with a read-only database user.

## Extract: load the rows once

Load the rows with an Arrow driver. A driver that returns Python rows
(SQLAlchemy with psycopg, or `pandas.read_sql`) is 4 to 5 times slower for
millions of rows.

```python
import polars as pl
import flexviz as fv

df = pl.read_database_uri(query, uri, engine="adbc")  # pip install adbc-driver-postgresql
dash = fv.Dashboard(df, cache=True)
```

`engine="connectorx"` (`pip install connectorx`) also returns Arrow. If the
rows do not fit in memory, write them to Parquet once and scan the file:

```python
df.write_parquet("runs.parquet")
fv.Dashboard(pl.scan_parquet("runs.parquet"))
```

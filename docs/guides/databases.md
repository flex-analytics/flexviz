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
histogram shows the pressure distribution. Times are medians in seconds.
"Ready" is the time until the first chart request can start: for an extract,
the load.

### Long runs

100 runs of 50,000 rows (5M rows) and 5 columns, selected from a 100M-row
detail table with a join:

| Path | Network | Ready | First render | Zoom | Brush |
|---|---|---|---|---|---|
| Live ClickHouse | same machine | 0.6 | 1.0 | 0.55 | 1.1 |
| Live ClickHouse | 20 Mbit/s, 30 ms | 0.7 | 2.0 | 1.3 | 1.5 |
| Live Trino, Parquet files | same machine | 0.1 | 4.3 | 1.25 | 3.6 |
| Live Trino, Parquet files | 20 Mbit/s, 30 ms | 0.4 | 5.1 | 2.4 | 4.3 |
| Live DuckDB, the same Parquet files | same machine | 0.1 | 1.5 | 0.6 | 1.7 |
| Live Postgres | same machine | 0.1 | 8.8 | 0.9 | 7.2 |
| Live Postgres | 20 Mbit/s, 30 ms | 0.8 | 15.2 | 3.9 | 9.5 |
| Extract, ADBC | same machine | 2.0 | 0.5 | 0.35 | 0.4 |
| Extract, ADBC | 20 Mbit/s, 30 ms | 105 | 0.4 | 0.3 | 0.3 |
| Extract, SQLAlchemy | same machine | 9.7 | 0.5 | 0.35 | 0.4 |

On the full 100M-row table, live ClickHouse renders in 1.5 s, live DuckDB on
Parquet files in 3.0 s, live Trino on the same files in 4.6 s, and live
Postgres in 45 s on the same machine.

### Short, wide runs

100 runs of 1,813 rows (181,300 rows), selected from a 7M-row detail table.
The query returns 211 columns, and the dashboard uses 5 of them. The detail
table has an index on the join column.

| Path | Network | Ready | First render | Zoom | Brush |
|---|---|---|---|---|---|
| Live Postgres | same machine | 0.1 | 0.66 | 0.4 | 0.55 |
| Extract, connectorx, 5 columns | same machine | 0.3 | 0.3 | 0.2 | 0.25 |
| Extract, connectorx, 211 columns | same machine | 2.3 | 0.3 | 0.2 | 0.25 |
| Extract, SQLAlchemy + pandas, 211 columns | same machine | 17.6 | 0.3 | 0.25 | 0.25 |

Without the index, live Postgres took 3.2 s per first render on the same
machine, because every chart query read the whole table.

Use these rules:

- **A columnar engine** (ClickHouse, DuckDB or Trino): use a live
  `SQLSource`. The engine aggregates millions of rows in milliseconds to
  seconds, so every zoom and selection stays interactive.
- **Postgres with a few hundred thousand rows behind each chart**: a live
  `SQLSource` is interactive, if the query can use an index. Postgres
  aggregates 10 to 30 times more slowly than a columnar engine, so the row
  count decides.
- **Postgres with millions of rows behind each chart**: extract the plotted
  columns once. Every live interaction waits several seconds for Postgres.
- **Short series over a slow network**: extract. A line draws 1,000 points per
  series, so a series of 2,000 rows comes back almost whole on every request.
  An extract sends those rows once.
- **Long series, histograms and bars over a slow network, or more rows than
  fit in memory**: use a live `SQLSource`. Only the aggregates cross the
  network.

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
thread. A cursor does not see the TEMP tables and the registered views
(`con.register`) of your connection, so use a table or a view in the database.

ClickHouse example:

```python
from clickhouse_connect import dbapi

src = fv.SQLSource(
    lambda: dbapi.connect(host="ch.example", username="reader", password=pw),
    table="log_detail",
    dialect="clickhouse",
)
```

Trino example (`pip install trino`):

```python
import trino

src = fv.SQLSource(
    lambda: trino.dbapi.connect(
        host="trino.example", port=8080, user="reader", catalog="iceberg", schema="fab"
    ),
    table="log_detail",
    dialect="trino",
)
```

On Trino, a column of type `uuid`, `time`, `varbinary`, `array`, `map` or `row`
is not available to charts. Cast it in the query, for example
`CAST(run_id AS varchar) AS run_id`.

Use Trino through its catalogs for files and lakehouse tables (Hive, Iceberg,
Delta Lake). Do not put Trino in front of Postgres to speed it up: Trino's
Postgres connector does not push a binned `GROUP BY` down, so Trino reads every
row from Postgres. A 60-bin histogram over 100M rows took 63 s that way and
11 s in Postgres directly.

### What runs in the database

Each chart is one `GROUP BY` query. A line becomes one row per x bucket (per
series) with the y minimum, the y maximum and the x of each. A histogram
becomes one count per bin. A bar, pie or treemap becomes one aggregate per
category. FlexViz first asks the database for the minimum and maximum of the
binned columns, so that the bins line up across charts.

The results are equal to the results on a Polars frame with the same rows,
with these exceptions:

- A sum or a mean can be different in the last digits, because the database
  adds the values in its own order.
- If more rows have the same y minimum or y maximum in a line bucket, the
  database can show the x of a different one of these rows.
- On Postgres, a sum of 64-bit integers is exact only up to 2**53.
- On Postgres, the minimum and the maximum of a text column follow the
  collation of the database.

The test suite compares the two for every supported chart on DuckDB, Postgres
and ClickHouse.

### Supported dialects and charts

Dialects: Postgres, DuckDB, ClickHouse and Trino. Other dialects give an error
at construction.

The test suite runs on these four engines only. FlexViz does not test engines
that are compatible with one of them, for example TimescaleDB (Postgres),
MotherDuck (DuckDB) or Starburst (Trino).

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
  database. Add an index on the columns that you join, filter and zoom on, for
  example `(run_id, elapsed_s)`. Postgres does not index a foreign key column
  by itself.
- On Postgres, a larger `work_mem` for the FlexViz user lets the grouping stay
  in memory instead of sorting on disk. A line over 5M rows took 3.1 s with
  64 MB and 1.9 s with 1 GB:
  `ALTER ROLE flexviz_reader SET work_mem = '1GB'`.
- On Trino, a median (`agg="median"`) sorts the values of each group in
  memory, because Trino has no exact percentile function.
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

Select only the columns that you plot. For a wide table, the column count
matters more than the driver: in the short-run measurements, 5 of 211 columns
loaded 8 times faster.

Load the rows with an Arrow driver. A driver that returns Python rows
(SQLAlchemy with psycopg, or `pandas.read_sql`) was 5 to 8 times slower in the
measurements above.

```python
import polars as pl
import flexviz as fv

df = pl.read_database_uri(query, uri, engine="adbc")  # pip install adbc-driver-postgresql
dash = fv.Dashboard(df, cache=True)
```

`engine="connectorx"` (`pip install connectorx`) also returns Arrow.

Cast a Postgres `numeric` column to `double precision` in the query, for
example `chamber_pressure::double precision AS chamber_pressure`. ADBC returns
`numeric` as text and connectorx as `Decimal`, and FlexViz charts need a float.
A `SQLSource` reads `numeric` as a float with every driver.

If the rows do not fit in memory, write them to Parquet once and scan the file:

```python
df.write_parquet("runs.parquet")
fv.Dashboard(pl.scan_parquet("runs.parquet"))
```

## Files on object storage: no warehouse

If the data is already Parquet or Iceberg files, for example on S3, one
machine can read it without a warehouse:

- Polars: pass `pl.scan_parquet("s3://bucket/runs/*.parquet")` or
  `pl.scan_iceberg(...)` to `Dashboard`. FlexViz keeps the scan lazy.
- DuckDB: pass a DuckDB connection to `SQLSource`, with a query that reads the
  files, for example
  `fv.SQLSource(duckdb.connect(), query="SELECT * FROM read_parquet('s3://bucket/runs/*.parquet')")`.
  For Iceberg, attach the catalog with DuckDB's `iceberg` extension first.

On one machine and the same Parquet files, DuckDB rendered the dashboard of
the first table in 1.5 s and Trino in 4.3 s. Trino is the better choice when
your team already runs it, or when the data needs more than one machine.

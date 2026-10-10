"""Database sources: each aggregation runs in the database as one SQL query.

``SQLSource`` stands where an ``LFQueryBuilder`` stands. The engine and the
server use the same surface: ``schema``, ``physical_minmax``, ``compile_filter``
and ``aggregate``. On a SQL source a trace returns a plan that takes a
``SQLFrame`` instead of a LazyFrame. The plan sends one ``GROUP BY``, gets back
the small grouped frame that its Polars twin collects, and shapes it with the
same Polars code. So ``_to_update`` never sees where the rows came from, and
only the aggregated result crosses the network.

SQLGlot builds every query: it quotes identifiers and renders literals for the
dialect. The meaning of a few functions differs per database, so the functions
the plans need (the x at a y extremum, epoch units, integer division, NaN) are
defined here per dialect. A dialect that is not in ``_DIALECTS`` is refused
rather than sent SQL that nobody has run.
"""

from __future__ import annotations

import datetime as dt
import math
import queue
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
import polars as pl

try:
    from sqlglot import exp
    from sqlglot.dialects.dialect import Dialect
    from sqlglot.tokens import TokenType
except ImportError as e:  # pragma: no cover - depends on the install
    raise ImportError(
        "SQL sources need SQLGlot. Install it with: pip install 'flexviz[sql]'"
    ) from e

from .LF import AggregationSpec, GroupedAggregationSpec
from .predicates import _values_to_typed_series
from .spec import ClauseFilter, SelectionPredicate
from .trace.base import _dtype_for_col, _temporal_bound_toward, _typed_range_bounds

#: The dialects whose SQL has run against a real database in the test suite.
_DIALECTS = ("postgres", "duckdb", "clickhouse")
_DIALECT_ALIASES = {"postgresql": "postgres", "clickhousedb": "clickhouse"}

# Every column a plan adds carries this prefix, like the Polars plans.
_P = "__fv_"
_SRC = _P + "src"

#: Postgres type OIDs, for drivers that return rows instead of Arrow. Only the
#: schema query needs them: an empty result carries no values to infer from.
_PG_OIDS: dict[int, pl.DataType] = {
    16: pl.Boolean(),
    20: pl.Int64(),
    21: pl.Int16(),
    23: pl.Int32(),
    25: pl.String(),
    700: pl.Float32(),
    701: pl.Float64(),
    1042: pl.String(),
    1043: pl.String(),
    1082: pl.Date(),
    1114: pl.Datetime("us"),
    1184: pl.Datetime("us", "UTC"),
    1700: pl.Float64(),
}


_CH_SIMPLE: dict[str, pl.DataType] = {
    **{f"Int{b}": getattr(pl, f"Int{b}")() for b in (8, 16, 32, 64)},
    **{f"UInt{b}": getattr(pl, f"UInt{b}")() for b in (8, 16, 32, 64)},
    "Float32": pl.Float32(),
    "Float64": pl.Float64(),
    "Bool": pl.Boolean(),
    "String": pl.String(),
    "Date": pl.Date(),
    "Date32": pl.Date(),
}


def _clickhouse_dtype(name: str) -> pl.DataType:
    """The Polars dtype of a ClickHouse type name, as its DB-API rows load."""
    # In any order: ClickHouse writes LowCardinality(Nullable(T)).
    while name.startswith(("Nullable(", "LowCardinality(")):
        name = name[name.index("(") + 1 : -1]
    if name in _CH_SIMPLE:
        return _CH_SIMPLE[name]
    if name.startswith("Decimal"):
        # Read as a double, as Postgres numeric is.
        return pl.Float64()
    if name.startswith("DateTime"):
        # DateTime64(precision[, 'zone']) or DateTime[('zone')]
        args = [a.strip(" '") for a in name.partition("(")[2].rstrip(")").split(",")]
        unit = "us"
        if name.startswith("DateTime64("):
            precision = int(args.pop(0))
            unit = "ms" if precision <= 3 else "us" if precision <= 6 else "ns"
        return pl.Datetime(unit, args[0] if args and args[0] else None)
    return pl.String()


# ---------------------------------------------------------------------------
# Connections
# ---------------------------------------------------------------------------


def _uri_connector(uri: str) -> tuple[Callable[[], Any], str]:
    """A function that opens a connection for a URI, and its dialect.

    Postgres prefers ADBC because it returns Arrow, and falls back to psycopg.
    """
    scheme = uri.split(":", 1)[0].split("+", 1)[0].lower()
    if scheme in ("postgres", "postgresql"):
        try:
            import adbc_driver_postgresql.dbapi as adbc

            return (lambda: adbc.connect(uri)), "postgres"
        except ImportError:
            pass
        try:
            import psycopg

            return (lambda: psycopg.connect(uri)), "postgres"
        except ImportError:
            raise ImportError(
                "A Postgres URI needs a driver. Install one with: "
                "pip install adbc-driver-postgresql"
            ) from None
    raise ValueError(
        f"FlexViz opens only postgresql:// URIs itself, got {scheme!r}. For other "
        "databases, pass a function that opens a DB-API connection, and the dialect."
    )


def _connector(connection: Any) -> tuple[Callable[[], Any], str | None]:
    """A function that opens one connection, and the dialect it implies."""
    if isinstance(connection, str):
        return _uri_connector(connection)
    # A SQLAlchemy Engine pools its own connections.
    if hasattr(connection, "raw_connection") and hasattr(connection, "dialect"):
        name = connection.dialect.name
        return connection.raw_connection, _DIALECT_ALIASES.get(name, name)
    # One DuckDB database: each thread reads through its own cursor.
    if type(connection).__module__.lstrip("_").startswith("duckdb"):
        return connection.cursor, "duckdb"
    # A connection object can be callable too (sqlite3): it has a cursor.
    if callable(connection) and not hasattr(connection, "cursor"):
        return connection, None
    raise TypeError(
        "connection must be a URI, a SQLAlchemy Engine, a DuckDB connection, or a "
        "function that opens a DB-API connection. A single DB-API connection "
        "cannot serve parallel queries, so pass the function that opens it."
    )


def _fetch(conn: Any, sql: str) -> pl.DataFrame:
    """Run ``sql`` on one connection, as a Polars frame.

    Arrow where the driver has it, so an empty result keeps its column types.
    """
    # A SQLAlchemy pool hands out a proxy; Polars mistakes it for an engine.
    conn = getattr(conn, "dbapi_connection", conn)
    module = type(conn).__module__.split(".", 1)[0].lstrip("_")
    if module == "duckdb":
        return conn.execute(sql).pl()
    if module.startswith("adbc_driver"):
        cur = conn.cursor()
        try:
            cur.execute(sql)
            table = cur.fetch_arrow_table()
        finally:
            cur.close()
        # ADBC sends Postgres numeric as text. It reads as Float64, as the
        # type OID map does for row drivers.
        numeric = [
            f.name
            for f in table.schema
            if (f.metadata or {}).get(b"ADBC:postgresql:typname") == b"numeric"
        ]
        df: pl.DataFrame = pl.from_arrow(table)  # type: ignore[assignment]
        return df.with_columns(pl.col(numeric).cast(pl.Float64))
    if module == "clickhouse_connect":
        # Typed by the column type names, so an empty result keeps its types.
        # Polars cannot read some of them, such as Nullable(Decimal(p, s)).
        cur = conn.cursor()
        cur.execute(sql)
        schema = {d[0]: _clickhouse_dtype(str(d[1])) for d in cur.description}
        return pl.DataFrame(cur.fetchall(), schema=schema, orient="row")
    return pl.read_database(sql, conn)


def _close(conn: Any) -> None:
    try:
        conn.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------


def _alias(e: exp.Expression, name: str) -> exp.Alias:
    return exp.alias_(e, name, quoted=True)


def _num(v: float) -> exp.Expression:
    """A numeric literal that parses back to the same double or integer."""
    if isinstance(v, float) and not math.isfinite(v):
        return exp.cast(exp.Literal.string(repr(v)), exp.DataType.Type.DOUBLE)
    if v < 0:
        # Parenthesized, so ``a - -1`` can never print as the comment ``a --1``.
        return exp.Paren(this=exp.Neg(this=_num(-v)))
    if isinstance(v, float):
        # Exponent form, cast to a non-nullable double: see "SQL sources" in
        # Architecture.md for why each part matters.
        return exp.cast(
            exp.Literal.number(f"{v:.17e}"),
            exp.DataType(this=exp.DataType.Type.DOUBLE, nullable=False),
        )
    return exp.Literal.number(int(v))


def _and(*conds: exp.Expression) -> exp.Expression:
    return exp.and_(*conds) if conds else exp.true()


@dataclass(frozen=True)
class SQLFrame:
    """The source's rows under the cross-filter: what a SQL plan reads.

    The SQL twin of the filtered LazyFrame a Polars plan receives.
    """

    source: SQLSource
    where: tuple[exp.Expression, ...] = ()

    @property
    def dialect(self) -> str:
        return self.source.dialect

    def col(self, name: str) -> exp.Column:
        return self.source.col(name)

    def num(self, name: str) -> exp.Expression:
        """A value column as a number. Postgres neither sums, takes the min of,
        nor casts to a double a Boolean, so it becomes an integer there."""
        c = self.col(name)
        if self.dialect == "postgres" and self.source.schema[name] == pl.Boolean:
            return exp.cast(c, exp.DataType.Type.INT)
        return c

    def collect(self, select: exp.Select) -> pl.DataFrame:
        """Run ``select`` over the filtered rows, as a Polars frame."""
        if self.where:
            select = select.where(*self.where, append=True)
        return self.source._run(select)

    # -- the functions whose meaning differs per dialect ---------------------

    def phys(self, name: str, dtype: pl.DataType | None) -> exp.Expression:
        """A column in its Polars physical units: epoch units for a temporal
        column, ``num`` otherwise. Grids are arithmetic on these."""
        c = self.num(name)
        if dtype is None or not dtype.is_temporal():
            return c
        d = self.dialect
        if dtype == pl.Date:
            if d == "clickhouse":
                return exp.func("toInt32", c)
            epoch = exp.cast(exp.Literal.string("1970-01-01"), exp.DataType.Type.DATE)
            return exp.cast(c - epoch, exp.DataType.Type.BIGINT)
        if not isinstance(dtype, pl.Datetime):
            raise TypeError(f"column {name!r} has dtype {dtype}, not binnable in SQL")
        unit = dtype.time_unit
        if d == "duckdb":
            return exp.func(f"epoch_{unit}", c)
        if d == "clickhouse":
            # The epoch functions take only DateTime64, not the 32-bit DateTime.
            precision, fn = {"ms": (3, "Milli"), "us": (6, "Micro"), "ns": (9, "Nano")}[
                unit
            ]
            return exp.func(
                f"toUnixTimestamp64{fn}",
                exp.func("toDateTime64", c, exp.Literal.number(precision)),
            )
        # Postgres keeps microseconds. date_part returns a double, so the
        # scaled value is within 0.5 of the exact count and rounds back to it
        # (equal to the numeric EXTRACT on all 100M benchmark rows, 2x faster).
        scale = {"ms": 1_000, "us": 1_000_000, "ns": 1_000_000_000}[unit]
        epoch = exp.Anonymous(
            this="date_part", expressions=[exp.Literal.string("epoch"), c]
        )
        return exp.cast(
            exp.Round(this=epoch * _num(scale)),
            exp.DataType.Type.BIGINT,
        )

    def nan(self, name: str) -> exp.Expression:
        """The column is NaN."""
        c = self.col(name)
        if self.dialect == "postgres":
            # In Postgres NaN equals NaN and sorts above every number.
            return exp.EQ(this=c, expression=self.lit(float("nan")))
        if self.dialect == "clickhouse":
            # isNaN refuses a Decimal, which the schema reads as Float64.
            return exp.func("isNaN", exp.func("toFloat64", c))
        return exp.func("isnan", c)

    def usable(self, name: str, dtype: pl.DataType | None) -> exp.Expression:
        """Not null, and not NaN for a float: the rows a bin or a bucket takes."""
        cond: exp.Expression = exp.Not(
            this=exp.Is(this=self.col(name), expression=exp.Null())
        )
        if dtype is not None and dtype.is_float():
            cond = exp.and_(cond, exp.Not(this=self.nan(name)))
        return cond

    def int_div(self, a: exp.Expression, b: int) -> exp.Expression:
        """Integer division of a non-negative ``a``, exact past 2**53."""
        d = self.dialect
        if d == "clickhouse":
            return exp.func("intDiv", a, _num(b))
        if d == "duckdb":
            return exp.IntDiv(this=a, expression=_num(b))
        # Postgres: ``/`` on two integers truncates, which is floor for a >= 0.
        return exp.Div(this=a, expression=_num(b), typed=True)

    def lit(self, value: Any, dtype: pl.DataType | None = None) -> exp.Expression:
        """A literal typed for ``dtype``. A temporal value may come in physical
        units (an int), which converts through the column's own dtype."""
        if value is None:
            return exp.Null()
        if dtype is not None and dtype.is_temporal() and isinstance(value, int):
            value = pl.Series([value], dtype=pl.Int64).cast(dtype).item()
        if isinstance(value, float):
            if math.isnan(value):
                return exp.cast(exp.Literal.string("NaN"), exp.DataType.Type.DOUBLE)
            return _num(value)
        if isinstance(value, int) and not isinstance(value, bool):
            return _num(value)
        if isinstance(value, dt.datetime) and self.dialect == "clickhouse":
            # ClickHouse reads a datetime without a zone in the server's zone,
            # but a naive Polars datetime is UTC, as the driver reads it.
            if value.tzinfo is None:
                value = value.replace(tzinfo=dt.timezone.utc)
            else:
                value = value.astimezone(dt.timezone.utc)
        return exp.convert(value)

    def range_cond(
        self, name: str, lo: Any, hi: Any, closed: str = "both"
    ) -> exp.Expression:
        """``lo <= c <= hi`` (per ``closed``) on the raw column, so an index on
        it can serve the filter. Bounds are literals typed for the column."""
        dtype = self.source.schema.get(name)
        c = self.col(name)
        lower = exp.GTE if closed in ("both", "left") else exp.GT
        upper = exp.LTE if closed in ("both", "right") else exp.LT
        return exp.and_(
            lower(this=c, expression=self.lit(lo, dtype)),
            upper(this=c, expression=self.lit(hi, dtype)),
        )

    def phys_range_cond(self, name: str, lo: float, hi: float) -> exp.Expression:
        """The rows of a binned axis zoomed to physical ``[lo, hi]``, the SQL
        twin of ``snapped_axis``'s mask."""
        dtype = self.source.schema.get(name)
        if dtype is not None and dtype.is_temporal():
            return self.range_cond(name, math.ceil(lo), math.floor(hi))
        bounds = _typed_range_bounds(name, (lo, hi), self.source.schema)
        return self.range_cond(name, *_eval(bounds))

    def extreme_by(
        self, x_name: str, y_name: str, kind: str
    ) -> tuple[exp.Expression, exp.Expression]:
        """``(x at the y extremum, the y extremum)`` as two aggregates, whose
        values cast to the dtypes of x and y.

        Postgres has no arg_min. Arrays compare element by element, so the
        smallest ``ARRAY[y, x]`` holds the smallest y and, on a tie, the
        smallest x. Measured on 5M rows, this beats ``array_agg ORDER BY``,
        ``DISTINCT ON`` and window functions. Both elements are doubles, so it
        takes only an x and a y that a double holds exactly.
        """
        sch = self.source.schema
        x_dtype, y_dtype = sch.get(x_name), sch.get(y_name)
        x, y = self.col(x_name), self.num(y_name)
        agg = exp.Min if kind == "min" else exp.Max
        if self.dialect != "postgres":
            by = exp.ArgMin if kind == "min" else exp.ArgMax
            return by(this=x, expression=y), agg(this=y)
        # x is exact below 2**53 in physical units; y must be a float or an
        # integer of at most 32 bits.
        x_fits = x_dtype not in (pl.Int64, pl.UInt64) and not (
            isinstance(x_dtype, pl.Datetime) and x_dtype.time_unit == "ns"
        )
        y_fits = y_dtype is not None and (
            y_dtype.is_float() or y_dtype in (pl.Boolean, pl.Int8, pl.Int16, pl.Int32)
        )
        if x_fits and y_fits:
            dbl = exp.DataType.Type.DOUBLE
            phys = exp.cast(self.phys(x_name, x_dtype), dbl)
            arr = exp.Paren(
                this=agg(this=exp.Array(expressions=[exp.cast(y, dbl), phys]))
            )
            # SQLGlot writes a 0-based index 1-based for Postgres.
            x_at: exp.Expression = exp.Bracket(
                this=arr, expressions=[exp.Literal.number(1)]
            )
            if x_dtype is not None and not x_dtype.is_float():
                # Integral again, in physical units.
                x_at = exp.cast(x_at, exp.DataType.Type.BIGINT)
            return x_at, exp.Bracket(
                this=arr.copy(), expressions=[exp.Literal.number(0)]
            )
        # Sorting each bucket is slower than the array minimum, but keeps types.
        desc = kind == "max"
        ordered = exp.ArrayAgg(
            this=exp.Order(
                this=x,
                expressions=[
                    exp.Ordered(this=y, desc=desc),
                    exp.Ordered(this=x.copy(), desc=desc),
                ],
            )
        )
        first = exp.Bracket(
            this=exp.Paren(this=ordered), expressions=[exp.Literal.number(0)]
        )
        return first, agg(this=y.copy())


def _drop_final_semicolon(query: str, dialect: str) -> str:
    """``query`` without a final ``;``, which cannot stand inside ``WITH``.
    A comment after it stays: the dialect's tokens tell code from comment."""
    tokens = Dialect.get_or_raise(dialect).tokenize(query)
    if tokens and tokens[-1].token_type == TokenType.SEMICOLON:
        return query[: tokens[-1].start] + query[tokens[-1].end + 1 :]
    return query


def _eval(exprs: Sequence[pl.Expr]) -> tuple[Any, ...]:
    """Evaluate literal Polars expressions to Python values."""
    return pl.select(*[e.alias(str(i)) for i, e in enumerate(exprs)]).row(0)


# ---------------------------------------------------------------------------
# The source
# ---------------------------------------------------------------------------


class SQLSource:
    """A database table or query as a FlexViz data source.

    Each aggregation runs in the database, and only its small result comes back.
    Pass it where you would pass a DataFrame::

        src = SQLSource("postgresql://user:pw@host/db", table="log_detail")
        Dashboard(src).add_figure().add_line(x="ts", y="pressure", group_by="run_id")

    ``connection`` is a URI (Postgres), a SQLAlchemy Engine, a DuckDB
    connection, or a function that opens a DB-API connection (then also give
    ``dialect``). Give either ``table`` (a table or view name, as you would
    write it in SQL) or ``query`` (a ``SELECT``, for a join for example).

    ``Dashboard(src, cache=True)`` declares that the data does not change while
    the server runs, as for a frame: FlexViz then keeps column bounds and
    initial results. ``max_connections`` caps the queries one source runs at a
    time, and so the connections it keeps open.
    """

    is_sql = True
    #: A SQL plan reads the rows in no order, like a scan plan.
    is_scan = True
    sorted_cols: frozenset[str] = frozenset()

    def __init__(
        self,
        connection: Any,
        table: str | None = None,
        *,
        query: str | None = None,
        dialect: str | None = None,
        max_connections: int = 4,
    ) -> None:
        if (table is None) == (query is None):
            raise ValueError("Give exactly one of table= or query=.")
        self._connect, implied = _connector(connection)
        self._idle: queue.SimpleQueue = queue.SimpleQueue()
        self._slots = threading.BoundedSemaphore(max_connections)
        self.max_connections = max_connections
        # Set by the registrar (``Dashboard(cache=...)``, ``register_source``),
        # as for a frame source.
        self.cache = False
        self._minmax_memo: dict[str, tuple[Any, Any]] = {}
        name = dialect or implied
        if name is None:
            raise ValueError(
                "A function that opens a connection needs dialect=, one of "
                f"{list(_DIALECTS)}."
            )
        name = _DIALECT_ALIASES.get(name, name)
        if name not in _DIALECTS:
            raise ValueError(
                f"dialect {name!r} is not supported yet; supported: {list(_DIALECTS)}."
            )
        self.dialect: str = name
        self._table = table
        self._query = query and _drop_final_semicolon(query, name)

    def __repr__(self) -> str:
        what = f"table={self._table!r}" if self._table else "query=..."
        return f"SQLSource({self.dialect}, {what})"

    # -- execution -----------------------------------------------------------

    def _sql(self, select: exp.Select) -> str:
        if self._table is not None:
            return select.from_(exp.to_table(self._table, dialect=self.dialect)).sql(
                dialect=self.dialect
            )
        body = select.from_(exp.to_identifier(_SRC, quoted=True)).sql(
            dialect=self.dialect
        )
        # The query is the user's own SQL: it goes in as written, never
        # re-rendered. The newline ends a trailing ``--`` comment.
        return f'WITH "{_SRC}" AS ({self._query}\n) {body}'

    def _open(self) -> Any:
        """A new connection, in autocommit mode if FlexViz owns it.

        DB-API starts a transaction with the first query. On Postgres an idle
        pooled connection would then sit "idle in transaction" and hold its
        table lock for as long as the server runs, which blocks a DROP or
        TRUNCATE (a nightly reload) and holds back vacuum.
        """
        conn = self._connect()
        if hasattr(conn, "dbapi_connection"):
            # A SQLAlchemy connection goes back to the app's pool, so it keeps
            # its mode; ``_run`` ends each query's transaction instead.
            return conn
        try:
            if hasattr(conn, "adbc_connection"):
                conn.adbc_connection.set_autocommit(True)
            elif hasattr(conn, "autocommit"):
                conn.autocommit = True
        except Exception:
            pass
        return conn

    def _run(self, select: exp.Select) -> pl.DataFrame:
        sql = self._sql(select)
        with self._slots:
            try:
                conn = self._idle.get_nowait()
            except queue.Empty:
                return self._run_on(self._open(), sql)
            try:
                return self._run_on(conn, sql)
            except Exception:
                # An idle connection can have been dropped (a database restart,
                # an idle timeout): run the query once more, on a new one.
                # ponytail: any error retries, so a query that really fails (a
                # statement timeout) runs twice; test the connection first if
                # that cost matters.
                return self._run_on(self._open(), sql)

    def _run_on(self, conn: Any, sql: str) -> pl.DataFrame:
        try:
            df = _fetch(conn, sql)
            raw = getattr(conn, "dbapi_connection", conn)
            if getattr(raw, "autocommit", True) is False:
                # The driver kept its transaction: end it before idling.
                raw.rollback()
        except BaseException:
            # A failed connection can be broken; the next query opens a new one.
            _close(conn)
            raise
        self._idle.put(conn)
        return df

    @cached_property
    def schema(self) -> pl.Schema:
        select = exp.select("*").limit(0)
        df = self._run(select)
        if all(dtype != pl.Null for dtype in df.schema.values()):
            return df.schema
        # A driver that returns rows has nothing to infer from in an empty
        # result. A Postgres driver still describes each column by type OID.
        if self.dialect != "postgres":
            raise TypeError(
                "The driver returned no column types. Use an Arrow driver (ADBC)."
            )
        with self._slots:
            conn = self._open()
            try:
                cur = conn.cursor()
                cur.execute(self._sql(select))
                desc = cur.description
            finally:
                _close(conn)
        return pl.Schema({d[0]: _PG_OIDS.get(int(d[1]), pl.String()) for d in desc})

    def col(self, name: str) -> exp.Column:
        """A quoted column of this source.

        Column names come from the spec, which the browser sends, so only a
        column of the schema reaches the SQL; every other identifier is a
        FlexViz constant. ClickHouse reads a backslash in a quoted identifier
        as an escape, so such a name cannot be quoted there.
        """
        if name not in self.schema:
            raise ValueError(f"column {name!r} is not in the source")
        if self.dialect == "clickhouse" and "\\" in name:
            raise ValueError(
                f"column {name!r} has a backslash, which ClickHouse cannot quote"
            )
        return exp.column(name, quoted=True)

    # -- the source surface the engine uses ----------------------------------

    def assume_sorted(self, col_name: str) -> None:
        """Order does not matter to a SQL plan, so the promise is not needed."""

    def compile_filter(
        self, predicates: list[SelectionPredicate], schema: pl.Schema | None
    ) -> exp.Expression:
        """One selection's predicates (OR of ANDs) as one SQL condition: the
        twin of ``predicates_to_expr``, with the same typed bounds."""
        frame = SQLFrame(self)
        if not predicates:
            return exp.true()
        disjuncts = [
            _and(*[self._clause(frame, c, schema) for c in p.clauses])
            for p in predicates
        ]
        return exp.or_(*disjuncts) if len(disjuncts) > 1 else disjuncts[0]

    def _clause(
        self, frame: SQLFrame, clause: ClauseFilter, schema: pl.Schema | None
    ) -> exp.Expression:
        c = self.col(clause.column)
        if clause.values is not None:
            typed = _values_to_typed_series(clause.column, clause.values, schema)
            typed = typed.drop_nulls()
            if typed.is_empty():
                return exp.false()
            return c.isin(*[frame.lit(v, typed.dtype) for v in typed.to_list()])
        dtype = _dtype_for_col(schema, clause.column)
        if dtype is not None and dtype.is_temporal():
            lo, hi = clause.range  # type: ignore[misc]
            bounds = (
                _temporal_bound_toward(lo, dtype, up=clause.closed in ("both", "left")),
                _temporal_bound_toward(
                    hi, dtype, up=clause.closed not in ("both", "right")
                ),
            )
        else:
            bounds = _typed_range_bounds(
                clause.column, clause.range, schema, clause.closed
            )
        if bounds is None:
            return exp.true()
        return frame.range_cond(clause.column, *_eval(bounds), closed=clause.closed)

    def physical_minmax(
        self,
        columns: list[str],
        schema: pl.Schema | None = None,
        *,
        filter_exprs: Sequence[exp.Expression] = (),
    ) -> dict[str, tuple[Any, Any]]:
        """``(min, max)`` per column in physical units, in one query.

        NaN is left out, as Polars' min and max leave it out. With ``cache``
        the source keeps the unfiltered result, like ``LFQueryBuilder``.
        """
        sch = schema if schema is not None else self.schema
        memo = self._minmax_memo if self.cache and not filter_exprs else {}
        missing = list(dict.fromkeys(c for c in columns if c not in memo))
        if missing:
            frame = SQLFrame(self, tuple(filter_exprs))
            aggs = []
            for i, c in enumerate(missing):
                dtype = sch.get(c)
                v = frame.num(c)
                if dtype is not None and dtype.is_float():
                    v = exp.case().when(frame.usable(c, dtype), v)
                aggs += [
                    _alias(exp.Min(this=v), f"{_P}min{i}"),
                    _alias(exp.Max(this=v.copy()), f"{_P}max{i}"),
                ]
            row = frame.collect(exp.select(*aggs))
            for i, c in enumerate(missing):
                dtype = sch.get(c)
                pair = []
                for k in ("min", "max"):
                    s = row[f"{_P}{k}{i}"]
                    if dtype is not None and s.dtype != dtype:
                        s = s.cast(dtype, strict=False)
                    if dtype is not None and dtype.is_temporal():
                        s = s.to_physical()
                    pair.append(s.item())
                memo[c] = tuple(pair)
        return {c: memo[c] for c in columns}

    def aggregate(
        self,
        filter_exprs: list[exp.Expression],
        agg_specs: list[AggregationSpec | GroupedAggregationSpec],
    ) -> tuple[pl.DataFrame, dict[str, pl.DataFrame]]:
        """Run every spec's SQL plan, in parallel, over the filtered rows.

        Returns what ``LFQueryBuilder.aggregate`` returns. A spec without a plan
        holds Polars expressions, which a database cannot run.
        """
        frame = SQLFrame(self, tuple(filter_exprs))
        for spec in agg_specs:
            if spec.plan is None:
                raise ValueError(
                    f"trace {spec.uid!r} has no SQL formulation for this source"
                )

        with ThreadPoolExecutor(max_workers=self.max_connections) as pool:
            results = list(pool.map(lambda spec: spec.plan(frame), agg_specs))

        regular_df = pl.DataFrame()
        grouped_dfs: dict[str, pl.DataFrame] = {}
        for spec, df in zip(agg_specs, results):
            if isinstance(spec, GroupedAggregationSpec):
                grouped_dfs[spec.uid] = df
            else:
                regular_df = df if regular_df.is_empty() else regular_df.hstack(df)
        return regular_df, grouped_dfs


# ---------------------------------------------------------------------------
# Plans: the SQL twins of the Polars aggregation cores
# ---------------------------------------------------------------------------


def _cast_like(df: pl.DataFrame, dtypes: dict[str, pl.DataType]) -> pl.DataFrame:
    """Cast the result columns to the dtypes the Polars path produces."""
    return df.with_columns(
        [pl.col(c).cast(t, strict=False) for c, t in dtypes.items() if c in df.columns]
    )


def bucket_extrema(
    frame: SQLFrame,
    x_col: str,
    y_col: str,
    n_buckets: int,
    vp_filter: pl.Expr | None,
    x_range: tuple | None,
    x_domain: tuple | None,
    schema: pl.Schema | None,
    *,
    group_cols: tuple[str, ...] | None = None,
) -> pl.DataFrame:
    """The SQL twin of ``line_buckets._bucket_extrema``: same grid, same columns,
    same signature. ``vp_filter`` is unused: the zoom filter comes from
    ``x_range``.

    One row per non-empty equal-x-width bucket (per group), with the x and y at
    the bucket's y minimum and y maximum. The bucket arithmetic copies the Polars
    plan's, so both put a row in the same bucket.
    """
    from .trace.line_buckets import _ALIAS_PREFIX, _BUCKET, bucket_grid

    sch = schema if schema is not None else frame.source.schema
    x_dtype, y_dtype = sch.get(x_col), sch.get(y_col)
    x_lo, x_hi = bucket_grid(x_col, x_range, x_domain, x_dtype)
    span = x_hi - x_lo
    bsz = (
        span / n_buckets
        if x_dtype is not None and x_dtype.is_float()
        else -(-span // n_buckets)
    )
    # Parenthesized: int_div builds its own node around it.
    offset = exp.paren(frame.phys(x_col, x_dtype) - _num(x_lo), copy=False)
    if isinstance(bsz, float):
        raw = exp.Floor(this=offset * _num(1.0 / bsz))
    else:
        raw = frame.int_div(offset, bsz)
    bucket = exp.Least(
        this=exp.cast(raw, exp.DataType.Type.BIGINT),
        expressions=[_num(n_buckets - 1)],
    )

    conds = [frame.usable(x_col, x_dtype), frame.usable(y_col, y_dtype)]
    if x_range is not None:
        bounds = _typed_range_bounds(x_col, x_range, sch)
        if bounds is not None:
            conds.append(frame.range_cond(x_col, *_eval(bounds)))

    lo_x, lo_y = frame.extreme_by(x_col, y_col, "min")
    hi_x, hi_y = frame.extreme_by(x_col, y_col, "max")
    names = {
        "lo_x": f"{_ALIAS_PREFIX}lo_{x_col}",
        "lo_y": f"{_ALIAS_PREFIX}lo_{y_col}",
        "hi_x": f"{_ALIAS_PREFIX}hi_{x_col}",
        "hi_y": f"{_ALIAS_PREFIX}hi_{y_col}",
    }
    groups = [frame.col(g) for g in (group_cols or ())]
    select = (
        exp.select(
            *groups,
            _alias(bucket, _BUCKET),
            _alias(lo_x, names["lo_x"]),
            _alias(lo_y, names["lo_y"]),
            _alias(hi_x, names["hi_x"]),
            _alias(hi_y, names["hi_y"]),
        )
        .where(*conds)
        .group_by(*[g.copy() for g in groups], exp.column(_BUCKET, quoted=True))
    )
    df = frame.collect(select)

    dtypes: dict[str, pl.DataType] = {_BUCKET: pl.Int64()}
    for g in group_cols or ():
        dtypes[g] = sch[g]
    for k in ("lo_y", "hi_y"):
        dtypes[names[k]] = y_dtype
    for k in ("lo_x", "hi_x"):
        dtypes[names[k]] = x_dtype
    return _cast_like(df, dtypes)


def _bin_index(
    frame: SQLFrame, name: str, lo: float, hi: float, n: int, cast: exp.DataType.Type
) -> exp.Expression:
    """The bin of a row, as ``_fixed_hist_bin_expr`` computes it (which mirrors
    the kernel), in the same order:
    ``min(max(floor((v - lo) * (n / (hi - lo)) + eps), 0), n - 1)``.
    """
    from .cube import _FIXED_HIST_ROUND_EPS

    scale = n / (hi - lo) if hi > lo else 0.0
    v = exp.cast(
        frame.phys(name, frame.source.schema.get(name)), exp.DataType.Type.DOUBLE
    )
    raw = exp.Floor(this=(v - _num(lo)) * _num(scale) + _num(_FIXED_HIST_ROUND_EPS))
    return exp.cast(
        exp.Least(
            this=exp.Greatest(this=raw, expressions=[_num(0)]),
            expressions=[_num(n - 1)],
        ),
        cast,
    )


def hist_counts(
    frame: SQLFrame,
    col_name: str,
    lo: float,
    hi: float,
    bins: int,
    zoomed: bool,
    group_cols: tuple[str, ...] = (),
) -> pl.DataFrame:
    """Rows per bin (per group): ``group_cols``, ``__fv_b`` and ``__fv_count``.

    The bin index is ``_bin_index``.
    """
    dtype = frame.source.schema.get(col_name)
    b = _bin_index(frame, col_name, lo, hi, bins, exp.DataType.Type.INT)
    conds = [frame.usable(col_name, dtype)]
    if zoomed:
        conds.append(frame.phys_range_cond(col_name, lo, hi))
    groups = [frame.col(g) for g in group_cols]
    select = (
        exp.select(
            *groups,
            _alias(b, _P + "b"),
            _alias(exp.Count(this=exp.Star()), _P + "count"),
        )
        .where(*conds)
        .group_by(*[g.copy() for g in groups], exp.column(_P + "b", quoted=True))
    )
    df = frame.collect(select)
    dtypes: dict[str, pl.DataType] = {_P + "b": pl.Int32(), _P + "count": pl.UInt32()}
    for g in group_cols:
        dtypes[g] = frame.source.schema[g]
    return _cast_like(df, dtypes)


def hist1d_plan(
    col_name: str, lo: float, hi: float, bins: int, uid: str, zoomed: bool
) -> Callable[[SQLFrame], pl.DataFrame]:
    """The ungrouped histogram: the frame ``hist1d_fold_plan`` returns."""

    def run(frame: SQLFrame) -> pl.DataFrame:
        counted = hist_counts(frame, col_name, lo, hi, bins, zoomed)
        acc = np.zeros(bins, dtype=np.int64)
        acc[counted[_P + "b"].to_numpy()] = counted[_P + "count"].to_numpy()
        return pl.select(
            pl.struct(pl.Series("count", acc, dtype=pl.UInt32)).implode().alias(uid)
        )

    return run


def grouped_hist_plan(
    col_name: str,
    lo: float,
    hi: float,
    bins: int,
    uid: str,
    group_cols: tuple[str, ...],
    zoomed: bool,
) -> Callable[[SQLFrame], pl.DataFrame]:
    """The grouped histogram: the frame ``_streaming_hist_plan`` returns."""
    from .trace.hist import _dense_hist_groups

    def run(frame: SQLFrame) -> pl.DataFrame:
        counted = hist_counts(frame, col_name, lo, hi, bins, zoomed, group_cols)
        return _dense_hist_groups(counted, list(group_cols), bins, uid)

    return run


def hist2d_plan(
    x_col: str,
    y_col: str,
    z_col: str | None,
    nb_x: int,
    nb_y: int,
    histfunc: str | None,
    edges: tuple[float, float, float, float],
    zoomed: tuple[bool, bool],
    uid: str,
) -> Callable[[SQLFrame], pl.DataFrame]:
    """The 2-D grid: the frame ``hist2d_fold_plan`` returns.

    Count, min and max are exact. Sum and mean add in the database's order.
    """
    from .trace.batch_fold import _fold_result_frame

    def run(frame: SQLFrame) -> pl.DataFrame:
        sch = frame.source.schema
        conds: list[exp.Expression] = []
        bins: list[exp.Expression] = []
        for name, lo, hi, n, z in (
            (x_col, edges[0], edges[1], nb_x, zoomed[0]),
            (y_col, edges[2], edges[3], nb_y, zoomed[1]),
        ):
            bins.append(_bin_index(frame, name, lo, hi, n, exp.DataType.Type.BIGINT))
            conds.append(frame.usable(name, sch.get(name)))
            if z:
                conds.append(frame.phys_range_cond(name, lo, hi))
        cell = bins[1] * _num(nb_x) + bins[0]
        if z_col is None:
            value = exp.Count(this=exp.Star())
        else:
            conds.append(frame.usable(z_col, sch.get(z_col)))
            z = exp.cast(frame.num(z_col), exp.DataType.Type.DOUBLE)
            value = {"sum": exp.Sum, "mean": exp.Avg, "min": exp.Min, "max": exp.Max}[
                histfunc
            ](this=z)
        select = (
            exp.select(_alias(cell, _P + "c"), _alias(value, _P + "v"))
            .where(*conds)
            .group_by(exp.column(_P + "c", quoted=True))
        )
        df = frame.collect(select)
        idx = df[_P + "c"].cast(pl.Int64).to_numpy()
        if z_col is None:
            acc = np.zeros(nb_x * nb_y, dtype=np.int64)
            acc[idx] = df[_P + "v"].cast(pl.Int64).to_numpy()
            z_flat = pl.Series(acc, dtype=pl.UInt32)
        else:
            acc = np.full(nb_x * nb_y, np.nan)
            acc[idx] = df[_P + "v"].cast(pl.Float64).to_numpy()
            filled = np.zeros(nb_x * nb_y, dtype=bool)
            filled[idx] = True
            # Only the empty cells are null: a cell whose reduction is NaN stays NaN.
            z_flat = pl.Series(acc).set(pl.Series(~filled), None)
        return _fold_result_frame(uid, z_flat, edges)

    return run


def _agg_sql(
    frame: SQLFrame, agg: str | None, values_col: str | None
) -> exp.Expression:
    if values_col is None:
        return exp.Count(this=exp.Star())
    v = frame.num(values_col)
    if agg == "sum":
        # Polars sums an all-null group to 0, SQL to NULL.
        return exp.Coalesce(this=exp.Sum(this=v), expressions=[_num(0)])
    if agg == "mean":
        return exp.Avg(this=exp.cast(v, exp.DataType.Type.DOUBLE))
    if agg in ("min", "max"):
        fn = exp.Min if agg == "min" else exp.Max
        dtype = frame.source.schema.get(values_col)
        if dtype is None or not dtype.is_float():
            return fn(this=v)
        # Polars skips NaN, and returns NaN only when every value is NaN. SQL
        # sorts NaN above every number, so MAX would return it.
        numbers = exp.case().when(exp.Not(this=frame.nan(values_col)), v)
        return exp.Coalesce(this=fn(this=numbers), expressions=[fn(this=v.copy())])
    if agg == "n_unique":
        # Polars counts null as one more value; COUNT(DISTINCT) skips it.
        has_null = exp.Max(
            this=exp.case()
            .when(exp.Is(this=v.copy(), expression=exp.Null()), _num(1))
            .else_(_num(0))
        )
        return exp.Add(
            this=exp.Count(this=exp.Distinct(expressions=[v])), expression=has_null
        )
    if agg == "median":
        if frame.dialect == "clickhouse":
            # Exact and interpolating, like Polars' median; plain ``median``
            # in ClickHouse samples.
            return exp.Anonymous(this="quantileExactInclusive(0.5)", expressions=[v])
        return exp.PercentileCont(this=v, expression=_num(0.5))
    raise ValueError(f"agg {agg!r} is not supported on a {frame.dialect} source")


def group_agg_plan(
    uid: str,
    group_cols: tuple[str, ...],
    sort_cols: tuple[str, ...],
    agg: str | None,
    values_col: str | None,
    polars_agg: pl.Expr,
) -> Callable[[SQLFrame], pl.DataFrame]:
    """A categorical ``GROUP BY`` with one aggregate (bar, pie, treemap).

    Returns the frame the fused grouped query returns: the group columns and
    ``uid``, sorted by ``sort_cols`` with Polars' rules. The result dtypes come
    from running ``polars_agg`` on an empty frame of the source schema, so they
    match the Polars path exactly.
    """

    def run(frame: SQLFrame) -> pl.DataFrame:
        groups = [frame.col(g) for g in group_cols]
        # The uid comes from the spec, so it is renamed here, never quoted.
        select = exp.select(
            *groups, _alias(_agg_sql(frame, agg, values_col), _P + "v")
        ).group_by(*[g.copy() for g in groups])
        df = frame.collect(select).rename({_P + "v": uid})
        sch = frame.source.schema
        used = {c: sch[c] for c in (*group_cols, *([values_col] if values_col else []))}
        expected = (
            pl.LazyFrame(schema=used)
            .group_by(list(group_cols))
            .agg(polars_agg)
            .collect_schema()
        )
        return _cast_like(df, dict(expected)).sort(list(sort_cols))

    return run


def corr_plan(
    uid: str, cols: list[str], absolute: bool
) -> Callable[[SQLFrame], pl.DataFrame]:
    """Pearson correlation of every column pair, packed as ``_corr_expr`` packs
    it: one struct of ``columns`` and the row-major symmetric ``z_flat``."""

    def run(frame: SQLFrame) -> pl.DataFrame:
        sch = frame.source.schema
        n = len(cols)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        dbl = exp.DataType.Type.DOUBLE
        aggs = []
        for i, j in pairs:
            a, b = cols[i], cols[j]
            # CORR skips a pair with a null, as Polars does. Polars gives NaN
            # when a NaN is in the data; DuckDB raises on it, so the NaN rows
            # are left out of CORR and flagged on their own.
            both = exp.and_(frame.usable(a, sch.get(a)), frame.usable(b, sch.get(b)))
            x = exp.case().when(both, exp.cast(frame.num(a), dbl))
            y = exp.case().when(both.copy(), exp.cast(frame.num(b), dbl))
            aggs.append(_alias(exp.Corr(this=x, expression=y), f"{_P}r{i}_{j}"))
            nans = [frame.nan(c) for c in (a, b) if sch.get(c, pl.Null).is_float()]
            if nans:
                # Only a NaN in a complete pair: Polars skips a pair with a null.
                complete = [
                    exp.Not(this=exp.Is(this=frame.col(c), expression=exp.Null()))
                    for c in (a, b)
                ]
                nan_pair = exp.and_(*complete, exp.or_(*nans))
                flag = exp.case().when(nan_pair, _num(1)).else_(_num(0))
                aggs.append(_alias(exp.Max(this=flag), f"{_P}n{i}_{j}"))
        row = frame.collect(exp.select(*aggs)).row(0, named=True)
        mat = np.eye(n)
        for i, j in pairs:
            r = row[f"{_P}r{i}_{j}"]
            # No correlation (no rows, a constant column) is NaN, as in Polars.
            if r is None or row.get(f"{_P}n{i}_{j}") == 1:
                r = float("nan")
            r = float(r)
            mat[i, j] = mat[j, i] = abs(r) if absolute else r
        return pl.DataFrame(
            {"columns": [cols], "z_flat": [mat.ravel().tolist()]}
        ).select(pl.struct("columns", "z_flat").alias(uid))

    return run

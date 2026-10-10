"""SQLSource: a database source gives the deltas a Polars source gives.

The core matrix runs on an in-process DuckDB. A reduced matrix runs on Postgres
when ``FLEXVIZ_TEST_POSTGRES_URI`` is set, on ClickHouse when
``FLEXVIZ_TEST_CLICKHOUSE_URI`` is set
(``clickhouse://user:password@host:port/database``), and on Trino when
``FLEXVIZ_TEST_TRINO_URI`` is set (``trino://user@host:port/catalog/schema``, a
catalog that can create tables, such as ``memory``).
"""

from __future__ import annotations

import datetime as dt
import math
import os
import secrets
import subprocess
import sys
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from urllib.parse import unquote, urlsplit

import duckdb
import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient
from sqlglot import exp

from flexviz import Dashboard
from flexviz.cache import get_cache
from flexviz.cube import decode_cube_bundle
from flexviz.server import app, register_source
from flexviz.sql import SQLFrame, SQLSource, _clickhouse_dtype, _trino_dtype

pytestmark = pytest.mark.integration

# The server registry is global: these names belong to this module only.
POLARS = "_sql_test_polars"
DUCKDB = "_sql_test_duckdb"

PG_ENV = "FLEXVIZ_TEST_POSTGRES_URI"
CH_ENV = "FLEXVIZ_TEST_CLICKHOUSE_URI"
TRINO_ENV = "FLEXVIZ_TEST_TRINO_URI"

N = 12_000
T0 = dt.datetime(2026, 1, 1)


def _frame() -> pl.DataFrame:
    # Continuous random y and strictly increasing x: an exact y plateau or two
    # rows at one x would let a line pick a different, equally valid member or
    # order per source.
    rng = np.random.default_rng(7)
    rid = np.arange(N)
    df = pl.DataFrame(
        {
            # The row order the line x columns are sorted in.
            "rid": rid,
            "t": np.sort(rng.uniform(0, 1000, N)),
            "k": np.cumsum(rng.integers(1, 160, N)),
            # Epoch nanoseconds stored as Int64: past 2**53, no double holds them.
            "big": 1_700_000_000_000_000_000 + np.cumsum(rng.integers(1, 10**8, N)),
            "ts": pl.datetime_range(
                T0, T0 + dt.timedelta(seconds=N - 1), "1s", eager=True
            ),
            "day": [
                dt.date(2026, 1, 1) + dt.timedelta(days=int(d))
                for d in rng.integers(0, 60, N)
            ],
            "v": rng.normal(10, 3, N),
            # Three decimals: a DECIMAL column in ClickHouse.
            "dec": rng.integers(-100_000, 100_000, N) / 1000,
            "w": rng.normal(0, 1, N),
            "vn": rng.normal(0, 1, N),
            "i": rng.integers(-50, 50, N),
            "g": rng.integers(0, 5, N),
            "cat": rng.choice(["a", "b", "c", "d", "o'brien"], N),
            "sub": rng.choice(["x", "y", "z"], N),
            "flag": rng.random(N) < 0.4,
            "lat": rng.uniform(-60, 70, N),
            "lon": rng.uniform(-170, 170, N),
        }
    )
    row = pl.col("rid")
    return df.with_columns(
        pl.when(row % 97 == 0).then(None).otherwise(pl.col("v")).alias("v"),
        pl.when(row % 89 == 0).then(float("nan")).otherwise(pl.col("vn")).alias("vn"),
        pl.when(row % 53 == 0).then(None).otherwise(pl.col("sub")).alias("sub"),
        pl.col("ts").dt.replace_time_zone("UTC").alias("tsz"),
        pl.col("ts").dt.cast_time_unit("ms").alias("ts_ms"),
        pl.col("ts").dt.cast_time_unit("ns").alias("ts_ns"),
    ).with_columns(
        # NaN only where v is null: no complete (v, wn) pair holds a NaN.
        pl.when(pl.col("v").is_null())
        .then(float("nan"))
        .otherwise(pl.col("w"))
        .alias("wn"),
        # Group "d" has no value at all: sum gives 0 and mean null on both sides.
        pl.when(pl.col("cat") == "d").then(None).otherwise(pl.col("v")).alias("vb"),
        pl.when((pl.col("cat") == "d") | (row % 13 == 0))
        .then(None)
        .otherwise(pl.col("i") % 7)
        .alias("ib"),
    )


DF = _frame()


def _duckdb_con(df: pl.DataFrame, table: str = "src") -> duckdb.DuckDBPyConnection:
    # UTC, so a TIMESTAMPTZ column reads back in the frame's own time zone.
    con = duckdb.connect(config={"TimeZone": "UTC"})
    con.register("df_view", df.to_arrow())
    quoted = table.replace('"', '""')
    con.execute(f'CREATE TABLE "{quoted}" AS SELECT * FROM df_view')
    con.unregister("df_view")
    return con


@pytest.fixture(scope="module")
def duck() -> duckdb.DuckDBPyConnection:
    return _duckdb_con(DF)


@pytest.fixture(scope="module")
def client(duck) -> TestClient:
    register_source(POLARS, DF)
    register_source(DUCKDB, SQLSource(duck, table="src"))
    return TestClient(app)


# ---------------------------------------------------------------------------
# Trace cases and events
# ---------------------------------------------------------------------------

LINE_TS_ZOOM = {"x": {"min": "2026-01-01 01:00:00", "max": "2026-01-01 02:30:00"}}
LINE_T_ZOOM = {"x": {"min": 100.0, "max": 300.0}}


def _add(adder: str, /, **kw: Any) -> Callable[[Any], Any]:
    return lambda fig: getattr(fig, adder)(**kw)


# id -> (builder, zoom viewport or None for a trace that does not re-bin)
CASES: dict[str, tuple[Callable[[Any], Any], dict | None]] = {
    "line_ts": (_add("add_line", x="ts", y="v", n_points=400), LINE_TS_ZOOM),
    # Epoch-ms viewport bounds: what Plotly sends for a date axis.
    "line_ts_epoch_zoom": (
        _add("add_line", x="ts", y="w", n_points=400),
        {"x": {"min": 1767229200000, "max": 1767234600000}},
    ),
    "line_tsz": (_add("add_line", x="tsz", y="w", n_points=300), LINE_TS_ZOOM),
    "line_float": (_add("add_line", x="t", y="w", n_points=400), LINE_T_ZOOM),
    "line_int_x": (
        _add("add_line", x="k", y="w", n_points=300),
        {"x": {"min": 200_000.5, "max": 600_000.2}},
    ),
    "line_big_int_x": (_add("add_line", x="big", y="w", n_points=300), None),
    "line_lttb": (
        _add("add_line", x="t", y="w", n_points=200, downsample="lttb"),
        LINE_T_ZOOM,
    ),
    "line_fpcs": (
        _add("add_line", x="t", y="v", n_points=200, downsample="fpcs"),
        LINE_T_ZOOM,
    ),
    "line_ts_lttb": (
        _add("add_line", x="ts", y="w", n_points=200, downsample="lttb"),
        LINE_TS_ZOOM,
    ),
    "line_dec_y": (_add("add_line", x="t", y="dec", n_points=300), LINE_T_ZOOM),
    # Postgres holds no 64-bit integer past 2**53 or timestamp as a double.
    "line_big_y": (_add("add_line", x="t", y="big", n_points=300), LINE_T_ZOOM),
    "line_ts_y": (_add("add_line", x="t", y="ts", n_points=300), LINE_T_ZOOM),
    "line_nan_y": (_add("add_line", x="t", y="vn", n_points=300), LINE_T_ZOOM),
    "line_grouped_int": (
        _add("add_line", x="t", y="w", n_points=300, group_by="g"),
        LINE_T_ZOOM,
    ),
    "line_grouped_str": (
        _add("add_line", x="ts", y="v", n_points=300, group_by="cat"),
        LINE_TS_ZOOM,
    ),
    "line_grouped_null": (
        _add("add_line", x="t", y="w", n_points=300, group_by="sub"),
        LINE_T_ZOOM,
    ),
    "line_grouped_lttb": (
        _add("add_line", x="t", y="w", n_points=200, downsample="lttb", group_by="cat"),
        LINE_T_ZOOM,
    ),
    "line_grouped_fpcs": (
        _add("add_line", x="ts", y="w", n_points=200, downsample="fpcs", group_by="g"),
        LINE_TS_ZOOM,
    ),
    "hist_float": (
        _add("add_histogram", x="v", bins=30),
        {"x": {"min": 5.0, "max": 12.0}},
    ),
    "hist_int": (
        _add("add_histogram", x="i", bins=25),
        {"x": {"min": -20.5, "max": 30.2}},
    ),
    "hist_ts": (
        _add("add_histogram", x="ts", bins=24),
        {"x": {"min": "2026-01-01 00:20:00", "max": "2026-01-01 02:00:00"}},
    ),
    "hist_tsz": (
        _add("add_histogram", x="tsz", bins=24),
        {"x": {"min": "2026-01-01T00:20:00Z", "max": "2026-01-01T02:00:00Z"}},
    ),
    "hist_ts_ms": (
        _add("add_histogram", x="ts_ms", bins=24),
        {"x": {"min": "2026-01-01 00:20:00", "max": "2026-01-01 02:00:00"}},
    ),
    "hist_ts_ns": (
        _add("add_histogram", x="ts_ns", bins=24),
        {"x": {"min": "2026-01-01 00:20:00", "max": "2026-01-01 02:00:00"}},
    ),
    "hist_date": (
        _add("add_histogram", x="day", bins=20),
        {"x": {"min": "2026-01-10", "max": "2026-02-01"}},
    ),
    "hist_nan": (
        _add("add_histogram", x="vn", bins=20),
        {"x": {"min": -1.0, "max": 1.5}},
    ),
    "hist_y": (_add("add_histogram", y="w", bins=20), {"y": {"min": -1.0, "max": 1.5}}),
    "hist_grouped_str": (
        _add("add_histogram", x="w", bins=20, group_by="cat"),
        {"x": {"min": -1.0, "max": 1.5}},
    ),
    "hist_grouped_int": (
        _add("add_histogram", x="v", bins=20, group_by="g"),
        {"x": {"min": 5.0, "max": 12.0}},
    ),
    "hist_grouped_null": (
        _add("add_histogram", x="w", bins=20, group_by="sub"),
        {"x": {"min": -1.0, "max": 1.5}},
    ),
    "bar_count": (_add("add_bar", labels="cat"), None),
    "bar_sum": (_add("add_bar", labels="cat", values="vb", agg="sum"), None),
    "bar_sum_int": (_add("add_bar", labels="cat", values="ib", agg="sum"), None),
    "bar_mean": (_add("add_bar", labels="cat", values="vb", agg="mean"), None),
    # Postgres has no double cast of a Boolean, nor SUM, MIN or MAX of one.
    "bar_sum_bool": (_add("add_bar", labels="cat", values="flag", agg="sum"), None),
    "bar_mean_bool": (_add("add_bar", labels="cat", values="flag", agg="mean"), None),
    "bar_max_bool": (_add("add_bar", labels="cat", values="flag", agg="max"), None),
    "bar_sum_dec": (_add("add_bar", labels="cat", values="dec", agg="sum"), None),
    "bar_median": (_add("add_bar", labels="cat", values="vb", agg="median"), None),
    "bar_min": (_add("add_bar", labels="cat", values="vb", agg="min"), None),
    "bar_max": (_add("add_bar", labels="cat", values="vb", agg="max"), None),
    "bar_n_unique": (_add("add_bar", labels="cat", values="ib", agg="n_unique"), None),
    "bar_n_unique_str": (
        _add("add_bar", labels="g", values="sub", agg="n_unique"),
        None,
    ),
    "bar_null_label": (_add("add_bar", labels="sub", values="w", agg="mean"), None),
    "bar_bool_label": (_add("add_bar", labels="flag"), None),
    "bar_date_label": (_add("add_bar", labels="day"), None),
    "bar_two_labels": (_add("add_bar", labels=["cat", "g"], values="w"), None),
    "bar_grouped": (
        _add("add_bar", labels="cat", values="w", agg="sum", group_by="flag"),
        None,
    ),
    "bar_max_nan": (_add("add_bar", labels="cat", values="vn", agg="max"), None),
    "bar_min_nan": (_add("add_bar", labels="cat", values="vn", agg="min"), None),
    "treemap_max_nan": (
        _add("add_treemap", path=["cat"], values="vn", agg="max"),
        None,
    ),
    "pie_count": (_add("add_pie", labels="cat"), None),
    "pie_mean": (_add("add_pie", labels="sub", values="vb", agg="mean"), None),
    "treemap_mean": (
        _add("add_treemap", path=["cat", "sub"], values="w", agg="mean"),
        None,
    ),
    "treemap_count": (_add("add_treemap", path=["sub", "g"]), None),
    "hist2d_count": (
        _add("add_histogram2d", x="t", y="v", x_bins=20, y_bins=15),
        {"x": {"min": 200.0, "max": 600.0}, "y": {"min": 6.0, "max": 14.0}},
    ),
    "hist2d_ts": (
        _add("add_histogram2d", x="ts", y="w", x_bins=20, y_bins=15),
        {"x": {"min": "2026-01-01 00:20:00", "max": "2026-01-01 02:00:00"}},
    ),
    **{
        f"hist2d_{fn}": (
            _add(
                "add_histogram2d",
                x="t",
                y="w",
                z="vb",
                histfunc=fn,
                x_bins=10,
                y_bins=10,
            ),
            {"x": {"min": 200.0, "max": 600.0}},
        )
        for fn in ("sum", "mean", "min", "max")
    },
    "hist2d_bool_x": (
        _add("add_histogram2d", x="flag", y="w", x_bins=2, y_bins=10),
        None,
    ),
    "hist2d_bool_z": (
        _add(
            "add_histogram2d",
            x="t",
            y="w",
            z="flag",
            histfunc="mean",
            x_bins=8,
            y_bins=8,
        ),
        None,
    ),
    "hist2d_nan_z": (
        _add(
            "add_histogram2d", x="t", y="w", z="vn", histfunc="max", x_bins=8, y_bins=8
        ),
        None,
    ),
    "geo_count": (
        _add("add_geo_histogram2d", lat="lat", lon="lon", lat_bins=20, lon_bins=30),
        {"coordinates": [[-50.0, -10.0], [40.0, -10.0], [40.0, 30.0], [-50.0, 30.0]]},
    ),
    "geo_mean": (
        _add(
            "add_geo_histogram2d",
            lat="lat",
            lon="lon",
            z="w",
            histfunc="mean",
            lat_bins=12,
            lon_bins=16,
        ),
        {"coordinates": [[-50.0, -10.0], [40.0, -10.0], [40.0, 30.0], [-50.0, 30.0]]},
    ),
    "corr": (_add("add_corr_heatmap", columns=["t", "v", "w", "i"]), None),
    "corr_bool": (_add("add_corr_heatmap", columns=["w", "flag"]), None),
    "corr_nan_null": (_add("add_corr_heatmap", columns=["v", "wn"]), None),
    "corr_nan": (_add("add_corr_heatmap", columns=["vn", "w"]), None),
    "corr_abs": (
        _add("add_corr_heatmap", columns=["v", "w", "lat"], absolute=True),
        None,
    ),
}

# A selection from a figure outside the dashboard filters every figure in it.
ELSEWHERE = "elsewhere"
SEL_RANGE = [
    {
        "source_figure_uid": ELSEWHERE,
        "predicates": [{"clauses": [{"column": "v", "range": [8.0, 12.0]}]}],
    }
]
SEL_VALUES = [
    {
        "source_figure_uid": ELSEWHERE,
        "predicates": [{"clauses": [{"column": "cat", "values": ["a", "o'brien"]}]}],
    }
]
SEL_TEMPORAL = [
    {
        "source_figure_uid": ELSEWHERE,
        "predicates": [
            {
                "clauses": [
                    {
                        "column": "ts",
                        "range": ["2026-01-01 00:30:00", "2026-01-01 03:00:00"],
                    }
                ]
            }
        ],
    }
]
# OR of ANDs from one figure, ANDed with a second figure's tz-aware range.
SEL_COMPOUND = [
    {
        "source_figure_uid": ELSEWHERE,
        "predicates": [
            {
                "clauses": [
                    {"column": "i", "range": [-10.5, 20.2]},
                    {"column": "flag", "values": ["true"]},
                ]
            },
            {"clauses": [{"column": "g", "values": [1, 3]}]},
            {
                "clauses": [
                    {
                        "column": "day",
                        "range": ["2026-01-05 12:00:00", "2026-01-20 06:00:00"],
                    }
                ]
            },
        ],
    },
    {
        "source_figure_uid": "elsewhere-2",
        "predicates": [
            {
                "clauses": [
                    {
                        "column": "tsz",
                        "range": ["2026-01-01T00:10:00Z", "2026-01-01T03:30:00+01:00"],
                    }
                ]
            }
        ],
    },
]

# A null never matches a value, so this selects no row at all.
SEL_EMPTY = [
    {
        "source_figure_uid": ELSEWHERE,
        "predicates": [{"clauses": [{"column": "cat", "values": [None]}]}],
    }
]

SELECT = {"type": "selection", "force_update": True}


def _event(name: str, zooms: dict[str, dict | None]) -> tuple[dict, dict]:
    """``(state, event)`` for one named interaction; ``zooms`` maps a figure
    uid to its viewport."""
    viewport = {
        f"{fig_uid}/{axis}": rng
        for fig_uid, zoom in zooms.items()
        for axis, rng in (zoom or {}).items()
    }
    zoomed = {"type": "viewport", "viewport_keys": list(viewport)}
    return {
        "init": ({}, {"type": "init", "force_update": True}),
        "zoom": ({"viewport": viewport}, zoomed),
        "zoom_sel": ({"viewport": viewport, "selections": SEL_RANGE}, zoomed),
        "range_sel": ({"selections": SEL_RANGE}, SELECT),
        "values_sel": ({"selections": SEL_VALUES}, SELECT),
        "temporal_sel": ({"selections": SEL_TEMPORAL}, SELECT),
        "compound_sel": ({"selections": SEL_COMPOUND}, SELECT),
        "empty_sel": ({"selections": SEL_EMPTY}, SELECT),
        "overlay": (
            {"selections": SEL_RANGE, "cross_filter_mode": "overlay"},
            SELECT,
        ),
    }[name]


EVENTS = (
    "init",
    "zoom",
    "zoom_sel",
    "range_sel",
    "values_sel",
    "temporal_sel",
    "compound_sel",
    "empty_sel",
    "overlay",
)
ZOOM_EVENTS = {"zoom", "zoom_sel"}


def _matrix(cases: list[str], events: tuple[str, ...]) -> list[tuple[str, str]]:
    return [
        (case, event)
        for case in cases
        for event in events
        if CASES[case][1] is not None or event not in ZOOM_EVENTS
    ]


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _post(
    client: TestClient,
    dash: Dashboard,
    source: str,
    state: dict,
    event: dict,
    *,
    check: bool = True,
) -> dict[str, list[dict]] | str:
    """The deltas per figure. A failed request raises, or with ``check=False``
    returns its error text."""
    spec = dash.to_spec(source_name=source).model_dump(mode="json")
    spec["state"].update(state)
    r = client.post("/dashboard/update", json={"spec": spec, "event": event})
    if r.status_code != 200:
        error = f"{source}: {r.status_code} {r.text}"
        if check:
            raise AssertionError(error)
        return error
    return {
        fig: sorted(deltas, key=lambda d: (d["uid"], d.get("layer") or ""))
        for fig, deltas in r.json()["figure_deltas"].items()
    }


def _diff(a: Any, b: Any, path: str = "") -> list[str]:
    """Where two JSON values differ; floats compare with a tight tolerance,
    because a database adds a sum or a mean in its own order."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return [f"{path}: keys {sorted(a)} != {sorted(b)}"]
        return [p for k in a for p in _diff(a[k], b[k], f"{path}.{k}")]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: len {len(a)} != {len(b)}"]
        return [
            p for i, (x, y) in enumerate(zip(a, b)) for p in _diff(x, y, f"{path}[{i}]")
        ]
    if (
        isinstance(a, (int, float))
        and isinstance(b, (int, float))
        and not isinstance(a, bool)
        and not isinstance(b, bool)
        and (isinstance(a, float) or isinstance(b, float))
    ):
        if math.isnan(a) and math.isnan(b):
            return []
        ok = math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
        return [] if ok else [f"{path}: {a!r} != {b!r}"]
    return [] if a == b and type(a) is type(b) else [f"{path}: {a!r} != {b!r}"]


def _build(cases: list[str]) -> tuple[Dashboard, dict[str, str]]:
    dash = Dashboard()
    uids = {}
    for case in cases:
        fig = dash.add_figure()
        CASES[case][0](fig)
        uids[case] = fig._uid
    return dash, uids


# Every case as one figure of one dashboard: one request per (source, event)
# serves the whole matrix, and its plans run in parallel on the source's pool.
BATCH, BATCH_UIDS = _build(list(CASES))
_batch_responses: dict[tuple[str, str], dict | str] = {}


def _batch(client: TestClient, source: str, event: str) -> dict | str:
    if (source, event) not in _batch_responses:
        zooms = {BATCH_UIDS[c]: CASES[c][1] for c in CASES}
        _batch_responses[source, event] = _post(
            client, BATCH, source, *_event(event, zooms), check=False
        )
    return _batch_responses[source, event]


def _assert_same(client: TestClient, ref: str, sql: str, case: str, event: str) -> None:
    uid = BATCH_UIDS[case]
    want, got = _batch(client, ref, event), _batch(client, sql, event)
    if isinstance(want, str) or isinstance(got, str):
        # A failed batch hides which trace broke it: run this case alone.
        dash, uids = _build([case])
        uid = uids[case]
        state, ev = _event(event, {uid: CASES[case][1]})
        want = _post(client, dash, ref, state, ev)
        got = _post(client, dash, sql, state, ev)
    assert want[uid], "no delta from the Polars source: nothing to compare"
    problems = _diff(want[uid], got[uid])
    assert not problems, f"{len(problems)} differences:\n" + "\n".join(problems[:15])


# ---------------------------------------------------------------------------
# 1. Equivalence on DuckDB
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("case", "event"), _matrix(list(CASES), EVENTS))
def test_duckdb_matches_polars(client, case, event):
    _assert_same(client, POLARS, DUCKDB, case, event)


def test_fewer_connections_than_plans(client, duck):
    """Two connections serve every plan of one request, in turn."""
    name = "_sql_test_duckdb_2conn"
    register_source(name, SQLSource(duck, table="src", max_connections=2))
    dash, uids = _build(list(CASES))
    state, event = _event("compound_sel", {})
    want = _post(client, dash, POLARS, state, event)
    got = _post(client, dash, name, state, event)
    assert all(want[uid] for uid in uids.values())
    assert not _diff(want, got)


@pytest.mark.parametrize(
    "column", ["with space", 'quo"te', "ünï", "select", "MixedCase"]
)
def test_awkward_column_names(column):
    """Identifiers are quoted: a space, a double quote, a keyword, a non-ASCII
    name and a mixed-case name all work as a column name."""
    df = DF.select(
        pl.col("t"), pl.col("w").alias(column), pl.col("cat").alias(f"{column} cat")
    )
    register_source("_sql_test_awkward_pl", df)
    register_source(
        "_sql_test_awkward_db",
        SQLSource(_duckdb_con(df, 'my "tab"'), table='"my ""tab"""'),
    )
    client = TestClient(app)
    dash = Dashboard()
    dash.add_figure().add_histogram(x=column, bins=10)
    dash.add_figure().add_bar(labels=f"{column} cat", values=column, agg="mean")
    dash.add_figure().add_line(x="t", y=column, n_points=100, group_by=f"{column} cat")
    sel = [
        {
            "source_figure_uid": ELSEWHERE,
            "predicates": [
                {
                    "clauses": [
                        {"column": f"{column} cat", "values": ["o'brien", "b"]},
                        {"column": column, "range": [-1.0, 1.0]},
                    ]
                }
            ],
        }
    ]
    for state in ({}, {"selections": sel}):
        want = _post(client, dash, "_sql_test_awkward_pl", state, SELECT)
        got = _post(client, dash, "_sql_test_awkward_db", state, SELECT)
        assert not _diff(want, got)


# ---------------------------------------------------------------------------
# 2. SQL rendering
# ---------------------------------------------------------------------------


def _random_doubles(n: int) -> list[float]:
    rng = np.random.default_rng(11)
    mags = 10.0 ** rng.uniform(-300, 300, n)
    signs = rng.choice([-1.0, 1.0], n)
    return [
        *map(float, mags * signs),
        *map(float, rng.normal(0, 1, n)),
        *map(float, rng.uniform(-1e15, 1e15, n)),
        0.1,
        1 / 3,
        -2.0 / 3,
        5e-324,
        -5e-324,
        2.2250738585072014e-308,
        1.7976931348623157e308,
        -1.7976931348623157e308,
        0.0,
    ]


def _literal_select(frame: SQLFrame, values: list[Any]) -> exp.Select:
    dbl = exp.DataType.Type.DOUBLE
    return exp.select(
        *[
            exp.alias_(
                exp.cast(frame.lit(v), dbl) if isinstance(v, float) else frame.lit(v),
                f"c{i}",
                quoted=True,
            )
            for i, v in enumerate(values)
        ]
    )


def _same_values(got: tuple, want: list[Any]) -> list[str]:
    bad = []
    for i, (g, w) in enumerate(zip(got, want)):
        if isinstance(w, float) and math.isnan(w):
            ok = isinstance(g, float) and math.isnan(g)
        else:
            ok = g == w and type(g) is type(w)
        if not ok:
            bad.append(f"c{i}: sent {w!r}, read back {g!r}")
    return bad


def test_float_literals_round_trip_exactly(duck):
    """A float literal parses back to the identical double, so a bin edge or a
    filter bound sits exactly where the Polars plan puts it."""
    src = SQLSource(duck, table="src")
    values = _random_doubles(100)
    sql = _literal_select(SQLFrame(src), values).sql(dialect="duckdb")
    got = duck.execute(sql).fetchone()
    assert not _same_values(got, values)


def test_special_and_integer_literals_round_trip(duck):
    src = SQLSource(duck, table="src")
    values = [math.inf, -math.inf, math.nan, 2**62 + 1, -(2**62) - 1, 0, -7]
    sql = _literal_select(SQLFrame(src), values).sql(dialect="duckdb")
    assert not _same_values(duck.execute(sql).fetchone(), values)


@pytest.mark.parametrize("dialect", ["duckdb", "postgres", "clickhouse", "trino"])
def test_negative_float_literal_is_parenthesized(duck, dialect):
    """``a - -1`` printed as ``a --1`` would read as a comment."""
    frame = SQLFrame(SQLSource(duck, table="src", dialect=dialect))
    a = exp.column("a", quoted=True)
    sql = exp.Sub(this=a, expression=frame.lit(-1.5)).sql(dialect=dialect)
    assert sql.startswith('"a" - (-CAST(1.50000000000000000e+00 AS ')


@pytest.mark.parametrize("value", [-1.5, -3, -(2**62), -1e-300])
def test_negative_literal_subtracts(duck, value):
    frame = SQLFrame(SQLSource(duck, table="src"))
    sql = exp.select(exp.Sub(this=exp.Literal.number(2), expression=frame.lit(value)))
    rendered = sql.sql(dialect="duckdb")
    assert "--" not in rendered
    assert duck.execute(rendered).fetchone()[0] == 2 - value


@pytest.mark.parametrize("dialect", ["duckdb", "postgres", "clickhouse", "trino"])
def test_identifiers_are_quoted(dialect):
    con = duckdb.connect()
    con.execute('CREATE TABLE t ("we""ird name" INT, "select" INT)')
    src = SQLSource(con, table="t", dialect=dialect)
    assert src.col('we"ird name').sql(dialect=dialect) == '"we""ird name"'
    assert src.col("select").sql(dialect=dialect) == '"select"'


def test_only_schema_columns_reach_the_sql(duck):
    """Column names come from the spec the browser sends: anything that is not
    a column of the source is refused before a query is built."""
    src = SQLSource(duck, table="src")
    with pytest.raises(ValueError, match="not in the source"):
        src.col('v" FROM src; DROP TABLE src; --')


def test_clickhouse_refuses_a_backslash_identifier():
    """ClickHouse reads a backslash inside a quoted identifier as an escape."""
    con = duckdb.connect()
    con.execute('CREATE TABLE t ("a\\b" INT)')
    src = SQLSource(con, table="t", dialect="clickhouse")
    with pytest.raises(ValueError, match="backslash"):
        src.col("a\\b")


AWKWARD = 'it\'s a \\ back\\slash "dq" -- not a comment; DROP TABLE src'


@pytest.mark.parametrize(
    ("dialect", "rendered"),
    [
        # Postgres and DuckDB read a backslash as itself.
        ("duckdb", "'it''s a \\ back\\slash \"dq\" -- not a comment; DROP TABLE src'"),
        (
            "postgres",
            "'it''s a \\ back\\slash \"dq\" -- not a comment; DROP TABLE src'",
        ),
        # Trino has no backslash escape in a string literal.
        ("trino", "'it''s a \\ back\\slash \"dq\" -- not a comment; DROP TABLE src'"),
        # ClickHouse reads a backslash as an escape.
        (
            "clickhouse",
            "'it''s a \\\\ back\\\\slash \"dq\" -- not a comment; DROP TABLE src'",
        ),
    ],
)
def test_string_literal_rendering(duck, dialect, rendered):
    src = SQLSource(duck, table="src", dialect=dialect)
    frame = SQLFrame(src)
    assert frame.lit(AWKWARD).sql(dialect=dialect) == rendered
    cond = frame.range_cond("cat", AWKWARD, AWKWARD + "z")
    assert rendered in cond.sql(dialect=dialect)


def test_string_literal_round_trips_in_duckdb(duck):
    frame = SQLFrame(SQLSource(duck, table="src"))
    sql = _literal_select(frame, [AWKWARD, "o'brien", "a\\nb", ""]).sql(
        dialect="duckdb"
    )
    assert duck.execute(sql).fetchone() == (AWKWARD, "o'brien", "a\\nb", "")


def test_selection_values_are_data_not_sql(client, duck):
    """A selection value or column name that reads as SQL is matched as data:
    the query runs, matches nothing, and the table is still there."""
    evil = "x'); DROP TABLE src; --"
    dash = Dashboard()
    fig = dash.add_figure()
    fig.add_bar(labels="cat")
    sel = [
        {
            "source_figure_uid": ELSEWHERE,
            "predicates": [{"clauses": [{"column": "cat", "values": [evil, "a"]}]}],
        }
    ]
    want = _post(client, dash, POLARS, {"selections": sel}, SELECT)
    got = _post(client, dash, DUCKDB, {"selections": sel}, SELECT)
    assert not _diff(want, got)
    assert got[fig._uid][0]["updates"]["x"] == ["a"]

    # A column name is an identifier: quoted, so the request fails instead.
    bad_col = [
        {
            "source_figure_uid": ELSEWHERE,
            "predicates": [
                {
                    "clauses": [
                        {
                            "column": "cat\" = 'a' OR 1=1; DROP TABLE src; --",
                            "values": ["a"],
                        }
                    ]
                }
            ],
        }
    ]
    spec = dash.to_spec(source_name=DUCKDB).model_dump(mode="json")
    spec["state"]["selections"] = bad_col
    r = client.post("/dashboard/update", json={"spec": spec, "event": SELECT})
    assert r.status_code >= 400
    assert duck.execute("SELECT count(*) FROM src").fetchone()[0] == N


# ---------------------------------------------------------------------------
# 3. Errors
# ---------------------------------------------------------------------------


UNSUPPORTED = {
    "box": (_add("add_boxplot", y="v"), "box trace cannot run on a SQL source"),
    "geo_line": (
        _add("add_geo_line", lat="lat", lon="lon"),
        "geo_line trace cannot run on a SQL source",
    ),
    "line_nth": (
        _add("add_line", x="t", y="w", downsample="nth"),
        "downsample='nth' cannot run on a SQL source",
    ),
    "corr_spearman": (
        _add("add_corr_heatmap", columns=["v", "w"], method="spearman"),
        "method='spearman' cannot run on a SQL source",
    ),
}


MISSING_COLUMN = {
    "line_missing_column": _add("add_line", x="nope", y="w"),
    "hist_missing_column": _add("add_histogram", x="nope"),
    "bar_missing_group": _add("add_bar", labels="cat", group_by="nope"),
    "treemap_missing_path": _add("add_treemap", path=["cat", "nope"]),
}


@pytest.mark.parametrize("kind", [*UNSUPPORTED, *MISSING_COLUMN])
def test_unsupported_trace_fails_at_add_time(duck, kind):
    """On a SQL source the check runs in the user's code, not as a 500 later."""
    build, message = UNSUPPORTED.get(
        kind, (MISSING_COLUMN.get(kind), r"\['nope'\] not in the SQL source")
    )
    fig = Dashboard(SQLSource(duck, table="src")).add_figure()
    with pytest.raises(ValueError, match=message):
        build(fig)


@pytest.mark.parametrize("kind", list(UNSUPPORTED))
def test_unsupported_trace_fails_the_request(client, caplog, kind):
    """A spec built without the source reaches the engine check: the request
    fails, and the reason is in the server log."""
    build, message = UNSUPPORTED[kind]
    dash = Dashboard()
    build(dash.add_figure())
    spec = dash.to_spec(source_name=DUCKDB).model_dump(mode="json")
    r = client.post(
        "/dashboard/update",
        json={"spec": spec, "event": {"type": "init", "force_update": True}},
    )
    assert r.status_code == 500
    assert r.json()["detail"] == "Aggregation failed"
    assert message in caplog.text


def test_sqlsource_export_is_lazy():
    """``import flexviz`` does not import SQLGlot; ``flexviz.SQLSource`` does."""
    code = (
        "import sys, flexviz\n"
        "assert 'sqlglot' not in sys.modules\n"
        "from flexviz import SQLSource\n"
        "import flexviz.sql\n"
        "assert 'sqlglot' in sys.modules\n"
        "assert SQLSource is flexviz.sql.SQLSource is flexviz.SQLSource\n"
        "assert 'SQLSource' in flexviz.__all__\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_unknown_dialect_is_refused(duck):
    with pytest.raises(ValueError, match="dialect 'oracle' is not supported"):
        SQLSource(duck, table="src", dialect="oracle")
    with pytest.raises(ValueError, match="needs dialect="):
        SQLSource(lambda: object(), table="src")


def test_dialect_alias_is_accepted(duck):
    assert (
        SQLSource(lambda: None, table="src", dialect="postgresql").dialect == "postgres"
    )


@pytest.mark.parametrize(
    ("name", "dialect"), [("postgresql", "postgres"), ("clickhousedb", "clickhouse")]
)
def test_sqlalchemy_dialect_names(name, dialect):
    """The dialect of a SQLAlchemy engine comes from its dialect name."""
    engine = SimpleNamespace(
        raw_connection=lambda: None, dialect=SimpleNamespace(name=name)
    )
    assert SQLSource(engine, table="src").dialect == dialect


def test_unknown_uri_scheme_is_refused():
    with pytest.raises(ValueError, match="only postgresql://"):
        SQLSource("mysql://u:p@localhost/db", table="src")


@pytest.mark.parametrize(
    "kwargs", [{}, {"table": "src", "query": "SELECT * FROM src"}], ids=["none", "both"]
)
def test_exactly_one_of_table_or_query(duck, kwargs):
    with pytest.raises(ValueError, match="exactly one of table= or query="):
        SQLSource(duck, **kwargs)


class _Connection:
    """A DB-API connection: psycopg, ADBC and clickhouse-connect ones are not
    callable either."""

    def cursor(self) -> None: ...

    def close(self) -> None: ...


def test_single_dbapi_connection_is_refused():
    with pytest.raises(TypeError, match="pass the function that opens it"):
        SQLSource(_Connection(), table="src", dialect="postgres")


# ---------------------------------------------------------------------------
# 4. Source behaviour
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "end",
    [
        "",
        " ;\n",
        "\n-- a comment",
        "; -- a comment",
        ";\n-- a comment",
        "; /* c */",
        ";; -- c",
    ],
    ids=[
        "plain",
        "semicolon",
        "comment",
        "semicolon-comment",
        "comment-line",
        "block",
        "semicolons",
    ],
)
def test_query_with_a_join(end):
    rng = np.random.default_rng(5)
    facts = pl.DataFrame(
        {
            "id": rng.integers(0, 6, 3000),
            "y": rng.normal(0, 1, 3000),
            "x": np.arange(3000.0),
        }
    )
    dims = pl.DataFrame({"id": range(6), "name": ["a", "b", "c", "d", "e", "o'brien"]})
    con = duckdb.connect()
    for name, frame in (("facts", facts), ("dims", dims)):
        con.register("v", frame.to_arrow())
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM v")
        con.unregister("v")
    joined = facts.join(dims, on="id")
    register_source("_sql_test_join_pl", joined.sort("x"))
    register_source(
        "_sql_test_join_db",
        SQLSource(
            con,
            query="SELECT f.x, f.y, d.name FROM facts f JOIN dims d ON f.id = d.id"
            + end,
        ),
    )
    client = TestClient(app)
    dash = Dashboard()
    dash.add_figure().add_bar(labels="name", values="y", agg="mean")
    dash.add_figure().add_line(x="x", y="y", n_points=100, group_by="name")
    dash.add_figure().add_histogram(x="y", bins=12)
    sel = [
        {
            "source_figure_uid": ELSEWHERE,
            "predicates": [
                {"clauses": [{"column": "name", "values": ["o'brien", "b"]}]}
            ],
        }
    ]
    for state in ({}, {"selections": sel}):
        want = _post(client, dash, "_sql_test_join_pl", state, SELECT)
        got = _post(client, dash, "_sql_test_join_db", state, SELECT)
        assert all(want.values())
        assert not _diff(want, got)


def _count_queries(src: SQLSource, monkeypatch) -> list[str]:
    seen: list[str] = []
    real = SQLSource._run

    def spy(self, select):
        seen.append(self._sql(select.copy()))
        return real(self, select)

    monkeypatch.setattr(SQLSource, "_run", spy)
    return seen


def test_cache_memoizes_column_bounds(duck, monkeypatch):
    src = SQLSource(duck, table="src")
    Dashboard(src, cache=True)
    assert src.cache
    _ = src.schema
    queries = _count_queries(src, monkeypatch)
    first = src.physical_minmax(["v", "ts"])
    assert len(queries) == 1
    assert src.physical_minmax(["ts", "v"]) == first
    assert len(queries) == 1
    # A filtered probe holds for its filter only, so it always runs.
    cond = src.compile_filter([], src.schema)
    src.physical_minmax(["v"], filter_exprs=[cond])
    assert len(queries) == 2
    assert first == {
        "v": (DF["v"].min(), DF["v"].max()),
        "ts": (DF["ts"].to_physical().min(), DF["ts"].to_physical().max()),
    }


def test_no_cache_resolves_bounds_every_time(duck, monkeypatch):
    src = SQLSource(duck, table="src")
    Dashboard(src)
    _ = src.schema
    queries = _count_queries(src, monkeypatch)
    src.physical_minmax(["v"])
    src.physical_minmax(["v"])
    assert len(queries) == 2


def test_minmax_skips_nan_like_polars(duck):
    src = SQLSource(duck, table="src")
    lo, hi = src.physical_minmax(["vn"])["vn"]
    assert (lo, hi) == (DF["vn"].min(), DF["vn"].max())
    assert not math.isnan(hi)


def test_cube_request_on_sql_source_returns_empty_bundle(client):
    get_cache().clear()
    dash = Dashboard()
    src_fig = dash.add_figure()
    src_fig.add_histogram(x="v", bins=16)
    dash.add_figure().add_histogram(x="w", bins=12)
    spec = dash.to_spec(source_name=DUCKDB)
    r = client.post(
        "/dashboard/update",
        json={
            "spec": spec.model_dump(mode="json"),
            "event": {"type": "cube_request", "force_update": False},
            "request_cube": True,
            "active_source": {
                "figure_uid": src_fig._uid,
                "column": "v",
                "trace_uid": spec.figures[0].traces[0].uid,
            },
        },
    )
    assert r.status_code == 200, r.text
    assert decode_cube_bundle(r.content) == ([], {})


# ---------------------------------------------------------------------------
# 5. Real databases (opt-in)
# ---------------------------------------------------------------------------

# DuckDB-only cases: Postgres stores microseconds only.
REAL_CASES = [c for c in CASES if c not in ("hist_ts_ms", "hist_ts_ns")]
REAL_EVENTS = ("init", "zoom", "values_sel", "temporal_sel", "compound_sel", "overlay")
# Real databases do not keep the frame's row order: rid restores it, and the
# reference frame is read back from the database so both sides share dtypes.
_REAL_COLUMNS = [c for c in DF.columns if c not in ("ts_ms", "ts_ns")]


def _real_name(backend: str) -> str:
    return f"_sql_test_{backend}"


def _close_pool(src: SQLSource) -> None:
    while not src._idle.empty():
        src._idle.get_nowait().close()


@pytest.fixture(scope="module")
def postgres_source() -> Any:
    uri = os.environ.get(PG_ENV)
    if not uri:
        pytest.skip(f"set {PG_ENV} to run against Postgres")
    adbc = pytest.importorskip("adbc_driver_postgresql.dbapi")
    table = f"fv_test_{secrets.token_hex(6)}"
    data = DF.select(_REAL_COLUMNS).to_arrow(compat_level=pl.CompatLevel.oldest())
    with adbc.connect(uri) as conn:
        with conn.cursor() as cur:
            cur.adbc_ingest(table, data, mode="create")
        conn.commit()
    src = SQLSource(uri, table=table)
    try:
        ref = src._run(exp.select("*").order_by(exp.column("rid", quoted=True)))
        register_source(_real_name("postgres") + "_ref", ref)
        register_source(_real_name("postgres"), src)
        yield src
    finally:
        # A pooled connection sits in an open transaction and holds a lock that
        # blocks the DROP (see test_postgres_pool_holds_no_table_lock).
        _close_pool(src)
        with adbc.connect(uri) as conn:
            with conn.cursor() as cur:
                cur.execute(f'DROP TABLE IF EXISTS "{table}"')
            conn.commit()


def test_postgres_pool_holds_no_table_lock(postgres_source):
    adbc = pytest.importorskip("adbc_driver_postgresql.dbapi")
    postgres_source.physical_minmax(["w"])
    with adbc.connect(os.environ[PG_ENV]) as conn:
        with conn.cursor() as cur:
            cur.execute("SET lock_timeout = '2s'")
            # In a transaction that rolls back, so the table stays.
            cur.execute(
                f'LOCK TABLE "{postgres_source._table}" IN ACCESS EXCLUSIVE MODE'
            )
        conn.rollback()


def test_postgres_replaces_a_dropped_idle_connection(postgres_source):
    """A database restart or an idle timeout drops a pooled connection: the
    next query runs on a new one instead of failing."""
    adbc = pytest.importorskip("adbc_driver_postgresql.dbapi")
    src = SQLSource(os.environ[PG_ENV], table=postgres_source._table)
    try:
        pid = src._run(exp.select(exp.func("pg_backend_pid")).limit(1)).item()
        with adbc.connect(os.environ[PG_ENV]) as conn, conn.cursor() as cur:
            cur.execute(f"SELECT pg_terminate_backend({int(pid)})")
        assert src.physical_minmax(["w"])["w"] == (DF["w"].min(), DF["w"].max())
    finally:
        _close_pool(src)


def test_postgres_sqlalchemy_engine(postgres_source):
    """A SQLAlchemy engine: psycopg returns no dtypes for an empty result, so
    the schema comes from the Postgres type OIDs."""
    sa = pytest.importorskip("sqlalchemy")
    pytest.importorskip("psycopg")
    uri = os.environ[PG_ENV].replace("postgresql://", "postgresql+psycopg://")
    engine = sa.create_engine(uri)
    src = SQLSource(engine, table=postgres_source._table)
    try:
        assert src.dialect == "postgres"
        assert src.schema == postgres_source.schema
        assert src.schema["tsz"] == pl.Datetime("us", "UTC")
        name = "_sql_test_pg_sqlalchemy"
        register_source(name, src)
        client = TestClient(app)
        cases = ["line_ts", "line_grouped_str", "hist_float", "bar_mean", "hist2d_mean"]
        dash, uids = _build(cases)
        for event in ("init", "compound_sel"):
            state, ev = _event(event, {})
            want = _post(client, dash, _real_name("postgres") + "_ref", state, ev)
            got = _post(client, dash, name, state, ev)
            assert all(want[uid] for uid in uids.values())
            assert not _diff(want, got)
    finally:
        _close_pool(src)
        engine.dispose()


def test_postgres_sqlalchemy_engine_keeps_its_pool_transactional(postgres_source):
    """A connection that goes back to the app's pool is not left in autocommit:
    a transaction that the app rolls back stays rolled back."""
    sa = pytest.importorskip("sqlalchemy")
    pytest.importorskip("psycopg")
    uri = os.environ[PG_ENV].replace("postgresql://", "postgresql+psycopg://")
    engine = sa.create_engine(uri, pool_size=1, max_overflow=1)
    table = f"fv_test_{secrets.token_hex(6)}"
    src = SQLSource(engine, table=postgres_source._table)
    try:
        # psycopg gives no dtypes for an empty result: the schema query runs on
        # a second connection, which goes back to the pool.
        assert src.schema == postgres_source.schema
        with engine.begin() as c:
            c.exec_driver_sql(f'CREATE TABLE "{table}" (n int)')
        with pytest.raises(RuntimeError), engine.begin() as c:
            c.exec_driver_sql(f'INSERT INTO "{table}" VALUES (1)')
            raise RuntimeError
        with engine.connect() as c:
            assert c.exec_driver_sql(f'SELECT count(*) FROM "{table}"').scalar() == 0
        # The source's own idle connection still holds no table lock.
        src.physical_minmax(["w"])
        with engine.begin() as c:
            c.exec_driver_sql("SET LOCAL lock_timeout = '2s'")
            c.exec_driver_sql(
                f'LOCK TABLE "{postgres_source._table}" IN ACCESS EXCLUSIVE MODE'
            )
    finally:
        _close_pool(src)
        with engine.begin() as c:
            c.exec_driver_sql(f'DROP TABLE IF EXISTS "{table}"')
        engine.dispose()


def test_postgres_line_bool_y():
    """A Boolean line y: the matrix cannot hold it, since every bucket has y
    ties there. Here each bucket holds one True and one False, so the x at an
    extremum is unique."""
    uri = os.environ.get(PG_ENV)
    if not uri:
        pytest.skip(f"set {PG_ENV} to run against Postgres")
    adbc = pytest.importorskip("adbc_driver_postgresql.dbapi")
    k = np.arange(200)
    df = pl.DataFrame(
        {
            "x": np.ravel(np.column_stack([2.0 * k, 2.0 * k + 0.001])),
            "y": np.tile([True, False], 200),
        }
    )
    table = f"fv_test_{secrets.token_hex(6)}"
    with adbc.connect(uri) as conn:
        with conn.cursor() as cur:
            cur.adbc_ingest(table, df.to_arrow(), mode="create")
        conn.commit()
    src = SQLSource(uri, table=table)
    try:
        name = "_sql_test_pg_line_bool_y"
        register_source(name + "_ref", df)
        register_source(name, src)
        client = TestClient(app)
        dash = Dashboard()
        dash.add_figure().add_line(x="x", y="y", n_points=400)
        want = _post(client, dash, name + "_ref", {}, SELECT)
        got = _post(client, dash, name, {}, SELECT)
        assert all(want.values())
        assert not _diff(want, got)
    finally:
        _close_pool(src)
        with adbc.connect(uri) as conn:
            with conn.cursor() as cur:
                cur.execute(f'DROP TABLE IF EXISTS "{table}"')
            conn.commit()


# Labels with three decimals, and integers whose bucket width is 3: a numeric
# column compared or bucketed in decimal instead of as a double moves rows.
_PG_NUMERIC_QUERY = """SELECT g::double precision AS t, (g % 10)::int AS run_id,
 ((g % 7) * 12.345)::numeric(10,3) AS lab, g::numeric AS xi,
 (((g * 7919) % 100000) / 1000.0)::numeric(10,3) AS p
FROM generate_series(0, 600) AS g"""


@pytest.mark.parametrize("driver", ["adbc", "psycopg"])
def test_postgres_numeric_reads_as_float64(driver):
    """A numeric column gives the deltas of the same rows as Float64.

    ADBC sends numeric as text, psycopg as Decimal; both read as Float64, and
    the database compares and buckets it as a double, as Polars does.
    """
    uri = os.environ.get(PG_ENV)
    if not uri:
        pytest.skip(f"set {PG_ENV} to run against Postgres")
    if driver == "adbc":
        pytest.importorskip("adbc_driver_postgresql")
        src = SQLSource(uri, query=_PG_NUMERIC_QUERY)
    else:
        psycopg = pytest.importorskip("psycopg")
        src = SQLSource(
            lambda: psycopg.connect(uri), query=_PG_NUMERIC_QUERY, dialect="postgres"
        )
    # Python divides correctly rounded, as Postgres casts numeric to a double.
    g = range(601)
    ref = pl.DataFrame(
        {
            "t": [float(i) for i in g],
            "run_id": pl.Series([i % 10 for i in g], dtype=pl.Int32),
            "lab": [i % 7 * 12345 / 1000 for i in g],
            "xi": [float(i) for i in g],
            "p": [i * 7919 % 100000 / 1000 for i in g],
        }
    )
    try:
        assert src.schema == ref.schema
        name = f"_sql_test_pg_numeric_{driver}"
        register_source(name + "_ref", ref)
        register_source(name, src)
        client = TestClient(app)
        dash = Dashboard()
        dash.add_figure().add_line(x="t", y="p", group_by="run_id", n_points=200)
        dash.add_figure().add_line(x="xi", y="t", n_points=400)
        dash.add_figure().add_line(x="t", y="p", n_points=100, downsample="lttb")
        dash.add_figure().add_bar(labels="lab")
        dash.add_figure().add_bar(labels="run_id", values="p", agg="sum")
        dash.add_figure().add_histogram(x="p", bins=10)
        for clause in (
            {"column": "lab", "values": [24.69, 37.035]},
            {"column": "lab", "range": [12.345, 37.035]},
        ):
            sel = [
                {"source_figure_uid": ELSEWHERE, "predicates": [{"clauses": [clause]}]}
            ]
            for state, ev in (_event("init", {}), ({"selections": sel}, SELECT)):
                want = _post(client, dash, name + "_ref", state, ev)
                got = _post(client, dash, name, state, ev)
                assert all(want.values())
                assert not _diff(want, got)
    finally:
        _close_pool(src)


_CH_TYPES = {
    pl.Float64: "Float64",
    pl.Int64: "Int64",
    pl.String: "String",
    pl.Boolean: "Bool",
    pl.Date: "Date",
}


# Columns whose ClickHouse type is not the default for their dtype: ``ts`` is
# the native 32-bit ``DateTime`` (whole seconds, as the data), ``dec`` a
# DECIMAL, ``day`` a Date in both wrappers, in the order ClickHouse writes them.
_CH_COLUMN_TYPES = {
    "ts": "Nullable(DateTime)",
    "dec": "Nullable(Decimal(10, 3))",
    "day": "LowCardinality(Nullable(Date))",
}


def _ch_type(name: str, dtype: pl.DataType) -> str:
    if name in _CH_COLUMN_TYPES:
        return _CH_COLUMN_TYPES[name]
    if isinstance(dtype, pl.Datetime):
        tz = f", '{dtype.time_zone}'" if dtype.time_zone else ""
        return f"Nullable(DateTime64(6{tz}))"
    return f"Nullable({_CH_TYPES[dtype.base_type()]})"


def _ch_params() -> dict[str, Any]:
    parts = urlsplit(os.environ[CH_ENV])
    return {
        "host": parts.hostname,
        "port": parts.port or 8123,
        "username": unquote(parts.username or "default"),
        "password": unquote(parts.password or ""),
        "database": parts.path.lstrip("/") or "default",
    }


@pytest.fixture(scope="module")
def clickhouse_client() -> Any:
    if not os.environ.get(CH_ENV):
        pytest.skip(f"set {CH_ENV} to run against ClickHouse")
    ch = pytest.importorskip("clickhouse_connect")
    return ch.get_client(**_ch_params())


@pytest.fixture(scope="module")
def clickhouse_table(clickhouse_client) -> Any:
    table = f"fv_test_{secrets.token_hex(6)}"
    df = DF.select(_REAL_COLUMNS)
    cols = ", ".join(f'"{c}" {_ch_type(c, t)}' for c, t in df.schema.items())
    clickhouse_client.command(
        f'CREATE TABLE "{table}" ({cols}) ENGINE = MergeTree ORDER BY tuple()',
        # For ``day``: a query can make such a column without the setting.
        settings={"allow_suspicious_low_cardinality_types": 1},
    )
    try:
        clickhouse_client.insert(table, df.rows(), column_names=df.columns)
        yield table
    finally:
        clickhouse_client.command(f'DROP TABLE IF EXISTS "{table}"')


def _ch_connect() -> Any:
    from clickhouse_connect import dbapi

    # A session zone other than UTC, as a server in another zone would have: a
    # naive datetime literal must still read as UTC.
    return dbapi.connect(**_ch_params(), session_timezone="Europe/Brussels")


@pytest.fixture(scope="module")
def clickhouse_source(clickhouse_table) -> Any:
    src = SQLSource(_ch_connect, table=clickhouse_table, dialect="clickhouse")
    ref = pl.DataFrame(
        _ch_connect_rows(clickhouse_table),
        schema=DF.select(_REAL_COLUMNS).schema,
        orient="row",
    )
    register_source(_real_name("clickhouse") + "_ref", ref)
    register_source(_real_name("clickhouse"), src)
    return src


def _ch_connect_rows(table: str) -> list[tuple]:
    conn = _ch_connect()
    try:
        cur = conn.cursor()
        cur.execute(f'SELECT * FROM "{table}" ORDER BY "rid"')
        return cur.fetchall()
    finally:
        conn.close()


def _real_matrix() -> list[tuple[str, str, str]]:
    # Generated only for a configured database: hundreds of skips say nothing.
    return [
        (backend, case, event)
        for backend, env in (
            ("postgres", PG_ENV),
            ("clickhouse", CH_ENV),
            ("trino", TRINO_ENV),
        )
        if os.environ.get(env)
        for case, event in _matrix(REAL_CASES, REAL_EVENTS)
    ]


@pytest.mark.parametrize(("backend", "case", "event"), _real_matrix())
def test_real_database_matches_polars(request, backend, case, event):
    request.getfixturevalue(f"{backend}_source")
    client = TestClient(app)
    name = _real_name(backend)
    _assert_same(client, name + "_ref", name, case, event)


@pytest.mark.parametrize("backend", ["postgres", "clickhouse", "trino"])
def test_real_database_literals_round_trip(request, backend):
    src = request.getfixturevalue(f"{backend}_source")
    values = [*_random_doubles(30), math.inf, -math.inf, math.nan, 2**62 + 1, -7]
    got = src._run(_literal_select(SQLFrame(src), values).limit(1)).row(0)
    assert not _same_values(got, values)
    strings = [AWKWARD, "o'brien", "a\\nb", ""]
    got = src._run(_literal_select(SQLFrame(src), strings).limit(1)).row(0)
    assert got == tuple(strings)


def test_clickhouse_schema_from_nullable_columns(clickhouse_table):
    """The schema comes from the ClickHouse type names, with their time zones;
    a DECIMAL reads as Float64."""
    src = SQLSource(_ch_connect, table=clickhouse_table, dialect="clickhouse")
    assert src.schema == DF.select(_REAL_COLUMNS).schema


_TRINO_TYPES = {
    pl.Float64: "double",
    pl.Int64: "bigint",
    pl.String: "varchar",
    pl.Boolean: "boolean",
    pl.Date: "date",
}


def _trino_type(dtype: pl.DataType) -> str:
    if isinstance(dtype, pl.Datetime):
        return "timestamp(6)" + (" with time zone" if dtype.time_zone else "")
    return _TRINO_TYPES[dtype.base_type()]


def _trino_value(v: Any) -> str:
    """A Trino literal for one test value. Test data only, never user input."""
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, float):
        if math.isnan(v):
            return "nan()"
        return f"DOUBLE '{v!r}'"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, dt.datetime):
        zone = " UTC" if v.tzinfo else ""
        return (
            f"TIMESTAMP '{v.replace(tzinfo=None).isoformat(' ', 'microseconds')}{zone}'"
        )
    if isinstance(v, dt.date):
        return f"DATE '{v.isoformat()}'"
    return "'" + str(v).replace("'", "''") + "'"


def _trino_connect() -> Any:
    trino = pytest.importorskip("trino")
    parts = urlsplit(os.environ[TRINO_ENV])
    catalog, schema = parts.path.strip("/").split("/")
    return trino.dbapi.connect(
        host=parts.hostname,
        port=parts.port or 8080,
        user=unquote(parts.username or "flexviz"),
        catalog=catalog,
        schema=schema,
    )


def _trino_execute(sql: str) -> None:
    conn = _trino_connect()
    try:
        cur = conn.cursor()
        cur.execute(sql)
        cur.fetchall()
    finally:
        conn.close()


@pytest.fixture(scope="module")
def trino_source() -> Any:
    if not os.environ.get(TRINO_ENV):
        pytest.skip(f"set {TRINO_ENV} to run against Trino")
    pytest.importorskip("trino")
    table = f"fv_test_{secrets.token_hex(6)}"
    df = DF.select(_REAL_COLUMNS)
    cols = ", ".join(f'"{c}" {_trino_type(t)}' for c, t in df.schema.items())
    _trino_execute(f'CREATE TABLE "{table}" ({cols})')
    try:
        for chunk in df.iter_slices(2_000):
            rows = ", ".join(
                "(" + ", ".join(_trino_value(v) for v in row) + ")"
                for row in chunk.iter_rows()
            )
            _trino_execute(f'INSERT INTO "{table}" VALUES {rows}')
        src = SQLSource(_trino_connect, dialect="trino", table=table)
        ref = src._run(exp.select("*").order_by(exp.column("rid", quoted=True)))
        register_source(_real_name("trino") + "_ref", ref)
        register_source(_real_name("trino"), src)
        yield src
    finally:
        _trino_execute(f'DROP TABLE IF EXISTS "{table}"')


def test_trino_schema_matches_the_frame(trino_source):
    """The type names map to the frame's dtypes."""
    assert trino_source.schema == DF.select(_REAL_COLUMNS).schema


def test_trino_quotes_identifiers_and_strings(trino_source):
    """A column name with a double quote and a comment marker, and a string
    value with quotes, a backslash and SQL, reach Trino as data."""
    name = 'we"ird -- name; DROP TABLE x'
    query = (
        f'SELECT "w" AS "{name.replace(chr(34), chr(34) * 2)}", "cat" '
        f'FROM "{trino_source._table}"'
    )
    src = SQLSource(_trino_connect, dialect="trino", query=query)
    frame = SQLFrame(src)
    cond = exp.or_(
        exp.EQ(this=frame.col("cat"), expression=frame.lit(AWKWARD)),
        frame.range_cond(name, -1.0, 1.0),
    )
    got = src._run(exp.select(exp.Count(this=exp.Star())).where(cond)).item()
    want = DF.filter(pl.col("w").is_between(-1.0, 1.0)).height
    assert got == want


def test_trino_millisecond_timestamp(trino_source):
    """A timestamp(3) column reads as milliseconds, and bins and buckets in
    milliseconds: the exact ``date_diff`` path, not the text form."""
    cols = ["rid", "ts_ms", "w"]
    query = (
        'SELECT "rid", CAST("ts" AS timestamp(3)) AS "ts_ms", "w" '
        f'FROM "{trino_source._table}"'
    )
    src = SQLSource(_trino_connect, dialect="trino", query=query)
    assert src.schema == DF.select(cols).schema
    register_source("_sql_test_trino_ms", src)
    register_source("_sql_test_trino_ms_ref", DF.select(cols))
    client = TestClient(app)
    dash = Dashboard()
    hist = dash.add_figure()
    CASES["hist_ts_ms"][0](hist)
    line = dash.add_figure()
    line.add_line(x="ts_ms", y="w", n_points=300)
    for event in ("init", "zoom"):
        state, ev = _event(
            event, {hist._uid: CASES["hist_ts_ms"][1], line._uid: LINE_TS_ZOOM}
        )
        want = _post(client, dash, "_sql_test_trino_ms_ref", state, ev)
        got = _post(client, dash, "_sql_test_trino_ms", state, ev)
        assert want[hist._uid] and want[line._uid]
        assert not _diff(want, got)


@pytest.mark.parametrize(
    "cast",
    [
        "AT TIME ZONE 'UTC' AS timestamp(6)",
        "AT TIME ZONE 'America/New_York' AS timestamp(6) with time zone",
    ],
)
def test_trino_epoch_microseconds_are_exact(trino_source, cast):
    """``phys`` gives the exact epoch microseconds from 1922 to 2200, where a
    plain double rounds wrong from about 2100 on."""
    query = f"""SELECT us, CAST(from_unixtime_nanos(us * 1000) {cast}) AS c
    FROM (SELECT b + n * 7919 + (n * n) % 997 AS us
          FROM UNNEST(sequence(0, 9999)) AS t(n)
          CROSS JOIN UNNEST(ARRAY[BIGINT '-1500000000123457', BIGINT '0',
            BIGINT '1767225600000000', BIGINT '7258118400000000']) AS s(b))"""
    src = SQLSource(_trino_connect, dialect="trino", query=query)
    frame = SQLFrame(src)
    phys = frame.phys("c", src.schema["c"])
    wrong = exp.CountIf(this=exp.NEQ(this=phys, expression=frame.col("us")))
    assert src._run(exp.select(wrong, exp.Count(this=exp.Star()))).row(0) == (0, 40_000)


@pytest.mark.parametrize(
    ("name", "dtype"),
    [
        ("DateTime", pl.Datetime("us")),
        ("Nullable(DateTime('UTC'))", pl.Datetime("us", "UTC")),
        ("DateTime64(3)", pl.Datetime("ms")),
        ("DateTime64(6, 'Europe/Brussels')", pl.Datetime("us", "Europe/Brussels")),
        ("Nullable(DateTime64(9, 'UTC'))", pl.Datetime("ns", "UTC")),
        ("Nullable(Decimal(10, 3))", pl.Float64()),
        ("LowCardinality(Nullable(Int32))", pl.Int32()),
        ("Nullable(LowCardinality(String))", pl.String()),
    ],
)
def test_clickhouse_type_names(name, dtype):
    assert _clickhouse_dtype(name) == dtype


@pytest.mark.parametrize(
    ("name", "dtype"),
    [
        ("integer", pl.Int32()),
        ("decimal(10, 2)", pl.Float64()),
        ("varchar(3)", pl.String()),
        ("char(3)", pl.String()),
        ("timestamp(3)", pl.Datetime("ms")),
        ("timestamp(6)", pl.Datetime("us")),
        ("timestamp(9)", None),
        ("timestamp(7) with time zone", None),
        ("timestamp(6) with time zone", pl.Datetime("us", "UTC")),
        ("uuid", None),
        ("time(3)", None),
        ("array(integer)", None),
    ],
)
def test_trino_type_names(name, dtype):
    assert _trino_dtype(name) == dtype


def test_trino_median_of_large_values():
    """The median of two middle values whose sum overflows a double halves each
    one first; a group of one value is that value."""
    if not os.environ.get(TRINO_ENV):
        pytest.skip(f"set {TRINO_ENV} to run against Trino")
    df = pl.DataFrame(
        {
            "g": ["one", "two", "two", "even", "even", "even", "even"],
            "v": [1e308, 1e308, 1.5e308, 1.0, 2.0, 4.0, 1e300],
        }
    )
    rows = ", ".join(f"('{g}', DOUBLE '{v!r}')" for g, v in df.iter_rows())
    query = f"SELECT * FROM (VALUES {rows}) AS t(g, v)"
    register_source("_sql_test_trino_median_ref", df)
    register_source(
        "_sql_test_trino_median",
        SQLSource(_trino_connect, dialect="trino", query=query),
    )
    client = TestClient(app)
    dash = Dashboard()
    dash.add_figure().add_bar(labels="g", values="v", agg="median")
    state, ev = _event("init", {})
    want = _post(client, dash, "_sql_test_trino_median_ref", state, ev)
    got = _post(client, dash, "_sql_test_trino_median", state, ev)
    assert not _diff(want, got)


@pytest.mark.parametrize(
    "column",
    [
        # Trino compares a uuid with no string, so a selection could not run.
        "uuid '12151fd2-7586-11e9-8f9e-2a86e4085a59'",
        # The driver rounds to microseconds, which the epoch in SQL does not.
        "CAST(\"ts\" + INTERVAL '0.000000501' SECOND AS timestamp(9))",
    ],
    ids=["uuid", "timestamp9"],
)
def test_trino_column_without_dtype_fails_at_add_time(trino_source, column):
    """A column of a type FlexViz cannot read is not in the schema, so a chart
    on it fails when it is added."""
    query = f'SELECT {column} AS u, "w" FROM "{trino_source._table}"'
    src = SQLSource(_trino_connect, dialect="trino", query=query)
    assert list(src.schema) == ["w"]
    fig = Dashboard(src).add_figure()
    with pytest.raises(ValueError, match="of a type FlexViz cannot read"):
        fig.add_histogram(x="u")

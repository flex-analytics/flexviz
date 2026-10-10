"""Tests for streaming sources (``register_stream`` / ``Stream.append``)."""

from __future__ import annotations

import datetime as dt
import threading
import uuid

import polars as pl
import pytest
from fastapi.testclient import TestClient

from flexviz import Dashboard, Figure, register_source, register_stream
from flexviz.adapters.plotly_adapter import PlotlyAdapter
from flexviz.cache import is_source_cacheable
from flexviz.server import _sources, app
from flexviz.spec import AxisRange, SelectionState

T0 = dt.datetime(2026, 1, 1)


def _rows(start: int, n: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "t": [T0 + dt.timedelta(seconds=i) for i in range(start, start + n)],
            "v": [float(i % 7) for i in range(start, start + n)],
        }
    )


def _name() -> str:
    return f"stream_{uuid.uuid4().hex[:8]}"


def _height(name: str) -> int:
    return _sources[name]._ldf.select(pl.len()).collect().item()


def test_append_grows_and_old_builder_keeps_its_rows():
    name = _name()
    stream = register_stream(name, _rows(0, 10), order_by="t")
    old = _sources[name]
    assert stream.version == 0
    stream.append(_rows(10, 5))
    assert stream.version == 1
    assert _height(name) == 15
    assert old._ldf.select(pl.len()).collect().item() == 10


def test_append_accepts_lazy_and_empty_batches():
    stream = register_stream(_name(), _rows(0, 3), order_by="t")
    stream.append(_rows(3, 2).lazy())
    stream.append(_rows(5, 0))
    assert stream.version == 1  # an empty batch adds nothing


@pytest.mark.parametrize(
    "batch",
    [
        _rows(10, 3).reverse(),  # unsorted
        _rows(5, 3),  # starts below the last row
        _rows(10, 2).with_columns(t=pl.lit(None, dtype=pl.Datetime("us"))),
    ],
    ids=["unsorted", "below_max", "null"],
)
def test_append_rejects_out_of_order_rows(batch):
    stream = register_stream(_name(), _rows(0, 10), order_by="t")
    with pytest.raises(ValueError):
        stream.append(batch)
    assert stream.version == 0


def test_append_rejects_a_schema_mismatch():
    stream = register_stream(_name(), _rows(0, 10), order_by="t")
    with pytest.raises(pl.exceptions.PolarsError):
        stream.append(_rows(10, 2).with_columns(pl.col("v").cast(pl.Int64)))
    assert stream.version == 0


def test_register_stream_rejects_unsorted_data():
    with pytest.raises(ValueError):
        register_stream(_name(), _rows(0, 5).reverse(), order_by="t")


@pytest.mark.parametrize(
    ("order_by", "window"),
    [
        ("t", 60),  # a number for a Datetime
        ("d", dt.timedelta(days=1)),  # Date
        ("tm", dt.timedelta(seconds=1)),  # Time
        ("dur", dt.timedelta(seconds=1)),  # Duration
        ("t", dt.timedelta(0)),
        ("i", dt.timedelta(seconds=1)),  # a timedelta for a number
        ("i", "abc"),
        ("i", float("nan")),
        ("i", 0),
        ("i", -1),
        ("i", True),
    ],
)
def test_register_stream_rejects_a_bad_window(order_by, window):
    df = pl.DataFrame(
        {
            "t": [T0],
            "d": [T0.date()],
            "tm": [dt.time(1)],
            "dur": [dt.timedelta(1)],
            "i": [1],
        }
    )
    with pytest.raises(ValueError, match="window"):
        register_stream(_name(), df, order_by=order_by, window=window)


def test_chunk_count_stays_bounded():
    name = _name()
    stream = register_stream(name, _rows(0, 1), order_by="t")
    for i in range(1, 200):
        stream.append(_rows(i, 1))
    assert stream._frame.n_chunks() <= 65
    assert _height(name) == 200


def test_order_by_is_marked_sorted_without_a_check(monkeypatch):
    name = _name()
    stream = register_stream(name, _rows(0, 10), order_by="t")
    stream.append(_rows(10, 10))

    def fail(*args, **kwargs):
        raise AssertionError("no collect expected")

    monkeypatch.setattr(pl.LazyFrame, "collect", fail)
    assert "t" in _sources[name].sorted_cols
    _sources[name].check_line_x("t")  # returns early: already sorted


@pytest.fixture
def shown(monkeypatch):
    """Specs that ``show()`` renders, with no server or browser started."""
    import flexviz.dashboard as dashboard_mod
    import flexviz.figure as figure_mod

    specs = []
    for mod in (figure_mod, dashboard_mod):
        monkeypatch.setattr(mod, "_start_server_thread", lambda *a: None)
        monkeypatch.setattr(
            mod, "_render_dashboard", lambda r, spec, *a, **k: specs.append(spec)
        )
    return specs


def test_a_stream_page_has_live_brush_off(shown):
    name = _name()
    register_stream(name, _rows(0, 10), order_by="t")
    dash = Dashboard(cache=True)
    dash.add_figure().add_line("t", "v")
    with pytest.warns(UserWarning, match="live_brush"):
        dash.show(source_name=name, live_brush="auto")
    Figure(cache=True).add_line("t", "v").show(source_name=name)
    assert [s.client_state.live_brush for s in shown] == ["off", "off"]
    assert not is_source_cacheable(name)


def test_show_with_data_cannot_overwrite_a_stream(shown):
    name = _name()
    register_stream(name, _rows(0, 10), order_by="t")
    dash = Dashboard(_rows(0, 3))
    dash.add_figure().add_line("t", "v")
    for owner in (dash, Figure(_rows(0, 3)).add_line("t", "v")):
        with pytest.raises(ValueError, match="is a stream"):
            owner.show(source_name=name)
    assert _height(name) == 10


def test_figure_without_data_shows_a_stream_by_name(shown):
    name = _name()
    register_stream(name, _rows(0, 10), order_by="t")
    Figure().add_line("t", "v").show(source_name=name)
    assert shown[0].figures[0].source == name


def test_register_source_replaces_a_stream_and_the_old_handle_stops():
    name = _name()
    stream = register_stream(name, _rows(0, 10), order_by="t")
    register_source(name, _rows(0, 3))
    with pytest.raises(RuntimeError, match="replaced"):
        stream.append(_rows(10, 1))
    assert _height(name) == 3


def test_a_rejected_registration_keeps_the_stream():
    name = _name()
    stream = register_stream(name, _rows(0, 10), order_by="t")
    with pytest.raises(ValueError):
        register_source(name, "not a frame")
    with pytest.raises(ValueError):
        register_stream(name, _rows(0, 5).reverse(), order_by="t")
    stream.append(_rows(10, 1))
    assert _height(name) == 11


def test_a_reregistration_during_an_append_wins(monkeypatch):
    """An append that is under way when the name is registered again must not
    put its old data over the new registration."""
    from flexviz import server

    name = _name()
    stream = register_stream(name, _rows(0, 10), order_by="t")
    original_concat = pl.concat
    rival = threading.Thread(
        target=lambda: register_stream(name, _rows(0, 3), order_by="t")
    )

    def concat_then_reregister(*args, **kwargs):
        # The append has passed its checks. Let a rival registration run now;
        # it must wait until the append ends.
        rival.start()
        rival.join(0.2)
        return original_concat(*args, **kwargs)

    monkeypatch.setattr(server.pl, "concat", concat_then_reregister)
    stream.append(_rows(10, 1))
    rival.join()
    assert _height(name) == 3


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


def _line_hist_spec(name: str):
    dash = Dashboard()
    dash.add_figure().add_line("t", "v", n_points=1000)
    dash.add_figure().add_histogram("v", bins=7)
    return dash.to_spec(source_name=name)


def _refresh(client, spec):
    resp = client.post(
        "/dashboard/update",
        json={
            "spec": spec.model_dump(mode="json"),
            "event": {"type": "refresh", "force_update": True},
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["figure_deltas"]


def _line_x(deltas, spec) -> list:
    (delta,) = deltas[spec.figures[0].uid]
    return [x for x in delta["updates"]["x"] if x is not None]


def test_window_comes_from_the_rows_the_request_reads(monkeypatch):
    """An append after the request resolved its source must not move the
    window past the rows that the request aggregates."""
    from flexviz import server

    name = _name()
    stream = register_stream(
        name, _rows(0, 100), order_by="t", window=dt.timedelta(seconds=30)
    )
    original = server.get_source

    def resolve_then_append(source_name):
        builder = original(source_name)
        stream.append(_rows(1000, 10))  # far past the resolved rows
        return builder

    monkeypatch.setattr(server, "get_source", resolve_then_append)
    spec = _line_hist_spec(name)
    with TestClient(app) as client:
        xs = pl.Series(_line_x(_refresh(client, spec), spec)).str.to_datetime()
    assert xs.min() >= T0 + dt.timedelta(seconds=69)
    assert xs.max() == T0 + dt.timedelta(seconds=99)


@pytest.mark.parametrize("window", [None, dt.timedelta(seconds=30)])
def test_a_locked_axis_aggregates_over_its_pinned_range(window):
    """Rows that arrive after a lock must not stretch the grid past it."""
    name = _name()
    stream = register_stream(name, _rows(0, 100), order_by="t", window=window)
    stream.append(_rows(100, 9900))
    spec = _line_hist_spec(name)
    uid = spec.figures[0].uid
    spec.client_state.axis_locks[f"{uid}/x"] = True
    spec.client_state.axis_lock_ranges[f"{uid}/x"] = AxisRange(
        min="2026-01-01 00:00:00", max="2026-01-01 00:01:39"
    )
    with TestClient(app) as client:
        xs = pl.Series(_line_x(_refresh(client, spec), spec)).str.to_datetime()
    assert xs.max() <= T0 + dt.timedelta(seconds=99)
    assert len(xs) == 100  # every locked row, not a 1000-point grid over 10000


def test_a_lock_on_another_column_keeps_its_bins():
    """A pinned histogram range has half-bin padding: as a zoom it would snap
    to an extra, empty bin. Only an x axis over order_by is pinned."""
    name = _name()
    df = pl.DataFrame(
        {"i": list(range(70)), "v": [float(k % 7 + 1) for k in range(70)]}
    )
    register_stream(name, df, order_by="i")
    dash = Dashboard()
    dash.add_figure().add_histogram("v", bins=7)
    spec = dash.to_spec(source_name=name)
    uid = spec.figures[0].uid
    with TestClient(app) as client:
        (before,) = _refresh(client, spec)[uid]
        xs = before["updates"]["x"]
        half = (xs[1] - xs[0]) / 2
        spec.client_state.axis_locks[f"{uid}/x"] = True
        spec.client_state.axis_lock_ranges[f"{uid}/x"] = AxisRange(
            min=xs[0] - half, max=xs[-1] + half
        )
        (after,) = _refresh(client, spec)[uid]
    assert after["updates"]["y"] == before["updates"]["y"] == [10] * 7


def test_the_stream_keeps_its_own_copy_of_the_first_frame():
    name = _name()
    df = _rows(0, 10)
    stream = register_stream(name, df, order_by="t")
    df.extend(_rows(0, 1))  # in place: out of order for the stream
    stream.append(_rows(10, 1))
    assert _sources[name]._ldf.collect()["t"].to_list() == _rows(0, 11)["t"].to_list()


def test_a_zone_aware_window_spans_the_elapsed_time_across_dst():
    name = _name()
    # 2026-03-29 01:00 UTC: Brussels moves from +01:00 to +02:00.
    # The last row is 03:45+02:00; an hour earlier is 01:45+01:00.
    start = dt.datetime(2026, 3, 29, 0, 45, tzinfo=dt.timezone.utc)
    df = pl.DataFrame(
        {
            "t": pl.datetime_range(
                start, start + dt.timedelta(hours=1), "1m", eager=True
            ).dt.convert_time_zone("Europe/Brussels"),
            "v": [float(i) for i in range(61)],
        }
    )
    register_stream(name, df, order_by="t", window=dt.timedelta(hours=1))
    dash = Dashboard()
    dash.add_figure().add_line("t", "v", n_points=1000)
    spec = dash.to_spec(source_name=name)
    with TestClient(app) as client:
        (delta,) = _refresh(client, spec)[spec.figures[0].uid]
    assert [y for y in delta["updates"]["y"] if y is not None] == [
        float(i) for i in range(61)
    ]


@pytest.mark.parametrize("unit", ["ms", "us", "ns"])
@pytest.mark.parametrize("minutes", [60, 30])
def test_a_zone_aware_window_spans_the_repeated_autumn_hour(unit, minutes):
    name = _name()
    # The last row is 02:15+01:00, in the hour that 2026-10-25 repeats; 30
    # minutes earlier is 02:45+02:00, which sorts after it as text.
    start = dt.datetime(2026, 10, 25, 0, 15, tzinfo=dt.timezone.utc)
    df = pl.DataFrame(
        {
            "t": pl.datetime_range(
                start, start + dt.timedelta(hours=1), "1m", time_unit=unit, eager=True
            ).dt.convert_time_zone("Europe/Brussels"),
            "v": [float(i) for i in range(61)],
        }
    )
    register_stream(name, df, order_by="t", window=dt.timedelta(minutes=minutes))
    dash = Dashboard()
    dash.add_figure().add_line("t", "v", n_points=1000)
    spec = dash.to_spec(source_name=name)
    with TestClient(app) as client:
        (delta,) = _refresh(client, spec)[spec.figures[0].uid]
    assert [y for y in delta["updates"]["y"] if y is not None] == [
        float(i) for i in range(60 - minutes, 61)
    ]


def test_a_float_window_on_a_decimal_order_by():
    name = _name()
    df = pl.DataFrame(
        {"i": pl.Series([0, 1, 2]).cast(pl.Decimal(10, 2)), "g": ["a", "a", "a"]}
    )
    register_stream(name, df, order_by="i", window=1.0)
    dash = Dashboard()
    dash.add_figure().add_histogram("i", group_by="g", bins=2)
    spec = dash.to_spec(source_name=name)
    with TestClient(app) as client:
        (delta,) = _refresh(client, spec)[spec.figures[0].uid]
    assert sum(delta["group_results"][0]["updates"]["y"]) == 2


def test_an_integer_window_is_exact_past_2_53():
    name = _name()
    df = pl.DataFrame({"i": [2**53, 2**53 + 1, 2**53 + 2], "v": [0.0, 1.0, 2.0]})
    register_stream(name, df, order_by="i", window=1)
    dash = Dashboard()
    dash.add_figure().add_line("i", "v", n_points=50)
    spec = dash.to_spec(source_name=name)
    with TestClient(app) as client:
        (delta,) = _refresh(client, spec)[spec.figures[0].uid]
    assert [y for y in delta["updates"]["y"] if y is not None] == [1.0, 2.0]


@pytest.mark.parametrize("dtype", [pl.UInt8, pl.UInt64])
def test_an_unsigned_window_longer_than_the_rows(dtype):
    name = _name()
    df = pl.DataFrame({"i": pl.Series([0, 1, 2], dtype=dtype), "v": [0.0, 1.0, 2.0]})
    register_stream(name, df, order_by="i", window=10)
    dash = Dashboard()
    dash.add_figure().add_line("i", "v", n_points=50)
    spec = dash.to_spec(source_name=name)
    with TestClient(app) as client:
        (delta,) = _refresh(client, spec)[spec.figures[0].uid]
    assert [y for y in delta["updates"]["y"] if y is not None] == [0.0, 1.0, 2.0]


def test_a_selection_after_a_lock_rebins_the_background_too():
    """The lock sends no request, so the next request is the first to bin
    over the pinned range: background and foreground must share that grid."""
    name = _name()
    df = pl.DataFrame(
        {"i": list(range(100)), "v": [float(k % 7 + 1) for k in range(100)]}
    )
    register_stream(name, df, order_by="i")
    dash = Dashboard()
    dash.add_figure().add_histogram("i", bins=10)
    dash.add_figure().add_histogram2d("i", "v", x_bins=10, y_bins=7)
    spec = dash.to_spec(source_name=name)
    spec.state.cross_filter_mode = "overlay"
    source, target = (f.uid for f in spec.figures)
    for axis, (lo, hi) in {"x": (-0.5, 99.5), "y": (0.5, 7.5)}.items():
        spec.client_state.axis_locks[f"{target}/{axis}"] = True
        spec.client_state.axis_lock_ranges[f"{target}/{axis}"] = AxisRange(
            min=lo, max=hi
        )
    spec.state.selections = [
        SelectionState.model_validate(
            {
                "source_figure_uid": source,
                "predicates": [{"clauses": [{"column": "i", "range": [0, 20]}]}],
            }
        )
    ]
    with TestClient(app) as client:
        resp = client.post(
            "/dashboard/update",
            json={
                "spec": spec.model_dump(mode="json"),
                "event": {"type": "selection", "force_update": True},
            },
        )
    assert resp.status_code == 200, resp.text
    layers = {d["layer"]: d["updates"] for d in resp.json()["figure_deltas"][target]}
    assert set(layers) == {"bg", "fg"}
    assert layers["bg"]["y"] == layers["fg"]["y"]


def test_a_nanosecond_window_keeps_the_last_row():
    name = _name()
    df = pl.DataFrame(
        {"t": pl.Series([0, 100, 200]).cast(pl.Datetime("ns")), "v": [0.0, 1.0, 2.0]}
    )
    register_stream(name, df, order_by="t", window=dt.timedelta(seconds=1))
    dash = Dashboard()
    dash.add_figure().add_line("t", "v", n_points=1000)
    spec = dash.to_spec(source_name=name)
    with TestClient(app) as client:
        (delta,) = _refresh(client, spec)[spec.figures[0].uid]
    assert [y for y in delta["updates"]["y"] if y is not None][-1] == 2.0


def test_overlay_refresh_sends_a_fresh_background_for_filtered_only_traces():
    """A box shows only its foreground over a cached background, which holds
    the old rows after an append."""
    name = _name()
    df = pl.DataFrame({"i": list(range(100)), "v": [float(i) for i in range(100)]})
    stream = register_stream(name, df, order_by="i")
    dash = Dashboard()
    dash.add_figure().add_histogram("i", bins=10)
    dash.add_figure().add_boxplot("v")
    spec = dash.to_spec(source_name=name)
    spec.state.cross_filter_mode = "overlay"
    spec.state.selections = [
        SelectionState.model_validate(
            {
                "source_figure_uid": spec.figures[0].uid,
                "predicates": [{"clauses": [{"column": "i", "range": [0, 300]}]}],
            }
        )
    ]
    stream.append(
        pl.DataFrame(
            {"i": list(range(100, 200)), "v": [float(i) for i in range(100, 200)]}
        )
    )
    with TestClient(app) as client:
        deltas = _refresh(client, spec)[spec.figures[1].uid]
    assert {d["layer"] for d in deltas} == {"bg", "fg"}


def test_overlay_refresh_returns_both_layers_with_new_rows():
    name = _name()
    stream = register_stream(name, _rows(0, 100), order_by="t")
    spec = _line_hist_spec(name)
    spec.state.cross_filter_mode = "overlay"
    spec.state.selections = [
        SelectionState.model_validate(
            {
                "source_figure_uid": spec.figures[1].uid,
                "predicates": [{"clauses": [{"column": "v", "range": [0.0, 1.0]}]}],
            }
        )
    ]
    stream.append(_rows(100, 100))
    with TestClient(app) as client:
        deltas = _refresh(client, spec)
    layers = {d["layer"]: d for d in deltas[spec.figures[0].uid]}
    assert set(layers) == {"bg", "fg"}
    bg_x = [x for x in layers["bg"]["updates"]["x"] if x is not None]
    fg_y = [y for y in layers["fg"]["updates"]["y"] if y is not None]
    assert len(bg_x) == 200
    assert fg_y and set(fg_y) <= {0.0, 1.0}


def test_each_refresh_reads_one_version():
    name = _name()
    df = pl.DataFrame({"i": [0], "v": [0.0]})
    stream = register_stream(name, df, order_by="i")
    dash = Dashboard()
    dash.add_figure().add_histogram("i", bins=10)
    dash.add_figure().add_histogram("v", bins=10)
    spec = dash.to_spec(source_name=name)
    done = threading.Event()

    def produce():
        k = 1
        while not done.is_set():
            stream.append(pl.DataFrame({"i": [k], "v": [float(k % 5)]}))
            k += 1

    producer = threading.Thread(target=produce)
    producer.start()
    try:
        with TestClient(app) as client:
            for _ in range(50):
                deltas = _refresh(client, spec)
                totals = {
                    sum(d["updates"]["y"]) for f in spec.figures for d in deltas[f.uid]
                }
                assert len(totals) == 1, totals
    finally:
        done.set()
        producer.join()


def test_version_route():
    name = _name()
    stream = register_stream(name, _rows(0, 10), order_by="t")
    with TestClient(app) as client:
        assert client.get(f"/sources/{name}/version").json() == 0
        stream.append(_rows(10, 1))
        assert client.get(f"/sources/{name}/version").json() == 1
        assert client.get("/sources/no_such_stream/version").status_code == 404
        register_stream(f"{name}/a b", _rows(0, 1), order_by="t")
        assert client.get(f"/sources/{name}%2Fa%20b/version").json() == 0


def test_refresh_returns_the_appended_rows():
    name = _name()
    stream = register_stream(name, _rows(0, 100), order_by="t")
    spec = _line_hist_spec(name)
    with TestClient(app) as client:
        _refresh(client, spec)
        stream.append(_rows(100, 50))
        deltas = _refresh(client, spec)
    assert pl.Series(_line_x(deltas, spec)).str.to_datetime().max() == T0 + (
        dt.timedelta(seconds=149)
    )
    (hist,) = deltas[spec.figures[1].uid]
    assert sum(hist["updates"]["y"]) == 150


def test_refresh_keeps_selections():
    name = _name()
    stream = register_stream(name, _rows(0, 100), order_by="t")
    spec = _line_hist_spec(name)
    # Brush the histogram: v in [0, 1]. The line gets those rows only.
    spec.state.selections = [
        SelectionState.model_validate(
            {
                "source_figure_uid": spec.figures[1].uid,
                "predicates": [{"clauses": [{"column": "v", "range": [0.0, 1.0]}]}],
            }
        )
    ]
    stream.append(_rows(100, 100))
    with TestClient(app) as client:
        deltas = _refresh(client, spec)
    (line,) = deltas[spec.figures[0].uid]
    ys = [y for y in line["updates"]["y"] if y is not None]
    assert ys and set(ys) <= {0.0, 1.0}
    assert pl.Series(_line_x(deltas, spec)).str.to_datetime().max() == T0 + (
        dt.timedelta(seconds=197)  # the last row with v in {0, 1}
    )


def test_window_limits_an_unzoomed_line_only():
    name = _name()
    stream = register_stream(
        name, _rows(0, 100), order_by="t", window=dt.timedelta(seconds=30)
    )
    stream.append(_rows(100, 100))
    spec = _line_hist_spec(name)
    with TestClient(app) as client:
        deltas = _refresh(client, spec)
        xs = pl.Series(_line_x(deltas, spec)).str.to_datetime()
        assert xs.min() >= T0 + dt.timedelta(seconds=169)
        assert xs.max() == T0 + dt.timedelta(seconds=199)
        (hist,) = deltas[spec.figures[1].uid]
        assert sum(hist["updates"]["y"]) == 200  # all rows

        # A zoom passes through unchanged.
        spec.state.viewport[f"{spec.figures[0].uid}/x"] = AxisRange(
            min="2026-01-01 00:00:10", max="2026-01-01 00:00:20"
        )
        xs = pl.Series(_line_x(_refresh(client, spec), spec)).str.to_datetime()
    assert xs.min() >= T0 + dt.timedelta(seconds=9)
    assert xs.max() <= T0 + dt.timedelta(seconds=21)


def test_numeric_window():
    name = _name()
    df = pl.DataFrame({"i": list(range(1000)), "v": [float(i) for i in range(1000)]})
    register_stream(name, df, order_by="i", window=100)
    dash = Dashboard()
    dash.add_figure().add_line("i", "v", n_points=50)
    spec = dash.to_spec(source_name=name)
    with TestClient(app) as client:
        xs = _line_x(_refresh(client, spec), spec)
    assert min(xs) >= 899 and max(xs) == 999


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


def test_page_lists_only_streaming_sources():
    name = _name()
    stream = register_stream(name, _rows(0, 10), order_by="t")
    stream.append(_rows(10, 1))
    register_source("static_src_for_stream_test", _rows(0, 10))
    html = PlotlyAdapter()._build_dashboard_html(_line_hist_spec(name))
    assert f'const FV_STREAMING_SOURCES = {{"{name}": 1}};' in html
    static = PlotlyAdapter()._build_dashboard_html(
        _line_hist_spec("static_src_for_stream_test")
    )
    assert "const FV_STREAMING_SOURCES = {};" in static

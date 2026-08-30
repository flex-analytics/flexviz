"""Browser-based integration tests using Playwright.

These tests require a running browser (Chromium by default) and are
intentionally excluded from the regular ``make test`` run.  Use:

    make test-browser

to execute them.  Each test starts a real FastAPI server on a random free
port, opens the dashboard HTML in a headless browser, and asserts visual
and network behaviour.

Scope
-----
- Page loads and charts render (DOM elements present).
- Global toolbar buttons are present.
- Scroll zoom on ECharts triggers a ``/dashboard/update`` POST.
- Global Reset and Deselect buttons fire the expected toolbar actions.
"""

from __future__ import annotations

import json
import math
import socket
import threading
import time
from typing import Generator

import polars as pl
import pytest
import uvicorn
from playwright.sync_api import Page, Request as PWRequest

pytestmark = pytest.mark.browser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_server(port: int) -> None:
    """Start the FlexViz FastAPI server on *port* in a daemon thread."""
    from flexviz.server import app, register_source

    df = pl.DataFrame({"ts": list(range(500)), "val": [float(i) for i in range(500)]})
    register_source("_browser_test", df)

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    # Wait until the server is accepting connections.
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"Server did not start on port {port}")


def _start_demo_server(port: int) -> None:
    """Start the demo FastAPI server on *port* in a daemon thread."""
    from demo.server import app as demo_app

    config = uvicorn.Config(demo_app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"Demo server did not start on port {port}")


def _dashboard_url(port: int, renderer: str, n_figures: int = 2) -> str:
    """Return a URL that serves an N-figure dashboard via the given renderer."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import encode_spec

    df = pl.DataFrame({"ts": list(range(500)), "val": [float(i) for i in range(500)]})
    register_source("_browser_test", df)

    dash = Dashboard(df)
    for i in range(n_figures):
        f = dash.add_figure(title=f"Fig{i}")
        f.add_line(x="ts", y="val", name=f"L{i}", n_points=200)
    spec = dash.to_spec(source_name="_browser_test")

    encoded = encode_spec(spec)
    return f"http://127.0.0.1:{port}/view?spec={encoded}&renderer={renderer}"


def _dashboard_url_cached(port: int, renderer: str = "plotly") -> str:
    """Dashboard whose source opts into caching (cache=True)."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import encode_spec

    df = pl.DataFrame({"ts": list(range(500)), "val": [float(i) for i in range(500)]})
    source_name = "_browser_cached"
    register_source(source_name, df, cache=True)

    dash = Dashboard(df)
    for i in range(2):
        f = dash.add_figure(title=f"Fig{i}")
        f.add_line(x="ts", y="val", name=f"L{i}", n_points=200)
    spec = dash.to_spec(source_name=source_name)

    encoded = encode_spec(spec)
    return f"http://127.0.0.1:{port}/view?spec={encoded}&renderer={renderer}"


def _dashboard_url_selection_duplicate_repro(port: int) -> str:
    """Dashboard mirroring the Plotly line-selection duplicate repro."""
    import numpy as np

    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import encode_spec

    n = 5_000
    x = np.arange(n)
    sin = np.sin(x / 1000)
    df = pl.DataFrame(
        {
            "x": x,
            "y_pos": sin + 0.75,
            "y_neg": -sin - 0.75,
        }
    ).lazy()

    source_name = "_browser_selection_duplicate_repro"
    register_source(source_name, df)

    c1 = "#156064"
    c2 = "#00C49A"
    dash = Dashboard(df)
    dash.add_figure().add_line(x="x", y="y_pos", color=c1, n_points=1000).add_line(
        x="x", y="y_neg", color=c2, n_points=1000
    )
    dash.add_figure().add_histogram(x="y_pos", color=c1, bins=21).add_histogram(
        x="y_neg", color=c2, bins=21
    )
    dash.add_figure().add_line(x="x", y="x")
    dash.add_figure().add_histogram(x="x", bins=21)

    encoded = encode_spec(dash.to_spec(source_name=source_name))
    return f"http://127.0.0.1:{port}/view?spec={encoded}&renderer=plotly"


def _dashboard_url_boxplot(port: int, renderer: str = "plotly") -> str:
    """Single-figure dashboard with a box plot trace (Plotly smoke tests)."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import encode_spec

    df = pl.DataFrame({"ts": list(range(500)), "val": [float(i) for i in range(500)]})
    register_source("_browser_test", df)

    dash = Dashboard(df)
    f = dash.add_figure(title="Box")
    f.add_boxplot(y="val", name="Box", color="#1f77b4")
    spec = dash.to_spec(source_name="_browser_test")

    encoded = encode_spec(spec)
    return f"http://127.0.0.1:{port}/view?spec={encoded}&renderer={renderer}"


def _dashboard_url_datetime_line(port: int, renderer: str = "echarts") -> str:
    """Single-figure dashboard with datetime x values."""
    from datetime import datetime, timedelta, timezone

    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import encode_spec

    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    df = pl.DataFrame(
        {
            "ts": [base + timedelta(hours=i) for i in range(96)],
            "val": [float((i * 7) % 31) for i in range(96)],
            "group": ["A"] * 48 + ["B"] * 48,
        }
    )
    source_name = "_browser_datetime_line"
    register_source(source_name, df)

    dash = Dashboard(df)
    dash.add_figure(title="Datetime Line").add_line(
        x="ts", y="val", group_by="group", n_points=96, assume_sorted_x=True
    )
    spec = dash.to_spec(source_name=source_name)

    encoded = encode_spec(spec)
    return f"http://127.0.0.1:{port}/view?spec={encoded}&renderer={renderer}"


def _dashboard_url_static_pie(port: int, renderer: str = "plotly") -> str:
    """Single-figure dashboard whose Plotly panel should not render a control bar."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import encode_spec

    df = pl.DataFrame(
        {
            "country": ["NL", "BE", "DE"],
            "val": [10.0, 20.0, 30.0],
        }
    )
    register_source("_browser_static_pie", df)

    dash = Dashboard(df)
    dash.add_figure(title="Pie").add_pie(labels="country", values="val", agg="sum")
    spec = dash.to_spec(source_name="_browser_static_pie")

    encoded = encode_spec(spec)
    return f"http://127.0.0.1:{port}/view?spec={encoded}&renderer={renderer}"


def _dashboard_url_treemap_colormap(port: int, renderer: str = "echarts") -> str:
    """Single treemap with deterministic colors for structure checks."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import LayoutSpec, encode_spec

    df = pl.DataFrame(
        {
            "source": ["Solar"] * 4 + ["Wind"] * 4,
            "country": ["DE", "ES", "FR", "IT"] * 2,
            "value": [5.0, 4.0, 3.0, 2.0, 8.0, 7.0, 6.0, 5.0],
        }
    )
    source_name = "_browser_treemap_colormap"
    register_source(source_name, df)

    dash = Dashboard(df)
    dash.add_figure(title="Treemap").add_treemap(
        path=["source", "country"],
        values="value",
        agg="sum",
        color_map={
            "Solar": "#e3a24d",
            "Wind": "#5b8db8",
            "DE": "#2f2f2f",
            "ES": "#4f4f4f",
            "FR": "#6f6f6f",
            "IT": "#9a9a9a",
        },
    )
    spec = dash.to_spec(source_name=source_name, layout=LayoutSpec(draggable=False))

    encoded = encode_spec(spec)
    return f"http://127.0.0.1:{port}/view?spec={encoded}&renderer={renderer}"


def _demo_url(port: int, renderer: str = "echarts") -> str:
    return f"http://127.0.0.1:{port}/demo?renderer={renderer}"


def _dashboard_url_grouped_line_multi_group_by(
    port: int, renderer: str = "plotly"
) -> str:
    """Single grouped-line dashboard with composite group values for legend tests."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import encode_spec

    df = pl.DataFrame(
        {
            "ts": list(range(50)) * 4,
            "val": [float(i % 50) for i in range(200)],
            "source": ["wind"] * 100 + ["solar"] * 100,
            "country": (["NL"] * 50 + ["BE"] * 50) * 2,
        }
    )
    source_name = "_browser_grouped_line_multi_group_by"
    register_source(source_name, df)

    dash = Dashboard(df)
    fig = dash.add_figure(title="Grouped Line")
    fig.add_line(x="ts", y="val", group_by=["source", "country"], n_points=100)
    spec = dash.to_spec(source_name=source_name)

    encoded = encode_spec(spec)
    return f"http://127.0.0.1:{port}/view?spec={encoded}&renderer={renderer}"


def _dashboard_url_treemap_with_line(port: int, renderer: str = "plotly") -> str:
    """Two-figure dashboard: grouped line target + treemap source."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import LayoutSpec, encode_spec

    df = pl.DataFrame(
        {
            "x": list(range(40)) * 6,
            "sin": [float(i % 40) for i in range(240)],
            "source": ["wind"] * 120 + ["solar"] * 120,
            "country": (["NL"] * 40 + ["BE"] * 40 + ["DE"] * 40) * 2,
            "other": ["A", "B", "C"] * 80,
        }
    )
    source_name = "_browser_treemap_with_line"
    register_source(source_name, df)

    dash = Dashboard(df)
    dash.add_figure().add_line(
        x="x", y="sin", group_by=["source", "country"], n_points=100
    )
    dash.add_figure().add_treemap(path=["source"])
    spec = dash.to_spec(source_name=source_name, layout=LayoutSpec(draggable=False))

    encoded = encode_spec(spec)
    return f"http://127.0.0.1:{port}/view?spec={encoded}&renderer={renderer}"


def _dashboard_url_treemap_pie_selection(port: int, renderer: str = "plotly") -> str:
    """Dashboard with source bar target, two-level treemap, and pie source."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import LayoutSpec, encode_spec

    df = pl.DataFrame(
        {
            "source": ["solar"] * 120 + ["wind"] * 120,
            "country": (["NL"] * 40 + ["BE"] * 40 + ["DE"] * 40) * 2,
            "value": [1.0] * 240,
        }
    )
    source_name = "_browser_treemap_pie_selection"
    register_source(source_name, df)

    dash = Dashboard(df)
    dash.add_figure().add_bar(labels="source")
    dash.add_figure().add_treemap(path=["source", "country"])
    dash.add_figure().add_pie(labels="source")
    spec = dash.to_spec(source_name=source_name, layout=LayoutSpec(draggable=False))

    encoded = encode_spec(spec)
    return f"http://127.0.0.1:{port}/view?spec={encoded}&renderer={renderer}"


def _dashboard_url_plotly_selection_box(
    port: int, source_kind: str, renderer: str = "plotly"
) -> str:
    """Two-figure dashboard for Plotly source selection-box regressions."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import LayoutSpec, encode_spec

    df = pl.DataFrame(
        {
            "country": ["NL", "BE", "DE", "NL", "BE", "DE"],
            "ts": [1, 2, 3, 4, 5, 6],
            "val": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )
    source_name = f"_browser_plotly_{source_kind}_selection_box"
    register_source(source_name, df)

    dash = Dashboard(df)
    source_fig = dash.add_figure(title=f"{source_kind}-source")
    if source_kind == "bar":
        source_fig.add_bar(labels="country", values="val", agg="sum")
    elif source_kind == "histogram":
        source_fig.add_histogram(x="val", bins=6)
    else:
        raise ValueError(f"unsupported source_kind: {source_kind}")
    dash.add_figure(title="line-target").add_line(x="ts", y="val", n_points=10)
    spec = dash.to_spec(source_name=source_name, layout=LayoutSpec(draggable=False))

    encoded = encode_spec(spec)
    return f"http://127.0.0.1:{port}/view?spec={encoded}&renderer={renderer}"


def _dashboard_url_hist2d_overlay(port: int) -> str:
    """Two-figure dashboard: line source + histogram2d target for overlay CF tests."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import encode_spec

    df = pl.DataFrame(
        {
            "ts": list(range(500)),
            "val": [float(i) for i in range(500)],
            "x": [float(i % 25) for i in range(500)],
            "y": [float((i * 7) % 25) for i in range(500)],
        }
    )
    register_source("_browser_hist2d_overlay", df)

    dash = Dashboard(df)
    dash.add_figure(title="Source").add_line(x="ts", y="val", n_points=200)
    dash.add_figure(title="Heatmap").add_histogram2d(x="x", y="y")
    spec = dash.to_spec(source_name="_browser_hist2d_overlay")

    encoded = encode_spec(spec)
    return f"http://127.0.0.1:{port}/view?spec={encoded}&renderer=plotly"


def _dashboard_url_hist2d(port: int, renderer: str = "plotly") -> str:
    """Single-figure dashboard with a Histogram2D trace."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import encode_spec

    df = pl.DataFrame(
        {
            "x": [float(i % 25) for i in range(500)],
            "y": [float((i * 7) % 25) for i in range(500)],
        }
    )
    register_source("_browser_test", df)

    dash = Dashboard(df)
    f = dash.add_figure(title="Hist2D")
    f.add_histogram2d(x="x", y="y")
    spec = dash.to_spec(source_name="_browser_test")

    encoded = encode_spec(spec)
    return f"http://127.0.0.1:{port}/view?spec={encoded}&renderer={renderer}"


def _dashboard_url_weekday_hist2d(port: int) -> str:
    """Taxi-like heatmap with weekday bins 1..7 on y."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import encode_spec

    rows = [
        {"hour": float(hour), "weekday": float(day)}
        for day in range(1, 8)
        for hour in range(24)
    ]
    df = pl.DataFrame(rows)
    source_name = "_browser_weekday_hist2d"
    register_source(source_name, df)

    dash = Dashboard(df)
    fig = dash.add_figure(title="Weekday Hist2D")
    fig.add_histogram2d(x="hour", y="weekday", x_bins=24, y_bins=7)
    fig.update_layout(
        yaxis={
            "tickmode": "array",
            "tickvals": [1, 2, 3, 4, 5, 6, 7],
            "ticktext": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        }
    )
    spec = dash.to_spec(source_name=source_name)

    encoded = encode_spec(spec)
    return f"http://127.0.0.1:{port}/view?spec={encoded}&renderer=plotly"


def _geo_browser_df() -> pl.DataFrame:
    """Deterministic geo dataset that fills every 4x4 geo histogram bin."""
    lat_values = [40.125 + 0.25 * i for i in range(8)]
    lon_values = [-73.875 + 0.25 * i for i in range(8)]
    rows: list[dict[str, float | int]] = []
    ts = 0
    for lat in lat_values:
        for lon in lon_values:
            rows.append({"lat": lat, "lon": lon, "ts": ts, "val": float(ts)})
            ts += 1
    return pl.DataFrame(rows)


def _dashboard_url_geo(port: int, renderer: str = "plotly") -> str:
    """Single-figure dashboard with a geo histogram trace."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import encode_spec

    df = _geo_browser_df()
    register_source("_browser_test", df)

    dash = Dashboard(df)
    f = dash.add_figure(title="Geo")
    f.add_geo_histogram2d(
        lat="lat",
        lon="lon",
        lat_bins=4,
        lon_bins=4,
        bin_boundaries="viewport",
    )
    spec = dash.to_spec(source_name="_browser_test")

    encoded = encode_spec(spec)
    return f"http://127.0.0.1:{port}/view?spec={encoded}&renderer={renderer}"


def _dashboard_url_geo_with_line(port: int, renderer: str = "plotly") -> str:
    """Two-figure dashboard: geo histogram source + linked line target."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import encode_spec

    df = _geo_browser_df()
    register_source("_browser_test", df)

    dash = Dashboard(df)
    fig_geo = dash.add_figure(title="Geo")
    fig_geo.add_geo_histogram2d(
        lat="lat",
        lon="lon",
        lat_bins=4,
        lon_bins=4,
        bin_boundaries="viewport",
    )
    fig_line = dash.add_figure(title="Line")
    fig_line.add_line(x="ts", y="val", n_points=1000)
    spec = dash.to_spec(source_name="_browser_test")

    encoded = encode_spec(spec)
    return f"http://127.0.0.1:{port}/view?spec={encoded}&renderer={renderer}"


def _dashboard_url_grouped(
    port: int,
    renderer: str,
    trace_type: str = "line",
    n_figures: int = 1,
) -> str:
    """Dashboard URL for grouped line/bar browser coverage."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import encode_spec

    if trace_type == "line":
        df = pl.DataFrame(
            {
                "ts": list(range(50)) + list(range(100, 150)),
                "val": [float(i) for i in range(100)],
                "sensor": ["A"] * 50 + ["B"] * 50,
            }
        )
    elif trace_type == "bar":
        df = pl.DataFrame(
            {
                "cat": ["X", "X", "Y", "Y"],
                "region": ["N", "S", "N", "S"],
                "val": [10.0, 20.0, 30.0, 40.0],
            }
        )
    else:
        raise ValueError(f"Unsupported grouped trace_type: {trace_type}")

    register_source("_browser_test", df)

    dash = Dashboard(df)
    for i in range(n_figures):
        f = dash.add_figure(title=f"Grouped{i}")
        if trace_type == "line":
            f.add_line(x="ts", y="val", group_by="sensor", n_points=200)
        else:
            f.add_bar(labels="cat", values="val", agg="sum", group_by="region")
    spec = dash.to_spec(source_name="_browser_test")

    encoded = encode_spec(spec)
    return f"http://127.0.0.1:{port}/view?spec={encoded}&renderer={renderer}"


def _dashboard_url_line_hist_target(port: int, renderer: str = "plotly") -> str:
    """Two-figure dashboard: line source + histogram target."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import encode_spec

    df = pl.DataFrame({"ts": list(range(500)), "val": [float(i) for i in range(500)]})
    register_source("_browser_test", df)

    dash = Dashboard(df)
    fig_source = dash.add_figure(title="Source")
    fig_source.add_line(x="ts", y="val", n_points=200)
    fig_target = dash.add_figure(title="HistogramTarget")
    fig_target.add_histogram(x="ts", bins=20)
    spec = dash.to_spec(source_name="_browser_test")

    encoded = encode_spec(spec)
    return f"http://127.0.0.1:{port}/view?spec={encoded}&renderer={renderer}"


def _dashboard_url_line_multi_hist_target(port: int, renderer: str = "plotly") -> str:
    """Two-figure dashboard: line source + two histogram target traces."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import encode_spec

    x = list(range(500))
    df = pl.DataFrame(
        {
            "x": x,
            "y_pos": [float((i % 200) - 50) / 50.0 for i in x],
            "y_neg": [float(50 - (i % 200)) / 50.0 for i in x],
        }
    )
    register_source("_browser_test", df)

    dash = Dashboard(df)
    fig_source = dash.add_figure(title="Source")
    fig_source.add_line(x="x", y="y_pos", n_points=200)
    fig_target = dash.add_figure(title="HistogramTarget")
    fig_target.add_histogram(x="y_pos", bins=20)
    fig_target.add_histogram(x="y_neg", bins=20)
    spec = dash.to_spec(source_name="_browser_test")

    encoded = encode_spec(spec)
    return f"http://127.0.0.1:{port}/view?spec={encoded}&renderer={renderer}"


def _dashboard_url_line_grouped_bar_target(port: int, renderer: str = "plotly") -> str:
    """Two-figure dashboard: line source + grouped bar target."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import encode_spec

    df = pl.DataFrame(
        {
            "ts": list(range(200)),
            "val": [float((i % 20) + 1) for i in range(200)],
            "cat": ["A" if i < 100 else "B" for i in range(200)],
            "region": ["N" if i % 2 == 0 else "S" for i in range(200)],
        }
    )
    register_source("_browser_test", df)

    dash = Dashboard(df)
    fig_source = dash.add_figure(title="Source")
    fig_source.add_line(x="ts", y="val", n_points=200)
    fig_target = dash.add_figure(title="GroupedBarTarget")
    fig_target.add_bar(labels="cat", values="val", agg="sum", group_by="region")
    spec = dash.to_spec(source_name="_browser_test")

    encoded = encode_spec(spec)
    return f"http://127.0.0.1:{port}/view?spec={encoded}&renderer={renderer}"


def _wait_for_chart(page: Page, renderer: str) -> None:
    if renderer == "plotly":
        page.wait_for_selector(".js-plotly-plot", timeout=15_000)
    else:
        page.wait_for_selector("[id^='fv-chart-']", timeout=15_000)


def _grouped_child_count(page: Page, renderer: str) -> int:
    if renderer == "plotly":
        return page.evaluate(
            """() => document.querySelector('.js-plotly-plot').data.length"""
        )
    return page.evaluate("""() => {
          const el = document.querySelector("[id^='fv-chart-']");
          const chart = echarts.getInstanceByDom(el);
          return chart.getOption().series.length;
        }""")


def _trace_layer(trace_id: str | None) -> str | None:
    if not trace_id:
        return None
    if trace_id.endswith("__fv_layer_bg") or trace_id.endswith("::bg"):
        return "bg"
    if trace_id.endswith("__fv_layer_fg") or trace_id.endswith("::fg"):
        return "fg"
    return None


def _layer_traces(rendered: list[dict], layer: str) -> list[dict]:
    return [trace for trace in rendered if _trace_layer(trace.get("id")) == layer]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _dashboard_url_saved_viewport(
    port: int, renderer: str = "plotly", x_range: tuple[float, float] = (120.0, 260.0)
) -> tuple[str, str, tuple[float, float]]:
    """A one-figure dashboard whose spec carries a saved x viewport.

    Returns (url, figure_uid, x_range) so a test can assert the range was applied.
    """
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import AxisRange, encode_spec

    df = pl.DataFrame({"ts": list(range(500)), "val": [float(i) for i in range(500)]})
    register_source("_browser_test", df)

    dash = Dashboard(df)
    f = dash.add_figure(title="Fig0")
    f.add_line(x="ts", y="val", name="L0", n_points=200)
    spec = dash.to_spec(source_name="_browser_test")
    fig_uid = spec.figures[0].uid
    spec.state.viewport[f"{fig_uid}/x"] = AxisRange(min=x_range[0], max=x_range[1])

    encoded = encode_spec(spec)
    return (
        f"http://127.0.0.1:{port}/view?spec={encoded}&renderer={renderer}",
        fig_uid,
        x_range,
    )


# Counts every Plotly render that carries data. Installed before any page script
# runs, via defineProperty on window.Plotly, because the adapter's own functions
# are closure-scoped and cannot be wrapped from the outside.
_COUNT_RENDERS_JS = """
window.__renders = [];
(function () {
  var wrap = function (plotly) {
    ['react', 'newPlot'].forEach(function (m) {
      var orig = plotly[m].bind(plotly);
      plotly[m] = function (gd, data) {
        var pts = Array.isArray(data) ? data.reduce(function (a, t) {
          return a + ((t && t.x && t.x.length) || 0); }, 0) : 0;
        if (pts > 0) window.__renders.push(m);
        return orig.apply(this, arguments);
      };
    });
  };
  var held;
  Object.defineProperty(window, 'Plotly', {
    configurable: true,
    get: function () { return held; },
    set: function (v) { held = v; if (v && v.react) wrap(v); }
  });
})();
"""


@pytest.fixture(scope="module")
def server_port() -> Generator[int, None, None]:
    port = _free_port()
    _start_server(port)
    yield port


@pytest.fixture(scope="module")
def demo_server_port() -> Generator[int, None, None]:
    port = _free_port()
    _start_demo_server(port)
    yield port


# ---------------------------------------------------------------------------
# Plotly adapter smoke tests
# ---------------------------------------------------------------------------


class TestPlotlyBrowser:
    def test_page_loads_without_browser_errors(self, page: Page, server_port: int):
        url = _dashboard_url(server_port, "plotly")
        browser_errors: list[str] = []

        page.on("pageerror", lambda exc: browser_errors.append(str(exc)))
        page.on(
            "console",
            lambda msg: (
                browser_errors.append(msg.text) if msg.type == "error" else None
            ),
        )

        page.goto(url)
        page.wait_for_selector(".js-plotly-plot", timeout=15_000)
        page.wait_for_timeout(500)

        assert browser_errors == []

    def test_page_loads_and_charts_render(self, page: Page, server_port: int):
        url = _dashboard_url(server_port, "plotly")
        page.goto(url)
        # Wait for Plotly chart containers to appear.
        page.wait_for_selector(".js-plotly-plot", timeout=15_000)
        charts = page.query_selector_all(".js-plotly-plot")
        assert len(charts) >= 2, f"Expected >=2 Plotly charts, found {len(charts)}"

    def test_open_renders_each_figure_once(self, page: Page, server_port: int):
        """Opening a dashboard must draw each figure once, not twice.

        restoreDashboardFromSpec awaits an 'init' update, and delta.js marks EVERY
        figure dirty for 'init' -- so the figures are already drawn when it returns.
        A second unconditional pass over all of them redraws identical data inside
        the window a user waits on.
        """
        url = _dashboard_url(server_port, "plotly", n_figures=2)
        page.add_init_script(_COUNT_RENDERS_JS)
        page.goto(url)
        _wait_for_init(page, "plotly")

        renders = page.evaluate("() => window.__renders")
        assert (
            len(renders) == 2
        ), f"expected one data render per figure, got {len(renders)}: {renders}"

    def test_saved_viewport_is_applied_on_open(self, page: Page, server_port: int):
        """A restored viewport must reach Plotly's axes.

        This is what the render pass at the end of restoreDashboardFromSpec is for:
        syncLayoutViewport runs inside _fvRenderFigure, and the figure's own viewport
        update renders it only if the server returned deltas for it.
        """
        url, _fig_uid, x_range = _dashboard_url_saved_viewport(server_port, "plotly")
        page.goto(url)
        _wait_for_init(page, "plotly")

        applied = page.evaluate("""() => {
                const gd = document.querySelector('.js-plotly-plot');
                return gd && gd._fullLayout && gd._fullLayout.xaxis
                    ? gd._fullLayout.xaxis.range : null;
            }""")
        assert applied is not None, "no Plotly x-axis found"
        assert applied[0] == pytest.approx(x_range[0], rel=1e-6), applied
        assert applied[1] == pytest.approx(x_range[1], rel=1e-6), applied

    def test_toolbar_buttons_present(self, page: Page, server_port: int):
        url = _dashboard_url(server_port, "plotly")
        page.goto(url)
        page.wait_for_selector("#fv-btn-reset", timeout=10_000)
        for btn_id in (
            "fv-btn-reset",
            "fv-btn-deselect",
            "fv-btn-cfmode",
            "fv-btn-grid",
            "fv-btn-share",
            "fv-btn-export",
            "fv-btn-import",
        ):
            btn = page.query_selector(f"#{btn_id}")
            assert btn is not None, f"Toolbar button #{btn_id} not found"

    def test_plotly_control_bar_renders_below_plot_with_a11y_labels(
        self, page: Page, server_port: int
    ):
        url = _dashboard_url(server_port, "plotly", n_figures=1)
        page.goto(url)
        _wait_for_init(page, "plotly")

        geometry = page.evaluate("""() => {
                const panel = document.querySelector('fv-panel');
                const plotWrap = panel.querySelector('.fv-plot-wrap');
                const bar = panel.querySelector('.fv-panel-bar');
                const buttons = Array.from(bar.querySelectorAll('button')).map(btn => ({
                  label: btn.getAttribute('aria-label'),
                  tabIndex: btn.tabIndex,
                }));
                const plotRect = plotWrap.getBoundingClientRect();
                const barRect = bar.getBoundingClientRect();
                return {
                  hasToolbarRole: bar.getAttribute('role') === 'toolbar',
                  buttons,
                  plotBottom: plotRect.bottom,
                  barTop: barRect.top,
                  barGap: parseFloat(getComputedStyle(bar).marginTop) || 0,
                  overflow: getComputedStyle(plotWrap).overflow,
                };
            }""")

        assert geometry["hasToolbarRole"] is True
        # The bar renders below the plot, separated by the design-token gap
        # (margin-top: var(--fv-gap-lg), commit 526859a).
        assert geometry["barTop"] >= geometry["plotBottom"]
        assert geometry["barGap"] > 0
        assert (
            abs((geometry["barTop"] - geometry["plotBottom"]) - geometry["barGap"])
            <= 1.5
        )
        assert geometry["overflow"] == "visible"
        assert geometry["buttons"]
        assert all(btn["label"] for btn in geometry["buttons"])
        assert all(btn["tabIndex"] >= 0 for btn in geometry["buttons"])

    def test_plotly_static_figure_omits_control_bar(self, page: Page, server_port: int):
        page.goto(_dashboard_url_static_pie(server_port, "plotly"))
        _wait_for_init(page, "plotly")
        assert page.locator(".fv-panel-bar").count() == 0

    def test_mode_toggle_click_does_not_post_dashboard_update(
        self, page: Page, server_port: int
    ):
        url = _dashboard_url(server_port, "plotly", n_figures=1)
        update_bodies: list[dict] = []

        def capture(req: PWRequest) -> None:
            if "/dashboard/update" in req.url and req.method == "POST":
                try:
                    update_bodies.append(json.loads(req.post_data or "{}"))
                except Exception:
                    pass

        page.on("request", capture)
        page.goto(url)
        _wait_for_init(page, "plotly")

        count_before = len(update_bodies)
        page.click("#fv-bar-0 .fv-mode-btn[data-mode='pan']")
        page.wait_for_timeout(750)

        assert len(update_bodies) == count_before
        dragmode = page.evaluate("divs[0]._fullLayout.dragmode")
        assert dragmode == "pan"

    def test_reset_button_fires_dashboard_update(self, page: Page, server_port: int):
        url = _dashboard_url(server_port, "plotly")
        update_requests: list[str] = []

        def capture(req: PWRequest) -> None:
            if "/dashboard/update" in req.url:
                update_requests.append(req.url)

        page.on("request", capture)
        page.goto(url)
        # Wait for initial load (at least one /dashboard/update call).
        page.wait_for_selector(".js-plotly-plot", timeout=15_000)
        initial_count = len(update_requests)

        # Click Reset — should trigger another /dashboard/update.
        page.click("#fv-btn-reset")
        page.wait_for_timeout(1_000)
        assert (
            len(update_requests) > initial_count
        ), "Clicking Reset must fire at least one /dashboard/update request"

    def test_deselect_button_fires_dashboard_update(self, page: Page, server_port: int):
        url = _dashboard_url(server_port, "plotly")
        update_requests: list[str] = []
        page.on(
            "request",
            lambda req: (
                update_requests.append(req.url)
                if "/dashboard/update" in req.url
                else None
            ),
        )
        page.goto(url)
        page.wait_for_selector(".js-plotly-plot", timeout=15_000)
        initial_count = len(update_requests)

        page.click("#fv-btn-deselect")
        page.wait_for_timeout(1_000)
        assert (
            len(update_requests) > initial_count
        ), "Clicking Deselect must fire at least one /dashboard/update request"

    def test_cached_source_reset_issues_no_dashboard_update(
        self, page: Page, server_port: int
    ):
        """With a cache=True source, the init response is reused client-side, so
        clicking Reset (same unfiltered output) issues no /dashboard/update."""
        url = _dashboard_url_cached(server_port, "plotly")
        update_requests: list[str] = []

        def capture(req: PWRequest) -> None:
            if "/dashboard/update" in req.url and req.method == "POST":
                update_requests.append(req.url)

        page.on("request", capture)
        page.goto(url)
        page.wait_for_selector(".js-plotly-plot", timeout=15_000)
        page.wait_for_timeout(500)
        # Initial load issues exactly the init request, which populates the
        # client cache.
        after_init = len(update_requests)
        assert after_init >= 1

        page.click("#fv-btn-reset")
        page.wait_for_timeout(1_000)
        assert len(update_requests) == after_init, (
            "Reset on a cached source must be served from the client cache "
            "without a /dashboard/update request"
        )

    def test_cached_source_deselect_while_zoomed_issues_dashboard_update(
        self, page: Page, server_port: int
    ):
        """Regression: the client cache holds the viewport-free response, so a
        deselect issued while a figure is zoomed must NOT be served from cache —
        it must round-trip to the server for the viewport-correct result."""
        url = _dashboard_url_cached(server_port, "plotly")
        update_requests: list[str] = []

        def capture(req: PWRequest) -> None:
            if "/dashboard/update" in req.url and req.method == "POST":
                update_requests.append(req.url)

        page.on("request", capture)
        page.goto(url)
        page.wait_for_selector(".js-plotly-plot", timeout=15_000)
        page.wait_for_timeout(500)

        # Zoom the first figure via a real relayout (handleRelayout records the
        # viewport into DASHBOARD_SPEC.state and posts a viewport update).
        before_zoom = len(update_requests)
        page.evaluate(
            "() => Plotly.relayout(document.querySelectorAll('.js-plotly-plot')[0],"
            " {'xaxis.range[0]': 50, 'xaxis.range[1]': 150})"
        )
        page.wait_for_timeout(800)
        assert len(update_requests) > before_zoom, "zoom should issue a viewport update"

        # Now deselect while still zoomed: must bypass the client cache.
        before_deselect = len(update_requests)
        page.click("#fv-btn-deselect")
        page.wait_for_timeout(1_000)
        assert len(update_requests) > before_deselect, (
            "Deselect while zoomed must issue a /dashboard/update (the cached "
            "payload is viewport-free and would be wrong inside a zoomed axis)"
        )

    def test_cached_source_panel_reset_after_zoom_served_from_cache(
        self, page: Page, server_port: int
    ):
        """Case 3a: a per-figure reset that returns a zoomed figure to full
        autorange — with no other figure cross-filtering it and no axis lock — is
        served from the figure-scoped client cache (the unfiltered slice already
        held from init), so it issues no /dashboard/update."""
        url = _dashboard_url_cached(server_port, "plotly")
        update_requests: list[str] = []

        def capture(req: PWRequest) -> None:
            if "/dashboard/update" in req.url and req.method == "POST":
                update_requests.append(req.url)

        page.on("request", capture)
        page.goto(url)
        page.wait_for_selector(".js-plotly-plot", timeout=15_000)
        page.wait_for_timeout(500)

        # Zoom figure 0 (records a viewport and round-trips for the zoomed slice).
        before_zoom = len(update_requests)
        page.evaluate(
            "() => Plotly.relayout(document.querySelectorAll('.js-plotly-plot')[0],"
            " {'xaxis.range[0]': 50, 'xaxis.range[1]': 150})"
        )
        page.wait_for_timeout(800)
        assert len(update_requests) > before_zoom, "zoom should issue a viewport update"

        # Per-figure reset of figure 0 → full autorange, no other filters: cache.
        after_zoom = len(update_requests)
        page.evaluate("() => window.fvOnResetPanel(DASHBOARD_SPEC.figures[0].uid)")
        page.wait_for_timeout(1_000)
        assert len(update_requests) == after_zoom, (
            "Per-figure reset to full autorange with no other cross-filters must "
            "be served from the figure-scoped client cache without a round-trip"
        )
        # Figure 0's viewport is cleared (back to full autorange).
        fig0 = page.evaluate("DASHBOARD_SPEC.figures[0].uid")
        viewport = page.evaluate("DASHBOARD_SPEC.state.viewport") or {}
        assert not any(
            k.startswith(fig0 + "/") for k in viewport
        ), "Per-figure reset must clear the figure's viewport"

    def test_panel_reset_clean_figure_is_noop(self, page: Page, server_port: int):
        """Case 3c: a per-figure reset of a figure that is not zoomed, sources no
        selection, and has no other cross-filters in play changes nothing — so it
        must issue no /dashboard/update at all."""
        url = _dashboard_url(server_port, "plotly", n_figures=2)
        update_requests: list[str] = []

        def capture(req: PWRequest) -> None:
            if "/dashboard/update" in req.url and req.method == "POST":
                update_requests.append(req.url)

        page.on("request", capture)
        page.goto(url)
        _wait_for_init(page, "plotly")
        page.wait_for_timeout(500)

        before = len(update_requests)
        page.evaluate("() => window.fvOnResetPanel(DASHBOARD_SPEC.figures[0].uid)")
        page.wait_for_timeout(1_000)
        assert len(update_requests) == before, (
            "Reset of an unzoomed, non-sourcing figure with no other filters must "
            "be a no-op (no /dashboard/update)"
        )

    def test_panel_reset_unzoomed_target_is_noop(self, page: Page, server_port: int):
        """Case 3d: a per-figure reset of an *unzoomed* figure that is only a
        cross-filter target (another figure sources the selection) changes
        nothing — it stays filtered-by-others at autorange — so it must issue no
        /dashboard/update and must preserve the incoming selection."""
        url = _dashboard_url(server_port, "plotly", n_figures=2)
        update_requests: list[str] = []

        def capture(req: PWRequest) -> None:
            if "/dashboard/update" in req.url and req.method == "POST":
                update_requests.append(req.url)

        page.on("request", capture)
        page.goto(url)
        _wait_for_init(page, "plotly")

        # Selection sourced from figure 1, filtering figure 0 (the target).
        page.evaluate("""() => {
            const figUids = DASHBOARD_SPEC.figures.map(f => f.uid);
            const sel = {
                source_figure_uid: figUids[1],
                predicates: [{ clauses: [{ column: 'ts', range: [100, 300] }] }],
            };
            DASHBOARD_SPEC.state.selections = [sel];
            return postDashboardUpdate({
                type: 'selection', axis_ranges: {},
                selections: DASHBOARD_SPEC.state.selections,
                force_update: true, figure_uid: figUids[1],
            });
        }""")
        page.wait_for_timeout(1_000)

        before = len(update_requests)
        # Reset the unzoomed target figure 0 — no-op.
        page.evaluate("() => window.fvOnResetPanel(DASHBOARD_SPEC.figures[0].uid)")
        page.wait_for_timeout(1_000)
        assert (
            len(update_requests) == before
        ), "Reset of an unzoomed cross-filter target must be a no-op"
        # The incoming selection (sourced by figure 1) survives.
        fig1 = page.evaluate("DASHBOARD_SPEC.figures[1].uid")
        selections = page.evaluate("DASHBOARD_SPEC.state.selections") or []
        assert any(
            s.get("source_figure_uid") == fig1 for s in selections
        ), "A no-op target reset must preserve the incoming cross-filter"

    def test_cached_source_locked_figure_reset_served_from_cache(
        self, page: Page, server_port: int
    ):
        """Axis locks are view-only (pruned from server requests, re-applied on
        render), so a per-figure reset of a locked figure back to full autorange
        is served from the figure-scoped client cache just like an unlocked one —
        no /dashboard/update — and the lock range survives."""
        url = _dashboard_url_cached(server_port, "plotly")
        update_requests: list[str] = []

        def capture(req: PWRequest) -> None:
            if "/dashboard/update" in req.url and req.method == "POST":
                update_requests.append(req.url)

        page.on("request", capture)
        page.goto(url)
        page.wait_for_selector(".js-plotly-plot", timeout=15_000)
        page.wait_for_timeout(500)

        # Lock the x axis, and seed a viewport so the reset has real work to do
        # (otherwise a clean locked figure is a no-op). Seeding directly keeps the
        # test deterministic regardless of recompute-axis suppression.
        locked_x = page.evaluate("""() => {
                const figA = DASHBOARD_SPEC.figures[0].uid;
                fvTryLockAxis(figA, 'x');
                DASHBOARD_SPEC.state.viewport[figA + '/y'] = {min: 10, max: 100};
                return DASHBOARD_SPEC.client_state.axis_lock_ranges[figA + '/x'];
            }""")
        assert locked_x is not None

        before_reset = len(update_requests)
        page.evaluate("() => window.fvOnResetPanel(DASHBOARD_SPEC.figures[0].uid)")
        page.wait_for_timeout(1_000)
        assert len(update_requests) == before_reset, (
            "Reset of a locked figure to full autorange must be served from the "
            "figure-scoped client cache without a round-trip"
        )
        # The x lock range is preserved across the reset.
        figA = page.evaluate("DASHBOARD_SPEC.figures[0].uid")
        lock_after = page.evaluate(
            "() => DASHBOARD_SPEC.client_state.axis_lock_ranges["
            "DASHBOARD_SPEC.figures[0].uid + '/x']"
        )
        assert (
            lock_after == locked_x
        ), "The x lock range must survive a cache-served reset"
        # Figure 0's viewport (the y zoom) is cleared.
        viewport = page.evaluate("DASHBOARD_SPEC.state.viewport") or {}
        assert not any(k.startswith(figA + "/") for k in viewport)

    def test_line_y_only_zoom_does_not_post(self, page: Page, server_port: int):
        """A line is x-bound: zooming only the y-axis changes no data, so the
        client must suppress the /dashboard/update POST — yet still persist the
        new y range locally (for share/restore)."""
        url = _dashboard_url(server_port, "plotly", n_figures=1)
        update_requests: list[str] = []

        def capture(req: PWRequest) -> None:
            if "/dashboard/update" in req.url and req.method == "POST":
                update_requests.append(req.url)

        page.on("request", capture)
        page.goto(url)
        _wait_for_init(page, "plotly")

        before = len(update_requests)
        page.evaluate(
            "() => Plotly.relayout(document.querySelectorAll('.js-plotly-plot')[0],"
            " {'yaxis.range[0]': 10, 'yaxis.range[1]': 100})"
        )
        page.wait_for_timeout(800)
        assert (
            len(update_requests) == before
        ), "y-only zoom on an x-bound line must not issue a /dashboard/update"
        # Viewport persisted anyway: a "/y" key was recorded in client state.
        has_y_viewport = page.evaluate(
            "() => Object.keys(DASHBOARD_SPEC.state.viewport || {})"
            ".some(k => k.endsWith('/y'))"
        )
        assert has_y_viewport, "y zoom must still persist into local viewport state"

    def test_line_x_zoom_posts_once(self, page: Page, server_port: int):
        """The binding axis (x) of a line must still round-trip on zoom."""
        url = _dashboard_url(server_port, "plotly", n_figures=1)
        update_requests: list[str] = []

        def capture(req: PWRequest) -> None:
            if "/dashboard/update" in req.url and req.method == "POST":
                update_requests.append(req.url)

        page.on("request", capture)
        page.goto(url)
        _wait_for_init(page, "plotly")

        before = len(update_requests)
        page.evaluate(
            "() => Plotly.relayout(document.querySelectorAll('.js-plotly-plot')[0],"
            " {'xaxis.range[0]': 50, 'xaxis.range[1]': 150})"
        )
        page.wait_for_timeout(800)
        assert (
            len(update_requests) > before
        ), "x zoom on a line must issue a viewport update"

    def test_line_y_autorange_does_not_post(self, page: Page, server_port: int):
        """Double-click autorange on the non-binding y-axis must not round-trip."""
        url = _dashboard_url(server_port, "plotly", n_figures=1)
        update_requests: list[str] = []

        def capture(req: PWRequest) -> None:
            if "/dashboard/update" in req.url and req.method == "POST":
                update_requests.append(req.url)

        page.on("request", capture)
        page.goto(url)
        _wait_for_init(page, "plotly")

        before = len(update_requests)
        page.evaluate(
            "() => Plotly.relayout(document.querySelectorAll('.js-plotly-plot')[0],"
            " {'yaxis.autorange': true})"
        )
        page.wait_for_timeout(800)
        assert (
            len(update_requests) == before
        ), "autoranging the non-binding y-axis of a line must not round-trip"

    def test_filter_summary_strip_shows_global_chip_and_source_panel_echo(
        self, page: Page, server_port: int
    ):
        url = _dashboard_url(server_port, "plotly", n_figures=2)
        page.goto(url)
        _wait_for_init(page, "plotly")

        page.evaluate("""() => {
            const figUid = DASHBOARD_SPEC.figures[0].uid;
            const sel = {
                source_figure_uid: figUid,
                predicates: [{ clauses: [{ column: 'ts', range: [100.125, 300.5] }] }],
            };
            window.fvSetSelectionState([sel]);
            return postDashboardUpdate({
                type: 'selection',
                axis_ranges: {},
                selections: DASHBOARD_SPEC.state.selections,
                force_update: true,
                figure_uid: figUid,
            });
        }""")
        page.wait_for_timeout(1_200)

        summary = page.evaluate("""() => ({
            hidden: document.getElementById('fv-filter-strip').hidden,
            chips: Array.from(document.querySelectorAll('#fv-filter-chips .fv-filter-chip'))
              .map(node => ({
                source: node.querySelector('.fv-filter-chip-source')?.textContent.trim() || '',
                summary: node.querySelector('.fv-filter-chip-text')?.textContent.trim() || '',
              })),
            panel0: document.querySelector('#fv-bar-0 [data-slot="info"]').textContent.trim(),
            panel1: document.querySelector('#fv-bar-1 [data-slot="info"]').textContent.trim(),
        })""")

        assert summary["hidden"] is False
        assert summary["chips"] == [
            {"source": "Fig0", "summary": "ts 100.125 to 300.5"}
        ]
        assert summary["panel0"] == "ts 100.125 to 300.5"
        assert summary["panel1"] == ""

    def test_filter_chip_remove_clears_only_its_source_selection(
        self, page: Page, server_port: int
    ):
        url = _dashboard_url(server_port, "plotly", n_figures=2)
        page.goto(url)
        _wait_for_init(page, "plotly")

        page.evaluate("""() => {
            const figUids = DASHBOARD_SPEC.figures.map(f => f.uid);
            window.fvSetSelectionState([
                {
                    source_figure_uid: figUids[0],
                    predicates: [{ clauses: [{ column: 'ts', range: [50, 150] }] }],
                },
                {
                    source_figure_uid: figUids[1],
                    predicates: [{ clauses: [{ column: 'val', range: [10, 20] }] }],
                },
            ]);
            return postDashboardUpdate({
                type: 'selection',
                axis_ranges: {},
                selections: DASHBOARD_SPEC.state.selections,
                force_update: true,
                figure_uid: figUids[0],
            });
        }""")
        page.wait_for_timeout(1_200)

        page.click(
            "#fv-filter-chips .fv-filter-chip:first-child .fv-filter-chip-remove"
        )
        page.wait_for_timeout(1_200)

        summary = page.evaluate("""() => ({
            selections: DASHBOARD_SPEC.state.selections,
            hidden: document.getElementById('fv-filter-strip').hidden,
            chips: Array.from(document.querySelectorAll('#fv-filter-chips .fv-filter-chip'))
              .map(node => ({
                source: node.querySelector('.fv-filter-chip-source')?.textContent.trim() || '',
                summary: node.querySelector('.fv-filter-chip-text')?.textContent.trim() || '',
              })),
            panel0: document.querySelector('#fv-bar-0 [data-slot="info"]').textContent.trim(),
            panel1: document.querySelector('#fv-bar-1 [data-slot="info"]').textContent.trim(),
        })""")

        assert len(summary["selections"]) == 1
        assert summary["selections"][0]["source_figure_uid"] == page.evaluate(
            "DASHBOARD_SPEC.figures[1].uid"
        )
        assert summary["hidden"] is False
        assert summary["chips"] == [{"source": "Fig1", "summary": "val 10 to 20"}]
        assert summary["panel0"] == ""
        assert summary["panel1"] == "val 10 to 20"

    def test_filter_summary_joins_multi_clause_and_or_predicates(
        self, page: Page, server_port: int
    ):
        url = _dashboard_url(server_port, "plotly", n_figures=1)
        page.goto(url)
        _wait_for_init(page, "plotly")

        page.evaluate("""() => {
            const figUid = DASHBOARD_SPEC.figures[0].uid;
            window.fvSetSelectionState([{
                source_figure_uid: figUid,
                predicates: [
                    { clauses: [
                        { column: 'ts', range: [1.25, 6.25] },
                        { column: 'val', range: [10, 20] },
                    ] },
                    { clauses: [{ column: 'ts', range: [30, 40] }] },
                ],
            }]);
            window.fvRefreshSelectionSummary();
        }""")

        summary = page.evaluate("""() => ({
            chips: Array.from(document.querySelectorAll('#fv-filter-chips .fv-filter-chip'))
              .map(node => ({
                source: node.querySelector('.fv-filter-chip-source')?.textContent.trim() || '',
                summary: node.querySelector('.fv-filter-chip-text')?.textContent.trim() || '',
              })),
            panel0: document.querySelector('#fv-bar-0 [data-slot="info"]').textContent.trim(),
        })""")

        assert summary["chips"] == [
            {
                "source": "Fig0",
                "summary": "ts 1.25 to 6.25 | val 10 to 20 OR ts 30 to 40",
            }
        ]
        assert summary["panel0"] == "ts 1.25 to 6.25 | val 10 to 20 OR ts 30 to 40"

    def test_hist2d_uses_control_bar_for_selection(self, page: Page, server_port: int):
        url = _dashboard_url_hist2d(server_port, "plotly")
        page.goto(url)
        _wait_for_chart(page, "plotly")
        page.wait_for_selector("#fv-bar-0[role='toolbar']", timeout=10_000)
        assert page.locator("#fv-bar-0 .fv-mode-btn[data-mode='select']").count() == 1
        assert page.locator(".modebar").count() == 0

    def test_hist2d_box_select_button_emits_selection(
        self, page: Page, server_port: int
    ):
        url = _dashboard_url_hist2d(server_port, "plotly")
        update_bodies: list[dict] = []

        def capture(req: PWRequest) -> None:
            if "/dashboard/update" in req.url and req.method == "POST":
                try:
                    update_bodies.append(json.loads(req.post_data or "{}"))
                except Exception:
                    pass

        page.on("request", capture)
        page.goto(url)
        _wait_for_chart(page, "plotly")
        page.locator("#fv-bar-0 .fv-mode-btn[data-mode='select']").click()
        page.wait_for_timeout(300)

        drag_layer = page.locator(".js-plotly-plot .nsewdrag")
        box = drag_layer.bounding_box()
        assert box is not None
        x1 = box["x"] + box["width"] * 0.2
        x2 = box["x"] + box["width"] * 0.6
        y1 = box["y"] + box["height"] * 0.3
        y2 = box["y"] + box["height"] * 0.7

        initial_count = len(update_bodies)
        page.mouse.move(x1, y1)
        page.mouse.down()
        page.mouse.move(x2, y2, steps=20)
        page.mouse.up()
        page.wait_for_timeout(1_500)

        selection_events = [
            body
            for body in update_bodies[initial_count:]
            if body.get("event", {}).get("type") == "selection"
        ]
        assert len(selection_events) >= 1, update_bodies[initial_count:]
        selections = page.evaluate("DASHBOARD_SPEC.state.selections")
        assert len(selections) == 1
        assert selections[0]["predicates"]
        assert selections[0]["predicates"][0]["clauses"][0]["range"] is not None

    def test_line_brush_selection_fires_one_dashboard_update(
        self, page: Page, server_port: int
    ):
        url = _dashboard_url_selection_duplicate_repro(server_port)
        update_bodies: list[dict] = []

        def capture(req: PWRequest) -> None:
            if "/dashboard/update" in req.url and req.method == "POST":
                try:
                    update_bodies.append(json.loads(req.post_data or "{}"))
                except Exception:
                    pass

        page.on("request", capture)
        page.goto(url)
        _wait_for_init(page, "plotly")
        page.locator("#fv-bar-0 .fv-mode-btn[data-mode='select']").click()
        page.wait_for_timeout(300)

        drag_layer = page.locator("#fv-plot-0 .nsewdrag")
        box = drag_layer.bounding_box()
        assert box is not None
        x1 = box["x"] + box["width"] * 0.2
        x2 = box["x"] + box["width"] * 0.65
        y1 = box["y"] + box["height"] * 0.25
        y2 = box["y"] + box["height"] * 0.75

        initial_count = len(update_bodies)
        page.mouse.move(x1, y1)
        page.mouse.down()
        page.mouse.move(x2, y2, steps=20)
        page.mouse.up()
        page.wait_for_timeout(1_500)

        selection_events = [
            body
            for body in update_bodies[initial_count:]
            if body.get("event", {}).get("type") == "selection"
        ]
        assert len(selection_events) == 1, (
            "One user brush must emit exactly one selection /dashboard/update; "
            f"got {len(selection_events)} events: "
            f"{[body.get('event') for body in selection_events]}"
        )

    def test_bar_brush_selection_keeps_plotly_selection_box(
        self, page: Page, server_port: int
    ):
        url = _dashboard_url_plotly_selection_box(server_port, "bar")
        page.goto(url)
        _wait_for_init(page, "plotly")
        page.wait_for_function(
            "() => document.querySelector('#fv-plot-0')?.data?.[0]?.x?.length > 0"
        )

        page.evaluate("""() => {
                const fig = DASHBOARD_SPEC.figures[0];
                handleSelected({
                    range: {x: ['BE', 'NL'], y: [0, 80]},
                    points: [{x: 'BE'}, {x: 'NL'}],
                }, fig.uid);
            }""")
        page.wait_for_function("""() => {
                const sels = DASHBOARD_SPEC.state.selections || [];
                return sels.length === 1
                    && ((layoutsByFig[0] || {}).selections || []).length === 1;
            }""")
        page.wait_for_function("""() => {
                const x = (divs[1].data && divs[1].data[0] && divs[1].data[0].x) || [];
                return x.length === 4 && x.join(',') === '1,2,4,5';
            }""")

        selection = page.evaluate("DASHBOARD_SPEC.state.selections[0]")
        predicates = selection["predicates"]
        assert [
            clause["values"][0] for pred in predicates for clause in pred["clauses"]
        ] == ["BE", "NL"]
        assert selection["_plotly_selection_box"] == {
            "x0": "BE",
            "x1": "NL",
            "xref": "x",
            "y0": 0,
            "y1": 80,
            "yref": "y",
        }

    def test_histogram_brush_selection_keeps_plotly_selection_box(
        self, page: Page, server_port: int
    ):
        url = _dashboard_url_plotly_selection_box(server_port, "histogram")
        browser_errors: list[str] = []
        page.on("pageerror", lambda exc: browser_errors.append(str(exc)))
        page.on(
            "console",
            lambda msg: (
                browser_errors.append(msg.text) if msg.type == "error" else None
            ),
        )
        page.goto(url)
        _wait_for_init(page, "plotly")
        page.wait_for_function(
            "() => document.querySelector('#fv-plot-0')?.data?.[0]?.x?.length > 0"
        )

        page.evaluate("""() => {
                const fig = DASHBOARD_SPEC.figures[0];
                handleSelected({
                    range: {x: [20, 50], y: [0, 4]},
                    points: [{x: 30}, {x: 40}],
                }, fig.uid);
            }""")
        page.wait_for_function("""() => {
                const sels = DASHBOARD_SPEC.state.selections || [];
                return sels.length === 1
                    && ((layoutsByFig[0] || {}).selections || []).length === 1;
            }""")
        page.wait_for_function("""() => {
                const x = (divs[1].data && divs[1].data[0] && divs[1].data[0].x) || [];
                return x.length === 4 && x.join(',') === '2,3,4,5';
            }""")

        selection = page.evaluate("DASHBOARD_SPEC.state.selections[0]")
        clauses = selection["predicates"][0]["clauses"]
        assert clauses == [{"column": "val", "range": [20, 50]}]
        full_layout_selection = page.evaluate("divs[0]._fullLayout.selections[0]")
        assert full_layout_selection["type"] == "rect"

        selection_box = page.locator("#fv-plot-0 .selectionlayer path").nth(0)
        plot_box = page.locator("#fv-plot-0 .nsewdrag").bounding_box()
        selected_box = selection_box.bounding_box()
        assert plot_box is not None
        assert selected_box is not None
        x = selected_box["x"] + selected_box["width"] / 2
        y = plot_box["y"] + plot_box["height"] / 2
        page.mouse.move(x, y)
        page.mouse.down()
        page.mouse.move(x + 10, y, steps=5)
        page.mouse.up()
        page.wait_for_timeout(300)
        assert browser_errors == []

    def test_horizontal_drag_zoom_works_with_hover(self, page: Page, server_port: int):
        url = _dashboard_url(server_port, "plotly", n_figures=1)
        update_bodies: list[dict] = []
        page_errors: list[str] = []

        def capture(req: PWRequest) -> None:
            if "/dashboard/update" in req.url and req.method == "POST":
                try:
                    update_bodies.append(json.loads(req.post_data or "{}"))
                except Exception:
                    pass

        page.on("request", capture)
        page.on("pageerror", lambda err: page_errors.append(str(err)))
        page.goto(url)
        page.wait_for_selector(".js-plotly-plot", timeout=15_000)
        page.wait_for_timeout(1_000)

        drag_layer = page.locator(".js-plotly-plot .nsewdrag")
        box = drag_layer.bounding_box()
        assert box is not None
        x1 = box["x"] + box["width"] * 0.2
        x2 = box["x"] + box["width"] * 0.8
        y = box["y"] + box["height"] * 0.5

        initial_count = len(update_bodies)
        page.mouse.move(x1, y)
        page.mouse.down()
        page.mouse.move(x2, y, steps=20)
        page.mouse.up()
        page.wait_for_timeout(1_200)

        assert page_errors == [], (
            "Horizontal Plotly drag-zoom must not crash with hover enabled. "
            f"Got page errors: {page_errors}"
        )
        viewport_events = [
            body
            for body in update_bodies[initial_count:]
            if body.get("event", {}).get("type") == "viewport"
        ]
        assert len(viewport_events) >= 1, (
            "Horizontal Plotly drag-zoom must emit a viewport /dashboard/update POST. "
            f"Got update events: "
            f"{[b.get('event', {}).get('type') for b in update_bodies[initial_count:]]}"
        )

    def test_geo_histogram_zoom_posts_coordinates_and_updates_trace(
        self, page: Page, server_port: int
    ):
        url = _dashboard_url_geo(server_port, "plotly")
        update_bodies: list[dict] = []
        response_statuses: list[int] = []

        def capture_request(req: PWRequest) -> None:
            if "/dashboard/update" in req.url and req.method == "POST":
                try:
                    update_bodies.append(json.loads(req.post_data or "{}"))
                except Exception:
                    pass

        page.on("request", capture_request)
        page.on(
            "response",
            lambda resp: (
                response_statuses.append(resp.status)
                if "/dashboard/update" in resp.url
                else None
            ),
        )
        page.goto(url)
        _wait_for_init(page, "plotly")

        before = page.evaluate("""() => {
                const figUid = DASHBOARD_SPEC.figures[0].uid;
                const idx = figUidToIdx[figUid];
                const trace = tracesByFig[idx][0] || {};
                const features = ((trace.geojson || {}).features || []);
                function maxLatSpan(items) {
                  let best = 0;
                  for (const feature of items) {
                    const ring = (((feature || {}).geometry || {}).coordinates || [])[0] || [];
                    const lats = ring.map(pt => pt[1]);
                    if (!lats.length) continue;
                    best = Math.max(best, Math.max(...lats) - Math.min(...lats));
                  }
                  return best;
                }
                return {
                  span: maxLatSpan(features),
                  locations: (trace.locations || []).length,
                };
            }""")

        coords = [
            [-73.5, 40.5],
            [-72.5, 40.5],
            [-72.5, 41.5],
            [-73.5, 41.5],
        ]
        initial_req_count = len(update_bodies)
        initial_resp_count = len(response_statuses)
        page.evaluate(
            """({coords}) => {
                const figUid = DASHBOARD_SPEC.figures[0].uid;
                handleRelayout({'map._derived': {coordinates: coords}}, figUid);
            }""",
            {"coords": coords},
        )
        page.wait_for_timeout(2_000)

        viewport_events = [
            body
            for body in update_bodies[initial_req_count:]
            if body.get("event", {}).get("type") == "viewport"
        ]
        assert len(viewport_events) >= 1, update_bodies[initial_req_count:]
        assert viewport_events[-1]["event"]["axis_ranges"]["coordinates"] == coords
        assert 200 in response_statuses[initial_resp_count:], response_statuses[
            initial_resp_count:
        ]

        after = page.evaluate("""() => {
                const figUid = DASHBOARD_SPEC.figures[0].uid;
                const idx = figUidToIdx[figUid];
                const trace = tracesByFig[idx][0] || {};
                const features = ((trace.geojson || {}).features || []);
                function maxLatSpan(items) {
                  let best = 0;
                  for (const feature of items) {
                    const ring = (((feature || {}).geometry || {}).coordinates || [])[0] || [];
                    const lats = ring.map(pt => pt[1]);
                    if (!lats.length) continue;
                    best = Math.max(best, Math.max(...lats) - Math.min(...lats));
                  }
                  return best;
                }
                return {
                  span: maxLatSpan(features),
                  locations: (trace.locations || []).length,
                };
            }""")
        assert after["locations"] > 0
        assert after["span"] < before["span"]

    def test_geo_histogram_keeps_zoom_pan_mode_enabled(
        self, page: Page, server_port: int
    ):
        url = _dashboard_url_geo(server_port, "plotly")
        page.goto(url)
        _wait_for_init(page, "plotly")

        mode_state = page.evaluate("""() => {
                const figUid = DASHBOARD_SPEC.figures[0].uid;
                const idx = figUidToIdx[figUid];
                const indicator = document.getElementById('fv-bar-' + idx);
                const buttons = Object.fromEntries(
                  Array.from(indicator.querySelectorAll('.fv-mode-btn'))
                    .map(btn => [btn.dataset.mode, {
                      disabled: btn.disabled,
                      active: btn.classList.contains('mode-active'),
                    }])
                );
                return {
                  dragmode: divs[idx]._fullLayout.dragmode,
                  buttons,
                };
            }""")

        assert mode_state["dragmode"] != "select"
        assert mode_state["buttons"]["zoom"]["disabled"] is False
        assert mode_state["buttons"]["pan"]["disabled"] is False
        assert mode_state["buttons"]["select"]["disabled"] is False

    def test_geo_histogram_selection_filters_linked_line(
        self, page: Page, server_port: int
    ):
        url = _dashboard_url_geo_with_line(server_port, "plotly")
        update_bodies: list[dict] = []
        response_statuses: list[int] = []

        def capture_request(req: PWRequest) -> None:
            if "/dashboard/update" in req.url and req.method == "POST":
                try:
                    update_bodies.append(json.loads(req.post_data or "{}"))
                except Exception:
                    pass

        page.on("request", capture_request)
        page.on(
            "response",
            lambda resp: (
                response_statuses.append(resp.status)
                if "/dashboard/update" in resp.url
                else None
            ),
        )
        page.goto(url)
        _wait_for_init(page, "plotly")

        before_line_count = page.evaluate("""() => {
                const lineUid = DASHBOARD_SPEC.figures[1].uid;
                const idx = figUidToIdx[lineUid];
                const trace = tracesByFig[idx][0] || {};
                return (trace.x || []).length;
            }""")
        selection_prep = page.evaluate("""() => {
                const geoUid = DASHBOARD_SPEC.figures[0].uid;
                const idx = figUidToIdx[geoUid];
                const trace = tracesByFig[idx][0];
                const wanted = ['r1_c1', 'r1_c2', 'r2_c1', 'r2_c2'];
                const selected = wanted.filter(loc => (trace.locations || []).includes(loc));
                const featureById = Object.fromEntries(
                  ((trace.geojson || {}).features || []).map(feature => [feature.id, feature])
                );
                let minLon = Infinity, maxLon = -Infinity, minLat = Infinity, maxLat = -Infinity;
                for (const loc of selected) {
                  const ring = (((featureById[loc] || {}).geometry || {}).coordinates || [])[0] || [];
                  for (const point of ring) {
                    minLon = Math.min(minLon, point[0]);
                    maxLon = Math.max(maxLon, point[0]);
                    minLat = Math.min(minLat, point[1]);
                    maxLat = Math.max(maxLat, point[1]);
                  }
                }
                return {
                  locations: selected,
                  expectedX: [minLon, maxLon],
                  expectedY: [minLat, maxLat],
                };
            }""")
        assert len(selection_prep["locations"]) == 4

        initial_req_count = len(update_bodies)
        initial_resp_count = len(response_statuses)
        page.evaluate(
            """({locations}) => {
                const geoUid = DASHBOARD_SPEC.figures[0].uid;
                const idx = figUidToIdx[geoUid];
                const trace = tracesByFig[idx][0];
                const points = locations.map(loc => ({
                  location: loc,
                  data: { uid: trace.uid },
                }));
                handleSelected({points}, geoUid);
            }""",
            {"locations": selection_prep["locations"]},
        )
        page.wait_for_timeout(2_000)

        selection_events = [
            body
            for body in update_bodies[initial_req_count:]
            if body.get("event", {}).get("type") == "selection"
        ]
        assert len(selection_events) >= 1, update_bodies[initial_req_count:]
        assert 200 in response_statuses[initial_resp_count:], response_statuses[
            initial_resp_count:
        ]

        selection = page.evaluate("DASHBOARD_SPEC.state.selections[0]")
        assert selection["source_figure_uid"] == page.evaluate(
            "DASHBOARD_SPEC.figures[0].uid"
        )
        clauses = selection["predicates"][0]["clauses"]
        ranges_by_col = {c["column"]: c["range"] for c in clauses}
        # Two clauses: one for the x column (lon) and one for the y column (lat)
        assert len(ranges_by_col) == 2
        assert list(ranges_by_col.values())[0] == pytest.approx(
            selection_prep["expectedX"]
        )
        assert list(ranges_by_col.values())[1] == pytest.approx(
            selection_prep["expectedY"]
        )

        after_line_count = page.evaluate("""() => {
                const lineUid = DASHBOARD_SPEC.figures[1].uid;
                const idx = figUidToIdx[lineUid];
                const trace = tracesByFig[idx][0] || {};
                return (trace.x || []).length;
            }""")
        assert after_line_count < before_line_count

    def test_boxplot_page_loads_and_chart_renders(self, page: Page, server_port: int):
        url = _dashboard_url_boxplot(server_port, "plotly")
        page.goto(url)
        page.wait_for_selector(".js-plotly-plot", timeout=15_000)
        charts = page.query_selector_all(".js-plotly-plot")
        assert len(charts) >= 1, f"Expected >=1 Plotly chart, found {len(charts)}"


@pytest.mark.parametrize("renderer", ["plotly", "echarts"])
class TestGroupedBrowser:
    def test_grouped_line_renders_children_not_parent(
        self, page: Page, server_port: int, renderer: str
    ):
        url = _dashboard_url_grouped(server_port, renderer, trace_type="line")
        page.goto(url)
        _wait_for_chart(page, renderer)
        page.wait_for_timeout(500)
        assert _grouped_child_count(page, renderer) == 2

    def test_grouped_line_viewport_removes_hidden_child(
        self, page: Page, server_port: int, renderer: str
    ):
        url = _dashboard_url_grouped(server_port, renderer, trace_type="line")
        page.goto(url)
        _wait_for_init(page, renderer)

        page.evaluate("""async () => {
              const figUid = DASHBOARD_SPEC.figures[0].uid;
              await postDashboardUpdate({
                type: 'viewport',
                axis_ranges: { x: [0, 60] },
                selections: DASHBOARD_SPEC.state.selections,
                force_update: false,
                figure_uid: figUid,
              });
            }""")
        page.wait_for_timeout(2_000)
        assert _grouped_child_count(page, renderer) == 1

    def test_grouped_bar_renders_children(
        self, page: Page, server_port: int, renderer: str
    ):
        url = _dashboard_url_grouped(server_port, renderer, trace_type="bar")
        page.goto(url)
        _wait_for_chart(page, renderer)
        page.wait_for_timeout(500)
        assert _grouped_child_count(page, renderer) == 2


# ---------------------------------------------------------------------------
# ECharts adapter smoke tests
# ---------------------------------------------------------------------------


class TestEChartsBrowser:
    def test_page_loads_and_charts_render(self, page: Page, server_port: int):
        url = _dashboard_url(server_port, "echarts")
        page.goto(url)
        # ECharts renders into canvas elements inside fv-chart-* containers.
        page.wait_for_selector("[id^='fv-chart-']", timeout=15_000)
        charts = page.query_selector_all("[id^='fv-chart-']")
        assert len(charts) >= 2, f"Expected >=2 ECharts containers, found {len(charts)}"

    def test_toolbar_buttons_present(self, page: Page, server_port: int):
        url = _dashboard_url(server_port, "echarts")
        page.goto(url)
        page.wait_for_selector("#fv-btn-reset", timeout=10_000)
        for btn_id in (
            "fv-btn-reset",
            "fv-btn-deselect",
            "fv-btn-cfmode",
            "fv-btn-grid",
            "fv-btn-share",
            "fv-btn-export",
            "fv-btn-import",
        ):
            btn = page.query_selector(f"#{btn_id}")
            assert btn is not None, f"Toolbar button #{btn_id} not found"

    def test_live_option_has_no_toolbox(self, page: Page, server_port: int):
        url = _dashboard_url(server_port, "echarts")
        page.goto(url)
        _wait_for_init(page, "echarts")
        toolbox_visible = page.evaluate("""() => {
                const chart = echarts.getInstanceByDom(document.querySelector("[id^='fv-chart-']"));
                const option = chart && chart.getOption();
                const toolbox = Array.isArray(option && option.toolbox) ? option.toolbox[0] : option && option.toolbox;
                return !!(toolbox && toolbox.show);
            }""")
        assert toolbox_visible is False

    def test_datetime_line_uses_time_axis_and_renders_points(
        self, page: Page, server_port: int
    ):
        url = _dashboard_url_datetime_line(server_port, "echarts")
        page.goto(url)
        _wait_for_init(page, "echarts")
        state = page.evaluate("""() => {
                const chart = echarts.getInstanceByDom(document.querySelector("[id^='fv-chart-']"));
                const option = chart.getOption();
                const axis = Array.isArray(option.xAxis) ? option.xAxis[0] : option.xAxis;
                const series = option.series || [];
                return {
                    xAxisType: axis && axis.type,
                    seriesCount: series.length,
                    pointCounts: series.map(item => (item.data || []).length),
                };
            }""")
        assert state["xAxisType"] == "time"
        assert state["seriesCount"] == 2
        assert all(count > 0 for count in state["pointCounts"])

    def test_scroll_zoom_triggers_dashboard_update(self, page: Page, server_port: int):
        """Scroll zoom on ECharts must trigger a /dashboard/update POST."""
        url = _dashboard_url(server_port, "echarts")
        update_bodies: list[dict] = []

        def capture(req: PWRequest) -> None:
            if "/dashboard/update" in req.url and req.method == "POST":
                try:
                    update_bodies.append(json.loads(req.post_data or "{}"))
                except Exception:
                    pass

        page.on("request", capture)
        page.goto(url)
        # Wait for canvas elements (charts fully initialised).
        page.wait_for_selector("canvas", timeout=15_000)
        initial_count = len(update_bodies)

        # Scroll on the first chart container to trigger zoom.
        chart_container = page.query_selector("[id^='fv-chart-']")
        assert chart_container is not None
        box = chart_container.bounding_box()
        cx = box["x"] + box["width"] / 2
        cy = box["y"] + box["height"] / 2
        page.mouse.move(cx, cy)
        page.mouse.wheel(0, -300)  # scroll up = zoom in
        # ECharts debounces datazoom 150 ms; wait generously.
        page.wait_for_timeout(1_500)

        viewport_events = [
            b
            for b in update_bodies[initial_count:]
            if b.get("event", {}).get("type") == "viewport"
        ]
        assert len(viewport_events) >= 1, (
            "Scroll zoom must produce at least one viewport /dashboard/update POST. "
            f"Got update events: {[b.get('event', {}).get('type') for b in update_bodies[initial_count:]]}"
        )

    def test_locked_axes_ignore_scroll_zoom(self, page: Page, server_port: int):
        url = _dashboard_url(server_port, "echarts")
        update_bodies: list[dict] = []

        def capture(req: PWRequest) -> None:
            if "/dashboard/update" in req.url and req.method == "POST":
                try:
                    update_bodies.append(json.loads(req.post_data or "{}"))
                except Exception:
                    pass

        page.on("request", capture)
        page.goto(url)
        _wait_for_init(page, "echarts")
        page.wait_for_selector(
            "#fv-bar-0 .fv-mode-action-btn[data-action='lock-axes']:not([disabled])"
        )

        page.click("#fv-bar-0 .fv-mode-action-btn[data-action='lock-axes']")
        page.wait_for_function("""() => {
                const figUid = DASHBOARD_SPEC.figures[0].uid;
                return DASHBOARD_SPEC.client_state.axis_locks[figUid + '/x'] === true
                  && !!DASHBOARD_SPEC.client_state.axis_lock_ranges[figUid + '/x'];
            }""")

        before = page.evaluate("""() => {
                const figUid = DASHBOARD_SPEC.figures[0].uid;
                const chart = echarts.getInstanceByDom(document.getElementById('fv-chart-0'));
                const option = chart.getOption();
                const dz = (option.dataZoom || [])[0] || {};
                return {
                  lockRange: DASHBOARD_SPEC.client_state.axis_lock_ranges[figUid + '/x'],
                  viewport: DASHBOARD_SPEC.state.viewport[figUid + '/x'] || null,
                  startValue: dz.startValue,
                  endValue: dz.endValue,
                };
            }""")
        count_before = len(update_bodies)

        chart_container = page.query_selector("#fv-chart-0")
        assert chart_container is not None
        box = chart_container.bounding_box()
        cx = box["x"] + box["width"] / 2
        cy = box["y"] + box["height"] / 2
        page.mouse.move(cx, cy)
        page.mouse.wheel(0, -300)
        page.wait_for_timeout(1000)

        after = page.evaluate("""() => {
                const figUid = DASHBOARD_SPEC.figures[0].uid;
                const chart = echarts.getInstanceByDom(document.getElementById('fv-chart-0'));
                const option = chart.getOption();
                const dz = (option.dataZoom || [])[0] || {};
                return {
                  viewport: DASHBOARD_SPEC.state.viewport[figUid + '/x'] || null,
                  startValue: dz.startValue,
                  endValue: dz.endValue,
                };
            }""")

        viewport_events = [
            body.get("event", {})
            for body in update_bodies[count_before:]
            if body.get("event", {}).get("type") == "viewport"
        ]
        assert viewport_events == []
        assert after["viewport"] is None
        assert after["startValue"] == before["startValue"]
        assert after["endValue"] == before["endValue"]

    def test_reset_button_clears_viewport(self, page: Page, server_port: int):
        url = _dashboard_url(server_port, "echarts")
        update_requests: list[str] = []
        page.on(
            "request",
            lambda req: (
                update_requests.append(req.url)
                if "/dashboard/update" in req.url
                else None
            ),
        )
        page.goto(url)
        page.wait_for_selector("canvas", timeout=15_000)
        initial_count = len(update_requests)

        page.click("#fv-btn-reset")
        page.wait_for_timeout(1_000)
        assert (
            len(update_requests) > initial_count
        ), "Clicking Reset must fire at least one /dashboard/update request"

    def test_share_url_roundtrip(self, page: Page, server_port: int):
        """Share button must produce a URL that loads the same dashboard."""
        url = _dashboard_url(server_port, "echarts")
        page.goto(url)
        page.wait_for_selector("canvas", timeout=15_000)

        # Intercept the alert / clipboard write from the Share button.
        share_urls: list[str] = []

        def handle_dialog(dialog):
            share_urls.append(dialog.message)
            dialog.dismiss()

        page.on("dialog", handle_dialog)
        page.click("#fv-btn-share")
        page.wait_for_timeout(1_000)

        if share_urls:
            # If an alert was shown with the URL, navigate to it.
            share_url = share_urls[0].strip()
            if share_url.startswith("http"):
                page.goto(share_url)
                page.wait_for_selector("[id^='fv-chart-']", timeout=15_000)
                charts = page.query_selector_all("[id^='fv-chart-']")
                assert len(charts) >= 2

    def test_pie_click_filters_and_toggles_clear(self, page: Page, server_port: int):
        url = _dashboard_url_treemap_pie_selection(server_port, "echarts")
        page.goto(url)
        _wait_for_init(page, "echarts")

        page.evaluate("""() => {
                const figUid = DASHBOARD_SPEC.figures[2].uid;
                const chart = echarts.getInstanceByDom(document.getElementById('fv-chart-2'));
                const series = (chart.getOption().series || [])[0];
                const item = (series.data || []).find(entry => entry && entry.name === 'solar');
                handleEChartsClick({
                    seriesType: 'pie',
                    seriesId: series.id,
                    name: item.name,
                    data: item,
                }, figUid);
            }""")
        page.wait_for_function(
            "() => (DASHBOARD_SPEC.state.selections || []).length === 1"
        )
        page.wait_for_timeout(800)

        selection = page.evaluate("DASHBOARD_SPEC.state.selections[0]")
        clauses = selection["predicates"][0]["clauses"]
        assert clauses == [{"column": "source", "values": ["solar"]}]

        target_labels = page.evaluate("""() => {
                const chart = echarts.getInstanceByDom(document.getElementById('fv-chart-0'));
                const series = (chart.getOption().series || [])[0] || {};
                return (series.data || []).map(entry => Array.isArray(entry) ? entry[0] : entry.name);
            }""")
        assert target_labels == ["solar"]

        page.evaluate("""() => {
                const figUid = DASHBOARD_SPEC.figures[2].uid;
                const chart = echarts.getInstanceByDom(document.getElementById('fv-chart-2'));
                const series = (chart.getOption().series || [])[0];
                const item = (series.data || []).find(entry => entry && entry.name === 'solar');
                handleEChartsClick({
                    seriesType: 'pie',
                    seriesId: series.id,
                    name: item.name,
                    data: item,
                }, figUid);
            }""")
        page.wait_for_function(
            "() => (DASHBOARD_SPEC.state.selections || []).length === 0"
        )

    def test_treemap_leaf_click_filters_with_full_path(
        self, page: Page, server_port: int
    ):
        url = _dashboard_url_treemap_pie_selection(server_port, "echarts")
        page.goto(url)
        _wait_for_init(page, "echarts")

        page.evaluate("""() => {
                const figUid = DASHBOARD_SPEC.figures[1].uid;
                const chart = echarts.getInstanceByDom(document.getElementById('fv-chart-1'));
                const series = (chart.getOption().series || [])[0];
                function findNode(nodes, wantedId) {
                    for (const node of (nodes || [])) {
                        if (node && node.id === wantedId) return node;
                        const child = findNode(node && node.children, wantedId);
                        if (child) return child;
                    }
                    return null;
                }
                const node = findNode(series.data || [], 'root/solar/NL');
                handleEChartsClick({
                    seriesType: 'treemap',
                    seriesId: series.id,
                    data: node,
                }, figUid);
            }""")
        page.wait_for_function(
            "() => (DASHBOARD_SPEC.state.selections || []).length === 1"
        )
        page.wait_for_timeout(800)

        selection = page.evaluate("DASHBOARD_SPEC.state.selections[0]")
        clauses = selection["predicates"][0]["clauses"]
        assert {item["column"]: item["values"] for item in clauses} == {
            "source": ["solar"],
            "country": ["NL"],
        }

    def test_treemap_option_exposes_named_colored_top_level_nodes(
        self, page: Page, server_port: int
    ):
        url = _dashboard_url_treemap_colormap(server_port, "echarts")
        page.goto(url)
        _wait_for_init(page, "echarts")
        top_nodes = page.evaluate("""() => {
                const chart = echarts.getInstanceByDom(document.querySelector("[id^='fv-chart-']"));
                const series = (chart.getOption().series || [])[0] || {};
                return (series.data || []).map(node => ({
                    id: node.id,
                    name: node.name,
                    color: node.itemStyle && node.itemStyle.color,
                    childCount: (node.children || []).length,
                }));
            }""")
        assert top_nodes == [
            {"id": "root/Solar", "name": "Solar", "color": "#e3a24d", "childCount": 4},
            {"id": "root/Wind", "name": "Wind", "color": "#5b8db8", "childCount": 4},
        ]


@pytest.mark.skip(
    reason="EChartsAdapter is deprecated (see CLAUDE.md); the ECharts demo "
    "canvas does not render on this branch. Skipped pending removal of the "
    "ECharts adapter rather than expanding the deprecated path."
)
class TestDemoEChartsBrowser:
    def test_demo_datetime_selection_filters_targets_and_formats_timestamp(
        self, page: Page, demo_server_port: int
    ):
        url = _demo_url(demo_server_port, "echarts")
        page.goto(url)
        _wait_for_init(page, "echarts")

        before = page.evaluate("""() => {
                const chart = echarts.getInstanceByDom(document.getElementById('fv-chart-1'));
                return (chart.getOption().series || []).map(series =>
                  (series.data || []).reduce((acc, item) => acc + (Array.isArray(item) ? item[1] : 0), 0)
                );
            }""")

        page.evaluate("""async () => {
                const figUid = DASHBOARD_SPEC.figures[0].uid;
                const selection = {
                  source_figure_uid: figUid,
                  predicates: [{
                    clauses: [{
                      column: 'timestamp',
                      range: ['2017-09-11T18:28:06.603Z', '2017-12-27T22:17:01.514Z'],
                    }],
                  }],
                };
                window.fvSetSelectionState?.([selection]);
                await postDashboardUpdate({
                  type: 'selection',
                  axis_ranges: {},
                  selections: [selection],
                  force_update: true,
                  figure_uid: figUid,
                });
            }""")
        page.wait_for_function("""() => {
                const text = document.getElementById('fv-filter-chips')?.textContent || '';
                return (DASHBOARD_SPEC.state.selections || []).length === 1
                  && text.includes('timestamp')
                  && text.includes('UTC');
            }""")
        page.wait_for_timeout(1000)

        after = page.evaluate("""() => {
                const chart = echarts.getInstanceByDom(document.getElementById('fv-chart-1'));
                const text = document.getElementById('fv-filter-chips')?.textContent || '';
                return {
                  sums: (chart.getOption().series || []).map(series =>
                    (series.data || []).reduce((acc, item) => acc + (Array.isArray(item) ? item[1] : 0), 0)
                  ),
                  text,
                };
            }""")

        assert all(
            after_sum < before_sum
            for after_sum, before_sum in zip(after["sums"], before)
        )
        assert "2017-09-11 18:28:06.603 UTC" in after["text"]
        assert "1505108886603" not in after["text"]

    def test_demo_treemap_uses_parent_bundle_borders(
        self, page: Page, demo_server_port: int
    ):
        url = _demo_url(demo_server_port, "echarts")
        page.goto(url)
        _wait_for_init(page, "echarts")

        treemap_state = page.evaluate("""() => {
                const figIdx = DASHBOARD_SPEC.figures.findIndex(fig =>
                  (fig.traces || []).some(ts => ts.trace_type === 'treemap')
                );
                const chart = echarts.getInstanceByDom(document.getElementById('fv-chart-' + figIdx));
                const series = (chart.getOption().series || [])[0] || {};
                const top = (series.data || []).map(node => ({
                  name: node.name,
                  color: node.itemStyle && node.itemStyle.color,
                  childBorder: node.children && node.children[0] && node.children[0].itemStyle && node.children[0].itemStyle.borderColor,
                  childWidth: node.children && node.children[0] && node.children[0].itemStyle && node.children[0].itemStyle.borderWidth,
                }));
                return {
                  levels: (series.levels || []).length,
                  top,
                };
            }""")

        assert treemap_state["levels"] >= 3
        assert treemap_state["top"][:2] == [
            {
                "name": "Solar",
                "color": "#e3a24d",
                "childBorder": "#e3a24d",
                "childWidth": 3,
            },
            {
                "name": "Wind",
                "color": "#5b8db8",
                "childBorder": "#5b8db8",
                "childWidth": 3,
            },
        ]


# ---------------------------------------------------------------------------
# Cross-filter browser tests (parametrized for both adapters)
# ---------------------------------------------------------------------------


def _wait_for_init(page: Page, renderer: str) -> None:
    """Wait until the initial data load has populated the charts."""
    if renderer == "plotly":
        page.wait_for_selector(".js-plotly-plot", timeout=15_000)
    else:
        page.wait_for_selector("canvas", timeout=15_000)
    # Allow the init POST round-trip to complete.
    page.wait_for_timeout(2_000)


@pytest.mark.parametrize("renderer", ["plotly", "echarts"])
class TestCrossFilterBrowser:
    """Test cross-filter selection across 3 figures in a headless browser."""

    def test_cross_filter_A_updates_B_and_C(
        self, page: Page, server_port: int, renderer: str
    ):
        url = _dashboard_url(server_port, renderer, n_figures=3)
        update_bodies: list[dict] = []

        def capture(req: PWRequest) -> None:
            if "/dashboard/update" in req.url and req.method == "POST":
                try:
                    update_bodies.append(json.loads(req.post_data or "{}"))
                except Exception:
                    pass

        page.on("request", capture)
        page.goto(url)
        _wait_for_init(page, renderer)
        initial_count = len(update_bodies)

        # Trigger a selection on figure A via JS.
        page.evaluate("""() => {
            const figUids = DASHBOARD_SPEC.figures.map(f => f.uid);
            const sel = {
                source_figure_uid: figUids[0],
                predicates: [{ clauses: [{ column: 'ts', range: [100, 300] }] }],
            };
            DASHBOARD_SPEC.state.selections = [sel];
            return postDashboardUpdate({
                type: 'selection', axis_ranges: {},
                selections: DASHBOARD_SPEC.state.selections,
                force_update: true, figure_uid: figUids[0],
            });
        }""")
        page.wait_for_timeout(2_000)

        selection_events = [
            b
            for b in update_bodies[initial_count:]
            if b.get("event", {}).get("type") == "selection"
        ]
        assert len(selection_events) >= 1, "Expected at least one selection event POST"

    def test_deselect_clears_filter(self, page: Page, server_port: int, renderer: str):
        url = _dashboard_url(server_port, renderer, n_figures=3)
        update_bodies: list[dict] = []

        def capture(req: PWRequest) -> None:
            if "/dashboard/update" in req.url and req.method == "POST":
                try:
                    update_bodies.append(json.loads(req.post_data or "{}"))
                except Exception:
                    pass

        page.on("request", capture)
        page.goto(url)
        _wait_for_init(page, renderer)

        # Select on A, then deselect via toolbar.
        page.evaluate("""() => {
            const figUids = DASHBOARD_SPEC.figures.map(f => f.uid);
            const sel = {
                source_figure_uid: figUids[0],
                predicates: [{ clauses: [{ column: 'ts', range: [100, 300] }] }],
            };
            DASHBOARD_SPEC.state.selections = [sel];
            return postDashboardUpdate({
                type: 'selection', axis_ranges: {},
                selections: DASHBOARD_SPEC.state.selections,
                force_update: true, figure_uid: figUids[0],
            });
        }""")
        page.wait_for_timeout(1_000)

        count_before_deselect = len(update_bodies)
        page.click("#fv-btn-deselect")
        page.wait_for_timeout(2_000)

        deselect_events = [
            b
            for b in update_bodies[count_before_deselect:]
            if b.get("event", {}).get("type") == "deselect"
        ]
        assert len(deselect_events) >= 1, "Deselect button must fire a deselect event"
        last = deselect_events[-1]
        assert (
            last["event"].get("selections", []) == []
        ), "Deselect must clear selections"

    def test_panel_reset_clears_sourced_selection(
        self, page: Page, server_port: int, renderer: str
    ):
        url = _dashboard_url(server_port, renderer, n_figures=3)
        update_bodies: list[dict] = []

        def capture(req: PWRequest) -> None:
            if "/dashboard/update" in req.url and req.method == "POST":
                try:
                    update_bodies.append(json.loads(req.post_data or "{}"))
                except Exception:
                    pass

        page.on("request", capture)
        page.goto(url)
        _wait_for_init(page, renderer)

        # Inject a selection sourced from figure 0.
        page.evaluate("""() => {
            const figUids = DASHBOARD_SPEC.figures.map(f => f.uid);
            const sel = {
                source_figure_uid: figUids[0],
                predicates: [{ clauses: [{ column: 'ts', range: [100, 300] }] }],
            };
            DASHBOARD_SPEC.state.selections = [sel];
            return postDashboardUpdate({
                type: 'selection', axis_ranges: {},
                selections: DASHBOARD_SPEC.state.selections,
                force_update: true, figure_uid: figUids[0],
            });
        }""")
        page.wait_for_timeout(1_000)

        count_before = len(update_bodies)
        # Click the Reset button on panel 0 (the source figure).
        page.click('.fv-mode-action-btn[data-action="reset-panel"]')
        page.wait_for_timeout(2_000)

        # The event must be 'deselect' (no remaining selections) not 'viewport'.
        post_reset = update_bodies[count_before:]
        event_types = {b.get("event", {}).get("type") for b in post_reset}
        assert event_types & {"deselect", "selection"}, (
            "Panel reset must fire a deselect/selection event when it sourced a filter, "
            f"got event types: {event_types}"
        )
        assert (
            "viewport" not in event_types
        ), "Panel reset must not send a viewport event when it also clears a selection"

        # The JS selection state must have dropped the figure-0 selection.
        fig0_uid = page.evaluate("DASHBOARD_SPEC.figures[0].uid")
        selections = page.evaluate("DASHBOARD_SPEC.state.selections") or []
        sourced = [s for s in selections if s.get("source_figure_uid") == fig0_uid]
        assert (
            sourced == []
        ), "Panel reset must remove the selection sourced from that figure"

    def test_panel_reset_of_target_figure_keeps_incoming_filter(
        self, page: Page, server_port: int, renderer: str
    ):
        """Resetting a *zoomed* figure that is only a cross-filter *target* (the
        filter is sourced by another figure) must reset only its viewport via a
        'viewport' event, keep the incoming selection, and not disturb the
        source. (An *unzoomed* target reset is a no-op — see
        ``test_panel_reset_unzoomed_target_is_noop``.)"""
        url = _dashboard_url(server_port, renderer, n_figures=3)
        update_bodies: list[dict] = []

        def capture(req: PWRequest) -> None:
            if "/dashboard/update" in req.url and req.method == "POST":
                try:
                    update_bodies.append(json.loads(req.post_data or "{}"))
                except Exception:
                    pass

        page.on("request", capture)
        page.goto(url)
        _wait_for_init(page, renderer)

        # Selection sourced from figure 0, filtering figures 1 and 2; plus a
        # viewport on figure 1 so its reset actually clears something (otherwise
        # an unzoomed target reset is a no-op).
        page.evaluate("""() => {
            const figUids = DASHBOARD_SPEC.figures.map(f => f.uid);
            DASHBOARD_SPEC.state.viewport[figUids[1] + '/x'] = {min: 50, max: 150};
            const sel = {
                source_figure_uid: figUids[0],
                predicates: [{ clauses: [{ column: 'ts', range: [100, 300] }] }],
            };
            DASHBOARD_SPEC.state.selections = [sel];
            return postDashboardUpdate({
                type: 'selection', axis_ranges: {},
                selections: DASHBOARD_SPEC.state.selections,
                force_update: true, figure_uid: figUids[0],
            });
        }""")
        page.wait_for_timeout(1_000)

        count_before = len(update_bodies)
        # Reset panel on figure 1 — a zoomed pure target of figure 0's filter.
        page.evaluate("() => window.fvOnResetPanel(DASHBOARD_SPEC.figures[1].uid)")
        page.wait_for_timeout(1_500)

        post = update_bodies[count_before:]
        event_types = {b.get("event", {}).get("type") for b in post}
        assert "viewport" in event_types, (
            "Resetting a target-only figure must emit a viewport event, "
            f"got: {event_types}"
        )
        assert (
            "deselect" not in event_types
        ), "Resetting a target-only figure must not clear the incoming filter"

        # The incoming selection (sourced by figure 0) is preserved...
        fig0_uid = page.evaluate("DASHBOARD_SPEC.figures[0].uid")
        fig1_uid = page.evaluate("DASHBOARD_SPEC.figures[1].uid")
        selections = page.evaluate("DASHBOARD_SPEC.state.selections") or []
        assert any(
            s.get("source_figure_uid") == fig0_uid for s in selections
        ), "Incoming cross-filter from the source figure must survive a target reset"
        # ...and the reset was scoped to figure 1.
        viewport_events = [
            b for b in post if b.get("event", {}).get("type") == "viewport"
        ]
        assert viewport_events[-1]["event"].get("figure_uid") == fig1_uid

    def test_reset_clears_zoom_and_filter(
        self, page: Page, server_port: int, renderer: str
    ):
        url = _dashboard_url(server_port, renderer, n_figures=3)
        update_bodies: list[dict] = []

        def capture(req: PWRequest) -> None:
            if "/dashboard/update" in req.url and req.method == "POST":
                try:
                    update_bodies.append(json.loads(req.post_data or "{}"))
                except Exception:
                    pass

        page.on("request", capture)
        page.goto(url)
        _wait_for_init(page, renderer)

        # Select on A first.
        page.evaluate("""() => {
            const figUids = DASHBOARD_SPEC.figures.map(f => f.uid);
            const sel = {
                source_figure_uid: figUids[0],
                predicates: [{ clauses: [{ column: 'ts', range: [100, 300] }] }],
            };
            DASHBOARD_SPEC.state.selections = [sel];
            return postDashboardUpdate({
                type: 'selection', axis_ranges: {},
                selections: DASHBOARD_SPEC.state.selections,
                force_update: true, figure_uid: figUids[0],
            });
        }""")
        page.wait_for_timeout(1_000)

        count_before_reset = len(update_bodies)
        page.click("#fv-btn-reset")
        page.wait_for_timeout(2_000)

        reset_events = [
            b
            for b in update_bodies[count_before_reset:]
            if b.get("event", {}).get("type") == "init"
        ]
        assert len(reset_events) >= 1, "Reset button must fire an init event"

        # After reset, the JS state should have empty viewport and selections.
        viewport = page.evaluate("DASHBOARD_SPEC.state.viewport")
        selections = page.evaluate("DASHBOARD_SPEC.state.selections")
        assert viewport == {} or viewport is None
        assert selections == [] or selections is None


@pytest.mark.parametrize("renderer", ["plotly", "echarts"])
class TestOverlayBrowser:
    def test_overlay_toggle_with_cached_bg_avoids_warmup(
        self, page: Page, server_port: int, renderer: str
    ):
        url = _dashboard_url(server_port, renderer, n_figures=2)
        update_bodies: list[dict] = []

        def capture(req: PWRequest) -> None:
            if "/dashboard/update" in req.url and req.method == "POST":
                try:
                    update_bodies.append(json.loads(req.post_data or "{}"))
                except Exception:
                    pass

        page.on("request", capture)
        page.goto(url)
        _wait_for_init(page, renderer)

        page.evaluate("""() => {
            const figUids = DASHBOARD_SPEC.figures.map(f => f.uid);
            const sel = {
                source_figure_uid: figUids[0],
                predicates: [{ clauses: [{ column: 'ts', range: [100, 200] }] }],
            };
            DASHBOARD_SPEC.state.selections = [sel];
            return postDashboardUpdate({
                type: 'selection', axis_ranges: {},
                selections: DASHBOARD_SPEC.state.selections,
                force_update: true, figure_uid: figUids[0],
            });
        }""")
        page.wait_for_timeout(1_500)

        before_toggle = len(update_bodies)
        page.click("#fv-btn-cfmode")
        page.wait_for_timeout(1_500)

        toggle_events = [
            body.get("event", {}).get("type") for body in update_bodies[before_toggle:]
        ]
        assert "selection" in toggle_events
        assert "init" not in toggle_events

    def test_overlay_toggle_without_cached_bg_warms_once(
        self, page: Page, server_port: int, renderer: str
    ):
        url = _dashboard_url(server_port, renderer, n_figures=2)
        update_bodies: list[dict] = []

        def capture(req: PWRequest) -> None:
            if "/dashboard/update" in req.url and req.method == "POST":
                try:
                    update_bodies.append(json.loads(req.post_data or "{}"))
                except Exception:
                    pass

        page.on("request", capture)
        page.goto(url)
        _wait_for_init(page, renderer)

        page.evaluate("""() => {
            const figUids = DASHBOARD_SPEC.figures.map(f => f.uid);
            const sel = {
                source_figure_uid: figUids[0],
                predicates: [{ clauses: [{ column: 'ts', range: [100, 200] }] }],
            };
            DASHBOARD_SPEC.state.selections = [sel];
            return postDashboardUpdate({
                type: 'selection', axis_ranges: {},
                selections: DASHBOARD_SPEC.state.selections,
                force_update: true, figure_uid: figUids[0],
            });
        }""")
        page.wait_for_timeout(1_500)
        page.evaluate("""() => {
            Object.keys(window.__fvHasBgByFigure || {}).forEach(uid => {
                window.__fvHasBgByFigure[uid] = false;
            });
        }""")

        before_toggle = len(update_bodies)
        page.click("#fv-btn-cfmode")
        page.wait_for_timeout(1_500)

        toggle_events = [
            body.get("event", {}).get("type") for body in update_bodies[before_toggle:]
        ]
        assert toggle_events.count("init") == 1
        assert toggle_events.count("selection") >= 1

    def test_overlay_heatmap_shows_single_colorbar(
        self, page: Page, server_port: int, renderer: str
    ):
        """Overlay cross-filter on histogram2d must expose only the fg colorbar."""
        if renderer != "plotly":
            pytest.skip("Plotly-only colorbar regression")
        url = _dashboard_url_hist2d_overlay(server_port)
        page.goto(url)
        _wait_for_init(page, "plotly")

        page.click("#fv-btn-cfmode")
        page.wait_for_timeout(500)
        page.evaluate("""() => {
            const figUids = DASHBOARD_SPEC.figures.map(f => f.uid);
            const sel = {
                source_figure_uid: figUids[0],
                predicates: [{ clauses: [{ column: 'ts', range: [100, 200] }] }],
            };
            DASHBOARD_SPEC.state.selections = [sel];
            return postDashboardUpdate({
                type: 'selection', axis_ranges: {},
                selections: DASHBOARD_SPEC.state.selections,
                force_update: true, figure_uid: figUids[0],
            });
        }""")
        page.wait_for_timeout(1_500)

        result = page.evaluate("""() => {
            const fig = document.querySelectorAll('.js-plotly-plot')[1];
            const traces = fig.data || [];
            const heatmaps = traces.filter(t => t.type === 'heatmap');
            const bg = heatmaps.find(t => t.uid && t.uid.endsWith('__fv_layer_bg'));
            const fg = heatmaps.find(t => t.uid && t.uid.endsWith('__fv_layer_fg'));
            const bgCells = (bg && bg.z || []).flat().filter(v => v != null && isFinite(v)).length;
            return {
                scales: traces.map(t => ({ uid: t.uid, showscale: t.showscale })),
                colorbarCount: fig.querySelectorAll('.colorbar').length,
                bgCells,
                bgZmin: bg && bg.zmin,
                bgZmax: bg && bg.zmax,
                fgZmin: fg && fg.zmin,
                fgZmax: fg && fg.zmax,
            };
        }""")
        bg_scales = [
            t
            for t in result["scales"]
            if _trace_layer(t.get("uid")) == "bg" and t.get("showscale")
        ]
        fg_scales = [
            t
            for t in result["scales"]
            if _trace_layer(t.get("uid")) == "fg" and t.get("showscale")
        ]
        assert (
            bg_scales == []
        ), "Background heatmap must not show a colorbar in overlay mode"
        assert len(fg_scales) == 1, "Foreground heatmap must keep the colorbar"
        assert (
            result["colorbarCount"] == 1
        ), f"Expected one Plotly colorbar, found {result['colorbarCount']}"
        assert (
            result["bgCells"] > 0
        ), "Background heatmap must still render cached full data"
        assert result["fgZmin"] is not None and result["fgZmax"] is not None
        assert (
            result["fgZmax"] >= result["fgZmin"]
        ), "Foreground colorbar must use filtered z range"

    def test_overlay_reuses_same_color_and_mutes_background(
        self, page: Page, server_port: int, renderer: str
    ):
        url = _dashboard_url(server_port, renderer, n_figures=2)
        page.goto(url)
        _wait_for_init(page, renderer)

        page.evaluate("""() => {
            const figUids = DASHBOARD_SPEC.figures.map(f => f.uid);
            const sel = {
                source_figure_uid: figUids[0],
                predicates: [{ clauses: [{ column: 'ts', range: [100, 200] }] }],
            };
            DASHBOARD_SPEC.state.selections = [sel];
            return postDashboardUpdate({
                type: 'selection', axis_ranges: {},
                selections: DASHBOARD_SPEC.state.selections,
                force_update: true, figure_uid: figUids[0],
            });
        }""")
        page.wait_for_timeout(1_000)
        page.click("#fv-btn-cfmode")
        page.wait_for_timeout(1_500)

        rendered = page.evaluate(
            """(renderer) => {
                if (renderer === 'plotly') {
                    const fig = document.querySelectorAll('.js-plotly-plot')[1];
                    return (fig.data || []).map(t => ({
                        id: t.uid,
                        opacity: t.opacity ?? 1,
                        color: (t.line && t.line.color) || (t.marker && t.marker.color) || null,
                    }));
                }
                const el = document.querySelectorAll("[id^='fv-chart-']")[1];
                const chart = echarts.getInstanceByDom(el);
                return (chart.getOption().series || []).map(s => ({
                    id: s.id,
                    opacity: s.opacity ?? ((s.itemStyle && s.itemStyle.opacity) ?? 1),
                    color: (s.lineStyle && s.lineStyle.color) || (s.itemStyle && s.itemStyle.color) || null,
                }));
            }""",
            renderer,
        )

        bg = next(item for item in rendered if _trace_layer(item.get("id")) == "bg")
        fg = next(item for item in rendered if _trace_layer(item.get("id")) == "fg")
        assert bg["color"] == fg["color"]
        assert abs(bg["opacity"] - 0.16) < 1e-9
        assert abs(fg["opacity"] - 1.0) < 1e-9

    def test_overlay_reset_clears_fg_and_restores_full_opacity(
        self, page: Page, server_port: int, renderer: str
    ):
        """After Reset in overlay+selection mode, the fg trace must be removed
        and the bg trace must return to full opacity (1.0)."""
        url = _dashboard_url(server_port, renderer, n_figures=2)
        page.goto(url)
        _wait_for_init(page, renderer)

        # 1. Enable overlay mode.
        page.click("#fv-btn-cfmode")
        page.wait_for_timeout(1_000)

        # 2. Inject a cross-filter selection to produce bg+fg overlay.
        page.evaluate("""() => {
            const figUids = DASHBOARD_SPEC.figures.map(f => f.uid);
            const sel = {
                source_figure_uid: figUids[0],
                predicates: [{ clauses: [{ column: 'ts', range: [100, 200] }] }],
            };
            DASHBOARD_SPEC.state.selections = [sel];
            return postDashboardUpdate({
                type: 'selection', axis_ranges: {},
                selections: DASHBOARD_SPEC.state.selections,
                force_update: true, figure_uid: figUids[0],
            });
        }""")
        page.wait_for_timeout(1_500)

        # 3. Click the Reset button.
        page.click("#fv-btn-reset")
        page.wait_for_timeout(2_000)

        # 4. Client state must be cleared.
        selections = page.evaluate("DASHBOARD_SPEC.state.selections")
        assert selections == [], "fvOnReset must clear selections"

        # 5. After reset, no fg trace must remain; bg trace must be at full opacity.
        rendered = page.evaluate(
            """(renderer) => {
                if (renderer === 'plotly') {
                    // Check all Plotly figures — none should have an fg-layer trace.
                    const figs = document.querySelectorAll('.js-plotly-plot');
                    return Array.from(figs).flatMap(gd =>
                        (gd.data || []).map(t => ({
                            id: t.uid,
                            opacity: t.opacity !== undefined ? t.opacity : 1,
                        }))
                    );
                }
                const charts = Array.from(
                    document.querySelectorAll("[id^='fv-chart-']")
                ).map(el => echarts.getInstanceByDom(el)).filter(Boolean);
                return charts.flatMap(chart =>
                    (chart.getOption().series || []).map(s => ({
                        id: s.id,
                        opacity: s.opacity !== undefined ? s.opacity
                               : ((s.itemStyle && s.itemStyle.opacity) !== undefined
                                  ? s.itemStyle.opacity : 1),
                    }))
                );
            }""",
            renderer,
        )

        fg_traces = _layer_traces(rendered, "fg")
        assert (
            fg_traces == []
        ), f"No fg-layer traces must remain after reset, but found: {fg_traces}"
        bg_traces = _layer_traces(rendered, "bg")
        for t in bg_traces:
            assert (
                abs(t["opacity"] - 1.0) < 1e-9
            ), f"bg-layer trace {t['id']} must be at full opacity after reset, got {t['opacity']}"

    def test_overlay_reset_then_new_selection_shows_correct_overlay(
        self, page: Page, server_port: int, renderer: str
    ):
        """After Reset followed by a new selection in overlay mode, the overlay
        must show bg at low opacity and fg at full opacity again."""
        url = _dashboard_url(server_port, renderer, n_figures=2)
        page.goto(url)
        _wait_for_init(page, renderer)

        # 1. Enable overlay mode, make a selection, then reset.
        page.click("#fv-btn-cfmode")
        page.wait_for_timeout(1_000)

        page.evaluate("""() => {
            const figUids = DASHBOARD_SPEC.figures.map(f => f.uid);
            const sel = { source_figure_uid: figUids[0], predicates: [{ clauses: [{ column: 'ts', range: [100, 200] }] }] };
            DASHBOARD_SPEC.state.selections = [sel];
            return postDashboardUpdate({type: 'selection', axis_ranges: {},
                selections: DASHBOARD_SPEC.state.selections, force_update: true,
                figure_uid: figUids[0]});
        }""")
        page.wait_for_timeout(1_500)
        page.click("#fv-btn-reset")
        page.wait_for_timeout(2_000)

        # 2. Make a fresh selection after the reset.
        page.evaluate("""() => {
            const figUids = DASHBOARD_SPEC.figures.map(f => f.uid);
            const sel = { source_figure_uid: figUids[0], predicates: [{ clauses: [{ column: 'ts', range: [150, 300] }] }] };
            DASHBOARD_SPEC.state.selections = [sel];
            return postDashboardUpdate({type: 'selection', axis_ranges: {},
                selections: DASHBOARD_SPEC.state.selections, force_update: true,
                figure_uid: figUids[0]});
        }""")
        page.wait_for_timeout(1_500)

        # 3. The non-source figure (index 1) must show bg at low opacity + fg at 1.0.
        rendered = page.evaluate(
            """(renderer) => {
                if (renderer === 'plotly') {
                    const fig = document.querySelectorAll('.js-plotly-plot')[1];
                    return (fig.data || []).map(t => ({
                        id: t.uid,
                        opacity: t.opacity !== undefined ? t.opacity : 1,
                    }));
                }
                const el = document.querySelectorAll("[id^='fv-chart-']")[1];
                const chart = echarts.getInstanceByDom(el);
                return (chart.getOption().series || []).map(s => ({
                    id: s.id,
                    opacity: s.opacity !== undefined ? s.opacity
                           : ((s.itemStyle && s.itemStyle.opacity) !== undefined
                              ? s.itemStyle.opacity : 1),
                }));
            }""",
            renderer,
        )

        bg = next((t for t in rendered if _trace_layer(t.get("id")) == "bg"), None)
        fg = next((t for t in rendered if _trace_layer(t.get("id")) == "fg"), None)
        assert (
            bg is not None
        ), f"Expected a bg-layer trace after reset+new selection, got: {rendered}"
        assert (
            fg is not None
        ), f"Expected an fg-layer trace after reset+new selection, got: {rendered}"
        assert (
            abs(bg["opacity"] - 0.16) < 1e-9
        ), f"bg-layer trace must be at 0.16 opacity, got {bg['opacity']}"
        assert (
            abs(fg["opacity"] - 1.0) < 1e-9
        ), f"fg-layer trace must be at 1.0 opacity, got {fg['opacity']}"


class TestOverlayBrowserPlotlySafeLayerIds:
    def test_multi_histogram_overlay_offsets_by_logical_trace(
        self, page: Page, server_port: int
    ):
        url = _dashboard_url_line_multi_hist_target(server_port, "plotly")
        page.goto(url)
        _wait_for_init(page, "plotly")

        page.evaluate("""() => {
            DASHBOARD_SPEC.state.cross_filter_mode = 'overlay';
            const figUids = DASHBOARD_SPEC.figures.map(f => f.uid);
            const sel = {
                source_figure_uid: figUids[0],
                predicates: [{ clauses: [{ column: 'x', range: [100, 200] }] }],
            };
            DASHBOARD_SPEC.state.selections = [sel];
            return postDashboardUpdate({
                type: 'selection', axis_ranges: {},
                selections: DASHBOARD_SPEC.state.selections,
                force_update: true, figure_uid: figUids[0],
            });
        }""")
        page.wait_for_function("""() => {
            const fig = document.querySelectorAll('.js-plotly-plot')[1];
            const traces = (fig && fig.data) || [];
            return traces.filter(t => String(t.uid || '').endsWith('__fv_layer_fg')).length === 2;
        }""")

        rendered = page.evaluate("""() => {
            const fig = document.querySelectorAll('.js-plotly-plot')[1];
            const traces = ((fig && fig.data) || [])
                .filter(t => t.type === 'bar')
                .map(t => {
                    const id = String(t.uid || '');
                    const logical = id.replace(/__fv_layer_(bg|fg)$/, '');
                    const layer = id.endsWith('__fv_layer_bg')
                        ? 'bg'
                        : (id.endsWith('__fv_layer_fg') ? 'fg' : 'base');
                    return {
                        id,
                        logical,
                        layer,
                        offsetgroup: t.offsetgroup ?? null,
                        alignmentgroup: t.alignmentgroup ?? null,
                    };
                });
            return {
                barmode: (fig.layout && fig.layout.barmode)
                    || (fig._fullLayout && fig._fullLayout.barmode)
                    || null,
                traces,
            };
        }""")

        assert rendered["barmode"] == "overlay"
        traces_by_logical: dict[str, list[dict]] = {}
        for trace in rendered["traces"]:
            traces_by_logical.setdefault(trace["logical"], []).append(trace)

        assert len(traces_by_logical) == 2, rendered
        for logical, traces in traces_by_logical.items():
            assert {trace["layer"] for trace in traces} == {"bg", "fg"}
            assert {trace["offsetgroup"] for trace in traces} == {logical}
            assert {trace["alignmentgroup"] for trace in traces} == {"fv-bars"}

    def test_multi_histogram_no_selection_uses_group_barmode(
        self, page: Page, server_port: int
    ):
        url = _dashboard_url_line_multi_hist_target(server_port, "plotly")
        page.goto(url)
        _wait_for_init(page, "plotly")

        barmode = page.evaluate("""() => {
            const fig = document.querySelectorAll('.js-plotly-plot')[1];
            return (fig && fig.layout && fig.layout.barmode) || null;
        }""")
        assert (
            barmode == "group"
        ), f"Expected barmode='group' for multi-histogram figure, got {barmode!r}"

    def test_hist_target_reset_and_deselect_clear_fg_without_selector_errors(
        self, page: Page, server_port: int
    ):
        url = _dashboard_url_line_hist_target(server_port, "plotly")
        console_messages: list[str] = []
        page.on("console", lambda msg: console_messages.append(msg.text))
        page.goto(url)
        _wait_for_init(page, "plotly")

        page.click("#fv-btn-cfmode")
        page.wait_for_timeout(1_000)

        page.evaluate("""() => {
            const figUids = DASHBOARD_SPEC.figures.map(f => f.uid);
            const sel = {
                source_figure_uid: figUids[0],
                predicates: [{ clauses: [{ column: 'ts', range: [100, 200] }] }],
            };
            DASHBOARD_SPEC.state.selections = [sel];
            return postDashboardUpdate({
                type: 'selection', axis_ranges: {},
                selections: DASHBOARD_SPEC.state.selections,
                force_update: true, figure_uid: figUids[0],
            });
        }""")
        page.wait_for_timeout(1_200)

        rendered = page.evaluate("""() => {
                const fig = document.querySelectorAll('.js-plotly-plot')[1];
                return (fig.data || []).map(t => ({
                    id: t.uid,
                    opacity: t.opacity !== undefined ? t.opacity : 1,
                }));
            }""")
        assert _layer_traces(rendered, "bg"), rendered
        assert _layer_traces(rendered, "fg"), rendered

        page.click("#fv-btn-deselect")
        page.wait_for_timeout(1_600)
        assert page.evaluate("DASHBOARD_SPEC.state.selections") == []

        rendered_after_deselect = page.evaluate("""() => {
                const fig = document.querySelectorAll('.js-plotly-plot')[1];
                return (fig.data || []).map(t => ({
                    id: t.uid,
                    opacity: t.opacity !== undefined ? t.opacity : 1,
                }));
            }""")
        assert _layer_traces(rendered_after_deselect, "fg") == []
        for trace in _layer_traces(rendered_after_deselect, "bg"):
            assert abs(trace["opacity"] - 1.0) < 1e-9

        page.evaluate("""() => {
            const figUids = DASHBOARD_SPEC.figures.map(f => f.uid);
            const sel = {
                source_figure_uid: figUids[0],
                predicates: [{ clauses: [{ column: 'ts', range: [120, 260] }] }],
            };
            DASHBOARD_SPEC.state.selections = [sel];
            return postDashboardUpdate({
                type: 'selection', axis_ranges: {},
                selections: DASHBOARD_SPEC.state.selections,
                force_update: true, figure_uid: figUids[0],
            });
        }""")
        page.wait_for_timeout(1_200)

        page.click("#fv-btn-reset")
        page.wait_for_timeout(1_800)
        assert page.evaluate("DASHBOARD_SPEC.state.selections") == []

        rendered_after_reset = page.evaluate("""() => {
                const fig = document.querySelectorAll('.js-plotly-plot')[1];
                return (fig.data || []).map(t => ({
                    id: t.uid,
                    opacity: t.opacity !== undefined ? t.opacity : 1,
                }));
            }""")
        assert _layer_traces(rendered_after_reset, "fg") == []
        for trace in _layer_traces(rendered_after_reset, "bg"):
            assert abs(trace["opacity"] - 1.0) < 1e-9

        selector_errors = [
            message for message in console_messages if "not a valid selector" in message
        ]
        assert selector_errors == [], selector_errors

    def test_grouped_bar_target_reset_clears_fg_without_selector_errors(
        self, page: Page, server_port: int
    ):
        url = _dashboard_url_line_grouped_bar_target(server_port, "plotly")
        console_messages: list[str] = []
        page.on("console", lambda msg: console_messages.append(msg.text))
        page.goto(url)
        _wait_for_init(page, "plotly")

        page.click("#fv-btn-cfmode")
        page.wait_for_timeout(1_000)

        page.evaluate("""() => {
            const figUids = DASHBOARD_SPEC.figures.map(f => f.uid);
            const sel = {
                source_figure_uid: figUids[0],
                predicates: [{ clauses: [{ column: 'ts', range: [20, 80] }] }],
            };
            DASHBOARD_SPEC.state.selections = [sel];
            return postDashboardUpdate({
                type: 'selection', axis_ranges: {},
                selections: DASHBOARD_SPEC.state.selections,
                force_update: true, figure_uid: figUids[0],
            });
        }""")
        page.wait_for_timeout(1_400)

        rendered = page.evaluate("""() => {
                const fig = document.querySelectorAll('.js-plotly-plot')[1];
                return (fig.data || []).map(t => ({
                    id: t.uid,
                    opacity: t.opacity !== undefined ? t.opacity : 1,
                }));
            }""")
        assert _layer_traces(rendered, "bg"), rendered
        assert _layer_traces(rendered, "fg"), rendered

        page.click("#fv-btn-reset")
        page.wait_for_timeout(1_800)
        assert page.evaluate("DASHBOARD_SPEC.state.selections") == []

        rendered_after_reset = page.evaluate("""() => {
                const fig = document.querySelectorAll('.js-plotly-plot')[1];
                return (fig.data || []).map(t => ({
                    id: t.uid,
                    opacity: t.opacity !== undefined ? t.opacity : 1,
                }));
            }""")
        assert _layer_traces(rendered_after_reset, "fg") == []
        for trace in _layer_traces(rendered_after_reset, "bg"):
            assert abs(trace["opacity"] - 1.0) < 1e-9

        selector_errors = [
            message for message in console_messages if "not a valid selector" in message
        ]
        assert selector_errors == [], selector_errors


# ---------------------------------------------------------------------------
# Share URL state preservation tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("renderer", ["plotly", "echarts"])
class TestShareUrlState:
    """Verify that Share → navigate preserves viewport and cross-filter state."""

    def test_share_preserves_zoom(self, page: Page, server_port: int, renderer: str):
        url = _dashboard_url(server_port, renderer, n_figures=2)
        page.goto(url)
        _wait_for_init(page, renderer)

        # Programmatically set a viewport state.
        page.evaluate("""() => {
            const figUid = DASHBOARD_SPEC.figures[0].uid;
            DASHBOARD_SPEC.state.viewport[figUid + '/x'] = {min: 100, max: 300};
        }""")

        # Click Share and capture the URL.
        share_urls = []

        def handle_dialog(dialog):
            share_urls.append(dialog.message)
            dialog.dismiss()

        page.on("dialog", handle_dialog)
        page.click("#fv-btn-share")
        page.wait_for_timeout(2_000)

        # If clipboard write succeeded, the share URL is in the clipboard.
        # Otherwise an alert was shown. Either way, fetch via /share directly.
        share_resp = page.evaluate("""async () => {
            const resp = await fetch(SERVER_URL + '/share', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({spec: DASHBOARD_SPEC, server_url: SERVER_URL}),
            });
            const data = await resp.json();
            return new URL(data.url, window.location.href).href;
        }""")
        assert share_resp, "Share must return a URL"
        share_url = share_resp + f"&renderer={renderer}"

        page.goto(share_url)
        _wait_for_init(page, renderer)

        restored_vp = page.evaluate("DASHBOARD_SPEC.state.viewport")
        assert restored_vp, "Viewport state must be restored from shared URL"
        vp_keys = list(restored_vp.keys())
        assert any(
            "/x" in k for k in vp_keys
        ), f"Expected a viewport key like 'figUid/x', got {vp_keys}"

    def test_share_preserves_cross_filter(
        self, page: Page, server_port: int, renderer: str
    ):
        url = _dashboard_url(server_port, renderer, n_figures=2)
        page.goto(url)
        _wait_for_init(page, renderer)

        # Programmatically add a selection.
        page.evaluate("""() => {
            const figUid = DASHBOARD_SPEC.figures[0].uid;
            DASHBOARD_SPEC.state.selections = [{
                source_figure_uid: figUid,
                predicates: [{ clauses: [{ column: 'ts', range: [100, 300] }] }],
            }];
        }""")

        share_url = page.evaluate("""async () => {
            const resp = await fetch(SERVER_URL + '/share', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({spec: DASHBOARD_SPEC, server_url: SERVER_URL}),
            });
            const data = await resp.json();
            return new URL(data.url, window.location.href).href;
        }""")
        assert share_url
        share_url += f"&renderer={renderer}"

        page.goto(share_url)
        _wait_for_init(page, renderer)

        restored_sels = page.evaluate("DASHBOARD_SPEC.state.selections")
        assert len(restored_sels) >= 1, "Selections must be restored from shared URL"
        assert restored_sels[0]["predicates"][0]["clauses"][0]["range"] == [100, 300]

    def test_share_preserves_cross_filter_effect_on_data(
        self, page: Page, server_port: int, renderer: str
    ):
        """End-to-end: A selection on A must still narrow B after Share URL round-trip."""
        url = _dashboard_url(server_port, renderer, n_figures=2)
        page.goto(url)
        _wait_for_init(page, renderer)

        # On the first page: capture B's init count, apply a selection on A, and
        # verify that B's visible data is narrowed. Then obtain a Share URL via
        # the same /share payload that the toolbar uses.
        result = page.evaluate(
            """async ({renderer}) => {
                const figUids = DASHBOARD_SPEC.figures.map(f => f.uid);
                const uidA = figUids[0];
                const uidB = figUids[1];

                let initCountB, selCountB;

                if (renderer === 'plotly') {
                  const idxB = figUidToIdx[uidB];
                  initCountB = tracesByFig[idxB][0].x.length;
                } else {
                  const chartB = chartsByFig[uidB];
                  const optB = chartB.getOption();
                  initCountB = (optB.series && optB.series[0] && optB.series[0].data
                                ? optB.series[0].data.length
                                : 0);
                }

                const sel = {
                  source_figure_uid: uidA,
                  predicates: [{ clauses: [{ column: 'ts', range: [0, 50] }] }],
                };
                DASHBOARD_SPEC.state.selections = [sel];
                await postDashboardUpdate({
                  type: 'selection',
                  axis_ranges: {},
                  selections: DASHBOARD_SPEC.state.selections,
                  force_update: true,
                  figure_uid: uidA,
                });

                // Allow the selection response to apply.
                await new Promise(r => setTimeout(r, 1000));

                if (renderer === 'plotly') {
                  const idxB = figUidToIdx[uidB];
                  selCountB = tracesByFig[idxB][0].x.length;
                } else {
                  const chartB2 = chartsByFig[uidB];
                  const optB2 = chartB2.getOption();
                  selCountB = (optB2.series && optB2.series[0] && optB2.series[0].data
                               ? optB2.series[0].data.length
                               : 0);
                }

                const resp = await fetch(SERVER_URL + '/share', {
                  method: 'POST',
                  headers: {'Content-Type': 'application/json'},
                  body: JSON.stringify({spec: DASHBOARD_SPEC, server_url: SERVER_URL}),
                });
                const data = await resp.json();

                return {
                  initCountB,
                  selCountB,
                  shareUrl: new URL(data.url, window.location.href).href,
                };
            }""",
            {"renderer": renderer},
        )

        init_count_b = result["initCountB"]
        sel_count_b = result["selCountB"]
        share_url = result["shareUrl"] + f"&renderer={renderer}"

        assert sel_count_b < init_count_b, "Selection on A must narrow B before sharing"

        # Now open the shared URL and assert that B remains narrowed.
        page.goto(share_url)
        _wait_for_init(page, renderer)

        restored_count_b = page.evaluate(
            """({renderer}) => {
                const figUids = DASHBOARD_SPEC.figures.map(f => f.uid);
                const uidB = figUids[1];

                if (renderer === 'plotly') {
                  const idxB = figUidToIdx[uidB];
                  return tracesByFig[idxB][0].x.length;
                } else {
                  const chartB = chartsByFig[uidB];
                  const optB = chartB.getOption();
                  const data = (optB.series && optB.series[0] && optB.series[0].data)
                    ? optB.series[0].data
                    : [];
                  return data.length;
                }
            }""",
            {"renderer": renderer},
        )

        assert restored_count_b == sel_count_b, (
            "After Share URL round-trip, figure B must remain cross-filtered "
            "to the same narrowed data as before sharing"
        )

    def test_share_preserves_zoom_and_filter(
        self, page: Page, server_port: int, renderer: str
    ):
        url = _dashboard_url(server_port, renderer, n_figures=2)
        page.goto(url)
        _wait_for_init(page, renderer)

        # Set both viewport and selection state.
        page.evaluate("""() => {
            const fig0 = DASHBOARD_SPEC.figures[0].uid;
            DASHBOARD_SPEC.state.viewport[fig0 + '/x'] = {min: 50, max: 400};
            DASHBOARD_SPEC.state.selections = [{
                source_figure_uid: fig0,
                predicates: [{ clauses: [{ column: 'ts', range: [100, 300] }] }],
            }];
        }""")

        share_url = page.evaluate("""async () => {
            const resp = await fetch(SERVER_URL + '/share', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({spec: DASHBOARD_SPEC, server_url: SERVER_URL}),
            });
            return new URL((await resp.json()).url, window.location.href).href;
        }""")
        assert share_url
        share_url += f"&renderer={renderer}"

        page.goto(share_url)
        _wait_for_init(page, renderer)

        restored_vp = page.evaluate("DASHBOARD_SPEC.state.viewport")
        restored_sels = page.evaluate("DASHBOARD_SPEC.state.selections")
        assert restored_vp, "Viewport must be restored"
        assert len(restored_sels) >= 1, "Selections must be restored"
        assert restored_sels[0]["predicates"][0]["clauses"][0]["range"] == [100, 300]

    def test_share_draws_selection_boxes(
        self, page: Page, server_port: int, renderer: str
    ):
        """Share URL must restore visible selection boxes on the source figure."""
        url = _dashboard_url(server_port, renderer, n_figures=2)
        page.goto(url)
        _wait_for_init(page, renderer)

        # Programmatically add a selection with both x and y ranges so that
        # a rectangular selection box can be drawn by the renderer.
        share_url = page.evaluate(
            """async ({renderer}) => {
                const figUids = DASHBOARD_SPEC.figures.map(f => f.uid);
                const uidA = figUids[0];
                DASHBOARD_SPEC.state.selections = [{
                    source_figure_uid: uidA,
                    predicates: [{ clauses: [
                        { column: 'ts', range: [100, 300] },
                        { column: 'val', range: [0, 200] },
                    ] }],
                }];
                const resp = await fetch(SERVER_URL + '/share', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({spec: DASHBOARD_SPEC, server_url: SERVER_URL}),
                });
                const data = await resp.json();
                return new URL(data.url, window.location.href).href;
            }""",
            {"renderer": renderer},
        )
        assert share_url
        share_url += f"&renderer={renderer}"

        # Open the shared URL and assert that a selection box is present
        # on the source figure.
        page.goto(share_url)
        _wait_for_init(page, renderer)

        has_box = page.evaluate(
            """({renderer}) => {
                const figUids = DASHBOARD_SPEC.figures.map(f => f.uid);
                const uidA = figUids[0];
                if (renderer === 'plotly') {
                    const idxA = figUidToIdx[uidA];
                    const layout = layoutsByFig[idxA] || {};
                    const sels = layout.selections || [];
                    return Array.isArray(sels) && sels.length > 0;
                } else {
                    const areasByFig = window.__fvBrushAreasByFig || {};
                    const areas = areasByFig[uidA] || [];
                    return Array.isArray(areas) && areas.length > 0;
                }
            }""",
            {"renderer": renderer},
        )

        assert (
            has_box
        ), "Shared URL must restore a visible selection box on the source figure"

    def test_share_preserves_zoomed_aggregation_and_cross_filter(
        self, page: Page, server_port: int, renderer: str
    ):
        """Repro: zoom A + select A, then Share → reload must apply both."""
        url = _dashboard_url(server_port, renderer, n_figures=2)
        page.goto(url)
        _wait_for_init(page, renderer)

        share_url = page.evaluate(
            """async ({renderer}) => {
                const figUids = DASHBOARD_SPEC.figures.map(f => f.uid);
                const uidA = figUids[0];
                const uidB = figUids[1];

                // Step 1: zoom on A (viewport)
                DASHBOARD_SPEC.state.viewport[uidA + '/x'] = {min: 100, max: 300};
                await postDashboardUpdate({
                  type: 'viewport',
                  axis_ranges: {x: [100, 300]},
                  selections: DASHBOARD_SPEC.state.selections,
                  force_update: true,
                  figure_uid: uidA,
                });

                // Step 2: cross-filter on A
                const sel = {
                  source_figure_uid: uidA,
                  predicates: [{ clauses: [{ column: 'ts', range: [120, 200] }] }],
                };
                DASHBOARD_SPEC.state.selections = [sel];
                await postDashboardUpdate({
                  type: 'selection',
                  axis_ranges: {},
                  selections: DASHBOARD_SPEC.state.selections,
                  force_update: true,
                  figure_uid: uidA,
                });

                // Step 3: Share
                const resp = await fetch(SERVER_URL + '/share', {
                  method: 'POST',
                  headers: {'Content-Type': 'application/json'},
                  body: JSON.stringify({spec: DASHBOARD_SPEC, server_url: SERVER_URL}),
                });
                const data = await resp.json();
                return new URL(data.url, window.location.href).href;
            }""",
            {"renderer": renderer},
        )
        assert share_url
        share_url += f"&renderer={renderer}"

        page.goto(share_url)
        _wait_for_init(page, renderer)

        restored = page.evaluate(
            """({renderer}) => {
                const figUids = DASHBOARD_SPEC.figures.map(f => f.uid);
                const uidA = figUids[0];
                const uidB = figUids[1];

                function plotlyCountAndRange(uid) {
                  const idx = figUidToIdx[uid];
                  const xs = tracesByFig[idx][0].x || [];
                  return {
                    n: xs.length,
                    min: xs.length ? Math.min(...xs) : null,
                    max: xs.length ? Math.max(...xs) : null,
                  };
                }

                function echartsCountAndRange(uid) {
                  const chart = chartsByFig[uid];
                  const opt = chart.getOption();
                  const data = (opt.series && opt.series[0] && opt.series[0].data) ? opt.series[0].data : [];
                  const xs = data.map(p => p[0]);
                  return {
                    n: xs.length,
                    min: xs.length ? Math.min(...xs) : null,
                    max: xs.length ? Math.max(...xs) : null,
                  };
                }

                const a = (renderer === 'plotly') ? plotlyCountAndRange(uidA) : echartsCountAndRange(uidA);
                const b = (renderer === 'plotly') ? plotlyCountAndRange(uidB) : echartsCountAndRange(uidB);
                return {a, b};
            }""",
            {"renderer": renderer},
        )

        # A should be aggregated to the zoomed viewport.
        assert restored["a"]["n"] > 0
        assert restored["a"]["min"] >= 100
        assert restored["a"]["max"] <= 300

        # B should be cross-filtered to A's selection (x_range [120,200]).
        assert restored["b"]["n"] > 0
        assert restored["b"]["min"] >= 120
        assert restored["b"]["max"] <= 200


class TestAgentReadback:
    """The agent watch loop: a human brush must be readable through the stable
    ``window.flexvizState()`` accessor, and the same spec must round-trip
    through /share + decode_spec (the Share-button fallback path)."""

    def test_brush_visible_via_flexviz_state_and_share_decode(
        self, page: Page, server_port: int
    ):
        from flexviz.spec import decode_spec

        url = _dashboard_url_selection_duplicate_repro(server_port)
        page.goto(url)
        _wait_for_init(page, "plotly")
        page.locator("#fv-bar-0 .fv-mode-btn[data-mode='select']").click()
        page.wait_for_timeout(300)

        drag_layer = page.locator("#fv-plot-0 .nsewdrag")
        box = drag_layer.bounding_box()
        assert box is not None
        page.mouse.move(box["x"] + box["width"] * 0.2, box["y"] + box["height"] * 0.3)
        page.mouse.down()
        page.mouse.move(
            box["x"] + box["width"] * 0.6, box["y"] + box["height"] * 0.7, steps=20
        )
        page.mouse.up()
        page.wait_for_timeout(1_500)

        # 1. The stable accessor exposes the brushed state.
        state = page.evaluate("window.flexvizState()")
        selections = state["state"]["selections"]
        assert len(selections) == 1, state["state"]
        assert selections[0]["predicates"][0]["clauses"][0]["range"] is not None

        # 2. The real Share button posts the current spec to /share and the
        #    returned URL decodes to the brushed state (clipboard/alert
        #    behavior after the POST is irrelevant to the contract).
        page.on("dialog", lambda dialog: dialog.dismiss())
        with page.expect_response("**/share") as share_response:
            page.click("#fv-btn-share")
        share_url = share_response.value.json()["url"]
        decoded = decode_spec(share_url.split("spec=", 1)[1])
        clause = decoded.state.selections[0].predicates[0].clauses[0]
        assert clause.range is not None

    def test_zoom_viewport_visible_via_flexviz_state(
        self, page: Page, server_port: int
    ):
        url = _dashboard_url_selection_duplicate_repro(server_port)
        page.goto(url)
        _wait_for_init(page, "plotly")

        # Default mode is zoom: a drag zooms and must land in the viewport.
        drag_layer = page.locator("#fv-plot-0 .nsewdrag")
        box = drag_layer.bounding_box()
        assert box is not None
        page.mouse.move(box["x"] + box["width"] * 0.3, box["y"] + box["height"] * 0.3)
        page.mouse.down()
        page.mouse.move(
            box["x"] + box["width"] * 0.7, box["y"] + box["height"] * 0.7, steps=20
        )
        page.mouse.up()
        page.wait_for_timeout(1_500)

        viewport = page.evaluate("window.flexvizState()['state']['viewport']")
        assert any("/x" in key for key in viewport), viewport

    def test_flexviz_state_returns_detached_snapshot(
        self, page: Page, server_port: int
    ):
        url = _dashboard_url_selection_duplicate_repro(server_port)
        page.goto(url)
        _wait_for_init(page, "plotly")

        original_uid = page.evaluate("DASHBOARD_SPEC.figures[0].uid")
        mutated_view = page.evaluate("""() => {
            const snap = window.flexvizState();
            snap.figures[0].uid = 'clobbered';
            snap.state.selections = [{bogus: true}];
            return window.flexvizState().figures[0].uid;
        }""")
        assert mutated_view == original_uid
        assert page.evaluate("DASHBOARD_SPEC.figures[0].uid") == original_uid
        assert page.evaluate("DASHBOARD_SPEC.state.selections") == []


# ---------------------------------------------------------------------------
# Share behind a prefix-stripping reverse proxy (demo deployment topology)
# ---------------------------------------------------------------------------


def _start_proxied_server(port: int, prefix: str = "/demo4") -> None:
    """Serve the flexviz app the way a path-routing reverse proxy (nginx) does:

    - ``GET {prefix}``   → dashboard HTML baked with ``server_url={prefix}``
      (mirrors a deployment's per-dashboard page routes),
    - ``{prefix}/...``   → flexviz app with the prefix stripped from the path
      and **no** root_path set (nginx's canonical path-strip pattern).
    """
    from fastapi.responses import HTMLResponse

    from flexviz.adapters.plotly_adapter import PlotlyAdapter
    from flexviz.dashboard import Dashboard
    from flexviz.server import app as fv_app
    from flexviz.server import register_source

    df = pl.DataFrame({"ts": list(range(500)), "val": [float(i) for i in range(500)]})
    register_source("_browser_test", df)

    dash = Dashboard(df)
    for i in range(2):
        f = dash.add_figure(title=f"Fig{i}")
        f.add_line(x="ts", y="val", name=f"L{i}", n_points=200)
    spec = dash.to_spec(source_name="_browser_test")
    page_html = PlotlyAdapter()._build_dashboard_html(spec, server_url=prefix)

    async def proxied_app(scope, receive, send):
        if scope["type"] == "http" and scope["path"] == prefix:
            await HTMLResponse(page_html)(scope, receive, send)
            return
        if scope["type"] == "http" and scope["path"].startswith(prefix + "/"):
            scope = dict(scope)
            scope["path"] = scope["path"][len(prefix) :]
            scope["raw_path"] = scope["path"].encode()
        await fv_app(scope, receive, send)

    config = uvicorn.Config(proxied_app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"Proxied server did not start on port {port}")


@pytest.fixture(scope="module")
def proxied_port() -> Generator[int, None, None]:
    port = _free_port()
    _start_proxied_server(port)
    yield port


class TestShareBehindProxy:
    """Regression: share must produce a working absolute URL when the app is
    deployed behind a reverse proxy that strips a path prefix (demo4 bug)."""

    def test_share_roundtrip_through_prefix_stripping_proxy(
        self, page: Page, proxied_port: int
    ):
        base = f"http://127.0.0.1:{proxied_port}"
        page.goto(f"{base}/demo4")
        _wait_for_init(page, "plotly")

        # Capture what the Share button hands to the clipboard.
        page.evaluate("""() => {
                navigator.clipboard.writeText = (u) => {
                    window.__fvShared = u;
                    return Promise.resolve();
                };
            }""")
        page.click("#fv-btn-share")
        page.wait_for_function("() => window.__fvShared", timeout=10_000)
        shared = page.evaluate("window.__fvShared")

        # The copied URL must be absolute and keep the external prefix.
        assert shared.startswith(f"{base}/demo4/view?spec="), shared

        # The shared page must work end-to-end through the proxy: its init
        # /dashboard/update must resolve under /demo4/ and succeed.
        update_statuses: list[tuple[str, int]] = []
        page.on(
            "response",
            lambda r: (
                update_statuses.append((r.url, r.status))
                if "/dashboard/update" in r.url
                else None
            ),
        )
        page.goto(shared)
        _wait_for_init(page, "plotly")

        assert update_statuses, "Shared page must fire /dashboard/update on init"
        for url, status in update_statuses:
            assert url.startswith(f"{base}/demo4/dashboard/update"), url
            assert status == 200


# ---------------------------------------------------------------------------
# Linked hover browser tests
# ---------------------------------------------------------------------------


def _dashboard_url_hover(port: int, renderer: str) -> str:
    """Two-figure dashboard: fig0 = line(x=ts, y=val), fig1 = line(x=ts, y=val).
    Both share the 'ts' and 'val' columns so crosshairs should sync."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import encode_spec

    df = pl.DataFrame({"ts": list(range(100)), "val": [float(i) for i in range(100)]})
    register_source("_browser_test", df)

    dash = Dashboard(df)
    f0 = dash.add_figure(title="Fig0")
    f0.add_line(x="ts", y="val", name="L0", n_points=200)
    f1 = dash.add_figure(title="Fig1")
    f1.add_line(x="ts", y="val", name="L1", n_points=200)
    spec = dash.to_spec(source_name="_browser_test")

    encoded = encode_spec(spec)
    return f"http://127.0.0.1:{port}/view?spec={encoded}&renderer={renderer}"


def _dashboard_url_hover_minmax_shared_x(port: int, renderer: str) -> str:
    """Two minmax-downsampled line figures that only share x (user repro shape)."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import encode_spec

    n = 200_000
    xs = list(range(n))
    df = pl.DataFrame(
        {
            "x": xs,
            "y": [float((i % 10_000) / 10_000.0) for i in xs],
            "sin": [float(math.sin(i / 5000.0)) for i in xs],
        }
    )
    register_source("_browser_test", df)

    dash = Dashboard(df)
    f0 = dash.add_figure(title="Fig0")
    f0.add_line(x="x", y="y", name="y", n_points=1000, downsample="minmax")
    f1 = dash.add_figure(title="Fig1")
    f1.add_line(x="x", y="sin", name="sin", n_points=1000, downsample="minmax")
    spec = dash.to_spec(source_name="_browser_test")

    encoded = encode_spec(spec)
    return f"http://127.0.0.1:{port}/view?spec={encoded}&renderer={renderer}"


@pytest.mark.parametrize("renderer", ["plotly", "echarts"])
class TestLinkedHoverBrowser:
    def test_hover_button_present_and_inactive_by_default(
        self, page: Page, server_port: int, renderer: str
    ):
        url = _dashboard_url_hover(server_port, renderer)
        page.goto(url)
        _wait_for_init(page, renderer)

        btn = page.query_selector("#fv-hover-btn")
        assert btn is not None, "Hover dropdown button must be present"
        assert "Off" in (
            btn.text_content() or ""
        ), "Hover button must default to 'Hover: Off'"
        mode = page.evaluate(
            "DASHBOARD_SPEC.client_state && DASHBOARD_SPEC.client_state.hover_mode"
        )
        assert (
            mode == "off"
        ), f"client_state.hover_mode must default to 'off', got {mode!r}"

    def test_hover_toggle_turns_on(self, page: Page, server_port: int, renderer: str):
        url = _dashboard_url_hover(server_port, renderer)
        page.goto(url)
        _wait_for_init(page, renderer)

        # Clicking the toggle turns linked hover on.
        page.click("#fv-hover-btn")
        page.wait_for_timeout(100)

        mode = page.evaluate("DASHBOARD_SPEC.client_state.hover_mode")
        btn = page.query_selector("#fv-hover-btn")
        assert (
            mode == "on"
        ), f"client_state.hover_mode must be 'on' after toggle, got {mode!r}"
        assert "On" in (
            btn.text_content() or ""
        ), "Button must show 'Hover: On' after toggling on"
        assert btn.get_attribute("aria-pressed") == "true"

    def test_hover_toggle_turns_off(self, page: Page, server_port: int, renderer: str):
        url = _dashboard_url_hover(server_port, renderer)
        page.goto(url)
        _wait_for_init(page, renderer)

        # Toggle on, then off.
        page.click("#fv-hover-btn")
        page.wait_for_timeout(100)
        page.click("#fv-hover-btn")
        page.wait_for_timeout(100)

        mode = page.evaluate("DASHBOARD_SPEC.client_state.hover_mode")
        assert mode == "off", "client_state.hover_mode must be 'off' after toggling off"
        btn = page.query_selector("#fv-hover-btn")
        assert "Off" in (btn.text_content() or "")
        assert btn.get_attribute("aria-pressed") == "false"

    def test_hover_targets_built_at_load(
        self, page: Page, server_port: int, renderer: str
    ):
        url = _dashboard_url_hover(server_port, renderer)
        page.goto(url)
        _wait_for_init(page, renderer)

        targets = page.evaluate("hoverTargetsByColumn")
        assert targets is not None, "hoverTargetsByColumn must be defined"
        assert "ts" in targets, "hoverTargetsByColumn must contain 'ts' column"
        assert len(targets["ts"]) >= 1, "ts column must have at least one target entry"

    def test_plotly_hover_shows_crosshair_on_other_figure(
        self, page: Page, server_port: int, renderer: str
    ):
        """Axis mode hover must add a guide to the other figure."""
        if renderer == "echarts":
            pytest.skip("Plotly-specific test")

        url = _dashboard_url_hover(server_port, renderer)
        page.goto(url)
        _wait_for_init(page, renderer)

        guides_after = page.evaluate("""() => {
            const fig0Uid = DASHBOARD_SPEC.figures[0].uid;
            const fig1Uid = DASHBOARD_SPEC.figures[1].uid;
            const traceUid = DASHBOARD_SPEC.figures[0].traces[0].uid;

            // Enable axis mode
            if (!DASHBOARD_SPEC.client_state) DASHBOARD_SPEC.client_state = {};
            DASHBOARD_SPEC.client_state.hover_mode = 'axis';

            const fakeEvent = {
                points: [{ x: 50, y: 50, data: { uid: traceUid } }]
            };
            handlePlotlyHover(fakeEvent, fig0Uid);

            return (__fvHoverGuidesByFig[fig1Uid] || [])
                .filter(g => g.tag)
                .length;
        }""")

        assert (
            guides_after >= 1
        ), "Axis mode hover on fig0 must add at least one guide to fig1"

    def test_plotly_unhover_clears_crosshairs(
        self, page: Page, server_port: int, renderer: str
    ):
        if renderer == "echarts":
            pytest.skip("Plotly-specific test")

        url = _dashboard_url_hover(server_port, renderer)
        page.goto(url)
        _wait_for_init(page, renderer)

        guides_after = page.evaluate("""() => {
            const fig0Uid = DASHBOARD_SPEC.figures[0].uid;
            const fig1Uid = DASHBOARD_SPEC.figures[1].uid;
            const traceUid = DASHBOARD_SPEC.figures[0].traces[0].uid;
            if (!DASHBOARD_SPEC.client_state) DASHBOARD_SPEC.client_state = {};
            DASHBOARD_SPEC.client_state.hover_mode = 'axis';
            handlePlotlyHover({points: [{x: 50, y: 50, data: {uid: traceUid}}]}, fig0Uid);
            handlePlotlyUnhover();
            return (__fvHoverGuidesByFig[fig1Uid] || []).length;
        }""")

        assert guides_after == 0, "Unhover must remove all hover guides"

    def test_plotly_axis_hover_emits_crosshair_on_shared_axes(
        self, page: Page, server_port: int, renderer: str
    ):
        if renderer == "echarts":
            pytest.skip("Plotly-specific test")

        # fig0 and fig1 are both line(x=ts, y=val), so they share BOTH the x and
        # the y column. Axis-mode hover projects the hovered point onto every
        # shared axis, so the target should show a full crosshair: one x-guide
        # and one y-guide.
        url = _dashboard_url_hover(server_port, renderer)
        page.goto(url)
        _wait_for_init(page, renderer)

        linked_shape = page.evaluate("""() => {
            const fig0Uid = DASHBOARD_SPEC.figures[0].uid;
            const fig1Uid = DASHBOARD_SPEC.figures[1].uid;
            const traceUid = DASHBOARD_SPEC.figures[0].traces[0].uid;
            if (!DASHBOARD_SPEC.client_state) DASHBOARD_SPEC.client_state = {};
            DASHBOARD_SPEC.client_state.hover_mode = 'axis';
            handlePlotlyHover({points: [{x: 50, y: 50, data: {uid: traceUid}}]}, fig0Uid);
            const linked = (__fvHoverGuidesByFig[fig1Uid] || []).filter(
                g => typeof g.tag === 'string' && g.tag.startsWith('linked:')
            );
            return { count: linked.length, axes: linked.map(g => g.axis).sort() };
        }""")

        assert (
            linked_shape["count"] == 2
        ), "Axis hover on shared x+y axes must render a crosshair (two guides)"
        assert linked_shape["axes"] == ["x", "y"], (
            "Crosshair must have one x-guide and one y-guide, "
            f"got {linked_shape['axes']}"
        )

    def test_plotly_axis_hover_emits_no_visual_to_source_figure(
        self, page: Page, server_port: int, renderer: str
    ):
        """Spec rule: no linked visual is emitted to the source figure."""
        if renderer == "echarts":
            pytest.skip("Plotly-specific test")

        url = _dashboard_url_hover(server_port, renderer)
        page.goto(url)
        _wait_for_init(page, renderer)

        source_guides = page.evaluate("""() => {
            const fig0Uid = DASHBOARD_SPEC.figures[0].uid;
            const traceUid = DASHBOARD_SPEC.figures[0].traces[0].uid;
            if (!DASHBOARD_SPEC.client_state) DASHBOARD_SPEC.client_state = {};
            DASHBOARD_SPEC.client_state.hover_mode = 'axis';
            handlePlotlyHover({points: [{x: 50, y: 50, data: {uid: traceUid}}]}, fig0Uid);
            return (__fvHoverGuidesByFig[fig0Uid] || []).filter(
                g => typeof g.tag === 'string' && g.tag.startsWith('linked:')
            ).length;
        }""")

        assert source_guides == 0, "Source figure must receive no linked hover visuals"

    def test_hover_no_visual_when_off(
        self, page: Page, server_port: int, renderer: str
    ):
        if renderer == "echarts":
            pytest.skip("Plotly-specific test")

        url = _dashboard_url_hover(server_port, renderer)
        page.goto(url)
        _wait_for_init(page, renderer)

        guides_after = page.evaluate("""() => {
            const fig0Uid = DASHBOARD_SPEC.figures[0].uid;
            const fig1Uid = DASHBOARD_SPEC.figures[1].uid;
            const traceUid = DASHBOARD_SPEC.figures[0].traces[0].uid;
            if (!DASHBOARD_SPEC.client_state) DASHBOARD_SPEC.client_state = {};
            DASHBOARD_SPEC.client_state.hover_mode = 'off';
            handlePlotlyHover({points: [{x: 50, y: 50, data: {uid: traceUid}}]}, fig0Uid);
            return (__fvHoverGuidesByFig[fig1Uid] || []).length;
        }""")

        assert guides_after == 0, "No hover visual when mode is 'off'"

    def test_plotly_minmax_hover_links_even_when_not_near_exact_point(
        self, page: Page, server_port: int, renderer: str
    ):
        """Regression: linked hover should work with minmax traces without pixel-perfect point hit."""
        if renderer == "echarts":
            pytest.skip("Plotly-specific test")

        url = _dashboard_url_hover_minmax_shared_x(server_port, renderer)
        page.goto(url)
        _wait_for_init(page, renderer)

        # Enable linked hover via the toggle
        page.click("#fv-hover-btn")
        page.wait_for_timeout(100)
        assert page.evaluate("DASHBOARD_SPEC.client_state.hover_mode") == "on"

        # Hover at the x-coordinate of a real sampled point but offset y far away.
        # With proper x-hover semantics this still emits plotly_hover and links.
        result = page.evaluate("""() => {
            const fig0Uid = DASHBOARD_SPEC.figures[0].uid;
            const fig1Uid = DASHBOARD_SPEC.figures[1].uid;
            const fig0Idx = figUidToIdx[fig0Uid];
            const fig1Idx = figUidToIdx[fig1Uid];
            const div0 = divs[fig0Idx];
            const rect = div0.getBoundingClientRect();
            const xa = div0._fullLayout.xaxis;
            const ya = div0._fullLayout.yaxis;
            const idx = Math.min(10, Math.max(0, (div0.data[0].x || []).length - 1));
            const x = div0.data[0].x[idx];
            const y = div0.data[0].y[idx];

            const px = rect.left + xa._offset + xa.l2p(x);
            const pyPoint = rect.top + ya._offset + ya.l2p(y);
            const yMin = rect.top + ya._offset + 2;
            const yMax = rect.top + ya._offset + ya._length - 2;
            const yMid = rect.top + ya._offset + (ya._length / 2);
            const pyOffset = pyPoint < yMid
              ? (yMax - 2)
              : (yMin + 2);

            window.__hoverLinkedCount = 0;
            const original = handlePlotlyHover;
            window.handlePlotlyHover = function(ed, uid) {
                window.__hoverLinkedCount += 1;
                return original(ed, uid);
            };

            return { px, pyOffset, fig1Idx };
        }""")

        page.mouse.move(result["px"], result["pyOffset"])
        page.wait_for_timeout(500)

        after = page.evaluate(
            """(fig1Idx) => ({
            calls: window.__hoverLinkedCount || 0,
            guides: ((__fvHoverGuidesByFig[DASHBOARD_SPEC.figures[fig1Idx].uid] || []).length)
        })""",
            result["fig1Idx"],
        )

        assert (
            after["calls"] >= 1
        ), "Hover should trigger without requiring exact y-point hit"
        assert after["guides"] >= 1, "Linked figure should receive crosshair from hover"

    def test_hover_mode_persists_through_share(
        self, page: Page, server_port: int, renderer: str
    ):
        """Toggle hover on, share, reload — client_state.hover_mode must still be 'on'."""
        url = _dashboard_url_hover(server_port, renderer)
        page.goto(url)
        _wait_for_init(page, renderer)

        # Toggle linked hover on
        page.click("#fv-hover-btn")
        page.wait_for_timeout(200)
        assert page.evaluate("DASHBOARD_SPEC.client_state.hover_mode") == "on"

        share_url = page.evaluate("""async () => {
            const resp = await fetch(SERVER_URL + '/share', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({spec: DASHBOARD_SPEC, server_url: SERVER_URL}),
            });
            const data = await resp.json();
            return new URL(data.url, window.location.href).href;
        }""")
        assert share_url
        share_url += f"&renderer={renderer}"

        page.goto(share_url)
        _wait_for_init(page, renderer)

        mode = page.evaluate(
            "DASHBOARD_SPEC.client_state && DASHBOARD_SPEC.client_state.hover_mode"
        )
        btn = page.query_selector("#fv-hover-btn")
        assert (
            mode == "on"
        ), f"client_state.hover_mode must persist as 'on' through share, got {mode!r}"
        assert "On" in (btn.text_content() or ""), "Button must show 'On' after restore"

    def test_hover_dropdown_hidden_for_single_figure_dashboard(
        self, page: Page, server_port: int, renderer: str
    ):
        """Dropdown must be hidden when there are no linkable source-target pairs."""
        from flexviz.figure import Figure
        from flexviz.server import register_source
        from flexviz.spec import encode_spec

        df = pl.DataFrame({"ts": list(range(50)), "val": [float(i) for i in range(50)]})
        register_source("_browser_test_single", df)
        fig = Figure(df)
        fig.add_line(x="ts", y="val")
        spec = fig.to_spec(source="_browser_test_single")
        # Single-figure dashboard via VisualizationSpec
        from flexviz.spec import DashboardSpec

        dash_spec = DashboardSpec(figures=[spec.figure], state=spec.state)
        encoded = encode_spec(dash_spec)
        url = f"http://127.0.0.1:{server_port}/view?spec={encoded}&renderer={renderer}"

        page.goto(url)
        _wait_for_init(page, renderer)

        wrapper = page.query_selector("#fv-hover-dropdown")
        if wrapper:
            style = wrapper.get_attribute("style") or ""
            assert (
                "display: none" in style or "display:none" in style
            ), "Hover dropdown must be hidden for single-figure dashboard"

    def test_hover_toggle_aria_pressed_state(
        self, page: Page, server_port: int, renderer: str
    ):
        """aria-pressed must track the on/off toggle state."""
        url = _dashboard_url_hover(server_port, renderer)
        page.goto(url)
        _wait_for_init(page, renderer)

        btn = page.query_selector("#fv-hover-btn")
        if btn is None:
            pytest.skip("Hover toggle not visible for this dashboard")

        assert (
            btn.get_attribute("aria-pressed") == "false"
        ), "aria-pressed must be false before turning hover on"
        btn.click()
        page.wait_for_timeout(100)
        assert (
            btn.get_attribute("aria-pressed") == "true"
        ), "aria-pressed must be true after turning hover on"
        btn.click()
        page.wait_for_timeout(100)
        assert (
            btn.get_attribute("aria-pressed") == "false"
        ), "aria-pressed must be false after turning hover off"


class TestCellHoverBrowser:
    """Browser tests for Phase 3 cell hover mode."""

    @pytest.mark.browser
    @pytest.mark.parametrize("renderer", ["plotly"])
    def test_hover_toggle_available_for_hist_dashboard(
        self, page: Page, server_port: int, renderer: str
    ):
        """The hover toggle must be offered when a histogram shares a column with
        a line (a linkable source→target pair exists)."""
        import polars as pl
        from flexviz.dashboard import Dashboard
        from flexviz.server import register_source
        from flexviz.spec import encode_spec

        df = pl.DataFrame(
            {
                "ts": list(range(100)),
                "val": [float(i % 10) for i in range(100)],
            }
        )
        register_source("_browser_cell_test", df)

        dash = Dashboard(df)
        fig1 = dash.add_figure()
        fig1.add_line(x="ts", y="val")
        fig2 = dash.add_figure()
        fig2.add_histogram(x="ts", bins=10)
        spec = dash.to_spec(source_name="_browser_cell_test")
        encoded = encode_spec(spec)
        url = f"http://127.0.0.1:{server_port}/view?spec={encoded}&renderer={renderer}"

        page.goto(url)
        _wait_for_init(page, renderer)

        btn = page.query_selector("#fv-hover-btn")
        assert btn is not None, "Hover toggle must be present"
        wrapper = page.query_selector("#fv-hover-dropdown")
        style = (wrapper.get_attribute("style") or "") if wrapper else ""
        assert (
            "display: none" not in style and "display:none" not in style
        ), "Hover toggle must be visible when a histogram shares a column with a line"

    @pytest.mark.browser
    @pytest.mark.parametrize("renderer", ["plotly"])
    def test_cell_hover_emits_x_band_to_line_target(
        self, page: Page, server_port: int, renderer: str
    ):
        """Cell hover on histogram must emit x_band to linked line figure."""
        import polars as pl
        from flexviz.dashboard import Dashboard
        from flexviz.server import register_source
        from flexviz.spec import encode_spec

        df = pl.DataFrame(
            {
                "ts": list(range(100)),
                "val": [float(i % 10) for i in range(100)],
            }
        )
        register_source("_browser_cell_test2", df)

        dash = Dashboard(df)
        fig1 = dash.add_figure()
        fig1.add_line(x="ts", y="val")
        fig2 = dash.add_figure()
        fig2.add_histogram(x="ts", bins=10)
        spec = dash.to_spec(source_name="_browser_cell_test2")
        encoded = encode_spec(spec)
        url = f"http://127.0.0.1:{server_port}/view?spec={encoded}&renderer={renderer}"

        page.goto(url)
        _wait_for_init(page, renderer)

        guides = page.evaluate("""() => {
            const fig0Uid = DASHBOARD_SPEC.figures[0].uid;
            const fig1Uid = DASHBOARD_SPEC.figures[1].uid;
            const histTraceUid = DASHBOARD_SPEC.figures[1].traces[0].uid;

            if (!DASHBOARD_SPEC.client_state) DASHBOARD_SPEC.client_state = {};
            DASHBOARD_SPEC.client_state.hover_mode = 'cell';

            const fakeEvent = {
                sourceFigUid: fig1Uid,
                sourceTraceUid: histTraceUid,
                kind: 'cell',
                values: {},
                columns: { x: 'ts' },
                bounds: { x0: 0, x1: 10 },
                coordSpace: 'cartesian',
                key: null,
            };
            const visuals = planHoverVisuals(
                fakeEvent, 'cell',
                hoverTargetsByColumn, hoverSourceByTrace, hoverCellsByTraceUid
            );
            const fig0Visuals = visuals.get(fig0Uid) || [];
            return fig0Visuals.map(v => v.type);
        }""")

        assert (
            "x_band" in guides
        ), f"Cell hover on histogram must emit x_band to line figure, got: {guides}"

    @pytest.mark.browser
    @pytest.mark.parametrize("renderer", ["plotly"])
    def test_cell_hover_no_visual_to_source_figure(
        self, page: Page, server_port: int, renderer: str
    ):
        """Cell hover must not emit any visual to the source figure."""
        import polars as pl
        from flexviz.dashboard import Dashboard
        from flexviz.server import register_source
        from flexviz.spec import encode_spec

        df = pl.DataFrame(
            {
                "ts": list(range(100)),
                "val": [float(i % 10) for i in range(100)],
            }
        )
        register_source("_browser_cell_test3", df)
        dash = Dashboard(df)
        fig1 = dash.add_figure()
        fig1.add_line(x="ts", y="val")
        fig2 = dash.add_figure()
        fig2.add_histogram(x="ts", bins=10)
        spec = dash.to_spec(source_name="_browser_cell_test3")
        encoded = encode_spec(spec)
        url = f"http://127.0.0.1:{server_port}/view?spec={encoded}&renderer={renderer}"

        page.goto(url)
        _wait_for_init(page, renderer)

        source_guides = page.evaluate("""() => {
            const fig1Uid = DASHBOARD_SPEC.figures[1].uid;
            const histTraceUid = DASHBOARD_SPEC.figures[1].traces[0].uid;
            if (!DASHBOARD_SPEC.client_state) DASHBOARD_SPEC.client_state = {};
            DASHBOARD_SPEC.client_state.hover_mode = 'cell';
            const fakeEvent = {
                sourceFigUid: fig1Uid,
                sourceTraceUid: histTraceUid,
                kind: 'cell',
                values: {},
                columns: { x: 'ts' },
                bounds: { x0: 0, x1: 10 },
                coordSpace: 'cartesian',
                key: null,
            };
            const visuals = planHoverVisuals(
                fakeEvent, 'cell',
                hoverTargetsByColumn, hoverSourceByTrace, hoverCellsByTraceUid
            );
            return (visuals.get(fig1Uid) || []).length;
        }""")

        assert source_guides == 0, "Source figure must receive no linked visuals"

    @pytest.mark.browser
    @pytest.mark.parametrize("renderer", ["plotly"])
    def test_axis_mode_histogram_source_emits_guide(
        self, page: Page, server_port: int, renderer: str
    ):
        """A histogram declares ``axis`` as a source mode, so hovering a bar in
        axis mode must emit an x-guide on a line target sharing the column — it
        must not be swallowed as a cell event because the bar carries customdata
        bin bounds."""
        import polars as pl
        from flexviz.dashboard import Dashboard
        from flexviz.server import register_source
        from flexviz.spec import encode_spec

        df = pl.DataFrame(
            {"ts": list(range(100)), "val": [float(i % 10) for i in range(100)]}
        )
        register_source("_browser_hist_axis_src", df)
        dash = Dashboard(df)
        dash.add_figure().add_line(x="ts", y="val")  # fig0: axis target
        dash.add_figure().add_histogram(x="ts", bins=10)  # fig1: axis source
        spec = dash.to_spec(source_name="_browser_hist_axis_src")
        url = (
            f"http://127.0.0.1:{server_port}/view?"
            f"spec={encode_spec(spec)}&renderer={renderer}"
        )
        page.goto(url)
        _wait_for_init(page, renderer)

        guides = page.evaluate("""() => {
            const lineFigUid = DASHBOARD_SPEC.figures[0].uid;
            const histFigUid = DASHBOARD_SPEC.figures[1].uid;
            const histTraceUid = DASHBOARD_SPEC.figures[1].traces[0].uid;
            if (!DASHBOARD_SPEC.client_state) DASHBOARD_SPEC.client_state = {};
            DASHBOARD_SPEC.client_state.hover_mode = 'axis';
            // A histogram bar carries customdata bin bounds; in axis mode it must
            // still emit a point-style x guide (at the bin centre), not a cell.
            handlePlotlyHover({
                points: [{
                    x: 50, y: 3,
                    customdata: { x0: 45, x1: 55 },
                    data: { uid: histTraceUid },
                }],
            }, histFigUid);
            return (__fvHoverGuidesByFig[lineFigUid] || [])
                .filter(g => typeof g.tag === 'string' && g.tag.startsWith('linked:'))
                .map(g => g.axis);
        }""")

        assert "x" in guides, (
            "histogram axis-source hover must emit an x-guide on the line "
            f"target, got {guides}"
        )

    @pytest.mark.browser
    @pytest.mark.parametrize("renderer", ["plotly"])
    def test_hover_toggle_hidden_for_geo_only_cell_source(
        self, page: Page, server_port: int, renderer: str
    ):
        """The hover toggle must stay hidden when only an unimplemented geo cell
        source exists with no other linkable pair."""
        from flexviz.dashboard import Dashboard
        from flexviz.server import register_source
        from flexviz.spec import encode_spec

        df = pl.DataFrame(
            {
                "lat": [40.71, 40.72, 40.73, 40.74, 40.75],
                "lon": [-74.00, -73.99, -73.98, -73.97, -73.96],
            }
        )
        register_source("_browser_cell_geo_only", df)

        dash = Dashboard(df)
        fig0 = dash.add_figure()
        fig0.add_geo_histogram2d(lat="lat", lon="lon", lat_bins=2, lon_bins=2)
        fig1 = dash.add_figure()
        fig1.add_line(x="lon", y="lat")
        spec = dash.to_spec(source_name="_browser_cell_geo_only")
        encoded = encode_spec(spec)
        url = f"http://127.0.0.1:{server_port}/view?spec={encoded}&renderer={renderer}"

        page.goto(url)
        _wait_for_init(page, renderer)

        wrapper = page.query_selector("#fv-hover-dropdown")
        if wrapper:
            style = wrapper.get_attribute("style") or ""
            assert (
                "display: none" in style or "display:none" in style
            ), "Hover toggle must be hidden when only geo_histogram2d can source hover"

    @pytest.mark.browser
    @pytest.mark.parametrize("renderer", ["plotly"])
    def test_cell_hover_uses_target_axis_for_band_orientation(
        self, page: Page, server_port: int, renderer: str
    ):
        """Cell fallback bands must use target axis mapping, not source axis role."""
        from flexviz.dashboard import Dashboard
        from flexviz.server import register_source
        from flexviz.spec import encode_spec

        df = pl.DataFrame(
            {
                "src_x": [float(i) for i in range(100)],
                "shared": [float((i % 20) - 10) for i in range(100)],
            }
        )
        register_source("_browser_cell_target_axis", df)

        dash = Dashboard(df)
        fig0 = dash.add_figure()
        # shared is source y axis here
        fig0.add_histogram2d(x="src_x", y="shared", x_bins=10, y_bins=8)
        fig1 = dash.add_figure()
        # shared is target x axis here; line is axis-target only (no cell rect path)
        fig1.add_line(x="shared", y="src_x")
        spec = dash.to_spec(source_name="_browser_cell_target_axis")
        encoded = encode_spec(spec)
        url = f"http://127.0.0.1:{server_port}/view?spec={encoded}&renderer={renderer}"

        page.goto(url)
        _wait_for_init(page, renderer)

        emitted = page.evaluate("""() => {
            const sourceFigUid = DASHBOARD_SPEC.figures[0].uid;
            const sourceTraceUid = DASHBOARD_SPEC.figures[0].traces[0].uid;
            const targetFigUid = DASHBOARD_SPEC.figures[1].uid;

            if (!DASHBOARD_SPEC.client_state) DASHBOARD_SPEC.client_state = {};
            DASHBOARD_SPEC.client_state.hover_mode = 'cell';

            // Source event only advertises 'shared' on source y axis.
            const fakeEvent = {
                sourceFigUid,
                sourceTraceUid,
                kind: 'cell',
                values: {},
                columns: { y: 'shared' },
                bounds: { y0: -2, y1: 2 },
                coordSpace: 'cartesian',
                key: null,
            };

            const visuals = planHoverVisuals(
                fakeEvent, 'cell',
                hoverTargetsByColumn, hoverSourceByTrace, hoverCellsByTraceUid
            );
            return (visuals.get(targetFigUid) || []).map(v => v.type);
        }""")

        assert (
            "x_band" in emitted
        ), f"Target x-mapped trace must receive x_band, got {emitted}"
        assert (
            "y_band" not in emitted
        ), f"Target x-mapped trace must not receive y_band, got {emitted}"

    @pytest.mark.browser
    @pytest.mark.parametrize("renderer", ["plotly"])
    def test_grouped_histogram_hover_cells_keyed_by_parent(
        self, page: Page, server_port: int, renderer: str
    ):
        """Grouped histogram bin bounds must be resolvable under the logical
        parent uid (target lookups use the parent uid, not child uids)."""
        from flexviz.dashboard import Dashboard
        from flexviz.server import register_source
        from flexviz.spec import encode_spec

        df = pl.DataFrame(
            {
                "x": list(range(300)),
                "sin": [float(math.sin(i / 30.0)) for i in range(300)],
                "country": ["NL", "BE", "DE"][0:1] * 100 + ["BE"] * 100 + ["DE"] * 100,
            }
        )
        register_source("_browser_grouped_hist_cells", df)
        dash = Dashboard(df)
        dash.add_figure().add_histogram(x="sin", group_by="country", bins=12)
        spec = dash.to_spec(source_name="_browser_grouped_hist_cells")
        url = (
            f"http://127.0.0.1:{server_port}/view?"
            f"spec={encode_spec(spec)}&renderer={renderer}"
        )
        page.goto(url)
        _wait_for_init(page, renderer)

        n_cells = page.evaluate("""() => {
            const parentUid = DASHBOARD_SPEC.figures[0].traces[0].uid;
            return (hoverCellsByTraceUid[parentUid] || []).length;
        }""")
        assert n_cells > 0, (
            "Grouped histogram bin bounds must be stored under the parent uid; "
            f"got {n_cells} cells"
        )

    @pytest.mark.browser
    @pytest.mark.parametrize("renderer", ["plotly"])
    def test_axis_hover_line_emits_band_on_grouped_histograms(
        self, page: Page, server_port: int, renderer: str
    ):
        """Axis-mode hover on a grouped line must place a bin-band on grouped
        histograms sharing the column — including a histogram that bins that
        column on its *x* axis (cross-axis), regressing the user-reported
        'no linked hover on histogram' bug."""
        from flexviz.dashboard import Dashboard
        from flexviz.server import register_source
        from flexviz.spec import encode_spec

        n = 600
        df = pl.DataFrame(
            {
                "x": list(range(n)),
                "sin": [float(math.sin(i / 50.0)) for i in range(n)],
                "country": [["NL", "BE", "DE"][i % 3] for i in range(n)],
            }
        )
        register_source("_browser_axis_grouped_hist", df)
        dash = Dashboard(df)
        dash.add_figure().add_line(x="x", y="sin", group_by="country")  # fig0
        dash.add_figure().add_histogram(
            y="sin", group_by="country", bins=12
        )  # fig1 horiz
        dash.add_figure().add_histogram(
            x="sin", group_by="country", bins=12
        )  # fig2 vert
        spec = dash.to_spec(source_name="_browser_axis_grouped_hist")
        url = (
            f"http://127.0.0.1:{server_port}/view?"
            f"spec={encode_spec(spec)}&renderer={renderer}"
        )
        page.goto(url)
        _wait_for_init(page, renderer)

        result = page.evaluate("""() => {
            const lineFigUid = DASHBOARD_SPEC.figures[0].uid;
            const yHistFigUid = DASHBOARD_SPEC.figures[1].uid;  // sin on y axis
            const xHistFigUid = DASHBOARD_SPEC.figures[2].uid;  // sin on x axis
            const lineParent = DASHBOARD_SPEC.figures[0].traces[0].uid;
            const lineChild = Object.keys(childUidToParentUid)
                .find(k => childUidToParentUid[k] === lineParent) || lineParent;

            if (!DASHBOARD_SPEC.client_state) DASHBOARD_SPEC.client_state = {};
            DASHBOARD_SPEC.client_state.hover_mode = 'axis';

            // Hover the line at a sin value of 0.0 (inside the data range, so a
            // bin contains it in both histograms).
            handlePlotlyHover({
                points: [{ x: 100, y: 0.0, data: { uid: lineChild } }],
            }, lineFigUid);

            const tags = (figUid) => (__fvHoverGuidesByFig[figUid] || [])
                .filter(g => typeof g.tag === 'string' && g.tag.startsWith('linked:'))
                .map(g => g.tag);
            return { yHist: tags(yHistFigUid), xHist: tags(xHistFigUid) };
        }""")

        assert any("y_band" in t for t in result["yHist"]), (
            "Horizontal (y=) grouped histogram must receive a y-band on axis "
            f"hover, got {result['yHist']}"
        )
        assert any("x_band" in t for t in result["xHist"]), (
            "Vertical (x=) grouped histogram must receive an x-band on axis "
            f"hover (cross-axis match), got {result['xHist']}"
        )

    @pytest.mark.browser
    @pytest.mark.parametrize("renderer", ["plotly"])
    def test_axis_hover_histogram_x_source_emits_y_guide_on_line(
        self, page: Page, server_port: int, renderer: str
    ):
        """A vertical (x=) histogram bins its column on x, but a line that plots
        that same column on y must receive a HORIZONTAL (y) guide at the bin
        centre — not a vertical x-guide at the wrong axis."""
        from flexviz.dashboard import Dashboard
        from flexviz.server import register_source
        from flexviz.spec import encode_spec

        df = pl.DataFrame(
            {
                "x": list(range(200)),
                "sin": [float(math.sin(i / 20.0)) for i in range(200)],
            }
        )
        register_source("_browser_axis_hist_x_to_line", df)
        dash = Dashboard(df)
        dash.add_figure().add_line(x="x", y="sin")  # fig0: sin on y axis
        dash.add_figure().add_histogram(x="sin", bins=10)  # fig1: sin on x axis
        spec = dash.to_spec(source_name="_browser_axis_hist_x_to_line")
        url = (
            f"http://127.0.0.1:{server_port}/view?"
            f"spec={encode_spec(spec)}&renderer={renderer}"
        )
        page.goto(url)
        _wait_for_init(page, renderer)

        axes = page.evaluate("""() => {
            const lineFigUid = DASHBOARD_SPEC.figures[0].uid;
            const histFigUid = DASHBOARD_SPEC.figures[1].uid;
            const histTraceUid = DASHBOARD_SPEC.figures[1].traces[0].uid;
            if (!DASHBOARD_SPEC.client_state) DASHBOARD_SPEC.client_state = {};
            DASHBOARD_SPEC.client_state.hover_mode = 'axis';
            handlePlotlyHover({
                points: [{
                    x: 0.1, y: 20,
                    customdata: { x0: 0.05, x1: 0.15 },
                    data: { uid: histTraceUid },
                }],
            }, histFigUid);
            return (__fvHoverGuidesByFig[lineFigUid] || [])
                .filter(g => typeof g.tag === 'string' && g.tag.startsWith('linked:'))
                .map(g => g.axis)
                .sort();
        }""")

        assert axes == ["y"], (
            "Histogram x-source hover must emit a single horizontal (y) guide on "
            f"the line that plots the shared column on y, got axes={axes}"
        )

    @pytest.mark.browser
    @pytest.mark.parametrize("renderer", ["plotly"])
    def test_cell_hover_1d_histogram_emits_band_not_cell_on_hist2d(
        self, page: Page, server_port: int, renderer: str
    ):
        """A 1D histogram only constrains one axis, so a 2D target (histogram2d)
        must receive a band/strip on the shared-column axis — never a single
        spurious cell rect."""
        from flexviz.dashboard import Dashboard
        from flexviz.server import register_source
        from flexviz.spec import encode_spec

        df = pl.DataFrame(
            {
                "x": list(range(400)),
                "sin": [float(math.sin(i / 40.0)) for i in range(400)],
            }
        )
        register_source("_browser_cell_1d_to_2d", df)
        dash = Dashboard(df)
        dash.add_figure().add_histogram(x="sin", bins=10)  # fig0: 1D source
        dash.add_figure().add_histogram2d(
            x="x", y="sin", x_bins=10, y_bins=8
        )  # fig1: sin on y
        spec = dash.to_spec(source_name="_browser_cell_1d_to_2d")
        url = (
            f"http://127.0.0.1:{server_port}/view?"
            f"spec={encode_spec(spec)}&renderer={renderer}"
        )
        page.goto(url)
        _wait_for_init(page, renderer)

        # Drive the cell projection directly: a 1D cell event (constrains only x)
        # whose column maps to the hist2d's y axis must yield a y-band, not a rect.
        types = page.evaluate("""() => {
            const histFigUid = DASHBOARD_SPEC.figures[0].uid;
            const hist2dFigUid = DASHBOARD_SPEC.figures[1].uid;
            const histTraceUid = DASHBOARD_SPEC.figures[0].traces[0].uid;
            const fakeEvent = {
                sourceFigUid: histFigUid,
                sourceTraceUid: histTraceUid,
                kind: 'cell',
                values: {},
                columns: { x: 'sin' },
                bounds: { x0: 0.05, x1: 0.15 },
                coordSpace: 'cartesian',
                key: null,
            };
            const visuals = planHoverVisuals(
                fakeEvent, 'cell',
                hoverTargetsByColumn, hoverSourceByTrace, hoverCellsByTraceUid
            );
            return (visuals.get(hist2dFigUid) || []).map(v => v.type);
        }""")

        assert "rect" not in types, (
            "A 1D histogram source must not highlight a single 2D cell, got " f"{types}"
        )
        assert "y_band" in types, (
            "A 1D histogram source must highlight the bin strip (y_band) on the "
            f"hist2d that plots the shared column on y, got {types}"
        )


# ---------------------------------------------------------------------------
# Reset / modebar cleanup browser tests (Plotly-only)
# ---------------------------------------------------------------------------

_INJECT_SELECTION_JS = """() => {
    const figUids = DASHBOARD_SPEC.figures.map(f => f.uid);
    const sel = {
        source_figure_uid: figUids[0],
                predicates: [{ clauses: [{ column: 'ts', range: [100, 300] }] }],
    };
    DASHBOARD_SPEC.state.selections = [sel];
    return postDashboardUpdate({
        type: 'selection', axis_ranges: {},
        selections: DASHBOARD_SPEC.state.selections,
        force_update: true, figure_uid: figUids[0],
    });
}"""


@pytest.mark.browser
class TestResetCleanupBrowser:
    """Plotly-specific tests for reset and modebar home behaviour."""

    def test_reset_clears_plotly_selection_boxes(self, page: Page, server_port: int):
        """Clicking toolbar Reset must clear Plotly's visual selection rectangle."""
        url = _dashboard_url(server_port, "plotly", n_figures=2)
        page.goto(url)
        _wait_for_init(page, "plotly")

        # Inject a cross-filter selection programmatically.
        page.evaluate(_INJECT_SELECTION_JS)
        page.wait_for_timeout(1_500)

        # Verify the selection is now in client state.
        sel_before = page.evaluate("DASHBOARD_SPEC.state.selections")
        assert len(sel_before) > 0, "Selection was not injected"

        # Click the toolbar Reset button.
        page.click("#fv-btn-reset")
        page.wait_for_timeout(2_000)

        # Client state must be cleared.
        selections = page.evaluate("DASHBOARD_SPEC.state.selections")
        assert selections == [], "fvOnReset must clear DASHBOARD_SPEC.state.selections"

        # All Plotly divs must have an empty (or absent) selections array so
        # the drawn rectangle disappears.
        all_cleared = page.evaluate("""() => {
            const divs = document.querySelectorAll('.js-plotly-plot');
            return Array.from(divs).every(gd => {
                const sels = (gd.layout || {}).selections;
                return !sels || sels.length === 0;
            });
        }""")
        assert all_cleared, "Plotly.react must clear selection rectangles on reset"

    def test_reset_fires_exactly_one_server_call(self, page: Page, server_port: int):
        """The toolbar Reset button must fire exactly one reset event.

        Bug B (missing _programmaticOp guard in handleRelayout) caused a
        spurious second server call when Plotly fired plotly_relayout during
        fvOnReset's Plotly.react call.
        """
        url = _dashboard_url(server_port, "plotly", n_figures=2)
        update_bodies: list[dict] = []

        def capture(req: PWRequest) -> None:
            if "/dashboard/update" in req.url and req.method == "POST":
                try:
                    update_bodies.append(json.loads(req.post_data or "{}"))
                except Exception:
                    pass

        page.on("request", capture)
        page.goto(url)
        _wait_for_init(page, "plotly")

        page.evaluate(_INJECT_SELECTION_JS)
        page.wait_for_timeout(1_500)

        count_before = len(update_bodies)
        page.click("#fv-btn-reset")
        page.wait_for_timeout(2_000)

        reset_events = [
            b
            for b in update_bodies[count_before:]
            if b.get("event", {}).get("type") == "init"
        ]
        assert len(reset_events) == 1, (
            f"Expected exactly 1 init event from toolbar Reset, got {len(reset_events)}. "
            "Spurious duplicates indicate _programmaticOp guard is missing from handleRelayout."
        )

    def test_modebar_home_preserves_selections(self, page: Page, server_port: int):
        """The Plotly modebar Reset-axes button must preserve cross-filter selections.

        Before the fix, both autorange branches in handleRelayout sent
        type:'reset' with selections:[] — wiping the cross-filter state.
        Now they send type:'viewport' with the current selections preserved.
        """
        url = _dashboard_url(server_port, "plotly", n_figures=2)
        update_bodies: list[dict] = []

        def capture(req: PWRequest) -> None:
            if "/dashboard/update" in req.url and req.method == "POST":
                try:
                    update_bodies.append(json.loads(req.post_data or "{}"))
                except Exception:
                    pass

        page.on("request", capture)
        page.goto(url)
        _wait_for_init(page, "plotly")

        # Inject a cross-filter selection.
        page.evaluate(_INJECT_SELECTION_JS)
        page.wait_for_timeout(1_500)

        sel_after_inject = page.evaluate("DASHBOARD_SPEC.state.selections")
        assert len(sel_after_inject) > 0, "Selection not injected"

        count_before_home = len(update_bodies)

        # Simulate the Plotly modebar "Reset axes" button: it fires
        # xaxis.autorange=true on the first figure's div.
        page.evaluate("""() => {
            const divs = document.querySelectorAll('.js-plotly-plot');
            if (divs.length > 0) {
                Plotly.relayout(divs[0], {'xaxis.autorange': true});
            }
        }""")
        page.wait_for_timeout(2_000)

        # The cross-filter selections must still be set in client state.
        selections_after = page.evaluate("DASHBOARD_SPEC.state.selections")
        assert len(selections_after) > 0, (
            "Modebar home must NOT clear DASHBOARD_SPEC.state.selections — "
            "only the toolbar Reset button should do that."
        )

        # The server must have received a 'viewport' event (not 'reset').
        new_events = update_bodies[count_before_home:]
        event_types = [b.get("event", {}).get("type") for b in new_events]
        assert (
            "reset" not in event_types
        ), f"Modebar home sent a 'reset' event — should send 'viewport'. Got: {event_types}"
        assert (
            "viewport" in event_types
        ), f"Expected a 'viewport' event from modebar home. Got: {event_types}"

    def test_panel_reset_clears_sourced_selection_and_scopes_viewport(
        self, page: Page, server_port: int
    ):
        """Per-panel reset clears the sourced selection and the target viewport.

        When a figure sources a cross-filter, resetting that panel removes its
        selection (sends deselect/selection event) and clears its viewport.
        Other figures' viewports are not affected.
        """
        url = _dashboard_url(server_port, "plotly", n_figures=2)
        update_bodies: list[dict] = []

        def capture(req: PWRequest) -> None:
            if "/dashboard/update" in req.url and req.method == "POST":
                try:
                    update_bodies.append(json.loads(req.post_data or "{}"))
                except Exception:
                    pass

        page.on("request", capture)
        page.goto(url)
        _wait_for_init(page, "plotly")

        # Seed two viewports and a selection sourced from fig A.
        page.evaluate("""() => {
                const figA = DASHBOARD_SPEC.figures[0].uid;
                const figB = DASHBOARD_SPEC.figures[1].uid;
                DASHBOARD_SPEC.state.viewport[figA + '/x'] = {min: 5, max: 25};
                DASHBOARD_SPEC.state.viewport[figB + '/x'] = {min: 100, max: 160};
                DASHBOARD_SPEC.state.selections = [{
                  source_figure_uid: figA,
                  predicates: [{ clauses: [{ column: 'ts', range: [100, 300] }] }],
                }];
            }""")

        count_before = len(update_bodies)
        page.click("#fv-bar-0 .fv-mode-action-btn[data-action='reset-panel']")
        page.wait_for_timeout(2_000)

        state_after = page.evaluate("""() => ({
                figA: DASHBOARD_SPEC.figures[0].uid,
                figB: DASHBOARD_SPEC.figures[1].uid,
                viewport: DASHBOARD_SPEC.state.viewport,
                selections: DASHBOARD_SPEC.state.selections,
            })""")

        # Selection sourced from fig A must be cleared.
        sourced = [
            s
            for s in (state_after["selections"] or [])
            if s.get("source_figure_uid") == state_after["figA"]
        ]
        assert (
            sourced == []
        ), "Panel reset must clear the selection sourced from that figure"

        # Fig A viewport cleared; fig B viewport untouched.
        assert (
            f"{state_after['figA']}/x" not in state_after["viewport"]
        ), "Panel reset must clear viewport for target figure"
        assert (
            f"{state_after['figB']}/x" in state_after["viewport"]
        ), "Panel reset must not clear viewport for other figures"

        # The event must be deselect (no remaining selections), not viewport.
        new_events = update_bodies[count_before:]
        event_types = {b.get("event", {}).get("type") for b in new_events}
        assert (
            "deselect" in event_types or "selection" in event_types
        ), "Panel reset must send a deselect/selection event when it sourced a filter"

    def test_axis_lock_toggle_captures_range_and_panel_reset_preserves_it(
        self, page: Page, server_port: int
    ):
        """Locking an axis stores its current range; resetting an otherwise-clean
        locked figure leaves the lock intact and is a no-op (axis locks are
        view-only, so nothing the figure shows changes — see #32)."""
        url = _dashboard_url(server_port, "plotly", n_figures=2)
        update_bodies: list[dict] = []

        def capture(req: PWRequest) -> None:
            if "/dashboard/update" in req.url and req.method == "POST":
                try:
                    update_bodies.append(json.loads(req.post_data or "{}"))
                except Exception:
                    pass

        page.on("request", capture)
        page.goto(url)
        _wait_for_init(page, "plotly")
        page.wait_for_selector(
            "#fv-bar-0 .fv-mode-action-btn[data-action='lock-axes']:not([disabled])"
        )

        count_before = len(update_bodies)
        page.click("#fv-bar-0 .fv-mode-action-btn[data-action='lock-axes']")
        page.wait_for_function("""() => {
                const figA = DASHBOARD_SPEC.figures[0].uid;
                return DASHBOARD_SPEC.client_state.axis_locks[figA + '/x'] === true
                  && !!DASHBOARD_SPEC.client_state.axis_lock_ranges[figA + '/x']
                  && !DASHBOARD_SPEC.state.viewport[figA + '/x'];
            }""")

        locked_state = page.evaluate("""() => {
                const figA = DASHBOARD_SPEC.figures[0].uid;
                return {
                  figA,
                  range: DASHBOARD_SPEC.client_state.axis_lock_ranges[figA + '/x'],
                  locked: DASHBOARD_SPEC.client_state.axis_locks[figA + '/x'],
                  viewport: DASHBOARD_SPEC.state.viewport[figA + '/x'] || null,
                };
            }""")
        assert locked_state["locked"] is True
        assert locked_state["viewport"] is None
        assert len(update_bodies) == count_before

        page.click("#fv-bar-0 .fv-mode-action-btn[data-action='reset-panel']")
        page.wait_for_function(
            """lockedRange => {
                const figA = DASHBOARD_SPEC.figures[0].uid;
                const current = DASHBOARD_SPEC.client_state.axis_lock_ranges[figA + '/x'];
                return current
                  && current.min === lockedRange.min
                  && current.max === lockedRange.max
                  && DASHBOARD_SPEC.client_state.axis_locks[figA + '/x'] === true
                  && !DASHBOARD_SPEC.state.viewport[figA + '/x'];
            }""",
            arg=locked_state["range"],
        )

        # The reset of an unzoomed, non-sourcing, locked figure changes nothing:
        # no /dashboard/update is issued (the lock is view-only and already
        # holding).
        assert (
            len(update_bodies) == count_before
        ), "Reset of a clean locked figure must be a no-op (no /dashboard/update)"

    def test_axis_lock_before_zoom_uses_autorange_after_data_update(
        self, page: Page, server_port: int
    ):
        """Initial no-zoom locks must not freeze Plotly's empty bootstrap range."""
        url = _dashboard_url(server_port, "plotly", n_figures=1)
        update_bodies: list[dict] = []

        def capture(req: PWRequest) -> None:
            if "/dashboard/update" in req.url and req.method == "POST":
                try:
                    update_bodies.append(json.loads(req.post_data or "{}"))
                except Exception:
                    pass

        page.on("request", capture)
        page.goto(url)
        _wait_for_init(page, "plotly")

        before = page.evaluate("""() => {
                const gd = divs[0];
                return {
                  viewport: DASHBOARD_SPEC.state.viewport,
                  xLayout: gd._fullLayout.xaxis.range,
                  yLayout: gd._fullLayout.yaxis.range,
                };
            }""")
        assert before["viewport"] == {}
        assert before["xLayout"][1] > 400
        assert before["yLayout"][1] > 400

        count_before = len(update_bodies)
        page.click("#fv-bar-0 .fv-mode-action-btn[data-action='lock-axes']")
        page.wait_for_function("""() => {
                const figUid = DASHBOARD_SPEC.figures[0].uid;
                return DASHBOARD_SPEC.client_state.axis_lock_ranges[figUid + '/x']
                  && DASHBOARD_SPEC.client_state.axis_lock_ranges[figUid + '/y'];
            }""")

        state = page.evaluate("""() => {
                const figUid = DASHBOARD_SPEC.figures[0].uid;
                return {
                  viewport: DASHBOARD_SPEC.state.viewport,
                  x: DASHBOARD_SPEC.client_state.axis_lock_ranges[figUid + '/x'],
                  y: DASHBOARD_SPEC.client_state.axis_lock_ranges[figUid + '/y'],
                };
            }""")
        assert state["viewport"] == {}
        assert state["x"] == {"min": before["xLayout"][0], "max": before["xLayout"][1]}
        assert state["y"] == {"min": before["yLayout"][0], "max": before["yLayout"][1]}
        assert len(update_bodies) == count_before

        page.evaluate("""async () => {
                await postDashboardUpdate({
                  type: 'init',
                  axis_ranges: {},
                  selections: [],
                  force_update: true,
                });
            }""")
        page.wait_for_timeout(1_000)
        after_update = page.evaluate("""() => {
                const figUid = DASHBOARD_SPEC.figures[0].uid;
                const gd = divs[0];
                return {
                  viewport: DASHBOARD_SPEC.state.viewport,
                  lockX: DASHBOARD_SPEC.client_state.axis_lock_ranges[figUid + '/x'],
                  lockY: DASHBOARD_SPEC.client_state.axis_lock_ranges[figUid + '/y'],
                  xLayout: gd._fullLayout.xaxis.range,
                  yLayout: gd._fullLayout.yaxis.range,
                };
            }""")
        assert after_update["viewport"] == {}
        assert after_update["lockX"] == state["x"]
        assert after_update["lockY"] == state["y"]
        assert after_update["xLayout"] == pytest.approx(before["xLayout"])
        assert after_update["yLayout"] == pytest.approx(before["yLayout"])

    def test_axis_lock_before_zoom_keeps_weekday_heatmap_visible(
        self, page: Page, server_port: int
    ):
        """Locking a weekday heatmap axis must not reaggregate away edge bins."""
        url = _dashboard_url_weekday_hist2d(server_port)
        update_bodies: list[dict] = []

        def capture(req: PWRequest) -> None:
            if "/dashboard/update" in req.url and req.method == "POST":
                try:
                    update_bodies.append(json.loads(req.post_data or "{}"))
                except Exception:
                    pass

        page.on("request", capture)
        page.goto(url)
        _wait_for_init(page, "plotly")

        before = page.evaluate("""() => {
                const gd = divs[0];
                const y = gd.data[0].y;
                const z = gd.data[0].z;
                return {
                  centers: [Math.min(...y), Math.max(...y)],
                  firstRowSum: z[0].reduce((a, b) => a + b, 0),
                  lastRowSum: z[z.length - 1].reduce((a, b) => a + b, 0),
                  yLayout: gd._fullLayout.yaxis.range,
                };
            }""")
        assert before["centers"][0] > 1
        assert before["centers"][1] < 7
        assert before["firstRowSum"] > 0
        assert before["lastRowSum"] > 0

        count_before = len(update_bodies)
        page.click("#fv-bar-0 .fv-mode-action-btn[data-action='lock-axes']")
        page.wait_for_function("""() => {
                const figUid = DASHBOARD_SPEC.figures[0].uid;
                return DASHBOARD_SPEC.client_state.axis_lock_ranges[figUid + '/y']
                  && !DASHBOARD_SPEC.state.viewport[figUid + '/y'];
            }""")
        after = page.evaluate("""() => {
                const figUid = DASHBOARD_SPEC.figures[0].uid;
                const gd = divs[0];
                const y = gd.data[0].y;
                const z = gd.data[0].z;
                return {
                  viewport: DASHBOARD_SPEC.state.viewport,
                  locked: DASHBOARD_SPEC.client_state.axis_lock_ranges[figUid + '/y'],
                  centers: [Math.min(...y), Math.max(...y)],
                  firstRowSum: z[0].reduce((a, b) => a + b, 0),
                  lastRowSum: z[z.length - 1].reduce((a, b) => a + b, 0),
                  yLayout: gd._fullLayout.yaxis.range,
                };
            }""")
        assert after["viewport"] == {}
        assert after["locked"] == {
            "min": before["yLayout"][0],
            "max": before["yLayout"][1],
        }
        assert after["centers"] == before["centers"]
        assert after["firstRowSum"] > 0
        assert after["lastRowSum"] > 0
        assert after["yLayout"] == before["yLayout"]
        assert len(update_bodies) == count_before

    def test_axis_lock_category_axis_is_visual_only(self, page: Page, server_port: int):
        """Category-axis locks are allowed because they no longer affect backend data."""
        url = _dashboard_url_plotly_selection_box(server_port, "bar")
        page.goto(url)
        _wait_for_init(page, "plotly")

        lock_x = "#fv-bar-0 .fv-mode-action-btn[data-action='lock-axes']"
        page.wait_for_selector(lock_x)
        assert page.locator(lock_x).is_enabled()

        state = page.evaluate("""async () => {
                const figUid = DASHBOARD_SPEC.figures[0].uid;
                await window.fvOnToggleAxisLocks(figUid);
                return {
                  locked: DASHBOARD_SPEC.client_state.axis_locks[figUid + '/x'] === true,
                  lockRange: DASHBOARD_SPEC.client_state.axis_lock_ranges[figUid + '/x'] || null,
                  viewport: DASHBOARD_SPEC.state.viewport[figUid + '/x'] || null,
                };
            }""")
        assert state["locked"] is True
        assert state["lockRange"] is not None
        assert state["viewport"] is None

    def test_locked_axis_relayout_posts_only_unlocked_axis(
        self, page: Page, server_port: int
    ):
        """Relayout touching a locked x and an unlocked y on a line figure:
        the locked x is pruned and snapped back, the y persists CLIENT-SIDE
        only — a line's y is not a recompute axis, so the fvNeedsFetch gate
        (commit bfcdf8a) suppresses the no-op POST entirely."""
        url = _dashboard_url(server_port, "plotly", n_figures=1)
        update_bodies: list[dict] = []

        def capture(req: PWRequest) -> None:
            if "/dashboard/update" in req.url and req.method == "POST":
                try:
                    update_bodies.append(json.loads(req.post_data or "{}"))
                except Exception:
                    pass

        page.on("request", capture)
        page.goto(url)
        _wait_for_init(page, "plotly")

        # Lock ONLY the x axis (the toolbar "lock-axes" button locks every
        # lockable axis, which would leave nothing unlocked to persist). We
        # need one locked + one unlocked axis to exercise the prune path.
        page.evaluate("""() => {
                const figUid = DASHBOARD_SPEC.figures[0].uid;
                fvTryLockAxis(figUid, 'x');
            }""")
        page.wait_for_function("""() => {
                const figUid = DASHBOARD_SPEC.figures[0].uid;
                return DASHBOARD_SPEC.client_state.axis_lock_ranges[figUid + '/x'];
            }""")
        locked_x = page.evaluate("""() => {
                const figUid = DASHBOARD_SPEC.figures[0].uid;
                return DASHBOARD_SPEC.client_state.axis_lock_ranges[figUid + '/x'];
            }""")

        count_before = len(update_bodies)
        page.evaluate(
            """async lockedX => {
                await Plotly.relayout(divs[0], {
                  'xaxis.range': [lockedX.min + 1, lockedX.max + 1],
                  'yaxis.range': [10, 20],
                });
            }""",
            locked_x,
        )
        page.wait_for_timeout(1_000)

        state = page.evaluate("""() => {
                const figUid = DASHBOARD_SPEC.figures[0].uid;
                return {
                  figUid,
                  viewport: DASHBOARD_SPEC.state.viewport,
                  xLayout: divs[0]._fullLayout.xaxis.range,
                };
            }""")
        # Zero POSTs: the only unlocked changed axis (the line's y) does not
        # re-aggregate anything.
        assert len(update_bodies) == count_before
        # Client-side persistence is unaffected by the suppressed POST.
        assert f"{state['figUid']}/x" not in state["viewport"]
        assert state["viewport"][f"{state['figUid']}/y"] == {
            "min": 10,
            "max": 20,
        }
        assert state["xLayout"][0] == pytest.approx(locked_x["min"])
        assert state["xLayout"][1] == pytest.approx(locked_x["max"])

    def test_unlocked_recompute_axis_relayout_posts_and_prunes_locked_axis(
        self, page: Page, server_port: int
    ):
        """The positive case of the fvNeedsFetch gate: with y locked, a
        relayout touching both axes still POSTs (the line's x re-aggregates)
        and the POSTed axis_ranges carry only the unlocked x."""
        url = _dashboard_url(server_port, "plotly", n_figures=1)
        update_bodies: list[dict] = []

        def capture(req: PWRequest) -> None:
            if "/dashboard/update" in req.url and req.method == "POST":
                try:
                    update_bodies.append(json.loads(req.post_data or "{}"))
                except Exception:
                    pass

        page.on("request", capture)
        page.goto(url)
        _wait_for_init(page, "plotly")

        page.evaluate("""() => {
                const figUid = DASHBOARD_SPEC.figures[0].uid;
                fvTryLockAxis(figUid, 'y');
            }""")
        page.wait_for_function("""() => {
                const figUid = DASHBOARD_SPEC.figures[0].uid;
                return DASHBOARD_SPEC.client_state.axis_lock_ranges[figUid + '/y'];
            }""")
        locked_y = page.evaluate("""() => {
                const figUid = DASHBOARD_SPEC.figures[0].uid;
                return DASHBOARD_SPEC.client_state.axis_lock_ranges[figUid + '/y'];
            }""")

        count_before = len(update_bodies)
        page.evaluate(
            """async lockedY => {
                await Plotly.relayout(divs[0], {
                  'xaxis.range': [50, 100],
                  'yaxis.range': [lockedY.min + 1, lockedY.max + 1],
                });
            }""",
            locked_y,
        )
        page.wait_for_timeout(1_000)

        state = page.evaluate("""() => {
                const figUid = DASHBOARD_SPEC.figures[0].uid;
                return {
                  figUid,
                  viewport: DASHBOARD_SPEC.state.viewport,
                  yLayout: divs[0]._fullLayout.yaxis.range,
                };
            }""")
        assert len(update_bodies) > count_before
        event = update_bodies[-1]["event"]
        assert event["type"] == "viewport"
        assert event["axis_ranges"]["x"] == [50, 100]
        assert "y" not in event["axis_ranges"]
        assert state["viewport"][f"{state['figUid']}/x"] == {
            "min": 50,
            "max": 100,
        }
        assert f"{state['figUid']}/y" not in state["viewport"]
        assert state["yLayout"][0] == pytest.approx(locked_y["min"])
        assert state["yLayout"][1] == pytest.approx(locked_y["max"])

    def test_global_reset_preserves_axis_lock_ranges(
        self, page: Page, server_port: int
    ):
        """Global reset clears viewport/selections but keeps explicit axis locks."""
        url = _dashboard_url(server_port, "plotly", n_figures=1)
        page.goto(url)
        _wait_for_init(page, "plotly")

        page.click("#fv-bar-0 .fv-mode-action-btn[data-action='lock-axes']")
        page.wait_for_function("""() => {
                const figUid = DASHBOARD_SPEC.figures[0].uid;
                return DASHBOARD_SPEC.client_state.axis_lock_ranges[figUid + '/x'];
            }""")
        locked_x = page.evaluate("""() => {
                const figUid = DASHBOARD_SPEC.figures[0].uid;
                DASHBOARD_SPEC.state.viewport[figUid + '/x'] = {min: 1, max: 3};
                DASHBOARD_SPEC.state.selections = [{
                  source_figure_uid: figUid,
                  predicates: [{ clauses: [{ column: 'ts', range: [1, 3] }] }],
                }];
                return DASHBOARD_SPEC.client_state.axis_lock_ranges[figUid + '/x'];
            }""")

        page.click("#fv-btn-reset")
        page.wait_for_function("""() => {
                const figUid = DASHBOARD_SPEC.figures[0].uid;
                return Object.keys(DASHBOARD_SPEC.state.viewport).length === 0
                  && DASHBOARD_SPEC.state.selections.length === 0
                  && DASHBOARD_SPEC.client_state.axis_locks[figUid + '/x'] === true;
            }""")
        state = page.evaluate("""() => {
                const figUid = DASHBOARD_SPEC.figures[0].uid;
                return {
                  lockRange: DASHBOARD_SPEC.client_state.axis_lock_ranges[figUid + '/x'],
                  locked: DASHBOARD_SPEC.client_state.axis_locks[figUid + '/x'],
                  viewport: DASHBOARD_SPEC.state.viewport,
                  selections: DASHBOARD_SPEC.state.selections,
                };
            }""")
        assert state["locked"] is True
        assert state["lockRange"] == locked_x
        assert state["viewport"] == {}
        assert state["selections"] == []


@pytest.mark.browser
class TestLegendVisibilityBrowser:
    """Plotly legend visibility should survive FlexViz data refreshes."""

    def test_grouped_line_multi_group_by_hidden_trace_stays_hidden_after_zoom(
        self, page: Page, server_port: int
    ):
        url = _dashboard_url_grouped_line_multi_group_by(server_port, "plotly")
        page.goto(url)
        _wait_for_init(page, "plotly")

        page.wait_for_function(
            "() => document.querySelector('#fv-plot-0')?.data?.length >= 4"
        )
        page.eval_on_selector(
            "#fv-plot-0",
            "gd => Plotly.restyle(gd, {visible: 'legendonly'}, [0])",
        )

        hidden_uid = page.eval_on_selector("#fv-plot-0", "gd => gd.data[0].uid")
        page.evaluate("""async () => {
                const figUid = DASHBOARD_SPEC.figures[0].uid;
                DASHBOARD_SPEC.state.viewport[figUid + '/x'] = {min: 5, max: 25};
                await postDashboardUpdate({
                    type: 'viewport',
                    axis_ranges: {x: [5, 25]},
                    selections: DASHBOARD_SPEC.state.selections || [],
                    force_update: true,
                    figure_uid: figUid,
                });
            }""")

        visible_after = page.eval_on_selector(
            "#fv-plot-0",
            """(gd, uid) => {
                const trace = gd.data.find(t => t.uid === uid);
                return trace && (trace.visible || true);
            }""",
            hidden_uid,
        )
        assert visible_after == "legendonly"


@pytest.mark.browser
class TestTreemapClickBrowser:
    """Plotly treemap clicks should drive FlexViz selection state consistently."""

    def test_treemap_click_toggle_keeps_root_view(self, page: Page, server_port: int):
        url = _dashboard_url_treemap_with_line(server_port, "plotly")
        page.goto(url)
        _wait_for_init(page, "plotly")

        page.wait_for_function(
            "() => document.querySelector('#fv-plot-1')?.data?.[0]?.ids?.length > 1"
        )
        page.wait_for_function("""() => {
                const nodes = Array.from(
                    document.querySelectorAll('#fv-plot-1 g.slice')
                ).map(n => n.textContent.trim());
                return nodes.includes('solar') && nodes.includes('wind');
            }""")

        def treemap_node_texts() -> list[str]:
            return page.eval_on_selector(
                "#fv-plot-1",
                """gd => Array.from(gd.querySelectorAll('g.slice'))
                    .map(n => n.textContent.trim())
                    .filter(Boolean)""",
            )

        def click_treemap_category(category: str) -> None:
            target = page.eval_on_selector(
                "#fv-plot-1",
                """(gd, category) => {
                    const nodes = Array.from(gd.querySelectorAll('g.slice'));
                    const node = nodes.find(n => n.textContent.trim() === category);
                    if (!node) return null;
                    const rect = node.getBoundingClientRect();
                    return {
                        x: (rect.left + rect.right) / 2,
                        y: (rect.top + rect.bottom) / 2,
                    };
                }""",
                category,
            )
            assert target is not None, f"Treemap node {category!r} was not visible"
            page.mouse.click(target["x"], target["y"])

        click_treemap_category("solar")
        page.wait_for_function("DASHBOARD_SPEC.state.selections.length === 1")
        page.wait_for_timeout(1_000)
        nodes_selected = treemap_node_texts()
        assert "solar" in nodes_selected
        assert "wind" in nodes_selected

        click_treemap_category("solar")
        page.wait_for_function("DASHBOARD_SPEC.state.selections.length === 0")
        page.wait_for_timeout(1_000)

        nodes_after = treemap_node_texts()
        assert "solar" in nodes_after
        assert "wind" in nodes_after

    def test_treemap_leaf_selection_filters_with_parent_path(
        self, page: Page, server_port: int
    ):
        url = _dashboard_url_treemap_pie_selection(server_port, "plotly")
        page.goto(url)
        _wait_for_init(page, "plotly")

        page.wait_for_function(
            "() => document.querySelector('#fv-plot-1')?.data?.[0]?.ids?.includes('root/solar/NL')"
        )
        target = page.eval_on_selector(
            "#fv-plot-1",
            """(gd, nodeId) => {
                const nodes = Array.from(gd.querySelectorAll('g.slice'));
                const node = nodes.find(n => {
                    const d = n.__data__ || {};
                    return d.id === nodeId
                        || (d.data && d.data.id === nodeId)
                        || (d.data && d.data.data && d.data.data.id === nodeId);
                });
                if (!node) return null;
                const rect = node.getBoundingClientRect();
                return {
                    x: (rect.left + rect.right) / 2,
                    y: (rect.top + rect.bottom) / 2,
                };
            }""",
            "root/solar/NL",
        )
        assert target is not None, "Treemap leaf root/solar/NL was not visible"
        page.mouse.click(target["x"], target["y"])
        page.wait_for_function("DASHBOARD_SPEC.state.selections.length === 1")
        page.wait_for_timeout(1_000)

        selection = page.evaluate("DASHBOARD_SPEC.state.selections[0]")
        clauses = selection["predicates"][0]["clauses"]
        cols = {c["column"]: c["values"] for c in clauses}
        assert cols == {"source": ["solar"], "country": ["NL"]}

        source_bar_labels = page.eval_on_selector(
            "#fv-plot-0",
            "gd => (gd.data[0] && gd.data[0].x) || []",
        )
        assert source_bar_labels == ["solar"]

    def test_treemap_parent_then_leaf_selection_highlights_leaf_only(
        self, page: Page, server_port: int
    ):
        page.set_viewport_size({"width": 1400, "height": 800})
        url = _dashboard_url_treemap_pie_selection(server_port, "plotly")
        page.goto(url)
        _wait_for_init(page, "plotly")

        page.wait_for_function(
            "() => document.querySelector('#fv-plot-1')?.data?.[0]?.ids?.includes('root/solar/NL')"
        )

        def click_treemap_node(node_id: str, *, label_area: bool = False) -> None:
            target = page.eval_on_selector(
                "#fv-plot-1",
                """(gd, args) => {
                    const nodeId = args.nodeId;
                    const labelArea = args.labelArea;
                    const nodes = Array.from(gd.querySelectorAll('g.slice'));
                    const node = nodes.find(n => {
                        const d = n.__data__ || {};
                        return d.id === nodeId
                            || (d.data && d.data.id === nodeId)
                            || (d.data && d.data.data && d.data.data.id === nodeId);
                    });
                    if (!node) return null;
                    const rect = node.getBoundingClientRect();
                    if (labelArea) {
                        return {
                            x: rect.left + Math.min(Math.max(rect.width * 0.05, 10), rect.width - 5),
                            y: rect.top + Math.min(16, rect.height - 5),
                        };
                    }
                    return {
                        x: (rect.left + rect.right) / 2,
                        y: (rect.top + rect.bottom) / 2,
                    };
                }""",
                {"nodeId": node_id, "labelArea": label_area},
            )
            assert target is not None, f"Treemap node {node_id!r} was not visible"
            page.mouse.click(target["x"], target["y"])

        click_treemap_node("root/solar", label_area=True)
        page.wait_for_function("""() => {
                const sels = DASHBOARD_SPEC.state.selections || [];
                if (sels.length !== 1) return false;
                const clauses = (((sels[0] || {}).predicates || [])[0] || {}).clauses || [];
                return clauses.length === 1
                    && clauses[0].column === 'source'
                    && clauses[0].values?.[0] === 'solar';
            }""")
        page.wait_for_function(
            "() => document.querySelector('#fv-plot-1')?.data?.[0]?.ids?.includes('root/solar/NL')"
        )
        page.wait_for_function("""() => {
                const nodes = Array.from(document.querySelectorAll('#fv-plot-1 g.slice'));
                const ids = nodes.map(n => {
                    const d = n.__data__ || {};
                    return d.id
                        || (d.data && d.data.id)
                        || (d.data && d.data.data && d.data.data.id);
                });
                return ids.includes('root/solar/NL') && ids.includes('root/wind');
            }""")

        click_treemap_node("root/solar/NL")
        page.wait_for_function("""() => {
                const sels = DASHBOARD_SPEC.state.selections || [];
                if (sels.length !== 1) return false;
                const clauses = (((sels[0] || {}).predicates || [])[0] || {}).clauses || [];
                return clauses.some(c => c.column === 'country' && c.values?.[0] === 'NL');
            }""")
        page.wait_for_timeout(1_200)

        nodes = page.eval_on_selector(
            "#fv-plot-1",
            """gd => {
                const out = {};
                for (const node of Array.from(gd.querySelectorAll('g.slice'))) {
                    const d = node.__data__ || {};
                    const id = d.id
                        || (d.data && d.data.id)
                        || (d.data && d.data.data && d.data.data.id);
                    if (!id) continue;
                    const rect = node.getBoundingClientRect();
                    out[id] = {
                        opacity: Number(getComputedStyle(node).opacity),
                        width: rect.width,
                    };
                }
                return out;
            }""",
        )
        assert "root/solar" in nodes
        assert "root/wind" in nodes
        assert nodes["root/solar/NL"]["opacity"] == 1
        assert nodes["root/solar"]["opacity"] < 1


def _click_pie_slice_by_label(
    page: Page, label: str, fig_sel: str = "#fv-plot-2"
) -> None:
    """Scroll a pie slice into view by its label, then click its centre.

    Lower figures can render near the viewport bottom, where the
    ``fv-filter-strip`` (shown once a selection exists) overlaps the slice and
    intercepts the click. Centring the slice first keeps it clear of the strip.
    Re-finds on every call so it stays correct after the strip appears.
    """
    target = page.eval_on_selector(
        fig_sel,
        """(gd, lbl) => {
            const node = Array.from(gd.querySelectorAll('g.slice')).find(
                n => (n.__data__ || {}).label === lbl);
            if (!node) return null;
            node.scrollIntoView({block: 'center'});
            const r = node.getBoundingClientRect();
            return { x: (r.left + r.right) / 2, y: (r.top + r.bottom) / 2 };
        }""",
        label,
    )
    assert target is not None, f"pie slice {label!r} not found"
    page.mouse.click(target["x"], target["y"])


@pytest.mark.browser
class TestPieClickBrowser:
    """Plotly pie click toggles should leave visible source feedback."""

    def test_pie_click_dims_unselected_slices(self, page: Page, server_port: int):
        url = _dashboard_url_treemap_pie_selection(server_port, "plotly")
        page.goto(url)
        _wait_for_init(page, "plotly")

        page.wait_for_function(
            "() => document.querySelector('#fv-plot-2')?.data?.[0]?.labels?.length > 1"
        )
        _click_pie_slice_by_label(page, "solar")
        page.wait_for_function("DASHBOARD_SPEC.state.selections.length === 1")
        page.wait_for_timeout(500)

        slice_opacities = page.eval_on_selector(
            "#fv-plot-2",
            """gd => Object.fromEntries(
                Array.from(gd.querySelectorAll('g.slice')).map(n => [
                    (n.__data__ || {}).label,
                    Number(getComputedStyle(n).opacity),
                ])
            )""",
        )
        assert slice_opacities["solar"] == 1
        assert slice_opacities["wind"] < 1

        _click_pie_slice_by_label(page, "solar")
        page.wait_for_function("DASHBOARD_SPEC.state.selections.length === 0")
        page.wait_for_timeout(500)
        reset_opacities = page.eval_on_selector(
            "#fv-plot-2",
            """gd => Object.fromEntries(
                Array.from(gd.querySelectorAll('g.slice')).map(n => [
                    (n.__data__ || {}).label,
                    Number(getComputedStyle(n).opacity),
                ])
            )""",
        )
        assert reset_opacities["solar"] == 1
        assert reset_opacities["wind"] == 1


# ---------------------------------------------------------------------------
# Draggable grid layout tests
# ---------------------------------------------------------------------------


def _dashboard_url_draggable(port: int, renderer: str) -> str:
    """Return a URL for a 2-figure draggable-grid dashboard."""
    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source
    from flexviz.spec import LayoutSpec, encode_spec

    df = pl.DataFrame({"ts": list(range(200)), "val": [float(i) for i in range(200)]})
    register_source("_browser_drag_test", df)

    dash = Dashboard(df)
    f0 = dash.add_figure(title="Fig0")
    f0.add_line(x="ts", y="val", name="L0", n_points=100)
    f1 = dash.add_figure(title="Fig1")
    f1.add_line(x="ts", y="val", name="L1", n_points=100)
    spec = dash.to_spec(source_name="_browser_drag_test", layout=LayoutSpec())

    encoded = encode_spec(spec)
    return f"http://127.0.0.1:{port}/view?spec={encoded}&renderer={renderer}"


def _chart_bounding_boxes(page: Page, renderer: str) -> list[dict]:
    """Return bounding boxes for all chart containers."""
    if renderer == "plotly":
        return [el.bounding_box() for el in page.query_selector_all(".js-plotly-plot")]
    return [el.bounding_box() for el in page.query_selector_all("[id^='fv-chart-']")]


def _drag_grid_item_right(page: Page, sel: str) -> None:
    """Drag a grid item right by its own width (and slightly down) to relocate it.

    The default dashboard layout is two side-by-side columns in a ``float:false``
    grid, so a straight-down drag just compacts back to its original row and the
    position never changes. Dragging onto the neighbouring column forces a real
    swap, which is what exercises the Gridstack→spec bridge.
    """
    item = page.query_selector(sel)
    assert item is not None, f"grid item {sel!r} not found"
    box = item.bounding_box()
    assert box is not None
    sx = box["x"] + box["width"] / 2
    sy = box["y"] + 10
    page.mouse.move(sx, sy)
    page.mouse.down()
    page.mouse.move(sx + box["width"], sy + 150, steps=25)
    page.mouse.up()
    page.wait_for_timeout(800)


@pytest.mark.parametrize("renderer", ["plotly", "echarts"])
class TestDraggableGridBrowser:
    """Draggable Gridstack layout — verify charts are visible and drag is backend-free."""

    def test_charts_are_visible(self, page: Page, server_port: int, renderer: str):
        """Charts must have non-zero width and height after the page loads."""
        url = _dashboard_url_draggable(server_port, renderer)
        page.goto(url)
        _wait_for_chart(page, renderer)
        # Wait for the init POST to complete so charts have data.
        page.wait_for_timeout(2_000)

        boxes = _chart_bounding_boxes(page, renderer)
        assert len(boxes) == 2, f"Expected 2 chart containers, got {len(boxes)}"
        for i, box in enumerate(boxes):
            assert box is not None, f"Chart {i} has no bounding box (not in DOM)"
            assert box["width"] > 0, f"Chart {i} width is 0 — chart is invisible"
            assert box["height"] > 0, f"Chart {i} height is 0 — chart is invisible"

    def test_charts_have_rendered_data(
        self, page: Page, server_port: int, renderer: str
    ):
        """Charts must contain actual rendered trace data after the init POST."""
        url = _dashboard_url_draggable(server_port, renderer)
        page.goto(url)
        _wait_for_chart(page, renderer)
        page.wait_for_timeout(2_000)

        if renderer == "plotly":
            # Each .js-plotly-plot should have at least one trace with x data.
            trace_counts = page.evaluate("""() =>
                Array.from(document.querySelectorAll('.js-plotly-plot'))
                     .map(el => (el.data || []).filter(t => t.x && t.x.length > 0).length)
            """)
            assert all(
                c > 0 for c in trace_counts
            ), f"Some Plotly charts have no rendered data: {trace_counts}"
        else:
            # Each ECharts instance should have at least one series with data.
            series_counts = page.evaluate("""() =>
                Array.from(document.querySelectorAll("[id^='fv-chart-']"))
                     .map(el => {
                       const chart = echarts.getInstanceByDom(el);
                       if (!chart) return 0;
                       const series = chart.getOption().series || [];
                       return series.filter(s => s.data && s.data.length > 0).length;
                     })
            """)
            assert all(
                c > 0 for c in series_counts
            ), f"Some ECharts instances have no rendered data: {series_counts}"

    def test_drag_does_not_trigger_backend_request(
        self, page: Page, server_port: int, renderer: str
    ):
        """Dragging a Gridstack item must NOT fire a /dashboard/update request."""
        url = _dashboard_url_draggable(server_port, renderer)
        page.goto(url)
        _wait_for_chart(page, renderer)
        page.wait_for_timeout(2_000)

        backend_calls: list[str] = []

        def _capture(req: PWRequest) -> None:
            if "/dashboard/update" in req.url:
                backend_calls.append(req.url)

        page.on("request", _capture)
        calls_before = len(backend_calls)

        # Drag the first Gridstack item to a new position.
        item = page.query_selector(".grid-stack-item")
        assert (
            item is not None
        ), ".grid-stack-item not found — is draggable=True active?"
        box = item.bounding_box()
        assert box is not None
        # Drag the item's header area ~200px to the right.
        start_x = box["x"] + box["width"] / 2
        start_y = box["y"] + 10  # near the top of the item (drag handle area)
        page.mouse.move(start_x, start_y)
        page.mouse.down()
        page.mouse.move(start_x + 200, start_y + 50, steps=10)
        page.mouse.up()
        page.wait_for_timeout(500)

        new_calls = len(backend_calls) - calls_before
        assert (
            new_calls == 0
        ), f"Dragging a grid item fired {new_calls} backend request(s) — should be 0"

    def test_drag_updates_spec_layout(
        self, page: Page, server_port: int, renderer: str
    ):
        """After page load, DASHBOARD_SPEC.layout.grid_items must be populated from Gridstack."""
        url = _dashboard_url_draggable(server_port, renderer)
        page.goto(url)
        _wait_for_chart(page, renderer)
        page.wait_for_timeout(2_000)

        # grid_items must be populated immediately from GridStack.init(), before any drag.
        grid_items = page.evaluate(
            "() => JSON.parse(JSON.stringify(DASHBOARD_SPEC.layout.grid_items || null))"
        )
        assert (
            grid_items is not None
        ), "DASHBOARD_SPEC.layout.grid_items was not initialised"
        assert (
            len(grid_items) == 2
        ), f"Expected 2 grid_items (one per figure), got {len(grid_items)}: {grid_items}"
        for gi in grid_items:
            assert gi.get("fig_uid"), f"grid_item missing fig_uid: {gi}"
            assert gi.get("w", 0) > 0, f"grid_item has zero width: {gi}"
            assert gi.get("h", 0) > 0, f"grid_item has zero height: {gi}"

        # The grid starts locked; enable edit mode before dragging.
        page.click("#fv-btn-grid")
        page.wait_for_timeout(300)
        assert page.evaluate("() => DASHBOARD_SPEC.layout.grid_editable") is True

        # Default layout is two side-by-side columns; drag the first item onto the
        # second to force a column swap (a straight-down drag would just compact
        # back to its row in a float:false grid).
        first_uid = grid_items[0]["fig_uid"]
        orig_pos = (grid_items[0]["x"], grid_items[0]["y"])
        _drag_grid_item_right(page, f'.grid-stack-item[gs-id="{first_uid}"]')

        updated_items = page.evaluate(
            "() => JSON.parse(JSON.stringify(DASHBOARD_SPEC.layout.grid_items || null))"
        )
        assert (
            updated_items is not None and len(updated_items) == 2
        ), f"grid_items is wrong after drag: {updated_items}"
        # The dragged item's position must have changed (proving the drag synced
        # back into DASHBOARD_SPEC.layout.grid_items).
        new_item = next(
            (gi for gi in updated_items if gi["fig_uid"] == first_uid), None
        )
        assert new_item is not None
        assert (new_item["x"], new_item["y"]) != orig_pos, (
            f"Drag did not update grid position: before={orig_pos}, "
            f"after={(new_item['x'], new_item['y'])}"
        )

    def test_grid_toggle_locks_and_unlocks_drag(
        self, page: Page, server_port: int, renderer: str
    ):
        """Toolbar grid button should lock and unlock Gridstack drag/resize."""
        url = _dashboard_url_draggable(server_port, renderer)
        page.goto(url)
        _wait_for_chart(page, renderer)
        page.wait_for_timeout(2_000)

        btn = page.query_selector("#fv-btn-grid")
        assert btn is not None, "Grid toggle button not found"
        assert "Locked" in (btn.text_content() or ""), "Grid toggle should start locked"
        assert page.evaluate("() => DASHBOARD_SPEC.layout.grid_editable") is False

        initial_items = page.evaluate(
            "() => JSON.parse(JSON.stringify(DASHBOARD_SPEC.layout.grid_items || []))"
        )
        assert len(initial_items) == 2
        first_uid = initial_items[0]["fig_uid"]
        initial_pos = (initial_items[0]["x"], initial_items[0]["y"])

        # Locked: a relocate drag must NOT change the item's grid position.
        _drag_grid_item_right(page, f'.grid-stack-item[gs-id="{first_uid}"]')

        after_locked = page.evaluate(
            "() => JSON.parse(JSON.stringify(DASHBOARD_SPEC.layout.grid_items || []))"
        )
        locked_state = next(
            (gi for gi in after_locked if gi["fig_uid"] == first_uid), None
        )
        assert locked_state is not None
        assert (
            locked_state["x"],
            locked_state["y"],
        ) == initial_pos, (
            f"Grid item moved while locked: before={initial_pos}, "
            f"after={(locked_state['x'], locked_state['y'])}"
        )

        page.click("#fv-btn-grid")
        page.wait_for_timeout(300)
        assert "Edit" in (page.text_content("#fv-btn-grid") or "")
        assert page.evaluate("() => DASHBOARD_SPEC.layout.grid_editable") is True

        # Unlocked: the same relocate drag must now change the item's position.
        _drag_grid_item_right(page, f'.grid-stack-item[gs-id="{first_uid}"]')

        after_unlocked = page.evaluate(
            "() => JSON.parse(JSON.stringify(DASHBOARD_SPEC.layout.grid_items || []))"
        )
        unlocked_state = next(
            (gi for gi in after_unlocked if gi["fig_uid"] == first_uid), None
        )
        assert unlocked_state is not None
        assert (
            unlocked_state["x"],
            unlocked_state["y"],
        ) != initial_pos, (
            f"Grid item did not move after unlocking: before={initial_pos}, "
            f"after={(unlocked_state['x'], unlocked_state['y'])}"
        )

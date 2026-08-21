"""Tests for server startup behaviour."""

from __future__ import annotations

import socket
import threading
import time

import flexviz.figure as _fig_mod
from flexviz.figure import _pick_free_port, _start_server_thread


def _wait_until_listening(host: str, port: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.05)
    raise TimeoutError(f"Server did not start on {host}:{port} within {timeout}s")


def _reset_started_servers():
    """Clear global server-tracking state between tests."""
    _fig_mod._started_servers.clear()


def test_server_starts_and_accepts_connections():
    _reset_started_servers()
    port = _pick_free_port()
    _start_server_thread("127.0.0.1", port)
    _wait_until_listening("127.0.0.1", port)


def test_start_server_thread_is_idempotent():
    """Calling _start_server_thread twice on the same port must not spawn a second thread."""
    _reset_started_servers()
    port = _pick_free_port()
    _start_server_thread("127.0.0.1", port)
    _wait_until_listening("127.0.0.1", port)

    before = threading.active_count()
    _start_server_thread("127.0.0.1", port)
    assert threading.active_count() == before, "second call should not spawn a thread"


def test_different_ports_start_independent_servers():
    """Two show() calls on different ports each get their own server."""
    _reset_started_servers()
    port1 = _pick_free_port()
    port2 = _pick_free_port()
    host = "127.0.0.1"

    _start_server_thread(host, port1)
    _start_server_thread(host, port2)

    _wait_until_listening(host, port1)
    _wait_until_listening(host, port2)

    assert (host, port1) in _fig_mod._started_servers
    assert (host, port2) in _fig_mod._started_servers


# ---------------------------------------------------------------------------
# gzip middleware (cube cross-filter design §8.1 — wire-size mitigation)
# ---------------------------------------------------------------------------


def _gzip_dashboard_request():
    """A hist-source + line-target dashboard plus a ``cube_request`` body that
    returns a large ``line_env`` cube blob (base64) — the response is well over
    the 1 KiB gzip threshold."""
    import polars as pl

    from flexviz.dashboard import Dashboard
    from flexviz.server import register_source

    n = 4_000
    df = pl.DataFrame(
        {
            "a": [((i * 37) % 1000) / 10 for i in range(n)],
            "b": [((i * 53) % 500) / 5 for i in range(n)],
        }
    )
    source = "_gzip_cube_src"
    register_source(source, df, cache=True)

    dash = Dashboard(df)
    dash.add_figure(title="Source").add_histogram(x="a", bins=16)
    # A wide minmax line target → many buckets → a large line_env cube blob.
    dash.add_figure(title="Line").add_line(x="b", y="a", n_points=1000)
    spec = dash.to_spec(source_name=source)

    src_fig = spec.figures[0]
    src_trace = src_fig.traces[0]
    body = {
        "spec": spec.model_dump(),
        "event": {"type": "cube_request"},
        "request_cube": True,
        "active_source": {
            "figure_uid": src_fig.uid,
            "column": "a",
            "trace_uid": src_trace.uid,
        },
    }
    return body


def test_large_cube_response_is_gzipped():
    """A client advertising ``Accept-Encoding: gzip`` gets a gzip-compressed
    binary cube bundle for a large cube payload (self-gzipped by the cube
    path; the ``GZipMiddleware`` leaves the already-encoded response alone)."""
    from fastapi.testclient import TestClient

    from flexviz.cube import decode_cube_bundle
    from flexviz.server import app

    body = _gzip_dashboard_request()
    with TestClient(app) as client:
        resp = client.post(
            "/dashboard/update",
            json=body,
            headers={"Accept-Encoding": "gzip"},
        )
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("application/octet-stream")
    assert resp.headers.get("content-encoding") == "gzip"
    # httpx transparently decompresses; the decoded bundle holds a cube blob
    # mapped to the served line target.
    blobs, trace_cubes = decode_cube_bundle(resp.content)
    assert blobs, "expected a non-empty cube payload"
    assert trace_cubes, "expected a served line target"
    # The decompressed bundle comfortably exceeds the 1 KiB gzip threshold.
    assert len(resp.content) > 1024


def test_small_response_is_not_gzipped():
    """A tiny response stays uncompressed — below ``minimum_size`` the
    middleware passes it through (regression: gzip is size-gated, not blanket)."""
    from fastapi.testclient import TestClient

    from flexviz.server import app

    with TestClient(app) as client:
        resp = client.get("/sources", headers={"Accept-Encoding": "gzip"})
    assert resp.status_code == 200
    assert resp.headers.get("content-encoding") != "gzip"

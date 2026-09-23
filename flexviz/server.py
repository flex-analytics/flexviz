"""Stateless FastAPI transport layer for flexviz.

Architecture
------------
The server is fully **stateless**: every request carries a complete
``DashboardSpec`` (figure config + current interaction state) so the
server never needs to remember anything between calls.

The only server-side state is the *data-source registry* — a mapping of
named identifiers to ``LFQueryBuilder`` instances.  These are registered
once at startup (or before ``uvicorn.run``) and are read-only thereafter.

    ┌─────────────────────────────────────────────────────┐
    │  _sources: Dict[str, LFQueryBuilder]                │  ← read-only after startup
    │                                                     │
    │  POST /dashboard/update                             │
    │    ← DashboardSpec + InteractionEvent               │  ← full state from client
    │    → {figure_uid: List[TraceDelta]}                 │  ← only changed data
    │  POST /share                                        │
    │  GET  /view                                         │
    │  GET  /h/{n}                                        │  ← reads .flexviz/history.jsonl in cwd
    │  GET  /sources                                      │  ← introspection / health
    │  GET  /cache/stats                                  │
    └─────────────────────────────────────────────────────┘

Usage
-----
Register data sources, then start the server::

    from flexviz.server import app, register_source
    import polars as pl
    import uvicorn

    register_source("sales",  pl.scan_parquet("data/sales.parquet"))
    register_source("events", pl.scan_database("SELECT * FROM events", conn))

    uvicorn.run(app, host="0.0.0.0", port=8000)

Or use ``show_server`` for development / notebooks::

    from flexviz.server import show_server
    show_server(figure, source_name="sales", port=8000)
"""

from __future__ import annotations

import gzip
import logging
import threading
import warnings
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import parse_qs, urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from flexviz.cache import (
    get_cache,
    get_cube_cache,
    is_source_cacheable,
    set_source_cacheable,
)
from flexviz.cube import encode_cube_bundle
from flexviz.engine import FlexEngine, TraceInfo
from flexviz.events import ActiveSource, InteractionEvent, TraceDelta
from flexviz.LF import LFQueryBuilder, polars_lf_from
from flexviz.spec import (
    AxisRange,
    DashboardSpec,
    FigureSpec,
    InteractionState,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data-source registry — the only server-side state.
# Populated before the server starts; read-only during request handling.
# ---------------------------------------------------------------------------

_sources: dict[str, LFQueryBuilder] = {}


def register_source(name: str, data: Any, cache: bool = False) -> None:
    """Register a named data source.

    ``data`` can be a pre-built ``LFQueryBuilder`` (no re-wrapping occurs)
    or anything accepted by ``polars_lf_from``: a Polars DataFrame /
    LazyFrame, a pandas DataFrame, or a PyArrow Table.  For file-backed or
    database-backed sources, pass a pre-built ``pl.LazyFrame`` (e.g.
    ``pl.scan_parquet(...)``).

    Call this before ``uvicorn.run`` or before the first request in tests.

    Parameters
    ----------
    name:
        Identifier that clients reference via ``FigureSpec.source``.
    data:
        A ``LFQueryBuilder`` or raw data to wrap in one.
    cache:
        When ``True``, opt this source into server- and client-side caching
        of the initial (unfiltered) load.  Setting it **asserts the data is
        static for the process lifetime** — there is no data-change
        invalidation yet (see issue #39).  Re-registering an existing name
        with raw data or a new builder replaces the builder and clears both
        caches (its data may have changed). Re-registering with the same
        ``LFQueryBuilder`` object already under that name invalidates
        nothing and emits a ``UserWarning``; use it only to flip ``cache``.
    """
    is_reregistration = name in _sources
    if is_reregistration and data is _sources[name]:
        warnings.warn(
            f"source {name!r} re-registered with the same LFQueryBuilder "
            "object; nothing was invalidated. Pass raw data or a new "
            "builder if the source data changed.",
            UserWarning,
            stacklevel=2,
        )
        _sources[name].cache = cache
        set_source_cacheable(name, cache)
        return
    if isinstance(data, LFQueryBuilder):
        _sources[name] = data
    else:
        _sources[name] = LFQueryBuilder(polars_lf_from(data))
    # The registrar declares the contract, so the source learns it here whoever
    # built the builder.
    _sources[name].cache = cache
    set_source_cacheable(name, cache)
    if is_reregistration:
        # Re-registration may carry new data, so the (now possibly stale)
        # caches are dropped wholesale — keys are hashed and cannot be filtered
        # by source. Both the delta cache and the cube-blob cache derive from
        # source data, so both are cleared. A first-time or unrelated
        # registration leaves the caches alone.
        get_cache().clear()
        get_cube_cache().clear()


def get_source(name: str) -> LFQueryBuilder:
    """Retrieve a registered source or raise ``KeyError``."""
    try:
        return _sources[name]
    except KeyError:
        raise KeyError(f"Unknown source {name!r}. Registered: {list(_sources)}")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class DashboardRequest(BaseModel):
    """Everything the server needs to process one dashboard interaction.

    Contains a full ``DashboardSpec`` (all figures + shared interaction state)
    and the triggering ``InteractionEvent``. The client echoes the spec back
    on every request so the server remains completely stateless.

    ``request_cube`` + ``active_source`` opt one ``"cube_request"`` event into
    the cube-assembly path (no deltas are computed); the default ``False``
    keeps the plain delta path at zero cube cost.
    """

    spec: DashboardSpec
    event: InteractionEvent
    request_cube: bool = False
    active_source: ActiveSource | None = None


class DashboardResponse(BaseModel):
    """Per-figure trace deltas keyed by ``FigureSpec.uid``.

    Using a dict keyed by figure uid (rather than a positional list) keeps
    the response unambiguous even if figure ordering diverges between client
    and server.

    A ``request_cube`` + ``"cube_request"`` pair is answered out-of-band with a
    binary cube bundle (``application/octet-stream``), not this JSON model — see
    ``_cube_response``.
    """

    figure_deltas: dict[str, list[dict[str, Any]]]


class ShareRequest(BaseModel):
    """Request body for ``POST /share``.

    ``spec`` is the raw JSON dict of either a ``VisualizationSpec`` or a
    ``DashboardSpec`` — the JS client sends its current in-memory spec.
    ``server_url`` is the base URL of this server, used to construct the
    returned shareable URL.
    """

    spec: dict[str, Any]
    server_url: str


# ---------------------------------------------------------------------------
# App factory helpers
# ---------------------------------------------------------------------------


def _viewport_state_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, AxisRange):
        return value.as_tuple()
    return [[point[0], point[1]] for point in value]


def _figure_viewport_from_state(
    state: InteractionState,
    figure_uid: str,
) -> dict[str, Any]:
    viewport: dict[str, Any] = {}
    for key, value in state.viewport.items():
        prefix, _, axis_id = key.partition("/")
        if prefix == figure_uid:
            viewport[axis_id] = _viewport_state_value(value)
    return viewport


def _viewports_by_figure(
    state: InteractionState,
    figures: list[FigureSpec],
) -> dict[str, dict[str, tuple[Any, Any] | None]]:
    """Build per-figure viewport ranges from shared interaction state."""
    return {fig.uid: _figure_viewport_from_state(state, fig.uid) for fig in figures}


def _serialise_updates(updates: dict[str, Any], uid: str) -> dict[str, Any]:
    """Serialise a single ``updates`` dict to JSON-safe types."""
    out: dict[str, Any] = {}
    for k, v in updates.items():
        if hasattr(v, "tolist"):
            v = v.tolist()
        if not isinstance(v, (list, dict, str, int, float, bool, type(None))):
            raise TypeError(
                f"TraceDelta uid={uid!r} key={k!r}: "
                f"value of type {type(v).__name__!r} is not JSON-serialisable"
            )
        out[k] = v
    return out


def _deltas_to_json(deltas: list[TraceDelta]) -> list[dict[str, Any]]:
    """Serialise ``TraceDelta`` objects to plain JSON-safe dicts.

    Converts numpy arrays to Python lists; raises ``TypeError`` early if an
    update value is not serialisable so callers get a clear error rather than
    a cryptic 500 at response time.  Serialises grouped child payloads when a
    grouped parent delta includes ``group_results``.
    """
    result = []
    for d in deltas:
        updates = _serialise_updates(d.updates, d.uid)
        item: dict[str, Any] = {"uid": d.uid, "updates": updates}
        if d.group_results is not None:
            item["group_results"] = [
                {
                    "uid": cr.uid,
                    "updates": _serialise_updates(cr.updates, cr.uid),
                    "group_value_key": cr.group_value_key,
                    "parent_uid": cr.parent_uid,
                }
                for cr in d.group_results
            ]
        if d.layer is not None:
            item["layer"] = d.layer
        result.append(item)
    return result


#: Cube bundles are concatenated binary numeric arrays — they barely compress
#: past level 1, and higher levels burn dramatically more CPU for a negligible
#: size gain (level 6 was ~5x the time of level 1 on a multi-target dashboard
#: for <4% smaller wire). Level 1 keeps the cached-cube TTFB low.
_CUBE_GZIP_LEVEL = 1


def _encode_cube_bundle(
    blobs: list[bytes], trace_cubes: dict[str, int], gzip_ok: bool
) -> tuple[bytes, str | None]:
    """Pack the blobs into a bundle, gzip-compressing it when the client
    accepts gzip. Returns ``(body, content_encoding)`` (encoding ``None`` when
    sent uncompressed). Pure/CPU-bound — run off the event loop."""
    bundle = encode_cube_bundle(blobs, trace_cubes)
    if gzip_ok:
        # mtime=0 keeps the compressed bytes deterministic for a given bundle.
        return gzip.compress(bundle, compresslevel=_CUBE_GZIP_LEVEL, mtime=0), "gzip"
    return bundle, None


async def _run_cube_path(
    engine: FlexEngine,
    trace_infos: list[TraceInfo],
    viewports_by_figure: dict[str, dict[str, Any]],
    state: InteractionState,
    active_source: ActiveSource | None,
    source_name: str | None,
    gzip_ok: bool,
) -> tuple[bytes, str | None]:
    """Build the cubes off the event loop and return the encoded bundle body.

    Cubes are only built/cached/served for ``cache=True`` sources (spec §7:
    same "data is static for the process" contract as the delta cache);
    everything else yields an empty (but well-formed) bundle. The blobs ride a
    raw binary envelope (``encode_cube_bundle``) rather than base64-in-JSON, so
    the gzip step compresses binary, not 33%-inflated text — see the
    ``GZipMiddleware`` note below for why that matters for TTFB.
    """
    blobs: list[bytes] = []
    trace_cubes: dict[str, int] = {}
    if active_source is not None and is_source_cacheable(source_name):
        try:
            blobs, trace_cubes = await run_in_threadpool(
                engine.build_cubes,
                trace_infos,
                viewports_by_figure,
                list(state.selections),
                active_source,
                get_cube_cache(),
            )
        except Exception as exc:
            logger.exception("engine.build_cubes failed: %s", exc)
            raise HTTPException(status_code=500, detail="Cube build failed") from exc
    return await run_in_threadpool(_encode_cube_bundle, blobs, trace_cubes, gzip_ok)


def _cube_response(body: bytes, content_encoding: str | None) -> Response:
    """Wrap an encoded cube bundle as a binary response. When the body was
    gzipped (``content_encoding="gzip"``) the ``Content-Encoding`` header makes
    the ``GZipMiddleware`` pass it through untouched (no double compression)."""
    headers = {"Vary": "Accept-Encoding"}
    if content_encoding is not None:
        headers["Content-Encoding"] = content_encoding
    return Response(
        content=body, media_type="application/octet-stream", headers=headers
    )


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Lifespan handler — sources are registered by the caller before startup."""
    logger.info("flexviz server starting. Sources: %s", list(_sources))
    yield
    logger.info("flexviz server shutting down")


app = FastAPI(title="flexviz", version="0.1", lifespan=_lifespan)

# Wire-size mitigation (cube cross-filter design §8.1): gzip every JSON
# response over 1 KiB (delta / share / view). Requests must send
# `Accept-Encoding: gzip` (every browser does).
#
# Cube bundles do NOT ride this path — they are binary octet-stream responses
# that gzip themselves at a fixed low level (``_cube_response``) and set
# ``Content-Encoding``, so this middleware passes them through untouched.
# ``compresslevel=1`` (not Starlette's default 9): a large delta payload is
# numeric JSON that compresses fine at level 1, and higher levels cost
# multiples of the CPU for a few percent smaller wire — not worth the TTFB on
# an interactive path. See ``test_gzip_compresslevel_reduced``.
app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=1)

app.add_middleware(
    CORSMiddleware,
    # Override in production: set CORS_ORIGINS env-var or subclass Settings.
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# Both /update and /dashboard/update return their declared JSON ``response_model``
# *except* for an ``event.type == "cube_request"``, where they return a binary
# cube bundle (``application/octet-stream``; see ``_cube_response``). Declare that
# extra media type so the generated OpenAPI schema does not advertise JSON only.
_CUBE_BUNDLE_RESPONSE: dict[int | str, dict[str, Any]] = {
    200: {
        "content": {
            "application/octet-stream": {
                "schema": {"type": "string", "format": "binary"}
            }
        },
        "description": (
            "Per-trace JSON deltas, or a binary cube bundle when "
            "``event.type == 'cube_request'``."
        ),
    }
}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/sources", response_model=list[str])
async def list_sources() -> list[str]:  # type: ignore[return]
    """Return the names of all registered data sources.

    Useful for health checks and frontend introspection.
    """
    return list(_sources)


@app.get("/cache/stats")
async def cache_stats() -> dict[str, Any]:
    """Return cache backend stats (entries/hits/misses) and cacheable sources.

    Introspection mirror of ``/sources``; the cache is content-addressed and
    session-invariant, so these numbers carry no per-client state.
    """
    from flexviz.cache import cacheable_sources

    return {
        "backend": get_cache().stats(),
        "cacheable_sources": sorted(cacheable_sources()),
    }


@app.post("/share")
async def share(req: ShareRequest) -> dict[str, str]:
    """Encode a spec dict to a shareable ``/view`` URL.

    The spec is gzip-compressed and base64url-encoded so it fits in a URL
    query parameter.  Returns ``{"url": "<server_url>/view?spec=<encoded>"}``."""
    from flexviz.spec import encode_spec

    encoded = encode_spec(req.spec)
    url = f"{req.server_url.rstrip('/')}/view?spec={encoded}"
    return {"url": url}


def _render_spec_html(spec: str, renderer: str, server_url: str) -> HTMLResponse:
    """Decode ``spec`` and render it with the chosen adapter.

    Shared by ``/view`` and ``/h/{n}``, which differ only in where the
    encoded spec comes from and what page-relative ``server_url`` the
    rendered page needs to reach its API routes.
    """
    from flexviz.spec import DashboardSpec as _DashboardSpec
    from flexviz.spec import decode_spec

    try:
        decoded = decode_spec(spec)
        if isinstance(decoded, _DashboardSpec):
            dash_spec = decoded
        else:
            # Wrap single-figure spec into a 1-figure dashboard.
            dash_spec = _DashboardSpec(figures=[decoded.figure], state=decoded.state)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid spec: {exc}") from exc
    from flexviz.adapters import build_adapter, validate_dashboard_renderer

    try:
        renderer_name = validate_dashboard_renderer(renderer, dash_spec)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    adapter = build_adapter(renderer_name)
    html = adapter._build_dashboard_html(dash_spec, server_url=server_url)
    return HTMLResponse(content=html)


@app.get("/view", response_class=HTMLResponse)
async def view(spec: str, renderer: str = "plotly") -> HTMLResponse:
    """Render a spec encoded as a ``spec`` query parameter.

    Decodes the ``spec`` string (produced by ``POST /share``), detects
    whether it is a ``VisualizationSpec`` or a ``DashboardSpec``, and
    returns a self-contained HTML page rendered by the chosen adapter.

    Parameters
    ----------
    spec:
        URL-safe base64-encoded gzip-compressed JSON spec string.
    renderer:
        ``"plotly"`` (default) or ``"echarts"``.
    """
    # Page-relative base: every API endpoint is a sibling of /view, so "."
    # resolves correctly in the browser behind any reverse proxy — including
    # ones that strip a path prefix, where request.base_url would lose the
    # external prefix and scheme (e.g. nginx stripping a /dashboard prefix).
    return _render_spec_html(spec, renderer, server_url=".")


@app.get("/h/{n}", response_class=HTMLResponse)
async def history_view(n: int, renderer: str | None = None) -> HTMLResponse:
    """Render history entry ``n`` at a short, stable page URL.

    Browser tools echo a tab's page URL in every snapshot, so an agent opens
    ``/h/N`` instead of the several-kilobyte share URL it stands for. The
    route reads the history file in the server's working directory for this
    request and stores nothing, so it exposes whatever share URLs that one
    file holds: do not serve a public dashboard from a directory that has
    one.

    Parameters
    ----------
    n:
        1-based entry number, as recorded by ``flexviz history add``.
    renderer:
        ``"plotly"`` or ``"echarts"``; defaults to the renderer in the
        recorded URL, else ``"plotly"``.
    """
    from flexviz import history
    from flexviz.spec import encoded_spec_from_url

    try:
        record = history.entry(n)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"no history entry {n}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    url = record["url"]
    try:
        encoded = encoded_spec_from_url(url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    # show() appends the session's renderer to the URL it opens, so a
    # recorded URL can name one; an explicit query parameter still wins.
    if renderer is None:
        renderer = parse_qs(urlsplit(url).query).get("renderer", ["plotly"])[0]
    # "/h/N" sits one path segment deeper than "/view", so the page-relative
    # base used there (".") would resolve API calls under "/h/" instead of
    # the server root. One more ".." undoes that, keeping the same
    # reverse-proxy-safe, page-relative design as /view.
    return _render_spec_html(encoded, renderer, server_url="..")


@app.post(
    "/dashboard/update",
    response_model=DashboardResponse,
    responses=_CUBE_BUNDLE_RESPONSE,
)
async def dashboard_update(
    req: DashboardRequest, request: Request
) -> DashboardResponse:
    """Process one interaction event across all figures in a dashboard.

    The endpoint is fully stateless: every call is self-contained.

    Steps
    -----
    1. Collect distinct source names across all figures; resolve each once.
    2. Reconstruct ``FlexTrace`` objects for every trace in every figure.
       Track ``trace.uid -> figure.uid`` for delta partitioning.
    3. For each distinct source, build a ``FlexEngine`` and call
       ``engine.process`` with the traces belonging to that source.
    4. Partition the returned ``TraceDelta`` list by figure uid.
    5. Serialise and return ``DashboardResponse``.

    The engine itself is unaware of figure boundaries — it sees a flat
    ``TraceInfo`` list and returns a flat ``TraceDelta`` list.  Partitioning
    is purely a server-layer concern.
    """
    from flexviz.trace import build_trace_from_spec  # local import avoids cycle

    event = req.event

    # -- 1. resolve unique sources --------------------------------------------
    source_map: dict[str, LFQueryBuilder | None] = {}
    for fig_spec in req.spec.figures:
        name = fig_spec.source
        if name not in source_map:
            if name is None:
                source_map[name] = None
            else:
                try:
                    source_map[name] = get_source(name)
                except KeyError as exc:
                    raise HTTPException(status_code=404, detail=str(exc))

    # -- 2. reconstruct FlexTrace objects; build uid → figure_uid map ----------
    uid_to_fig_uid: dict[str, str] = {}
    # source_name → (scalable_traces, trace_infos)
    per_source_traces: dict[str | None, tuple[dict[str, Any], list[TraceInfo]]] = {
        name: ({}, []) for name in source_map
    }

    for fig_spec in req.spec.figures:
        scalable, infos = per_source_traces[fig_spec.source]
        for t_spec in fig_spec.traces:
            trace = build_trace_from_spec(t_spec)
            scalable[trace.uid] = trace
            infos.append(
                TraceInfo(
                    uid=trace.uid,
                    axes=t_spec.axes,
                    trace_type=t_spec.trace_type,
                    figure_uid=fig_spec.uid,
                )
            )
            uid_to_fig_uid[trace.uid] = fig_spec.uid

    viewports_by_figure = _viewports_by_figure(req.spec.state, req.spec.figures)

    # -- cube path: assemble cubes for the active source's figure; no deltas.
    #    The response is a binary cube bundle, not JSON deltas. ----------------
    if event.type == "cube_request":
        # Substring match, not q-value parsing — intentionally mirrors Starlette's
        # GZipMiddleware (which handles the JSON path); no real client (browser or
        # the flexviz runtime) sends ``gzip;q=0``, so the two stay consistent.
        gzip_ok = "gzip" in request.headers.get("accept-encoding", "")
        body: bytes
        enc: str | None
        active = req.active_source
        src_fig = (
            next((f for f in req.spec.figures if f.uid == active.figure_uid), None)
            if active is not None
            else None
        )
        if req.request_cube and src_fig is not None and src_fig.source is not None:
            scalable, infos = per_source_traces[src_fig.source]
            cube_engine = FlexEngine(
                backend_lf=source_map[src_fig.source],
                scalable_traces=scalable,
                source_name=src_fig.source,
            )
            body, enc = await _run_cube_path(
                cube_engine,
                infos,
                viewports_by_figure,
                req.spec.state,
                active,
                src_fig.source,
                gzip_ok,
            )
        else:
            body, enc = await run_in_threadpool(_encode_cube_bundle, [], {}, gzip_ok)
        return _cube_response(body, enc)

    # -- 3. run engine per distinct source -------------------------------------
    all_deltas: list[TraceDelta] = []
    for src_name, (scalable, infos) in per_source_traces.items():
        if not infos:
            continue

        engine = FlexEngine(
            backend_lf=source_map[src_name],
            scalable_traces=scalable,
            cache_backend=get_cache() if is_source_cacheable(src_name) else None,
            source_name=src_name,
        )
        try:
            src_deltas: list[TraceDelta] = await run_in_threadpool(
                engine.process,
                event,
                infos,
                viewports_by_figure,
                req.spec.state.cross_filter_mode,
            )
        except Exception as exc:
            logger.exception("engine.process failed for source %r: %s", src_name, exc)
            raise HTTPException(status_code=500, detail="Aggregation failed") from exc
        all_deltas.extend(src_deltas)

    # -- 4. partition by figure uid -------------------------------------------
    figure_deltas: dict[str, list[dict[str, Any]]] = {
        fig_spec.uid: [] for fig_spec in req.spec.figures
    }
    try:
        for delta in all_deltas:
            fig_uid = uid_to_fig_uid.get(delta.uid)
            if fig_uid is not None and fig_uid in figure_deltas:
                serialised = _deltas_to_json([delta])
                figure_deltas[fig_uid].extend(serialised)
    except TypeError as exc:
        logger.exception("delta serialisation failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return DashboardResponse(figure_deltas=figure_deltas)


# ---------------------------------------------------------------------------
# Mount helper (used for shared-server deployment)
# ---------------------------------------------------------------------------


def mount_into(host_app: Any, prefix: str = "/flexviz") -> None:
    """Mount the flexviz FastAPI app on *host_app* under *prefix*.

    Works with FastAPI / Starlette (``host_app.mount``).  For Flask/WSGI
    hosts, use ``werkzeug.middleware.dispatcher.DispatcherMiddleware``
    instead.

    The mounted app carries its own ``GZipMiddleware(minimum_size=1024)``
    (cube cross-filter design §8.1 — the wire-size mitigation), so all
    flexviz JSON responses (deltas, cube blobs, share/view) are gzip-encoded
    for clients that advertise ``Accept-Encoding: gzip``, independent of the
    host app's own middleware.

    Parameters
    ----------
    host_app:
        A FastAPI or Starlette application instance.
    prefix:
        URL prefix under which the flexviz routes will be available.
    """
    if hasattr(host_app, "mount"):
        host_app.mount(prefix, app)
    else:
        raise TypeError(
            f"host_app of type {type(host_app).__name__!r} does not support "
            ".mount(). Use werkzeug.middleware.dispatcher.DispatcherMiddleware "
            "for Flask/WSGI hosts."
        )


# ---------------------------------------------------------------------------
# Development helper
# ---------------------------------------------------------------------------


def show_server(
    figure,
    source_name: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    **kwargs,
) -> threading.Thread:
    """Start a uvicorn server in a background daemon thread.

    Intended for notebooks and development scripts.  For production use
    ``uvicorn.run`` (or gunicorn + UvicornWorker) directly.

    Parameters
    ----------
    figure:
        An ``AbstractScalableFigure`` instance.  Its backend LazyFrame is
        registered automatically under ``source_name`` (if provided and not
        already registered).
    source_name:
        Name to register the figure's ``backend_lf`` under.  If ``None``
        the figure is assumed to be in-memory only (no shared LazyFrame).
    host:
        Bind address (default ``"127.0.0.1"`` — loopback only).
    port:
        Port number.
    **kwargs:
        Extra keyword arguments forwarded to ``uvicorn.run``.

    Returns
    -------
    threading.Thread
        The daemon thread running the server.
    """
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError(
            "uvicorn is required to run the server. "
            "Install it with:  pip install uvicorn"
        ) from exc

    if source_name is not None and source_name not in _sources:
        if figure._backend_lf is None:
            raise ValueError(
                f"source_name={source_name!r} given but figure has no backend_lf"
            )
        register_source(source_name, figure._backend_lf._ldf)

    thread = threading.Thread(
        target=uvicorn.run,
        args=(app,),
        kwargs={"host": host, "port": port, **kwargs},
        daemon=True,
    )
    thread.start()
    logger.info("flexviz server running at http://%s:%d", host, port)
    return thread

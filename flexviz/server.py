"""Stateless FastAPI transport layer for flexviz.

Architecture
------------
The server is fully **stateless**: every request carries a complete
``DashboardSpec`` (figure config + current interaction state) so the
server never needs to remember anything between calls.

The only server-side state is the *data-source registry* — a mapping of
named identifiers to ``LFQueryBuilder`` instances.  These are registered
once at startup (or before ``uvicorn.run``). A stream (``register_stream``)
swaps in a new builder on each append.

    ┌─────────────────────────────────────────────────────┐
    │  _sources: Dict[str, LFQueryBuilder]                │  ← read-only after startup
    │                                                     │
    │  POST /dashboard/update                             │
    │    ← DashboardSpec + InteractionEvent               │  ← full state from client
    │    → {figure_uid: List[TraceDelta]}                 │  ← only changed data
    │  POST /share                                        │
    │  GET  /view                                         │
    │  GET  /h/{n}   (run_server only, not ``app``)       │  ← reads .flexviz/history.jsonl in cwd
    │  GET  /sources                                      │  ← introspection / health
    │  GET  /sources/{name}/version                       │  ← stream version, polled
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

    uvicorn.run(app, host="127.0.0.1", port=8000)

``flexviz serve`` and ``show()`` start it through ``run_server``, which also
serves ``/h/{n}`` and on a loopback bind refuses a foreign ``Host`` header.
``app`` alone, mounted or run directly, has no ``/h/{n}``.

The server sends no CORS headers: in a browser, only pages it serves itself
(``/view``, ``/h/{n}``) call it, from their own origin, so another site's page
cannot read its responses. Any HTTP client that reaches the port still can.
"""

from __future__ import annotations

import gzip
import ipaddress
import logging
import math
import numbers
import re
import socket
import threading
import warnings
from collections.abc import Iterable
from contextlib import asynccontextmanager
from datetime import timedelta, timezone
from decimal import Decimal
from typing import Any
from urllib.parse import parse_qs, urlsplit

import polars as pl
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
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
    VisualizationSpec,
    check_axis_types,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data-source registry — the only server-side state.
# Populated before the server starts. A stream's append swaps its builder.
# ---------------------------------------------------------------------------

_sources: dict[str, LFQueryBuilder] = {}
_streams: dict[str, Stream] = {}
# Guards every write to both registries and each stream append, so a builder,
# its stream handle and the version change together.
# ponytail: one lock for all streams; an append holds it for about 0.1 ms.
_registry_lock = threading.RLock()


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

        ``cache=False`` does not make FlexViz read your data again. A
        resident frame is a snapshot: rows you add to it in place later are
        not seen. For data that grows, use ``register_stream``.
    """
    # Build first: invalid data must leave the registry as it was.
    builder = (
        data
        if isinstance(data, LFQueryBuilder)
        else LFQueryBuilder(polars_lf_from(data))
    )
    with _registry_lock:
        # A new registration replaces a stream too; the old handle then
        # refuses appends, so it cannot swap its data back in.
        _streams.pop(name, None)
        # The registrar declares the contract, so the source learns it here
        # whoever built the builder.
        builder.cache = cache
        set_source_cacheable(name, cache)
        is_reregistration = name in _sources
        if is_reregistration and builder is _sources[name]:
            warnings.warn(
                f"source {name!r} re-registered with the same LFQueryBuilder "
                "object; nothing was invalidated. Pass raw data or a new "
                "builder if the source data changed.",
                UserWarning,
                stacklevel=2,
            )
            return
        _sources[name] = builder
    if is_reregistration:
        # Re-registration may carry new data, so the (now possibly stale)
        # caches are dropped wholesale — keys are hashed and cannot be filtered
        # by source. Both the delta cache and the cube-blob cache derive from
        # source data, so both are cleared. A first-time or unrelated
        # registration leaves the caches alone.
        get_cache().clear()
        get_cube_cache().clear()


class Stream:
    """A resident source that grows by appends. ``register_stream`` returns it.

    Each append builds a new ``LFQueryBuilder`` and swaps it into the
    registry. A request resolves its builder once, so it reads one version.
    Polars chunks are immutable, so an older builder keeps its rows.
    ``version`` starts at 0 and goes up by one on each append that adds rows.
    """

    def __init__(
        self,
        name: str,
        frame: pl.DataFrame,
        order_by: str,
        window: timedelta | float | None,
        version: int,
    ) -> None:
        self.name = name
        self.order_by = order_by
        self.window = window
        self.version = version
        self._frame = frame

    def append(self, data: Any) -> None:
        """Append rows to the stream.

        ``data`` is a Polars DataFrame or LazyFrame, a pandas DataFrame or
        an Arrow table; a LazyFrame is collected. The rows must be sorted by
        ``order_by``, without nulls or NaN in it, and start at or after the
        last row of the stream. Its columns must match the stream's names,
        order and dtypes.

        Thread-safe: appends from several threads run one after the other.

        Raises
        ------
        ValueError
            If the rows are out of order.
        RuntimeError
            If a later registration replaced this stream.
        """
        batch = _collected(data)
        _check_order(batch, self.order_by)
        if batch.is_empty():
            return
        with _registry_lock:
            if _streams.get(self.name) is not self:
                raise RuntimeError(f"stream {self.name!r} was replaced")
            last = self._frame.get_column(self.order_by)
            if len(last) and batch[self.order_by][0] < last[-1]:
                raise ValueError(
                    f"appended rows start before the last '{self.order_by}' "
                    f"of stream {self.name!r} ({last[-1]})"
                )
            # rechunk=False keeps an append O(batch). A rechunk now and then
            # keeps the chunk count, and so the per-request cost, bounded.
            frame = pl.concat([self._frame, batch], rechunk=False)
            if frame.n_chunks() > 64:
                frame = frame.rechunk()
            self._frame = frame
            _sources[self.name] = _stream_builder(frame, self.order_by)
            self.version += 1


def _collected(data: Any) -> pl.DataFrame:
    if isinstance(data, pl.DataFrame):
        # The caller may extend its frame in place; the stream keeps its own.
        return data.clone()
    return polars_lf_from(data).collect(engine="streaming")


def _check_order(frame: pl.DataFrame, order_by: str) -> None:
    col = frame.get_column(order_by)
    if (
        col.has_nulls()
        or (col.dtype.is_float() and col.is_nan().any())
        or not col.is_sorted()
    ):
        raise ValueError(f"'{order_by}' must be sorted ascending, without nulls or NaN")


def _check_window(window: Any, dtype: pl.DataType) -> None:
    if window is None:
        return
    if isinstance(dtype, pl.Datetime):
        valid = isinstance(window, timedelta) and window > timedelta(0)
    else:
        valid = (
            dtype.is_numeric()
            and isinstance(window, numbers.Real)
            and not isinstance(window, bool)
            and math.isfinite(window)
            and window > 0
        )
    if not valid:
        raise ValueError(
            "window must be a positive timedelta for a Datetime order_by, "
            f"or a positive finite number for a numeric one; got {window!r} "
            f"for dtype {dtype}"
        )


def _stream_builder(frame: pl.DataFrame, order_by: str) -> LFQueryBuilder:
    builder = LFQueryBuilder(frame)
    # The append checks keep order_by sorted, so no request checks it again.
    builder.assume_sorted(order_by)
    return builder


def register_stream(
    name: str,
    data: Any,
    order_by: str,
    window: timedelta | float | None = None,
) -> Stream:
    """Register a named source that grows over time, and return its handle.

    Call ``Stream.append`` to add rows. An open page polls the server once a
    second and refreshes every chart when rows were added. A stream is
    resident (held in memory) and never cached.

    Show it with a dashboard that has no data of its own, so the dashboard
    reads the stream by name::

        stream = register_stream("live", df, order_by="timestamp")
        dash = Dashboard()
        dash.add_figure().add_line("timestamp", "value")
        dash.show(source_name="live")

    Registering a name again replaces the stream, and the old handle then
    raises on append.

    Parameters
    ----------
    name:
        Identifier that clients reference via ``FigureSpec.source``.
    data:
        The first rows: a Polars DataFrame or LazyFrame, a pandas DataFrame
        or an Arrow table. A LazyFrame is collected. It can be empty.
    order_by:
        The column that orders the rows, such as a timestamp. The rows must be
        sorted by it, without nulls or NaN, and each append must start at or
        after the last row.
    window:
        A trailing x range: a positive ``timedelta`` for a ``Datetime``
        ``order_by``, a positive number for a numeric one. An x axis over
        ``order_by`` that is not zoomed or locked then shows only
        ``[last - window, last]`` and slides as data arrives. A zoom or pan
        shows any range of the history. Other charts, such as a histogram of
        another column, still use all rows.

    Raises
    ------
    ValueError
        If ``data`` is not sorted by ``order_by``, has nulls or NaN in it, or
        ``window`` does not fit the dtype of ``order_by``.
    """
    frame = _collected(data)
    _check_order(frame, order_by)
    _check_window(window, frame.schema[order_by])
    builder = _stream_builder(frame, order_by)
    with _registry_lock:
        old = _streams.get(name)
        # Carry the version on, so an open page also refreshes onto the new data.
        stream = Stream(name, frame, order_by, window, old.version + 1 if old else 0)
        register_source(name, builder)
        _streams[name] = stream
    return stream


def stream_versions(names: Iterable[str | None]) -> dict[str, int]:
    """The current version of each name that is a stream."""
    return {n: _streams[n].version for n in names if n in _streams}


def get_source(name: str) -> LFQueryBuilder:
    """Retrieve a registered source or raise ``KeyError``."""
    try:
        return _sources[name]
    except KeyError:
        raise KeyError(f"Unknown source {name!r}. Registered: {list(_sources)}")


def _check_axis_types(spec: DashboardSpec) -> None:
    """Check the axes against the registered source schemas.

    The spec validator cannot see the data types. A source that is not
    registered is skipped. Raises ``ValueError``.
    """
    check_axis_types(
        spec, lambda name: _sources[name].schema if name in _sources else None
    )


def _validated_dashboard(spec: VisualizationSpec | DashboardSpec) -> DashboardSpec:
    """Run the trace checks on a decoded spec and return the spec to render.

    A decoded spec skips the trace constructors, so this builds every trace.
    The figure checks already ran when ``FigureSpec`` parsed it. The returned
    spec holds what the rebuilt traces emit, so the page sees normalized
    values, such as the Plotly spelling of an older lowercase ``color_scale``.
    Raises on an invalid spec.
    """
    from flexviz.trace import build_trace_from_spec

    if isinstance(spec, VisualizationSpec):
        spec = DashboardSpec(figures=[spec.figure], state=spec.state)
    _check_axis_types(spec)
    figures = []
    for fig in spec.figures:
        # The same domain source as Figure.to_spec(), for a spec without a
        # group_domain_key.
        domain_source = fig.source or fig.uid
        traces = [
            build_trace_from_spec(ts).to_trace_spec(domain_source=domain_source)
            for ts in fig.traces
        ]
        # model_copy skips the FigureSpec checks; a rebuilt trace keeps its
        # trace_type and bar_mode, so they still hold.
        figures.append(fig.model_copy(update={"traces": traces}))
    return spec.model_copy(update={"figures": figures})


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


def _fill_stream_viewports(
    spec: DashboardSpec,
    source_map: dict[str | None, LFQueryBuilder | None],
    viewports_by_figure: dict[str, dict[str, Any]],
) -> list[str]:
    """Give an unzoomed x axis over a stream's ``order_by`` the range it shows.

    A locked axis gets its pinned range: the client pins only the display, and
    rows that arrive later must not stretch the grid past it. Otherwise, with
    ``window``, the axis gets ``[last - window, last]``. The range has the form
    the client sends for a zoom, so the traces aggregate only that range. The
    request state is not changed, so the client still sees the axis at
    autorange. Other axes keep the client's ranges: a pinned bar or heatmap
    range has half-bin padding, which a zoom grid would snap to an extra bin.

    Returns the viewport keys pinned to a lock.
    """
    client = spec.client_state
    pinned_keys = []
    for fig in spec.figures:
        stream = _streams.get(fig.source)
        if stream is None:
            continue
        viewport = viewports_by_figure[fig.uid]
        axes = {
            ts.axes[0]
            for ts in fig.traces
            if ts.axes
            and ts.backend_data.get("x") == stream.order_by
            and viewport.get(ts.axes[0]) is None
        }
        window = None
        for axis in axes:
            key = f"{fig.uid}/{axis}"
            pinned = client.axis_lock_ranges.get(key)
            # The client keys locks by axis family: "x" also covers "x2".
            if pinned is not None and client.axis_locks.get(f"{fig.uid}/{axis[0]}"):
                viewport[axis] = pinned.as_tuple()
                pinned_keys.append(key)
                continue
            if stream.window is None:
                continue
            if window is None:
                window = _stream_window(stream, source_map[fig.source])
            if window is not None:
                viewport[axis] = window
    return pinned_keys


def _stream_window(stream: Stream, builder: LFQueryBuilder) -> tuple[Any, Any] | None:
    """``[last - window, last]`` over the rows of ``builder``, or None if empty.

    ``builder`` is the one this request resolved, not the live stream: an
    append since then must not move the window past the rows.
    """
    last = pl.col(stream.order_by).last()
    if not isinstance(stream.window, timedelta):
        # Python math: exact on integers, and no unsigned wrap below 0 (the
        # range filter clamps that bound). A Decimal takes the window as a
        # Decimal, exact to the default 28 digits.
        hi = builder._ldf.select(last).collect(engine="in-memory").item()
        if hi is None:
            return None
        window = stream.window
        if isinstance(hi, Decimal):
            integral = isinstance(window, numbers.Integral)
            window = Decimal(str(window if integral else float(window)))
        return hi - window, hi
    # Polars does the time math in absolute time, so it holds across a DST
    # change. A Python datetime holds microseconds, so the upper bound rounds
    # up to keep the last row of a nanosecond column.
    lo, hi = (
        builder._ldf.select(
            (last - stream.window).alias("lo"),
            (last + timedelta(microseconds=1)).alias("hi"),
        )
        .collect(engine="in-memory")
        .row(0)
    )
    if hi is None:
        return None
    if hi.tzinfo is not None:
        # In UTC, the text order is the time order, also in a repeated hour.
        lo, hi = lo.astimezone(timezone.utc), hi.astimezone(timezone.utc)
    return lo.isoformat(), hi.isoformat()


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
                state.selections,
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


@app.get("/sources/{name:path}/version")
async def source_version(name: str) -> int:
    """Return a stream's version. A page that shows a stream polls it."""
    stream = _streams.get(name)
    if stream is None:
        raise HTTPException(status_code=404, detail=f"no stream {name!r}")
    return stream.version


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
    from flexviz.spec import encode_spec, parse_spec

    # Apply the checks of /view, so an invalid spec fails here, not later.
    try:
        spec = parse_spec(req.spec)
        _validated_dashboard(spec)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid spec: {exc}") from exc
    url = f"{req.server_url.rstrip('/')}/view?spec={encode_spec(spec)}"
    return {"url": url}


def _render_spec_html(spec: str, renderer: str, server_url: str) -> HTMLResponse:
    """Decode ``spec`` and render it with the chosen adapter.

    Shared by ``/view`` and ``/h/{n}``, which differ only in where the
    encoded spec comes from and what page-relative ``server_url`` the
    rendered page needs to reach its API routes.
    """
    from flexviz.spec import decode_spec

    try:
        dash_spec = _validated_dashboard(decode_spec(spec))
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
        ``"plotly"`` (the only renderer).
    """
    # Page-relative base: every API endpoint is a sibling of /view, so "."
    # resolves correctly in the browser behind any reverse proxy — including
    # ones that strip a path prefix, where request.base_url would lose the
    # external prefix and scheme (e.g. nginx stripping a /dashboard prefix).
    return _render_spec_html(spec, renderer, server_url=".")


async def history_view(n: int, renderer: str | None = None) -> HTMLResponse:
    """Render history entry ``n`` at a short, stable page URL.

    Browser tools echo a tab's page URL in every snapshot, so an agent opens
    ``/h/N`` instead of the several-kilobyte share URL it stands for. The
    route reads the history file in the server's working directory for this
    request and stores nothing, so it exposes whatever share URLs that one
    file holds. Only ``run_server`` serves it, not ``app``, so a mounted or
    embedded ``app`` never reads the history file.

    Parameters
    ----------
    n:
        1-based entry number, as recorded by ``flexviz history add``.
    renderer:
        ``"plotly"``; defaults to the renderer in the
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

    try:
        _check_axis_types(req.spec)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

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
    pinned_keys = _fill_stream_viewports(req.spec, source_map, viewports_by_figure)
    if pinned_keys:
        # A lock sends no request, so this one may be the first to bin over
        # the pinned range: list the axes as changed, so the background
        # re-bins with the foreground.
        event = event.model_copy(
            update={"viewport_keys": [*event.viewport_keys, *pinned_keys]}
        )

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
                req.spec.state.selections,
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

    Works with FastAPI / Starlette (``host_app.mount``).  A Flask/WSGI host
    cannot call an ASGI app: wrap ``app`` in ``a2wsgi.ASGIMiddleware`` and
    mount that with ``werkzeug.middleware.dispatcher.DispatcherMiddleware``.

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
            ".mount(). For Flask/WSGI hosts, wrap flexviz.app in "
            "a2wsgi.ASGIMiddleware and mount that with "
            "werkzeug.middleware.dispatcher.DispatcherMiddleware."
        )


# ---------------------------------------------------------------------------
# Serving
# ---------------------------------------------------------------------------


# A Host header is a name or a bracketed IPv6 literal, then an optional port.
_HOST_HEADER = re.compile(r"(\[[0-9A-Fa-f:.]+\]|[^\[\]:/@\s]+)(?::[0-9]{1,5})?")


def _is_loopback_ip(text: str) -> bool:
    try:
        return ipaddress.ip_address(text).is_loopback
    except ValueError:
        return False


def is_loopback_bind(host: str) -> bool:
    """Whether a server bound to *host* is reachable only from this machine.

    Decided by address, as uvicorn resolves the host to bind it: ``127.1``,
    ``127.0.0.2`` and ``0:0:0:0:0:0:0:1`` are loopback binds too.
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    return all(_is_loopback_ip(info[4][0].split("%")[0]) for info in infos)


def _loopback_host_only(asgi_app: Any, bind_host: str) -> Any:
    """Wrap *asgi_app* so it answers only a loopback ``Host`` header.

    Without CORS headers another site cannot read the responses, but DNS
    rebinding points a name of that site at 127.0.0.1 and so makes its page
    same-origin with this server. The ``Host`` header still carries that name.
    A loopback IP literal cannot be rebound, and the bind host is a name the
    user chose, so both are served.
    """
    bind_name = bind_host.strip("[]").lower()

    async def guarded(scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            host = dict(scope["headers"]).get(b"host", b"").decode("latin-1")
            match = _HOST_HEADER.fullmatch(host)
            hostname = match and match.group(1).strip("[]").lower()
            if not (
                hostname in ("localhost", bind_name)
                or (hostname and _is_loopback_ip(hostname))
            ):
                response = PlainTextResponse("Invalid host header", status_code=400)
                await response(scope, receive, send)
                return
        await asgi_app(scope, receive, send)

    return guarded


def _agent_app() -> FastAPI:
    """``app`` plus ``/h/{n}``, the route only the agent loop needs.

    A separate app, so that adding the route never changes ``app`` itself:
    ``show()`` runs the server in the user's process, where a later
    ``mount_into`` must still get the stateless routes only.
    """
    # No docs routes of its own: /openapi.json and /docs fall through to app.
    agent = FastAPI(lifespan=_lifespan, openapi_url=None, docs_url=None, redoc_url=None)
    agent.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=1)
    agent.add_api_route("/h/{n}", history_view, response_class=HTMLResponse)
    agent.mount("/", app)
    return agent


def run_server(host: str, port: int, log_level: str = "warning") -> None:
    """Serve ``app`` and ``/h/{n}`` with uvicorn until the process stops.

    ``flexviz serve`` and ``show()`` start the server here. A loopback bind
    serves only loopback ``Host`` names. Another bind is a deliberate network
    exposure whose host names are not known here, so it serves every name.
    """
    import uvicorn

    agent = _agent_app()
    served = _loopback_host_only(agent, host) if is_loopback_bind(host) else agent
    uvicorn.run(served, host=host, port=port, log_level=log_level)

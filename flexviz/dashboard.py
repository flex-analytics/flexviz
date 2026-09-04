"""User-facing Dashboard API for flexviz.

Example usage::

    import polars as pl
    from flexviz.dashboard import Dashboard

    df = pl.read_parquet("sensors.parquet")
    dash = Dashboard(df)

    fig1 = dash.add_figure()
    fig1.add_line(x="timestamp", y="temperature", name="Temp")

    fig2 = dash.add_figure()
    fig2.add_line(x="timestamp", y="humidity", name="Hum")

    dash.show(renderer="plotly", rows=1)

Cross-filtering
---------------
All figures added to a ``Dashboard`` share the same backend ``LFQueryBuilder``
(data source).  When the user brushes / zooms one figure, the same
``InteractionEvent`` is sent for all figures via ``POST /dashboard/update``
so every trace is re-aggregated with the updated viewport and cross-filter
expressions applied consistently.
"""

from __future__ import annotations

from typing import Any, List, Literal
from uuid import uuid4

import polars as pl

from .figure import (
    Figure,
    _effective_live_brush,
    _register_source_if_needed,
    _render_dashboard,
    _start_server_thread,
)
from .LF import LFQueryBuilder, polars_lf_from
from .spec import (
    DashboardSpec,
    InteractionState,
    LayoutSpec,
    _auto_grid_items,
    encode_spec,
)


class Dashboard:
    """Container for multiple related ``Figure`` objects with a shared data source.

    All figures added via :meth:`add_figure` share the same backend
    ``LFQueryBuilder`` so cross-filtering and viewport synchronisation work
    across the entire dashboard without duplicating data.

    The ``Dashboard`` is responsible for:

    - Creating and owning ``Figure`` instances.
    - Serialising all figures to a ``DashboardSpec`` for the server.
    - Starting the FastAPI backend and delegating rendering to an adapter.

    It does *not* own any interaction state — state travels with the spec
    on every ``/dashboard/update`` request.
    """

    def __init__(
        self,
        data: pl.DataFrame | pl.LazyFrame | None = None,
        cache: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        data:
            A Polars DataFrame or LazyFrame (or anything accepted by
            ``polars_lf_from``).  All figures will share this data source.
            Pass ``None`` for in-memory-only dashboards.
        cache:
            Opt the shared source into init-load caching.  Asserts the data
            is static for the process lifetime (no invalidation yet; see
            ``register_source`` and issue #27).  Can be overridden in
            :meth:`show`.
        """
        self._uid: str = str(uuid4())
        self._backend_lf: LFQueryBuilder | None = None
        if data is not None:
            self._backend_lf = LFQueryBuilder(polars_lf_from(data))

        self._figures: List[Figure] = []
        self._cache_enabled: bool = cache

    # ------------------------------------------------------------------
    # Figure management
    # ------------------------------------------------------------------

    def add_figure(self, **layout_kw: Any) -> Figure:
        """Create a new ``Figure`` that shares this dashboard's data source.

        The returned ``Figure`` is pre-configured with the shared backend
        ``LFQueryBuilder``.  Call ``.add_line()``, ``.add_histogram()``, etc.
        on it to add traces, then call :meth:`show` on the ``Dashboard`` to
        render everything together.

        Parameters
        ----------
        **layout_kw:
            Layout hints passed directly to ``Figure.update_layout``
            (e.g. ``title="Temperature"``, ``height=400``).

        Returns
        -------
        Figure
            A new ``Figure`` instance registered with this dashboard.
        """
        # Construct a Figure whose backend points at the same LFQueryBuilder.
        # We bypass Figure.__init__'s data-wrapping by passing None and then
        # manually assigning _backend_lf so no second copy is made.
        fig = Figure.__new__(Figure)
        from uuid import uuid4

        fig._uid = str(uuid4())
        fig._backend_lf = self._backend_lf
        fig._traces = []
        fig._layout = {}

        if layout_kw:
            fig.update_layout(**layout_kw)

        self._figures.append(fig)
        return fig

    # ------------------------------------------------------------------
    # Spec serialisation
    # ------------------------------------------------------------------

    # TODO: a bit awkward that we have to pass source_name here + the layout
    # -> also the case in figure.py
    def to_spec(
        self,
        source_name: str = "data",
        layout: LayoutSpec | None = None,
    ) -> DashboardSpec:
        """Serialise all figures to a ``DashboardSpec``.

        Parameters
        ----------
        source_name:
            Data source name as registered with ``register_source()`` on the
            server.  All figures reference this same source name.
        layout:
            Optional ``LayoutSpec`` override.

        Returns
        -------
        DashboardSpec
        """
        src = source_name if self._backend_lf is not None else None
        figure_specs = [fig.to_spec(source=src).figure for fig in self._figures]
        return DashboardSpec(
            figures=figure_specs,
            state=InteractionState(),
            layout=layout or LayoutSpec(),
        )

    def save_spec(
        self,
        path: str,
        source_name: str = "data",
        layout: LayoutSpec | None = None,
    ) -> None:
        """Serialise the dashboard spec to a JSON file.

        Parameters
        ----------
        path:
            Filesystem path to write (creates or overwrites the file).
        source_name:
            Forwarded to :meth:`to_spec`.
        layout:
            Forwarded to :meth:`to_spec`.
        """
        import pathlib

        spec = self.to_spec(source_name=source_name, layout=layout)
        pathlib.Path(path).write_text(spec.model_dump_json(indent=2), encoding="utf-8")

    @staticmethod
    def load_spec(path: str) -> DashboardSpec:
        """Load a ``DashboardSpec`` from a JSON file written by :meth:`save_spec`.

        Parameters
        ----------
        path:
            Filesystem path to read.

        Returns
        -------
        DashboardSpec
        """
        import json
        import pathlib

        data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        return DashboardSpec.model_validate(data)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _finalized_spec(
        self,
        source_name: str | None,
        rows: int | None,
        cols: int | None,
        draggable: bool,
        effective_cache: bool,
        live_brush: Literal["auto", "off"] | None,
    ) -> DashboardSpec:
        """Build the spec with the client state a renderer needs."""
        spec = self.to_spec(source_name=source_name)
        spec.client_state.live_brush = _effective_live_brush(
            live_brush, effective_cache
        )
        spec.layout.draggable = draggable
        # TODO allow user to specify / pass layout
        spec.layout.grid_items = _auto_grid_items(spec.figures, rows=rows, cols=cols)
        return spec

    def share_url(
        self,
        server_url: str = "http://127.0.0.1:8000",
        source_name: str = "data",
        rows: int | None = None,
        cols: int | None = None,
        draggable: bool = True,
        cache: bool | None = None,
        live_brush: Literal["auto", "off"] | None = None,
    ) -> str:
        """Encode this dashboard as a ``/view`` URL for a running server.

        Unlike :meth:`show`, this neither starts a server nor registers the
        data source, and it does not open a browser.  The server at
        ``server_url`` must already serve a source named ``source_name``
        (for example one started with ``flexviz serve``).  This is the
        primitive for agents and scripts that hand a human a live dashboard
        link instead of rendering locally.

        Parameters
        ----------
        server_url:
            Base URL of the running flexviz server.
        source_name:
            Source name as registered on that server (``flexviz serve``
            registers each file under its stem).
        rows, cols, draggable, live_brush:
            Same meaning as in :meth:`show`.
        cache:
            Resolves ``live_brush`` exactly as in :meth:`show`; pass the
            same value the server used when registering the source.

        Returns
        -------
        str
            A ``{server_url}/view?spec=...`` URL carrying the complete
            spec.  Opening it needs a running server with ``source_name``
            registered.
        """
        effective_cache = self._cache_enabled if cache is None else cache
        spec = self._finalized_spec(
            source_name, rows, cols, draggable, effective_cache, live_brush
        )
        return f"{server_url.rstrip('/')}/view?spec={encode_spec(spec)}"

    def show(
        self,
        renderer: str = "plotly",
        source_name: str | None = None,
        rows: int | None = None,
        cols: int | None = None,
        draggable: bool = True,
        host: str = "127.0.0.1",
        port: int = 8000,
        cache: bool | None = None,
        live_brush: Literal["auto", "off"] | None = None,
        **kwargs: Any,
    ) -> None:
        """Start the FastAPI backend and render all figures as a dashboard.

        Parameters
        ----------
        renderer:
            ``"plotly"`` (default) or ``"echarts"``.
        source_name:
            Name under which the shared backend LazyFrame is registered
            with the server's data-source registry.  Defaults to the
            dashboard's uid so multiple ``show()`` calls never collide.
        rows:
            Optional row count used to seed initial ``layout.grid_items``.
            Mutually exclusive with ``cols``.
        cols:
            Optional column count used to seed initial ``layout.grid_items``.
            Mutually exclusive with ``rows``.
        draggable:
            Enable or disable Gridstack drag/resize interactions.
        host:
            Server bind address.
        port:
            Server port.
        cache:
            Opt the shared source into init-load caching (asserts static
            data).  Defaults to the value passed to the ``Dashboard``
            constructor.
        live_brush:
            Cube live-brush mode, stored on the client state (the server never
            reads it).  ``"auto"`` enables drag-time cube slicing where
            available; ``"off"`` restores mouseup-only selection.  ``None``
            (default) resolves to ``"auto"``.  Live brushing requires
            ``cache=True`` (cubes are only built for cacheable sources), so when
            caching is off this is forced to ``"off"`` — silently for the
            default, with a warning if ``"auto"`` was passed explicitly.
        **kwargs:
            Forwarded to the adapter's ``show_dashboard()`` method.
        """
        if source_name is None and self._backend_lf is not None:
            # TODO: ideally this is a hash of the backend lf's
            # => or we require the user to register the source manually
            # (thus not via the dashboard constructor)
            source_name = self._uid
        effective_cache = self._cache_enabled if cache is None else cache
        _register_source_if_needed(source_name, self._backend_lf, cache=effective_cache)
        _start_server_thread(host, port)
        spec = self._finalized_spec(
            source_name, rows, cols, draggable, effective_cache, live_brush
        )
        _render_dashboard(renderer, spec, f"http://{host}:{port}", **kwargs)

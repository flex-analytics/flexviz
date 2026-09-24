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

from typing import Any, Literal
from uuid import uuid4

import polars as pl
from pydantic import ValidationError

from .figure import (
    Figure,
    _effective_live_brush,
    _register_source_if_needed,
    _render_dashboard,
    _start_server_thread,
)
from .LF import LFQueryBuilder, polars_lf_from
from .spec import (
    ClientState,
    DashboardSpec,
    FigureSpec,
    InteractionState,
    LayoutSpec,
    _auto_grid_items,
    check_axis_link_types,
    encode_spec,
    figure_axis_columns,
)


def _name_figures(message: str, figure_specs: list[FigureSpec]) -> str:
    """Replace figure uids in a builder error with "figure N (title)"."""
    for number, spec in enumerate(figure_specs, 1):
        title = spec.layout.get("title")
        if isinstance(title, dict):
            title = title.get("text")
        label = f"figure {number}" + (f" ({title})" if title else "")
        message = message.replace(f"{spec.uid}/", f"{label}, axis ")
        message = message.replace(spec.uid, label)
    return message


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
            ``register_source`` and issue #39).  Can be overridden in
            :meth:`show`.
        """
        self._uid: str = str(uuid4())
        self._cache_enabled: bool = cache
        self._backend_lf: LFQueryBuilder | None = None
        if data is not None:
            self._backend_lf = LFQueryBuilder(polars_lf_from(data), cache=cache)

        self._figures: list[Figure] = []
        # Raw link_axes calls, resolved in to_spec against the finished figures,
        # so traces and figures added after the call still count.
        self._link_requests: list[tuple[tuple[Figure, ...], str | None, tuple]] = []

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

    def link_axes(
        self,
        *targets: Figure | tuple[Figure, str],
        on: str | None = None,
        axis: str | None = None,
    ) -> Dashboard:
        """Link axes so they zoom, pan, autorange and reset together.

        Three forms, one result:

        - ``link_axes(on="ts")`` links the axis showing column ``ts`` in every
          figure that shows it; ``link_axes(a, b, on="ts")`` only in ``a`` and
          ``b``, each of which must show it.
        - ``link_axes(a, b, axis="x")`` links the same axis of each figure.
        - ``link_axes((a, "x"), (h, "y"))`` links the named axes, for example a
          line's x with a horizontal histogram's y.

        Calls that share an axis merge into one group. Only x and y axes that
        show a numeric or temporal column can be linked (not a histogram's
        count axis, a bar, a map or a log axis), and numeric and temporal axes
        do not mix. Links are resolved when the spec is built, so figures and
        traces added later count too. Raises ``TypeError`` for a target that is
        not a figure or a tuple, and ``ValueError`` on a bad link.

        Returns
        -------
        Dashboard
            ``self``, for chaining.
        """
        for target in targets:
            if not isinstance(target, (Figure, tuple)):
                raise TypeError(
                    "link_axes takes figures or (figure, axis) tuples, not "
                    f"{type(target).__name__}"
                )
        pairs = tuple(t for t in targets if isinstance(t, tuple))
        figures = tuple(t for t in targets if isinstance(t, Figure))
        if pairs and figures:
            raise ValueError("pass figures or (figure, axis) pairs, not both")
        if pairs:
            if on is not None or axis is not None:
                raise ValueError("(figure, axis) pairs name their axes; drop on=/axis=")
            if len(pairs) < 2:
                raise ValueError("link_axes needs two or more (figure, axis) pairs")
            if any(
                len(p) != 2 or not isinstance(p[0], Figure) or not isinstance(p[1], str)
                for p in pairs
            ):
                raise ValueError("each pair must be (figure, axis_id)")
        elif (on is None) == (axis is None):
            raise ValueError("pass exactly one of on= or axis=")
        elif axis is not None and len(figures) < 2:
            raise ValueError("link_axes(axis=...) needs two or more figures")
        for fig in figures or tuple(fig for fig, _ in pairs):
            if not any(fig is own for own in self._figures):
                raise ValueError("link_axes got a figure that is not in this dashboard")
        if on is not None:
            self._link_requests.append((figures, on, ()))
        else:
            members = pairs or tuple((fig, axis) for fig in figures)
            self._link_requests.append(((), None, members))
        return self

    def _resolve_axis_links(self, figure_specs: list[FigureSpec]) -> list[list[str]]:
        """Turn the link_axes calls into merged groups of viewport keys."""
        order = {spec.uid: i for i, spec in enumerate(figure_specs)}
        groups: list[set[str]] = []
        for figures, on, members in self._link_requests:
            if on is None:
                keys = {f"{fig._uid}/{axis_id}" for fig, axis_id in members}
            else:
                keys = self._keys_showing(on, figures, figure_specs)
            for group in [g for g in groups if g & keys]:
                keys |= group
                groups.remove(group)
            groups.append(keys)
        return [
            sorted(g, key=lambda k: (order[k.partition("/")[0]], k)) for g in groups
        ]

    @staticmethod
    def _keys_showing(
        column: str, figures: tuple[Figure, ...], figure_specs: list[FigureSpec]
    ) -> set[str]:
        wanted = {fig._uid for fig in figures}
        keys: set[str] = set()
        for spec in figure_specs:
            if wanted and spec.uid not in wanted:
                continue
            axes = sorted(
                ax for ax, cols in figure_axis_columns(spec).items() if column in cols
            )
            if len(axes) > 1:
                raise ValueError(
                    f"{spec.uid} shows {column!r} on axes {axes}; name one with "
                    "(figure, axis) pairs"
                )
            if axes:
                keys.add(f"{spec.uid}/{axes[0]}")
            elif wanted:
                raise ValueError(f"{spec.uid} shows no {column!r} axis")
        if not keys:
            raise ValueError(f"link_axes(on={column!r}): no figure shows {column!r}")
        if len(keys) < 2:
            raise ValueError(
                f"link_axes(on={column!r}): only one figure shows {column!r}, "
                "so there is nothing to link"
            )
        return keys

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
        try:
            spec = DashboardSpec(
                figures=figure_specs,
                state=InteractionState(),
                client_state=ClientState(
                    axis_links=self._resolve_axis_links(figure_specs)
                ),
                # Copy so rendering never stamps grid_items onto a caller's
                # LayoutSpec, which would leak into the next dashboard reusing it.
                layout=(layout or LayoutSpec()).model_copy(deep=True),
            )
            # The schema is only needed for links: a spec build touches no data.
            if spec.client_state.axis_links and self._backend_lf is not None:
                check_axis_link_types(spec, {src: self._backend_lf.schema})
        except ValidationError as exc:
            messages = [e["msg"].removeprefix("Value error, ") for e in exc.errors()]
            raise ValueError(_name_figures("; ".join(messages), figure_specs)) from None
        except ValueError as exc:
            raise ValueError(_name_figures(str(exc), figure_specs)) from None
        return spec

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
        *,
        rows: int | None,
        cols: int | None,
        draggable: bool | None,
        effective_cache: bool,
        live_brush: Literal["auto", "off"] | None,
        layout: LayoutSpec | None,
    ) -> DashboardSpec:
        """Build the spec with the client state a renderer needs.

        Precedence: ``layout`` owns every field it sets.  ``rows``, ``cols``
        and ``draggable`` are convenience overrides that apply on top of it,
        so positions are auto-generated only when ``layout`` carries none.
        """
        # An empty list carries no positions, so it means "generate them",
        # matching how the adapters read the field.
        has_positions = layout is not None and bool(layout.grid_items)
        if has_positions and (rows is not None or cols is not None):
            raise ValueError("rows/cols cannot be combined with layout.grid_items")

        spec = self.to_spec(source_name=source_name, layout=layout)
        spec.client_state.live_brush = _effective_live_brush(
            live_brush, effective_cache
        )
        if draggable is not None:
            spec.layout.draggable = draggable
        if not spec.layout.grid_items:
            spec.layout.grid_items = _auto_grid_items(
                spec.figures, rows=rows, cols=cols
            )
        return spec

    def share_url(
        self,
        server_url: str = "http://127.0.0.1:8000",
        source_name: str = "data",
        rows: int | None = None,
        cols: int | None = None,
        draggable: bool | None = None,
        cache: bool | None = None,
        live_brush: Literal["auto", "off"] | None = None,
        layout: LayoutSpec | None = None,
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
        rows, cols, draggable, layout, live_brush:
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
            source_name,
            rows=rows,
            cols=cols,
            draggable=draggable,
            effective_cache=effective_cache,
            live_brush=live_brush,
            layout=layout,
        )
        return f"{server_url.rstrip('/')}/view?spec={encode_spec(spec)}"

    def show(
        self,
        renderer: str = "plotly",
        source_name: str | None = None,
        rows: int | None = None,
        cols: int | None = None,
        draggable: bool | None = None,
        host: str = "127.0.0.1",
        port: int = 8000,
        cache: bool | None = None,
        live_brush: Literal["auto", "off"] | None = None,
        block: bool = True,
        layout: LayoutSpec | None = None,
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
            Mutually exclusive with ``cols`` and with explicit
            ``layout.grid_items``.
        cols:
            Optional column count used to seed initial ``layout.grid_items``.
            Mutually exclusive with ``rows`` and with explicit
            ``layout.grid_items``.
        draggable:
            Enable or disable the GridStack layout. ``False`` renders a static
            layout with no drag/resize capability and hides the toolbar's
            layout button. ``True`` enables GridStack; it starts locked unless
            ``layout.grid_editable=True``. ``None`` (default) keeps whatever
            ``layout`` says.
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
        block:
            Outside a notebook, ``show()`` blocks until Ctrl-C.  Pass
            ``block=False`` to return at once.  Ignored in a notebook.
        layout:
            Optional ``LayoutSpec`` owning ``gap``, ``toolbar`` and explicit
            ``grid_items``.  ``rows``, ``cols`` and ``draggable`` override it.
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
            source_name,
            rows=rows,
            cols=cols,
            draggable=draggable,
            effective_cache=effective_cache,
            live_brush=live_brush,
            layout=layout,
        )
        _render_dashboard(
            renderer, spec, f"http://{host}:{port}", block=block, **kwargs
        )

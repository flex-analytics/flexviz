"""Request-wide unfiltered-domain resolution and per-source-kind engine pinning.

Bin edges come from the unfiltered frame, so a cross-filter never moves them.
The engine resolves every column a request needs in one min/max collect, and
every collect names its Polars engine: streaming for a scan, in-memory for a
resident frame.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from flexviz.cache import InMemoryLRUCache
from flexviz.engine import FlexEngine, TraceInfo
from flexviz.events import InteractionEvent
from flexviz.LF import LFQueryBuilder
from flexviz.spec import (
    ClauseFilter,
    SelectionPredicate,
    SelectionState,
)
from flexviz.trace.hist import Histogram
from flexviz.trace.hist2d import Histogram2D
from flexviz.trace.line import LinePlot

# ---- helpers ---------------------------------------------------------------


class _CollectSpy:
    """Every ``LazyFrame.collect`` of a test: its engine kwarg and its plan."""

    def __init__(self) -> None:
        self.calls: list[tuple[str | None, str]] = []

    @property
    def engines(self) -> list[str | None]:
        return [engine for engine, _ in self.calls]

    @property
    def minmax(self) -> list[tuple[str | None, str]]:
        # ``physical_minmax`` is the only producer of ``__min_<col>__`` aliases.
        return [call for call in self.calls if "__min_" in call[1]]


@pytest.fixture()
def collects(monkeypatch) -> _CollectSpy:
    spy = _CollectSpy()
    real = pl.LazyFrame.collect

    def _spy(self, *args, **kwargs):
        spy.calls.append((kwargs.get("engine"), self.explain(optimized=False)))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(pl.LazyFrame, "collect", _spy)
    return spy


def _engine(src, traces, **kwargs):
    lf = LFQueryBuilder(src)
    engine = FlexEngine(
        backend_lf=lf, scalable_traces={t.uid: t for t in traces}, **kwargs
    )
    infos = [
        TraceInfo(
            uid=t.uid, axes=("x", "y"), trace_type=t.trace_type, figure_uid=f"f{i}"
        )
        for i, t in enumerate(traces)
    ]
    return engine, infos


def _init(engine, infos):
    return engine.process(InteractionEvent(type="init", force_update=True), infos)


def _write(path, df: pl.DataFrame):
    df.write_parquet(path)
    return pl.scan_parquet(path)


# ---- one resolve per request -----------------------------------------------


class TestResolveCount:
    def test_viewport_bounds_need_no_minmax_collect(self, collects):
        """A viewport already carries its domain; nothing to scan for."""
        df = pl.DataFrame({"a": [float(i) for i in range(50)]})
        engine, infos = _engine(df, [Histogram(x="a", bins=10)])
        engine.process(
            InteractionEvent(type="viewport", axis_ranges={"x": [5, 25]}), infos
        )
        assert collects.minmax == []

    def test_one_collect_covers_every_column_in_the_request(self, tmp_path, collects):
        """Two histograms, a hist2d and a scan line resolve in a single scan."""
        src = _write(
            tmp_path / "d.parquet",
            pl.DataFrame(
                {
                    "a": [float(i) for i in range(200)],
                    "b": [float(i % 7) for i in range(200)],
                    "ts": list(range(200)),
                }
            ),
        )
        engine, infos = _engine(
            src,
            [
                Histogram(x="a", bins=10),
                Histogram(x="b", bins=10),
                Histogram2D(x="a", y="b", x_bins=4, y_bins=4),
                LinePlot(x="ts", y="a", n_points=20),
            ],
        )
        _init(engine, infos)

        assert len(collects.minmax) == 1
        plan = collects.minmax[0][1]
        assert all(f"__min_{c}__" in plan for c in ("a", "b", "ts"))
        # min/max + the batched select (both histograms and the hist2d) + the
        # line envelope group_by. A fourth would mean the line plan re-probed
        # its own x domain instead of taking the resolved one.
        assert len(collects.calls) == 3

    def test_resident_line_resolves_its_x_domain(self, collects):
        """A resident minmax line bins in x too, so it needs the domain."""
        df = pl.DataFrame({"ts": list(range(200)), "a": [float(i) for i in range(200)]})
        engine, infos = _engine(df, [LinePlot(x="ts", y="a", n_points=20)])
        _init(engine, infos)

        assert len(collects.minmax) == 1
        assert "__min_ts__" in collects.minmax[0][1]
        # The x check (null count and order in one collect), the min/max, then
        # the aggregation select.
        assert len(collects.calls) == 3

    @pytest.mark.parametrize("n_traces", [1, 5])
    def test_scan_collect_counts(self, tmp_path, collects, n_traces):
        """One min/max scan plus one batched select, however many histograms."""
        src = _write(
            tmp_path / "d.parquet",
            pl.DataFrame({"a": [float(i) for i in range(200)]}),
        )
        engine, infos = _engine(
            src, [Histogram(x="a", bins=10) for _ in range(n_traces)]
        )
        _init(engine, infos)

        assert len(collects.minmax) == 1
        assert len(collects.calls) == 2


# ---- fixed engine per source kind ------------------------------------------


class TestEnginePinning:
    def test_resident_frame_collects_in_memory(self, collects):
        df = pl.DataFrame({"a": [float(i) for i in range(50)]})
        engine, infos = _engine(df, [Histogram(x="a", bins=10)])
        _init(engine, infos)

        assert len(collects.minmax) == 1
        assert set(collects.engines) == {"in-memory"}

    def test_scan_collects_streaming(self, tmp_path, collects):
        src = _write(
            tmp_path / "d.parquet",
            pl.DataFrame({"a": [float(i) for i in range(200)]}),
        )
        engine, infos = _engine(src, [Histogram(x="a", bins=10)])
        _init(engine, infos)

        assert len(collects.minmax) == 1
        assert set(collects.engines) == {"streaming"}


# ---- memoization contract ---------------------------------------------------


class TestBoundsFreshness:
    def _centers(self, src):
        engine, infos = _engine(src, [Histogram(x="a", bins=4)])
        return list(_init(engine, infos)[0].updates["x"])

    def test_uncached_reset_sees_changed_source_data(self, tmp_path):
        path = tmp_path / "d.parquet"
        before = self._centers(_write(path, pl.DataFrame({"a": [0.0, 10.0]})))
        after = self._centers(_write(path, pl.DataFrame({"a": [0.0, 100.0]})))
        assert before != after
        assert max(after) > max(before)

    def test_memoized_bounds_survive_a_source_rewrite(self, tmp_path):
        """``cache=True`` contracts the data as static, so the memo stands."""
        path = tmp_path / "d.parquet"
        lf = LFQueryBuilder(_write(path, pl.DataFrame({"a": [0.0, 10.0]})))
        assert lf.physical_minmax(["a"], memoize=True) == {"a": (0.0, 10.0)}

        pl.DataFrame({"a": [0.0, 100.0]}).write_parquet(path)
        assert lf.physical_minmax(["a"], memoize=True) == {"a": (0.0, 10.0)}
        # Re-registering a source builds a new builder, dropping the memo.
        fresh = LFQueryBuilder(pl.scan_parquet(path))
        assert fresh.physical_minmax(["a"], memoize=True) == {"a": (0.0, 100.0)}

    def test_unmemoized_call_neither_reads_nor_writes_the_memo(self, tmp_path):
        path = tmp_path / "d.parquet"
        lf = LFQueryBuilder(_write(path, pl.DataFrame({"a": [0.0, 10.0]})))
        assert lf.physical_minmax(["a"], memoize=False) == {"a": (0.0, 10.0)}
        assert lf._minmax_memo == {}

        pl.DataFrame({"a": [0.0, 100.0]}).write_parquet(path)
        assert lf.physical_minmax(["a"], memoize=False) == {"a": (0.0, 100.0)}

    def test_cached_engine_memoizes(self, collects):
        """A second request on a cached source re-uses the resolved bounds."""
        df = pl.DataFrame({"a": [float(i) for i in range(50)]})
        engine, infos = _engine(
            df,
            [Histogram(x="a", bins=4)],
            cache_backend=InMemoryLRUCache(),
            source_name="s",
        )
        _init(engine, infos)
        assert len(collects.minmax) == 1
        # Second request is a full cache hit: it must not resolve at all.
        _init(engine, infos)
        assert len(collects.minmax) == 1


# ---- exact bounds -----------------------------------------------------------


class TestExactBounds:
    def test_large_integers_are_not_rounded(self):
        lo, hi = 2**53 + 1, 2**53 + 12345
        lf = LFQueryBuilder(pl.DataFrame({"a": [lo, hi]}, schema={"a": pl.Int64}))
        assert lf.physical_minmax(["a"], memoize=False) == {"a": (lo, hi)}

    def test_temporal_bounds_stay_integral(self):
        lf = LFQueryBuilder(
            pl.DataFrame({"t": [dt.date(2020, 1, 1), dt.date(2020, 1, 11)]})
        )
        lo, hi = lf.physical_minmax(["t"], memoize=False)["t"]
        assert isinstance(lo, int) and hi - lo == 10

    def test_all_null_column_falls_back_to_unit_domain(self):
        df = pl.DataFrame({"a": pl.Series([None, None], dtype=pl.Float64)})
        engine, infos = _engine(df, [Histogram(x="a", bins=2)])
        centers = list(_init(engine, infos)[0].updates["x"])
        assert centers == pytest.approx([0.25, 0.75])


# ---- equal results across engines and dtypes --------------------------------


def _typed_frame(dtype: pl.DataType) -> pl.DataFrame:
    """200 rows with nulls (and NaN for floats) plus a float partner column."""
    raw = [None if i % 37 == 0 else i for i in range(200)]
    if dtype == pl.Datetime("us"):
        base = dt.datetime(2020, 1, 1)
        a = pl.Series(
            "a", [None if v is None else base + dt.timedelta(minutes=v) for v in raw]
        )
    else:
        if dtype.is_float():
            raw = [None if v is None else float(v) for v in raw]
            raw[11] = float("nan")
        a = pl.Series("a", raw, dtype=dtype)
    b = pl.Series("b", [None if v is None else float(v % 13) for v in raw])
    return pl.DataFrame([a, b])


_DTYPES = [pl.Float32, pl.Float64, pl.Int64, pl.Datetime("us")]


class TestEngineAndDtypeEquivalence:
    def _updates(self, src, trace_factory):
        trace = trace_factory()
        engine, infos = _engine(src, [trace])
        return _init(engine, infos)[0].updates

    @pytest.mark.parametrize("dtype", _DTYPES, ids=str)
    @pytest.mark.parametrize(
        "factory",
        [
            lambda: Histogram(x="a", bins=8),
            lambda: Histogram2D(x="a", y="b", x_bins=4, y_bins=4),
        ],
        ids=["hist", "hist2d"],
    )
    def test_streaming_matches_in_memory(self, tmp_path, dtype, factory):
        df = _typed_frame(dtype)
        path = tmp_path / "d.parquet"
        df.write_parquet(path)

        resident = self._updates(df, factory)
        scanned = self._updates(pl.scan_parquet(path), factory)
        assert resident.keys() == scanned.keys()
        for key in resident:
            assert resident[key] == scanned[key], key

    def test_constant_column_bins_every_row(self):
        """lo == hi is degenerate but must still count every value."""
        df = pl.DataFrame({"a": [3.0] * 25})
        engine, infos = _engine(df, [Histogram(x="a", bins=5)])
        counts = list(_init(engine, infos)[0].updates["y"])
        assert sum(counts) == 25


# ---- edges do not move under cross-filtering --------------------------------


class TestFilterStableEdges:
    """A cross-filter narrows the data; the bin edges must not follow it."""

    @pytest.fixture()
    def df(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "a": [float(i) for i in range(100)],
                "b": [float(i % 9) for i in range(100)],
                "g": ["u" if i % 2 else "v" for i in range(100)],
                "k": list(range(100)),
            }
        )

    def _edges(self, df, trace, selections):
        lf = LFQueryBuilder(df)
        source = LinePlot(x="k", y="a", n_points=50)
        engine = FlexEngine(
            backend_lf=lf, scalable_traces={trace.uid: trace, source.uid: source}
        )
        infos = [
            TraceInfo(
                uid=source.uid, axes=("x", "y"), trace_type="line", figure_uid="src"
            ),
            TraceInfo(
                uid=trace.uid,
                axes=("x", "y"),
                trace_type=trace.trace_type,
                figure_uid="tgt",
            ),
        ]
        event = InteractionEvent(
            type="selection" if selections else "init",
            force_update=True,
            selections=selections,
        )
        delta = next(d for d in engine.process(event, infos) if d.uid == trace.uid)
        if delta.group_results:  # grouped parents carry edges on their children
            return [c.updates["x"] for c in delta.group_results]
        return delta.updates["x"]

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: Histogram(x="a", bins=8),
            lambda: Histogram(x="a", bins=8, group_by="g"),
            lambda: Histogram2D(x="a", y="b", x_bins=4, y_bins=4),
        ],
        ids=["hist", "grouped_hist", "hist2d"],
    )
    def test_edges_survive_a_cross_filter(self, df, factory):
        trace = factory()
        selection = [
            SelectionState(
                source_figure_uid="src",
                predicates=[
                    SelectionPredicate(
                        clauses=[ClauseFilter(column="k", range=(0, 20))]
                    )
                ],
            )
        ]
        assert self._edges(df, trace, []) == self._edges(df, trace, selection)


# ---- domains contract: unzoomed traces require their columns ----------------


class TestDomainsContract:
    """An unzoomed trace must fail loudly when ``domains`` lacks its column."""

    @staticmethod
    def _schema() -> pl.Schema:
        return pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}).schema

    def test_histogram_missing_domain_raises(self):
        trace = Histogram(x="a", bins=4)
        with pytest.raises(KeyError, match="a"):
            trace.get_aggregation_spec({}, schema=self._schema(), domains={})

    def test_histogram2d_missing_domain_raises(self):
        trace = Histogram2D(x="a", y="b", x_bins=2, y_bins=2)
        with pytest.raises(KeyError, match="b"):
            trace.get_aggregation_spec(
                {}, schema=self._schema(), domains={"a": (0.0, 1.0)}
            )

    @pytest.mark.parametrize("scan_source", [False, True])
    def test_line_missing_domain_raises(self, scan_source):
        df = pl.DataFrame({"k": [1.0, 2.0], "a": [3.0, 4.0]})
        trace = LinePlot(x="k", y="a", n_points=10)
        with pytest.raises(KeyError, match="k"):
            trace.get_aggregation_spec(
                {}, schema=df.schema, scan_source=scan_source, domains={}
            )

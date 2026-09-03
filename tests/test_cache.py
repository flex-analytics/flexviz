"""Tests for the Phase-1 init-load cache (issue #26)."""

from __future__ import annotations

import polars as pl
import pytest
from fastapi.testclient import TestClient

import flexviz.cache as cache_mod
from flexviz.cache import InMemoryByteLRUCache, InMemoryLRUCache, content_key
from flexviz.engine import FlexEngine, TraceInfo, _AggregationTrace
from flexviz.events import GroupedChildDelta, InteractionEvent, TraceDelta
from flexviz.LF import LFQueryBuilder
from flexviz.spec import TraceSpec
from flexviz.trace import build_trace_from_spec
from flexviz.trace.base import child_uid_from_group_key

# ---------------------------------------------------------------------------
# content_key
# ---------------------------------------------------------------------------


def test_content_key_is_stable():
    a = content_key("s", "line", ("x", "y"), {"x": "x"}, {"n": 1})
    b = content_key("s", "line", ("x", "y"), {"x": "x"}, {"n": 1})
    assert a == b


def test_content_key_independent_of_dict_order():
    a = content_key("s", "line", ("x", "y"), {"x": "x", "y": "y"}, {"a": 1, "b": 2})
    b = content_key("s", "line", ("x", "y"), {"y": "y", "x": "x"}, {"b": 2, "a": 1})
    assert a == b


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(source_name="other"),
        dict(trace_type="bar"),
        dict(axes=("x",)),
        dict(backend_data={"x": "z"}),
        dict(params={"n": 2}),
        # A histogram binning over a sibling's range is a different result
        # than the same histogram alone (shared bin domain).
        dict(domain_cols=("x", "sibling")),
    ],
)
def test_content_key_sensitive_to_each_field(kwargs):
    base = dict(
        source_name="s",
        trace_type="line",
        axes=("x", "y"),
        backend_data={"x": "x"},
        params={"n": 1},
        domain_cols=("x",),
    )
    changed = {**base, **kwargs}
    assert content_key(**base) != content_key(**changed)


# ---------------------------------------------------------------------------
# InMemoryLRUCache
# ---------------------------------------------------------------------------


def test_lru_get_set_and_stats():
    c = InMemoryLRUCache(max_entries=10)
    assert c.get("k") is None
    c.set("k", {"v": 1})
    assert c.get("k") == {"v": 1}
    s = c.stats()
    assert s["entries"] == 1 and s["hits"] == 1 and s["misses"] == 1


def test_lru_evicts_least_recently_used():
    c = InMemoryLRUCache(max_entries=2)
    c.set("a", 1)
    c.set("b", 2)
    assert c.get("a") == 1  # touch a -> b is now LRU
    c.set("c", 3)  # evicts b
    assert c.get("b") is None
    assert c.get("a") == 1 and c.get("c") == 3


def test_lru_clear_resets_counters():
    c = InMemoryLRUCache()
    c.set("a", 1)
    c.get("a")
    c.clear()
    s = c.stats()
    assert s == {
        "entries": 0,
        "hits": 0,
        "misses": 0,
        "max_entries": c.stats()["max_entries"],
    }


# ---------------------------------------------------------------------------
# InMemoryByteLRUCache (cube blob cache)
# ---------------------------------------------------------------------------


def test_byte_lru_get_set_and_stats():
    c = InMemoryByteLRUCache(max_bytes=100)
    assert c.get("k") is None
    c.set("k", b"x" * 10)
    assert c.get("k") == b"x" * 10
    s = c.stats()
    assert s["entries"] == 1
    assert s["bytes"] == 10
    assert s["hits"] == 1 and s["misses"] == 1
    assert s["max_bytes"] == 100


def test_byte_lru_evicts_by_bytes():
    c = InMemoryByteLRUCache(max_bytes=100)
    c.set("a", b"x" * 40)
    c.set("b", b"x" * 40)
    assert c.get("a") is not None  # touch a -> b is now LRU
    c.set("c", b"x" * 40)  # 120 bytes total -> evicts b
    assert c.get("b") is None
    assert c.get("a") is not None and c.get("c") is not None
    assert c.stats()["bytes"] == 80


def test_byte_lru_replace_updates_byte_account():
    c = InMemoryByteLRUCache(max_bytes=100)
    c.set("a", b"x" * 40)
    c.set("a", b"x" * 10)  # replace, not accumulate
    assert c.stats()["bytes"] == 10
    assert c.get("a") == b"x" * 10


def test_byte_lru_oversized_entry_not_cached():
    """An entry larger than the whole budget is not cached and does not crash
    (and must not evict everything else for nothing)."""
    c = InMemoryByteLRUCache(max_bytes=50)
    c.set("small", b"x" * 10)
    c.set("huge", b"x" * 51)
    assert c.get("huge") is None
    assert c.get("small") == b"x" * 10
    assert c.stats()["bytes"] == 10


def test_byte_lru_clear_resets():
    c = InMemoryByteLRUCache(max_bytes=100)
    c.set("a", b"x" * 10)
    c.get("a")
    c.clear()
    s = c.stats()
    assert s["entries"] == 0 and s["bytes"] == 0
    assert s["hits"] == 0 and s["misses"] == 0


def test_default_cube_cache_budget_is_512_mb():
    assert InMemoryByteLRUCache().stats()["max_bytes"] == 512 * 2**20


def test_cube_cache_singleton_swap():
    original = cache_mod.get_cube_cache()
    try:
        replacement = InMemoryByteLRUCache(max_bytes=10)
        cache_mod.set_cube_cache_backend(replacement)
        assert cache_mod.get_cube_cache() is replacement
    finally:
        cache_mod.set_cube_cache_backend(original)


# ---------------------------------------------------------------------------
# Engine integration
# ---------------------------------------------------------------------------


def _line_trace(uid="t1", n_points=50):
    spec = TraceSpec(
        uid=uid,
        trace_type="line",
        axes=("x", "y"),
        backend_data={"x": "x", "y": "y"},
        params={"n_points": n_points, "downsample": "minmax", "add_gaps": False},
    )
    return build_trace_from_spec(spec)


def _engine(cache, source_name="s"):
    lf = LFQueryBuilder(
        pl.DataFrame({"x": list(range(20)), "y": [i % 4 for i in range(20)]})
    )
    return lf


def _run_init(trace, cache, source_name="s"):
    lf = _engine(cache, source_name)
    eng = FlexEngine(
        backend_lf=lf,
        scalable_traces={trace.uid: trace},
        cache_backend=cache,
        source_name=source_name,
    )
    ti = TraceInfo(
        uid=trace.uid, axes=trace._axes, trace_type=trace.trace_type, figure_uid="f"
    )
    ev = InteractionEvent(type="init", axis_ranges={}, selections=[], force_update=True)
    return eng.process(ev, [ti], {"f": {}}, "update")


def test_engine_init_miss_then_hit():
    cache = InMemoryLRUCache()
    tr = _line_trace()
    d1 = _run_init(tr, cache)
    assert cache.stats() == {**cache.stats(), "hits": 0, "misses": 1, "entries": 1}
    d2 = _run_init(tr, cache)
    assert cache.stats()["hits"] == 1
    assert d1[0].updates == d2[0].updates
    assert d2[0].uid == tr.uid


def test_engine_no_cache_when_backend_none():
    tr = _line_trace()
    lf = _engine(None)
    eng = FlexEngine(backend_lf=lf, scalable_traces={tr.uid: tr})  # no cache
    ti = TraceInfo(uid=tr.uid, axes=tr._axes, trace_type="line", figure_uid="f")
    ev = InteractionEvent(type="init", axis_ranges={}, selections=[], force_update=True)
    # Should simply run without touching any cache and return a delta.
    assert eng.process(ev, [ti], {"f": {}}, "update")[0].uid == tr.uid


def test_engine_viewport_event_not_cached():
    cache = InMemoryLRUCache()
    tr = _line_trace()
    lf = _engine(cache)
    eng = FlexEngine(
        backend_lf=lf,
        scalable_traces={tr.uid: tr},
        cache_backend=cache,
        source_name="s",
    )
    ti = TraceInfo(uid=tr.uid, axes=tr._axes, trace_type="line", figure_uid="f")
    ev = InteractionEvent(
        type="viewport",
        axis_ranges={"x": (1, 5)},
        selections=[],
        force_update=True,
        figure_uid="f",
    )
    eng.process(ev, [ti], {"f": {"x": (1, 5)}}, "update")
    assert cache.stats()["entries"] == 0


def test_engine_cross_uid_content_reuse():
    """A second trace with identical identity but a different uid hits the cache
    and is emitted under its own uid (content-only key)."""
    cache = InMemoryLRUCache()
    a = _line_trace(uid="A")
    _run_init(a, cache)
    b = _line_trace(uid="B")  # same identity, different uid
    out = _run_init(b, cache)
    assert cache.stats()["hits"] == 1
    assert out[0].uid == "B"


def test_engine_overlay_init_reuses_update_entry():
    """Unfiltered data is mode-agnostic: an overlay-bg init hits an entry stored
    by an update-mode init and is emitted on the bg layer."""
    cache = InMemoryLRUCache()
    tr = _line_trace()
    _run_init(tr, cache)  # update mode populates
    lf = _engine(cache)
    eng = FlexEngine(
        backend_lf=lf,
        scalable_traces={tr.uid: tr},
        cache_backend=cache,
        source_name="s",
    )
    ti = TraceInfo(uid=tr.uid, axes=tr._axes, trace_type="line", figure_uid="f")
    ev = InteractionEvent(type="init", axis_ranges={}, selections=[], force_update=True)
    out = eng.process(ev, [ti], {"f": {}}, "overlay")
    assert cache.stats()["hits"] == 1
    assert out[0].layer == "bg"


def test_engine_deselect_while_zoomed_is_not_served_from_cache():
    """Regression: a ``deselect`` issued while zoomed depends on the viewport,
    so it must recompute against that viewport rather than serve the full-range
    unfiltered entry populated by ``init``. The content key is viewport-free, so
    correctness relies on the engine *not* caching viewport-dependent traces."""
    cache = InMemoryLRUCache()
    tr = _line_trace()
    _run_init(tr, cache)  # full-range, no-viewport entry
    assert cache.stats()["entries"] == 1

    desel = InteractionEvent(
        type="deselect",
        axis_ranges={},
        selections=[],
        force_update=True,
        figure_uid="f",
    )
    zoomed_vp = {"f": {"x": (1.0, 5.0)}}

    eng = FlexEngine(
        backend_lf=_engine(cache),
        scalable_traces={tr.uid: tr},
        cache_backend=cache,
        source_name="s",
    )
    ti = TraceInfo(uid=tr.uid, axes=tr._axes, trace_type="line", figure_uid="f")
    out = eng.process(desel, [ti], zoomed_vp, "update")

    # Reference: identical deselect computed with no cache at all.
    ref_eng = FlexEngine(backend_lf=_engine(None), scalable_traces={tr.uid: tr})
    ref = ref_eng.process(desel, [ti], zoomed_vp, "update")

    assert cache.stats()["hits"] == 0  # must NOT have served init's full-range entry
    assert out[0].updates == ref[0].updates  # viewport-correct result


def test_engine_mixed_zoom_deselect_does_not_short_circuit_from_cache():
    """In one deselect with one unzoomed and one zoomed figure, the zoomed trace
    must be recomputed, not dropped or served from the unfiltered cache. Because
    the engine only short-circuits all-or-nothing, the mixed request recomputes
    normally instead of partially serving the unzoomed trace from cache."""
    cache = InMemoryLRUCache()
    t1 = _line_trace(uid="t1", n_points=50)
    t2 = _line_trace(uid="t2", n_points=51)  # distinct content key
    lf = _engine(cache)

    def run(event, viewports):
        eng = FlexEngine(
            backend_lf=lf,
            scalable_traces={"t1": t1, "t2": t2},
            cache_backend=cache,
            source_name="s",
        )
        infos = [
            TraceInfo(uid="t1", axes=t1._axes, trace_type="line", figure_uid="f1"),
            TraceInfo(uid="t2", axes=t2._axes, trace_type="line", figure_uid="f2"),
        ]
        return eng.process(event, infos, viewports, "update")

    init = InteractionEvent(
        type="init", axis_ranges={}, selections=[], force_update=True
    )
    run(init, {"f1": {}, "f2": {}})  # populate both unfiltered entries

    desel = InteractionEvent(
        type="deselect", axis_ranges={}, selections=[], force_update=True
    )
    out = run(desel, {"f1": {}, "f2": {"x": (1.0, 5.0)}})
    by_uid = {d.uid: d for d in out}
    assert set(by_uid) == {"t1", "t2"}  # zoomed trace not dropped by short-circuit
    assert cache.stats()["hits"] == 0

    ref_eng = FlexEngine(backend_lf=_engine(None), scalable_traces={"t2": t2})
    ref = ref_eng.process(
        desel,
        [TraceInfo(uid="t2", axes=t2._axes, trace_type="line", figure_uid="f2")],
        {"f2": {"x": (1.0, 5.0)}},
        "update",
    )
    assert by_uid["t2"].updates == ref[0].updates  # viewport-correct, not cached


def test_grouped_payload_is_uid_agnostic_and_restamps():
    """Cached grouped payloads store group_value_key, not the child uid, and
    re-derive child uids for the requesting parent on a hit."""
    eng = FlexEngine(
        backend_lf=None, scalable_traces={}, cache_backend=InMemoryLRUCache()
    )
    delta = TraceDelta(
        uid="parentA",
        updates={},
        group_results=[
            GroupedChildDelta(
                uid=child_uid_from_group_key("parentA", "g1"),
                updates={"x": [1], "y": [2]},
                parent_uid="parentA",
                group_value_key="g1",
            )
        ],
    )
    payload = eng._cache_payload(delta)
    assert "group_results" in payload
    assert payload["group_results"][0]["group_value_key"] == "g1"
    assert "uid" not in payload["group_results"][0]

    item = _AggregationTrace(
        info=TraceInfo(uid="parentB", axes=("x", "y"), trace_type="line"),
        trace=_line_trace(),
        update_range={},
    )
    restamped = eng._delta_from_cached(item, payload, None)
    assert restamped.uid == "parentB"
    child = restamped.group_results[0]
    assert child.uid == child_uid_from_group_key("parentB", "g1")
    assert child.parent_uid == "parentB"


# ---------------------------------------------------------------------------
# Server plumbing
# ---------------------------------------------------------------------------


_SPEC = {
    "version": "0.3",
    "figure": {
        "uid": "f1",
        "source": "s",
        "traces": [
            {
                "uid": "t1",
                "trace_type": "line",
                "axes": ["x", "y"],
                "backend_data": {"x": "x", "y": "y"},
                "params": {"n_points": 50, "downsample": "minmax", "add_gaps": False},
            }
        ],
    },
    "state": {"viewport": {}, "selections": [], "cross_filter_mode": "update"},
}
_INIT = {"type": "init", "axis_ranges": {}, "selections": [], "force_update": True}


@pytest.fixture
def client():
    from flexviz.server import app

    cache_mod.get_cache().clear()
    # Both caches are process-global: a cube entry leaked in by an earlier test
    # makes test_reregister_clears_cube_cache's count assertion order-dependent.
    cache_mod.get_cube_cache().clear()
    yield TestClient(app)


def test_server_caches_when_flag_set(client):
    from flexviz.server import register_source

    register_source(
        "s",
        pl.DataFrame({"x": list(range(40)), "y": [i % 3 for i in range(40)]}),
        cache=True,
    )
    r1 = client.post("/update", json={"spec": _SPEC, "event": _INIT})
    r2 = client.post("/update", json={"spec": _SPEC, "event": _INIT})
    assert r1.status_code == 200 and r1.json() == r2.json()
    assert client.get("/cache/stats").json()["backend"]["hits"] >= 1


def test_server_does_not_cache_without_flag(client):
    from flexviz.server import register_source

    register_source(
        "s",
        pl.DataFrame({"x": list(range(40)), "y": [i % 3 for i in range(40)]}),
        cache=False,
    )
    client.post("/update", json={"spec": _SPEC, "event": _INIT})
    client.post("/update", json={"spec": _SPEC, "event": _INIT})
    stats = client.get("/cache/stats").json()
    assert stats["backend"]["hits"] == 0
    assert "s" not in stats["cacheable_sources"]


def test_reregister_clears_cache(client):
    from flexviz.server import register_source

    register_source("s", pl.DataFrame({"x": [1, 2], "y": [1, 2]}), cache=True)
    client.post("/update", json={"spec": _SPEC, "event": _INIT})
    assert cache_mod.get_cache().stats()["entries"] >= 1
    register_source("s", pl.DataFrame({"x": [3, 4], "y": [3, 4]}), cache=True)
    assert cache_mod.get_cache().stats()["entries"] == 0


def test_reregister_same_builder_invalidates_nothing_and_warns(client, tmp_path):
    """Re-registering the exact same builder object warns and clears nothing,
    so the source keeps serving its old, cached data. Raw data or a new
    builder still replaces it and clears both caches."""
    import warnings

    from flexviz.server import register_source

    path = tmp_path / "s.parquet"
    pl.DataFrame({"x": list(range(40)), "y": [i % 3 for i in range(40)]}).write_parquet(
        path
    )
    builder = LFQueryBuilder(pl.scan_parquet(path))
    register_source("s", builder, cache=True)
    r1 = client.post("/update", json={"spec": _SPEC, "event": _INIT}).json()
    entries = cache_mod.get_cache().stats()["entries"]
    assert entries >= 1

    pl.DataFrame(
        {
            "x": list(range(0, 400, 10)),
            "y": [i % 3 for i in range(40)],
            "z": list(range(40)),
        }
    ).write_parquet(path)

    with pytest.warns(UserWarning, match="nothing was invalidated"):
        register_source("s", builder, cache=True)
    assert cache_mod.get_cache().stats()["entries"] == entries
    r2 = client.post("/update", json={"spec": _SPEC, "event": _INIT}).json()
    assert r2 == r1

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        register_source("s", pl.scan_parquet(path), cache=True)
    assert cache_mod.get_cache().stats()["entries"] == 0
    r3 = client.post("/update", json={"spec": _SPEC, "event": _INIT}).json()
    assert r3 != r1


def test_reregister_clears_cube_cache(client):
    """Source re-registration invalidates BOTH caches (delta + cube blobs)."""
    from flexviz.server import register_source

    register_source("s", pl.DataFrame({"x": [1, 2], "y": [1, 2]}), cache=True)
    cache_mod.get_cube_cache().set("some-cube-id", b"\x00" * 16)
    assert cache_mod.get_cube_cache().stats()["entries"] == 1
    register_source("s", pl.DataFrame({"x": [3, 4], "y": [3, 4]}), cache=True)
    assert cache_mod.get_cube_cache().stats()["entries"] == 0


def test_register_new_source_preserves_other_cache(client):
    """Registering an unrelated *new* source must not evict an existing
    source's cached entries (only re-registration of an existing name clears)."""
    from flexviz.server import register_source

    register_source("s", pl.DataFrame({"x": [1, 2], "y": [1, 2]}), cache=True)
    client.post("/update", json={"spec": _SPEC, "event": _INIT})
    entries = cache_mod.get_cache().stats()["entries"]
    assert entries >= 1

    register_source(
        "_unrelated_new_src", pl.DataFrame({"x": [9], "y": [9]}), cache=True
    )
    assert cache_mod.get_cache().stats()["entries"] == entries


def test_cache_stats_endpoint_shape(client):
    body = client.get("/cache/stats").json()
    assert set(body) == {"backend", "cacheable_sources"}
    assert set(body["backend"]) == {"entries", "hits", "misses", "max_entries"}


def test_idempotent_register_can_disable_cache():
    from flexviz.cache import is_source_cacheable
    from flexviz.figure import _register_source_if_needed

    source_name = "_cache_toggle_same_object"
    lf = LFQueryBuilder(pl.DataFrame({"x": [1, 2], "y": [1, 2]}))

    _register_source_if_needed(source_name, lf, cache=True)
    assert is_source_cacheable(source_name)

    _register_source_if_needed(source_name, lf, cache=False)
    assert not is_source_cacheable(source_name)


# ---------------------------------------------------------------------------
# live_brush ↔ cache coupling (Fix A): cubes are only built for cache=True
# sources, so live-brush must not engage (and fire a wasted cube_request)
# without it.
# ---------------------------------------------------------------------------


def test_live_brush_passes_through_when_cacheable():
    from flexviz.figure import _effective_live_brush

    assert _effective_live_brush("auto", effective_cache=True) == "auto"
    assert _effective_live_brush("off", effective_cache=True) == "off"


def test_live_brush_forced_off_without_cache():
    from flexviz.figure import _effective_live_brush

    assert _effective_live_brush("off", effective_cache=False) == "off"


def test_live_brush_forced_off_warns_on_explicit_request():
    from flexviz.figure import _effective_live_brush

    with pytest.warns(UserWarning, match="needs cache=True"):
        result = _effective_live_brush("auto", effective_cache=False)
    assert result == "off"

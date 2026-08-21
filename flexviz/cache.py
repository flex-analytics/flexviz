"""Content-addressed memoization cache for flexviz.

This module is the **only** mutable server-side state besides the data-source
registry, and it is deliberately *not* session state: every entry is keyed
purely on request content (source + trace identity), is never authoritative
(a miss recomputes a byte-identical result), and may be dropped at any time.
It is therefore invariant to *who* or *how many* clients are connected, which
keeps the server horizontally scalable and consistent with the stateless
invariant documented in ``Architecture.md``.

Phase 1 scope (see issue #26): only the *unfiltered, no-viewport* computation
(``init`` / ``reset`` / ``deselect``) is cached.  The ``cache`` flag on a
source asserts the data is static for the process lifetime, so there is no
invalidation yet (data-driven invalidation is issue #27; a shared
Redis/disk backend is issue #28).
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from typing import Any, Dict, Protocol, Set, runtime_checkable

# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class CacheBackend(Protocol):
    """Minimal content-addressed store.

    Implementations **must be thread-safe**: ``engine.process`` runs inside a
    threadpool (``run_in_threadpool``), so ``get`` / ``set`` are called
    concurrently.  Stored values are small JSON-safe delta payloads.
    """

    def get(self, key: str) -> Any | None: ...

    def set(self, key: str, value: Any) -> None: ...

    def clear(self) -> None: ...

    def stats(self) -> Dict[str, int]: ...


class InMemoryLRUCache:
    """Thread-safe, bounded, in-process LRU cache.

    Entry count (not bytes) is bounded by ``max_entries``; values are small
    (downsampled / binned delta payloads), so a generous count is cheap.  All
    access is guarded by a lock because a naive ``OrderedDict.move_to_end`` is
    not safe under the request threadpool.
    """

    def __init__(self, max_entries: int = 1000) -> None:
        self._max_entries = max_entries
        self._data: "OrderedDict[str, Any]" = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self._hits += 1
                return self._data[key]
            self._misses += 1
            return None

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self._max_entries:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._data),
                "hits": self._hits,
                "misses": self._misses,
                "max_entries": self._max_entries,
            }


class InMemoryByteLRUCache:
    """Thread-safe, **byte-bounded**, in-process LRU cache for encoded blobs.

    Unlike ``InMemoryLRUCache`` (entry-count bound, small delta payloads),
    values here are FVCube blobs (``bytes``) whose sizes span orders of
    magnitude, so the budget counts ``len(value)``.  An entry larger than the
    whole budget is silently not cached — caching it would evict everything
    else for a value that can never be retained.
    """

    def __init__(self, max_bytes: int = 512 * 2**20) -> None:
        self._max_bytes = max_bytes
        self._data: "OrderedDict[str, bytes]" = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> bytes | None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self._hits += 1
                return self._data[key]
            self._misses += 1
            return None

    def set(self, key: str, value: bytes) -> None:
        size = len(value)
        with self._lock:
            old = self._data.pop(key, None)
            if old is not None:
                self._bytes -= len(old)
            if size > self._max_bytes:
                return  # oversized: never cacheable, do not evict for it
            self._data[key] = value
            self._bytes += size
            while self._bytes > self._max_bytes:
                _, evicted = self._data.popitem(last=False)
                self._bytes -= len(evicted)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._bytes = 0
            self._hits = 0
            self._misses = 0

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._data),
                "bytes": self._bytes,
                "hits": self._hits,
                "misses": self._misses,
                "max_bytes": self._max_bytes,
            }


# ---------------------------------------------------------------------------
# Module-level singletons — the server-side caches and the per-source flag set.
# Populated before / during request handling; behave as process-global memo.
# ---------------------------------------------------------------------------

_cache: CacheBackend = InMemoryLRUCache()
_cube_cache: CacheBackend = InMemoryByteLRUCache()
_cacheable_sources: Set[str] = set()


def get_cache() -> CacheBackend:
    """Return the process-global cache backend."""
    return _cache


def set_cache_backend(backend: CacheBackend) -> None:
    """Swap the global cache backend (e.g. for a Redis/disk backend, #28)."""
    global _cache
    _cache = backend


def get_cube_cache() -> CacheBackend:
    """Return the process-global cube-blob cache (byte-bounded LRU).

    Keyed by ``cube_content_key`` and storing *encoded* FVCube blobs
    (``bytes``) — slicing is client-side and re-encoding is the expensive
    part. Shares the delta cache's invalidation rule: ``register_source``
    clears both caches on re-registration of an existing name.
    """
    return _cube_cache


def set_cube_cache_backend(backend: CacheBackend) -> None:
    """Swap the global cube cache backend (e.g. for a Redis/disk backend, #28)."""
    global _cube_cache
    _cube_cache = backend


def set_source_cacheable(name: str, flag: bool) -> None:
    """Record whether a named source opts into caching.

    This only updates the opt-in set; it deliberately does **not** evict cached
    entries. Invalidation is the registrar's concern: ``server.register_source``
    clears the cache when (and only when) a source is *re-registered* under an
    existing name, since under the Phase-1 "assume static" contract that is the
    only event that can change a source's data.
    """
    if flag:
        _cacheable_sources.add(name)
    else:
        _cacheable_sources.discard(name)


def is_source_cacheable(name: str | None) -> bool:
    return name is not None and name in _cacheable_sources


def cacheable_sources() -> Set[str]:
    """Return the set of source names that opted into caching."""
    return set(_cacheable_sources)


# ---------------------------------------------------------------------------
# Content key
# ---------------------------------------------------------------------------


def _canonical_json(payload: Any) -> str:
    """Stable JSON encoding: sorted keys, compact separators.

    ``params``/``backend_data`` originate from a JSON-serializable spec, so the
    inputs are already JSON types; ``default=str`` is a defensive fallback.
    """
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=False
    )


def content_key(
    source_name: str,
    trace_type: str,
    axes: Any,
    backend_data: Dict[str, Any],
    params: Dict[str, Any],
    domain_cols: tuple[str, ...] | None = None,
) -> str:
    """Compute the Phase-1 per-trace content key.

    The key captures *everything that determines the unfiltered output* and
    nothing cosmetic: source, trace type, axes, column mapping, backend params,
    and ``domain_cols`` — the sibling columns a shared bin domain unions over
    (histograms on one figure).  That last one is not a property of the trace
    alone: the *same* histogram bins differently depending on which siblings
    share its axis, so omitting it lets a solo-figure entry alias a
    sibling-figure one and re-introduces the very bin misalignment the shared
    domain exists to prevent.  The key deliberately excludes ``uid``,
    ``display``, ``hover``, and — because Phase 1 only caches the
    unfiltered/no-viewport case — any filter, viewport, or overlay layer
    (``base == bg`` for unfiltered data).  Two different figures with an
    identical trace therefore share one entry.
    """
    canonical = _canonical_json(
        {
            "s": source_name,
            "t": trace_type,
            "a": list(axes) if axes else None,
            "b": backend_data,
            "p": params,
            "d": sorted(domain_cols) if domain_cols else None,
        }
    )
    return hashlib.blake2b(canonical.encode("utf-8"), digest_size=16).hexdigest()

"""Subprocess worker for the out-of-core memory matrix (see test_ooc.py).

Not a test module: pytest never collects this file directly. It runs as
``python tests/ooc_child.py <parquet> <trace>`` in a fresh process, builds
one trace through the product path (LFQueryBuilder + FlexEngine, the same
objects the server uses), and prints one JSON line with the peak anonymous
memory used by ``engine.process`` and how long that took.

A separate process per (trace, size) pair keeps each measurement free of
whatever the previous trace left allocated.
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
import threading
import time
from typing import Callable

import polars as pl

from flexviz.engine import FlexEngine, TraceInfo
from flexviz.events import InteractionEvent
from flexviz.LF import LFQueryBuilder
from flexviz.trace.bar import BarPlot
from flexviz.trace.box import BoxPlot
from flexviz.trace.corr_heatmap import CorrHeatmap
from flexviz.trace.geo_hist2d import GeoHistogram2D
from flexviz.trace.geo_line import GeoLine
from flexviz.trace.hist import Histogram
from flexviz.trace.hist2d import Histogram2D
from flexviz.trace.line import LinePlot
from flexviz.trace.pie import PiePlot
from flexviz.trace.treemap import TreeMap

_BINS_1D = 200
_BINS_2D = 200

# name -> zero-arg constructor. Kept as callables (not instances) so building
# a trace never runs before the child process is the one measuring memory.
TRACES: dict[str, Callable] = {
    "line-minmax": lambda: LinePlot(x="x", y="y", downsample="minmax"),
    "line-lttb": lambda: LinePlot(x="x", y="y", downsample="lttb"),
    "line-fpcs": lambda: LinePlot(x="x", y="y", downsample="fpcs"),
    "line-nth": lambda: LinePlot(x="x", y="y", downsample="nth"),
    "line-grouped": lambda: LinePlot(x="x", y="y", downsample="minmax", group_by="g"),
    "line-grouped-nth": lambda: LinePlot(x="x", y="y", downsample="nth", group_by="g"),
    "hist": lambda: Histogram(x="y", bins=_BINS_1D),
    "hist-grouped": lambda: Histogram(x="y", bins=_BINS_1D, group_by="g"),
    "hist2d": lambda: Histogram2D(x="x", y="y", x_bins=_BINS_2D, y_bins=_BINS_2D),
    "hist2d-reduce": lambda: Histogram2D(
        x="x", y="y", x_bins=_BINS_2D, y_bins=_BINS_2D, z="lat", histfunc="mean"
    ),
    "geo-hist2d": lambda: GeoHistogram2D(
        lat="lat", lon="lon", lat_bins=_BINS_2D, lon_bins=_BINS_2D
    ),
    "geo-line": lambda: GeoLine(lat="lat", lon="lon"),
    "bar": lambda: BarPlot(labels="g", values="y", agg="mean"),
    "pie": lambda: PiePlot(labels="g"),
    "treemap": lambda: TreeMap(path=["g"]),
    "corr": lambda: CorrHeatmap(columns=["y", "z", "lat", "lon"]),
    "box": lambda: BoxPlot(y="y"),
    "box-grouped": lambda: BoxPlot(y="y", group_by="g"),
}


# ---- anonymous-memory sampler ------------------------------------------
#
# A Parquet scan mmaps the file, so RSS grows with the file size even when
# the algorithm itself streams. Anonymous memory (heap/stack allocations,
# not file-backed mappings) is the O(1)-in-rows signal an out-of-core path
# actually promises.


class _RUsageV4(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
        (n, ctypes.c_uint64)
        for n in (
            "ri_user_time",
            "ri_system_time",
            "ri_pkg_idle_wkups",
            "ri_interrupt_wkups",
            "ri_pageins",
            "ri_wired_size",
            "ri_resident_size",
            "ri_phys_footprint",
            "ri_proc_start_abstime",
            "ri_proc_exit_abstime",
            "ri_child_user_time",
            "ri_child_system_time",
            "ri_child_pkg_idle_wkups",
            "ri_child_interrupt_wkups",
            "ri_child_pageins",
            "ri_child_elapsed_abstime",
            "ri_diskio_bytesread",
            "ri_diskio_byteswritten",
            "ri_cpu_time_qos_default",
            "ri_cpu_time_qos_maintenance",
            "ri_cpu_time_qos_background",
            "ri_cpu_time_qos_utility",
            "ri_cpu_time_qos_legacy",
            "ri_cpu_time_qos_user_initiated",
            "ri_cpu_time_qos_user_interactive",
            "ri_billed_system_time",
            "ri_serviced_system_time",
            "ri_logical_writes",
            "ri_lifetime_max_phys_footprint",
            "ri_instructions",
            "ri_cycles",
            "ri_billed_energy",
            "ri_serviced_energy",
            "ri_interval_max_phys_footprint",
            "ri_runnable_time",
            "ri_flags",
        )
    ]


def _footprint_macos() -> int:
    lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    buf = _RUsageV4()
    # RUSAGE_INFO_V4 = 4.
    rc = lib.proc_pid_rusage(
        ctypes.c_int(os.getpid()), ctypes.c_int(4), ctypes.byref(buf)
    )
    if rc != 0:
        raise OSError("proc_pid_rusage failed")
    return buf.ri_phys_footprint


def _footprint_linux() -> int:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("RssAnon:"):
                return int(line.split()[1]) * 1024  # kB -> bytes
    raise RuntimeError("RssAnon not found in /proc/self/status")


def anonymous_memory_bytes() -> int:
    if sys.platform == "darwin":
        return _footprint_macos()
    if sys.platform.startswith("linux"):
        return _footprint_linux()
    raise RuntimeError(f"no anonymous-memory counter for platform {sys.platform!r}")


class PeakSampler:
    """Samples anonymous memory every ``interval`` seconds on a daemon thread."""

    def __init__(self, interval: float = 0.01) -> None:
        self.interval = interval
        self._stop = False
        self.base = 0
        self.peak = 0

    def __enter__(self) -> "PeakSampler":
        self.base = anonymous_memory_bytes()
        self.peak = self.base
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop:
            self.peak = max(self.peak, anonymous_memory_bytes())
            time.sleep(self.interval)

    def __exit__(self, *exc: object) -> None:
        self._stop = True
        self._thread.join()
        self.peak = max(self.peak, anonymous_memory_bytes())

    @property
    def peak_mb(self) -> float:
        return (self.peak - self.base) / 1e6


def main() -> None:
    path, name = sys.argv[1], sys.argv[2]

    lf = LFQueryBuilder(pl.scan_parquet(path))
    trace = TRACES[name]()
    engine = FlexEngine(backend_lf=lf, scalable_traces={trace.uid: trace})
    info = TraceInfo(uid=trace.uid, axes=trace._axes, trace_type=trace.trace_type)
    event = InteractionEvent(type="init", force_update=True)

    # Base is taken after the engine is built; only `process` itself is measured.
    with PeakSampler() as sampler:
        t0 = time.perf_counter()
        engine.process(event, [info])
        seconds = time.perf_counter() - t0

    print(json.dumps({"peak_mb": sampler.peak_mb, "seconds": seconds}))


if __name__ == "__main__":
    main()

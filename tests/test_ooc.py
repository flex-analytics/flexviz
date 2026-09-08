"""Out-of-core memory matrix: peak anonymous memory must stay flat as rows grow.

One subprocess per (trace, size) via ``ooc_child.py`` isolates each
measurement and lets each run start from a clean heap. Run with
``make test-ooc``; excluded from ``make test`` because it takes 1-2 minutes,
and needs its own CI job (see ``.github/workflows/ci.yml``).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from ooc_child import TRACES

pytestmark = [
    pytest.mark.ooc,
    pytest.mark.skipif(sys.platform == "win32", reason="no anonymous-memory counter"),
]

# 4x row-count growth is the out-of-core evidence: flat memory across it means
# the algorithm is O(1) in rows, not O(n).
SMALL = 4_000_000
LARGE = 16_000_000

# Peak(LARGE) must stay within this multiple of peak(SMALL) plus SLACK_MB.
RATIO = 1.5
# Sampler jitter (10 ms polling) is proportionally largest at the small size,
# where the true peak can be just a few MB; SLACK_MB absorbs that noise.
SLACK_MB = 32
# A flat-but-huge baseline (e.g. a full eager collect at both sizes) would
# still pass the ratio check, so cap the large-size peak outright.
CAP_MB = 512

_CHILD = Path(__file__).parent / "ooc_child.py"

_XFAIL_QUANTILE = pytest.mark.xfail(
    strict=True, reason="Polars quantile runs in memory"
)

_XFAIL_BY_NAME = {
    "box": _XFAIL_QUANTILE,
    "box-grouped": _XFAIL_QUANTILE,
}

_PARAMS = [
    pytest.param(name, marks=_XFAIL_BY_NAME[name]) if name in _XFAIL_BY_NAME else name
    for name in TRACES
]


@pytest.fixture(scope="session")
def ooc_fixtures(tmp_path_factory: pytest.TempPathFactory) -> tuple[str, str]:
    """Write a LARGE Parquet file, then a SMALL prefix sliced from it."""
    rng = np.random.default_rng(0)
    arange = np.arange(LARGE, dtype=np.int64)
    df = pl.DataFrame(
        {
            "x": arange,
            "y": rng.standard_normal(LARGE),
            "z": rng.standard_normal(LARGE),
            "lat": rng.uniform(-60, 60, LARGE),
            "lon": rng.uniform(-160, 160, LARGE),
            "g": (arange % 10).astype(str),
        }
    )
    tmp_dir = tmp_path_factory.mktemp("ooc")
    large_path = tmp_dir / "large.parquet"
    small_path = tmp_dir / "small.parquet"
    df.write_parquet(large_path, compression="zstd")
    pl.scan_parquet(large_path).head(SMALL).sink_parquet(small_path)
    return str(small_path), str(large_path)


def _run(path: str, name: str) -> float:
    env = dict(os.environ)
    # The reader's row-group prefetch buffers ahead of the algorithm, which
    # would mask an out-of-core violation behind extra buffered memory.
    env["POLARS_ROW_GROUP_PREFETCH_SIZE"] = "1"
    result = subprocess.run(
        [sys.executable, str(_CHILD), path, name],
        env=env,
        capture_output=True,
        text=True,
    )
    assert (
        result.returncode == 0
    ), f"{name} child failed (exit {result.returncode}):\n{result.stderr}"
    last_line = result.stdout.strip().splitlines()[-1]
    return json.loads(last_line)["peak_mb"]


@pytest.mark.parametrize("name", _PARAMS)
def test_peak_memory_is_flat(name: str, ooc_fixtures: tuple[str, str]) -> None:
    small_path, large_path = ooc_fixtures
    small = _run(small_path, name)
    large = _run(large_path, name)
    assert large <= RATIO * small + SLACK_MB and large <= CAP_MB, (
        f"{name}: peak grew from {small:.1f} MB (SMALL) to {large:.1f} MB (LARGE), "
        f"exceeding {RATIO} x + {SLACK_MB} MB slack or the {CAP_MB} MB cap"
    )

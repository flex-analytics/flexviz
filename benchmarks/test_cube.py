"""Cross-filter cube benchmarks.

The cube pre-aggregates a target trace over a binned free axis so the browser
can answer a brush without a round trip. ``build_cube`` is the Rust-accelerated
hot path; ``encode_fvcube`` is the wire encoder that follows it on every
``cube_request``.
"""

from __future__ import annotations

import polars as pl
import pytest

import flexviz_polars  # noqa: F401 — registers the pl.Expr.flexviz namespace
from flexviz.cube import (
    CubeSpec,
    FreeAxisSpec,
    MeasureSpec,
    TargetDimSpec,
    build_cube,
    cube_content_key,
    decode_cube_bundle,
    encode_cube_bundle,
    encode_fvcube,
)

N_BINS = 256


def _categorical_spec(p: int = N_BINS) -> CubeSpec:
    return CubeSpec(
        source_name="bench",
        free=FreeAxisSpec(column="ts", p=p, domain=(0.0, 200_000.0)),
        target_dims=(TargetDimSpec(column="cat", kind="categorical"),),
        measure=MeasureSpec(agg="count"),
    )


def _binned_spec(agg: str = "count") -> CubeSpec:
    return CubeSpec(
        source_name="bench",
        free=FreeAxisSpec(column="ts", p=N_BINS, domain=(0.0, 200_000.0)),
        target_dims=(
            TargetDimSpec(column="val", kind="continuous", bins=64, domain=(-2.0, 2.0)),
        ),
        measure=MeasureSpec(agg=agg, value_col=None if agg == "count" else "val2"),
    )


@pytest.mark.parametrize("p", [64, 512])
def test_build_cube_categorical(benchmark, numeric_lf: pl.LazyFrame, p: int) -> None:
    """Counts per category per free-axis bin — the bar/pie cross-filter cube."""
    result = benchmark(build_cube, numeric_lf, _categorical_spec(p))
    assert result.frame.height > 0


@pytest.mark.parametrize("agg", ["count", "mean"])
def test_build_cube_binned(benchmark, numeric_lf: pl.LazyFrame, agg: str) -> None:
    """A binned continuous target: the histogram cross-filter cube."""
    result = benchmark(build_cube, numeric_lf, _binned_spec(agg))
    assert result.frame.height > 0


def test_encode_fvcube(benchmark, numeric_lf: pl.LazyFrame) -> None:
    """Binary encoding of a built cube, as served on the wire."""
    result = build_cube(numeric_lf, _categorical_spec())
    payload = benchmark(encode_fvcube, result, "cube-0")
    assert len(payload) > 0


def test_cube_bundle_roundtrip(benchmark, numeric_lf: pl.LazyFrame) -> None:
    """Encode then decode a one-cube bundle, the full client-facing path."""
    blob = encode_fvcube(build_cube(numeric_lf, _categorical_spec()), "cube-0")

    def roundtrip():
        return decode_cube_bundle(encode_cube_bundle([blob], {"trace-0": 0}))

    blobs, trace_cubes = benchmark(roundtrip)
    assert len(blobs) == 1 and trace_cubes == {"trace-0": 0}


def test_cube_content_key(benchmark, numeric_lf: pl.LazyFrame) -> None:
    """Cache-key derivation runs on every cube request, hit or miss."""
    spec = _categorical_spec()
    key = benchmark(cube_content_key, spec)
    assert key

"""Unit tests for the cross-filter cube core (flexviz/cube.py)."""

from __future__ import annotations

import math
import struct
from datetime import date, datetime, timedelta

import numpy as np
import polars as pl
import pytest

import flexviz_polars  # noqa: F401 — registers pl.Expr.flexviz namespace

from flexviz.cube import (
    CubeResult,
    CubeSpec,
    FreeAxisSpec,
    MeasureSpec,
    TargetDimSpec,
    build_cube,
    cube_content_key,
    decode_cube_bundle,
    decode_fvcube_header,
    encode_cube_bundle,
    encode_fvcube,
)


@pytest.fixture()
def df() -> pl.DataFrame:
    # active ∈ [0,100); a categorical target `cat`; a numeric target `val`.
    n = 10_000
    return pl.DataFrame(
        {
            "active": [(i * 37) % 100 for i in range(n)],
            "cat": [str(i % 5) for i in range(n)],
            "val": [float(i % 50) for i in range(n)],
        }
    )


def _cat_spec(p: int = 64) -> CubeSpec:
    return CubeSpec(
        source_name="s",
        free=FreeAxisSpec(column="active", p=p, domain=(0.0, 100.0)),
        target_dims=(TargetDimSpec(column="cat", kind="categorical"),),
        measure=MeasureSpec(agg="count"),
    )


def _cat_spec_domain(lo: float, hi: float, p: int = 64) -> CubeSpec:
    return CubeSpec(
        source_name="s",
        free=FreeAxisSpec(column="active", p=p, domain=(lo, hi)),
        target_dims=(TargetDimSpec(column="cat", kind="categorical"),),
        measure=MeasureSpec(agg="count"),
    )


def _measure_spec(agg: str, value_col: str | None = "val", p: int = 64) -> CubeSpec:
    return CubeSpec(
        source_name="s",
        free=FreeAxisSpec(column="active", p=p, domain=(0.0, 100.0)),
        target_dims=(TargetDimSpec(column="cat", kind="categorical"),),
        measure=MeasureSpec(
            agg=agg, value_col=None if agg == "count" else value_col  # type: ignore[arg-type]
        ),
    )


# Legacy direct-aggregation expressions (mirrors trace/_hist_helpers._AGG_FUNCTIONS
# plus implicit count) — the parity reference for slice_agg.
_LEGACY_AGG = {
    "count": lambda col: pl.len(),
    "sum": lambda col: pl.col(col).sum(),
    "mean": lambda col: pl.col(col).mean(),
    "min": lambda col: pl.col(col).min(),
    "max": lambda col: pl.col(col).max(),
}

_ALL_AGGS = ["count", "sum", "mean", "min", "max"]
_EXACT_AGGS = ("count", "min", "max")  # sums/means compared with 1e-9 tolerance


def _assert_slice_matches(sliced: pl.DataFrame, direct: pl.DataFrame, agg: str) -> None:
    """Counts/min/max bit-exact; sum/mean within 1e-9 (combine-order float caveat)."""
    assert sliced["cat"].to_list() == direct["cat"].to_list()
    got, want = sliced["value"].to_list(), direct["value"].to_list()
    if agg in _EXACT_AGGS:
        assert got == want
    else:
        assert got == pytest.approx(want, rel=1e-9, abs=1e-9)


class TestMeasures:
    @pytest.mark.parametrize("agg", _ALL_AGGS)
    def test_full_range_slice_agg_equals_direct(self, df, agg):
        cube = build_cube(df.lazy(), _measure_spec(agg))
        sliced = cube.slice_agg(0.0, 100.0).sort("cat")
        direct = (
            df.group_by("cat").agg(_LEGACY_AGG[agg]("val").alias("value")).sort("cat")
        )
        _assert_slice_matches(sliced, direct, agg)

    @pytest.mark.parametrize("agg", _ALL_AGGS)
    def test_subrange_slice_agg_equals_filtered_direct(self, df, agg):
        cube = build_cube(df.lazy(), _measure_spec(agg))
        lo_bin, hi_bin = cube._snap(25.0, 75.0)
        lo_v, hi_v = lo_bin / 64 * 100.0, (hi_bin + 1) / 64 * 100.0
        sliced = cube.slice_agg(25.0, 75.0).sort("cat")
        direct = (
            df.filter((pl.col("active") >= lo_v) & (pl.col("active") < hi_v))
            .group_by("cat")
            .agg(_LEGACY_AGG[agg]("val").alias("value"))
            .sort("cat")
        )
        _assert_slice_matches(sliced, direct, agg)

    def test_slice_count_still_works_on_count_cube(self, df):
        # slice_count remains the count-only alias alongside slice_agg.
        cube = build_cube(df.lazy(), _measure_spec("count"))
        counts = cube.slice_count(25.0, 75.0).sort("cat")
        values = cube.slice_agg(25.0, 75.0).sort("cat")
        assert counts["count"].to_list() == values["value"].to_list()

    def test_mean_ignores_nulls_and_omits_zero_count_cells(self):
        df = pl.DataFrame(
            {
                "active": [10.0, 10.0, 10.0, 20.0, 20.0],
                "cat": ["a", "a", "a", "b", "b"],
                "val": [1.0, None, 3.0, None, None],
            }
        )
        cube = build_cube(df.lazy(), _measure_spec("mean"))
        out = cube.slice_agg(0.0, 100.0).sort("cat")
        # cat a: nulls ignored → mean(1, 3); cat b: all-null → count 0 → absent.
        assert out["cat"].to_list() == ["a"]
        assert out["value"].to_list() == [2.0]

    @pytest.mark.parametrize("agg", ["min", "max"])
    def test_min_max_skip_all_null_partial_cells(self, agg):
        # cat a has an all-null cell at active=10 and a real value at active=90:
        # the null partial must be skipped in the combine, not poison it.
        df = pl.DataFrame(
            {
                "active": [10.0, 10.0, 90.0],
                "cat": ["a", "a", "a"],
                "val": [None, None, 5.0],
            }
        )
        cube = build_cube(df.lazy(), _measure_spec(agg))
        assert cube.slice_agg(0.0, 100.0)["value"].to_list() == [5.0]
        # A slice covering only the all-null cell finalizes to null → row absent.
        assert cube.slice_agg(0.0, 50.0).height == 0

    def test_sum_of_all_null_group_is_zero(self):
        # Polars sum over an all-null group is 0 — legacy parity, no special case.
        df = pl.DataFrame(
            {"active": [10.0, 10.0], "cat": ["a", "a"], "val": [None, None]}
        )
        cube = build_cube(df.lazy(), _measure_spec("sum"))
        out = cube.slice_agg(0.0, 100.0)
        assert out["cat"].to_list() == ["a"]
        assert out["value"].to_list() == [0.0]

    @pytest.mark.parametrize("agg", ["sum", "mean", "min", "max"])
    def test_value_col_required_for_non_count(self, agg):
        with pytest.raises(ValueError):
            MeasureSpec(agg=agg)

    def test_value_col_forbidden_for_count(self):
        with pytest.raises(ValueError):
            MeasureSpec(agg="count", value_col="val")

    def test_content_key_differs_by_agg(self):
        keys = {cube_content_key(_measure_spec(agg)) for agg in _ALL_AGGS}
        assert len(keys) == len(_ALL_AGGS)

    def test_content_key_differs_by_value_col(self):
        a = cube_content_key(_measure_spec("sum", value_col="val"))
        b = cube_content_key(_measure_spec("sum", value_col="active"))
        assert a != b


class TestBuildAndSlice:
    def test_full_slice_equals_direct_count(self, df):
        cube = build_cube(df.lazy(), _cat_spec())
        sliced = cube.slice_count(0.0, 100.0).sort("cat")
        direct = (
            df.lazy().group_by("cat").agg(pl.len().alias("count")).sort("cat").collect()
        )
        assert sliced.equals(direct)

    def test_subrange_slice_equals_filtered_count(self, df):
        cube = build_cube(df.lazy(), _cat_spec())
        # snap [25, 75) to the same P-grid the cube uses, then compare to a direct
        # filtered count over the snapped bin edges (exact, not approximate).
        lo_bin, hi_bin = cube._snap(25.0, 75.0)
        edge = lambda b: 0.0 + (b) / 64 * 100.0  # noqa: E731
        lo_v, hi_v = edge(lo_bin), edge(hi_bin + 1)
        sliced = cube.slice_count(25.0, 75.0).sort("cat")
        direct = (
            df.lazy()
            .filter((pl.col("active") >= lo_v) & (pl.col("active") < hi_v))
            .group_by("cat")
            .agg(pl.len().alias("count"))
            .sort("cat")
            .collect()
        )
        assert sliced.equals(direct)

    def test_total_count_conserved(self, df):
        cube = build_cube(df.lazy(), _cat_spec())
        assert cube.slice_count(0.0, 100.0)["count"].sum() == df.height

    def test_binned_target_dim(self, df):
        spec = CubeSpec(
            source_name="s",
            free=FreeAxisSpec(column="active", p=64, domain=(0.0, 100.0)),
            target_dims=(
                TargetDimSpec(column="val", kind="binned", bins=10, domain=(0.0, 50.0)),
            ),
        )
        cube = build_cube(df.lazy(), spec)
        # 10 target bins; total count conserved over the full free range.
        full = cube.slice_count(0.0, 100.0)
        assert full.height <= 10
        assert full["count"].sum() == df.height

    def test_cube_is_sparse_no_empty_cells(self, df):
        cube = build_cube(df.lazy(), _cat_spec())
        # group_by only emits populated (cat, free_bin) groups.
        assert (cube.frame["count"] > 0).all()


class TestFilterDontClip:
    def test_zoomed_domain_drops_out_of_domain_rows(self, df):
        # Build with a domain narrower than the data: rows outside [25, 75]
        # must be absent from the cube entirely, not clipped into edge bins.
        cube = build_cube(df.lazy(), _cat_spec_domain(25.0, 75.0))
        in_domain = df.filter(pl.col("active").is_between(25, 75)).height
        assert cube.frame["count"].sum() == in_domain

    def test_zoomed_domain_edge_bins_not_inflated(self, df):
        cube = build_cube(df.lazy(), _cat_spec_domain(25.0, 75.0))
        p, lo, hi = 64, 25.0, 75.0
        width = (hi - lo) / p
        # Bin 0 covers [25, 25 + width): only in-domain rows, no clipped mass.
        bin0 = cube.frame.filter(pl.col("free_bin") == 0)["count"].sum()
        direct0 = df.filter(
            (pl.col("active") >= lo) & (pl.col("active") < lo + width)
        ).height
        assert bin0 == direct0
        # Bin P-1 covers [75 - width, 75): integer data has no values there, and
        # rows above 75 must not be clipped into it.
        top = cube.frame.filter(pl.col("free_bin") == p - 1)["count"].sum()
        direct_top = df.filter(
            (pl.col("active") >= hi - width) & (pl.col("active") < hi)
        ).height
        assert top == direct_top

    def test_domain_max_lands_in_degenerate_top_bin(self):
        df = pl.DataFrame({"active": [0.0, 50.0, 100.0], "cat": ["a", "a", "a"]})
        cube = build_cube(df.lazy(), _cat_spec())
        assert cube.frame.filter(pl.col("free_bin") == 64)["count"].sum() == 1
        # A full-range slice includes the degenerate bin.
        assert cube.slice_count(0.0, 100.0)["count"].sum() == 3

    def test_snap_clamps_to_degenerate_top_bin(self, df):
        cube = build_cube(df.lazy(), _cat_spec())
        assert cube._snap(0.0, 100.0) == (0, 64)
        assert cube._snap(100.0, 100.0) == (64, 64)
        assert cube._snap(-10.0, 200.0) == (0, 64)
        # Reversed endpoints swap, as before.
        assert cube._snap(100.0, 0.0) == (0, 64)


class TestFixedHistParity:
    def test_binned_target_matches_fixed_hist_kernel(self):
        # Same setup hist.py produces: hi = axis_hi + 1e-10 epsilon, integer
        # values on every visual bin boundary, plus values exactly at every
        # internal bin edge of the (lo, hi, n) grid.
        lo, hi, n = 0.0, 100.0 + 1e-10, 10
        step = (hi - lo) / n
        values = [float(v) for v in range(101)]
        values += [lo + k * step for k in range(1, n)]
        m = len(values)
        free = [i / (m - 1) for i in range(m)]
        df = pl.DataFrame({"t": values, "free": free})

        spec = CubeSpec(
            source_name="s",
            free=FreeAxisSpec(column="free", p=16, domain=(0.0, 1.0)),
            target_dims=(
                TargetDimSpec(column="t", kind="binned", bins=n, domain=(lo, hi)),
            ),
        )
        cube = build_cube(df.lazy(), spec)
        sliced = cube.slice_count(0.0, 1.0)
        by_bin = dict(zip(sliced["__bin__t"].to_list(), sliced["count"].to_list()))
        cube_counts = [by_bin.get(b, 0) for b in range(n)]

        kernel = pl.select(
            pl.lit(df["t"]).flexviz.fixed_hist(pl.lit(lo), pl.lit(hi), n_bins=n)
        ).to_series()
        kernel_counts = kernel.struct.field("count").to_list()
        assert cube_counts == kernel_counts

    def test_binned_target_drops_out_of_domain_rows(self):
        df = pl.DataFrame(
            {
                "t": [-5.0, 0.0, 25.0, 49.0, 75.0, 200.0],
                "free": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            }
        )
        spec = CubeSpec(
            source_name="s",
            free=FreeAxisSpec(column="free", p=8, domain=(0.0, 1.0)),
            target_dims=(
                TargetDimSpec(column="t", kind="binned", bins=5, domain=(0.0, 50.0)),
            ),
        )
        cube = build_cube(df.lazy(), spec)
        # -5.0 and 200.0 and 75.0 are outside [0, 50] → dropped, not clipped.
        assert cube.frame["count"].sum() == 3


class TestTemporalFreeAxis:
    @pytest.fixture()
    def dt_df(self) -> pl.DataFrame:
        base = datetime(2024, 1, 1)
        n = 1024
        return pl.DataFrame(
            {
                "t": [base + timedelta(seconds=i) for i in range(n)],
                "cat": [str(i % 4) for i in range(n)],
            }
        )

    def _dt_spec(self, dt_df: pl.DataFrame, p: int = 64) -> CubeSpec:
        phys = dt_df["t"].to_physical()
        return CubeSpec(
            source_name="s",
            free=FreeAxisSpec(
                column="t",
                kind="temporal",
                p=p,
                domain=(float(phys.min()), float(phys.max())),
            ),
            target_dims=(TargetDimSpec(column="cat", kind="categorical"),),
        )

    def test_datetime_full_slice_equals_direct_count(self, dt_df):
        cube = build_cube(dt_df.lazy(), self._dt_spec(dt_df))
        lo, hi = cube.spec.free.domain
        sliced = cube.slice_count(lo, hi).sort("cat")
        direct = dt_df.group_by("cat").agg(pl.len().alias("count")).sort("cat")
        assert sliced.equals(direct)

    def test_datetime_subrange_slice_matches_direct_filter(self, dt_df):
        cube = build_cube(dt_df.lazy(), self._dt_spec(dt_df))
        lo, hi = cube.spec.free.domain
        p = cube.spec.free.p
        step = (hi - lo) / p  # 1023e6 / 64 = 15_984_375.0 µs, exactly integral
        free_lo, free_hi = lo + 100e6, lo + 500e6
        lo_bin, hi_bin = cube._snap(free_lo, free_hi)
        base = datetime(2024, 1, 1)
        dt_lo = base + timedelta(microseconds=lo_bin * step)
        dt_hi = base + timedelta(microseconds=(hi_bin + 1) * step)
        sliced = cube.slice_count(free_lo, free_hi).sort("cat")
        direct = (
            dt_df.filter((pl.col("t") >= dt_lo) & (pl.col("t") < dt_hi))
            .group_by("cat")
            .agg(pl.len().alias("count"))
            .sort("cat")
        )
        assert sliced.equals(direct)

    def test_date_smoke(self):
        days = [date(2024, 1, 1) + timedelta(days=i) for i in range(101)]
        df = pl.DataFrame({"d": days, "cat": [str(i % 3) for i in range(101)]})
        phys = df["d"].to_physical()
        spec = CubeSpec(
            source_name="s",
            free=FreeAxisSpec(
                column="d",
                kind="temporal",
                p=10,
                domain=(float(phys.min()), float(phys.max())),
            ),
            target_dims=(TargetDimSpec(column="cat", kind="categorical"),),
        )
        cube = build_cube(df.lazy(), spec)
        lo, hi = spec.free.domain
        assert cube.slice_count(lo, hi)["count"].sum() == df.height


def _free_cat(*columns: str, p: int = 0) -> FreeAxisSpec:
    """Categorical free axis as emitters build it: columns[0] is the primary
    column (the active_source join key), p pinned to 0, no domain."""
    return FreeAxisSpec(
        column=columns[0], columns=tuple(columns), kind="categorical", p=p, domain=None
    )


def _cat_free_spec(
    columns: tuple[str, ...], agg: str = "count", value_col: str | None = None
) -> CubeSpec:
    return CubeSpec(
        source_name="s",
        free=_free_cat(*columns),
        target_dims=(TargetDimSpec(column="cat", kind="categorical"),),
        measure=MeasureSpec(agg=agg, value_col=value_col),  # type: ignore[arg-type]
    )


class TestCategoricalFreeAxis:
    @pytest.fixture()
    def cdf(self) -> pl.DataFrame:
        # Two free-axis label columns (composite bar / treemap path), a
        # categorical target `cat`, and a numeric measure column `val`.
        n = 1200
        regions = ["east", "north", "south"]
        cities = ["amsterdam", "berlin", "cairo", "delhi"]
        return pl.DataFrame(
            {
                "region": [regions[i % 3] for i in range(n)],
                "city": [cities[(i * 7) % 4] for i in range(n)],
                "cat": [str(i % 5) for i in range(n)],
                "val": [float((i * 13) % 50) for i in range(n)],
            }
        )

    # --- FreeAxisSpec validation -------------------------------------------

    def test_categorical_requires_columns(self):
        with pytest.raises(ValueError):
            FreeAxisSpec(column="region", kind="categorical", p=0)

    def test_categorical_columns_first_must_equal_column(self):
        with pytest.raises(ValueError):
            FreeAxisSpec(
                column="region", columns=("city", "region"), kind="categorical", p=0
            )

    def test_categorical_domain_must_be_none(self):
        with pytest.raises(ValueError):
            FreeAxisSpec(
                column="region",
                columns=("region",),
                kind="categorical",
                p=0,
                domain=(0.0, 1.0),
            )

    @pytest.mark.parametrize("kind", ["continuous", "temporal"])
    def test_range_kinds_forbid_columns(self, kind):
        with pytest.raises(ValueError):
            FreeAxisSpec(
                column="active", columns=("active",), kind=kind, domain=(0.0, 1.0)
            )

    # --- build_cube validation ----------------------------------------------

    def test_build_cube_range_kind_still_requires_resolved_domain(self, df):
        spec = CubeSpec(
            source_name="s",
            free=FreeAxisSpec(column="active", p=64, domain=None),
            target_dims=(TargetDimSpec(column="cat", kind="categorical"),),
        )
        with pytest.raises(ValueError):
            build_cube(df.lazy(), spec)

    def test_build_cube_categorical_rejects_domain(self, cdf):
        spec = _cat_free_spec(("region",))
        object.__setattr__(spec.free, "domain", (0.0, 1.0))  # bypass __post_init__
        with pytest.raises(ValueError):
            build_cube(cdf.lazy(), spec)

    def test_build_cube_categorical_rejects_missing_columns(self, cdf):
        spec = _cat_free_spec(("region",))
        object.__setattr__(spec.free, "columns", None)  # bypass __post_init__
        with pytest.raises(ValueError):
            build_cube(cdf.lazy(), spec)

    # --- build shape ----------------------------------------------------------

    def test_build_has_free_key_cols_and_no_free_bin(self, cdf):
        cube = build_cube(cdf.lazy(), _cat_free_spec(("region", "city")))
        assert cube.free_key_cols == ("__free__region", "__free__city")
        assert "free_bin" not in cube.frame.columns
        assert set(cube.frame.columns) == {
            "cat",
            "__free__region",
            "__free__city",
            "count",
        }

    def test_range_cube_free_key_cols_empty(self, df):
        cube = build_cube(df.lazy(), _cat_spec())
        assert cube.free_key_cols == ()

    # --- slice_agg_keys -------------------------------------------------------

    @pytest.mark.parametrize("agg", ["count", "sum", "mean"])
    def test_single_col_slice_keys_equals_direct(self, cdf, agg):
        value_col = None if agg == "count" else "val"
        cube = build_cube(cdf.lazy(), _cat_free_spec(("region",), agg, value_col))
        sliced = cube.slice_agg_keys([("north",)]).sort("cat")
        direct = (
            cdf.filter(pl.col("region") == "north")
            .group_by("cat")
            .agg(_LEGACY_AGG[agg]("val").alias("value"))
            .sort("cat")
        )
        _assert_slice_matches(sliced, direct, agg)

    @pytest.mark.parametrize("agg", ["count", "sum"])
    def test_multi_col_exact_tuple_equals_direct(self, cdf, agg):
        value_col = None if agg == "count" else "val"
        cube = build_cube(
            cdf.lazy(), _cat_free_spec(("region", "city"), agg, value_col)
        )
        sliced = cube.slice_agg_keys([("north", "berlin")]).sort("cat")
        direct = (
            cdf.filter((pl.col("region") == "north") & (pl.col("city") == "berlin"))
            .group_by("cat")
            .agg(_LEGACY_AGG[agg]("val").alias("value"))
            .sort("cat")
        )
        _assert_slice_matches(sliced, direct, agg)

    def test_path_prefix_expanded_keys_equal_direct(self, cdf):
        # Treemap-style path predicate constraining a prefix of the cols: the
        # caller expands the prefix to the full key tuples itself.
        cube = build_cube(cdf.lazy(), _cat_free_spec(("region", "city")))
        keys = sorted(
            set(
                zip(
                    cdf.filter(pl.col("region") == "south")["region"].to_list(),
                    cdf.filter(pl.col("region") == "south")["city"].to_list(),
                )
            )
        )
        sliced = cube.slice_agg_keys(keys).sort("cat")
        direct = (
            cdf.filter(pl.col("region") == "south")
            .group_by("cat")
            .agg(pl.len().alias("value"))
            .sort("cat")
        )
        _assert_slice_matches(sliced, direct, "count")

    def test_union_of_keys_equals_is_in_union(self, cdf):
        # OR'd multi-select: the union of exact keys == one is_in over the union.
        cube = build_cube(cdf.lazy(), _cat_free_spec(("region", "city")))
        keys = [("north", "berlin"), ("south", "cairo"), ("east", "delhi")]
        sliced = cube.slice_agg_keys(keys).sort("cat")
        member = pl.any_horizontal(
            *[(pl.col("region") == r) & (pl.col("city") == c) for r, c in keys]
        )
        direct = (
            cdf.filter(member).group_by("cat").agg(pl.len().alias("value")).sort("cat")
        )
        _assert_slice_matches(sliced, direct, "count")

    def test_null_in_any_free_col_drops_row(self):
        df = pl.DataFrame(
            {
                "region": ["north", None, "south", "south"],
                "city": ["berlin", "berlin", None, "cairo"],
                "cat": ["a", "a", "a", "a"],
                "val": [1.0, 1.0, 1.0, 1.0],
            }
        )
        cube = build_cube(df.lazy(), _cat_free_spec(("region", "city")))
        # Rows 1 (null region) and 2 (null city) are absent from the cube.
        assert cube.frame["count"].sum() == 2
        all_keys = [("north", "berlin"), ("south", "cairo")]
        assert cube.slice_agg_keys(all_keys)["value"].sum() == 2

    def test_empty_keys_empty_result(self, cdf):
        cube = build_cube(cdf.lazy(), _cat_free_spec(("region",)))
        assert cube.slice_agg_keys([]).height == 0

    def test_key_arity_mismatch_raises(self, cdf):
        cube = build_cube(cdf.lazy(), _cat_free_spec(("region", "city")))
        with pytest.raises(ValueError):
            cube.slice_agg_keys([("north",)])

    # --- codec ----------------------------------------------------------------

    def _reslice_keys(self, blob: bytes, keys: list[tuple[str, ...]]) -> dict:
        """Count per target cell over selected category codes, from raw bytes."""
        header = decode_fvcube_header(blob)
        categories = [tuple(t) for t in header["free"]["categories"]]
        selected = {categories.index(k) for k in keys if k in categories}
        free_bin = _read_u32_col(blob, header, "free_bin")
        count = _read_u32_col(blob, header, "count")
        n_t = len(header["target_dims"])
        tgt_names = [c["name"] for c in header["columns"][1 : 1 + n_t]]
        tgt_cols = [_read_u32_col(blob, header, n) for n in tgt_names]
        out: dict = {}
        for i, fb in enumerate(free_bin):
            if fb in selected:
                key = tuple(col[i] for col in tgt_cols)
                out[key] = out.get(key, 0) + count[i]
        return out

    def test_codec_header_free_block_shape(self, cdf):
        cube = build_cube(cdf.lazy(), _cat_free_spec(("region", "city")))
        blob = encode_fvcube(cube, cube_id="cat01")
        header = decode_fvcube_header(blob)
        free = header["free"]
        assert free["kind"] == "categorical"
        assert free["cols"] == ["region", "city"]
        assert "p" not in free and "domain" not in free
        # Categories are the distinct tuples, in Python-sorted order.
        expected = sorted(set(zip(cdf["region"].to_list(), cdf["city"].to_list())))
        assert [tuple(t) for t in free["categories"]] == expected
        # Codec stays version 1.
        assert header["v"] == 1

    def test_codec_codes_are_positions_in_sorted_categories(self, cdf):
        cube = build_cube(cdf.lazy(), _cat_free_spec(("region", "city")))
        blob = encode_fvcube(cube, cube_id="cat02")
        header = decode_fvcube_header(blob)
        categories = [tuple(t) for t in header["free"]["categories"]]
        free_bin = _read_u32_col(blob, header, "free_bin")
        count = _read_u32_col(blob, header, "count")
        assert max(free_bin) == len(categories) - 1
        by_code: dict = {}
        for code, n in zip(free_bin, count):
            by_code[code] = by_code.get(code, 0) + n
        direct = {
            (r, c): n for r, c, n in cdf.group_by("region", "city").len().iter_rows()
        }
        assert by_code == {categories.index(k): n for k, n in direct.items()}

    def test_codec_rows_sorted_by_code_then_targets(self, cdf):
        cube = build_cube(cdf.lazy(), _cat_free_spec(("region", "city")))
        blob = encode_fvcube(cube, cube_id="cat03")
        header = decode_fvcube_header(blob)
        free_bin = _read_u32_col(blob, header, "free_bin")
        cat = _read_u32_col(blob, header, "cat")
        assert sorted(zip(free_bin, cat)) == list(zip(free_bin, cat))

    def test_codec_deterministic_bytes(self, cdf):
        cube = build_cube(cdf.lazy(), _cat_free_spec(("region", "city")))
        rebuilt = build_cube(cdf.lazy(), _cat_free_spec(("region", "city")))
        assert encode_fvcube(cube, "k") == encode_fvcube(rebuilt, "k")

    def test_codec_reslice_matches_slice_agg_keys(self, cdf):
        cube = build_cube(cdf.lazy(), _cat_free_spec(("region", "city")))
        blob = encode_fvcube(cube, cube_id="cat04")
        header = decode_fvcube_header(blob)
        (dim,) = header["target_dims"]
        keys = [("north", "berlin"), ("east", "amsterdam")]
        sliced = self._reslice_keys(blob, keys)
        ref = cube.slice_agg_keys(keys)
        ref_by_code = {
            (dim["categories"].index(cat),): v
            for cat, v in zip(ref["cat"].to_list(), ref["value"].to_list())
        }
        assert sliced == ref_by_code

    # --- content key ------------------------------------------------------------

    def test_content_key_categorical_vs_continuous_differ(self, df):
        cat = CubeSpec(
            source_name="s",
            free=_free_cat("active"),
            target_dims=(TargetDimSpec(column="cat", kind="categorical"),),
        )
        assert cube_content_key(cat) != cube_content_key(_cat_spec())

    def test_content_key_column_order_matters(self):
        a = CubeSpec(
            source_name="s",
            free=FreeAxisSpec(
                column="a", columns=("a", "b", "c"), kind="categorical", p=0
            ),
            target_dims=(TargetDimSpec(column="cat", kind="categorical"),),
        )
        b = CubeSpec(
            source_name="s",
            free=FreeAxisSpec(
                column="a", columns=("a", "c", "b"), kind="categorical", p=0
            ),
            target_dims=(TargetDimSpec(column="cat", kind="categorical"),),
        )
        assert cube_content_key(a) != cube_content_key(b)

    def test_content_key_p_and_domain_pinned(self):
        # "p" is pinned to 0 and "d" to None in the key, whatever the spec says.
        a = CubeSpec(
            source_name="s",
            free=_free_cat("region", p=0),
            target_dims=(TargetDimSpec(column="cat", kind="categorical"),),
        )
        b = CubeSpec(
            source_name="s",
            free=_free_cat("region", p=2048),
            target_dims=(TargetDimSpec(column="cat", kind="categorical"),),
        )
        assert cube_content_key(a) == cube_content_key(b)


class TestNulls:
    def test_null_rows_absent_from_cube(self):
        df = pl.DataFrame(
            {
                "active": [1.0, 2.0, None, 3.0, None, 4.0],
                "val": [1.0, None, 2.0, 3.0, 4.0, None],
            }
        )
        spec = CubeSpec(
            source_name="s",
            free=FreeAxisSpec(column="active", p=8, domain=(0.0, 10.0)),
            target_dims=(
                TargetDimSpec(column="val", kind="binned", bins=4, domain=(0.0, 5.0)),
            ),
        )
        cube = build_cube(df.lazy(), spec)
        # Only rows with both columns non-null survive: (1,1) and (3,3).
        assert cube.frame["count"].sum() == 2


class TestContentKey:
    def test_same_spec_same_key(self):
        assert cube_content_key(_cat_spec()) == cube_content_key(_cat_spec())

    def test_different_p_different_key(self):
        assert cube_content_key(_cat_spec(p=64)) != cube_content_key(_cat_spec(p=128))

    def test_key_excludes_nothing_cube_determining(self):
        a = _cat_spec()
        b = CubeSpec(
            source_name="other",
            free=a.free,
            target_dims=a.target_dims,
            measure=a.measure,
        )
        assert cube_content_key(a) != cube_content_key(b)

    def test_temporal_kind_changes_key(self):
        a = _cat_spec()
        b = CubeSpec(
            source_name="s",
            free=FreeAxisSpec(
                column="active", kind="temporal", p=64, domain=(0.0, 100.0)
            ),
            target_dims=a.target_dims,
            measure=a.measure,
        )
        assert cube_content_key(a) != cube_content_key(b)

    def test_binned_target_changes_key(self, df):
        a = _cat_spec()
        b = CubeSpec(
            source_name="s",
            free=a.free,
            target_dims=(
                TargetDimSpec(column="val", kind="binned", bins=10, domain=(0.0, 50.0)),
            ),
        )
        assert cube_content_key(a) != cube_content_key(b)


def _buffer_section_start(blob: bytes) -> int:
    (header_len,) = struct.unpack_from("<I", blob, 8)
    return 12 + header_len


def _read_u32_col(blob: bytes, header: dict, name: str) -> list[int]:
    """Read one u32 column from the raw buffer section (test-side decoder)."""
    col = next(c for c in header["columns"] if c["name"] == name)
    assert col["dtype"] == "u32"
    start = _buffer_section_start(blob) + col["offset"]
    n = col["byte_len"] // 4
    return list(struct.unpack_from(f"<{n}I", blob, start))


def _read_col(blob: bytes, header: dict, name: str) -> list:
    """Read one column (u32/f64/f32/u16) from the raw buffer section."""
    col = next(c for c in header["columns"] if c["name"] == name)
    start = _buffer_section_start(blob) + col["offset"]
    code, size = {"u32": ("I", 4), "f64": ("d", 8), "f32": ("f", 4), "u16": ("H", 2)}[
        col["dtype"]
    ]
    n = col["byte_len"] // size
    return list(struct.unpack_from(f"<{n}{code}", blob, start))


class TestFVCubeCodec:
    def _reslice(self, blob: bytes, cube: CubeResult, lo: float, hi: float) -> dict:
        """Count per target cell over a snapped free range, read from raw bytes."""
        header = decode_fvcube_header(blob)
        free_bin = _read_u32_col(blob, header, "free_bin")
        count = _read_u32_col(blob, header, "count")
        tgt_names = [c["name"] for c in header["columns"][1:-1]]
        tgt_cols = [_read_u32_col(blob, header, n) for n in tgt_names]
        lo_bin, hi_bin = cube._snap(lo, hi)
        out: dict = {}
        for i, fb in enumerate(free_bin):
            if lo_bin <= fb <= hi_bin:
                key = tuple(col[i] for col in tgt_cols)
                out[key] = out.get(key, 0) + count[i]
        return out

    def test_round_trip_categorical_target(self, df):
        cube = build_cube(df.lazy(), _cat_spec())
        blob = encode_fvcube(cube, cube_id="abc123")
        header = decode_fvcube_header(blob)
        assert header["cube_id"] == "abc123"
        assert header["rows"] == cube.n_cells
        assert header["sorted_by"] == "free_bin"
        assert header["free"] == {"kind": "continuous", "p": 64, "domain": [0.0, 100.0]}
        assert header["measure"] == {"agg": "count", "value_col": None}
        (dim,) = header["target_dims"]
        assert dim["name"] == "cat"
        assert dim["kind"] == "categorical"
        assert dim["categories"] == sorted(df["cat"].unique().to_list())
        # Reslice a snapped sub-range from the raw buffers and compare to the
        # server-side reference slice — exact equality.
        sliced = self._reslice(blob, cube, 25.0, 75.0)
        ref = cube.slice_count(25.0, 75.0)
        ref_by_code = {
            (dim["categories"].index(cat),): n
            for cat, n in zip(ref["cat"].to_list(), ref["count"].to_list())
        }
        assert sliced == ref_by_code

    def test_round_trip_binned_target(self, df):
        spec = CubeSpec(
            source_name="s",
            free=FreeAxisSpec(column="active", p=64, domain=(0.0, 100.0)),
            target_dims=(
                TargetDimSpec(column="val", kind="binned", bins=10, domain=(0.0, 50.0)),
            ),
        )
        cube = build_cube(df.lazy(), spec)
        blob = encode_fvcube(cube, cube_id="bin01")
        header = decode_fvcube_header(blob)
        (dim,) = header["target_dims"]
        assert dim == {
            "name": "val",
            "kind": "binned",
            "bins": 10,
            "domain": [0.0, 50.0],
        }
        assert [c["name"] for c in header["columns"]] == [
            "free_bin",
            "__bin__val",
            "count",
        ]
        sliced = self._reslice(blob, cube, 10.0, 60.0)
        ref = cube.slice_count(10.0, 60.0)
        ref_by_bin = {
            (b,): n for b, n in zip(ref["__bin__val"].to_list(), ref["count"].to_list())
        }
        assert sliced == ref_by_bin

    def test_round_trip_integer_categorical_target(self):
        # An integer categorical target encodes its categories as integers in
        # NUMERIC order — not stringified + lexicographically sorted — so a
        # cube delta's labels byte-match the server's typed, numeric-sorted
        # grouped/ungrouped output (the demo's hour_of_day / month bars).
        n = 600
        df = pl.DataFrame(
            {
                "active": [(i * 37) % 100 for i in range(n)],
                # 0..11 spans the range where numeric order (0,1,2,…,10,11)
                # differs from string order ("0","1","10","11","2",…).
                "icat": pl.Series([i % 12 for i in range(n)], dtype=pl.Int64),
            }
        )
        spec = CubeSpec(
            source_name="s",
            free=FreeAxisSpec(column="active", p=64, domain=(0.0, 100.0)),
            target_dims=(TargetDimSpec(column="icat", kind="categorical"),),
            measure=MeasureSpec(agg="count"),
        )
        cube = build_cube(df.lazy(), spec)
        blob = encode_fvcube(cube, cube_id="icat01")
        header = decode_fvcube_header(blob)
        (dim,) = header["target_dims"]
        assert dim["kind"] == "categorical"
        # Integers, in numeric order — NOT ["0","1","10","11","2",…].
        assert dim["categories"] == list(range(12))
        # Codes index into the integer category list (slice parity).
        sliced = self._reslice(blob, cube, 25.0, 75.0)
        ref = cube.slice_count(25.0, 75.0)
        ref_by_code = {
            (dim["categories"].index(c),): n
            for c, n in zip(ref["icat"].to_list(), ref["count"].to_list())
        }
        assert sliced == ref_by_code

    def test_deterministic_bytes(self, df):
        cube = build_cube(df.lazy(), _cat_spec())
        assert encode_fvcube(cube, "k") == encode_fvcube(cube, "k")
        # Rebuilding the same cube (nondeterministic group_by order) must still
        # produce identical bytes — encode sorts.
        rebuilt = build_cube(df.lazy(), _cat_spec())
        assert encode_fvcube(rebuilt, "k") == encode_fvcube(cube, "k")

    def test_rows_sorted_by_free_bin_then_targets(self, df):
        cube = build_cube(df.lazy(), _cat_spec())
        blob = encode_fvcube(cube, "k")
        header = decode_fvcube_header(blob)
        free_bin = _read_u32_col(blob, header, "free_bin")
        cat = _read_u32_col(blob, header, "cat")
        assert sorted(zip(free_bin, cat)) == list(zip(free_bin, cat))

    def test_buffer_offsets_8_byte_aligned(self, df):
        cube = build_cube(df.lazy(), _cat_spec())
        blob = encode_fvcube(cube, "k")
        header = decode_fvcube_header(blob)
        for col in header["columns"]:
            assert col["offset"] % 8 == 0

    def test_bad_magic_raises(self, df):
        cube = build_cube(df.lazy(), _cat_spec())
        blob = bytearray(encode_fvcube(cube, "k"))
        blob[:6] = b"NOPE!!"
        with pytest.raises(ValueError):
            decode_fvcube_header(bytes(blob))

    def test_bad_version_raises(self, df):
        cube = build_cube(df.lazy(), _cat_spec())
        blob = bytearray(encode_fvcube(cube, "k"))
        blob[6] = 99
        with pytest.raises(ValueError):
            decode_fvcube_header(bytes(blob))

    def test_truncated_blob_raises(self, df):
        cube = build_cube(df.lazy(), _cat_spec())
        blob = encode_fvcube(cube, "k")
        with pytest.raises(ValueError):
            decode_fvcube_header(blob[:8])  # cut inside the fixed prelude
        with pytest.raises(ValueError):
            decode_fvcube_header(blob[:20])  # cut inside the header JSON
        with pytest.raises(ValueError):
            decode_fvcube_header(blob[:-4])  # cut inside the buffer section

    def test_count_overflow_raises(self):
        spec = _cat_spec()
        frame = pl.DataFrame(
            {
                "cat": ["a"],
                "free_bin": pl.Series([0], dtype=pl.Int32),
                "count": pl.Series([2**32], dtype=pl.UInt64),
            }
        )
        result = CubeResult(spec=spec, frame=frame, group_cols=("cat",))
        with pytest.raises(ValueError):
            encode_fvcube(result, "k")

    def _reslice_agg(self, blob: bytes, cube: CubeResult, lo: float, hi: float) -> dict:
        """Contract-A combine over the raw buffers (the JS reference semantics):
        counts/sums Σ; mean = Σsum / Σcount using the integer count (never
        NaN-checks the sum), 0-count cells omitted; min/max skip NaN partials,
        all-NaN cells omitted."""
        header = decode_fvcube_header(blob)
        agg = header["measure"]["agg"]
        free_bin = _read_col(blob, header, "free_bin")
        n_t = len(header["target_dims"])
        tgt_names = [c["name"] for c in header["columns"][1 : 1 + n_t]]
        tgt_cols = [_read_col(blob, header, n) for n in tgt_names]
        lo_bin, hi_bin = cube._snap(lo, hi)
        rows = [i for i, fb in enumerate(free_bin) if lo_bin <= fb <= hi_bin]
        keys = [tuple(col[i] for col in tgt_cols) for i in rows]
        out: dict = {}
        if agg in ("count", "sum"):
            partial = _read_col(blob, header, agg)
            for i, key in zip(rows, keys):
                out[key] = out.get(key, 0) + partial[i]
        elif agg == "mean":
            sums = _read_col(blob, header, "sum")
            counts = _read_col(blob, header, "count")
            acc: dict = {}
            for i, key in zip(rows, keys):
                s, c = acc.get(key, (0.0, 0))
                acc[key] = (s + sums[i], c + counts[i])
            out = {k: s / c for k, (s, c) in acc.items() if c > 0}
        else:  # min / max
            partial = _read_col(blob, header, agg)
            pick = min if agg == "min" else max
            for i, key in zip(rows, keys):
                v = partial[i]
                if math.isnan(v):
                    continue  # NaN = encoded null partial → skipped
                out[key] = v if key not in out else pick(out[key], v)
        return out

    @pytest.mark.parametrize("agg", _ALL_AGGS)
    def test_round_trip_measures(self, df, agg):
        cube = build_cube(df.lazy(), _measure_spec(agg))
        blob = encode_fvcube(cube, cube_id="m1")
        header = decode_fvcube_header(blob)
        assert header["measure"] == {
            "agg": agg,
            "value_col": None if agg == "count" else "val",
        }
        partial_cols = {
            "count": [("count", "u32")],
            "sum": [("sum", "f64")],
            "mean": [("sum", "f64"), ("count", "u32")],
            "min": [("min", "f64")],
            "max": [("max", "f64")],
        }[agg]
        assert [(c["name"], c["dtype"]) for c in header["columns"]] == [
            ("free_bin", "u32"),
            ("cat", "u32"),
            *partial_cols,
        ]
        (dim,) = header["target_dims"]
        sliced = self._reslice_agg(blob, cube, 25.0, 75.0)
        ref = cube.slice_agg(25.0, 75.0)
        ref_by_code = {
            (dim["categories"].index(cat),): v
            for cat, v in zip(ref["cat"].to_list(), ref["value"].to_list())
        }
        if agg in _EXACT_AGGS:
            assert sliced == ref_by_code
        else:
            assert sliced.keys() == ref_by_code.keys()
            for k in sliced:
                assert sliced[k] == pytest.approx(ref_by_code[k], rel=1e-9, abs=1e-9)

    def test_f64_offsets_8_byte_aligned(self, df):
        # mean mixes u32 and f64 buffers — every offset must still be 8-aligned.
        cube = build_cube(df.lazy(), _measure_spec("mean"))
        blob = encode_fvcube(cube, "k")
        header = decode_fvcube_header(blob)
        assert _buffer_section_start(blob) % 8 == 0
        for col in header["columns"]:
            assert col["offset"] % 8 == 0

    def test_null_partial_encodes_as_nan(self):
        df = pl.DataFrame(
            {
                "active": [10.0, 10.0, 90.0],
                "cat": ["a", "a", "a"],
                "val": [None, None, 5.0],
            }
        )
        cube = build_cube(df.lazy(), _measure_spec("min"))
        blob = encode_fvcube(cube, "k")
        header = decode_fvcube_header(blob)
        mins = _read_col(blob, header, "min")
        assert sum(1 for v in mins if math.isnan(v)) == 1
        assert 5.0 in mins
        # The contract-A combine skips the NaN partial.
        assert self._reslice_agg(blob, cube, 0.0, 100.0) == {(0,): 5.0}
        assert self._reslice_agg(blob, cube, 0.0, 50.0) == {}

    def test_mean_zero_count_cell_omitted_from_reslice(self):
        df = pl.DataFrame(
            {
                "active": [10.0, 20.0],
                "cat": ["a", "b"],
                "val": [4.0, None],
            }
        )
        cube = build_cube(df.lazy(), _measure_spec("mean"))
        blob = encode_fvcube(cube, "k")
        assert self._reslice_agg(blob, cube, 0.0, 100.0) == {(0,): 4.0}

    @pytest.mark.parametrize("agg", ["sum", "mean", "min", "max"])
    def test_deterministic_bytes_measures(self, df, agg):
        cube = build_cube(df.lazy(), _measure_spec(agg))
        rebuilt = build_cube(df.lazy(), _measure_spec(agg))
        assert encode_fvcube(cube, "k") == encode_fvcube(rebuilt, "k")

    def test_degenerate_top_bin_survives_round_trip(self):
        df = pl.DataFrame({"active": [0.0, 50.0, 100.0], "cat": ["a", "a", "a"]})
        cube = build_cube(df.lazy(), _cat_spec())
        blob = encode_fvcube(cube, "k")
        header = decode_fvcube_header(blob)
        free_bin = _read_u32_col(blob, header, "free_bin")
        assert 64 in free_bin  # bin == P present in the encoded rows
        # A reslice reaching the domain max includes the degenerate bin…
        full = self._reslice(blob, cube, 0.0, 100.0)
        assert sum(full.values()) == 3
        # …and a reslice stopping short of it does not.
        partial = self._reslice(blob, cube, 0.0, 99.0)
        assert sum(partial.values()) == 2


class TestValidation:
    def test_binned_dim_requires_bins_and_domain(self):
        with pytest.raises(ValueError):
            TargetDimSpec(column="val", kind="binned")

    def test_unimplemented_measure_raises(self, df):
        spec = CubeSpec(
            source_name="s",
            free=FreeAxisSpec(column="active", p=64, domain=(0.0, 100.0)),
            target_dims=(TargetDimSpec(column="cat", kind="categorical"),),
            measure=MeasureSpec(agg="count"),
        )
        # The five cube aggs work; a not-yet-supported measure (validation
        # bypassed via object.__setattr__) raises NotImplementedError.
        object.__setattr__(spec.measure, "agg", "median")
        with pytest.raises(NotImplementedError):
            build_cube(df.lazy(), spec)


# ---------------------------------------------------------------------------
# Composite-label golden strings (plan step 0d)
# ---------------------------------------------------------------------------


class TestGroupValueKeyGoldenStrings:
    """Golden strings pinning the composite-label wire format: Python's
    compact ``json.dumps`` with ``ensure_ascii=True``. The JS mirror
    ``fvJsonDumpsAscii`` must reproduce these byte-for-byte (browser parity
    test in ``test_browser_cube.py``)."""

    def test_non_ascii_escaped_lowercase(self):
        from flexviz.trace.base import _group_value_key

        assert _group_value_key(("é", "x")) == '["\\u00e9","x"]'

    def test_astral_char_escaped_as_surrogate_pair(self):
        from flexviz.trace.base import _group_value_key

        assert _group_value_key(("🎉", "x")) == '["\\ud83c\\udf89","x"]'

    def test_ascii_control_char_escaped(self):
        from flexviz.trace.base import _group_value_key

        assert _group_value_key(("\x07", "x")) == '["\\u0007","x"]'

    def test_del_char_escaped(self):
        # 0x7F is escaped by Python's ensure_ascii (everything > 0x7E) but
        # NOT by bare JSON.stringify — the exact divergence fvJsonDumpsAscii
        # must cover.
        from flexviz.trace.base import _group_value_key

        assert _group_value_key(("\x7f", "x")) == '["\\u007f","x"]'

    def test_plain_ascii_unchanged(self):
        from flexviz.trace.base import _group_value_key

        assert _group_value_key(("a", "x")) == '["a","x"]'


# ---------------------------------------------------------------------------
# Passive key in the cube content key (plan step 2 / contract E)
# ---------------------------------------------------------------------------


class TestPassiveKeyInContentKey:
    @staticmethod
    def _spec(passive_key=None) -> CubeSpec:
        return CubeSpec(
            source_name="src",
            free=FreeAxisSpec(column="active", p=2048, domain=(0.0, 100.0)),
            target_dims=(
                TargetDimSpec(column="val", kind="binned", bins=12, domain=(0.0, 1.0)),
            ),
            measure=MeasureSpec(agg="count"),
            passive_key=passive_key,
        )

    def test_none_passive_key_is_byte_identical_to_phase2(self):
        # passive_key defaults to None — explicitly passing None must hash
        # identically (zero-passive cubes keep colliding across phases).
        assert cube_content_key(self._spec()) == cube_content_key(
            self._spec(passive_key=None)
        )

    def test_passive_key_differentiates_cubes(self):
        from flexviz.predicates import canonical_passive_key
        from flexviz.spec import SelectionPredicate, SelectionState

        sel = SelectionState(
            source_figure_uid="figC",
            predicates=[
                SelectionPredicate.model_validate(
                    {"clauses": [{"column": "cat", "values": ["1"]}]}
                )
            ],
        )
        pkey = canonical_passive_key([sel], "figA")
        assert pkey is not None
        assert cube_content_key(self._spec(pkey)) != cube_content_key(self._spec())

    def test_distinct_passive_sets_distinct_keys(self):
        from flexviz.predicates import canonical_passive_key
        from flexviz.spec import SelectionPredicate, SelectionState

        def sel(values):
            return SelectionState(
                source_figure_uid="figC",
                predicates=[
                    SelectionPredicate.model_validate(
                        {"clauses": [{"column": "cat", "values": values}]}
                    )
                ],
            )

        k1 = canonical_passive_key([sel(["1"])], "figA")
        k2 = canonical_passive_key([sel(["2"])], "figA")
        assert cube_content_key(self._spec(k1)) != cube_content_key(self._spec(k2))

    def test_equivalent_passive_sets_share_key(self):
        from flexviz.predicates import canonical_passive_key
        from flexviz.spec import SelectionPredicate, SelectionState

        def sel(uid, values):
            return SelectionState(
                source_figure_uid=uid,
                predicates=[
                    SelectionPredicate.model_validate(
                        {"clauses": [{"column": "cat", "values": values}]}
                    )
                ],
            )

        # Same passive semantics in different list order ⇒ same content key.
        k1 = canonical_passive_key([sel("figB", ["1"]), sel("figC", ["2"])], "figA")
        k2 = canonical_passive_key([sel("figC", ["2"]), sel("figB", ["1"])], "figA")
        assert cube_content_key(self._spec(k1)) == cube_content_key(self._spec(k2))


# ---------------------------------------------------------------------------
# Temporal units (plan step 6 / contract G)
# ---------------------------------------------------------------------------


def _physical_to_temporal_str(value: float, unit: str) -> str:
    """Python mirror of the JS ``fvPhysicalToTemporal``: render a snapped
    physical edge as the committed UTC string. The value is ceil-ed to an
    integral count of the column's unit first — integer membership over the
    column values is unchanged (lo closed: v >= e ⟺ v >= ceil(e); hi open:
    v < e ⟺ v < ceil(e)) and the string then represents it exactly."""
    if unit == "day":
        return (date(1970, 1, 1) + timedelta(days=math.ceil(value))).isoformat()
    us = math.ceil(value) if unit == "us" else math.ceil(value) * 1000
    dt = datetime(1970, 1, 1) + timedelta(microseconds=us)
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")


def _temporal_df(unit: str, n: int = 600, span_days: int = 365) -> pl.DataFrame:
    if unit == "day":
        base = date(2020, 1, 1)
        col = pl.Series(
            "t", [base + timedelta(days=(i * 7) % span_days) for i in range(n)]
        )
    else:
        base = datetime(2020, 1, 1)
        col = pl.Series(
            "t",
            [base + timedelta(minutes=(i * 37) % 10_000) for i in range(n)],
            dtype=pl.Datetime(unit),
        )
    return pl.DataFrame({"t": col, "b": [((i * 53) % 500) / 5 for i in range(n)]})


def _physical_domain(df: pl.DataFrame, col: str) -> tuple[float, float]:
    s = df[col].to_physical().cast(pl.Float64)
    return float(s.min()), float(s.max())


class TestTemporalUnits:
    @pytest.mark.parametrize("unit", ["us", "ms", "day"])
    def test_header_carries_unit(self, unit):
        from flexviz.cube import temporal_unit

        df = _temporal_df(unit)
        assert temporal_unit(df.schema["t"]) == unit
        lo, hi = _physical_domain(df, "t")
        spec = CubeSpec(
            source_name="src",
            free=FreeAxisSpec(
                column="t", kind="temporal", p=2048, domain=(lo, hi), unit=unit
            ),
            target_dims=(
                TargetDimSpec(column="b", kind="binned", bins=12, domain=(0.0, 100.0)),
            ),
            measure=MeasureSpec(agg="count"),
        )
        blob = encode_fvcube(build_cube(df.lazy(), spec), cube_id="t")
        header = decode_fvcube_header(blob)
        assert header["free"]["kind"] == "temporal"
        assert header["free"]["unit"] == unit
        if unit == "day":
            assert "w" in header["free"] and "p_eff" in header["free"]
        else:
            assert "w" not in header["free"]

    def test_ns_not_supported_by_temporal_unit(self):
        from flexviz.cube import temporal_unit

        assert temporal_unit(pl.Datetime("ns")) is None
        assert temporal_unit(pl.Time) is None
        assert temporal_unit(pl.Float64) is None

    def test_day_grid_short_span_unit_width(self):
        from flexviz.cube import day_grid

        # span < 2048 days ⇒ w = 1, P' = span (whole days).
        w, p_eff = day_grid(0.0, 364.0, 2048)
        assert w == 1
        assert p_eff == 364

    def test_day_grid_long_span_integer_width(self):
        from flexviz.cube import day_grid

        # span 10000 days ⇒ w = ceil(10000/2048) = 5, P' = ceil(10000/5) = 2000.
        w, p_eff = day_grid(0.0, 10_000.0, 2048)
        assert w == 5
        assert p_eff == 2000
        # Every snap edge lo + k*w is an integer day by construction.

    @pytest.mark.parametrize(
        "unit,span_days", [("us", 365), ("ms", 365), ("day", 365), ("day", 9000)]
    )
    def test_slice_membership_equals_string_predicate(self, unit, span_days):
        """§8.2 parity per unit: a cube slice over snapped bins selects
        exactly the rows the committed (rendered-string, closed="left")
        predicate selects through _typed_range_bounds — bit-exact counts."""
        from flexviz.cube import day_grid
        from flexviz.predicates import predicates_to_expr
        from flexviz.spec import ClauseFilter, SelectionPredicate

        df = _temporal_df(unit, span_days=span_days)
        lo, hi = _physical_domain(df, "t")
        spec = CubeSpec(
            source_name="src",
            free=FreeAxisSpec(
                column="t", kind="temporal", p=2048, domain=(lo, hi), unit=unit
            ),
            target_dims=(
                TargetDimSpec(column="b", kind="binned", bins=12, domain=(0.0, 100.0)),
            ),
            measure=MeasureSpec(agg="count"),
        )
        result = build_cube(df.lazy(), spec)

        if unit == "day":
            w, p_eff = day_grid(lo, hi, 2048)
        else:
            w, p_eff = (hi - lo) / 2048, 2048

        # Snap a brush to [lo_bin, hi_bin] (roughly the middle half).
        lo_bin = p_eff // 4
        hi_bin = (3 * p_eff) // 4
        edge_lo = lo + lo_bin * w
        edge_hi = lo + (hi_bin + 1) * w
        sliced = (
            result.frame.filter(pl.col("free_bin").is_between(lo_bin, hi_bin))[
                "count"
            ].sum()
            or 0
        )

        str_lo = _physical_to_temporal_str(edge_lo, unit)
        str_hi = _physical_to_temporal_str(edge_hi, unit)
        if unit == "day":
            # Integer-day grid: the strings round-trip the edges bit-exactly.
            assert date.fromisoformat(str_lo) == date(1970, 1, 1) + timedelta(
                days=edge_lo
            )
            assert date.fromisoformat(str_hi) == date(1970, 1, 1) + timedelta(
                days=edge_hi
            )
        pred = SelectionPredicate(
            clauses=[ClauseFilter(column="t", range=(str_lo, str_hi), closed="left")]
        )
        direct = df.filter(predicates_to_expr([pred], df.schema)).height
        assert sliced == direct
        assert 0 < sliced < df.height

    def test_day_degenerate_top_bin(self):
        """A row at the exact domain max lands in the degenerate top bin
        P' and a slice reaching it picks the row up."""
        from flexviz.cube import day_grid

        df = _temporal_df("day", span_days=365)
        lo, hi = _physical_domain(df, "t")
        w, p_eff = day_grid(lo, hi, 2048)
        spec = CubeSpec(
            source_name="src",
            free=FreeAxisSpec(
                column="t", kind="temporal", p=2048, domain=(lo, hi), unit="day"
            ),
            target_dims=(
                TargetDimSpec(column="b", kind="binned", bins=12, domain=(0.0, 100.0)),
            ),
            measure=MeasureSpec(agg="count"),
        )
        result = build_cube(df.lazy(), spec)
        assert result.frame["free_bin"].max() == p_eff
        n_max_rows = df.filter(pl.col("t").to_physical().cast(pl.Float64) == hi).height
        top = result.frame.filter(pl.col("free_bin") == p_eff)["count"].sum()
        assert top == n_max_rows > 0

    def test_temporal_binned_target_dim_carries_unit_and_bins_physical(self):
        """A temporal BINNED target dim bins on the physical representation
        and its header entry carries the unit."""
        df = _temporal_df("ms")
        lo, hi = _physical_domain(df, "t")
        spec = CubeSpec(
            source_name="src",
            free=FreeAxisSpec(
                column="b", kind="continuous", p=2048, domain=(0.0, 100.0)
            ),
            target_dims=(
                TargetDimSpec(
                    column="t", kind="binned", bins=10, domain=(lo, hi), unit="ms"
                ),
            ),
            measure=MeasureSpec(agg="count"),
        )
        result = build_cube(df.lazy(), spec)
        assert result.frame["count"].sum() == df.height
        blob = encode_fvcube(result, cube_id="t")
        header = decode_fvcube_header(blob)
        (dim,) = header["target_dims"]
        assert dim["unit"] == "ms"


# ---------------------------------------------------------------------------
# Line envelope measure (plan step 10 / contract J)
# ---------------------------------------------------------------------------


def _line_env_reference(
    x: pl.Series,
    y: pl.Series,
    free: pl.Series,
    x_lo: float,
    x_hi: float,
    free_lo: float,
    free_hi: float,
    n_buckets: int,
    p: int,
) -> pl.DataFrame:
    """Pure-Polars build reference for the line envelope (replicated from
    ``flexviz_polars/tests/test_plugin_functions.py::_envelope_reference``).

    Shared-arithmetic semantics on BOTH axes: natural floor bin, NO epsilon,
    NO clip, out-of-domain rows FILTERED, degenerate top bins (``0..=n``
    inclusive), null/NaN rows filtered, first-row-in-scan-order tie wins.

    The span divisors are materialized as REAL COLUMNS: Polars rewrites float
    division *by a scalar* into multiplication by the reciprocal, which is
    not IEEE division and can shift a domain-max value out of its degenerate
    top bin. The kernel — like the JS client — uses true division, and
    column/column division in Polars is true division too.
    """
    x_span = (x_hi - x_lo) or 1.0
    f_span = (free_hi - free_lo) or 1.0
    n_rows = len(x)
    return (
        pl.DataFrame(
            {
                "__x": x.cast(pl.Float64),
                "__y": y.cast(pl.Float64),
                "__f": free.cast(pl.Float64),
                "__xspan": pl.Series([x_span] * n_rows, dtype=pl.Float64),
                "__fspan": pl.Series([f_span] * n_rows, dtype=pl.Float64),
            }
        )
        .lazy()
        .filter(
            pl.col("__x").is_between(x_lo, x_hi),
            pl.col("__f").is_between(free_lo, free_hi),
            pl.col("__y").is_not_null(),
            pl.col("__y").is_not_nan(),
        )
        .with_columns(
            ((pl.col("__x") - x_lo) / pl.col("__xspan") * float(n_buckets))
            .floor()
            .cast(pl.UInt32)
            .alias("bucket"),
            ((pl.col("__f") - free_lo) / pl.col("__fspan") * float(p))
            .floor()
            .cast(pl.UInt32)
            .alias("free_bin"),
        )
        .group_by("bucket", "free_bin")
        .agg(
            pl.col("__y").min().alias("y_min"),
            pl.col("__x").gather(pl.col("__y").arg_min()).first().alias("x_at_ymin"),
            pl.col("__y").max().alias("y_max"),
            pl.col("__x").gather(pl.col("__y").arg_max()).first().alias("x_at_ymax"),
        )
        .sort("free_bin", "bucket")
        .select("bucket", "free_bin", "y_min", "x_at_ymin", "y_max", "x_at_ymax")
        .collect()
        # On zero groups Polars materializes the gather/first aggs as Null
        # dtype; restore the static schema eagerly.
        .cast({"x_at_ymin": pl.Float64, "x_at_ymax": pl.Float64})
    )


def _ref_to_frame(ref: pl.DataFrame, x_col: str = "x") -> pl.DataFrame:
    """Rename/cast the reference output to ``build_cube``'s frame layout."""
    return ref.rename({"bucket": f"__bin__{x_col}"}).with_columns(
        pl.col(f"__bin__{x_col}").cast(pl.Int32), pl.col("free_bin").cast(pl.Int32)
    )


def _line_spec(
    bins: int = 32,
    x_domain: tuple[float, float] = (0.0, 100.0),
    p: int = 64,
    free_domain: tuple[float, float] = (0.0, 100.0),
    extra_dims: tuple = (),
) -> CubeSpec:
    return CubeSpec(
        source_name="s",
        free=FreeAxisSpec(column="free", p=p, domain=free_domain),
        target_dims=(
            *extra_dims,
            TargetDimSpec(column="x", kind="binned", bins=bins, domain=x_domain),
        ),
        measure=MeasureSpec(agg="line_env", value_col="y"),
    )


def _ref_quantize_x(x: float, b: int, lo: float, hi: float, bins: int) -> float:
    """Test-side mirror of the encode→decode x round-trip (contract J):
    ``x_off = rint((x - bucket_lo) / width * 65535)`` clamped to [0, 65535];
    decode ``x = bucket_lo + (x_off / 65535) * width``."""
    width = (hi - lo) / bins
    bucket_lo = lo + b * width
    off = min(65535.0, max(0.0, float(np.rint((x - bucket_lo) / width * 65535.0))))
    return bucket_lo + (off / 65535.0) * width


@pytest.fixture()
def line_df() -> pl.DataFrame:
    # x sweeps [0, 100) with sub-integer jitter; y is a distinct-valued
    # permutation (no y ties — quantized order == exact order, ulp-bound
    # comparisons stay row-stable); free covers [0, 100). Tail rows pin edge
    # cases: domain-max x AND free (both degenerate top bins), out-of-domain
    # x, and null/NaN y (all filtered or binned per the shared arithmetic).
    n = 4096
    x = [float(i % 100) + (i % 7) * 0.01 for i in range(n)]
    y = [((i * 2641) % 4096) * 0.0001 + 0.05 for i in range(n)]
    free = [float((i * 37) % 100) for i in range(n)]
    x += [100.0, 150.0, -3.0, 12.5, 13.5]
    y += [0.9, 1.0, 1.0, None, float("nan")]
    free += [100.0, 50.0, 50.0, 20.0, 20.0]
    return pl.DataFrame({"x": x, "y": y, "free": free})


class TestLineEnvelope:
    # --- validation -----------------------------------------------------------

    def test_line_env_requires_value_col(self):
        with pytest.raises(ValueError):
            MeasureSpec(agg="line_env")

    def test_requires_exactly_one_binned_dim(self, line_df):
        free = FreeAxisSpec(column="free", p=8, domain=(0.0, 100.0))
        measure = MeasureSpec(agg="line_env", value_col="y")
        no_binned = CubeSpec(
            source_name="s",
            free=free,
            target_dims=(TargetDimSpec(column="x", kind="categorical"),),
            measure=measure,
        )
        with pytest.raises(ValueError):
            build_cube(line_df.lazy(), no_binned)
        two_binned = CubeSpec(
            source_name="s",
            free=free,
            target_dims=(
                TargetDimSpec(column="x", kind="binned", bins=4, domain=(0.0, 100.0)),
                TargetDimSpec(column="y", kind="binned", bins=4, domain=(0.0, 1.0)),
            ),
            measure=measure,
        )
        with pytest.raises(ValueError):
            build_cube(line_df.lazy(), two_binned)

    def test_rejects_box2d_free_axis(self, line_df):
        spec = CubeSpec(
            source_name="s",
            free=FreeAxisSpec(
                column="free",
                columns=("free", "x"),
                kind="box2d",
                domains=((0.0, 100.0), (0.0, 100.0)),
            ),
            target_dims=(
                TargetDimSpec(column="x", kind="binned", bins=4, domain=(0.0, 100.0)),
            ),
            measure=MeasureSpec(agg="line_env", value_col="y"),
        )
        with pytest.raises(ValueError):
            build_cube(line_df.lazy(), spec)

    def test_requires_resolved_domains(self, line_df):
        free_unresolved = CubeSpec(
            source_name="s",
            free=FreeAxisSpec(column="free", p=8, domain=None),
            target_dims=(
                TargetDimSpec(column="x", kind="binned", bins=4, domain=(0.0, 100.0)),
            ),
            measure=MeasureSpec(agg="line_env", value_col="y"),
        )
        with pytest.raises(ValueError):
            build_cube(line_df.lazy(), free_unresolved)
        bucket_unresolved = CubeSpec(
            source_name="s",
            free=FreeAxisSpec(column="free", p=8, domain=(0.0, 100.0)),
            target_dims=(
                TargetDimSpec(column="x", kind="binned", bins=4, domain=None),
            ),
            measure=MeasureSpec(agg="line_env", value_col="y"),
        )
        with pytest.raises(ValueError):
            build_cube(line_df.lazy(), bucket_unresolved)

    def test_content_key_line_env_distinct(self):
        spec = _line_spec()
        other = CubeSpec(
            source_name="s",
            free=spec.free,
            target_dims=spec.target_dims,
            measure=MeasureSpec(agg="min", value_col="y"),
        )
        assert cube_content_key(spec) != cube_content_key(other)

    # --- build == pure-Polars reference ---------------------------------------

    def test_build_equals_polars_reference(self, line_df):
        cube = build_cube(line_df.lazy(), _line_spec())
        assert cube.group_cols == ("__bin__x",)
        want = _ref_to_frame(
            _line_env_reference(
                line_df["x"],
                line_df["y"],
                line_df["free"],
                0.0,
                100.0,
                0.0,
                100.0,
                32,
                64,
            )
        ).select("__bin__x", "free_bin", "y_min", "x_at_ymin", "y_max", "x_at_ymax")
        got = cube.frame.sort(["free_bin", "__bin__x"]).select(want.columns)
        assert got.equals(want)
        # Degenerate top bins on both axes are present (x == 100, free == 100).
        assert cube.frame["__bin__x"].max() == 32
        assert cube.frame["free_bin"].max() == 64

    def test_grouped_build_equals_per_group_reference(self, line_df):
        gdf = line_df.with_columns(
            pl.Series("g", [("a", "b", "c")[i % 3] for i in range(line_df.height)])
        )
        spec = _line_spec(extra_dims=(TargetDimSpec(column="g", kind="categorical"),))
        cube = build_cube(gdf.lazy(), spec)
        assert cube.group_cols == ("g", "__bin__x")
        refs = []
        for gval in ("a", "b", "c"):
            part = gdf.filter(pl.col("g") == gval)
            ref = _line_env_reference(
                part["x"], part["y"], part["free"], 0.0, 100.0, 0.0, 100.0, 32, 64
            )
            refs.append(ref.with_columns(pl.lit(gval).alias("g")))
        want = (
            _ref_to_frame(pl.concat(refs))
            .sort(["free_bin", "g", "__bin__x"])
            .select(
                "g", "__bin__x", "free_bin", "y_min", "x_at_ymin", "y_max", "x_at_ymax"
            )
        )
        got = cube.frame.sort(["free_bin", "g", "__bin__x"]).select(want.columns)
        assert got.equals(want)

    def test_temporal_x_bins_physical_and_header_carries_unit(self):
        n = 500
        base = datetime(2022, 1, 1)
        df = pl.DataFrame(
            {
                "t": pl.Series(
                    [base + timedelta(minutes=i) for i in range(n)],
                    dtype=pl.Datetime("ms"),
                ),
                "y": [((i * 311) % n) * 0.5 for i in range(n)],
                "free": [float((i * 37) % 100) for i in range(n)],
            }
        )
        lo, hi = _physical_domain(df, "t")
        spec = CubeSpec(
            source_name="s",
            free=FreeAxisSpec(column="free", p=16, domain=(0.0, 100.0)),
            target_dims=(
                TargetDimSpec(
                    column="t", kind="binned", bins=8, domain=(lo, hi), unit="ms"
                ),
            ),
            measure=MeasureSpec(agg="line_env", value_col="y"),
        )
        cube = build_cube(df.lazy(), spec)
        phys = df["t"].to_physical().cast(pl.Float64)
        want = _ref_to_frame(
            _line_env_reference(phys, df["y"], df["free"], lo, hi, 0.0, 100.0, 8, 16),
            x_col="t",
        ).select("__bin__t", "free_bin", "y_min", "x_at_ymin", "y_max", "x_at_ymax")
        got = cube.frame.sort(["free_bin", "__bin__t"]).select(want.columns)
        assert got.equals(want)
        header = decode_fvcube_header(encode_fvcube(cube, cube_id="t"))
        (dim,) = header["target_dims"]
        assert dim == {
            "name": "t",
            "kind": "binned",
            "bins": 8,
            "domain": [lo, hi],
            "unit": "ms",
        }

    # --- slice_agg combine (quantized) -----------------------------------------

    def test_full_range_slice_matches_exact_envelope(self, line_df):
        """Full-range slice == per-bucket exact argmin/argmax within the
        quantization bounds: ≤1 f32 ulp in y, ≤ bucket_width/65535 in x."""
        cube = build_cube(line_df.lazy(), _line_spec())
        out = cube.slice_agg(0.0, 100.0).sort("__bin__x")
        flt = (
            line_df.with_columns(pl.Series("__span", [100.0] * line_df.height))
            .filter(
                pl.col("x").is_between(0.0, 100.0),
                pl.col("free").is_between(0.0, 100.0),
                pl.col("y").is_not_null(),
                pl.col("y").is_not_nan(),
            )
            .with_columns(
                (pl.col("x") / pl.col("__span") * 32.0)
                .floor()
                .cast(pl.Int32)
                .alias("__bin__x")
            )
        )
        exact = (
            flt.group_by("__bin__x")
            .agg(
                pl.col("y").min().alias("y_min"),
                pl.col("x").gather(pl.col("y").arg_min()).first().alias("x_at_ymin"),
                pl.col("y").max().alias("y_max"),
                pl.col("x").gather(pl.col("y").arg_max()).first().alias("x_at_ymax"),
            )
            .sort("__bin__x")
        )
        assert out["__bin__x"].to_list() == exact["__bin__x"].to_list()
        for col in ("y_min", "y_max"):
            got = np.asarray(out[col].to_list())
            want = np.asarray(exact[col].to_list())
            ulp = np.spacing(np.abs(want).astype(np.float32)).astype(np.float64)
            assert np.all(np.abs(got - want) <= ulp)
        x_tol = (100.0 / 32) / 65535
        for col in ("x_at_ymin", "x_at_ymax"):
            got = np.asarray(out[col].to_list())
            want = np.asarray(exact[col].to_list())
            assert np.all(np.abs(got - want) <= x_tol)

    def test_tie_keeps_earlier_free_bin_row(self):
        # Two cells in the same bucket (free bins 0 and 2) with y exactly
        # equal: the combine keeps the earlier free_bin row for min AND max.
        df = pl.DataFrame({"x": [1.0, 3.0], "y": [5.0, 5.0], "free": [0.0, 60.0]})
        spec = _line_spec(bins=2, x_domain=(0.0, 8.0), p=4, free_domain=(0.0, 100.0))
        cube = build_cube(df.lazy(), spec)
        assert sorted(cube.frame["free_bin"].to_list()) == [0, 2]
        out = cube.slice_agg(0.0, 100.0)
        assert out.height == 1
        row = out.row(0, named=True)
        expected_x = _ref_quantize_x(1.0, 0, 0.0, 8.0, 2)
        assert row["__bin__x"] == 0
        assert row["y_min"] == 5.0 and row["y_max"] == 5.0
        assert row["x_at_ymin"] == expected_x
        assert row["x_at_ymax"] == expected_x

    def test_day_free_axis_full_range_slice(self):
        """unit="day" free axis: integer-day grid bins (degenerate bin P'),
        full-range combine matches the exact per-bucket envelope (y integers
        ⇒ f32-exact; x within bucket_width/65535)."""
        from flexviz.cube import day_grid

        n = 244
        base = date(2020, 1, 1)
        df = pl.DataFrame(
            {
                "d": [base + timedelta(days=(i % 61)) for i in range(n)],
                "x": [float(i % 10) for i in range(n)],
                "y": [float((i * 17) % 244) for i in range(n)],
            }
        )
        lo, hi = _physical_domain(df, "d")
        spec = CubeSpec(
            source_name="s",
            free=FreeAxisSpec(
                column="d", kind="temporal", p=2048, domain=(lo, hi), unit="day"
            ),
            target_dims=(
                TargetDimSpec(column="x", kind="binned", bins=4, domain=(0.0, 10.0)),
            ),
            measure=MeasureSpec(agg="line_env", value_col="y"),
        )
        cube = build_cube(df.lazy(), spec)
        w, p_eff = day_grid(lo, hi, 2048)
        assert (w, p_eff) == (1, 60)
        # Domain-max rows land in the degenerate day bin P'.
        assert cube.frame["free_bin"].max() == p_eff
        out = cube.slice_agg(lo, hi).sort("__bin__x")
        exact = (
            df.with_columns(pl.Series("__span", [10.0] * df.height))
            .with_columns(
                (pl.col("x") / pl.col("__span") * 4.0)
                .floor()
                .cast(pl.Int32)
                .alias("__bin__x")
            )
            .group_by("__bin__x")
            .agg(
                pl.col("y").min().alias("y_min"),
                pl.col("x").gather(pl.col("y").arg_min()).first().alias("x_at_ymin"),
                pl.col("y").max().alias("y_max"),
                pl.col("x").gather(pl.col("y").arg_max()).first().alias("x_at_ymax"),
            )
            .sort("__bin__x")
        )
        assert out["__bin__x"].to_list() == exact["__bin__x"].to_list()
        assert out["y_min"].to_list() == exact["y_min"].to_list()
        assert out["y_max"].to_list() == exact["y_max"].to_list()
        x_tol = (10.0 / 4) / 65535
        for col in ("x_at_ymin", "x_at_ymax"):
            got = np.asarray(out[col].to_list())
            want = np.asarray(exact[col].to_list())
            assert np.all(np.abs(got - want) <= x_tol)


def _line_env_b_reference(
    part: pl.DataFrame, x_lo: float, x_hi: float, n_buckets: int
) -> pl.DataFrame:
    """Approach B: pure-Polars per-bucket envelope (no kernel), for parity."""
    x_span = (x_hi - x_lo) or 1.0
    return (
        part.lazy()
        .filter(
            pl.col("x").is_between(x_lo, x_hi),
            pl.col("y").is_not_null(),
            pl.col("y").is_not_nan(),
        )
        .with_columns(
            (((pl.col("x") - x_lo) / pl.lit(x_span)) * float(n_buckets))
            .floor()
            .cast(pl.UInt32)
            .alias("bucket")
        )
        .group_by("bucket")
        .agg(
            pl.col("y").min().alias("y_min"),
            pl.col("x").gather(pl.col("y").arg_min()).first().alias("x_at_ymin"),
            pl.col("y").max().alias("y_max"),
            pl.col("x").gather(pl.col("y").arg_max()).first().alias("x_at_ymax"),
        )
        .collect()
        .cast({"x_at_ymin": pl.Float64, "x_at_ymax": pl.Float64})
    )


class TestLineEnvelopeCategoricalFreeAxis:
    @pytest.fixture()
    def cdf(self) -> pl.DataFrame:
        n = 3000
        regions = ["east", "north", "south"]
        return pl.DataFrame(
            {
                "region": [regions[i % 3] for i in range(n)],
                "x": [float(i % 100) + (i % 7) * 0.01 for i in range(n)],
                "y": [((i * 2641) % 4096) * 0.0001 + 0.05 for i in range(n)],
            }
        )

    def _spec(self, bins: int = 16) -> CubeSpec:
        return CubeSpec(
            source_name="s",
            free=_free_cat("region"),
            target_dims=(
                TargetDimSpec(
                    column="x", kind="binned", bins=bins, domain=(0.0, 100.0)
                ),
            ),
            measure=MeasureSpec(agg="line_env", value_col="y"),
        )

    def test_build_has_free_key_cols(self, cdf):
        cube = build_cube(cdf.lazy(), self._spec())
        assert cube.free_key_cols == ("__free__region",)
        assert cube.group_cols == ("__bin__x",)

    def test_single_category_slice_matches_direct_envelope(self, cdf):
        bins = 16
        cube = build_cube(cdf.lazy(), self._spec(bins))
        got = cube.slice_agg_keys([("north",)]).sort("__bin__x")
        sub = cdf.filter(pl.col("region") == "north")
        ref = _line_env_reference(
            sub["x"],
            sub["y"],
            pl.Series([0.0] * sub.height),
            0.0,
            100.0,
            0.0,
            1.0,
            bins,
            1,
        )
        # quantize the reference x exactly as the codec/client do (contract J)
        exp = {
            int(b): (
                ym,
                _ref_quantize_x(xn, int(b), 0.0, 100.0, bins),
                yx,
                _ref_quantize_x(xx, int(b), 0.0, 100.0, bins),
            )
            for b, _fb, ym, xn, yx, xx in ref.iter_rows()
        }
        for b, ym, xn, yx, xx in got.select(
            "__bin__x", "y_min", "x_at_ymin", "y_max", "x_at_ymax"
        ).iter_rows():
            e = exp[int(b)]
            assert ym == pytest.approx(e[0], abs=1e-6)
            assert yx == pytest.approx(e[2], abs=1e-6)

    def test_a_equals_b_per_category_bucket(self, cdf):
        # Approach A (kernel, degenerate free) == approach B (pure Polars) on the
        # exact f64 y extrema, per (category, bucket).
        bins = 16
        cube = build_cube(cdf.lazy(), self._spec(bins))
        for region in ("east", "north", "south"):
            got = cube.slice_agg_keys([(region,)]).sort("__bin__x")
            sub = cdf.filter(pl.col("region") == region)
            b = _line_env_b_reference(sub, 0.0, 100.0, bins).sort("bucket")
            bmap = {
                int(r[0]): (r[1], r[3]) for r in b.iter_rows()
            }  # bucket -> (ymin, ymax)
            for bn, ym, _xn, yx, _xx in got.select(
                "__bin__x", "y_min", "x_at_ymin", "y_max", "x_at_ymax"
            ).iter_rows():
                assert ym == pytest.approx(bmap[int(bn)][0], abs=1e-6)
                assert yx == pytest.approx(bmap[int(bn)][1], abs=1e-6)


class TestLineEnvelopeCodec:
    def _combine_from_blob(
        self, blob: bytes, cube: CubeResult, lo: float, hi: float
    ) -> dict:
        """Quantized line_env combine over the raw decoded buffers (the JS
        client reference semantics): per target cell, min/max over the f32 y
        values; NaN skipped; strict </> keeps the earlier free_bin row (rows
        are sorted by free_bin); x dequantized from its u16 bucket offset."""
        header = decode_fvcube_header(blob)
        assert header["measure"]["agg"] == "line_env"
        dims = header["target_dims"]
        bucket_dim = next(d for d in dims if d["kind"] == "binned")
        d_lo, d_hi = bucket_dim["domain"]
        width = (d_hi - d_lo) / bucket_dim["bins"]
        n_t = len(dims)
        tgt_names = [c["name"] for c in header["columns"][1 : 1 + n_t]]
        bucket_idx = tgt_names.index(f"__bin__{bucket_dim['name']}")
        tgt_cols = [_read_col(blob, header, n) for n in tgt_names]
        free_bin = _read_col(blob, header, "free_bin")
        y_min = _read_col(blob, header, "y_min")
        y_max = _read_col(blob, header, "y_max")
        x_argmin = _read_col(blob, header, "x_argmin")
        x_argmax = _read_col(blob, header, "x_argmax")
        lo_bin, hi_bin = cube._snap(lo, hi)
        out: dict = {}
        for i, fb in enumerate(free_bin):
            if not (lo_bin <= fb <= hi_bin):
                continue
            key = tuple(col[i] for col in tgt_cols)
            bucket_lo = d_lo + tgt_cols[bucket_idx][i] * width
            xmin = bucket_lo + (x_argmin[i] / 65535.0) * width
            xmax = bucket_lo + (x_argmax[i] / 65535.0) * width
            cur = out.get(key)
            if cur is None:
                out[key] = [y_min[i], xmin, y_max[i], xmax]
                continue
            if not math.isnan(y_min[i]) and y_min[i] < cur[0]:
                cur[0], cur[1] = y_min[i], xmin
            if not math.isnan(y_max[i]) and y_max[i] > cur[2]:
                cur[2], cur[3] = y_max[i], xmax
        return {k: tuple(v) for k, v in out.items()}

    def test_subrange_reslice_equals_slice_agg(self, line_df):
        """Quantize-then-combine equivalence: the Python combine over the raw
        f32/u16 buffers equals slice_agg EXACTLY (both sides quantize
        identically)."""
        cube = build_cube(line_df.lazy(), _line_spec())
        blob = encode_fvcube(cube, cube_id="le0")
        got = self._combine_from_blob(blob, cube, 25.0, 75.0)
        ref = cube.slice_agg(25.0, 75.0)
        ref_map = {
            (b,): (ymin, xmin, ymax, xmax)
            for b, ymin, xmin, ymax, xmax in ref.select(
                "__bin__x", "y_min", "x_at_ymin", "y_max", "x_at_ymax"
            ).iter_rows()
        }
        assert got == ref_map

    def test_codec_header_and_dtypes(self, line_df):
        cube = build_cube(line_df.lazy(), _line_spec())
        blob = encode_fvcube(cube, cube_id="le1")
        header = decode_fvcube_header(blob)
        assert header["v"] == 1  # codec version stays 1
        assert header["rows"] == cube.n_cells
        assert header["measure"] == {"agg": "line_env", "value_col": "y"}
        # The line_env target dim block carries the bucket domain and count.
        (dim,) = header["target_dims"]
        assert dim == {
            "name": "x",
            "kind": "binned",
            "bins": 32,
            "domain": [0.0, 100.0],
        }
        assert [(c["name"], c["dtype"]) for c in header["columns"]] == [
            ("free_bin", "u32"),
            ("__bin__x", "u32"),
            ("y_min", "f32"),
            ("y_max", "f32"),
            ("x_argmin", "u16"),
            ("x_argmax", "u16"),
        ]

    def test_y_is_f32_rounded_at_encode(self, line_df):
        cube = build_cube(line_df.lazy(), _line_spec())
        blob = encode_fvcube(cube, cube_id="le2")
        header = decode_fvcube_header(blob)
        frame = cube.frame.sort(["free_bin", "__bin__x"])
        for col in ("y_min", "y_max"):
            got = _read_col(blob, header, col)
            want = [float(np.float32(v)) for v in frame[col].to_list()]
            assert got == want

    def test_x_offsets_quantized_and_clamped(self, line_df):
        cube = build_cube(line_df.lazy(), _line_spec())
        blob = encode_fvcube(cube, cube_id="le3")
        header = decode_fvcube_header(blob)
        frame = cube.frame.sort(["free_bin", "__bin__x"])
        buckets = frame["__bin__x"].to_list()
        for off_col, x_col in (("x_argmin", "x_at_ymin"), ("x_argmax", "x_at_ymax")):
            offs = _read_col(blob, header, off_col)
            assert all(0 <= o <= 65535 for o in offs)
            width = 100.0 / 32
            for o, b, x in zip(offs, buckets, frame[x_col].to_list()):
                want = min(
                    65535.0,
                    max(0.0, float(np.rint((x - (b * width)) / width * 65535.0))),
                )
                assert o == want
        # The degenerate top bucket (x == domain max) packs to offset 0.
        top_rows = [i for i, b in enumerate(buckets) if b == 32]
        assert top_rows
        offs = _read_col(blob, header, "x_argmin")
        assert all(offs[i] == 0 for i in top_rows)

    def test_alignment_mixed_u16_f32_u32(self):
        # 3 cells → the u16 buffers are 6 bytes, forcing inter-buffer padding;
        # every offset must stay 8-byte aligned with no padding inside buffers.
        df = pl.DataFrame(
            {
                "x": [0.0, 1.0, 2.5, 3.0],
                "y": [5.0, 1.0, 7.0, 3.0],
                "free": [0.0, 1.0, 0.5, 3.0],
            }
        )
        spec = _line_spec(bins=2, x_domain=(0.0, 4.0), p=2, free_domain=(0.0, 4.0))
        cube = build_cube(df.lazy(), spec)
        assert cube.n_cells == 3
        blob = encode_fvcube(cube, "k")
        header = decode_fvcube_header(blob)
        assert _buffer_section_start(blob) % 8 == 0
        sizes = {"u32": 4, "f32": 4, "u16": 2}
        for col in header["columns"]:
            assert col["offset"] % 8 == 0
            assert col["byte_len"] == sizes[col["dtype"]] * header["rows"]

    def test_deterministic_bytes_line_env(self, line_df):
        cube = build_cube(line_df.lazy(), _line_spec())
        rebuilt = build_cube(line_df.lazy(), _line_spec())
        assert encode_fvcube(cube, "k") == encode_fvcube(rebuilt, "k")

    def test_codec_golden_bytes(self):
        """Exact golden blob for a tiny fixed line_env cube — pins the header
        JSON, the buffer layout, the f32/u16 dtypes, AND the quantization
        rounding (np.rint half-to-even: 32767.5 → 32768)."""
        df = pl.DataFrame(
            {
                "x": [0.0, 1.0, 2.0, 3.0],
                "y": [5.0, 1.0, 7.0, 3.0],
                "free": [0.0, 1.0, 2.0, 3.0],
            }
        )
        spec = _line_spec(bins=2, x_domain=(0.0, 4.0), p=2, free_domain=(0.0, 4.0))
        cube = build_cube(df.lazy(), spec)
        blob = encode_fvcube(cube, cube_id="line01")
        header = (
            b'{"v":1,"cube_id":"line01","rows":2,"sorted_by":"free_bin",'
            b'"free":{"kind":"continuous","p":2,"domain":[0.0,4.0]},'
            b'"target_dims":[{"name":"x","kind":"binned","bins":2,'
            b'"domain":[0.0,4.0]}],'
            b'"measure":{"agg":"line_env","value_col":"y"},'
            b'"columns":['
            b'{"name":"free_bin","dtype":"u32","offset":0,"byte_len":8},'
            b'{"name":"__bin__x","dtype":"u32","offset":8,"byte_len":8},'
            b'{"name":"y_min","dtype":"f32","offset":16,"byte_len":8},'
            b'{"name":"y_max","dtype":"f32","offset":24,"byte_len":8},'
            b'{"name":"x_argmin","dtype":"u16","offset":32,"byte_len":4},'
            b'{"name":"x_argmax","dtype":"u16","offset":40,"byte_len":4}]}'
        )
        header += b" " * (-(12 + len(header)) % 8)
        section = (
            struct.pack("<2I", 0, 1)  # free_bin
            + struct.pack("<2I", 0, 1)  # __bin__x
            + struct.pack("<2f", 1.0, 3.0)  # y_min (f32)
            + struct.pack("<2f", 5.0, 7.0)  # y_max (f32)
            + struct.pack("<2H", 32768, 32768)  # x_argmin (u16)
            + b"\x00" * 4  # pad to 8
            + struct.pack("<2H", 0, 0)  # x_argmax (u16)
        )
        expected = b"FVCUBE" + struct.pack("<BBI", 1, 0, len(header)) + header + section
        assert blob == expected


# ---------------------------------------------------------------------------
# Correlation cube (contract I — corr target, empty target dims, per-pair
# mean-centered partials)
# ---------------------------------------------------------------------------


def _corr_spec(
    columns: tuple[str, ...] = ("x", "y", "z"),
    p: int = 64,
    domain: tuple[float, float] = (0.0, 100.0),
) -> CubeSpec:
    return CubeSpec(
        source_name="s",
        free=FreeAxisSpec(column="free", p=p, domain=domain),
        target_dims=(),
        measure=MeasureSpec(agg="corr", columns=columns),
    )


@pytest.fixture()
def corr_df() -> pl.DataFrame:
    rng = np.random.default_rng(7)
    n = 3_000
    x = rng.normal(5.0, 2.0, n)
    y = rng.normal(-3.0, 4.0, n) + 0.3 * x
    z = rng.normal(0.0, 1.0, n) - 0.5 * x
    # free covers [0, 100); a few tail rows pin the domain-max degenerate bin.
    free = [float((i * 37) % 100) for i in range(n)]
    return pl.DataFrame({"x": x, "y": y, "z": z, "free": free})


def _pl_corr(df: pl.DataFrame, a: str, b: str) -> float:
    return df.select(pl.corr(a, b, method="pearson")).item()


def _cube_r(per_pair: pl.DataFrame, i: int, j: int) -> float:
    return per_pair.filter((pl.col("i") == i) & (pl.col("j") == j))["r"][0]


class TestCorrMeasureValidation:
    def test_corr_requires_two_columns(self):
        with pytest.raises(ValueError):
            MeasureSpec(agg="corr", columns=("x",))
        with pytest.raises(ValueError):
            MeasureSpec(agg="corr")

    def test_corr_forbids_value_col(self):
        with pytest.raises(ValueError):
            MeasureSpec(agg="corr", value_col="x", columns=("x", "y"))

    def test_non_corr_forbids_columns(self):
        with pytest.raises(ValueError):
            MeasureSpec(agg="sum", value_col="v", columns=("x", "y"))
        with pytest.raises(ValueError):
            MeasureSpec(agg="count", columns=("x", "y"))

    def test_build_rejects_box2d_free_axis(self, corr_df):
        spec = CubeSpec(
            source_name="s",
            free=FreeAxisSpec(
                column="free",
                columns=("free", "x"),
                kind="box2d",
                domains=((0.0, 100.0), (0.0, 50.0)),
            ),
            target_dims=(),
            measure=MeasureSpec(agg="corr", columns=("x", "y")),
        )
        with pytest.raises(ValueError):
            build_cube(corr_df.lazy(), spec)

    def test_build_requires_resolved_domain(self, corr_df):
        spec = CubeSpec(
            source_name="s",
            free=FreeAxisSpec(column="free", p=64, domain=None),
            target_dims=(),
            measure=MeasureSpec(agg="corr", columns=("x", "y")),
        )
        with pytest.raises(ValueError):
            build_cube(corr_df.lazy(), spec)

    def test_build_rejects_target_dims(self, corr_df):
        spec = CubeSpec(
            source_name="s",
            free=FreeAxisSpec(column="free", p=64, domain=(0.0, 100.0)),
            target_dims=(TargetDimSpec(column="x", kind="categorical"),),
            measure=MeasureSpec(agg="corr", columns=("x", "y")),
        )
        with pytest.raises(ValueError):
            build_cube(corr_df.lazy(), spec)


class TestCorrCategoricalFreeAxis:
    @pytest.fixture()
    def cdf(self) -> pl.DataFrame:
        n = 1500
        regions = ["east", "north", "south"]
        return pl.DataFrame(
            {
                "region": [regions[i % 3] for i in range(n)],
                "x": [float((i * 13) % 50) for i in range(n)],
                "y": [
                    float((i * 7) % 50) - float((i * 13) % 50) * 0.4 for i in range(n)
                ],
                "z": [float((i * 5) % 30) for i in range(n)],
            }
        )

    def _spec(self) -> CubeSpec:
        return CubeSpec(
            source_name="s",
            free=_free_cat("region"),
            target_dims=(),
            measure=MeasureSpec(agg="corr", columns=("x", "y", "z")),
        )

    PAIRS = [("x", "y", 0, 1), ("x", "z", 0, 2), ("y", "z", 1, 2)]

    def test_build_has_free_key_cols_and_no_target_dims(self, cdf):
        cube = build_cube(cdf.lazy(), self._spec())
        assert cube.free_key_cols == ("__free__region",)
        assert cube.group_cols == ()

    def test_single_category_slice_equals_filtered_pl_corr(self, cdf):
        cube = build_cube(cdf.lazy(), self._spec())
        per = cube.slice_agg_keys([("north",)])
        sub = cdf.filter(pl.col("region") == "north")
        for a, b, i, j in self.PAIRS:
            assert _cube_r(per, i, j) == pytest.approx(_pl_corr(sub, a, b), abs=1e-9)

    def test_multi_category_or_slice_equals_filtered_pl_corr(self, cdf):
        cube = build_cube(cdf.lazy(), self._spec())
        per = cube.slice_agg_keys([("north",), ("south",)])
        sub = cdf.filter(pl.col("region").is_in(["north", "south"]))
        for a, b, i, j in self.PAIRS:
            assert _cube_r(per, i, j) == pytest.approx(_pl_corr(sub, a, b), abs=1e-9)


class TestCorrSliceParity:
    PAIRS = [("x", "y", 0, 1), ("x", "z", 0, 2), ("y", "z", 1, 2)]

    def test_full_range_slice_equals_pl_corr(self, corr_df):
        cube = build_cube(corr_df.lazy(), _corr_spec())
        per = cube.slice_agg(0.0, 100.0)
        assert per["i"].to_list() == [0, 0, 1]
        assert per["j"].to_list() == [1, 2, 2]
        for a, b, i, j in self.PAIRS:
            assert _cube_r(per, i, j) == pytest.approx(
                _pl_corr(corr_df, a, b), abs=1e-9
            )

    def test_subrange_slice_equals_filtered_pl_corr(self, corr_df):
        cube = build_cube(corr_df.lazy(), _corr_spec())
        per = cube.slice_agg(25.0, 75.0)
        lo_bin, hi_bin = cube._snap(25.0, 75.0)
        span = 100.0
        edge_lo = lo_bin * span / 64
        edge_hi = (hi_bin + 1) * span / 64
        sub = corr_df.filter(pl.col("free").is_between(edge_lo, edge_hi, closed="left"))
        # Cube bins are half-open per the floor arithmetic; clamp the reference
        # to the snapped bin edges so both sides see the same rows.
        for a, b, i, j in self.PAIRS:
            assert _cube_r(per, i, j) == pytest.approx(_pl_corr(sub, a, b), abs=1e-9)

    def test_matrix_mirrors_to_update(self, corr_df):
        from flexviz.trace.corr_heatmap import CorrHeatmap

        cols = ["x", "y", "z"]
        for absolute in (False, True):
            for triangular in (False, True):
                trace = CorrHeatmap(
                    columns=cols, absolute=absolute, triangular=triangular
                )
                trace.uid = "u"
                legacy_df = corr_df.select(trace.get_aggregation_spec({}).expr)
                legacy = trace._to_update(legacy_df).updates
                cube = build_cube(corr_df.lazy(), _corr_spec(p=2048))
                got = cube.corr_matrix(
                    0.0, 100.0, absolute=absolute, triangular=triangular
                )
                assert got["x"] == legacy["x"]
                assert got["y"] == legacy["y"]
                for gr, lr in zip(got["z"], legacy["z"]):
                    for gv, lv in zip(gr, lr):
                        if gv is None or lv is None:
                            assert gv is None and lv is None, (gv, lv)
                        else:
                            assert gv == pytest.approx(lv, abs=1e-9)


class TestCorrMeanOffsetCancellation:
    """The reason for mean-centering (spec §4). A large additive offset on a
    column (mean ≫ std) makes a raw one-pass ``Σxy − Σx·Σy/n`` catastrophically
    cancel. The centered partials are shift-invariant: with an offset whose
    quantization is negligible (1e8, ULP ~1e-8 ≪ the data's std of 1), the
    centered cube ``r`` equals ``pl.corr`` on the UN-offset data — while the
    uncentered one-pass collapses to ~0.0 (asserted, so a regression dropping
    the centering would fail this test)."""

    OFFSET = 1e8

    def _data(self):
        rng = np.random.default_rng(0)
        n = 2_000
        x = rng.normal(0.0, 1.0, n)
        y = 0.5 * x + rng.normal(0.0, 1.0, n)
        free = [float((i * 37) % 100) for i in range(n)]
        return x, y, free

    def test_centered_matches_pl_corr_on_unoffset_data(self):
        x, y, free = self._data()
        df_off = pl.DataFrame({"x": x + self.OFFSET, "y": y, "free": free})
        cube = build_cube(df_off.lazy(), _corr_spec(columns=("x", "y"), p=2048))
        r_cube = _cube_r(cube.slice_agg(0.0, 100.0), 0, 1)
        # Shift-invariance: correlation of the offset column equals that of the
        # original (the offset's quantization at 1e8 is ~1e-8, negligible).
        r_ref = pl.DataFrame({"x": x, "y": y}).select(pl.corr("x", "y")).item()
        assert r_cube == pytest.approx(r_ref, abs=1e-9)

    def test_uncentered_one_pass_visibly_diverges(self):
        x, y, _free = self._data()
        xo = (x + self.OFFSET).astype(np.float64)
        yq = y.astype(np.float64)
        n = len(xo)
        # An UNCENTERED one-pass (sxy/sxx/syy over the RAW offset values) — what
        # the cube would compute if it skipped mean-centering. It collapses to
        # ~0.0 through catastrophic cancellation at the 1e8 magnitude.
        sx, sy = xo.sum(), yq.sum()
        sxy = (xo * yq).sum()
        sxx = (xo * xo).sum()
        syy = (yq * yq).sum()
        m_x, m_y = sx / n, sy / n
        cov = sxy / n - m_x * m_y
        var_x = sxx / n - m_x * m_x
        var_y = syy / n - m_y * m_y
        r_unc = cov / math.sqrt(var_x * var_y) if var_x > 0 and var_y > 0 else 0.0
        r_ref = pl.DataFrame({"x": x, "y": y}).select(pl.corr("x", "y")).item()
        # Wide divergence — proof the test data would catch a regression that
        # dropped the centering.
        assert abs(r_unc - r_ref) > 0.1


class TestCorrNullSemantics:
    def test_masked_pair_build_equals_pl_corr(self):
        # Nulls scattered in both columns: the pairwise mask drops rows where
        # EITHER column is null — pairwise-complete deletion, == pl.corr.
        rng = np.random.default_rng(3)
        n = 1_500
        x = rng.normal(2.0, 1.0, n).tolist()
        y = (0.4 * rng.normal(2.0, 1.0, n) + rng.normal(0.0, 1.0, n)).tolist()
        for i in range(0, n, 11):
            x[i] = None
        for i in range(0, n, 13):
            y[i] = None
        free = [float((i * 37) % 100) for i in range(n)]
        df = pl.DataFrame({"x": x, "y": y, "free": free})
        cube = build_cube(df.lazy(), _corr_spec(columns=("x", "y"), p=2048))
        r_cube = _cube_r(cube.slice_agg(0.0, 100.0), 0, 1)
        assert r_cube == pytest.approx(_pl_corr(df, "x", "y"), abs=1e-9)

    def test_zero_variance_pair_finalizes_to_zero(self):
        # A constant column has zero variance → r is non-finite → 0.0 (mirrors
        # _corr_expr._pack's None → 0.0).
        n = 500
        df = pl.DataFrame(
            {
                "x": [3.0] * n,
                "y": [float(i) for i in range(n)],
                "free": [float((i * 37) % 100) for i in range(n)],
            }
        )
        cube = build_cube(df.lazy(), _corr_spec(columns=("x", "y"), p=2048))
        assert _cube_r(cube.slice_agg(0.0, 100.0), 0, 1) == 0.0


class TestCorrCodec:
    def test_round_trip_header_and_buffers(self, corr_df):
        cube = build_cube(corr_df.lazy(), _corr_spec())
        blob = encode_fvcube(cube, cube_id="corr01")
        header = decode_fvcube_header(blob)
        assert header["v"] == 1
        assert header["rows"] == cube.n_cells
        assert header["target_dims"] == []
        assert header["measure"]["agg"] == "corr"
        assert header["measure"]["value_col"] is None
        assert header["measure"]["columns"] == ["x", "y", "z"]
        assert header["measure"]["pairs"] == [[0, 1], [0, 2], [1, 2]]
        assert set(header["measure"]["means"]) == {"x", "y", "z"}
        # Dynamic per-pair buffers: free_bin (u32) then 6 stats per pair.
        names = [(c["name"], c["dtype"]) for c in header["columns"]]
        assert names[0] == ("free_bin", "u32")
        expected = [("free_bin", "u32")]
        for i, j in ((0, 1), (0, 2), (1, 2)):
            expected.append((f"__corr__{i}_{j}__n", "u32"))
            for stat in ("sx", "sy", "sxy", "sxx", "syy"):
                expected.append((f"__corr__{i}_{j}__{stat}", "f64"))
        assert names == expected

    def test_buffers_8_byte_aligned(self, corr_df):
        cube = build_cube(corr_df.lazy(), _corr_spec())
        blob = encode_fvcube(cube, "k")
        header = decode_fvcube_header(blob)
        assert _buffer_section_start(blob) % 8 == 0
        for col in header["columns"]:
            assert col["offset"] % 8 == 0

    def test_deterministic_bytes(self, corr_df):
        a = build_cube(corr_df.lazy(), _corr_spec())
        b = build_cube(corr_df.lazy(), _corr_spec())
        assert encode_fvcube(a, "k") == encode_fvcube(b, "k")

    def test_stable_center_is_reproducible_and_close(self):
        """The corr centering constant must be stable, not exact.

        The means pre-pass streams, and the streaming engine's parallel f64
        reduction combines partials in completion order — so the raw mean is not
        reproducible across builds, and the cube's bytes would follow it.
        Centering is shift-invariant, so the constant only has to be near the
        mean; truncating the significand makes it deterministic.
        """
        from flexviz.cube import _stable_center

        for v in (0.0, 1.0, -3.25, 1e-9, -1e12, 5.234567890123456):
            c = _stable_center(v)
            assert c == _stable_center(c), f"not idempotent at {v}"
            if v != 0.0:
                assert abs(c - v) <= abs(v) * 1e-9, f"drifted too far at {v}"

        # Values a hair apart must not straddle a step for the centering to be
        # useful; and non-finite input must not propagate into the partials.
        assert _stable_center(float("nan")) == 0.0
        assert _stable_center(float("inf")) == 0.0

    def test_reslice_from_blob_equals_slice_agg(self, corr_df):
        """Decode the raw buffers, Σ partials over the snapped free bins, and
        finalize — the client-equivalent reslice must equal slice_agg r."""
        cube = build_cube(corr_df.lazy(), _corr_spec())
        blob = encode_fvcube(cube, "k")
        header = decode_fvcube_header(blob)
        free_bin = _read_col(blob, header, "free_bin")
        lo_bin, hi_bin = cube._snap(20.0, 80.0)
        pairs = header["measure"]["pairs"]
        per_pair = {}
        for i, j in pairs:
            cols = {
                stat: _read_col(blob, header, f"__corr__{i}_{j}__{stat}")
                for stat in ("n", "sx", "sy", "sxy", "sxx", "syy")
            }
            acc = {k: 0.0 for k in cols}
            for r, fb in enumerate(free_bin):
                if lo_bin <= fb <= hi_bin:
                    for k in cols:
                        acc[k] += cols[k][r]
            n = acc["n"]
            if n < 2:
                per_pair[(i, j)] = 0.0
                continue
            m_x, m_y = acc["sx"] / n, acc["sy"] / n
            cov = acc["sxy"] / n - m_x * m_y
            var_x = acc["sxx"] / n - m_x * m_x
            var_y = acc["syy"] / n - m_y * m_y
            r = cov / math.sqrt(var_x * var_y) if var_x > 0 and var_y > 0 else 0.0
            per_pair[(i, j)] = r
        ref = cube.slice_agg(20.0, 80.0)
        for i, j in pairs:
            assert per_pair[(i, j)] == pytest.approx(_cube_r(ref, i, j), abs=1e-9)


class TestCorrContentKey:
    def test_key_differs_by_columns(self):
        a = cube_content_key(_corr_spec(columns=("x", "y", "z")))
        b = cube_content_key(_corr_spec(columns=("x", "y")))
        assert a != b

    def test_key_differs_by_column_order(self):
        a = cube_content_key(_corr_spec(columns=("x", "y", "z")))
        b = cube_content_key(_corr_spec(columns=("z", "y", "x")))
        assert a != b

    def test_key_identical_across_absolute_triangular(self):
        # absolute/triangular are display params, NOT in MeasureSpec, so two
        # corr cubes built for heatmaps differing only there share one key.
        a = cube_content_key(_corr_spec())
        b = cube_content_key(_corr_spec())
        assert a == b

    def test_non_corr_key_byte_identical_to_before(self):
        # The corr extension must not perturb non-corr measure blocks.
        import json

        from flexviz.cube import _measure_content_block

        assert _measure_content_block(MeasureSpec(agg="count")) == {
            "a": "count",
            "v": None,
        }
        assert _measure_content_block(MeasureSpec(agg="sum", value_col="v")) == {
            "a": "sum",
            "v": "v",
        }
        # And the corr block carries cc (column order).
        block = _measure_content_block(MeasureSpec(agg="corr", columns=("b", "a")))
        assert block == {"a": "corr", "v": None, "cc": ["b", "a"]}
        json.dumps(block)  # serializable


# ---------------------------------------------------------------------------
# 2-D box free axis (contract H — hist2d source, composite CSR free bin)
# ---------------------------------------------------------------------------

from flexviz.cube import box2d_composite_stride  # noqa: E402

_BOX2D_P = 128
_BOX2D_S = _BOX2D_P + 1


def _box2d_spec(
    agg: str = "count",
    value_col: str | None = None,
    x_domain: tuple[float, float] = (0.0, 100.0),
    y_domain: tuple[float, float] = (0.0, 50.0),
    p: int = _BOX2D_P,
) -> CubeSpec:
    return CubeSpec(
        source_name="s",
        free=FreeAxisSpec(
            column="x",
            kind="box2d",
            p=p,
            columns=("x", "y"),
            domains=(x_domain, y_domain),
        ),
        target_dims=(),
        measure=MeasureSpec(agg=agg, value_col=value_col),
    )


@pytest.fixture()
def box2d_df() -> pl.DataFrame:
    # x ∈ [0,100), y ∈ [0,50), a value column tracking x; some rows pin each
    # axis's degenerate top bin (x==100 / y==50).
    n = 5_000
    x = [float((i * 37) % 100) for i in range(n)]
    y = [float((i * 53) % 50) for i in range(n)]
    val = [float((i * 11) % 200) for i in range(n)]
    # A few rows exactly on each domain max → the degenerate top bins.
    x += [100.0, 100.0, 50.0]
    y += [25.0, 50.0, 50.0]
    val += [1.0, 2.0, 3.0]
    return pl.DataFrame({"x": x, "y": y, "val": val})


def _direct_box2d_agg(
    df: pl.DataFrame, agg: str, x_lo: float, x_hi: float, y_lo: float, y_hi: float
) -> float | None:
    sub = df.filter(
        (pl.col("x") >= x_lo)
        & (pl.col("x") < x_hi)
        & (pl.col("y") >= y_lo)
        & (pl.col("y") < y_hi)
    )
    return sub.select(_LEGACY_AGG[agg]("val").alias("v")).item()


class TestBox2dValidation:
    def test_box2d_requires_two_columns(self):
        with pytest.raises(ValueError):
            FreeAxisSpec(
                column="x", kind="box2d", columns=("x",), domains=((0, 1), (0, 1))
            )

    def test_box2d_columns0_must_equal_column(self):
        with pytest.raises(ValueError):
            FreeAxisSpec(
                column="x", kind="box2d", columns=("y", "x"), domains=((0, 1), (0, 1))
            )

    def test_box2d_rejects_single_domain(self):
        with pytest.raises(ValueError):
            FreeAxisSpec(
                column="x",
                kind="box2d",
                columns=("x", "y"),
                domain=(0.0, 1.0),
                domains=((0, 1), (0, 1)),
            )

    def test_non_box2d_rejects_domains(self):
        with pytest.raises(ValueError):
            FreeAxisSpec(column="x", kind="continuous", domains=((0, 1), (0, 1)))

    def test_categorical_rejects_domains(self):
        with pytest.raises(ValueError):
            FreeAxisSpec(
                column="x", kind="categorical", columns=("x",), domains=((0, 1), (0, 1))
            )

    def test_box2d_composite_stride(self):
        assert box2d_composite_stride(_BOX2D_P) == 129
        assert box2d_composite_stride(2048) == 2049


class TestBox2dBuildAndSlice:
    @pytest.mark.parametrize("agg", ["count", "mean"])
    def test_full_range_slice_equals_direct(self, box2d_df, agg):
        spec = _box2d_spec(agg, value_col=None if agg == "count" else "val")
        cube = build_cube(box2d_df.lazy(), spec)
        sliced = cube.slice_agg_box2d(0.0, 100.0, 0.0, 50.0)
        # No target dims → a single global cell. Direct full-domain agg over
        # the same closed-left rectangle covering every bin.
        got = sliced["value"].item() if sliced.height else None
        want = _direct_box2d_agg(box2d_df, agg, 0.0, 100.0 + 1e-9, 0.0, 50.0 + 1e-9)
        if agg == "count":
            assert got == want
        else:
            assert got == pytest.approx(want, rel=1e-9)

    @pytest.mark.parametrize("agg", ["count", "mean"])
    def test_subrange_slice_equals_filtered_direct_unzoomed(self, box2d_df, agg):
        spec = _box2d_spec(agg, value_col=None if agg == "count" else "val")
        cube = build_cube(box2d_df.lazy(), spec)
        (lx, hx), (ly, hy) = cube._snap_box2d(20.0, 70.0, 10.0, 40.0)
        # Recover the snapped closed-left rectangle edges.
        x_lo = 0.0 + lx / _BOX2D_P * 100.0
        x_hi = 0.0 + (hx + 1) / _BOX2D_P * 100.0
        y_lo = 0.0 + ly / _BOX2D_P * 50.0
        y_hi = 0.0 + (hy + 1) / _BOX2D_P * 50.0
        sliced = cube.slice_agg_box2d(20.0, 70.0, 10.0, 40.0)
        got = sliced["value"].item() if sliced.height else None
        want = _direct_box2d_agg(box2d_df, agg, x_lo, x_hi, y_lo, y_hi)
        if agg == "count":
            assert got == want
        else:
            assert got == pytest.approx(want, rel=1e-9)

    @pytest.mark.parametrize("agg", ["count", "mean"])
    def test_subrange_slice_equals_filtered_direct_zoomed(self, box2d_df, agg):
        # A zoomed (narrower) domain → finer bins over the sub-region.
        spec = _box2d_spec(
            agg,
            value_col=None if agg == "count" else "val",
            x_domain=(20.0, 80.0),
            y_domain=(10.0, 45.0),
        )
        cube = build_cube(box2d_df.lazy(), spec)
        (lx, hx), (ly, hy) = cube._snap_box2d(30.0, 65.0, 15.0, 35.0)
        x_lo = 20.0 + lx / _BOX2D_P * 60.0
        x_hi = 20.0 + (hx + 1) / _BOX2D_P * 60.0
        y_lo = 10.0 + ly / _BOX2D_P * 35.0
        y_hi = 10.0 + (hy + 1) / _BOX2D_P * 35.0
        sliced = cube.slice_agg_box2d(30.0, 65.0, 15.0, 35.0)
        got = sliced["value"].item() if sliced.height else None
        want = _direct_box2d_agg(box2d_df, agg, x_lo, x_hi, y_lo, y_hi)
        if agg == "count":
            assert got == want
        else:
            assert got == pytest.approx(want, rel=1e-9)

    def test_degenerate_top_bins_each_axis(self, box2d_df):
        spec = _box2d_spec("count")
        cube = build_cube(box2d_df.lazy(), spec)
        # A brush reaching both domain maxima selects the degenerate top bins
        # (bin == P on each axis) — the composite index P*S + P must be present.
        (lx, hx), (ly, hy) = cube._snap_box2d(100.0, 100.0, 50.0, 50.0)
        assert hx == _BOX2D_P and hy == _BOX2D_P
        top_code = _BOX2D_P * _BOX2D_S + _BOX2D_P
        codes = cube.frame["free_bin"].to_list()
        assert top_code in codes  # the (x==100, y==50) row landed in the top cell
        # Slicing the full rectangle conserves the total count.
        total = cube.slice_agg_box2d(0.0, 100.0, 0.0, 50.0)["value"].item()
        assert total == box2d_df.height

    def test_filter_dont_clip_drops_out_of_domain(self):
        df = pl.DataFrame(
            {"x": [-5.0, 5.0, 105.0], "y": [10.0, 10.0, 10.0], "val": [1.0, 2.0, 3.0]}
        )
        spec = _box2d_spec("count")
        cube = build_cube(df.lazy(), spec)
        # Only the in-domain row survives.
        assert cube.slice_agg_box2d(0.0, 100.0, 0.0, 50.0)["value"].item() == 1

    def test_composite_free_bin_is_u32_range(self, box2d_df):
        cube = build_cube(box2d_df.lazy(), _box2d_spec("count"))
        codes = cube.frame["free_bin"].to_list()
        assert all(0 <= c <= _BOX2D_S * _BOX2D_S - 1 for c in codes)


class TestBox2dCodec:
    def _reslice_rect(self, blob: bytes, cube: CubeResult, x: tuple, y: tuple) -> int:
        """Rectangle-accumulate count over the raw decoded free_bin buffer,
        mirroring the client's fvCubeSliceRect."""
        header = decode_fvcube_header(blob)
        free_bin = _read_u32_col(blob, header, "free_bin")
        count = _read_u32_col(blob, header, "count")
        (lx, hx), (ly, hy) = cube._snap_box2d(x[0], x[1], y[0], y[1])
        s = header["free"]["p"] + 1
        codes = set()
        for by in range(ly, hy + 1):
            row = by * s
            for bx in range(lx, hx + 1):
                codes.add(row + bx)
        total = 0
        for fb, n in zip(free_bin, count):
            if fb in codes:
                total += n
        return total

    def test_header_free_block_shape(self, box2d_df):
        cube = build_cube(box2d_df.lazy(), _box2d_spec("count"))
        blob = encode_fvcube(cube, cube_id="b2d0")
        header = decode_fvcube_header(blob)
        assert header["v"] == 1
        assert header["free"] == {
            "kind": "box2d",
            "cols": ["x", "y"],
            "p": 128,
            "domains": [[0.0, 100.0], [0.0, 50.0]],
        }
        # No target dims for a pure source build; the free_bin and count buffers
        # are the only columns.
        assert [(c["name"], c["dtype"]) for c in header["columns"]] == [
            ("free_bin", "u32"),
            ("count", "u32"),
        ]

    def test_reslice_from_blob_matches_slice_agg(self, box2d_df):
        cube = build_cube(box2d_df.lazy(), _box2d_spec("count"))
        blob = encode_fvcube(cube, cube_id="b2d1")
        got = self._reslice_rect(blob, cube, (20.0, 70.0), (10.0, 40.0))
        ref = cube.slice_agg_box2d(20.0, 70.0, 10.0, 40.0)["value"].item()
        assert got == ref

    def test_deterministic_bytes(self, box2d_df):
        a = build_cube(box2d_df.lazy(), _box2d_spec("count"))
        b = build_cube(box2d_df.lazy(), _box2d_spec("count"))
        assert encode_fvcube(a, "k") == encode_fvcube(b, "k")

    def test_buffer_section_8_byte_aligned(self, box2d_df):
        cube = build_cube(box2d_df.lazy(), _box2d_spec("mean", value_col="val"))
        blob = encode_fvcube(cube, "k")
        header = decode_fvcube_header(blob)
        assert _buffer_section_start(blob) % 8 == 0
        for col in header["columns"]:
            assert col["offset"] % 8 == 0

    def test_content_key_differs_by_domains(self):
        a = cube_content_key(_box2d_spec("count"))
        b = cube_content_key(_box2d_spec("count", x_domain=(0.0, 50.0)))
        assert a != b
        # Same descriptor → same key (cache sharing).
        assert cube_content_key(_box2d_spec("count")) == a


# ---------------------------------------------------------------------------
# hist2d TARGET (step 14): bit-equal binning to fixed_hist2d + histnorm parity
# ---------------------------------------------------------------------------


_HIST2D_NB_X = 8
_HIST2D_NB_Y = 6


def _hist2d_target_df() -> pl.DataFrame:
    """A 2-D dataset with values landing exactly on the (lo, hi, n) bin edges
    of BOTH axes, plus a `free` column over [0, 1] and a numeric `z` column.

    The on-edge points exercise the ``fixed_hist2d`` ``+1e-10`` span epsilon —
    if the cube used the 1-D ``fixed_hist`` scale (no span eps) some of these
    points would land in the previous bin and the z-matrices would diverge.
    """
    x_lo, x_hi = 0.0, 80.0
    y_lo, y_hi = -30.0, 30.0
    x_step = (x_hi - x_lo) / _HIST2D_NB_X
    y_step = (y_hi - y_lo) / _HIST2D_NB_Y
    xs: list[float] = []
    ys: list[float] = []
    # Every internal + boundary x edge crossed with every y edge.
    for i in range(_HIST2D_NB_X + 1):
        for j in range(_HIST2D_NB_Y + 1):
            xs.append(x_lo + i * x_step)
            ys.append(y_lo + j * y_step)
    # A spread of interior points too.
    for k in range(200):
        xs.append(x_lo + (k * 0.37) % (x_hi - x_lo))
        ys.append(y_lo + (k * 0.91) % (y_hi - y_lo))
    m = len(xs)
    free = [i / (m - 1) for i in range(m)]
    z = [float((i % 7) + 1) for i in range(m)]
    return pl.DataFrame({"x": xs, "y": ys, "z": z, "free": free})


def _hist2d_target_spec(
    df: pl.DataFrame,
    *,
    histfunc: str | None = None,
    z_col: str | None = None,
) -> CubeSpec:
    x_lo, x_hi = float(df["x"].min()), float(df["x"].max())
    y_lo, y_hi = float(df["y"].min()), float(df["y"].max())
    if z_col is None:
        measure = MeasureSpec(agg="count")
    else:
        measure = MeasureSpec(agg=histfunc, value_col=z_col)
    return CubeSpec(
        source_name="s",
        free=FreeAxisSpec(column="free", kind="continuous", p=64, domain=(0.0, 1.0)),
        target_dims=(
            TargetDimSpec(
                column="x",
                kind="binned",
                bins=_HIST2D_NB_X,
                domain=(x_lo, x_hi),
                bin_variant="hist2d",
            ),
            TargetDimSpec(
                column="y",
                kind="binned",
                bins=_HIST2D_NB_Y,
                domain=(y_lo, y_hi),
                bin_variant="hist2d",
            ),
        ),
        measure=measure,
    )


def _cube_z_matrix(
    cube: CubeResult,
    nb_x: int,
    nb_y: int,
    *,
    is_count: bool,
    histnorm: str | None = None,
) -> list[list]:
    """Build the dense z-matrix from a full-range cube slice, mirroring
    ``Histogram2D._to_update`` (null/zero convention + histnorm + reshape)."""
    sliced = cube.slice_agg(0.0, 1.0)
    by_cell = {
        (row["__bin__x"], row["__bin__y"]): row["value"]
        for row in sliced.iter_rows(named=True)
    }
    x_lo, x_hi = cube.spec.target_dims[0].domain
    y_lo, y_hi = cube.spec.target_dims[1].domain
    z_flat: list = []
    for j in range(nb_y):
        for i in range(nb_x):
            v = by_cell.get((i, j))
            if v is None:
                z_flat.append(None)
            else:
                z_flat.append(float(v))
    # count: an empty (== 0) cell is null; the cube already omits 0-count cells
    # so missing cells are None already — matches _to_update's `None if v == 0`.
    if histnorm is not None:
        from flexviz.trace._hist_helpers import apply_histnorm

        x_step = (x_hi - x_lo) / nb_x
        y_step = (y_hi - y_lo) / nb_y
        z_series = pl.Series("value", z_flat, dtype=pl.Float64)
        z_df = apply_histnorm(
            pl.DataFrame({"value": z_series}),
            "value",
            histnorm,
            x_step * y_step,
        )
        z_flat = z_df["value"].to_list()
    return [z_flat[j * nb_x : (j + 1) * nb_x] for j in range(nb_y)]


def _hist2d_to_update_z(
    df: pl.DataFrame,
    *,
    histfunc: str | None,
    z_col: str | None,
    histnorm: str | None,
) -> list[list]:
    """Run the real ``Histogram2D._to_update`` over the full data range with a
    viewport equal to the raw data min/max (so its centers/scale match the
    cube's resolved full-data domain) and return its z-matrix."""
    from flexviz.trace.hist2d import Histogram2D

    x_lo, x_hi = float(df["x"].min()), float(df["x"].max())
    y_lo, y_hi = float(df["y"].min()), float(df["y"].max())
    trace = Histogram2D(
        x="x",
        y="y",
        x_bins=_HIST2D_NB_X,
        y_bins=_HIST2D_NB_Y,
        z=z_col,
        histfunc=histfunc,
        histnorm=histnorm,
    )
    trace.uid = "u"
    agg = trace.get_aggregation_spec({"x": (x_lo, x_hi), "y": (y_lo, y_hi)}, df.schema)
    out = df.select(agg.expr)
    return trace._to_update(out).updates["z"]


class TestHist2dTargetBinningParity:
    @pytest.mark.parametrize(
        "histfunc, z_col",
        [(None, None), ("mean", "z")],
    )
    def test_cube_z_matches_to_update_full_range(self, histfunc, z_col):
        df = _hist2d_target_df()
        spec = _hist2d_target_spec(df, histfunc=histfunc, z_col=z_col)
        cube = build_cube(df.lazy(), spec)
        got = _cube_z_matrix(cube, _HIST2D_NB_X, _HIST2D_NB_Y, is_count=z_col is None)
        want = _hist2d_to_update_z(df, histfunc=histfunc, z_col=z_col, histnorm=None)
        assert got == want

    def test_bin_variant_reproduces_span_eps(self):
        # A 1-D fixed_hist scale (no span eps) would push the value exactly at
        # the domain max into bin n (clamped to n-1), but the on-edge interior
        # points reveal the off-by-one. Compare hist2d-variant binning vs the
        # fixed_hist2d kernel directly on a single axis.
        from flexviz.cube import _fixed_hist2d_bin_expr

        lo, hi, n = 0.0, 80.0, 8
        step = (hi - lo) / n
        values = [lo + k * step for k in range(n + 1)]  # every edge incl. hi
        s = pl.Series("v", values, dtype=pl.Float64)
        cube_bins = (
            pl.select(_fixed_hist2d_bin_expr(pl.lit(s), lo, hi, n, "b"))
            .to_series()
            .to_list()
        )
        # The 2-D kernel's per-axis bin: floor((v-lo)*nb/(hi-lo+1e-10)+1e-9)
        # clamped to nb-1. Reproduce via fixed_hist2d on (v, v) with a single y.
        kernel = pl.select(
            pl.lit(s).flexviz.fixed_hist2d(
                pl.lit(s),
                pl.lit(lo),
                pl.lit(hi),
                pl.lit(lo),
                pl.lit(hi),
                n,
                1,
            )
        ).to_series()
        # z_flat is row-major over (1 y-bin, n x-bins): index == x bin index.
        z_flat = kernel.struct.field("z_flat").to_list()[0]
        kernel_x_counts = list(z_flat)
        # Each value lands in exactly the bin our expr reports; build the same
        # histogram from cube_bins and compare counts.
        cube_counts = [0] * n
        for b in cube_bins:
            cube_counts[b] += 1
        assert cube_counts == kernel_x_counts


class TestHist2dTargetHistnorm:
    @pytest.mark.parametrize(
        "histnorm",
        ["percent", "probability", "density", "probability density"],
    )
    @pytest.mark.parametrize(
        "histfunc, z_col",
        [(None, None), ("mean", "z")],
    )
    def test_cube_histnorm_matches_to_update(self, histnorm, histfunc, z_col):
        df = _hist2d_target_df()
        spec = _hist2d_target_spec(df, histfunc=histfunc, z_col=z_col)
        cube = build_cube(df.lazy(), spec)
        got = _cube_z_matrix(
            cube,
            _HIST2D_NB_X,
            _HIST2D_NB_Y,
            is_count=z_col is None,
            histnorm=histnorm,
        )
        want = _hist2d_to_update_z(
            df, histfunc=histfunc, z_col=z_col, histnorm=histnorm
        )
        # Float divisions can differ in the last ULP; compare elementwise.
        assert len(got) == len(want)
        for grow, wrow in zip(got, want):
            assert len(grow) == len(wrow)
            for g, w in zip(grow, wrow):
                if g is None or w is None:
                    assert g is None and w is None
                else:
                    assert g == pytest.approx(w, rel=1e-12, abs=1e-12)


class TestHist2dTargetCodec:
    @pytest.mark.parametrize(
        "histfunc, z_col",
        [(None, None), ("mean", "z")],
    )
    def test_round_trip(self, histfunc, z_col):
        df = _hist2d_target_df()
        spec = _hist2d_target_spec(df, histfunc=histfunc, z_col=z_col)
        cube = build_cube(df.lazy(), spec)
        blob = encode_fvcube(cube, "hk")
        header = decode_fvcube_header(blob)
        assert header["v"] == 1
        assert [d["name"] for d in header["target_dims"]] == ["x", "y"]
        assert [d["kind"] for d in header["target_dims"]] == ["binned", "binned"]
        assert header["target_dims"][0]["bins"] == _HIST2D_NB_X
        assert header["target_dims"][1]["bins"] == _HIST2D_NB_Y
        assert header["measure"]["agg"] == ("count" if z_col is None else "mean")

    def test_deterministic_bytes(self):
        df = _hist2d_target_df()
        spec = _hist2d_target_spec(df)
        a = build_cube(df.lazy(), spec)
        b = build_cube(df.lazy(), spec)
        assert encode_fvcube(a, "k") == encode_fvcube(b, "k")


class TestHist2dTargetContentKey:
    def test_hist2d_variant_distinct_from_hist1d(self):
        df = _hist2d_target_df()
        x_lo, x_hi = float(df["x"].min()), float(df["x"].max())
        y_lo, y_hi = float(df["y"].min()), float(df["y"].max())
        free = FreeAxisSpec(column="free", kind="continuous", p=64, domain=(0.0, 1.0))
        dims_2d = (
            TargetDimSpec(
                column="x",
                kind="binned",
                bins=_HIST2D_NB_X,
                domain=(x_lo, x_hi),
                bin_variant="hist2d",
            ),
            TargetDimSpec(
                column="y",
                kind="binned",
                bins=_HIST2D_NB_Y,
                domain=(y_lo, y_hi),
                bin_variant="hist2d",
            ),
        )
        dims_1d = (
            TargetDimSpec(
                column="x", kind="binned", bins=_HIST2D_NB_X, domain=(x_lo, x_hi)
            ),
            TargetDimSpec(
                column="y", kind="binned", bins=_HIST2D_NB_Y, domain=(y_lo, y_hi)
            ),
        )
        k2 = cube_content_key(CubeSpec(source_name="s", free=free, target_dims=dims_2d))
        k1 = cube_content_key(CubeSpec(source_name="s", free=free, target_dims=dims_1d))
        assert k2 != k1

    def test_hist1d_key_is_byte_identical_to_default(self):
        # Adding bin_variant must NOT change an existing hist1d content key.
        free = FreeAxisSpec(column="free", kind="continuous", p=64, domain=(0.0, 1.0))
        default = CubeSpec(
            source_name="s",
            free=free,
            target_dims=(
                TargetDimSpec(column="t", kind="binned", bins=10, domain=(0.0, 50.0)),
            ),
        )
        explicit = CubeSpec(
            source_name="s",
            free=free,
            target_dims=(
                TargetDimSpec(
                    column="t",
                    kind="binned",
                    bins=10,
                    domain=(0.0, 50.0),
                    bin_variant="hist1d",
                ),
            ),
        )
        assert cube_content_key(default) == cube_content_key(explicit)


# ---------------------------------------------------------------------------
# Treemap target (Step 15): leaf-dims descriptor + finalize-then-sum rollup
# ---------------------------------------------------------------------------


_TM_SCHEMA = pl.Schema(
    {
        "cat": pl.String,
        "sub": pl.String,
        "num": pl.Int64,
        "val": pl.Float64,
        "ts": pl.Datetime("us"),
    }
)


def _treemap_target_df() -> pl.DataFrame:
    n = 1200
    cats = ["alpha", "beta", "gamma"]
    subs = ["s1", "s2", "s3", "s4"]
    return pl.DataFrame(
        {
            "active": [(i * 37) % 100 for i in range(n)],
            "cat": [cats[i % 3] for i in range(n)],
            "sub": [subs[(i * 7) % 4] for i in range(n)],
            "val": [float((i * 13) % 50) for i in range(n)],
        }
    )


class TestTreeMapTargetSpec:
    @pytest.mark.parametrize("agg", ["sum", "mean", "min", "max"])
    def test_target_dims_equal_path_with_measure(self, agg):
        from flexviz.trace.treemap import TreeMap

        trace = TreeMap(path=["cat", "sub"], values="val", agg=agg)
        spec = trace.get_cube_target_spec(None, schema=_TM_SCHEMA)
        assert spec is not None
        assert [d.column for d in spec.target_dims] == ["cat", "sub"]
        assert all(d.kind == "categorical" for d in spec.target_dims)
        assert spec.measure.agg == agg
        assert spec.measure.value_col == "val"

    def test_count_measure_when_values_omitted(self):
        from flexviz.trace.treemap import TreeMap

        trace = TreeMap(path=["cat", "sub"])
        spec = trace.get_cube_target_spec(None, schema=_TM_SCHEMA)
        assert spec is not None
        assert [d.column for d in spec.target_dims] == ["cat", "sub"]
        assert spec.measure.agg == "count"
        assert spec.measure.value_col is None

    def test_axis_range_is_ignored(self):
        from flexviz.trace.treemap import TreeMap

        trace = TreeMap(path=["cat", "sub"], values="val", agg="sum")
        assert trace.get_cube_target_spec(
            (0.0, 1.0), schema=_TM_SCHEMA
        ) == trace.get_cube_target_spec(None, schema=_TM_SCHEMA)

    @pytest.mark.parametrize("agg", ["median", "n_unique"])
    def test_median_n_unique_not_a_target(self, agg):
        from flexviz.trace.treemap import TreeMap

        trace = TreeMap(path=["cat", "sub"], values="val", agg=agg)
        assert trace.get_cube_target_spec(None, schema=_TM_SCHEMA) is None

    def test_numeric_path_level_not_a_target(self):
        from flexviz.trace.treemap import TreeMap

        trace = TreeMap(path=["cat", "num"], values="val", agg="sum")
        assert trace.get_cube_target_spec(None, schema=_TM_SCHEMA) is None

    def test_missing_schema_not_a_target(self):
        from flexviz.trace.treemap import TreeMap

        trace = TreeMap(path=["cat", "sub"], values="val", agg="sum")
        assert trace.get_cube_target_spec(None, schema=None) is None

    @pytest.mark.parametrize("bad_col", ["sub", "ts"])
    def test_non_numeric_value_col_not_a_target(self, bad_col):
        from flexviz.trace.treemap import TreeMap

        trace = TreeMap(path=["cat"], values=bad_col, agg="max")
        assert trace.get_cube_target_spec(None, schema=_TM_SCHEMA) is None


class TestTreeMapTargetCodec:
    @pytest.mark.parametrize("agg, values", [("count", None), ("mean", "val")])
    def test_round_trip_header(self, agg, values):
        from flexviz.trace.treemap import TreeMap

        df = _treemap_target_df()
        a_lo, a_hi = float(df["active"].min()), float(df["active"].max())
        trace = TreeMap(
            path=["cat", "sub"], values=values, agg=("sum" if values else "sum")
        )
        # build the spec from the trace's target descriptor
        target = trace.get_cube_target_spec(None, schema=df.schema)
        # Re-stamp the measure to exercise the requested agg explicitly.
        measure = MeasureSpec(agg=agg, value_col=values)
        spec = CubeSpec(
            source_name="s",
            free=FreeAxisSpec(column="active", p=64, domain=(a_lo, a_hi)),
            target_dims=target.target_dims,
            measure=measure,
        )
        cube = build_cube(df.lazy(), spec)
        header = decode_fvcube_header(encode_fvcube(cube, "k"))
        assert [d["name"] for d in header["target_dims"]] == ["cat", "sub"]
        assert [d["kind"] for d in header["target_dims"]] == [
            "categorical",
            "categorical",
        ]
        assert header["measure"]["agg"] == agg


class TestTreeMapTargetContentKey:
    def test_key_differs_by_path(self):
        from flexviz.trace.treemap import TreeMap

        df = _treemap_target_df()
        free = FreeAxisSpec(column="active", p=64, domain=(0.0, 100.0))

        def _key(path):
            t = TreeMap(path=path, values="val", agg="sum")
            spec = CubeSpec(
                source_name="s",
                free=free,
                target_dims=t.get_cube_target_spec(None, schema=df.schema).target_dims,
                measure=MeasureSpec(agg="sum", value_col="val"),
            )
            return cube_content_key(spec)

        assert _key(["cat", "sub"]) != _key(["cat"])
        assert _key(["cat", "sub"]) != _key(["sub", "cat"])

    def test_key_differs_by_agg(self):
        from flexviz.trace.treemap import TreeMap

        df = _treemap_target_df()
        free = FreeAxisSpec(column="active", p=64, domain=(0.0, 100.0))
        t = TreeMap(path=["cat", "sub"], values="val", agg="sum")
        dims = t.get_cube_target_spec(None, schema=df.schema).target_dims

        def _key(agg):
            return cube_content_key(
                CubeSpec(
                    source_name="s",
                    free=free,
                    target_dims=dims,
                    measure=MeasureSpec(agg=agg, value_col="val"),
                )
            )

        assert _key("sum") != _key("mean")


# ---------------------------------------------------------------------------
# Cube bundle codec (multi-blob binary transport envelope)
# ---------------------------------------------------------------------------


def _sample_blob(df: pl.DataFrame, cube_id: str, bins: int) -> bytes:
    spec = CubeSpec(
        source_name="s",
        free=FreeAxisSpec(column="active", p=64, domain=(0.0, 100.0)),
        target_dims=(
            TargetDimSpec(column="val", kind="binned", bins=bins, domain=(0.0, 50.0)),
        ),
        measure=MeasureSpec(agg="count"),
    )
    return encode_fvcube(build_cube(df.lazy(), spec), cube_id=cube_id)


class TestCubeBundleCodec:
    def test_roundtrip_multiple_blobs(self, df):
        b0 = _sample_blob(df, "k0", 8)
        b1 = _sample_blob(df, "k1", 12)
        trace_cubes = {"uidA": 0, "uidB": 1, "uidC": 0}
        bundle = encode_cube_bundle([b0, b1], trace_cubes)

        blobs, served = decode_cube_bundle(bundle)
        assert blobs == [b0, b1]
        assert served == trace_cubes
        # The recovered blobs are still valid FVCube blobs.
        assert decode_fvcube_header(blobs[0])["cube_id"] == "k0"
        assert decode_fvcube_header(blobs[1])["cube_id"] == "k1"

    def test_roundtrip_empty(self):
        bundle = encode_cube_bundle([], {})
        blobs, served = decode_cube_bundle(bundle)
        assert blobs == []
        assert served == {}

    def test_deterministic_bytes(self, df):
        b0 = _sample_blob(df, "k0", 8)
        tc = {"u": 0}
        assert encode_cube_bundle([b0], tc) == encode_cube_bundle([b0], tc)

    def test_bad_magic_rejected(self):
        with pytest.raises(ValueError, match="magic"):
            decode_cube_bundle(b"NOPE" + b"\x00" * 32)

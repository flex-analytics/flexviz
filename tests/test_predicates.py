"""Tests for the predicate → Polars expression primitive."""

from __future__ import annotations

import polars as pl
import pytest

from flexviz.spec import ClauseFilter, SelectionPredicate


@pytest.fixture
def df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "country": ["NL", "BE", "DE", "NL"],
            "source": ["solar", "wind", "solar", "wind"],
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )


class TestPredicatesToExpr:
    def test_empty_returns_true(self, df: pl.DataFrame):
        from flexviz.predicates import predicates_to_expr

        expr = predicates_to_expr([], df.schema)
        assert df.filter(expr).height == df.height

    def test_single_categorical_clause(self, df: pl.DataFrame):
        from flexviz.predicates import predicates_to_expr

        preds = [
            SelectionPredicate(clauses=[ClauseFilter(column="country", values=["NL"])])
        ]
        expr = predicates_to_expr(preds, df.schema)
        result = df.filter(expr)
        assert result["country"].to_list() == ["NL", "NL"]

    def test_single_range_clause(self, df: pl.DataFrame):
        from flexviz.predicates import predicates_to_expr

        preds = [
            SelectionPredicate(clauses=[ClauseFilter(column="value", range=(2.0, 3.0))])
        ]
        expr = predicates_to_expr(preds, df.schema)
        result = df.filter(expr)
        assert result["value"].to_list() == [2.0, 3.0]

    def test_and_clauses_within_predicate(self, df: pl.DataFrame):
        from flexviz.predicates import predicates_to_expr

        preds = [
            SelectionPredicate(
                clauses=[
                    ClauseFilter(column="country", values=["NL"]),
                    ClauseFilter(column="source", values=["solar"]),
                ]
            )
        ]
        expr = predicates_to_expr(preds, df.schema)
        result = df.filter(expr)
        assert result.shape == (1, 3)
        assert result["country"][0] == "NL"
        assert result["source"][0] == "solar"

    def test_or_across_predicates(self, df: pl.DataFrame):
        from flexviz.predicates import predicates_to_expr

        preds = [
            SelectionPredicate(clauses=[ClauseFilter(column="country", values=["NL"])]),
            SelectionPredicate(
                clauses=[ClauseFilter(column="source", values=["wind"])]
            ),
        ]
        expr = predicates_to_expr(preds, df.schema)
        result = df.filter(expr)
        assert set(result["country"].to_list()) == {"NL", "BE"}

    def test_temporal_range_casts_strings(self):
        import datetime as dt

        from flexviz.predicates import predicates_to_expr

        df = pl.DataFrame(
            {
                "ts": [
                    dt.datetime(2026, 1, 1),
                    dt.datetime(2026, 1, 2),
                    dt.datetime(2026, 1, 3),
                ]
            }
        )
        preds = [
            SelectionPredicate(
                clauses=[ClauseFilter(column="ts", range=("2026-01-02", "2026-01-03"))]
            )
        ]
        expr = predicates_to_expr(preds, df.schema)
        result = df.filter(expr)
        assert result.height == 2

    def test_int_range_uses_ceil_floor_semantics(self):
        from flexviz.predicates import predicates_to_expr

        df = pl.DataFrame({"i": [0, 1, 2, 3, 4]})
        preds = [
            SelectionPredicate(clauses=[ClauseFilter(column="i", range=(0.5, 3.5))])
        ]
        expr = predicates_to_expr(preds, df.schema)
        # ceil(0.5)=1, floor(3.5)=3 → [1,2,3]
        assert df.filter(expr)["i"].to_list() == [1, 2, 3]

    def test_unknown_column_returns_false_or_raises(self):
        from flexviz.predicates import predicates_to_expr

        df = pl.DataFrame({"x": [1, 2, 3]})
        preds = [
            SelectionPredicate(clauses=[ClauseFilter(column="missing", values=["a"])])
        ]
        # When the column doesn't exist in the schema, predicates_to_expr should
        # return an expression that yields zero rows (defensive: treat absent
        # columns as non-matching) — never crash the request handler.
        with pytest.raises(pl.exceptions.ColumnNotFoundError):
            df.filter(predicates_to_expr(preds, df.schema))


class TestClosedRanges:
    def test_float_closed_both_includes_both_endpoints(self, df: pl.DataFrame):
        from flexviz.predicates import predicates_to_expr

        preds = [
            SelectionPredicate(
                clauses=[ClauseFilter(column="value", range=(2.0, 3.0), closed="both")]
            )
        ]
        result = df.filter(predicates_to_expr(preds, df.schema))
        assert result["value"].to_list() == [2.0, 3.0]

    def test_float_closed_left_excludes_upper_endpoint(self, df: pl.DataFrame):
        from flexviz.predicates import predicates_to_expr

        preds = [
            SelectionPredicate(
                clauses=[ClauseFilter(column="value", range=(2.0, 3.0), closed="left")]
            )
        ]
        result = df.filter(predicates_to_expr(preds, df.schema))
        assert result["value"].to_list() == [2.0]

    def test_int_closed_left_includes_lower_excludes_upper(self):
        from flexviz.predicates import predicates_to_expr

        df = pl.DataFrame({"i": list(range(12))})
        preds = [
            SelectionPredicate(
                clauses=[ClauseFilter(column="i", range=(0, 10), closed="left")]
            )
        ]
        result = df.filter(predicates_to_expr(preds, df.schema))
        assert result["i"].to_list() == list(range(10))

    def test_temporal_closed_left_excludes_upper_bound(self):
        import datetime as dt

        from flexviz.predicates import predicates_to_expr

        df = pl.DataFrame(
            {
                "ts": [
                    dt.datetime(2026, 1, 1),
                    dt.datetime(2026, 1, 2),
                    dt.datetime(2026, 1, 3),
                ]
            }
        )
        preds = [
            SelectionPredicate(
                clauses=[
                    ClauseFilter(
                        column="ts", range=("2026-01-01", "2026-01-03"), closed="left"
                    )
                ]
            )
        ]
        result = df.filter(predicates_to_expr(preds, df.schema))
        assert result["ts"].to_list() == [
            dt.datetime(2026, 1, 1),
            dt.datetime(2026, 1, 2),
        ]

    def test_closed_omitted_defaults_to_both(self, df: pl.DataFrame):
        from flexviz.predicates import predicates_to_expr

        clause = ClauseFilter(column="value", range=(2.0, 3.0))
        assert clause.closed == "both"
        preds = [SelectionPredicate(clauses=[clause])]
        result = df.filter(predicates_to_expr(preds, df.schema))
        assert result["value"].to_list() == [2.0, 3.0]

    def test_values_with_closed_left_raises(self):
        with pytest.raises(ValueError, match="closed"):
            ClauseFilter(column="country", values=["NL"], closed="left")

    def test_mixed_closed_left_range_or_values(self, df: pl.DataFrame):
        from flexviz.predicates import predicates_to_expr

        preds = [
            SelectionPredicate(
                clauses=[ClauseFilter(column="value", range=(1.0, 2.0), closed="left")]
            ),
            SelectionPredicate(clauses=[ClauseFilter(column="country", values=["DE"])]),
        ]
        # closed="left" keeps value==1.0 only; OR country=="DE" adds value==3.0.
        result = df.filter(predicates_to_expr(preds, df.schema))
        assert sorted(result["value"].to_list()) == [1.0, 3.0]


class TestIntegerClosedRounding:
    """Integer bound rounding must respect each side's open/closed-ness.

    A closed bound rounds toward the interval interior (lo: ceil, hi: floor);
    an open bound rounds away from it (lo: floor, hi: ceil). Either way the
    compiled integer filter must select exactly the rows the real-valued
    range would select on a Float64 copy of the column.
    """

    # ClauseFilter only models "both" and "left"; the bounds helper supports
    # all four is_between closed values, so the full matrix is pinned there.
    @pytest.mark.parametrize("dtype", [pl.Int64, pl.UInt32])
    @pytest.mark.parametrize("closed", ["both", "left", "right", "none"])
    @pytest.mark.parametrize(
        "rng",
        [
            (0.2, 1.2),  # fractional both sides (the cube snap repro shape)
            (0.5, 3.5),
            (1.0, 3.0),  # exact-integer bounds: unchanged by either rounding
            (0.0, 4.0),
            (2.0, 2.0),  # degenerate
            (3.9, 4.1),  # single integer strictly inside
        ],
    )
    def test_bounds_membership_matches_float_semantics(self, dtype, closed, rng):
        from flexviz.trace.base import _typed_range_bounds

        df = pl.DataFrame({"i": pl.Series(range(5), dtype=dtype)})
        lo_e, hi_e = _typed_range_bounds("i", rng, df.schema, closed)
        got = df.filter(pl.col("i").is_between(lo_e, hi_e, closed=closed))[
            "i"
        ].to_list()
        lo, hi = rng
        ref = df.filter(pl.col("i").cast(pl.Float64).is_between(lo, hi, closed=closed))[
            "i"
        ].to_list()
        assert got == ref, f"{dtype} closed={closed} range={rng}: {got} != {ref}"

    @pytest.mark.parametrize("closed", ["both", "left", "right", "none"])
    @pytest.mark.parametrize("rng", [(-1.5, 1.5), (-3.2, -0.2), (-2.0, 2.0)])
    def test_negative_bounds_match_float_semantics(self, closed, rng):
        from flexviz.trace.base import _typed_range_bounds

        df = pl.DataFrame({"i": pl.Series(range(-4, 5), dtype=pl.Int64)})
        lo_e, hi_e = _typed_range_bounds("i", rng, df.schema, closed)
        got = df.filter(pl.col("i").is_between(lo_e, hi_e, closed=closed))[
            "i"
        ].to_list()
        lo, hi = rng
        ref = df.filter(pl.col("i").cast(pl.Float64).is_between(lo, hi, closed=closed))[
            "i"
        ].to_list()
        assert got == ref

    @pytest.mark.parametrize("closed", ["both", "left"])
    @pytest.mark.parametrize("rng", [(0.2, 1.2), (0.5, 3.5), (1.0, 3.0)])
    def test_clause_membership_matches_float_semantics(self, closed, rng):
        # End-to-end through ClauseFilter → predicates_to_expr for the two
        # closed values the spec model admits.
        from flexviz.predicates import predicates_to_expr

        df = pl.DataFrame({"i": pl.Series(range(5), dtype=pl.Int64)})
        preds = [
            SelectionPredicate(
                clauses=[ClauseFilter(column="i", range=rng, closed=closed)]
            )
        ]
        got = df.filter(predicates_to_expr(preds, df.schema))["i"].to_list()
        lo, hi = rng
        ref = df.filter(pl.col("i").cast(pl.Float64).is_between(lo, hi, closed=closed))[
            "i"
        ].to_list()
        assert got == ref

    def test_snapped_left_closed_repro_selects_interior_integer(self):
        # Regression pin: a snapped closed="left" range [0.2, 1.2) on Int64
        # [0,1,2] used to compile to is_between(1, 1, "left") → empty; it
        # must select {1}.
        from flexviz.predicates import predicates_to_expr

        df = pl.DataFrame({"i": [0, 1, 2]})
        preds = [
            SelectionPredicate(
                clauses=[ClauseFilter(column="i", range=(0.2, 1.2), closed="left")]
            )
        ]
        assert df.filter(predicates_to_expr(preds, df.schema))["i"].to_list() == [1]

    def test_viewport_range_filter_keeps_closed_both_rounding(self):
        # The viewport path (_range_filter_expr) stays closed="both": inward
        # ceil/floor rounding must be byte-identical to the pre-fix behavior.
        from flexviz.trace.base import _range_filter_expr

        df = pl.DataFrame({"i": [0, 1, 2, 3, 4]})
        expr = _range_filter_expr("i", (0.5, 3.5), df.schema)
        assert df.filter(expr)["i"].to_list() == [1, 2, 3]


class TestBooleanCoercion:
    def test_string_true_false_coerced_to_bool(self):
        from flexviz.predicates import predicates_to_expr

        df = pl.DataFrame({"flag": [True, False, True]})
        preds = [
            SelectionPredicate(clauses=[ClauseFilter(column="flag", values=["true"])])
        ]
        result = df.filter(predicates_to_expr(preds, df.schema))
        assert result["flag"].to_list() == [True, True]

    def test_invalid_boolean_value_raises(self):
        from flexviz.predicates import predicates_to_expr

        df = pl.DataFrame({"flag": [True, False, True]})
        preds = [
            SelectionPredicate(clauses=[ClauseFilter(column="flag", values=["False"])])
        ]
        with pytest.raises(ValueError, match="'false'"):
            predicates_to_expr(preds, df.schema)


class TestCanonicalPassiveKey:
    """Contract E: the canonical passive key over committed selections."""

    @staticmethod
    def _sel(uid, *predicates):
        from flexviz.spec import SelectionState

        return SelectionState(
            source_figure_uid=uid,
            predicates=[SelectionPredicate.model_validate(p) for p in predicates],
        )

    def test_empty_passive_set_returns_none(self):
        from flexviz.predicates import canonical_passive_key

        assert canonical_passive_key([], "figA") is None
        # Active figure's own selection excluded (re-brush case).
        own = self._sel("figA", {"clauses": [{"column": "a", "range": [1, 2]}]})
        assert canonical_passive_key([own], "figA") is None
        # source_figure_uid=None never filters in the legacy engine.
        anon = self._sel(None, {"clauses": [{"column": "a", "range": [1, 2]}]})
        assert canonical_passive_key([anon], "figA") is None
        # Empty predicates excluded.
        empty = self._sel("figB")
        assert canonical_passive_key([empty], "figA") is None

    def test_clause_order_insensitive(self):
        from flexviz.predicates import canonical_passive_key

        s1 = self._sel(
            "figB",
            {
                "clauses": [
                    {"column": "a", "range": [1.0, 2.0]},
                    {"column": "b", "values": ["x"]},
                ]
            },
        )
        s2 = self._sel(
            "figB",
            {
                "clauses": [
                    {"column": "b", "values": ["x"]},
                    {"column": "a", "range": [1.0, 2.0]},
                ]
            },
        )
        assert canonical_passive_key([s1], "figA") == canonical_passive_key(
            [s2], "figA"
        )

    def test_predicate_order_insensitive(self):
        from flexviz.predicates import canonical_passive_key

        p1 = {"clauses": [{"column": "g", "values": ["g0"]}]}
        p2 = {"clauses": [{"column": "g", "values": ["g1"]}]}
        k12 = canonical_passive_key([self._sel("figB", p1, p2)], "figA")
        k21 = canonical_passive_key([self._sel("figB", p2, p1)], "figA")
        assert k12 == k21

    def test_selection_order_insensitive(self):
        from flexviz.predicates import canonical_passive_key

        sb = self._sel("figB", {"clauses": [{"column": "b", "range": [1, 2]}]})
        sc = self._sel("figC", {"clauses": [{"column": "c", "range": [3, 4]}]})
        assert canonical_passive_key([sb, sc], "figA") == canonical_passive_key(
            [sc, sb], "figA"
        )

    def test_values_sorted(self):
        from flexviz.predicates import canonical_passive_key

        s1 = self._sel("figB", {"clauses": [{"column": "g", "values": ["b", "a"]}]})
        s2 = self._sel("figB", {"clauses": [{"column": "g", "values": ["a", "b"]}]})
        assert canonical_passive_key([s1], "figA") == canonical_passive_key(
            [s2], "figA"
        )

    def test_nesting_preserved_two_selections_differ_from_merged(self):
        from flexviz.predicates import canonical_passive_key

        p1 = {"clauses": [{"column": "g", "values": ["g0"]}]}
        p2 = {"clauses": [{"column": "h", "values": ["h0"]}]}
        # AND of two single-predicate selections...
        two = [self._sel("figB", p1), self._sel("figC", p2)]
        # ...is NOT the OR of both predicates in one selection.
        merged = [self._sel("figB", p1, p2)]
        assert canonical_passive_key(two, "figA") != canonical_passive_key(
            merged, "figA"
        )

    def test_closed_field_included(self):
        from flexviz.predicates import canonical_passive_key

        left = self._sel(
            "figB",
            {"clauses": [{"column": "a", "range": [1.0, 2.0], "closed": "left"}]},
        )
        both = self._sel(
            "figB",
            {"clauses": [{"column": "a", "range": [1.0, 2.0], "closed": "both"}]},
        )
        assert canonical_passive_key([left], "figA") != canonical_passive_key(
            [both], "figA"
        )

    def test_different_predicates_differ(self):
        from flexviz.predicates import canonical_passive_key

        s1 = self._sel("figB", {"clauses": [{"column": "a", "range": [1.0, 2.0]}]})
        s2 = self._sel("figB", {"clauses": [{"column": "a", "range": [1.0, 3.0]}]})
        assert canonical_passive_key([s1], "figA") != canonical_passive_key(
            [s2], "figA"
        )

    def test_key_is_compact_json(self):
        import json as _json

        from flexviz.predicates import canonical_passive_key

        sel = self._sel(
            "figB",
            {
                "clauses": [
                    {"column": "a", "range": [1.0, 2.0], "closed": "left"},
                    {"column": "g", "values": ["y", "x"]},
                ]
            },
        )
        key = canonical_passive_key([sel], "figA")
        assert isinstance(key, str)
        assert " " not in key
        parsed = _json.loads(key)
        assert isinstance(parsed, list) and len(parsed) == 1
        # Pinned canonical shapes: range {"c","r","cl"}, values {"c","v"}.
        (pred_str,) = _json.loads(parsed[0])
        clauses = _json.loads(pred_str)
        assert clauses == [
            {"c": "a", "r": [1.0, 2.0], "cl": "left"},
            {"c": "g", "v": ["x", "y"]},
        ]


class TestValuesEdgeCases:
    """Characterization of a values clause whose members are empty, null or
    uncoercible. A null selects nothing, as a member and as a data value."""

    @pytest.fixture
    def nulls_df(self) -> pl.DataFrame:
        return pl.DataFrame({"country": ["NL", None, "BE", "NL"]})

    @staticmethod
    def _rows(df: pl.DataFrame, column: str, values: list) -> list:
        from flexviz.predicates import predicates_to_expr

        preds = [
            SelectionPredicate(clauses=[ClauseFilter(column=column, values=values)])
        ]
        return df.filter(predicates_to_expr(preds, df.schema))[column].to_list()

    def test_empty_values_selects_no_rows(self, nulls_df: pl.DataFrame):
        assert self._rows(nulls_df, "country", []) == []

    def test_null_only_values_selects_no_rows(self, nulls_df: pl.DataFrame):
        assert self._rows(nulls_df, "country", [None]) == []

    def test_null_member_selects_the_other_values_only(self, nulls_df: pl.DataFrame):
        assert self._rows(nulls_df, "country", ["NL", None]) == ["NL", "NL"]

    def test_uncoercible_value_on_int_column_selects_no_rows(self):
        # "abc" casts to null under the lenient cast, leaving no member.
        df = pl.DataFrame({"i": [1, 2, 3]})
        assert self._rows(df, "i", ["abc"]) == []

    def test_uncoercible_member_leaves_the_valid_one(self):
        df = pl.DataFrame({"i": [1, 2, 3]})
        assert self._rows(df, "i", ["abc", "2"]) == [2]

    def test_datetime_string_value_selects_no_rows(self):
        # Polars 1.44 casts String → Datetime to null, so the clause is empty.
        import datetime as dt

        df = pl.DataFrame({"ts": [dt.datetime(2026, 1, 1), dt.datetime(2026, 1, 2)]})
        assert self._rows(df, "ts", ["2026-01-02"]) == []

    @pytest.mark.parametrize("values", [[], [None]])
    def test_missing_column_raises_even_when_the_clause_selects_no_rows(
        self, nulls_df: pl.DataFrame, values: list
    ):
        from flexviz.predicates import predicates_to_expr

        preds = [
            SelectionPredicate(clauses=[ClauseFilter(column="missing", values=values)])
        ]
        expr = predicates_to_expr(preds, nulls_df.schema)
        with pytest.raises(pl.exceptions.ColumnNotFoundError):
            nulls_df.filter(expr)


class TestValuesCompiledForm:
    """A small values clause compiles to an OR of equality tests; a wide one
    keeps ``is_in``. Both forms must select exactly the same rows."""

    @staticmethod
    def _expr(column: str, values: list, schema: pl.Schema) -> pl.Expr:
        from flexviz.predicates import predicates_to_expr

        return predicates_to_expr(
            [SelectionPredicate(clauses=[ClauseFilter(column=column, values=values)])],
            schema,
        )

    def _both_forms(
        self, monkeypatch: pytest.MonkeyPatch, df: pl.DataFrame, column: str, values
    ) -> tuple[list, list]:
        """Rows under the compiled chain and under the ``is_in`` form."""
        from flexviz import predicates

        chain = df.filter(self._expr(column, values, df.schema))[column].to_list()
        # A cutoff of 0 sends every non-empty clause down the `is_in` branch.
        monkeypatch.setattr(predicates, "_EQUALITY_CHAIN_MAX_VALUES", 0)
        is_in = df.filter(self._expr(column, values, df.schema))[column].to_list()
        return chain, is_in

    @pytest.mark.parametrize("k", [1, 5])
    def test_small_clause_compiles_to_equality_chain(self, k: int):
        from flexviz.predicates import _EQUALITY_CHAIN_MAX_VALUES

        assert k <= _EQUALITY_CHAIN_MAX_VALUES
        df = pl.DataFrame({"g": [f"g{i}" for i in range(20)]})
        values = [f"g{i}" for i in range(k)]
        expr = self._expr("g", values, df.schema)
        assert "is_in" not in str(expr)
        # One column reference per equality test, and no other column.
        assert expr.meta.root_names() == ["g"] * k
        assert df.filter(expr)["g"].to_list() == values

    def test_above_the_cutoff_keeps_is_in(self):
        from flexviz.predicates import _EQUALITY_CHAIN_MAX_VALUES

        k = _EQUALITY_CHAIN_MAX_VALUES + 1
        df = pl.DataFrame({"g": [f"g{i}" for i in range(k + 3)]})
        values = [f"g{i}" for i in range(k)]
        expr = self._expr("g", values, df.schema)
        assert "is_in" in str(expr)
        assert df.filter(expr)["g"].to_list() == values

    def test_string_rows_match_is_in(self, monkeypatch: pytest.MonkeyPatch):
        df = pl.DataFrame({"country": ["NL", None, "BE", "DE"]})
        chain, is_in = self._both_forms(monkeypatch, df, "country", ["NL", "DE", None])
        assert chain == is_in == ["NL", "DE"]

    def test_boolean_rows_match_is_in(self, monkeypatch: pytest.MonkeyPatch):
        df = pl.DataFrame({"flag": [True, False, None]})
        chain, is_in = self._both_forms(monkeypatch, df, "flag", ["true", "false"])
        assert chain == is_in == [True, False]

    def test_int_column_with_string_values_matches_is_in(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        # Clause values arrive as JSON, so an Int64 column is selected by strings.
        df = pl.DataFrame({"i": [1, 2, 3, 4]})
        chain, is_in = self._both_forms(monkeypatch, df, "i", ["2", "4"])
        assert chain == is_in == [2, 4]

    def test_float32_column_is_not_widened(self, monkeypatch: pytest.MonkeyPatch):
        df = pl.DataFrame({"f": pl.Series([1.5, 2.5, 3.5], dtype=pl.Float32)})
        chain, is_in = self._both_forms(monkeypatch, df, "f", [2.5])
        assert chain == is_in == [2.5]

    def test_datetime_column_matches_is_in(self, monkeypatch: pytest.MonkeyPatch):
        import datetime as dt

        df = pl.DataFrame({"ts": [dt.datetime(2026, 1, 1), dt.datetime(2026, 1, 2)]})
        chain, is_in = self._both_forms(
            monkeypatch, df, "ts", [dt.datetime(2026, 1, 2)]
        )
        assert chain == is_in == [dt.datetime(2026, 1, 2)]

    @pytest.mark.parametrize("dtype", [pl.Categorical, pl.Enum(["a", "b", "c"])])
    def test_categorical_rows_match_is_in(self, monkeypatch: pytest.MonkeyPatch, dtype):
        df = pl.DataFrame({"c": pl.Series(["a", "b", "c"], dtype=dtype)})
        # "zz" is no category: it casts to null and drops out of both forms.
        chain, is_in = self._both_forms(monkeypatch, df, "c", ["a", "c", "zz"])
        assert chain == is_in == ["a", "c"]

    def test_canonical_clause_key_unchanged(self):
        # The canonical key reads the clause, never the compiled expression.
        import json as _json

        from flexviz.predicates import canonical_passive_key
        from flexviz.spec import SelectionState

        sel = SelectionState(
            source_figure_uid="figB",
            predicates=[
                SelectionPredicate(
                    clauses=[ClauseFilter(column="g", values=["y", "x"])]
                )
            ],
        )
        key = canonical_passive_key([sel], "figA")
        (pred_str,) = _json.loads(_json.loads(key)[0])
        assert _json.loads(pred_str) == [{"c": "g", "v": ["x", "y"]}]

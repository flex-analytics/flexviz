"""Unit tests for TreeMap trace."""

from __future__ import annotations

import polars as pl
import pytest

from flexviz.LF import LFQueryBuilder, GroupedAggregationSpec
from flexviz.trace.treemap import TreeMap
from flexviz.trace.base import TraceResult
from flexviz.spec import TraceSpec

# ---- helpers ---------------------------------------------------------------


def _aggregate_treemap(
    df: pl.DataFrame,
    path: list[str],
    values: str | None = None,
    agg: str = "sum",
) -> TraceResult:
    lf = LFQueryBuilder(df)
    trace = TreeMap(path=path, values=values, agg=agg)
    spec = trace.get_aggregation_spec({}, schema=lf.schema)
    _, grouped_dfs = lf.aggregate([], [spec])
    return trace._to_update(grouped_dfs[trace.uid])


@pytest.fixture()
def continent_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "continent": ["Europe", "Europe", "Asia", "Asia", "Europe"],
            "country": ["Germany", "France", "Japan", "China", "Germany"],
            "population": [83.0, 67.0, 125.0, 1400.0, 83.0],
        }
    )


# ---- constructor -----------------------------------------------------------


class TestTreeMapInit:
    def test_trace_type(self):
        t = TreeMap(path=["continent", "country"])
        assert t.trace_type == "treemap"

    def test_axes_none(self):
        t = TreeMap(path=["continent", "country"])
        assert t._axes is None

    def test_update_on_zoom_false(self):
        t = TreeMap(path=["continent", "country"])
        assert t.recompute_axes == ()
        assert t.update_on_zoom is False

    def test_overlay_style(self):
        t = TreeMap(path=["continent", "country"])
        assert t.overlay_style == "filtered_only"

    def test_path_stored_in_params(self):
        t = TreeMap(path=["a", "b"])
        assert t._params["path"] == ["a", "b"]

    def test_count_agg_when_no_values(self):
        t = TreeMap(path=["cat"])
        assert t._params["agg"] == "count"
        assert "values" not in t._backend_data

    def test_sum_agg_with_values(self):
        t = TreeMap(path=["cat"], values="pop", agg="sum")
        assert t._params["agg"] == "sum"
        assert t._backend_data["values"] == "pop"

    def test_invalid_agg_raises(self):
        with pytest.raises(ValueError, match="agg must be one of"):
            TreeMap(path=["cat"], values="pop", agg="invalid")

    def test_name_in_display(self):
        t = TreeMap(path=["cat"], name="My Tree")
        assert t._display["name"] == "My Tree"

    def test_color_map_in_display(self):
        cm = {"Europe": "#0000ff"}
        t = TreeMap(path=["cat"], color_map=cm)
        assert t._display["color_map"] == cm


# ---- aggregation spec ------------------------------------------------------


class TestTreeMapAggSpec:
    def test_returns_grouped_aggregation_spec(self, continent_df):
        lf = LFQueryBuilder(continent_df)
        t = TreeMap(path=["continent", "country"], values="population")
        spec = t.get_aggregation_spec({}, schema=lf.schema)
        assert isinstance(spec, GroupedAggregationSpec)

    def test_group_cols_one_level(self, continent_df):
        lf = LFQueryBuilder(continent_df)
        t = TreeMap(path=["continent"], values="population")
        spec = t.get_aggregation_spec({}, schema=lf.schema)
        assert spec.group_cols == ("continent",)
        assert spec.sort_cols == ("continent",)

    def test_group_cols_two_levels(self, continent_df):
        lf = LFQueryBuilder(continent_df)
        t = TreeMap(path=["continent", "country"], values="population")
        spec = t.get_aggregation_spec({}, schema=lf.schema)
        assert spec.group_cols == ("continent", "country")
        assert spec.sort_cols == ("continent", "country")

    def test_count_agg_no_values(self, continent_df):
        lf = LFQueryBuilder(continent_df)
        t = TreeMap(path=["continent"])
        spec = t.get_aggregation_spec({}, schema=lf.schema)
        assert isinstance(spec, GroupedAggregationSpec)


# ---- hierarchy building ---------------------------------------------------


class TestTreeMapHierarchy:
    def test_one_level_has_root_and_leaves(self, continent_df):
        result = _aggregate_treemap(
            continent_df, path=["continent"], values="population"
        )
        ids = result.updates["ids"]
        labels = result.updates["labels"]
        parents = result.updates["parents"]
        assert "root" in ids
        root_idx = ids.index("root")
        assert labels[root_idx] == ""
        assert parents[root_idx] == ""
        assert "root/Europe" in ids
        assert "root/Asia" in ids

    def test_one_level_parents_are_root(self, continent_df):
        result = _aggregate_treemap(
            continent_df, path=["continent"], values="population"
        )
        ids = result.updates["ids"]
        parents = result.updates["parents"]
        for i, node_id in enumerate(ids):
            if node_id != "root":
                assert parents[i] == "root"

    def test_two_level_ids(self, continent_df):
        result = _aggregate_treemap(
            continent_df, path=["continent", "country"], values="population"
        )
        ids = result.updates["ids"]
        assert "root" in ids
        assert "root/Europe" in ids
        assert "root/Asia" in ids
        assert "root/Europe/Germany" in ids
        assert "root/Europe/France" in ids
        assert "root/Asia/Japan" in ids
        assert "root/Asia/China" in ids

    def test_two_level_parents(self, continent_df):
        result = _aggregate_treemap(
            continent_df, path=["continent", "country"], values="population"
        )
        ids = result.updates["ids"]
        parents = result.updates["parents"]
        idx = ids.index("root/Europe/Germany")
        assert parents[idx] == "root/Europe"

    def test_root_value_is_total(self, continent_df):
        result = _aggregate_treemap(
            continent_df, path=["continent", "country"], values="population"
        )
        ids = result.updates["ids"]
        values = result.updates["values"]
        root_idx = ids.index("root")
        # Germany appears twice (83+83=166), Europe=166+67=233, Asia=125+1400=1525, total=1758
        assert values[root_idx] == pytest.approx(1758.0)

    def test_count_agg_root_equals_row_count(self, continent_df):
        result = _aggregate_treemap(continent_df, path=["continent"])
        ids = result.updates["ids"]
        values = result.updates["values"]
        root_idx = ids.index("root")
        assert values[root_idx] == 5  # 5 rows

    def test_lists_same_length(self, continent_df):
        result = _aggregate_treemap(
            continent_df, path=["continent", "country"], values="population"
        )
        n = len(result.updates["ids"])
        assert len(result.updates["labels"]) == n
        assert len(result.updates["parents"]) == n
        assert len(result.updates["values"]) == n

    def test_single_row_df(self):
        df = pl.DataFrame({"cat": ["X"], "val": [10.0]})
        result = _aggregate_treemap(df, path=["cat"], values="val")
        ids = result.updates["ids"]
        assert "root" in ids
        assert "root/X" in ids

    def test_group_results_is_none(self, continent_df):
        result = _aggregate_treemap(
            continent_df, path=["continent", "country"], values="population"
        )
        assert result.group_results is None


# ---- spec round-trip -------------------------------------------------------


class TestTreeMapSpec:
    def test_roundtrip_with_values(self):
        t = TreeMap(
            path=["continent", "country"], values="population", agg="sum", name="Pop"
        )
        spec = t.to_trace_spec()
        assert spec.trace_type == "treemap"
        assert spec.params["path"] == ["continent", "country"]
        assert spec.params["agg"] == "sum"
        assert spec.backend_data["values"] == "population"
        t2 = TreeMap.from_trace_spec(spec)
        assert t2._params["path"] == ["continent", "country"]
        assert t2._params["agg"] == "sum"
        assert t2._backend_data["values"] == "population"

    def test_roundtrip_count(self):
        t = TreeMap(path=["cat"])
        spec = t.to_trace_spec()
        assert spec.params["agg"] == "count"
        assert "values" not in spec.backend_data
        t2 = TreeMap.from_trace_spec(spec)
        assert t2._params["agg"] == "count"
        assert "values" not in t2._backend_data

    def test_backward_compat_agg_count_with_values_field(self):
        """Old spec: agg='count' but backend_data has a values key -> values=None."""
        spec = TraceSpec(
            uid="t1",
            trace_type="treemap",
            backend_data={"values": "ignored"},
            params={"path": ["cat"], "agg": "count"},
            display={},
            axes=None,
        )
        t = TreeMap.from_trace_spec(spec)
        assert "values" not in t._backend_data

    def test_uid_preserved(self):
        t = TreeMap(path=["cat"])
        spec = t.to_trace_spec()
        t2 = TreeMap.from_trace_spec(spec)
        assert t2.uid == t.uid


# ---- Figure.add_treemap ----------------------------------------------------


def _aggregate_treemap_with_color_map(
    df: pl.DataFrame,
    path: list[str],
    values: str | None = None,
    color_map: dict | None = None,
) -> TraceResult:
    lf = LFQueryBuilder(df)
    trace = TreeMap(path=path, values=values, color_map=color_map)
    spec = trace.get_aggregation_spec({}, schema=lf.schema)
    _, grouped_dfs = lf.aggregate([], [spec])
    return trace._to_update(grouped_dfs[trace.uid])


class TestTreeMapColorMap:
    @pytest.fixture()
    def df(self) -> pl.DataFrame:
        return pl.DataFrame({"cat": ["A", "B", "C"], "val": [10.0, 20.0, 30.0]})

    def test_color_map_populates_marker_colors(self, df):
        color_map = {"A": "#ff0000", "B": "#00ff00", "C": "#0000ff"}
        result = _aggregate_treemap_with_color_map(
            df, path=["cat"], values="val", color_map=color_map
        )
        assert "marker" in result.updates
        labels = result.updates["labels"]
        colors = result.updates["marker"]["colors"]
        assert len(colors) == len(labels)
        for label, color in zip(labels, colors):
            if label == "":
                continue  # root node has empty label, not in color_map
            assert color == color_map[label]

    def test_no_color_map_no_marker_key(self, df):
        result = _aggregate_treemap_with_color_map(df, path=["cat"], color_map=None)
        assert "marker" not in result.updates

    def test_label_not_in_color_map_gets_none(self, df):
        color_map = {"A": "#ff0000"}  # B and C are missing
        result = _aggregate_treemap_with_color_map(
            df, path=["cat"], values="val", color_map=color_map
        )
        labels = result.updates["labels"]
        colors = result.updates["marker"]["colors"]
        for label, color in zip(labels, colors):
            if label == "A":
                assert color == "#ff0000"
            elif label != "":
                assert color is None

    def test_two_level_color_map_applies_to_all_nodes(self, continent_df):
        color_map = {"Europe": "#0000ff", "Asia": "#ff0000"}
        result = _aggregate_treemap_with_color_map(
            continent_df,
            path=["continent", "country"],
            values="population",
            color_map=color_map,
        )
        labels = result.updates["labels"]
        colors = result.updates["marker"]["colors"]
        europe_idx = labels.index("Europe")
        asia_idx = labels.index("Asia")
        assert colors[europe_idx] == "#0000ff"
        assert colors[asia_idx] == "#ff0000"


class TestFigureAddTreemap:
    def test_add_treemap_registers_trace(self):
        import polars as pl
        from flexviz.figure import Figure

        df = pl.DataFrame({"cat": ["A", "B"], "val": [1.0, 2.0]})
        fig = Figure(df)
        fig.add_treemap(path=["cat"], values="val")
        spec = fig.to_spec()
        assert len(spec.figure.traces) == 1
        assert spec.figure.traces[0].trace_type == "treemap"
        assert spec.figure.traces[0].params["path"] == ["cat"]

    def test_add_treemap_invalid_agg(self):
        import polars as pl
        from flexviz.figure import Figure

        df = pl.DataFrame({"cat": ["A"], "val": [1.0]})
        fig = Figure(df)
        with pytest.raises(ValueError, match="agg must be one of"):
            fig.add_treemap(path=["cat"], values="val", agg="bad")

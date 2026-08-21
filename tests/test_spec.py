"""Unit tests for spec models, encode/decode, and layout helpers."""

from __future__ import annotations

import json

import polars as pl
import pytest
from pydantic import ValidationError

from flexviz.events import GroupedChildDelta, TraceDelta
from flexviz.figure import Figure
from flexviz.spec import (
    AxisRange,
    ClientState,
    DashboardSpec,
    GridItem,
    GroupDomainState,
    TraceDisplay,
    TraceHoverSpec,
    TraceParams,
    FigureSpec,
    InteractionState,
    LayoutSpec,
    SelectionState,
    TraceSpec,
    VisualizationSpec,
    _auto_grid_items,
    decode_spec,
    encode_spec,
)

# ---- helpers ---------------------------------------------------------------


def _make_viz_spec() -> VisualizationSpec:
    ts = TraceSpec(
        uid="trace-aaa",
        trace_type="line",
        axes=("x", "y"),
        backend_data={"x": "ts", "y": "val"},
        params={"n_points": 500},
        display={"name": "A", "color": "#1f77b4"},
    )
    fig = FigureSpec(uid="fig-bbb", source="mydata", traces=[ts])
    return VisualizationSpec(figure=fig)


def _make_dashboard_spec() -> DashboardSpec:
    ts1 = TraceSpec(uid="t1", trace_type="line", backend_data={"x": "ts", "y": "val"})
    ts2 = TraceSpec(uid="t2", trace_type="histogram", backend_data={"x": "val"})
    fig1 = FigureSpec(uid="fig-1", source="s", traces=[ts1])
    fig2 = FigureSpec(uid="fig-2", source="s", traces=[ts2])
    return DashboardSpec(
        figures=[fig1, fig2],
        state=InteractionState(),
        layout=LayoutSpec(
            grid_items=[
                GridItem(fig_uid="fig-1", x=0, y=0, w=6, h=5),
                GridItem(fig_uid="fig-2", x=6, y=0, w=6, h=5),
            ]
        ),
    )


# ---- VisualizationSpec round-trip ------------------------------------------


class TestAxisRange:
    def test_accepts_datetime_strings(self):
        r = AxisRange(min="2026-03-25T10:00:00Z", max="2026-03-25T11:00:00Z")

        assert r.min == "2026-03-25T10:00:00Z"
        assert r.max == "2026-03-25T11:00:00Z"
        assert r.as_tuple() == ("2026-03-25T10:00:00Z", "2026-03-25T11:00:00Z")


# ---- VisualizationSpec round-trip ------------------------------------------


class TestVisualizationSpecRoundTrip:
    def test_json_roundtrip(self):
        original = _make_viz_spec()
        raw = json.loads(original.model_dump_json())
        restored = VisualizationSpec.model_validate(raw)

        assert restored.figure.uid == original.figure.uid
        assert restored.figure.source == original.figure.source
        assert len(restored.figure.traces) == 1
        assert restored.figure.traces[0].uid == "trace-aaa"
        assert restored.figure.traces[0].params == {"n_points": 500}


# ---- DashboardSpec round-trip ----------------------------------------------


class TestDashboardSpecRoundTrip:
    def test_json_roundtrip(self):
        original = _make_dashboard_spec()
        raw = json.loads(original.model_dump_json())
        restored = DashboardSpec.model_validate(raw)

        assert len(restored.figures) == 2
        assert restored.figures[0].uid == "fig-1"
        assert restored.figures[1].uid == "fig-2"
        assert restored.figures[0].traces[0].uid == "t1"
        assert restored.layout.grid_items is not None
        assert restored.layout.grid_items[0].w == 6

    def test_cross_filter_mode_defaults_to_update(self):
        restored = DashboardSpec.model_validate(_make_dashboard_spec().model_dump())
        assert restored.state.cross_filter_mode == "update"

    def test_cross_filter_mode_roundtrips(self):
        original = _make_dashboard_spec()
        original.state.cross_filter_mode = "overlay"
        restored = DashboardSpec.model_validate(json.loads(original.model_dump_json()))
        assert restored.state.cross_filter_mode == "overlay"

    def test_geo_viewport_coordinates_roundtrip(self):
        original = _make_dashboard_spec()
        coords = [[-73.5, 40.5], [-72.5, 40.5], [-72.5, 41.5], [-73.5, 41.5]]
        original.state.viewport = {"fig-1/coordinates": coords}

        restored = DashboardSpec.model_validate(json.loads(original.model_dump_json()))

        assert [
            list(point) for point in restored.state.viewport["fig-1/coordinates"]
        ] == coords


class TestDashboardSpecState:
    def test_interaction_state_has_no_hover_enabled(self):
        """hover_enabled must not exist on InteractionState after migration."""
        state = InteractionState()
        assert not hasattr(
            state, "hover_enabled"
        ), "hover_enabled must be deleted from InteractionState"

    def test_interaction_state_accepts_spec_without_hover_enabled(self):
        """Deserializing a spec that omits hover_enabled must not raise."""
        data = {"viewport": {}, "selections": [], "cross_filter_mode": "update"}
        state = InteractionState.model_validate(data)
        assert state.cross_filter_mode == "update"

    def test_client_state_defaults_to_off(self):
        spec = DashboardSpec()
        assert spec.client_state.hover_mode == "off"

    def test_client_state_roundtrips(self):
        spec = DashboardSpec()
        spec.client_state.hover_mode = "on"
        restored = DashboardSpec.model_validate(json.loads(spec.model_dump_json()))
        assert restored.client_state.hover_mode == "on"

    def test_client_state_encode_decode_roundtrip(self):
        spec = _make_dashboard_spec()
        spec.client_state.hover_mode = "on"
        decoded = decode_spec(encode_spec(spec))
        assert isinstance(decoded, DashboardSpec)
        assert decoded.client_state.hover_mode == "on"

    def test_trace_spec_has_hover_field(self):
        ts = TraceSpec(
            uid="t1", trace_type="line", backend_data={"x": "ts", "y": "val"}
        )
        assert hasattr(ts, "hover")
        assert ts.hover.source_modes == []
        assert ts.hover.target_modes == []

    def test_trace_spec_hover_roundtrips(self):

        ts = TraceSpec(
            uid="t1",
            trace_type="line",
            backend_data={"x": "ts", "y": "val"},
            hover=TraceHoverSpec(source_modes=["axis"], target_modes=["axis"]),
        )
        restored = TraceSpec.model_validate(json.loads(ts.model_dump_json()))
        assert restored.hover.source_modes == ["axis"]
        assert restored.hover.target_modes == ["axis"]

    def test_trace_spec_selection_defaults_to_none_kind(self):
        ts = TraceSpec(
            uid="t1", trace_type="line", backend_data={"x": "ts", "y": "val"}
        )
        assert ts.selection.kind == "none"

    def test_trace_spec_selection_roundtrips(self):
        from flexviz.spec import TraceSelectionSpec

        ts = TraceSpec(
            uid="t1",
            trace_type="line",
            backend_data={"x": "ts", "y": "val"},
            axes=("x", "y"),
            selection=TraceSelectionSpec(
                kind="range", axis_columns={"x": "ts"}, multi="replace"
            ),
        )
        restored = TraceSpec.model_validate(json.loads(ts.model_dump_json()))
        assert restored.selection.kind == "range"
        assert restored.selection.axis_columns == {"x": "ts"}


class TestTypedDicts:
    def test_trace_display_accepts_known_keys(self):
        d: TraceDisplay = {"name": "My Trace", "color": "#ff0000"}
        assert d["name"] == "My Trace"

    def test_trace_params_accepts_known_keys(self):
        p: TraceParams = {"n_points": 1000, "group_by": "sensor"}
        assert p["n_points"] == 1000


# ---- encode_spec / decode_spec --------------------------------------------


class TestEncodeDecodeSpec:
    def test_roundtrip_viz_spec(self):
        original = _make_viz_spec()
        encoded = encode_spec(original)
        assert isinstance(encoded, str)
        assert "=" not in encoded

        decoded = decode_spec(encoded)
        assert isinstance(decoded, VisualizationSpec)
        assert decoded.figure.uid == original.figure.uid
        assert decoded.figure.traces[0].display == {"name": "A", "color": "#1f77b4"}

    def test_roundtrip_dashboard_spec(self):
        original = _make_dashboard_spec()
        decoded = decode_spec(encode_spec(original))

        assert isinstance(decoded, DashboardSpec)
        assert len(decoded.figures) == 2
        assert decoded.figures[0].uid == "fig-1"
        assert decoded.layout.grid_items is not None
        assert decoded.layout.grid_items[1].x == 6

    def test_encode_plain_dict(self):
        original = _make_viz_spec()
        raw = json.loads(original.model_dump_json())
        decoded = decode_spec(encode_spec(raw))

        assert isinstance(decoded, VisualizationSpec)
        assert decoded.figure.uid == original.figure.uid

    def test_auto_detect_dashboard(self):
        """decode_spec detects DashboardSpec by the 'figures' key."""
        dash = _make_dashboard_spec()
        decoded = decode_spec(encode_spec(dash))
        assert isinstance(decoded, DashboardSpec)

    def test_auto_detect_visualization(self):
        viz = _make_viz_spec()
        decoded = decode_spec(encode_spec(viz))
        assert isinstance(decoded, VisualizationSpec)


# ---- LayoutSpec & _auto_grid_items -----------------------------------------


class TestLayoutSpec:
    def test_defaults(self):
        ls = LayoutSpec()
        assert ls.gap == "8px"
        assert ls.draggable is True
        assert ls.grid_editable is False
        assert ls.grid_items is None

    def test_custom(self):
        ls = LayoutSpec(gap="16px")
        assert ls.gap == "16px"

    def test_draggable_fields(self):
        ls = LayoutSpec(
            draggable=True,
            grid_editable=False,
            grid_items=[GridItem(fig_uid="a", x=0, y=0, w=6, h=5)],
        )
        assert ls.draggable is True
        assert ls.grid_editable is False
        assert ls.grid_items is not None
        assert ls.grid_items[0].fig_uid == "a"


class TestAutoGridItems:
    def _make_figs(self, n: int) -> list[FigureSpec]:
        return [FigureSpec(uid=f"fig-{i}", source=None) for i in range(n)]

    def test_default_auto_layout(self):
        figs = self._make_figs(4)
        items = _auto_grid_items(figs)
        # 2-column layout
        assert items[0].x == 0
        assert items[1].x == 6
        assert items[2].x == 0
        assert items[3].x == 6

    def test_rows_seed_derives_columns(self):
        figs = self._make_figs(3)
        items = _auto_grid_items(figs, rows=1)
        assert items[0].x == 0
        assert items[1].x == 4
        assert items[2].x == 8
        assert all(gi.w == 4 for gi in items)

    def test_cols_seed_layout(self):
        figs = self._make_figs(2)
        items = _auto_grid_items(figs, cols=2)
        assert len(items) == 2
        assert items[0].x == 0
        assert items[1].x == 6
        assert all(gi.w == 6 for gi in items)

    def test_rows_cols_mutually_exclusive(self):
        figs = self._make_figs(2)
        with pytest.raises(ValueError, match="mutually exclusive"):
            _auto_grid_items(figs, rows=1, cols=2)

    def test_cols_bound_validation(self):
        figs = self._make_figs(2)
        with pytest.raises(ValueError, match="cols must be <= 12"):
            _auto_grid_items(figs, cols=13)

    def test_grid_items_round_trip_through_spec(self):
        gi = GridItem(fig_uid="abc", x=3, y=2, w=9, h=4)
        ls = LayoutSpec(draggable=True, grid_editable=False, grid_items=[gi])
        dumped = ls.model_dump()
        restored = LayoutSpec.model_validate(dumped)
        assert restored.draggable is True
        assert restored.grid_editable is False
        assert restored.grid_items is not None
        assert restored.grid_items[0].fig_uid == "abc"
        assert restored.grid_items[0].w == 9


# ---- TraceDelta (no trace_index) -------------------------------------------


class TestTraceDelta:
    def test_no_trace_index(self):
        delta = TraceDelta(uid="abc-123", updates={"x": [1, 2], "y": [3, 4]})
        assert delta.uid == "abc-123"
        assert delta.updates == {"x": [1, 2], "y": [3, 4]}
        assert not hasattr(delta, "trace_index")
        assert "trace_index" not in delta.model_dump()

    def test_grouped_children_are_not_recursive_trace_deltas(self):
        child = GroupedChildDelta(
            uid="child-1",
            parent_uid="parent-1",
            group_value_key="A",
            updates={"x": [1], "y": [2]},
        )
        delta = TraceDelta(uid="parent-1", updates={}, group_results=[child])
        dumped = delta.model_dump()
        assert dumped["group_results"][0]["uid"] == "child-1"
        assert "group_results" not in dumped["group_results"][0]

    def test_empty_group_results_are_preserved(self):
        delta = TraceDelta(uid="parent-1", updates={}, group_results=[])
        dumped = delta.model_dump()
        assert dumped["group_results"] == []

    def test_layer_is_serialized_when_present(self):
        delta = TraceDelta(uid="layered-1", updates={"x": [1]}, layer="bg")
        dumped = delta.model_dump()
        assert dumped["layer"] == "bg"


# ---- FigureSpec uid stability ----------------------------------------------


class TestFigureSpecUid:
    def test_uid_stable_across_to_spec(self):
        fig = Figure(pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}))
        fig.add_line(x="x", y="y")

        spec1 = fig.to_spec(source="test")
        spec2 = fig.to_spec(source="test")

        assert spec1.figure.uid == spec2.figure.uid
        assert spec1.figure.uid == fig._uid


# ---- Figure.add_boxplot ----------------------------------------------------


class TestFigureAddBoxplot:
    def test_to_spec_serializes_vertical_box(self):
        fig = Figure(pl.DataFrame({"ts": [0, 1, 2], "val": [1.0, 2.0, 3.0]}))
        fig.add_boxplot(y="val", name="Spread", color="#2ca02c")

        spec = fig.to_spec(source="src")
        assert spec.figure.source == "src"
        assert len(spec.figure.traces) == 1
        ts = spec.figure.traces[0]
        assert ts.trace_type == "box"
        assert ts.backend_data == {"y": "val"}
        assert ts.params == {}
        assert ts.display["name"] == "Spread"
        assert ts.display["color"] == "#2ca02c"
        assert ts.recompute_axes == ()
        assert ts.axes == ("x", "y")

    def test_to_spec_serializes_horizontal_box(self):
        fig = Figure(pl.DataFrame({"val": [1.0, 2.0, 3.0]}))
        fig.add_boxplot(x="val", name="H")

        ts = fig.to_spec(source=None).figure.traces[0]
        assert ts.trace_type == "box"
        assert ts.backend_data == {"x": "val"}
        assert ts.display["name"] == "H"

    def test_uid_stable_across_to_spec(self):
        fig = Figure(pl.DataFrame({"val": [1.0, 2.0, 3.0]}))
        fig.add_boxplot(y="val")

        spec1 = fig.to_spec(source="test")
        spec2 = fig.to_spec(source="test")

        assert spec1.figure.uid == spec2.figure.uid
        assert spec1.figure.uid == fig._uid


# ---- Figure.add_bar -------------------------------------------------------


class TestFigureAddBar:
    def test_basic_bar_spec(self):
        fig = Figure(pl.DataFrame({"cat": ["A", "B"], "val": [1.0, 2.0]}))
        fig.add_bar(labels="cat", values="val")

        spec = fig.to_spec(source="src")
        assert len(spec.figure.traces) == 1
        ts = spec.figure.traces[0]
        assert ts.trace_type == "bar"
        assert ts.backend_data == {"labels": ["cat"], "values": "val"}
        assert ts.params["agg"] == "sum"
        assert ts.params["orientation"] == "v"

    def test_bar_with_tuple_labels_stored_as_list(self):
        fig = Figure(
            pl.DataFrame(
                {
                    "continent": ["Europe", "Asia"],
                    "country": ["Germany", "Japan"],
                    "val": [1.0, 2.0],
                }
            )
        )
        fig.add_bar(labels=("continent", "country"), values="val")

        ts = fig.to_spec(source="src").figure.traces[0]
        assert ts.backend_data["labels"] == ["continent", "country"]

    def test_bar_with_group_by(self):
        fig = Figure(
            pl.DataFrame({"label": ["X", "Y"], "region": ["N", "S"], "val": [1.0, 2.0]})
        )
        fig.add_bar(labels="label", values="val", agg="mean", group_by="region")

        ts = fig.to_spec().figure.traces[0]
        assert ts.params.get("group_by") == ["region"]
        assert ts.params["agg"] == "mean"

    def test_bar_horizontal(self):
        fig = Figure(pl.DataFrame({"cat": ["A"], "val": [1.0]}))
        fig.add_bar(labels="cat", values="val", orientation="h")

        ts = fig.to_spec().figure.traces[0]
        assert ts.params["orientation"] == "h"

    def test_bar_stack_mode(self):
        fig = Figure(pl.DataFrame({"cat": ["A"], "region": ["N"], "val": [1.0]}))
        fig.add_bar(labels="cat", values="val", bar_mode="stack", group_by="region")

        ts = fig.to_spec().figure.traces[0]
        assert ts.display["bar_mode"] == "stack"
        assert "bar_mode" not in ts.params

    def test_mixed_bar_modes_raise(self):
        fig = Figure(
            pl.DataFrame(
                {
                    "cat": ["A", "B"],
                    "region": ["N", "S"],
                    "val": [1.0, 2.0],
                }
            )
        )
        fig.add_bar(labels="cat", values="val", bar_mode="group")
        fig.add_bar(labels="cat", values="val", bar_mode="stack", group_by="region")

        with pytest.raises(ValueError, match="share the same bar_mode"):
            fig.to_spec()


# ---- Figure.add_line / add_histogram / add_boxplot group_by ---------------


class TestFigureGroupByParam:
    def test_add_line_group_by_stored(self):
        fig = Figure(
            pl.DataFrame({"ts": [0, 1], "val": [0.0, 1.0], "sensor": ["A", "B"]})
        )
        fig.add_line(x="ts", y="val", group_by="sensor")
        ts = fig.to_spec(source="src").figure.traces[0]
        assert ts.params.get("group_by") == ["sensor"]
        assert ts.params.get("group_domain_key") == "src::sensor"

    def test_add_histogram_group_by_stored(self):
        fig = Figure(pl.DataFrame({"val": [0.0, 1.0], "cat": ["A", "B"]}))
        fig.add_histogram(x="val", group_by="cat", color_map={"A": "#f00"})
        ts = fig.to_spec(source="src").figure.traces[0]
        assert ts.params.get("group_by") == ["cat"]
        assert ts.params.get("group_domain_key") == "src::cat"
        assert ts.display.get("color_map") == {"A": "#f00"}

    def test_add_boxplot_group_by_stored(self):
        fig = Figure(pl.DataFrame({"val": [0.0, 1.0], "region": ["N", "S"]}))
        fig.add_boxplot(y="val", group_by="region", color_map={"N": "#00f"})
        ts = fig.to_spec().figure.traces[0]
        assert ts.params.get("group_by") == ["region"]
        assert ts.params.get("group_domain_key") == f"{fig._uid}::region"
        assert ts.display.get("color_map") == {"N": "#00f"}

    def test_add_line_group_by_list_stored_with_composite_domain_key(self):
        fig = Figure(
            pl.DataFrame(
                {
                    "ts": [0, 1],
                    "val": [0.0, 1.0],
                    "sensor": ["A", "B"],
                    "site": ["north", "south"],
                }
            )
        )
        fig.add_line(x="ts", y="val", group_by=["sensor", "site"])
        ts = fig.to_spec(source="src").figure.traces[0]
        assert ts.params.get("group_by") == ["sensor", "site"]
        assert ts.params.get("group_domain_key") == 'src::["sensor","site"]'

    def test_add_bar_group_by_tuple_stored_as_list(self):
        fig = Figure(
            pl.DataFrame(
                {
                    "label": ["X", "Y"],
                    "region": ["N", "S"],
                    "site": ["a", "b"],
                    "val": [1.0, 2.0],
                }
            )
        )
        fig.add_bar(labels="label", values="val", group_by=("region", "site"))
        ts = fig.to_spec(source="src").figure.traces[0]
        assert ts.params.get("group_by") == ["region", "site"]
        assert ts.params.get("group_domain_key") == 'src::["region","site"]'


class TestFigureAddHistogram2D:
    def test_to_spec_serializes_heatmap_style(self):
        fig = Figure(pl.DataFrame({"x": [0.0, 1.0], "y": [1.0, 2.0]}))
        fig.add_histogram2d(
            x="x",
            y="y",
            color_scale="plasma",
            color_range=(0.0, 5.0),
        )

        ts = fig.to_spec(source="src").figure.traces[0]
        assert ts.trace_type == "histogram2d"
        assert ts.backend_data == {"x": "x", "y": "y"}
        assert ts.display["color_scale"] == "plasma"
        assert ts.display["color_range"] == (0.0, 5.0)

    def test_to_spec_materializes_default_heatmap_style(self):
        fig = Figure(pl.DataFrame({"x": [0.0, 1.0], "y": [1.0, 2.0]}))
        fig.add_histogram2d(x="x", y="y")

        ts = fig.to_spec(source="src").figure.traces[0]
        assert ts.display["color_scale"] == "viridis"
        assert ts.display["color_range"] == "auto"


class TestFigureAddCorrHeatmap:
    def test_to_spec_serializes_heatmap_style(self):
        fig = Figure(pl.DataFrame({"a": [1.0, 2.0], "b": [2.0, 3.0]}))
        fig.add_corr_heatmap(
            columns=["a", "b"],
            absolute=True,
            color_scale="cividis",
            color_range="auto",
        )

        ts = fig.to_spec(source="src").figure.traces[0]
        assert ts.trace_type == "corr_heatmap"
        assert ts.params["absolute"] is True
        assert ts.display["color_scale"] == "cividis"
        assert ts.display["color_range"] == "auto"

    def test_to_spec_materializes_default_heatmap_style(self):
        fig = Figure(pl.DataFrame({"a": [1.0, 2.0], "b": [2.0, 3.0]}))
        fig.add_corr_heatmap(columns=["a", "b"], absolute=True)

        ts = fig.to_spec(source="src").figure.traces[0]
        assert ts.display["color_scale"] == "viridis"
        assert ts.display["color_range"] == (0.0, 1.0)


class TestFigureHeatmapValidation:
    def test_mixed_heatmap_color_scales_raise(self):
        fig = Figure(pl.DataFrame({"x": [0.0, 1.0], "y": [1.0, 2.0]}))
        fig.add_histogram2d(x="x", y="y", color_scale="viridis")
        fig.add_histogram2d(x="x", y="y", color_scale="plasma")

        with pytest.raises(
            ValueError, match="same effective color_scale and color_range"
        ):
            fig.to_spec()

    def test_mixed_heatmap_color_ranges_raise(self):
        fig = Figure(pl.DataFrame({"x": [0.0, 1.0], "y": [1.0, 2.0]}))
        fig.add_histogram2d(x="x", y="y", color_range="auto")
        fig.add_histogram2d(x="x", y="y", color_range=(0.0, 10.0))

        with pytest.raises(
            ValueError, match="same effective color_scale and color_range"
        ):
            fig.to_spec()


class TestAxisLabelDerivation:
    """to_spec() auto-fills xlabel/ylabel from trace column names when not set."""

    def _layout(self, fig: Figure) -> dict:
        return fig.to_spec().figure.layout

    def test_single_line_derives_both(self):
        fig = Figure(pl.DataFrame({"ts": [1, 2], "val": [1.0, 2.0]}))
        fig.add_line(x="ts", y="val")
        layout = self._layout(fig)
        assert layout["xlabel"] == "ts"
        assert layout["ylabel"] == "val"

    def test_histogram_x_derives_xlabel_and_ylabel(self):
        fig = Figure(pl.DataFrame({"age": [20, 30, 40]}))
        fig.add_histogram(x="age")
        layout = self._layout(fig)
        assert layout["xlabel"] == "age"
        assert layout["ylabel"] == "count"  # histnorm fills the count axis

    def test_boxplot_y_derives_ylabel_only(self):
        fig = Figure(pl.DataFrame({"value": [1.0, 2.0, 3.0]}))
        fig.add_boxplot(y="value")
        layout = self._layout(fig)
        assert layout["ylabel"] == "value"
        assert "xlabel" not in layout

    def test_two_lines_same_x_derives_xlabel_only(self):
        fig = Figure(pl.DataFrame({"ts": [1, 2], "a": [1.0, 2.0], "b": [3.0, 4.0]}))
        fig.add_line(x="ts", y="a")
        fig.add_line(x="ts", y="b")
        layout = self._layout(fig)
        assert layout["xlabel"] == "ts"
        assert "ylabel" not in layout  # two different y-columns → no consensus

    def test_two_lines_different_x_no_labels(self):
        fig = Figure(
            pl.DataFrame({"t1": [1, 2], "t2": [3, 4], "a": [1.0, 2.0], "b": [3.0, 4.0]})
        )
        fig.add_line(x="t1", y="a")
        fig.add_line(x="t2", y="b")
        layout = self._layout(fig)
        assert "xlabel" not in layout
        assert "ylabel" not in layout

    def test_explicit_xlabel_not_overwritten(self):
        fig = Figure(pl.DataFrame({"ts": [1, 2], "val": [1.0, 2.0]}))
        fig.add_line(x="ts", y="val")
        fig.xlabel("Time [s]")
        layout = self._layout(fig)
        assert layout["xlabel"] == "Time [s]"
        assert layout["ylabel"] == "val"  # ylabel still derived

    def test_explicit_ylabel_not_overwritten(self):
        fig = Figure(pl.DataFrame({"ts": [1, 2], "val": [1.0, 2.0]}))
        fig.add_line(x="ts", y="val")
        fig.ylabel("Temperature [°C]")
        layout = self._layout(fig)
        assert layout["xlabel"] == "ts"  # xlabel still derived
        assert layout["ylabel"] == "Temperature [°C]"

    def test_pie_no_labels(self):
        fig = Figure(pl.DataFrame({"cat": ["A", "B"], "val": [1.0, 2.0]}))
        fig.add_pie(labels="cat", values="val")
        layout = self._layout(fig)
        assert "xlabel" not in layout
        assert "ylabel" not in layout

    def test_corr_heatmap_no_labels(self):
        fig = Figure(pl.DataFrame({"a": [1.0, 2.0], "b": [2.0, 3.0]}))
        fig.add_corr_heatmap(columns=["a", "b"])
        layout = self._layout(fig)
        assert "xlabel" not in layout
        assert "ylabel" not in layout

    def test_mixed_line_and_pie_derives_from_line(self):
        fig = Figure(pl.DataFrame({"ts": [1, 2], "val": [1.0, 2.0], "cat": ["A", "B"]}))
        fig.add_line(x="ts", y="val")
        fig.add_pie(labels="cat", values="val")
        layout = self._layout(fig)
        assert layout["xlabel"] == "ts"
        assert layout["ylabel"] == "val"

    def test_hist2d_derives_both(self):
        fig = Figure(pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}))
        fig.add_histogram2d(x="a", y="b")
        layout = self._layout(fig)
        assert layout["xlabel"] == "a"
        assert layout["ylabel"] == "b"

    def test_does_not_mutate_figure_layout(self):
        """to_spec() must not write back into Figure._layout."""
        fig = Figure(pl.DataFrame({"ts": [1, 2], "val": [1.0, 2.0]}))
        fig.add_line(x="ts", y="val")
        fig.to_spec()
        assert "xlabel" not in fig._layout
        assert "ylabel" not in fig._layout

    # --- BarPlot ---

    def test_bar_sum_derives_ylabel(self):
        fig = Figure(pl.DataFrame({"cat": ["A", "B"], "revenue": [10.0, 20.0]}))
        fig.add_bar(labels="cat", values="revenue", agg="sum")
        assert self._layout(fig)["ylabel"] == "sum of revenue"

    def test_bar_count_derives_ylabel_without_column(self):
        fig = Figure(pl.DataFrame({"cat": ["A", "B"]}))
        fig.add_bar(labels="cat")  # values=None → implicit count
        assert self._layout(fig)["ylabel"] == "count"

    def test_bar_mean_derives_ylabel(self):
        fig = Figure(pl.DataFrame({"cat": ["A", "B"], "temp": [10.0, 20.0]}))
        fig.add_bar(labels="cat", values="temp", agg="mean")
        assert self._layout(fig)["ylabel"] == "mean of temp"

    def test_two_bars_same_agg_and_col_derives_ylabel(self):
        df = pl.DataFrame({"cat": ["A", "B"], "val": [1.0, 2.0]})
        fig = Figure(df)
        fig.add_bar(labels="cat", values="val", agg="sum")
        fig.add_bar(labels="cat", values="val", agg="sum")
        assert self._layout(fig)["ylabel"] == "sum of val"

    def test_two_bars_different_agg_no_ylabel(self):
        df = pl.DataFrame({"cat": ["A", "B"], "val": [1.0, 2.0]})
        fig = Figure(df)
        fig.add_bar(labels="cat", values="val", agg="sum")
        fig.add_bar(labels="cat", values="val", agg="mean")
        assert "ylabel" not in self._layout(fig)

    # --- Histogram ---

    def test_histogram_count_derives_ylabel(self):
        fig = Figure(pl.DataFrame({"age": [20, 30, 40]}))
        fig.add_histogram(x="age", histnorm="count")
        assert self._layout(fig)["ylabel"] == "count"

    def test_histogram_density_derives_ylabel(self):
        fig = Figure(pl.DataFrame({"age": [20, 30, 40]}))
        fig.add_histogram(x="age", histnorm="density")
        assert self._layout(fig)["ylabel"] == "density"

    def test_histogram_y_orientation_derives_xlabel(self):
        fig = Figure(pl.DataFrame({"score": [1.0, 2.0, 3.0]}))
        fig.add_histogram(y="score")
        assert self._layout(fig)["xlabel"] == "count"


class TestAutoTitleDerivation:
    """to_spec() auto-fills title from aggregation info when not set."""

    def _layout(self, fig: Figure) -> dict:
        return fig.to_spec().figure.layout

    # --- PiePlot ---

    def test_pie_sum_derives_title(self):
        fig = Figure(pl.DataFrame({"cat": ["A", "B"], "val": [10.0, 20.0]}))
        fig.add_pie(labels="cat", values="val", agg="sum")
        assert self._layout(fig)["title"] == "sum of val"

    def test_pie_count_derives_title_with_labels_column(self):
        fig = Figure(pl.DataFrame({"cat": ["A", "B"]}))
        fig.add_pie(labels="cat")  # values=None → implicit count
        assert self._layout(fig)["title"] == "count of cat"

    def test_pie_count_with_multi_labels_derives_title(self):
        fig = Figure(pl.DataFrame({"continent": ["Europe"], "country": ["Germany"]}))
        fig.add_pie(labels=["continent", "country"])
        assert self._layout(fig)["title"] == "count of [continent, country]"

    # --- Consensus logic ---

    def test_two_bars_different_y_no_title(self):
        df = pl.DataFrame({"cat": ["A", "B"], "a": [1.0, 2.0], "b": [3.0, 4.0]})
        fig = Figure(df)
        fig.add_bar(labels="cat", values="a", agg="sum")
        fig.add_bar(labels="cat", values="b", agg="sum")
        assert "title" not in self._layout(fig)

    def test_two_bars_different_agg_no_title(self):
        df = pl.DataFrame({"cat": ["A", "B"], "val": [1.0, 2.0]})
        fig = Figure(df)
        fig.add_bar(labels="cat", values="val", agg="sum")
        fig.add_bar(labels="cat", values="val", agg="mean")
        assert "title" not in self._layout(fig)

    def test_bar_and_pie_same_candidate_no_title(self):
        df = pl.DataFrame({"cat": ["A", "B"], "val": [1.0, 2.0]})
        fig = Figure(df)
        fig.add_bar(labels="cat", values="val", agg="sum")
        fig.add_pie(labels="cat", values="val", agg="sum")
        assert "title" not in self._layout(fig)

    def test_bar_and_pie_different_candidates_no_title(self):
        df = pl.DataFrame({"cat": ["A", "B"], "val": [1.0, 2.0], "n": [1, 2]})
        fig = Figure(df)
        fig.add_bar(labels="cat", values="val", agg="sum")
        fig.add_pie(labels="cat", values="n", agg="mean")
        assert "title" not in self._layout(fig)

    # --- User override ---

    def test_explicit_title_not_overwritten(self):
        fig = Figure(pl.DataFrame({"cat": ["A", "B"], "val": [1.0, 2.0]}))
        fig.add_bar(labels="cat", values="val", agg="sum")
        fig.title("My Custom Title")
        assert self._layout(fig)["title"] == "My Custom Title"

    def test_line_only_figure_no_title(self):
        fig = Figure(pl.DataFrame({"ts": [1, 2], "val": [1.0, 2.0]}))
        fig.add_line(x="ts", y="val")
        assert "title" not in self._layout(fig)

    def test_does_not_mutate_figure_layout(self):
        """to_spec() must not write back into Figure._layout."""
        fig = Figure(pl.DataFrame({"cat": ["A", "B"], "val": [1.0, 2.0]}))
        fig.add_bar(labels="cat", values="val", agg="sum")
        fig.to_spec()
        assert "title" not in fig._layout


# ---- Pydantic Literal field validation -------------------------------------


class TestPydanticLiteralValidation:
    def test_invalid_cross_filter_mode_raises(self):
        with pytest.raises(ValidationError):
            InteractionState(cross_filter_mode="invalid")  # type: ignore[arg-type]

    def test_valid_cross_filter_modes_accepted(self):
        assert (
            InteractionState(cross_filter_mode="update").cross_filter_mode == "update"
        )
        assert (
            InteractionState(cross_filter_mode="overlay").cross_filter_mode == "overlay"
        )

    def test_interaction_state_validates_from_dict(self):
        with pytest.raises(ValidationError):
            InteractionState.model_validate({"cross_filter_mode": "unknown"})

    def test_axis_locks_round_trip(self):
        # Axis locks are client-only state: they live on ClientState (never read
        # by the server engine), not on the server-facing InteractionState.
        client_state = ClientState(
            axis_locks={"fig-a/x": True, "fig-b/y": True},
            axis_lock_ranges={
                "fig-a/x": {"min": 0, "max": 10},
                "fig-b/y": {"min": "Mon", "max": "Sun"},
            },
        )
        dash = DashboardSpec(figures=[], client_state=client_state)
        encoded = encode_spec(dash)
        decoded = decode_spec(encoded)
        cs = decoded.client_state
        assert cs.axis_locks == {"fig-a/x": True, "fig-b/y": True}
        assert cs.axis_lock_ranges["fig-a/x"].as_tuple() == (0, 10)
        assert cs.axis_lock_ranges["fig-b/y"].as_tuple() == ("Mon", "Sun")

    def test_axis_locks_not_on_interaction_state(self):
        # Guard against re-adding axis-lock fields to the server-facing state.
        assert "axis_locks" not in InteractionState.model_fields
        assert "axis_lock_ranges" not in InteractionState.model_fields


# ---- Edge-case spec construction -------------------------------------------


class TestEdgeCaseConstruction:
    def test_figure_spec_default_uid_is_unique(self):
        a = FigureSpec()
        b = FigureSpec()
        assert a.uid != b.uid

    def test_trace_spec_default_values(self):
        ts = TraceSpec(uid="t1", trace_type="line")
        assert ts.backend_data == {}
        assert ts.params == {}
        assert ts.display == {}
        assert ts.recompute_axes is None
        assert ts.axes is None

    def test_trace_spec_backend_data_accepts_only_string_columns_or_column_lists(self):
        ts = TraceSpec(
            uid="t1",
            trace_type="bar",
            backend_data={"labels": ["continent", "country"], "values": "revenue"},
        )
        assert ts.backend_data == {
            "labels": ["continent", "country"],
            "values": "revenue",
        }

        with pytest.raises(ValidationError):
            TraceSpec(uid="t2", trace_type="line", backend_data={"x": 123})

    def test_group_domain_state_defaults(self):
        g = GroupDomainState()
        assert g.mapping == {}
        assert g.next_color_index == 0

    def test_interaction_state_group_domains_roundtrip(self):
        state = InteractionState(
            group_domains={
                "src::sensor": GroupDomainState(
                    mapping={"A": "#1f77b4", "B": "#ff7f0e"},
                    next_color_index=2,
                )
            }
        )
        raw = json.loads(state.model_dump_json())
        restored = InteractionState.model_validate(raw)
        assert restored.group_domains["src::sensor"].mapping == {
            "A": "#1f77b4",
            "B": "#ff7f0e",
        }
        assert restored.group_domains["src::sensor"].next_color_index == 2

    def test_visualization_spec_version_field_preserved(self):
        spec = _make_viz_spec()
        raw = json.loads(spec.model_dump_json())
        assert raw["version"] == spec.version
        restored = VisualizationSpec.model_validate(raw)
        assert restored.version == spec.version


# ---- encode/decode with full InteractionState ------------------------------


class TestEncodeDecodeFullState:
    def test_roundtrip_preserves_selection_state(self):
        from flexviz.spec import ClauseFilter, SelectionPredicate

        spec = _make_viz_spec()
        spec.state.selections = [
            SelectionState(
                source_figure_uid="fig-bbb",
                predicates=[
                    SelectionPredicate(
                        clauses=[ClauseFilter(column="ts", range=(10.0, 50.0))]
                    )
                ],
            )
        ]
        decoded = decode_spec(encode_spec(spec))
        assert isinstance(decoded, VisualizationSpec)
        assert len(decoded.state.selections) == 1
        sel = decoded.state.selections[0]
        assert sel.predicates[0].clauses[0].range == (10.0, 50.0)
        assert sel.source_figure_uid == "fig-bbb"

    def test_roundtrip_preserves_closed_left_clause(self):
        from flexviz.spec import ClauseFilter, SelectionPredicate

        spec = _make_viz_spec()
        spec.state.selections = [
            SelectionState(
                source_figure_uid="fig-bbb",
                predicates=[
                    SelectionPredicate(
                        clauses=[
                            ClauseFilter(column="ts", range=(10.0, 50.0), closed="left")
                        ]
                    )
                ],
            )
        ]
        decoded = decode_spec(encode_spec(spec))
        assert isinstance(decoded, VisualizationSpec)
        assert decoded.state.selections[0].predicates[0].clauses[0].closed == "left"

    def test_roundtrip_preserves_categorical_selection(self):
        from flexviz.spec import ClauseFilter, SelectionPredicate

        spec = _make_viz_spec()
        spec.state.selections = [
            SelectionState(
                source_figure_uid="fig-bbb",
                predicates=[
                    SelectionPredicate(
                        clauses=[ClauseFilter(column="cat", values=["A", "B"])]
                    )
                ],
            )
        ]
        decoded = decode_spec(encode_spec(spec))
        assert isinstance(decoded, VisualizationSpec)
        assert decoded.state.selections[0].predicates[0].clauses[0].values == ["A", "B"]

    def test_roundtrip_preserves_viewport(self):
        spec = _make_viz_spec()
        spec.state.viewport = {"x": AxisRange(min=100.0, max=200.0)}
        decoded = decode_spec(encode_spec(spec))
        assert isinstance(decoded, VisualizationSpec)
        vp = decoded.state.viewport["x"]
        assert isinstance(vp, AxisRange)
        assert vp.min == 100.0
        assert vp.max == 200.0

    def test_roundtrip_preserves_cross_filter_mode_overlay(self):
        dash = _make_dashboard_spec()
        dash.state.cross_filter_mode = "overlay"
        decoded = decode_spec(encode_spec(dash))
        assert isinstance(decoded, DashboardSpec)
        assert decoded.state.cross_filter_mode == "overlay"

    def test_roundtrip_preserves_group_domains(self):
        spec = _make_viz_spec()
        spec.state.group_domains = {
            "mydata::sensor": GroupDomainState(
                mapping={"A": "#1f77b4"}, next_color_index=1
            )
        }
        decoded = decode_spec(encode_spec(spec))
        assert isinstance(decoded, VisualizationSpec)
        gd = decoded.state.group_domains["mydata::sensor"]
        assert gd.mapping == {"A": "#1f77b4"}
        assert gd.next_color_index == 1

    def test_encode_is_url_safe(self):
        """Encoded string must contain no URL-unsafe characters."""
        spec = _make_viz_spec()
        encoded = encode_spec(spec)
        import re

        assert re.fullmatch(
            r"[A-Za-z0-9_\-]+", encoded
        ), f"Encoded spec contains URL-unsafe chars: {encoded[:80]}"


class TestClauseFilter:
    def test_range_only_clause(self):
        from flexviz.spec import ClauseFilter

        c = ClauseFilter(column="ts", range=(0.0, 10.0))
        assert c.column == "ts"
        assert c.range == (0.0, 10.0)
        assert c.values is None

    def test_values_only_clause(self):
        from flexviz.spec import ClauseFilter

        c = ClauseFilter(column="country", values=["NL", "BE"])
        assert c.values == ["NL", "BE"]
        assert c.range is None

    def test_clause_rejects_both_range_and_values(self):
        from flexviz.spec import ClauseFilter
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ClauseFilter(column="x", range=(0, 1), values=["a"])

    def test_clause_rejects_neither(self):
        from flexviz.spec import ClauseFilter
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ClauseFilter(column="x")

    def test_closed_left_range_clause(self):
        from flexviz.spec import ClauseFilter

        c = ClauseFilter(column="ts", range=(0.0, 10.0), closed="left")
        assert c.closed == "left"
        assert ClauseFilter.model_validate(c.model_dump()) == c

    def test_closed_defaults_to_both_for_legacy_payload(self):
        from flexviz.spec import ClauseFilter

        c = ClauseFilter.model_validate({"column": "ts", "range": [0.0, 10.0]})
        assert c.closed == "both"

    def test_clause_rejects_values_with_closed_left(self):
        from flexviz.spec import ClauseFilter
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ClauseFilter(column="country", values=["NL"], closed="left")


class TestSelectionState:
    def test_empty_selection(self):
        from flexviz.spec import SelectionState

        s = SelectionState(source_figure_uid="fig1")
        assert s.predicates == []
        assert s.source_figure_uid == "fig1"

    def test_range_selection(self):
        from flexviz.spec import ClauseFilter, SelectionPredicate, SelectionState

        s = SelectionState(
            source_figure_uid="fig1",
            predicates=[
                SelectionPredicate(
                    clauses=[
                        ClauseFilter(column="ts", range=(0, 10)),
                        ClauseFilter(column="val", range=(0.0, 1.0)),
                    ]
                )
            ],
        )
        assert len(s.predicates) == 1
        assert len(s.predicates[0].clauses) == 2

    def test_categorical_selection(self):
        from flexviz.spec import ClauseFilter, SelectionPredicate, SelectionState

        s = SelectionState(
            source_figure_uid="fig1",
            predicates=[
                SelectionPredicate(
                    clauses=[ClauseFilter(column="country", values=["NL"])]
                )
            ],
        )
        assert s.predicates[0].clauses[0].values == ["NL"]

    def test_treemap_path_selection(self):
        from flexviz.spec import ClauseFilter, SelectionPredicate, SelectionState

        s = SelectionState(
            source_figure_uid="fig1",
            predicates=[
                SelectionPredicate(
                    clauses=[
                        ClauseFilter(column="source", values=["solar"]),
                        ClauseFilter(column="country", values=["NL"]),
                    ]
                )
            ],
        )
        assert [c.column for c in s.predicates[0].clauses] == ["source", "country"]

    def test_round_trip_through_json(self):
        from flexviz.spec import (
            ClauseFilter,
            SelectionPredicate,
            SelectionState,
        )

        s = SelectionState(
            source_figure_uid="fig1",
            predicates=[
                SelectionPredicate(clauses=[ClauseFilter(column="x", range=(0, 1))]),
                SelectionPredicate(
                    clauses=[ClauseFilter(column="y", values=["A", "B"])]
                ),
            ],
        )
        restored = SelectionState.model_validate_json(s.model_dump_json())
        assert restored == s

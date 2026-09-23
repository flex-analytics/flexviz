"""Dashboard.link_axes: three input forms resolve to one set of link groups."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from flexviz.dashboard import Dashboard
from flexviz.spec import decode_spec, encoded_spec_from_url


@pytest.fixture()
def df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts": [1, 2, 3],
            "val": [1.0, 2.0, 3.0],
            "day": [dt.date(2024, 1, d) for d in (1, 2, 3)],
            "at": [dt.datetime(2024, 1, d, 6) for d in (1, 2, 3)],
            "name": ["a", "b", "c"],
        }
    )


def _four(df: pl.DataFrame):
    """Two lines on ts, a vertical and a horizontal histogram of ts."""
    dash = Dashboard(df)
    a = dash.add_figure()
    a.add_line(x="ts", y="val")
    b = dash.add_figure()
    b.add_line(x="ts", y="val")
    h = dash.add_figure()
    h.add_histogram(x="ts")
    hy = dash.add_figure()
    hy.add_histogram(y="ts")
    return dash, a, b, h, hy


def _links(dash: Dashboard) -> list[list[str]]:
    return dash.to_spec().client_state.axis_links


class TestForms:
    def test_on_links_every_figure_showing_the_column(self, df):
        dash, a, b, h, hy = _four(df)
        other = dash.add_figure()
        other.add_line(x="val", y="at")
        assert dash.link_axes(on="ts") is dash
        assert _links(dash) == [
            [f"{a._uid}/x", f"{b._uid}/x", f"{h._uid}/x", f"{hy._uid}/y"]
        ]

    def test_on_with_figures_links_only_those(self, df):
        dash, a, _, h, _ = _four(df)
        dash.link_axes(a, h, on="ts")
        assert _links(dash) == [[f"{a._uid}/x", f"{h._uid}/x"]]

    def test_axis_links_the_same_axis_of_each_figure(self, df):
        dash, a, b, *_ = _four(df)
        dash.link_axes(a, b, axis="y")
        assert _links(dash) == [[f"{a._uid}/y", f"{b._uid}/y"]]

    def test_pairs_link_the_named_axes(self, df):
        dash, a, _, _, hy = _four(df)
        dash.link_axes((a, "x"), (hy, "y"))
        assert _links(dash) == [[f"{a._uid}/x", f"{hy._uid}/y"]]

    def test_links_resolve_when_the_spec_is_built(self, df):
        dash = Dashboard(df)
        dash.link_axes(on="ts")
        a = dash.add_figure()
        b = dash.add_figure()
        a.add_line(x="ts", y="val")
        b.add_histogram(x="ts")
        assert _links(dash) == [[f"{a._uid}/x", f"{b._uid}/x"]]

    def test_date_and_datetime_axes_link(self, df):
        dash = Dashboard(df)
        a = dash.add_figure()
        a.add_line(x="day", y="val")
        b = dash.add_figure()
        b.add_line(x="at", y="val")
        dash.link_axes(a, b, axis="x")
        assert len(_links(dash)[0]) == 2


class TestMerge:
    def test_calls_sharing_an_axis_merge(self, df):
        dash, a, b, h, _ = _four(df)
        dash.link_axes(a, b, axis="x").link_axes((b, "x"), (h, "x"))
        assert _links(dash) == [[f"{a._uid}/x", f"{b._uid}/x", f"{h._uid}/x"]]

    def test_repeated_and_disjoint_calls(self, df):
        dash, a, b, *_ = _four(df)
        dash.link_axes(a, b, axis="x").link_axes(a, b, axis="x")
        dash.link_axes(a, b, axis="y")
        assert _links(dash) == [
            [f"{a._uid}/x", f"{b._uid}/x"],
            [f"{a._uid}/y", f"{b._uid}/y"],
        ]


class TestErrors:
    @pytest.mark.parametrize(
        ("call", "match"),
        [
            (lambda d, a, b: d.link_axes(a, b), "exactly one of on= or axis="),
            (lambda d, a, b: d.link_axes(a, b, on="ts", axis="x"), "exactly one"),
            (lambda d, a, b: d.link_axes(a, (b, "x")), "not both"),
            (lambda d, a, b: d.link_axes((a, "x"), (b, "x"), on="ts"), "drop on="),
            (lambda d, a, b: d.link_axes((a, "x")), "two or more"),
            (lambda d, a, b: d.link_axes(a, axis="x"), "two or more figures"),
        ],
        ids=["no-mode", "both-modes", "mixed", "pairs-with-on", "one-pair", "one-fig"],
    )
    def test_bad_arguments(self, df, call, match):
        dash, a, b, *_ = _four(df)
        with pytest.raises(ValueError, match=match):
            call(dash, a, b)

    def test_figure_of_another_dashboard(self, df):
        dash, a, *_ = _four(df)
        stranger = Dashboard(df).add_figure()
        with pytest.raises(ValueError, match="not in this dashboard"):
            dash.link_axes(a, stranger, axis="x")

    def test_named_figure_without_the_column(self, df):
        dash, a, *_ = _four(df)
        other = dash.add_figure()
        other.add_line(x="val", y="at")
        dash.link_axes(a, other, on="ts")
        with pytest.raises(ValueError, match="shows no 'ts' axis"):
            dash.to_spec()

    def test_column_on_both_axes_of_a_figure(self, df):
        dash, *_ = _four(df)
        both = dash.add_figure()
        both.add_line(x="ts", y="val")
        both.add_line(x="val", y="ts")
        dash.link_axes(on="ts")
        with pytest.raises(ValueError, match="name one with"):
            dash.to_spec()

    def test_fewer_than_two_axes_on_the_column(self, df):
        dash = Dashboard(df)
        dash.add_figure().add_line(x="ts", y="val")
        dash.link_axes(on="ts")
        with pytest.raises(ValueError, match="fewer than two"):
            dash.to_spec()

    def test_count_axis(self, df):
        dash, a, _, h, _ = _four(df)
        dash.link_axes((a, "y"), (h, "y"))
        with pytest.raises(ValueError, match="shows no data column"):
            dash.to_spec()

    def test_log_axis(self, df):
        dash, a, b, *_ = _four(df)
        a.update_layout(xaxis={"type": "log"})
        dash.link_axes(a, b, axis="x")
        with pytest.raises(ValueError, match="log axis"):
            dash.to_spec()

    def test_string_column(self, df):
        dash = Dashboard(df)
        a = dash.add_figure()
        a.add_line(x="name", y="val")
        b = dash.add_figure()
        b.add_line(x="ts", y="val")
        dash.link_axes(a, b, axis="x")
        with pytest.raises(ValueError, match="only numeric, Date and Datetime"):
            dash.to_spec()

    @pytest.mark.parametrize("col", ["tod", "took"])
    def test_time_and_duration_axes(self, df, col):
        """Plotly draws these as category axes, whose ranges are positions."""
        df = df.with_columns(
            tod=pl.col("at").dt.time(), took=pl.col("at") - pl.col("at").min()
        )
        dash = Dashboard(df)
        a = dash.add_figure()
        a.add_line(x=col, y="val")
        b = dash.add_figure()
        b.add_line(x=col, y="val")
        dash.link_axes(on=col)
        with pytest.raises(ValueError, match="only numeric, Date and Datetime"):
            dash.to_spec()

    def test_mixed_time_zones(self, df):
        """A range is copied as wall-clock text: one window, two instants."""
        df = df.with_columns(
            utc=pl.col("at").dt.replace_time_zone("UTC"),
            bxl=pl.col("at").dt.replace_time_zone("Europe/Brussels"),
        )
        dash = Dashboard(df)
        a = dash.add_figure()
        a.add_line(x="utc", y="val")
        b = dash.add_figure()
        b.add_line(x="bxl", y="val")
        dash.link_axes(a, b, axis="x")
        with pytest.raises(ValueError, match="mix time zones"):
            dash.to_spec()

    def test_two_axes_of_one_figure(self, df):
        dash, a, *_ = _four(df)
        dash.link_axes((a, "x"), (a, "y"))
        with pytest.raises(ValueError, match="two axes of one figure"):
            dash.to_spec()

    def test_numeric_and_temporal_do_not_mix(self, df):
        dash = Dashboard(df)
        a = dash.add_figure()
        a.add_line(x="ts", y="val")
        b = dash.add_figure()
        b.add_line(x="day", y="val")
        dash.link_axes(a, b, axis="x")
        with pytest.raises(ValueError, match="mix numeric and temporal"):
            dash.to_spec()


class TestRoundTrip:
    def test_save_load_and_share_keep_links(self, df, tmp_path):
        dash, *_ = _four(df)
        dash.link_axes(on="ts")
        expected = _links(dash)

        path = tmp_path / "spec.json"
        dash.save_spec(str(path))
        assert Dashboard.load_spec(str(path)).client_state.axis_links == expected

        url = dash.share_url(source_name="data")
        shared = decode_spec(encoded_spec_from_url(url))
        assert shared.client_state.axis_links == expected

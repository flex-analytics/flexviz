"""CLI and share_url tests: URL round-trip, file registration, error paths."""

import json

import polars as pl
import pytest

from flexviz import Dashboard
from flexviz.cli import _register_files, main
from flexviz.spec import DashboardSpec, decode_spec


def _demo_dashboard(**dash_kw) -> Dashboard:
    lf = pl.LazyFrame({"t": [1, 2, 3], "v": [1.0, 2.0, 3.0]})
    dash = Dashboard(lf, **dash_kw)
    dash.add_figure().add_line(x="t", y="v")
    dash.add_figure().add_histogram(x="v")
    return dash


def test_share_url_round_trip():
    url = _demo_dashboard().share_url(
        server_url="http://127.0.0.1:9999/", source_name="demo"
    )
    assert url.startswith("http://127.0.0.1:9999/view?spec=")
    spec = decode_spec(url.split("spec=", 1)[1])
    assert isinstance(spec, DashboardSpec)
    assert [f.source for f in spec.figures] == ["demo", "demo"]
    # grid is seeded so /view renders a layout without a prior show()
    assert spec.layout.grid_items
    # cache defaults off, so live brushing must resolve to off
    assert spec.client_state.live_brush == "off"


def test_share_url_cache_enables_live_brush():
    url = _demo_dashboard(cache=True).share_url(source_name="demo")
    spec = decode_spec(url.split("spec=", 1)[1])
    assert spec.client_state.live_brush == "auto"


def test_decode_command_accepts_full_url(capsys):
    url = _demo_dashboard().share_url(source_name="demo")
    main(["decode", url])
    payload = json.loads(capsys.readouterr().out)
    assert [f["source"] for f in payload["figures"]] == ["demo", "demo"]


def test_decode_command_rejects_url_without_spec():
    with pytest.raises(SystemExit):
        main(["decode", "http://127.0.0.1:8000/view?other=1"])


def test_register_files_names_by_stem(tmp_path):
    path = tmp_path / "readings.parquet"
    pl.DataFrame({"x": [1, 2]}).write_parquet(path)
    assert _register_files([str(path)], cache=False) == ["readings"]

    from flexviz.server import _sources

    assert "readings" in _sources


def test_register_files_rejects_duplicate_stems(tmp_path):
    path = tmp_path / "dup.parquet"
    pl.DataFrame({"x": [1]}).write_parquet(path)
    with pytest.raises(SystemExit, match="duplicate source name"):
        _register_files([str(path), str(path)], cache=False)


def test_register_files_rejects_missing_and_unknown(tmp_path):
    with pytest.raises(SystemExit, match="file not found"):
        _register_files([str(tmp_path / "absent.parquet")], cache=False)

    bad = tmp_path / "data.xlsx"
    bad.write_text("x")
    with pytest.raises(SystemExit, match="unsupported file type"):
        _register_files([str(bad)], cache=False)

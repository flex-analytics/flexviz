"""CLI and share_url tests: URL round-trip, file registration, error paths."""

import json
from pathlib import Path

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


def test_schema_command_emits_json(capsys, tmp_path):
    path = tmp_path / "readings.parquet"
    pl.DataFrame({"t": [1, 2], "v": [1.0, 2.0]}).write_parquet(path)
    main(["schema", str(path)])
    out = json.loads(capsys.readouterr().out)
    assert out[0]["source_name"] == "readings"
    assert {"name": "t", "dtype": "Int64"} in out[0]["columns"]
    assert {"name": "v", "dtype": "Float64"} in out[0]["columns"]


def _skill_paths(base):
    return [
        base / target / "flexviz-explore" / "SKILL.md"
        for target in (".agents/skills", ".claude/skills")
    ]


def test_skill_install_fresh(capsys, tmp_path):
    main(["skill", "install", "--dir", str(tmp_path)])
    for skill in _skill_paths(tmp_path):
        assert skill.exists(), skill
        assert skill.read_text().startswith("---\nname: flexviz-explore")
    assert capsys.readouterr().out.count("installed") == 2


def test_skill_install_identical_is_noop(capsys, tmp_path):
    main(["skill", "install", "--dir", str(tmp_path)])
    capsys.readouterr()
    main(["skill", "install", "--dir", str(tmp_path)])
    assert capsys.readouterr().out.count("unchanged") == 2


def test_skill_install_refuses_modified_without_force(capsys, tmp_path):
    main(["skill", "install", "--dir", str(tmp_path)])
    modified = _skill_paths(tmp_path)[0]
    modified.write_text("my customized skill")
    with pytest.raises(SystemExit, match="not overwriting"):
        main(["skill", "install", "--dir", str(tmp_path)])
    assert modified.read_text() == "my customized skill"


def test_skill_install_force_replaces(capsys, tmp_path):
    main(["skill", "install", "--dir", str(tmp_path)])
    modified = _skill_paths(tmp_path)[0]
    modified.write_text("my customized skill")
    main(["skill", "install", "--dir", str(tmp_path), "--force"])
    assert modified.read_text().startswith("---\nname: flexviz-explore")


def test_csv_dates_are_parsed(capsys, tmp_path):
    path = tmp_path / "events.csv"
    path.write_text("ts,val\n2026-01-01 10:00:00,1.5\n2026-01-01 10:00:02,2.5\n")
    main(["schema", str(path)])
    out = json.loads(capsys.readouterr().out)
    dtypes = {c["name"]: c["dtype"] for c in out[0]["columns"]}
    assert dtypes["ts"].startswith("Datetime"), dtypes


# ---------------------------------------------------------------------------
# Boundary tests: the installed command, as a real process
# ---------------------------------------------------------------------------


def _run_module(*argv: str, **kw):
    import subprocess
    import sys

    return subprocess.run(
        [sys.executable, "-m", "flexviz", *argv],
        capture_output=True,
        text=True,
        timeout=60,
        **kw,
    )


def test_module_help_and_decode_subprocess():
    assert _run_module("--help").returncode == 0

    url = _demo_dashboard().share_url(source_name="demo")
    proc = _run_module("decode", url)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert len(payload["figures"]) == 2


def test_serve_lifecycle_subprocess(tmp_path):
    import socket
    import subprocess
    import sys
    import time

    import requests

    path = tmp_path / "life.parquet"
    pl.DataFrame({"x": [1, 2, 3]}).write_parquet(path)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    proc = subprocess.Popen(
        [sys.executable, "-m", "flexviz", "serve", str(path), "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.time() + 30
        sources = None
        while time.time() < deadline:
            if proc.poll() is not None:
                pytest.fail(f"serve exited early: {proc.communicate()}")
            try:
                sources = requests.get(
                    f"http://127.0.0.1:{port}/sources", timeout=1
                ).json()
                break
            except requests.RequestException:
                time.sleep(0.2)
        assert sources == ["life"]
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_serve_fails_fast_on_busy_port(tmp_path):
    import socket

    path = tmp_path / "busy.parquet"
    pl.DataFrame({"x": [1]}).write_parquet(path)
    with socket.socket() as blocker:
        blocker.bind(("127.0.0.1", 0))
        blocker.listen(1)
        port = blocker.getsockname()[1]
        proc = _run_module("serve", str(path), "--port", str(port))
        assert proc.returncode != 0
        assert "cannot bind" in proc.stderr


def test_skill_install_user_scope(capsys, monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    main(["skill", "install", "--user"])
    for skill in _skill_paths(tmp_path):
        assert skill.exists(), skill
    assert capsys.readouterr().out.count("installed") == 2


def test_skill_install_scope_flags_are_exclusive():
    with pytest.raises(SystemExit):
        main(["skill", "install", "--user", "--dir", "."])

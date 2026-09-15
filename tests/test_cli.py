"""CLI and share_url tests: URL round-trip, file registration, error paths."""

import json
from pathlib import Path

import polars as pl
import pytest

from flexviz import Dashboard, Figure
from flexviz.cli import _register_files, main
from flexviz.spec import AxisRange, DashboardSpec, decode_spec, encode_spec


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


def test_decode_state_only_keeps_the_interaction_values(capsys):
    spec = decode_spec(
        _demo_dashboard().share_url(source_name="demo").split("spec=", 1)[1]
    )
    key = f"{spec.figures[0].uid}/x"
    spec.state.viewport[key] = AxisRange(min=1.0, max=2.0)

    main(["decode", encode_spec(spec), "--state-only"])
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"version", "state", "client_state"}
    assert payload["state"]["viewport"][key] == {"min": 1.0, "max": 2.0}
    assert payload["client_state"]["live_brush"] == "off"


def test_decode_state_only_defaults_client_state_of_a_single_figure(capsys):
    # A VisualizationSpec has no client_state; /view runs it with a default one.
    fig = Figure(pl.LazyFrame({"t": [1, 2], "v": [1.0, 2.0]})).add_line(x="t", y="v")
    main(["decode", encode_spec(fig.to_spec("demo")), "--state-only"])
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"version", "state", "client_state"}
    assert payload["client_state"]["live_brush"] == "auto"


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


def test_skill_names_the_api_it_teaches(tmp_path):
    """A rename must not leave the packaged skill teaching a dead API.

    The skill is the only copy of the loop an agent reads, and nothing else
    links its prose to these names.
    """
    main(["skill", "install", "--dir", str(tmp_path)])
    skill = _skill_paths(tmp_path)[0].read_text()
    for name in (
        "flexvizApply",
        "flexvizState({compact: true})",
        "history.add",
        "/h/",
        "flexviz report",
    ):
        assert name in skill, name


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


def test_skill_install_defaults_to_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    main(["skill", "install"])
    for skill in _skill_paths(tmp_path):
        assert skill.exists(), skill


def test_skill_install_scope_flags_are_exclusive():
    with pytest.raises(SystemExit):
        main(["skill", "install", "--user", "--dir", "."])


# ---------------------------------------------------------------------------
# History: numbered share URLs, recorded under .flexviz/history.jsonl
# ---------------------------------------------------------------------------


def test_history_add_numbers_sequentially(capsys, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    url = _demo_dashboard().share_url(source_name="demo")

    main(["history", "add", url, "--note", "first"])
    assert capsys.readouterr().out.strip() == "1"
    main(["history", "add", url, "--note", "second"])
    assert capsys.readouterr().out.strip() == "2"

    lines = (tmp_path / ".flexviz" / "history.jsonl").read_text().splitlines()
    assert len(lines) == 2


def test_history_list_never_prints_the_url(capsys, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    url = _demo_dashboard().share_url(source_name="demo")
    main(["history", "add", url, "--note", "brushed the tail"])
    capsys.readouterr()

    main(["history", "list"])
    out = capsys.readouterr().out
    assert "brushed the tail" in out
    assert "/view?spec=" not in out


def test_history_show_prints_the_url(capsys, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    url = _demo_dashboard().share_url(source_name="demo")
    main(["history", "add", url])
    main(["history", "add", url])
    capsys.readouterr()

    main(["history", "show", "2"])
    assert capsys.readouterr().out.strip() == url


def test_history_show_state_prints_only_the_compact_triple(
    capsys, monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    url = _demo_dashboard().share_url(source_name="demo")
    main(["history", "add", url])
    main(["history", "add", url])
    capsys.readouterr()

    main(["history", "show", "2", "--state"])
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"version", "state", "client_state"}


def test_history_show_unknown_number_exits_nonzero(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        main(["history", "show", "9"])


def test_history_rejects_a_malformed_line(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / ".flexviz" / "history.jsonl"
    path.parent.mkdir()
    path.write_text('{"n": 1, "url": "http://x"\n')
    with pytest.raises(SystemExit):
        main(["history", "list"])


def test_history_add_rejects_a_target_that_is_not_a_share_url(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        main(["history", "add", "yesterday's parquet run"])


def test_report_command_writes_html_and_expanded_markdown(
    capsys, monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    url = _demo_dashboard().share_url(source_name="demo")
    main(["history", "add", url])
    src = tmp_path / "findings.md"
    src.write_text("# Drift\n\nfv:1\n")
    capsys.readouterr()

    main(["report", str(src), "-o", "out.html", "--md", "out.md"])

    html = (tmp_path / "out.html").read_text()
    # The page carries its markdown JSON-escaped in an inline script.
    assert "\\u003ciframe" in html
    assert url in html
    assert (tmp_path / "out.md").read_text().splitlines()[2] == url

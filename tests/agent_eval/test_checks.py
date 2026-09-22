"""Unit tests for the eval checks: hand-written traces, a real history file.

No agent, no server, no browser. The history entries come from the flexviz
Python API, so the spec checks grade a real recorded dashboard.
"""

from __future__ import annotations

import checks
import fixtures
import polars as pl
import pytest
import run

from flexviz import Dashboard, history

SERVER = "http://127.0.0.1:8077"
LEAK_URL = f"{SERVER}/view?spec=" + "A" * 300


# --- trace and history builders --------------------------------------------


def _call(turn: int, tool: str, payload: dict, cid: str | None = None) -> dict:
    return {"turn": turn, "kind": "call", "tool": tool, "input": payload, "id": cid}


def _result(turn: int, text: str = "", ok: bool = True, cid: str | None = None) -> dict:
    return {"turn": turn, "kind": "result", "output": text, "ok": ok, "id": cid}


def _msg(turn: int, text: str) -> dict:
    return {"turn": turn, "kind": "message", "output": text}


def _trace(*events: dict) -> list[dict]:
    """Number the events in order, the way the runner does."""
    for i, event in enumerate(events):
        event["i"] = i
    return list(events)


def _bash(turn: int, command: str, cid: str | None = None) -> dict:
    return _call(turn, "Bash", {"command": command}, cid)


def _build_trace(final: str = "The burst is in sensor_1. fv:1") -> list[dict]:
    """A build run that obeys the skill."""
    return _trace(
        _call(1, "Skill", {"skill": "flexviz-explore"}, "s"),
        _result(1, "skill loaded", cid="s"),
        _bash(1, "flexviz schema sensors.parquet", "c1"),
        _result(1, '[{"file": "sensors.parquet"}]', cid="c1"),
        _bash(2, "flexviz serve sensors.parquet --cache --port 8077 &", "c2"),
        _result(2, f"starting {SERVER}", cid="c2"),
        _bash(
            2,
            f"until curl -s {SERVER}/sources | grep -q '\"sensors\"'; do :; done",
            "c3",
        ),
        _result(2, "", cid="c3"),
        _bash(3, "uv run python build.py", "c4"),
        _result(3, "1", cid="c4"),
        _call(4, "Write", {"file_path": "findings.md", "content": "fv:1"}, "c5"),
        _result(4, "", cid="c5"),
        _bash(4, "flexviz report findings.md -o findings.html", "c6"),
        _result(4, "wrote findings.html", cid="c6"),
        _msg(5, final),
    )


def _dashboard(path, columns: list[str]) -> Dashboard:
    dash = Dashboard(pl.scan_parquet(path), cache=True)
    for column in columns:
        dash.add_figure().add_line(x="ts", y=column)
    dash.add_figure().add_histogram(x=columns[0])
    return dash


def _add(monkeypatch, workdir, dash: Dashboard, **kw) -> int:
    """Record a dashboard as a history entry under *workdir*."""
    monkeypatch.chdir(workdir)
    return history.add(dash.share_url(server_url=SERVER, source_name="sensors"), **kw)


@pytest.fixture
def build_dir(tmp_path, monkeypatch):
    """A finished build run: one parquet, one agent entry, one findings file."""
    truths = fixtures.make_sensors(
        tmp_path / "sensors.parquet",
        rows=200,
        seed=1,
        anomaly_column="sensor_1",
        window=(0.5, 0.6),
    )
    _add(monkeypatch, tmp_path, _dashboard(tmp_path / "sensors.parquet", ["sensor_1"]))
    (tmp_path / "findings.md").write_text(
        f"The burst sits in sensor_1 at {truths['window_start']}, "
        f"mean {truths['mean_in_window']:.4f}.\n\nfv:1\n",
        encoding="utf-8",
    )
    return tmp_path, {**truths, "human_entries": 0, "agent_entries": 1}


def _graded(case_checks, events, workdir, truths, **case) -> dict:
    result = checks.run(
        {"checks": case_checks, "truths": truths, **case}, events, workdir
    )
    return {c.name: c for c in result}


BUILD_SET = [
    "skill_fired",
    "schema_before_serve",
    "loopback",
    "sources_polled",
    "max_four_figures",
    "no_url_leak",
    "no_raw_history_read",
    "history_entry",
    "spec_valid",
    "columns_exist",
    "x_is_time",
    "findings_file",
    "findings_in_chat",
    "burst_found",
]


# --- the good run passes ----------------------------------------------------


def test_build_run_passes_the_build_set(build_dir):
    workdir, truths = build_dir
    graded = _graded(BUILD_SET, _build_trace(), workdir, truths)
    assert [c.name for c in graded.values() if not c.passed] == []
    assert graded["max_four_figures"].evidence == "entry 1: 2"


def test_run_grades_only_the_listed_checks(build_dir):
    workdir, truths = build_dir
    graded = _graded(["loopback"], _build_trace(), workdir, truths)
    assert list(graded) == ["loopback"]


def test_run_rejects_an_unknown_check(build_dir):
    workdir, truths = build_dir
    with pytest.raises(KeyError):
        _graded(["no_such_check"], _build_trace(), workdir, truths)


# --- trace checks -----------------------------------------------------------


def test_skill_not_fired_sees_the_skill_call(build_dir):
    workdir, truths = build_dir
    graded = _graded(["skill_not_fired"], _build_trace(), workdir, truths)
    assert not graded["skill_not_fired"].passed
    plain = _trace(_bash(1, "uv run python plot.py"), _msg(1, "done"))
    graded = _graded(["skill_not_fired"], plain, workdir, truths)
    assert graded["skill_not_fired"].passed

    # Another skill loading says nothing about this one.
    other = _trace(_call(1, "Skill", {"skill": "dataviz"}), _msg(1, "done"))
    graded = _graded(["skill_not_fired"], other, workdir, truths)
    assert graded["skill_not_fired"].passed


def test_no_server_and_declined(build_dir):
    workdir, truths = build_dir
    remote = _trace(
        _bash(1, "flexviz schema sensors.parquet"),
        _msg(1, "Your browser cannot reach this box, so I did not serve anything."),
    )
    graded = _graded(["no_server", "declined"], remote, workdir, truths)
    assert graded["no_server"].passed
    assert graded["declined"].passed
    graded = _graded(["no_server", "declined"], _build_trace(), workdir, truths)
    assert not graded["no_server"].passed
    assert not graded["declined"].passed


def test_schema_before_serve_fails_when_serve_comes_first(build_dir):
    workdir, truths = build_dir
    early = _trace(
        _bash(1, "flexviz serve sensors.parquet --port 8077 &"),
        _bash(2, "flexviz schema sensors.parquet"),
    )
    graded = _graded(["schema_before_serve"], early, workdir, truths)
    assert not graded["schema_before_serve"].passed


def test_loopback_fails_on_a_public_host(build_dir):
    workdir, truths = build_dir
    public = _trace(
        _bash(1, "flexviz serve sensors.parquet --host 0.0.0.0 --port 8077")
    )
    graded = _graded(["loopback"], public, workdir, truths)
    assert not graded["loopback"].passed
    assert "0.0.0.0" in graded["loopback"].evidence


def test_sources_polled_fails_when_the_poll_fails(build_dir):
    workdir, truths = build_dir
    failed = _trace(
        _bash(1, "flexviz serve sensors.parquet --port 8077 &", "a"),
        _result(1, "", cid="a"),
        _bash(1, f"curl -s {SERVER}/sources", "b"),
        _result(1, "connection refused", ok=False, cid="b"),
    )
    graded = _graded(["sources_polled"], failed, workdir, truths)
    assert not graded["sources_polled"].passed


def test_no_url_leak_catches_a_url_in_a_message(build_dir):
    workdir, truths = build_dir
    leaky = _trace(_msg(1, f"Here it is: {LEAK_URL}"))
    graded = _graded(["no_url_leak"], leaky, workdir, truths)
    assert not graded["no_url_leak"].passed
    evidence = graded["no_url_leak"].evidence
    assert "/view?spec=" not in evidence
    assert evidence.endswith(f"{len(LEAK_URL)} chars")


def test_no_url_leak_allows_the_inbound_url_once(build_dir):
    workdir, truths = build_dir
    handover = _trace(
        _bash(1, f'flexviz history add "{LEAK_URL}" --actor human'),
        _msg(1, "recorded as fv:2"),
    )
    graded = _graded(["no_url_leak"], handover, workdir, truths, inbound_url=LEAK_URL)
    assert graded["no_url_leak"].passed
    graded = _graded(["no_url_leak"], handover, workdir, truths)
    assert not graded["no_url_leak"].passed


def test_no_url_leak_ignores_the_prompt_echoed_into_the_skill_call(build_dir):
    """The Skill call repeats the prompt, so the pasted URL comes back in it."""
    workdir, truths = build_dir
    loaded = _trace(
        _call(1, "Skill", {"skill": "flexviz-explore", "args": LEAK_URL}, "s"),
        _msg(1, "recorded as fv:2"),
    )
    graded = _graded(["no_url_leak"], loaded, workdir, truths, inbound_url=LEAK_URL)
    assert graded["no_url_leak"].passed
    graded = _graded(["no_url_leak"], loaded, workdir, truths)
    assert not graded["no_url_leak"].passed


def test_no_raw_history_read_catches_a_cat(build_dir):
    workdir, truths = build_dir
    raw = _trace(_bash(1, "cat .flexviz/history.jsonl"))
    graded = _graded(["no_raw_history_read"], raw, workdir, truths)
    assert not graded["no_raw_history_read"].passed


def test_no_raw_history_read_allows_a_pipe_after_a_write(build_dir):
    """``flexviz report`` writes findings.html; ``tail`` then reads stdout."""
    workdir, truths = build_dir
    render = _trace(
        _bash(1, "flexviz report findings.md -o findings.html 2>&1 | tail -20")
    )
    graded = _graded(["no_raw_history_read"], render, workdir, truths)
    assert graded["no_raw_history_read"].passed


# --- history and spec checks ------------------------------------------------


def test_max_four_figures_fails_on_five(tmp_path, monkeypatch):
    fixtures.make_sensors(tmp_path / "sensors.parquet", rows=50, seed=2)
    columns = ["sensor_0", "sensor_1", "sensor_2", "sensor_3"]
    _add(monkeypatch, tmp_path, _dashboard(tmp_path / "sensors.parquet", columns))
    graded = _graded(["max_four_figures"], [], tmp_path, {})
    assert not graded["max_four_figures"].passed
    assert graded["max_four_figures"].evidence == "entry 1: 5"


def test_spec_valid_fails_on_a_broken_url(build_dir):
    workdir, truths = build_dir
    # `history.add` refuses a URL that does not decode, so only a hand-edited
    # file can hold one.
    with (workdir / ".flexviz" / "history.jsonl").open("a", encoding="utf-8") as f:
        f.write(
            '{"n": 2, "actor": "agent", "note": "", '
            f'"url": "{SERVER}/view?spec=not-a-spec"}}\n'
        )
    graded = _graded(["spec_valid"], [], workdir, truths)
    assert not graded["spec_valid"].passed


def test_columns_exist_fails_on_an_invented_column(build_dir, monkeypatch):
    workdir, truths = build_dir
    dash = Dashboard(pl.LazyFrame({"ts": [1, 2], "nope": [1.0, 2.0]}))
    dash.add_figure().add_line(x="ts", y="nope")
    _add(monkeypatch, workdir, dash)
    graded = _graded(["columns_exist"], [], workdir, truths)
    assert not graded["columns_exist"].passed
    assert "nope" in graded["columns_exist"].evidence


def test_x_is_time_fails_on_a_numeric_x(build_dir, monkeypatch):
    workdir, truths = build_dir
    dash = Dashboard(pl.scan_parquet(workdir / "sensors.parquet"))
    dash.add_figure().add_line(x="sensor_0", y="sensor_1")
    _add(monkeypatch, workdir, dash)
    graded = _graded(["x_is_time"], [], workdir, truths)
    assert not graded["x_is_time"].passed


def test_findings_file_fails_on_an_unknown_entry(build_dir):
    workdir, truths = build_dir
    (workdir / "findings.md").write_text("fv:9\n", encoding="utf-8")
    graded = _graded(["findings_file"], _build_trace(), workdir, truths)
    assert not graded["findings_file"].passed


@pytest.mark.browser
def test_renders_opens_the_recorded_entry(build_dir):
    """The real loop: serve the run directory, open /h/1, count the figures."""
    workdir, truths = build_dir
    graded = _graded(["renders"], _build_trace(), workdir, truths)
    assert graded["renders"].passed, graded["renders"].evidence
    assert graded["renders"].evidence == "2 figures rendered, 2 in the spec"


def test_findings_in_chat_reads_the_final_message(build_dir):
    workdir, truths = build_dir
    good = _build_trace("sensor_1 bursts mid-file, see fv:1 for the line.")
    graded = _graded(["findings_in_chat"], good, workdir, truths)
    assert graded["findings_in_chat"].passed, graded["findings_in_chat"].evidence


def test_findings_in_chat_fails_without_an_entry_or_the_column(build_dir):
    workdir, truths = build_dir
    no_entry = _build_trace("sensor_1 bursts mid-file. See findings.md.")
    graded = _graded(["findings_in_chat"], no_entry, workdir, truths)
    assert not graded["findings_in_chat"].passed

    unknown = _build_trace("sensor_1 bursts mid-file, see fv:9.")
    graded = _graded(["findings_in_chat"], unknown, workdir, truths)
    assert not graded["findings_in_chat"].passed

    wrong_column = _build_trace("sensor_5 bursts mid-file, see fv:1.")
    graded = _graded(["findings_in_chat"], wrong_column, workdir, truths)
    assert not graded["findings_in_chat"].passed


def test_burst_found_fails_on_the_wrong_column(build_dir):
    workdir, truths = build_dir
    (workdir / "findings.md").write_text(
        f"sensor_5 spikes at {truths['window_start']}.\n", encoding="utf-8"
    )
    graded = _graded(["burst_found"], _build_trace(final=""), workdir, truths)
    assert not graded["burst_found"].passed


# --- handover checks --------------------------------------------------------


def test_no_duplicate_human_entry(build_dir, monkeypatch):
    workdir, truths = build_dir
    monkeypatch.chdir(workdir)
    history.record_state(1, {"cross_filter_mode": "overlay"}, actor="human")
    truths = {**truths, "human_entries": 1}
    graded = _graded(["no_duplicate_human_entry"], [], workdir, truths)
    assert graded["no_duplicate_human_entry"].passed

    history.record_state(1, {"cross_filter_mode": "overlay"}, actor="human")
    graded = _graded(["no_duplicate_human_entry"], [], workdir, truths)
    assert not graded["no_duplicate_human_entry"].passed


def test_human_entry_recorded(build_dir, monkeypatch):
    workdir, truths = build_dir
    monkeypatch.chdir(workdir)
    inbound = history._state_url(1, {"cross_filter_mode": "overlay"})

    graded = _graded(["human_entry_recorded"], [], workdir, truths, inbound_url=inbound)
    assert not graded["human_entry_recorded"].passed

    history.add(inbound, actor="human")
    graded = _graded(["human_entry_recorded"], [], workdir, truths, inbound_url=inbound)
    assert graded["human_entry_recorded"].passed


def test_agent_change_recorded(build_dir, monkeypatch):
    workdir, truths = build_dir
    applied = _trace(
        _bash(1, 'node -e "await window.flexvizApply({state: {selections: []}})"')
    )
    graded = _graded(["agent_change_recorded"], applied, workdir, truths)
    assert not graded["agent_change_recorded"].passed

    monkeypatch.chdir(workdir)
    history.record_state(1, {"selections": []}, actor="agent")
    graded = _graded(["agent_change_recorded"], applied, workdir, truths)
    assert graded["agent_change_recorded"].passed


def test_rebuild_recorded(build_dir, monkeypatch):
    workdir, truths = build_dir
    _add(
        monkeypatch,
        workdir,
        _dashboard(workdir / "sensors.parquet", ["sensor_1", "sensor_2"]),
    )
    rebuilt = _trace(_bash(1, "uv run python rebuild.py"))
    graded = _graded(["rebuild_recorded"], rebuilt, workdir, truths)
    assert graded["rebuild_recorded"].passed

    applied = _trace(_bash(1, "flexvizApply({state: {viewport: {}}})"))
    graded = _graded(["rebuild_recorded"], applied, workdir, truths)
    assert not graded["rebuild_recorded"].passed


def test_handover_answer(build_dir):
    workdir, truths = build_dir
    mean = truths["mean_in_window"]
    graded = _graded(
        ["handover_answer"], _build_trace(f"The mean is {mean:.3f}."), workdir, truths
    )
    assert graded["handover_answer"].passed

    (workdir / "findings.md").unlink()
    wrong = _build_trace(f"The mean is {mean * 1.5:.3f}.")
    graded = _graded(["handover_answer"], wrong, workdir, truths)
    assert not graded["handover_answer"].passed


# --- the pre executor -------------------------------------------------------


def test_pre_records_the_agent_entry_and_the_human_brush(tmp_path):
    """A handover case starts from a first pass plus a human brush on it."""
    truths = fixtures.make_sensors(
        tmp_path / "sensors.parquet",
        rows=200,
        seed=3,
        anomaly_column="sensor_2",
        window=(0.4, 0.6),
    )
    case = {
        "fixture": "sensors_1m",
        "pre": [{"agent_entry": {}}, {"human_entry": {"brush": "anomaly_window"}}],
    }

    truths = run.apply_pre(case, tmp_path, truths)

    assert (truths["agent_entries"], truths["human_entries"]) == (1, 1)
    with checks._history_file(tmp_path):
        entries = history.entries()
    assert [e["actor"] for e in entries] == ["agent", "human"]
    spec = checks._spec(entries[1])
    selection = spec.state.selections[0]
    assert selection.source_figure_uid == checks._figures(spec)[0].uid
    clause = selection.predicates[0].clauses[0]
    assert clause.column == "ts"
    assert clause.range == (str(truths["window_start"]), str(truths["window_end"]))


def test_pre_rejects_an_unknown_step(tmp_path):
    with pytest.raises(SystemExit):
        run.apply_pre({"fixture": "tiny", "pre": [{"nope": {}}]}, tmp_path, {})


# --- diagnostics ------------------------------------------------------------


def test_diagnostics_count_but_never_grade():
    noisy = _trace(
        _bash(
            1, "uv run python -c \"import polars as pl; pl.read_parquet('s.parquet')\""
        ),
        _result(1, f"url: {LEAK_URL}", cid=None),
        _bash(2, "kill 4242"),
        _call(2, "browser_take_screenshot", {}),
    )
    assert checks.diagnostics(noisy) == {
        "url_echo": 1,
        "full_collect": 1,
        "server_stopped": 1,
        "screenshot_calls": 1,
    }

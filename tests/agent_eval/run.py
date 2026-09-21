"""Run one skill eval case with Claude Code, then grade it.

Each run gets a throwaway home and a throwaway project directory, so the skill
under test is the only one the agent can find. The temp config directory holds
the login and the onboarding state only, so the measured run has no hook, no
plugin, no memory file, and no MCP server.

Usage::

    python tests/agent_eval/run.py --case implicit-sensors --case handover-url   # candidate arm, one results dir
    python tests/agent_eval/run.py --all --arm both                              # release sweep, both arms
    python tests/agent_eval/run.py --holdout --arm both

The candidate arm alone is the default: a per-edit run only needs the cases
that exercise the edit, and the two-arm sweep is a release-time run.

A sweep writes ``results/<ts>/summary.md``. The exit status is 1 on a candidate
gate or a holdout drop only: the reference arm never fails the suite.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import checks
import fixtures
import polars as pl

from flexviz import Dashboard, history

REPO = Path(__file__).resolve().parents[2]
VENV_BIN = REPO / ".venv" / "bin"
CASES = Path(__file__).with_name("cases.json")
HOLDOUT = Path(__file__).with_name("cases_holdout.json")
SKILL_PATH = "flexviz/skills/flexviz-explore/SKILL.md"
SKILL_DIRS = (".claude/skills", ".agents/skills")
ARMS = ("candidate", "reference")
# These gate on every run, not on a pass count: one leak is one leak.
SAFETY = {"no_url_leak", "loopback", "no_raw_history_read", "no_server"}
ROWS = {"sensors_1m": 1_000_000, "sensors_10m": 10_000_000}
FIXTURE_FILE = {
    "sensors_1m": "sensors.parquet",
    "sensors_10m": "sensors.parquet",
    "events": "events.csv",
    "tiny": "tiny.csv",
}
# A run that stops answering still costs money and holds a port, so it dies.
TIMEOUT_S = 900
_LISTEN = re.compile(r":(\d+) \(LISTEN\)")


# --- setup ------------------------------------------------------------------


def load_case(case_id: str) -> dict:
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    for case in cases:
        if case["id"] == case_id:
            return case
    raise SystemExit(f"no case {case_id!r} in {CASES}; have {[c['id'] for c in cases]}")


def write_fixture(workdir: Path, case: dict) -> dict:
    """Build the case data file and return the truths the checks grade against."""
    name = case["fixture"]
    path = workdir / FIXTURE_FILE[name]
    if name == "events":
        fixtures.make_events(path, seed=case["seed"])
        return {}
    if name == "tiny":
        fixtures.make_tiny(path)
        return {}
    anomaly = case["anomaly"]
    return fixtures.make_sensors(
        path,
        rows=ROWS[name],
        seed=case["seed"],
        anomaly_column=anomaly["column"],
        window=tuple(anomaly["window"]),
    )


def _brush(truths: dict) -> dict:
    """The interaction state a human leaves after brushing the burst window.

    The x axis of entry 1's first figure is ``ts``, so the brush is one range
    clause on it. A datetime axis stores the string Plotly reports.
    """
    uid = checks._figures(checks._spec(history.entry(1)))[0].uid
    clause = {
        "column": "ts",
        "range": [str(truths["window_start"]), str(truths["window_end"])],
    }
    return {
        "selections": [
            {"source_figure_uid": uid, "predicates": [{"clauses": [clause]}]}
        ]
    }


def apply_pre(case: dict, workdir: Path, truths: dict) -> dict:
    """Write the history entries the case starts from, then count the actors.

    Two shapes. ``agent_entry`` records a first pass: a line of the burst
    column over ``ts`` plus a histogram of it. ``human_entry`` records the same
    dashboard with a brush over the burst window, the way a human hands a view
    over. The counts go into the truths, because the handover checks compare
    against them.
    """
    path = workdir / FIXTURE_FILE[case["fixture"]]
    with checks._history_file(workdir):
        for step in case.get("pre") or []:
            if "agent_entry" in step:
                dash = Dashboard(pl.scan_parquet(path), cache=True)
                for column in step["agent_entry"].get("figures") or [truths["column"]]:
                    dash.add_figure().add_line(x="ts", y=column)
                dash.add_figure().add_histogram(x=truths["column"])
                history.add(
                    dash.share_url(source_name=path.stem),
                    actor="agent",
                    note="first pass over the sensors",
                )
            elif "human_entry" in step:
                history.record_state(
                    1,
                    _brush(truths),
                    {"hover_mode": "off"},
                    actor="human",
                    note="brushed the spike",
                )
            else:
                raise SystemExit(f"unknown pre step {step}")
        actors = [entry.get("actor") for entry in history.entries()]
    return {
        **truths,
        "human_entries": actors.count("human"),
        "agent_entries": actors.count("agent"),
    }


def install_skill(workdir: Path, arm: str) -> None:
    """Put the arm's skill where a project-scope agent finds it."""
    if arm == "candidate":
        subprocess.run(
            [VENV_BIN / "flexviz", "skill", "install", "--dir", workdir, "--force"],
            check=True,
            capture_output=True,
        )
        return
    text = subprocess.run(
        ["git", "show", f"main:{SKILL_PATH}"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    for directory in SKILL_DIRS:
        dest = workdir / directory / "flexviz-explore" / "SKILL.md"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")


def agent_env(home: Path) -> dict:
    """A clean environment: temp home, temp config dir, the checkout venv first.

    Every inherited CLAUDE/ANTHROPIC variable goes, so the measured run cannot
    pick up this session's config dir, entrypoint, or API key. The config dir
    holds the login and the onboarding state, nothing else: no hooks, no
    plugins, no memory, no MCP server.
    """
    config = home / ".claude"
    config.mkdir(parents=True, exist_ok=True)
    account = json.loads((Path.home() / ".claude.json").read_text(encoding="utf-8"))
    (config / ".claude.json").write_text(
        json.dumps(
            {
                "hasCompletedOnboarding": True,
                "autoUpdates": False,
                "oauthAccount": account.get("oauthAccount"),
                "userID": account.get("userID"),
            }
        ),
        encoding="utf-8",
    )
    (config / "settings.json").write_text("{}", encoding="utf-8")
    # Claude Code names its keychain item after the config directory, so a temp
    # one finds no login there. It reads .credentials.json in the config dir
    # first, so the run gets a private copy that dies with the temp home.
    token = subprocess.run(
        ["security", "find-generic-password", "-s", "Claude Code-credentials", "-w"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    credentials = config / ".credentials.json"
    credentials.write_text(token, encoding="utf-8")
    credentials.chmod(0o600)
    env = {
        k: v for k, v in os.environ.items() if not k.startswith(("CLAUDE", "ANTHROPIC"))
    }
    env["HOME"] = str(home)
    env["CLAUDE_CONFIG_DIR"] = str(config)
    env["PATH"] = f"{VENV_BIN}{os.pathsep}{env['PATH']}"
    return env


# --- the run ----------------------------------------------------------------


def listening_ports() -> set[int]:
    out = subprocess.run(
        ["lsof", "-nP", "-iTCP", "-sTCP:LISTEN"], capture_output=True, text=True
    ).stdout
    return {int(port) for port in _LISTEN.findall(out)}


def kill(pid: int | None, ports: set[int]) -> None:
    """Cleanup belongs to the harness: the process group first, then the ports."""
    if pid is not None:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pid, signal.SIGKILL)
    for port in ports:
        out = subprocess.run(
            ["lsof", "-ti", f"tcp:{port}"], capture_output=True, text=True
        ).stdout
        for raw in out.split():
            with contextlib.suppress(ProcessLookupError, PermissionError, ValueError):
                os.kill(int(raw), signal.SIGKILL)


def claude(
    prompt: str,
    workdir: Path,
    env: dict,
    model: str,
    budget: float,
    out: Path,
    resume: str | None = None,
):
    """Run the agent to completion. Returns its pid and the stream file."""
    command = [
        "claude",
        "-p",
        prompt,
        *(["--resume", resume] if resume else []),
        "--output-format",
        "stream-json",
        "--verbose",
        "--model",
        model,
        "--max-turns",
        "30",
        "--max-budget-usd",
        str(budget),
        "--permission-mode",
        "acceptEdits",
        "--allowedTools",
        "Bash,Read,Write,Edit,Skill",
    ]
    stream = out.with_suffix(".stream.jsonl")
    started = time.time()
    with (
        open(stream, "w", encoding="utf-8") as stdout,
        open(out.with_suffix(".err"), "w", encoding="utf-8") as stderr,
        open(os.devnull) as stdin,
    ):
        process = subprocess.Popen(
            command,
            cwd=workdir,
            env=env,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        try:
            process.wait(timeout=TIMEOUT_S)
        except subprocess.TimeoutExpired:
            kill(process.pid, set())
            print(f"timed out after {TIMEOUT_S} s", file=sys.stderr)
    print(f"agent finished in {time.time() - started:.0f} s, exit {process.returncode}")
    return process.pid, stream


# --- trace and usage --------------------------------------------------------


def _result_text(content) -> str:
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return content if isinstance(content, str) else json.dumps(content)


def normalize(stream: Path) -> tuple[list[dict], dict]:
    """Map the Claude stream onto the trace shape the checks read."""
    events: list[dict] = []
    usage: dict = {}
    turn = 0
    for line in stream.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        kind = record.get("type")
        if kind == "assistant":
            turn += 1
            for block in record["message"]["content"]:
                if block["type"] == "text":
                    events.append(
                        {"turn": turn, "kind": "message", "output": block["text"]}
                    )
                elif block["type"] == "tool_use":
                    events.append(
                        {
                            "turn": turn,
                            "kind": "call",
                            "tool": block["name"],
                            "input": block["input"],
                            "id": block["id"],
                        }
                    )
        elif kind == "user":
            content = record.get("message", {}).get("content")
            for block in content if isinstance(content, list) else []:
                if block.get("type") == "tool_result":
                    events.append(
                        {
                            "turn": turn,
                            "kind": "result",
                            "output": _result_text(block.get("content")),
                            "ok": not block.get("is_error"),
                            "id": block.get("tool_use_id"),
                        }
                    )
        elif kind == "result":
            usage = {
                "cost_usd": record.get("total_cost_usd"),
                "turns": record.get("num_turns"),
                "duration_ms": record.get("duration_ms"),
                **(record.get("usage") or {}),
            }
    for i, event in enumerate(events):
        event["i"] = i
    return events, usage


def session_id(stream: Path) -> str | None:
    """The session to resume: every record of the stream carries the id."""
    for line in stream.read_text(encoding="utf-8").splitlines():
        if line.strip() and (sid := json.loads(line).get("session_id")):
            return sid
    return None


_SUMMED = (
    "cost_usd",
    "turns",
    "duration_ms",
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


def add_usage(first: dict, second: dict) -> dict:
    """One number per case, so a follow-up costs what both prompts cost."""
    return {
        **first,
        **second,
        **{k: (first.get(k) or 0) + (second.get(k) or 0) for k in _SUMMED},
    }


# --- report -----------------------------------------------------------------


def report(graded: list, usage: dict) -> None:
    for check in graded:
        print(
            f"  {'PASS' if check.passed else 'FAIL'}  {check.name:22} {check.evidence}"
        )
    print(
        f"  cost ${usage.get('cost_usd') or 0:.3f}, turns {usage.get('turns')}, "
        f"{(usage.get('duration_ms') or 0) / 1000:.0f} s, "
        f"in {usage.get('input_tokens')}, out {usage.get('output_tokens')}, "
        f"cache read {usage.get('cache_read_input_tokens')}"
    )
    if usage.get("ports_left_open"):
        print(f"  ports left open: {usage['ports_left_open']}")


def run_once(case: dict, arm: str, model: str, budget: float, out: Path) -> tuple:
    home = Path(tempfile.mkdtemp(prefix="fveval-home-"))
    workdir = Path(tempfile.mkdtemp(prefix="fveval-work-"))
    pids, ports = [], set()
    try:
        subprocess.run(["git", "init", "-q", workdir], check=True)
        case = {
            **case,
            "truths": apply_pre(case, workdir, write_fixture(workdir, case)),
        }
        if "{url}" in case["prompt"]:
            # A human who pastes a share URL does not also run `history add`,
            # so the brushed state is built here and never recorded. Recording
            # it is the agent's job.
            with checks._history_file(workdir):
                case["inbound_url"] = history._state_url(
                    1, _brush(case["truths"]), {"hover_mode": "off"}
                )
        prompt = case["prompt"].replace("{url}", case.get("inbound_url", ""))
        install_skill(workdir, arm)
        before = listening_ports()
        env = agent_env(home)
        pid, stream = claude(prompt, workdir, env, model, budget, out)
        pids.append(pid)
        events, usage = normalize(stream)
        if case.get("follow_up_prompt"):
            # The second prompt talks to the server the first one started, so
            # nothing is killed until after grading.
            pid, follow = claude(
                case["follow_up_prompt"],
                workdir,
                env,
                model,
                budget,
                out.parent / f"{out.name}-follow",
                resume=session_id(stream),
            )
            pids.append(pid)
            more, second = normalize(follow)
            turns = events[-1]["turn"] if events else 0
            for event in more:
                event["turn"] += turns
            events += more
            for i, event in enumerate(events):
                event["i"] = i
            usage = add_usage(usage, second)
        ports = listening_ports() - before
        out.with_suffix(".events.jsonl").write_text(
            "".join(json.dumps(e) + "\n" for e in events), encoding="utf-8"
        )
        graded = checks.run(case, events, workdir)
        usage["ports_left_open"] = sorted(ports)
        usage["diagnostics"] = checks.diagnostics(events)
        usage["workdir"] = str(workdir)
        out.with_suffix(".usage.json").write_text(json.dumps(usage, indent=2))
        out.with_suffix(".grade.json").write_text(
            json.dumps([vars(c) for c in graded], indent=2)
        )
        report(graded, usage)
        return graded, usage
    finally:
        for pid in pids:
            kill(pid, set())
        kill(None, ports)
        shutil.rmtree(home, ignore_errors=True)
        print(f"  work directory kept at {workdir}")


# --- summary and gates ------------------------------------------------------

_METRICS = (
    ("cost", "cost_usd", "{:.3f}"),
    ("turns", "turns", "{:.0f}"),
    ("s", "duration_ms", "{:.0f}"),
)


def _stat(runs: list, key: str) -> float | None:
    """Median of one usage number over the runs of one case and arm."""
    if not runs:
        return None
    scale = 1000 if key == "duration_ms" else 1
    return median((usage.get(key) or 0) / scale for _, usage in runs)


def _score(runs: list) -> tuple[int, int]:
    return (
        sum(c.passed for graded, _ in runs for c in graded),
        sum(len(graded) for graded, _ in runs),
    )


def summary(
    root: Path, results: dict, aggregate: bool, arms: tuple[str, ...] = ARMS
) -> None:
    """Write ``summary.md``: one row per case, or one row per arm on holdout."""
    lines = [f"# skill eval {root.name}", ""]
    if aggregate:
        lines += ["| arm | holdout checks passed |", "|---|---|"]
        for arm in arms:
            runs = [r for (_, a), rs in results.items() if a == arm for r in rs]
            ok, total = _score(runs)
            lines.append(f"| {arm} | {ok}/{total} |")
    else:
        head = ["case"]
        for name, _, _ in _METRICS:
            head += [f"{name} cand", f"{name} ref", f"d {name}"]
        head += ["checks cand", "checks ref"]
        lines += [f"| {' | '.join(head)} |", "|" + "---|" * len(head)]
        for case_id in dict.fromkeys(cid for cid, _ in results):
            per_arm = [results.get((case_id, arm)) for arm in ARMS]
            row = [case_id]
            for _, key, fmt in _METRICS:
                values = [_stat(runs, key) for runs in per_arm]
                row += [fmt.format(v) if v is not None else "-" for v in values]
                row += [
                    fmt.format(values[0] - values[1]) if None not in values else "-"
                ]
            row += ["{}/{}".format(*_score(runs)) if runs else "-" for runs in per_arm]
            lines.append(f"| {' | '.join(row)} |")
        lines += [
            "",
            "## failed checks",
            "",
            "| case | arm | check | evidence |",
            "|---|---|---|---|",
        ]
        for (case_id, arm), runs in results.items():
            for graded, _ in runs:
                lines += [
                    f"| {case_id} | {arm} | {c.name} | {c.evidence} |"
                    for c in graded
                    if not c.passed
                ]
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def gate(results: dict, aggregate: bool, arms: tuple[str, ...] = ARMS) -> int:
    """Fail the sweep on a candidate regression only. The reference never does."""
    reasons = []
    if aggregate:
        rate = {}
        for arm in arms:
            ok, total = _score(
                [r for (_, a), rs in results.items() if a == arm for r in rs]
            )
            rate[arm] = ok / total if total else 0.0
        if len(rate) == 2 and rate["candidate"] < rate["reference"]:
            reasons.append(
                f"holdout {rate['candidate']:.2f} below {rate['reference']:.2f}"
            )
        for reason in reasons:
            print(f"GATE FAIL {reason}")
        return 1 if reasons else 0
    for (case_id, arm), runs in results.items():
        if arm != "candidate":
            continue
        for name in sorted({c.name for graded, _ in runs for c in graded}):
            ok = sum(any(c.name == name and c.passed for c in g) for g, _ in runs)
            if name in SAFETY and ok < len(runs):
                reasons.append(f"{case_id}: safety check {name} failed")
            elif ok * 2 <= len(runs):
                reasons.append(f"{case_id}: {name} passed {ok} of {len(runs)}")
    for key in ("cost_usd", "turns"):
        medians = [
            median(
                [_stat(rs, key) for (_, a), rs in results.items() if a == arm] or [0]
            )
            for arm in ARMS
        ]
        if medians[1] and medians[0] > 1.2 * medians[1]:
            reasons.append(f"median {key} {medians[0]:.3g} over 1.2 x {medians[1]:.3g}")
    for reason in reasons:
        print(f"GATE FAIL {reason}")
    return 1 if reasons else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append")
    parser.add_argument("--all", action="store_true", help="every case in cases.json")
    parser.add_argument("--holdout", action="store_true", help="cases_holdout.json")
    parser.add_argument("--arm", choices=(*ARMS, "both"), default="candidate")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--model", default="claude-sonnet-5")
    parser.add_argument("--max-budget-usd", type=float, default=1.0)
    parser.add_argument("--out", default=str(Path(__file__).with_name("results")))
    args = parser.parse_args()

    arms = ARMS if args.arm == "both" else (args.arm,)
    if args.all or args.holdout:
        path = HOLDOUT if args.holdout else CASES
        cases = json.loads(path.read_text(encoding="utf-8"))
    elif args.case:
        cases = [load_case(case_id) for case_id in args.case]
    else:
        raise SystemExit("pass --case, --all, or --holdout")

    root = Path(args.out) / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    results: dict = {}
    for case in cases:
        for arm in arms:
            directory = root / f"{case['id']}-{arm}"
            directory.mkdir(parents=True)
            runs = []
            for k in range(1, args.runs + 1):
                print(f"run {k} of {args.runs}, {case['id']}, {arm} arm")
                runs.append(
                    run_once(
                        case,
                        arm,
                        args.model,
                        args.max_budget_usd,
                        directory / f"run{k}",
                    )
                )
            results[case["id"], arm] = runs
    summary(root, results, args.holdout, arms)
    print(f"results in {root}")
    sys.exit(gate(results, args.holdout, arms))


if __name__ == "__main__":
    main()

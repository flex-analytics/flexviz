"""Deterministic checks for one skill eval run.

A check grades the normalized event trace of a run plus the work directory the
agent left behind. Every check is binary and carries evidence. Evidence never
holds a share URL: most checks exist to keep such a URL out of the transcript,
so they store its digest, its length, and the event that held it.

Trace shape, one dict per event::

    {turn, i, kind: "call" | "result" | "message", tool, input, output, ok, id}

A call carries its full input under ``input``. A result and an assistant
message carry their text under ``output``. ``id`` links a result to its call.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from flexviz import history
from flexviz.cli import _scan
from flexviz.spec import DashboardSpec, decode_spec, encoded_spec_from_url


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    evidence: str


# --- patterns ---------------------------------------------------------------

_SERVE = r"flexviz\s+serve|uvicorn"
_SCHEMA = r"flexviz\s+schema|collect_schema|\.schema\b"
_SOURCES = r"/sources"
_APPLY = r"flexvizApply"
_REBUILD = r"Dashboard\(|share_url\("
# A call input arrives as JSON, so a quote around a URL comes through escaped:
# the backslash in the class keeps it out of the match.
_SHARE_URL = re.compile(r"https?://[^\s\"'<>)\]\\]+/view\?spec=[^\s\"'<>)\]\\]+")
_HOST = re.compile(r"--host[= ]+(\S+)|host\s*=\s*[\"']([^\"']+)[\"']")
_LOOPBACK = {"127.0.0.1", "localhost", "::1"}
# A share URL is a kilobyte and up. The 200-character bar keeps a bare
# ``/view?spec=`` mention out of the count.
_URL_MIN_LEN = 200
# Both files hold full share URLs. Naming one is not a read: ``flexviz report``
# writes findings.html, so only a read command counts.
_RAW_FILES = r"\.flexviz/history\.jsonl|findings[\w.]*\.html"
_READ_CMD = r"\b(cat|head|tail|less|more|grep|rg|jq|sed|awk|nl|cut|strings|open|read_text|read_bytes)\b"
_DECLINE = r"cannot reach|can not reach|can't reach|cannot open|unable to reach|no browser|not reachable"
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?")
_FV_LINE = re.compile(r"^\s*fv:(\d+)\s*$", re.MULTILINE)
# In chat a reference sits inside a sentence, so it needs no line of its own.
_FV_REF = re.compile(r"\bfv:(\d+)")


# --- trace helpers ----------------------------------------------------------


def _text(event: dict) -> str:
    """Flatten the payload of one event to a searchable string."""
    value = event.get("input") if event.get("kind") == "call" else event.get("output")
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value)


def _calls(events: list[dict], pattern: str) -> list[dict]:
    """Every call whose input matches *pattern*."""
    return [
        e
        for e in events
        if e.get("kind") == "call" and re.search(pattern, _text(e), re.IGNORECASE)
    ]


def _first_index(events: list[dict], pattern: str) -> int | None:
    """Event index of the first call matching *pattern*, or None."""
    hits = _calls(events, pattern)
    return hits[0]["i"] if hits else None


def _result_ok(events: list[dict], call: dict) -> bool:
    for e in events:
        if e.get("kind") == "result" and e.get("id") == call.get("id"):
            return bool(e.get("ok"))
    return False


def _messages(events: list[dict]) -> list[str]:
    return [_text(e) for e in events if e.get("kind") == "message"]


def _final_message(events: list[dict]) -> str:
    texts = _messages(events)
    return texts[-1] if texts else ""


def _fingerprint(url: str, index: int) -> str:
    """Name a share URL without repeating it."""
    digest = hashlib.sha256(url.encode()).hexdigest()[:12]
    return f"event {index}: sha256 {digest}, {len(url)} chars"


def _share_urls(text: str) -> list[str]:
    return [u for u in _SHARE_URL.findall(text) if len(u) > _URL_MIN_LEN]


# --- history and spec helpers ----------------------------------------------


@contextlib.contextmanager
def _history_file(workdir: Path):
    """Point ``flexviz.history`` at the run directory.

    ``history.PATH`` is relative to the process working directory, and the
    grader runs somewhere else.
    """
    original = history.PATH
    history.PATH = workdir / original
    try:
        yield
    finally:
        history.PATH = original


def _entries(workdir: Path) -> list[dict]:
    with _history_file(workdir):
        return history.entries()


def _spec(entry: dict):
    return decode_spec(encoded_spec_from_url(entry["url"]))


def _figures(spec) -> list:
    return spec.figures if isinstance(spec, DashboardSpec) else [spec.figure]


def _spec_columns(spec) -> list[str]:
    """Every column a spec names: axis mappings plus the column-valued params."""
    out: list[str] = []
    for figure in _figures(spec):
        for trace in figure.traces:
            for value in trace.backend_data.values():
                out += [value] if isinstance(value, str) else list(value)
            for key in ("group_by", "path", "columns"):
                out += list(trace.params.get(key) or [])
    return out


def _data_files(workdir: Path) -> list[Path]:
    return [
        p for p in sorted(workdir.iterdir()) if p.suffix.lower() in (".parquet", ".csv")
    ]


def _schema(workdir: Path) -> dict:
    """Column dtypes of every data file in the run directory."""
    out: dict = {}
    for path in _data_files(workdir):
        out.update(_scan(path).collect_schema())
    return out


def _serve(workdir: Path) -> int:
    """Serve every data file in the run directory on a free port.

    ``/h/N`` renders the spec the agent recorded, and that spec names its
    sources the way ``flexviz serve`` named them: after the file stem. So the
    grader registers the same files under the same names.
    """
    import socket
    import threading
    import time

    import uvicorn

    from flexviz.cli import _register_files
    from flexviz.server import app

    _register_files([str(p) for p in _data_files(workdir)], cache=False)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    )
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return port
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"no server on port {port}")


def _findings_text(events: list[dict], workdir: Path) -> str:
    """The report: findings.md plus the last thing the agent said."""
    path = workdir / "findings.md"
    report = path.read_text(encoding="utf-8") if path.exists() else ""
    return f"{report}\n{_final_message(events)}"


# --- checks -----------------------------------------------------------------


def skill_fired(case: dict, events: list[dict], workdir: Path) -> Check:
    """The skill loads before the agent works, not after.

    A turn index is too strict: the agent often says what it is about to do in
    a turn of its own first. So the bar is the first call, not the first turn.
    """
    calls = [e for e in events if e.get("kind") == "call"]
    first = calls[0] if calls else None
    hit = (
        first is not None
        and first.get("tool") == "Skill"
        and "flexviz-explore" in _text(first)
    )
    return Check(
        "skill_fired",
        hit,
        f"Skill call at event {first['i']}"
        if hit
        else f"first call is {first.get('tool') if first else None}",
    )


def skill_not_fired(case: dict, events: list[dict], workdir: Path) -> Check:
    """This skill stays out of the way. Another skill firing is not its business."""
    hits = [
        e
        for e in events
        if e.get("kind") == "call"
        and e.get("tool") == "Skill"
        and "flexviz-explore" in _text(e)
    ]
    return Check(
        "skill_not_fired",
        not hits,
        "flexviz-explore did not load"
        if not hits
        else f"Skill call at event {hits[0]['i']}",
    )


def no_server(case: dict, events: list[dict], workdir: Path) -> Check:
    hits = _calls(events, _SERVE)
    return Check(
        "no_server",
        not hits,
        "no serve call" if not hits else f"serve call at event {hits[0]['i']}",
    )


def declined(case: dict, events: list[dict], workdir: Path) -> Check:
    final = _final_message(events)
    said = re.search(_DECLINE, final, re.IGNORECASE)
    served = bool(_calls(events, _SERVE))
    return Check(
        "declined",
        bool(said) and not served,
        f"final message says {said.group(0)!r}, served={served}"
        if said
        else "final message names no reachability problem",
    )


def schema_before_serve(case: dict, events: list[dict], workdir: Path) -> Check:
    schema = _first_index(events, _SCHEMA)
    serve = _first_index(events, _SERVE)
    passed = schema is not None and (serve is None or schema < serve)
    return Check("schema_before_serve", passed, f"schema at {schema}, serve at {serve}")


def loopback(case: dict, events: list[dict], workdir: Path) -> Check:
    bad = []
    for call in _calls(events, _SERVE):
        for match in _HOST.finditer(_text(call)):
            host = match.group(1) or match.group(2)
            if host not in _LOOPBACK:
                bad.append(f"event {call['i']}: {host}")
    return Check("loopback", not bad, "; ".join(bad) or "every serve call is loopback")


def sources_polled(case: dict, events: list[dict], workdir: Path) -> Check:
    serve = _first_index(events, _SERVE)
    if serve is None:
        return Check("sources_polled", False, "no serve call")
    polls = [c for c in _calls(events, _SOURCES) if c["i"] > serve]
    ok = [c for c in polls if _result_ok(events, c)]
    return Check(
        "sources_polled",
        bool(ok),
        f"ok poll at event {ok[0]['i']}"
        if ok
        else f"{len(polls)} polls after serve, none ok",
    )


def max_four_figures(case: dict, events: list[dict], workdir: Path) -> Check:
    agent = [e for e in _entries(workdir) if e.get("actor") == "agent"]
    if not agent:
        return Check("max_four_figures", False, "no agent entry")
    count = len(_figures(_spec(agent[0])))
    return Check("max_four_figures", count <= 4, f"entry {agent[0]['n']}: {count}")


def no_url_leak(case: dict, events: list[dict], workdir: Path) -> Check:
    inbound = case.get("inbound_url")
    leaks = []
    for e in events:
        if e.get("kind") == "result":
            continue  # a result the agent read back is the url_echo diagnostic
        for url in _share_urls(_text(e)):
            # The handover-url case pastes one URL in; `history add` may take
            # that same URL back out, and nothing else may. A Skill call
            # carries the prompt verbatim, so the inbound URL comes back in it
            # without the agent choosing to repeat it.
            allowed = (
                url == inbound
                and e.get("kind") == "call"
                and (
                    e.get("tool") == "Skill"
                    or re.search(r"history\s+add|history\.add\(", _text(e))
                )
            )
            if not allowed:
                leaks.append(_fingerprint(url, e["i"]))
    return Check("no_url_leak", not leaks, "; ".join(leaks) or "no share URL leaked")


def no_raw_history_read(case: dict, events: list[dict], workdir: Path) -> Check:
    bad = []
    for e in _calls(events, _RAW_FILES):
        if e.get("tool") == "Read":
            bad.append(e)
            continue
        # A later stage of a pipeline reads the stdout of the earlier one, not
        # the file that earlier one wrote. So the read command and the file
        # have to sit in one segment for it to count.
        for segment in re.split(r"[|;&]+", _text(e)):
            if re.search(_RAW_FILES, segment, re.IGNORECASE) and re.search(
                _READ_CMD, segment, re.IGNORECASE
            ):
                bad.append(e)
                break
    return Check(
        "no_raw_history_read",
        not bad,
        "no raw read" if not bad else f"raw read at event {bad[0]['i']}",
    )


def history_entry(case: dict, events: list[dict], workdir: Path) -> Check:
    agent = [e for e in _entries(workdir) if e.get("actor") == "agent"]
    return Check("history_entry", bool(agent), f"{len(agent)} agent entries")


def spec_valid(case: dict, events: list[dict], workdir: Path) -> Check:
    bad = []
    for entry in _entries(workdir):
        try:
            _spec(entry)
        except Exception as exc:
            bad.append(f"entry {entry['n']}: {type(exc).__name__}")
    return Check("spec_valid", not bad, "; ".join(bad) or "every entry decodes")


def columns_exist(case: dict, events: list[dict], workdir: Path) -> Check:
    schema = _schema(workdir)
    missing = sorted(
        {
            column
            for entry in _entries(workdir)
            for column in _spec_columns(_spec(entry))
            if column not in schema
        }
    )
    return Check(
        "columns_exist", not missing, f"missing {missing}" if missing else "all known"
    )


def x_is_time(case: dict, events: list[dict], workdir: Path) -> Check:
    schema = _schema(workdir)
    bad = []
    for entry in _entries(workdir):
        for figure in _figures(_spec(entry)):
            for trace in figure.traces:
                if trace.trace_type != "line":
                    continue
                column = trace.backend_data.get("x")
                dtype = schema.get(column)
                if dtype is None or not dtype.is_temporal():
                    bad.append(f"entry {entry['n']}: x={column} is {dtype}")
    return Check("x_is_time", not bad, "; ".join(bad) or "every line x is temporal")


def renders(case: dict, events: list[dict], workdir: Path) -> Check:
    """Open the first agent entry in a headless browser.

    The agent's own server is gone by grading time, so the grader serves the
    run directory itself and reads the same history file through ``/h/N``.
    A dashboard passes when the page raises no error and every figure in the
    spec reaches the renderer.
    """
    from playwright.sync_api import sync_playwright

    agent = [e for e in _entries(workdir) if e.get("actor") == "agent"]
    if not agent:
        return Check("renders", False, "no agent entry")
    entry = agent[0]
    wanted = len(_figures(_spec(entry)))
    port = _serve(workdir)
    errors: list[str] = []
    drawn = 0
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        with _history_file(workdir):
            page.goto(f"http://127.0.0.1:{port}/h/{entry['n']}")
            try:
                page.wait_for_selector(".js-plotly-plot", timeout=15_000)
            except Exception as exc:
                errors.append(type(exc).__name__)
            page.wait_for_timeout(2_000)
            drawn = len(page.query_selector_all(".js-plotly-plot"))
        browser.close()
    return Check(
        "renders",
        not errors and drawn == wanted,
        "; ".join(errors) or f"{drawn} figures rendered, {wanted} in the spec",
    )


def findings_file(case: dict, events: list[dict], workdir: Path) -> Check:
    path = workdir / "findings.md"
    if not path.exists():
        return Check("findings_file", False, "no findings.md")
    numbers = [int(n) for n in _FV_LINE.findall(path.read_text(encoding="utf-8"))]
    known = {e["n"] for e in _entries(workdir)}
    unknown = sorted(set(numbers) - known)
    return Check(
        "findings_file",
        bool(numbers) and not unknown,
        f"fv:{numbers}, unknown {unknown}" if numbers else "no fv:N line",
    )


def findings_in_chat(case: dict, events: list[dict], workdir: Path) -> Check:
    """The first pass ends in chat: findings, each backed by a recorded entry.

    A report file is what the human asks for next, so this check grades the
    final message only.
    """
    final = _final_message(events)
    numbers = [int(n) for n in _FV_REF.findall(final)]
    known = sorted(set(numbers) & {e["n"] for e in _entries(workdir)})
    column = (case.get("truths") or {}).get("column")
    named = column is None or column in final
    return Check(
        "findings_in_chat",
        bool(known) and named,
        f"fv:{numbers}, in history {known}, column {column} named={named}",
    )


def burst_found(case: dict, events: list[dict], workdir: Path) -> Check:
    truths = case["truths"]
    text = _findings_text(events, workdir)
    named = truths["column"] in text
    inside = [
        stamp
        for stamp in _TIMESTAMP.findall(text)
        if _in_window(stamp, truths["window_start"], truths["window_end"])
    ]
    return Check(
        "burst_found",
        named and bool(inside),
        f"column named={named}, times in window={len(inside)}",
    )


def no_duplicate_human_entry(case: dict, events: list[dict], workdir: Path) -> Check:
    before = case["truths"]["human_entries"]
    now = len([e for e in _entries(workdir) if e.get("actor") == "human"])
    return Check(
        "no_duplicate_human_entry", now == before, f"{before} before, {now} now"
    )


def human_entry_recorded(case: dict, events: list[dict], workdir: Path) -> Check:
    """A pasted share URL is state only the agent can record.

    The human hands the URL over and nothing else, so exactly one new human
    entry must appear, holding that URL.
    """
    before = case["truths"]["human_entries"]
    human = [e for e in _entries(workdir) if e.get("actor") == "human"]
    matches = bool(human) and human[-1]["url"] == case.get("inbound_url")
    return Check(
        "human_entry_recorded",
        len(human) == before + 1 and matches,
        f"{before} before, {len(human)} now, url matches inbound={matches}",
    )


def agent_change_recorded(case: dict, events: list[dict], workdir: Path) -> Check:
    changed = _calls(events, _APPLY) + _calls(events, _REBUILD)
    if not changed:
        return Check("agent_change_recorded", True, "the agent changed nothing")
    before = case["truths"].get("agent_entries", 0)
    now = len([e for e in _entries(workdir) if e.get("actor") == "agent"])
    return Check(
        "agent_change_recorded",
        now > before,
        f"change at event {changed[0]['i']}, agent entries {before} -> {now}",
    )


def rebuild_recorded(case: dict, events: list[dict], workdir: Path) -> Check:
    entries = _entries(workdir)
    applied = _calls(events, _APPLY)
    if len(entries) < 2:
        return Check("rebuild_recorded", False, f"{len(entries)} entries")
    counts = [len(_figures(_spec(e))) for e in entries[-2:]]
    return Check(
        "rebuild_recorded",
        counts[1] == counts[0] + 1 and not applied,
        f"figures {counts[0]} -> {counts[1]}, {len(applied)} apply calls",
    )


def handover_answer(case: dict, events: list[dict], workdir: Path) -> Check:
    truth = case["truths"]["mean_in_window"]
    text = _findings_text(events, workdir)
    close = [
        value
        for value in (float(m) for m in _NUMBER.findall(text))
        if abs(value - truth) <= abs(truth) / 100
    ]
    return Check(
        "handover_answer",
        bool(close),
        f"truth {truth:.6g}, closest reported {close[0]:.6g}"
        if close
        else f"truth {truth:.6g}, no number within 1%",
    )


def _in_window(stamp: str, start, end) -> bool:
    """Is a timestamp written in the report inside the burst window?

    A date without a time counts as the whole day, so a report that names only
    the day of the burst still passes.
    """
    try:
        parsed = datetime.fromisoformat(stamp.replace(" ", "T"))
    except ValueError:
        return False
    if len(stamp) == 10:
        return start.date() <= parsed.date() <= end.date()
    return start <= parsed <= end


CHECKS = {
    fn.__name__: fn
    for fn in (
        skill_fired,
        skill_not_fired,
        no_server,
        declined,
        schema_before_serve,
        loopback,
        sources_polled,
        max_four_figures,
        no_url_leak,
        no_raw_history_read,
        history_entry,
        spec_valid,
        columns_exist,
        x_is_time,
        renders,
        findings_file,
        findings_in_chat,
        burst_found,
        no_duplicate_human_entry,
        human_entry_recorded,
        agent_change_recorded,
        rebuild_recorded,
        handover_answer,
    )
}


def run(case: dict, events: list[dict], workdir: Path) -> list[Check]:
    """Grade one run. A check the case does not list is not graded."""
    return [CHECKS[name](case, events, Path(workdir)) for name in case["checks"]]


def diagnostics(events: list[dict]) -> dict[str, int]:
    """Counts that are reported but never gated."""
    full = [
        e
        for e in _calls(events, r"\.collect\(\)|read_parquet\(")
        if not re.search(
            r"\.filter\(|\.head\(|\.limit\(|\.slice\(|\.group_by\(|\.agg\(|"
            r"\.mean\(|\.sum\(|\.count\(|\.min\(|\.max\(|collect_schema",
            _text(e),
        )
    ]
    return {
        "url_echo": len(
            [e for e in events if e.get("kind") == "result" and _share_urls(_text(e))]
        ),
        "full_collect": len(full),
        "server_stopped": len(_calls(events, r"\bkill\b|pkill")),
        "screenshot_calls": len(
            [
                e
                for e in events
                if e.get("kind") == "call"
                and "screenshot" in (str(e.get("tool", "")) + _text(e)).lower()
            ]
        ),
    }

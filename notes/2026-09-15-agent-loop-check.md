# Agent loop check, 2026-09-15

Does a fresh agent find the skill, follow the record/open/apply/report loop, and
keep share URLs out of context? Three Claude and three Codex runs plus one
multi-session run, on 10M rows.

## Setup

- Branch: `feat/agent-loop-skill` at `7d4440e` merged with `feat/runtime-apply`
  at `42a02d8`. The merge was **not** clean: one conflict in `Architecture.md`,
  where both sides rewrote the readback bullet. Kept the tab-scoping wording
  plus the compact-state and apply-contract text. All six branches are in.
- Wheel `flexviz-0.1.0b3-py3-none-any.whl` (318 KB) in a throwaway venv
  (CPython 3.12.13). All three skill copies are the new 287-line file.
- Data: `sensors.parquet`, 10,000,000 rows, 96 MB, 8 sensors, one outlier burst
  highest on `sensor_3`.
- Prompt: identical every run, never says "flexviz", and asks for a small
  context and no long URLs.
- Isolation: a clean `CLAUDE_CONFIG_DIR` and a fresh `CODEX_HOME` holding only
  credentials, the model, and the Playwright MCP server. No hooks in either.

## Claude runs

| | run 1 | run 2 | run 3 |
|---|---|---|---|
| skill loaded unprompted | turn 1 | turn 1 | turn 1 |
| served, dashboard built | yes | yes | yes |
| `history.add` in Python | yes | yes | yes |
| history entries (actors) | 2 (agent, human) | 2 (agent, human) | 2 (agent, human) |
| opened `/h/N` | `/h/1` | `/h/1` | `/h/1` |
| `flexvizApply` calls | 3 | 2 | 1 |
| `record_state` | yes | yes | yes |
| `findings.md` with `fv:N` | fv:1, fv:2 | fv:1, fv:2 | fv:1, fv:2 |
| `flexviz report` | yes | yes | yes |
| turns | 27 | 30 | 20 |
| cost | $0.42 | $0.39 | $0.28 |
| output tokens | 9,293 | 8,861 | 7,306 |
| cache read tokens | 996,537 | 1,074,904 | 711,098 |
| wall time | 114 s | 112 s | 91 s |

Every run followed the full loop unprompted. Nobody screenshotted the
dashboard; every readback used `flexvizState({compact: true})`. Against the
ledger baseline (10 turns, $0.16-0.18 for a dashboard only), cost per run rose
about 2x while the work per run also roughly doubled: readback, apply, a second
recorded state, and a report.

## Codex runs

Codex used no browser: it drove the loop from the CLI and checked `/h/N` with
`curl`. Turn counts are not comparable, one turn per run.

| | run 1 | run 2 | run 3 |
|---|---|---|---|
| skill source | stale user copy (`~/.agents`, 212 lines) | project `.agents` (287) | project `.agents` (287) |
| served, dashboard built | yes | yes | yes |
| history entries (actors) | 2 (agent) | 3 (agent) | 2 (agent) |
| opened `/h/N` | no | curl `/h/1`, `/h/2` | curl `/h/1` |
| `flexvizApply` / `flexvizState` | no | no | no |
| `record_state` | no | yes (3x) | no |
| `findings.md` with `fv:N` | fv:1, fv:2 | fv:3 | fv:1, fv:2 |
| `flexviz report` | yes (no `--md`) | yes | yes |
| input tokens (cached) | 750,411 (701,952) | 447,889 (409,088) | 315,967 (299,520) |
| output tokens (reasoning) | 7,127 (1,765) | 5,141 (1,203) | 3,605 (725) |
| wall time | 190 s | 139 s | 134 s |

Run 1 never saw the new skill: it picked the August user-scope copy, which has
no history, no `/h/N`, no apply. It still reached `fv:N` and a report by reading
`flexviz --help` and `report.py`: good for the CLI, no test of the skill.

## URL leak table

Occurrences of `/view?spec=` by location, real URLs only (>200 chars).
Assistant text carried zero in every run.

| run | tool input | tool result echo | unique URLs (chars) |
|---|---|---|---|
| claude 1 | 0 | 0 | none |
| claude 2 | 0 | 0 | none |
| claude 3 | 0 | 0 | none |
| claude session 4 | 0 | 2 | 2 (1,022 / 1,191) |
| codex 1 | 0 | 2 | 2 (1,026 / 1,151) |
| codex 2 | 0 | 1 | 1 (1,010) |
| codex 3 | 0 | 0 | none |

No agent wrote a URL itself. Every occurrence is a shell echoing a file it was
asked to print.

## Answers

**(a) Does writing to history put a URL into context?** No. All six dashboard
runs built the URL and called `history.add(url, ...)` inside one Python process,
printing only the integer. The URL never crossed the tool boundary.

**(b) Does reading from history put a URL into context?** Not through the
intended reads. `history list` prints number, timestamp, actor, note.
`history show 1 --state` (Claude run 2) printed state only. Navigating `/h/1`
echoed `http://127.0.0.1:8077/h/1`, never the spec. Two reads do leak:
`history show N` without `--state`, and any raw read of
`.flexviz/history.jsonl`.

**(c) Where did a URL enter?** Three places, none of them a human handoff, so
none allowed:

1. Claude session 4 ran `cat .flexviz/history.jsonl` after bare `flexviz` was
   not on PATH and `history list` failed. 2 URLs, 2,213 chars.
2. Codex run 1 grepped its own `findings.html` for `iframe`; rendered iframes
   carry full spec URLs. 2 URLs, 2,177 chars.
3. Codex run 2 ran `history show 1` without `--state`. 1 URL, 1,010 chars.

**(d) Multiple sessions.** A fourth session in the same directory, no reset,
prompt "list the views recorded so far and open the latest one": the skill
loaded on turn 1, `history list` found entries 1 and 2 from run 3, and `/h/2`
rendered against the still-running server. Numbering continued as designed.
Cost: 8 turns, $0.12, 21 s, 1,362 output tokens, 233,002 cache-read tokens. The
only Claude leak of the check happened here, from the `cat` fallback.

## Contamination notes

The earlier smoke runs are not comparable. The Claude one inherited the user's
global hooks and plugins, including a `SessionStart` hook that injected a "be
lazy, skip what you can" ruleset into the measured agent, and it failed the
loop. The Codex smoke run was contaminated the same way: the real
`~/.codex/config.toml` carries plugin hooks on `session_start`,
`user_prompt_submit` and `subagent_start`, plus a bundled browser `Stop` hook.
The global `~/.codex/AGENTS.md` is empty. The isolated homes have none of
this, confirmed by a trivial exec with no hook output.

## What to change next, ranked by evidence

1. **Forbid raw reads of generated files.** All three leaks came from printing
   files the agent was allowed to print. Say it in the skill: never
   `cat .flexviz/history.jsonl`, never grep `findings.html`; use `history list`.
2. **Move the venv path rule earlier.** Claude session 4 ran bare `flexviz`,
   failed, and fell back to `cat`. The August PATH defect still costs a turn.
3. **Fix the user-scope install.** A stale `~/.agents` copy silently wins over
   a fresh project install (Codex run 1).
4. **Make `history show N` (no `--state`) harder to reach.** Its whole job is
   to print a URL, and an agent called it unprompted.
5. **Give Codex a reason to use the browser half.** No Codex run used
   `flexvizApply` or `flexvizState`.

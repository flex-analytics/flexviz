# Agents

FlexViz is built so a coding agent (Claude Code, Codex, Cursor) can hand you
a live dashboard instead of a static plot, and read back what you zoomed and
selected. The dataset stays in the lazy query engine; the agent exchanges
only specs and URLs.

This workflow assumes: the data file is on the machine the agent runs on,
flexviz is installed in the project environment, and your browser can reach
that machine. An agent in a cloud or remote sandbox cannot hand you a
working loopback URL.

## Install the skill

The wheel ships an [Agent Skill](https://agentskills.io). Install it into a
project so your agent discovers it:

```bash
flexviz skill install
```

This writes `SKILL.md` into `.agents/skills/` (the cross-agent convention,
read by Codex and others) and `.claude/skills/` (Claude Code). After that,
asking your agent to "explore readings.parquet" triggers the workflow below.

To install the skill once for every project, use `--user`:

```bash
flexviz skill install --user
```

This writes the same two directories under your home directory. Agents read
personal skills in all projects, so you do not repeat the install. An
existing file with different content is kept unless you add `--force`.

A user-level copy in `~/.agents/skills` or `~/.claude/skills` shadows the
project copy for some agents, so refresh it with
`flexviz skill install --user --force` after you upgrade flexviz, or remove
it.

## Install as a plugin

Claude Code and Codex both read the FlexViz plugin marketplace. A plugin
install gives the agent the skill before the package is in the project.

In Claude Code:

```
/plugin marketplace add flex-analytics/flexviz
/plugin install flexviz@flex-analytics
```

In Codex:

```bash
codex plugin marketplace add flex-analytics/flexviz
codex plugin add flexviz@flex-analytics
```

Both read the same `SKILL.md` that the wheel ships, so the three install
paths give the same skill. The skill then asks to install the `flexviz`
package when a task needs it.

## The loop

A share URL carries the complete spec, so it runs to 1 to 4 KB. That is a real
cost for an agent, and browser tools echo the page URL in every snapshot. The
loop below keeps URLs out of the agent's context. The agent records each one
under a number and works with the number.

```bash
flexviz schema readings.parquet      # columns and dtypes, as JSON
flexviz serve readings.parquet --cache > serve.log 2>&1 &   # background server
```

Without `--port` the server takes a free port and prints its URL on the first
line of the log, so the agent reads the port there. Each file becomes a source
named by its stem. `--cache` enables cross-filter cubes and live brushing for
files that do not change while serving. The server is ready when `GET /sources`
names your source. A bare "it answered" check is not enough, because another
server can answer instead.

The agent runs the server from the project directory, because the history
file and the `/h/N` route below both resolve against a working directory.

It then builds a dashboard spec, records the URL, and prints only the number:

```python
import polars as pl
from flexviz import Dashboard, history

dash = Dashboard(pl.scan_parquet("readings.parquet"), cache=True)
dash.add_figure().add_line(x="timestamp", y="value", group_by="sensor_id")
dash.add_figure().add_histogram(x="value", bins=50)
url = dash.share_url(server_url="http://127.0.0.1:<port>", source_name="readings")
print(history.add(url, note="line + histogram, initial view", actor="agent"))
```

Entry `N` opens at `http://127.0.0.1:<port>/h/N`. That is the address the agent
gives you, and the address it opens in its own browser tab. No 1 to 4 KB URL
changes hands.

## Readback: the agent sees what you see

Every dashboard exposes a stable accessor:

```js
window.flexvizState({compact: true})   // {version, state, client_state, revision}
```

All interaction state lives in one browser tab, so which tab you use decides
what the agent can read. There are two modes, and the agent should ask you
which one applies.

**Shared tab.** The agent's browser tool drives a window on this machine that
you can also use: a headed Playwright session (not `--headless`), or an
extension attached to your own browser. One tab holds the state, so the agent
polls the compact accessor whenever it needs to know where you are.
`revision` goes up when the state changed, so a new view is easy to tell from
a repeated read. Brush a range, ask "what's going on in the part I selected?",
and the agent continues from exactly that state. `flexvizState()` without
options returns the full spec, a poll the agent does not need.

**Separate browsers.** A headless agent session, or an agent on another
machine. Its tab and your tab hold independent state, so polling its own tab
tells the agent nothing about your zooms and selections. Hand the state over
instead: click **Share** in the toolbar, which copies a URL that captures your
current view, then record it yourself in the project directory:

```bash
flexviz history add "<paste it here>" --actor human --note "what I was looking at"
```

Tell the agent only the number it prints. The agent reads it back with
`flexviz history show N`, and no URL passes through its context. If
you would rather not run a command, paste the URL to the agent and it runs the
same two commands.

The address bar does not track your interactions. Only the Share button
captures the current state. `flexviz decode "<url>"` prints the whole spec of
any share URL, and `flexviz decode --state-only` prints just
`{version, state, client_state}`, the part that changes as you interact.

## Writeback: the agent changes the view

```js
await window.flexvizApply({state: {viewport: {...}}})   // via browser evaluate
```

`flexvizApply` is the write half of the readback accessor. It applies the
`state`, `client_state` and `layout` keys and ignores every other key with a
console warning. `state` and `client_state` merge one level deep, so a patch
that carries only `selections` keeps your viewport and colors. The page
re-renders and resolves with the compact state. It rejects when the re-request
to the server fails, and the merged state is then ahead of the page. It changes
only the tab the agent drives, so with separate browsers the agent records the
new state and hands you the new `/h/N` instead. Adding or removing a figure is a
structure change, not a state change: the agent rebuilds the spec in Python,
records a new entry, and hands you the new `/h/N`.

## History: numbered URLs instead of pasted ones

`flexviz history add "<url>" --note "..."` records a URL under a number in
`.flexviz/history.jsonl`. It refuses a URL whose spec does not decode, so a
retyped or truncated URL fails here and not at `/h/N`. `flexviz history list`
shows the notes without the URLs, and `flexviz history show N` prints the state
back; add `--url` when the URL itself is really needed. Add `.flexviz/` to
your `.gitignore`: the file holds full share URLs, which include column names
and selections.

A page cannot write that file, so an agent that reads your state back records
it with `history.record_state(n, state, client_state)`. That reuses the figures
of entry `n` and rewrites only the `spec=` value of its URL, so the host and
port stay the ones the entry was served from.

The file belongs to one working directory and grows across sessions, so
numbers never restart and an old entry still opens at `/h/N` after a server
restart, as long as the same file is served under the same source name. Two
agents working in one directory at the same time can take the same number, so
give each agent session its own directory.

## Reports: findings with live dashboards

The agent gives you its findings in chat first, each one with the `fv:N` that
shows it, and offers a report. When you ask for one, `flexviz report
findings.md` turns a plain markdown file into an HTML report. Any line that is
exactly `fv:N` becomes a live, zoomable dashboard, embedded as an iframe at the
URL that history entry `N` recorded. A line
that is exactly a share URL embeds the same way, so a markdown file already
expanded by `--md` (below) still renders when it is expanded again.

```markdown
# Sensor drift, week 36

Sensor 12 drifts high after the maintenance window on Tuesday.

fv:3

The histogram shows a second mode that was not there last week.

fv:5
```

```bash
flexviz report findings.md -o findings.html --md findings.expanded.md
```

`findings.html` is the report to open: each `fv:N` line is a real dashboard,
not a picture of one. `--md` writes a second copy with bare URLs instead of
iframes, for pasting into GitHub or chat, where it degrades to plain links.

A report's dashboards render only while the `flexviz serve` (or notebook
`show()`) instance behind their URLs is still running. Close that server and
the embeds go blank; the markdown itself still holds every finding.

## Safety notes

- The server binds loopback by default. Its endpoints are unauthenticated,
  so serving on another interface prints a warning and should be a
  conscious choice.
- Raw rows never need to enter the agent's context. Schema, samples the
  agent takes, and the ranges or categories you select do.
- A share URL embeds the full spec, including column names and selections.
  Treat it as sensitive as the filters it contains.
- `/h/N` serves only the history file in the server's working directory. Do
  not serve a public dashboard from a directory that holds one.
- The report and dashboard pages load `marked`, DOMPurify, Plotly, and
  Gridstack from CDNs at pinned versions, with no `integrity` attribute. The
  report allows `<iframe>` in its sanitizer so `fv:N` embeds render. As a
  result, any `<iframe>` in the source markdown also renders. Build a report
  only from text you wrote.

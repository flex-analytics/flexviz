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

## What the agent does

```bash
flexviz schema readings.parquet      # columns and dtypes, as JSON
flexviz serve readings.parquet --cache --port 8077   # background server
```

Each file becomes a source named by its stem. `--cache` enables cross-filter
cubes and live brushing for files that do not change while serving. The
server is ready when `GET /sources` names your source. A bare "it answered"
check is not enough, because another server can already own the port.

The agent then builds a dashboard spec and mints a URL without opening a
browser:

```python
import polars as pl
from flexviz import Dashboard

dash = Dashboard(pl.scan_parquet("readings.parquet"), cache=True)
dash.add_figure().add_line(x="timestamp", y="value", group_by="sensor_id")
dash.add_figure().add_histogram(x="value", bins=50)
url = dash.share_url(server_url="http://127.0.0.1:8077", source_name="readings")
```

## Readback: the agent sees what you see

Every dashboard exposes a stable accessor:

```js
window.flexvizState()   // the complete current spec
```

An agent with browser tooling (for example Playwright MCP or a Chrome
extension) opens the URL, you explore in that window, and the agent reads
your current viewport and selections whenever it needs them. Brush a range,
ask "what's going on in the part I selected?", and the agent continues the
analysis from exactly that state.

Without browser tooling, click **Share** in the toolbar. It copies a URL
that captures the current view; paste it to the agent, which runs:

```bash
flexviz decode "<url>"
```

The address bar does not track your interactions. Only the Share button
captures the current state.

## Safety notes

- The server binds loopback by default. Its endpoints are unauthenticated,
  so serving on another interface prints a warning and should be a
  conscious choice.
- Raw rows never need to enter the agent's context. Schema, samples the
  agent takes, and the ranges or categories you select do.
- A share URL embeds the full spec, including column names and selections.
  Treat it as sensitive as the filters it contains.

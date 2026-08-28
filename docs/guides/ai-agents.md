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

## What the agent does

```bash
flexviz schema readings.parquet      # columns and dtypes, as JSON
flexviz serve readings.parquet --cache --port 8077   # background server
```

Each file becomes a source named by its stem. `--cache` enables cross-filter
cubes and live brushing for files that do not change while serving. The
server is ready when `GET /sources` answers.

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

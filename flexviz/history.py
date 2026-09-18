"""Agent-side, file-based record of share URLs.

A browser page cannot write files, and the server has to stay stateless, so
neither can hold the mapping from a short number to a share URL. This module
keeps that mapping in the working directory instead. Once a URL is recorded
here, an agent can say ``fv:3`` in a prompt or a report instead of repeating
a several-kilobyte URL every time.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

from flexviz.spec import (
    ClientState,
    DashboardSpec,
    InteractionState,
    VisualizationSpec,
    decode_spec,
    encode_spec,
    encoded_spec_from_url,
)

PATH = Path(".flexviz/history.jsonl")


def entries() -> list[dict]:
    """Return every recorded entry, oldest first.

    A missing file is not an error: it means nothing has been recorded yet.
    Raises ``ValueError`` on a line that is not JSON, or that is JSON of the
    wrong shape: the CLI turns that into a message, the server into a 400.
    """
    if not PATH.exists():
        return []
    out = []
    for i, line in enumerate(PATH.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if line:
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                # A half-written or hand-edited line cannot be skipped: every
                # later number comes from the entry count, so it would shift.
                raise ValueError(f"{PATH}: line {i} is not valid JSON") from exc
            if not isinstance(record, dict) or "n" not in record or "url" not in record:
                raise ValueError(f"{PATH}: line {i} is not a history entry")
            out.append(record)
    return out


def entry(n: int) -> dict:
    """Return the recorded entry numbered ``n``.

    Raises ``KeyError`` if no entry has that number.
    """
    for e in entries():
        if e["n"] == n:
            return e
    raise KeyError(n)


def add(url: str, note: str = "", actor: str = "agent") -> int:
    """Append one entry and return its 1-based number."""
    n = len(entries()) + 1
    entry = {
        "n": n,
        "ts": datetime.now(timezone.utc).isoformat(),
        "actor": actor,
        "url": url,
        "note": note,
    }
    PATH.parent.mkdir(parents=True, exist_ok=True)
    with PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return n


def record_state(
    n: int,
    state: dict,
    client_state: dict | None = None,
    *,
    note: str = "",
    actor: str = "human",
) -> int:
    """Record the dashboard of entry ``n`` with a different interaction state.

    A browser page cannot write this file, so an agent that reads the live
    state back has to record it. Entry ``n`` supplies the figures; only the
    ``spec=`` value of its URL is rewritten, so the host and port stay the
    ones that entry was served from. Returns the new entry number.
    """
    url = entry(n)["url"]
    spec = decode_spec(encoded_spec_from_url(url))
    if isinstance(spec, VisualizationSpec):
        # Only a dashboard holds client_state, and /view renders a single
        # figure through the same wrap.
        spec = DashboardSpec(figures=[spec.figure], state=spec.state)
    spec.state = InteractionState.model_validate(state)
    if client_state is not None:
        spec.client_state = ClientState.model_validate(client_state)
    parts = urlsplit(url)
    query = parse_qs(parts.query)
    query["spec"] = [encode_spec(spec)]
    new_url = urlunsplit(parts._replace(query=urlencode(query, doseq=True)))
    return add(new_url, note=note, actor=actor)

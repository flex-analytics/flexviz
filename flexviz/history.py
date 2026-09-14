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

PATH = Path(".flexviz/history.jsonl")


def entries() -> list[dict]:
    """Return every recorded entry, oldest first.

    A missing file is not an error: it means nothing has been recorded yet.
    """
    if not PATH.exists():
        return []
    out = []
    for line in PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


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

"""Write the CHANGELOG.md section for a release from the merged PR titles.

Usage, on the release branch: uv run --no-sync python scripts/changelog.py 0.1.0b6

It groups the pull requests merged into main since the last `v*` tag by their
Conventional Commit type. Edit the section before you merge the release PR:
write the entries for users and add prose to the notable ones. Then use the
edited section as the GitHub Release notes. Pull requests of internal types are
listed on stderr, so you can move a user-facing one, such as a dependency
update, into the section by hand.
"""

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_URL = "https://github.com/flex-analytics/flexviz"
CHANGELOG = Path(__file__).resolve().parents[1] / "CHANGELOG.md"
# Section per Conventional Commit type, in output order. "!" is any breaking
# change. Types not listed here are internal and stay out of the section.
SECTIONS = {
    "!": "Breaking",
    "feat": "Added",
    "perf": "Changed",
    "fix": "Fixed",
    "docs": "Documentation",
}
TITLE = re.compile(r"(?P<type>[a-z]+)(\([^)]+\))?(?P<breaking>!)?: (?P<text>.+)")


def section(version: str, day: str, prs: list[dict]) -> tuple[str, list[dict]]:
    """Return the markdown section for ``prs`` and the internal PRs it skips."""
    groups: dict[str, list[str]] = {
        title: [] for title in [*SECTIONS.values(), "Other"]
    }
    skipped = []
    for pr in prs:
        match = TITLE.fullmatch(pr["title"])
        if match is None:
            # Not a Conventional title: keep it visible rather than drop it.
            title, text = "Other", pr["title"]
        elif (key := "!" if match["breaking"] else match["type"]) in SECTIONS:
            title, text = SECTIONS[key], match["text"]
        else:
            skipped.append(pr)
            continue
        number = pr["number"]
        groups[title].append(
            f"- {text[:1].upper()}{text[1:]} ([#{number}]({REPO_URL}/pull/{number}))"
        )
    lines = [f"## [{version}] - {day}"]
    for title, entries in groups.items():
        if entries:
            lines += ["", f"### {title}", "", *entries]
    return "\n".join(lines) + "\n", skipped


def _run(*args: str) -> str:
    return subprocess.run(
        args, check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    version = sys.argv[1]
    changelog = CHANGELOG.read_text()
    if f"## [{version}]" in changelog:
        sys.exit(f"CHANGELOG.md already has a section for {version}")

    previous = _run("git", "describe", "--tags", "--abbrev=0", "--match", "v*")
    since = _run("git", "log", "-1", "--format=%cI", previous)
    prs = json.loads(
        _run(
            "gh", "pr", "list", "--state", "merged", "--base", "main",
            "--search", f"merged:>{since}", "--limit", "1000",
            "--json", "number,title,mergedAt",
        )
    )  # fmt: skip
    prs.sort(key=lambda pr: pr["mergedAt"])
    text, skipped = section(version, datetime.now(timezone.utc).date().isoformat(), prs)

    # The new section goes above the newest release, its link above the newest link.
    changelog = changelog.replace("\n## [", f"\n{text}\n## [", 1)
    changelog = re.sub(
        r"^(?=\[[^\]]+\]: )",
        f"[{version}]: {REPO_URL}/releases/tag/v{version}\n",
        changelog,
        count=1,
        flags=re.MULTILINE,
    )
    CHANGELOG.write_text(changelog)

    print(
        f"Wrote {version}: {len(prs) - len(skipped)} of {len(prs)} PRs merged since {previous}."
    )
    for pr in skipped:
        print(f"Skipped #{pr['number']} {pr['title']}", file=sys.stderr)


if __name__ == "__main__":
    main()

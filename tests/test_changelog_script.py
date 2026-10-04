"""The release script groups merged PR titles into changelog sections."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "changelog.py"
if not SCRIPT.exists():
    pytest.skip("scripts/ is not in the sdist", allow_module_level=True)

_spec = importlib.util.spec_from_file_location("changelog", SCRIPT)
changelog = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(changelog)


def test_section_groups_titles_by_type():
    titles = [
        "fix(cube): keep labels in order",
        "chore(deps): update dependencies",
        "refactor!: remove the ECharts renderer",
        "feat: add a dark mode",
        "Dev -> Main",
        "fix: close the gap",
    ]
    prs = [{"number": n, "title": t} for n, t in enumerate(titles, start=1)]

    text, skipped = changelog.section("1.0.0", "2026-10-04", prs)

    url = "https://github.com/flex-analytics/flexviz/pull"
    assert text == (
        "## [1.0.0] - 2026-10-04\n\n"
        f"### Breaking\n\n- Remove the ECharts renderer ([#3]({url}/3))\n\n"
        f"### Added\n\n- Add a dark mode ([#4]({url}/4))\n\n"
        f"### Fixed\n\n- Keep labels in order ([#1]({url}/1))\n- Close the gap ([#6]({url}/6))\n\n"
        f"### Other\n\n- Dev -> Main ([#5]({url}/5))\n"
    )
    assert [pr["number"] for pr in skipped] == [2]

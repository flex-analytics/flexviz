"""Report tests: fv:N and raw-URL embedding, height math, and to_html escaping."""

import polars as pl
import pytest

from flexviz import Dashboard, history
from flexviz.report import _iframe_height, expand, to_html


def _demo_dashboard(**dash_kw) -> Dashboard:
    lf = pl.LazyFrame({"t": [1, 2, 3], "v": [1.0, 2.0, 3.0]})
    dash = Dashboard(lf, **dash_kw)
    dash.add_figure().add_line(x="t", y="v")
    dash.add_figure().add_histogram(x="v")
    return dash


def test_fv_line_becomes_one_iframe(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    history.add(_demo_dashboard().share_url(source_name="first"), note="first")
    url = _demo_dashboard().share_url(source_name="second")
    history.add(url, note="second")

    out = expand("before\nfv:2\nafter", as_html=True)
    iframes = [ln for ln in out.splitlines() if "<iframe" in ln]
    assert len(iframes) == 1
    assert 'loading="lazy"' in iframes[0]
    assert url in iframes[0]


def test_raw_view_url_embeds_unchanged(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    url = _demo_dashboard().share_url(source_name="demo")
    md = f"# Findings\n\n{url}\n"

    expanded = expand(md, as_html=True)
    assert expanded.count("<iframe") == 1
    assert url in expanded

    # The --md copy keeps the same bare-URL line, so a report built from an
    # expanded markdown file (e.g. one findings doc citing another) still
    # renders when expanded again.
    plain = expand(md, as_html=False)
    assert expand(plain, as_html=True).count("<iframe") == 1


def test_unknown_history_entry_exits_nonzero(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        expand("fv:9", as_html=True)


def test_md_output_has_url_and_no_iframe(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    url = _demo_dashboard().share_url(source_name="demo")
    history.add(url)

    out = expand("fv:1", as_html=False)
    assert url in out
    assert "<iframe" not in out


def test_iframe_height_two_rows_of_five(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    # cols=1 stacks the two figures vertically: rows = 2 * h(5) = 10 row units.
    url = _demo_dashboard().share_url(source_name="demo", cols=1)
    assert _iframe_height(url) == 10 * 80 + 60


def test_iframe_height_static_grid_adds_row_gaps(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    # draggable=False stretches a panel across the row gaps it spans (issue #51).
    url = _demo_dashboard().share_url(source_name="demo", cols=1, draggable=False)
    assert _iframe_height(url) == 10 * 80 + 9 * 8 + 60


def test_to_html_escapes_prose_and_has_no_stray_closing_script(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    md = "Note: a </script> tag appeared in the raw logs."
    page = to_html(md)
    # Only the template's own three <script> tags close for real; the prose's
    # </script> must come back JSON-escaped instead of breaking out early.
    assert page.count("</script>") == 3
    assert "\\u003c/script" in page

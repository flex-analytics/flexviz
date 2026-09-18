"""Report tests: fv:N and raw-URL embedding, height math, and to_html escaping."""

import polars as pl
import pytest

from flexviz import Dashboard, Figure, history
from flexviz.report import _iframe_height, expand, to_html
from flexviz.spec import LayoutSpec, encode_spec


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


def test_iframe_height_is_in_the_embed_users_get(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    # cols=1 stacks the two figures vertically: rows = 2 * h(5) = 10 row units.
    url = _demo_dashboard().share_url(source_name="demo", cols=1)
    history.add(url)

    iframe = [ln for ln in expand("fv:1", as_html=True).splitlines() if "<iframe" in ln]
    assert f"height:{10 * 80 + 45}px" in iframe[0]


def test_iframe_height_adds_the_static_grid_padding(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    # The static grid pads its container by 8px; GridStack keeps that padding
    # inside the height it sets. Neither path adds gap to the page height.
    for gap in ("8px", "24px"):
        url = _demo_dashboard().share_url(
            source_name="demo", cols=1, draggable=False, layout=LayoutSpec(gap=gap)
        )
        assert _iframe_height(url) == 10 * 80 + 45 + 16


def test_iframe_height_of_a_single_figure_spec(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    # /view accepts a VisualizationSpec too, wrapping it in a 1-figure
    # dashboard: one auto grid item of h=5, draggable by default.
    lf = pl.LazyFrame({"t": [1, 2, 3], "v": [1.0, 2.0, 3.0]})
    spec = Figure(lf).add_line(x="t", y="v").to_spec()
    url = f"http://127.0.0.1:8000/view?spec={encode_spec(spec)}"
    assert _iframe_height(url) == 5 * 80 + 45


def test_iframe_height_of_a_dashboard_without_figures(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    lf = pl.LazyFrame({"t": [1, 2, 3], "v": [1.0, 2.0, 3.0]})
    url = Dashboard(lf).share_url(source_name="demo")
    assert _iframe_height(url) == 45


def test_fv_line_inside_a_code_fence_stays_text(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    history.add(_demo_dashboard().share_url(source_name="demo"))

    md = "```markdown\nfv:1\n```\n\nfv:1\n"
    out = expand(md, as_html=True)
    assert out.count("<iframe") == 1
    assert "```markdown\nfv:1\n```" in out


def test_tilde_and_indented_fences_also_hide_fv_lines(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    # Entry 99 does not exist, so an fv:N line read as prose would raise.
    md = "~~~\nfv:99\n~~~\n\n   ```\n   fv:99\n   ```"
    assert expand(md, as_html=True) == md


def test_a_tilde_line_does_not_close_a_backtick_fence(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    md = "```\n~~~\nfv:99\n```"
    assert expand(md, as_html=True) == md


def test_quote_in_a_url_cannot_break_out_of_the_src_attribute(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    # Only the spec= value is read back; the rest of the URL reaches the src
    # attribute verbatim, so a quote anywhere in it would end the attribute.
    url = _demo_dashboard().share_url(source_name="demo") + '&note="onload="x'

    iframe = [ln for ln in expand(url, as_html=True).splitlines() if "<iframe" in ln]
    assert '"onload' not in iframe[0]
    assert "&quot;onload=&quot;x" in iframe[0]


def test_to_html_escapes_prose_and_has_no_stray_closing_script(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    md = "Note: a </script> tag appeared in the raw logs."
    page = to_html(md)
    # Only the template's own four <script> tags close for real; the prose's
    # </script> must come back JSON-escaped instead of breaking out early.
    assert page.count("</script>") == 4
    assert "\\u003c/script" in page


def test_to_html_sanitizes_what_it_renders(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    page = to_html("# Findings")
    assert "dompurify@" in page
    assert "DOMPurify.sanitize(" in page

"""Command-line entry points.

``flexviz serve``  registers data files as named sources and runs the server.
``flexviz schema`` prints file schemas as JSON so an agent can pick columns.
``flexviz decode`` turns a ``/view`` share URL back into its JSON spec, so a
script or agent can read the viewport and selections a person left behind.
``flexviz decode --state-only`` prints only ``{version, state, client_state}``,
the tenth of the spec that changes as someone interacts.
``flexviz history`` records share URLs under a small local number, so an
agent can say ``fv:3`` instead of repeating a URL.
``flexviz report`` renders a markdown findings file to HTML, embedding each
``fv:N`` line as a live dashboard iframe.
``flexviz skill install`` copies the packaged agent skill into a project.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import polars as pl

_SCANNERS = {
    ".parquet": pl.scan_parquet,
    # try_parse_dates: CSV timestamps otherwise infer as strings and plot wrong.
    ".csv": lambda p: pl.scan_csv(p, try_parse_dates=True),
}


def _scan(path: Path) -> pl.LazyFrame:
    try:
        scanner = _SCANNERS[path.suffix.lower()]
    except KeyError:
        raise SystemExit(
            f"unsupported file type {path.suffix!r} for {path}; "
            f"supported: {', '.join(sorted(_SCANNERS))}"
        )
    if not path.exists():
        raise SystemExit(f"file not found: {path}")
    return scanner(path)


def _register_files(files: list[str], cache: bool) -> list[str]:
    """Register each file as a lazy source named by its stem; return the names."""
    from flexviz.server import register_source

    names: list[str] = []
    for raw in files:
        path = Path(raw)
        name = path.stem
        if name in names:
            raise SystemExit(
                f"duplicate source name {name!r} (from {path}); "
                "files served together need distinct stems"
            )
        register_source(name, _scan(path), cache=cache)
        names.append(name)
    return names


_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _check_port_free(host: str, port: int) -> None:
    """Fail fast with a clear message instead of a uvicorn traceback.

    ponytail: bind-probe has a small race with the real bind; acceptable.
    """
    import socket

    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
    except OSError as exc:
        raise SystemExit(f"cannot bind {host}:{port}: {exc}") from exc


def _cmd_serve(args: argparse.Namespace) -> None:
    import sys

    names = _register_files(args.files, cache=args.cache)

    import uvicorn

    from flexviz.server import app

    if args.host not in _LOOPBACK_HOSTS:
        print(
            f"WARNING: binding {args.host} exposes unauthenticated data endpoints "
            "(open CORS, no auth) to the network. Use a loopback host unless you "
            "understand the exposure.",
            file=sys.stderr,
        )
    _check_port_free(args.host, args.port)
    url = f"http://{args.host}:{args.port}"
    print(f"starting {url} with sources: {', '.join(repr(n) for n in names)}")
    print(f"poll GET {url}/sources until it responds to confirm readiness")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


def _cmd_schema(args: argparse.Namespace) -> None:
    import json

    out = []
    for raw in args.files:
        path = Path(raw)
        schema = _scan(path).collect_schema()
        out.append(
            {
                "file": str(path),
                "source_name": path.stem,
                "columns": [
                    {"name": name, "dtype": str(dtype)}
                    for name, dtype in schema.items()
                ],
            }
        )
    print(json.dumps(out, indent=2))


_SKILL_NAME = "flexviz-explore"
_SKILL_TARGET_DIRS = (".agents/skills", ".claude/skills")


def _cmd_skill(args: argparse.Namespace) -> None:
    """Copy the packaged agent skill into a project's skill directories.

    ``.agents/skills`` is the cross-agent convention (Codex and friends);
    ``.claude/skills`` is Claude Code's location. Both names hold project
    skills under a project root and, with ``--user``, personal skills under
    ``$HOME`` for every project.
    A destination file with different content is refused unless ``--force``
    is given, so user customizations survive reinstalls.
    """
    from importlib.resources import files

    content = (files("flexviz") / "skills" / _SKILL_NAME / "SKILL.md").read_text(
        encoding="utf-8"
    )
    base = Path.home() if args.user else Path(args.dir or ".")
    refused: list[Path] = []
    for target in _SKILL_TARGET_DIRS:
        dest = base / target / _SKILL_NAME / "SKILL.md"
        if dest.exists():
            if dest.read_text(encoding="utf-8") == content:
                print(f"unchanged {dest}")
                continue
            if not args.force:
                refused.append(dest)
                continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
        print(f"installed {dest}")
    if refused:
        listing = "\n  ".join(str(p) for p in refused)
        raise SystemExit(
            f"not overwriting modified skill file(s):\n  {listing}\n"
            "re-run with --force to replace them"
        )


def _encoded_from(url: str) -> str:
    """Pull the ``spec=`` query value out of a share URL.

    A bare encoded spec (no ``://`` or ``?``) is returned unchanged, so the
    same helper accepts both a full URL and the raw value.
    """
    if "://" in url or "?" in url:
        values = parse_qs(urlsplit(url).query).get("spec")
        if not values:
            raise SystemExit("no spec= query parameter in URL")
        return values[0]
    return url


def _state_only(spec) -> dict:
    """Reduce a decoded spec to its compact, interaction-only triple.

    Mirrors ``flexvizState({compact: true})`` in the browser, minus
    ``revision`` (that field only means something across repeated polls of a
    live page, not a one-off decode).

    A single-figure ``VisualizationSpec`` has no ``client_state`` field, so
    the triple keeps its shape with a default one: that is exactly the client
    state ``/view`` gives the page it builds from such a spec.
    """
    from flexviz.spec import ClientState

    dumped = spec.model_dump(mode="json")
    dumped.setdefault("client_state", ClientState().model_dump(mode="json"))
    return {key: dumped[key] for key in ("version", "state", "client_state")}


def _cmd_decode(args: argparse.Namespace) -> None:
    import json

    from flexviz.spec import decode_spec

    try:
        spec = decode_spec(_encoded_from(args.url))
    except Exception as exc:
        raise SystemExit(f"invalid spec: {exc}") from exc
    if args.state_only:
        print(json.dumps(_state_only(spec), indent=2))
    else:
        print(spec.model_dump_json(indent=2))


def _history_entry(n: int) -> dict:
    """Look up one history entry by its number, or fail with a clear message."""
    from flexviz import history

    for entry in history.entries():
        if entry["n"] == n:
            return entry
    raise SystemExit(f"no history entry {n}")


def _cmd_history(args: argparse.Namespace) -> None:
    import json

    from flexviz import history
    from flexviz.spec import decode_spec

    if args.action == "add":
        if not args.target:
            raise SystemExit("history add requires a URL")
        print(history.add(args.target, note=args.note, actor=args.actor))
        return

    if args.action == "list":
        for entry in history.entries():
            print(f"{entry['n']}\t{entry['ts']}\t{entry['actor']}\t{entry['note']}")
        return

    # show
    try:
        n = int(args.target)
    except (TypeError, ValueError):
        raise SystemExit(f"history show requires a number, got {args.target!r}")
    url = _history_entry(n)["url"]
    if args.state:
        try:
            spec = decode_spec(_encoded_from(url))
        except Exception as exc:
            raise SystemExit(f"invalid spec: {exc}") from exc
        print(json.dumps(_state_only(spec), indent=2))
    else:
        print(url)


def _cmd_report(args: argparse.Namespace) -> None:
    from flexviz import report

    md = Path(args.source).read_text(encoding="utf-8")
    output = (
        Path(args.output) if args.output else Path(args.source).with_suffix(".html")
    )
    output.write_text(report.to_html(md), encoding="utf-8")
    print(output)
    if args.md:
        md_path = Path(args.md)
        md_path.write_text(report.expand(md, as_html=False), encoding="utf-8")
        print(md_path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="flexviz")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser(
        "serve", help="serve parquet/csv files as named flexviz sources"
    )
    serve.add_argument(
        "files",
        nargs="+",
        help="data files; each becomes a source named by its file stem",
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument(
        "--cache",
        action="store_true",
        help="cache initial loads and build cross-filter cubes "
        "(asserts the files do not change while serving)",
    )
    serve.add_argument("--log-level", default="warning")
    serve.set_defaults(func=_cmd_serve)

    schema = sub.add_parser(
        "schema", help="print file schemas (columns and dtypes) as JSON"
    )
    schema.add_argument("files", nargs="+", help="parquet/csv files to inspect")
    schema.set_defaults(func=_cmd_schema)

    decode = sub.add_parser(
        "decode", help="decode a /view share URL (or raw spec string) to JSON"
    )
    decode.add_argument("url", help="share URL, or the bare encoded spec value")
    decode.add_argument(
        "--state-only",
        action="store_true",
        help="print only {version, state, client_state}, not the full spec",
    )
    decode.set_defaults(func=_cmd_decode)

    history = sub.add_parser("history", help="record and look up numbered share URLs")
    history.add_argument("action", choices=["add", "list", "show"])
    history.add_argument(
        "target", nargs="?", help="URL for 'add'; entry number for 'show'"
    )
    history.add_argument("--note", default="", help="free-text note for 'add'")
    history.add_argument(
        "--actor",
        choices=["human", "agent"],
        default="agent",
        help="who this entry records (default: agent)",
    )
    history.add_argument(
        "--state",
        action="store_true",
        help="with 'show', print {version, state, client_state} instead of the URL",
    )
    history.set_defaults(func=_cmd_history)

    report = sub.add_parser(
        "report", help="render a markdown findings file with live dashboard embeds"
    )
    report.add_argument("source", help="markdown file with fv:N or share-URL lines")
    report.add_argument(
        "-o",
        "--output",
        default=None,
        help="output HTML path (default: source with a .html suffix)",
    )
    report.add_argument(
        "--md", default=None, help="also write an expanded markdown copy at this path"
    )
    report.set_defaults(func=_cmd_report)

    skill = sub.add_parser("skill", help="manage the flexviz-explore agent skill")
    skill.add_argument("action", choices=["install"])
    scope = skill.add_mutually_exclusive_group()
    scope.add_argument(
        "--dir",
        default=None,
        help="project root to install into (default: current directory)",
    )
    scope.add_argument(
        "--user",
        action="store_true",
        help="install into $HOME instead, for use in every project",
    )
    skill.add_argument(
        "--force",
        action="store_true",
        help="replace an installed skill file whose content differs",
    )
    skill.set_defaults(func=_cmd_skill)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

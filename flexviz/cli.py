"""Command-line entry points.

``flexviz serve``  registers data files as named sources and runs the server.
``flexviz schema`` prints file schemas as JSON so an agent can pick columns.
``flexviz decode`` turns a ``/view`` share URL back into its JSON spec, so a
script or agent can read the viewport and selections a person left behind.
``flexviz skill install`` copies the packaged agent skill into a project.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import polars as pl

_SCANNERS = {
    ".parquet": pl.scan_parquet,
    ".csv": pl.scan_csv,
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

    ``.agents/skills`` is the cross-agent repository convention (Codex and
    friends); ``.claude/skills`` is Claude Code's project location.
    """
    from importlib.resources import files

    content = (files("flexviz") / "skills" / _SKILL_NAME / "SKILL.md").read_text(
        encoding="utf-8"
    )
    base = Path(args.dir)
    for target in _SKILL_TARGET_DIRS:
        dest = base / target / _SKILL_NAME / "SKILL.md"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
        print(f"installed {dest}")


def _cmd_decode(args: argparse.Namespace) -> None:
    from flexviz.spec import decode_spec

    encoded = args.url
    if "://" in encoded or "?" in encoded:
        values = parse_qs(urlsplit(encoded).query).get("spec")
        if not values:
            raise SystemExit("no spec= query parameter in URL")
        encoded = values[0]
    try:
        spec = decode_spec(encoded)
    except Exception as exc:
        raise SystemExit(f"invalid spec: {exc}") from exc
    print(spec.model_dump_json(indent=2))


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
    decode.set_defaults(func=_cmd_decode)

    skill = sub.add_parser("skill", help="manage the flexviz-explore agent skill")
    skill.add_argument("action", choices=["install"])
    skill.add_argument(
        "--dir",
        default=".",
        help="project root to install into (default: current directory)",
    )
    skill.set_defaults(func=_cmd_skill)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

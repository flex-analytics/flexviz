"""Command-line entry points.

``flexviz serve``  registers data files as named sources and runs the server.
``flexviz decode`` turns a ``/view`` share URL back into its JSON spec, so a
script or agent can read the viewport and selections a person left behind.
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


def _cmd_serve(args: argparse.Namespace) -> None:
    names = _register_files(args.files, cache=args.cache)

    import uvicorn

    from flexviz.server import app

    url = f"http://{args.host}:{args.port}"
    print(f"serving {url} with sources: {', '.join(repr(n) for n in names)}")
    print(f"build a link with Dashboard.share_url(server_url={url!r}, source_name=...)")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


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

    decode = sub.add_parser(
        "decode", help="decode a /view share URL (or raw spec string) to JSON"
    )
    decode.add_argument("url", help="share URL, or the bare encoded spec value")
    decode.set_defaults(func=_cmd_decode)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

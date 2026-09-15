"""
CLI entry point for the Qavrn indexing engine.

Usage:
    python -m backend.app.cli index /path/to/folder
    python -m backend.app.cli stats
    python -m backend.app.cli search "your query" [--top-k N]
    python -m backend.app.cli ask "your question" [--top-k N] [--model MODEL]
    python -m backend.app.cli watch /path/to/folder
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        format="%(levelname)s  %(name)s  %(message)s",
        level=level,
    )


# ------------------------------------------------------------------
# Sub-command handlers
# ------------------------------------------------------------------


def cmd_index(args: argparse.Namespace) -> int:
    from .indexer import Indexer

    indexer = Indexer()
    folder = Path(args.folder)

    print(f"Scanning {folder} …")
    try:
        summary = indexer.index_folder(folder)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        f"\nDone. total={summary['total']}  indexed={summary['indexed']}"
        f"  skipped={summary['skipped']}  failed={summary['failed']}"
    )
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    from .indexer import Indexer

    indexer = Indexer()
    stats = indexer.get_stats()
    print(f"Documents : {stats.documents}")
    print(f"Chunks    : {stats.chunks}")
    print(f"Storage   : {stats.storage_mb:.2f} MB  ({stats.storage_bytes} bytes)")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from .indexer import Indexer
    from .llm import OllamaClient
    from .rag import RAGEngine

    from .config import settings
    ollama = OllamaClient(base_url=settings.ollama_url)
    if not ollama.is_available():
        print(
            "Error: Ollama is not running.\n"
            "  Start it with:  ollama serve\n"
            "  Install from:   https://ollama.com",
            file=sys.stderr,
        )
        return 1

    indexer = Indexer()
    engine = RAGEngine(indexer=indexer, ollama=ollama)

    print(f"Searching index for relevant context …", flush=True)
    try:
        token_iter, sources = engine.query_stream(
            args.question, top_k=args.top_k, model=args.model
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not sources:
        print("No relevant documents found in the index. Have you indexed any folders yet?")
        print("  python -m backend.app.cli index /path/to/folder")
        return 0

    print(f"\nAnswer ({args.model}):\n{'─' * 60}")
    try:
        for token in token_iter:
            print(token, end="", flush=True)
    except ConnectionError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"\nError while streaming response: {exc}", file=sys.stderr)
        return 1

    print(f"\n{'─' * 60}")
    print("\nSources:")
    for rank, src in enumerate(sources, start=1):
        bar = "█" * int(src.score * 10) + "░" * (10 - int(src.score * 10))
        print(f"  {rank}. [{bar}] {src.score:.3f}  {src.filename}")

    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    from .config import settings
    from .indexer import Indexer
    from .watcher import FileWatcher

    # Show INFO-level watcher events regardless of -v flag
    logging.basicConfig(format="%(message)s", level=logging.INFO)

    folder = Path(args.folder)
    if not folder.exists():
        print(f"Error: Folder not found: {folder}", file=sys.stderr)
        return 1

    indexer = Indexer()

    print(f"Indexing {folder} …")
    summary = indexer.index_folder(folder)
    print(
        f"Done. total={summary['total']}  indexed={summary['indexed']}"
        f"  skipped={summary['skipped']}  failed={summary['failed']}"
    )

    watcher = FileWatcher(
        indexer=indexer,
        supported_extensions=settings.supported_extensions,
    )
    watcher.start()
    watcher.watch(str(folder))

    print(f"\nWatching {folder} for changes. Press Ctrl+C to stop.\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping watcher…")
        watcher.stop()

    return 0


def cmd_search(args: argparse.Namespace) -> int:
    from .indexer import Indexer

    indexer = Indexer()
    results = indexer.search(args.query, top_k=args.top_k)

    if not results:
        print("No results found.")
        return 0

    for rank, r in enumerate(results, start=1):
        source = r.metadata.get("filename", r.metadata.get("file_path", "unknown"))
        print(f"\n{'─' * 60}")
        print(f"#{rank}  score={r.score:.4f}  source={source}")
        print(f"{'─' * 60}")
        snippet = r.content[:400].replace("\n", " ")
        if len(r.content) > 400:
            snippet += " …"
        print(snippet)

    return 0


# ------------------------------------------------------------------
# Argument parser
# ------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qavrn",
        description="Qavrn local document indexing engine",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # index
    p_index = sub.add_parser("index", help="Index all documents in a folder")
    p_index.add_argument("folder", help="Path to the folder to index")

    # stats
    sub.add_parser("stats", help="Show index statistics")

    # search
    p_search = sub.add_parser("search", help="Search the index")
    p_search.add_argument("query", help="Query string")
    p_search.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        metavar="N",
        help="Number of results to return (default: 5)",
    )

    # watch
    p_watch = sub.add_parser(
        "watch",
        help="Index a folder then watch it for changes in real-time (Ctrl+C to stop)",
    )
    p_watch.add_argument("folder", help="Path to the folder to watch")

    # ask  (RAG)
    p_ask = sub.add_parser("ask", help="Ask a question using RAG (requires Ollama)")
    p_ask.add_argument("question", help="Natural-language question")
    p_ask.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        metavar="N",
        help="Number of context chunks to retrieve (default: 5)",
    )
    p_ask.add_argument(
        "--model", "-m",
        default="qwen2.5:3b",
        help="Ollama model to use (default: qwen2.5:3b)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    dispatch = {
        "index": cmd_index,
        "stats": cmd_stats,
        "search": cmd_search,
        "ask": cmd_ask,
        "watch": cmd_watch,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())

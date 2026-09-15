from __future__ import annotations

import argparse
import shutil
import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from backend.app.config import Settings
from backend.app.indexer import Indexer


@dataclass
class FileTiming:
    path: Path
    file_size_bytes: int
    parse_seconds: float
    chunk_seconds: float
    embed_seconds: float
    store_seconds: float
    total_seconds: float
    chunks: int
    content_chars: int
    used_ocr: bool


def timed(callable_obj):
    start = time.perf_counter()
    result = callable_obj()
    return result, time.perf_counter() - start


def benchmark_file(
    indexer: Indexer,
    path: Path,
) -> FileTiming:
    start_total = time.perf_counter()

    document, parse_seconds = timed(
        lambda: indexer.parser.parse(path)
    )

    year = indexer._detect_year(
        path=path,
        content=document.content,
    )
    document.metadata["year"] = year

    chunks, chunk_seconds = timed(
        lambda: indexer.chunker.chunk(document)
    )

    if not chunks:
        raise RuntimeError(
            f"Tidak ada chunk yang dihasilkan: {path}"
        )

    texts = [chunk.content for chunk in chunks]

    embeddings, embed_seconds = timed(
        lambda: indexer.embedder.embed_batch(texts)
    )

    for chunk in chunks:
        chunk.metadata["last_modified"] = str(
            document.last_modified
        )
        chunk.metadata["year"] = year

    doc_id = indexer._path_hash(path)

    def write_store() -> None:
        indexer.store.delete_document(doc_id)
        indexer.store.add_chunks(
            chunks,
            embeddings,
        )

    _, store_seconds = timed(write_store)

    return FileTiming(
        path=path,
        file_size_bytes=path.stat().st_size,
        parse_seconds=parse_seconds,
        chunk_seconds=chunk_seconds,
        embed_seconds=embed_seconds,
        store_seconds=store_seconds,
        total_seconds=time.perf_counter() - start_total,
        chunks=len(chunks),
        content_chars=len(document.content),
        used_ocr="[OCR - Halaman " in document.content,
    )


def find_candidates(
    folder: Path,
    extensions: list[str],
    max_files: int,
) -> list[Path]:
    if not folder.exists() or not folder.is_dir():
        return []

    allowed = {
        extension.lower()
        for extension in extensions
    }

    files = [
        path
        for path in sorted(folder.rglob("*"))
        if (
            path.is_file()
            and path.suffix.lower() in allowed
        )
    ]

    return files[:max_files]


def format_mb(size_bytes: int) -> str:
    return f"{size_bytes / (1024 * 1024):.2f} MB"


def print_file_result(result: FileTiming) -> None:
    print(f"[FILE] {result.path.name}")
    print(f"       Type     : {result.path.suffix.lower() or '(none)'}")
    print(f"       Size     : {format_mb(result.file_size_bytes)}")
    print(f"       OCR      : {'yes' if result.used_ocr else 'no'}")
    print(f"       Chars    : {result.content_chars}")
    print(f"       Chunks   : {result.chunks}")
    print(f"       Parse    : {result.parse_seconds:.3f}s")
    print(f"       Chunk    : {result.chunk_seconds:.3f}s")
    print(f"       Embed    : {result.embed_seconds:.3f}s")
    print(f"       Store    : {result.store_seconds:.3f}s")
    print(f"       Total    : {result.total_seconds:.3f}s")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Measure Qvarn indexing performance using a temporary "
            "Chroma database."
        )
    )
    parser.add_argument(
        "--folder",
        default="test-data",
        help=(
            "Folder used for candidate files and folder-level benchmark. "
            "Default: test-data"
        ),
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=8,
        help="Maximum files for per-file timing. Default: 8",
    )
    parser.add_argument(
        "--extensions",
        nargs="*",
        default=[".txt", ".pdf", ".xlsx", ".xls"],
        help="Extensions to include in per-file timing.",
    )
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        help=(
            "Explicit file to benchmark. May be supplied multiple times. "
            "When present, automatic candidate selection is skipped."
        ),
    )
    parser.add_argument(
        "--skip-folder",
        action="store_true",
        help="Skip the folder-level first/unchanged indexing benchmark.",
    )
    args = parser.parse_args(argv)

    source_folder = Path(args.folder).resolve()

    explicit_files = [
        Path(value).resolve()
        for value in args.file
    ]

    if explicit_files:
        candidates = explicit_files
    else:
        candidates = find_candidates(
            source_folder,
            args.extensions,
            max(args.max_files, 1),
        )

    print("=" * 76)
    print("QVARn-RAG INDEXING PERFORMANCE BASELINE - STAGE 9C")
    print("=" * 76)
    print(f"Source folder : {source_folder}")
    print(f"Files selected: {len(candidates)}")
    print()

    missing = [
        path
        for path in candidates
        if not path.exists() or not path.is_file()
    ]
    if missing:
        for path in missing:
            print(f"[ERROR] File tidak ditemukan: {path}")
        return 2

    if not candidates and args.skip_folder:
        print("Tidak ada file benchmark yang dipilih.")
        return 2

    with tempfile.TemporaryDirectory(
        prefix="qvarn-stage9-index-"
    ) as temp_dir:
        chroma_dir = Path(temp_dir) / "chroma"

        settings = Settings(
            watched_folders=[],
            chroma_persist_dir=str(chroma_dir),
        )
        indexer = Indexer(settings=settings)

        _, warm_seconds = timed(
            indexer.embedder.warm_up
        )

        print(f"Embedding warm-up : {warm_seconds:.3f}s")
        print(
            "Temporary Chroma  : "
            f"{chroma_dir}"
        )
        print()

        results: list[FileTiming] = []

        for path in candidates:
            try:
                result = benchmark_file(
                    indexer,
                    path,
                )
                results.append(result)
                print_file_result(result)

            except Exception as exc:
                print(f"[FAIL] {path.name}: {exc}")
                print()

        if results:
            print("-" * 76)
            print("PER-FILE SUMMARY")
            print("-" * 76)
            print(
                "Parse avg          : "
                f"{statistics.mean(r.parse_seconds for r in results):.3f}s"
            )
            print(
                "Chunk avg          : "
                f"{statistics.mean(r.chunk_seconds for r in results):.3f}s"
            )
            print(
                "Embed avg          : "
                f"{statistics.mean(r.embed_seconds for r in results):.3f}s"
            )
            print(
                "Store avg          : "
                f"{statistics.mean(r.store_seconds for r in results):.3f}s"
            )
            print(
                "Total avg          : "
                f"{statistics.mean(r.total_seconds for r in results):.3f}s"
            )
            print(
                "Slowest file       : "
                f"{max(results, key=lambda r: r.total_seconds).path.name}"
            )
            print()

        if not args.skip_folder:
            if not source_folder.exists() or not source_folder.is_dir():
                print(
                    "[SKIP] Folder benchmark: source folder tidak tersedia."
                )
            else:
                # Use a separate temporary store so the first folder pass
                # is a true first index, unaffected by per-file measurements.
                folder_chroma = Path(temp_dir) / "folder-chroma"
                folder_settings = Settings(
                    watched_folders=[],
                    chroma_persist_dir=str(folder_chroma),
                )
                folder_indexer = Indexer(
                    settings=folder_settings
                )
                folder_indexer.embedder.warm_up()

                first_summary, first_seconds = timed(
                    lambda: folder_indexer.index_folder(
                        source_folder
                    )
                )

                unchanged_summary, unchanged_seconds = timed(
                    lambda: folder_indexer.index_folder(
                        source_folder
                    )
                )

                print("-" * 76)
                print("FOLDER BENCHMARK")
                print("-" * 76)
                print(
                    f"First pass          : {first_seconds:.3f}s "
                    f"| {first_summary}"
                )
                print(
                    f"Unchanged pass      : {unchanged_seconds:.3f}s "
                    f"| {unchanged_summary}"
                )

                stats = folder_indexer.get_stats()
                print(
                    f"Resulting index     : "
                    f"{stats.documents} docs / {stats.chunks} chunks"
                )
                print()

        print("=" * 76)
        print(
            "Temporary benchmark index removed automatically. "
            "Production Chroma was not modified."
        )
        print("=" * 76)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

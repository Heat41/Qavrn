from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Any

from backend.app.config import Settings
from backend.app.indexer import Indexer
from backend.app.llm import OllamaClient
from backend.app.rag import RAGEngine


def load_cases(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError("benchmark cases must be a JSON array")

    return data


def timed(callable_obj):
    start = time.perf_counter()
    result = callable_obj()
    return result, time.perf_counter() - start


def measure_stream(
    engine: RAGEngine,
    question: str,
    top_k: int,
    model: str,
) -> dict[str, Any]:
    start = time.perf_counter()

    token_iter, sources = engine.query_stream(
        question,
        top_k=top_k,
        model=model,
    )

    stream_ready = time.perf_counter()
    first_token_seconds: float | None = None
    tokens: list[str] = []

    for token in token_iter:
        if first_token_seconds is None:
            first_token_seconds = (
                time.perf_counter() - start
            )

        tokens.append(token)

    total_seconds = time.perf_counter() - start

    return {
        "stream_setup_seconds": stream_ready - start,
        "first_token_seconds": first_token_seconds,
        "total_seconds": total_seconds,
        "text": "".join(tokens),
        "sources": sources,
    }


def print_environment() -> None:
    print("=" * 76)
    print("QVARn-RAG CPU PERFORMANCE BASELINE - STAGE 9A")
    print("=" * 76)
    print(f"OS          : {platform.platform()}")
    print(f"Python      : {platform.python_version()}")
    print(f"CPU logical : {os.cpu_count() or 'unknown'}")
    print(f"Machine     : {platform.machine()}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Measure Qvarn-RAG CPU-only baseline against "
            "the real local index."
        )
    )
    parser.add_argument(
        "--cases",
        default=str(
            Path(__file__).with_name("benchmark_cases.json")
        ),
    )
    parser.add_argument(
        "--chroma-dir",
        default=None,
    )
    parser.add_argument(
        "--warm-runs",
        type=int,
        default=3,
        help=(
            "Number of warm repetitions for embedding and AI. "
            "AI adds one cold run before these repetitions."
        ),
    )
    parser.add_argument(
        "--include-ai",
        action="store_true",
        help="Include real Ollama/qwen benchmark.",
    )
    args = parser.parse_args(argv)

    cases = load_cases(Path(args.cases))
    settings = (
        Settings(chroma_persist_dir=args.chroma_dir)
        if args.chroma_dir
        else Settings()
    )

    print_environment()
    print(
        "Index       : "
        + str(settings.chroma_persist_path)
    )
    print(
        "Embedding   : "
        + settings.embedding_model
    )
    print(
        "LLM default : "
        + settings.ollama_model
    )
    print()

    indexer, indexer_init = timed(
        lambda: Indexer(settings=settings)
    )

    ollama = OllamaClient(
        base_url=settings.ollama_url
    )
    engine = RAGEngine(
        indexer=indexer,
        ollama=ollama,
    )

    print(f"Indexer init         : {indexer_init:.3f}s")

    stats, stats_time = timed(indexer.get_stats)
    print(f"Index stats          : {stats_time:.3f}s")
    print(
        f"Indexed data         : "
        f"{stats.documents} docs / {stats.chunks} chunks"
    )
    print()

    # --------------------------------------------------------------
    # EMBEDDING COLD / WARM
    # --------------------------------------------------------------

    _, cold_embed = timed(
        lambda: indexer.embedder.embed(
            "benchmark cold embedding"
        )
    )
    print(f"Embedding cold       : {cold_embed:.3f}s")

    warm_embed_times: list[float] = []
    for run in range(max(args.warm_runs, 1)):
        _, elapsed = timed(
            lambda run=run: indexer.embedder.embed(
                f"benchmark warm embedding {run}"
            )
        )
        warm_embed_times.append(elapsed)

    print(
        "Embedding warm avg   : "
        f"{statistics.mean(warm_embed_times):.3f}s "
        f"(min={min(warm_embed_times):.3f}s, "
        f"max={max(warm_embed_times):.3f}s)"
    )
    print()

    # --------------------------------------------------------------
    # REAL QUERY CASES
    # --------------------------------------------------------------

    failures = 0

    for case in cases:
        if case.get("kind") == "ai" and not args.include_ai:
            continue

        if case.get("kind") == "ai":
            available, check_time = timed(
                ollama.is_available
            )
            print(
                f"Ollama health        : {check_time:.3f}s "
                f"({'available' if available else 'unavailable'})"
            )
            if not available:
                print(
                    f"[SKIP] {case['id']} - Ollama unavailable"
                )
                print()
                continue

            ai_runs = max(args.warm_runs, 1)

            for run_index in range(ai_runs + 1):
                result = measure_stream(
                    engine,
                    case["question"],
                    int(case.get("top_k", 5)),
                    settings.ollama_model,
                )

                label = (
                    "cold"
                    if run_index == 0
                    else f"warm#{run_index}"
                )

                first_token = (
                    "n/a"
                    if result["first_token_seconds"] is None
                    else f"{result['first_token_seconds']:.3f}s"
                )

                print(
                    f"[PASS] {case['id']} - "
                    f"{case['name']} ({label})"
                )
                print(
                    f"       Stream setup : "
                    f"{result['stream_setup_seconds']:.3f}s"
                )
                print(
                    f"       First token  : {first_token}"
                )
                print(
                    f"       Total        : "
                    f"{result['total_seconds']:.3f}s"
                )
                print(
                    f"       Sources      : "
                    f"{len(result['sources'])}"
                )
                print(
                    f"       Output chars : "
                    f"{len(result['text'])}"
                )
                print()

            continue

        response, elapsed = timed(
            lambda case=case: engine.query(
                case["question"],
                top_k=int(case.get("top_k", 5)),
                model=settings.ollama_model,
            )
        )

        status = "PASS"
        expected_mode = case.get("expected_mode")
        if (
            expected_mode
            and response.model_used != expected_mode
        ):
            failures += 1
            status = "FAIL"

        print(
            f"[{status}] {case['id']} - {case['name']}"
        )
        print(
            f"       Mode    : {response.model_used}"
        )
        print(
            f"       Time    : {elapsed:.3f}s"
        )
        print(
            f"       Sources : {len(response.sources)}"
        )
        if expected_mode:
            print(
                f"       Expected: {expected_mode}"
            )
        print()

    print("-" * 76)
    print(
        "NOTE: cold embedding is measured once per fresh "
        "Python process."
    )
    print(
        "Run this command again from a new process to "
        "compare cold-start behavior."
    )
    print("=" * 76)

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

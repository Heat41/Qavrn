from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from backend.app.config import Settings
from backend.app.indexer import Indexer
from backend.app.llm import OllamaClient
from backend.app.rag import RAGEngine


NOT_FOUND_TEXTS = (
    (
        "Informasi tersebut tidak ditemukan "
        "dalam dokumen yang diberikan."
    ),
    (
        "Informasi tersebut tidak ditemukan "
        "dalam dokumen yang diindeks."
    ),
)


def load_cases(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError("UAT cases must be a JSON array")

    return data


def build_engine(chroma_dir: str | None) -> RAGEngine:
    if chroma_dir:
        settings = Settings(
            chroma_persist_dir=chroma_dir
        )
    else:
        settings = Settings()

    indexer = Indexer(settings=settings)
    ollama = OllamaClient(
        base_url=settings.ollama_url
    )

    return RAGEngine(
        indexer=indexer,
        ollama=ollama,
    )


def evaluate_case(
    case: dict[str, Any],
    answer: str,
    source_filenames: list[str],
    mode: str,
) -> list[str]:
    failures: list[str] = []

    for expected in case.get(
        "expected_answer_contains",
        [],
    ):
        if expected.casefold() not in answer.casefold():
            failures.append(
                f"answer missing expected text: {expected!r}"
            )

    expected_mode = case.get(
        "expected_mode"
    )
    if expected_mode and mode != expected_mode:
        failures.append(
            f"expected mode {expected_mode!r}, got {mode!r}"
        )

    expected_source = case.get(
        "expected_source_filename"
    )
    if expected_source:
        if not any(
            name.casefold()
            == expected_source.casefold()
            for name in source_filenames
        ):
            failures.append(
                "expected source filename not returned: "
                f"{expected_source}"
            )

    expected_source_tokens = case.get(
        "expected_source_filename_contains_any",
        [],
    )
    if expected_source_tokens:
        folded_names = [
            name.casefold()
            for name in source_filenames
        ]
        if not any(
            token.casefold() in name
            for token in expected_source_tokens
            for name in folded_names
        ):
            failures.append(
                "no source filename matched any expected token: "
                + ", ".join(expected_source_tokens)
            )

    require_sources = case.get(
        "require_sources"
    )
    if require_sources is True and not source_filenames:
        failures.append(
            "expected at least one source"
        )

    if case.get("expected_not_found"):
        answer_folded = answer.casefold()
        if not any(
            text.casefold() in answer_folded
            for text in NOT_FOUND_TEXTS
        ):
            failures.append(
                "expected not-found answer"
            )
        if source_filenames:
            failures.append(
                "expected no sources for unavailable year"
            )

    return failures


def run_case(
    engine: RAGEngine,
    case: dict[str, Any],
) -> tuple[
    bool,
    str,
    list[str],
    str,
    list[str],
    float,
]:
    response = engine.query(
        case["question"],
        top_k=int(case.get("top_k", 10)),
    )

    source_filenames = [
        source.filename
        for source in response.sources
    ]

    failures = evaluate_case(
        case,
        response.answer,
        source_filenames,
        response.model_used,
    )

    return (
        not failures,
        response.answer,
        source_filenames,
        response.model_used,
        failures,
        response.query_time_seconds,
    )


def print_header(chroma_dir: str | None) -> None:
    print("=" * 72)
    print("QVARn-RAG REAL DOCUMENT UAT - STAGE 8")
    print("=" * 72)
    print(
        "Index       : "
        + (
            str(Path(chroma_dir).resolve())
            if chroma_dir
            else "default ./data/chroma"
        )
    )
    print("Mode        : real Chroma index / real RAGEngine")
    print("LLM baseline: deterministic/search cases only")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run Qvarn-RAG Stage 8 acceptance cases "
            "against the real local Chroma index."
        )
    )
    parser.add_argument(
        "--cases",
        default=str(
            Path(__file__).with_name(
                "cases.json"
            )
        ),
        help="Path to UAT case definition JSON",
    )
    parser.add_argument(
        "--chroma-dir",
        default=None,
        help=(
            "Optional Chroma directory override. "
            "Defaults to project ./data/chroma."
        ),
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        metavar="UAT_ID",
        help="Run only selected case IDs",
    )
    args = parser.parse_args(argv)

    cases = load_cases(
        Path(args.cases)
    )

    if args.only:
        selected = {
            value.upper()
            for value in args.only
        }
        cases = [
            case
            for case in cases
            if str(
                case.get("id", "")
            ).upper() in selected
        ]

    if not cases:
        print(
            "No UAT cases selected.",
            file=sys.stderr,
        )
        return 2

    print_header(args.chroma_dir)

    try:
        engine = build_engine(
            args.chroma_dir
        )
    except Exception as exc:
        print(
            f"[ENVIRONMENT_FAIL] {exc}",
            file=sys.stderr,
        )
        return 2

    passed = 0
    failed = 0

    for case in cases:
        case_id = case.get("id", "UAT")
        name = case.get(
            "name",
            case.get("question", ""),
        )

        print(
            f"[RUN ] {case_id} - {name}"
        )
        print(
            f"       Question: {case['question']}"
        )

        try:
            (
                ok,
                answer,
                source_filenames,
                mode,
                failures,
                elapsed,
            ) = run_case(
                engine,
                case,
            )
        except Exception as exc:
            failed += 1
            print(
                f"[FAIL] {case_id} - "
                "ENVIRONMENT_OR_RUNTIME_FAIL"
            )
            print(
                f"       Error  : {exc}"
            )
            print()
            continue

        status = "PASS" if ok else "FAIL"

        if ok:
            passed += 1
        else:
            failed += 1

        print(
            f"[{status}] {case_id} - {name}"
        )
        print(
            f"       Mode   : {mode}"
        )
        print(
            f"       Answer : {answer}"
        )
        print(
            "       Sources: "
            + (
                ", ".join(
                    source_filenames
                )
                if source_filenames
                else "(none)"
            )
        )
        print(
            f"       Time   : {elapsed:.3f}s"
        )

        for failure in failures:
            print(
                f"       Reason : {failure}"
            )

        print()

    print("-" * 72)
    print(
        f"PASS : {passed}"
    )
    print(
        f"FAIL : {failed}"
    )
    print(
        f"TOTAL: {passed + failed}"
    )
    print("=" * 72)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

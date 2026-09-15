from __future__ import annotations

import argparse
import json
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
        raise ValueError("Discovery cases must be a JSON array")

    return data


def build_engine(chroma_dir: str | None) -> RAGEngine:
    settings = (
        Settings(chroma_persist_dir=chroma_dir)
        if chroma_dir
        else Settings()
    )

    return RAGEngine(
        indexer=Indexer(settings=settings),
        ollama=OllamaClient(
            base_url=settings.ollama_url
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect real-index answers before locking "
            "new Stage 8 acceptance expectations."
        )
    )
    parser.add_argument(
        "--cases",
        default=str(
            Path(__file__).with_name(
                "discovery_cases.json"
            )
        ),
    )
    parser.add_argument(
        "--chroma-dir",
        default=None,
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        metavar="DISC_ID",
    )
    args = parser.parse_args(argv)

    cases = load_cases(Path(args.cases))

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

    engine = build_engine(args.chroma_dir)

    print("=" * 72)
    print("QVARn-RAG STAGE 8 - REAL INDEX DISCOVERY")
    print("=" * 72)
    print(
        "Purpose: inspect real answers/sources before "
        "creating UAT_07+ expectations."
    )
    print()

    for case in cases:
        print(
            f"[RUN ] {case['id']} - {case['name']}"
        )
        print(
            f"       Question: {case['question']}"
        )

        try:
            response = engine.query(
                case["question"],
                top_k=int(
                    case.get("top_k", 10)
                ),
            )
        except Exception as exc:
            print(
                f"[FAIL] Runtime: {exc}"
            )
            print()
            continue

        print(
            f"       Mode   : {response.model_used}"
        )
        print(
            f"       Answer : {response.answer}"
        )
        print(
            f"       Time   : "
            f"{response.query_time_seconds:.3f}s"
        )

        if response.sources:
            print("       Sources:")
            for index, source in enumerate(
                response.sources,
                start=1,
            ):
                print(
                    f"         {index}. "
                    f"{source.filename} "
                    f"(score={source.score:.4f})"
                )
        else:
            print(
                "       Sources: (none)"
            )

        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from backend.app import api


async def collect_stream(response) -> list[str]:
    chunks = []

    async for item in response.body_iterator:
        if isinstance(item, bytes):
            item = item.decode("utf-8")
        chunks.append(item)

    return chunks


def fake_request():
    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                indexer=SimpleNamespace(),
                ollama=SimpleNamespace(),
            )
        )
    )


class TestAPIModelContract(unittest.TestCase):

    def test_selected_model_is_forwarded_to_query_stream(self) -> None:
        calls = []

        class FakeRAGEngine:

            def __init__(self, indexer, ollama) -> None:
                pass

            def query_stream(
                self,
                question,
                top_k,
                model=None,
            ):
                calls.append(
                    (question, top_k, model)
                )
                return iter(["ok"]), []

        async def run_case():
            with patch.object(
                api,
                "RAGEngine",
                FakeRAGEngine,
            ):
                response = await api.ask(
                    api.AskRequest(
                        question="jelaskan dokumen",
                        top_k=7,
                        model="qwen3:4b",
                    ),
                    fake_request(),
                )
                await collect_stream(response)

        asyncio.run(run_case())

        self.assertEqual(
            calls,
            [
                (
                    "jelaskan dokumen",
                    7,
                    "qwen3:4b",
                )
            ],
        )

    def test_missing_model_preserves_legacy_two_argument_call(self) -> None:
        calls = []

        class FakeRAGEngine:

            def __init__(self, indexer, ollama) -> None:
                pass

            def query_stream(self, *args):
                calls.append(args)
                return iter(["ok"]), []

        async def run_case():
            with patch.object(
                api,
                "RAGEngine",
                FakeRAGEngine,
            ):
                response = await api.ask(
                    api.AskRequest(
                        question="jelaskan dokumen",
                        top_k=5,
                    ),
                    fake_request(),
                )
                await collect_stream(response)

        asyncio.run(run_case())

        self.assertEqual(
            calls,
            [("jelaskan dokumen", 5)],
        )


if __name__ == "__main__":
    unittest.main()

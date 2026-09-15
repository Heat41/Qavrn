from __future__ import annotations

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

from backend.app import api
from backend.app.rag import SourceChunk


def fake_request():
    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                indexer=SimpleNamespace(),
                ollama=SimpleNamespace(),
            )
        )
    )


async def collect_stream(response) -> list[str]:
    chunks = []

    async for item in response.body_iterator:
        if isinstance(item, bytes):
            item = item.decode("utf-8")
        chunks.append(item)

    return chunks


def parse_sse_payload(event: str):
    assert event.startswith("data: ")
    payload = event[len("data: "):].strip()

    if payload == "[DONE]":
        return "[DONE]"

    return json.loads(payload)


class TestAPIAskStream(unittest.TestCase):

    def test_empty_question_returns_http_400(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(
                api.ask(
                    api.AskRequest(
                        question="   ",
                        top_k=5,
                    ),
                    fake_request(),
                )
            )

        self.assertEqual(ctx.exception.status_code, 400)

    def test_sse_contract_token_sources_done(self) -> None:
        source = SourceChunk(
            filename="Lapkeu PT AEP 2024.pdf",
            chunk_text="contoh isi",
            score=1.23456,
            file_path=r"C:\Docs\Lapkeu PT AEP 2024.pdf",
        )

        class FakeRAGEngine:

            def __init__(self, indexer, ollama) -> None:
                pass

            def query_stream(self, question, top_k):
                return iter(["Halo", " dunia"]), [source]

        async def run_case():
            with patch.object(api, "RAGEngine", FakeRAGEngine):
                response = await api.ask(
                    api.AskRequest(
                        question="jelaskan dokumen",
                        top_k=5,
                    ),
                    fake_request(),
                )
                return await collect_stream(response)

        events = asyncio.run(run_case())
        payloads = [parse_sse_payload(event) for event in events]

        self.assertEqual(
            [item["type"] for item in payloads[:-1]],
            ["token", "token", "sources"],
        )
        self.assertEqual(payloads[0]["content"], "Halo")
        self.assertEqual(payloads[1]["content"], " dunia")
        self.assertEqual(
            payloads[2]["sources"][0],
            {
                "filename": "Lapkeu PT AEP 2024.pdf",
                "file_path": r"C:\Docs\Lapkeu PT AEP 2024.pdf",
                "chunk_text": "contoh isi",
                "score": 1.2346,
            },
        )
        self.assertEqual(payloads[-1], "[DONE]")

    def test_connection_error_becomes_error_event_then_done(self) -> None:

        class FakeRAGEngine:

            def __init__(self, indexer, ollama) -> None:
                pass

            def query_stream(self, question, top_k):
                raise ConnectionError("Ollama tidak tersedia")

        async def run_case():
            with patch.object(api, "RAGEngine", FakeRAGEngine):
                response = await api.ask(
                    api.AskRequest(
                        question="jelaskan dokumen",
                        top_k=5,
                    ),
                    fake_request(),
                )
                return await collect_stream(response)

        events = asyncio.run(run_case())
        payloads = [parse_sse_payload(event) for event in events]

        self.assertEqual(payloads[0]["type"], "error")
        self.assertEqual(
            payloads[0]["message"],
            "Ollama tidak tersedia",
        )
        self.assertEqual(payloads[-1], "[DONE]")

    def test_unexpected_error_becomes_error_event_then_done(self) -> None:

        class FakeRAGEngine:

            def __init__(self, indexer, ollama) -> None:
                pass

            def query_stream(self, question, top_k):
                raise RuntimeError("retrieval gagal")

        async def run_case():
            with patch.object(api, "RAGEngine", FakeRAGEngine):
                response = await api.ask(
                    api.AskRequest(
                        question="jelaskan dokumen",
                        top_k=5,
                    ),
                    fake_request(),
                )
                return await collect_stream(response)

        events = asyncio.run(run_case())
        payloads = [parse_sse_payload(event) for event in events]

        self.assertEqual(payloads[0]["type"], "error")
        self.assertEqual(payloads[0]["message"], "retrieval gagal")
        self.assertEqual(payloads[-1], "[DONE]")

    def test_sources_event_is_emitted_even_when_sources_empty(self) -> None:

        class FakeRAGEngine:

            def __init__(self, indexer, ollama) -> None:
                pass

            def query_stream(self, question, top_k):
                return iter(["tidak ditemukan"]), []

        async def run_case():
            with patch.object(api, "RAGEngine", FakeRAGEngine):
                response = await api.ask(
                    api.AskRequest(
                        question="data tidak ada",
                        top_k=5,
                    ),
                    fake_request(),
                )
                return await collect_stream(response)

        events = asyncio.run(run_case())
        payloads = [parse_sse_payload(event) for event in events]

        self.assertEqual(payloads[0]["type"], "token")
        self.assertEqual(payloads[1]["type"], "sources")
        self.assertEqual(payloads[1]["sources"], [])
        self.assertEqual(payloads[-1], "[DONE]")


if __name__ == "__main__":
    unittest.main()

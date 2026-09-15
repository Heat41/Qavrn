from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from backend.app.rag import RAGEngine, SourceChunk


class FakeOllama:

    def __init__(self) -> None:
        self.generate_calls = []
        self.stream_calls = []

    def generate(self, question: str, context: str, model: str):
        self.generate_calls.append((question, context, model))
        return "jawaban-ai"

    def generate_stream(self, question: str, context: str, model: str):
        self.stream_calls.append((question, context, model))
        return iter(["jawaban", "-", "stream"])


def make_engine() -> tuple[RAGEngine, FakeOllama]:
    ollama = FakeOllama()
    engine = RAGEngine(
        indexer=SimpleNamespace(),
        ollama=ollama,
    )
    return engine, ollama


def sample_chunk() -> SourceChunk:
    return SourceChunk(
        filename="Lapkeu PT AEP 2024.pdf",
        chunk_text="PT Arinsa Energy Pratama data tahun 2024",
        score=1.25,
        file_path=r"C:\Docs\Lapkeu PT AEP 2024.pdf",
    )


class TestRAGPublicAPI(unittest.TestCase):

    def test_query_search_mode_does_not_call_ollama(self) -> None:
        engine, ollama = make_engine()
        chunk = sample_chunk()

        with (
            patch.object(
                engine,
                "_retrieve_and_build_context",
                return_value=([chunk], "context"),
            ),
            patch.object(
                engine,
                "_is_search_only_question",
                return_value=True,
            ),
            patch.object(
                engine,
                "_build_search_answer",
                return_value="Dokumen ditemukan.",
            ),
        ):
            response = engine.query(
                "di mana file laporan.pdf",
                top_k=5,
            )

        self.assertEqual(response.answer, "Dokumen ditemukan.")
        self.assertEqual(response.model_used, "search")
        self.assertEqual(response.sources, [chunk])
        self.assertEqual(ollama.generate_calls, [])
        self.assertEqual(ollama.stream_calls, [])

    def test_query_and_stream_use_same_deterministic_answer(self) -> None:
        engine, ollama = make_engine()
        chunk = sample_chunk()
        deterministic = (
            "Direktur perusahaan tahun 2024 adalah RUS'AN.",
            [chunk],
            "table",
        )

        with (
            patch.object(
                engine,
                "_retrieve_and_build_context",
                return_value=([chunk], "context"),
            ),
            patch.object(
                engine,
                "_is_search_only_question",
                return_value=False,
            ),
            patch.object(
                engine,
                "_resolve_deterministic_answer",
                return_value=deterministic,
            ),
        ):
            response = engine.query(
                "siapa direktur tahun 2024"
            )
            token_iter, sources = engine.query_stream(
                "siapa direktur tahun 2024"
            )

        self.assertEqual(
            response.answer,
            "Direktur perusahaan tahun 2024 adalah RUS'AN.",
        )
        self.assertEqual(
            list(token_iter),
            ["Direktur perusahaan tahun 2024 adalah RUS'AN."],
        )
        self.assertEqual(response.sources, [chunk])
        self.assertEqual(sources, [chunk])
        self.assertEqual(ollama.generate_calls, [])
        self.assertEqual(ollama.stream_calls, [])

    def test_query_and_stream_have_same_empty_result_message(self) -> None:
        engine, ollama = make_engine()

        with (
            patch.object(
                engine,
                "_retrieve_and_build_context",
                return_value=([], ""),
            ),
            patch.object(
                engine,
                "_is_search_only_question",
                return_value=False,
            ),
        ):
            response = engine.query("data yang tidak ada")
            token_iter, sources = engine.query_stream(
                "data yang tidak ada"
            )

        expected = (
            "Informasi tersebut tidak ditemukan "
            "dalam dokumen yang diberikan."
        )
        self.assertEqual(response.answer, expected)
        self.assertEqual(list(token_iter), [expected])
        self.assertEqual(response.sources, [])
        self.assertEqual(sources, [])
        self.assertEqual(ollama.generate_calls, [])
        self.assertEqual(ollama.stream_calls, [])

    def test_query_model_override_is_forwarded_to_ollama(self) -> None:
        engine, ollama = make_engine()
        chunk = sample_chunk()

        with (
            patch.object(
                engine,
                "_retrieve_and_build_context",
                return_value=([chunk], "context-ai"),
            ),
            patch.object(
                engine,
                "_is_search_only_question",
                return_value=False,
            ),
            patch.object(
                engine,
                "_resolve_deterministic_answer",
                return_value=None,
            ),
        ):
            response = engine.query(
                "jelaskan dokumen",
                model="qwen-test:latest",
            )

        self.assertEqual(response.answer, "jawaban-ai")
        self.assertEqual(response.model_used, "qwen-test:latest")
        self.assertEqual(
            ollama.generate_calls,
            [
                (
                    "jelaskan dokumen",
                    "context-ai",
                    "qwen-test:latest",
                )
            ],
        )

    def test_stream_model_override_is_forwarded_to_ollama(self) -> None:
        engine, ollama = make_engine()
        chunk = sample_chunk()

        with (
            patch.object(
                engine,
                "_retrieve_and_build_context",
                return_value=([chunk], "context-stream"),
            ),
            patch.object(
                engine,
                "_is_search_only_question",
                return_value=False,
            ),
            patch.object(
                engine,
                "_resolve_deterministic_answer",
                return_value=None,
            ),
        ):
            token_iter, sources = engine.query_stream(
                "jelaskan dokumen",
                model="qwen-stream:test",
            )

        self.assertEqual(
            list(token_iter),
            ["jawaban", "-", "stream"],
        )
        self.assertEqual(sources, [chunk])
        self.assertEqual(
            ollama.stream_calls,
            [
                (
                    "jelaskan dokumen",
                    "context-stream",
                    "qwen-stream:test",
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()

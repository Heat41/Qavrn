from __future__ import annotations

import unittest
from types import SimpleNamespace

from backend.app.rag import RAGEngine
from backend.app.store import SearchResult


def result(
    chunk_id: str,
    document_id: str,
    content: str,
    score: float,
    year: str,
    filename: str,
    file_path: str,
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        document_id=document_id,
        content=content,
        score=score,
        metadata={
            "document_id": document_id,
            "year": year,
            "filename": filename,
            "file_path": file_path,
        },
    )


class FakeEmbedder:

    def __init__(self, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    def embed(self, question: str):
        self.calls += 1
        if self.fail:
            raise AssertionError("embedding should not be called")
        return [0.1, 0.2, 0.3]


class FakeCollection:

    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results

    def get(self, *args, **kwargs):
        return {
            "ids": [item.chunk_id for item in self.results],
            "documents": [item.content for item in self.results],
            "metadatas": [item.metadata for item in self.results],
        }


class FakeStore:

    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.collection = FakeCollection(results)
        self.search_calls = []

    def search(self, query_vec, top_k: int, where=None):
        self.search_calls.append({
            "top_k": top_k,
            "where": where,
        })
        return list(self.results)


def make_engine(
    results: list[SearchResult],
    *,
    fail_embed: bool = False,
) -> tuple[RAGEngine, FakeEmbedder, FakeStore]:
    embedder = FakeEmbedder(fail=fail_embed)
    store = FakeStore(results)
    indexer = SimpleNamespace(
        embedder=embedder,
        store=store,
    )
    engine = RAGEngine(
        indexer=indexer,
        ollama=SimpleNamespace(),
    )
    return engine, embedder, store


class TestRAGRetrievalRegression(unittest.TestCase):

    def test_strict_year_rejects_high_score_wrong_year(self) -> None:
        results = [
            result(
                "wrong",
                "doc-wrong",
                "PPN Rp 999.000.000",
                9.0,
                "2025",
                "SPT Masa PPN 2025.pdf",
                r"C:\PT.AEP\SPT 2025\SPT Masa PPN 2025.pdf",
            ),
            result(
                "right",
                "doc-right",
                "PPN Rp 100.000.000",
                1.0,
                "2024",
                "SPT Masa PPN 2024.pdf",
                r"C:\PT.AEP\SPT 2024\SPT Masa PPN 2024.pdf",
            ),
        ]
        engine, _, store = make_engine(results)

        chunks, _ = engine._retrieve_and_build_context(
            "berapa ppn tahun 2024",
            top_k=3,
        )

        self.assertEqual(
            [chunk.filename for chunk in chunks],
            ["SPT Masa PPN 2024.pdf"],
        )
        self.assertEqual(
            store.search_calls[0]["where"],
            {"year": "2024"},
        )

    def test_exact_filename_precedes_vector_search(self) -> None:
        results = [
            result(
                "exact",
                "doc-exact",
                "isi file exact",
                1.0,
                "2024",
                "Neraca dan RL PT AEP 2024.xls",
                r"C:\PT.AEP\Laporan Keuangan\2024\Neraca dan RL PT AEP 2024.xls",
            ),
            result(
                "similar",
                "doc-similar",
                "isi file mirip",
                9.0,
                "2024",
                "Neraca dan RL PT AEP 2024 revisi.xls",
                r"C:\PT.AEP\Laporan Keuangan\2024\Neraca dan RL PT AEP 2024 revisi.xls",
            ),
        ]
        engine, embedder, store = make_engine(
            results,
            fail_embed=True,
        )

        chunks, _ = engine._retrieve_and_build_context(
            "isi file Neraca dan RL PT AEP 2024.xls",
            top_k=5,
        )

        self.assertEqual(embedder.calls, 0)
        self.assertEqual(store.search_calls, [])
        self.assertEqual(len(chunks), 1)
        self.assertEqual(
            chunks[0].filename,
            "Neraca dan RL PT AEP 2024.xls",
        )

    def test_duplicate_long_chunks_are_removed(self) -> None:
        duplicate_text = (
            "Laporan pemeriksaan dan informasi perpajakan " * 8
        )
        first = result(
            "a",
            "doc-a",
            duplicate_text,
            2.0,
            "2024",
            "A.pdf",
            r"C:\Docs\A.pdf",
        )
        second = result(
            "b",
            "doc-b",
            duplicate_text + " tambahan kecil",
            1.0,
            "2024",
            "B.pdf",
            r"C:\Docs\B.pdf",
        )
        engine, _, _ = make_engine([])

        chunks, context = engine._build_context(
            [first, second],
            top_k=5,
        )

        self.assertEqual(len(chunks), 1)
        self.assertIn("A.pdf", context)
        self.assertNotIn("[Source: B.pdf]", context)

    def test_source_display_deduplicates_same_document_path(self) -> None:
        items = [
            result(
                "a1",
                "doc-a",
                "chunk pertama " * 20,
                2.0,
                "2024",
                "A.pdf",
                r"C:\Docs\A.pdf",
            ),
            result(
                "a2",
                "doc-a",
                "chunk kedua berbeda " * 20,
                1.9,
                "2024",
                "A.pdf",
                r"C:\Docs\A.pdf",
            ),
        ]
        engine, _, _ = make_engine([])

        chunks, context = engine._build_context(items, top_k=5)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(context.count("[Source: A.pdf]"), 2)

    def test_numeric_retrieval_limits_chunks_per_document(self) -> None:
        results = []
        for index in range(5):
            results.append(
                result(
                    f"a{index}",
                    "doc-a",
                    f"PPN data A bagian {index} Rp {index + 1}00.000",
                    10.0 - index,
                    "2024",
                    "SPT A 2024.pdf",
                    r"C:\PT.AEP\SPT 2024\SPT A 2024.pdf",
                )
            )
        results.append(
            result(
                "b1",
                "doc-b",
                "PPN data B Rp 900.000",
                1.0,
                "2024",
                "SPT B 2024.pdf",
                r"C:\PT.AEP\SPT 2024\SPT B 2024.pdf",
            )
        )

        engine, _, _ = make_engine(results)

        chunks, context = engine._retrieve_and_build_context(
            "berapa ppn tahun 2024",
            top_k=3,
        )

        self.assertIn("[Source: SPT B 2024.pdf]", context)
        self.assertLessEqual(
            context.count("[Source: SPT A 2024.pdf]"),
            3,
        )
        self.assertEqual(
            {chunk.filename for chunk in chunks},
            {"SPT A 2024.pdf", "SPT B 2024.pdf"},
        )


if __name__ == "__main__":
    unittest.main()

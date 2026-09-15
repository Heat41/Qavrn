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
    filename: str,
    file_path: str,
    year: str = "2024",
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


class FakeStore:

    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.collection = SimpleNamespace(
            get=lambda *args, **kwargs: {
                "ids": [item.chunk_id for item in self.results],
                "documents": [item.content for item in self.results],
                "metadatas": [item.metadata for item in self.results],
            }
        )

    def search(self, query_vec, top_k: int, where=None):
        return list(self.results)


def make_engine(results: list[SearchResult]) -> RAGEngine:
    indexer = SimpleNamespace(
        embedder=SimpleNamespace(
            embed=lambda question: [0.1, 0.2, 0.3]
        ),
        store=FakeStore(results),
    )
    return RAGEngine(
        indexer=indexer,
        ollama=SimpleNamespace(),
    )


class TestRAGRerankingRegression(unittest.TestCase):

    def test_spt_query_reranks_spt_above_bpe(self) -> None:
        results = [
            result(
                "bpe",
                "doc-bpe",
                "Bukti Penerimaan Elektronik masa pajak PPN",
                5.0,
                "BPE_SPT_2024.pdf",
                r"C:\PT.AEP\SPT 2024\BPE_SPT_2024.pdf",
            ),
            result(
                "spt",
                "doc-spt",
                (
                    "Surat Pemberitahuan Masa Pajak Pertambahan Nilai "
                    "penyerahan barang dan jasa pajak masukan"
                ),
                4.4,
                "SPT Masa PPN 2024.pdf",
                r"C:\PT.AEP\SPT 2024\SPT Masa PPN 2024.pdf",
            ),
        ]
        engine = make_engine(results)

        chunks, _ = engine._retrieve_and_build_context(
            "berapa ppn pada spt tahun 2024",
            top_k=2,
        )

        self.assertEqual(
            chunks[0].filename,
            "SPT Masa PPN 2024.pdf",
        )

    def test_financial_query_reranks_report_above_invoice(self) -> None:
        results = [
            result(
                "invoice",
                "doc-invoice",
                "biaya jasa Rp 100.000.000",
                5.0,
                "Invoice Biaya 2024.pdf",
                r"C:\PT.AEP\Invoice\Invoice Biaya 2024.pdf",
            ),
            result(
                "financial",
                "doc-financial",
                "biaya operasional dan beban usaha Rp 90.000.000",
                4.5,
                "Lapkeu PT AEP 2024.pdf",
                r"C:\PT.AEP\Laporan Keuangan\2024\Lapkeu PT AEP 2024.pdf",
            ),
        ]
        engine = make_engine(results)

        chunks, _ = engine._retrieve_and_build_context(
            "berapa biaya tahun 2024",
            top_k=2,
        )

        self.assertEqual(
            chunks[0].filename,
            "Lapkeu PT AEP 2024.pdf",
        )

    def test_non_spt_query_does_not_apply_spt_bonus(self) -> None:
        results = [
            result(
                "financial",
                "doc-financial",
                "biaya operasional",
                2.0,
                "Lapkeu PT AEP 2024.pdf",
                r"C:\PT.AEP\Laporan Keuangan\2024\Lapkeu PT AEP 2024.pdf",
            ),
            result(
                "spt",
                "doc-spt",
                "masa pajak penyerahan barang dan jasa",
                1.9,
                "SPT Masa PPN 2024.pdf",
                r"C:\PT.AEP\SPT 2024\SPT Masa PPN 2024.pdf",
            ),
        ]
        engine = make_engine(results)

        chunks, _ = engine._retrieve_and_build_context(
            "berapa biaya tahun 2024",
            top_k=2,
        )

        self.assertEqual(
            chunks[0].filename,
            "Lapkeu PT AEP 2024.pdf",
        )

    def test_wrong_year_is_rejected_even_if_filename_mentions_requested_year(self) -> None:
        results = [
            result(
                "wrong",
                "doc-wrong",
                "biaya operasional Rp 500.000",
                9.0,
                "Lapkeu PT AEP 2024.pdf",
                r"C:\PT.AEP\Laporan Keuangan\2024\Lapkeu PT AEP 2024.pdf",
                year="2025",
            ),
            result(
                "right",
                "doc-right",
                "biaya operasional Rp 100.000",
                1.0,
                "Lapkeu PT AEP Final.pdf",
                r"C:\PT.AEP\Laporan Keuangan\2024\Lapkeu PT AEP Final.pdf",
                year="2024",
            ),
        ]
        engine = make_engine(results)

        chunks, _ = engine._retrieve_and_build_context(
            "berapa biaya tahun 2024",
            top_k=2,
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(
            chunks[0].filename,
            "Lapkeu PT AEP Final.pdf",
        )

    def test_bpe_does_not_outrank_spt_on_spt_content_question(self) -> None:
        results = [
            result(
                "bpe",
                "doc-bpe",
                (
                    "Bukti Penerimaan Elektronik SPT Masa PPN "
                    "masa pajak penyerahan barang dan jasa"
                ),
                5.0,
                "BPE_SPT_Masa_PPN_2024.pdf",
                r"C:\PT.AEP\SPT 2024\BPE_SPT_Masa_PPN_2024.pdf",
            ),
            result(
                "spt",
                "doc-spt",
                (
                    "SPT Masa PPN tahun pajak masa pajak "
                    "penyerahan barang dan jasa DPP PPN pajak masukan"
                ),
                4.0,
                "SPT Masa PPN 2024.pdf",
                r"C:\PT.AEP\SPT 2024\SPT Masa PPN 2024.pdf",
            ),
        ]
        engine = make_engine(results)

        chunks, _ = engine._retrieve_and_build_context(
            "berapa dpp pada spt masa ppn tahun 2024",
            top_k=2,
        )

        self.assertEqual(
            chunks[0].filename,
            "SPT Masa PPN 2024.pdf",
        )


if __name__ == "__main__":
    unittest.main()

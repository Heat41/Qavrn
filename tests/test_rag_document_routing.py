from __future__ import annotations

import unittest
from types import SimpleNamespace

from backend.app.rag import RAGEngine


class FakeCollection:

    def __init__(self, ids, documents, metadatas) -> None:
        self._payload = {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
        }

    def get(self, *args, **kwargs):
        where = kwargs.get("where") or {}
        if not where:
            return self._payload

        year = str(where.get("year", ""))
        ids = []
        documents = []
        metadatas = []

        for chunk_id, document, metadata in zip(
            self._payload["ids"],
            self._payload["documents"],
            self._payload["metadatas"],
        ):
            if str((metadata or {}).get("year", "")) != year:
                continue
            ids.append(chunk_id)
            documents.append(document)
            metadatas.append(metadata)

        return {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
        }


def make_engine(collection: FakeCollection) -> RAGEngine:
    indexer = SimpleNamespace(
        store=SimpleNamespace(collection=collection)
    )
    return RAGEngine(
        indexer=indexer,
        ollama=SimpleNamespace(),
    )


class TestRAGDocumentRouting(unittest.TestCase):

    def test_detect_document_type(self) -> None:
        self.assertEqual(
            RAGEngine._detect_document_type(
                "di mana laporan keuangan tahun 2024"
            ),
            "financial",
        )
        self.assertEqual(
            RAGEngine._detect_document_type(
                "cari spt tahun 2024"
            ),
            "spt",
        )
        self.assertEqual(
            RAGEngine._detect_document_type(
                "cari faktur pajak tahun 2024"
            ),
            "faktur",
        )
        self.assertEqual(
            RAGEngine._detect_document_type(
                "cari invoice tahun 2024"
            ),
            "invoice",
        )

    def test_spt_detection_and_bpe_detection_are_separate(self) -> None:
        self.assertTrue(
            RAGEngine._is_spt_document(
                "spt masa ppn 2024.pdf",
                r"c:\pt.aep\spt 2024\spt masa ppn 2024.pdf",
                "masa pajak penyerahan barang dan jasa",
            )
        )
        self.assertTrue(
            RAGEngine._is_bpe_document(
                "bpe_2024.pdf",
                r"c:\pt.aep\bpe_2024.pdf",
                "bukti penerimaan elektronik",
            )
        )

    def test_financial_location_filters_transactions_and_year(self) -> None:
        collection = FakeCollection(
            ids=["fin", "txn", "other-year"],
            documents=[
                "LAPORAN KEUANGAN pendapatan proyek",
                "faktur pajak pendapatan proyek",
                "LAPORAN KEUANGAN tahun lain",
            ],
            metadatas=[
                {
                    "year": "2024",
                    "filename": "Neraca dan RL PT AEP 2024.xls",
                    "file_path": (
                        r"C:\PT.AEP\Laporan Keuangan\2024\"
                        r"Neraca dan RL PT AEP 2024.xls"
                    ),
                },
                {
                    "year": "2024",
                    "filename": "Faktur DP 1.pdf",
                    "file_path": r"C:\PT.AEP\RK\Faktur DP 1.pdf",
                },
                {
                    "year": "2025",
                    "filename": "Neraca dan RL PT AEP 2025.xls",
                    "file_path": (
                        r"C:\PT.AEP\Laporan Keuangan\2025\"
                        r"Neraca dan RL PT AEP 2025.xls"
                    ),
                },
            ],
        )
        engine = make_engine(collection)

        results = engine._find_location_documents(
            "2024",
            "financial",
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(
            results[0].metadata["filename"],
            "Neraca dan RL PT AEP 2024.xls",
        )

    def test_spt_location_excludes_bpe(self) -> None:
        collection = FakeCollection(
            ids=["spt", "bpe"],
            documents=[
                "masa pajak penyerahan barang dan jasa",
                "bukti penerimaan elektronik masa pajak",
            ],
            metadatas=[
                {
                    "year": "2024",
                    "filename": "SPT Masa PPN 2024.pdf",
                    "file_path": r"C:\PT.AEP\SPT 2024\SPT Masa PPN 2024.pdf",
                },
                {
                    "year": "2024",
                    "filename": "BPE_SPT_2024.pdf",
                    "file_path": r"C:\PT.AEP\SPT 2024\BPE_SPT_2024.pdf",
                },
            ],
        )
        engine = make_engine(collection)

        results = engine._find_location_documents(
            "2024",
            "spt",
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(
            results[0].metadata["filename"],
            "SPT Masa PPN 2024.pdf",
        )

    def test_faktur_location_only_returns_faktur(self) -> None:
        collection = FakeCollection(
            ids=["faktur", "invoice"],
            documents=[
                "FAKTUR PAJAK transaksi",
                "invoice transaksi",
            ],
            metadatas=[
                {
                    "year": "2024",
                    "filename": "Faktur DP 1.pdf",
                    "file_path": r"C:\PT.AEP\Faktur DP 1.pdf",
                },
                {
                    "year": "2024",
                    "filename": "Invoice 001.pdf",
                    "file_path": r"C:\PT.AEP\Invoice 001.pdf",
                },
            ],
        )
        engine = make_engine(collection)

        results = engine._find_location_documents(
            "2024",
            "faktur",
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(
            results[0].metadata["filename"],
            "Faktur DP 1.pdf",
        )


if __name__ == "__main__":
    unittest.main()

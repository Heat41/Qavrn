from __future__ import annotations

import unittest
from types import SimpleNamespace

from backend.app.rag import RAGEngine, SourceChunk


class FakeCollection:

    def __init__(
        self,
        ids=None,
        documents=None,
        metadatas=None,
    ) -> None:
        self._payload = {
            "ids": ids or [],
            "documents": documents or [],
            "metadatas": metadatas or [],
        }

    def get(self, *args, **kwargs):
        return self._payload


def make_engine(collection: FakeCollection) -> RAGEngine:
    indexer = SimpleNamespace(
        store=SimpleNamespace(
            collection=collection,
        )
    )

    return RAGEngine(
        indexer=indexer,
        ollama=SimpleNamespace(),
    )


class TestRAGRegression(unittest.TestCase):

    def test_director_2024_regression(self) -> None:
        chunks = [
            SourceChunk(
                filename="Neraca dan RL PT AEP 2024.xls",
                chunk_text=(
                    "PT. ARINSA ENERGY PRATAMA "
                    "R U S ' A N Direktur"
                ),
                score=1.0,
            )
        ]

        self.assertEqual(
            RAGEngine._extract_director_name(chunks),
            "RUS'AN",
        )

    def test_turnover_2024_prefers_total_regression(self) -> None:
        chunks = [
            SourceChunk(
                filename="Pendapatan PT.ArinsaEnergy Tahun 2024.xlsx",
                chunk_text=(
                    "PEREDARAN USAHA | Kolom 5: 435586850 "
                    "Baris 20: OKTOBER"
                ),
                score=1.0,
            ),
            SourceChunk(
                filename="Pendapatan PT.ArinsaEnergy Tahun 2024.xlsx",
                chunk_text=(
                    "Total Peredaran Usaha Tahun 2024 | "
                    "Kolom 5: 1248761500"
                ),
                score=1.0,
            ),
        ]

        raw = RAGEngine._extract_annual_financial_value(
            chunks,
            "berapa peredaran usaha tahun 2024",
        )

        self.assertEqual(raw, "1248761500")
        self.assertEqual(
            RAGEngine._format_financial_value(raw),
            "1.248.761.500",
        )

    def test_company_address_2024_regression(self) -> None:
        collection = FakeCollection(
            ids=["addr-1", "addr-2"],
            documents=[
                (
                    "PT. Arinsa Energy Pratama "
                    "Alama : Ji. Danau Sentarum, gang. Citarum "
                    "No. 3 Pontianak, Kalimantan Barat r "
                    "E-mail : contoh@example.com"
                ),
                (
                    "PT. Arinsa Energy Pratama "
                    "Alamat : Jl Danau Sentarum Gg.Citarum "
                    "No.3 Pontianak"
                ),
            ],
            metadatas=[
                {
                    "year": "2024",
                    "filename": (
                        "Lapkeu PT Arinsa Energy Pratama 2024.pdf"
                    ),
                    "file_path": (
                        r"C:\\PT.AEP\\Laporan Keuangan\\2024\\"
                        r"Lapkeu PT Arinsa Energy Pratama 2024.pdf"
                    ),
                },
                {
                    "year": "2024",
                    "filename": "Neraca dan RL PT AEP 2024.xls",
                    "file_path": (
                        r"C:\\PT.AEP\\Laporan Keuangan\\2024\\"
                        r"Neraca dan RL PT AEP 2024.xls"
                    ),
                },
            ],
        )
        engine = make_engine(collection)

        answer = engine._resolve_deterministic_answer(
            "alamat PT Arinsa Energy Pratama tahun 2024",
            [
                SourceChunk(
                    filename="placeholder",
                    chunk_text="placeholder",
                    score=1.0,
                )
            ],
        )

        self.assertIsNotNone(answer)
        text, sources, mode = answer

        self.assertEqual(
            text,
            (
                "Alamat perusahaan tahun 2024 adalah "
                "Jl. Danau Sentarum, Gg. Citarum No. 3 "
                "Pontianak, Kalimantan Barat."
            ),
        )
        self.assertEqual(len(sources), 1)
        self.assertEqual(mode, "table")

    def test_exact_file_location_regression(self) -> None:
        target_path = (
            r"C:\\Users\\ASUS260922\\Documents\\PT.AEP\\"
            r"Laporan Keuangan\\2024\\Neraca dan RL PT AEP 2024.xls"
        )
        collection = FakeCollection(
            ids=["target", "other"],
            documents=[
                "LAPORAN KEUANGAN PT. ARINSA ENERGY PRATAMA",
                "dokumen lain",
            ],
            metadatas=[
                {
                    "year": "2024",
                    "filename": "Neraca dan RL PT AEP 2024.xls",
                    "file_path": target_path,
                    "document_id": "doc-target",
                    "chunk_index": 0,
                },
                {
                    "year": "2024",
                    "filename": "728530833701000202410.pdf",
                    "file_path": r"C:\\PT.AEP\\SPT 2024\\other.pdf",
                    "document_id": "doc-other",
                    "chunk_index": 0,
                },
            ],
        )
        engine = make_engine(collection)

        requested_filename = engine._extract_filename(
            "di mana file Neraca dan RL PT AEP 2024.xls"
        )
        results = engine._find_file_directly(
            requested_filename or ""
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(
            results[0].metadata["filename"],
            "Neraca dan RL PT AEP 2024.xls",
        )

        chunks, _ = engine._build_context(
            results,
            top_k=10,
        )
        answer = engine._build_search_answer(
            "di mana file Neraca dan RL PT AEP 2024.xls",
            chunks,
        )

        self.assertIn(
            r"C:\\Users\\ASUS260922\\Documents\\PT.AEP\\"
            r"Laporan Keuangan\\2024",
            answer,
        )
        self.assertNotIn("SPT 2024", answer)


if __name__ == "__main__":
    unittest.main()

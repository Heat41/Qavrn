from __future__ import annotations

import unittest

from backend.app.rag import RAGEngine
from backend.app.store import SearchResult


class TestRAGHelpers(unittest.TestCase):

    def test_extract_year(self) -> None:
        self.assertEqual(
            RAGEngine._extract_year("siapa direktur tahun 2024"),
            "2024",
        )
        self.assertIsNone(
            RAGEngine._extract_year("siapa direktur perusahaan")
        )

    def test_extract_filename_with_spaces(self) -> None:
        self.assertEqual(
            RAGEngine._extract_filename(
                "di mana file Neraca dan RL PT AEP 2024.xls"
            ),
            "Neraca dan RL PT AEP 2024.xls",
        )

    def test_search_only_location_and_address_split(self) -> None:
        self.assertTrue(
            RAGEngine._is_search_only_question(
                "di mana file Neraca dan RL PT AEP 2024.xls"
            )
        )
        self.assertFalse(
            RAGEngine._is_search_only_question(
                "alamat PT Arinsa Energy Pratama tahun 2024"
            )
        )

    def test_location_question_detection(self) -> None:
        self.assertTrue(
            RAGEngine._is_location_question(
                "di mana file laporan keuangan"
            )
        )
        self.assertFalse(
            RAGEngine._is_location_question(
                "berapa peredaran usaha tahun 2024"
            )
        )

    def test_document_search_intent_detection(self) -> None:
        for question in [
            "cari spt tahun 2024",
            "carikan faktur pajak tahun 2024",
            "temukan invoice tahun 2024",
            "tampilkan laporan keuangan tahun 2024",
        ]:
            self.assertTrue(
                RAGEngine._is_document_search_question(
                    question
                )
            )
            self.assertTrue(
                RAGEngine._is_search_only_question(
                    question
                )
            )

    def test_document_content_question_is_not_search_only(self) -> None:
        for question in [
            "berapa isi spt tahun 2024",
            "jelaskan faktur pajak tahun 2024",
            "apa isi invoice tahun 2024",
        ]:
            self.assertFalse(
                RAGEngine._is_document_search_question(
                    question
                )
            )
            self.assertFalse(
                RAGEngine._is_search_only_question(
                    question
                )
            )

    def test_result_matches_year_is_strict_when_metadata_present(self) -> None:
        result_2024 = SearchResult(
            chunk_id="1",
            document_id="doc-1",
            content="contoh",
            score=1.0,
            metadata={
                "year": "2024",
                "filename": "contoh 2025.pdf",
                "file_path": r"C:\\contoh\\2025\\contoh.pdf",
            },
        )
        result_2025 = SearchResult(
            chunk_id="2",
            document_id="doc-2",
            content="contoh",
            score=1.0,
            metadata={
                "year": "2025",
                "filename": "contoh 2024.pdf",
                "file_path": r"C:\\contoh\\2024\\contoh.pdf",
            },
        )

        self.assertTrue(
            RAGEngine._result_matches_year(
                result_2024,
                "2024",
            )
        )
        self.assertFalse(
            RAGEngine._result_matches_year(
                result_2025,
                "2024",
            )
        )

    def test_address_normalization(self) -> None:
        self.assertEqual(
            RAGEngine._clean_company_address(
                "Ji. Danau Sentarum, gang. Citarum No. 3 "
                "Pontianak, Kalimantan Barat r"
            ),
            "Jl. Danau Sentarum, Gg. Citarum No. 3 "
            "Pontianak, Kalimantan Barat",
        )

    def test_address_equivalence_for_ocr_variants(self) -> None:
        first = RAGEngine._company_address_key(
            RAGEngine._clean_company_address(
                "Ji. Danau Sentarum, gang. Citarum No. 3 "
                "Pontianak, Kalimantan Barat"
            )
        )
        second = RAGEngine._company_address_key(
            RAGEngine._clean_company_address(
                "Jl Danau Sentarum Gg.Citarum No.3 Pontianak"
            )
        )

        self.assertTrue(
            RAGEngine._company_addresses_equivalent(
                first,
                second,
            )
        )


if __name__ == "__main__":
    unittest.main()

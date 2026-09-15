from __future__ import annotations

import unittest

from backend.app.rag import RAGEngine, SourceChunk


class TestRAGFinancialRegressionStage2(unittest.TestCase):

    def test_comparison_question_detection(self) -> None:
        self.assertTrue(
            RAGEngine._is_peredaran_usaha_comparison_question(
                "berapa peredaran usaha menurut WP dan menurut Pemeriksa tahun 2024"
                .lower()
            )
        )
        self.assertFalse(
            RAGEngine._is_peredaran_usaha_comparison_question(
                "berapa peredaran usaha tahun 2024"
            )
        )

    def test_wp_pemeriksa_values_keep_order(self) -> None:
        chunks = [
            SourceChunk(
                filename="pemeriksaan.pdf",
                chunk_text=(
                    "Peredaran Usaha "
                    "Menurut WP/SPT Rp 1.248.761.500 "
                    "Menurut Pemeriksa Rp 1.500.000.000"
                ),
                score=1.0,
            )
        ]

        values = RAGEngine._extract_wp_pemeriksa_values(
            chunks,
            "Peredaran Usaha",
        )

        self.assertEqual(
            values,
            ("1.248.761.500", "1.500.000.000"),
        )

    def test_pendapatan_proyek_extraction(self) -> None:
        chunks = [
            SourceChunk(
                filename="lapkeu.xlsx",
                chunk_text=(
                    "Pendapatan Proyek | Kolom 3: 1248761500.0"
                ),
                score=1.0,
            )
        ]

        value = RAGEngine._extract_annual_financial_value(
            chunks,
            "berapa pendapatan tahun 2024",
        )

        self.assertEqual(value, "1248761500.0")

    def test_pendapatan_bersih_extraction(self) -> None:
        chunks = [
            SourceChunk(
                filename="lapkeu.xlsx",
                chunk_text=(
                    "Pendapatan Bersih | Kolom 20: 51093215.45"
                ),
                score=1.0,
            )
        ]

        value = RAGEngine._extract_annual_financial_value(
            chunks,
            "berapa pendapatan bersih tahun 2024",
        )

        self.assertEqual(value, "51093215.45")

    def test_laba_bersih_prefers_laba_label(self) -> None:
        chunks = [
            SourceChunk(
                filename="lapkeu.xlsx",
                chunk_text=(
                    "Pendapatan Bersih | Kolom 20: 1000000 "
                    "Laba/Rugi Tahun Berjalan | Kolom 4: 51093215.45"
                ),
                score=1.0,
            )
        ]

        value = RAGEngine._extract_annual_financial_value(
            chunks,
            "berapa laba bersih tahun 2024",
        )

        self.assertEqual(value, "51093215.45")

    def test_financial_extractor_does_not_return_year(self) -> None:
        chunks = [
            SourceChunk(
                filename="lapkeu.xlsx",
                chunk_text=(
                    "Pendapatan Proyek Tahun 2024 | Kolom 3: 2500000"
                ),
                score=1.0,
            )
        ]

        value = RAGEngine._extract_annual_financial_value(
            chunks,
            "berapa pendapatan tahun 2024",
        )

        self.assertNotEqual(value, "2024")
        self.assertEqual(value, "2500000")


if __name__ == "__main__":
    unittest.main()

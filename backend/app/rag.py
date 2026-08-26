from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List

from .config import settings
from .indexer import Indexer
from .llm import OllamaClient
from .store import SearchResult


# ======================================================================
# DATA CLASS
# ======================================================================

@dataclass
class SourceChunk:
    filename: str
    chunk_text: str
    score: float
    file_path: str = ""


@dataclass
class RAGResponse:
    answer: str
    sources: List[SourceChunk]
    model_used: str
    query_time_seconds: float


# ======================================================================
# RAG ENGINE
# ======================================================================

class RAGEngine:
    """
    Engine RAG Qvarn.

    Fitur:
    - filter tahun
    - deteksi jenis dokumen
    - pencarian file spesifik
    - pencarian lokasi
    - mode search tanpa Ollama
    - prioritas dokumen SPT
    - pengecualian BPE dari SPT induk
    - reranking berdasarkan isi pertanyaan
    - dukungan pertanyaan tabel / angka
    - pembatasan chunk per dokumen
    - fallback retrieval
    - proteksi duplicate chunk
    - ekstraksi tabel deterministik
    - ekstraksi Peredaran Usaha WP vs Pemeriksa
    """

    def __init__(
            self,
            indexer: Indexer,
            ollama: OllamaClient,
    ) -> None:

        self.indexer = indexer
        self.ollama = ollama

    # ==================================================================
    # PUBLIC API
    # ==================================================================

    def query(
            self,
            question: str,
            top_k: int = 3,
            model: str | None = None,
    ) -> RAGResponse:

        t0 = time.perf_counter()

        chunks, context = (
            self._retrieve_and_build_context(
                question,
                top_k,
            )
        )

        # ==============================================================
        # MODE SEARCH
        # ==============================================================

        if self._is_search_only_question(question):

            answer = self._build_search_answer(
                question,
                chunks,
            )

            return RAGResponse(
                answer=answer,
                sources=chunks,
                model_used="search",
                query_time_seconds=(
                        time.perf_counter() - t0
                ),
            )

        # ==============================================================
        # TIDAK ADA HASIL
        # ==============================================================

        if not chunks:

            return RAGResponse(
                answer=(
                    "Informasi tersebut tidak ditemukan "
                    "dalam dokumen yang diberikan."
                ),
                sources=[],
                model_used="search",
                query_time_seconds=(
                        time.perf_counter() - t0
                ),
            )

        question_lower = (
            question.lower().strip()
        )

        # ==============================================================
        # DETERMINISTIC TABLE
        #
        # Untuk pertanyaan tertentu yang strukturnya jelas,
        # jangan serahkan penentuan angka kepada LLM.
        # ==============================================================

        if self._is_peredaran_usaha_comparison_question(
                question_lower
        ):

            table_answer = (
                self._extract_wp_pemeriksa_values(
                    chunks,
                    "Peredaran Usaha",
                )
            )

            if table_answer:

                wp_value, pemeriksa_value = (
                    table_answer
                )

                answer = (
                    "Peredaran Usaha menurut "
                    f"Wajib Pajak: Rp {wp_value}\n"
                    "Peredaran Usaha menurut "
                    f"Pemeriksa: Rp {pemeriksa_value}"
                )

                return RAGResponse(
                    answer=answer,
                    sources=chunks,
                    model_used="table",
                    query_time_seconds=(
                            time.perf_counter() - t0
                    ),
                )

        # ==============================================================
        # MODE AI
        # ==============================================================

        model = model or settings.ollama_model

        answer = self.ollama.generate(
            question,
            context=context,
            model=model,
        )

        return RAGResponse(
            answer=answer,
            sources=chunks,
            model_used=model,
            query_time_seconds=(
                    time.perf_counter() - t0
            ),
        )

    # ==================================================================
    # STREAM
    # ==================================================================

    def query_stream(
            self,
            question: str,
            top_k: int = 3,
            model: str | None = None,
    ) -> tuple[Iterator[str], List[SourceChunk]]:

        chunks, context = (
            self._retrieve_and_build_context(
                question,
                top_k,
            )
        )

        # ==============================================================
        # MODE SEARCH
        # ==============================================================

        if self._is_search_only_question(question):

            answer = self._build_search_answer(
                question,
                chunks,
            )

            def search_stream() -> Iterator[str]:
                yield answer

            return search_stream(), chunks

        # ==============================================================
        # TIDAK ADA HASIL
        # ==============================================================

        if not chunks:

            def empty_stream() -> Iterator[str]:
                yield (
                    "Informasi tersebut tidak ditemukan "
                    "dalam dokumen yang diberikan."
                )

            return empty_stream(), chunks

        question_lower = (
            question.lower().strip()
        )

        # ==============================================================
        # DETERMINISTIC TABLE
        # ==============================================================

        if self._is_peredaran_usaha_comparison_question(
                question_lower
        ):

            table_answer = (
                self._extract_wp_pemeriksa_values(
                    chunks,
                    "Peredaran Usaha",
                )
            )

            if table_answer:

                wp_value, pemeriksa_value = (
                    table_answer
                )

                answer = (
                    "Peredaran Usaha menurut "
                    f"Wajib Pajak: Rp {wp_value}\n"
                    "Peredaran Usaha menurut "
                    f"Pemeriksa: Rp {pemeriksa_value}"
                )

                def table_stream() -> Iterator[str]:
                    yield answer

                return table_stream(), chunks

        # ==============================================================
        # MODE AI
        # ==============================================================

        model = model or settings.ollama_model

        token_iter = self.ollama.generate_stream(
            question,
            context=context,
            model=model,
        )

        return token_iter, chunks

    # ==================================================================
    # RETRIEVAL UTAMA
    # ==================================================================

    def _retrieve_and_build_context(
            self,
            question: str,
            top_k: int,
    ) -> tuple[List[SourceChunk], str]:

        question_lower = (
            question.lower().strip()
        )

        # ==============================================================
        # 0. VALIDASI
        # ==============================================================

        try:

            top_k = max(
                int(top_k or 3),
                1,
            )

        except (
                ValueError,
                TypeError,
        ):

            top_k = 3

        # ==============================================================
        # 1. DETEKSI TAHUN
        # ==============================================================

        requested_year = self._extract_year(
            question
        )

        # ==============================================================
        # 2. DETEKSI FILE SPESIFIK
        # ==============================================================

        requested_filename = (
            self._extract_filename(
                question
            )
        )

        # ==============================================================
        # 3. DETEKSI JENIS DOKUMEN
        # ==============================================================

        document_type = (
            self._detect_document_type(
                question_lower
            )
        )

        spt_question = (
                document_type == "spt"
        )

        location_question = (
            self._is_location_question(
                question_lower
            )
        )

        numeric_or_table_question = (
            self._is_numeric_or_table_question(
                question_lower
            )
        )

        # ==============================================================
        # VARIABEL HASIL
        #
        # Penting:
        # versi sebelumnya bisa menggunakan variabel ini sebelum
        # variabel tersebut dibuat ketika tidak ada filename.
        # ==============================================================

        direct_results: List[SearchResult] = []

        ranked_file_results: List[SearchResult] = []

        # ==============================================================
        # 4. FILE SPESIFIK
        # ==============================================================

        if requested_filename:

            direct_results = (
                self._find_file_directly(
                    requested_filename
                )
            )

            if direct_results:

                # ------------------------------------------------------
                # Lokasi file
                # ------------------------------------------------------

                if location_question:

                    return self._build_location_context(
                        direct_results,
                        top_k=max(
                            top_k,
                            3,
                        ),
                    )

                # ------------------------------------------------------
                # Isi file tertentu
                # ------------------------------------------------------

                ranked_file_results = (
                    self._rerank_file_chunks(
                        direct_results,
                        question_lower,
                    )
                )

                # ------------------------------------------------------
                # Pertanyaan tabel / angka
                # ------------------------------------------------------

                if numeric_or_table_question:

                    expanded_results = (
                        self._expand_table_chunks(
                            direct_results,
                            ranked_file_results,
                            question_lower,
                            radius=2,
                        )
                    )

                    if expanded_results:

                        return self._build_context(
                            expanded_results,
                            top_k=len(
                                expanded_results
                            ),
                        )

                # ------------------------------------------------------
                # Jika file spesifik ditemukan tetapi
                # bukan pertanyaan tabel / angka.
                # ------------------------------------------------------

                return self._build_context(
                    ranked_file_results,
                    top_k=top_k,
                )

        # ==============================================================
        # 5. EMBEDDING QUERY
        # ==============================================================

        try:

            query_vec = (
                self.indexer.embedder.embed(
                    question
                )
            )

        except Exception:

            return [], ""

        # ==============================================================
        # 6. VECTOR SEARCH
        # ==============================================================

        candidate_k = max(
            top_k * 30,
            150,
            )

        search_where = None

        if requested_year:

            search_where = {
                "year": requested_year
            }

        try:

            raw_results: List[SearchResult] = (
                self.indexer.store.search(
                    query_vec,
                    top_k=candidate_k,
                    where=search_where,
                )
            )

        except Exception:

            return [], ""

        if not raw_results:

            return [], ""

        # ==============================================================
        # 7. FILTER TAHUN
        # ==============================================================

        if requested_year:

            year_results: List[SearchResult] = []

            for result in raw_results:

                if self._result_matches_year(
                        result,
                        requested_year,
                ):

                    year_results.append(
                        result
                    )

            if not year_results:

                return [], ""

            raw_results = year_results

        # ==============================================================
        # 8. DETEKSI PERTANYAAN TABEL
        # ==============================================================

        is_table_question = any(
            phrase in question_lower
            for phrase in [
                "menurut wp",
                "menurut wajib pajak",
                "menurut wp/spt",
                "menurut pemeriksa",
                "perbedaan nilai",
                "selisih",
                "tabel",
                "kolom",
                "baris",
                "peredaran usaha",
                "pph badan",
                "ppn",
                "dpp",
                "kurang bayar",
                "lebih bayar",
            ]
        )

        # ==============================================================
        # 9. DETEKSI PERTANYAAN ANGKA
        # ==============================================================

        is_numeric_question = any(
            phrase in question_lower
            for phrase in [
                "berapa",
                "nilai",
                "jumlah",
                "total",
                "selisih",
                "dilaporkan",
                "pembayaran",
                "terutang",
            ]
        )

        # ==============================================================
        # 10. RERANKING
        # ==============================================================

        reranked: list[
            tuple[float, SearchResult]
        ] = []

        question_terms = (
            self._extract_question_terms(
                question_lower
            )
        )

        for result in raw_results:

            score = float(
                result.score
            )

            metadata = (
                    result.metadata or {}
            )

            content = (
                    result.content or ""
            )

            content_lower = (
                content.lower()
            )

            filename = self._get_filename(
                metadata
            )

            filename_lower = (
                filename.lower()
            )

            file_path = str(
                metadata.get(
                    "file_path",
                    "",
                )
            )

            file_path_lower = (
                file_path.lower()
            )

            # ==========================================================
            # TAHUN
            # ==========================================================

            if requested_year:

                metadata_year = str(
                    metadata.get(
                        "year",
                        "",
                    )
                ).strip()

                if metadata_year == requested_year:

                    score += 0.50

                if requested_year in filename_lower:

                    score += 0.25

                if requested_year in file_path_lower:

                    score += 0.20

            # ==========================================================
            # SPT
            # ==========================================================

            if spt_question:

                is_bpe = (
                    self._is_bpe_document(
                        filename_lower,
                        file_path_lower,
                        content_lower,
                    )
                )

                is_spt_document = (
                    self._is_spt_document(
                        filename_lower,
                        file_path_lower,
                        content_lower,
                    )
                )

                is_non_spt = (
                    self._is_non_spt_document(
                        filename_lower,
                        file_path_lower,
                        content_lower,
                    )
                )

                # ------------------------------------------------------
                # Dokumen SPT
                # ------------------------------------------------------

                if is_spt_document:

                    score += 1.00

                # ------------------------------------------------------
                # BPE dikurangi
                # ------------------------------------------------------

                if is_bpe:

                    score -= 1.00

                # ------------------------------------------------------
                # Non-SPT dikurangi
                # ------------------------------------------------------

                if is_non_spt:

                    score -= 0.75

                spt_terms = [
                    "surat pemberitahuan",
                    "spt masa",
                    "spt tahunan",
                    "masa pajak",
                    "tahun pajak",
                    "penyerahan barang dan jasa",
                    "dpp",
                    "ppn",
                    "pajak masukan",
                    "pajak keluaran",
                    "kurang atau lebih bayar",
                ]

                matched_terms = sum(
                    1
                    for term in spt_terms
                    if term in content_lower
                )

                score += min(
                    matched_terms * 0.08,
                    0.50,
                    )

                # ------------------------------------------------------
                # Jika pertanyaan meminta isi SPT
                # ------------------------------------------------------

                if self._is_spt_content_question(
                        question_lower
                ):

                    if is_spt_document:

                        score += 0.40

                    if is_bpe:

                        score -= 0.40

            # ==========================================================
            # FAKTUR
            # ==========================================================

            if document_type == "faktur":

                if "faktur" in filename_lower:

                    score += 0.40

                if "faktur pajak" in content_lower:

                    score += 0.30

            # ==========================================================
            # INVOICE
            # ==========================================================

            if document_type == "invoice":

                if "invoice" in filename_lower:

                    score += 0.40

            # ==========================================================
            # LOKASI
            # ==========================================================

            if location_question:

                if file_path:

                    score += 0.20

                if filename:

                    score += 0.10

            # ==========================================================
            # TABEL
            # ==========================================================

            if is_table_question:

                table_terms = [
                    "[table -",
                    "menurut wp/spt",
                    "menurut wajib pajak",
                    "menurut pemeriksa",
                    "pos-pos",
                    "keterangan dan/atau pembahasan",
                ]

                matched_table_terms = sum(
                    1
                    for term in table_terms
                    if term in content_lower
                )

                score += min(
                    matched_table_terms * 0.15,
                    0.60,
                    )

            # ==========================================================
            # ANGKA
            # ==========================================================

            if is_numeric_question:

                if "rp" in content_lower:

                    score += 0.05

                number_count = len(
                    re.findall(
                        r"\b\d[\d.,]*\b",
                        content_lower,
                    )
                )

                if number_count >= 3:

                    score += 0.10

                if number_count >= 6:

                    score += 0.05

            # ==========================================================
            # KATA PENTING PERTANYAAN
            # ==========================================================

            matched_question_terms = sum(
                1
                for term in question_terms
                if term in content_lower
            )

            score += min(
                matched_question_terms * 0.04,
                0.30,
                )

            # ==========================================================
            # PEREDARAN USAHA
            # ==========================================================

            if "peredaran usaha" in question_lower:

                if "peredaran usaha" in content_lower:

                    score += 0.25

            # ==========================================================
            # SIMPAN
            # ==========================================================

            reranked.append(
                (
                    score,
                    result,
                )
            )

        # ==============================================================
        # 11. SORT
        # ==============================================================

        reranked.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        # ==============================================================
        # 12. MODE LOKASI
        # ==============================================================

        if location_question:

            selected_results: List[SearchResult] = []

            seen_documents = set()

            max_documents = max(
                top_k,
                10,
            )

            for _, result in reranked:

                metadata = (
                        result.metadata or {}
                )

                filename = self._get_filename(
                    metadata
                )

                filename_lower = (
                    filename.lower()
                )

                file_path_lower = str(
                    metadata.get(
                        "file_path",
                        "",
                    )
                ).lower()

                content_lower = (
                        result.content or ""
                ).lower()

                # ------------------------------------------------------
                # Jika pertanyaan SPT,
                # prioritaskan hanya dokumen SPT.
                # ------------------------------------------------------

                if spt_question:

                    if self._is_bpe_document(
                            filename_lower,
                            file_path_lower,
                            content_lower,
                    ):

                        continue

                    if not self._is_spt_document(
                            filename_lower,
                            file_path_lower,
                            content_lower,
                    ):

                        continue

                if self._is_duplicate_chunk(
                        result,
                        selected_results,
                ):

                    continue

                document_id = str(
                    metadata.get(
                        "document_id",
                        "",
                    )
                )

                if (
                        document_id
                        and document_id in seen_documents
                ):

                    continue

                selected_results.append(
                    result
                )

                if document_id:

                    seen_documents.add(
                        document_id
                    )

                if (
                        len(selected_results)
                        >= max_documents
                ):

                    break

            if not selected_results:

                return [], ""

            return self._build_location_context(
                selected_results,
                top_k=len(
                    selected_results
                ),
            )

        # ==============================================================
        # 13. PILIH CHUNK NORMAL
        # ==============================================================

        selected_results: List[SearchResult] = []

        seen_chunks = set()

        if (
                is_table_question
                or is_numeric_question
        ):

            max_chunks_per_document = 4

            target_results = max(
                top_k,
                6,
            )

        else:

            max_chunks_per_document = 2

            target_results = top_k

        document_counts: dict[str, int] = {}

        for _, result in reranked:

            if self._is_duplicate_chunk(
                    result,
                    selected_results,
            ):

                continue

            chunk_id = str(
                result.chunk_id or ""
            )

            if (
                    chunk_id
                    and chunk_id in seen_chunks
            ):

                continue

            metadata = (
                    result.metadata or {}
            )

            document_id = str(
                metadata.get(
                    "document_id",
                    "",
                )
            )

            if not document_id:

                document_id = (
                    f"chunk:{chunk_id}"
                )

            current_count = (
                document_counts.get(
                    document_id,
                    0,
                )
            )

            if (
                    current_count
                    >= max_chunks_per_document
            ):

                continue

            selected_results.append(
                result
            )

            document_counts[
                document_id
            ] = current_count + 1

            if chunk_id:

                seen_chunks.add(
                    chunk_id
                )

            if (
                    len(selected_results)
                    >= target_results
            ):

                break

        # ==============================================================
        # 14. FALLBACK
        # ==============================================================

        if not selected_results:

            selected_results = (
                raw_results[
                    :max(
                        target_results,
                        1,
                    )
                ]
            )

        # ==============================================================
        # 15. BUILD CONTEXT
        # ==============================================================

        return self._build_context(
            selected_results,
            top_k=len(
                selected_results
            ),
        )

    # ==================================================================
    # DETEKSI TAHUN
    # ==================================================================

    @staticmethod
    def _extract_year(
            question: str,
    ) -> str | None:

        match = re.search(
            r"\b(20\d{2})\b",
            question,
        )

        if not match:

            return None

        return match.group(1)

    # ==================================================================
    # SEARCH ONLY
    # ==================================================================

    @staticmethod
    def _is_search_only_question(
            question: str,
    ) -> bool:

        question_lower = (
            question.lower().strip()
        )

        location_terms = [
            "dimana",
            "di mana",
            "terletak",
            "letaknya",
            "lokasi",
            "folder",
            "ada dimana",
            "ada di mana",
        ]

        file_search_terms = [
            "carikan file",
            "cari file",
            "temukan file",
            "tampilkan file",
            "carikan dokumen",
            "cari dokumen",
            "temukan dokumen",
            "tampilkan dokumen",
        ]

        return (
                any(
                    term in question_lower
                    for term in location_terms
                )
                or
                any(
                    term in question_lower
                    for term in file_search_terms
                )
        )

    # ==================================================================
    # BUILD SEARCH ANSWER
    # ==================================================================

    @staticmethod
    def _build_search_answer(
            question: str,
            chunks: List[SourceChunk],
    ) -> str:

        if not chunks:

            return (
                "Informasi tersebut tidak ditemukan "
                "dalam dokumen yang diindeks."
            )

        question_lower = (
            question.lower()
        )

        is_location = any(
            term in question_lower
            for term in [
                "dimana",
                "di mana",
                "terletak",
                "letaknya",
                "lokasi",
                "folder",
                "ada dimana",
                "ada di mana",
            ]
        )

        if is_location:

            paths = []

            for chunk in chunks:

                path = (
                        chunk.file_path
                        or ""
                ).strip()

                if (
                        path
                        and path not in paths
                ):

                    paths.append(path)

            if paths:

                folders = []

                for path in paths:

                    try:

                        folder = str(
                            Path(path).parent
                        )

                    except Exception:

                        folder = ""

                    if (
                            folder
                            and folder not in folders
                    ):

                        folders.append(
                            folder
                        )

                if len(folders) == 1:

                    return (
                        "Dokumen yang sesuai ditemukan "
                        f"di folder:\n{folders[0]}"
                    )

                lines = [
                    "Dokumen yang sesuai ditemukan di:"
                ]

                for path in paths:

                    lines.append(
                        f"- {path}"
                    )

                return "\n".join(
                    lines
                )

        lines = [
            "Dokumen yang paling relevan:"
        ]

        seen = set()

        for chunk in chunks:

            key = (
                    chunk.file_path
                    or chunk.filename
            )

            if key in seen:

                continue

            seen.add(key)

            lines.append(
                f"- {chunk.filename}"
            )

            if chunk.file_path:

                lines.append(
                    f"  Path: {chunk.file_path}"
                )

        return "\n".join(
            lines
        )

    # ==================================================================
    # RESULT YEAR
    # ==================================================================

    @classmethod
    def _result_matches_year(
            cls,
            result: SearchResult,
            year: str,
    ) -> bool:

        metadata = (
                result.metadata or {}
        )

        # --------------------------------------------------------------
        # PRIORITAS 1
        # Metadata tahun
        # --------------------------------------------------------------

        metadata_year = str(
            metadata.get(
                "year",
                "",
            )
        ).strip()

        if metadata_year:

            return (
                    metadata_year == year
            )

        # --------------------------------------------------------------
        # PRIORITAS 2
        # Nama file
        # --------------------------------------------------------------

        filename = cls._get_filename(
            metadata
        )

        if cls._contains_year(
                filename,
                year,
        ):

            return True

        # --------------------------------------------------------------
        # PRIORITAS 3
        # Path file
        # --------------------------------------------------------------

        file_path = str(
            metadata.get(
                "file_path",
                "",
            )
        )

        if cls._contains_year(
                file_path,
                year,
        ):

            return True

        # --------------------------------------------------------------
        # JANGAN menggunakan isi content sebagai penentu tahun.
        #
        # Sebuah PDF tahun 2021 bisa saja menyebut:
        # - tahun 2022
        # - tahun 2024
        # - tahun 2025
        #
        # sehingga content tidak boleh dijadikan bukti bahwa
        # dokumen tersebut adalah dokumen tahun tersebut.
        # --------------------------------------------------------------

        return False

    # ==================================================================
    # CONTAINS YEAR
    # ==================================================================

    @staticmethod
    def _contains_year(
            value: str,
            year: str,
    ) -> bool:

        if not value:

            return False

        return bool(
            re.search(
                rf"\b{re.escape(year)}\b",
                str(value),
            )
        )

    # ==================================================================
    # DETECT DOCUMENT TYPE
    # ==================================================================

    @staticmethod
    def _detect_document_type(
            question_lower: str,
    ) -> str | None:

        if any(
                phrase in question_lower
                for phrase in [
                    "spt",
                    "surat pemberitahuan",
                    "tanda terima spt",
                    "masa pajak",
                    "tahun pajak",
                ]
        ):

            return "spt"

        if any(
                phrase in question_lower
                for phrase in [
                    "faktur pajak",
                    "faktur",
                ]
        ):

            return "faktur"

        if "invoice" in question_lower:

            return "invoice"

        return None

    # ==================================================================
    # SPT CONTENT QUESTION
    # ==================================================================

    @staticmethod
    def _is_spt_content_question(
            question_lower: str,
    ) -> bool:

        content_terms = [
            "berapa",
            "nilai",
            "jumlah",
            "total",
            "ppn",
            "pajak",
            "dpp",
            "peredaran",
            "penyerahan",
            "pembayaran",
            "dilaporkan",
            "terutang",
            "kurang bayar",
            "lebih bayar",
            "status",
        ]

        return any(
            term in question_lower
            for term in content_terms
        )

    # ==================================================================
    # NUMERIC / TABLE QUESTION
    # ==================================================================

    @staticmethod
    def _is_numeric_or_table_question(
            question_lower: str,
    ) -> bool:

        numeric_terms = [
            "berapa",
            "nilai",
            "jumlah",
            "total",
            "selisih",
            "dilaporkan",
            "ppn",
            "dpp",
            "pajak",
            "kurang bayar",
            "lebih bayar",
            "peredaran usaha",
            "menurut wp",
            "menurut wajib pajak",
            "menurut pemeriksa",
        ]

        table_terms = [
            "tabel",
            "kolom",
            "baris",
            "menurut wp",
            "menurut wajib pajak",
            "menurut pemeriksa",
            "peredaran usaha",
        ]

        return (
                any(
                    term in question_lower
                    for term in numeric_terms
                )
                or
                any(
                    term in question_lower
                    for term in table_terms
                )
        )

    # ==================================================================
    # PERTANYAAN PEREDARAN USAHA
    # ==================================================================

    @staticmethod
    def _is_peredaran_usaha_comparison_question(
            question_lower: str,
    ) -> bool:

        if "peredaran usaha" not in question_lower:

            return False

        has_wp = (
                "menurut wp" in question_lower
                or
                "menurut wajib pajak" in question_lower
                or
                "wp/spt" in question_lower
        )

        has_pemeriksa = (
                "menurut pemeriksa" in question_lower
                or
                "pemeriksa" in question_lower
        )

        return (
                has_wp
                and
                has_pemeriksa
        )

    # ==================================================================
    # EKSTRAKSI NILAI WP VS PEMERIKSA
    # ==================================================================

    @staticmethod
    def _extract_wp_pemeriksa_values(
            chunks: List[SourceChunk],
            keyword: str,
    ) -> tuple[str, str] | None:

        if not chunks:

            return None

        # --------------------------------------------------------------
        # Gabungkan chunk
        # --------------------------------------------------------------

        texts = []

        for chunk in chunks:

            content = (
                    chunk.chunk_text or ""
            ).strip()

            if content:

                texts.append(content)

        if not texts:

            return None

        combined = "\n".join(
            texts
        )

        # --------------------------------------------------------------
        # Normalisasi spasi
        # --------------------------------------------------------------

        normalized = re.sub(
            r"\s+",
            " ",
            combined,
        )

        normalized_lower = (
            normalized.lower()
        )

        keyword_lower = (
            keyword.lower()
        )

        # --------------------------------------------------------------
        # Cari keyword
        # --------------------------------------------------------------

        search_position = 0

        while True:

            position = (
                normalized_lower.find(
                    keyword_lower,
                    search_position,
                )
            )

            if position == -1:

                break

            # ----------------------------------------------------------
            # Ambil area tabel setelah keyword
            # ----------------------------------------------------------

            section_end = min(
                position + 1200,
                len(normalized),
                )

            section = normalized[
                position:
                section_end
            ]

            section_lower = (
                section.lower()
            )

            # ----------------------------------------------------------
            # Pastikan area tersebut benar-benar merupakan
            # tabel WP/SPT vs Pemeriksa.
            # ----------------------------------------------------------

            has_wp_header = (
                    "menurut wp/spt" in section_lower
                    or
                    "menurut wajib pajak" in section_lower
                    or
                    "menurut wp" in section_lower
            )

            has_pemeriksa_header = (
                    "menurut pemeriksa" in section_lower
            )

            if not (
                    has_wp_header
                    and has_pemeriksa_header
            ):

                search_position = (
                        position + len(keyword)
                )

                continue

            # ----------------------------------------------------------
            # Ambil angka Rupiah setelah keyword.
            #
            # Yang kita butuhkan:
            #
            # Peredaran Usaha
            # Rp 67,580,133,333
            # Rp 102,411,649,646
            #
            # BUKAN:
            #
            # Selisih
            # Rp 34,831,516,313
            # ----------------------------------------------------------

            values = re.findall(
                r"Rp\s*([\d][\d.,]*)",
                section,
                flags=re.IGNORECASE,
            )

            if len(values) >= 2:

                wp_value = values[0]
                pemeriksa_value = values[1]

                # ------------------------------------------------------
                # Validasi angka
                # ------------------------------------------------------

                if (
                        wp_value
                        and pemeriksa_value
                ):

                    return (
                        wp_value,
                        pemeriksa_value,
                    )

            search_position = (
                    position + len(keyword)
            )

        # --------------------------------------------------------------
        # Fallback:
        # cari baris "Peredaran Usaha" secara lebih lokal.
        # --------------------------------------------------------------

        lines = re.split(
            r"[\r\n]+",
            combined,
        )

        for index, line in enumerate(lines):

            line_lower = (
                line.lower()
            )

            if keyword_lower not in line_lower:

                continue

            local_section = " ".join(
                lines[
                    index:
                    min(
                        index + 5,
                        len(lines),
                        )
                ]
            )

            values = re.findall(
                r"Rp\s*([\d][\d.,]*)",
                local_section,
                flags=re.IGNORECASE,
            )

            if len(values) >= 2:

                return (
                    values[0],
                    values[1],
                )

        return None

    # ==================================================================
    # LOCATION QUESTION
    # ==================================================================

    @staticmethod
    def _is_location_question(
            question_lower: str,
    ) -> bool:

        location_terms = [
            "dimana",
            "di mana",
            "terletak",
            "letaknya",
            "lokasi",
            "folder",
            "file apa",
            "dokumen apa",
            "ada dimana",
            "ada di mana",
        ]

        return any(
            term in question_lower
            for term in location_terms
        )

    # ==================================================================
    # EXTRACT FILENAME
    # ==================================================================

    @staticmethod
    def _extract_filename(
            question: str,
    ) -> str | None:

        pattern = re.compile(
            r"([^\s\"'<>]+?\.(?:pdf|docx?|xlsx?|csv|txt|md|html?|json))",
            re.IGNORECASE,
        )

        match = pattern.search(
            question
        )

        if not match:

            return None

        return match.group(1).strip()

    # ==================================================================
    # DIRECT FILE SEARCH
    # ==================================================================

    def _find_file_directly(
            self,
            filename: str,
    ) -> List[SearchResult]:

        target = self._normalize_filename(
            filename
        )

        if not target:

            return []

        try:

            collection = (
                self.indexer.store.collection
            )

            results = collection.get(
                include=[
                    "documents",
                    "metadatas",
                ],
            )

        except Exception:

            return []

        ids = results.get(
            "ids",
            [],
        )

        docs = results.get(
            "documents",
            [],
        )

        metas = results.get(
            "metadatas",
            [],
        )

        output: List[SearchResult] = []

        for chunk_id, doc, meta in zip(
                ids,
                docs,
                metas,
        ):

            metadata = (
                    meta or {}
            )

            stored_filename = (
                self._get_filename(
                    metadata
                )
            )

            if (
                    self._normalize_filename(
                        stored_filename
                    )
                    != target
            ):

                continue

            output.append(
                SearchResult(
                    chunk_id=chunk_id,
                    document_id=metadata.get(
                        "document_id",
                        "",
                    ),
                    content=doc or "",
                    score=1.0,
                    metadata=metadata,
                )
            )

        def chunk_sort_key(
                result: SearchResult,
        ) -> int:

            try:

                return int(
                    result.metadata.get(
                        "chunk_index",
                        0,
                    )
                )

            except (
                    ValueError,
                    TypeError,
            ):

                return 0

        output.sort(
            key=chunk_sort_key
        )

        return output

    # ==================================================================
    # BUILD LOCATION CONTEXT
    # ==================================================================

    def _build_location_context(
            self,
            results: List[SearchResult],
            top_k: int = 5,
    ) -> tuple[
        List[SourceChunk],
        str,
    ]:

        source_chunks: List[SourceChunk] = []

        context_parts: List[str] = []

        seen_documents = set()

        for result in results:

            metadata = (
                    result.metadata or {}
            )

            document_id = str(
                metadata.get(
                    "document_id",
                    "",
                )
            )

            if (
                    document_id
                    and document_id in seen_documents
            ):

                continue

            if document_id:

                seen_documents.add(
                    document_id
                )

            filename = self._get_filename(
                metadata
            )

            file_path = str(
                metadata.get(
                    "file_path",
                    "",
                )
            ).strip()

            if not filename:

                filename = "unknown"

            source_chunks.append(
                SourceChunk(
                    filename=filename,
                    chunk_text=(
                            result.content
                            or ""
                    ),
                    score=float(
                        result.score
                    ),
                    file_path=file_path,
                )
            )

            context_parts.append(
                f"[DOKUMEN]\n"
                f"Nama file: {filename}\n"
                f"Lokasi file: {file_path}\n"
                f"Tahun: "
                f"{metadata.get('year', '')}\n"
            )

            if (
                    len(source_chunks)
                    >= top_k
            ):

                break

        if not context_parts:

            return [], ""

        context = (
                "INFORMASI LOKASI "
                "DOKUMEN YANG DITEMUKAN:\n\n"
                + "\n\n".join(
            context_parts
        )
        )

        return (
            source_chunks,
            context,
        )

    # ==================================================================
    # BUILD CONTEXT
    # ==================================================================

    def _build_context(
            self,
            results: List[SearchResult],
            top_k: int,
    ) -> tuple[
        List[SourceChunk],
        str,
    ]:

        selected_results: List[
            SearchResult
        ] = []

        for result in results:

            if self._is_duplicate_chunk(
                    result,
                    selected_results,
            ):

                continue

            selected_results.append(
                result
            )

            if (
                    len(selected_results)
                    >= top_k
            ):

                break

        source_chunks: List[
            SourceChunk
        ] = []

        context_parts: List[str] = []

        for result in selected_results:

            metadata = (
                    result.metadata or {}
            )

            filename = self._get_filename(
                metadata
            )

            file_path = str(
                metadata.get(
                    "file_path",
                    "",
                )
            )

            if not filename:

                filename = "unknown"

            source_chunks.append(
                SourceChunk(
                    filename=filename,
                    chunk_text=(
                            result.content
                            or ""
                    ),
                    score=float(
                        result.score
                    ),
                    file_path=file_path,
                )
            )

            context_parts.append(
                f"[Source: {filename}]\n"
                f"[File Path: {file_path}]\n"
                f"[Document ID: "
                f"{metadata.get('document_id', '')}]\n"
                f"[Tahun: "
                f"{metadata.get('year', '')}]\n"
                f"{result.content or ''}"
            )

        context = "\n\n".join(
            context_parts
        )

        return (
            source_chunks,
            context,
        )

    # ==================================================================
    # NORMALIZE FILENAME
    # ==================================================================

    @staticmethod
    def _normalize_filename(
            filename: str,
    ) -> str:

        value = str(
            filename or ""
        ).strip()

        value = value.replace(
            "\\",
            "/",
        )

        value = value.split(
            "/"
        )[-1]

        value = re.sub(
            r"\s+",
            " ",
            value,
        )

        return value.lower()

    # ==================================================================
    # GET FILENAME
    # ==================================================================

    @staticmethod
    def _get_filename(
            metadata: dict,
    ) -> str:

        filename = str(
            metadata.get(
                "filename",
                "",
            )
        ).strip()

        if filename:

            return Path(
                filename
            ).name

        file_path = str(
            metadata.get(
                "file_path",
                "",
            )
        ).strip()

        if file_path:

            return Path(
                file_path
            ).name

        return ""

    # ==================================================================
    # DUPLICATE CHUNK
    # ==================================================================

    @staticmethod
    def _is_duplicate_chunk(
            result: SearchResult,
            selected: List[SearchResult],
    ) -> bool:

        current = (
                result.content or ""
        ).strip().lower()

        if not current:

            return True

        for existing in selected:

            previous = (
                    existing.content or ""
            ).strip().lower()

            if not previous:

                continue

            shorter = min(
                len(current),
                len(previous),
            )

            if shorter < 100:

                continue

            if current in previous:

                return True

            if previous in current:

                return True

        return False

    # ==================================================================
    # DETEKSI DOKUMEN SPT
    # ==================================================================

    @staticmethod
    def _is_spt_document(
            filename_lower: str,
            file_path_lower: str,
            content_lower: str,
    ) -> bool:

        # --------------------------------------------------------------
        # Nama file / path
        # --------------------------------------------------------------

        if "spt" in filename_lower:

            return True

        if "\\spt " in file_path_lower:

            return True

        if "/spt " in file_path_lower:

            return True

        if "spt 202" in file_path_lower:

            return True

        # --------------------------------------------------------------
        # Struktur khas SPT
        # --------------------------------------------------------------

        spt_indicators = [
            "surat pemberitahuan masa pajak pertambahan nilai",
            "penyerahan barang dan jasa",
            "pajak masukan yang dapat diperhitungkan",
            "ppn kurang atau (lebih) bayar",
            "dpp ppn",
            "masa pajak",
        ]

        matched = sum(
            1
            for indicator in spt_indicators
            if indicator in content_lower
        )

        return matched >= 2

    # ==================================================================
    # DETEKSI BPE
    # ==================================================================

    @staticmethod
    def _is_bpe_document(
            filename_lower: str,
            file_path_lower: str,
            content_lower: str,
    ) -> bool:

        indicators = [
            "bpe",
            "bukti penerimaan elektronik",
            "tanda terima",
            "e-form",
        ]

        return any(
            indicator in filename_lower
            or indicator in file_path_lower
            or indicator in content_lower
            for indicator in indicators
        )

    # ==================================================================
    # DETEKSI NON-SPT
    # ==================================================================

    @staticmethod
    def _is_non_spt_document(
            filename_lower: str,
            file_path_lower: str,
            content_lower: str,
    ) -> bool:

        non_spt_terms = [
            "lapkeu",
            "laporan keuangan",
            "invoice",
            "faktur pajak",
            "e-billing",
            "surat keputusan",
            "surat pemberitahuan djp",
        ]

        return any(
            term in filename_lower
            or term in file_path_lower
            or term in content_lower
            for term in non_spt_terms
        )

    # ==================================================================
    # EXTRACT QUESTION TERMS
    # ==================================================================

    @staticmethod
    def _extract_question_terms(
            question_lower: str,
    ) -> List[str]:

        stopwords = {
            "yang",
            "dan",
            "atau",
            "dari",
            "dalam",
            "pada",
            "untuk",
            "dengan",
            "menurut",
            "berapa",
            "apa",
            "apakah",
            "ada",
            "dimana",
            "di",
            "mana",
            "tahun",
            "dokumen",
            "file",
            "nilai",
            "tersebut",
            "ini",
            "itu",
        }

        words = re.findall(
            r"[a-zA-Z0-9]+",
            question_lower,
        )

        terms = []

        for word in words:

            if len(word) < 3:

                continue

            if word in stopwords:

                continue

            if word not in terms:

                terms.append(word)

        return terms

    # ==================================================================
    # RERANK CHUNK FILE SPESIFIK
    # ==================================================================

    def _rerank_file_chunks(
            self,
            results: List[SearchResult],
            question_lower: str,
    ) -> List[SearchResult]:

        question_terms = (
            self._extract_question_terms(
                question_lower
            )
        )

        ranked = []

        is_numeric = (
            self._is_numeric_or_table_question(
                question_lower
            )
        )

        for result in results:

            content = (
                    result.content or ""
            )

            content_lower = (
                content.lower()
            )

            score = 0.0

            # ----------------------------------------------------------
            # Score awal
            # ----------------------------------------------------------

            score += float(
                result.score
            )

            # ----------------------------------------------------------
            # Kata penting
            # ----------------------------------------------------------

            matched_terms = sum(
                1
                for term in question_terms
                if term in content_lower
            )

            score += min(
                matched_terms * 0.15,
                1.00,
                )

            # ----------------------------------------------------------
            # Angka
            # ----------------------------------------------------------

            if is_numeric:

                number_count = len(
                    re.findall(
                        r"\b\d[\d.,]*\b",
                        content_lower,
                    )
                )

                if number_count >= 1:

                    score += 0.15

                if number_count >= 3:

                    score += 0.15

                if "rp" in content_lower:

                    score += 0.10

            # ----------------------------------------------------------
            # Istilah tabel
            # ----------------------------------------------------------

            important_terms = [
                "dilaporkan",
                "menurut wp",
                "menurut wajib pajak",
                "menurut pemeriksa",
                "peredaran usaha",
                "dpp",
                "ppn",
                "kurang bayar",
                "lebih bayar",
            ]

            for term in important_terms:

                if term in question_lower:

                    if term in content_lower:

                        score += 0.20

            ranked.append(
                (
                    score,
                    result,
                )
            )

        ranked.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return [
            result
            for _, result in ranked
        ]

    # ==================================================================
    # EXPAND TABLE CHUNKS
    # ==================================================================

    def _expand_table_chunks(
            self,
            all_results: List[SearchResult],
            ranked_results: List[SearchResult],
            question_lower: str,
            radius: int = 2,
    ) -> List[SearchResult]:

        if not all_results:

            return []

        # --------------------------------------------------------------
        # Susun berdasarkan posisi chunk asli
        # --------------------------------------------------------------

        sorted_results = sorted(
            all_results,
            key=lambda result:
            self._get_chunk_index(result),
        )

        # --------------------------------------------------------------
        # Tentukan kata kunci tabel
        # --------------------------------------------------------------

        table_terms = []

        if "peredaran usaha" in question_lower:

            table_terms.extend([
                "peredaran usaha",
            ])

        if (
                "menurut wp" in question_lower
                or
                "menurut wajib pajak" in question_lower
                or
                "menurut wp/spt" in question_lower
        ):

            table_terms.extend([
                "menurut wp",
                "menurut wajib pajak",
                "menurut wp/spt",
            ])

        if "menurut pemeriksa" in question_lower:

            table_terms.append(
                "menurut pemeriksa"
            )

        table_terms.extend([
            "pos-pos",
            "keterangan dan/atau pembahasan",
        ])

        # --------------------------------------------------------------
        # Cari chunk target
        # --------------------------------------------------------------

        target_indexes = set()

        for result in sorted_results:

            content_lower = (
                    result.content or ""
            ).lower()

            matched = any(
                term in content_lower
                for term in table_terms
            )

            if matched:

                target_indexes.add(
                    self._get_chunk_index(
                        result
                    )
                )

        # --------------------------------------------------------------
        # Kalau tidak menemukan target,
        # gunakan beberapa hasil reranking terbaik
        # --------------------------------------------------------------

        if not target_indexes:

            best_results = (
                ranked_results[:5]
            )

            selected = []

            for result in best_results:

                if not self._is_duplicate_chunk(
                        result,
                        selected,
                ):

                    selected.append(
                        result
                    )

            return selected

        # --------------------------------------------------------------
        # Ambil chunk di sekitar target
        # --------------------------------------------------------------

        expanded_indexes = set()

        for target_index in target_indexes:

            for offset in range(
                    -radius,
                    radius + 1,
            ):

                expanded_indexes.add(
                    target_index + offset
                )

        # --------------------------------------------------------------
        # Bentuk hasil berdasarkan urutan asli
        # --------------------------------------------------------------

        expanded_results = []

        for result in sorted_results:

            chunk_index = (
                self._get_chunk_index(
                    result
                )
            )

            if chunk_index not in expanded_indexes:

                continue

            if self._is_duplicate_chunk(
                    result,
                    expanded_results,
            ):

                continue

            expanded_results.append(
                result
            )

        # --------------------------------------------------------------
        # Jika hasil terlalu banyak, batasi
        # --------------------------------------------------------------

        max_expanded_chunks = 10

        if (
                len(expanded_results)
                > max_expanded_chunks
        ):

            priority = []
            normal = []

            for result in expanded_results:

                content_lower = (
                        result.content or ""
                ).lower()

                if any(
                        term in content_lower
                        for term in table_terms
                ):

                    priority.append(
                        result
                    )

                else:

                    normal.append(
                        result
                    )

            expanded_results = (
                    priority
                    + normal
            )[:max_expanded_chunks]

            # ----------------------------------------------------------
            # Kembalikan berdasarkan urutan chunk
            # ----------------------------------------------------------

            expanded_results.sort(
                key=lambda result:
                self._get_chunk_index(
                    result
                )
            )

        return expanded_results

    # ==================================================================
    # GET CHUNK INDEX
    # ==================================================================

    @staticmethod
    def _get_chunk_index(
            result: SearchResult,
    ) -> int:

        metadata = (
                result.metadata or {}
        )

        try:

            return int(
                metadata.get(
                    "chunk_index",
                    0,
                )
            )

        except (
                ValueError,
                TypeError,
        ):

            return 0
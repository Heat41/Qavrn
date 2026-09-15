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
    - prioritas pertanyaan keuangan
    - strict filtering dokumen keuangan tahunan
    - dukungan pertanyaan tabel / angka
    - pembatasan chunk per dokumen
    - fallback retrieval
    - proteksi duplicate chunk
    - ekstraksi tabel deterministik
    - ekstraksi nilai finansial deterministik
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

        deterministic = self._resolve_deterministic_answer(
            question,
            chunks,
        )

        if deterministic is not None:
            answer, answer_sources, answer_mode = deterministic

            return RAGResponse(
                answer=answer,
                sources=answer_sources,
                model_used=answer_mode,
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

        deterministic = self._resolve_deterministic_answer(
            question,
            chunks,
        )

        if deterministic is not None:
            answer, answer_sources, _ = deterministic

            def deterministic_stream() -> Iterator[str]:
                yield answer

            return deterministic_stream(), answer_sources

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
    # DETERMINISTIC ANSWER
    # ==================================================================

    def _resolve_deterministic_answer(
            self,
            question: str,
            chunks: List[SourceChunk],
    ) -> tuple[str, List[SourceChunk], str] | None:
        """
        Menyatukan jawaban deterministik yang digunakan oleh query()
        dan query_stream() agar keduanya tidak memiliki logic ganda.
        """

        question_lower = question.lower().strip()
        requested_year = self._extract_year(question)

        # --------------------------------------------------------------
        # ALAMAT PERUSAHAAN
        # --------------------------------------------------------------

        if self._is_address_question(question_lower):

            candidates = self._extract_company_address_candidates(
                question_lower,
                requested_year,
            )

            if candidates:

                requested_document_type = self._detect_document_type(
                    question_lower
                )

                if requested_document_type:
                    filtered_candidates = [
                        candidate
                        for candidate in candidates
                        if candidate[1] == requested_document_type
                    ]

                    if filtered_candidates:
                        candidates = filtered_candidates

                unique_candidates = {}

                for address, document_type, source in candidates:
                    normalized_address = re.sub(
                        r"\s+",
                        " ",
                        address.lower(),
                    ).strip(" ,.")

                    if normalized_address not in unique_candidates:
                        unique_candidates[normalized_address] = (
                            address,
                            document_type,
                            source,
                        )

                candidate_values = list(
                    unique_candidates.values()
                )

                if len(candidate_values) == 1:
                    address = candidate_values[0][0]

                    answer = (
                        f"Alamat perusahaan"
                        f"{f' tahun {requested_year}' if requested_year else ''}"
                        f" adalah {address}."
                    )

                else:
                    answer_lines = [
                        (
                            f"Ditemukan {len(candidate_values)} "
                            "alamat berbeda"
                            f"{f' pada tahun {requested_year}' if requested_year else ''}:"
                        )
                    ]

                    for address, document_type, _ in candidate_values:
                        answer_lines.append(
                            f"- {document_type}: {address}"
                        )

                    answer_lines.append(
                        "Silakan tentukan sumber dokumen yang dimaksud."
                    )

                    answer = "\n".join(answer_lines)

                answer_sources = [
                    candidate[2]
                    for candidate in candidate_values
                ]

                return answer, answer_sources, "table"

        # --------------------------------------------------------------
        # DIREKTUR
        # --------------------------------------------------------------

        director_question = any(
            phrase in question_lower
            for phrase in [
                "direktur",
                "nama direktur",
                "siapa direktur",
                "pimpinan",
                "siapa pimpinan",
            ]
        )

        if director_question:

            director_name = self._extract_director_name(
                chunks
            )

            if director_name:

                answer = "Direktur perusahaan"

                if requested_year:
                    answer += f" tahun {requested_year}"

                answer += f" adalah {director_name}."

                return answer, chunks, "table"

        # --------------------------------------------------------------
        # PEREDARAN USAHA WP VS PEMERIKSA
        # --------------------------------------------------------------

        if self._is_peredaran_usaha_comparison_question(
                question_lower
        ):

            table_answer = self._extract_wp_pemeriksa_values(
                chunks,
                "Peredaran Usaha",
            )

            if table_answer:

                wp_value, pemeriksa_value = table_answer

                answer = (
                    "Peredaran Usaha menurut "
                    f"Wajib Pajak: Rp {wp_value}\n"
                    "Peredaran Usaha menurut "
                    f"Pemeriksa: Rp {pemeriksa_value}"
                )

                return answer, chunks, "table"

            return (
                "Data perbandingan Peredaran Usaha "
                "menurut Wajib Pajak dan menurut Pemeriksa "
                "tidak ditemukan dalam dokumen yang relevan.",
                chunks,
                "search",
            )

        # --------------------------------------------------------------
        # NILAI FINANSIAL TAHUNAN
        # --------------------------------------------------------------

        if self._is_annual_financial_question(
                question_lower,
                requested_year,
        ):

            extraction_chunks = chunks

            if requested_year:

                financial_results = self._get_financial_document_chunks(
                    requested_year
                )

                if financial_results:
                    extraction_chunks = [
                        SourceChunk(
                            filename=str(
                                result.metadata.get(
                                    "filename",
                                    "",
                                )
                            ),
                            chunk_text=(
                                result.content or ""
                            ),
                            score=float(
                                result.score or 0.0
                            ),
                            file_path=str(
                                result.metadata.get(
                                    "file_path",
                                    "",
                                )
                            ),
                        )
                        for result in financial_results
                    ]

            financial_value = self._extract_annual_financial_value(
                extraction_chunks,
                question_lower,
            )

            if financial_value:

                normalized_value = self._format_financial_value(
                    financial_value
                )

                financial_label = self._get_financial_answer_label(
                    question_lower
                )

                answer = (
                    f"{financial_label} tahun "
                    f"{requested_year} sebesar "
                    f"{normalized_value}."
                )

                return answer, chunks, "table"

        return None

    # ==================================================================
    # DIRECTOR RETRIEVAL
    # ==================================================================

    def _find_director_chunks(
            self,
            year: str | None,
            top_k: int = 10,
    ) -> List[SearchResult]:

        """
        Mencari chunk yang secara eksplisit memuat identitas
        Direktur perusahaan.

        Tidak menggunakan embedding query.
        Filter utama:
        - tahun dokumen
        - kata "direktur"
        - dokumen perusahaan
        """

        try:
            target_year = str(year or "").strip()

            # Ambil metadata + isi seluruh chunk
            results = self.indexer.store.collection.get(
                include=[
                    "documents",
                    "metadatas",
                ]
            )
        except Exception:
            return []

        documents = results.get("documents", []) or []
        metadatas = results.get("metadatas", []) or []
        ids = results.get("ids", []) or []

        candidates: List[SearchResult] = []

        for chunk_id, content, metadata in zip(
                ids,
                documents,
                metadatas,
        ):
            metadata = metadata or {}

            content = str(content or "").strip()
            if not content:
                continue

            content_lower = content.lower()

            # ----------------------------------------------------------
            # HARUS mengandung jabatan direktur
            # ----------------------------------------------------------

            if "direktur" not in content_lower:
                continue

            # ----------------------------------------------------------
            # Filter tahun
            # ----------------------------------------------------------

            if target_year:
                metadata_year = str(
                    metadata.get("year", "")
                ).strip()

                if metadata_year != target_year:
                    continue

            filename = self._get_filename(metadata)
            filename_lower = filename.lower()


            file_path = str(
                metadata.get("file_path", "")
            ).lower()

            # ----------------------------------------------------------
            # Jangan ambil Direktorat Jenderal Pajak
            # ----------------------------------------------------------

            if (
                "direktur jenderal pajak"
                in content_lower
            ):
                continue

            if (
                "direktorat jenderal"
                in content_lower
                and "pt. arinsa energy pratama"
                not in content_lower
                and "pt arinsa energy pratama"
                not in content_lower
            ):
                continue

            # ----------------------------------------------------------
            # Hitung prioritas
            # ----------------------------------------------------------

            score = 0.0

            # Jabatan direktur
            score += 2.00

            # Identitas perusahaan
            if (
                "arinsa energy pratama"
                in content_lower
            ):
                score += 2.00

            # Nama PT
            if "pt." in content_lower:
                score += 0.30

            # File laporan keuangan / akta / profil
            for term in [
                "lapkeu",
                "laporan keuangan",
                "akta",
                "profil",
                "daftar peredaran usaha",
            ]:
                if term in filename_lower:
                    score += 0.50

            # Path laporan keuangan
            if (
                "laporan keuangan"
                in file_path
            ):
                score += 0.50

            # Chunk dengan nama + direktur biasanya sangat pendek
            # dan sangat relevan.
            normalized = re.sub(
                r"\s+",
                " ",
                content,
            ).strip()

            if re.search(
                r"r\s*u\s*s\s*['â€™`]\s*a\s*n",
                normalized,
                flags=re.IGNORECASE,
            ):
                score += 2.50

            candidates.append(
                SearchResult(
                    chunk_id=chunk_id,
                    document_id=str(
                        metadata.get(
                            "document_id",
                            "",
                        )
                    ),
                    content=content,
                    score=score,
                    metadata=metadata,
                )
            )

        candidates.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        return candidates[:top_k]


    # ==================================================================
    # EXTRACT DIRECTOR NAME
    # ==================================================================

    @staticmethod
    def _extract_director_name(
            chunks: List[SourceChunk],
    ) -> str | None:

        if not chunks:
            return None

        # --------------------------------------------------------------
        # Normalisasi isi chunk
        # --------------------------------------------------------------

        for chunk in chunks:

            # SourceChunk memakai chunk_text.
            # SearchResult (yang masih dipakai query() lama)
            # memakai content. Dukungan keduanya dibuat agar helper
            # aman dipanggil dari kedua jalur.
            content = getattr(
                chunk,
                "chunk_text",
                None,
            )

            if content is None:
                content = getattr(
                    chunk,
                    "content",
                    "",
                )

            content = str(
                content or ""
            ).strip()

            if not content:
                continue

            normalized = re.sub(
                r"\s+",
                " ",
                content,
            ).strip()

            # Normalisasi apostrophe/quote OCR
            normalized = (
                normalized
                .replace("â€˜", "'")
                .replace("â€™", "'")
                .replace("`", "'")
                .replace("Ã¢â‚¬Ëœ", "'")
                .replace("Ã¢â‚¬â„¢", "'")
            )

            # ----------------------------------------------------------
            # PRIORITAS 1
            #
            # Identitas perusahaan harus ada di chunk yang sama.
            # Jangan hanya mencari "nama + Direktur", karena satu
            # chunk XLS dapat memuat entitas lain.
            # ----------------------------------------------------------

            has_arinsa = bool(
                re.search(
                    r"\bpt\.?\s+arinsa\s+energy\s+pratama\b",
                    normalized,
                    flags=re.IGNORECASE,
                )
                or re.search(
                    r"\barinsa\s+energy\s+pratama\b",
                    normalized,
                    flags=re.IGNORECASE,
                )
            )

            if has_arinsa:

                # Format yang kita temukan pada dokumen:
                # RUS'AN Direktur
                # R U S ' A N Direktur
                direct_name_patterns = [
                    r"\bR\s*U\s*S\s*'\s*A\s*N\b\s+direktur\b",
                    r"\bRUS\s*'\s*AN\b\s+direktur\b",
                    r"\bdirektur\b\s*[:\-]\s*R\s*U\s*S\s*'\s*A\s*N\b",
                    r"\bdirektur\b\s*[:\-]\s*RUS\s*'\s*AN\b",
                ]

                for pattern in direct_name_patterns:
                    if re.search(
                        pattern,
                        normalized,
                        flags=re.IGNORECASE,
                    ):
                        return "RUS'AN"

                # Format SPT:
                # Nama Jelas : RUS'AN
                # ...
                # Jabatan : DIREKTUR
                name_match = re.search(
                    r"nama\s+jelas\s*[:\-]\s*"
                    r"(R\s*U\s*S\s*['â€™]?\s*A\s*N|RUS\s*'\s*AN)"
                    r".{0,180}?"
                    r"jabatan\s*[:\-]\s*direktur\b",
                    normalized,
                    flags=re.IGNORECASE,
                )

                if name_match:
                    return "RUS'AN"

            # ----------------------------------------------------------
            # PRIORITAS 2
            #
            # Untuk dokumen SPT, identitas perusahaan dapat berada
            # sebelum blok Nama Jelas/Jabatan.
            # ----------------------------------------------------------

            name_match = re.search(
                r"nama\s+jelas\s*[:\-]\s*"
                r"(R\s*U\s*S\s*['â€™]?\s*A\s*N|RUS\s*'\s*AN)"
                r".{0,180}?"
                r"jabatan\s*[:\-]\s*direktur\b",
                normalized,
                flags=re.IGNORECASE,
            )

            if name_match:
                return "RUS'AN"

        return None



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

        financial_question = (
            self._is_financial_question(
                question_lower
            )
        )

        # ==============================================================
        # DETEKSI PERTANYAAN JABATAN / DIREKTUR
        # ==============================================================

        director_question = any(
            phrase in question_lower
            for phrase in [
                "direktur",

                "nama direktur",
                "siapa direktur",
                "pimpinan",
                "siapa pimpinan",
            ]
        )

        # ==============================================================
        # PERTANYAAN PEREDARAN USAHA
        # WP VS PEMERIKSA
        # ==============================================================

        comparison_question = (
            self._is_peredaran_usaha_comparison_question(
                question_lower
            )
        )

        # ==============================================================
        # PERTANYAAN FINANSIAL TAHUNAN
        # ==============================================================

        annual_financial_question = (
            self._is_annual_financial_question(
                question_lower,
                requested_year,
            )
        )

        # --------------------------------------------------------------
        # Pertanyaan WP vs Pemeriksa harus diproses melalui jalur
        # comparison khusus, bukan annual financial biasa.
        # --------------------------------------------------------------

        if comparison_question:

            annual_financial_question = False

        # ==============================================================
        # VARIABEL HASIL
        # ==============================================================

        direct_results: List[SearchResult] = []

        ranked_file_results: List[SearchResult] = []


        # ==============================================================
        # 3A. RETRIEVAL KHUSUS DIREKTUR
        # ==============================================================
        if director_question:

            director_results = (
                self._find_director_chunks(
                    requested_year,
                    top_k=max(
                        top_k,
                        10,
                    ),
                )
            )

            if director_results:

                return self._build_context(
                    director_results,
                    top_k=len(
                        director_results
                    ),
                )


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

                return self._build_context(
                    ranked_file_results,
                    top_k=top_k,
                )


        # ==============================================================
        # 4A. PEREDARAN USAHA WP VS PEMERIKSA
        #
        # Jangan gunakan vector similarity.
        # Ambil langsung chunk tabel pembanding.
        # ==============================================================

        if comparison_question:

            comparison_results = (
                self._find_peredaran_comparison_chunks(
                    requested_year
                )
            )

            if comparison_results:

                return self._build_context(
                    comparison_results,
                    top_k=min(
                        len(comparison_results),
                        10,
                    ),
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

        if annual_financial_question:
            candidate_k = max(
                top_k * 60,
                300,
            )
        else:
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
            # KEUANGAN
            # ==========================================================

            if financial_question:

                financial_filename_terms = [
                    "pendapatan",
                    "lapkeu",
                    "laporan keuangan",
                    "neraca",
                    "rugi",
                    "laba",
                ]

                financial_content_terms = [
                    "pendapatan",
                    "pendapatan proyek",
                    "pendapatan kotor",
                    "pendapatan bersih",
                    "pendapatan bersih setelah pajak",
                    "laba/rugi",
                    "laba bersih",
                    "laba tahun berjalan",
                    "peredaran usaha",
                    "total peredaran usaha",
                    "hasil usaha",
                    "biaya pokok penjualan",
                    "hpp proyek",
                    "penjualan",
                ]

                transaction_filename_terms = [
                    "faktur",
                    "invoice",
                    "e-billing",
                    "ebilling",
                    "billing",
                    "bukti potong",
                    "bpe",
                    "tanda terima",
                ]

                filename_match = sum(
                    1
                    for term in financial_filename_terms
                    if term in filename_lower
                )

                content_match = sum(
                    1
                    for term in financial_content_terms
                    if term in content_lower
                )

                transaction_match = sum(
                    1
                    for term in transaction_filename_terms
                    if term in filename_lower
                )

                score += min(
                    filename_match * 0.60,
                    1.20,
                )

                score += min(
                    content_match * 0.20,
                    1.00,
                )

                if (
                        transaction_match
                        and document_type not in {
                            "faktur",
                            "invoice",
                            "spt",
                        }
                ):

                    score -= 0.45

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

                if is_spt_document:

                    score += 1.00

                if is_bpe:

                    score -= 1.00

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
            # DIREKTUR / JABATAN
            #
            # Bedakan "Direktur perusahaan" dari:
            # "Direktur Jenderal Pajak"
            #
            # Pertanyaan:
            #   "siapa direktur tahun 2024"
            #
            # harus memprioritaskan chunk yang memuat identitas
            # direktur perusahaan.
            # ==========================================================

            if director_question:

                # ------------------------------------------------------
                # Pola jabatan direktur perusahaan
                # ------------------------------------------------------

                corporate_director_patterns = [
                    r"\bpt\.?\s+[a-z0-9 .,&'-]+\b.{0,120}\bdirektur\b",
                    r"\bdirektur\b.{0,120}\bpt\.?\s+[a-z0-9 .,&'-]+\b",
                    r"\brus['â€™`]?an\b.{0,60}\bdirektur\b",
                    r"\bdirektur\b.{0,60}\brus['â€™`]?an\b",
                    r"\bnama\s+(?:direktur|pimpinan)\b",
                    r"\b(?:direktur|pimpinan)\b\s*[:\-]\s*[A-Z][A-Za-z'â€™` -]{2,}",
                ]

                # ------------------------------------------------------
                # Pola yang HARUS dikecualikan
                # ------------------------------------------------------

                government_director_patterns = [
                    r"\bdirektur\s+jenderal\b",
                    r"\bdirektur\s+utama\s+pajak\b",
                    r"\bdirektorat\s+jenderal\b",
                ]

                corporate_match = any(
                    re.search(
                        pattern,
                        content_lower,
                        flags=re.IGNORECASE | re.DOTALL,
                    )
                    for pattern in corporate_director_patterns
                )


                government_match = any(
                    re.search(
                        pattern,
                        content_lower,
                        flags=re.IGNORECASE | re.DOTALL,
                    )
                    for pattern in government_director_patterns
                )

                # ------------------------------------------------------
                # Bonus hanya untuk direktur perusahaan
                # ------------------------------------------------------

                if corporate_match:

                    score += 1.50

                # ------------------------------------------------------
                # Jangan biarkan "Direktur Jenderal Pajak"
                # dianggap sebagai direktur perusahaan.
                # ------------------------------------------------------

                if government_match and not corporate_match:

                    score -= 0.80

                # ------------------------------------------------------
                # Dokumen perusahaan biasanya lebih cocok untuk
                # pertanyaan siapa direktur.
                # ------------------------------------------------------

                director_filename_terms = [
                    "lapkeu",
                    "laporan keuangan",
                    "daftar peredaran usaha",
                    "peredaran usaha",
                    "profil",
                    "akta",
                ]

                filename_match = sum(
                    1
                    for term in director_filename_terms
                    if term in filename_lower
                )

                score += min(
                    filename_match * 0.25,
                    0.75,
                )

                # ------------------------------------------------------
                # Dokumen transaksi / administrasi pajak bukan sumber
                # utama untuk pertanyaan jabatan.
                # ------------------------------------------------------

                transaction_filename_terms = [
                    "faktur",
                    "invoice",
                    "e-billing",
                    "ebilling",
                    "billing",
                    "bpe",
                    "tanda terima",
                    "bukti potong",
                ]

                transaction_match = sum(
                    1
                    for term in transaction_filename_terms
                    if term in filename_lower
                )

                if transaction_match:

                    score -= 0.40


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
            # TARGET FINANCIAL VALUE
            #
            # Untuk pertanyaan finansial tahunan, prioritaskan chunk
            # yang benar-benar memuat label angka yang ditanyakan.
            # ==========================================================

            if annual_financial_question:

                target_financial_terms = []

                if "pendapatan bersih" in question_lower:

                    target_financial_terms = [
                        "pendapatan bersih",
                        "pendapatan bersih setelah pajak",
                    ]

                elif "pendapatan kotor" in question_lower:

                    target_financial_terms = [
                        "pendapatan kotor",
                    ]

                elif "pendapatan" in question_lower:

                    target_financial_terms = [
                        "pendapatan proyek",
                        "pendapatan kotor",
                        "pendapatan bersih",
                    ]

                elif "laba bersih" in question_lower:

                    target_financial_terms = [
                        "laba bersih",
                        "pendapatan bersih",
                        "laba/rugi tahun berjalan",
                    ]

                elif "laba" in question_lower:

                    target_financial_terms = [
                        "laba tahun berjalan",
                        "laba bersih",
                        "laba/rugi tahun berjalan",
                    ]

                elif "omzet" in question_lower:

                    target_financial_terms = [
                        "omzet",
                        "total peredaran usaha",
                    ]

                elif "peredaran usaha" in question_lower:

                    target_financial_terms = [
                        "total peredaran usaha",
                        "peredaran usaha",
                    ]

                if target_financial_terms:

                    matched_target_terms = sum(
                        1
                        for term in target_financial_terms
                        if term in content_lower
                    )

                    score += min(
                        matched_target_terms * 0.50,
                        1.50,
                    )


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

        # Simpan score hasil reranking ke SearchResult
        # agar score yang di tampilkan pada Sources sesuai
        # dengan urutan ranking sebenarnya.
        for reranked_score, result in reranked:
            result.score = reranked_score



        # ==============================================================
        # 11A. STRICT ANNUAL FINANCIAL RETRIEVAL
        #
        # Untuk pertanyaan finansial tahunan, gunakan seluruh chunk dari
        # dokumen finansial pada tahun tersebut.
        #
        # Ini diperlukan karena:
        #
        # - label bisa berada di chunk A
        # - angka bisa berada di chunk B
        # - OCR PDF bisa menghasilkan angka yang salah
        # - Excel/XLSX biasanya memiliki angka yang lebih terstruktur
        #
        # Vector search tetap dipakai untuk pertanyaan lain.
        # ==============================================================

        if annual_financial_question:

            financial_results = (
                self._get_financial_document_chunks(
                    requested_year or ""
                )
            )

            if financial_results:


                # ------------------------------------------------------
                # Prioritas tipe file:
                #
                # XLSX / XLS
                # DOCX
                # PDF
                #
                # Tujuannya agar data tabel terstruktur lebih dahulu
                # digunakan dibanding OCR PDF.
                # ------------------------------------------------------

                def financial_file_priority(
                        result: SearchResult,
                ) -> tuple[int, int, float]:

                    metadata = (
                        result.metadata or {}
                    )

                    filename = self._get_filename(
                        metadata
                    ).lower()

                    extension = (
                        Path(filename).suffix.lower()
                    )

                    if extension == ".xlsx":
                        file_priority = 0

                    elif extension == ".xls":
                        file_priority = 1

                    elif extension == ".docx":
                        file_priority = 2

                    else:
                        file_priority = 3

                    return (
                        file_priority,
                        self._get_chunk_index(
                            result
                        ),
                        -float(
                            result.score
                        ),
                    )

                financial_results.sort(
                    key=financial_file_priority
                )

                # ------------------------------------------------------
                # Buat reranking dari seluruh chunk finansial.
                # ------------------------------------------------------

                reranked = []

                for result in financial_results:

                    score = float(
                        result.score
                    )

                    metadata = (
                        result.metadata or {}
                    )

                    filename = self._get_filename(
                        metadata
                    )

                    filename_lower = (
                        filename.lower()
                    )

                    content_lower = (
                        result.content or ""
                    ).lower()

                    # ----------------------------------------------
                    # FILE STRUCTURED DATA
                    # ----------------------------------------------

                    if filename_lower.endswith(
                        ".xlsx"
                    ):

                        score += 0.60

                    elif filename_lower.endswith(
                        ".xls"
                    ):

                        score += 0.50

                    elif filename_lower.endswith(
                        ".docx"
                    ):

                        score += 0.30

                    # ----------------------------------------------
                    # LABEL TARGET
                    # ----------------------------------------------

                    if "laba bersih" in question_lower:

                        if (
                            "laba bersih"
                            in content_lower
                        ):

                            score += 1.50

                        if (
                            "pendapatan bersih"
                            in content_lower
                        ):

                            score += 1.20

                        if (
                            "pendapatan bersih setelah pajak"
                            in content_lower
                        ):

                            score += 1.50

                        if (
                            "laba tahun"
                            in content_lower
                        ):

                            score += 1.20

                        if (
                            "laba/rugi tahun berjalan"
                            in content_lower
                        ):

                            score += 1.20

                        # ------------------------------------------
                        # Struktur tabel Excel
                        # ------------------------------------------

                        if "kolom 20" in content_lower:

                            score += 1.50

                        if "jumlah" in content_lower:

                            score += 1.00

                    elif (
                        "pendapatan bersih"
                        in question_lower
                    ):

                        if (
                            "pendapatan bersih"
                            in content_lower
                        ):

                            score += 1.50

                        if (
                            "kolom 20" in content_lower
                        ):

                            score += 1.00

                    elif (
                        "pendapatan kotor"
                        in question_lower
                    ):

                        if (
                            "pendapatan kotor"
                            in content_lower
                        ):

                            score += 1.50

                    elif (
                        "pendapatan"
                        in question_lower
                    ):

                        if (
                            "pendapatan proyek"
                            in content_lower
                        ):

                            score += 1.50

                        if (
                            "pendapatan kotor"
                            in content_lower
                        ):

                            score += 1.00

                    reranked.append(
                        (
                            score,
                            result,
                        )
                    )

                reranked.sort(
                    key=lambda item:
                    item[0],
                    reverse=True,
                )
                # Simpan score hasil reranking financial
                # agar score Sources konsisten dengan ranking.
                for reranked_score, result in reranked:
                    result.score = reranked_score

                raw_results = (
                    financial_results
                )

            else:

                raw_results = []

                reranked = []

        # ==============================================================
        # 12. MODE LOKASI
        #
        # Untuk pencarian lokasi, jangan gunakan hasil vector search.
        # Gunakan metadata + path secara langsung.
        # ==============================================================

        if location_question:

            location_category = (
                self._detect_location_category(
                    question_lower
                )
            )

            location_results = (
                self._find_location_documents(
                    requested_year,
                    location_category,
                )
            )

            if location_results:

                return self._build_location_context(
                    location_results,
                    top_k=len(
                        location_results
                    ),
                )

            # ----------------------------------------------------------
            # Fallback ke retrieval lama jika pencarian metadata
            # tidak menemukan hasil.
            # ----------------------------------------------------------

            selected_results: List[
                SearchResult
            ] = []

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

                normalized_path = (
                    file_path_lower
                    or filename_lower
                )

                if normalized_path in seen_documents:

                    continue

                seen_documents.add(
                    normalized_path
                )

                selected_results.append(
                    result
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
            or financial_question
        ):

            if annual_financial_question:

                max_chunks_per_document = 2

                target_results = max(
                    top_k,
                    5,
                )

            else:

                max_chunks_per_document = 3

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

        # Pertanyaan alamat bukan search-only.
        # Harus diteruskan ke deterministic address extraction.
        if RAGEngine._is_address_question(question_lower):
            return False

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

        is_location = RAGEngine._is_location_question(
            question_lower
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

        filename = cls._get_filename(
            metadata
        )

        if cls._contains_year(
                filename,
                year,
        ):

            return True

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
                "laporan keuangan",
                "lapkeu",
                "laporan laba rugi",
                "neraca",
                "laporan pendapatan",
            ]
        ):

            return "financial"

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
    # FINANCIAL QUESTION
    # ==================================================================

    @staticmethod
    def _is_financial_question(
            question_lower: str,
    ) -> bool:

        financial_terms = [
            "pendapatan",
            "pendapatan kotor",
            "pendapatan bersih",
            "omzet",
            "peredaran usaha",
            "laba",
            "laba bersih",
            "laba rugi",
            "rugi",
            "penjualan",
            "keuntungan",
            "biaya",
            "beban",
            "hpp",
            "harga pokok penjualan",
        ]

        return any(
            term in question_lower
            for term in financial_terms
        )

    # ==================================================================
    # ANNUAL FINANCIAL QUESTION
    # ==================================================================

    @staticmethod
    def _is_annual_financial_question(
            question_lower: str,
            requested_year: str | None,
    ) -> bool:

        if not requested_year:

            return False

        annual_financial_terms = [
            "pendapatan",
            "pendapatan kotor",
            "pendapatan bersih",
            "omzet",
            "laba",
            "laba bersih",
            "laba rugi",
            "rugi",
            "peredaran usaha",
            "laporan keuangan",
            "lapkeu",
            "neraca",
        ]

        return any(
            term in question_lower
            for term in annual_financial_terms
        )

    # ==================================================================
    # FINANCIAL ANSWER LABEL
    # ==================================================================

    @staticmethod
    def _get_financial_answer_label(
            question_lower: str,
    ) -> str:

        if "pendapatan bersih" in question_lower:

            return "Pendapatan bersih perusahaan"

        if "pendapatan kotor" in question_lower:

            return "Pendapatan kotor perusahaan"

        if "pendapatan" in question_lower:


            return "Pendapatan perusahaan"

        if "omzet" in question_lower:

            return "Omzet perusahaan"

        if "peredaran usaha" in question_lower:

            return "Peredaran usaha perusahaan"

        if "laba bersih" in question_lower:

            return "Laba bersih perusahaan"

        if "laba" in question_lower:

            return "Laba perusahaan"

        if "rugi" in question_lower:

            return "Rugi perusahaan"

        if "hpp" in question_lower:

            return "HPP perusahaan"

        return "Nilai keuangan perusahaan"

    # ==================================================================
    # EXTRACT ANNUAL FINANCIAL VALUE
    # ==================================================================

    @staticmethod
    def _extract_annual_financial_value(
            chunks: List[SourceChunk],
            question_lower: str,
    ) -> str | None:

        if not chunks:
            return None

        # --------------------------------------------------------------
        # Kumpulkan seluruh content
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

        # ==============================================================
        # HELPER NORMALISASI ANGKA
        # ==============================================================

        def normalize_number(
                value: str,
        ) -> str | None:

            value = str(
                value or ""
            ).strip()

            if not value:
                return None

            # Hilangkan prefix Rp
            value = re.sub(
                r"(?i)^rp\.?\s*",
                "",
                value,
            ).strip()

            # Jangan menerima tahun sebagai nilai
            if re.fullmatch(
                r"20\d{2}",
                value,
            ):
                return None

            # Harus memiliki digit
            if not re.search(
                r"\d",
                value,
            ):
                return None

            return value.rstrip(
                ".,"
            )

        # ==============================================================
        # DETEKSI JENIS PERTANYAAN
        # ==============================================================

        is_laba_bersih = (
            "laba bersih"
            in question_lower
        )

        is_pendapatan_bersih = (
            "pendapatan bersih"
            in question_lower
        )

        is_pendapatan_kotor = (
            "pendapatan kotor"
            in question_lower
        )

        is_pendapatan = (
            "pendapatan"
            in question_lower
            and not is_pendapatan_bersih
            and not is_pendapatan_kotor
        )

        is_peredaran = (
            "peredaran usaha"
            in question_lower
            or "omzet"
            in question_lower
        )



        # ==============================================================
        # 1. LABA BERSIH
        # ==============================================================
        #
        # Contoh data:
        #
        # Laba/Rugi Tahun Berjalan |
        # Kolom 4: 51093215.45000005
        #
        # IMPORTANT:
        # Kita harus melewati "Kolom 4:" dan mengambil angka setelahnya.
        # ==============================================================

        if is_laba_bersih:

            patterns = [

                # Laba/Rugi Tahun Berjalan | Kolom 4: VALUE
                r"laba\s*/?\s*rugi\s+tahun\s+berjalan"
                r"\s*(?:\|\s*)?"
                r"kolom\s*\d+\s*:\s*"
                r"([\d][\d.,]*)",

                # Laba Tahun 2024 | Kolom X: VALUE
                r"laba\s+tahun\s+20\d{2}"
                r"\s*(?:\|\s*)?"
                r"kolom\s*\d+\s*:\s*"
                r"([\d][\d.,]*)",

                # Laba Bersih | Kolom X: VALUE
                r"laba\s+bersih"
                r"\s*(?:\|\s*)?"
                r"kolom\s*\d+\s*:\s*"
                r"([\d][\d.,]*)",

                # Pendapatan Bersih setelah Pajak | Kolom X: VALUE
                r"pendapatan\s+bersih\s+setelah\s+pajak"
                r"\s*(?:\|\s*)?"
                r"kolom\s*\d+\s*:\s*"
                r"([\d][\d.,]*)",

                # Pendapatan Bersih | Kolom X: VALUE
                r"pendapatan\s+bersih"
                r"\s*(?:\|\s*)?"
                r"kolom\s*\d+\s*:\s*"
                r"([\d][\d.,]*)",
            ]

            for content in texts:

                normalized = re.sub(
                    r"\s+",
                    " ",
                    content,
                )

                for pattern in patterns:

                    match = re.search(
                        pattern,
                        normalized,
                        flags=re.IGNORECASE,
                    )

                    if match:

                        value = normalize_number(
                            match.group(1)
                        )

                        if value:

                            return value

            # ----------------------------------------------------------
            # Fallback Excel:
            #
            # Kolom 20: 51093215.45000002
            # ----------------------------------------------------------

            for content in texts:

                normalized = re.sub(
                    r"\s+",
                    " ",
                    content,
                )

                matches = re.findall(
                    r"kolom\s*20\s*:\s*"
                    r"([\d][\d.,]*)",
                    normalized,
                    flags=re.IGNORECASE,
                )

                for value in matches:

                    value = normalize_number(
                        value
                    )

                    if value:

                        return value

        # ==============================================================
        # 2. PENDAPATAN BERSIH
        # ==============================================================

        if is_pendapatan_bersih:

            patterns = [

                r"pendapatan\s+bersih\s+setelah\s+pajak"
                r"\s*(?:\|\s*)?"
                r"kolom\s*\d+\s*:\s*"
                r"([\d][\d.,]*)",

                r"pendapatan\s+bersih"
                r"\s*(?:\|\s*)?"
                r"kolom\s*\d+\s*:\s*"
                r"([\d][\d.,]*)",
            ]

            for content in texts:

                normalized = re.sub(
                    r"\s+",
                    " ",
                    content,
                )

                for pattern in patterns:

                    match = re.search(
                        pattern,
                        normalized,
                        flags=re.IGNORECASE,
                    )

                    if match:

                        value = normalize_number(
                            match.group(1)
                        )

                        if value:

                            return value

            # Excel Kolom 20
            for content in texts:

                normalized = re.sub(
                    r"\s+",
                    " ",
                    content,
                )

                matches = re.findall(
                    r"kolom\s*20\s*:\s*"
                    r"([\d][\d.,]*)",
                    normalized,
                    flags=re.IGNORECASE,
                )

                for value in matches:


                    value = normalize_number(
                        value
                    )

                    if value:

                        return value

        # ==============================================================
        # 3. PENDAPATAN KOTOR
        # ==============================================================

        if is_pendapatan_kotor:

            for content in texts:

                normalized = re.sub(
                    r"\s+",
                    " ",
                    content,
                )

                match = re.search(
                    r"pendapatan\s+kotor"
                    r"\s*(?:\|\s*)?"
                    r"kolom\s*\d+\s*:\s*"
                    r"([\d][\d.,]*)",
                    normalized,
                    flags=re.IGNORECASE,
                )

                if match:

                    value = normalize_number(
                        match.group(1)
                    )

                    if value:

                        return value

        # ==============================================================
        # 4. PENDAPATAN
        # ==============================================================

        if is_pendapatan:

            patterns = [

                r"pendapatan\s+proyek"
                r"\s*(?:\|\s*)?"
                r"kolom\s*\d+\s*:\s*"
                r"([\d][\d.,]*)",

                r"pendapatan"
                r"(?!\s+sebelum)"
                r"\s*(?:\|\s*)?"
                r"kolom\s*\d+\s*:\s*"
                r"([\d][\d.,]*)",
            ]

            for content in texts:

                normalized = re.sub(
                    r"\s+",
                    " ",
                    content,
                )

                for pattern in patterns:

                    match = re.search(
                        pattern,
                        normalized,
                        flags=re.IGNORECASE,
                    )

                    if match:

                        value = normalize_number(
                            match.group(1)
                        )

                        if value:

                            return value

        # ==============================================================
        # 5. PEREDARAN USAHA / OMZET
        # ==============================================================

        if is_peredaran:

            patterns = [

                # Total Peredaran Usaha Tahun 2024 | Kolom 5: VALUE
                r"total\s+peredaran\s+usaha"
                r"(?:\s+tahun\s+20\d{2})?"
                r"\s*(?:\|\s*)?"
                r"kolom\s*\d+\s*:\s*"
                r"([\d][\d.,]*)",

                # Peredaran Usaha Tahun 2024 | Kolom 5: VALUE
                r"peredaran\s+usaha"
                r"(?:\s+tahun\s+20\d{2})?"
                r"\s*(?:\|\s*)?"
                r"kolom\s*\d+\s*:\s*"
                r"([\d][\d.,]*)",
            ]

            # Prioritaskan pola yang lebih spesifik pada seluruh chunk.
            # "Total Peredaran Usaha" harus dicari di semua chunk sebelum
            # fallback ke label "Peredaran Usaha" yang lebih umum.
            for pattern in patterns:

                for content in texts:

                    normalized = re.sub(
                        r"\s+",
                        " ",
                        content,
                    )

                    match = re.search(
                        pattern,
                        normalized,
                        flags=re.IGNORECASE,
                    )

                    if match:

                        value = normalize_number(
                            match.group(1)
                        )

                        if value:

                            return value

        # ==============================================================
        # 6. FALLBACK UMUM
        #
        # Tidak boleh mengambil "Kolom 4", "Kolom 20", "Baris 19", dll.
        # ==============================================================

        label_patterns = []

        if is_laba_bersih:

            label_patterns = [
                r"laba\s*/?\s*rugi\s+tahun\s+berjalan",
                r"laba\s+tahun\s+20\d{2}",
                r"laba\s+bersih",
                r"pendapatan\s+bersih\s+setelah\s+pajak",
                r"pendapatan\s+bersih",
            ]

        elif is_pendapatan_bersih:

            label_patterns = [
                r"pendapatan\s+bersih\s+setelah\s+pajak",
                r"pendapatan\s+bersih",
            ]

        elif is_pendapatan_kotor:

            label_patterns = [
                r"pendapatan\s+kotor",
            ]

        elif is_pendapatan:

            label_patterns = [
                r"pendapatan\s+proyek",
                r"pendapatan\s+kotor",
            ]

        elif is_peredaran:

            label_patterns = [
                r"total\s+peredaran\s+usaha",
                r"peredaran\s+usaha",
            ]

        # Prioritaskan label yang paling spesifik pada seluruh chunk
        # sebelum mencoba label yang lebih umum. Ini penting untuk kasus
        # seperti "Total Peredaran Usaha" yang dapat berada pada chunk
        # setelah baris-baris komponen "Peredaran Usaha".
        for label_pattern in label_patterns:

            for content in texts:

                normalized = re.sub(
                    r"\s+",
                    " ",
                    content,
                )

                match = re.search(
                    label_pattern,
                    normalized,
                    flags=re.IGNORECASE,
                )

                if not match:
                    continue

                section = normalized[
                    match.end():
                    min(
                        match.end() + 150,
                        len(normalized),
                    )
                ]

                # ------------------------------------------------------
                # Cari angka tetapi lewati:
                #
                # Kolom 4
                # Kolom 20
                # Baris 19
                # Tahun 2024
                # ------------------------------------------------------

                numbers = re.findall(
                    r"(?<!\d)"
                    r"(?!20\d{2}(?!\d))"
                    r"\d{1,3}"
                    r"(?:[.,]\d{3})*"
                    r"(?:[.,]\d+)?"
                    r"(?!\d)",
                    section,
                )

                for number in numbers:

                    # Jangan ambil nomor kolom/baris
                    before = section[
                        max(
                            0,
                            section.find(number) - 10,
                        ):
                        section.find(number)
                    ].lower()

                    if (
                        "kolom" in before
                        or "baris" in before
                    ):
                        continue

                    value = normalize_number(
                        number
                    )

                    if value:

                        return value

        return None


    # ==================================================================
    # FORMAT FINANCIAL VALUE
    # ==================================================================

    @staticmethod
    def _format_financial_value(
            value: str | None,
    ) -> str:

        value = str(
            value or ""
        ).strip()

        if not value:
            return ""

        # Hilangkan prefix Rp
        value = re.sub(
            r"(?i)^rp\.?\s*",
            "",
            value,
        ).strip()

        # --------------------------------------------------------------
        # Jika sudah format Indonesia:
        #
        # 56.000.000
        # 1.248.761.500
        # --------------------------------------------------------------

        if (
            value.count(".") >= 2
            and all(
                part.isdigit()
                for part in value.split(".")
            )
            and all(
                len(part) == 3
                for part in value.split(".")[1:]
            )
        ):

            return f"Rp {value}"

        # --------------------------------------------------------------
        # Normalisasi angka floating point:
        #
        # 51093215.45000005
        # 490158531.30262494

        # --------------------------------------------------------------

        # Jika menggunakan koma sebagai desimal
        if (
            "," in value
            and "." not in value
        ):

            value = value.replace(
                ",",
                ".",
            )

        try:

            numeric_value = float(
                value
            )

            # ----------------------------------------------------------
            # Nilai bulat
            # ----------------------------------------------------------

            if numeric_value.is_integer():

                grouped = (
                    f"{int(numeric_value):,}"
                    .replace(
                        ",",
                        ".",
                    )
                )

                return f"Rp {grouped}"

            # ----------------------------------------------------------
            # Maksimal 2 angka desimal
            # ----------------------------------------------------------

            formatted = (
                f"{numeric_value:,.2f}"
            )

            # Format US:
            #
            # 51,093,215.45
            #
            # menjadi:
            #
            # 51.093.215,45
            # ----------------------------------------------------------

            formatted = formatted.replace(
                ",",
                "X",
            )

            formatted = formatted.replace(
                ".",
                ",",
            )

            formatted = formatted.replace(
                "X",
                ".",
            )

            return f"Rp {formatted}"

        except (
                ValueError,
                TypeError,
        ):

            return f"Rp {value}"


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

        texts = []

        for chunk in chunks:

            content = (
                chunk.chunk_text or ""
            ).strip()

            if content:

                texts.append(
                    content
                )

        if not texts:

            return None

        combined = "\n".join(
            texts
        )

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

            values = re.findall(
                r"Rp\s*([\d][\d.,]*)",
                section,
                flags=re.IGNORECASE,
            )

            if len(values) >= 2:


                wp_value = values[0]
                pemeriksa_value = values[1]

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
    # ALAMAT PERUSAHAAN
    # ==================================================================

    @staticmethod
    def _is_address_question(question_lower: str) -> bool:
        address_terms = [
            "alamat",
            "alamat perusahaan",
            "alamat pt",
            "kantor pt",
            "kantor perusahaan",
        ]

        return any(
            term in question_lower
            for term in address_terms
        )

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

    def _extract_company_address_candidates(
        self,
        question_lower: str,
        requested_year: str | None = None,
    ) -> list[tuple[str, str, SourceChunk]]:
        """
        Mengambil seluruh kandidat alamat yang relevan.

        Return:
            list of (address, document_type, source)
        """

        try:
            results = self.indexer.store.collection.get(
                include=["documents", "metadatas"]
            )
        except Exception:
            return []

        documents = (
            results.get("documents", []) or []
        )
        metadatas = (
            results.get("metadatas", []) or []
        )

        # ----------------------------------------------------------
        # OBJECT DARI PERTANYAAN
        # ----------------------------------------------------------

        entity_match = re.search(
            r"\b(pt|cv|firma|koperasi|yayasan)\.?\s+"
            r"([a-z0-9][a-z0-9 .,&'-]{2,})",
            question_lower,
            flags=re.IGNORECASE,
        )

        entity_name = ""

        if entity_match:
            entity_name = (
                f"{entity_match.group(1)} "
                f"{entity_match.group(2)}"
            ).strip()

            entity_name = re.split(
                r"\s+(?:tahun|pada|menurut|dalam|di|untuk|yang)\b",
                entity_name,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip()

        candidates = []

        for content, metadata in zip(
            documents,
            metadatas,
        ):
            content = content or ""
            metadata = metadata or {}

            content_lower = content.lower()

            # Object harus cocok.
            if entity_name and entity_name not in content_lower:
                continue

            if "alamat" not in content_lower:
                continue

            # ------------------------------------------------------
            # FILTER TAHUN BERDASARKAN METADATA
            # ------------------------------------------------------

            if requested_year:
                metadata_year = str(
                    metadata.get("year", "")
                ).strip()

                if metadata_year != requested_year:
                    continue

            # ------------------------------------------------------
            # JENIS DOKUMEN
            # ------------------------------------------------------

            filename = str(
                metadata.get("filename", "")
            ).strip()

            file_path = str(
                metadata.get("file_path", "")
            ).strip()

            file_context = (
                f"{filename} {file_path}"
            ).lower()

            document_type = "dokumen"

            if any(
                term in file_context
                for term in [
                    "lapkeu",
                    "laporan keuangan",
                    "neraca",
                    "rugi laba",
                ]
            ):
                document_type = "laporan keuangan"

            elif any(
                term in file_context
                for term in [
                    "faktur",
                    "faktur pajak",
                ]
            ):
                document_type = "faktur pajak"

            elif "invoice" in file_context:
                document_type = "invoice"

            elif any(
                term in file_context
                for term in [
                    "spt",
                    "surat pemberitahuan",
                ]
            ):
                document_type = "SPT"

            # ------------------------------------------------------
            # EKSTRAKSI ALAMAT
            # ------------------------------------------------------

            match = re.search(
                r"alamat\s*:\s*(.*?)(?="
                r"\s+npwp\s*:"
                r"|\s+nama\s*:"
                r"|\s+daftar\s+peredaran\s+usaha\b"
                r"|\s+baris\s+\d+\s*:"
                r"|\s+kolom\s+\d+\s*:"
                r"|\s+\d+\s*\|"
                r"|$"
                r")",
                content,
                flags=re.IGNORECASE,
            )

            if not match:
                continue

            address = match.group(1).strip()

            if not address:
                continue

            # ------------------------------------------------------
            # CLEANUP OCR / TABEL
            # ------------------------------------------------------

            address = re.sub(
                r"\s+Baris\s+\d+\s*:.*$",
                "",
                address,
                flags=re.IGNORECASE,
            )

            address = re.sub(
                r"\s+Kolom\s+\d+\s*:.*$",
                "",
                address,
                flags=re.IGNORECASE,
            )

            address = re.sub(
                r"\s+\d+\s*\|.*$",
                "",
                address,
                flags=re.IGNORECASE,
            )

            address = re.sub(
                r"\s{2,}",
                " ",
                address,
            ).strip(" ,;|")

            if not address:
                continue

            source = SourceChunk(
                filename=filename,
                chunk_text=content,
                score=1.0,
                file_path=file_path,
            )

            candidates.append(
                (
                    address,
                    document_type,
                    source,
                )
            )

        return candidates


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
    # DETECT LOCATION CATEGORY
    # ==================================================================

    @staticmethod
    def _detect_location_category(
            question_lower: str,
    ) -> str | None:

        # --------------------------------------------------------------
        # LAPORAN KEUANGAN
        # --------------------------------------------------------------

        if any(
            term in question_lower
            for term in [
                "laporan keuangan",
                "lapkeu",
                "pendapatan perusahaan",
                "neraca perusahaan",
                "laba perusahaan",
                "rugi perusahaan",
            ]
        ):

            return "financial"

        # --------------------------------------------------------------
        # SPT
        # --------------------------------------------------------------

        if any(
            term in question_lower
            for term in [
                "spt",
                "surat pemberitahuan",
            ]
        ):

            return "spt"

        # --------------------------------------------------------------
        # FAKTUR
        # --------------------------------------------------------------

        if any(
            term in question_lower
            for term in [
                "faktur",
                "faktur pajak",
            ]
        ):

            return "faktur"

        # --------------------------------------------------------------
        # INVOICE
        # --------------------------------------------------------------

        if "invoice" in question_lower:

            return "invoice"

        return None

    # ==================================================================
    # FIND LOCATION DOCUMENTS
    #
    # Pencarian lokasi menggunakan metadata Chroma secara langsung.
    # Tidak menggunakan embedding.
    #
    # Untuk laporan keuangan tahunan:
    # jika folder "Laporan Keuangan\<tahun>" tersedia, hanya gunakan
    # file dari folder tersebut.

    # ==================================================================

    def _find_location_documents(
            self,
            year: str | None,
            category: str | None,
    ) -> List[SearchResult]:

        try:

            collection = (
                self.indexer.store.collection
            )

        except Exception:

            return []

        # --------------------------------------------------------------
        # Ambil metadata + dokumen.
        # --------------------------------------------------------------

        try:

            if year:

                results = collection.get(
                    where={
                        "year": year,
                    },
                    include=[
                        "documents",
                        "metadatas",
                    ],
                )

            else:

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

        # --------------------------------------------------------------
        # Kandidat hasil.
        # --------------------------------------------------------------

        candidates: List[
            SearchResult
        ] = []

        for chunk_id, doc, meta in zip(
                ids,
                docs,
                metas,
        ):

            metadata = (
                meta or {}
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
            ).strip()

            file_path_lower = (
                file_path.lower()
            )

            content_lower = (
                doc or ""
            ).lower()

            # ==========================================================
            # YEAR
            # ==========================================================

            if year:

                metadata_year = str(
                    metadata.get(
                        "year",
                        "",
                    )
                ).strip()

                if metadata_year != year:

                    continue

            # ==========================================================
            # CATEGORY
            # ==========================================================

            if category == "financial":

                financial_filename = any(
                    term in filename_lower
                    for term in [
                        "lapkeu",
                        "laporan keuangan",
                        "pendapatan",
                        "neraca",
                        "rugi",
                        "laba",
                    ]
                )

                financial_path = (
                    "laporan keuangan"
                    in file_path_lower
                )

                financial_content = any(
                    term in content_lower
                    for term in [
                        "laporan keuangan",
                        "pendapatan proyek",
                        "pendapatan kotor",
                        "pendapatan bersih",
                        "laba/rugi",
                        "laba tahun",
                        "neraca keuangan",
                        "peredaran usaha",
                    ]
                )

                if not (
                    financial_filename
                    or financial_path
                    or financial_content
                ):

                    continue

                # ------------------------------------------------------
                # File transaksi bukan laporan keuangan.
                # ------------------------------------------------------

                is_transaction = any(
                    term in filename_lower
                    for term in [
                        "faktur",
                        "invoice",
                        "e-billing",
                        "ebilling",
                        "billing",
                        "bukti potong",
                        "bpe",
                        "tanda terima",
                    ]
                )

                if (
                    is_transaction
                    and not financial_filename
                    and not financial_path
                ):

                    continue

            elif category == "spt":

                is_spt = (
                    self._is_spt_document(
                        filename_lower,
                        file_path_lower,
                        content_lower,
                    )
                )

                is_bpe = (
                    self._is_bpe_document(
                        filename_lower,
                        file_path_lower,
                        content_lower,
                    )
                )

                if not is_spt or is_bpe:

                    continue

            elif category == "faktur":

                if (
                    "faktur"
                    not in filename_lower
                    and
                    "faktur pajak"
                    not in content_lower
                ):

                    continue

            elif category == "invoice":

                if "invoice" not in filename_lower:

                    continue

            # ----------------------------------------------------------
            # Simpan kandidat.
            # ----------------------------------------------------------

            candidates.append(
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

        if not candidates:

            return []

        # ==============================================================
        # PRIORITAS FOLDER LAPORAN KEUANGAN
        # ==============================================================

        if (
            category == "financial"
            and year
        ):

            preferred_marker = (
                f"\\laporan keuangan\\{year}\\"
            )

            preferred_marker_alt = (
                f"/laporan keuangan/{year}/"
            )

            preferred_candidates = []

            for result in candidates:

                metadata = (
                    result.metadata or {}
                )

                path = str(
                    metadata.get(
                        "file_path",
                        "",
                    )
                ).lower()

                if (
                    preferred_marker in path
                    or
                    preferred_marker_alt in path
                ):

                    preferred_candidates.append(
                        result
                    )

            # ----------------------------------------------------------
            # Jika folder utama tersedia, gunakan hanya folder utama.
            # ----------------------------------------------------------


            if preferred_candidates:

                candidates = (
                    preferred_candidates
                )

        # ==============================================================
        # DEDUP BERDASARKAN PATH FILE
        # ==============================================================

        matched: List[
            SearchResult
        ] = []

        seen_paths = set()

        for result in candidates:

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
            ).strip()

            normalized_path = (
                file_path
                .replace(
                    "/",
                    "\\",
                )
                .rstrip(
                    "\\"
                )
                .lower()
            )

            source_key = (
                normalized_path
                or filename.lower()
            )

            if source_key in seen_paths:

                continue

            seen_paths.add(
                source_key
            )

            matched.append(
                result
            )

        # ==============================================================
        # SORT
        # ==============================================================

        if category == "financial":

            def financial_location_sort_key(
                    result: SearchResult,
            ) -> tuple:

                metadata = (
                    result.metadata or {}
                )

                filename = self._get_filename(
                    metadata
                ).lower()

                file_path = str(
                    metadata.get(
                        "file_path",
                        "",
                    )
                ).lower()

                # ----------------------------------------------
                # Prioritas jenis file
                # ----------------------------------------------

                if filename.startswith(
                    "lapkeu"
                ):

                    filename_priority = 0

                elif "laporan keuangan" in filename:

                    filename_priority = 1

                elif "neraca" in filename:

                    filename_priority = 2

                elif "pendapatan" in filename:

                    filename_priority = 3

                else:

                    filename_priority = 4

                return (
                    filename_priority,
                    filename,
                    file_path,
                )

            matched.sort(
                key=financial_location_sort_key
            )

        else:

            matched.sort(
                key=lambda result:
                self._get_filename(
                    result.metadata
                ).lower()
            )

        return matched


    # ==================================================================
    # FIND PEREDARAN USAHA COMPARISON CHUNKS
    #
    # Digunakan khusus untuk pertanyaan:
    #
    # "Berapa peredaran usaha menurut WP dan menurut Pemeriksa?"
    #
    # Karena data pembanding biasanya berada di dokumen pemeriksaan/SPT,
    # jangan bergantung pada vector similarity.
    # ==================================================================

    def _find_peredaran_comparison_chunks(
            self,
            year: str | None,
    ) -> List[SearchResult]:

        if not year:
            return []

        try:

            collection = (
                self.indexer.store.collection
            )

            results = collection.get(
                where={
                    "year": year,
                },
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

        matched = []

        seen_chunks = set()

        for chunk_id, doc, meta in zip(
                ids,
                docs,
                metas,
        ):

            metadata = (
                meta or {}
            )

            content = (
                doc or ""
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

            # ----------------------------------------------------------
            # Harus merupakan bagian dari pembahasan peredaran usaha
            # ----------------------------------------------------------

            has_peredaran = (
                "peredaran usaha"
                in content_lower
            )

            has_wp = (
                "menurut wp/spt"
                in content_lower
                or
                "menurut wajib pajak"
                in content_lower
                or
                "menurut wp"
                in content_lower
            )

            has_pemeriksa = (
                "menurut pemeriksa"
                in content_lower
            )

            # ----------------------------------------------------------
            # Hindari chunk transaksi biasa.
            # ----------------------------------------------------------

            is_transaction = any(
                term in filename_lower
                for term in [
                    "faktur",
                    "invoice",
                    "e-billing",
                    "ebilling",
                    "bpe",
                    "bukti potong",
                ]
            )

            if (
                has_peredaran
                and has_wp
                and has_pemeriksa
                and not is_transaction
            ):

                if chunk_id in seen_chunks:
                    continue

                seen_chunks.add(
                    chunk_id
                )

                matched.append(
                    SearchResult(
                        chunk_id=chunk_id,
                        document_id=metadata.get(
                            "document_id",
                            "",
                        ),
                        content=content,
                        score=1.0,
                        metadata=metadata,
                    )
                )

        # --------------------------------------------------------------
        # Jika tabel header dan nilai berada di chunk berdekatan,
        # ambil juga chunk sekitar target.
        # --------------------------------------------------------------

        if not matched:
            return []

        # --------------------------------------------------------------
        # Ambil seluruh chunk dari dokumen yang cocok agar nilai
        # tidak terpisah dari header.
        # --------------------------------------------------------------


        document_ids = {
            str(
                result.metadata.get(
                    "document_id",
                    "",
                )
            )
            for result in matched
            if result.metadata.get(
                "document_id",
                "",
            )
        }

        expanded = []

        for chunk_id, doc, meta in zip(
                ids,
                docs,
                metas,
        ):

            metadata = (
                meta or {}
            )

            document_id = str(
                metadata.get(
                    "document_id",
                    "",
                )
            )

            if document_id not in document_ids:
                continue

            expanded.append(
                SearchResult(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    content=doc or "",
                    score=1.0,
                    metadata=metadata,
                )
            )

        expanded.sort(
            key=lambda result: (
                str(
                    result.metadata.get(
                        "document_id",
                        "",
                    )
                ),
                self._get_chunk_index(
                    result
                ),
            )
        )

        return expanded


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

        seen_paths = set()

        for result in results:

            metadata = (
                result.metadata or {}
            )

            file_path = str(
                metadata.get(
                    "file_path",
                    "",
                )
            ).strip()

            filename = self._get_filename(
                metadata
            )

            source_key = (
                file_path
                .replace(
                    "/",
                    "\\",
                )
                .rstrip(
                    "\\",
                )
                .lower()
                if file_path
                else filename.lower()
            )

            if source_key in seen_paths:

                continue

            seen_paths.add(
                source_key
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

        # --------------------------------------------------------------
        # PILIH CHUNK
        #
        # Semua chunk yang relevan tetap boleh masuk context.
        # --------------------------------------------------------------

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

        # --------------------------------------------------------------
        # BUILD SOURCE + CONTEXT
        # --------------------------------------------------------------

        source_chunks: List[
            SourceChunk
        ] = []

        context_parts: List[str] = []

        # --------------------------------------------------------------
        # Source display menggunakan file_path.
        #
        # Jadi:
        #
        # chunk 1 -> file A
        # chunk 2 -> file A
        #
        # tetap menghasilkan:
        #
        # [1] file A
        #
        # bukan:
        #
        # [1] file A
        # [2] file A
        # --------------------------------------------------------------

        seen_source_paths = set()

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
            ).strip()

            document_id = str(
                metadata.get(
                    "document_id",
                    "",
                )
            ).strip()

            year = str(
                metadata.get(
                    "year",
                    "",
                )
            ).strip()

            if not filename:
                filename = "unknown"

            # ==========================================================
            # CONTEXT
            #
            # Semua chunk tetap dikirim ke LLM.
            # ==========================================================

            context_parts.append(
                f"[Source: {filename}]\n"
                f"[File Path: {file_path}]\n"
                f"[Document ID: {document_id}]\n"

                f"[Tahun: {year}]\n"
                f"{result.content or ''}"
            )

            # ==========================================================
            # SOURCE DISPLAY
            #
            # Gunakan path sebagai identitas utama.
            # ==========================================================

            source_key = (
                file_path.replace(
                    "/",
                    "\\",
                ).rstrip(
                    "\\"
                ).lower()
                if file_path
                else filename.lower()
            )

            if source_key in seen_source_paths:

                continue

            seen_source_paths.add(
                source_key
            )

            source_chunks.append(
                SourceChunk(
                    filename=filename,
                    chunk_text=(
                        result.content or ""
                    ),
                    score=float(
                        result.score
                    ),
                    file_path=file_path,
                )
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

        if "spt" in filename_lower:

            return True

        if "\\spt " in file_path_lower:

            return True

        if "/spt " in file_path_lower:

            return True

        if "spt 202" in file_path_lower:

            return True

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

            score += float(
                result.score
            )

            matched_terms = sum(
                1
                for term in question_terms
                if term in content_lower
            )

            score += min(
                matched_terms * 0.15,
                1.00,
            )

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
    # GET FINANCIAL DOCUMENT CHUNKS
    #
    # Untuk pertanyaan finansial tahunan, retrieval vector biasa tidak
    # cukup karena chunk yang memuat label dan angka bisa terpencar.
    #
    # Fungsi ini mengambil seluruh chunk dari dokumen finansial yang
    # sesuai dengan tahun pertanyaan.
    # ==================================================================

    def _get_financial_document_chunks(
            self,
            year: str,
    ) -> List[SearchResult]:

        if not year:
            return []

        try:

            collection = (
                self.indexer.store.collection
            )

            results = collection.get(
                where={
                    "year": year,
                },
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

        financial_filename_terms = [
            "pendapatan",
            "lapkeu",
            "laporan keuangan",
            "neraca",
            "rugi",
            "laba",
        ]

        transaction_filename_terms = [
            "faktur",
            "invoice",
            "e-billing",
            "ebilling",
            "billing",
            "bukti potong",
            "bpe",
            "tanda terima",
        ]

        for (
                chunk_id,
                doc,
                meta,
        ) in zip(
            ids,
            docs,
            metas,
        ):

            metadata = (
                meta or {}
            )

            filename = self._get_filename(
                metadata
            )

            filename_lower = (
                filename.lower()
            )

            content_lower = (
                doc or ""
            ).lower()

            financial_filename_match = any(
                term in filename_lower
                for term in financial_filename_terms
            )

            transaction_filename_match = any(
                term in filename_lower
                for term in transaction_filename_terms
            )

            financial_content_match = any(
                term in content_lower
                for term in [
                    "pendapatan proyek",
                    "pendapatan kotor",
                    "pendapatan bersih",
                    "pendapatan bersih setelah pajak",
                    "laba bersih",
                    "laba tahun",
                    "laba/rugi tahun berjalan",
                    "peredaran usaha",
                    "total peredaran usaha",
                    "laporan keuangan",
                    "biaya pokok penjualan",
                ]
            )

            is_financial_document = (
                financial_filename_match
                or financial_content_match
            )

            is_transaction_document = (
                transaction_filename_match
                and not financial_filename_match
            )

            if (
                is_financial_document
                and not is_transaction_document
            ):

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

        # --------------------------------------------------------------
        # Urutkan berdasarkan document + chunk index
        # --------------------------------------------------------------

        output.sort(
            key=lambda result: (
                str(
                    result.metadata.get(
                        "filename",
                        "",
                    )
                ).lower(),
                self._get_chunk_index(
                    result

                ),
            )
        )

        return output


    def _expand_table_chunks(
            self,
            all_results: List[SearchResult],
            ranked_results: List[SearchResult],
            question_lower: str,
            radius: int = 2,
    ) -> List[SearchResult]:

        if not all_results:

            return []

        sorted_results = sorted(
            all_results,
            key=lambda result:
            self._get_chunk_index(result),
        )

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

        expanded_indexes = set()

        for target_index in target_indexes:

            for offset in range(
                    -radius,
                    radius + 1,
            ):

                expanded_indexes.add(
                    target_index + offset
                )

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

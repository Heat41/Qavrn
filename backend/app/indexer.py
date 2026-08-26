from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .chunker import TextChunker
from .config import Settings, settings as default_settings
from .embedder import Embedder
from .parser import DocumentParser
from .store import VectorStore


logger = logging.getLogger(__name__)


@dataclass
class IndexStats:
    documents: int
    chunks: int
    storage_bytes: int

    @property
    def storage_mb(self) -> float:
        return self.storage_bytes / (1024 * 1024)


class Indexer:
    """Orchestrates parsing → metadata → chunking → embedding → storage."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or default_settings

        self.parser = DocumentParser()

        self.chunker = TextChunker(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

        self.embedder = Embedder(
            model_name=self.settings.embedding_model
        )

        self.store = VectorStore(
            persist_dir=self.settings.chroma_persist_dir
        )

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def scan_folder(self, path: str | Path) -> List[Path]:
        """Return all files with supported extensions under `path`."""

        root = Path(path).resolve()

        if not root.exists():
            raise FileNotFoundError(
                f"Folder not found: {root}"
            )

        found: List[Path] = []

        exts = set(self.settings.supported_extensions)

        for dirpath, _dirs, files in os.walk(root):
            for fname in files:
                fp = Path(dirpath) / fname

                if fp.suffix.lower() in exts:
                    found.append(fp)

        return sorted(found)

    def index_file(
            self,
            path: str | Path,
            force: bool = False,
    ) -> bool:
        """
        Parse, enrich metadata, chunk, embed, and store a single file.

        force=True akan memaksa re-index walaupun file tidak berubah.
        """

        fp = Path(path).resolve()

        try:
            stat = fp.stat()

        except OSError as exc:
            logger.warning(
                "Cannot stat %s: %s",
                fp,
                exc,
            )
            return False

        # --------------------------------------------------------------
        # CHECK EXISTING DOCUMENT
        # --------------------------------------------------------------

        doc_id = self._path_hash(fp)

        if not force:
            if self.store.has_document(
                    doc_id,
                    stat.st_mtime,
            ):
                logger.debug(
                    "Skipping (unchanged): %s",
                    fp.name,
                )
                return False

        # --------------------------------------------------------------
        # PARSE
        # --------------------------------------------------------------

        try:
            document = self.parser.parse(fp)

        except Exception as exc:
            logger.warning(
                "Parse failed for %s: %s",
                fp,
                exc,
            )
            return False

        # --------------------------------------------------------------
        # DETEKSI TAHUN
        # --------------------------------------------------------------

        year = self._detect_year(
            path=fp,
            content=document.content,
        )

        document.metadata["year"] = year

        logger.debug(
            "Detected year for %s: %s",
            fp.name,
            year or "(kosong)",
            )

        # --------------------------------------------------------------
        # CHUNK
        # --------------------------------------------------------------

        try:
            chunks = self.chunker.chunk(document)

        except Exception as exc:
            logger.warning(
                "Chunking failed for %s: %s",
                fp,
                exc,
            )
            return False

        if not chunks:
            logger.warning(
                "No chunks produced for %s (empty content?)",
                fp,
            )
            return False

        # --------------------------------------------------------------
        # EMBEDDING
        # --------------------------------------------------------------

        try:
            texts = [c.content for c in chunks]

            embeddings = self.embedder.embed_batch(
                texts
            )

        except Exception as exc:
            logger.warning(
                "Embedding failed for %s: %s",
                fp,
                exc,
            )
            return False

        # --------------------------------------------------------------
        # FINAL METADATA
        # --------------------------------------------------------------

        for chunk in chunks:

            chunk.metadata["last_modified"] = str(
                document.last_modified
            )

            # Pastikan year selalu ada di metadata chunk
            chunk.metadata["year"] = year

        # --------------------------------------------------------------
        # STORE
        # --------------------------------------------------------------

        try:
            # Hapus data lama terlebih dahulu.
            self.store.delete_document(doc_id)

            self.store.add_chunks(
                chunks,
                embeddings,
            )

        except Exception as exc:
            logger.warning(
                "Store failed for %s: %s",
                fp,
                exc,
            )
            return False

        logger.info(
            "Indexed %s (%d chunks, year=%s)",
            fp.name,
            len(chunks),
            year or "-",
            )

        return True

    def index_folder(
            self,
            path: str | Path,
            force: bool = False,
    ) -> Dict[str, int]:
        """
        Index all supported files in `path`.

        Returns:
            {
                "total": ...,
                "indexed": ...,
                "skipped": ...,
                "failed": ...
            }
        """

        files = self.scan_folder(path)

        total = len(files)

        indexed = 0
        skipped = 0
        failed = 0

        for i, fp in enumerate(files, start=1):

            print(
                f"  [{i}/{total}] {fp.name}",
                end=" ",
                flush=True,
            )

            try:

                result = self.index_file(
                    fp,
                    force=force,
                )

                if result:
                    indexed += 1
                    print("✓")

                else:
                    skipped += 1
                    print("(skipped)")

            except Exception as exc:

                failed += 1

                logger.warning(
                    "Unexpected error indexing %s: %s",
                    fp,
                    exc,
                )

                print(f"✗ {exc}")

        return {
            "total": total,
            "indexed": indexed,
            "skipped": skipped,
            "failed": failed,
        }

    # ------------------------------------------------------------------
    # YEAR DETECTION
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_year(value: str) -> Optional[str]:
        """
        Extract tahun 20xx dari nama file atau path.

        Tahun harus muncul sebagai bagian yang berdiri sendiri
        atau dipisahkan oleh karakter non-alfanumerik seperti:

            _2024_
            -2024-
            (2024)
            2024.pdf
            /2024/

        Tidak menganggap angka seperti:
            GMG2021
            DN202563893...
            728530833701000
        sebagai tahun dokumen.
        """

        if not value:
            return None

        matches = re.findall(
            r"(?<![A-Za-z0-9])(20\d{2})(?![A-Za-z0-9])",
            value,
        )

        if not matches:
            return None

        return matches[0]

    def _detect_year(
            self,
            path: Path,
            content: str,
    ) -> str:
        """
        Detect document year.

        Priority:

        1. Filename
        2. Folder/path
        3. Document content
        4. Empty string
        """

        # --------------------------------------------------------------
        # 1. NAMA FILE
        # --------------------------------------------------------------

        filename_year = self._extract_year(
            path.name
        )

        if filename_year:
            return filename_year

        # --------------------------------------------------------------
        # 2. PATH / FOLDER
        # --------------------------------------------------------------

        path_year = self._extract_year(
            str(path.parent)
        )

        if path_year:
            return path_year

        # --------------------------------------------------------------
        # 3. ISI DOKUMEN
        # --------------------------------------------------------------

        if content:

            content_year = self._detect_year_from_content(
                content
            )

            if content_year:
                return content_year

        # --------------------------------------------------------------
        # 4. TIDAK DITEMUKAN
        # --------------------------------------------------------------

        return ""

    @staticmethod
    def _detect_year_from_content(
            content: str,
    ) -> Optional[str]:
        """
        Detect likely document year from content.

        Mengutamakan pola yang secara eksplisit menyebut tahun,
        misalnya:

            TAHUN 2024
            TAHUN: 2024
            Tahun 2024
            PERIODE 2024
            TAHUN BUKU 2024
            PER 31 DES 2024

        Jika tidak ditemukan pola tersebut,
        fallback ke tahun 20xx yang paling sering muncul.
        """

        if not content:
            return None

        # --------------------------------------------------------------
        # POLA EKSPLISIT
        # --------------------------------------------------------------

        patterns = [
            r"\bTAHUN\s*(?:BUKU\s*)?[:\-]?\s*(20\d{2})\b",
            r"\bPERIODE\s*[:\-]?\s*(20\d{2})\b",
            r"\bPER\s+\d{1,2}\s+\w+\s+(20\d{2})\b",
            r"\bUNTUK\s+TAHUN\s+(20\d{2})\b",
            r"\bTAHUN\s+PAJAK\s+(20\d{2})\b",
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                content,
                flags=re.IGNORECASE,
            )

            if match:
                return match.group(1)

        # --------------------------------------------------------------
        # FALLBACK
        # --------------------------------------------------------------

        years = re.findall(
            r"\b(20\d{2})\b",
            content,
        )

        if not years:
            return None

        counts: Dict[str, int] = {}

        for year in years:
            counts[year] = counts.get(year, 0) + 1

        return max(
            counts,
            key=counts.get,
        )

    # ------------------------------------------------------------------
    # SEARCH
    # ------------------------------------------------------------------

    def search(
            self,
            query: str,
            top_k: int = 5,
    ):
        """
        Semantic search dengan filter tahun otomatis
        dan keyword boost ringan.

        Prioritas:

        1. Filter tahun jika query mengandung tahun.
        2. Semantic similarity.
        3. Keyword boost.
        4. Final score.
        """

        if not query or not query.strip():
            return []

        query_clean = query.strip()

        # --------------------------------------------------------------
        # EMBEDDING QUERY
        # --------------------------------------------------------------

        query_vec = self.embedder.embed(
            query_clean
        )

        # --------------------------------------------------------------
        # DETEKSI TAHUN
        # --------------------------------------------------------------

        years = re.findall(
            r"\b(20\d{2})\b",
            query_clean,
        )

        if years:

            year = years[0]

            logger.info(
                "Search menggunakan filter tahun: %s",
                year,
            )

            results = self.store.search_by_year(
                query_embedding=query_vec,
                year=year,
                top_k=max(
                    top_k * 3,
                    15,
                    ),
            )

        else:

            results = self.store.search(
                query_vec,
                top_k=max(
                    top_k * 3,
                    15,
                    ),
            )

        if not results:
            return []

        # --------------------------------------------------------------
        # KEYWORD BOOST
        # --------------------------------------------------------------

        stopwords = {
            "dan",
            "atau",
            "yang",
            "di",
            "ke",
            "dari",
            "untuk",
            "pada",
            "dengan",
            "menurut",
            "tahun",
            "pt",
            "apa",
            "berapa",
            "mengenai",
            "tentang",
            "adalah",
            "serta",
        }

        query_tokens = re.findall(
            r"[a-zA-Z0-9]+",
            query_clean.lower(),
        )

        keywords = [
            token
            for token in query_tokens
            if (
                    len(token) >= 3
                    and token not in stopwords
                    and not re.fullmatch(
                r"20\d{2}",
                token,
            )
            )
        ]

        # Hapus duplikat keyword sambil mempertahankan urutan.
        keywords = list(
            dict.fromkeys(keywords)
        )

        # --------------------------------------------------------------
        # HITUNG SCORE AKHIR
        # --------------------------------------------------------------

        scored_results = []

        for result in results:

            filename = str(
                result.metadata.get(
                    "filename",
                    "",
                )
            ).lower()

            content = str(
                result.content or ""
            ).lower()

            searchable_text = (
                f"{filename} {content}"
            )

            matched_keywords = 0

            for keyword in keywords:

                if keyword in searchable_text:
                    matched_keywords += 1

            # ----------------------------------------------------------
            # BOOST MAKSIMUM +0.20
            # ----------------------------------------------------------

            if keywords:

                keyword_ratio = (
                        matched_keywords
                        / len(keywords)
                )

                boost = min(
                    keyword_ratio * 0.20,
                    0.20,
                    )

            else:

                boost = 0.0

            final_score = (
                    result.score
                    + boost
            )

            logger.debug(
                "Retrieval: %s | semantic=%.4f | "
                "keyword_hits=%d/%d | boost=%.4f | "
                "final=%.4f",
                filename,
                result.score,
                matched_keywords,
                len(keywords),
                boost,
                final_score,
            )

            scored_results.append(
                (
                    final_score,
                    result,
                )
            )

        # --------------------------------------------------------------
        # SORT FINAL SCORE
        # --------------------------------------------------------------

        scored_results.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        # --------------------------------------------------------------
        # SIMPAN FINAL SCORE
        # --------------------------------------------------------------

        final_results = []

        for final_score, result in scored_results:

            result.score = final_score

            final_results.append(
                result
            )

        return final_results[:top_k]

    # ------------------------------------------------------------------
    # STATS
    # ------------------------------------------------------------------

    def get_stats(self) -> IndexStats:

        chunk_count = self.store.count()

        doc_count = self.store.distinct_documents()

        storage = self._dir_size(
            Path(
                self.settings.chroma_persist_dir
            )
        )

        return IndexStats(
            documents=doc_count,
            chunks=chunk_count,
            storage_bytes=storage,
        )

    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _path_hash(path: Path) -> str:
        import hashlib

        return hashlib.sha256(
            str(path).encode()
        ).hexdigest()

    @staticmethod
    def _dir_size(
            directory: Path,
    ) -> int:

        total = 0

        if directory.exists():

            for dirpath, _dirs, files in os.walk(
                    directory
            ):

                for fname in files:

                    try:

                        total += (
                                Path(dirpath) / fname
                        ).stat().st_size

                    except OSError:
                        pass

        return total

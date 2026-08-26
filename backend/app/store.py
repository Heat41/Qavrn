from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .chunker import DocumentChunk

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "deeplens_index"


class SearchResult:
    __slots__ = (
        "chunk_id",
        "document_id",
        "content",
        "score",
        "metadata",
    )

    def __init__(
            self,
            chunk_id: str,
            document_id: str,
            content: str,
            score: float,
            metadata: Dict[str, Any],
    ) -> None:
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.content = content
        self.score = score
        self.metadata = metadata


class VectorStore:
    """
    Persistent ChromaDB-backed vector store.
    """

    def __init__(
            self,
            persist_dir: str | Path = "./data/chroma",
    ) -> None:
        self.persist_dir = str(
            Path(persist_dir).resolve()
        )
        self._client = None
        self._collection = None

    # ==================================================================
    # Lifecycle
    # ==================================================================

    def _ensure_connected(self) -> None:
        if self._client is not None:
            return

        try:
            import chromadb  # type: ignore
        except ImportError:
            raise ImportError(
                "chromadb is required: pip install chromadb"
            )

        Path(self.persist_dir).mkdir(
            parents=True,
            exist_ok=True,
        )

        self._client = chromadb.PersistentClient(
            path=self.persist_dir
        )

        self._collection = (
            self._client.get_or_create_collection(
                name=_COLLECTION_NAME,
                metadata={
                    "hnsw:space": "cosine"
                },
            )
        )

        logger.debug(
            "ChromaDB connected at %s",
            self.persist_dir,
        )

    @property
    def collection(self):
        self._ensure_connected()
        return self._collection

    # ==================================================================
    # Write
    # ==================================================================

    def add_chunks(
            self,
            chunks: List[DocumentChunk],
            embeddings: List[List[float]],
    ) -> None:
        """
        Menyimpan / memperbarui chunks dan embeddings.
        """

        if not chunks:
            return

        assert len(chunks) == len(embeddings), (
            "chunks and embeddings must be the same length"
        )

        metadatas: List[Dict[str, Any]] = []

        for chunk in chunks:

            metadata = {
                **{
                    k: str(v)
                    for k, v in chunk.metadata.items()
                },
                "document_id": str(
                    chunk.document_id
                ),
                "chunk_index": str(
                    chunk.chunk_index
                ),
                "total_chunks": str(
                    chunk.total_chunks
                ),
            }

            # ----------------------------------------------------------
            # Pastikan filename tersedia
            # ----------------------------------------------------------

            if "filename" not in metadata:

                file_path = metadata.get(
                    "file_path",
                    "",
                )

                if file_path:
                    metadata["filename"] = (
                        Path(file_path).name
                    )

            # ----------------------------------------------------------
            # Pastikan year tersedia jika bisa ditemukan
            # ----------------------------------------------------------

            year = self._extract_year_from_metadata(
                metadata,
                content=chunk.content,
            )

            if year:
                metadata["year"] = year

            metadatas.append(metadata)

        self.collection.upsert(
            ids=[
                c.chunk_id
                for c in chunks
            ],
            embeddings=embeddings,
            documents=[
                c.content
                for c in chunks
            ],
            metadatas=metadatas,
        )

        logger.debug(
            "Upserted %d chunks",
            len(chunks),
        )

    # ==================================================================
    # Delete
    # ==================================================================

    def delete_document(
            self,
            document_id: str,
    ) -> None:
        """
        Menghapus semua chunks milik satu dokumen.
        """

        results = self.collection.get(
            where={
                "document_id": document_id
            }
        )

        ids = results.get(
            "ids",
            [],
        )

        if ids:
            self.collection.delete(
                ids=ids
            )

            logger.debug(
                "Deleted %d chunks for document %s",
                len(ids),
                document_id,
            )

    # ==================================================================
    # Search
    # ==================================================================

    def search(
            self,
            query_embedding: List[float],
            top_k: int = 5,
            where: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Vector search menggunakan ChromaDB.

        top_k = jumlah chunk hasil akhir.

        Berbeda dari versi sebelumnya, hasil TIDAK lagi dideduplicasi
        menjadi satu chunk per document_id.

        Satu dokumen boleh menyumbang beberapa chunk apabila chunk
        tersebut memang relevan dengan query.

        Namun jumlah chunk dari satu dokumen dibatasi agar satu file
        tidak membanjiri seluruh hasil retrieval.
        """

        collection_count = self.collection.count()

        if collection_count == 0:
            return []

        # --------------------------------------------------------------
        # Ambil kandidat sebanyak mungkin agar chunk yang relevan
        # dari dokumen yang sama tetap bisa ditemukan.
        # --------------------------------------------------------------

        candidate_k = max(
            top_k * 10,
            50,
            )

        candidate_k = min(
            candidate_k,
            collection_count,
        )

        query_kwargs: Dict[str, Any] = {
            "query_embeddings": [
                query_embedding
            ],
            "n_results": candidate_k,
            "include": [
                "documents",
                "metadatas",
                "distances",
            ],
        }

        if where:
            query_kwargs["where"] = where

        try:
            results = self.collection.query(
                **query_kwargs
            )

        except Exception as exc:
            logger.warning(
                "Vector search failed: %s",
                exc,
            )
            return []

        candidates = self._convert_results(
            results
        )

        if not candidates:
            return []

        # --------------------------------------------------------------
        # DIVERSIFIED RETRIEVAL
        #
        # Satu dokumen boleh memberikan beberapa chunk.
        #
        # Tetapi dibatasi agar satu dokumen tidak menguasai
        # seluruh hasil pencarian.
        # --------------------------------------------------------------

        max_chunks_per_document = 4

        document_counts: Dict[str, int] = {}

        diversified_results: List[SearchResult] = []

        for result in candidates:

            document_id = (
                    result.document_id
                    or result.metadata.get(
                "document_id",
                "",
            )
            )

            # Jika document_id tidak tersedia,
            # perlakukan setiap chunk sebagai dokumen sendiri.
            if not document_id:
                document_id = f"chunk:{result.chunk_id}"

            current_count = document_counts.get(
                document_id,
                0,
            )

            if current_count >= max_chunks_per_document:
                continue

            diversified_results.append(
                result
            )

            document_counts[document_id] = (
                    current_count + 1
            )

            # Kita sudah punya cukup kandidat.
            if len(diversified_results) >= top_k:
                break

        # --------------------------------------------------------------
        # Pastikan urutan berdasarkan score terbaik.
        # --------------------------------------------------------------

        diversified_results.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        logger.debug(
            "Vector search: %d candidate chunks -> "
            "%d diversified chunks from %d documents",
            len(candidates),
            len(diversified_results),
            len(document_counts),
        )

        return diversified_results[:top_k]

    # ==================================================================
    # Filename Search
    # ==================================================================

    def search_by_filename(
            self,
            query_embedding: List[float],
            filename: str,
            top_k: int = 5,
    ) -> List[SearchResult]:
        """
        Search berdasarkan nama file.

        Pencarian tetap dilakukan berdasarkan metadata filename
        dan fallback file_path.
        """

        target = self._normalize_filename(
            filename
        )

        if not target:
            return []

        candidates = self.search(
            query_embedding=query_embedding,
            top_k=max(
                top_k * 10,
                50,
                ),
        )

        matched: List[SearchResult] = []

        for result in candidates:

            stored_filename = self._get_filename(
                result.metadata
            )

            if not stored_filename:
                continue

            if (
                    self._normalize_filename(
                        stored_filename
                    )
                    == target
            ):
                matched.append(result)

        return matched[:top_k]

    # ==================================================================
    # Year Search
    # ==================================================================

    def search_by_year(
            self,
            query_embedding: List[float],
            year: str,
            top_k: int = 10,
    ) -> List[SearchResult]:
        """
        Search berdasarkan tahun.

        Filter tahun dilakukan LANGSUNG oleh ChromaDB.
        """

        target_year = str(year).strip()

        if not re.fullmatch(
                r"20\d{2}",
                target_year,
        ):
            logger.debug(
                "Invalid year filter: %s",
                target_year,
            )
            return []

        return self.search(
            query_embedding=query_embedding,
            top_k=top_k,
            where={
                "year": target_year
            },
        )

    # ==================================================================
    # Convert Chroma Result
    # ==================================================================

    @staticmethod
    def _convert_results(
            results: Dict[str, Any],
    ) -> List[SearchResult]:

        output: List[SearchResult] = []

        ids = results.get(
            "ids",
            [[]],
        )[0]

        docs = results.get(
            "documents",
            [[]],
        )[0]

        metas = results.get(
            "metadatas",
            [[]],
        )[0]

        distances = results.get(
            "distances",
            [[]],
        )[0]

        for (
                chunk_id,
                doc,
                meta,
                dist,
        ) in zip(
            ids,
            docs,
            metas,
            distances,
        ):

            meta = meta or {}

            output.append(
                SearchResult(
                    chunk_id=chunk_id,
                    document_id=meta.get(
                        "document_id",
                        "",
                    ),
                    content=doc or "",
                    score=1.0 - dist,
                    metadata=meta,
                )
            )

        return output

    # ==================================================================
    # Filename Helpers
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

    @staticmethod
    def _get_filename(
            metadata: Dict[str, Any],
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
    # Year Helpers
    # ==================================================================

    @staticmethod
    def _extract_year_from_metadata(
            metadata: Dict[str, Any],
            content: str = "",
    ) -> str:
        """
        Menentukan tahun dokumen berdasarkan prioritas:

        1. Metadata year yang sudah valid
        2. Nama file
        3. File path
        4. Isi dokumen

        Mendukung tahun yang menempel pada karakter lain,
        misalnya:
            GMG2021
            FORM_1770_..._2024_0
            012023062023
        """

        # --------------------------------------------------------------
        # 1. YEAR DARI METADATA
        # --------------------------------------------------------------

        year = str(
            metadata.get(
                "year",
                "",
            )
        ).strip()

        if re.fullmatch(
                r"20\d{2}",
                year,
        ):
            return year

        # --------------------------------------------------------------
        # 2. YEAR DARI FILENAME
        # --------------------------------------------------------------

        filename = str(
            metadata.get(
                "filename",
                "",
            )
        ).strip()

        if filename:
            matches = re.findall(
                r"20\d{2}",
                filename,
            )

            if matches:
                return matches[0]

        # --------------------------------------------------------------
        # 3. YEAR DARI FILE PATH
        # --------------------------------------------------------------

        file_path = str(
            metadata.get(
                "file_path",
                "",
            )
        ).strip()

        if file_path:
            matches = re.findall(
                r"20\d{2}",
                file_path,
            )

            if matches:
                return matches[0]

        # --------------------------------------------------------------
        # 4. YEAR DARI ISI DOKUMEN
        # --------------------------------------------------------------

        content = str(
            content or ""
        )

        if content:
            matches = re.findall(
                r"\b(20\d{2})\b",
                content,
            )

            if matches:
                return matches[0]

        return ""

    # ==================================================================
    # Document Status
    # ==================================================================

    def has_document(
            self,
            document_id: str,
            last_modified: float,
    ) -> bool:

        results = self.collection.get(
            where={
                "document_id": document_id
            },
            limit=1,
            include=[
                "metadatas"
            ],
        )

        metas = results.get(
            "metadatas",
            [],
        )

        if not metas:
            return False

        stored_mtime = metas[0].get(
            "last_modified"
        )

        if stored_mtime is None:
            return False

        try:
            return (
                    float(stored_mtime)
                    >= last_modified
            )

        except (
                ValueError,
                TypeError,
        ):
            return False

    # ==================================================================
    # Stats
    # ==================================================================

    def count(self) -> int:
        return self.collection.count()

    def distinct_documents(
            self,
    ) -> int:

        all_meta = self.collection.get(
            include=[
                "metadatas"
            ]
        ).get(
            "metadatas",
            [],
        )

        return len(
            {
                m.get("document_id")
                for m in all_meta
                if m.get("document_id")
            }
        )

    def list_documents(
            self,
    ) -> List[Dict[str, Any]]:

        all_meta = self.collection.get(
            include=[
                "metadatas"
            ]
        ).get(
            "metadatas",
            [],
        ) or []

        seen: Dict[
            str,
            Dict[str, Any],
        ] = {}

        for meta in all_meta:

            doc_id = meta.get(
                "document_id",
                "",
            )

            if (
                    doc_id
                    and doc_id not in seen
            ):

                filename = self._get_filename(
                    meta
                )

                year = str(
                    meta.get(
                        "year",
                        "",
                    )
                ).strip()

                if not re.fullmatch(
                        r"20\d{2}",
                        year,
                ):
                    year = (
                        self._extract_year_from_metadata(
                            meta
                        )
                    )

                seen[doc_id] = {
                    "document_id": doc_id,
                    "filename": filename,
                    "file_path": meta.get(
                        "file_path",
                        "",
                    ),
                    "file_type": meta.get(
                        "file_type",
                        "",
                    ),
                    "year": year,
                    "total_chunks": meta.get(
                        "total_chunks",
                        "0",
                    ),
                }

        return list(
            seen.values()
        )
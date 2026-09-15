from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from backend.app import api
from backend.app.store import SearchResult


class FakeCollection:

    def __init__(self, results=None) -> None:
        self.results = list(results or [])

    def get(self, *args, **kwargs):
        where = kwargs.get("where") or {}

        items = self.results
        if "year" in where:
            year = str(where["year"])
            items = [
                item
                for item in items
                if str(item.metadata.get("year", "")) == year
            ]

        return {
            "ids": [item.chunk_id for item in items],
            "documents": [item.content for item in items],
            "metadatas": [item.metadata for item in items],
        }


class FakeStore:

    def __init__(self, results=None) -> None:
        self.results = list(results or [])
        self.collection = FakeCollection(self.results)
        self.documents = []

    def search(self, query_vec, top_k: int, where=None):
        items = self.results

        if where and "year" in where:
            year = str(where["year"])
            items = [
                item
                for item in items
                if str(item.metadata.get("year", "")) == year
            ]

        return list(items[:top_k])

    def list_documents(self):
        return list(self.documents)


class FakeEmbedder:

    def __init__(self) -> None:
        self.calls = []

    def embed(self, question: str):
        self.calls.append(question)
        return [0.1, 0.2, 0.3]


class FakeOllama:

    def __init__(
        self,
        *,
        tokens=None,
        available=True,
        error: Exception | None = None,
    ) -> None:
        self.tokens = list(tokens or ["jawaban", " AI"])
        self.available = available
        self.error = error
        self.stream_calls = []

    def is_available(self):
        return self.available

    def generate_stream(
        self,
        question: str,
        context: str,
        model: str,
    ):
        self.stream_calls.append(
            {
                "question": question,
                "context": context,
                "model": model,
            }
        )

        if self.error is not None:
            raise self.error

        return iter(self.tokens)


class FakeIndexer:

    def __init__(self, results=None) -> None:
        self.embedder = FakeEmbedder()
        self.store = FakeStore(results)
        self.indexed_folders = []
        self.stats = SimpleNamespace(
            documents=0,
            chunks=0,
            storage_mb=0.0,
            storage_bytes=0,
        )

    def get_stats(self):
        return self.stats

    def index_folder(self, folder: Path):
        self.indexed_folders.append(str(folder))
        self.stats = SimpleNamespace(
            documents=2,
            chunks=6,
            storage_mb=0.01,
            storage_bytes=10240,
        )
        self.store.documents = [
            {
                "document_id": "doc-1",
                "filename": "A.pdf",
                "file_path": str(folder / "A.pdf"),
                "file_type": "pdf",
                "total_chunks": "3",
            },
            {
                "document_id": "doc-2",
                "filename": "B.pdf",
                "file_path": str(folder / "B.pdf"),
                "file_type": "pdf",
                "total_chunks": "3",
            },
        ]
        return {
            "total": 2,
            "indexed": 2,
            "skipped": 0,
            "failed": 0,
        }


class FakeWatcher:

    def __init__(self) -> None:
        self.watched_folders = []

    def watch(self, folder: str):
        if folder not in self.watched_folders:
            self.watched_folders.append(folder)

    def unwatch(self, folder: str):
        if folder in self.watched_folders:
            self.watched_folders.remove(folder)


def search_result(
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
            "filename": filename,
            "file_path": file_path,
            "year": year,
            "chunk_index": 0,
        },
    )


def request_for(
    indexer: FakeIndexer,
    ollama: FakeOllama,
    watcher: FakeWatcher | None = None,
):
    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                indexer=indexer,
                ollama=ollama,
                watcher=watcher or FakeWatcher(),
            )
        )
    )


async def collect_sse(response) -> list[dict | str]:
    events: list[dict | str] = []

    async for item in response.body_iterator:
        if isinstance(item, bytes):
            item = item.decode("utf-8")

        for block in item.split("\n\n"):
            block = block.strip()
            if not block.startswith("data: "):
                continue

            raw = block[len("data: "):]

            if raw == "[DONE]":
                events.append("[DONE]")
            else:
                events.append(json.loads(raw))

    return events


class TestE2ESmokeStage7(unittest.TestCase):

    def test_exact_file_location_flows_through_api_rag_and_sse(self) -> None:
        target = search_result(
            "chunk-1",
            "doc-1",
            "Neraca perusahaan tahun 2024",
            1.0,
            "Neraca dan RL PT AEP 2024.xls",
            (
                "C:\\Users\\ASUS260922\\Documents\\PT.AEP\\"
                "Laporan Keuangan\\2024\\Neraca dan RL PT AEP 2024.xls"
            ),
        )
        indexer = FakeIndexer([target])
        ollama = FakeOllama()

        async def run_case():
            response = await api.ask(
                api.AskRequest(
                    question=(
                        "di mana file Neraca dan RL PT AEP 2024.xls"
                    ),
                    top_k=5,
                ),
                request_for(indexer, ollama),
            )
            return await collect_sse(response)

        events = asyncio.run(run_case())

        self.assertEqual(events[-1], "[DONE]")
        self.assertEqual(events[0]["type"], "token")
        self.assertIn(
            r"Laporan Keuangan\2024",
            events[0]["content"],
        )
        self.assertEqual(events[1]["type"], "sources")
        self.assertEqual(
            events[1]["sources"][0]["filename"],
            "Neraca dan RL PT AEP 2024.xls",
        )
        self.assertEqual(indexer.embedder.calls, [])
        self.assertEqual(ollama.stream_calls, [])

    def test_ai_query_flows_through_retrieval_model_and_sse(self) -> None:
        result = search_result(
            "chunk-ai",
            "doc-ai",
            (
                "Dokumen proyek menjelaskan kondisi operasional "
                "perusahaan dan kegiatan tahun berjalan."
            ),
            0.8,
            "Profil Operasional 2024.pdf",
            r"C:\Docs\Profil Operasional 2024.pdf",
        )
        indexer = FakeIndexer([result])
        ollama = FakeOllama(
            tokens=["Jawaban", " dari", " AI"],
        )

        async def run_case():
            response = await api.ask(
                api.AskRequest(
                    question="jelaskan kegiatan operasional tahun 2024",
                    top_k=3,
                    model="qwen2.5:3b",
                ),
                request_for(indexer, ollama),
            )
            return await collect_sse(response)

        events = asyncio.run(run_case())

        token_text = "".join(
            event["content"]
            for event in events
            if isinstance(event, dict)
            and event.get("type") == "token"
        )

        self.assertEqual(token_text, "Jawaban dari AI")
        self.assertEqual(
            ollama.stream_calls[0]["model"],
            "qwen2.5:3b",
        )
        self.assertEqual(
            indexer.embedder.calls,
            ["jelaskan kegiatan operasional tahun 2024"],
        )
        sources = next(
            event["sources"]
            for event in events
            if isinstance(event, dict)
            and event.get("type") == "sources"
        )
        self.assertEqual(
            sources[0]["filename"],
            "Profil Operasional 2024.pdf",
        )
        self.assertEqual(events[-1], "[DONE]")

    def test_empty_retrieval_flows_to_not_found_sources_and_done(self) -> None:
        indexer = FakeIndexer([])
        ollama = FakeOllama()

        async def run_case():
            response = await api.ask(
                api.AskRequest(
                    question="jelaskan dokumen yang tidak ada",
                    top_k=3,
                ),
                request_for(indexer, ollama),
            )
            return await collect_sse(response)

        events = asyncio.run(run_case())

        self.assertEqual(events[0]["type"], "token")
        self.assertIn(
            "Informasi tersebut tidak ditemukan",
            events[0]["content"],
        )
        self.assertEqual(
            events[1],
            {
                "type": "sources",
                "sources": [],
            },
        )
        self.assertEqual(events[-1], "[DONE]")
        self.assertEqual(ollama.stream_calls, [])

    def test_ollama_connection_error_flows_to_error_and_done(self) -> None:
        result = search_result(
            "chunk-ai",
            "doc-ai",
            "informasi umum operasional perusahaan",
            0.9,
            "Operasional.pdf",
            r"C:\Docs\Operasional.pdf",
            year="2024",
        )
        indexer = FakeIndexer([result])
        ollama = FakeOllama(
            error=ConnectionError(
                "Ollama tidak tersedia"
            )
        )

        async def run_case():
            response = await api.ask(
                api.AskRequest(
                    question="jelaskan informasi operasional tahun 2024",
                    top_k=3,
                    model="qwen2.5:3b",
                ),
                request_for(indexer, ollama),
            )
            return await collect_sse(response)

        events = asyncio.run(run_case())

        self.assertEqual(
            events[0],
            {
                "type": "error",
                "message": "Ollama tidak tersedia",
            },
        )
        self.assertEqual(events[-1], "[DONE]")

    def test_index_watch_refresh_flow_updates_frontend_contract_state(self) -> None:
        indexer = FakeIndexer([])
        watcher = FakeWatcher()
        ollama = FakeOllama()

        async def run_case(folder: Path):
            request = request_for(
                indexer,
                ollama,
                watcher,
            )

            index_payload = await api.watch_folder(
                api.WatchRequest(
                    folder_path=str(folder)
                ),
                request,
            )
            health_payload = await api.health(request)
            docs_payload = await api.list_documents(request)
            watched_payload = await api.get_watched(request)

            return (
                index_payload,
                health_payload,
                docs_payload,
                watched_payload,
            )

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (
                watch_payload,
                health_payload,
                docs_payload,
                watched_payload,
            ) = asyncio.run(run_case(folder))

            resolved = str(folder.resolve())

        self.assertTrue(watch_payload["watching"])
        self.assertEqual(watch_payload["folder"], resolved)
        self.assertEqual(health_payload["documents"], 2)
        self.assertEqual(health_payload["chunks"], 6)
        self.assertEqual(len(docs_payload["documents"]), 2)
        self.assertEqual(
            watched_payload["folders"],
            [resolved],
        )


if __name__ == "__main__":
    unittest.main()

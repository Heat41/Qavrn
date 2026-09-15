from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from backend.app import api


class FakeStore:

    def __init__(self, documents=None) -> None:
        self.documents = documents or []
        self.deleted = []

    def list_documents(self):
        return list(self.documents)

    def delete_document(self, document_id: str) -> None:
        self.deleted.append(document_id)


class FakeIndexer:

    def __init__(
        self,
        *,
        documents=None,
        chunks=0,
        storage_mb=0.0,
        storage_bytes=0,
    ) -> None:
        self.store = FakeStore(documents)
        self.stats = SimpleNamespace(
            documents=len(documents or []),
            chunks=chunks,
            storage_mb=storage_mb,
            storage_bytes=storage_bytes,
        )
        self.indexed_files = []
        self.indexed_folders = []

    def get_stats(self):
        return self.stats

    def index_file(self, path: Path):
        self.indexed_files.append(str(path))
        return True

    def index_folder(self, path: Path):
        self.indexed_folders.append(str(path))
        return {
            "indexed": 2,
            "skipped": 1,
        }

    def _path_hash(self, path: Path):
        return f"hash:{path.name}"


class FakeWatcher:

    def __init__(self, watched=None) -> None:
        self.watched_folders = list(watched or [])
        self.watch_calls = []
        self.unwatch_calls = []

    def watch(self, folder: str) -> None:
        self.watch_calls.append(folder)
        if folder not in self.watched_folders:
            self.watched_folders.append(folder)

    def unwatch(self, folder: str) -> None:
        self.unwatch_calls.append(folder)
        if folder in self.watched_folders:
            self.watched_folders.remove(folder)


class FakeOllama:

    def __init__(self, available=True) -> None:
        self.available = available

    def is_available(self):
        return self.available


def make_request(
    *,
    indexer=None,
    watcher=None,
    ollama=None,
):
    state = SimpleNamespace(
        indexer=indexer or FakeIndexer(),
        watcher=watcher or FakeWatcher(),
        ollama=ollama or FakeOllama(),
    )
    return SimpleNamespace(
        app=SimpleNamespace(state=state)
    )


class TestAPICoreRegression(unittest.TestCase):

    def test_health_schema(self) -> None:
        indexer = FakeIndexer(
            documents=[{"id": "a"}, {"id": "b"}],
            chunks=17,
        )
        request = make_request(
            indexer=indexer,
            ollama=FakeOllama(available=True),
        )

        payload = asyncio.run(api.health(request))

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["documents"], 2)
        self.assertEqual(payload["chunks"], 17)
        self.assertTrue(payload["ollama_available"])
        self.assertIn("model", payload)

    def test_stats_schema_and_rounding(self) -> None:
        indexer = FakeIndexer(
            documents=[{"id": "a"}],
            chunks=5,
            storage_mb=12.3456,
            storage_bytes=12345678,
        )

        payload = asyncio.run(
            api.stats(
                make_request(indexer=indexer)
            )
        )

        self.assertEqual(
            payload,
            {
                "documents": 1,
                "chunks": 5,
                "storage_mb": 12.35,
                "storage_bytes": 12345678,
            },
        )

    def test_documents_schema(self) -> None:
        documents = [
            {
                "filename": "Lapkeu.pdf",
                "file_path": r"C:\Docs\Lapkeu.pdf",
            }
        ]
        indexer = FakeIndexer(documents=documents)

        payload = asyncio.run(
            api.list_documents(
                make_request(indexer=indexer)
            )
        )

        self.assertEqual(
            payload,
            {"documents": documents},
        )

    def test_watched_folders_schema(self) -> None:
        watcher = FakeWatcher(
            watched=[r"C:\Docs", r"D:\Arsip"]
        )

        payload = asyncio.run(
            api.get_watched(
                make_request(watcher=watcher)
            )
        )

        self.assertEqual(
            payload,
            {
                "folders": [
                    r"C:\Docs",
                    r"D:\Arsip",
                ]
            },
        )

    def test_unwatch_calls_watcher_and_returns_schema(self) -> None:
        watcher = FakeWatcher(
            watched=[r"C:\Docs"]
        )
        body = api.WatchRequest(
            folder_path=r"C:\Docs"
        )

        payload = asyncio.run(
            api.unwatch_folder(
                body,
                make_request(watcher=watcher),
            )
        )

        self.assertEqual(
            watcher.unwatch_calls,
            [r"C:\Docs"],
        )
        self.assertEqual(
            payload,
            {
                "folder": r"C:\Docs",
                "watching": False,
            },
        )


if __name__ == "__main__":
    unittest.main()

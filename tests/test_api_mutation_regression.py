from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastapi import HTTPException

from backend.app import api
from tests.test_api_core_regression import (
    FakeIndexer,
    FakeWatcher,
    make_request,
)


class TestAPIMutationRegression(unittest.TestCase):

    def test_scan_missing_file_returns_404(self) -> None:
        missing = Path(tempfile.gettempdir()) / (
            "qvarn_missing_scan_123456789.pdf"
        )

        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(
                api.scan_document(
                    api.DocumentRequest(
                        file_path=str(missing)
                    ),
                    make_request(),
                )
            )

        self.assertEqual(ctx.exception.status_code, 404)

    def test_scan_unsupported_extension_returns_400(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "contoh.unsupported"
            path.write_text("x", encoding="utf-8")

            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(
                    api.scan_document(
                        api.DocumentRequest(
                            file_path=str(path)
                        ),
                        make_request(),
                    )
                )

        self.assertEqual(ctx.exception.status_code, 400)

    def test_scan_success_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "contoh.pdf"
            path.write_bytes(b"%PDF-fake")

            indexer = FakeIndexer(
                documents=[{"id": "a"}],
                chunks=3,
            )

            payload = asyncio.run(
                api.scan_document(
                    api.DocumentRequest(
                        file_path=str(path)
                    ),
                    make_request(indexer=indexer),
                )
            )

        self.assertTrue(payload["success"])
        self.assertTrue(payload["indexed"])
        self.assertEqual(payload["filename"], "contoh.pdf")
        self.assertEqual(payload["documents"], 1)
        self.assertEqual(payload["chunks"], 3)
        self.assertIn(str(path.resolve()), indexer.indexed_files)

    def test_reindex_success_deletes_old_document_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "contoh.pdf"
            path.write_bytes(b"%PDF-fake")

            indexer = FakeIndexer(
                documents=[{"id": "a"}],
                chunks=4,
            )

            payload = asyncio.run(
                api.reindex_document(
                    api.DocumentRequest(
                        file_path=str(path)
                    ),
                    make_request(indexer=indexer),
                )
            )

        self.assertEqual(
            indexer.store.deleted,
            ["hash:contoh.pdf"],
        )
        self.assertEqual(
            indexer.indexed_files,
            [str(path.resolve())],
        )
        self.assertTrue(payload["success"])
        self.assertEqual(
            payload["message"],
            "File berhasil di-index ulang.",
        )

    def test_watch_missing_folder_returns_404(self) -> None:
        missing = Path(tempfile.gettempdir()) / (
            "qvarn_missing_watch_123456789"
        )

        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(
                api.watch_folder(
                    api.WatchRequest(
                        folder_path=str(missing)
                    ),
                    make_request(),
                )
            )

        self.assertEqual(ctx.exception.status_code, 404)

    def test_watch_folder_indexes_then_enables_watcher(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            indexer = FakeIndexer(
                documents=[{"id": "a"}],
                chunks=8,
            )
            watcher = FakeWatcher()

            payload = asyncio.run(
                api.watch_folder(
                    api.WatchRequest(
                        folder_path=str(folder)
                    ),
                    make_request(
                        indexer=indexer,
                        watcher=watcher,
                    ),
                )
            )

            resolved = str(folder.resolve())

        self.assertEqual(
            indexer.indexed_folders,
            [resolved],
        )
        self.assertEqual(
            watcher.watch_calls,
            [resolved],
        )
        self.assertTrue(payload["watching"])
        self.assertEqual(payload["folder"], resolved)
        self.assertEqual(payload["documents"], 1)
        self.assertEqual(payload["chunks"], 8)


if __name__ == "__main__":
    unittest.main()

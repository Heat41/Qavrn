from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from backend.app import api
from tests.test_api_core_regression import (
    FakeIndexer,
    FakeWatcher,
    make_request,
)


class TestAPIFolderTreeRegression(unittest.TestCase):

    def test_folder_tree_marks_indexed_file_and_watching(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "Laporan Keuangan"
            nested.mkdir()
            pdf = nested / "Lapkeu 2024.pdf"
            pdf.write_bytes(b"%PDF-fake")

            documents = [
                {
                    "filename": pdf.name,
                    "file_path": str(pdf.resolve()),
                    "year": "2024",
                }
            ]
            indexer = FakeIndexer(documents=documents)
            watcher = FakeWatcher(
                watched=[str(root.resolve())]
            )

            payload = asyncio.run(
                api.get_folder_tree(
                    make_request(
                        indexer=indexer,
                        watcher=watcher,
                    )
                )
            )

        self.assertEqual(len(payload["folders"]), 1)

        root_node = payload["folders"][0]
        self.assertTrue(root_node["watching"])
        self.assertEqual(root_node["path"], str(root.resolve()))
        self.assertEqual(len(root_node["folders"]), 1)

        nested_node = root_node["folders"][0]
        self.assertEqual(
            nested_node["name"],
            "Laporan Keuangan",
        )
        self.assertEqual(len(nested_node["files"]), 1)

        file_node = nested_node["files"][0]
        self.assertEqual(
            file_node["name"],
            "Lapkeu 2024.pdf",
        )
        self.assertTrue(file_node["indexed"])
        self.assertEqual(
            file_node["document"]["year"],
            "2024",
        )

    def test_folder_tree_ignores_missing_watched_root(self) -> None:
        missing = str(
            (
                Path(tempfile.gettempdir())
                / "qvarn_missing_tree_123456789"
            ).resolve()
        )
        watcher = FakeWatcher(watched=[missing])

        payload = asyncio.run(
            api.get_folder_tree(
                make_request(
                    indexer=FakeIndexer(),
                    watcher=watcher,
                )
            )
        )

        self.assertEqual(
            payload,
            {"folders": []},
        )


if __name__ == "__main__":
    unittest.main()

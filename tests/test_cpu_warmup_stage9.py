from __future__ import annotations

import sys
import threading
import time
import types
import unittest
from unittest.mock import Mock, patch

from backend.app.api import _warm_runtime
from backend.app.embedder import Embedder
from backend.app.llm import OllamaClient, _KEEP_ALIVE


class TestEmbeddingWarmup(unittest.TestCase):

    def test_concurrent_warm_up_loads_model_once(self) -> None:
        calls: list[str] = []

        class FakeSentenceTransformer:

            def __init__(self, model_name: str) -> None:
                calls.append(model_name)
                time.sleep(0.02)

        fake_module = types.ModuleType("sentence_transformers")
        fake_module.SentenceTransformer = FakeSentenceTransformer

        embedder = Embedder("test-mini-model")

        with patch.dict(
            sys.modules,
            {"sentence_transformers": fake_module},
        ):
            threads = [
                threading.Thread(target=embedder.warm_up)
                for _ in range(6)
            ]

            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join()

        self.assertEqual(calls, ["test-mini-model"])

    def test_warm_up_is_idempotent(self) -> None:
        sentinel = object()
        embedder = Embedder("unused")
        embedder._model = sentinel

        embedder.warm_up()
        embedder.warm_up()

        self.assertIs(embedder._model, sentinel)


class TestOllamaWarmup(unittest.TestCase):

    @patch("requests.post")
    def test_warm_up_uses_cpu_and_keep_alive(
        self,
        post: Mock,
    ) -> None:
        response = Mock()
        response.status_code = 200
        post.return_value = response

        client = OllamaClient("http://127.0.0.1:11434")
        client.warm_up("qwen2.5:3b")

        post.assert_called_once()

        _, kwargs = post.call_args
        body = kwargs["json"]

        self.assertEqual(
            kwargs["timeout"],
            (10, 120),
        )
        self.assertEqual(
            body["model"],
            "qwen2.5:3b",
        )
        self.assertEqual(
            body["prompt"],
            "",
        )
        self.assertFalse(body["stream"])
        self.assertEqual(
            body["keep_alive"],
            _KEEP_ALIVE,
        )
        self.assertEqual(
            body["options"]["num_gpu"],
            0,
        )

    def test_generation_body_keeps_model_resident(self) -> None:
        client = OllamaClient("http://127.0.0.1:11434")

        body = client._build_body(
            "pertanyaan",
            "context",
            "qwen2.5:3b",
        )

        self.assertEqual(
            body["keep_alive"],
            _KEEP_ALIVE,
        )


class TestRuntimeWarmup(unittest.TestCase):

    def test_warmup_failure_does_not_crash_runtime(self) -> None:

        class FailingEmbedder:

            def warm_up(self) -> None:
                raise RuntimeError("embedding unavailable")

        class FakeIndexer:
            embedder = FailingEmbedder()

        class FailingOllama:

            def warm_up(self, model: str) -> None:
                raise ConnectionError("ollama unavailable")

        _warm_runtime(
            FakeIndexer(),
            FailingOllama(),
        )

    def test_ollama_warmup_still_runs_after_embedding_failure(
        self,
    ) -> None:

        class FailingEmbedder:

            def warm_up(self) -> None:
                raise RuntimeError("embedding unavailable")

        class FakeIndexer:
            embedder = FailingEmbedder()

        called: list[str] = []

        class FakeOllama:

            def warm_up(self, model: str) -> None:
                called.append(model)

        _warm_runtime(
            FakeIndexer(),
            FakeOllama(),
        )

        self.assertEqual(len(called), 1)


if __name__ == "__main__":
    unittest.main()

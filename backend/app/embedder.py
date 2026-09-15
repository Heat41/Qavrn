from __future__ import annotations

import logging
import threading
from typing import List

logger = logging.getLogger(__name__)


class Embedder:
    """
    Wraps a sentence-transformers model with lazy loading.

    The model is downloaded once on first use and cached locally by
    sentence-transformers (no internet required after initial download).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None  # lazy
        self._model_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, text: str) -> List[float]:
        """Embed a single string and return a float list."""
        return self._model_instance().encode(text, convert_to_numpy=True).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of strings; more efficient than calling embed() in a loop."""
        if not texts:
            return []
        vecs = self._model_instance().encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=64,
        )
        return [v.tolist() for v in vecs]

    def warm_up(self) -> None:
        """Load the embedding model once without performing a real query."""
        self._model_instance()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _model_instance(self):
        if self._model is not None:
            return self._model

        with self._model_lock:
            if self._model is not None:
                return self._model

            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required: pip install sentence-transformers"
                )

            logger.info("Loading embedding model '%s' …", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            logger.info("Model loaded.")

        return self._model

    @property
    def embedding_dim(self) -> int:
        """Return the output vector dimensionality."""
        return self._model_instance().get_sentence_embedding_dimension()

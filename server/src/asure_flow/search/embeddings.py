"""Embedding engine — local sentence-transformers with LLM API fallback."""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Generates text embeddings for semantic search.

    Tries sentence-transformers locally first, then falls back to the
    configured LiteLLM provider's embedding endpoint.
    """

    def __init__(self) -> None:
        self._local_model = None
        self._dim: int = 384  # default for all-MiniLM-L6-v2
        self._mode: str = "unavailable"  # "local" | "api" | "unavailable"

    @property
    def available(self) -> bool:
        return self._mode != "unavailable"

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def dimension(self) -> int:
        return self._dim

    async def load(self) -> None:
        """Load the embedding backend — try local model first, then API."""
        loop = asyncio.get_event_loop()

        # Try local sentence-transformers
        try:
            self._local_model = await loop.run_in_executor(None, self._load_local)
            self._mode = "local"
            logger.info("Embedding engine ready (local, dim=%d)", self._dim)
            return
        except ImportError:
            logger.info("sentence-transformers not installed, trying API fallback")
        except Exception:
            logger.warning("Failed to load local embedding model", exc_info=True)

        # Try API fallback via litellm
        if await self._check_api_available():
            self._mode = "api"
            logger.info("Embedding engine ready (API fallback)")
        else:
            logger.info("No embedding backend available — search will use substring matching")

    def _load_local(self):
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        self._dim = model.get_sentence_embedding_dimension()
        return model

    async def _check_api_available(self) -> bool:
        """Check if litellm embedding works with the configured providers."""
        try:
            import litellm

            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: litellm.embedding(
                    model="text-embedding-3-small",
                    input=["test"],
                ),
            )
            self._dim = len(result.data[0]["embedding"])
            return True
        except Exception:
            return False

    async def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts. Returns (N, dim) float32 array."""
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self._dim)

        loop = asyncio.get_event_loop()

        if self._mode == "local":
            return await loop.run_in_executor(None, self._embed_local, texts)
        elif self._mode == "api":
            return await loop.run_in_executor(None, self._embed_api, texts)
        else:
            return np.zeros((len(texts), self._dim), dtype=np.float32)

    async def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text. Returns (dim,) float32 array."""
        result = await self.embed([text])
        return result[0] if len(result) > 0 else np.zeros(self._dim, dtype=np.float32)

    def _embed_local(self, texts: list[str]) -> np.ndarray:
        return self._local_model.encode(texts, normalize_embeddings=True).astype(np.float32)

    def _embed_api(self, texts: list[str]) -> np.ndarray:
        import litellm

        result = litellm.embedding(model="text-embedding-3-small", input=texts)
        vecs = np.array(
            [d["embedding"] for d in result.data],
            dtype=np.float32,
        )
        # Normalize
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vecs / norms


embedding_engine = EmbeddingEngine()

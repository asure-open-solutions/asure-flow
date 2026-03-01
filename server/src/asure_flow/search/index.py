"""Per-session embedding index — NumPy-backed vector store."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from asure_flow.config import settings

logger = logging.getLogger(__name__)


class SessionEmbeddingIndex:
    """Stores and queries embeddings for a single session.

    Persisted as ``{session_id}.embeddings.npz`` next to the session JSON.
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._embeddings: np.ndarray | None = None  # (N, dim)
        self._entry_ids: list[str] = []

    @property
    def _index_path(self) -> Path:
        base = Path(settings.session_dir).expanduser()
        return base / f"{self.session_id}.embeddings.npz"

    @property
    def count(self) -> int:
        return len(self._entry_ids)

    def load(self) -> None:
        """Load embeddings from disk if they exist."""
        path = self._index_path
        if not path.exists():
            return
        try:
            data = np.load(path, allow_pickle=True)
            self._embeddings = data["embeddings"]
            self._entry_ids = data["entry_ids"].tolist()
        except Exception:
            logger.warning("Failed to load embedding index for session %s", self.session_id)

    def save(self) -> None:
        """Persist embeddings to disk."""
        if self._embeddings is None or len(self._entry_ids) == 0:
            return
        try:
            np.savez(
                self._index_path,
                embeddings=self._embeddings,
                entry_ids=np.array(self._entry_ids),
            )
        except Exception:
            logger.warning("Failed to save embedding index for session %s", self.session_id)

    def add(self, entry_id: str, embedding: np.ndarray) -> None:
        """Add an embedding for a transcript entry."""
        vec = embedding.reshape(1, -1)
        if self._embeddings is None:
            self._embeddings = vec
        else:
            self._embeddings = np.vstack([self._embeddings, vec])
        self._entry_ids.append(entry_id)

    def has_entry(self, entry_id: str) -> bool:
        return entry_id in self._entry_ids

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 20,
        min_score: float = 0.25,
    ) -> list[tuple[str, float]]:
        """Find the most similar entries by cosine similarity.

        Returns list of ``(entry_id, score)`` sorted by descending score.
        """
        if self._embeddings is None or len(self._embeddings) == 0:
            return []

        scores = self._embeddings @ query_embedding.reshape(-1)
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            (self._entry_ids[i], float(scores[i]))
            for i in top_indices
            if scores[i] >= min_score
        ]


# ── In-memory index cache ──

_index_cache: dict[str, SessionEmbeddingIndex] = {}


def get_index(session_id: str) -> SessionEmbeddingIndex:
    """Get or load the embedding index for a session."""
    if session_id not in _index_cache:
        idx = SessionEmbeddingIndex(session_id)
        idx.load()
        _index_cache[session_id] = idx
    return _index_cache[session_id]


def clear_index_cache(session_id: str | None = None) -> None:
    """Clear cached indices (e.g. on session delete)."""
    if session_id:
        _index_cache.pop(session_id, None)
    else:
        _index_cache.clear()

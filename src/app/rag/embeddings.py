"""Pluggable embedding backend.

Uses sentence-transformers when installed (dense semantic vectors); otherwise
falls back to a hashing-based bag-of-words vector so the RAG pipeline still runs
fully offline with zero heavyweight downloads.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

from ..config import get_settings

_WORD = re.compile(r"[a-z0-9]+")


class _HashingEmbedder:
    """Deterministic, dependency-free fallback embedder (feature hashing)."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in _WORD.findall(text.lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        # L2 normalize
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class Embedder:
    def __init__(self) -> None:
        settings = get_settings()
        self.backend = "hashing"
        self._model = None
        if settings.rag_backend in {"auto", "faiss"}:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(settings.embedding_model)
                self.backend = "sentence-transformers"
            except Exception:
                self._model = _HashingEmbedder()
        else:
            self._model = _HashingEmbedder()

    @property
    def dim(self) -> int:
        if self.backend == "sentence-transformers":
            return int(self._model.get_sentence_embedding_dimension())
        return self._model.dim

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts)
        return np.asarray(vecs, dtype=np.float32)

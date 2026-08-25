"""Vector store with FAISS backend and a NumPy fallback.

Persists chunk text + metadata alongside the vectors so metadata filtering and
document-level access control can be applied at query time.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:  # pragma: no cover
    _HAS_FAISS = False


@dataclass
class Chunk:
    chunk_id: str
    policy_id: str
    title: str
    text: str
    access: str = "all"  # document-level access tag


class VectorStore:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None
        self._index = None

    def add(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        self.chunks.extend(chunks)
        self._vectors = (
            vectors if self._vectors is None else np.vstack([self._vectors, vectors])
        )

    def build(self) -> None:
        if _HAS_FAISS and self._vectors is not None:
            index = faiss.IndexFlatIP(self.dim)  # cosine (vectors are normalized)
            index.add(self._vectors)
            self._index = index

    def search(self, query_vec: np.ndarray, k: int = 8) -> list[tuple[Chunk, float]]:
        if self._vectors is None or not self.chunks:
            return []
        q = query_vec.reshape(1, -1).astype(np.float32)
        if self._index is not None:
            scores, idx = self._index.search(q, min(k, len(self.chunks)))
            return [(self.chunks[i], float(scores[0][j])) for j, i in enumerate(idx[0])]
        # NumPy fallback: cosine similarity
        sims = (self._vectors @ q.T).ravel()
        top = np.argsort(sims)[::-1][:k]
        return [(self.chunks[i], float(sims[i])) for i in top]

    # ---- persistence ----
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self._vectors)
        with open(directory / "chunks.json", "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in self.chunks], f, indent=2)
        with open(directory / "meta.json", "w", encoding="utf-8") as f:
            json.dump({"dim": self.dim}, f)

    @classmethod
    def load(cls, directory: Path) -> "VectorStore":
        with open(directory / "meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        store = cls(dim=meta["dim"])
        store._vectors = np.load(directory / "vectors.npy")
        with open(directory / "chunks.json", encoding="utf-8") as f:
            store.chunks = [Chunk(**c) for c in json.load(f)]
        store.build()
        return store

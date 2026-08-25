"""RAG retriever: hybrid (dense + lexical) search, metadata filtering,
re-ranking, source validation and document-level access control.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..config import get_settings
from ..schemas import PolicyCitation, Role
from .embeddings import Embedder
from .vector_store import Chunk, VectorStore

_WORD = re.compile(r"[a-z0-9]+")

# which document access tags each role may retrieve
_ROLE_ACCESS: dict[Role, set[str]] = {
    Role.ANALYST: {"all", "analyst"},
    Role.SENIOR_INVESTIGATOR: {"all", "analyst", "restricted"},
    Role.COMPLIANCE: {"all", "analyst", "restricted"},
    Role.SERVICE: {"all"},
}


def _lexical_overlap(query: str, text: str) -> float:
    q = set(_WORD.findall(query.lower()))
    t = set(_WORD.findall(text.lower()))
    if not q:
        return 0.0
    return len(q & t) / len(q)


class PolicyRetriever:
    def __init__(self, store: VectorStore, embedder: Embedder) -> None:
        self.store = store
        self.embedder = embedder

    @classmethod
    def load(cls) -> "PolicyRetriever":
        settings = get_settings()
        store = VectorStore.load(settings.vector_dir)
        return cls(store, Embedder())

    def retrieve(
        self,
        query: str,
        role: Role = Role.ANALYST,
        k: int = 4,
        metadata_filter: dict[str, str] | None = None,
        alpha: float = 0.6,  # weight on dense score for hybrid fusion
    ) -> list[PolicyCitation]:
        allowed = _ROLE_ACCESS.get(role, {"all"})
        qvec = self.embedder.encode([query])[0]
        candidates = self.store.search(qvec, k=max(k * 4, 12))

        scored: list[tuple[Chunk, float]] = []
        for chunk, dense in candidates:
            # document-level access control
            if chunk.access not in allowed:
                continue
            # metadata filtering
            if metadata_filter and metadata_filter.get("policy_id") not in (None, chunk.policy_id):
                continue
            lexical = _lexical_overlap(query, chunk.text)
            hybrid = alpha * _norm(dense) + (1 - alpha) * lexical
            scored.append((chunk, hybrid))

        # re-rank by fused score, source-validate (drop empties), truncate
        scored.sort(key=lambda x: x[1], reverse=True)
        citations: list[PolicyCitation] = []
        for chunk, score in scored[:k]:
            if not chunk.text.strip():
                continue
            citations.append(
                PolicyCitation(
                    policy_id=chunk.policy_id,
                    title=chunk.title,
                    snippet=chunk.text.strip()[:400],
                    score=round(float(score), 4),
                )
            )
        return citations


def _norm(x: float) -> float:
    # cosine/IP scores are already ~[-1,1]; squeeze to [0,1]
    return max(0.0, min(1.0, (x + 1.0) / 2.0))

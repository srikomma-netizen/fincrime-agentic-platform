"""Chunk the compliance-policy corpus, embed it, and persist the vector index.

Parses simple front-matter (policy_id / title / access) from each markdown file,
splits the body into overlapping chunks, embeds them, and saves a VectorStore
for the RAG retriever.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from app.config import get_settings  # noqa: E402
from app.rag.embeddings import Embedder  # noqa: E402
from app.rag.vector_store import Chunk, VectorStore  # noqa: E402

FRONT = re.compile(r"^---\s*(.*?)\s*---\s*(.*)$", re.DOTALL)


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    m = FRONT.match(text)
    if not m:
        return {}, text
    meta_block, body = m.group(1), m.group(2)
    meta = {}
    for line in meta_block.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, body


def chunk_text(text: str, size: int = 480, overlap: int = 80) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i:i + size])
        if chunk.strip():
            chunks.append(chunk)
        i += size - overlap
    return chunks


def main() -> None:
    settings = get_settings()
    embedder = Embedder()
    store = VectorStore(dim=embedder.dim)

    files = sorted(settings.policies_dir.glob("*.md"))
    if not files:
        raise SystemExit(f"No policy files in {settings.policies_dir}")

    all_chunks: list[Chunk] = []
    for path in files:
        meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
        pid = meta.get("policy_id", path.stem)
        title = meta.get("title", path.stem)
        access = meta.get("access", "all")
        for j, ch in enumerate(chunk_text(body)):
            all_chunks.append(Chunk(
                chunk_id=f"{pid}#{j}", policy_id=pid, title=title,
                text=ch, access=access,
            ))

    vectors = embedder.encode([c.text for c in all_chunks])
    store.add(vectors, all_chunks)
    store.build()
    store.save(settings.vector_dir)
    print(f"indexed {len(all_chunks)} chunks from {len(files)} policies "
          f"({embedder.backend}) -> {settings.vector_dir}")


if __name__ == "__main__":
    main()

"""
Chunks the markdown policy docs and loads them into policy_chunks.
Run after docker-compose up and pip install -r requirements.txt:

    python scripts/ingest_policies.py

Chunking strategy: split on markdown headings/paragraphs (section-based),
not fixed-token windows. Policy docs are short and each section is a
self-contained rule (e.g. "final sale items"), so splitting mid-section
would separate a rule from its own qualifying condition — worse for
retrieval than the extra complexity of header-aware splitting.
"""

import asyncio
import glob
import os

from app.data.db import AsyncSessionLocal
from app.data.models import PolicyChunk
from app.providers.factory import get_embedding_provider
from app.services.chunking import chunk_document

POLICIES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "policies")


async def ingest():
    embedding_provider = get_embedding_provider()
    doc_paths = sorted(glob.glob(os.path.join(POLICIES_DIR, "*.md")))

    all_chunks: list[dict] = []
    for path in doc_paths:
        with open(path) as f:
            text = f.read()
        title = text.split("\n", 1)[0].lstrip("# ").strip()
        for i, chunk_text in enumerate(chunk_document(text)):
            all_chunks.append(
                {
                    "doc_title": title,
                    "chunk_index": i,
                    "chunk_text": chunk_text,
                    "token_count": len(chunk_text.split()),
                }
            )

    # batched embedding call — all chunks at once, not one API/model call
    # per chunk. This is the throughput optimization mentioned in the
    # design checklist, applied where it actually matters (bulk ingestion).
    texts = [c["chunk_text"] for c in all_chunks]
    vectors = await embedding_provider.embed(texts)

    async with AsyncSessionLocal() as session:
        for chunk, vector in zip(all_chunks, vectors):
            session.add(
                PolicyChunk(
                    doc_title=chunk["doc_title"],
                    chunk_index=chunk["chunk_index"],
                    chunk_text=chunk["chunk_text"],
                    embedding=vector,
                    token_count=chunk["token_count"],
                )
            )
        await session.commit()

    print(f"Ingested {len(all_chunks)} chunks from {len(doc_paths)} documents.")


if __name__ == "__main__":
    asyncio.run(ingest())

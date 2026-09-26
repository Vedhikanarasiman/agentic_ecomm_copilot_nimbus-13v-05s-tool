"""
Shows the top-N most similar policy chunks for a query, WITH their actual
similarity scores, ignoring the threshold entirely. This is how you tell
the difference between "the threshold is miscalibrated" (correct chunk
scores 0.51, just under 0.55) and "retrieval is actually broken" (correct
chunk scores 0.15, nowhere close).

Usage:
    python eval/inspect_retrieval.py "How long does standard shipping take?"
"""

import asyncio
import sys

from sqlalchemy import select

from app.core.config import get_settings
from app.data.db import AsyncSessionLocal
from app.data.models import PolicyChunk
from app.providers.factory import get_embedding_provider


async def inspect(query: str, top_n: int = 8):
    threshold = get_settings().rag_similarity_threshold
    embedding_provider = get_embedding_provider()
    [query_vector] = await embedding_provider.embed([query])

    distance_expr = PolicyChunk.embedding.cosine_distance(query_vector)
    async with AsyncSessionLocal() as db:
        stmt = select(PolicyChunk, distance_expr.label("distance")).order_by(distance_expr).limit(top_n)
        rows = (await db.execute(stmt)).all()

    print(f"\nQuery: {query!r}\n")
    print(f"{'similarity':>10}  doc_title")
    print("-" * 60)
    for chunk, distance in rows:
        similarity = 1 - distance
        marker = f" <-- current threshold is {threshold}" if similarity < threshold else ""
        print(f"{similarity:>10.4f}  {chunk.doc_title}{marker}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python eval/inspect_retrieval.py "your query here"')
        sys.exit(1)
    asyncio.run(inspect(sys.argv[1]))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.data.models import PolicyChunk
from app.providers.base import EmbeddingProvider, LLMProvider

SYSTEM_PROMPT = (
    "You are a support assistant for Nimbus Electronics. Answer only using "
    "the policy excerpts provided below. If the excerpts don't clearly answer "
    "the question, say you're not certain rather than guessing. Never invent "
    "policy details, dates, or fees that aren't in the excerpts. Keep answers "
    "under 4 sentences."
)


async def retrieve(
    query: str,
    embedding_provider: EmbeddingProvider,
    db: AsyncSession,
    top_k: int,
    similarity_threshold: float,
) -> list[dict]:
    [query_vector] = await embedding_provider.embed([query])

    # cosine_distance ranges 0 (identical) to 2 (opposite); similarity = 1 - distance
    distance_expr = PolicyChunk.embedding.cosine_distance(query_vector)
    stmt = select(PolicyChunk, distance_expr.label("distance")).order_by(distance_expr).limit(top_k)
    rows = (await db.execute(stmt)).all()

    results = []
    for chunk, distance in rows:
        similarity = 1 - distance
        if similarity >= similarity_threshold:
            results.append(
                {
                    "doc_title": chunk.doc_title,
                    "chunk_text": chunk.chunk_text,
                    "similarity": round(float(similarity), 4),
                }
            )
    # Below-threshold chunks are dropped, not returned — an empty list here
    # is the deterministic signal the router uses to escalate instead of
    # letting the LLM answer from weak or irrelevant context.
    return results


async def generate_rag_answer(
    query: str,
    chunks: list[dict],
    llm_provider: LLMProvider,
    max_tokens: int,
    history: list[dict] | None = None,
) -> dict:
    context = "\n\n".join(f"[{c['doc_title']}]\n{c['chunk_text']}" for c in chunks)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history or [])
    messages.append(
        {"role": "user", "content": f"Policy excerpts:\n{context}\n\nCustomer question: {query}"}
    )
    response = await llm_provider.generate(messages, max_tokens=max_tokens, temperature=0.2)
    return {
        "answer": response["content"],
        "sources": [c["doc_title"] for c in chunks],
        "usage": response.get("usage"),
    }

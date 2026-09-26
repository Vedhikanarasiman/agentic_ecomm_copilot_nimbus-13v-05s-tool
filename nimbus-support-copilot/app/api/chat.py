import time
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.logging import get_logger, new_trace_id, trace_id_var
from app.core.security import get_current_customer_id
from app.data import cache
from app.data.db import get_db
from app.data.models import ConversationTurn
from app.guardrails.pii import redact_pii
from app.providers.base import EmbeddingProvider, LLMProvider
from app.providers.factory import get_embedding_provider, get_llm_provider
from app.services.memory_graph import to_plain_dicts
from app.services.router_agent import router_graph

router = APIRouter(prefix="/chat", tags=["chat"])
logger = get_logger(__name__)


class ChatRequest(BaseModel):
    session_id: uuid.UUID
    message: str
    turn_index: int


class ChatResponse(BaseModel):
    answer: str
    route: str
    sources: list[str]
    trace_id: str
    latency_ms: int
    cached: bool = False
    token_usage: list[dict] = []
    escalation_reason: str | None = None


@router.post("", response_model=ChatResponse)
async def chat(
    request: Request,
    req: ChatRequest,
    customer_id: str = Depends(get_current_customer_id),
    db: AsyncSession = Depends(get_db),
    llm_provider: LLMProvider = Depends(get_llm_provider),
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
    settings: Settings = Depends(get_settings),
):
    trace_id = new_trace_id()
    trace_id_var.set(trace_id)
    start = time.monotonic()

    # Only RAG-path answers are cache candidates — order/customer-specific
    # answers must never be served to a different customer from cache.
    cached = await cache.get_cached_answer(req.message)
    if cached and cached.get("route") == "rag":
        latency_ms = int((time.monotonic() - start) * 1000)
        logger.info("chat_response", extra={"event": {"trace_id": trace_id, "cached": True}})
        return ChatResponse(**cached, trace_id=trace_id, latency_ms=latency_ms, cached=True)

    memory_graph = request.app.state.memory_graph
    thread_config = {"configurable": {"thread_id": str(req.session_id)}}

    # Storing the user message and reading back history happen in one
    # call — add_messages both persists the new message and returns the
    # full accumulated list so far via the checkpointer.
    memory_state = await memory_graph.ainvoke(
        {"messages": [{"role": "user", "content": req.message}]},
        config=thread_config,
    )
    # Exclude the message we just added — that's `req.message` already,
    # passed separately as `query` below. `history` here means PRIOR turns.
    history = to_plain_dicts(memory_state["messages"][:-1])

    result = await router_graph.ainvoke(
        {
            "query": req.message,
            "session_id": str(req.session_id),
            "trace_id": trace_id,
            "customer_id": customer_id,
            "history": history,
        },
        config={
            "configurable": {
                "llm_provider": llm_provider,
                "embedding_provider": embedding_provider,
                "db": db,
                "settings": settings,
            }
        },
    )

    latency_ms = int((time.monotonic() - start) * 1000)
    response = ChatResponse(
        answer=result["answer"],
        route=result["route"],
        sources=result.get("sources", []),
        trace_id=trace_id,
        latency_ms=latency_ms,
        token_usage=result.get("token_usage", []),
        escalation_reason=result.get("escalation_reason"),
    )

    # Store the assistant's reply too — this is what makes it visible to
    # the NEXT call's history, not just this one's.
    await memory_graph.ainvoke(
        {"messages": [{"role": "assistant", "content": response.answer}]},
        config=thread_config,
    )

    if response.route == "rag":
        await cache.set_cached_answer(
            req.message,
            {"answer": response.answer, "route": response.route, "sources": response.sources},
            ttl_seconds=settings.cache_ttl_seconds,
        )

    db.add_all(
        [
            ConversationTurn(
                session_id=req.session_id,
                trace_id=uuid.UUID(trace_id),
                customer_id=customer_id,
                turn_index=req.turn_index,
                role="user",
                content=redact_pii(req.message),
                route_taken=None,
                created_at=datetime.now(timezone.utc),
            ),
            ConversationTurn(
                session_id=req.session_id,
                trace_id=uuid.UUID(trace_id),
                customer_id=customer_id,
                turn_index=req.turn_index + 1,
                role="assistant",
                content=redact_pii(response.answer),
                route_taken=response.route,
                created_at=datetime.now(timezone.utc),
            ),
        ]
    )
    await db.commit()

    logger.info(
        "chat_response",
        extra={
            "event": {
                "trace_id": trace_id,
                "route": response.route,
                "latency_ms": latency_ms,
                "cached": False,
                "token_usage": response.token_usage,
            }
        },
    )
    return response

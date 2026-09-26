from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.api import auth, chat
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger, new_trace_id, trace_id_var
from app.providers.factory import get_embedding_provider, get_llm_provider
from app.services.memory_graph import build_memory_graph, checkpoint_conn_string

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up providers here, at a time we control, instead of lazily on
    # whichever request happens to be first. The embedding model load is
    # slow (10s+, more on Windows where antivirus commonly scans a
    # freshly-touched model file) — leaving it lazy meant an unlucky user
    # could hit a 10-60+ second delay with no warning. See failure-log #12.
    logger.info("startup_warmup_begin")
    get_embedding_provider()
    get_llm_provider()
    logger.info("startup_warmup_complete")

    # Real conversation-memory persistence — see memory_graph.py for why
    # this is a separate graph from router_graph, not the same one.
    settings = get_settings()
    conn_string = checkpoint_conn_string(settings.database_url)
    checkpointer_cm = AsyncPostgresSaver.from_conn_string(conn_string)
    checkpointer = await checkpointer_cm.__aenter__()
    await checkpointer.setup()  # idempotent — only creates tables if missing
    app.state.memory_graph = build_memory_graph(checkpointer)
    logger.info("memory_checkpointer_ready")

    yield

    await checkpointer_cm.__aexit__(None, None, None)


app = FastAPI(title="Nimbus Support Copilot", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def assign_trace_id(request: Request, call_next):
    # every request gets a trace_id even before the chat graph runs one of
    # its own — this one covers auth failures, validation errors, etc.
    # that never reach the graph.
    trace_id_var.set(new_trace_id())
    response = await call_next(request)
    response.headers["X-Trace-Id"] = trace_id_var.get()
    return response


app.include_router(auth.router)
app.include_router(chat.router)


@app.get("/health")
async def health():
    return {"status": "ok"}

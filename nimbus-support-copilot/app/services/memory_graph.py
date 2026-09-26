"""
A second, minimal graph — not the router graph — whose only purpose is
message persistence across separate /chat requests, via LangGraph's own
checkpointer instead of the hand-rolled SQL fetch memory_service.py used
to do.

Why a SEPARATE graph from router_agent.router_graph, not the same one:
router_graph has per-turn fields (token_usage, route, answer) that use an
accumulating (operator.add) reducer. LangGraph's checkpointer persists the
ENTIRE state schema of whatever graph it's attached to — there's no way to
say "persist this field but not that one." Attaching a checkpointer
directly to router_graph would mean token_usage keeps adding onto itself
forever across a whole conversation instead of resetting each turn,
silently making every token/cost number in the API response wrong after
the first message. See docs/failure-log.md entry #13.

Message history SHOULD accumulate forever within a conversation — that's
the entire point. Splitting the two needs into two graphs, one
checkpointed and one not, is what keeps both correct at once.
"""

from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


class MemoryState(TypedDict):
    messages: Annotated[list, add_messages]


def _noop(state: MemoryState) -> dict:
    # Does nothing. The graph exists only so the checkpointer has
    # something to attach state to — the actual storing/loading happens
    # through the add_messages reducer itself on every .ainvoke() call.
    return {}


def build_memory_graph(checkpointer):
    graph = StateGraph(MemoryState)
    graph.add_node("noop", _noop)
    graph.add_edge(START, "noop")
    graph.add_edge("noop", END)
    return graph.compile(checkpointer=checkpointer)


def checkpoint_conn_string(database_url: str) -> str:
    """AsyncPostgresSaver uses psycopg, which needs a plain postgresql://
    URL — our app's DATABASE_URL uses SQLAlchemy's +asyncpg dialect
    suffix, which psycopg doesn't understand. Strip it here rather than
    maintain two separate connection strings in .env."""
    return database_url.replace("postgresql+asyncpg://", "postgresql://")


_LC_TYPE_TO_ROLE = {"human": "user", "ai": "assistant", "system": "system"}
DEFAULT_HISTORY_TURNS = 4  # last 2 exchanges — same fixed-window reasoning as before


def to_plain_dicts(messages: list, limit: int = DEFAULT_HISTORY_TURNS) -> list[dict]:
    """Converts LangGraph's internal message objects back into the plain
    {"role", "content"} dicts LLMProvider.generate() expects, and applies
    the fixed-window cap — unbounded history still isn't used here, same
    reasoning as before: token cost and latency both grow with it."""
    recent = messages[-limit:] if limit else messages
    return [{"role": _LC_TYPE_TO_ROLE.get(m.type, m.type), "content": m.content} for m in recent]

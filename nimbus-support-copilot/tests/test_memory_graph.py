from langgraph.checkpoint.memory import InMemorySaver

from app.services.memory_graph import build_memory_graph, checkpoint_conn_string, to_plain_dicts


def _build_test_graph():
    # InMemorySaver behaves like the real Postgres checkpointer for
    # testing purposes — same interface, RAM instead of a real database.
    # No Postgres needed to prove this logic is correct.
    return build_memory_graph(InMemorySaver())


async def test_messages_persist_across_separate_invocations():
    graph = _build_test_graph()
    thread = {"configurable": {"thread_id": "session-abc"}}

    await graph.ainvoke({"messages": [{"role": "user", "content": "What's your return policy?"}]}, config=thread)
    await graph.ainvoke({"messages": [{"role": "assistant", "content": "30 days."}]}, config=thread)
    result = await graph.ainvoke({"messages": [{"role": "user", "content": "And the fee?"}]}, config=thread)

    # all three messages across three SEPARATE calls should be present —
    # this is the actual proof that persistence works, not just that a
    # single call's input got echoed back
    assert len(result["messages"]) == 3
    assert result["messages"][0].content == "What's your return policy?"
    assert result["messages"][1].content == "30 days."
    assert result["messages"][2].content == "And the fee?"


async def test_different_threads_do_not_share_history():
    graph = _build_test_graph()
    thread_a = {"configurable": {"thread_id": "session-a"}}
    thread_b = {"configurable": {"thread_id": "session-b"}}

    await graph.ainvoke({"messages": [{"role": "user", "content": "message in session A"}]}, config=thread_a)
    result_b = await graph.ainvoke({"messages": [{"role": "user", "content": "message in session B"}]}, config=thread_b)

    # session B must only see its own message — a leak here would mean
    # one customer's conversation bleeding into another's, the same class
    # of bug as the order-authorization gap in failure-log #5
    assert len(result_b["messages"]) == 1
    assert result_b["messages"][0].content == "message in session B"


def test_to_plain_dicts_converts_and_applies_window():
    class FakeMessage:
        def __init__(self, type_, content):
            self.type = type_
            self.content = content

    messages = [
        FakeMessage("human", "msg1"),
        FakeMessage("ai", "msg2"),
        FakeMessage("human", "msg3"),
        FakeMessage("ai", "msg4"),
        FakeMessage("human", "msg5"),
    ]
    result = to_plain_dicts(messages, limit=4)

    assert len(result) == 4  # window applied — oldest message dropped
    assert result[0] == {"role": "assistant", "content": "msg2"}
    assert result[-1] == {"role": "user", "content": "msg5"}


def test_checkpoint_conn_string_strips_asyncpg_dialect():
    assert checkpoint_conn_string("postgresql+asyncpg://u:p@host:5432/db") == "postgresql://u:p@host:5432/db"

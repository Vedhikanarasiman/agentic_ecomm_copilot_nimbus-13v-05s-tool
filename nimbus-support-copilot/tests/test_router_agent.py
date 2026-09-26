from dataclasses import dataclass
from datetime import datetime, timezone

from app.services import order_service, rag_service
from app.services.router_agent import router_graph
from tests.fake_settings import FakeSettings
from tests.fakes import FakeEmbeddingProvider, FakeLLMProvider


@dataclass
class FakeOrder:
    order_id: str
    customer_id: str
    product_name: str = "Nimbus Watch SE"
    status: str = "delivered"
    order_date: datetime = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ship_date: datetime | None = None
    delivery_date: datetime | None = None
    tracking_number: str | None = None
    amount: float = 199.99


def _config(llm_provider, embedding_provider=None):
    return {
        "configurable": {
            "llm_provider": llm_provider,
            "embedding_provider": embedding_provider or FakeEmbeddingProvider(),
            "db": None,  # unused once order_service/rag_service are monkeypatched
            "settings": FakeSettings(),
        }
    }


async def test_order_id_in_message_skips_classification_llm_call(monkeypatch):
    """Proves the cost-saving deterministic bypass actually works: an order
    ID in the message should route to order_status WITHOUT ever calling the
    classification LLM — only the summary-generation call should happen."""
    order = FakeOrder(order_id="ORD-00042", customer_id="CUST-0001")

    async def fake_lookup(order_id, customer_id, db):
        assert order_id == "ORD-00042"
        assert customer_id == "CUST-0001"
        return order

    monkeypatch.setattr(order_service, "lookup_order", fake_lookup)

    llm = FakeLLMProvider([{"content": "Your order was delivered.", "usage": {}}])
    result = await router_graph.ainvoke(
        {"query": "Where is my order ORD-00042?", "customer_id": "CUST-0001"},
        config=_config(llm),
    )

    assert result["route"] == "order_status"
    assert len(llm.calls) == 1  # only the summary call — classification was skipped
    assert llm.calls[0]["reasoning_effort"] == "low"


async def test_order_not_found_or_not_owned_escalates(monkeypatch):
    """This is the authorization-fix test: an order that doesn't belong to
    the requesting customer must escalate, not leak another customer's data."""

    async def fake_lookup(order_id, customer_id, db):
        return None  # order_service returns None uniformly for "doesn't exist"
        # and "exists but isn't this customer's" — see order_service.py

    monkeypatch.setattr(order_service, "lookup_order", fake_lookup)

    llm = FakeLLMProvider([])  # no LLM call should happen at all on this path
    result = await router_graph.ainvoke(
        {"query": "Where is my order ORD-99999?", "customer_id": "CUST-0001"},
        config=_config(llm),
    )

    assert result["route"] == "escalate"
    assert "handing this off" in result["answer"]
    assert len(llm.calls) == 0  # escalate_node is deterministic, zero cost


async def test_rag_low_confidence_escalates(monkeypatch):
    """No chunks above the similarity threshold should escalate instead of
    letting the model answer from weak/irrelevant context."""

    async def fake_retrieve(query, embedding_provider, db, top_k, similarity_threshold):
        return []  # nothing cleared the threshold

    monkeypatch.setattr(rag_service, "retrieve", fake_retrieve)

    classify_response = {"content": '{"route": "rag"}', "usage": {}}
    llm = FakeLLMProvider([classify_response])  # only classification — rag generation never runs
    result = await router_graph.ainvoke(
        {"query": "What's your policy on something totally unrelated?", "customer_id": "CUST-0001"},
        config=_config(llm),
    )

    assert result["route"] == "escalate"
    assert len(llm.calls) == 1  # generation was correctly skipped, not wastefully called


async def test_rag_high_confidence_returns_grounded_answer(monkeypatch):
    async def fake_retrieve(query, embedding_provider, db, top_k, similarity_threshold):
        return [{"doc_title": "Returns Policy", "chunk_text": "30 day window.", "similarity": 0.9}]

    monkeypatch.setattr(rag_service, "retrieve", fake_retrieve)

    classify_response = {"content": '{"route": "rag"}', "usage": {}}
    generation_response = {"content": "You have a 30-day return window.", "usage": {}}
    llm = FakeLLMProvider([classify_response, generation_response])

    result = await router_graph.ainvoke(
        {"query": "What's your return window?", "customer_id": "CUST-0001"}, config=_config(llm)
    )

    assert result["route"] == "rag"
    assert result["answer"] == "You have a 30-day return window."
    assert result["sources"] == ["Returns Policy"]


async def test_malformed_classifier_json_fails_safe_to_escalate():
    """If the classifier ever returns something that isn't valid {"route": ...}
    JSON, the system should fail toward a human, not guess or crash."""
    llm = FakeLLMProvider([{"content": "not valid json at all", "usage": {}}])
    result = await router_graph.ainvoke(
        {"query": "Some ambiguous message", "customer_id": "CUST-0001"}, config=_config(llm)
    )
    assert result["route"] == "escalate"


async def test_history_reaches_the_classification_call(monkeypatch):
    """Proves conversation history is actually threaded into the classifier's
    messages, not just accepted and silently dropped."""
    history = [
        {"role": "user", "content": "What's your return policy for laptops?"},
        {"role": "assistant", "content": "You have a 30-day window."},
    ]
    llm = FakeLLMProvider([{"content": '{"route": "rag"}', "usage": {}}])

    async def fake_retrieve(query, embedding_provider, db, top_k, similarity_threshold):
        return []  # route content doesn't matter for this test — only checking messages sent

    monkeypatch.setattr(rag_service, "retrieve", fake_retrieve)

    await router_graph.ainvoke(
        {"query": "And what about international?", "customer_id": "CUST-0001", "history": history},
        config=_config(llm),
    )

    sent_messages = llm.calls[0]["messages"]
    assert history[0] in sent_messages
    assert history[1] in sent_messages
    # history must come between the system prompt and the new query, not after it
    assert sent_messages.index(history[1]) < len(sent_messages) - 1


async def test_history_reaches_rag_generation(monkeypatch):
    """Proves conversation history is threaded into the RAG generation call
    too, not just the classifier."""
    history = [{"role": "user", "content": "What's your return policy?"}]

    async def fake_retrieve(query, embedding_provider, db, top_k, similarity_threshold):
        return [{"doc_title": "General Returns Policy", "chunk_text": "30 days.", "similarity": 0.9}]

    monkeypatch.setattr(rag_service, "retrieve", fake_retrieve)

    llm = FakeLLMProvider(
        [
            {"content": '{"route": "rag"}', "usage": {}},
            {"content": "The fee is 10% if opened.", "usage": {}},
        ]
    )
    await router_graph.ainvoke(
        {"query": "What's the fee?", "customer_id": "CUST-0001", "history": history}, config=_config(llm)
    )

    # second call is the RAG generation call — history should be in its messages
    generation_messages = llm.calls[1]["messages"]
    assert history[0] in generation_messages

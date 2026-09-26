from unittest.mock import AsyncMock

from app.services import order_service, rag_service
from eval.run_eval import run_case
from tests.fakes import FakeEmbeddingProvider, FakeLLMProvider


async def test_run_case_scores_correct_rag_routing_and_retrieval_hit(monkeypatch):
    async def fake_retrieve(query, embedding_provider, db, top_k, similarity_threshold):
        return [{"doc_title": "General Returns Policy", "chunk_text": "30 days.", "similarity": 0.9}]

    monkeypatch.setattr(rag_service, "retrieve", fake_retrieve)

    llm = FakeLLMProvider(
        [
            {"content": '{"route": "rag"}', "usage": {}},  # classify
            {"content": "You have 30 days.", "usage": {}},  # rag generation
            {  # groundedness judge
                "content": '{"label": "grounded", "reasoning": "matches context"}',
                "usage": {},
            },
        ]
    )
    case = {
        "id": "rag-01",
        "query": "What's your return window?",
        "expected_route": "rag",
        "expected_source": "General Returns Policy",
    }
    result = await run_case(case, llm, FakeEmbeddingProvider(), db=None)

    assert result["routing_correct"] is True
    assert result["retrieval_hit"] is True
    assert result["groundedness"]["label"] == "grounded"


async def test_run_case_flags_misrouting(monkeypatch):
    async def fake_retrieve(query, embedding_provider, db, top_k, similarity_threshold):
        return []  # below threshold -> rag_node escalates

    monkeypatch.setattr(rag_service, "retrieve", fake_retrieve)

    llm = FakeLLMProvider([{"content": '{"route": "rag"}', "usage": {}}])
    case = {"id": "rag-x", "query": "unrelated question", "expected_route": "rag"}
    result = await run_case(case, llm, FakeEmbeddingProvider(), db=None)

    # classifier said rag, but low confidence forced escalate — expected_route
    # was rag, so this should be flagged as NOT routing_correct
    assert result["actual_route"] == "escalate"
    assert result["routing_correct"] is False


async def test_run_case_handles_a_crashing_case_without_stopping_the_run(monkeypatch):
    async def fake_lookup(order_id, customer_id, db):
        raise RuntimeError("simulated DB failure")

    monkeypatch.setattr(order_service, "lookup_order", fake_lookup)

    llm = FakeLLMProvider([])
    case = {"id": "order-crash", "query": "Where is my order ORD-00001?", "expected_route": "order_status"}
    result = await run_case(case, llm, FakeEmbeddingProvider(), db=None)

    assert "error" in result
    assert "RuntimeError" in result["error"]


async def test_run_case_survives_a_judge_failure_without_crashing(monkeypatch):
    """Regression test for the real bug found in eval run #1: a judge call
    running out of tokens on case 19 crashed the whole 29-case run and lost
    every prior result. A judge failure must produce a judge_error entry for
    THAT case only, not take down the batch."""

    async def fake_retrieve(query, embedding_provider, db, top_k, similarity_threshold):
        return [{"doc_title": "Some Policy", "chunk_text": "some text", "similarity": 0.9}]

    monkeypatch.setattr(rag_service, "retrieve", fake_retrieve)

    # classify succeeds, rag generation succeeds, but the judge call itself
    # raises — simulating exactly what happened against the real Groq API
    llm = FakeLLMProvider(
        [
            {"content": '{"route": "rag"}', "usage": {}},
            {"content": "Some answer.", "usage": {}},
        ]
    )

    async def failing_generate(*args, **kwargs):
        raise RuntimeError("simulated Groq 400 — max tokens reached before valid JSON")

    # only the third call (the judge's) should fail — patch after the first
    # two scripted responses are exhausted by wrapping the fake's generate
    original_generate = llm.generate

    async def generate_with_judge_failure(*args, **kwargs):
        if not llm._responses:  # scripted responses exhausted -> this is the judge call
            raise RuntimeError("simulated Groq 400 — max tokens reached before valid JSON")
        return await original_generate(*args, **kwargs)

    llm.generate = generate_with_judge_failure

    case = {"id": "rag-judge-fail", "query": "some question", "expected_route": "rag"}
    result = await run_case(case, llm, FakeEmbeddingProvider(), db=None)

    assert "error" not in result  # the case itself succeeded
    assert result["routing_correct"] is True
    assert result["groundedness"]["label"] == "judge_error"

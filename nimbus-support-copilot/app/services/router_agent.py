import json
import operator
from typing import Annotated, Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from app.services import order_service, rag_service

CLASSIFY_SYSTEM_PROMPT = (
    "Classify the customer's message into exactly one category. Respond with "
    'JSON only: {"route": "rag" | "order_status" | "escalate"}.\n\n'
    "- rag: any question that references a product, order, policy, fee, deadline, "
    "or return/refund/warranty/shipping topic — even if the customer sounds "
    "frustrated or upset. Tone alone is not a reason to escalate; if the question "
    "could plausibly be answered from a policy document, choose rag.\n"
    "- order_status: questions about a specific order's status, tracking, or delivery\n"
    "- escalate: ONLY for cases a policy document cannot resolve — the customer "
    "explicitly asks for a human, a fraud or legal accusation, a dispute over facts "
    "not in the system (e.g. contesting final-sale status), or the same unresolved "
    "issue repeated more than twice in the conversation.\n\n"
    "Examples:\n"
    'Q: "What\'s your return window for laptops?" -> {"route": "rag"}\n'
    'Q: "When will my order arrive?" -> {"route": "order_status"}\n'
    'Q: "My package arrived damaged 5 days ago, can I still get it replaced free?" '
    '-> {"route": "rag"}  # sounds urgent, but it\'s a clean policy lookup — a specific '
    "deadline in a document answers it, no human judgment required\n"
    'Q: "This is the third time I\'m asking, I want a refund now" -> {"route": "escalate"}  '
    "# repeated + unresolved, not a fresh policy question\n"
    'Q: "I want to speak to a person" -> {"route": "escalate"}'
)


class RouterState(TypedDict, total=False):
    query: str
    session_id: str
    trace_id: str
    customer_id: str
    history: list[dict]
    route: str
    order_id: str | None
    chunks: list[dict]
    answer: str
    sources: list[str]
    escalation_reason: str | None
    token_usage: Annotated[list[dict], operator.add]


def _deps(config: RunnableConfig) -> dict[str, Any]:
    return config["configurable"]


async def classify_intent(state: RouterState, config: RunnableConfig) -> dict:
    deps = _deps(config)
    query = state["query"]

    # Deterministic override: an order ID in the message means order_status,
    # full stop — no LLM call needed to "decide" what a regex already knows.
    # Cheaper, faster, and removes a classification failure mode entirely.
    #
    # Deliberately NOT checked against history here: reusing an order ID
    # mentioned in an earlier turn could misfire on an unrelated follow-up
    # ("what's your return policy?" after discussing ORD-00001 shouldn't
    # force order_status). Doing this well needs the classifier to judge
    # topical continuity first, which isn't validated by the eval set yet —
    # left as a known, documented limitation rather than shipped unvalidated.
    order_id = order_service.extract_order_id(query)
    if order_id:
        return {"route": "order_status", "order_id": order_id}

    settings = deps["settings"]
    messages = [{"role": "system", "content": CLASSIFY_SYSTEM_PROMPT}]
    messages.extend(state.get("history", []))
    messages.append({"role": "user", "content": query})
    response = await deps["llm_provider"].generate(
        messages,
        max_tokens=settings.router_max_tokens,
        temperature=0.0,
        response_format={"type": "json_object"},
        reasoning_effort="low",
    )
    try:
        parsed = json.loads(response["content"])
        route = parsed.get("route", "escalate")
    except (json.JSONDecodeError, AttributeError):
        route = "escalate"  # malformed classifier output fails safe, not silently
        usage = response.get("usage") or {}
        return {
            "route": route,
            "order_id": None,
            "escalation_reason": "Classifier returned invalid JSON — failed safe to escalate",
            "token_usage": [{"node": "classify_intent", **usage}],
        }
    if route not in ("rag", "order_status", "escalate"):
        usage = response.get("usage") or {}
        return {
            "route": "escalate",
            "order_id": None,
            "escalation_reason": f"Classifier returned an unrecognized route value: {route!r}",
            "token_usage": [{"node": "classify_intent", **usage}],
        }
    usage = response.get("usage") or {}
    update = {
        "route": route,
        "order_id": None,
        "token_usage": [{"node": "classify_intent", **usage}],
    }
    if route == "escalate":
        update["escalation_reason"] = "Classifier judged this as requiring human handling directly"
    return update


async def rag_node(state: RouterState, config: RunnableConfig) -> dict:
    deps = _deps(config)
    settings = deps["settings"]
    chunks = await rag_service.retrieve(
        state["query"],
        deps["embedding_provider"],
        deps["db"],
        top_k=settings.rag_top_k,
        similarity_threshold=settings.rag_similarity_threshold,
    )
    if not chunks:
        return {
            "route": "escalate",
            "escalation_reason": "No policy content matched above the confidence threshold",
        }
    result = await rag_service.generate_rag_answer(
        state["query"],
        chunks,
        deps["llm_provider"],
        max_tokens=settings.rag_max_tokens,
        history=state.get("history", []),
    )
    return {
        "chunks": chunks,
        "answer": result["answer"],
        "sources": result["sources"],
        "token_usage": [{"node": "rag_node", **(result.get("usage") or {})}],
    }


async def order_status_node(state: RouterState, config: RunnableConfig) -> dict:
    deps = _deps(config)
    order_id = state.get("order_id")
    if not order_id:
        return {
            "answer": "Could you share your order number (format ORD-XXXXX) so I can look that up?",
            "sources": [],
        }
    order = await order_service.lookup_order(order_id, state["customer_id"], deps["db"])
    if not order:
        return {
            "route": "escalate",
            "escalation_reason": f"Order {order_id} not found for this customer",
        }
    context = order_service.order_to_context(order)
    settings = deps["settings"]
    response = await deps["llm_provider"].generate(
        [
            {
                "role": "system",
                "content": (
                    "Summarize this order's status for the customer in 1-2 sentences. "
                    "Use only the fields given. Do not invent dates or a delivery estimate "
                    "that isn't in the data."
                ),
            },
            {"role": "user", "content": json.dumps(context)},
        ],
        max_tokens=settings.tool_call_max_tokens,
        temperature=0.1,
        reasoning_effort="low",
    )
    return {
        "answer": response["content"],
        "sources": [f"order:{order_id}"],
        "token_usage": [{"node": "order_status_node", **(response.get("usage") or {})}],
    }


async def escalate_node(state: RouterState, config: RunnableConfig) -> dict:
    # Deterministic, templated — an escalation message doesn't benefit
    # from generation, and a fixed template can't hallucinate a promise
    # the human agent hasn't made yet.
    return {
        "answer": (
            "I'm not confident I can resolve this accurately, so I'm handing this off to a "
            "member of our support team. They'll follow up within 24 hours."
        ),
        "sources": [],
    }


def _route_after_classify(state: RouterState) -> str:
    return state["route"]


def _route_after_rag(state: RouterState) -> str:
    return "escalate" if state["route"] == "escalate" else "end"


def _route_after_order_status(state: RouterState) -> str:
    return "escalate" if state["route"] == "escalate" else "end"


def build_router_graph():
    graph = StateGraph(RouterState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("rag_node", rag_node)
    graph.add_node("order_status_node", order_status_node)
    graph.add_node("escalate_node", escalate_node)

    graph.add_edge(START, "classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        _route_after_classify,
        {"rag": "rag_node", "order_status": "order_status_node", "escalate": "escalate_node"},
    )
    graph.add_conditional_edges("rag_node", _route_after_rag, {"escalate": "escalate_node", "end": END})
    graph.add_conditional_edges(
        "order_status_node", _route_after_order_status, {"escalate": "escalate_node", "end": END}
    )
    graph.add_edge("escalate_node", END)
    return graph.compile()


# Compiled once at import time — the graph structure itself has no
# per-request state, so there's no reason to rebuild it on every call.
router_graph = build_router_graph()

"""
Runs the full eval suite against the REAL system — real Groq calls, real
embeddings, real Postgres. This costs real API calls and takes a few
minutes; it's not meant to run on every commit (that's what the mocked
tests in tests/test_router_agent.py are for — fast, free, run constantly).
Run this when you actually want to measure system quality, not every time
you save a file.

Usage:
    python eval/run_eval.py
"""

import asyncio
import json
import statistics
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import get_settings
from app.data.db import AsyncSessionLocal
from app.providers.factory import get_embedding_provider, get_llm_provider
from app.services.router_agent import router_graph
from eval.judge import judge_groundedness

EVAL_DIR = Path(__file__).parent
TEST_SET_PATH = EVAL_DIR / "test_set.jsonl"
RESULTS_DIR = EVAL_DIR / "results"


def load_test_set() -> list[dict]:
    with open(TEST_SET_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


async def run_case(case: dict, llm_provider, embedding_provider, db) -> dict:
    start = time.monotonic()
    trace_id = str(uuid.uuid4())
    try:
        result = await router_graph.ainvoke(
            {
                "query": case["query"],
                "session_id": str(uuid.uuid4()),
                "trace_id": trace_id,
                "customer_id": case.get("customer_id") or "CUST-EVAL",
            },
            config={
                "configurable": {
                    "llm_provider": llm_provider,
                    "embedding_provider": embedding_provider,
                    "db": db,
                    "settings": get_settings(),
                }
            },
        )
    except Exception as exc:  # noqa: BLE001 — a crashed case is still a result worth recording
        return {
            "id": case["id"],
            "error": f"{type(exc).__name__}: {exc}",
            "latency_ms": int((time.monotonic() - start) * 1000),
        }

    latency_ms = int((time.monotonic() - start) * 1000)
    actual_route = result.get("route", "")
    actual_sources = result.get("sources", [])
    actual_answer = result.get("answer", "")
    token_usage = result.get("token_usage", [])
    total_tokens = sum(u.get("completion_tokens", 0) + u.get("prompt_tokens", 0) for u in token_usage)

    routing_correct = actual_route == case["expected_route"]

    retrieval_hit = None
    if "expected_source" in case:
        retrieval_hit = any(case["expected_source"] in s for s in actual_sources)

    answer_contains_check = None
    if "expected_answer_contains" in case:
        answer_contains_check = case["expected_answer_contains"].lower() in actual_answer.lower()

    groundedness = None
    if actual_route == "rag" and actual_answer:
        context = "\n".join(c.get("chunk_text", "") for c in result.get("chunks", []))
        try:
            groundedness = await judge_groundedness(case["query"], context, actual_answer, llm_provider)
        except Exception as exc:  # noqa: BLE001 — a judge failure must not crash the whole run
            groundedness = {"label": "judge_error", "reasoning": f"{type(exc).__name__}: {exc}"}

    return {
        "id": case["id"],
        "category": case.get("category"),
        "query": case["query"],
        "expected_route": case["expected_route"],
        "actual_route": actual_route,
        "routing_correct": routing_correct,
        "escalation_reason": result.get("escalation_reason"),
        "retrieval_hit": retrieval_hit,
        "answer_contains_check": answer_contains_check,
        "answer": actual_answer,
        "sources": actual_sources,
        "groundedness": groundedness,
        "latency_ms": latency_ms,
        "total_tokens": total_tokens,
        "token_usage": token_usage,
    }


async def main():
    import sys

    cases = load_test_set()
    if len(sys.argv) > 1 and sys.argv[1] == "--ids":
        wanted = set(sys.argv[2].split(","))
        cases = [c for c in cases if c["id"] in wanted]
        print(f"Filtered to {len(cases)} case(s): {[c['id'] for c in cases]}")

    llm_provider = get_llm_provider()
    embedding_provider = get_embedding_provider()

    print(f"Running {len(cases)} eval cases against the live system...")
    results = []
    async with AsyncSessionLocal() as db:
        for i, case in enumerate(cases, 1):
            print(f"  [{i}/{len(cases)}] {case['id']}...")
            try:
                results.append(await run_case(case, llm_provider, embedding_provider, db))
            except Exception as exc:  # noqa: BLE001 — one case's crash must never lose the whole run
                print(f"    ! unexpected failure on {case['id']}: {exc}")
                results.append({"id": case["id"], "error": f"{type(exc).__name__}: {exc}", "latency_ms": 0})

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    raw_path = RESULTS_DIR / f"run_{timestamp}.json"
    with open(raw_path, "w") as f:
        json.dump(results, f, indent=2)

    report_path = RESULTS_DIR / f"report_{timestamp}.md"
    write_report(results, report_path, timestamp)
    print(f"\nRaw results: {raw_path}")
    print(f"Report:      {report_path}")


def write_report(results: list[dict], path: Path, timestamp: str):
    total = len(results)
    errored = [r for r in results if "error" in r]
    scored = [r for r in results if "error" not in r]

    routing_correct = sum(1 for r in scored if r["routing_correct"])
    routing_accuracy = routing_correct / len(scored) * 100 if scored else 0

    retrieval_cases = [r for r in scored if r["retrieval_hit"] is not None]
    retrieval_hits = sum(1 for r in retrieval_cases if r["retrieval_hit"])
    retrieval_hit_rate = retrieval_hits / len(retrieval_cases) * 100 if retrieval_cases else None

    grounded_cases = [r for r in scored if r["groundedness"]]
    grounded_count = sum(1 for r in grounded_cases if r["groundedness"]["label"] == "grounded")
    groundedness_rate = grounded_count / len(grounded_cases) * 100 if grounded_cases else None

    latencies = sorted(r["latency_ms"] for r in scored)
    p50 = statistics.median(latencies) if latencies else 0
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0

    total_tokens = sum(r.get("total_tokens", 0) for r in scored)

    misrouted = [r for r in scored if not r["routing_correct"]]
    not_grounded = [
        r for r in grounded_cases if r["groundedness"]["label"] in ("not_grounded", "judge_error")
    ]

    lines = [
        f"# Eval Report — {timestamp}",
        "",
        f"**Total cases:** {total} ({len(errored)} errored, {len(scored)} scored)",
        "",
        "## Headline metrics",
        "",
        f"- **Routing accuracy:** {routing_accuracy:.1f}% ({routing_correct}/{len(scored)})",
        f"- **Retrieval hit rate:** {retrieval_hit_rate:.1f}%" if retrieval_hit_rate is not None else "- **Retrieval hit rate:** n/a",
        f"- **Groundedness rate:** {groundedness_rate:.1f}%" if groundedness_rate is not None else "- **Groundedness rate:** n/a",
        f"- **Latency p50 / p95:** {p50}ms / {p95}ms",
        f"- **Total tokens used:** {total_tokens}",
        "",
        "## Misrouted cases",
        "",
    ]
    if misrouted:
        for r in misrouted:
            reason = f" [{r['escalation_reason']}]" if r.get("escalation_reason") else ""
            lines.append(f"- `{r['id']}`: expected `{r['expected_route']}`, got `{r['actual_route']}`{reason} — {r['query'][:80]}")
    else:
        lines.append("None.")

    lines += ["", "## Not-grounded / judge-error cases", ""]
    if not_grounded:
        for r in not_grounded:
            lines.append(
                f"- `{r['id']}` ({r['groundedness']['label']}): {r['groundedness']['reasoning']}"
            )
    else:
        lines.append("None.")

    if errored:
        lines += ["", "## Errored cases", ""]
        for r in errored:
            lines.append(f"- `{r['id']}`: {r['error']}")

    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    asyncio.run(main())

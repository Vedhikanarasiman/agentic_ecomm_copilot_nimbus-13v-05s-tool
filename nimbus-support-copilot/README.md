# Nimbus Support Copilot

A production-shaped agentic support system for a fictional e-commerce
electronics retailer (Nimbus Electronics). Built as a portfolio project to
demonstrate end-to-end AI engineering: not just "a RAG chatbot," but the
routing, authorization, evaluation, and failure-analysis discipline around
one.

**Read this first if you're reviewing the project:**
[`docs/engineering-trade-offs.md`](docs/engineering-trade-offs.md) explains
every architecture decision and why alternatives were rejected.
[`docs/failure-log.md`](docs/failure-log.md) documents 11 real bugs found
and fixed during development — not hypothetical failure modes, actual ones,
with root cause and fix for each.

## What it does

A multi-agent router (LangGraph) handles customer support queries across
three paths:

- **Policy questions** (returns, warranty, shipping, refunds) → retrieval
  over 20 policy documents, grounded generation, confidence-based fallback
  to human handoff when retrieval doesn't find a good match.
- **Order status** (`"where is my order ORD-00042?"`) → deterministic order
  lookup, restricted to orders the authenticated customer actually owns.
- **Escalation** — explicit human requests, disputed facts, fraud/legal
  concerns, or anything the confidence checks above don't clear.

## Stack, and why

| Layer | Choice | Why (full reasoning in trade-offs doc) |
|---|---|---|
| Orchestration | LangGraph | Explicit state graph, not an open-ended agent loop — bounded latency/cost per request |
| API | FastAPI, async | Handles concurrent requests without blocking |
| Database | Postgres + pgvector | One database for both order data and policy embeddings |
| Cache | Redis | Exact-match caching for repeated policy questions |
| Generation | `openai/gpt-oss-20b` via Groq | Open-weight, hosted for real concurrent-request speed |
| Embeddings | `Qwen3-Embedding-0.6B`, local | Open-weight, small enough to stay off the CPU bottleneck |
| Auth | JWT + database-level authorization | JWT proves who you are; a separate ownership check proves it's your data |

Every model/provider is swappable via `app/providers/` without touching
business logic — see the `EmbeddingProvider`/`LLMProvider` interfaces.

## Quickstart

```bash
# 1. Install
pip install -r requirements.txt
pip install -e .          # makes `app` importable from anywhere

# 2. Configure
cp .env .env      # fill in GROQ_API_KEY

# 3. Start infrastructure
docker-compose up -d      # Postgres + Redis

# 4. Load data
psql ... < data/seed_orders.sql          # or the Windows Get-Content equivalent
python scripts/ingest_policies.py        # chunks, embeds, loads 20 policy docs

# 5. Run
uvicorn app.main:app --reload

# 6. Test
pytest tests/             # 21 tests, fast, no API calls — mocked providers
python eval/run_eval.py   # 29 cases, real API calls, real evaluation
```

Full step-by-step (including Windows/PowerShell specifics) in
[`docs/setup-walkthrough.md`](docs/setup-walkthrough.md).

## Evaluation

29-case eval set across all three routing paths plus adversarial cases
(prompt injection, out-of-scope queries). Methodology, including an
important caveat about *why not every "misrouted" case is actually a bug*,
is in [`docs/eval-methodology.md`](docs/eval-methodology.md).

Latest results: routing accuracy, retrieval hit rate, groundedness rate
(LLM-judge, cross-checked against blind human labels), and latency
percentiles are in `eval/results/` — see the most recent `report_*.md`.

## Project structure

```
app/
  api/          FastAPI routers (chat, auth)
  core/         config, logging, security
  data/         DB models, session management, cache
  guardrails/   PII detection/redaction
  providers/    swappable LLM/embedding provider interfaces
  services/     business logic — order lookup, retrieval, the router graph
data/           synthetic orders + policy documents (source content)
eval/           test set, LLM-judge, eval runner, human-review workflow
tests/          mocked-provider unit tests (fast, free, run constantly)
docs/           trade-offs, failure log, eval methodology, setup walkthrough
scripts/        data ingestion
```

## Known limitations (stated up front, not discovered by a reviewer)

- Embedding model (0.6B, chosen for local CPU speed) has a measured
  retrieval-quality ceiling on indirectly-phrased queries — one documented,
  accepted case in the failure log rather than chased to zero.
- `/auth/token` issues a JWT for any `customer_id` with no real identity
  verification — intentional simplification to demonstrate the
  JWT-protected request path, not a real auth system.
- No load test has been run against real concurrent traffic yet (Locust/k6
  planned, not yet executed) — latency numbers in `eval/results/` are
  single-request, not under load.

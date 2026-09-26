# Engineering Trade-offs

Why this system looks the way it does — decisions made, alternatives
considered and rejected, and what would change at real scale. Written for
someone reviewing the repo cold, not someone who watched it get built.

## Orchestration: LangGraph, not LangChain agents, CrewAI, or Haystack

The router needs bounded, predictable latency and cost per request — it's
serving concurrent customers, not running an open-ended research task.

- **CrewAI** — rejected. Role-based agents converse autonomously until they
  decide they're done. No hard bound on latency or cost per query.
- **LangChain's `AgentExecutor`** — rejected. Same problem: a non-deterministic
  ReAct loop where the model decides when to stop reasoning.
- **Haystack** — close second. Strong for retrieval-heavy pipelines, but its
  agent orchestration is less mature than LangGraph's for an explicit
  multi-path router like this one.
- **LangGraph** — chosen. It's an explicit state graph: every path
  (classify → rag / order_status / escalate) is defined up front, not
  decided freely at runtime. Bounded, debuggable, and the compiled graph's
  structure can be printed and inspected directly — see the mermaid diagram
  generated from `router_agent.build_router_graph()`.

## Data layer: Postgres + pgvector, not a dedicated vector DB

- **Chroma** — rejected for this project. Single-node, not built for
  concurrent production reads/writes at scale. Fine for prototyping.
- **Pinecone / Weaviate** — rejected. Real production tools, but add a
  second database to operate for a corpus this small (20 policy docs).
- **Postgres + pgvector** — chosen. One database for both order data (needs
  ACID transactions) and policy embeddings. Trade-off made explicitly: a
  lower similarity-search ceiling than a dedicated vector DB, in exchange
  for one system to run and back up instead of two kept in sync. This
  mirrors how many real mid-size companies actually do it before justifying
  a dedicated vector store.
- **At real scale:** if the policy corpus grew past a few thousand
  documents, or query-per-second requirements exceeded what a single
  Postgres instance handles well, this would be the first component to
  reconsider — not before.

## Models: open-weight, hosted inference — not self-hosted, not proprietary

- **Generation:** `openai/gpt-oss-20b` via Groq. Open-weight (so it's
  swappable and inspectable), hosted on Groq's LPU hardware for genuine
  concurrent-request speed no local GPU could match without real
  infrastructure investment. Native structured-output and tool-calling
  support avoids prompt-engineering workarounds for reliable JSON.
- **Embeddings:** `Qwen3-Embedding-0.6B`, run locally via
  `sentence-transformers`. Small enough to stay off the CPU concurrency
  bottleneck; open-weight and local, so no per-call cost or external
  dependency for retrieval.
- **Why not self-host the LLM too:** tested against the actual requirement
  — handle concurrent customers without degradation. Local CPU inference of
  a model capable enough for this task would not meet that bar without
  dedicated GPU infrastructure this project doesn't have. "Open-weight"
  describes the model's license, not where it has to run.
- **Known limitation, accepted rather than chased further:** the 0.6B
  embedding model has a real retrieval-quality ceiling on indirectly-phrased
  queries — see failure-log entry #11. A larger variant (Qwen3-Embedding-4B)
  is the untested next step if this ever needs to improve; the provider
  abstraction (`app/providers/`) makes that a one-line config change, not a
  rewrite.

## Deterministic rules vs. LLM judgment — applied deliberately, not everywhere

The system uses the LLM only where actual judgment is needed, and regex/
hard rules everywhere a fixed pattern exists:

- **Order-ID extraction:** regex (`ORD-\d{5}`), never the LLM. There's one
  fixed format; using a model to extract it adds latency and a new failure
  mode (hallucinated IDs) for zero benefit. This also skips the
  classification LLM call entirely when it fires — see
  `test_order_id_in_message_skips_classification_llm_call`.
- **Authorization:** a database filter (`order_id` AND `customer_id`
  together), never a judgment call. See failure-log entry #5 — this was a
  real gap found by asking "what could go wrong," not by anything breaking.
- **Retrieval-confidence fallback:** a numeric threshold
  (`RAG_SIMILARITY_THRESHOLD`), not the model deciding whether it's
  confident. Calibrated against real measured scores (failure-log entry #8),
  not guessed.
- **Escalation message:** a fixed template, not generated. An escalation
  handoff doesn't benefit from generation, and a fixed template can't
  promise something a human agent hasn't actually committed to.
- **Where the LLM IS used for judgment:** intent classification (genuine
  3-way ambiguity), RAG answer synthesis (requires reading and condensing
  retrieved text), and order-status summarization (structured data to
  natural language). All three got `reasoning_effort="low"` after discovering
  that gpt-oss's default reasoning verbosity was silently starving these
  lightweight tasks of their token budget — see failure-log entries #1-2.

## Caching: exact-match Redis, not semantic caching

Repeated identical policy questions are cached with a TTL. Semantic caching
(matching paraphrased questions via embedding similarity) was considered and
deliberately deferred — it adds a similarity-search call on *every* request
just to check the cache, which is a real latency/cost trade-off that isn't
justified at this project's traffic volume. Worth adding if a production
deployment showed a high rate of paraphrased-but-identical questions.

## What's explicitly out of scope, and why

- **GPU utilization / self-hosted-model monitoring** — doesn't apply; this
  system uses hosted inference for generation, so GPU metrics are Groq's
  concern, not this codebase's. Worth stating explicitly rather than leaving
  as an unexplained gap.
- **Queue-length metrics, prompt-caching for repeated system-prompt tokens,
  batched embedding calls at request time** — real production optimizations,
  premature at this traffic volume. Named here rather than half-built badly.
- **Fine-tuning or custom model training** — everything here runs on hosted
  or off-the-shelf open-weight models. A deliberate scope choice, not an
  oversight — the system-design and evaluation discipline was the point of
  this project, not model training.

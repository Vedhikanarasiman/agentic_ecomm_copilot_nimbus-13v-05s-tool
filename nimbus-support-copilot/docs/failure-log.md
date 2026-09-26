# Failure Log

Real issues found while building and testing this system, logged as they
happened — not reconstructed afterward. Each entry: what broke, why, how it
was actually fixed (not just patched), and why it matters.

This is a living document. New entries get added as they're found, not
batched up at the end.

---

## 1. Router classification failed with a Groq 400 error

**Symptom:** `groq.BadRequestError: 400 - json_validate_failed` on every
`/chat` request, even valid ones.

**Root cause:** `classify_intent` requested structured JSON output
(`response_format={"type": "json_object"}`) with `max_tokens=50`. gpt-oss-20b
is a reasoning model — it spends completion tokens on internal reasoning
*before* writing the final JSON. At 50 tokens, the budget ran out mid-reasoning,
before any valid JSON was produced, and Groq rejected the incomplete output
outright.

**Fix:** Two parts, not one:
1. Widened `ROUTER_MAX_TOKENS` from 50 → 200 (immediate unblock).
2. Set `reasoning_effort="low"` on the classification call, which addresses
   the actual mechanism — a 3-way intent classification doesn't need deep
   reasoning, so capping reasoning effort keeps the model from spending tokens
   it doesn't need to. Confirmed via captured `reasoning_tokens` in the usage
   payload: dropped to single digits after this change on simple queries.

**Why it matters:** step 1 alone would have "worked" but left a fragile
system — any query needing slightly more reasoning would hit the same wall
again at a higher token count. Step 2 is the actual fix; step 1 without step 2
is a bandage.

---

## 2. Order-status responses came back with an empty answer

**Symptom:** `/chat` returned `200 OK` with `route: order_status` and
`sources: [order:ORD-00017]` (so retrieval and lookup both succeeded) but
`answer: ""`. No error — silently wrong, which is worse than a crash.

**Root cause:** Same mechanism as #1, different symptom. `TOOL_CALL_MAX_TOKENS`
was set to 100 for the order-summary generation call, with default (higher)
reasoning effort. Reasoning consumed the entire token budget, leaving nothing
for the actual sentence. Because the API call itself succeeded (just with
empty content), this failed silently instead of raising an error — a more
dangerous failure mode than #1, since nothing alerted us except manual
inspection of the response.

**Fix:** `reasoning_effort="low"` on this call too. Confirmed via captured
usage: `reasoning_tokens: 15` out of `completion_tokens: 72` on a real order
lookup — well within the 200-token budget, with room to spare.

**Why it matters:** this is the stronger example of the two for an interview
— a silent wrong-but-successful response is harder to catch in production
than a loud error, and it's exactly the kind of failure a good eval set (not
just manual testing) is supposed to catch systematically instead of by luck.

---

## 3. Stale pinned dependency blocked a feature the code assumed existed

**Symptom:** `TypeError: Completions.create() got an unexpected keyword
argument 'reasoning_effort'` — after the fix for #1 and #2 was written and
deployed.

**Root cause:** `requirements.txt` pinned `groq==0.13.*`. Groq added
`reasoning_effort` support to the Python SDK in v0.30.0. The API itself
supported the parameter; the installed client library predated it.

**Fix:** Relaxed the pin to `groq>=0.30.0`.

**Why it matters:** a real example of why "it works when I wrote it" isn't
the same as "it works" — a dependency pin chosen early in the project quietly
became incompatible with a feature added later in the same project, and only
surfaced as a runtime error, not an install-time warning. Worth a line in the
engineering trade-offs doc about pin strategy (exact-pin vs. minimum-version)
for exactly this reason.

---

## 4. Classifier escalated an answerable question because it "sounded" urgent

**Symptom:** "My package arrived damaged 5 days ago, I just noticed it today,
can I still get it replaced for free?" was routed straight to `escalate`,
even though the answer (no — the 48-hour damaged-in-transit window had
passed) was sitting in a policy document the whole time. `token_usage` showed
only the `classify_intent` node ran — retrieval was never attempted.

**Root cause:** the original classifier prompt treated "sounds like a
complaint" as equivalent to "needs a human." The retrieval-confidence
fallback in `rag_node` (escalate when similarity is below threshold) was
correctly designed to be the system's real safety net for uncertain
questions — but this query never reached it, because the classifier
pre-emptively escalated based on tone, not content.

**Fix:** rewrote `CLASSIFY_SYSTEM_PROMPT` to narrow `escalate` to genuine
human-only cases (explicit request for a person, fraud/legal accusation,
disputed facts not in the system, repeated unresolved issue) and added a
counter-example few-shot showing an urgent-sounding-but-answerable question
routing to `rag`. Re-tested the same query after the fix: correctly routed
to `rag`, retrieved both "Defective Item Replacement Process" and
"Damaged-in-Transit Claims," and gave an accurate answer explaining the
48-hour cutoff and what happens after it.

**Why it matters:** this is the most interview-relevant entry in this log.
It's not an infrastructure bug (like #1-3) — it's a genuine design flaw in
how confidence and escalation were divided between two different components.
The fix wasn't "add more escalation," it was "escalate less, and trust the
retrieval-confidence check that already existed to do its job." Worth
stating plainly: an agent that escalates too eagerly isn't safe, it's just
failing differently — it silently defeats the purpose of building the RAG
path at all.

---

## Still to test

- Concurrent-user load test (Locust/k6) — not yet run
- All four escalation triggers now defined in the classifier prompt (explicit
  human request, fraud/legal, disputed facts, repeated issue) — only "final
  sale dispute" and "repeated issue" have been tested so far

---

## 5. Order lookup checked authentication, not authorization

**Symptom:** none observed in testing — found by design review, not by a
failure. `order_status_node` looked up any order by ID alone. A valid JWT
proves *someone* is logged in; it says nothing about whether the order
belongs to them. Order IDs are sequential (`ORD-00001`, `ORD-00002`...),
so any authenticated customer could enumerate and read other customers'
order details, including name, address, and amount.

**Fix:** `order_service.lookup_order` now filters by `order_id` AND
`customer_id` together, and returns `None` uniformly whether the order
doesn't exist or simply isn't this customer's — so the response can't be
used to distinguish "wrong ID" from "not yours," which would itself leak
which IDs are valid. Covered by
`test_order_not_found_or_not_owned_escalates` in `tests/test_router_agent.py`.

**Why it matters:** the single most interview-relevant item in this log.
"We added JWT auth" is a checklist answer; "JWT gives you authentication,
and we separately enforce authorization on every data access" is the answer
that shows you understand what the checklist item actually protects against.
Worth stating explicitly: this was found by asking "what could go wrong,"
not by anything breaking in a test.

---

## 6. Provider factory eagerly imported every implementation, and settings
   loaded eagerly at import time

**Symptom:** importing the app for testing failed with
`ModuleNotFoundError: No module named 'sentence_transformers'` — even
though the test never touches the embedding provider at all. Separately,
`ValidationError: database_url Field required` — even for tests using fully
mocked providers with no real database.

**Root cause:** two separate eager-loading problems. (1) `factory.py`
imported both `GroqLLM` and `Qwen3LocalEmbedding` at module level,
regardless of which one `LLM_PROVIDER`/`EMBEDDING_PROVIDER` actually
selects — so using only Groq still required `sentence_transformers`/`torch`
to be installed. (2) `app/data/db.py` and `app/data/cache.py` called
`get_settings()` at module import time, so simply importing the app
required a real `.env` file or real environment variables to exist.

**Fix:** made provider imports lazy (moved `from app.providers.groq_llm
import GroqLLM` etc. inside the functions that use them, not the module
top). Added `tests/conftest.py` setting dummy env vars via
`os.environ.setdefault` before any app import happens, so the test suite
never depends on a real `.env` or real secrets.

**Why it matters:** this isn't just a testing convenience — it's the same
class of bug as #3 (the stale groq pin): code silently assumed something
was true (all providers' dependencies are installed; a real `.env` always
exists) instead of that assumption being enforced or made explicit. A CI
pipeline has neither. This would have blocked automated testing entirely
in a real deployment pipeline, not just in a sandbox.

---

## 7. Eval harness crashed on case 19 and lost all 18 prior results

**Symptom:** running the 29-case eval suite against the real system, the
groundedness judge call on case 19 hit the same token-starvation pattern as
failure-log entries #1 and #2 — Groq returned a 400 (`max completion tokens
reached before generating a valid document`). Because that call wasn't
wrapped in error handling, the exception propagated all the way up and
killed the entire run. 18 cases' worth of real Groq calls — real money and
real time — were spent and then discarded, since nothing was written to
disk until the full run finished.

**Root cause:** the eval harness treated the judge call as if it couldn't
fail, even though the exact same failure mode had already been found and
documented twice in this same system (entries #1, #2). The harness itself
wasn't held to the same reliability standard as the thing it was testing.

**Fix:** wrapped the judge call in its own try/except, recording a
`judge_error` result for that one case instead of crashing. Added a second,
outer layer of error handling around each case in the main loop, so even a
completely unexpected failure inside `run_case` can't take down the batch —
worst case, one case becomes one error entry. Also widened the judge's
token budget (200 → 350) to reduce how often this triggers at all. Covered
by a new regression test,
`test_run_case_survives_a_judge_failure_without_crashing`, which simulates
the exact Groq failure and asserts the case still returns a usable result.

**Why it matters:** this is a distinct lesson from #1/#2, not a repeat of
them — those were about the LLM provider's behavior; this is about **eval
harness engineering discipline**. A harness that runs real, paid API calls
against 29 cases needs to be at least as resilient as the system it's
testing, or a single flaky case can silently invalidate an entire expensive
run. This is exactly the kind of operational maturity question a good
interviewer asks about eval infrastructure specifically — not "does your
eval work," but "what happens when your eval itself fails partway through."

---

## 8. Similarity threshold was set without ever measuring real scores

**Symptom:** first full eval run: 79.3% routing accuracy, 6 misroutes. 4 of
them (`rag-04`, `rag-06`, `rag-07`, `rag-08`) shared the identical
`escalation_reason`: "No policy content matched above the confidence
threshold" — on plain, directly-phrased questions like "how long does
standard shipping take?"

**Root cause:** `RAG_SIMILARITY_THRESHOLD=0.55` was a guessed default, never
checked against Qwen3-Embedding-0.6B's actual score distribution. Built
`eval/inspect_retrieval.py` to check: the correct document ("Domestic
Shipping") scored 0.5372 for that query — the right answer, rejected by
0.013.

**Fix:** lowered the threshold to 0.50 based on this evidence — clears the
4 failing cases and the next-closest adjacent shipping docs (0.4965,
0.4765), while staying well above the steep drop-off into genuinely
irrelevant content (0.40 and below on the same query).

**Why it matters:** this is the cleanest example in the whole project of
evidence-based tuning versus guessing. The instinct on seeing "escalated
instead of answered" is to loosen something — the discipline is building a
tool that shows *where* the correct answer actually scored before touching
the number, so the fix is a measured calibration, not a hopeful guess. Worth
noting for the trade-offs doc: this number should be re-checked any time the
embedding model changes, since it's specific to Qwen3-Embedding-0.6B's score
distribution, not a universal constant.

## 9. Two test-set labels were wrong, not the system

**Symptom:** `rag-09` and `adversarial-01` both showed as misrouted.

**On inspection:** both were the system behaving *correctly* —
`rag-09` triggered the intended graceful "please provide your order number"
fallback for a genuinely order-shaped question; `adversarial-01` produced a
grounded refusal that cited real policy and explicitly declined to skip
the policy check, a stronger outcome than a bare escalate would have been.

**Fix:** relabeled both cases' `expected_route` to match the correct
behavior, with a `notes` field explaining why, so the reasoning isn't lost
next time this test set is read.

**Why it matters:** an eval set is not ground truth by default — it's a
hypothesis about correct behavior, and it needs the same scrutiny as the
system it's testing. Treating every red X in a report as "the system is
wrong" without checking the label itself is a common eval mistake. This is
a good, concrete example for an interview of *iterating on the eval set*,
not just running it once and reporting the number.

---

## 10. Eval harness silently ignored real config — hardcoded duplicate settings

**Symptom:** after lowering `RAG_SIMILARITY_THRESHOLD` to 0.50 (entry #8)
and confirming via `get_settings()` that the value was correctly 0.50, the
same 4 cases failed identically in the next eval run — same exact
escalation reason, no change at all.

**Root cause:** `eval/run_eval.py` never called the real `get_settings()`.
It used a hardcoded `_EvalSettings` class, written once when Phase 3 was
built, with `rag_similarity_threshold = 0.55` frozen into it — completely
disconnected from `.env` or `app/core/config.py`. The live `/chat` endpoint
used real settings; the eval harness used a stale snapshot. Nothing ever
errored, because both were syntactically valid — they just silently
diverged.

**Fix:** removed `_EvalSettings` entirely; `run_case` now calls the real
`get_settings()` directly, the same function every other part of the app
uses.

**Why it matters:** this is the most subtle and arguably most dangerous bug
in this log, because it fails silently and specifically undermines the eval
suite's entire purpose — trusting eval numbers requires trusting that eval
is testing the same system you're actually running. A hardcoded config
duplicate is a classic way for that trust to quietly become false. Worth
naming directly in an interview: an eval harness needs the same "one source
of truth" discipline as the production code, not an exemption from it
because it's "just testing infrastructure."

---

## 11. rag-06 — known, accepted retrieval limitation (not fixed further)

**Symptom:** "My headphones stopped working after 2 weeks, what happens?"
fails to retrieve "Defective Item Replacement Process" (the correct doc) —
it scores 0.3755, ranked 5th, well below the top match "Returns Policy —
Opened Electronics" (0.4985), which not coincidentally itself contains a
cross-reference redirecting to the correct policy.

**Diagnosis:** unlike entry #8, this is not a threshold miscalibration —
the correct document doesn't rank near the top at all, so no reasonable
threshold adjustment fixes it without also admitting noise on unrelated
queries (Payment Methods sits at a similar score, 0.3579).

**Decision: accepted as a known limitation, not fixed.** Considered three
options: nudge the threshold further (treats a ranking problem as a
calibration problem — doesn't actually fix it), swap to a larger embedding
model (Qwen3-Embedding-4B — plausible real fix, untested, adds local
compute/latency cost), or document and move on. Chose the third: this is
one case out of 29, the 0.6B embedding model was a deliberate choice for
local CPU speed, and chasing every remaining percentage point trades
project time for a marginal, unverified gain.

**Why it matters:** knowing when to stop tuning and document a limitation
is itself the engineering judgment call worth being able to defend —
distinct from entry #8, where the same "reduce the score gap" instinct
was the RIGHT move because it was evidence of miscalibration, not a
capability ceiling. The lesson isn't "always tune" or "never tune" — it's
correctly diagnosing which one you're looking at before deciding.

---

## 12. Embedding model loaded lazily on a random first request, causing an unpredictable client timeout

**Symptom:** Streamlit UI showed "Request failed: Read timed out (timeout=60)"
on an `order_status` query — a query type that never touches retrieval or
embeddings at all. The uvicorn log showed `sentence_transformers` loading
`Qwen3-Embedding-0.6B` right before the request completed.

**Root cause:** `embedding_provider` is declared as a FastAPI `Depends()`
parameter on the `/chat` endpoint, so it's resolved on every request
regardless of which route the query actually takes. It's `@lru_cache`'d, so
the expensive model load only happens once per server lifetime — but
"once" meant "on whichever request happened to land first," which was this
one. The load itself took ~14 seconds in the log; the original failed
attempt likely took longer still (a fresh ML model load is a common target
for Windows Defender's real-time file scanning on first touch), long
enough to exceed the client's 60-second timeout.

**Fix:** moved provider construction into a FastAPI `lifespan` handler, so
both providers load once at server startup — a cost paid predictably while
watching the terminal, not silently dropped on whichever user's request
happens to be first.

**Why it matters:** this is a different flavor of the same underlying
lesson as entries #1/#2/#7 — an expensive operation with no explicit
handling of its own latency will eventually surprise someone at the worst
possible moment. The general principle: warm up expensive resources at a
time you control (startup), never lazily on user-facing request paths.
Also worth naming as a follow-up, not yet fixed: neither the Redis client
nor the Groq client currently has an explicit connection/read timeout set
anywhere in this codebase — meaning a slow or unreachable dependency can
still hang a request indefinitely instead of failing fast with a clear
error. That's a real, still-open gap.

---

## 13. Checkpointing conversation memory couldn't share a graph with per-turn state

**Context:** replacing the hand-rolled SQL memory fetch (`memory_service.py`) with LangGraph's real checkpointer, as originally planned two turns earlier in this project's build.

**The problem, caught during planning, not after shipping it:** LangGraph's checkpointer persists the *entire* state schema of whatever graph it's attached to — there's no way to persist one field and not another. `router_graph`'s `token_usage` field uses an accumulating (`operator.add`) reducer. Attaching a checkpointer directly to `router_graph` would mean `token_usage` keeps adding onto its own total across the whole lifetime of a conversation instead of resetting each turn — every cost/token number in the API response would silently become wrong starting with the second message in any session, with no error or warning.

**Fix:** built a second, deliberately minimal graph (`memory_graph.py`) whose only job is message persistence, checkpointed independently of `router_graph`. `router_graph` stays exactly as it was — computed fresh every invocation, never checkpointed, so its per-turn fields stay correct. Message history, which genuinely *should* accumulate forever within a conversation, gets the real checkpointer; per-turn bookkeeping, which should never accumulate, doesn't.

**Why it matters:** the fix that seemed obvious at first — "just reset the field at the start of each turn" — doesn't actually work, because an accumulating reducer has no "clear" operation, only "add." Passing an empty value doesn't reset anything; `old + [] = old`. This is the kind of bug that wouldn't show up in a quick manual test — it needs a real multi-turn conversation to surface, by which point it would already be live. Worth the general lesson: when adopting a persistence mechanism, check what its unit of persistence actually is before assuming "just turn it on" is the whole task.

## 14. `langgraph-checkpoint-postgres` installed without a working Postgres driver backend

**Symptom:** `ImportError: no pq wrapper available` when importing `app.main`, immediately after adding the checkpointer dependency — caught by running the full test suite here before handing the change over, not discovered live.

**Root cause:** `langgraph-checkpoint-postgres` depends on `psycopg`, but the bare `psycopg` package has no working backend on its own — it needs either a system-installed `libpq` library, or the self-contained binary wheel (`psycopg[binary]`), explicitly requested. Neither was specified.

**Fix:** added `psycopg[binary]>=3.1` explicitly to `requirements.txt`.

**Why it matters:** same category as failure-log entry #3 (the stale groq pin) — a transitive dependency's real requirements weren't fully specified, and it would have failed at import time on any fresh install, not just eventually. The general habit worth keeping: after adding any new dependency, actually import the module that uses it before considering the change done — `pip install` succeeding is not the same claim as "it works."

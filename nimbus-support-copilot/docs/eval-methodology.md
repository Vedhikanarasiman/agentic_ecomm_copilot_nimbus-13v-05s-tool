# Eval Methodology

## What's measured

- **Routing accuracy** — did the system's final route match the expected one
- **Retrieval hit rate** — for RAG cases, did the expected policy doc actually
  get retrieved (not just "did it answer," but "did it use the right source")
- **Groundedness** — LLM-judge verdict on whether the answer's claims are
  actually supported by the retrieved context, not invented
- **Judge/human agreement** — a sample of judge verdicts checked against a
  blind human label, to know how much to trust the judge (see
  `export_for_human_review.py` / `compute_agreement.py`)
- **Latency (p50/p95) and token cost** — per the real run, not estimated

## Important: not all "misrouted" cases are equally bad

The harness marks a case as `routing_correct: False` any time the final
route doesn't match `expected_route`. But there are two very different ways
that happens:

1. **Genuine misclassification** — e.g. a policy question got routed to
   `order_status`. This is a real routing bug.
2. **Safe fallback** — e.g. a policy question was classified as `rag`, but
   retrieval confidence came back too low, so `rag_node` correctly escalated
   instead of guessing. This is `routing_correct: False` by the strict
   definition, but it's the system doing exactly what it was designed to do
   when uncertain — the alternative (forcing an answer from weak context)
   is worse, not better.

**When reading a report, always check `actual_route` on every misrouted
case before treating it as a bug.** A cluster of type-1 misroutes means the
classifier prompt needs work. A cluster of type-2 fallbacks might just mean
the similarity threshold (`RAG_SIMILARITY_THRESHOLD`, currently 0.55) is
tuned conservatively — worth a deliberate decision, not an automatic "fix."

## How to run it

1. Make sure Docker (Postgres + Redis) is running and policy docs are
   ingested — same prerequisites as normal `/chat` testing.
2. `python eval/run_eval.py` — runs all 29 cases against the real system.
   Costs real Groq calls, takes a few minutes. Produces
   `eval/results/run_<timestamp>.json` (raw) and `report_<timestamp>.md`
   (readable summary).
3. `python eval/export_for_human_review.py eval/results/run_<timestamp>.json`
   — samples 15 RAG cases into a CSV, judge labels hidden.
4. Open that CSV, read each answer against its sources, fill in
   `human_label` for each row yourself, blind.
5. `python eval/compute_agreement.py eval/results/run_<timestamp>.json eval/results/run_<timestamp>_human_review.csv`
   — prints the actual judge/human agreement percentage.

## What "good" looks like here (a starting point, not a target to force)

There's no universal passing threshold — these are reference points to
notice when a run is *surprising*, not numbers to chase:

- Routing accuracy on the 19 rag-basic/rag-numeric/rag-reasoning +
  escalate-explicit/legal/repeated cases (clear-cut ones) should be very
  high — these aren't ambiguous by design.
- The 2 adversarial cases and the classifier-boundary regression case
  (`rag-16`) are the ones actually worth reading carefully every run, not
  just checking the aggregate percentage on.
- Judge/human agreement below ~80% on the sample means the judge prompt
  itself needs work before trusting its verdicts on the full set.

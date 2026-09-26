"""
LLM-as-judge for groundedness: does the answer actually follow from the
retrieved policy context, or does it state something the context doesn't
support (hallucination)?

Used as one signal, not the whole eval — see docs/eval-methodology.md for
why a subset is also human-labeled and compared against this judge's output.
"""

import json

from app.providers.base import LLMProvider

JUDGE_SYSTEM_PROMPT = (
    "You are evaluating whether a customer-support answer is grounded in the "
    "policy context it was given. Grounded means every factual claim in the "
    "answer (numbers, deadlines, fees, conditions) is actually stated in the "
    "context — not invented, not assumed, not extrapolated.\n\n"
    "Respond with JSON only: "
    '{"label": "grounded" | "partially_grounded" | "not_grounded", "reasoning": "<one sentence>"}\n\n'
    "- grounded: every claim traces directly to the context\n"
    "- partially_grounded: the core answer is supported, but it adds at least "
    "one unsupported detail (a number, a condition, a caveat not in the context)\n"
    "- not_grounded: the answer's main claim isn't supported by the context, "
    "or contradicts it"
)


async def judge_groundedness(
    query: str, context: str, answer: str, llm_provider: LLMProvider
) -> dict:
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context given to the assistant:\n{context}\n\n"
            f"Customer question: {query}\n\nAssistant's answer: {answer}",
        },
    ]
    response = await llm_provider.generate(
        messages,
        max_tokens=350,
        temperature=0.0,
        response_format={"type": "json_object"},
        reasoning_effort="low",
    )
    try:
        parsed = json.loads(response["content"])
        label = parsed.get("label", "not_grounded")
        reasoning = parsed.get("reasoning", "")
    except (json.JSONDecodeError, AttributeError):
        # judge itself failing shouldn't crash the eval run — record it as a
        # judge failure, distinct from an actual "not_grounded" verdict, so
        # you can tell the two apart when reading results later
        label, reasoning = "judge_error", "Judge did not return valid JSON"
    if label not in ("grounded", "partially_grounded", "not_grounded"):
        label = "judge_error"
    return {"label": label, "reasoning": reasoning}

import re

_PATTERNS = {
    "EMAIL": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "PHONE": re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b"),
    "CARD_NUMBER": re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
}


def redact_pii(text: str) -> str:
    """Applied before any request/response gets written to logs or the
    conversation_turns table. Deterministic on purpose — a model-based
    redactor can miss things unpredictably, which is not acceptable for
    a compliance-relevant guardrail."""
    redacted = text
    for label, pattern in _PATTERNS.items():
        redacted = pattern.sub(f"[REDACTED_{label}]", redacted)
    return redacted


def contains_pii(text: str) -> bool:
    return any(pattern.search(text) for pattern in _PATTERNS.values())

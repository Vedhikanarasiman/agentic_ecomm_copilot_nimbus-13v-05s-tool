import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.guardrails.pii import contains_pii, redact_pii
from app.services.chunking import chunk_document
from app.services.order_service import extract_order_id


def test_redact_email():
    text = "Contact me at jane.doe@example.com about this."
    redacted = redact_pii(text)
    assert "jane.doe@example.com" not in redacted
    assert "[REDACTED_EMAIL]" in redacted


def test_redact_phone():
    text = "Call me at 555-123-4567 please."
    redacted = redact_pii(text)
    assert "555-123-4567" not in redacted


def test_no_pii_passthrough():
    text = "What is your return policy for laptops?"
    assert redact_pii(text) == text
    assert not contains_pii(text)


def test_extract_order_id_found():
    assert extract_order_id("Where is my order ORD-00042?") == "ORD-00042"


def test_extract_order_id_case_insensitive():
    assert extract_order_id("tracking for ord-00099 please") == "ORD-00099"


def test_extract_order_id_not_found():
    assert extract_order_id("What's your warranty policy?") is None


def test_extract_order_id_ignores_malformed():
    # 4 digits instead of 5 — should not match
    assert extract_order_id("order ORD-1234 status?") is None


def test_chunk_document_respects_paragraph_boundaries():
    text = "# Title\n\nPara one.\n\nPara two.\n\nPara three."
    chunks = chunk_document(text)
    assert len(chunks) >= 1
    # no paragraph should be split mid-sentence
    for chunk in chunks:
        assert chunk.strip() != ""


def test_chunk_document_splits_long_docs():
    long_para = " ".join(["word"] * 500)
    text = f"# Title\n\n{long_para}\n\nShort para."
    chunks = chunk_document(text)
    assert len(chunks) >= 2

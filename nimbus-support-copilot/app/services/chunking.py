MAX_CHUNK_TOKENS = 400  # approx, using whitespace-word count as a cheap proxy
OVERLAP_WORDS = 40


def chunk_document(text: str) -> list[str]:
    """Splits on paragraph boundaries, not fixed-token windows. Policy docs
    are short and each paragraph is usually a self-contained rule, so
    splitting mid-paragraph would separate a rule from its own qualifying
    condition — worse for retrieval than the extra bookkeeping here.
    Overlap carries the tail of one chunk into the next for continuity
    across a split.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para.split())
        if current_len + para_len > MAX_CHUNK_TOKENS and current:
            chunks.append("\n\n".join(current))
            overlap_text = " ".join(" ".join(current).split()[-OVERLAP_WORDS:])
            current = [overlap_text, para]
            current_len = len(overlap_text.split()) + para_len
        else:
            current.append(para)
            current_len += para_len

    if current:
        chunks.append("\n\n".join(current))
    return chunks

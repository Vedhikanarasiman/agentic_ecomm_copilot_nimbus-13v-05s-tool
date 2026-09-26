-- Nimbus Support Copilot — database schema
-- One database, two purposes: structured order data (needs ACID) and
-- policy embeddings (needs similarity search). See architecture notes
-- on why pgvector was chosen over a dedicated vector DB.

CREATE EXTENSION IF NOT EXISTS vector;

-- ── Orders (structured, tool-call path reads this) ──────────────────────────
CREATE TYPE order_status AS ENUM (
    'processing', 'shipped', 'delivered', 'returned', 'cancelled'
);

CREATE TABLE orders (
    order_id         VARCHAR(12) PRIMARY KEY,       -- format: ORD-XXXXX
    customer_id      VARCHAR(12) NOT NULL,
    customer_name    VARCHAR(120) NOT NULL,
    email            VARCHAR(160) NOT NULL,
    product_name     VARCHAR(200) NOT NULL,
    category         VARCHAR(80) NOT NULL,           -- e.g. 'laptops', 'audio', 'accessories'
    amount           NUMERIC(10, 2) NOT NULL,
    payment_method   VARCHAR(40) NOT NULL,
    status           order_status NOT NULL DEFAULT 'processing',
    order_date       TIMESTAMPTZ NOT NULL,
    ship_date        TIMESTAMPTZ,
    delivery_date    TIMESTAMPTZ,
    tracking_number  VARCHAR(40),
    shipping_address VARCHAR(300) NOT NULL
);

CREATE INDEX idx_orders_customer_id ON orders (customer_id);
CREATE INDEX idx_orders_status ON orders (status);

-- ── Policy chunks (RAG path reads this) ─────────────────────────────────────
-- vector(1024) assumes Qwen3-Embedding-0.6B configured at 1024 dims.
-- Confirm against the model card before ingesting — Qwen3 embedding
-- models support multiple output dimensions (Matryoshka-style truncation),
-- so this is a config choice, not a fixed constant.
CREATE TABLE policy_chunks (
    id              SERIAL PRIMARY KEY,
    doc_title        VARCHAR(200) NOT NULL,
    chunk_index      INT NOT NULL,
    chunk_text       TEXT NOT NULL,
    embedding        vector(1024) NOT NULL,
    token_count      INT NOT NULL
);

-- HNSW index for approximate nearest-neighbor search. Chosen over
-- ivfflat because it doesn't need a training/list-count step at this
-- small a corpus size, and query-time recall is more predictable.
CREATE INDEX idx_policy_chunks_embedding
    ON policy_chunks USING hnsw (embedding vector_cosine_ops);

-- ── Conversation memory (per-session short-term memory) ─────────────────────
CREATE TABLE conversation_turns (
    id              SERIAL PRIMARY KEY,
    session_id       UUID NOT NULL,
    trace_id         UUID NOT NULL,           -- links a turn across router -> retrieval/tool -> generation in logs
    customer_id      VARCHAR(12),
    turn_index       INT NOT NULL,
    role             VARCHAR(20) NOT NULL,    -- 'user' | 'assistant' | 'system'
    content          TEXT NOT NULL,
    route_taken      VARCHAR(20),             -- 'rag' | 'tool_call' | 'escalate'
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_conversation_session ON conversation_turns (session_id, turn_index);

-- ── Feedback (grows the eval set post-launch) ───────────────────────────────
CREATE TABLE feedback (
    id              SERIAL PRIMARY KEY,
    trace_id         UUID NOT NULL,
    rating           SMALLINT,                -- 1 = thumbs up, -1 = thumbs down, NULL = no explicit feedback
    implicit_signal  VARCHAR(40),              -- e.g. 'escalation_triggered', 'repeated_question'
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

"""Postgres schema: one row per chunk, with a pgvector embedding and a full-text index."""
from __future__ import annotations

import psycopg
from pgvector.psycopg import register_vector

from .config import settings

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS chunks (
    id        serial PRIMARY KEY,
    doc_path  text NOT NULL,
    title     text NOT NULL,
    section   text NOT NULL,
    url       text NOT NULL,
    content   text NOT NULL,
    embedding vector({dim}) NOT NULL,
    tsv       tsvector GENERATED ALWAYS AS (
                setweight(to_tsvector('english', title || ' ' || section), 'A') ||
                setweight(to_tsvector('english', content), 'B')) STORED
);
CREATE INDEX IF NOT EXISTS chunks_tsv ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_emb ON chunks USING hnsw (embedding vector_cosine_ops);
"""


def connect() -> psycopg.Connection:
    conn = psycopg.connect(settings.database_url, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def create_schema(conn: psycopg.Connection, reset: bool = False) -> None:
    if reset:
        conn.execute("DROP TABLE IF EXISTS chunks")
    conn.execute(SCHEMA.format(dim=settings.embed_dim))

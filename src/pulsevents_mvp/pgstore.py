"""Vector store PostgreSQL + pgvector (remplace FAISS du POC).

PGVector implémente la même interface VectorStore que FAISS : le retriever
hybride, le RAG et l'évaluation de P11 fonctionnent sans modification.

Détails qui comptent (questions soutenance) :
- ``embedding_length=1024`` : pgvector REFUSE de créer un index HNSW sur une
  colonne ``vector`` sans dimension fixe. mistral-embed = 1024.
- Index ``HNSW`` (vs IVFFlat) : meilleur rappel/latence ; ``ef_search`` règle
  le compromis rappel ↔ latence à la requête.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import psycopg
from langchain_core.documents import Document
from langchain_postgres import PGVector

from pulsevents_mvp.config import Settings
from pulsevents_mvp.embeddings import get_embeddings, split_documents

logger = logging.getLogger(__name__)

EMBED_DIM = 1024  # dimension mistral-embed


def _psycopg_dsn(settings: Settings) -> str:
    db = settings.database
    return f"postgresql://{db.user}:{db.password}@{db.host}:{db.port}/{db.name}"


def get_vectorstore(settings: Settings, embeddings=None,
                    cache_dir: Optional[Path] = None) -> PGVector:
    """Instancie le PGVector (collection 'events')."""
    embeddings = embeddings or get_embeddings(settings, cache_dir=cache_dir)
    return PGVector(
        embeddings=embeddings,
        collection_name=settings.database.collection,
        connection=settings.database.connection_string,
        embedding_length=EMBED_DIM,
        use_jsonb=True,
    )


def ensure_hnsw_index(settings: Settings) -> None:
    """Crée l'index HNSW (cosine) et fixe ef_search au niveau base."""
    db = settings.database
    with psycopg.connect(_psycopg_dsn(settings), autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_hnsw "
            "ON langchain_pg_embedding USING hnsw (embedding vector_cosine_ops) "
            f"WITH (m = {db.hnsw_m}, ef_construction = {db.hnsw_ef_construction});"
        )
        conn.execute(f"ALTER DATABASE {db.name} SET hnsw.ef_search = {db.hnsw_ef_search};")
    logger.info(
        "Index HNSW prêt (m=%s, ef_construction=%s, ef_search=%s)",
        db.hnsw_m, db.hnsw_ef_construction, db.hnsw_ef_search,
    )


def count_embeddings(settings: Settings) -> int:
    try:
        with psycopg.connect(_psycopg_dsn(settings), autocommit=True) as conn:
            row = conn.execute("SELECT count(*) FROM langchain_pg_embedding;").fetchone()
            return int(row[0]) if row else 0
    except psycopg.Error:
        return 0


def build_index(documents: list[Document], settings: Settings,
                cache_dir: Optional[Path] = None, batch: int = 256) -> PGVector:
    """Découpe, embedde et insère dans pgvector, puis crée l'index HNSW."""
    chunks = split_documents(documents, settings)
    vs = get_vectorstore(settings, cache_dir=cache_dir)
    for i in range(0, len(chunks), batch):
        vs.add_documents(chunks[i:i + batch])
        logger.info("pgvector : %d / %d chunks insérés", min(i + batch, len(chunks)), len(chunks))
    ensure_hnsw_index(settings)
    return vs

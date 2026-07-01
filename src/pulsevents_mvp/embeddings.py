"""Embeddings Mistral (cache disque) + splitter + chargement BM25.

Repris de P11 (vectorstore.py) sans la partie FAISS. Le cache d'embeddings
est réutilisé depuis P11 -> ré-embedding gratuit au chargement de pgvector.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from pulsevents_mvp.config import Settings
from pulsevents_mvp.preprocessing import to_documents
from pulsevents_mvp.tokenizer_fr import fr_preprocess

logger = logging.getLogger(__name__)


class RateLimitedMistralEmbeddings(Embeddings):
    """Rate-limit + retry autour de MistralAIEmbeddings (anti-429, cf. P11)."""

    def __init__(self, base: MistralAIEmbeddings, min_interval_s: float = 1.1,
                 batch_size: int = 24):
        self._base = base
        self._min_interval = min_interval_s
        self._batch_size = batch_size
        self._last_ts = 0.0

    @retry(
        retry=retry_if_not_exception_type((ValueError, TypeError, NotImplementedError)),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        stop=stop_after_attempt(6),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _safe_call(self, fn, *args, **kwargs):
        elapsed = time.time() - self._last_ts
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        try:
            return fn(*args, **kwargs)
        finally:
            self._last_ts = time.time()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            chunk = texts[start:start + self._batch_size]
            out.extend(self._safe_call(self._base.embed_documents, chunk))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._safe_call(self._base.embed_query, text)


def split_documents(documents: list[Document], settings: Settings) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunking.chunk_size,
        chunk_overlap=settings.chunking.chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", ", ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    logger.info("Découpage : %d documents -> %d chunks", len(documents), len(chunks))
    return chunks


def get_embeddings(settings: Settings, cache_dir: Optional[Path] = None,
                   use_cache: bool = True) -> Embeddings:
    """Embeddings Mistral, éventuellement cachés sur disque.

    Args:
        cache_dir: dossier de cache. Pour le chargement initial, passer
            ``settings.p11_embed_cache`` (réutilise les vecteurs déjà payés).
            Pour les requêtes runtime, laisser défaut (cache local writable).
    """
    if not settings.mistral_api_key:
        raise ValueError("MISTRAL_API_KEY absente (voir .env).")
    base = MistralAIEmbeddings(
        model=settings.models.embedding_model, api_key=settings.mistral_api_key
    )
    rate_limited = RateLimitedMistralEmbeddings(
        base,
        min_interval_s=settings.models.embedding_min_interval_s,
        batch_size=settings.models.embedding_batch_size,
    )
    if not use_cache:
        return rate_limited
    cache_dir = cache_dir or settings.local_embed_cache
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    store = LocalFileStore(str(cache_dir))
    return CacheBackedEmbeddings.from_bytes_store(
        rate_limited, store, namespace=settings.models.embedding_model,
        query_embedding_cache=True,   # cache AUSSI les embeddings de requête
                                      # (sinon chaque retrieval rappelle l'API : latence faussée)
    )


def load_documents_for_bm25(settings: Settings) -> list[Document]:
    """Recharge les Documents (parquet P11) pour alimenter BM25Retriever."""
    parquet = settings.p11_parquet
    if not parquet.exists():
        raise FileNotFoundError(f"Parquet introuvable : {parquet}")
    df = pd.read_parquet(parquet)
    docs = to_documents(df, settings.preprocessing.min_description_length)
    logger.info("BM25 : %d Documents rechargés depuis %s", len(docs), parquet.name)
    return docs


def build_bm25_retriever(settings: Settings, k: int) -> BM25Retriever:
    """Construit le BM25Retriever (point unique : rag + bench passent par ici).

    Injecte le tokeniseur FR ``fr_preprocess`` si ``settings.retrieval.use_fr_tokenizer``,
    sinon laisse le découpage par défaut de langchain (split sur les espaces).
    """
    docs = load_documents_for_bm25(settings)
    if settings.retrieval.use_fr_tokenizer:
        bm25 = BM25Retriever.from_documents(docs, preprocess_func=fr_preprocess)
    else:
        bm25 = BM25Retriever.from_documents(docs)
    bm25.k = k
    return bm25

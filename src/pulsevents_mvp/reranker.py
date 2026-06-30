"""Re-ranking cross-encoder bge-reranker-v2-m3 (reco v1 soutenance P11).

Encapsule le cross-encoder derrière une interface isolée (comme pgstore.py
isole le vector store). ``build_reranker`` renvoie un compressor LangChain
prêt à enrober un retriever, ou ``None`` si le modèle ne peut pas être chargé
(réseau, cache absent) — le RAG retombe alors proprement sur le retriever nu.
"""
from __future__ import annotations

import logging
from typing import Optional

from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents.compressor import BaseDocumentCompressor

from pulsevents_mvp.config import Settings

logger = logging.getLogger(__name__)


def build_reranker(settings: Settings) -> Optional[BaseDocumentCompressor]:
    """Charge le cross-encoder et renvoie un CrossEncoderReranker (top_n = k).

    Renvoie None si le chargement échoue, pour que la chaîne RAG continue
    sans re-ranking au lieu de planter.
    """
    model_name = settings.retrieval.reranker_model
    try:
        model = HuggingFaceCrossEncoder(model_name=model_name)
    except Exception:  # chargement modèle : réseau / cache / OOM
        logger.warning(
            "Cross-encoder '%s' non chargé : re-ranking désactivé pour cette session.",
            model_name, exc_info=True,
        )
        return None
    logger.info("Cross-encoder '%s' chargé (top_n=%d).", model_name, settings.retrieval.k)
    return CrossEncoderReranker(model=model, top_n=settings.retrieval.k)

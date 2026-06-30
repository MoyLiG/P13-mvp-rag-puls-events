# src/pulsevents_mvp/loader.py
"""Chargement (idempotent) de l'index pgvector depuis les données P11.

Appelé au démarrage de l'API (auto-load) et par scripts/load_pgvector.py.
Réutilise le cache d'embeddings P11 -> ré-embedding gratuit.
"""
from __future__ import annotations

import logging
import shutil

import pandas as pd

from pulsevents_mvp.config import Settings
from pulsevents_mvp.pgstore import build_index, count_embeddings
from pulsevents_mvp.preprocessing import to_documents

log = logging.getLogger(__name__)


def seed_cache(settings: Settings) -> None:
    """Copie le cache d'embeddings P11 (ro) vers le cache local (writable)."""
    src, dst = settings.p11_embed_cache, settings.local_embed_cache
    if dst.exists() and any(dst.iterdir()):
        log.info("Cache local déjà présent (%s).", dst)
        return
    if not src.exists():
        log.warning("Cache P11 absent (%s) : ré-embedding payant ~0,30 €.", src)
        return
    log.info("Copie du cache d'embeddings P11 -> local (peut prendre ~1 min)...")
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)
    log.info("Cache copié.")


def load_if_needed(settings: Settings) -> int:
    """Charge pgvector si la collection est vide. Renvoie le nb de chunks final."""
    existing = count_embeddings(settings)
    if existing > 0:
        log.info("Index déjà peuplé (%d chunks). Rien à faire.", existing)
        return existing
    if not settings.p11_parquet.exists():
        raise SystemExit(f"Parquet P11 introuvable : {settings.p11_parquet}")
    seed_cache(settings)
    df = pd.read_parquet(settings.p11_parquet)
    log.info("Parquet chargé : %d événements.", len(df))
    docs = to_documents(df, settings.preprocessing.min_description_length)
    build_index(docs, settings, cache_dir=settings.local_embed_cache)
    total = count_embeddings(settings)
    log.info("Terminé : %d chunks dans pgvector.", total)
    return total

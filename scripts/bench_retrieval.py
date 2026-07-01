#!/usr/bin/env python
"""Benchmark retrieval : recall@k (dense vs hybride) + latence (FAISS vs pgvector).

Retrieval seul, sans appel LLM -> rapide et quasi gratuit (seules les
questions sont embeddées, et elles sont cachées). Produit le tableau de
chiffres défendables en soutenance.

Lancement : docker compose run --rm api python scripts/bench_retrieval.py
"""
from __future__ import annotations

import json
import logging
import statistics
import time
from pathlib import Path

from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from pulsevents_mvp.reranker import build_reranker
from langchain_community.vectorstores import FAISS

from pulsevents_mvp.config import load_settings
from pulsevents_mvp.embeddings import get_embeddings, build_bm25_retriever
from pulsevents_mvp.pgstore import get_vectorstore
from pulsevents_mvp.eval_utils import summarize_delta

logging.basicConfig(level=logging.WARNING)


def _dense(vs, settings):
    return vs.as_retriever(
        search_type=settings.retrieval.search_type,
        search_kwargs={"k": settings.retrieval.k, "fetch_k": settings.retrieval.fetch_k},
    )


def _hybrid(dense, settings):
    bm25 = build_bm25_retriever(settings, settings.retrieval.k)
    w = settings.retrieval.bm25_weight
    return EnsembleRetriever(retrievers=[bm25, dense], weights=[w, 1.0 - w])


def _hybrid_pool(vs, settings, k_pool):
    """Hybride avec pool élargi (candidats à reclasser)."""
    fetch_k = max(settings.retrieval.fetch_k, k_pool * 2)
    dense = vs.as_retriever(
        search_type=settings.retrieval.search_type,
        search_kwargs={"k": k_pool, "fetch_k": fetch_k},
    )
    bm25 = build_bm25_retriever(settings, k_pool)
    w = settings.retrieval.bm25_weight
    return EnsembleRetriever(retrievers=[bm25, dense], weights=[w, 1.0 - w])


def _reranked(vs, settings):
    """Hybride pool + cross-encoder (None si modèle indisponible)."""
    compressor = build_reranker(settings)
    if compressor is None:
        return None
    pool = _hybrid_pool(vs, settings, settings.retrieval.rerank_candidates)
    return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=pool)


def hit_rate(retriever, dataset, k):
    # Cap a k : l'EnsembleRetriever renvoie l'union RRF (~2k docs), pas k. Sans cap,
    # "recall@k" comparerait un nombre de docs different selon la variante (deloyal).
    hits = tot = 0
    for item in dataset:
        if item.get("out_of_scope"):
            continue
        exp = set(item.get("expected_source_uids") or [])
        if not exp:
            continue
        got = {d.metadata.get("uid") for d in retriever.invoke(item["question"])[:k]}
        hits += int(bool(exp & got))
        tot += 1
    return (hits / tot if tot else None), tot


def hit_flags(retriever, dataset, k):
    """Hit (bool) par question évaluable, dans l'ordre — pour comparer OFF vs ON."""
    flags = []
    for item in dataset:
        if item.get("out_of_scope"):
            continue
        exp = set(item.get("expected_source_uids") or [])
        if not exp:
            continue
        got = {d.metadata.get("uid") for d in retriever.invoke(item["question"])[:k]}
        flags.append(bool(exp & got))
    return flags


def latency_ms(retriever, questions, repeats=5):
    # passe de chauffe (non chronométrée) : met en cache les embeddings de requête
    # pour mesurer la recherche vectorielle, pas l'appel API d'embedding.
    for q in questions:
        retriever.invoke(q)
    xs = []
    for _ in range(repeats):
        for q in questions:
            t = time.perf_counter()
            retriever.invoke(q)
            xs.append((time.perf_counter() - t) * 1000)
    xs.sort()
    p95 = xs[min(len(xs) - 1, int(0.95 * len(xs)))]
    return statistics.median(xs), p95


def main():
    settings = load_settings()
    dataset = json.loads(settings.p11_qa_dataset.read_text(encoding="utf-8"))
    questions = [d["question"] for d in dataset if not d.get("out_of_scope")]
    emb = get_embeddings(settings, cache_dir=settings.local_embed_cache)

    # pgvector
    pg = get_vectorstore(settings, embeddings=emb)
    pg_dense = _dense(pg, settings)
    pg_hybrid = _hybrid(pg_dense, settings)

    k = settings.retrieval.k
    rows = []
    hr_dense, n = hit_rate(pg_dense, dataset, k)
    hr_hybrid, _ = hit_rate(pg_hybrid, dataset, k)
    rows.append(("recall@%d dense (pgvector)" % k, f"{hr_dense:.2f}" if hr_dense is not None else "n/a"))
    rows.append(("recall@%d hybride BM25+dense" % k, f"{hr_hybrid:.2f}" if hr_hybrid is not None else "n/a"))

    # --- Tokeniseur FR : gain OFF vs ON (bench loyal, cap @k) ---
    settings.retrieval.use_fr_tokenizer = False
    flags_off = hit_flags(_hybrid(pg_dense, settings), dataset, k)
    settings.retrieval.use_fr_tokenizer = True
    flags_on = hit_flags(_hybrid(pg_dense, settings), dataset, k)
    delta = summarize_delta(flags_off, flags_on)
    rows.append(("recall@%d hybride (tokeniseur FR OFF)" % k, f"{delta['recall_off']:.2f}"))
    rows.append(("recall@%d hybride (tokeniseur FR ON)" % k, f"{delta['recall_on']:.2f}"))
    rows.append(("  gain tokeniseur FR (pts)", f"{delta['gain']:+.2f}"))
    rows.append(("  questions améliorées / dégradées / inchangées",
                 f"{delta['improved']} / {delta['degraded']} / {delta['unchanged']}"))

    rr = _reranked(pg, settings)
    if rr is not None:
        hr_rerank, _ = hit_rate(rr, dataset, k)
        rows.append(("recall@%d hybride + re-ranking bge" % k,
                     f"{hr_rerank:.2f}" if hr_rerank is not None else "n/a"))
        med_rr, p95_rr = latency_ms(rr, questions)
        rows.append(("latence retrieval + re-ranking — médiane (ms)", f"{med_rr:.1f}"))
        rows.append(("latence retrieval + re-ranking — p95 (ms)", f"{p95_rr:.1f}"))
    else:
        rows.append(("re-ranking bge", "modèle non chargé"))

    med_pg, p95_pg = latency_ms(pg_dense, questions)
    rows.append(("latence retrieval pgvector — médiane (ms)", f"{med_pg:.1f}"))
    rows.append(("latence retrieval pgvector — p95 (ms)", f"{p95_pg:.1f}"))

    # FAISS (comparatif latence) si l'index P11 est monté
    if (settings.p11_faiss_dir / "index.faiss").exists():
        faiss = FAISS.load_local(str(settings.p11_faiss_dir), emb,
                                 allow_dangerous_deserialization=True)
        med_fa, p95_fa = latency_ms(_dense(faiss, settings), questions)
        rows.append(("latence retrieval FAISS — médiane (ms)", f"{med_fa:.1f}"))
        rows.append(("latence retrieval FAISS — p95 (ms)", f"{p95_fa:.1f}"))
    else:
        rows.append(("latence FAISS", "index P11 non monté"))

    print("\n=== Benchmark retrieval (n=%d questions) ===" % n)
    for label, val in rows:
        print(f"  {label:<48} {val}")

    out = settings.resolved_path("data/eval/bench_retrieval.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("metric,value\n" + "\n".join(f'"{l}",{v}' for l, v in rows), encoding="utf-8")
    print(f"\nCSV -> {out}")


if __name__ == "__main__":
    main()

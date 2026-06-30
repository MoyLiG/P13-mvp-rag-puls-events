"""Utilitaires d'évaluation retrieval : comparer deux séries de hits (OFF vs ON)."""
from __future__ import annotations


def summarize_delta(off_hits: list[bool], on_hits: list[bool]) -> dict:
    """Résume le gain entre deux variantes, hit par hit (même ordre de questions).

    Retourne recall_off, recall_on, gain (pts), et le compte de questions
    améliorées / dégradées / inchangées — pour distinguer un gain net d'un échange.
    """
    if len(off_hits) != len(on_hits):
        raise ValueError("off_hits et on_hits doivent avoir la même longueur")
    n = len(off_hits)
    recall_off = sum(off_hits) / n if n else 0.0
    recall_on = sum(on_hits) / n if n else 0.0
    improved = sum(1 for o, x in zip(off_hits, on_hits) if x and not o)
    degraded = sum(1 for o, x in zip(off_hits, on_hits) if o and not x)
    return {
        "n": n,
        "recall_off": recall_off,
        "recall_on": recall_on,
        "gain": recall_on - recall_off,
        "improved": improved,
        "degraded": degraded,
        "unchanged": n - improved - degraded,
    }

"""Tokeniseur FR pour BM25 : dé-accent + lowercase + stopwords + stemming Snowball.

Fonction pure, sans I/O. Injectée comme ``preprocess_func`` dans
``BM25Retriever`` (appliquée aux documents ET à la requête -> symétrie).
But : précision lexicale (le corpus d'événements en a besoin, cf. trace re-ranking).
"""
from __future__ import annotations

import re
import unicodedata

import snowballstemmer

_STEMMER = snowballstemmer.stemmer("french")

# Stopwords FR usuels, déjà dé-accentués et en minuscules (le filtre s'applique
# APRÈS dé-accent + lowercase). Liste volontairement courte (pas de download).
_STOPWORDS_FR = frozenset({
    "le", "la", "les", "un", "une", "des", "de", "du", "au", "aux",
    "ce", "ces", "cet", "cette", "dans", "en", "pour", "par", "sur", "sous",
    "avec", "sans", "que", "qui", "quoi", "dont", "ou", "est", "sont", "ete",
    "etre", "il", "elle", "ils", "elles", "je", "tu", "nous", "vous", "on",
    "se", "sa", "son", "ses", "leur", "leurs", "mon", "ma", "mes", "ton", "ta",
    "tes", "notre", "nos", "votre", "vos", "ne", "pas", "plus", "moins", "tout",
    "tous", "toute", "toutes", "si", "mais", "donc", "or", "ni", "car", "comme",
    "quand", "aussi", "tres", "trop", "peu", "y", "a", "the", "of", "and",
})

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def fr_preprocess(text: str) -> list[str]:
    text = _strip_accents(text).lower()
    tokens = [t for t in _TOKEN_RE.findall(text)
              if len(t) > 1 and t not in _STOPWORDS_FR]
    return _STEMMER.stemWords(tokens)

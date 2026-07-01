#!/usr/bin/env python
"""Mesure l'effet de la mémoire conversationnelle.

Pour chaque scénario : une question de base établit un contexte (ville),
puis une question de suivi à référence implicite (« et le week-end prochain ? »).
On compare la résolution de la question de suivi SANS historique vs AVEC.

Lancement : docker compose run --rm api python scripts/bench_memory.py
"""
from __future__ import annotations

import logging
import unicodedata

from langchain_core.messages import AIMessage, HumanMessage
from langchain_mistralai import ChatMistralAI

from pulsevents_mvp.config import load_settings
from pulsevents_mvp.rag import MvpRagPipeline, _contextualize_prompt

logging.basicConfig(level=logging.WARNING)

SCENARIOS = [
    {"base": "Quels concerts à Nantes ce mois-ci ?",
     "follow": "Et le week-end prochain ?", "expect": "nantes"},
    {"base": "Y a-t-il des expositions à Saint-Nazaire ?",
     "follow": "Et des spectacles plutôt ?", "expect": "saint-nazaire"},
    {"base": "Des festivals en Loire-Atlantique cet été ?",
     "follow": "Lesquels sont gratuits ?", "expect": "loire"},
    {"base": "Quelles pièces de théâtre à Nantes ?",
     "follow": "Et pour les enfants ?", "expect": "nantes"},
    {"base": "Des concerts de musique classique à Nantes ?",
     "follow": "Et ce mois-ci précisément ?", "expect": "nantes"},
]


def _norm(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s.lower())
                   if not unicodedata.combining(c))


def _resolved(answer: str, expect: str) -> bool:
    """Suivi 'résolu' = réponse non-refus mentionnant le lieu attendu."""
    n = _norm(answer)
    refusal = "pas trouve" in n or "n'ai pas" in n or "aucun evenement" in n
    return (not refusal) and (_norm(expect) in n)


def main():
    settings = load_settings()

    # --- Preuve principale : la reformulation history-aware ---
    # Sans historique, une question de suivi est ambiguë ; avec, elle porte
    # le contexte (lieu, thème). C'est l'effet mémoire, démontré directement.
    llm = ChatMistralAI(model=settings.models.llm_model, temperature=0,
                        api_key=settings.mistral_api_key)
    reformulate = _contextualize_prompt() | llm
    print("\n=== Preuve mémoire : reformulation de la question de suivi ===")
    for sc in SCENARIOS:
        hist = [HumanMessage(content=sc["base"]),
                AIMessage(content="(réponse de l'assistant)")]
        sans = reformulate.invoke({"input": sc["follow"], "chat_history": []}).content
        avec = reformulate.invoke({"input": sc["follow"], "chat_history": hist}).content
        print(f"\n  base   : {sc['base']}")
        print(f"  suivi  : {sc['follow']}")
        print(f"  SANS mémoire -> {sans.strip()}")
        print(f"  AVEC mémoire -> {avec.strip()}")

    # --- Indicateur secondaire (imparfait, cf. README) ---
    pipe = MvpRagPipeline(settings)
    without = with_ = 0
    print("\n=== Effet mémoire (suivi résolu ?) ===")
    for sc in SCENARIOS:
        # SANS historique : la question de suivi est ambiguë
        ans_wo = pipe.answer(sc["follow"], chat_history=[]).answer
        ok_wo = _resolved(ans_wo, sc["expect"])

        # AVEC historique : on rejoue la base puis le suivi
        base_ans = pipe.answer(sc["base"], chat_history=[]).answer
        hist = [HumanMessage(content=sc["base"]), AIMessage(content=base_ans)]
        ans_w = pipe.answer(sc["follow"], chat_history=hist).answer
        ok_w = _resolved(ans_w, sc["expect"])

        without += ok_wo
        with_ += ok_w
        print(f"  [{ '✓' if ok_wo else '✗'} sans | {'✓' if ok_w else '✗'} avec] "
              f"{sc['base']} -> {sc['follow']}")

    n = len(SCENARIOS)
    print(f"\nSuivi résolu SANS mémoire : {without}/{n}")
    print(f"Suivi résolu AVEC mémoire : {with_}/{n}")


if __name__ == "__main__":
    main()

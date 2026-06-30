"""answer() route vers la chaîne rerankée ou plain selon le flag use_reranker."""
from unittest.mock import MagicMock

from pulsevents_mvp.config import load_settings
from pulsevents_mvp.rag import MvpRagPipeline


def _pipeline_with_fake_chains():
    # bypass __init__ : on ne veut ni pgvector ni Mistral, juste tester le routage.
    pipe = MvpRagPipeline.__new__(MvpRagPipeline)
    pipe.settings = load_settings()
    plain = MagicMock()
    plain.invoke.return_value = {"answer": "PLAIN", "context": []}
    reranked = MagicMock()
    reranked.invoke.return_value = {"answer": "RERANK", "context": []}
    pipe._chain_plain = plain
    pipe._chain_reranked = reranked
    return pipe


def test_answer_uses_reranked_chain_by_default():
    pipe = _pipeline_with_fake_chains()
    assert pipe.answer("concerts à Nantes").answer == "RERANK"


def test_answer_uses_plain_chain_when_disabled():
    pipe = _pipeline_with_fake_chains()
    assert pipe.answer("concerts à Nantes", use_reranker=False).answer == "PLAIN"

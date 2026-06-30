"""Config du re-ranking : défauts + override par variable d'environnement."""
from pulsevents_mvp.config import load_settings


def test_reranker_defaults():
    s = load_settings()
    assert s.retrieval.use_reranker is False
    assert s.retrieval.reranker_model == "BAAI/bge-reranker-v2-m3"
    assert s.retrieval.rerank_candidates == 24


def test_use_reranker_env_override_false(monkeypatch):
    monkeypatch.setenv("USE_RERANKER", "false")
    s = load_settings()
    assert s.retrieval.use_reranker is False


def test_use_reranker_env_override_true(monkeypatch):
    monkeypatch.setenv("USE_RERANKER", "1")
    s = load_settings()
    assert s.retrieval.use_reranker is True

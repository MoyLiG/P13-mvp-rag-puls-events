"""build_bm25_retriever : injecte fr_preprocess selon le flag use_fr_tokenizer."""
from langchain_core.documents import Document

from pulsevents_mvp import embeddings as emb
from pulsevents_mvp.config import load_settings
from pulsevents_mvp.tokenizer_fr import fr_preprocess

_FAKE_DOCS = [
    Document(page_content="concert gratuit à Nantes", metadata={"uid": "1"}),
    Document(page_content="exposition pour enfants à Saint-Nazaire", metadata={"uid": "2"}),
]


def test_injects_fr_preprocess_when_enabled(monkeypatch):
    s = load_settings()
    s.retrieval.use_fr_tokenizer = True
    monkeypatch.setattr(emb, "load_documents_for_bm25", lambda settings: list(_FAKE_DOCS))
    r = emb.build_bm25_retriever(s, k=6)
    assert r.preprocess_func is fr_preprocess
    assert r.k == 6


def test_default_tokenizer_when_disabled(monkeypatch):
    s = load_settings()
    s.retrieval.use_fr_tokenizer = False
    monkeypatch.setattr(emb, "load_documents_for_bm25", lambda settings: list(_FAKE_DOCS))
    r = emb.build_bm25_retriever(s, k=4)
    assert r.preprocess_func is not fr_preprocess
    assert r.k == 4

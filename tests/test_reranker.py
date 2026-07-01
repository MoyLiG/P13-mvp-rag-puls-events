"""build_reranker : construit le compressor, ou None si le modèle ne charge pas."""
from pulsevents_mvp.config import load_settings
from pulsevents_mvp import reranker as rr


def test_build_reranker_uses_settings(monkeypatch):
    s = load_settings()
    captured = {}

    class FakeCrossEncoder:
        def __init__(self, model_name=None, **kw):
            captured["model_name"] = model_name

    class FakeReranker:
        def __init__(self, model=None, top_n=None, **kw):
            captured["top_n"] = top_n
            self.top_n = top_n

    monkeypatch.setattr(rr, "HuggingFaceCrossEncoder", FakeCrossEncoder)
    monkeypatch.setattr(rr, "CrossEncoderReranker", FakeReranker)
    compressor = rr.build_reranker(s)
    assert compressor is not None
    assert captured["model_name"] == s.retrieval.reranker_model
    assert captured["top_n"] == s.retrieval.k


def test_build_reranker_returns_none_on_failure(monkeypatch):
    s = load_settings()

    def boom(*a, **k):
        raise RuntimeError("téléchargement impossible")

    monkeypatch.setattr(rr, "HuggingFaceCrossEncoder", boom)
    assert rr.build_reranker(s) is None

"""Flag use_fr_tokenizer : défaut False, surchargé par l'env USE_FR_TOKENIZER."""
from pulsevents_mvp.config import load_settings


def test_use_fr_tokenizer_default_false(monkeypatch):
    monkeypatch.delenv("USE_FR_TOKENIZER", raising=False)
    s = load_settings()
    assert s.retrieval.use_fr_tokenizer is False


def test_use_fr_tokenizer_env_true(monkeypatch):
    monkeypatch.setenv("USE_FR_TOKENIZER", "true")
    s = load_settings()
    assert s.retrieval.use_fr_tokenizer is True


def test_use_fr_tokenizer_env_off(monkeypatch):
    monkeypatch.setenv("USE_FR_TOKENIZER", "0")
    s = load_settings()
    assert s.retrieval.use_fr_tokenizer is False

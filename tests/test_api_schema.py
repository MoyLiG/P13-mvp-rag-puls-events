"""ChatRequest : use_reranker présent, défaut True."""
from pulsevents_mvp.api import ChatRequest


def test_chat_request_reranker_default_true():
    req = ChatRequest(message="x")
    assert req.use_reranker is True


def test_chat_request_reranker_can_be_disabled():
    req = ChatRequest(message="x", use_reranker=False)
    assert req.use_reranker is False

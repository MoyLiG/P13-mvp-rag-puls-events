import pytest
import requests

from pulsevents_mvp.ui_client import safe_url, format_sources, ChatClient, ApiError, ChatResult


def test_safe_url_accepte_http_https():
    assert safe_url("http://a.fr") == "http://a.fr"
    assert safe_url("https://a.fr/x") == "https://a.fr/x"


def test_safe_url_rejette_scheme_dangereux_et_none():
    assert safe_url("javascript:alert(1)") is None
    assert safe_url("data:text/html,x") is None
    assert safe_url(None) is None
    assert safe_url("") is None


def test_format_sources_complet():
    src = [{"title": "Concert", "city": "Nantes",
            "daterange": "12-13 juin", "url": "https://oa.fr/e"}]
    out = format_sources(src)
    assert out == ["**Concert** — Nantes — _12-13 juin_\n\n[Voir sur Open Agenda](https://oa.fr/e)"]


def test_format_sources_champs_manquants_et_url_dangereuse():
    src = [{"title": None, "url": "javascript:x"}]
    out = format_sources(src)
    assert out == ["**?**"]


# ---------------------------------------------------------------------------
# Helpers pour les tests ChatClient
# ---------------------------------------------------------------------------

class _Resp:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json = json_data or {}

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")


class _Session:
    def __init__(self, response=None, exc=None):
        self.response = response
        self.exc = exc
        self.last_url = None
        self.last_json = None
        self.last_params = None

    def post(self, url, json=None, params=None, timeout=None):
        if self.exc:
            raise self.exc
        self.last_url, self.last_json, self.last_params = url, json, params
        return self.response

    def get(self, url, timeout=None):
        if self.exc:
            raise self.exc
        self.last_url = url
        return self.response


# ---------------------------------------------------------------------------
# Tests ChatClient
# ---------------------------------------------------------------------------

def test_send_construit_le_payload_complet():
    resp = _Resp(200, {"answer": "ok", "sources": [], "session_id": "s1", "history_len": 2})
    sess = _Session(resp)
    client = ChatClient("http://api:8000", session=sess)
    result = client.send("Salut", session_id="s1", use_memory=True)
    assert isinstance(result, ChatResult)
    assert result.answer == "ok" and result.history_len == 2
    assert sess.last_url == "http://api:8000/chat"
    assert sess.last_json == {
        "message": "Salut", "session_id": "s1",
        "use_memory": True, "use_reranker": True,
    }


def test_send_passe_les_flags_a_false():
    resp = _Resp(200, {"answer": "ok", "sources": [], "session_id": "s1", "history_len": 0})
    sess = _Session(resp)
    ChatClient("http://api:8000", session=sess).send(
        "x", "s1", use_memory=False, use_reranker=False
    )
    assert sess.last_json["use_memory"] is False
    assert sess.last_json["use_reranker"] is False


def test_send_400_leve_apierror_avec_detail():
    sess = _Session(_Resp(400, {"detail": "Question trop longue"}))
    with pytest.raises(ApiError, match="trop longue"):
        ChatClient("http://api:8000", session=sess).send("x", "s1")


def test_send_503_leve_apierror_initialisation():
    sess = _Session(_Resp(503, {"detail": "init"}))
    with pytest.raises(ApiError, match="initialisation"):
        ChatClient("http://api:8000", session=sess).send("x", "s1")


def test_send_connection_error_se_propage():
    sess = _Session(exc=requests.ConnectionError("refused"))
    with pytest.raises(requests.ConnectionError):
        ChatClient("http://api:8000", session=sess).send("x", "s1")


def test_health_retourne_le_json():
    sess = _Session(_Resp(200, {"status": "ok", "pgvector_chunks": 2302, "redis": True}))
    out = ChatClient("http://api:8000", session=sess).health()
    assert out["pgvector_chunks"] == 2302


def test_reset_appelle_endpoint_avec_param():
    sess = _Session(_Resp(200, {"reset": "s1"}))
    ChatClient("http://api:8000", session=sess).reset("s1")
    assert sess.last_url == "http://api:8000/reset"
    assert sess.last_params == {"session_id": "s1"}

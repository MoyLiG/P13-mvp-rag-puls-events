"""Client HTTP de l'UI Streamlit vers l'API du slice.

Module PUR (pas d'import Streamlit) : toute la logique testable de l'UI vit ici.
"""
from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

import requests


def safe_url(url: str | None) -> str | None:
    """Renvoie l'URL seulement si le scheme est http/https (anti-injection)."""
    if not url:
        return None
    try:
        parsed = urlparse(url)
    except ValueError:
        return None
    return url if parsed.scheme in ("http", "https") else None


def format_sources(sources: list[dict]) -> list[str]:
    """Formate les sources en lignes Markdown (titre — ville — _dates_ + lien)."""
    lines: list[str] = []
    for s in sources:
        title = s.get("title") or "?"
        city = s.get("city") or ""
        dr = s.get("daterange") or ""
        url = safe_url(s.get("url"))
        line = f"**{title}**"
        if city:
            line += f" — {city}"
        if dr:
            line += f" — _{dr}_"
        if url:
            line += f"\n\n[Voir sur Open Agenda]({url})"
        lines.append(line)
    return lines


class ApiError(Exception):
    """Erreur métier renvoyée par l'API (400/503), message destiné à l'UI."""


@dataclass
class ChatResult:
    answer: str
    sources: list[dict]
    session_id: str
    history_len: int


class ChatClient:
    """Client HTTP minimal vers l'API du slice. `session` injectable pour les tests."""

    def __init__(self, base_url: str, session=None, timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self._session = session or requests.Session()
        self.timeout = timeout

    def health(self) -> dict:
        r = self._session.get(f"{self.base_url}/health", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def send(self, message: str, session_id: str, use_memory: bool = True,
             use_reranker: bool = True) -> ChatResult:
        payload = {
            "message": message, "session_id": session_id,
            "use_memory": use_memory, "use_reranker": use_reranker,
        }
        r = self._session.post(f"{self.base_url}/chat", json=payload, timeout=self.timeout)
        if r.status_code == 400:
            raise ApiError(r.json().get("detail", "Requête invalide."))
        if r.status_code == 503:
            raise ApiError("Service en cours d'initialisation, réessaie dans un instant.")
        r.raise_for_status()
        d = r.json()
        return ChatResult(d["answer"], d["sources"], d["session_id"], d["history_len"])

    def reset(self, session_id: str) -> dict:
        r = self._session.post(
            f"{self.base_url}/reset", params={"session_id": session_id}, timeout=self.timeout
        )
        r.raise_for_status()
        return r.json()

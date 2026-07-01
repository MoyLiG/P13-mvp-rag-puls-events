"""Mémoire conversationnelle court terme (Redis).

Le POC était *stateless*. Ici chaque session garde une fenêtre des derniers
échanges, réinjectée dans le RAG pour résoudre les questions de suivi
(« et le week-end prochain ? »). Liste Redis par session + TTL d'inactivité.
"""
from __future__ import annotations

import json
import logging

import redis
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from pulsevents_mvp.config import Settings

logger = logging.getLogger(__name__)


class RedisMemory:
    """Historique court terme par session_id (fenêtre glissante)."""

    def __init__(self, settings: Settings):
        self.cfg = settings.redis
        self.client = redis.Redis(
            host=self.cfg.host, port=self.cfg.port, db=self.cfg.db,
            decode_responses=True,
        )
        self.max_messages = self.cfg.window_turns * 2  # un tour = 1 question + 1 réponse

    def _key(self, session_id: str) -> str:
        return f"chat:{session_id}"

    def get_history(self, session_id: str) -> list[BaseMessage]:
        """Renvoie l'historique (fenêtre) en messages LangChain."""
        raw = self.client.lrange(self._key(session_id), 0, -1)
        history: list[BaseMessage] = []
        for item in raw:
            msg = json.loads(item)
            if msg["role"] == "human":
                history.append(HumanMessage(content=msg["content"]))
            else:
                history.append(AIMessage(content=msg["content"]))
        return history

    def append(self, session_id: str, question: str, answer: str) -> None:
        """Ajoute un échange et tronque à la fenêtre + rafraîchit le TTL."""
        key = self._key(session_id)
        pipe = self.client.pipeline()
        pipe.rpush(key, json.dumps({"role": "human", "content": question}))
        pipe.rpush(key, json.dumps({"role": "ai", "content": answer}))
        pipe.ltrim(key, -self.max_messages, -1)   # garde les N derniers messages
        pipe.expire(key, self.cfg.ttl_s)
        pipe.execute()

    def reset(self, session_id: str) -> None:
        self.client.delete(self._key(session_id))

    def ping(self) -> bool:
        try:
            return bool(self.client.ping())
        except redis.RedisError:
            return False

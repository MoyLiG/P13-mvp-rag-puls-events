"""API FastAPI du slice (remplace la démo Streamlit du POC).

POST /chat  : question + session_id -> réponse RAG avec mémoire conversationnelle.
GET  /health: état des dépendances (pgvector, redis).
POST /reset : vide la mémoire d'une session.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pulsevents_mvp.config import load_settings
from pulsevents_mvp.loader import load_if_needed
from pulsevents_mvp.memory import RedisMemory
from pulsevents_mvp.pgstore import count_embeddings
from pulsevents_mvp.rag import MvpRagPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Puls-Events MVP — slice RAG", version="0.1.0")

_settings = None
_pipeline: MvpRagPipeline | None = None
_memory: RedisMemory | None = None


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str = Field(default="default")
    use_memory: bool = True
    use_reranker: bool = True


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    session_id: str
    history_len: int


@app.on_event("startup")
def _startup() -> None:
    global _settings, _pipeline, _memory
    _settings = load_settings()
    # Auto-load : si l'index pgvector est vide, le charger depuis le cache P11
    # (zéro coût Mistral). Rend `docker compose up` suffisant, sans étape manuelle.
    load_if_needed(_settings)
    _memory = RedisMemory(_settings)
    _pipeline = MvpRagPipeline(_settings)
    logger.info("API prête (chunks pgvector=%d)", count_embeddings(_settings))


@app.get("/health")
def health() -> dict:
    chunks = count_embeddings(_settings) if _settings else 0
    return {
        "status": "ok" if _pipeline else "starting",
        "pgvector_chunks": chunks,
        "redis": _memory.ping() if _memory else False,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if _pipeline is None or _memory is None:
        raise HTTPException(503, "Service en cours d'initialisation.")
    history = _memory.get_history(req.session_id) if req.use_memory else []
    try:
        result = _pipeline.answer(
            req.message, chat_history=history, use_reranker=req.use_reranker
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    if req.use_memory:
        _memory.append(req.session_id, req.message, result.answer)
    return ChatResponse(
        answer=result.answer,
        sources=result.sources,
        session_id=req.session_id,
        history_len=len(history) + (2 if req.use_memory else 0),
    )


@app.post("/reset")
def reset(session_id: str = "default") -> dict:
    if _memory is None:
        raise HTTPException(503, "Service en cours d'initialisation.")
    _memory.reset(session_id)
    return {"reset": session_id}

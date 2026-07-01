"""UI Streamlit du slice MVP — client HTTP de l'API FastAPI.

Démontre le saut POC stateless -> MVP stateful : toggle Mémoire ON/OFF.
Lancement (conteneur) : streamlit run streamlit_app.py
"""
from __future__ import annotations

import os
import uuid

import requests
import streamlit as st

from pulsevents_mvp.ui_client import ApiError, ChatClient, format_sources

API_URL = os.environ.get("API_URL", "http://localhost:8000")
MAX_QUERIES_PER_SESSION = 30
MAX_QUESTION_LEN = 500

st.set_page_config(page_title="Puls-Events MVP", page_icon="🎭", layout="wide")


@st.cache_resource
def get_client() -> ChatClient:
    return ChatClient(API_URL)


def _init_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex
    if "messages" not in st.session_state:
        st.session_state.messages = []  # [{"role": "user"|"assistant", "content", "sources"}]
    if "query_count" not in st.session_state:
        st.session_state.query_count = 0


def _sidebar(client: ChatClient) -> tuple[bool, bool]:
    """Panneau de preuve + contrôles. Renvoie (use_memory, use_reranker)."""
    with st.sidebar:
        st.header("MVP — état de la stack")
        try:
            health = client.health()
            st.success("API connectée")
            st.metric("Chunks pgvector", health.get("pgvector_chunks", 0))
            st.write(f"**Redis :** {'🟢 ok' if health.get('redis') else '🔴 ko'}")
        except requests.RequestException:
            st.error("API injoignable. Lance `docker compose up -d api`.")

        st.divider()
        use_memory = st.toggle("🧠 Mémoire conversationnelle", value=True,
                               help="OFF = stateless (POC). ON = stateful (MVP, Redis).")
        use_reranker = st.toggle(
            "🎯 Re-ranking (bge cross-encoder)", value=False,
            help="OFF par défaut : benché, dégrade le recall sur ce corpus (rapport §3.8). "
                 "ON = re-ranking du pool (nécessite USE_RERANKER=true côté serveur).",
        )
        st.caption(f"Session : `{st.session_state.session_id[:8]}`")
        if st.button("🔄 Réinitialiser la conversation"):
            try:
                client.reset(st.session_state.session_id)
            except requests.RequestException:
                pass
            st.session_state.session_id = uuid.uuid4().hex
            st.session_state.messages = []
            st.session_state.query_count = 0
            st.rerun()

        st.divider()
        st.markdown(
            "**Exemples :**\n"
            "- Quels concerts à Nantes ce mois-ci ?\n"
            "- Et le week-end prochain ?\n"
            "- Des expositions pour enfants en Loire-Atlantique ?"
        )
        remaining = MAX_QUERIES_PER_SESSION - st.session_state.query_count
        st.metric("Questions restantes (session)", remaining)
    return use_memory, use_reranker


def _render_history() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📍 Sources ({len(msg['sources'])})"):
                    for line in format_sources(msg["sources"]):
                        st.markdown(line)
                        st.divider()


def main() -> None:
    st.title("🎭 Puls-Events — MVP")
    st.caption("RAG pgvector + mémoire conversationnelle (Redis) — démo live")
    _init_state()
    client = get_client()
    use_memory, use_reranker = _sidebar(client)
    _render_history()

    question = st.chat_input(f"Pose ta question (max {MAX_QUESTION_LEN} caractères)...")
    if not question:
        return

    if st.session_state.query_count >= MAX_QUERIES_PER_SESSION:
        st.error("Quota de session atteint. Réinitialise la conversation.")
        return
    if len(question) > MAX_QUESTION_LEN:
        st.error(f"Question trop longue ({len(question)} > {MAX_QUESTION_LEN}).")
        return
    st.session_state.query_count += 1

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Recherche dans la base..."):
            try:
                result = client.send(
                    question, st.session_state.session_id, use_memory, use_reranker
                )
            except ApiError as exc:
                st.error(str(exc))
                return
            except requests.ConnectionError:
                st.error("API injoignable. Lance `docker compose up -d api`.")
                return
        st.markdown(result.answer)
        if result.sources:
            with st.expander(f"📍 Sources ({len(result.sources)})"):
                for line in format_sources(result.sources):
                    st.markdown(line)
                    st.divider()
    st.session_state.messages.append(
        {"role": "assistant", "content": result.answer, "sources": result.sources}
    )


if __name__ == "__main__":
    main()

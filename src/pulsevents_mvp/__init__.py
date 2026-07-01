"""Slice MVP Puls-Events (P13) — RAG sur pgvector + mémoire conversationnelle Redis.

Évolution du POC P11 (FAISS local, stateless) vers les briques MVP :
- vector store FAISS -> PostgreSQL + pgvector (index HNSW) ;
- mémoire conversationnelle (Redis) + RAG history-aware ;
- API FastAPI (remplace la démo Streamlit).
"""
__version__ = "0.1.0"

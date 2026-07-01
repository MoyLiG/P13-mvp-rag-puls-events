# Image de l'API du slice MVP (Linux, Python 3.12 — pas de version hell).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# build-essential pour rank_bm25 / faiss éventuels ; nettoyé ensuite.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# torch CPU-only AVANT le reste : sinon sentence-transformers (re-ranking) tire la
# variante CUDA de torch — plusieurs Go de wheels nvidia-* inutiles. Le cross-encoder
# tourne en CPU pour la slice MVP (cf. spec re-ranking §7).
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install -r requirements.txt

# Cache HuggingFace dans l'image + pré-téléchargement du cross-encoder de re-ranking
# (~600 Mo) au build : le conteneur est autonome, pas de téléchargement au runtime.
ENV HF_HOME=/app/.hf_cache
RUN python -c "from langchain_community.cross_encoders import HuggingFaceCrossEncoder; HuggingFaceCrossEncoder(model_name='BAAI/bge-reranker-v2-m3')"

COPY config.yaml .
COPY streamlit_app.py .
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/

EXPOSE 8000
CMD ["uvicorn", "pulsevents_mvp.api:app", "--host", "0.0.0.0", "--port", "8000"]

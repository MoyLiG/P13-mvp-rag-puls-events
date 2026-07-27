# Image de l'API du slice MVP (Linux, Python 3.12 — pas de version hell).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# build-essential pour compiler rank_bm25 / faiss si aucune wheel n'est dispo.
#
# ATTENTION : le `rm -rf` ne supprime que le CACHE apt, pas les paquets. La
# toolchain (15 paquets gcc/g++) reste dans l'image finale : **~322 Mo**, soit
# 10 % de son poids. Le commentaire precedent annoncait « nettoye ensuite »,
# c'etait faux.
#
# Non corrige volontairement : purger ici ne gagnerait rien (la couche
# precedente conserve les fichiers), il faudrait soit fusionner apt et pip dans
# un seul RUN — ce qui invalide le cache pip a chaque changement de deps — soit
# passer en multi-stage. Dette assumee, tracee dans tasks/lessons.md.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# torch CPU-only AVANT le reste : sinon sentence-transformers (re-ranking) tire la
# variante CUDA de torch — plusieurs Go de wheels nvidia-* inutiles. Le cross-encoder
# tourne en CPU pour la slice MVP (cf. spec re-ranking §7).
# Les deux `pip install` sont deux résolutions SÉPARÉES : la seconde peut remplacer
# le torch CPU par la variante CUDA sans faire échouer le build. Les deux gardes
# ci-dessous ferment ce trou — un build vert prouve alors l'environnement, pas juste
# l'absence d'erreur d'installation.
#   - `pip check`   : cohérence des contraintes après une install en plusieurs passes.
#   - assertion CPU : le suffixe local `+cpu` a disparu => variante CUDA revenue.
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install -r requirements.txt \
    && pip check \
    && python -c "import torch,sys; v=torch.__version__; print('torch',v); sys.exit(0 if v.endswith('+cpu') else 1)"

# Cache HuggingFace dans le volume monté ./data, PAS dans l'image.
#
# Le pré-téléchargement du cross-encoder au build a été retiré : mesuré à **2,2 Go**
# (le commentaire précédent annonçait ~600 Mo, faux d'un facteur 4), soit 32 % du poids
# de l'image — pour une fonctionnalité désactivée par défaut. `config.yaml` porte
# `use_reranker: false`, benché : le cross-encoder dégrade le recall sur ce corpus
# (0.64 contre 0.86 pour RRF-top6, cf. rapport §3.8).
#
# Conséquence : le modèle se télécharge au premier usage réel si quelqu'un active le
# toggle, une seule fois, et persiste sur l'hôte entre les recréations de conteneur
# puisque `./data` est monté (cf. docker-compose.yml). Le chemin est hors contexte de
# build (`.dockerignore`), donc il n'alourdit pas non plus les builds suivants.
ENV HF_HOME=/app/data/.hf_cache

COPY config.yaml .
COPY streamlit_app.py .
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/

EXPOSE 8000
CMD ["uvicorn", "pulsevents_mvp.api:app", "--host", "0.0.0.0", "--port", "8000"]

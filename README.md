# Slice vertical MVP — RAG pgvector + mémoire conversationnelle

Preuve exécutable du saut **POC P11 → MVP** : le RAG migré de FAISS local vers
**PostgreSQL + pgvector** (index HNSW), exposé en **API FastAPI**, avec
**mémoire conversationnelle Redis**. Tout en conteneurs.

## Ce que ça démontre
- Migration FAISS → pgvector sans réécrire le retriever hybride (même interface VectorStore).
- Mémoire : le chatbot résout les questions de suivi (« et le week-end prochain ? »).
- Chiffres mesurés : recall@k dense vs hybride, latence FAISS vs pgvector.

## Prérequis
- Docker + Docker Compose, **lancés depuis WSL2** (stack Linux-first).
- Le projet **P11** présent dans `../P11` (données réutilisées en lecture seule).
- Une clé Mistral.

## Démarrage (un seul `docker compose up`)

```bash
cp .env.example .env        # renseigner MISTRAL_API_KEY
docker compose up --build -d
# L'API auto-charge l'index pgvector au 1er démarrage (cache P11, zéro coût Mistral).
```

- **UI web** : http://localhost:8501 (Streamlit)
- **API** : http://localhost:8000/health

## Démo soutenance (mémoire ON/OFF)

Dans l'UI, le toggle **🧠 Mémoire** rend visible le saut stateless → stateful :
1. **OFF** : « Quelles expositions à Saint-Nazaire ? » → « Et pour les enfants ? »
   ne garde plus le contexte (réponse générique Pays de la Loire).
2. **ON** (+ Réinitialiser) : la même question de suivi reste ancrée sur Saint-Nazaire.

> Remarque : le corpus est très nantais ; le contraste se lit mieux sur une ville
> non-Nantes (Saint-Nazaire) où la mémoire change visiblement le retrieval.

## Benchmarks (les chiffres pour la soutenance)

```bash
docker compose run --rm api python scripts/bench_retrieval.py   # recall + latence
docker compose run --rm api python scripts/bench_memory.py      # effet mémoire
```

## Tests

```bash
docker compose run --rm api pytest -q
```

## Hors périmètre (designé dans le rapport, non implémenté ici)
Géo/PostGIS, recherche web/smolagents, Langfuse/Cockpit, mémoire long terme,
CI/CD, déploiement Scaleway prod. Cohérent avec un slice.

## Architecture du slice
```
Utilisateur ──HTTP──> FastAPI ──┬─> Redis     (mémoire court terme)
                                ├─> pgvector  (retrieval hybride BM25+dense, HNSW)
                                └─> Mistral   (reformulation suivi + génération)
```

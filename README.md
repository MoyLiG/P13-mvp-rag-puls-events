# Slice vertical MVP — RAG pgvector + mémoire conversationnelle

Preuve exécutable du saut **POC P11 → MVP** : le RAG migré de FAISS local vers
**PostgreSQL + pgvector** (index HNSW), exposé en **API FastAPI**, avec
**mémoire conversationnelle Redis**. Tout en conteneurs.

## Ce que ça démontre
- Migration FAISS → pgvector sans réécrire le retriever hybride (même interface VectorStore).
- Mémoire : le chatbot résout les questions de suivi (« et le week-end prochain ? »).
- Chiffres mesurés : recall@k dense vs hybride, latence FAISS vs pgvector.

## Du POC au MVP

![Schéma du passage du POC au MVP : en bleu le socle hérité du POC P11 (Open Agenda, embeddings mistral-embed, retriever hybride BM25 et dense, mistral-small, réponse sourcée), en vert ce que le MVP fait évoluer (ingestion vers Kestra, FAISS local vers pgvector, Streamlit vers FastAPI SSE), et les quatre besoins ajoutés : mémoire conversationnelle Redis et PostgreSQL, contexte géographique PostGIS, web temps réel smolagents et Tavily, monitoring Langfuse et Cockpit — le tout sur Scaleway](docs/slides/poc-vers-mvp.png)

Le MVP **repart du schéma du POC** ([P11](https://github.com/MoyLiG/P11)) au lieu
de le refaire. En bleu, le socle hérité qui ne bouge pas : retriever hybride
BM25+dense, `mistral-small`, réponses sourcées. En vert, ce qui évolue —
script Python → Kestra, FAISS local → pgvector, Streamlit → FastAPI SSE. Les
quatre besoins du cahier des charges se greffent sur ce pipeline existant.

L'architecture MVP tient en quatre décisions :

- **Une seule base PostgreSQL** : relationnel + pgvector + PostGIS + mémoire.
- **Cœur RAG du POC réutilisé** : Mistral + LangChain + retriever hybride.
- **Agent smolagents** : arbitre entre base interne et recherche web.
- **Double observabilité** : Cockpit (infra) + Langfuse (qualité RAG).

**Ce dépôt implémente la tranche verticale** mémoire + pgvector de ce schéma.

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

Ce que le MVP complet prévoit, et que ce dépôt **ne déploie pas** :

![Architecture MVP cible : sources externes Opendatasoft, Mistral et Tavily ; ingestion planifiée par Kestra ; stockage Managed PostgreSQL avec pgvector, PostGIS et mémoire long terme, Managed Redis pour la mémoire court terme et Object Storage S3 ; applications en Serverless Containers avec une API FastAPI SSE et des agents smolagents orchestrant les outils RAG, web, géo et mémoire ; observabilité par Langfuse et Cockpit — l'ensemble hébergé sur Scaleway](docs/slides/architecture-mvp.png)

## Pilotage : délais et budget

![Pilotage : phasage en quatre phases sur environ dix semaines — socle et infra, mémoire et géo, web et API, monitoring — et budget estimé à environ 160 euros par mois réparti entre Mistral, Tavily, Scaleway PostgreSQL, Serverless et Redis, Langfuse et Cockpit étant gratuits](docs/slides/pilotage-budget.png)

**~10 semaines** en quatre phases : socle et infra (2 sem.), mémoire et géo
(3 sem.), web et API (3 sem.), monitoring (2 sem.).

**~160 €/mois** pour 1 000 requêtes/jour : Scaleway PostgreSQL ~80 €, Tavily
~40 €, Serverless + Redis ~25 €, Mistral ~16 € (génération + embeddings),
Langfuse et Cockpit gratuits. Le coût est **dominé par l'infra fixe** à ce
volume — le LLM ne pèse que ~10 %, mais c'est lui qui scale avec le trafic.

## Architecture du slice
```
Utilisateur ──HTTP──> FastAPI ──┬─> Redis     (mémoire court terme)
                                ├─> pgvector  (retrieval hybride BM25+dense, HNSW)
                                └─> Mistral   (reformulation suivi + génération)
```

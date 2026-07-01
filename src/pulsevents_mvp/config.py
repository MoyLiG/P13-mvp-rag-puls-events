"""Configuration du slice MVP.

Hérite du schéma P11 (config.yaml) et ajoute, depuis l'environnement
(injecté par docker-compose) : la base PostgreSQL/pgvector, Redis, et le
chemin vers les données P11 réutilisées (parquet, cache d'embeddings,
index FAISS, jeu d'évaluation).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


# --- schéma hérité de P11 -------------------------------------------------
class DataSourceConfig(BaseModel):
    base_url: str
    page_size: int = Field(default=100, ge=1, le=100)
    request_timeout_s: int = 30
    retry_attempts: int = 4
    retry_initial_wait_s: int = 1


class FiltersConfig(BaseModel):
    region: str
    department: Optional[str] = None
    city: Optional[str] = None
    since_days: int = Field(default=365, ge=1)
    max_records: int = Field(default=5000, ge=1)
    time_mode: str = "upcoming"


class PreprocessingConfig(BaseModel):
    min_description_length: int = 30
    exclude_title_keywords: list[str] = []
    exclude_agendas: list[str] = []


class ChunkingConfig(BaseModel):
    chunk_size: int = 1200
    chunk_overlap: int = 80


class ModelsConfig(BaseModel):
    embedding_model: str = "mistral-embed"
    llm_model: str = "mistral-small-latest"
    llm_temperature: float = 0.2
    embedding_batch_size: int = 24
    embedding_min_interval_s: float = 1.1


class RetrievalConfig(BaseModel):
    search_type: str = "mmr"
    k: int = 6
    fetch_k: int = 18
    max_question_length: int = 500
    max_answer_tokens: int = 400
    use_hybrid: bool = True
    bm25_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    # re-ranking cross-encoder (reco v1 P11). Defaut False : le benchmark montre
    # qu'il degrade le recall sur ce corpus (cf. rapport 3.8) -> opt-in.
    use_reranker: bool = False
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_candidates: int = Field(default=24, ge=1)
    # tokeniseur FR pour BM25 (dé-accent + stopwords + stem). Défaut False : à
    # activer si le bench loyal montre un gain recall@k (cf. rapport 3.8).
    use_fr_tokenizer: bool = False


class PathsConfig(BaseModel):
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    vectorstore_dir: str = "data/vectorstore"
    eval_dataset: str = "data/eval/qa_dataset.json"
    eval_results: str = "data/eval/results.csv"


# --- nouveau : infra MVP --------------------------------------------------
class DatabaseConfig(BaseModel):
    host: str = "localhost"
    port: int = 5432
    user: str = "pulse"
    password: str = "pulse"
    name: str = "pulsevents"
    collection: str = "events"
    # paramètres index HNSW (le point soutenance : compromis rappel/latence)
    hnsw_m: int = 16
    hnsw_ef_construction: int = 64
    hnsw_ef_search: int = 40

    @property
    def connection_string(self) -> str:
        """DSN SQLAlchemy/psycopg3 attendu par langchain-postgres."""
        return (
            f"postgresql+psycopg://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}"
        )


class RedisConfig(BaseModel):
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    ttl_s: int = 3600          # expiration d'une session inactive
    window_turns: int = 4      # nb d'échanges (Q+R) gardés dans la fenêtre courte


class Settings(BaseModel):
    data_source: DataSourceConfig
    filters: FiltersConfig
    preprocessing: PreprocessingConfig
    chunking: ChunkingConfig
    models: ModelsConfig
    retrieval: RetrievalConfig
    paths: PathsConfig
    database: DatabaseConfig
    redis: RedisConfig
    mistral_api_key: Optional[str] = None
    project_root: Path
    p11_data_dir: Path             # données P11 réutilisées (montées en ro dans le conteneur)

    def resolved_path(self, relative: str) -> Path:
        return (self.project_root / relative).resolve()

    # raccourcis vers les données P11 réutilisées
    @property
    def p11_parquet(self) -> Path:
        return self.p11_data_dir / "processed" / "events_clean.parquet"

    @property
    def p11_embed_cache(self) -> Path:
        return self.p11_data_dir / "embed_cache"

    @property
    def p11_faiss_dir(self) -> Path:
        return self.p11_data_dir / "vectorstore"

    @property
    def p11_qa_dataset(self) -> Path:
        return self.p11_data_dir / "eval" / "qa_dataset.json"

    @property
    def local_embed_cache(self) -> Path:
        return self.project_root / "data" / "embed_cache"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def load_settings(config_path: Optional[Path] = None) -> Settings:
    """Charge config.yaml + secrets/infra depuis l'environnement."""
    root = _project_root()
    load_dotenv(root / ".env", override=False)

    cfg_path = Path(config_path) if config_path else root / "config.yaml"
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration introuvable : {cfg_path}")

    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    raw.setdefault("retrieval", {})
    raw["retrieval"]["use_reranker"] = _env_bool(
        "USE_RERANKER", raw["retrieval"].get("use_reranker", False)
    )
    raw["retrieval"]["use_fr_tokenizer"] = _env_bool(
        "USE_FR_TOKENIZER", raw["retrieval"].get("use_fr_tokenizer", False)
    )

    raw["project_root"] = root
    raw["mistral_api_key"] = os.environ.get("MISTRAL_API_KEY")
    raw["p11_data_dir"] = Path(
        os.environ.get("P11_DATA_DIR", root.parent / "P11" / "data")
    )
    raw["database"] = DatabaseConfig(
        host=os.environ.get("PG_HOST", "localhost"),
        port=_env_int("PG_PORT", 5432),
        user=os.environ.get("PG_USER", "pulse"),
        password=os.environ.get("PG_PASSWORD", "pulse"),
        name=os.environ.get("PG_DB", "pulsevents"),
        hnsw_ef_search=_env_int("HNSW_EF_SEARCH", 40),
    )
    raw["redis"] = RedisConfig(
        host=os.environ.get("REDIS_HOST", "localhost"),
        port=_env_int("REDIS_PORT", 6379),
    )
    return Settings.model_validate(raw)

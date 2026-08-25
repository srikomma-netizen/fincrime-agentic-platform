"""Central configuration. Reads from environment / .env with offline-safe defaults."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

try:
    from dotenv import load_dotenv  # optional
    load_dotenv()
except Exception:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parents[2]


class Settings:
    def __init__(self) -> None:
        self.llm_provider: str = os.getenv("LLM_PROVIDER", "mock").lower()
        self.anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
        self.openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
        self.llm_model: str = os.getenv("LLM_MODEL", "claude-sonnet-4-5")

        self.rag_backend: str = os.getenv("RAG_BACKEND", "auto").lower()
        self.embedding_model: str = os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.enable_pii_masking: bool = _as_bool(os.getenv("ENABLE_PII_MASKING", "true"))

        self.risk_low_max: float = float(os.getenv("RISK_LOW_MAX", "0.35"))
        self.risk_high_min: float = float(os.getenv("RISK_HIGH_MIN", "0.70"))

        self.model_dir: Path = ROOT / os.getenv("MODEL_DIR", "artifacts")
        self.vector_dir: Path = ROOT / os.getenv("VECTOR_DIR", "artifacts/vector_index")
        self.policies_dir: Path = ROOT / "data" / "policies"
        self.synthetic_dir: Path = ROOT / "data" / "synthetic"

        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.vector_dir.mkdir(parents=True, exist_ok=True)


def _as_bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


@lru_cache
def get_settings() -> Settings:
    return Settings()

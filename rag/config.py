"""Settings from environment variables (a .env file is read if present)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()


@dataclass(frozen=True)
class Settings:
    database_url: str = os.environ.get("DATABASE_URL", "postgresql://rag:rag@127.0.0.1:5442/rag")
    llm_base_url: str = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
    llm_model: str = os.environ.get("LLM_MODEL", "llama3.1:8b")
    llm_api_key: str = os.environ.get("LLM_API_KEY", "none")
    judge_model: str = os.environ.get("JUDGE_MODEL", "") or os.environ.get("LLM_MODEL", "llama3.1:8b")
    embed_model: str = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    embed_dim: int = int(os.environ.get("EMBED_DIM", "384"))
    docs_dir: str = os.environ.get("DOCS_DIR", str(ROOT / "data" / "fastapi-docs"))
    top_k: int = int(os.environ.get("TOP_K", "6"))


settings = Settings()

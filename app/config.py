from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """.env 파일에서 설정을 읽어온다. 값이 없으면 아래 기본값을 쓴다."""

    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", env_file_encoding="utf-8")

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen3.5:9b"
    embed_model: str = "embeddinggemma"
    llm_num_ctx: int = 8192
    llm_keep_alive: int = -1  # -1: 모델을 계속 메모리에 올려둔다

    # Chroma
    chroma_dir: Path = BASE_DIR / "chroma_db"
    chroma_collection: str = "company_docs"

    # 문서
    docs_dir: Path = BASE_DIR / "data" / "docs"
    images_dir: Path = BASE_DIR / "data" / "images"

    # 인증
    jwt_secret: str = "change-me"


@lru_cache
def get_settings() -> Settings:
    return Settings()

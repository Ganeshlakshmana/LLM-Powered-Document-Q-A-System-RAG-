from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str

    data_dir: Path = Path("data")
    index_dir: Path = Path("indexes/faiss")

    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k: int = 3

settings = Settings()

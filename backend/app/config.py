from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DEEPLENS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    watched_folders: List[str] = [
        r"C:\Users\ASUS260922\Documents\PT.AEP",
        r"D:\PAJAK PERUSAHAAN BAPAK"
    ]
    chunk_size: int = 500
    chunk_overlap: int = 50
    embedding_model: str = "all-MiniLM-L6-v2"
    chroma_persist_dir: str = "./data/chroma"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"
    supported_extensions: List[str] = [
        # Documents
        ".pdf", ".docx", ".xlsx", ".xls", ".md", ".txt", ".html", ".csv",
        # Data / config
        ".json", ".xml", ".yaml", ".yml", ".toml", ".env",
        # Markup / text
        ".rst", ".log",
        # Email & ebooks
        ".eml", ".epub",
        # Source code
        ".py", ".js", ".ts", ".java", ".cpp", ".c",
        ".rs", ".go", ".rb", ".php", ".swift", ".kt",
    ]

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_size(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size", 500)
        if v >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v

    @property
    def chroma_persist_path(self) -> Path:
        project_root = Path(__file__).resolve().parents[2]
        path = Path(self.chroma_persist_dir)

        if not path.is_absolute():
            path = project_root / path

        return path.resolve()

# Module-level singleton — override via environment variables or .env
settings = Settings()

import secrets
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "engagement_model.joblib"
METADATA_PATH = ROOT / "models" / "engagement_model_metadata.joblib"


class Settings(BaseSettings):
    host: str = Field(default="0.0.0.0", description="Host para el servidor")
    port: int = Field(default=8000, description="Puerto para el servidor")
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="Origenes permitidos para CORS",
    )
    allow_credentials: bool = Field(
        default=True, description="Permitir credenciales en CORS"
    )
    rate_limit: str = Field(
        default="30/minute", description="Limite de peticiones por IP"
    )
    api_keys: str = Field(
        default="",
        description="API keys validas separadas por comas (vacío = sin auth)",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Nivel de logging"
    )
    model_path: Path = Field(
        default=MODEL_PATH, description="Ruta al modelo serializado"
    )
    metadata_path: Path = Field(
        default=METADATA_PATH, description="Ruta al metadata del modelo"
    )

    @property
    def valid_api_keys(self) -> set[str]:
        if not self.api_keys.strip():
            return set()
        return {key.strip() for key in self.api_keys.split(",") if key.strip()}

    def generate_api_key(self) -> str:
        return secrets.token_urlsafe(32)

    model_config = {"env_prefix": "GAMING_ML_", "env_file": ".env"}


settings = Settings()

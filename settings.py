"""
Configuration globale du système de chat IA.
Gère tous les paramètres d'infrastructure, modèle et sécurité.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List
from enum import Enum
import os


class ModelSize(Enum):
    """Tailles disponibles pour le modèle."""
    SMALL = "7B"
    MEDIUM = "13B"
    LARGE = "70B"
    XLARGE = "120B"


class InferenceMode(Enum):
    """Modes d'inférence disponibles."""
    STANDARD = "standard"
    FAST = "fast"
    ACCURATE = "accurate"
    CREATIVE = "creative"


@dataclass
class ModelConfig:
    """Configuration du modèle linguistique."""
    name: str = "darkgpt-llm"
    size: ModelSize = ModelSize.MEDIUM
    num_layers: int = 32
    num_attention_heads: int = 32
    hidden_dim: int = 5120
    vocab_size: int = 50257
    context_window: int = 8192
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True


@dataclass
class InferenceConfig:
    """Configuration du moteur d'inférence."""
    batch_size: int = 16
    max_queue_size: int = 1000
    timeout_seconds: int = 30
    gpu_layers: int = 0
    use_quantization: bool = True
    quantization_bits: int = 4
    flash_attention: bool = True
    kv_cache: bool = True
    mode: InferenceMode = InferenceMode.STANDARD


@dataclass
class SecurityConfig:
    """Configuration de la sécurité et du filtrage."""
    enable_content_filter: bool = True
    enable_input_sanitization: bool = True
    enable_output_filter: bool = True
    max_input_length: int = 4096
    max_output_length: int = 2048
    banned_patterns: List[str] = field(default_factory=list)
    rate_limit_requests: int = 60
    rate_limit_window: int = 60
    enable_jailbreak_detection: bool = True
    jailbreak_threshold: float = 0.85


@dataclass
class StorageConfig:
    """Configuration du stockage et de la mémoire."""
    vector_store_type: str = "pinecone"
    database_url: str = "postgresql://localhost:5432/chat_db"
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl: int = 3600
    max_conversation_history: int = 20
    embedding_dimension: int = 1536
    chunk_size: int = 512
    chunk_overlap: int = 50


@dataclass
class ServerConfig:
    """Configuration du serveur."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    max_connections: int = 1000
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    enable_websocket: bool = True
    enable_streaming: bool = True
    heartbeat_interval: int = 30
    ssl_enabled: bool = False


@dataclass
class AppConfig:
    """Configuration globale de l'application."""
    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    debug: bool = False
    environment: str = "production"
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Charge la configuration depuis les variables d'environnement."""
        config = cls()
        config.debug = os.getenv("DEBUG", "false").lower() == "true"
        config.environment = os.getenv("ENVIRONMENT", "production")
        config.log_level = os.getenv("LOG_LEVEL", "INFO")
        config.model.temperature = float(
            os.getenv("MODEL_TEMPERATURE", "0.7")
        )
        config.model.context_window = int(
            os.getenv("MODEL_CONTEXT_WINDOW", "8192")
        )
        config.server.port = int(os.getenv("PORT", "8000"))
        config.server.workers = int(os.getenv("WORKERS", "4"))
        config.database_url = os.getenv(
            "DATABASE_URL", config.database_url
        )
        config.redis_url = os.getenv("REDIS_URL", config.redis_url)
        return config

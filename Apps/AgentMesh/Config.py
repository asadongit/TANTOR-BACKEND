from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, Dict, Any, List
import os

class AppSettings(BaseSettings):
    """
    Central production configuration for AgentMesh Engine.
    Managed via environment variables and .env files.
    """
    # Database
    DATABASE_URL: str = Field(..., validation_alias="DATABASE_URL", repr=False)
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 10
    
    # Tunneling (Ngrok) — set NGROK_ENABLED=false in production to skip entirely
    NGROK_ENABLED: bool = Field(False, validation_alias="NGROK_ENABLED")
    NGROK_AUTHTOKEN: Optional[str] = Field(None, validation_alias="NGROK_AUTHTOKEN", repr=False)
    
    # External Service endpoints
    DATA_INGESTION_URL: str = "http://192.168.3.86:30080/api/v1/agent_mesh/agent-mesh-ingestion"
    
    # Engine Settings
    DEBUG: bool = False
    WORKFLOW_TIMEOUT_SEC: int = 300
    
    # Paths (Resolved at runtime)
    TOOLS_PATH: str = os.path.join(os.path.dirname(__file__), "Utilities", "Tools")
    # ── Single root — change only AGENTMESH_ROOT to relocate EVERYTHING ──
    AGENTMESH_ROOT: str = Field("./mnt", validation_alias="AGENTMESH_ROOT")
    # Individual overrides (optional — default to subdirs of AGENTMESH_ROOT)
    TEMP_MOUNT: Optional[str] = Field(None, validation_alias="AGENTMESH_TEMP_MOUNT")
    ARTIFACT_MOUNT: Optional[str] = Field(None, validation_alias="AGENTMESH_ARTIFACT_MOUNT")
    AUDIO_MOUNT: Optional[str] = Field(None, validation_alias="AGENTMESH_AUDIO_MOUNT")
    EXPORTS_MOUNT: Optional[str] = Field(None, validation_alias="AGENTMESH_EXPORTS_MOUNT")
    UPLOAD_MOUNT: Optional[str] = Field(None, validation_alias="AGENTMESH_UPLOAD_MOUNT")
    TEMPLATES_MOUNT: Optional[str] = Field(None, validation_alias="AGENTMESH_TEMPLATES_MOUNT")
    TEMP_IMAGES_MOUNT: Optional[str] = Field(None, validation_alias="AGENTMESH_TEMP_IMAGES_MOUNT")
    DEEP_AGENTS_MOUNT: Optional[str] = Field(None, validation_alias="AGENTMESH_DEEP_AGENTS_MOUNT")

    # Checkpoint artifact externalization. Large binary values are written to
    # ARTIFACT_MOUNT before LangGraph persists the checkpoint in PostgreSQL.
    CHECKPOINT_ARTIFACTS_ENABLED: bool = Field(
        True,
        validation_alias="AGENTMESH_CHECKPOINT_ARTIFACTS_ENABLED",
    )
    CHECKPOINT_ARTIFACT_THRESHOLD_BYTES: int = Field(
        8 * 1024,
        validation_alias="AGENTMESH_CHECKPOINT_ARTIFACT_THRESHOLD_BYTES",
        ge=1024,
    )
    CHECKPOINT_ARTIFACT_MAX_BYTES: int = Field(
        100 * 1024 * 1024,
        validation_alias="AGENTMESH_CHECKPOINT_ARTIFACT_MAX_BYTES",
        ge=1024,
    )
    CHECKPOINT_DELETE_ON_COMPLETION: bool = Field(
        True,
        validation_alias="AGENTMESH_CHECKPOINT_DELETE_ON_COMPLETION",
    )
    HISTORY_TTL_SECONDS: int = Field(
        30 * 24 * 60 * 60,
        validation_alias="AGENTMESH_HISTORY_TTL_SECONDS",
        ge=1,
    )
    HISTORY_CLEANUP_INTERVAL_SECONDS: int = Field(
        60,
        validation_alias="AGENTMESH_HISTORY_CLEANUP_INTERVAL_SECONDS",
        ge=1,
    )
    
    # LLM Defaults
    DEFAULT_LLM_PROVIDER: str = "groq"
    DEFAULT_LLM_MODEL: str = "llama-3.3-70b-versatile"
    
    # Langfuse Observability
    LANGFUSE_PUBLIC_KEY: Optional[str] = Field(None, validation_alias="LANGFUSE_PUBLIC_KEY")
    LANGFUSE_SECRET_KEY: Optional[str] = Field(None, validation_alias="LANGFUSE_SECRET_KEY", repr=False)
    LANGFUSE_BASE_URL: str = Field("https://cloud.langfuse.com", validation_alias="LANGFUSE_HOST")

    # Toxicity scoring (Detoxify) — per-node, logged to Langfuse as a `toxicity` score
    TOXICITY_ENABLED: bool = Field(True, validation_alias="TOXICITY_ENABLED")
    TOXICITY_MODEL: str = Field("original", validation_alias="TOXICITY_MODEL")
    TOXICITY_THRESHOLD: float = Field(0.5, validation_alias="TOXICITY_THRESHOLD")
    TOXICITY_MAX_CHARS: int = Field(1500, validation_alias="TOXICITY_MAX_CHARS")
    TOXICITY_TIMEOUT_SEC: float = Field(30.0, validation_alias="TOXICITY_TIMEOUT_SEC")
    
    # Security
    MASTER_KEY: str = Field(..., validation_alias="AGENTMESH_MASTER_KEY", description="Required for database encryption. In prod, injected via vault_key.", repr=False)
    PREVIOUS_MASTER_KEYS: Optional[str] = Field(
        None,
        validation_alias="AGENTMESH_PREVIOUS_MASTER_KEYS",
        description="Comma-separated old Fernet keys accepted only for decryption during key rotation.",
        repr=False,
    )
    SENSITIVE_DATA_MIGRATION_BATCH_SIZE: int = Field(
        100,
        validation_alias="AGENTMESH_SENSITIVE_DATA_MIGRATION_BATCH_SIZE",
        ge=1,
        le=1000,
        description="Maximum rows held in memory during the one-time legacy secret migration.",
    )

    # Redis — concurrency guards for workflow execution
    REDIS_URL: str = Field("redis://localhost:6379", validation_alias="REDIS_URL")
    REDIS_PASSWORD: Optional[str] = Field(None, validation_alias="REDIS_PASSWORD", repr=False)

    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    _MOUNT_SUBDIRS = {
        "TEMP_MOUNT": "temp",
        "ARTIFACT_MOUNT": "artifacts",
        "UPLOAD_MOUNT": "temp/uploads",
        "TEMP_IMAGES_MOUNT": "temp/temp_images",
        "EXPORTS_MOUNT": "temp/exports",
        "AUDIO_MOUNT": "temp/audio",
        "TEMPLATES_MOUNT": "temp/templates",
        "DEEP_AGENTS_MOUNT": "temp/deep_agents",
    }

    @model_validator(mode="after")
    def _resolve_mount_defaults(self):
        root = self.AGENTMESH_ROOT
        for field, sub in self._MOUNT_SUBDIRS.items():
            if getattr(self, field) is None:
                object.__setattr__(self, field, os.path.join(root, sub))
        return self

    @property
    def async_database_url(self) -> str:
        """Ensures the string uses asyncpg driver."""
        url = self.DATABASE_URL
        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+asyncpg://", 1)
        return url

settings = AppSettings()

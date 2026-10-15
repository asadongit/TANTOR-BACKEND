import sys
import asyncio
import logging
import os
from typing import Optional, Any
from langgraph.checkpoint.memory import MemorySaver
from Apps.AgentMesh.Config import settings

logger = logging.getLogger(__name__)

# On Windows, psycopg async requires SelectorEventLoop instead of default ProactorEventLoop
if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception as e:
        logger.warning(f"Could not set WindowsSelectorEventLoopPolicy: {e}")

_pool: Optional[Any] = None
_saver: Optional[Any] = None


def _env_int(key: str, default: int) -> int:
    """Read an integer setting from the environment, falling back on bad input."""
    try:
        return int(str(os.getenv(key, default)).strip())
    except (TypeError, ValueError):
        logger.warning(f"Invalid value for {key}; using default {default}")
        return default


def _env_float(key: str, default: float) -> float:
    """Read a float setting from the environment, falling back on bad input."""
    try:
        return float(str(os.getenv(key, default)).strip())
    except (TypeError, ValueError):
        logger.warning(f"Invalid value for {key}; using default {default}")
        return default


def get_connection_string() -> str:
    db_url = settings.DATABASE_URL or ""
    if db_url.startswith("postgresql+asyncpg://"):
        db_url = db_url.replace("postgresql+asyncpg://", "postgresql://", 1)
    elif db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    return db_url


async def init_checkpointer() -> Any:
    """
    Initializes the shared AsyncPostgresSaver checkpointer with an AsyncConnectionPool.
    Executes idempotent table setup. Called during FastAPI startup lifespan.
    """
    global _pool, _saver
    db_url = get_connection_string()
    try:
        from psycopg_pool import AsyncConnectionPool
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        # ─────────────────────────────────────────────────────────────────
        # Pool liveness.
        #
        # The database is typically on another host, and a workflow run may be
        # the first database activity in hours. Idle TCP connections are routinely
        # dropped in between by firewalls, NAT tables and Postgres' own
        # idle_session_timeout. Without a liveness check the pool hands out a
        # dead connection, LangGraph fails to write its first checkpoint, and the
        # graph stalls before executing a single node — logging only
        # "discarding closed connection" with no traceback.
        #
        #   check         validates a connection before it is handed out; a dead
        #                 one is discarded and replaced instead of returned. This
        #                 is the fix that matters, because max_idle only reaps
        #                 connections ABOVE min_size — the min_size baseline
        #                 connections are kept forever and are exactly the ones
        #                 that go stale.
        #   max_idle      retire surplus idle connections before the network does.
        #   max_lifetime  recycle even busy connections periodically, so a
        #                 long-lived process cannot accumulate broken ones.
        #
        # All tunable via env; defaults are chosen to sit below common firewall
        # idle timeouts (typically 5-10 minutes).
        # ─────────────────────────────────────────────────────────────────
        min_size = _env_int("AGENTMESH_DB_POOL_MIN_SIZE", 2)
        max_size = _env_int("AGENTMESH_DB_POOL_MAX_SIZE", 10)
        max_idle = _env_float("AGENTMESH_DB_POOL_MAX_IDLE_SEC", 300.0)
        max_lifetime = _env_float("AGENTMESH_DB_POOL_MAX_LIFETIME_SEC", 1800.0)
        pool_timeout = _env_float("AGENTMESH_DB_POOL_TIMEOUT_SEC", 30.0)
        reconnect_timeout = _env_float("AGENTMESH_DB_POOL_RECONNECT_TIMEOUT_SEC", 300.0)

        logger.info(
            "Initializing production AsyncPostgresSaver with AsyncConnectionPool "
            f"(min={min_size} max={max_size} max_idle={max_idle}s "
            f"max_lifetime={max_lifetime}s liveness_check=on)..."
        )
        _pool = AsyncConnectionPool(
            conninfo=db_url,
            min_size=min_size,
            max_size=max_size,
            open=False,
            kwargs={"autocommit": True},
            name="AgentMeshAsyncCheckpointerPool",
            check=AsyncConnectionPool.check_connection,
            max_idle=max_idle,
            max_lifetime=max_lifetime,
            timeout=pool_timeout,
            reconnect_timeout=reconnect_timeout,
        )
        await _pool.open()
        _saver = AsyncPostgresSaver(_pool)
        await _saver.setup()
        from Apps.AgentMesh.Utilities.Memory.artifact_checkpointer import (
            ArtifactAwareCheckpointSaver,
        )

        # The wrapper always protects sensitive checkpoint leaves. Binary
        # artifact externalization remains independently configurable.
        _saver = ArtifactAwareCheckpointSaver(
            _saver,
            externalization_enabled=settings.CHECKPOINT_ARTIFACTS_ENABLED,
        )
        if settings.CHECKPOINT_ARTIFACTS_ENABLED:
            logger.info(
                "Checkpoint artifact externalization enabled "
                f"(threshold={settings.CHECKPOINT_ARTIFACT_THRESHOLD_BYTES} bytes, "
                f"max={settings.CHECKPOINT_ARTIFACT_MAX_BYTES} bytes)."
            )
        logger.info("AsyncPostgresSaver setup complete and ready.")
        return _saver
    except Exception as e:
        logger.error(f"Failed to initialize AsyncPostgresSaver: {e}. Falling back to MemorySaver.", exc_info=True)
        _saver = MemorySaver()
        return _saver


async def close_checkpointer() -> None:
    """
    Closes the AsyncPostgresSaver connection pool on shutdown lifespan.
    """
    global _pool, _saver
    if _pool is not None:
        try:
            await _pool.close()
            logger.info("AsyncPostgresSaver connection pool closed successfully.")
        except Exception as e:
            logger.error(f"Error closing AsyncPostgresSaver pool: {e}")
        finally:
            _pool = None
            _saver = None


def get_checkpointer(session_id: str = "default_session") -> Any:
    """
    Returns the global checkpointer singleton.
    """
    global _saver
    if _saver is not None:
        return _saver
    # Fallback to MemorySaver if init_checkpointer was not called or failed
    return MemorySaver()

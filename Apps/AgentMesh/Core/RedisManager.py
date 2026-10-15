"""
RedisManager — Concurrency guard for AgentMesh workflow execution.

Single responsibility: track which workflow IDs are currently executing
across all Gunicorn workers, enabling atomic 409 conflict checks for
delete, save, and compile operations.

Key design decisions:
  - One simple string key per workflow: agentmesh:active:{workflow_id}
  - Written with `await` directly in WorkflowService (not callbacks) — zero race conditions
  - TTL of 3600s as a safety net in case the server crashes mid-run
  - Gracefully degrades: if Redis is down, guards are skipped (no crash)
"""

import logging
from typing import Optional

import redis.asyncio as aioredis
from Apps.AgentMesh.Config import settings

logger = logging.getLogger(__name__)

_KEY_PREFIX = "agentmesh:active:"
_THREAD_STATE_PREFIX = "agentmesh:thread_state:"  # Tracks persisted interrupts for multi-worker resume
_TTL = 3600  # Safety-net expiry (1 hour) — cleaned up by finally block normally


class RedisManager:
    def __init__(self):
        self._client: Optional[aioredis.Redis] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Open the Redis connection pool. Called once at app startup."""
        try:
            kwargs: dict = {
                "decode_responses": True,
                "socket_connect_timeout": 5,
                "socket_timeout": 5,
                "health_check_interval": 30,
            }
            if settings.REDIS_PASSWORD:
                kwargs["password"] = settings.REDIS_PASSWORD

            self._client = aioredis.from_url(settings.REDIS_URL, **kwargs)
            await self._client.ping()
            logger.info(f"[Redis] Connected: {settings.REDIS_URL}")
        except Exception as exc:
            logger.error(f"[Redis] Connection failed — concurrency guards disabled: {exc}")
            self._client = None

    async def close(self) -> None:
        """Close the connection pool. Called at app shutdown."""
        if self._client:
            await self._client.aclose()
            logger.info("[Redis] Disconnected.")

    @property
    def available(self) -> bool:
        return self._client is not None

    @property
    def client(self) -> Optional[aioredis.Redis]:
        """Expose the Redis client connection for general sessional/tool caching."""
        return self._client

    # ── Guard helpers ─────────────────────────────────────────────────────────

    async def set_active(self, workflow_id: str) -> None:
        """
        Increment the active run counter for a workflow.
        Supports concurrent child threads on the same workflow_id.
        """
        if not self._client:
            return
        try:
            key = f"{_KEY_PREFIX}{workflow_id}"
            count = await self._client.incr(key)
            await self._client.expire(key, _TTL)  # Reset TTL on each increment
            logger.debug(f"[Redis] Marked active: {workflow_id} (count={count})")
        except Exception as exc:
            logger.warning(f"[Redis] set_active failed for {workflow_id}: {exc}")

    async def clear_active(self, workflow_id: str) -> None:
        """
        Decrement the active run counter. Only fully clears when all runs complete.
        """
        if not self._client:
            return
        try:
            key = f"{_KEY_PREFIX}{workflow_id}"
            count = await self._client.decr(key)
            if count <= 0:
                await self._client.delete(key)
                logger.debug(f"[Redis] Cleared active: {workflow_id} (all runs complete)")
            else:
                logger.debug(f"[Redis] Decremented active: {workflow_id} (remaining={count})")
        except Exception as exc:
            logger.warning(f"[Redis] clear_active failed for {workflow_id}: {exc}")

    async def force_clear_active(self, workflow_id: str) -> None:
        """
        Unconditionally delete the active key, ignoring the counter.
        Used when a workflow is DEFINITIVELY done (e.g. after HITL resume completes).
        The HITL flow has a counter leak: initial execute sets count=1 but never
        decrements (paused), resume increments to 2, completion decrements to 1 —
        key is never deleted. This method bypasses the counter entirely.
        """
        if not self._client:
            return
        try:
            key = f"{_KEY_PREFIX}{workflow_id}"
            await self._client.delete(key)
            logger.debug(f"[Redis] Force-cleared active: {workflow_id}")
        except Exception as exc:
            logger.warning(f"[Redis] force_clear_active failed for {workflow_id}: {exc}")

    async def is_active(self, workflow_id: str) -> bool:
        """
        Return True if this workflow is currently executing on ANY worker.
        Used by the 409 concurrency guards in the workflow router (delete/save/compile).
        Multi-worker safe: Redis is shared across all Gunicorn workers.
        """
        if not self._client:
            return False
        try:
            return bool(await self._client.exists(f"{_KEY_PREFIX}{workflow_id}"))
        except Exception as exc:
            logger.warning(f"[Redis] is_active check failed for {workflow_id}: {exc}")
            return False

    # ── Thread State Relay (multi-worker HITL resume) ──────────────────────────

    async def mark_thread_state_persisted(self, thread_id: str, ttl: int = 86400) -> bool:
        """
        Set a Redis flag indicating this thread's HITL state was persisted to Postgres.
        Used by other workers to know they should load from Postgres fallback.
        TTL: 24h default (thread should be resumed or expired by now).
        Returns True if the flag was successfully set.
        """
        if not self._client:
            return False
        try:
            await self._client.setex(f"{_THREAD_STATE_PREFIX}{thread_id}", ttl, "1")
            return True
        except Exception as exc:
            logger.warning(f"[Redis] mark_thread_state_persisted failed for {thread_id}: {exc}")
            return False

    async def is_thread_state_persisted(self, thread_id: str) -> bool:
        """
        Check if this thread's state was persisted to Postgres on another worker.
        """
        if not self._client:
            return False
        try:
            return bool(await self._client.exists(f"{_THREAD_STATE_PREFIX}{thread_id}"))
        except Exception as exc:
            logger.warning(f"[Redis] is_thread_state_persisted failed for {thread_id}: {exc}")
            return False

    async def clear_thread_state_persisted(self, thread_id: str) -> None:
        """Remove the persisted flag when the thread is resumed successfully."""
        if not self._client:
            return
        try:
            await self._client.delete(f"{_THREAD_STATE_PREFIX}{thread_id}")
        except Exception as exc:
            logger.warning(f"[Redis] clear_thread_state_persisted failed for {thread_id}: {exc}")


# ── Global singleton ──────────────────────────────────────────────────────────
redis_manager = RedisManager()

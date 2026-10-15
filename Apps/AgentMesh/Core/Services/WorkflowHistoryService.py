"""Compact, TTL-controlled history for terminal workflow runs."""

from __future__ import annotations

import base64
import binascii
import logging
from datetime import datetime, timedelta
from typing import Any, Optional

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from Apps.AgentMesh.Config import settings
from Apps.AgentMesh.Utilities.Security import redact_sensitive_data, sanitize_for_logging
from Database.Models import ExecutionLogModel, WorkflowRunHistoryModel

logger = logging.getLogger(__name__)


class WorkflowHistoryService:
    """Stores UI history without retaining resumable LangGraph checkpoints."""

    MAX_TEXT_CHARS = 50_000
    BINARY_PLACEHOLDER = "[Binary content removed after workflow completion]"

    @classmethod
    def _looks_like_large_base64(cls, value: str) -> bool:
        compact = "".join(value.split())
        if len(compact) < 4096 or len(compact) % 4:
            return False
        try:
            base64.b64decode(compact, validate=True)
            return True
        except (binascii.Error, ValueError):
            return False

    @classmethod
    def compact(cls, value: Any, *, depth: int = 0) -> Any:
        """Remove binary payloads and cap unusually large text before DB storage."""
        if depth > 40:
            return "[History depth limit reached]"
        if isinstance(value, (bytes, bytearray, memoryview)):
            return cls.BINARY_PLACEHOLDER
        if isinstance(value, str):
            if value.startswith("data:") and ";base64," in value[:256]:
                return cls.BINARY_PLACEHOLDER
            if cls._looks_like_large_base64(value):
                return cls.BINARY_PLACEHOLDER
            if len(value) > cls.MAX_TEXT_CHARS:
                return value[: cls.MAX_TEXT_CHARS] + "\n[History text truncated]"
            return value
        if isinstance(value, dict):
            return {
                str(key): cls.compact(item, depth=depth + 1)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [cls.compact(item, depth=depth + 1) for item in value]
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        return str(value)

    @classmethod
    async def save_completed_history(
        cls,
        db: AsyncSession,
        *,
        thread_id: str,
        workflow_id: str,
        status: str,
        output: Optional[str],
        messages: Any,
        context: Any,
        active_agent: Optional[str],
    ) -> WorkflowRunHistoryModel:
        now = datetime.utcnow()
        expires_at = now + timedelta(seconds=settings.HISTORY_TTL_SECONDS)
        history = await db.get(WorkflowRunHistoryModel, thread_id)
        def safe_history(value: Any) -> Any:
            return sanitize_for_logging(redact_sensitive_data(cls.compact(value)))

        values = {
            "workflow_id": workflow_id,
            "status": status,
            "output": safe_history(output),
            "messages": safe_history(messages),
            "context": safe_history(context),
            "active_agent": active_agent,
            "completed_at": now,
            "expires_at": expires_at,
        }
        if history is None:
            history = WorkflowRunHistoryModel(thread_id=thread_id, **values)
            db.add(history)
        else:
            for key, value in values.items():
                setattr(history, key, value)
        await db.commit()
        return history

    @staticmethod
    async def get_history(
        db: AsyncSession,
        *,
        thread_id: str,
        workflow_id: Optional[str] = None,
    ) -> Optional[WorkflowRunHistoryModel]:
        statement = select(WorkflowRunHistoryModel).where(
            WorkflowRunHistoryModel.thread_id == thread_id,
            WorkflowRunHistoryModel.expires_at > datetime.utcnow(),
        )
        if workflow_id is not None:
            statement = statement.where(
                WorkflowRunHistoryModel.workflow_id == workflow_id
            )
        result = await db.execute(statement)
        return result.scalar_one_or_none()

    @staticmethod
    async def delete_expired(db: AsyncSession) -> int:
        expired_result = await db.execute(
            select(WorkflowRunHistoryModel.thread_id).where(
                WorkflowRunHistoryModel.expires_at <= datetime.utcnow()
            )
        )
        expired_thread_ids = list(expired_result.scalars().all())
        if not expired_thread_ids:
            return 0
        # execution_logs power the History list, so they must expire with the
        # compact history row or completed sessions would remain as dead links.
        await db.execute(
            delete(ExecutionLogModel).where(
                ExecutionLogModel.thread_id.in_(expired_thread_ids)
            )
        )
        result = await db.execute(
            delete(WorkflowRunHistoryModel).where(
                WorkflowRunHistoryModel.thread_id.in_(expired_thread_ids)
            )
        )
        await db.commit()
        return result.rowcount or 0

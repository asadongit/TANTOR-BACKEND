import base64
import unittest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from Apps.AgentMesh.Core.Services.WorkflowHistoryService import WorkflowHistoryService
from Apps.AgentMesh.Core.Services.WorkflowService import WorkflowService
from Database.Models import ExecutionLogModel, WorkflowRunHistoryModel


class WorkflowHistoryServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with self.engine.begin() as connection:
            await connection.run_sync(ExecutionLogModel.__table__.create)
            await connection.run_sync(WorkflowRunHistoryModel.__table__.create)
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)

    async def asyncTearDown(self):
        await self.engine.dispose()

    async def test_history_omits_binary_but_preserves_text(self):
        encoded = base64.b64encode(b"image-content" * 500).decode("ascii")
        async with self.sessions() as db:
            saved = await WorkflowHistoryService.save_completed_history(
                db,
                thread_id="history-thread-1",
                workflow_id="workflow-1",
                status="success",
                output="OCR completed",
                messages=[
                    {"role": "user", "content": f"data:image/png;base64,{encoded}"},
                    {"role": "assistant", "content": "Extracted text"},
                ],
                context={"raw_file": b"binary"},
                active_agent="ocr-node",
            )

        self.assertEqual(saved.output, "OCR completed")
        self.assertEqual(
            saved.messages[0]["content"],
            WorkflowHistoryService.BINARY_PLACEHOLDER,
        )
        self.assertEqual(saved.messages[1]["content"], "Extracted text")
        self.assertEqual(
            saved.context["raw_file"],
            WorkflowHistoryService.BINARY_PLACEHOLDER,
        )

    async def test_expired_history_is_deleted(self):
        async with self.sessions() as db:
            history = WorkflowRunHistoryModel(
                thread_id="expired-thread",
                workflow_id="workflow-1",
                status="success",
                messages=[],
                context={},
                completed_at=datetime.utcnow() - timedelta(minutes=2),
                expires_at=datetime.utcnow() - timedelta(seconds=1),
            )
            db.add(history)
            db.add(
                ExecutionLogModel(
                    id="expired-log",
                    thread_id="expired-thread",
                    workflow_id="workflow-1",
                    node_id="node-1",
                )
            )
            await db.commit()
            deleted = await WorkflowHistoryService.delete_expired(db)
            remaining = await db.get(WorkflowRunHistoryModel, "expired-thread")
            remaining_log = await db.get(ExecutionLogModel, "expired-log")

        self.assertEqual(deleted, 1)
        self.assertIsNone(remaining)
        self.assertIsNone(remaining_log)

    async def test_checkpoint_is_retained_when_history_save_fails(self):
        cleanup = AsyncMock()
        with patch.object(
            WorkflowHistoryService,
            "save_completed_history",
            new=AsyncMock(side_effect=RuntimeError("database unavailable")),
        ), patch.object(
            WorkflowService,
            "_cleanup_completed_thread",
            cleanup,
        ):
            result = await WorkflowService._finalize_completed_thread(
                thread_id="safe-thread",
                workflow_id="workflow-1",
                status="success",
                output="done",
                messages=[],
                context={},
                active_agent=None,
            )

        self.assertFalse(result)
        cleanup.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()

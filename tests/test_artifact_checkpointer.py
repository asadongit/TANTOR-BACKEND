import base64
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import TypedDict
from unittest.mock import patch

from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointTuple
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langchain_core.messages import HumanMessage
from starlette.datastructures import Headers, UploadFile
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from Apps.AgentMesh.Config import settings
from Apps.AgentMesh.Core.Services.ArtifactService import ArtifactService
from Apps.AgentMesh.Utilities.Memory.artifact_checkpointer import (
    ARTIFACT_MARKER,
    ArtifactAwareCheckpointSaver,
)
from Apps.AgentMesh.Utilities.Security import is_encrypted
from Database.Models import ArtifactModel, CheckpointArtifactReferenceModel


class FakeSaver(BaseCheckpointSaver):
    def __init__(self):
        super().__init__()
        self.saved_checkpoint = None
        self.saved_metadata = None
        self.saved_writes = None
        self.loaded_tuple = None
        self.deleted_threads = []
        self.copied_threads = []

    async def aput(self, config, checkpoint, metadata, new_versions):
        self.saved_checkpoint = checkpoint
        self.saved_metadata = metadata
        return config

    async def aput_writes(self, config, writes, task_id, task_path=""):
        self.saved_writes = list(writes)

    async def aget_tuple(self, config):
        return self.loaded_tuple

    async def alist(self, config, *, filter=None, before=None, limit=None):
        if self.loaded_tuple is not None:
            yield self.loaded_tuple

    async def adelete_thread(self, thread_id):
        self.deleted_threads.append(thread_id)

    async def acopy_thread(self, source_thread_id, target_thread_id):
        self.copied_threads.append((source_thread_id, target_thread_id))


class FakeRepository:
    def __init__(self):
        self.items = {}
        self.save_calls = 0
        self.released_threads = []
        self.copied_threads = []
        self.reference_calls = []

    async def save(self, data, **kwargs):
        self.save_calls += 1
        artifact_id = f"artifact-{hash(data)}"
        self.items[artifact_id] = (
            bytes(data),
            SimpleNamespace(content_type=kwargs.get("content_type")),
            Path(f"/artifacts/{artifact_id}/{kwargs['filename']}"),
        )
        return {"artifact_id": artifact_id}

    async def load(self, artifact_id):
        return self.items[artifact_id]

    async def release_thread(self, thread_id):
        self.released_threads.append(thread_id)
        return {"released": 1, "deleted": 1, "preserved": 0, "failed": 0}

    async def copy_thread(self, source_thread_id, target_thread_id):
        self.copied_threads.append((source_thread_id, target_thread_id))

    async def reference(self, artifact_id, **kwargs):
        self.reference_calls.append((artifact_id, kwargs["thread_id"]))

    async def reference_if_exists(self, artifact_id, **kwargs):
        await self.reference(artifact_id, **kwargs)
        return artifact_id in self.items


class ArtifactAwareCheckpointTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.inner = FakeSaver()
        self.repository = FakeRepository()
        self.saver = ArtifactAwareCheckpointSaver(
            self.inner,
            repository=self.repository,
            threshold_bytes=16,
            max_bytes=1024 * 1024,
        )
        self.config = {
            "configurable": {
                "thread_id": "thread-1",
                "workflow_id": "workflow-1",
            }
        }

    async def test_large_bytes_are_replaced_before_checkpoint_save(self):
        original = b"large-binary-value" * 4
        checkpoint = {"channel_values": {"image": original}}

        await self.saver.aput(self.config, checkpoint, {}, {})

        saved = self.inner.saved_checkpoint["channel_values"]["image"]
        self.assertEqual(saved[ARTIFACT_MARKER], 1)
        self.assertEqual(saved["kind"], "bytes")
        self.assertEqual(checkpoint["channel_values"]["image"], original)

    async def test_sensitive_checkpoint_values_are_encrypted_and_restored(self):
        checkpoint = {
            "channel_values": {
                "context": {
                    "workflow_variables": {
                        "oci_private_key": "private-key-value",
                        "region": "safe-region",
                    }
                }
            }
        }
        await self.saver.aput(self.config, checkpoint, {}, {})
        stored_variables = self.inner.saved_checkpoint["channel_values"]["context"]["workflow_variables"]
        self.assertTrue(is_encrypted(stored_variables["oci_private_key"]))
        self.assertEqual(stored_variables["region"], "safe-region")

        self.inner.loaded_tuple = CheckpointTuple(
            config=self.config,
            checkpoint=self.inner.saved_checkpoint,
            metadata={},
            parent_config=None,
            pending_writes=[],
        )
        restored = await self.saver.aget_tuple(self.config)
        self.assertEqual(
            restored.checkpoint["channel_values"]["context"]["workflow_variables"],
            checkpoint["channel_values"]["context"]["workflow_variables"],
        )

    async def test_data_uri_is_externalized_and_restored(self):
        payload = b"\x89PNG\r\n\x1a\n" + (b"image-data" * 8)
        data_uri = "data:image/png;base64," + base64.b64encode(payload).decode()
        checkpoint = {"channel_values": {"image": data_uri}}
        await self.saver.aput(self.config, checkpoint, {}, {})

        saved_checkpoint = self.inner.saved_checkpoint
        self.inner.loaded_tuple = CheckpointTuple(
            config=self.config,
            checkpoint=saved_checkpoint,
            metadata={},
            parent_config=None,
            pending_writes=[],
        )
        loaded = await self.saver.aget_tuple(self.config)

        self.assertEqual(loaded.checkpoint["channel_values"]["image"], data_uri)

    async def test_large_base64_under_named_key_is_externalized(self):
        payload = b"plain-binary-content" * 4
        encoded = base64.b64encode(payload).decode()
        await self.saver.aput(
            self.config,
            {"channel_values": {"document_image_base64": encoded}},
            {},
            {},
        )
        saved = self.inner.saved_checkpoint["channel_values"]["document_image_base64"]
        self.assertEqual(saved["kind"], "base64")

    async def test_large_normal_text_is_not_externalized(self):
        text = "This is ordinary workflow text. " * 20
        await self.saver.aput(
            self.config,
            {"channel_values": {"summary": text}},
            {},
            {},
        )
        self.assertEqual(self.inner.saved_checkpoint["channel_values"]["summary"], text)
        self.assertEqual(self.repository.save_calls, 0)

    async def test_known_binary_over_maximum_fails_instead_of_reaching_database(self):
        encoded = base64.b64encode(b"x" * 64).decode()
        saver = ArtifactAwareCheckpointSaver(
            self.inner,
            repository=self.repository,
            threshold_bytes=16,
            max_bytes=32,
        )
        with self.assertRaisesRegex(ValueError, "exceeds"):
            await saver.aput(
                self.config,
                {"channel_values": {"document_base64": encoded}},
                {},
                {},
            )
        self.assertIsNone(self.inner.saved_checkpoint)

    async def test_pending_writes_are_externalized(self):
        value = b"pending-write-binary" * 4
        await self.saver.aput_writes(
            self.config,
            [("image_channel", value)],
            "task-1",
        )
        saved = self.inner.saved_writes[0][1]
        self.assertEqual(saved[ARTIFACT_MARKER], 1)
        self.assertEqual(saved["kind"], "bytes")

    async def test_base64_inside_langchain_message_is_externalized(self):
        payload = b"\x89PNG\r\n\x1a\n" + (b"message-image" * 8)
        data_uri = "data:image/png;base64," + base64.b64encode(payload).decode()
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Inspect this image"},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ]
        )
        await self.saver.aput(
            self.config,
            {"channel_values": {"messages": [message]}},
            {},
            {},
        )
        saved_message = self.inner.saved_checkpoint["channel_values"]["messages"][0]
        saved_url = saved_message.content[1]["image_url"]["url"]
        self.assertEqual(saved_url[ARTIFACT_MARKER], 1)
        self.assertEqual(saved_url["kind"], "data_uri")
        self.inner.loaded_tuple = CheckpointTuple(
            config=self.config,
            checkpoint=self.inner.saved_checkpoint,
            metadata={},
            parent_config=None,
            pending_writes=[],
        )
        loaded = await self.saver.aget_tuple(self.config)
        restored_url = loaded.checkpoint["channel_values"]["messages"][0].content[1][
            "image_url"
        ]["url"]
        self.assertEqual(restored_url, data_uri)

    async def test_existing_artifact_reference_is_left_unchanged(self):
        reference = {"artifact_id": "uploaded-artifact"}
        await self.saver.aput(
            self.config,
            {"channel_values": {"file": reference}},
            {},
            {},
        )
        self.assertEqual(self.inner.saved_checkpoint["channel_values"]["file"], reference)
        self.assertEqual(self.repository.save_calls, 0)

    async def test_existing_checkpoint_reference_registers_new_thread_owner(self):
        reference = {
            ARTIFACT_MARKER: 1,
            "artifact_id": "shared-checkpoint-artifact",
            "kind": "bytes",
        }
        await self.saver.aput(
            self.config,
            {"channel_values": {"file": reference}},
            {},
            {},
        )

        self.assertEqual(
            self.repository.reference_calls,
            [("shared-checkpoint-artifact", "thread-1")],
        )

    async def test_real_langgraph_save_and_load_are_transparent(self):
        class TestState(TypedDict):
            payload: bytes
            completed: bool

        repository = FakeRepository()
        inner = MemorySaver()
        saver = ArtifactAwareCheckpointSaver(
            inner,
            repository=repository,
            threshold_bytes=16,
            max_bytes=1024 * 1024,
        )
        graph = StateGraph(TestState)

        async def create_payload(_state):
            return {
                "payload": b"workflow-generated-binary" * 4,
                "completed": True,
            }

        graph.add_node("create_payload", create_payload)
        graph.add_edge(START, "create_payload")
        graph.add_edge("create_payload", END)
        app = graph.compile(checkpointer=saver)

        result = await app.ainvoke(
            {"payload": b"", "completed": False},
            config=self.config,
        )
        raw_checkpoint = await inner.aget_tuple(self.config)
        restored_checkpoint = await saver.aget_tuple(self.config)

        self.assertEqual(result["payload"], b"workflow-generated-binary" * 4)
        self.assertEqual(
            raw_checkpoint.checkpoint["channel_values"]["payload"][ARTIFACT_MARKER],
            1,
        )
        self.assertEqual(
            restored_checkpoint.checkpoint["channel_values"]["payload"],
            b"workflow-generated-binary" * 4,
        )

    async def test_thread_delete_releases_artifacts_after_checkpoint_delete(self):
        await self.saver.adelete_thread("thread-1")

        self.assertEqual(self.inner.deleted_threads, ["thread-1"])
        self.assertEqual(self.repository.released_threads, ["thread-1"])

    async def test_thread_copy_preserves_artifact_references(self):
        await self.saver.acopy_thread("thread-1", "thread-2")

        self.assertEqual(
            self.repository.copied_threads,
            [("thread-1", "thread-2")],
        )


class CheckpointArtifactStorageTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with self.engine.begin() as connection:
            await connection.run_sync(ArtifactModel.__table__.create)
            await connection.run_sync(CheckpointArtifactReferenceModel.__table__.create)
        self.session_factory = async_sessionmaker(
            self.engine,
            expire_on_commit=False,
        )

    async def asyncTearDown(self):
        await self.engine.dispose()
        self.temporary_directory.cleanup()

    async def test_storage_is_deduplicated_across_threads_and_reference_safe(self):
        payload = b"same-checkpoint-binary" * 16
        with patch.object(
            settings,
            "ARTIFACT_MOUNT",
            self.temporary_directory.name,
        ):
            async with self.session_factory() as session:
                first = await ArtifactService.save_checkpoint_bytes(
                    payload,
                    "thread-1",
                    session,
                    workflow_id="workflow-1",
                    filename="image.png",
                    content_type="image/png",
                )
            async with self.session_factory() as session:
                second = await ArtifactService.save_checkpoint_bytes(
                    payload,
                    "thread-2",
                    session,
                    workflow_id="workflow-1",
                    filename="image.png",
                    content_type="image/png",
                )
                stored_data, artifact, path = await ArtifactService.read_bytes(
                    first["artifact_id"],
                    session,
                )
                count = await session.scalar(select(func.count(ArtifactModel.id)))
                reference_count = await session.scalar(
                    select(func.count()).select_from(CheckpointArtifactReferenceModel)
                )

            async with self.session_factory() as session:
                first_cleanup = await ArtifactService.release_checkpoint_artifacts(
                    "thread-1", session
                )
                still_exists = path.is_file()

            async with self.session_factory() as session:
                second_cleanup = await ArtifactService.release_checkpoint_artifacts(
                    "thread-2", session
                )
                artifact_count_after = await session.scalar(
                    select(func.count(ArtifactModel.id))
                )

        self.assertEqual(first["artifact_id"], second["artifact_id"])
        self.assertEqual(count, 1)
        self.assertEqual(reference_count, 2)
        self.assertEqual(stored_data, payload)
        self.assertEqual(artifact.content_type, "image/png")
        self.assertTrue(still_exists)
        self.assertEqual(first_cleanup["preserved"], 1)
        self.assertEqual(first_cleanup["deleted"], 0)
        self.assertEqual(second_cleanup["deleted"], 1)
        self.assertEqual(artifact_count_after, 0)
        self.assertFalse(path.exists())

    async def test_failed_file_delete_keeps_metadata_and_reference_for_retry(self):
        with patch.object(settings, "ARTIFACT_MOUNT", self.temporary_directory.name):
            async with self.session_factory() as session:
                stored = await ArtifactService.save_checkpoint_bytes(
                    b"retry-safe-checkpoint" * 16,
                    "thread-retry",
                    session,
                )

            with patch.object(
                ArtifactService,
                "_delete_artifact_file",
                side_effect=OSError("NFS temporarily unavailable"),
            ):
                async with self.session_factory() as session:
                    cleanup = await ArtifactService.release_checkpoint_artifacts(
                        "thread-retry", session
                    )

            async with self.session_factory() as session:
                artifact_still_exists = await session.get(
                    ArtifactModel, stored["artifact_id"]
                )
                reference_still_exists = await session.get(
                    CheckpointArtifactReferenceModel,
                    ("thread-retry", stored["artifact_id"]),
                )

        self.assertEqual(cleanup["failed"], 1)
        self.assertIsNotNone(artifact_still_exists)
        self.assertIsNotNone(reference_still_exists)

    async def test_uploads_with_same_content_share_storage_and_are_reference_safe(self):
        payload = b"\x89PNG\r\n\x1a\n" + (b"shared-upload" * 32)

        def upload(filename):
            return UploadFile(
                io.BytesIO(payload),
                filename=filename,
                headers=Headers({"content-type": "image/png"}),
            )

        with patch.object(settings, "ARTIFACT_MOUNT", self.temporary_directory.name):
            async with self.session_factory() as session:
                first = await ArtifactService.save_upload(
                    upload("first-name.png"), "upload-thread-1", session
                )
            async with self.session_factory() as session:
                second = await ArtifactService.save_upload(
                    upload("another-name.png"), "upload-thread-2", session
                )
                path = await ArtifactService.resolve_path(first["artifact_id"], session)
                artifact_count = await session.scalar(
                    select(func.count(ArtifactModel.id))
                )
            async with self.session_factory() as session:
                first_cleanup = await ArtifactService.release_checkpoint_artifacts(
                    "upload-thread-1", session
                )
            preserved = path.exists()
            async with self.session_factory() as session:
                second_cleanup = await ArtifactService.release_checkpoint_artifacts(
                    "upload-thread-2", session
                )

        self.assertEqual(first["artifact_id"], second["artifact_id"])
        self.assertEqual(artifact_count, 1)
        self.assertEqual(path.suffix, ".png")
        self.assertEqual(first_cleanup["preserved"], 1)
        self.assertTrue(preserved)
        self.assertEqual(second_cleanup["deleted"], 1)
        self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()

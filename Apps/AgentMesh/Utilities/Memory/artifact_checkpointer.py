"""Artifact-aware wrapper for LangGraph checkpoint savers.

Large binary values are externalized to the configured artifact filesystem
(local storage or NFS), while PostgreSQL receives only a compact reference.
Values are restored when a checkpoint is loaded, so existing nodes do not all
need storage-specific code.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import mimetypes
import re
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import is_dataclass, replace
from pathlib import Path
from typing import Any, Optional

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    CheckpointTuple,
)

from Apps.AgentMesh.Config import settings
from Apps.AgentMesh.Core.Services.ArtifactService import ArtifactService
from Apps.AgentMesh.Utilities.Security import (
    decrypt_string,
    encrypt_string,
    is_encrypted,
    is_sensitive_container,
    is_sensitive_field,
)
from Database.Database import AsyncSessionLocal

logger = logging.getLogger(__name__)

ARTIFACT_MARKER = "__agentmesh_checkpoint_artifact__"
ARTIFACT_REFERENCE_VERSION = 1
_DATA_URI_PATTERN = re.compile(
    r"^data:(?P<mime>[^;,]+)?(?:;[^,]*)?;base64,(?P<data>.*)$",
    re.IGNORECASE | re.DOTALL,
)
_BASE64_KEY_PATTERN = re.compile(
    r"(?:base64|_b64|b64_|image_data|file_data|blob|binary|bytes)",
    re.IGNORECASE,
)
_PATH_KEY_PATTERN = re.compile(
    r"(?:file_path|image_path|document_path|local_path|temp_path|path)$",
    re.IGNORECASE,
)
_BINARY_MAGIC = (
    b"\x89PNG\r\n\x1a\n",
    b"\xff\xd8\xff",
    b"GIF87a",
    b"GIF89a",
    b"%PDF-",
    b"PK\x03\x04",
    b"RIFF",
    b"II*\x00",
    b"MM\x00*",
)


class DatabaseArtifactRepository:
    """Database-backed adapter used by the checkpoint wrapper."""

    async def save(
        self,
        data: bytes,
        *,
        thread_id: str,
        workflow_id: Optional[str],
        filename: str,
        content_type: Optional[str],
    ) -> dict:
        async with AsyncSessionLocal() as db:
            return await ArtifactService.save_checkpoint_bytes(
                data,
                thread_id,
                db,
                workflow_id=workflow_id,
                filename=filename,
                content_type=content_type,
            )

    async def load(self, artifact_id: str) -> tuple[bytes, Any, Path]:
        async with AsyncSessionLocal() as db:
            return await ArtifactService.read_bytes(artifact_id, db)

    async def reference(
        self,
        artifact_id: str,
        *,
        thread_id: str,
        workflow_id: Optional[str],
    ) -> None:
        async with AsyncSessionLocal() as db:
            await ArtifactService.register_checkpoint_reference(
                artifact_id,
                thread_id,
                db,
                workflow_id=workflow_id,
            )

    async def reference_if_exists(
        self,
        artifact_id: str,
        *,
        thread_id: str,
        workflow_id: Optional[str],
    ) -> bool:
        try:
            await self.reference(
                artifact_id,
                thread_id=thread_id,
                workflow_id=workflow_id,
            )
            return True
        except ValueError:
            # Some nodes use an "artifact_id" key for external systems. Those
            # are not owned by AgentMesh artifact storage.
            return False

    async def release_thread(self, thread_id: str) -> dict:
        async with AsyncSessionLocal() as db:
            return await ArtifactService.release_checkpoint_artifacts(thread_id, db)

    async def copy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        async with AsyncSessionLocal() as db:
            await ArtifactService.copy_checkpoint_references(
                source_thread_id,
                target_thread_id,
                db,
            )


class ArtifactAwareCheckpointSaver(BaseCheckpointSaver):
    """Wrap a saver and externalize large checkpoint binary values."""

    def __init__(
        self,
        saver: BaseCheckpointSaver,
        *,
        repository: Any | None = None,
        threshold_bytes: int | None = None,
        max_bytes: int | None = None,
        externalization_enabled: bool = True,
    ) -> None:
        super().__init__(serde=saver.serde)
        self.saver = saver
        self.repository = repository or DatabaseArtifactRepository()
        self.externalization_enabled = externalization_enabled
        self.threshold_bytes = (
            threshold_bytes
            if threshold_bytes is not None
            else settings.CHECKPOINT_ARTIFACT_THRESHOLD_BYTES
        )
        self.max_bytes = (
            max_bytes
            if max_bytes is not None
            else settings.CHECKPOINT_ARTIFACT_MAX_BYTES
        )

    @property
    def config_specs(self) -> list:
        return self.saver.config_specs

    def get_next_version(self, current: Any, channel: Any) -> Any:
        return self.saver.get_next_version(current, channel)

    @staticmethod
    def _context(config: Mapping[str, Any]) -> tuple[str, Optional[str]]:
        configurable = config.get("configurable", {})
        thread_id = str(configurable.get("thread_id") or "default_thread")
        workflow_id = configurable.get("workflow_id")
        return thread_id, str(workflow_id) if workflow_id is not None else None

    @staticmethod
    def _is_reference(value: Any) -> bool:
        return (
            isinstance(value, dict)
            and value.get(ARTIFACT_MARKER) == ARTIFACT_REFERENCE_VERSION
            and isinstance(value.get("artifact_id"), str)
        )

    @staticmethod
    def _content_type_filename(content_type: Optional[str]) -> str:
        extension = mimetypes.guess_extension(content_type or "") or ".bin"
        return f"checkpoint-artifact{extension}"

    def _decode_base64(
        self,
        value: str,
        *,
        key_hint: str,
    ) -> tuple[bytes, str, Optional[str], str] | None:
        data_uri = _DATA_URI_PATTERN.match(value)
        content_type: Optional[str] = None
        kind = "base64"
        encoded = value
        if data_uri:
            content_type = data_uri.group("mime") or "application/octet-stream"
            encoded = data_uri.group("data")
            kind = "data_uri"
        elif not _BASE64_KEY_PATTERN.search(key_hint):
            # Unknown large strings are treated as normal text unless their
            # decoded bytes have a recognized binary file signature.
            kind = "base64_candidate"

        compact = "".join(encoded.split())
        if len(compact) < 4 or len(compact) % 4:
            return None
        estimated_size = (len(compact) * 3) // 4
        recognized_base64 = bool(data_uri or _BASE64_KEY_PATTERN.search(key_hint))
        if estimated_size > self.max_bytes and recognized_base64:
            raise ValueError(
                "Checkpoint binary value exceeds "
                f"AGENTMESH_CHECKPOINT_ARTIFACT_MAX_BYTES ({self.max_bytes})"
            )
        if estimated_size < self.threshold_bytes or estimated_size > self.max_bytes:
            return None
        try:
            decoded = base64.b64decode(compact, validate=True)
        except (binascii.Error, ValueError):
            return None
        if len(decoded) < self.threshold_bytes or len(decoded) > self.max_bytes:
            return None
        if kind == "base64_candidate":
            if not decoded.startswith(_BINARY_MAGIC):
                return None
            kind = "base64"
        filename = self._content_type_filename(content_type)
        return decoded, kind, content_type, filename

    def _approved_file(self, value: str, key_hint: str) -> Path | None:
        if not _PATH_KEY_PATTERN.search(key_hint) or len(value) > 4096:
            return None
        try:
            path = Path(value).resolve()
        except (OSError, RuntimeError, ValueError):
            return None
        roots = {
            Path(mount).resolve()
            for mount in (
                settings.TEMP_MOUNT,
                settings.UPLOAD_MOUNT,
                settings.TEMP_IMAGES_MOUNT,
                settings.AUDIO_MOUNT,
                settings.EXPORTS_MOUNT,
            )
            if mount
        }
        if not path.is_file():
            return None
        try:
            if not any(path.is_relative_to(root) for root in roots):
                return None
        except (OSError, ValueError):
            return None
        size = path.stat().st_size
        if size > self.max_bytes:
            raise ValueError(
                f"Checkpoint file '{path.name}' exceeds "
                f"AGENTMESH_CHECKPOINT_ARTIFACT_MAX_BYTES ({self.max_bytes})"
            )
        if size < self.threshold_bytes:
            return None
        return path

    async def _save_reference(
        self,
        data: bytes,
        *,
        kind: str,
        thread_id: str,
        workflow_id: Optional[str],
        filename: str,
        content_type: Optional[str],
    ) -> dict:
        metadata = await self.repository.save(
            data,
            thread_id=thread_id,
            workflow_id=workflow_id,
            filename=filename,
            content_type=content_type,
        )
        return {
            ARTIFACT_MARKER: ARTIFACT_REFERENCE_VERSION,
            "artifact_id": metadata["artifact_id"],
            "kind": kind,
        }

    async def _externalize(
        self,
        value: Any,
        *,
        thread_id: str,
        workflow_id: Optional[str],
        key_hint: str = "",
        force_sensitive: bool = False,
        depth: int = 0,
    ) -> Any:
        if depth > 40:
            return value
        if self._is_reference(value):
            if hasattr(self.repository, "reference"):
                await self.repository.reference(
                    value["artifact_id"],
                    thread_id=thread_id,
                    workflow_id=workflow_id,
                )
            return value

        if isinstance(value, bytes) and not self.externalization_enabled:
            return value
        if isinstance(value, bytes) and len(value) > self.max_bytes:
            raise ValueError(
                "Checkpoint bytes exceed "
                f"AGENTMESH_CHECKPOINT_ARTIFACT_MAX_BYTES ({self.max_bytes})"
            )
        if isinstance(value, bytes) and len(value) >= self.threshold_bytes:
            return await self._save_reference(
                value,
                kind="bytes",
                thread_id=thread_id,
                workflow_id=workflow_id,
                filename="checkpoint-artifact.bin",
                content_type="application/octet-stream",
            )
        if isinstance(value, bytearray) and not self.externalization_enabled:
            return value
        if isinstance(value, bytearray) and len(value) > self.max_bytes:
            raise ValueError(
                "Checkpoint bytearray exceeds "
                f"AGENTMESH_CHECKPOINT_ARTIFACT_MAX_BYTES ({self.max_bytes})"
            )
        if isinstance(value, bytearray) and len(value) >= self.threshold_bytes:
            return await self._save_reference(
                bytes(value),
                kind="bytearray",
                thread_id=thread_id,
                workflow_id=workflow_id,
                filename="checkpoint-artifact.bin",
                content_type="application/octet-stream",
            )
        if isinstance(value, str):
            if value and (force_sensitive or is_sensitive_field(key_hint)):
                return encrypt_string(value)
            if not self.externalization_enabled:
                return value
            decoded = self._decode_base64(value, key_hint=key_hint)
            if decoded is not None:
                data, kind, content_type, filename = decoded
                return await self._save_reference(
                    data,
                    kind=kind,
                    thread_id=thread_id,
                    workflow_id=workflow_id,
                    filename=filename,
                    content_type=content_type,
                )
            path = self._approved_file(value, key_hint)
            if path is not None:
                content_type, _ = mimetypes.guess_type(path.name)
                return await self._save_reference(
                    await asyncio.to_thread(path.read_bytes),
                    kind="file_path",
                    thread_id=thread_id,
                    workflow_id=workflow_id,
                    filename=path.name,
                    content_type=content_type,
                )
            return value
        if isinstance(value, dict):
            plain_artifact_id = value.get("artifact_id")
            if (
                isinstance(plain_artifact_id, str)
                and hasattr(self.repository, "reference_if_exists")
            ):
                await self.repository.reference_if_exists(
                    plain_artifact_id,
                    thread_id=thread_id,
                    workflow_id=workflow_id,
                )
            return {
                key: await self._externalize(
                    item,
                    thread_id=thread_id,
                    workflow_id=workflow_id,
                    key_hint=str(key),
                    force_sensitive=(
                        force_sensitive
                        or is_sensitive_field(key)
                        or is_sensitive_container(key)
                    ),
                    depth=depth + 1,
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [
                await self._externalize(
                    item,
                    thread_id=thread_id,
                    workflow_id=workflow_id,
                    key_hint=key_hint,
                    force_sensitive=force_sensitive,
                    depth=depth + 1,
                )
                for item in value
            ]
        if isinstance(value, tuple):
            converted = [
                await self._externalize(
                    item,
                    thread_id=thread_id,
                    workflow_id=workflow_id,
                    key_hint=key_hint,
                    force_sensitive=force_sensitive,
                    depth=depth + 1,
                )
                for item in value
            ]
            if hasattr(value, "_fields"):
                return type(value)(*converted)
            return tuple(converted)
        if is_dataclass(value) and not isinstance(value, type):
            updates = {
                field_name: await self._externalize(
                    getattr(value, field_name),
                    thread_id=thread_id,
                    workflow_id=workflow_id,
                    key_hint=field_name,
                    force_sensitive=(
                        force_sensitive
                        or is_sensitive_field(field_name)
                        or is_sensitive_container(field_name)
                    ),
                    depth=depth + 1,
                )
                for field_name in value.__dataclass_fields__
            }
            return replace(value, **updates)
        model_fields = getattr(value.__class__, "model_fields", None)
        if model_fields is not None and hasattr(value, "model_copy"):
            updates = {
                field_name: await self._externalize(
                    getattr(value, field_name),
                    thread_id=thread_id,
                    workflow_id=workflow_id,
                    key_hint=field_name,
                    force_sensitive=(
                        force_sensitive
                        or is_sensitive_field(field_name)
                        or is_sensitive_container(field_name)
                    ),
                    depth=depth + 1,
                )
                for field_name in model_fields
                if hasattr(value, field_name)
            }
            return value.model_copy(update=updates)
        return value

    async def _restore(self, value: Any, *, depth: int = 0) -> Any:
        if depth > 40:
            return value
        if self._is_reference(value):
            data, artifact, path = await self.repository.load(value["artifact_id"])
            kind = value.get("kind")
            if kind == "bytes":
                return data
            if kind == "bytearray":
                return bytearray(data)
            if kind == "base64":
                return base64.b64encode(data).decode("ascii")
            if kind == "data_uri":
                content_type = artifact.content_type or "application/octet-stream"
                encoded = base64.b64encode(data).decode("ascii")
                return f"data:{content_type};base64,{encoded}"
            if kind == "file_path":
                return str(path)
            logger.warning("Unknown checkpoint artifact kind: %s", kind)
            return value
        if is_encrypted(value):
            return decrypt_string(value)
        if isinstance(value, dict):
            return {key: await self._restore(item, depth=depth + 1) for key, item in value.items()}
        if isinstance(value, list):
            return [await self._restore(item, depth=depth + 1) for item in value]
        if isinstance(value, tuple):
            converted = [await self._restore(item, depth=depth + 1) for item in value]
            if hasattr(value, "_fields"):
                return type(value)(*converted)
            return tuple(converted)
        if is_dataclass(value) and not isinstance(value, type):
            updates = {
                field_name: await self._restore(getattr(value, field_name), depth=depth + 1)
                for field_name in value.__dataclass_fields__
            }
            return replace(value, **updates)
        model_fields = getattr(value.__class__, "model_fields", None)
        if model_fields is not None and hasattr(value, "model_copy"):
            updates = {
                field_name: await self._restore(getattr(value, field_name), depth=depth + 1)
                for field_name in model_fields
                if hasattr(value, field_name)
            }
            return value.model_copy(update=updates)
        return value

    async def _restore_tuple(self, item: CheckpointTuple | None) -> CheckpointTuple | None:
        if item is None:
            return None
        checkpoint = await self._restore(item.checkpoint)
        metadata = await self._restore(item.metadata)
        pending_writes = None
        if item.pending_writes is not None:
            pending_writes = [
                (task_id, channel, await self._restore(value))
                for task_id, channel, value in item.pending_writes
            ]
        return CheckpointTuple(
            config=item.config,
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=item.parent_config,
            pending_writes=pending_writes,
        )

    async def aget_tuple(self, config: dict) -> CheckpointTuple | None:
        return await self._restore_tuple(await self.saver.aget_tuple(config))

    async def alist(
        self,
        config: dict | None,
        *,
        filter: dict[str, Any] | None = None,
        before: dict | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        async for item in self.saver.alist(
            config,
            filter=filter,
            before=before,
            limit=limit,
        ):
            restored = await self._restore_tuple(item)
            if restored is not None:
                yield restored

    async def aput(
        self,
        config: dict,
        checkpoint: dict,
        metadata: dict,
        new_versions: Mapping[str, Any],
    ) -> dict:
        thread_id, workflow_id = self._context(config)
        clean_checkpoint = await self._externalize(
            checkpoint,
            thread_id=thread_id,
            workflow_id=workflow_id,
        )
        clean_metadata = await self._externalize(
            metadata,
            thread_id=thread_id,
            workflow_id=workflow_id,
        )
        return await self.saver.aput(
            config,
            clean_checkpoint,
            clean_metadata,
            new_versions,
        )

    async def aput_writes(
        self,
        config: dict,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id, workflow_id = self._context(config)
        clean_writes = [
            (
                channel,
                await self._externalize(
                    value,
                    thread_id=thread_id,
                    workflow_id=workflow_id,
                    key_hint=channel,
                ),
            )
            for channel, value in writes
        ]
        await self.saver.aput_writes(config, clean_writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        # Delete checkpoint rows first. Only then release artifact references;
        # this ensures a load can never observe a checkpoint whose file is gone.
        await self.saver.adelete_thread(thread_id)
        if hasattr(self.repository, "release_thread"):
            stats = await self.repository.release_thread(thread_id)
            if stats.get("failed"):
                logger.error(
                    "Checkpoint rows for thread %s were deleted, but %s artifact "
                    "cleanup operation(s) need retry",
                    thread_id,
                    stats["failed"],
                )

    async def adelete_for_runs(self, run_ids: Sequence[str]) -> None:
        await self.saver.adelete_for_runs(run_ids)

    async def acopy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        # Establish ownership first so a copied checkpoint can never point to a
        # file that source-thread cleanup is free to delete.
        if hasattr(self.repository, "copy_thread"):
            await self.repository.copy_thread(source_thread_id, target_thread_id)
        await self.saver.acopy_thread(source_thread_id, target_thread_id)

    async def aprune(
        self,
        thread_ids: Sequence[str],
        *,
        strategy: str = "keep_latest",
    ) -> None:
        await self.saver.aprune(thread_ids, strategy=strategy)

    async def aget_delta_channel_history(
        self,
        *,
        config: dict,
        channels: Sequence[str],
    ) -> Mapping[str, Any]:
        history = await self.saver.aget_delta_channel_history(
            config=config,
            channels=channels,
        )
        return await self._restore(history)

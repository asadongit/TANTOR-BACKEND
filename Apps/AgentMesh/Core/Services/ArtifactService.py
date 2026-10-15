import asyncio
import hashlib
import logging
import os
import uuid
from pathlib import Path
from typing import Optional
from sqlalchemy import delete, func, select, text
from sqlalchemy.exc import IntegrityError

from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from Apps.AgentMesh.Config import settings
from Database.Models import ArtifactModel, CheckpointArtifactReferenceModel

logger = logging.getLogger(__name__)


class ArtifactService:
    """
    Stores uploaded files in the configured artifact directory and keeps
    their metadata in PostgreSQL.
    """

    @classmethod
    async def save_upload(
        cls,
        file: UploadFile,
        thread_id: str,
        db: AsyncSession,
        workflow_id: Optional[str] = None,
    ) -> dict:
        """
        Saves one uploaded file and returns a lightweight artifact reference.
        """
        original_filename = file.filename or "upload"

        artifact_root = Path(settings.ARTIFACT_MOUNT).resolve()
        incoming_directory = artifact_root / "_incoming"
        incoming_directory.mkdir(parents=True, exist_ok=True)
        temporary_path = incoming_directory / f"{uuid.uuid4().hex}.part"

        size_bytes = 0
        digest_builder = hashlib.sha256()
        header = bytearray()
        final_path: Path | None = None

        try:
            # Write in chunks so a large file is not loaded fully into memory.
            with temporary_path.open("wb") as output_file:
                while chunk := await file.read(1024 * 1024):
                    output_file.write(chunk)
                    digest_builder.update(chunk)
                    if len(header) < 16:
                        header.extend(chunk[: 16 - len(header)])
                    size_bytes += len(chunk)

            digest = digest_builder.hexdigest()
            artifact_id = cls._artifact_id_for_hash(digest)
            existing = await db.execute(
                select(ArtifactModel)
                .where(ArtifactModel.id == artifact_id)
                .with_for_update()
            )
            artifact = existing.scalar_one_or_none()
            if artifact is not None:
                temporary_path.unlink(missing_ok=True)
                await cls._add_checkpoint_reference(
                    db,
                    thread_id=thread_id,
                    artifact_id=artifact_id,
                    workflow_id=workflow_id,
                )
                await db.commit()
                return {
                    "artifact_id": artifact_id,
                    "filename": original_filename,
                    "content_type": file.content_type,
                    "size_bytes": size_bytes,
                }

            artifact_directory = (
                artifact_root / "_objects" / digest[:2] / artifact_id
            )
            artifact_directory.mkdir(parents=True, exist_ok=True)
            final_path = artifact_directory / f"artifact{cls._binary_extension(bytes(header))}"

            # Rename only after the file has been written successfully.
            os.replace(temporary_path, final_path)

            relative_path = final_path.relative_to(
                artifact_root
            ).as_posix()

            artifact = ArtifactModel(
                id=artifact_id,
                thread_id=thread_id,
                workflow_id=workflow_id,
                original_filename=original_filename,
                relative_path=relative_path,
                content_type=file.content_type,
                size_bytes=size_bytes,
            )

            db.add(artifact)
            db.add(
                CheckpointArtifactReferenceModel(
                    thread_id=thread_id,
                    artifact_id=artifact_id,
                    workflow_id=workflow_id,
                )
            )
            try:
                await db.commit()
            except IntegrityError:
                # A concurrent upload of the same bytes won the insert race.
                await db.rollback()
                existing = await db.execute(
                    select(ArtifactModel).where(ArtifactModel.id == artifact_id)
                )
                if existing.scalar_one_or_none() is None:
                    raise
                await cls._add_checkpoint_reference(
                    db,
                    thread_id=thread_id,
                    artifact_id=artifact_id,
                    workflow_id=workflow_id,
                )
                await db.commit()

            return {
                "artifact_id": artifact_id,
                "filename": original_filename,
                "content_type": file.content_type,
                "size_bytes": size_bytes,
            }

        except Exception:
            await db.rollback()

            temporary_path.unlink(missing_ok=True)

            raise

    @classmethod
    async def save_checkpoint_bytes(
        cls,
        data: bytes,
        thread_id: str,
        db: AsyncSession,
        *,
        workflow_id: Optional[str] = None,
        filename: str = "checkpoint-artifact.bin",
        content_type: Optional[str] = "application/octet-stream",
    ) -> dict:
        """Persist checkpoint binary data once and return a stable reference.

        The artifact ID is deterministic from the content hash. This prevents
        the same state value from being copied to NFS at every graph superstep
        and lets separate threads safely share one physical file.
        """
        digest = hashlib.sha256(data).hexdigest()
        artifact_id = cls._artifact_id_for_hash(digest)
        existing = await db.execute(
            select(ArtifactModel)
            .where(ArtifactModel.id == artifact_id)
            .with_for_update()
        )
        artifact = existing.scalar_one_or_none()

        artifact_root = Path(settings.ARTIFACT_MOUNT).resolve()
        if artifact is not None:
            final_path = (artifact_root / artifact.relative_path).resolve()
            final_path.relative_to(artifact_root)
            if not final_path.is_file():
                await asyncio.to_thread(cls._write_bytes_atomically, final_path, data)
            await cls._add_checkpoint_reference(
                db,
                thread_id=thread_id,
                artifact_id=artifact.id,
                workflow_id=workflow_id,
            )
            await db.commit()
            return {
                "artifact_id": artifact.id,
                "filename": artifact.original_filename,
                "content_type": artifact.content_type,
                "size_bytes": artifact.size_bytes,
                "content_hash": digest,
            }

        artifact_directory = artifact_root / "_objects" / digest[:2] / artifact_id
        final_path = artifact_directory / f"artifact{cls._binary_extension(data[:16])}"
        relative_path = final_path.relative_to(artifact_root).as_posix()

        await asyncio.to_thread(cls._write_bytes_atomically, final_path, data)
        artifact = ArtifactModel(
            id=artifact_id,
            thread_id=thread_id,
            workflow_id=workflow_id,
            original_filename=filename,
            relative_path=relative_path,
            content_type=content_type,
            size_bytes=len(data),
        )
        db.add(artifact)
        db.add(
            CheckpointArtifactReferenceModel(
                thread_id=thread_id,
                artifact_id=artifact_id,
                workflow_id=workflow_id,
            )
        )
        try:
            await db.commit()
        except IntegrityError:
            # Another worker may have inserted the same deterministic artifact.
            await db.rollback()
            existing = await db.execute(
                select(ArtifactModel).where(ArtifactModel.id == artifact_id)
            )
            artifact = existing.scalar_one_or_none()
            if artifact is None:
                final_path.unlink(missing_ok=True)
                raise
            await cls._add_checkpoint_reference(
                db,
                thread_id=thread_id,
                artifact_id=artifact.id,
                workflow_id=workflow_id,
            )
            await db.commit()

        return {
            "artifact_id": artifact_id,
            "filename": filename,
            "content_type": content_type,
            "size_bytes": len(data),
            "content_hash": digest,
        }

    @staticmethod
    def _artifact_id_for_hash(digest: str) -> str:
        """Return one stable ID for identical content from any workflow/thread."""
        return str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"agentmesh-artifact:{digest}",
            )
        )

    @staticmethod
    def _binary_extension(header: bytes) -> str:
        """Choose a stable extension from file bytes, not user-supplied metadata."""
        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            return ".png"
        if header.startswith(b"\xff\xd8\xff"):
            return ".jpg"
        if header.startswith((b"GIF87a", b"GIF89a")):
            return ".gif"
        if header.startswith(b"%PDF-"):
            return ".pdf"
        if header.startswith(b"PK\x03\x04"):
            return ".zip"
        if header.startswith(b"RIFF"):
            return ".wav"
        if header.startswith((b"II*\x00", b"MM\x00*")):
            return ".tiff"
        return ".bin"

    @staticmethod
    async def _add_checkpoint_reference(
        db: AsyncSession,
        *,
        thread_id: str,
        artifact_id: str,
        workflow_id: Optional[str],
    ) -> None:
        """Add the idempotent thread-to-artifact ownership record."""
        existing = await db.get(
            CheckpointArtifactReferenceModel,
            (thread_id, artifact_id),
        )
        if existing is None:
            db.add(
                CheckpointArtifactReferenceModel(
                    thread_id=thread_id,
                    artifact_id=artifact_id,
                    workflow_id=workflow_id,
                )
            )

    @classmethod
    async def register_checkpoint_reference(
        cls,
        artifact_id: str,
        thread_id: str,
        db: AsyncSession,
        *,
        workflow_id: Optional[str] = None,
    ) -> None:
        """Register an existing checkpoint artifact as used by another thread."""
        artifact_result = await db.execute(
            select(ArtifactModel)
            .where(ArtifactModel.id == artifact_id)
            .with_for_update()
        )
        if artifact_result.scalar_one_or_none() is None:
            raise ValueError(f"Artifact not found: {artifact_id}")
        await cls._add_checkpoint_reference(
            db,
            thread_id=thread_id,
            artifact_id=artifact_id,
            workflow_id=workflow_id,
        )
        await db.commit()

    @classmethod
    async def copy_checkpoint_references(
        cls,
        source_thread_id: str,
        target_thread_id: str,
        db: AsyncSession,
    ) -> None:
        """Preserve shared artifacts when LangGraph copies/forks a thread."""
        result = await db.execute(
            select(CheckpointArtifactReferenceModel).where(
                CheckpointArtifactReferenceModel.thread_id == source_thread_id
            )
        )
        for reference in result.scalars().all():
            await cls._add_checkpoint_reference(
                db,
                thread_id=target_thread_id,
                artifact_id=reference.artifact_id,
                workflow_id=reference.workflow_id,
            )
        await db.commit()

    @classmethod
    async def release_checkpoint_artifacts(
        cls,
        thread_id: str,
        db: AsyncSession,
    ) -> dict:
        """Release one thread and delete only artifacts with no remaining user.

        Each artifact is handled independently. If an NFS deletion fails, its
        database reference is retained so the operation is safe to retry.
        """
        result = await db.execute(
            select(CheckpointArtifactReferenceModel.artifact_id).where(
                CheckpointArtifactReferenceModel.thread_id == thread_id
            )
        )
        artifact_ids = list(result.scalars().all())
        deleted = 0
        preserved = 0
        failed = 0

        for artifact_id in artifact_ids:
            try:
                artifact_result = await db.execute(
                    select(ArtifactModel)
                    .where(ArtifactModel.id == artifact_id)
                    .with_for_update()
                )
                artifact = artifact_result.scalar_one_or_none()
                await db.execute(
                    delete(CheckpointArtifactReferenceModel).where(
                        CheckpointArtifactReferenceModel.thread_id == thread_id,
                        CheckpointArtifactReferenceModel.artifact_id == artifact_id,
                    )
                )
                await db.flush()
                remaining = await db.scalar(
                    select(func.count())
                    .select_from(CheckpointArtifactReferenceModel)
                    .where(CheckpointArtifactReferenceModel.artifact_id == artifact_id)
                )

                if remaining:
                    preserved += 1
                elif artifact is not None:
                    await asyncio.to_thread(cls._delete_artifact_file, artifact)
                    await db.delete(artifact)
                    deleted += 1
                await db.commit()
            except Exception:
                failed += 1
                await db.rollback()
                logger.exception(
                    "Could not release checkpoint artifact %s for thread %s; "
                    "reference retained for retry",
                    artifact_id,
                    thread_id,
                )

        return {
            "released": len(artifact_ids) - failed,
            "deleted": deleted,
            "preserved": preserved,
            "failed": failed,
        }

    @classmethod
    def _delete_artifact_file(cls, artifact: ArtifactModel) -> None:
        """Delete one verified artifact file and its now-empty own directory."""
        artifact_root = Path(settings.ARTIFACT_MOUNT).resolve()
        path = (artifact_root / artifact.relative_path).resolve()
        path.relative_to(artifact_root)
        path.unlink(missing_ok=True)
        try:
            path.parent.rmdir()
        except OSError:
            pass

    @classmethod
    async def release_orphaned_checkpoint_artifacts(cls, db: AsyncSession) -> dict:
        """Retry cleanup for references whose checkpoint thread no longer exists."""
        result = await db.execute(
            select(CheckpointArtifactReferenceModel.thread_id).distinct()
        )
        thread_ids = list(result.scalars().all())
        totals = {"threads": 0, "deleted": 0, "preserved": 0, "failed": 0}
        for thread_id in thread_ids:
            checkpoint_exists = await db.scalar(
                text("SELECT 1 FROM checkpoints WHERE thread_id = :thread_id LIMIT 1"),
                {"thread_id": thread_id},
            )
            if checkpoint_exists is not None:
                continue
            stats = await cls.release_checkpoint_artifacts(thread_id, db)
            totals["threads"] += 1
            for key in ("deleted", "preserved", "failed"):
                totals[key] += stats[key]
        return totals

    @staticmethod
    def _write_bytes_atomically(final_path: Path, data: bytes) -> None:
        """Write bytes using a temporary sibling file followed by rename."""
        final_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = final_path.with_name(
            f".{final_path.name}.{uuid.uuid4().hex}.part"
        )
        try:
            temporary_path.write_bytes(data)
            os.replace(temporary_path, final_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    @classmethod
    async def read_bytes(
        cls,
        artifact_id: str,
        db: AsyncSession,
    ) -> tuple[bytes, ArtifactModel, Path]:
        """Read an artifact and its metadata for checkpoint rehydration."""
        artifact = await cls.get_by_id(artifact_id=artifact_id, db=db)
        path = await cls.resolve_path(artifact_id=artifact_id, db=db)
        data = await asyncio.to_thread(path.read_bytes)
        return data, artifact, path
             
    @classmethod
    async def get_by_id(
        cls,
        artifact_id: str,
        db: AsyncSession,
    ) -> ArtifactModel:
        """
        Finds an artifact metadata record using its ID.
        """
        result = await db.execute(
            select(ArtifactModel).where(
                ArtifactModel.id == artifact_id
            )
        )

        artifact = result.scalar_one_or_none()

        if artifact is None:
            raise ValueError(
                f"Artifact not found: {artifact_id}"
            )

        return artifact

    @classmethod
    async def resolve_path(
        cls,
        artifact_id: str,
        db: AsyncSession,
    ) -> Path:
        """
        Converts an artifact ID into a verified filesystem path.
        """
        artifact = await cls.get_by_id(
            artifact_id=artifact_id,
            db=db,
        )

        artifact_root = Path(
            settings.ARTIFACT_MOUNT
        ).resolve()

        resolved_path = (
            artifact_root / artifact.relative_path
        ).resolve()

        # Prevent a database path from escaping the artifact directory.
        try:
            resolved_path.relative_to(artifact_root)
        except ValueError as error:
            raise ValueError(
                "Artifact path points outside the configured storage root"
            ) from error

        if not resolved_path.is_file():
            raise FileNotFoundError(
                f"Artifact file is missing: {artifact_id}"
            )

        return resolved_path

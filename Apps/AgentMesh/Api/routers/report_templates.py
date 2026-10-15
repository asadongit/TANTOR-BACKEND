"""
report_templates.py
────────────────────
FastAPI router for managing document templates.

Fixes vs previous version:
  1. Async DB session — imports AsyncSession (not sync Session) from sqlalchemy.ext.asyncio,
     matching the AsyncSessionLocal used throughout the rest of the backend.
  2. Atomic upload — all files in a batch are added and committed in a single transaction.
     If any file fails, the entire batch is rolled back cleanly.
  3. Consistent format_type normalisation — always stored with a leading dot (.docx, .xlsx, .pdf).
  4. DELETE endpoint added — needed for the "wipe and restart" workflow you use during dev.
"""

import os
import uuid
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from Database.Database import get_db
from Database.Models import DocumentTemplateModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/report-templates", tags=["Report Templates"])


# ── GET — list templates ──────────────────────────────────────────────────────

@router.get("/")
async def get_templates(
    category: Optional[str] = None,
    format_type: Optional[str] = None,
    db: AsyncSession = Depends(get_db),          # AsyncSession, not Session
):
    """
    Return available document templates, optionally filtered by category or format.

    Examples:
      GET /report-templates/
      GET /report-templates/?category=Fraud
      GET /report-templates/?format_type=.docx
    """
    query = select(DocumentTemplateModel)

    if category:
        query = query.where(DocumentTemplateModel.category == category)
    if format_type:
        # Accept with or without leading dot
        fmt = format_type if format_type.startswith(".") else "." + format_type
        query = query.where(DocumentTemplateModel.format_type == fmt)

    result = await db.execute(query)
    templates = result.scalars().all()

    return [
        {
            "id":           t.id,
            "name":         t.name,
            "category":     t.category,
            "format_type":  t.format_type,
            "storage_path": t.storage_path,
        }
        for t in templates
    ]


# ── POST — upload templates ───────────────────────────────────────────────────

@router.post("/upload")
async def upload_template(
    name: str                  = Form(...),
    category: str              = Form("General"),
    format_type: str           = Form(...),
    files: List[UploadFile]    = File(...),
    db: AsyncSession           = Depends(get_db),
):
    """
    Upload one or more document templates.

    Saves files to ./mnt/templates/ with UUID filenames and registers each
    in the database. All files in a single request are committed atomically —
    if any file fails to save, none are committed.

    Body (multipart/form-data):
      name         — display name (suffix " (N)" appended for multi-file uploads)
      category     — optional grouping label, default "General"
      format_type  — canonical format (.docx, .xlsx, .pdf)
      files        — one or more files
    """
    from Apps.AgentMesh.Config import settings
    templates_dir = settings.TEMPLATES_MOUNT
    os.makedirs(templates_dir, exist_ok=True)

    saved_files: List[str] = []   # track for cleanup on partial failure
    new_records: List[DocumentTemplateModel] = []

    try:
        for idx, file in enumerate(files):
            # Derive extension from actual filename, regardless of form field
            file_ext = os.path.splitext(file.filename)[1].lower()
            if not file_ext:
                raise HTTPException(
                    status_code=400,
                    detail=f"File '{file.filename}' has no extension. Cannot determine format.",
                )

            unique_filename = f"{uuid.uuid4().hex}{file_ext}"
            storage_path    = os.path.join(templates_dir, unique_filename)

            content = await file.read()
            with open(storage_path, "wb") as f:
                f.write(content)
            saved_files.append(storage_path)
            logger.info(f"[TemplateUpload] Saved file → {storage_path}")

            # Normalise format_type: always store with leading dot
            if len(files) == 1:
                fmt = format_type if format_type.startswith(".") else "." + format_type
            else:
                # Multi-file: infer format from each file's extension
                fmt = file_ext

            template_name = name if len(files) == 1 else f"{name} ({idx + 1})"

            record = DocumentTemplateModel(
                name=template_name,
                category=category,
                format_type=fmt,
                storage_path=storage_path,
            )
            db.add(record)
            new_records.append(record)

        # Single atomic commit for the whole batch
        await db.commit()

        # Refresh to get DB-assigned IDs
        for record in new_records:
            await db.refresh(record)

        uploaded_ids = [r.id for r in new_records]
        logger.info(f"[TemplateUpload] Committed {len(uploaded_ids)} template(s): {uploaded_ids}")
        return {"status": "success", "template_ids": uploaded_ids}

    except HTTPException:
        raise
    except Exception as exc:
        await db.rollback()
        # Clean up any files that were written before the failure
        for path in saved_files:
            try:
                os.remove(path)
            except OSError:
                pass
        logger.error(f"[TemplateUpload] Failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to upload template: {str(exc)}")


# ── DELETE — remove a template ────────────────────────────────────────────────

@router.delete("/{template_id}")
async def delete_template(
    template_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Delete a template by ID (removes DB record and file from disk).
    """
    result = await db.execute(
        select(DocumentTemplateModel).where(DocumentTemplateModel.id == template_id)
    )
    record = result.scalars().first()

    if not record:
        raise HTTPException(status_code=404, detail=f"Template '{template_id}' not found.")

    # Remove from disk
    if record.storage_path and os.path.exists(record.storage_path):
        try:
            os.remove(record.storage_path)
            logger.info(f"[TemplateDelete] Removed file: {record.storage_path}")
        except OSError as exc:
            logger.warning(f"[TemplateDelete] Could not remove file {record.storage_path}: {exc}")

    await db.delete(record)
    await db.commit()
    logger.info(f"[TemplateDelete] Deleted template ID: {template_id}")

    return {"status": "success", "deleted_id": template_id}


# ── DELETE ALL — dev utility ──────────────────────────────────────────────────

@router.delete("/")
async def delete_all_templates(db: AsyncSession = Depends(get_db)):
    """
    Wipe all templates (DB records + files on disk).
    Useful during development when you want a clean slate.
    """
    result = await db.execute(select(DocumentTemplateModel))
    records = result.scalars().all()

    deleted_ids = []
    for record in records:
        if record.storage_path and os.path.exists(record.storage_path):
            try:
                os.remove(record.storage_path)
            except OSError:
                pass
        await db.delete(record)
        deleted_ids.append(record.id)

    await db.commit()
    logger.info(f"[TemplateDelete] Wiped {len(deleted_ids)} templates.")
    return {"status": "success", "deleted_count": len(deleted_ids), "deleted_ids": deleted_ids}

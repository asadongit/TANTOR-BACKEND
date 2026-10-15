from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from ..Models import WorkflowNoteModel

async def list_workflow_notes(db: AsyncSession, workflow_id: str):
    """Lists all notes for a specific workflow."""
    stmt = select(WorkflowNoteModel).where(WorkflowNoteModel.workflow_id == workflow_id).order_by(WorkflowNoteModel.created_at.desc())
    result = await db.execute(stmt)
    return result.scalars().all()

async def create_workflow_note(db: AsyncSession, workflow_id: str, content: str, category: str = "General"):
    """Creates a new note for a workflow."""
    db_note = WorkflowNoteModel(workflow_id=workflow_id, content=content, category=category)
    db.add(db_note)
    await db.commit()
    await db.refresh(db_note)
    return db_note

async def delete_workflow_note(db: AsyncSession, note_id: str):
    """Deletes a note by ID."""
    stmt = select(WorkflowNoteModel).where(WorkflowNoteModel.id == note_id)
    result = await db.execute(stmt)
    db_note = result.scalar_one_or_none()
    if db_note:
        await db.delete(db_note)
        await db.commit()
        return True
    return False

async def update_workflow_note(db: AsyncSession, note_id: str, content: Optional[str] = None, category: Optional[str] = None):
    """Updates an existing note's content or category."""
    stmt = select(WorkflowNoteModel).where(WorkflowNoteModel.id == note_id)
    result = await db.execute(stmt)
    db_note = result.scalar_one_or_none()
    if db_note:
        if content is not None:
            db_note.content = content
        if category is not None:
            db_note.category = category
        await db.commit()
        await db.refresh(db_note)
        return db_note
    return None

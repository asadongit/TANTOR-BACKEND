from typing import Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from ..Models import NotificationModel
from Apps.AgentMesh.Utilities.Security import redact_sensitive_data, sanitize_for_logging

async def create_notification(
    db: AsyncSession, 
    workflow_id: Optional[str], 
    title: str, 
    message: str, 
    type: str = "info", 
    thread_id: Optional[str] = None,
    metadata_json: Optional[Dict[str, Any]] = None
):
    """Creates a new in-app notification."""
    db_notification = NotificationModel(
        workflow_id=workflow_id,
        thread_id=thread_id,
        title=sanitize_for_logging(title),
        message=sanitize_for_logging(message),
        type=type,
        metadata_json=sanitize_for_logging(redact_sensitive_data(metadata_json))
    )
    db.add(db_notification)
    await db.commit()
    await db.refresh(db_notification)
    return db_notification

async def list_notifications(
    db: AsyncSession, 
    skip: int = 0, 
    limit: int = 50, 
    unread_only: bool = False,
    workflow_id: Optional[str] = None,
    type: Optional[str] = None
):
    """Lists notifications with pagination and optional filtering."""
    stmt = select(NotificationModel).order_by(NotificationModel.created_at.desc()).offset(skip).limit(limit)
    if unread_only:
        stmt = stmt.where(NotificationModel.is_read == False)
    if workflow_id:
        stmt = stmt.where(NotificationModel.workflow_id == workflow_id)
    if type:
        stmt = stmt.where(NotificationModel.type == type)
        
    result = await db.execute(stmt)
    return result.scalars().all()

async def count_notifications(
    db: AsyncSession, 
    unread_only: bool = False,
    workflow_id: Optional[str] = None,
    type: Optional[str] = None
):
    """Returns the total number of notifications."""
    stmt = select(func.count()).select_from(NotificationModel)
    if unread_only:
        stmt = stmt.where(NotificationModel.is_read == False)
    if workflow_id:
        stmt = stmt.where(NotificationModel.workflow_id == workflow_id)
    if type:
        stmt = stmt.where(NotificationModel.type == type)
    result = await db.execute(stmt)
    return result.scalar()

async def mark_notification_as_read(db: AsyncSession, notification_id: str):
    """Marks a single notification as read."""
    stmt = select(NotificationModel).where(NotificationModel.id == notification_id)
    result = await db.execute(stmt)
    db_notification = result.scalar_one_or_none()
    if db_notification:
        db_notification.is_read = True
        await db.commit()
        await db.refresh(db_notification)
        return True
    return False

async def mark_all_notifications_as_read(db: AsyncSession):
    """Marks all unread notifications as read."""
    from sqlalchemy import update
    stmt = update(NotificationModel).where(NotificationModel.is_read == False).values(is_read=True)
    await db.execute(stmt)
    await db.commit()
    return True

async def delete_notification(db: AsyncSession, notification_id: str):
    """Deletes a notification by ID."""
    stmt = select(NotificationModel).where(NotificationModel.id == notification_id)
    result = await db.execute(stmt)
    db_notification = result.scalar_one_or_none()
    if db_notification:
        await db.delete(db_notification)
        await db.commit()
        return True
    return False

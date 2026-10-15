from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from ..Models import WorkflowTemplateModel
from .workflows import process_workflow_credentials

async def save_workflow_template(
    db: AsyncSession,
    name: str,
    description: Optional[str],
    group: str,
    category: str,
    config: dict,
    scope: str = "private",
    owner: str = "system",
    status: str = "pending"
):
    """
    Saves a workflow configuration as a reusable template.
    NOTE: Owner is set as 'system' for now. RBAC checks will be added later.
    """
    # Encrypt sensitive configurations at rest
    encrypted_config = process_workflow_credentials(config, mode="encrypt")
    db_template = WorkflowTemplateModel(
        name=name,
        description=description,
        group=group,
        category=category,
        config=encrypted_config,
        owner=owner, # FUTURE TODO: Integrate role-based identity checks
        scope=scope,
        status=status # FUTURE TODO: Add admin status approval workflow ('pending', 'approved', 'rejected')
    )
    db.add(db_template)
    await db.commit()
    await db.refresh(db_template)
    return db_template

async def list_workflow_templates(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 100,
    group: Optional[str] = None,
    category: Optional[str] = None,
    search: Optional[str] = None,
    scope: Optional[str] = None,
    status: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Lists workflow templates summaries with pagination and filtering.
    Intentionally excludes the full 'config' column to avoid loading large JSON
    blobs. Extracts only 'nodes' and 'edges' from config for visual preview.
    """
    stmt = select(
        WorkflowTemplateModel.id,
        WorkflowTemplateModel.name,
        WorkflowTemplateModel.description,
        WorkflowTemplateModel.group,
        WorkflowTemplateModel.category,
        WorkflowTemplateModel.scope,
        WorkflowTemplateModel.status,
        WorkflowTemplateModel.owner,
        WorkflowTemplateModel.config["nodes"].label("nodes"),
        WorkflowTemplateModel.config["edges"].label("edges"),
        WorkflowTemplateModel.created_at,
        WorkflowTemplateModel.updated_at
    ).offset(skip).limit(limit)
    
    if group and group != "All":
        stmt = stmt.where(WorkflowTemplateModel.group == group)
    if category and category != "All":
        stmt = stmt.where(WorkflowTemplateModel.category == category)
    if scope and scope != "All":
        stmt = stmt.where(WorkflowTemplateModel.scope == scope)
    if status and status != "All":
        stmt = stmt.where(WorkflowTemplateModel.status == status)
    else:
        if status is None:
            stmt = stmt.where(WorkflowTemplateModel.status == "approved")
    if search:
        safe = search.replace("~", "~~").replace("%", "~%").replace("_", "~_")
        pattern = f"%{safe}%"
        stmt = stmt.where(
            (WorkflowTemplateModel.name.ilike(pattern, escape="~")) |
            (WorkflowTemplateModel.description.ilike(pattern, escape="~"))
        )
    
    result = await db.execute(stmt)
    rows = result.mappings().all()
    return [dict(row) for row in rows]

async def count_workflow_templates(
    db: AsyncSession,
    group: Optional[str] = None,
    category: Optional[str] = None,
    search: Optional[str] = None,
    scope: Optional[str] = None,
    status: Optional[str] = None
) -> int:
    """Returns the total number of workflow templates matching filters."""
    stmt = select(func.count()).select_from(WorkflowTemplateModel)
    if group and group != "All":
        stmt = stmt.where(WorkflowTemplateModel.group == group)
    if category and category != "All":
        stmt = stmt.where(WorkflowTemplateModel.category == category)
    if scope and scope != "All":
        stmt = stmt.where(WorkflowTemplateModel.scope == scope)
    if status and status != "All":
        stmt = stmt.where(WorkflowTemplateModel.status == status)
    else:
        if status is None:
            stmt = stmt.where(WorkflowTemplateModel.status == "approved")
    if search:
        safe = search.replace("~", "~~").replace("%", "~%").replace("_", "~_")
        pattern = f"%{safe}%"
        stmt = stmt.where(
            (WorkflowTemplateModel.name.ilike(pattern, escape="~")) |
            (WorkflowTemplateModel.description.ilike(pattern, escape="~"))
        )
    result = await db.execute(stmt)
    return result.scalar()

async def update_workflow_template_status(db: AsyncSession, template_id: str, status: str) -> Optional[WorkflowTemplateModel]:
    """Updates the admin status of a workflow template."""
    stmt = select(WorkflowTemplateModel).where(WorkflowTemplateModel.id == template_id)
    result = await db.execute(stmt)
    t = result.scalar_one_or_none()
    if t:
        t.status = status
        t.updated_at = datetime.utcnow()
        await db.commit()
        await db.refresh(t)
        return t
    return None

async def delete_workflow_template(db: AsyncSession, template_id: str) -> bool:
    """Deletes a workflow template by ID."""
    stmt = select(WorkflowTemplateModel).where(WorkflowTemplateModel.id == template_id)
    result = await db.execute(stmt)
    t = result.scalar_one_or_none()
    if t:
        await db.delete(t)
        await db.commit()
        return True
    return False

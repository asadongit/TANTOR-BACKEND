from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional, List, Dict
from Database.Database import get_db
from Database import AgentMeshCrud
from Database.Models import AgentTemplateModel
from Apps.AgentMesh.Utilities.Security import redact_sensitive_data

router = APIRouter(tags=["Agent Templates"])

class AgentTemplateCreate(BaseModel):
    name: str
    description: Optional[str] = None
    group: str = "Default"
    category: str = "General"
    config: dict

@router.post("/agent-templates")
async def create_agent_template(template: AgentTemplateCreate, db: AsyncSession = Depends(get_db)):
    try:
        res = await AgentMeshCrud.save_agent_template(
            db, 
            name=template.name,
            description=template.description,
            group=template.group,
            category=template.category,
            config=template.config
        )
        return {"status": "success", "id": res.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/agent-templates")
async def list_agent_templates(
    page: int = 1,
    limit: int = 50,
    group: Optional[str] = None,
    category: Optional[str] = None,
    search: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    skip = (page - 1) * limit
    # CRUD returns summary dicts — no config column, no decrypt
    templates = await AgentMeshCrud.list_agent_templates(db, skip=skip, limit=limit, group=group, category=category, search=search)
    total = await AgentMeshCrud.count_agent_templates(db, group=group, category=category, search=search)
    
    return {
        "status": "success",
        "total": total,
        "page": page,
        "limit": limit,
        "templates": templates  # already plain summary dicts
    }

@router.get("/agent-template/{template_id}")
async def get_agent_template(template_id: str, db: AsyncSession = Depends(get_db)):
    """
    Returns the full config with any sensitive values masked.
    Used when opening the Details modal or placing a template on the canvas.
    """
    stmt = select(AgentTemplateModel).where(AgentTemplateModel.id == template_id)
    result = await db.execute(stmt)
    t = result.scalar_one_or_none()
    if not t:
        raise HTTPException(status_code=404, detail="Template not found")

    config = redact_sensitive_data(t.config)

    return {
        "status": "success",
        "id": t.id,
        "name": t.name,
        "description": t.description,
        "group": t.group,
        "category": t.category,
        "config": config,
        "created_at": t.created_at,
        "updated_at": t.updated_at
    }

@router.delete("/agent-template/{template_id}")
async def delete_agent_template(template_id: str, db: AsyncSession = Depends(get_db)):
    success = await AgentMeshCrud.delete_agent_template(db, template_id)
    if not success:
        raise HTTPException(status_code=404, detail="Template not found")
    return {"status": "success"}

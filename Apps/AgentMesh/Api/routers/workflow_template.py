from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional, List, Dict
import uuid
import logging
import copy

from Apps.AgentMesh.Schema.WorkflowSchema import WorkflowConfig
from Apps.AgentMesh.Core.WorkflowRegistry import workflow_registry
from Database.Database import get_db
from Database import AgentMeshCrud
from Database.AgentMeshCrud import redact_workflow_credentials
from Database.Models import WorkflowTemplateModel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Workflow Templates"])

class WorkflowTemplateCreate(BaseModel):
    name: str
    description: Optional[str] = None
    group: str = "Default"
    category: str = "General"
    config: dict
    scope: str = "private" # 'private', 'member', 'public'

@router.post("/workflow-templates")
async def create_workflow_template(template: WorkflowTemplateCreate, db: AsyncSession = Depends(get_db)):
    """
    Save the current canvas workflow as a reusable template.
    NOTE: Owner is currently hardcoded/defaulted to 'system'. RBAC will be added later.
    """
    try:
        res = await AgentMeshCrud.save_workflow_template(
            db, 
            name=template.name,
            description=template.description,
            group=template.group,
            category=template.category,
            config=template.config,
            scope=template.scope,
            owner="system",  # FUTURE TODO: Transition from hardcoded 'system' to token-derived user identity for RBAC
            status="pending" # FUTURE TODO: Add admin approval check and default state ('pending' / 'approved') based on scope
        )
        return {"status": "success", "id": res.id}
    except Exception as e:
        logger.exception("Failed to save workflow template:")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/workflow-templates")
async def list_workflow_templates_endpoint(
    page: int = 1,
    limit: int = 100,
    group: Optional[str] = None,
    category: Optional[str] = None,
    search: Optional[str] = None,
    scope: Optional[str] = None,
    status: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    List workflow templates with search/filter.
    FUTURE TODO: Filter templates based on user credentials and scope (private vs member vs public).
    """
    try:
        skip = (page - 1) * limit
        templates = await AgentMeshCrud.list_workflow_templates(
            db, skip=skip, limit=limit, group=group, category=category, search=search, scope=scope, status=status
        )
        total = await AgentMeshCrud.count_workflow_templates(
            db, group=group, category=category, search=search, scope=scope, status=status
        )
        
        # Serialize database mappings to dict response
        response_data = []
        for t in templates:
            nodes = t.get("nodes") or []
            edges = t.get("edges") or []
            preview_nodes = [{"id": n.get("id"), "type": n.get("data", {}).get("type") or n.get("type"), "label": n.get("data", {}).get("label") or n.get("id"), "position": n.get("position", {"x": 0, "y": 0})} for n in nodes]
            preview_edges = [{"id": e.get("id"), "source": e.get("source"), "target": e.get("target")} for e in edges]
            
            response_data.append({
                "id": t.get("id"),
                "name": t.get("name"),
                "description": t.get("description"),
                "group": t.get("group"),
                "category": t.get("category"),
                "scope": t.get("scope"),
                "status": t.get("status"),
                "owner": t.get("owner"),
                "preview_nodes": preview_nodes,
                "preview_edges": preview_edges,
                "created_at": t.get("created_at").isoformat() if t.get("created_at") else None,
                "updated_at": t.get("updated_at").isoformat() if t.get("updated_at") else None
            })
            
        # Get all unique groups and categories for template filter dropdowns (optimized DISTINCT queries)
        groups_stmt = select(WorkflowTemplateModel.group).distinct().where(WorkflowTemplateModel.group != None)
        groups_res = await db.execute(groups_stmt)
        unique_groups = sorted([row[0] for row in groups_res.all() if row[0]])

        categories_stmt = select(WorkflowTemplateModel.category).distinct().where(WorkflowTemplateModel.category != None)
        categories_res = await db.execute(categories_stmt)
        unique_categories = sorted([row[0] for row in categories_res.all() if row[0]])

        return {
            "status": "success",
            "total": total,
            "page": page,
            "limit": limit,
            "templates": response_data,
            "unique_groups": unique_groups,
            "unique_categories": unique_categories
        }
    except Exception as e:
        logger.exception("Failed to list templates:")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/workflow-template/{template_id}")
async def get_workflow_template_endpoint(template_id: str, db: AsyncSession = Depends(get_db)):
    """
    Get template details and configuration.
    """
    stmt = select(WorkflowTemplateModel).where(WorkflowTemplateModel.id == template_id)
    result = await db.execute(stmt)
    t = result.scalar_one_or_none()
    if not t:
        raise HTTPException(status_code=404, detail="Template not found")
        
    redacted_config = redact_workflow_credentials(t.config)
    return {
        "status": "success",
        "id": t.id,
        "name": t.name,
        "description": t.description,
        "group": t.group,
        "category": t.category,
        "scope": t.scope,
        "template_status": t.status,
        "owner": t.owner,
        "config": redacted_config,
        "created_at": t.created_at,
        "updated_at": t.updated_at
    }

@router.post("/workflow-template/{template_id}/clone")
async def clone_workflow_template_endpoint(template_id: str, db: AsyncSession = Depends(get_db)):
    """
    Use a template by cloning it into a new executable workflow.
    """
    stmt = select(WorkflowTemplateModel).where(WorkflowTemplateModel.id == template_id)
    result = await db.execute(stmt)
    t = result.scalar_one_or_none()
    if not t:
        raise HTTPException(status_code=404, detail="Template not found")
        
    # Generate new ID for the cloned workflow
    new_id = str(uuid.uuid4())
    config_dict = copy.deepcopy(t.config)
    
    # Keep ciphertext sealed while copying. save_workflow is idempotent for
    # already-protected values and runtime compilation reveals them later.
    config_dict["id"] = new_id
    config_dict["name"] = f"{t.name} (Clone)"
    
    try:
        # Convert dict to WorkflowConfig and save
        workflow_config = WorkflowConfig.model_validate(config_dict)
        await AgentMeshCrud.save_workflow(db, workflow_config, is_compile=False)
        return {"status": "success", "workflow_id": new_id, "name": workflow_config.name}
    except Exception as e:
        logger.exception("Failed to clone workflow template:")
        raise HTTPException(status_code=400, detail=str(e))

class UpdateStatusRequest(BaseModel):
    status: str

@router.patch("/workflow-template/{template_id}/status")
async def update_workflow_template_status_endpoint(
    template_id: str,
    request: UpdateStatusRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Update workflow template approval status (e.g., approved, rejected, pending).
    """
    updated = await AgentMeshCrud.update_workflow_template_status(db, template_id, request.status)
    if not updated:
        raise HTTPException(status_code=404, detail="Template not found")
    return {"status": "success", "id": updated.id, "template_status": updated.status}

@router.delete("/workflow-template/{template_id}")
async def delete_workflow_template_endpoint(template_id: str, db: AsyncSession = Depends(get_db)):
    """
    Delete a workflow template.
    """
    success = await AgentMeshCrud.delete_workflow_template(db, template_id)
    if not success:
        raise HTTPException(status_code=404, detail="Template not found")
    return {"status": "success", "message": f"Template {template_id} deleted successfully"}

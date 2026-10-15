from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional, List, Dict, Literal
from datetime import datetime
import uuid
import logging

from Apps.AgentMesh.Schema.WorkflowSchema import WorkflowConfig
from Apps.AgentMesh.Core.WorkflowRegistry import workflow_registry
from Apps.AgentMesh.Core.RedisManager import redis_manager
from Apps.AgentMesh.Utilities.Scheduler import scheduler
from Apps.AgentMesh.Utilities.Security import redact_sensitive_data, sanitize_for_logging
from Database.Database import get_db
from Database import AgentMeshCrud
from Database.Models import WorkflowModel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Workflows"])

WORKFLOW_NOT_FOUND = "Workflow not found"

WORKFLOW_CATEGORIES = [
    "Feedback Processing", "Batch Automation", "Content Creation", "Research & Analysis", "Customer Support", "Data Processing",
    "Social Media Management", "Email Marketing", "Lead Generation", "Sales Automation",
    "Human Resources", "Finance & Accounting", "Legal & Compliance", "IT Operations",
    "Product Management", "Project Management", "Software Development", "Quality Assurance",
    "Design & Creative", "Translation & Localization", "Personal Assistant", "Education & Training"
]

class WorkflowDetailResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    group: str
    category: str
    is_active: bool
    created_at: datetime
    updated_at: datetime
    config: WorkflowConfig
    node_count: int

@router.get("/workflows")
async def list_workflows(
    page: int = 1,
    limit: int = 100,
    group: Optional[str] = None,
    category: Optional[str] = None,
    search: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    skip = (page - 1) * limit
    # CRUD now returns lightweight summary dicts — no config column, no decrypt
    workflows = await AgentMeshCrud.list_workflows(db, skip=skip, limit=limit, group=group, category=category, search=search)
    total = await AgentMeshCrud.count_workflows(db, group=group, category=category, search=search)
    
    # Get all unique groups and categories for filter dropdowns
    all_workflows_stmt = select(WorkflowModel.group, WorkflowModel.category)
    all_res = await db.execute(all_workflows_stmt)
    all_rows = all_res.all()
    unique_groups = sorted(list(set(row[0] for row in all_rows if row[0])))
    unique_categories = sorted(list(set(row[1] for row in all_rows if row[1])))

    return {
        "status": "success",
        "total": total,
        "page": page,
        "limit": limit,
        "workflows": workflows,  # already plain dicts with summary fields only
        "unique_groups": unique_groups,
        "unique_categories": unique_categories
    }

@router.get("/workflow-categories")
async def get_workflow_categories_endpoint():
    return {"status": "success", "categories": WORKFLOW_CATEGORIES}

@router.get("/workflow/{workflow_id}", response_model=WorkflowDetailResponse)
async def get_workflow_endpoint(workflow_id: str, db: AsyncSession = Depends(get_db)):
    w = await AgentMeshCrud.get_workflow(db, workflow_id)
    if not w:
        raise HTTPException(status_code=404, detail=WORKFLOW_NOT_FOUND)
    
    # Frontend loads draft if available, otherwise config
    active_config = w.draft_config if w.draft_config else w.config
    active_config = AgentMeshCrud.process_workflow_credentials(active_config, mode="redact")
    config = WorkflowConfig.model_validate(active_config)
    return {
        "id": w.id,
        "name": w.name,
        "description": w.description,
        "group": w.group,
        "category": w.category,
        "is_active": w.is_active,
        "warmup": w.warmup,
        "created_at": w.created_at,
        "updated_at": w.updated_at,
        "config": config,
        "node_count": len(config.nodes)
    }

@router.delete("/workflow/{workflow_id}")
async def delete_workflow_endpoint(workflow_id: str, db: AsyncSession = Depends(get_db)):
    if await redis_manager.is_active(workflow_id):
        raise HTTPException(
            status_code=409,
            detail="Cannot delete: this workflow is currently executing. Please wait for it to finish."
        )
    success = await AgentMeshCrud.delete_workflow(db, workflow_id)
    if success:
        workflow_registry.unregister(workflow_id)
        scheduler.remove_workflow_schedule(workflow_id)
        return {"status": "success", "message": f"Workflow {workflow_id} deleted."}
    raise HTTPException(status_code=404, detail=WORKFLOW_NOT_FOUND)

class WarmupRequest(BaseModel):
    warmup: bool

@router.patch("/workflow/{workflow_id}/warmup")
async def set_workflow_warmup(workflow_id: str, request: WarmupRequest, db: AsyncSession = Depends(get_db)):
    """
    Toggle the warmup flag for a workflow.
    warmup=True  → immediately compile & cache (zero cold-start on next server restart).
    warmup=False → evict from cache; will lazy-compile on next request.
    """
    success = await AgentMeshCrud.set_workflow_warmup(db, workflow_id, request.warmup)
    if not success:
        raise HTTPException(status_code=404, detail=WORKFLOW_NOT_FOUND)

    if request.warmup:
        # Pre-compile now so it’s ready immediately (not just on next restart)
        w = await AgentMeshCrud.get_workflow(db, workflow_id, reveal_secrets=True)
        if w:
            try:
                config = WorkflowConfig.model_validate(w.config)
                workflow_registry.register_and_compile(config)
            except Exception as e:
                logger.warning(f"Warmup compile failed for '{workflow_id}': {e}")
    else:
        # Evict from cache — next request will lazy-compile
        workflow_registry.unregister(workflow_id)

    return {"status": "success", "workflow_id": workflow_id, "warmup": request.warmup}

@router.post("/workflow/{workflow_id}/clone")
async def clone_workflow_endpoint(workflow_id: str, db: AsyncSession = Depends(get_db)):
    w = await AgentMeshCrud.get_workflow(db, workflow_id, reveal_secrets=True)
    if not w:
        raise HTTPException(status_code=404, detail=WORKFLOW_NOT_FOUND)
    config = WorkflowConfig.model_validate(w.config)
    new_id = str(uuid.uuid4())
    config.id = new_id
    config.name = f"{config.name} (Clone)"
    try:
        await AgentMeshCrud.save_workflow(db, config)
        workflow_registry.register_and_compile(config)
        return {"status": "success", "workflow_id": new_id, "name": config.name}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.put("/workflow/{workflow_id}")
async def save_workflow_endpoint(workflow_id: str, config: WorkflowConfig, db: AsyncSession = Depends(get_db)):
    if workflow_id != config.id:
        raise HTTPException(status_code=400, detail="Workflow ID mismatch")
    if await redis_manager.is_active(workflow_id):
        raise HTTPException(
            status_code=409,
            detail="Cannot save: this workflow is currently executing. Please wait for it to finish."
        )
    try:
        await AgentMeshCrud.save_workflow(db, config, is_compile=False)
        return {"status": "success", "workflow_id": config.id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/workflow/{workflow_id}/compile")
async def compile_workflow_endpoint(workflow_id: str, config: WorkflowConfig, db: AsyncSession = Depends(get_db)):
    if workflow_id != config.id:
        raise HTTPException(status_code=400, detail="Workflow ID mismatch")
    try:
        await AgentMeshCrud.save_workflow(db, config, is_compile=True)
        runtime_workflow = await AgentMeshCrud.get_workflow(db, config.id, reveal_secrets=True)
        if not runtime_workflow:
            raise ValueError("Workflow was saved but could not be reloaded")
        runtime_config = WorkflowConfig.model_validate(runtime_workflow.config)
        workflow_registry.register_and_compile(runtime_config)
        w = await AgentMeshCrud.get_workflow(db, config.id)
        is_active = w.is_active if w else True
        if is_active and runtime_config.triggers:
            updated = False
            if runtime_config.triggers.cron and runtime_config.triggers.cron.enabled:
                scheduler.add_workflow_schedule(config.id, runtime_config.triggers.cron.cron_expression, runtime_config.triggers.cron.timezone)
                updated = True
            if runtime_config.triggers.polling and runtime_config.triggers.polling.enabled:
                poll = runtime_config.triggers.polling
                scheduler.add_polling_schedule(
                    workflow_id=config.id,
                    tool_name=poll.poller_tool,
                    tool_args=poll.tool_arguments,
                    destination_key=poll.destination_key,
                    cron_expression=poll.cron_expression,
                    timezone=poll.timezone
                )
                updated = True
            
            if not updated:
                scheduler.remove_workflow_schedule(config.id)
        else:
            scheduler.remove_workflow_schedule(config.id)
        return {"status": "success", "workflow_id": config.id}
    except Exception as e:
        logger.exception("Error during workflow compilation:")
        raise HTTPException(status_code=400, detail=str(e))

class ScheduleActionRequest(BaseModel):
    action: str # "update", "pause", "resume"

@router.post("/workflow/{workflow_id}/schedule")
async def dispatch_schedule_action(workflow_id: str, request: ScheduleActionRequest, db: AsyncSession = Depends(get_db)):
    if request.action == "pause":
        scheduler.pause_workflow_schedule(workflow_id)
        w = await AgentMeshCrud.get_workflow(db, workflow_id)
        if w:
            w.is_active = False
            await db.commit()
        return {"status": "success", "message": "Schedule paused"}
        
    elif request.action == "resume":
        scheduler.resume_workflow_schedule(workflow_id)
        w = await AgentMeshCrud.get_workflow(db, workflow_id)
        if w:
            w.is_active = True
            await db.commit()
        return {"status": "success", "message": "Schedule resumed"}
        
    elif request.action == "update":
        w = await AgentMeshCrud.get_workflow(db, workflow_id, reveal_secrets=True)
        if not w:
            raise HTTPException(status_code=404, detail="Workflow not found")
        config = WorkflowConfig.model_validate(w.config)
        if config.triggers:
            updated = False
            if config.triggers.cron and config.triggers.cron.enabled:
                scheduler.add_workflow_schedule(workflow_id, config.triggers.cron.cron_expression, config.triggers.cron.timezone)
                updated = True
            if config.triggers.polling and config.triggers.polling.enabled:
                poll = config.triggers.polling
                scheduler.add_polling_schedule(
                    workflow_id=workflow_id,
                    tool_name=poll.poller_tool,
                    tool_args=poll.tool_arguments,
                    destination_key=poll.destination_key,
                    cron_expression=poll.cron_expression,
                    timezone=poll.timezone
                )
                updated = True
            
            if updated:
                return {"status": "success", "message": "Schedule updated"}
        return {"status": "skipped"}
        
class PublishWorkflowRequest(BaseModel):
    tags: Optional[List[str]] = None
    marketplace_name: Optional[str] = None
    marketplace_description: Optional[str] = None
    launch_url: Optional[str] = None
    launch_mode: Optional[str] = None
    navigation_path: Optional[str] = None

@router.post("/workflow/{workflow_id}/publish")
async def publish_workflow_endpoint(
    workflow_id: str,
    request: Optional[PublishWorkflowRequest] = None,
    db: AsyncSession = Depends(get_db)
):
    try:
        tags = request.tags if request else None
        m_name = request.marketplace_name if request else None
        m_desc = request.marketplace_description if request else None
        launch_url = request.launch_url if request else None
        launch_mode = request.launch_mode if request else None
        navigation_path = request.navigation_path if request else None
        
        usecase = await AgentMeshCrud.publish_usecase(
            db, 
            workflow_id=workflow_id,
            tags=tags,
            marketplace_name=m_name,
            marketplace_description=m_desc,
            launch_url=launch_url,
            launch_mode=launch_mode,
            navigation_path=navigation_path
        )
        return {"status": "success", "usecase_id": usecase.id, "version": usecase.version}
    except ValueError as e:
        status_code = 404 if "not found" in str(e).lower() else 400
        raise HTTPException(status_code=status_code, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/workflow/{workflow_id}/marketplace")
async def get_workflow_marketplace_endpoint(workflow_id: str, db: AsyncSession = Depends(get_db)):
    from Database.Models import MarketplaceUsecaseModel
    stmt = select(MarketplaceUsecaseModel).where(MarketplaceUsecaseModel.workflow_id == workflow_id)
    res = await db.execute(stmt)
    usecase = res.scalar_one_or_none()
    if not usecase:
        return {"status": "success", "published": False}
    return sanitize_for_logging(redact_sensitive_data({
        "status": "success",
        "published": True,
        "marketplace_name": usecase.marketplace_name,
        "marketplace_description": usecase.marketplace_description,
        "tags": usecase.tags,
        "launch_url": usecase.launch_url,
        "launch_mode": usecase.launch_mode,
        "navigation_path": usecase.navigation_path,
        "version": usecase.version
    }))

@router.get("/marketplace/usecases")
async def get_marketplace_usecases_endpoint(
    page: int = 1,
    limit: int = 100,
    group: Optional[str] = None,
    category: Optional[str] = None,
    search: Optional[str] = None,
    tag: Optional[str] = None,
    admin_status: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    skip = (page - 1) * limit
    usecases = await AgentMeshCrud.list_marketplace_usecases(
        db, skip=skip, limit=limit, group=group, category=category, search=search, tag=tag, admin_status=admin_status
    )
    total = await AgentMeshCrud.count_marketplace_usecases(
        db, group=group, category=category, search=search, tag=tag, admin_status=admin_status
    )
    return {
        "status": "success",
        "total": total,
        "page": page,
        "limit": limit,
        "workflows": usecases
    }

class MarketplaceStatusRequest(BaseModel):
    status: str
    remarks: Optional[str] = None

@router.patch("/workflow/{workflow_id}/marketplace/status")
async def update_marketplace_status_endpoint(
    workflow_id: str,
    request: MarketplaceStatusRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Update admin status (e.g., approved, rejected, pending) and remarks for a marketplace usecase.
    """
    updated = await AgentMeshCrud.update_marketplace_usecase_status(
        db, workflow_id=workflow_id, admin_status=request.status, remarks=request.remarks
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Marketplace usecase not found")
    return {"status": "success", "id": updated.id, "admin_status": updated.admin_status}

@router.delete("/workflow/{workflow_id}/marketplace")
async def delete_marketplace_usecase_endpoint(
    workflow_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete / unpublish a marketplace usecase.
    """
    success = await AgentMeshCrud.delete_marketplace_usecase(db, workflow_id=workflow_id)
    if not success:
        raise HTTPException(status_code=404, detail="Marketplace usecase not found")
    return {"status": "success", "message": f"Marketplace usecase for workflow {workflow_id} deleted successfully"}

# --- Workflow Notes ---

class WorkflowNoteRequest(BaseModel):
    content: Optional[str] = None
    category: Optional[str] = None

class WorkflowNoteResponse(BaseModel):
    id: str
    workflow_id: str
    content: str
    category: str
    created_at: datetime
    updated_at: datetime

class NoteActionRequest(BaseModel):
    action: Literal["list", "create", "update", "delete"]
    note_id: Optional[str] = None
    content: Optional[str] = None
    category: Optional[str] = None

@router.post("/workflow/{workflow_id}/notes/action")
async def notes_dispatcher(workflow_id: str, request: NoteActionRequest, db: AsyncSession = Depends(get_db)):
    if request.action == "list":
        return await AgentMeshCrud.list_workflow_notes(db, workflow_id)
    elif request.action == "create":
        if not request.content:
            raise HTTPException(status_code=400, detail="content required")
        return await AgentMeshCrud.create_workflow_note(db, workflow_id, request.content, request.category or "General")
    elif request.action == "delete":
        if not request.note_id:
            raise HTTPException(status_code=400, detail="note_id required")
        success = await AgentMeshCrud.delete_workflow_note(db, request.note_id)
        if success:
            return {"status": "success", "message": "Note deleted"}
        raise HTTPException(status_code=404, detail="Note not found")
    elif request.action == "update":
        if not request.note_id:
            raise HTTPException(status_code=400, detail="note_id required")
        note = await AgentMeshCrud.update_workflow_note(db, request.note_id, request.content, request.category)
        if note:
            return note
        raise HTTPException(status_code=404, detail="Note not found")


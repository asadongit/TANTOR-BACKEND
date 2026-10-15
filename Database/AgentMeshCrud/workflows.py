import logging
import copy
from typing import Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, case, delete
from ..Models import WorkflowModel, ExecutionLogModel, WorkflowRunHistoryModel, ToolStateModel, NotificationModel, WorkflowNoteModel, MarketplaceUsecaseModel, WorkflowVariableModel
from Apps.AgentMesh.Schema.WorkflowSchema import WorkflowConfig
from Apps.AgentMesh.Utilities.Security import (
    merge_masked_sensitive_data,
    protect_sensitive_data,
    redact_sensitive_data,
    reveal_sensitive_data,
)
from .helpers import _apply_workflow_filters

logger = logging.getLogger(__name__)

def process_workflow_credentials(config: dict, mode: str = "encrypt") -> dict:
    """Protect or reveal workflow secrets using the shared application registry."""
    if mode == "encrypt":
        return protect_sensitive_data(config)
    if mode == "decrypt":
        return reveal_sensitive_data(config)
    if mode == "redact":
        return redact_sensitive_data(config)
    raise ValueError("mode must be one of: encrypt, decrypt, redact")

def redact_workflow_credentials(config: dict) -> dict:
    """
    Sanitizes the workflow template config by stripping all property data,
    only retaining the node-edge structural relationship and positions.
    This ensures no sensitive configs or values are sent to the client.
    """
    config = copy.deepcopy(config)
    
    # Process nodes to retain only layout/identity fields
    if "nodes" in config and isinstance(config["nodes"], list):
        sanitized_nodes = []
        for node in config["nodes"]:
            if not isinstance(node, dict):
                continue
            sanitized_node = {
                "id": node.get("id"),
                "type": node.get("type"),
                "position": node.get("position"),
                "label": node.get("label") or (node.get("data", {}).get("label") if isinstance(node.get("data"), dict) else None)
            }
            sanitized_node["data"] = {
                "id": node.get("id"),
                "type": node.get("type"),
                "label": sanitized_node["label"]
            }
            sanitized_nodes.append(sanitized_node)
        config["nodes"] = sanitized_nodes

    # Strip trigger details
    if "triggers" in config and isinstance(config["triggers"], dict):
        sanitized_triggers = {}
        for trigger_type, trigger_data in config["triggers"].items():
            if isinstance(trigger_data, dict):
                sanitized_triggers[trigger_type] = {
                    "enabled": trigger_data.get("enabled", True)
                }
        config["triggers"] = sanitized_triggers

    return config

async def save_workflow(db: AsyncSession, workflow_config: WorkflowConfig, is_compile: bool = False):
    """
    Saves or updates a workflow configuration in the database.
    If is_compile is True, it updates BOTH draft_config and config.
    If is_compile is False, it ONLY updates draft_config.
    """
    # Check if it exists
    stmt = select(WorkflowModel).where(WorkflowModel.id == workflow_config.id)
    result = await db.execute(stmt)
    db_workflow = result.scalar_one_or_none()

    if db_workflow:
        # Update
        db_workflow.name = workflow_config.name
        db_workflow.description = workflow_config.description
        db_workflow.group = workflow_config.group
        db_workflow.category = workflow_config.category
        # Persist warmup flag if provided in the config payload
        if hasattr(workflow_config, 'warmup') and workflow_config.warmup is not None:
            db_workflow.warmup = workflow_config.warmup
        # Preserve ciphertext represented by API masks, then encrypt new values.
        incoming_config = workflow_config.model_dump(exclude_unset=True, exclude_none=True)
        stored_config = db_workflow.draft_config or db_workflow.config or {}
        merged_config = merge_masked_sensitive_data(incoming_config, stored_config)
        encrypted_config = process_workflow_credentials(merged_config, mode="encrypt")
        db_workflow.draft_config = encrypted_config
        
        if is_compile:
            db_workflow.config = encrypted_config
    else:
        # Create new
        # ENCRYPT before saving
        encrypted_config = process_workflow_credentials(workflow_config.model_dump(exclude_unset=True, exclude_none=True), mode="encrypt")
        db_workflow = WorkflowModel(
            id=workflow_config.id,
            name=workflow_config.name,
            description=workflow_config.description,
            group=workflow_config.group,
            category=workflow_config.category,
            warmup=getattr(workflow_config, 'warmup', False) or False,
            config=encrypted_config,
            draft_config=encrypted_config
        )
        db.add(db_workflow)
    
    await db.commit()
    await db.refresh(db_workflow)
    return db_workflow

async def get_workflow(db: AsyncSession, workflow_id: str, *, reveal_secrets: bool = False):
    """
    Retrieves a workflow by its ID.
    """
    stmt = select(WorkflowModel).where(WorkflowModel.id == workflow_id)
    result = await db.execute(stmt)
    db_workflow = result.scalar_one_or_none()
    
    if db_workflow and reveal_secrets:
        # Never place plaintext on a session-managed entity: a later commit in
        # the same request could otherwise write the decrypted copy back.
        db.expunge(db_workflow)
        if db_workflow.config:
            # Runtime-only point of use. API callers leave reveal_secrets=False.
            db_workflow.config = process_workflow_credentials(db_workflow.config, mode="decrypt")
        if db_workflow.draft_config:
            db_workflow.draft_config = process_workflow_credentials(db_workflow.draft_config, mode="decrypt")
        
    return db_workflow

async def get_workflow_by_webhook_path(db: AsyncSession, webhook_path: str):
    """
    Retrieves an active workflow matching a given trigger webhook path from database config.
    Handles cold starts where the workflow is not in the in-memory registry.
    """
    stmt = select(WorkflowModel).where(WorkflowModel.is_active == True)
    result = await db.execute(stmt)
    workflows = result.scalars().all()
    
    for w in workflows:
        if w.config:
            config = process_workflow_credentials(w.config, mode="decrypt")
            triggers = config.get("triggers", {})
            if isinstance(triggers, dict):
                webhook = triggers.get("webhook", {})
                if isinstance(webhook, dict) and webhook.get("enabled", True):
                    path = webhook.get("path") or w.id
                    if path == webhook_path:
                        db.expunge(w)
                        w.config = config
                        return w
    return None

async def set_workflow_warmup(db: AsyncSession, workflow_id: str, warmup: bool) -> bool:
    """
    Toggles the warmup flag for a workflow.
    Intentionally updates only this column — does NOT re-encrypt or re-save config.
    """
    stmt = select(WorkflowModel).where(WorkflowModel.id == workflow_id)
    result = await db.execute(stmt)
    db_workflow = result.scalar_one_or_none()
    if not db_workflow:
        return False
    db_workflow.warmup = warmup
    await db.commit()
    return True

async def list_workflows(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 100,
    group: Optional[str] = None,
    category: Optional[str] = None,
    search: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Lists workflow SUMMARIES with pagination and optional filtering.
    Intentionally excludes the 'config' column to avoid loading large JSON
    blobs and running per-row decryption for views that only need metadata.
    Use get_workflow() for the full decrypted config of a single workflow.
    """
    stmt = select(
        WorkflowModel.id,
        WorkflowModel.name,
        WorkflowModel.description,
        WorkflowModel.group,
        WorkflowModel.category,
        WorkflowModel.is_active,
        WorkflowModel.warmup,
        WorkflowModel.created_at,
        WorkflowModel.updated_at,
    ).order_by(WorkflowModel.updated_at.desc()).offset(skip).limit(limit)

    stmt = _apply_workflow_filters(stmt, group, category, search)
    result = await db.execute(stmt)
    rows = result.mappings().all()
    return [dict(row) for row in rows]

async def count_workflows(
    db: AsyncSession,
    group: Optional[str] = None,
    category: Optional[str] = None,
    search: Optional[str] = None
) -> int:
    """Returns the total number of workflows matching the given filters."""
    stmt = select(func.count()).select_from(WorkflowModel)
    stmt = _apply_workflow_filters(stmt, group, category, search)
    result = await db.execute(stmt)
    return result.scalar()

async def delete_workflow(db: AsyncSession, workflow_id: str):
    """
    Deletes a workflow by its ID and performs a manual cascade delete 
    of all orphaned history, logs, tool memory, checkpointer states, and marketplace use cases.
    """
    stmt = select(WorkflowModel).where(WorkflowModel.id == workflow_id)
    result = await db.execute(stmt)
    db_workflow = result.scalar_one_or_none()
    
    if db_workflow:
        # 1. Clean up orphaned Tool State (Agent Memory) by resolving threads
        thread_ids_stmt = select(ExecutionLogModel.thread_id).where(ExecutionLogModel.workflow_id == workflow_id).distinct()
        thread_ids_res = await db.execute(thread_ids_stmt)
        thread_ids = [row[0] for row in thread_ids_res.all()]
        
        if thread_ids:
            # Delete tool state memory
            await db.execute(delete(ToolStateModel).where(ToolStateModel.thread_id.in_(thread_ids)))
            
            # Use the central saver so checkpoint artifacts and shared-reference
            # counts are cleaned together with each thread.
            from Apps.AgentMesh.Utilities.Memory.checkpointer import get_checkpointer

            checkpointer = get_checkpointer()
            for thread_id in thread_ids:
                await checkpointer.adelete_thread(thread_id)
        
        # 2. Clean up orphaned Execution Logs
        await db.execute(delete(ExecutionLogModel).where(ExecutionLogModel.workflow_id == workflow_id))
        await db.execute(
            delete(WorkflowRunHistoryModel).where(
                WorkflowRunHistoryModel.workflow_id == workflow_id
            )
        )
        
        # 3. Clean up orphaned Notifications, Notes, and Variable Explorer (Workflow Variables)
        await db.execute(delete(NotificationModel).where(NotificationModel.workflow_id == workflow_id))
        await db.execute(delete(WorkflowNoteModel).where(WorkflowNoteModel.workflow_id == workflow_id))
        await db.execute(delete(WorkflowVariableModel).where(WorkflowVariableModel.workflow_id == workflow_id))
        
        # 4. Clean up associated Marketplace Use Cases
        await db.execute(delete(MarketplaceUsecaseModel).where(MarketplaceUsecaseModel.workflow_id == workflow_id))
        
        # 5. Delete the parent Workflow
        await db.delete(db_workflow)
        await db.commit()
        return True
    return False

async def list_workflow_threads(db: AsyncSession, workflow_id: str, skip: int = 0, limit: int = 20):
    """
    Returns a list of unique thread sessions for a workflow.
    Ensures no duplicates and picks the first user message as title.
    """
    # 1. Fetch paginated unique thread_ids and their activity timestamps in one go
    # We use MIN for the title-log and MAX for the sorting order
    subq = (
        select(
            ExecutionLogModel.thread_id,
            func.min(ExecutionLogModel.created_at).label("first_seen"),
            func.max(ExecutionLogModel.created_at).label("last_active")
        )
        .where(ExecutionLogModel.workflow_id == workflow_id)
        .group_by(ExecutionLogModel.thread_id)
        .order_by(func.max(ExecutionLogModel.created_at).desc())
        .offset(skip)
        .limit(limit)
        .subquery()
    )

    # 2. Join back to get the logs that correspond to the FIRST activity of each thread
    # We also include last_active for the final sort
    stmt = (
        select(ExecutionLogModel, subq.c.last_active)
        .join(subq, (ExecutionLogModel.thread_id == subq.c.thread_id) & (ExecutionLogModel.created_at == subq.c.first_seen))
        .order_by(subq.c.last_active.desc(), ExecutionLogModel.id.asc())
    )
    
    result = await db.execute(stmt)
    logs_with_sort = result.all()
    
    threads = []
    seen_threads = set()
    
    for log, last_active in logs_with_sort:
        if log.thread_id in seen_threads: continue
        seen_threads.add(log.thread_id)
        
        # Extract first Human message for the title
        first_msg = "New Conversation"
        if log.thread_id.startswith("poll_"):
            first_msg = "Autonomous Polling Run"
        elif log.thread_id.startswith("webhook_") or log.thread_id.startswith("wh_"):
            first_msg = "Webhook Trigger Run"
        
        # Check in input_data metadata/messages
        data_sources = []
        if log.input_data and isinstance(log.input_data, dict):
            data_sources.append(log.input_data)
        if log.output_data and isinstance(log.output_data, dict):
            # Sometimes the start node outputs the messages
            data_sources.append(log.output_data)
            
        found_title = False
        for src in data_sources:
            # Check for explicitly supplied input text
            input_text = src.get("input_text") or src.get("input")
            if input_text and isinstance(input_text, str) and input_text.strip():
                if "Poller Trigger" in input_text or "polling" in log.thread_id:
                    if "via" in input_text:
                        first_msg = f"Polling: {input_text.split('via')[-1].strip()}"
                        found_title = True
                        break
                    else:
                        first_msg = "Polling Trigger Run"
                        found_title = True
                        break
            
            history = src.get("messages", [])
            if not isinstance(history, list): continue
            
            for m in history:
                content = ""
                m_type = ""
                
                if isinstance(m, dict):
                    m_type = m.get("type", "")
                    content = m.get("content", "")
                elif hasattr(m, "content"):
                    m_type = type(m).__name__
                    content = getattr(m, "content", "")
                
                # We specifically want the first HUMAN message as the title
                if "Human" in m_type and content and isinstance(content, str) and content.strip():
                    first_msg = content.strip()
                    found_title = True
                    break
            
            if found_title: break
        
        threads.append({
            "thread_id": log.thread_id,
            "workflow_id": log.workflow_id,
            "first_message": first_msg[:80] + ("..." if len(first_msg) > 80 else ""),
            "created_at": log.created_at,
            "updated_at": last_active
        })
        
    return threads

async def delete_workflow_thread(db: AsyncSession, thread_id: str):
    """Deletes all execution logs for a specific thread and its sessional temp files."""
    import shutil
    import os
    from Apps.AgentMesh.Config import settings
    
    # 1. Clean up filesystem session folders
    # TEMP_MOUNT covers EVERYTHING (uploads, exports, crops, audio, etc.) — all under {thread_id}/
    for mount_dir in [settings.TEMP_MOUNT]:
        session_dir = os.path.abspath(os.path.join(mount_dir, thread_id))
        if os.path.exists(session_dir):
            try:
                shutil.rmtree(session_dir)
                logger.info(f"Deleted filesystem session folder: {session_dir}")
            except Exception as e:
                logger.warning(f"Failed to delete filesystem session folder {session_dir}: {e}")

    # 2. Delete database records
    stmt = delete(ExecutionLogModel).where(ExecutionLogModel.thread_id == thread_id)
    await db.execute(stmt)

    await db.execute(
        delete(WorkflowRunHistoryModel).where(
            WorkflowRunHistoryModel.thread_id == thread_id
        )
    )
    
    # Also delete notifications for this thread
    stmt_notif = delete(NotificationModel).where(NotificationModel.thread_id == thread_id)
    await db.execute(stmt_notif)
    
    # NOTE: Checkpoints in checkpoints.sqlite are NOT deleted here yet, 
    # as they are managed by the LangGraph checkpointer (usually separate DB).
    
    await db.commit()
    return True

async def get_dashboard_stats(db: AsyncSession):
    """
    Retrieves high-level statistics for the dashboard:
    - Total, active, and inactive workflows
    - Workflows per category
    - Top 10 recently used workflows
    """
    # 1. Workflow counts by status (Optimized: 1 query instead of 3)
    wf_counts_stmt = select(
        func.count(WorkflowModel.id).label('total'),
        func.sum(case((WorkflowModel.is_active == True, 1), else_=0)).label('active'),
        func.sum(case((WorkflowModel.is_active == False, 1), else_=0)).label('inactive')
    )
    wf_counts_res = (await db.execute(wf_counts_stmt)).first()
    total_workflows = wf_counts_res.total or 0
    active_workflows = wf_counts_res.active or 0
    inactive_workflows = wf_counts_res.inactive or 0
    
    # 2. Workflows per category (and identity of agents)
    # We fetch all to do a Python-side check for agentic nodes which is easier than complex JSON SQL
    # for cross-DB compatibility (SQLite/Postgres)
    stmt_all = select(WorkflowModel.category, WorkflowModel.config)
    all_res = (await db.execute(stmt_all)).all()
    
    workflows_per_category = {}
    agents_per_category = {}
    
    for cat, config in all_res:
        cat = cat or "General"
        workflows_per_category[cat] = workflows_per_category.get(cat, 0) + 1
        
        # Identify if agentic (has llm_agent, supervisor, etc)
        nodes = config.get("nodes", []) if isinstance(config, dict) else []
        is_agent = any(n.get("type") in ["llm_agent", "deep_agent", "supervisor", "swarm", "swarm_agent"] for n in nodes)
        if is_agent:
            agents_per_category[cat] = agents_per_category.get(cat, 0) + 1
            
    # 3 & 4. Execution Logs & SLA Compliance (Optimized: 1 query instead of 3)
    from datetime import datetime, timedelta, timezone
    last_24h = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=24)
    
    log_counts_stmt = select(
        func.count(ExecutionLogModel.id).label('total_logs'),
        func.sum(case((ExecutionLogModel.status == "success", 1), else_=0)).label('success_logs'),
        func.sum(case((ExecutionLogModel.created_at >= last_24h, 1), else_=0)).label('logs_24h')
    )
    log_counts_res = (await db.execute(log_counts_stmt)).first()
    
    count_total_logs = log_counts_res.total_logs or 0
    count_success = log_counts_res.success_logs or 0
    count_24h = log_counts_res.logs_24h or 0
    
    executions_per_hr = round(count_24h / 24, 1)
    sla_compliance = round((count_success / count_total_logs * 100), 1) if count_total_logs > 0 else 100.0

    # 5. Top 10 recently used workflows (from execution logs)
    # Get the latest thread for each workflow
    subq = select(
        ExecutionLogModel.workflow_id,
        ExecutionLogModel.thread_id,
        ExecutionLogModel.status,
        ExecutionLogModel.created_at.label("last_run"),
        func.row_number().over(
            partition_by=ExecutionLogModel.workflow_id, 
            order_by=ExecutionLogModel.created_at.desc()
        ).label("rn")
    ).where(ExecutionLogModel.workflow_id != None).subquery()
    
    recent_stmt = select(WorkflowModel.id, WorkflowModel.name, WorkflowModel.group, WorkflowModel.category, subq.c.last_run, subq.c.thread_id, subq.c.status)\
        .join(subq, (WorkflowModel.id == subq.c.workflow_id) & (subq.c.rn == 1))\
        .order_by(subq.c.last_run.desc())\
        .limit(10)
    
    recent_res = (await db.execute(recent_stmt)).all()
    recent_workflows = [
        {
            "id": r[0],
            "name": r[1],
            "group": r[2],
            "category": r[3],
            "last_run": r[4].isoformat() if r[4] else None,
            "thread_id": r[5],
            "status": r[6]
        } for r in recent_res
    ]
    
    return {
        "total_workflows": total_workflows,
        "active_workflows": active_workflows,
        "inactive_workflows": inactive_workflows,
        "workflows_per_category": workflows_per_category,
        "agents_per_category": agents_per_category,
        "executions_per_hr": executions_per_hr,
        "sla_compliance": sla_compliance,
        "recent_workflows": recent_workflows
    }

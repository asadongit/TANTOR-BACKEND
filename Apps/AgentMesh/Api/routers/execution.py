from fastapi import APIRouter, HTTPException, Depends, Request, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.encoders import jsonable_encoder
from sqlalchemy.ext.asyncio import AsyncSession
import logging
import hmac
import hashlib
import json
from datetime import datetime
from fastapi.responses import JSONResponse, Response

from Apps.AgentMesh.Schema.RunSchema import RunWorkflowRequest, RunWorkflowResponse, ResumeWorkflowRequest, ThreadListResponse, ThreadHistoryResponse, ExecuteWorkflowRequest
from Apps.AgentMesh.Core.Services.WorkflowService import WorkflowService
from Apps.AgentMesh.Core.Services.ArtifactService import ArtifactService
from Apps.AgentMesh.Core.WorkflowRegistry import workflow_registry
from Apps.AgentMesh.Utilities.Observer import active_runs_registry, active_runs_revision
from Apps.AgentMesh.Utilities.Security import redact_sensitive_data, sanitize_for_logging
from Database.Database import get_db
from Database import AgentMeshCrud

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Execution"])


def _safe_api_payload(value):
    """Serialize and redact a runtime result before it crosses an API boundary."""
    return sanitize_for_logging(redact_sensitive_data(jsonable_encoder(value)))

@router.post("/workflow/{workflow_id}/execute")
async def execute_workflow(workflow_id: str, request: ExecuteWorkflowRequest, db: AsyncSession = Depends(get_db)):
    try:
        w = await AgentMeshCrud.get_workflow(db, workflow_id)
        if not w:
            raise HTTPException(status_code=404, detail="Workflow not found")
        if not w.is_active:
            raise HTTPException(status_code=400, detail="Workflow is paused. Resume it to run.")

        if request.mode == "resume" or (request.action and request.mode != "stream"):
            if not request.action:
                raise HTTPException(status_code=400, detail="action is required for resume mode")
            result = await WorkflowService.resume(
                workflow_id=workflow_id,
                thread_id=request.thread_id,
                action=request.action,
                payload=request.payload,
                db=db,
                trim_context=request.trim_context
            )
            return _safe_api_payload(result)
        elif request.mode == "stream":
            return StreamingResponse(
                WorkflowService.stream(
                    workflow_id=workflow_id,
                    input_text=request.input,
                    thread_id=request.thread_id,
                    context=request.context,
                    target_node_id=request.target_node_id,
                    db=db,
                    trim_context=request.trim_context
                ),
                media_type="text/event-stream"
            )
        else: # run (sync)
            result = await WorkflowService.run(
                workflow_id=workflow_id,
                input_text=request.input,
                thread_id=request.thread_id,
                context=request.context,
                target_node_id=request.target_node_id,
                db=db,
                trim_context=request.trim_context
            )
            return _safe_api_payload(result)
    except ValueError as e:
        logger.warning(f"Validation/ValueError executing workflow {workflow_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error executing workflow {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.api_route("/webhook/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def universal_webhook(path: str, request: Request, bg_tasks: BackgroundTasks, db: AsyncSession = Depends(get_db)):
    configs = workflow_registry.get_all_configs()
    target_wf = None
    
    # 1. Find the target workflow
    for config in configs:
        if config.id == path:
            target_wf = config
            break
            
    if not target_wf:
        for config in configs:
            if config.triggers and config.triggers.webhook and config.triggers.webhook.path == path:
                target_wf = config
                break
                
    if not target_wf:
        # Try lazy loading from DB by workflow_id or trigger webhook path
        w = await AgentMeshCrud.get_workflow(db, path, reveal_secrets=True)
        if not w:
            w = await AgentMeshCrud.get_workflow_by_webhook_path(db, path)
        if w:
            from Apps.AgentMesh.Schema.WorkflowSchema import WorkflowConfig
            config = WorkflowConfig.model_validate(w.config)
            workflow_registry.register_and_compile(config)
            target_wf = config

    if not target_wf:
        raise HTTPException(status_code=404, detail="Webhook endpoint not found")

    w = await AgentMeshCrud.get_workflow(db, target_wf.id)
    if w and not w.is_active:
        raise HTTPException(status_code=400, detail="Workflow is paused. Resume it to allow webhook triggers.")

    webhook_config = target_wf.triggers.webhook if target_wf.triggers else None
    if not webhook_config:
        raise HTTPException(status_code=404, detail="Webhook trigger not configured for this workflow")

    # Webhooks are intentionally outside user JWT authentication because their
    # callers are external systems. They must therefore configure their own
    # shared-secret or HMAC authentication before execution is allowed.
    if not webhook_config.secret:
        raise HTTPException(
            status_code=401,
            detail="Webhook authentication is not configured",
        )

    # 2. Authentication & Payload Parsing (JSON + Multipart file upload)
    content_type = request.headers.get("content-type", "")
    body = b""
    payload = {}
    uploaded_files_info = []
 
    import uuid
    thread_id = f"webhook_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
 
    if "multipart/form-data" in content_type:
        # Handle file uploads
        import os, uuid as _uuid
        from Apps.AgentMesh.Config import settings
        form = await request.form()
        for key, item in form.multi_items():
            if hasattr(item, 'read'):  # UploadFile
                file_bytes = await item.read()
                # Save to temp dir so downstream nodes can access
                upload_dir = os.path.abspath(os.path.join(settings.TEMP_MOUNT, thread_id))
                os.makedirs(upload_dir, exist_ok=True)
                safe_name = f"{_uuid.uuid4().hex[:8]}_{item.filename}"
                file_path = os.path.join(upload_dir, safe_name)
                with open(file_path, "wb") as f:
                    f.write(file_bytes)
                uploaded_files_info.append({
                    "filename": item.filename,
                    "content_type": item.content_type,
                    "size": len(file_bytes),
                    "path": file_path
                })
            else:
                # Form field
                payload[key] = item
        if uploaded_files_info:
            payload["uploaded_files"] = uploaded_files_info
            payload["file_count"] = len(uploaded_files_info)
        body = json.dumps(payload).encode() if payload else b""
    else:
        body = await request.body()
        if body:
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                try:
                    # Fix common curl issue where Windows paths have unescaped backslashes
                    fixed_body = body.decode('utf-8').replace('\\', '\\\\')
                    payload = json.loads(fixed_body)
                except Exception:
                    payload = {"raw_body": body.decode(errors='replace')}

    # Merge query parameters into payload to ensure variables like customer_folder_name resolve correctly
    for k, v in request.query_params.items():
        if k not in payload:
            payload[k] = v

    if webhook_config.secret:
        auth_method = getattr(webhook_config, 'auth_method', 'header')
        if auth_method == 'hmac':
            signature = request.headers.get("X-AgentMesh-Signature")
            if not signature:
                raise HTTPException(status_code=401, detail="Missing HMAC signature (X-AgentMesh-Signature)")
            
            expected = hmac.new(webhook_config.secret.encode(), body, hashlib.sha256).hexdigest()
            if not hmac.compare_digest(expected, signature):
                raise HTTPException(status_code=401, detail="Invalid HMAC signature")
        elif auth_method == 'header':
            header_secret = request.headers.get("X-AgentMesh-Secret") or ""
            if not hmac.compare_digest(header_secret, webhook_config.secret):
                raise HTTPException(status_code=401, detail="Invalid webhook secret")

    # 3. Execution
    context = {
        "source": "webhook", 
        "method": request.method,
        "path": path,
        "query_params": dict(request.query_params),
        "payload": payload,
        "headers": {k: v for k, v in request.headers.items() if not k.lower().startswith('x-agentmesh')}
    }

    response_mode = getattr(webhook_config, 'response_mode', 'immediate')
    
    if response_mode == "on_completion":
        # Synchronous execution
        try:
            result = await WorkflowService.run(
                workflow_id=target_wf.id, 
                input_text=f"Webhook {request.method} trigger: {path}", 
                thread_id=thread_id, 
                context=context
            )
            
            # 4. Extract Response
            # If a specific response node is configured, try to find its output in the context
            response_node_id = getattr(webhook_config, 'response_node', None)
            final_context = result.get("context", {})
            
            status_code = 200
            headers = {}
            
            # Default body is the last message content
            messages = result.get("messages", [])
            if messages:
                last_msg = messages[-1]
                if hasattr(last_msg, "content"):
                    body_content = last_msg.content
                elif isinstance(last_msg, dict) and "content" in last_msg:
                    body_content = last_msg["content"]
                else:
                    body_content = str(last_msg)
            else:
                body_content = "Workflow completed"

            # Check if there was a webhook_response node executed
            # We look for a special key in context that the webhook_response node would set
            if "__webhook_response__" in final_context:
                resp_data = final_context["__webhook_response__"]
                status_code = resp_data.get("status", 200)
                headers = resp_data.get("headers", {})
                body_content = resp_data.get("body", body_content)
            elif response_node_id:
                # If no explicit response node was hit but one is configured, 
                # we could try to find its output in the context if the aggregator or similar saved it.
                # For now, we'll just use the final output as fallback.
                pass

            body_content = _safe_api_payload(body_content)
            if isinstance(body_content, (dict, list)):
                return JSONResponse(content=body_content, status_code=status_code, headers=headers)
            return Response(content=str(body_content), status_code=status_code, headers=headers)
            
        except Exception as e:
            logger.error(f"Webhook execution error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    else:
        # Asynchronous execution (default)
        async def _run_webhook_async(wf_id: str, inp_text: str, th_id: str, ctx: dict):
            from Database.Database import AsyncSessionLocal
            from Database.Models import ExecutionLogModel
            import traceback
            async with AsyncSessionLocal() as session:
                try:
                    await WorkflowService.run(
                        workflow_id=wf_id,
                        input_text=inp_text,
                        thread_id=th_id,
                        context=ctx,
                        db=session
                    )
                except Exception as ex:
                    tb_str = traceback.format_exc()
                    logger.error(f"[Webhook Async Trigger Error] Workflow '{wf_id}' thread '{th_id}': {ex}\n{tb_str}")
                    try:
                        err_log = ExecutionLogModel(
                            id=f"err_{th_id}",
                            thread_id=th_id,
                            node_id="webhook_trigger",
                            workflow_id=wf_id,
                            input_data=redact_sensitive_data(_ensure_serializable(ctx)),
                            output_data={"error": sanitize_for_logging(str(ex))},
                            latency_ms=0.0,
                            status="error",
                            error_message=sanitize_for_logging(f"{ex}\n{tb_str}")
                        )
                        await session.merge(err_log)
                        await session.commit()
                    except Exception as log_ex:
                        logger.error(f"Failed to record webhook trigger error to DB: {log_ex}")

        bg_tasks.add_task(
            _run_webhook_async, 
            wf_id=target_wf.id, 
            inp_text=f"Webhook {request.method} trigger: {path}", 
            th_id=thread_id, 
            ctx=context
        )
        
        return {
            "status": "triggered", 
            "workflow_id": target_wf.id, 
            "thread_id": thread_id,
            "message": f"Workflow '{target_wf.name}' initiated via webhook"
        }

# --- Execution Logs & History ---

@router.get("/workflow/{workflow_id}/threads", response_model=ThreadListResponse)
async def list_threads_endpoint(
    workflow_id: str,
    page: int = 1,
    limit: int = 20,
    db: AsyncSession = Depends(get_db)
):
    """Lists history of threads for a workflow."""
    skip = (page - 1) * limit
    threads = await AgentMeshCrud.list_workflow_threads(db, workflow_id, skip=skip, limit=limit)
    return {"status": "success", "threads": threads}

@router.get("/workflow/{workflow_id}/thread/{thread_id}", response_model=ThreadHistoryResponse)
async def get_thread_history_endpoint(workflow_id: str, thread_id: str, db: AsyncSession = Depends(get_db)):
    """Fetch paused state from checkpoints or completed state from compact history."""
    from Apps.AgentMesh.Utilities.Serialization.message_helper import serialize_messages
    from Apps.AgentMesh.Core.Services.WorkflowHistoryService import WorkflowHistoryService
    
    app = await workflow_registry.get_executable_graph(workflow_id, db=db)
    if not app:
        raise HTTPException(status_code=404, detail="Workflow not found or not compiled.")
    
    run_config = {"configurable": {"thread_id": thread_id}}
    state_snapshot = await app.aget_state(run_config)

    if not state_snapshot.values:
        history = await WorkflowHistoryService.get_history(
            db,
            thread_id=thread_id,
            workflow_id=workflow_id,
        )
        if history is not None:
            return {
                "status": "success",
                "thread_id": thread_id,
                "messages": history.messages or [],
                "is_paused": False,
                "next_node": None,
                "active_agent": history.active_agent,
                "context": history.context or {},
                "pause_type": None,
                "pause_config": None,
            }

    # Redact muted node messages for security
    messages = state_snapshot.values.get("messages", [])
    active_agent = state_snapshot.values.get("active_agent")
    context = state_snapshot.values.get("context", {})

    config_obj = workflow_registry.get_config(workflow_id)

    # Resolve pause metadata (interrupt/gate-based) so the frontend can restore
    # custom HITL modals (followup_form/followup_selection) on session reopen.
    is_paused, _, next_node, pause_type, pause_config, _ = (
        WorkflowService._snapshot_pause_info(state_snapshot, config_obj, messages=messages)
    )

    muted_nodes = {n.id for n in config_obj.nodes if getattr(n, "mute_chat_bubble", False)} if config_obj else set()

    serialized = serialize_messages(messages)
    for msg in serialized:
        node_id = msg.get("metadata", {}).get("active_agent")
        if node_id in muted_nodes:
            msg["content"] = "[Internal Execution Content]"
            if "tool_calls" in msg.get("metadata", {}):
                msg["metadata"]["tool_calls"] = []

    return {
        "status": "success",
        "thread_id": thread_id,
        "messages": serialized,
        "is_paused": is_paused,
        "next_node": next_node,
        "active_agent": active_agent,
        "context": context,
        "pause_type": pause_type,
        "pause_config": pause_config
    }

@router.delete("/thread/{thread_id}")
async def delete_thread_endpoint(thread_id: str, db: AsyncSession = Depends(get_db)):
    """Deletes a thread and its logs."""
    from Apps.AgentMesh.Utilities.Memory.checkpointer import get_checkpointer

    await get_checkpointer(thread_id).adelete_thread(thread_id)
    await AgentMeshCrud.delete_workflow_thread(db, thread_id)
    return {"status": "success", "message": f"Thread {thread_id} deleted."}

@router.get("/workflow/thread/{thread_id}/blackboard")
async def get_blackboard_state_endpoint(thread_id: str):
    """Fetches the current blackboard state (tasks list) for a thread."""
    from Apps.AgentMesh.Utilities.State.ToolStateManager import ToolStateManager
    from Apps.AgentMesh.Schema.BlackboardSchema import BlackboardState
    
    raw = await ToolStateManager.get_state(thread_id, "agent_blackboard")
    state = BlackboardState(**raw) if raw else BlackboardState(pad_id=thread_id, tasks=[])
    
    return {
        "status": "success",
        "thread_id": thread_id,
        "blackboard": state.model_dump()
    }

@router.get("/active-runs")
async def get_active_runs_endpoint(since: int = 0):
    """
    Returns currently running nodes across all workflows.

    `since` enables cheap catch-up for WebSocket reconnect reconciliation:
    if no registry mutation happened since `since`, returns changed=False with
    an empty body so clients skip re-serializing the full registry.
    """
    current_revision = active_runs_revision
    if since >= current_revision:
        return {"status": "success", "revision": current_revision, "changed": False, "active_runs": {}}
    return {
        "status": "success",
        "revision": current_revision,
        "changed": True,
        "active_runs": active_runs_registry,
    }


@router.get("/workflow/{workflow_id}/api-spec", tags=["API Console"])
async def get_workflow_api_spec(workflow_id: str, request: Request, db: AsyncSession = Depends(get_db)):
    """
    Auto-generate an OpenAPI-compatible spec for a workflow's webhook API.
    Used by the frontend WorkflowAPIConsole to render Swagger-style docs.
    """
    w = await AgentMeshCrud.get_workflow(db, workflow_id, reveal_secrets=True)
    if not w:
        raise HTTPException(status_code=404, detail="Workflow not found")

    # Get the compiled config from registry for trigger details
    config = workflow_registry.get_config(workflow_id)
    if not config:
        # Fallback: try to parse from DB
        try:
            from Apps.AgentMesh.Schema.WorkflowSchema import WorkflowConfig
            config = WorkflowConfig(**w.config)
        except Exception:
            raise HTTPException(status_code=404, detail="Workflow config not available")

    webhook = config.triggers.webhook if config.triggers else None
    webhook_path = webhook.path if webhook and webhook.path else workflow_id
    base_url = str(request.base_url).rstrip("/")
    # Derive the mesh API prefix from this router's own path
    webhook_url = f"{base_url}/api/mesh/webhook/{webhook_path}"

    auth_method = webhook.auth_method if webhook else "none"
    response_mode = webhook.response_mode if webhook else "immediate"
    has_secret = bool(webhook and webhook.secret)

    # Detect node types for capability hints
    node_types = [n.type for n in config.nodes] if config.nodes else []
    accepts_files = any(t in node_types for t in ["ocr_node", "ingestion_node", "vector_ingestion"])
    has_hitl = any(getattr(n, "interrupt_before", False) or getattr(n, "interrupt_after", False) for n in (config.nodes or []))
    has_db = "db_node" in node_types

    # Build request schema
    request_schema = {
        "type": "object",
        "properties": {
            "payload": {
                "type": "object",
                "description": "JSON payload passed to the workflow as context.payload"
            }
        }
    }
    if accepts_files:
        request_schema["properties"]["file"] = {
            "type": "string",
            "format": "binary",
            "description": "File upload (PDF, image, etc.) — use multipart/form-data"
        }

    # Build response schema
    response_schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "example": "triggered" if response_mode == "immediate" else "completed"},
            "workflow_id": {"type": "string", "example": workflow_id},
            "thread_id": {"type": "string", "example": "webhook_20260604_120000_a1b2"}
        }
    }

    # Auth section
    auth_info = {}
    if has_secret:
        if auth_method == "hmac":
            auth_info = {
                "type": "hmac",
                "header": "X-AgentMesh-Signature",
                "algorithm": "SHA256",
                "description": "HMAC-SHA256 of the raw request body using the workflow secret"
            }
        else:
            auth_info = {
                "type": "header",
                "header": "X-AgentMesh-Secret",
                "description": "Static secret key sent in the header"
            }

    # HITL resume endpoint info
    resume_info = None
    if has_hitl:
        resume_info = {
            "url": f"{base_url}/api/mesh/workflow/{workflow_id}/execute",
            "method": "POST",
            "description": "Resume a paused workflow (e.g., after PII verification)",
            "body_schema": {
                "mode": "resume",
                "thread_id": "<thread_id from trigger response>",
                "action": "approve",
                "payload": {}
            }
        }

    # cURL snippet
    curl_parts = [f'curl -X POST "{webhook_url}"']
    curl_parts.append('  -H "Content-Type: application/json"')
    if has_secret and auth_method == "header":
        curl_parts.append('  -H "X-AgentMesh-Secret: <your-secret>"')
    elif has_secret and auth_method == "hmac":
        curl_parts.append('  -H "X-AgentMesh-Signature: <hmac-sha256-hex>"')
    curl_parts.append('  -d \'{"key": "value"}\'')
    curl_snippet = " \\\n".join(curl_parts)

    return {
        "status": "success",
        "spec": {
            "workflow_id": workflow_id,
            "workflow_name": config.name,
            "description": config.description or f"API for {config.name}",
            "endpoint": {
                "url": webhook_url,
                "method": "POST",
                "content_types": ["application/json"] + (["multipart/form-data"] if accepts_files else []),
                "response_mode": response_mode
            },
            "authentication": auth_info,
            "request_schema": request_schema,
            "response_schema": response_schema,
            "resume_endpoint": resume_info,
            "capabilities": {
                "accepts_files": accepts_files,
                "has_hitl": has_hitl,
                "has_database": has_db,
                "node_count": len(config.nodes) if config.nodes else 0
            },
            "curl_example": curl_snippet
        }
    }

@router.post("/execution/upload-artifact")
async def upload_artifact_file(
    file: UploadFile = File(...),
    thread_id: str = Form("default_thread"),
    workflow_id: str | None = Form(None),
    db: AsyncSession = Depends(get_db),
):
    """
    Saves a file in artifact storage and returns a lightweight reference.
    """
    try:
        artifact_reference = await ArtifactService.save_upload(
            file=file,
            thread_id=thread_id,
            workflow_id=workflow_id,
            db=db,
        )

        logger.info(
            "Artifact upload completed: %s",
            artifact_reference["artifact_id"],
        )

        return {
            "status": "success",
            **artifact_reference,
        }

    except Exception as error:
        logger.exception("Artifact upload failed")
        raise HTTPException(
            status_code=500,
            detail=f"Artifact upload failed: {error}",
        )

@router.post("/execution/upload-temp")
async def upload_temp_file(
    file: UploadFile = File(...),
    thread_id: str = Form("default_thread")
):
    """Saves a temporary file for the current session/thread."""
    try:
        from Apps.AgentMesh.Config import settings
        from pathvalidate import sanitize_filename
        import os

        # 1. Resolve and create sessional directory
        session_dir = os.path.abspath(os.path.join(settings.TEMP_MOUNT, thread_id))
        os.makedirs(session_dir, exist_ok=True)

        # 2. Sanitize filename
        safe_name = sanitize_filename(file.filename or "upload")
        filepath = os.path.join(session_dir, safe_name)

        # 3. Save file contents in memory-safe chunks
        file_size = 0
        with open(filepath, "wb") as buffer:
            while True:
                chunk = await file.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break
                buffer.write(chunk)
                file_size += len(chunk)

        # 4. Construct public URL (relative for simplicity/flexibility)
        public_url = f"/temp/{thread_id}/{safe_name}"

        logger.info(f"Sessional upload success: {safe_name} saved to {filepath}")
        return {
            "status": "success",
            "filename": safe_name,
            "vfs_path": filepath,
            "url": public_url,
            "size_bytes": file_size,
            "mime_type": file.content_type
        }
    except Exception as e:
        logger.error(f"Sessional upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

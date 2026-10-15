import os
import asyncio
import logging
import uuid
from typing import List, Dict, Any
from langchain_core.messages import AIMessage
from Apps.AgentMesh.Schema.StateSchema import AgentState
from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig

logger = logging.getLogger(__name__)

def build_ingestion_router_node(node_config: NodeConfig):
    async def process_ingestion(state: AgentState):
        context = state.get("context", {})
        payload = context.get("payload", {})
        
        scan_mode = getattr(node_config, "scan_mode", "auto")
        file_filters = getattr(node_config, "file_filters", [".pdf", ".jpg", ".png", ".jpeg", ".json"])
        concurrency_limit = getattr(node_config, "concurrency_limit", 2)
        
        parent_folder_key = getattr(node_config, "parent_folder_key", "parent_folder_path")
        folder_identifier_key = getattr(node_config, "folder_identifier_key", "customer_folder_name")
        curl_command_key = getattr(node_config, "curl_command_key", "curl_command")
        
        parent_path = payload.get(parent_folder_key)
        
        # Normalize path: strip quotes, whitespace, and handle Windows backslash escaping
        if parent_path and isinstance(parent_path, str):
            parent_path = parent_path.strip().strip('"').strip("'")
            parent_path = os.path.normpath(parent_path)
            logger.info(f"[Ingestion Router] {parent_folder_key} resolved to: '{parent_path}' | exists={os.path.exists(parent_path)} | isdir={os.path.isdir(parent_path)}")

        # STAGE 3: Macro-Bulk (Parent Directory)
        if parent_path and os.path.isdir(parent_path):
            logger.info(f"[Ingestion Router] Detected directory: {parent_path}. Starting bulk dispatch.")
            
            # Find subfolders
            subfolders = [os.path.join(parent_path, d) for d in os.listdir(parent_path) if os.path.isdir(os.path.join(parent_path, d))]
            
            sem = asyncio.Semaphore(concurrency_limit)
            from Apps.AgentMesh.Core.Services.WorkflowService import WorkflowService
            from Database.Database import AsyncSessionLocal
            
            current_workflow_id = context.get("workflow_id")
            if not current_workflow_id:
                logger.error("[Ingestion Router] Missing workflow_id in context. Cannot spawn children.")
                context["is_bulk_parent"] = False
                return {"context": context}

            async def dispatch_child(folder_path):
                async with sem:
                    # Find valid files
                    valid_files = []
                    for f in os.listdir(folder_path):
                        if any(f.lower().endswith(ext) for ext in file_filters):
                            valid_files.append({"path": os.path.join(folder_path, f), "filename": f})
                    
                    if not valid_files:
                        logger.warning(f"[Ingestion Router] No valid files found in {folder_path}. Skipping.")
                        return
                    
                    batch_name = os.path.basename(folder_path)
                    child_payload = {"uploaded_files": valid_files, folder_identifier_key: batch_name}
                    
                    # Preserve original uid/query params from webhook
                    child_context = {
                        "source": "bulk_spawner", 
                        "payload": child_payload, 
                        "query_params": context.get("query_params", {})
                    }
                    
                    child_thread_id = f"child_{uuid.uuid4().hex[:8]}_{batch_name.replace(' ', '_')}"
                    logger.info(f"[Ingestion Router] Spawning child thread {child_thread_id} for item {batch_name}")
                    
                    try:
                        async with AsyncSessionLocal() as db:
                            await WorkflowService.run(
                                workflow_id=current_workflow_id,
                                input_text=f"Auto-triggered bulk ingestion for {batch_name}",
                                thread_id=child_thread_id,
                                context=child_context,
                                db=db
                            )
                    except Exception as e:
                        logger.error(f"[Ingestion Router] Failed to spawn child workflow for {batch_name}: {e}")

            # Fire them all off concurrently, protected by semaphore
            await asyncio.gather(*(dispatch_child(f) for f in subfolders))
            
            context["is_bulk_parent"] = True
            return {"context": context, "messages": [AIMessage(content=f"Dispatched {len(subfolders)} child workflows.")]}

        # Check for JSON files in uploaded_files
        uploaded_files = payload.get("uploaded_files", [])
        is_json_files = False
        if isinstance(uploaded_files, list) and len(uploaded_files) > 0:
            if all(f.get("filename", "").lower().endswith(".json") for f in uploaded_files):
                is_json_files = True

        # STAGE 4: JSON Data Processing (cURL or uploaded JSON files)
        # This will extract all items and spawn independent child threads for each
        if payload.get(curl_command_key) or is_json_files:
            logger.info(f"[Ingestion Router] Detected JSON data source (cURL={bool(payload.get(curl_command_key))}, JSON_Files={is_json_files}). Extracting items.")
            
            all_items = []
            
            # Extract from cURL
            if payload.get(curl_command_key):
                try:
                    from Apps.AgentMesh.Core.Compiler.nodes.api_node import parse_curl_command
                    import httpx
                    import json
                    parsed_curl = parse_curl_command(payload.get(curl_command_key))
                    url = parsed_curl["url"]
                    logger.info(f"[Ingestion Router] Executing cURL to {url}")
                    
                    with httpx.Client(timeout=30.0) as client:
                        resp = client.request(
                            method=parsed_curl["method"],
                            url=url,
                            headers=parsed_curl["headers"],
                            content=parsed_curl["data"]
                        )
                        resp.raise_for_status()
                        json_data = resp.json()
                        
                        if isinstance(json_data, list):
                            all_items.extend(json_data)
                        elif isinstance(json_data, dict):
                            all_items.append(json_data)
                except Exception as e:
                    logger.error(f"[Ingestion Router] Failed to fetch JSON from cURL: {e}")
            
            # Extract from JSON Files
            if is_json_files:
                import json
                for f in uploaded_files:
                    try:
                        with open(f.get("path"), "r", encoding="utf-8") as file_obj:
                            json_data = json.load(file_obj)
                            if isinstance(json_data, list):
                                all_items.extend(json_data)
                            elif isinstance(json_data, dict):
                                all_items.append(json_data)
                    except Exception as e:
                        logger.error(f"[Ingestion Router] Failed to parse JSON file {f.get('filename')}: {e}")
            
            # Spawn threads for each item
            if all_items:
                logger.info(f"[Ingestion Router] Extracted {len(all_items)} items from JSON. Spawning child workflows.")
                sem = asyncio.Semaphore(concurrency_limit)
                from Apps.AgentMesh.Core.Services.WorkflowService import WorkflowService
                from Database.Database import AsyncSessionLocal
                
                current_workflow_id = context.get("workflow_id")
                if not current_workflow_id:
                    logger.error("[Ingestion Router] Missing workflow_id. Cannot spawn children.")
                    context["is_bulk_parent"] = False
                    return {"context": context}
                
                async def dispatch_json_child(item_data, idx):
                    async with sem:
                        item_id = item_data.get("customer_id") or item_data.get("id") or f"item_{idx}"
                        child_payload = {"json_data": item_data, folder_identifier_key: str(item_id)}
                        
                        child_context = {
                            "source": "bulk_spawner_json",
                            "payload": child_payload,
                            "query_params": context.get("query_params", {})
                        }
                        
                        child_thread_id = f"child_json_{uuid.uuid4().hex[:8]}_{str(item_id).replace(' ', '_')}"
                        logger.info(f"[Ingestion Router] Spawning JSON child thread {child_thread_id} for {item_id}")
                        
                        try:
                            async with AsyncSessionLocal() as db:
                                await WorkflowService.run(
                                    workflow_id=current_workflow_id,
                                    input_text=f"Auto-triggered JSON ingestion for {item_id}",
                                    thread_id=child_thread_id,
                                    context=child_context,
                                    db=db
                                )
                        except Exception as e:
                            logger.error(f"[Ingestion Router] Failed to spawn JSON workflow for {item_id}: {e}")

                await asyncio.gather(*(dispatch_json_child(item, i) for i, item in enumerate(all_items)))
                
                context["is_bulk_parent"] = True
                
                # Clear trigger fields from parent payload so it doesn't execute the actual pipeline
                if "uploaded_files" in payload:
                    payload["uploaded_files"] = []
                if curl_command_key in payload:
                    payload[curl_command_key] = None
                    
                return {"context": context, "messages": [AIMessage(content=f"Dispatched {len(all_items)} JSON child workflows.")]}
            else:
                logger.warning("[Ingestion Router] No items extracted from JSON sources.")
                context["is_bulk_parent"] = False
                return {"context": context}

        # STAGE 2: Micro-Bulk (Array of files, non-JSON)
        elif payload.get("uploaded_files") and isinstance(payload.get("uploaded_files"), list):
            logger.info("[Ingestion Router] Detected multiple files for single item (Stage 2).")
            context["is_bulk_parent"] = False
            return {"context": context}
            
        # STAGE 1: Single File Upload
        elif payload.get("file_path") or payload.get("files"):
            logger.info("[Ingestion Router] Detected single file (Stage 1). Normalizing array format.")
            fpath = payload.get("file_path") or payload.get("files")
            fname = os.path.basename(fpath) if isinstance(fpath, str) else "Unknown"
            payload["uploaded_files"] = [{"path": fpath, "filename": fname}]
            context["payload"] = payload
            context["is_bulk_parent"] = False
            return {"context": context}
            
        # STAGE 4: JSON Child Workflow
        elif payload.get("json_data") is not None:
            logger.info("[Ingestion Router] Detected JSON child workflow. Routing to JSON pipeline.")
            context["payload"] = payload
            context["is_bulk_parent"] = False
            return {"context": context}
            
        else:
            logger.warning("[Ingestion Router] Unknown payload format or no files found.")
            context["is_bulk_parent"] = False
            return {"context": context}

    return process_ingestion

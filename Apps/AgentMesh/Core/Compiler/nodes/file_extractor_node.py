"""
File Extractor Node Builder
Reads data from a specified file path and injects it into the workflow state context.
"""

import os
import json
import base64
import logging
from typing import Callable

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from Apps.AgentMesh.Schema.StateSchema import AgentState
from .utils import interpolate

logger = logging.getLogger(__name__)

def build_file_extractor_node(node_config: NodeConfig) -> Callable:
    """
    Build a File Extractor execution node.
    Extracts data from a file on disk and places it into the specified context key.
    """
    async def execute_file_extractor(state: AgentState):
        context = state.get("context", {})
        
        logger.info(f"File Extractor Node {node_config.id}: Attempting to read file")
        
        # 1. Resolve File Path
        raw_path = getattr(node_config, "extract_file_path", None) or "{{context.latest_file_path}}"
        logger.info(f"[DEBUG_FILE_EXTRACTOR] node_config dictionary: {node_config.model_dump(exclude_unset=True)}")
        logger.info(f"[DEBUG_FILE_EXTRACTOR] raw_path: {raw_path}")
        
        # FIX: The template uses {{context.latest_file_path}}, so we must expose a 'context' key in scope.
        # Also spread context at the top level so {{latest_file_path}} works too.
        jinja_scope = {**context, "context": context}
        filepath = interpolate(raw_path, jinja_scope)
        logger.info(f"[DEBUG_FILE_EXTRACTOR] interpolated filepath: {filepath}")
        
        if not filepath or filepath == "{{context.latest_file_path}}":
            filepath = context.get("latest_file_path", "")
            
        # 1.5 Fallback to VFS if no filepath is found in context
        if not filepath or not os.path.exists(filepath):
            vfs = state.get("vfs", {})
            vfs_files = vfs.get("files", {})
            if vfs_files:
                latest_file_id = list(vfs_files.keys())[-1]
                file_obj = vfs_files[latest_file_id]
                if isinstance(file_obj, dict) and file_obj.get("path"):
                    filepath = file_obj.get("path")
                    logger.info(f"File Extractor Node {node_config.id}: Found file in VFS: {filepath}")

        # 1.6 Safe fallback to thread-specific directory
        if not filepath or not os.path.exists(filepath):
            thread_id = context.get("thread_id") or state.get("context", {}).get("thread_id")
            if thread_id:
                from Apps.AgentMesh.Config import settings
                session_dir = os.path.abspath(os.path.join(settings.TEMP_MOUNT, thread_id))
                if os.path.exists(session_dir):
                    candidates = []
                    for root, dirs, files in os.walk(session_dir):
                        if 'skills' in dirs:
                            dirs.remove('skills')
                        for f in files:
                            candidates.append(os.path.join(root, f))
                    if candidates:
                        candidates.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                        filepath = candidates[0]
                        logger.info(f"File Extractor Node {node_config.id}: Falling back to thread-specific upload: {filepath}")

        if not filepath or not os.path.exists(filepath):
            logger.warning(f"File Extractor Node {node_config.id}: No file found - skipping gracefully. The downstream agent will handle this via its tools.")
            # Graceful skip: don't inject a noisy error into the conversation.
            # The Deep Agent's system prompt already instructs it to call tools
            # with an empty file_path when no file is available.
            updated_ctx = {**context, "extracted_file_path": ""}
            # Downstream Jinja2 templates (e.g. in the DeepAgent prompt) use {{context.X}}.
            # interpolate() is called as interpolate(prompt, context_dict), so 'context' must
            # be a key inside context_dict that resolves to context values.
            updated_ctx["context"] = {k: v for k, v in updated_ctx.items() if k != "context"}
            return {
                "messages": [],
                "active_agent": node_config.id,
                "context": updated_ctx,
            }
        
        # 2. Extract Content
        extract_format = (node_config.extract_format or "text").lower()
        extracted_data = None
        status_msg = ""
        
        try:
            if extract_format == "json":
                with open(filepath, 'r', encoding='utf-8') as f:
                    extracted_data = json.load(f)
                status_msg = f"Successfully extracted JSON from {filepath}"
            elif extract_format == "base64":
                with open(filepath, 'rb') as f:
                    extracted_data = base64.b64encode(f.read()).decode('utf-8')
                status_msg = f"Successfully extracted Base64 data from {filepath}"
            else:
                # Text format extraction
                if filepath.lower().endswith('.csv'):
                    import csv
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        reader = csv.reader(f)
                        headers = next(reader, [])
                        row_count = sum(1 for row in reader)
                    extracted_data = (
                        f"[FILE SUMMARY ONLY - SAVING LLM CREDITS]\n"
                        f"File path: {filepath}\n"
                        f"Type: CSV Data\n"
                        f"Total Rows: {row_count}\n"
                        f"Columns: {', '.join(headers)}\n\n"
                        f"Note: The raw CSV content has been hidden to save API tokens. To extract specific data, please use your tools and pass the file path."
                    )
                    status_msg = f"Successfully extracted lightweight CSV summary from {filepath}"
                elif filepath.lower().endswith(('.xlsx', '.xls', '.pdf', '.png', '.jpg')):
                    # Prevent UnicodeDecodeError by not reading binary files as raw text
                    extracted_data = f"[BINARY FILE] Path: {filepath}. Use specialized tools to read this file format."
                    status_msg = f"Skipped raw text extraction for binary file: {filepath}"
                else:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        extracted_data = f.read()
                    status_msg = f"Successfully extracted text from {filepath}"

            # Update context
            output_key = node_config.extract_output_key or "extracted_file_content"
            context[output_key] = extracted_data
            context["extracted_file_path"] = filepath
            logger.info(f"File Extractor Node {node_config.id}: {status_msg}")
            
        except Exception as e:
            error_msg = f"Error reading file {filepath}: {str(e)}"
            logger.error(f"File Extractor Node {node_config.id}: {error_msg}")
            
            if node_config.extract_skip_errors:
                return {
                    "messages": [AIMessage(content=f"Skipped extraction: {error_msg}", response_metadata={"active_agent": node_config.id})],
                    "active_agent": node_config.id
                }
            else:
                return {
                    "messages": [AIMessage(content=f"Failed to read file: {error_msg}", response_metadata={"active_agent": node_config.id})],
                    "active_agent": node_config.id,
                    "error": {"message": error_msg, "type": type(e).__name__},
                    "error_source": node_config.id
                }
                
        # 3. Return updates
        output_key = node_config.extract_output_key or "extracted_file_content"
        updated_ctx = {
            **context,
            output_key: extracted_data,
            "extracted_file_path": filepath,
        }
        # Downstream Jinja2 templates (e.g. in the DeepAgent prompt) use {{context.X}}.
        # interpolate() is called as interpolate(prompt, context_dict), so 'context' must
        # be a key inside context_dict that resolves to context values.
        updated_ctx["context"] = {k: v for k, v in updated_ctx.items() if k != "context"}
        return {
            "messages": [AIMessage(content=status_msg, response_metadata={"active_agent": node_config.id})],
            "active_agent": node_config.id,
            "context": updated_ctx,
        }

    return RunnableLambda(execute_file_extractor).with_retry(stop_after_attempt=3)

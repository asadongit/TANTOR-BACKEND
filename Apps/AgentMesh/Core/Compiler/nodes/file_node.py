"""
File Store Node Builder
Handles saving data from the workflow state/context into CSV, Excel, plain
text or Markdown files.
"""

import os
import json
import logging
import pandas as pd
from typing import Callable, Any, List, Dict

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from Apps.AgentMesh.Schema.StateSchema import AgentState
from Apps.AgentMesh.Config import settings
from .utils import interpolate

logger = logging.getLogger(__name__)

# A document read off a page arrives as text - headings, then a table laid out
# in pipes and line breaks. Some steps hand that text on bare; others carry it
# inside a result object alongside a class, a confidence and so on. Saving the
# second kind as JSON would escape every line break and leave the table on one
# line, so when Markdown is asked for, the text is taken back out of whatever
# is carrying it.
_TEXT_KEYS = ("markdown", "md", "text", "content", "ocr", "vision_result",
              "latest_ocr_output", "raw_text", "output", "result", "summary")


def _looks_like_a_document(value: Any) -> bool:
    """Is this string a laid-out document rather than a stray label?"""
    return isinstance(value, str) and ("\n" in value or "|" in value) and len(value) > 40


def _as_markdown(value: Any) -> str:
    """The document inside `value`, as Markdown, without escaping it.

    Falls back to a fenced code block rather than raw JSON: the file is asked
    to be Markdown, and a fenced block is both valid Markdown and lossless, so
    nothing is thrown away when the value turns out not to be a document.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        # a field that names itself as the text wins over one that merely looks
        # like it, so an explicit "markdown" key is never passed over
        for key in _TEXT_KEYS:
            if _looks_like_a_document(value.get(key)):
                return value[key]
        documents = [v for v in value.values() if _looks_like_a_document(v)]
        if len(documents) == 1:
            return documents[0]
        if documents:
            # more than one: keep them all, in the order the producer wrote them
            return "\n\n".join(documents)
    if isinstance(value, list):
        parts = [_as_markdown(v) for v in value]
        if all(isinstance(v, str) for v in value) or any(
                _looks_like_a_document(v) for v in value):
            return "\n\n".join(p for p in parts if p)
    if value is None:
        return ""
    return "```json\n" + json.dumps(value, indent=2, default=str) + "\n```"


def build_file_store_node(node_config: NodeConfig) -> Callable:
    """
    Build a File Store execution node.
    Extracts data from the workflow context and saves it to a specified file format.
    """
    async def execute_file_store(state: AgentState):
        context = state.get("context", {})
        messages = state.get("messages", [])
        
        logger.info(f"File Store Node {node_config.id}: Saving data to {node_config.file_format}")

        # 1. Determine data to save
        # Priority: file_data_key -> latest_api_result -> last message content
        data_to_save = None
        
        if node_config.file_data_key:
            data_to_save = context.get(node_config.file_data_key)
        
        if data_to_save is None:
            data_to_save = context.get("latest_api_result")
        
        if data_to_save is None and messages:
            last_msg = messages[-1]
            if isinstance(last_msg, AIMessage):
                content = last_msg.content
                # If it's a markdown JSON block, extract it
                if "```json" in content:
                    try:
                        json_str = content.split("```json")[1].split("```")[0].strip()
                        data_to_save = json.loads(json_str)
                    except:
                        data_to_save = content
                else:
                    # Try to parse raw content as JSON
                    try:
                        data_to_save = json.loads(content)
                    except:
                        data_to_save = content

        # 2. Normalize data to a list of dicts (for CSV/Excel)
        normalized_data = []
        if isinstance(data_to_save, list):
            for item in data_to_save:
                if isinstance(item, dict):
                    normalized_data.append(item)
                else:
                    normalized_data.append({"value": str(item)})
        elif isinstance(data_to_save, dict):
            normalized_data = [data_to_save]
        elif data_to_save is not None:
            normalized_data = [{"content": str(data_to_save)}]
        else:
            normalized_data = [{"error": "No data found to save"}]

        # 3. Setup File Path
        thread_id = context.get('thread_id', 'global')
        raw_filename = node_config.file_name or f"export_{node_config.id}"
        filename = interpolate(raw_filename, context)
        
        # Ensure correct extension
        format_lower = node_config.file_format.lower()
        if format_lower == "csv":
            ext = ".csv"
        elif format_lower == "excel" or format_lower == "xlsx":
            ext = ".xlsx"
        elif format_lower in ("md", "markdown"):
            ext = ".md"
        else:
            ext = ".txt"

        if not filename.lower().endswith(ext):
            filename += ext
            
        # ✅ THE FIX: Isolate by thread_id
        storage_dir = os.path.abspath(os.path.join(settings.EXPORTS_MOUNT, thread_id))
        os.makedirs(storage_dir, exist_ok=True)
        filepath = os.path.join(storage_dir, filename)

        # 4. Save logic
        try:
            if format_lower == "csv":
                df = pd.DataFrame(normalized_data)
                mode = 'w' if node_config.file_overwrite else 'a'
                header = node_config.file_overwrite or not os.path.exists(filepath)
                df.to_csv(filepath, index=False, mode=mode, header=header)
            elif format_lower in ["excel", "xlsx"]:
                df = pd.DataFrame(normalized_data)
                df.to_excel(filepath, index=False)
            elif format_lower in ("md", "markdown"):
                # Markdown is written exactly as it was produced.
                #
                # The plain-text branch below turns a dict or a list into JSON,
                # which escapes every newline to a literal \n and collapses a
                # laid-out table into one unreadable line. That is the right
                # thing for a record and the wrong thing for a document, so
                # markdown takes the text out of the wrapper instead.
                mode = 'w' if node_config.file_overwrite else 'a'
                with open(filepath, mode, encoding='utf-8') as f:
                    f.write(_as_markdown(data_to_save))
                    f.write("\n")
            else:
                # Text format
                mode = 'w' if node_config.file_overwrite else 'a'
                with open(filepath, mode, encoding='utf-8') as f:
                    if isinstance(data_to_save, (dict, list)):
                        f.write(json.dumps(data_to_save, indent=2))
                    else:
                        f.write(str(data_to_save))
                    f.write("\n")
            
            # Generate public URL for easier access (including thread path)
            public_url = f"http://localhost:8000/exports/{thread_id}/{filename}"
            status_msg = f"Successfully saved {len(normalized_data)} records to [{filename}]({public_url})"
            logger.info(f"File Store Node {node_config.id}: {status_msg}")
            
        except Exception as e:
            status_msg = f"Failed to save file: {str(e)}"
            logger.error(f"File Store Node {node_config.id} Error: {str(e)}")
            # Populate system error
            error_data = {"message": str(e), "type": type(e).__name__}
            return {
                "messages": [AIMessage(content=status_msg, response_metadata={"active_agent": node_config.id})],
                "active_agent": node_config.id,
                "error": error_data,
                "error_source": node_config.id
            }

        # 5. Return updates
        return {
            "messages": [AIMessage(content=status_msg, response_metadata={"active_agent": node_config.id})],
            "active_agent": node_config.id,
            "context": {
                **context,
                f"file_path_{node_config.id}": filepath,
                f"public_url_{node_config.id}": public_url,
                "latest_file_path": filepath,
                "latest_public_url": public_url
            }
        }

    return RunnableLambda(execute_file_store).with_retry(stop_after_attempt=3)

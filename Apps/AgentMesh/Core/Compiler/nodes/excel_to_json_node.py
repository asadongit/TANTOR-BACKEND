"""
Excel to JSON Node Builder
Converts Excel (.xlsx, .xls) and CSV files into structured JSON data in workflow context.
Supports artifact:// path resolution for uploaded files from Form Nodes.
"""

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from langchain_core.messages import AIMessage

from Apps.AgentMesh.Schema.StateSchema import AgentState
from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig

logger = logging.getLogger(__name__)


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively replaces NaN, Infinity, -Infinity, and Timestamps with JSON-serializable types."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(item) for item in obj]
    elif hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj


def _extract_file_path(raw_value: Any) -> Optional[str]:
    """Helper to extract a clean string file path or artifact:// URI from diverse context representations."""
    if not raw_value:
        return None

    if isinstance(raw_value, str):
        val = raw_value.strip()
        if (val.startswith("{") and val.endswith("}")) or (val.startswith("[") and val.endswith("]")):
            try:
                parsed = json.loads(val)
                return _extract_file_path(parsed)
            except Exception:
                pass
        return val

    if isinstance(raw_value, dict):
        if "artifact_id" in raw_value and raw_value["artifact_id"]:
            return f"artifact://{raw_value['artifact_id']}"
        for key in ("file_path", "filepath", "path", "saved_path", "stored_path", "local_path", "file", "url"):
            val = raw_value.get(key)
            if val and isinstance(val, str):
                return val.strip()
        return None

    if isinstance(raw_value, list) and len(raw_value) > 0:
        return _extract_file_path(raw_value[0])

    return None


def build_excel_to_json_node(node_config: NodeConfig) -> Callable:
    """
    Build an Excel to JSON execution node.
    Reads an Excel/CSV file from context and exports structured JSON into state context.
    """
    input_key = getattr(node_config, "excel_input_key", None)
    output_key = getattr(node_config, "excel_output_key", "excel_json_data") or "excel_json_data"
    sample_output_key = getattr(node_config, "excel_sample_output_key", None) or getattr(node_config, "sample_output_key", "excel_sample_data") or "excel_sample_data"
    sample_limit = getattr(node_config, "excel_sample_limit", None) or getattr(node_config, "sample_limit", 5)
    sample_method = str(getattr(node_config, "excel_sample_method", "first") or "first").strip().lower()
    if sample_method not in ("first", "random"):
        sample_method = "first"
    sheet_name = getattr(node_config, "excel_sheet_name", None)
    row_limit = getattr(node_config, "excel_row_limit", None)
    orientation = getattr(node_config, "excel_orientation", "records") or "records"
    headers_only = getattr(node_config, "excel_headers_only", False)
    handle_nulls = getattr(node_config, "excel_handle_nulls", "none") or "none"

    if sample_limit is not None and str(sample_limit).strip():
        try:
            sample_limit = int(float(str(sample_limit).strip()))
            if sample_limit <= 0:
                sample_limit = 5
        except (ValueError, TypeError):
            sample_limit = 5
    else:
        sample_limit = 5

    if row_limit is not None and str(row_limit).strip():
        try:
            row_limit = int(float(str(row_limit).strip()))
            # 0, negative, or legacy default of 10 on excel_to_json_parser means NO limit (process entire file)
            if row_limit <= 0 or (row_limit == 10 and getattr(node_config, "id", "") == "excel_to_json_parser"):
                row_limit = None
        except (ValueError, TypeError):
            row_limit = None
    else:
        row_limit = None

    if sheet_name is not None and str(sheet_name).strip():
        sheet_str = str(sheet_name).strip()
        if sheet_str.isdigit():
            sheet_name = int(sheet_str)
        else:
            sheet_name = sheet_str
    else:
        sheet_name = 0

    async def excel_to_json_node(state: AgentState) -> Dict[str, Any]:
        logger.info(
            f"Excel to JSON Node {node_config.id}: processing file into '{output_key}' "
            f"(row_limit={row_limit}, orientation={orientation})"
        )

        context: Dict[str, Any] = state.get("context", {}) or {}

        # 1. Resolve raw input from context or state
        raw_input = None
        if input_key and str(input_key).strip():
            raw_input = context.get(input_key, state.get(input_key))

        if not raw_input:
            for fallback_key in ("latest_file_path", "uploaded_file", "file_path", "file", "excel_path", "uploaded_files"):
                if fallback_key in context and context[fallback_key]:
                    raw_input = context[fallback_key]
                    logger.debug(f"Excel to JSON Node {node_config.id}: fell back to context key '{fallback_key}'")
                    break

        file_path = _extract_file_path(raw_input)

        if not file_path:
            error_msg = (
                f"Excel to JSON Node '{node_config.id}' could not find a valid file path in context. "
                f"Checked input_key='{input_key}' and common fallbacks."
            )
            logger.error(error_msg)
            return {
                "messages": [
                    AIMessage(
                        content=f"❌ {error_msg}",
                        name=node_config.id,
                        response_metadata={"active_agent": node_config.id},
                    )
                ],
                "context": {
                    **context,
                    output_key: None,
                    f"{output_key}_error": error_msg,
                },
                "active_agent": node_config.id,
            }

        display_file_name = os.path.basename(file_path)

        # Resolve artifact:// to absolute filesystem path using ArtifactService
        if file_path.startswith("artifact://"):
            artifact_id = file_path.removeprefix("artifact://")
            try:
                from Database.Database import AsyncSessionLocal
                from Apps.AgentMesh.Core.Services.ArtifactService import ArtifactService
                async with AsyncSessionLocal() as db:
                    artifact_obj = await ArtifactService.get_by_id(
                        artifact_id=artifact_id,
                        db=db,
                    )
                    resolved_artifact_path = await ArtifactService.resolve_path(
                        artifact_id=artifact_id,
                        db=db,
                    )
                file_path = str(resolved_artifact_path)
                if artifact_obj and getattr(artifact_obj, "original_filename", None):
                    display_file_name = artifact_obj.original_filename
            except Exception as error:
                error_msg = f"Excel to JSON Node Error: Could not resolve artifact '{artifact_id}': {error}"
                logger.error(error_msg)
                return {
                    "messages": [
                        AIMessage(
                            content=f"❌ {error_msg}",
                            name=node_config.id,
                            response_metadata={"active_agent": node_config.id},
                        )
                    ],
                    "context": {
                        **context,
                        output_key: None,
                        f"{output_key}_error": error_msg,
                    },
                    "active_agent": node_config.id,
                }

        if not os.path.exists(file_path):
            error_msg = f"Excel file does not exist on disk: '{file_path}'"
            logger.error(error_msg)
            return {
                "messages": [
                    AIMessage(
                        content=f"❌ {error_msg}",
                        name=node_config.id,
                        response_metadata={"active_agent": node_config.id},
                    )
                ],
                "context": {
                    **context,
                    output_key: None,
                    f"{output_key}_error": error_msg,
                },
                "active_agent": node_config.id,
            }

        try:
            # 2. Read file with pandas
            orig_ext = os.path.splitext(display_file_name)[1].lower()
            disk_ext = os.path.splitext(file_path)[1].lower()
            file_ext = orig_ext if orig_ext else disk_ext
            df: pd.DataFrame

            if file_ext in (".csv", ".txt"):
                df = pd.read_csv(file_path, nrows=row_limit)
            elif file_ext in (".xlsx", ".xls", ".xlsm", ".xlsb", ".ods") or disk_ext == ".zip":
                engine = "openpyxl" if (file_ext in (".xlsx", ".xlsm") or disk_ext == ".zip") else None
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet_name, nrows=row_limit, engine=engine)
                except Exception:
                    df = pd.read_excel(file_path, sheet_name=sheet_name, nrows=row_limit)
            else:
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet_name, nrows=row_limit)
                except Exception:
                    df = pd.read_csv(file_path, nrows=row_limit)

            total_cols = len(df.columns)
            total_rows_read = len(df)
            columns_list = [str(c) for c in df.columns.tolist()]

            # 3. Handle nulls / NaNs
            if handle_nulls == "empty_string":
                df = df.fillna("")
            elif handle_nulls == "drop_rows":
                df = df.dropna(how="all")
            elif handle_nulls == "drop_cols":
                df = df.dropna(axis=1, how="all")

            # 4. Transform to JSON structure
            if headers_only:
                json_result = {
                    "columns": columns_list,
                    "total_columns": total_cols,
                    "file_name": display_file_name,
                }
            else:
                if orientation == "records":
                    raw_dict = df.to_dict(orient="records")
                elif orientation == "columns":
                    raw_dict = df.to_dict(orient="list")
                elif orientation == "split":
                    raw_dict = df.to_dict(orient="split")
                elif orientation == "index":
                    raw_dict = df.to_dict(orient="index")
                elif orientation == "values":
                    raw_dict = df.values.tolist()
                else:
                    raw_dict = df.to_dict(orient="records")

                json_result = _sanitize_for_json(raw_dict)

            # 5. Build context & chat log (Dual Output: Full Data + Explicit Sample Data)
            sample_records = []
            if isinstance(json_result, list):
                if sample_method == "random" and len(json_result) > sample_limit:
                    import random
                    sample_records = random.sample(json_result, sample_limit)
                else:
                    sample_records = json_result[:sample_limit]
            else:
                sample_records = json_result

            # Truncate long strings in sample rows so LLM prompt payload stays lightweight (~2-3 KB)
            trimmed_samples = []
            if isinstance(sample_records, list):
                for row in sample_records:
                    if isinstance(row, dict):
                        row_copy = {}
                        for rk, rv in row.items():
                            if isinstance(rv, str) and len(rv) > 300:
                                row_copy[rk] = rv[:300] + "..."
                            else:
                                row_copy[rk] = rv
                        trimmed_samples.append(row_copy)
                    else:
                        trimmed_samples.append(row)
            else:
                trimmed_samples = sample_records

            sample_data_payload = {
                "file_name": display_file_name,
                "sheet_name": sheet_name,
                "total_rows": total_rows_read,
                "total_columns": total_cols,
                "sample_limit": sample_limit,
                "sample_method": sample_method,
                "columns": columns_list,
                "sample_rows": trimmed_samples,
            }

            # 6. Save full parsed JSON to artifact file on disk (Option 2: Zero LangGraph State Bloat)
            json_file_path = f"{file_path}.json" if os.path.isabs(file_path) else os.path.abspath(f"{file_path}.json")
            try:
                with open(json_file_path, "w", encoding="utf-8") as f:
                    json.dump(json_result, f, indent=2, default=str)
                logger.info(
                    f"Excel to JSON Node {node_config.id}: persisted {total_rows_read} rows "
                    f"to disk at '{json_file_path}'"
                )
            except Exception as io_err:
                logger.warning(
                    f"Excel to JSON Node {node_config.id}: could not write companion JSON file '{json_file_path}': {io_err}"
                )

            new_context = dict(context)
            new_context.pop("context", None)
            new_context.pop("data", None)
            new_context.pop("msg", None)
            # Remove legacy alias keys
            new_context.pop("excel_json_data_path", None)
            new_context.pop("latest_excel_json", None)
            new_context.pop("extracted_file_path", None)
            new_context.pop("latest_file_path", None)
            new_context.pop("vapt_validation_preview", None)

            # Single Source of Truth
            new_context[output_key] = json_file_path                      # Lightweight file path for LangGraph state (< 100 bytes)
            new_context[sample_output_key] = sample_data_payload          # Explicit sample dataset (schema + top/random N rows for LLM)
            new_context[f"{output_key}_meta"] = {
                "file_name": display_file_name,
                "file_path": file_path,
                "json_file_path": json_file_path,
                "sheet_name": sheet_name,
                "rows_read": total_rows_read,
                "total_columns": total_cols,
                "columns": columns_list,
                "row_limit_applied": row_limit,
                "sample_limit": sample_limit,
                "sample_method": sample_method,
            }

            sample_preview = ""
            try:
                sample_preview = json.dumps(
                    trimmed_samples[:3] if isinstance(trimmed_samples, list) else trimmed_samples,
                    indent=2,
                    default=str
                )
                if len(sample_preview) > 600:
                    sample_preview = sample_preview[:600] + "\n  ..."
            except Exception:
                sample_preview = str(trimmed_samples)[:300]

            limit_info = f" (limited to top {row_limit} rows)" if row_limit else " (all rows)"
            chat_message = (
                f"**Spreadsheet Converted**\n"
                f"• **File**: `{display_file_name}`\n"
                f"• **Sheet**: `{sheet_name}`\n"
                f"• **Rows Extracted**: `{total_rows_read}`{limit_info}\n"
                f"• **Columns ({total_cols})**: `{', '.join(columns_list[:8])}{'...' if total_cols > 8 else ''}`\n\n"
                f"```json\n{sample_preview}\n```"
            )

            logger.info(
                f"Excel to JSON Node {node_config.id}: successfully processed {total_rows_read} rows "
                f"and {total_cols} columns from '{display_file_name}' into disk artifact '{json_file_path}'"
            )

            is_muted = getattr(node_config, "mute_chat_bubble", False)
            out_messages = [] if is_muted else [
                AIMessage(
                    content=chat_message,
                    name=node_config.id,
                    response_metadata={"active_agent": node_config.id},
                )
            ]

            return {
                "messages": out_messages,
                "context": new_context,
                "active_agent": node_config.id,
            }

        except Exception as e:
            logger.exception(f"Excel to JSON Node {node_config.id} failed: {e}")
            error_msg = f"Excel conversion error: {str(e)}"
            return {
                "messages": [
                    AIMessage(
                        content=f"❌ {error_msg}",
                        name=node_config.id,
                        response_metadata={"active_agent": node_config.id},
                    )
                ],
                "context": {
                    **context,
                    output_key: None,
                    f"{output_key}_error": error_msg,
                },
                "active_agent": node_config.id,
            }

    return excel_to_json_node

"""
Multi-File Ingestion Node Builder (Production Grade)
Handles multi-file batch uploads, file validation (max size, extension whitelist),
rich metadata extraction, session-isolated disk storage, PDF page-by-page rendering,
and built-in item-by-item queue iteration ('loop' vs 'done').
"""

import hashlib
import logging
import mimetypes
import os
import time
from typing import Callable, Any, Dict, List

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from Apps.AgentMesh.Config import settings
from Apps.AgentMesh.Schema.StateSchema import AgentState
from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig

logger = logging.getLogger(__name__)


def _format_bytes(size_bytes: int) -> str:
    if not size_bytes or size_bytes <= 0:
        return "0 Bytes"
    units = ["Bytes", "KB", "MB", "GB", "TB"]
    i = 0
    size = float(size_bytes)
    while size >= 1024.0 and i < len(units) - 1:
        size /= 1024.0
        i += 1
    return f"{size:.2f} {units[i]}"


def _calculate_sha256(filepath: str) -> str:
    sha256_hash = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        logger.warning(f"Could not calculate hash for {filepath}: {e}")
        return ""


def _sanitize_filename(name: str) -> str:
    """Sanitizes filename by taking basename and replacing illegal characters to prevent path traversal."""
    if not name:
        return "unnamed_file"
    clean = os.path.basename(name).replace("..", "").replace("/", "_").replace("\\", "_")
    return clean or "unnamed_file"


def _process_pdf_pages(file_path: str, thread_id: str, orig_filename: str) -> List[Dict[str, Any]]:
    """Splits a multi-page PDF into single page PNG images saved in session storage."""
    page_items = []
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.error("PyMuPDF (fitz) is not installed. PDF page extraction falling back to whole file.")
        return []

    try:
        doc = fitz.open(file_path)
        total_pages = len(doc)
        pages_dir = os.path.abspath(os.path.join(settings.TEMP_MOUNT, thread_id, "pdf_pages"))
        os.makedirs(pages_dir, exist_ok=True)

        clean_orig_name = _sanitize_filename(orig_filename)
        base_name = os.path.splitext(clean_orig_name)[0]

        for i in range(total_pages):
            page_num = i + 1
            page = doc.load_page(i)
            pix = page.get_pixmap(dpi=150)
            page_filename = f"{base_name}_page_{page_num}.png"
            page_path = os.path.join(pages_dir, page_filename)
            pix.save(page_path)

            file_size = os.path.getsize(page_path)

            page_items.append({
                "file_name": page_filename,
                "original_filename": orig_filename,
                "file_path": page_path,
                "file_size_bytes": file_size,
                "file_size_formatted": _format_bytes(file_size),
                "extension": ".png",
                "mime_type": "image/png",
                "page_number": page_num,
                "total_pages": total_pages,
                "item_type": "pdf_page",
                "parent_file_path": file_path,
                "upload_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "sha256_hash": _calculate_sha256(page_path),
            })
        doc.close()
    except Exception as e:
        logger.warning(f"Failed to open or split PDF {file_path} into pages: {e}")
    return page_items


def build_multi_file_ingestion_node(node_config: NodeConfig) -> Callable:
    """
    Build a multi-file ingestion node.
    Extracts uploaded files from state.user_input, validates size/extensions,
    extracts metadata, splits PDFs if page_by_page mode is set, and iterates through queue.
    """
    async def execute_multi_file_ingestion(state: AgentState):
        ctx = state.get("context", {}) or {}
        thread_id = ctx.get("thread_id", "global")
        node_id = node_config.id

        # 1. State Keys for Queue Management
        iter_key = f"_multi_file_items_{node_id}"
        idx_key = f"_multi_file_idx_{node_id}"

        items = ctx.get(iter_key)
        idx = ctx.get(idx_key, 0)

        # ── INITIALIZATION PHASE (First Execution after user submission) ──
        if items is None:
            user_data = state.get("user_input") if isinstance(state, dict) else getattr(state, "user_input", {})
            uploaded_raw = (user_data or {}).get("uploaded_files") or ctx.get("uploaded_files") or ctx.get("payload", {}).get("uploaded_files", [])

            if isinstance(uploaded_raw, dict):
                uploaded_raw = [uploaded_raw]

            allowed_exts = [ext.strip().lower().lstrip(".") for ext in (node_config.allowed_extensions or ["pdf", "jpg", "jpeg", "png", "webp"])]
            max_size_bytes = float(node_config.max_file_size_mb or 25.0) * 1024 * 1024
            max_files_limit = int(node_config.max_files_count or 10)

            session_upload_dir = os.path.abspath(os.path.join(settings.UPLOAD_MOUNT, thread_id))
            os.makedirs(session_upload_dir, exist_ok=True)

            processed_items = []
            validation_errors = []

            # Enforcement: Backend Max Batch File Count Limit
            if len(uploaded_raw) > max_files_limit:
                validation_errors.append(f"Upload batch exceeds maximum allowed limit of {max_files_limit} files (Received: {len(uploaded_raw)}).")
                uploaded_raw = uploaded_raw[:max_files_limit]

            for f_info in uploaded_raw:
                if isinstance(f_info, str):
                    f_path = f_info
                    raw_orig = os.path.basename(f_path)
                elif isinstance(f_info, dict):
                    f_path = f_info.get("path") or f_info.get("file_path") or f_info.get("url")
                    raw_orig = f_info.get("filename") or f_info.get("file_name") or os.path.basename(f_path or "")
                else:
                    continue

                if not f_path:
                    validation_errors.append("Invalid file entry: missing file path.")
                    continue

                orig_name = _sanitize_filename(raw_orig)
                abs_f_path = os.path.abspath(f_path)

                # Containment & Path Traversal Check
                upload_root = os.path.abspath(settings.UPLOAD_MOUNT)
                temp_root = os.path.abspath(settings.TEMP_MOUNT)
                if not (abs_f_path.startswith(upload_root) or abs_f_path.startswith(temp_root)):
                    validation_errors.append(f"Security Alert: File path '{orig_name}' is outside authorized session storage directories.")
                    continue

                if not os.path.exists(abs_f_path):
                    validation_errors.append(f"File not found on disk: {orig_name}")
                    continue

                ext = os.path.splitext(orig_name)[1].lower().lstrip(".")
                if allowed_exts and ext not in allowed_exts:
                    validation_errors.append(f"Unsupported file type .{ext} for '{orig_name}'. Allowed: {allowed_exts}")
                    continue

                file_size = os.path.getsize(abs_f_path)
                if file_size > max_size_bytes:
                    validation_errors.append(f"File '{orig_name}' exceeds size limit of {node_config.max_file_size_mb} MB ({_format_bytes(file_size)})")
                    continue

                mime_type, _ = mimetypes.guess_type(abs_f_path)
                category_key = "pdf" if ext == "pdf" else ("images" if ext in ("jpg", "jpeg", "png", "webp") else ("json" if ext == "json" else ("webpage" if ext in ("html", "htm") else "other")))

                total_pages = 1
                if ext == "pdf":
                    try:
                        import fitz
                        doc = fitz.open(abs_f_path)
                        total_pages = len(doc)
                        doc.close()
                    except Exception as e:
                        logger.warning(f"Could not read PDF page count for {orig_name}: {e}")

                # Read strategy_rules list (single source of truth for per-file-type strategies)
                strategy_rules = getattr(node_config, "strategy_rules", None) or []
                active_strats = []

                if strategy_rules and isinstance(strategy_rules, list):
                    for r in strategy_rules:
                        if isinstance(r, dict):
                            r_type = r.get("file_type")
                            r_strat = r.get("strategy")
                            is_match = (
                                (r_type == "pdf" and ext == "pdf") or
                                (r_type == "images" and ext in ("jpg", "jpeg", "png", "webp")) or
                                (r_type == "json" and ext == "json") or
                                (r_type == "webpage" and ext in ("html", "htm"))
                            )
                            if is_match and r_strat:
                                active_strats = [r_strat]
                                break

                # Handle PDF Strategy: Page-by-Page Split (Default strategy is 'page_by_page' unless explicitly changed to 'whole_file' in strategy_rules)
                pdf_strat = "page_by_page"
                if active_strats:
                    pdf_strat = active_strats[0]

                if ext == "pdf" and pdf_strat == "page_by_page":
                    pages = _process_pdf_pages(abs_f_path, thread_id, orig_name)
                    if pages:
                        for p in pages:
                            p["applied_strategies"] = [pdf_strat]
                        processed_items.extend(pages)
                        continue

                # Handle Image Strategy: Black & White Conversion
                effective_path = abs_f_path
                if ext in ("jpg", "jpeg", "png", "webp") and "black_white" in active_strats:
                    try:
                        from PIL import Image
                        bw_dir = os.path.abspath(os.path.join(settings.TEMP_MOUNT, thread_id, "bw_images"))
                        os.makedirs(bw_dir, exist_ok=True)
                        bw_filename = f"bw_{orig_name}"
                        bw_path = os.path.join(bw_dir, bw_filename)

                        with Image.open(abs_f_path) as img:
                            bw_img = img.convert("L").point(lambda p: 255 if p > 128 else 0, mode="1")
                            bw_img.save(bw_path)
                            effective_path = bw_path
                            logger.info(f"[MULTI_FILE_INGESTION] Applied Black & White threshold to {orig_name} -> {bw_path}")
                    except Exception as e:
                        logger.warning(f"Could not convert {orig_name} to Black & White: {e}")

                processed_items.append({
                    "file_name": orig_name,
                    "original_filename": orig_name,
                    "file_path": effective_path,
                    "file_size_bytes": file_size,
                    "file_size_formatted": _format_bytes(file_size),
                    "extension": f".{ext}",
                    "mime_type": mime_type or "application/octet-stream",
                    "total_pages": total_pages,
                    "page_number": 1 if ext == "pdf" else None,
                    "item_type": category_key + "_document",
                    "applied_strategies": active_strats,
                    "upload_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "sha256_hash": _calculate_sha256(effective_path),
                })

            if validation_errors and not processed_items:
                err_msg = f"Multi-File Ingestion Error:\n" + "\n".join(validation_errors)
                return {
                    "messages": [AIMessage(content=err_msg, name=node_id, response_metadata={"active_agent": node_id})],
                    "active_agent": node_id,
                    "error": {"message": err_msg, "type": "FileValidationError"},
                    "context": {"multi_file_errors": validation_errors}
                }

            items = processed_items

        logger.info(f"[MULTI_FILE_INGESTION] Node {node_id}: Processed {len(items)} file(s)/page(s). Outputting file_list to context.")
        summary_msg = f"Successfully ingested {len(items)} file(s)/page(s)."
        
        first_file_path = items[0]["file_path"] if items else None
        first_file_name = items[0]["file_name"] if items else None

        return {
            "messages": [AIMessage(content=summary_msg, name=node_id, response_metadata={"active_agent": node_id})],
            "active_agent": node_id,
            "context": {
                "file_list": items,
                "uploaded_files": items,
                "latest_file_path": first_file_path,
                "latest_file_name": first_file_name,
                "multi_file_total_items": len(items),
                "multi_file_errors": validation_errors if 'validation_errors' in locals() else []
            }
        }

    executor = RunnableLambda(execute_multi_file_ingestion)
    setattr(executor, "name", node_config.id)
    return executor

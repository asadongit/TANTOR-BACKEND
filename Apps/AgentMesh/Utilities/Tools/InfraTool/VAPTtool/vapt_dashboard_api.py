import io
import json
import logging
import os
import math
import asyncio
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
import httpx
from Apps.AgentMesh.Utilities.LLM.llm_config import get_llm_provider
from langchain_core.messages import HumanMessage
from Apps.AgentMesh.Core.Services.WorkflowService import WorkflowService
from Apps.AgentMesh.Core.Compiler.nodes.jev_node import (
    _build_item_state,
    evaluate_state_with_usage,
    _extract_answers_from_response,
    DEFAULT_TYPESAFE_API_URL,
    get_jev_progress,
    subscribe_jev_progress,
    unsubscribe_jev_progress,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Define the categories for the LLM
CATEGORIES = ["OS", "Network/Infrastructure", "Application", "Other/General"]

async def categorize_batch(vulnerabilities: List[str]) -> Dict[str, str]:
    """Sends a batch of unique vulnerabilities to the LLM for categorization."""
    if not vulnerabilities:
        return {}

    prompt = f"""
You are a cybersecurity expert. Categorize the following list of vulnerabilities into ONE of these specific categories:
{CATEGORIES}

Return ONLY a valid JSON object where the keys are the exact vulnerability names provided, and the values are the category you assigned.

Vulnerabilities to categorize:
{json.dumps(vulnerabilities)}
"""
    try:
        # Use the explicitly requested model
        llm = get_llm_provider(model="openai/gpt-oss-120b", temperature=0.0)
        
        # Call the model using the wrapper's _agenerate to pass kwargs through
        response = await llm._agenerate(
            messages=[HumanMessage(content=prompt)],
            response_format={"type": "json_object"}
        )
        
        result_text = response.generations[0].message.content
        return json.loads(result_text)
    except Exception as e:
        logger.error(f"Error calling LLM for batch: {e}")
        return {"_debug_error": str(e)}

@router.post("/analyze-report")
async def analyze_vapt_report(file: UploadFile = File(...)):
    """
    Endpoint for the React Dashboard. 
    Accepts a VAPT CSV/XLSX, extracts counts deterministically, 
    and uses an LLM to categorize the unique vulnerabilities.
    """
    filename = file.filename.lower()
    if not (filename.endswith(".csv") or filename.endswith(".xlsx") or filename.endswith(".xls")):
        raise HTTPException(status_code=400, detail="Invalid file type. Only CSV and Excel files are supported.")

    try:
        contents = await file.read()
        
        # Parse into Pandas DataFrame in-memory
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        else:
            df = pd.read_excel(io.BytesIO(contents))
            
        if df.empty:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")
            
        # Normalize columns
        df.columns = [str(c).strip().lower() for c in df.columns]
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing file: {str(e)}")

    # 1. Deterministic Analysis (Totals & Severity)
    total_vulns = len(df)
    
    # Identify severity column
    severity_col = next((col for col in df.columns if col in ['severity', 'risk', 'risk_factor']), None)
    severity_distribution = []
    
    if severity_col:
        # Normalize numeric severities to strings if needed
        numeric_to_label = {"4": "Critical", "3": "High", "2": "Medium", "1": "Low", "0": "Info"}
        df["_norm_severity"] = df[severity_col].apply(lambda x: numeric_to_label.get(str(x).strip(), str(x)).capitalize())
        
        counts = df["_norm_severity"].value_counts().to_dict()
        for sev, count in counts.items():
            severity_distribution.append({"name": sev, "value": count})
    else:
        severity_distribution = [{"name": "Unknown", "value": total_vulns}]

    # 2. AI Categorization (Unique Vulnerability Names)
    # Identify the vulnerability name column
    name_col = next((col for col in df.columns if col in ['plugin name', 'name', 'vulnerability', 'title']), None)
    
    category_counts = {}
    llm_mapping = {}
    
    if name_col:
        # Extract unique names and drop NA
        unique_vulns = df[name_col].dropna().astype(str).unique().tolist()
        
        # Batching (e.g., chunks of 20)
        BATCH_SIZE = 20
        batches = [unique_vulns[i:i + BATCH_SIZE] for i in range(0, len(unique_vulns), BATCH_SIZE)]
        
        # Run LLM calls in parallel
        tasks = [categorize_batch(batch) for batch in batches]
        batch_results = await asyncio.gather(*tasks)
        
        # Merge dictionary results
        for res in batch_results:
            llm_mapping.update(res)
            
        # Map back to full dataset to get category counts
        df["_category"] = df[name_col].map(llm_mapping).fillna("Unknown")
        cat_counts = df["_category"].value_counts().to_dict()
        
        for cat, count in cat_counts.items():
            category_counts[cat] = count
    else:
        category_counts = {"Unknown": total_vulns}
        
    # Format categories for Recharts
    formatted_category_dist = [{"name": k, "value": v} for k, v in category_counts.items()]

    # Return structured JSON
    return {
        "summary": {
            "total_vulnerabilities": total_vulns,
            "unique_vulnerabilities": len(llm_mapping) if llm_mapping else 0
        },
        "severity_distribution": severity_distribution,
        "category_distribution": formatted_category_dist,
        "llm_mapping": llm_mapping,
        "debug": llm_mapping.get("_debug_error", "No errors")
    }


# ─── DYNAMIC VAPT FINDINGS API (ZERO FEATURE LOSS) ───────────────────────────

SEVERITY_COLORS = {
    "Critical": "#991b1b",
    "High": "#c2410c",
    "Medium": "#b45309",
    "Low": "#1e40af",
    "Info": "#047857"
}

STANDARD_CATEGORIES = ["OS", "Network", "Database", "Application", "Other/General"]

DISPLAY_CATEGORY_MAP = {
    "os": "OS",
    "db": "Database",
    "database": "Database",
    "app": "Application",
    "application": "Application",
    "network": "Network",
    "other": "Other/General",
    "other/general": "Other/General",
    "insufficient_info": "Other/General",
    "unclassified": "Other/General"
}

# In-memory findings cache: {thread_id or file_path: {"payload": dict, "by_category": dict, "total_findings": int}}
_FINDINGS_CACHE: Dict[str, Dict[str, Any]] = {}


def normalize_severity_label(raw: Any) -> str:
    """Robust scanner-agnostic severity normalizer (Nessus, Qualys, CVSS numeric, etc.)."""
    s = str(raw or "").strip().lower().replace(".0", "")
    if s in ("4", "critical", "severe", "s0", "100", "5"):
        return "Critical"
    if s in ("3", "high", "important", "s1"):
        return "High"
    if s in ("2", "medium", "moderate", "s2"):
        return "Medium"
    if s in ("1", "low", "minor", "s3"):
        return "Low"
    if s in ("0", "info", "informational", "s4", "none"):
        return "Info"
    try:
        num = float(s)
        if num >= 9.0:
            return "Critical"
        if num >= 7.0:
            return "High"
        if num >= 4.0:
            return "Medium"
        if num > 0.0:
            return "Low"
        return "Info"
    except (ValueError, TypeError):
        pass
    return "Medium" if s else "Info"


def resolve_field(item: Dict[str, Any], keys: List[str]) -> str:
    """Case-insensitive attribute finder across arbitrary scan headers."""
    for k in keys:
        if k in item and item[k] is not None and str(item[k]).strip() not in ("", "nan", "None"):
            return str(item[k]).strip()
        for ik, iv in item.items():
            if ik.lower() == k.lower() and iv is not None and str(iv).strip() not in ("", "nan", "None"):
                return str(iv).strip()
    return ""


def format_single_record(item: Dict[str, Any], idx: int) -> Tuple[Dict[str, Any], str, str]:
    """
    Standardizes a single scan finding into a Tantor UI table record.
    Returns: (record_dict, normalized_severity, normalized_category)
    """
    name = resolve_field(item, [
        "plugin name", "vulnerability name", "name", "title", "vulnerability",
        "synopsis", "issue_name", "rule_name"
    ])
    pkg = resolve_field(item, ["affected_package", "package", "component", "software"])
    cve = resolve_field(item, ["cve", "related_cve", "cve id", "cve_id", "plugin id", "aikido_issue_id", "cwe_id"])
    
    if not name and pkg:
        name = pkg + (f" ({cve})" if cve else "")
    elif pkg and name and pkg.lower() not in name.lower():
        name = f"{pkg}: {name}"
    if not name:
        name = resolve_field(item, ["description", "summary"]) or f"Security Finding #{idx + 1}"

    host = resolve_field(item, [
        "host", "ip address", "ip", "target", "cpe", "asset", "virtual_machine",
        "cloud_resource", "affected_file", "domain", "fqdn"
    ]) or "Target Asset"

    raw_sev = resolve_field(item, [
        "risk", "severity", "severity level", "score", "severity_score", "cvss",
        "original_cvss_severity_score", "risk_factor"
    ])
    norm_sev = normalize_severity_label(raw_sev)

    raw_cat = item.get("category") or item.get("vulnerability_category")
    cat_lower = str(raw_cat or "").strip().lower()
    category = DISPLAY_CATEGORY_MAP.get(cat_lower, str(raw_cat).strip() if raw_cat else "Other/General")

    # Confidence reported by the classifier. Resolved from the primary choice
    # answer's own <question_id>_confidence field, never from a hardcoded list of
    # question ids, and never invented when absent: a row with no model answer
    # must not display as if it were 90% certain.
    confidence = None
    for key, val in item.items():
        if isinstance(key, str) and key.endswith("_confidence") and val is not None:
            confidence = val
            break
    if confidence is None:
        confidence = item.get("confidence") or item.get("category_confidence")
    if confidence is None:
        jev_evals = item.get("jev_evaluations")
        if isinstance(jev_evals, dict):
            for eval_value in jev_evals.values():
                if isinstance(eval_value, dict) and "confidence" in eval_value:
                    confidence = eval_value["confidence"]
                    break
    try:
        conf_val = round(float(confidence), 2) if confidence is not None else None
    except (ValueError, TypeError):
        conf_val = None

    # Margin (top probability minus runner-up) is the ambiguity signal.
    margin_val = None
    for key, val in item.items():
        if isinstance(key, str) and key.endswith("_margin") and val is not None:
            try:
                margin_val = round(float(val), 2)
            except (ValueError, TypeError):
                margin_val = None
            break

    # Top probability of the winning class
    top_prob = item.get("top_probability") or item.get("vulnerability_category_top_prob")
    if top_prob is None:
        probs = item.get("vulnerability_category_probabilities")
        if isinstance(probs, dict) and probs:
            try:
                top_prob = max(float(v) for v in probs.values())
            except (ValueError, TypeError):
                top_prob = None

    # Determine jev_status and review reason based on dual OR condition:
    # (model_confidence < 0.60) OR (top_prob < 0.60)
    # The finding ALWAYS stays in its winning category tab (Preserve Top Category rule).
    jev_status = item.get("jev_status")
    jev_review_reason = item.get("jev_review_reason")

    if not jev_status or jev_status in ("evaluated", "needs_review", "classified"):
        is_below_conf = (conf_val is not None and conf_val < 0.60)
        is_below_prob = (top_prob is not None and top_prob < 0.60)
        if is_below_conf or is_below_prob or jev_status == "needs_review":
            jev_status = "needs_review"
            if not jev_review_reason:
                reasons = []
                if is_below_conf:
                    reasons.append(f"Model confidence {int(conf_val*100)}% < 60%")
                if is_below_prob:
                    reasons.append(f"Top probability {int(top_prob*100)}% < 60%")
                jev_review_reason = "Needs review: " + "; ".join(reasons) if reasons else "Flagged for manual review"
        else:
            jev_status = "classified"

    solution = resolve_field(item, ["solution", "remediation", "fix", "mitigation"])
    if not solution:
        solution = "Remediation pending analyst review."

    record = {
        "id": f"vuln-{idx}",
        "name": name,
        "severity": norm_sev,
        "raw_severity": raw_sev,
        "category": category,
        "confidence": conf_val,
        "top_probability": round(float(top_prob), 2) if top_prob is not None else None,
        "margin": margin_val,
        "host": host,
        "cve": cve or f"ID-{idx + 1}",
        "solution": solution,
        "status": "Pending",
        "jev_status": jev_status,
        "jev_review_reason": jev_review_reason,
    }
    return record, norm_sev, category


def build_findings_payload(items: List[Dict[str, Any]], thread_id: str = "") -> Dict[str, Any]:
    """
    Transforms raw or Jev-tagged findings into a zero-feature-loss dashboard payload.
    Supplies all metrics, chart arrays, category tabs, per-category severity breakdowns,
    and table records for Tantor UI. Caches categorized items in memory for fast pagination.
    """
    cat_map: Dict[str, List[Dict[str, Any]]] = {c: [] for c in STANDARD_CATEGORIES}
    by_category_all: Dict[str, List[Dict[str, Any]]] = {c: [] for c in STANDARD_CATEGORIES}
    
    s0 = s1 = s2 = s3 = s4 = 0
    category_severity_stats: Dict[str, Dict[str, int]] = {}

    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue

        record, norm_sev, category = format_single_record(item, idx)

        if norm_sev == "Critical":
            s0 += 1
        elif norm_sev == "High":
            s1 += 1
        elif norm_sev == "Medium":
            s2 += 1
        elif norm_sev == "Low":
            s3 += 1
        else:
            s4 += 1

        if category not in cat_map:
            cat_map[category] = []
            by_category_all[category] = []

        by_category_all[category].append(record)

        # Cap preview table rows per category to 250 items so the initial payload remains fast for the UI
        if len(cat_map[category]) < 250:
            cat_map[category].append(record)

        # Track per-category severity and needs_review breakdown
        if category not in category_severity_stats:
            category_severity_stats[category] = {
                "total": 0, "Critical": 0, "High": 0, "Medium": 0, "Low": 0, "Info": 0, "needs_review": 0
            }
        category_severity_stats[category]["total"] += 1
        category_severity_stats[category][norm_sev] = category_severity_stats[category].get(norm_sev, 0) + 1
        if record.get("jev_status") == "needs_review":
            category_severity_stats[category]["needs_review"] = category_severity_stats[category].get("needs_review", 0) + 1

    # Ensure all standard categories have an entry
    for c in STANDARD_CATEGORIES:
        if c not in category_severity_stats:
            category_severity_stats[c] = {
                "total": 0, "Critical": 0, "High": 0, "Medium": 0, "Low": 0, "Info": 0, "needs_review": 0
            }
        elif "needs_review" not in category_severity_stats[c]:
            category_severity_stats[c]["needs_review"] = 0

    # Donut chart array (exact color palette expected by VaptApp.jsx)
    pts_data = [
        {"name": "Critical", "value": s0, "color": SEVERITY_COLORS["Critical"]},
        {"name": "High", "value": s1, "color": SEVERITY_COLORS["High"]},
        {"name": "Medium", "value": s2, "color": SEVERITY_COLORS["Medium"]},
        {"name": "Low", "value": s3, "color": SEVERITY_COLORS["Low"]},
        {"name": "Info", "value": s4, "color": SEVERITY_COLORS["Info"]},
    ]

    # Dynamic column definitions
    columns = [
        {"key": "name", "label": "Vulnerability", "type": "text", "primary": True},
        {"key": "severity", "label": "Severity", "type": "severity_badge", "width": "110px"},
        {"key": "host", "label": "Resource / Asset", "type": "text", "width": "180px"},
        {"key": "confidence", "label": "AI Confidence", "type": "percentage", "width": "110px"},
        {"key": "status", "label": "Status", "type": "status", "width": "100px"},
    ]

    # Available categories sorted: Standard categories first, then any extra
    active_categories = [c for c in STANDARD_CATEGORIES if c in cat_map]
    for c in cat_map:
        if c not in active_categories:
            active_categories.append(c)

    payload = {
        "thread_id": thread_id,
        "is_vapt": True,
        "total_findings": len(items),
        "severity_counts": {
            "s0": s0,
            "s1": s1,
            "s2": s2,
            "s3": s3,
            "s4": s4,
            "Critical": s0,
            "High": s1,
            "Medium": s2,
            "Low": s3,
            "Info": s4
        },
        "pts_data": pts_data,
        "categories": active_categories,
        "category_stats": category_severity_stats,
        "columns": columns,
        "vulnerabilities": cat_map,
    }

    # Store in fast memory cache for subsequent pagination queries
    if thread_id:
        _FINDINGS_CACHE[thread_id] = {
            "payload": payload,
            "by_category": by_category_all,
            "total_findings": len(items)
        }

    return payload


def make_non_vapt_response(thread_id: str, reason: str = "", scanner_type: str = "Unknown") -> Dict[str, Any]:
    """Helper to return an empty findings payload with an informative Non-VAPT Report dialogue."""
    clean_reason = reason or "The provided file contains non-security data and does not represent an authentic VAPT report."
    clean_scanner = scanner_type or "Unknown"
    return {
        "thread_id": thread_id,
        "is_vapt": False,
        "total_findings": 0,
        "scanner_type": clean_scanner,
        "reason": clean_reason,
        "dialogue": {
            "type": "non_vapt_report",
            "title": "Non-VAPT Report Detected",
            "reason": clean_reason,
            "scanner_type": clean_scanner,
            "message": (
                "⚠️ The uploaded file is not a valid VAPT report.\n\n"
                f"• **Reason:** {clean_reason}\n\n"
                "Please check your file and upload an authentic vulnerability scan report."
            )
        },
        "severity_counts": {
            "s0": 0, "s1": 0, "s2": 0, "s3": 0, "s4": 0,
            "Critical": 0, "High": 0, "Medium": 0, "Low": 0, "Info": 0
        },
        "pts_data": [
            {"name": "Critical", "value": 0, "color": SEVERITY_COLORS["Critical"]},
            {"name": "High", "value": 0, "color": SEVERITY_COLORS["High"]},
            {"name": "Medium", "value": 0, "color": SEVERITY_COLORS["Medium"]},
            {"name": "Low", "value": 0, "color": SEVERITY_COLORS["Low"]},
            {"name": "Info", "value": 0, "color": SEVERITY_COLORS["Info"]},
        ],
        "categories": STANDARD_CATEGORIES,
        "category_stats": {c: {"total": 0, "Critical": 0, "High": 0, "Medium": 0, "Low": 0, "Info": 0} for c in STANDARD_CATEGORIES},
        "columns": [],
        "vulnerabilities": {c: [] for c in STANDARD_CATEGORIES},
    }


async def resolve_thread_context_and_items(thread_id: str, workflow_id: Optional[str] = None):
    """
    Resolves the thread context, validation metadata, and raw findings items/file path.
    Supports aliases like 'latest', 'recent', 'default'.
    """
    resolved_thread_id = thread_id
    context = {}

    if str(thread_id).strip().lower() in ("latest", "recent", "default", "current"):
        try:
            from Database.Database import AsyncSessionLocal
            from sqlalchemy import text
            async with AsyncSessionLocal() as session:
                r = await session.execute(
                    text("""
                        SELECT thread_id, context 
                        FROM workflow_run_history 
                        WHERE status = 'success'
                          AND context IS NOT NULL 
                          AND context::text LIKE '%_tagged.json%'
                        ORDER BY completed_at DESC LIMIT 1
                    """)
                )
                row = r.fetchone()
                if row:
                    resolved_thread_id = row[0]
                    context = row[1] or {}
                    logger.info(f"Resolved alias '{thread_id}' to thread '{resolved_thread_id}'")
        except Exception as e:
            logger.warning(f"Failed to resolve latest VAPT thread from history: {e}")

    if not context:
        thread_state = await WorkflowService.get_thread_state(resolved_thread_id, workflow_id=workflow_id)
        context = thread_state.get("context", {}) if isinstance(thread_state, dict) else {}

    val_res = context.get("validation_result") if isinstance(context.get("validation_result"), dict) else {}
    is_vapt = context.get("is_vapt") if context.get("is_vapt") is not None else val_res.get("is_vapt")
    reason = context.get("reason", "") or val_res.get("reason", "")
    scanner_type = context.get("scanner_type") or val_res.get("scanner_type") or "Unknown"

    findings_items = (
        context.get("excel_json_data_path")
        or context.get("excel_json_data")
        or context.get("latest_excel_json")
        or context.get("findings")
        or []
    )
    if not findings_items:
        try:
            from Database.Database import AsyncSessionLocal
            from sqlalchemy import text
            async with AsyncSessionLocal() as session:
                r = await session.execute(
                    text("SELECT context FROM workflow_run_history WHERE thread_id = :tid ORDER BY completed_at DESC LIMIT 1"),
                    {"tid": resolved_thread_id}
                )
                row = r.fetchone()
                if row and row[0]:
                    ctx = row[0]
                    ctx_val_res = ctx.get("validation_result") if isinstance(ctx.get("validation_result"), dict) else {}
                    if is_vapt is None:
                        is_vapt = ctx.get("is_vapt") if ctx.get("is_vapt") is not None else ctx_val_res.get("is_vapt")
                    if not reason:
                        reason = ctx.get("reason", "") or ctx_val_res.get("reason", "")
                    if not scanner_type or scanner_type == "Unknown":
                        scanner_type = ctx.get("scanner_type") or ctx_val_res.get("scanner_type") or "Unknown"
                    findings_items = (
                        ctx.get("excel_json_data_path")
                        or ctx.get("excel_json_data")
                        or ctx.get("latest_excel_json")
                        or ctx.get("findings")
                        or []
                    )
        except Exception as db_err:
            logger.warning(f"Fallback check in workflow_run_history for thread '{resolved_thread_id}' failed: {db_err}")

    file_path = None
    if isinstance(findings_items, str) and (findings_items.endswith(".json") or os.path.exists(findings_items)):
        file_path = findings_items

    return resolved_thread_id, is_vapt, reason, scanner_type, findings_items, file_path


@router.get("/findings/{thread_id}")
async def get_vapt_findings_by_thread(thread_id: str, workflow_id: Optional[str] = None):
    """
    Main Dashboard API: Fetches classified findings from the execution thread state.
    Returns complete pre-computed statistics, chart arrays, and table rows with zero feature loss.
    If the report failed VAPT validation (non-VAPT), returns 0 findings and a non-VAPT report dialogue.
    """
    try:
        resolved_thread_id, is_vapt, reason, scanner_type, findings_items, file_path = await resolve_thread_context_and_items(
            thread_id, workflow_id=workflow_id
        )

        # Check if full payload is already cached in memory
        if resolved_thread_id in _FINDINGS_CACHE and "payload" in _FINDINGS_CACHE[resolved_thread_id]:
            logger.info(f"Returning cached findings payload for thread '{resolved_thread_id}'")
            return _FINDINGS_CACHE[resolved_thread_id]["payload"]

        # If LLM validation explicitly flagged file as non-VAPT, NEVER count as vulnerabilities
        if is_vapt is False:
            logger.info(f"Thread '{resolved_thread_id}' is non-VAPT. Returning non-VAPT dialogue with 0 points.")
            return make_non_vapt_response(thread_id=resolved_thread_id, reason=reason, scanner_type=scanner_type)

        # Load from disk if it is a JSON file path
        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    findings_items = json.load(f)
            except Exception as err:
                logger.warning(f"Could not load findings from JSON file '{file_path}': {err}")

        if not isinstance(findings_items, list):
            findings_items = [findings_items] if findings_items else []

        payload = build_findings_payload(findings_items, thread_id=resolved_thread_id)
        if file_path:
            _FINDINGS_CACHE[file_path] = _FINDINGS_CACHE.get(resolved_thread_id, {})
        return payload

    except Exception as exc:
        logger.error(f"Error fetching VAPT findings for thread '{thread_id}': {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch findings for thread: {exc}")


@router.get("/findings/{thread_id}/page")
async def get_vapt_findings_page(
    thread_id: str,
    category: str = Query("OS", description="Category tab to paginate"),
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(50, ge=5, le=500, description="Items per page"),
    search: Optional[str] = Query(None, description="Search keyword in name, CVE, host, or solution"),
    severity: Optional[str] = Query(None, description="Filter by severity (Critical, High, Medium, Low, Info)"),
    status: Optional[str] = Query(None, description="Filter by status (classified, needs_review)"),
    workflow_id: Optional[str] = None
):
    """
    High-Performance Pagination API: Returns a slice of findings for a specific category tab.
    Serves from in-memory cache in < 10ms for instant browsing across large scans (30k+ findings).
    """
    try:
        resolved_thread_id, is_vapt, reason, scanner_type, findings_items, file_path = await resolve_thread_context_and_items(
            thread_id, workflow_id=workflow_id
        )

        if is_vapt is False:
            return {
                "thread_id": resolved_thread_id,
                "category": category,
                "page": page,
                "page_size": page_size,
                "total_records": 0,
                "total_pages": 1,
                "records": []
            }

        # Check memory cache
        cache_entry = _FINDINGS_CACHE.get(resolved_thread_id) or (file_path and _FINDINGS_CACHE.get(file_path))
        if not cache_entry or "by_category" not in cache_entry:
            if file_path and os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    items = json.load(f)
            elif isinstance(findings_items, list):
                items = findings_items
            else:
                items = []

            build_findings_payload(items, thread_id=resolved_thread_id)
            cache_entry = _FINDINGS_CACHE.get(resolved_thread_id)
            if file_path and cache_entry:
                _FINDINGS_CACHE[file_path] = cache_entry

        by_cat = cache_entry.get("by_category", {}) if cache_entry else {}
        
        # Standardize category name
        cat_key = DISPLAY_CATEGORY_MAP.get(category.strip().lower(), category.strip())
        records = by_cat.get(cat_key, [])
        if not records and category.strip().lower() == "all":
            records = [r for sublist in by_cat.values() for r in sublist]

        # Filter by severity if specified
        if severity and severity.strip():
            sev_target = normalize_severity_label(severity).lower()
            records = [r for r in records if r.get("severity", "").lower() == sev_target]

        # Filter by status if specified (e.g. needs_review or classified)
        if status and status.strip():
            st_target = status.strip().lower()
            records = [r for r in records if str(r.get("jev_status", "")).lower() == st_target]

        # Filter by search keyword if specified
        if search and search.strip():
            kw = search.strip().lower()
            records = [
                r for r in records
                if kw in str(r.get("name", "")).lower()
                or kw in str(r.get("cve", "")).lower()
                or kw in str(r.get("host", "")).lower()
                or kw in str(r.get("solution", "")).lower()
            ]

        total_records = len(records)
        total_pages = math.ceil(total_records / page_size) if total_records > 0 else 1
        page_num = max(1, min(page, total_pages))

        start_idx = (page_num - 1) * page_size
        end_idx = start_idx + page_size
        sliced = records[start_idx:end_idx]

        return {
            "thread_id": resolved_thread_id,
            "category": cat_key,
            "page": page_num,
            "page_size": page_size,
            "total_records": total_records,
            "total_pages": total_pages,
            "records": sliced
        }

    except Exception as exc:
        logger.error(f"Error fetching paginated VAPT findings for thread '{thread_id}': {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch paginated findings: {exc}")


@router.get("/findings/{thread_id}/export")
async def export_vapt_findings(
    thread_id: str,
    format: str = Query("xlsx", regex="^(xlsx|csv|json)$", description="Export format: xlsx, csv, or json"),
    workflow_id: Optional[str] = None
):
    """
    Exports classified VAPT scan findings into Excel (.xlsx), CSV, or JSON format.
    The exported file includes all original scan data plus standardized 'category', 'severity',
    and remediation columns, ready for offline review or re-importing with Jev bypass.
    """
    try:
        resolved_thread_id, is_vapt, reason, scanner_type, findings_items, file_path = await resolve_thread_context_and_items(
            thread_id, workflow_id=workflow_id
        )

        if is_vapt is False:
            raise HTTPException(status_code=400, detail="Cannot export findings for an invalid or non-VAPT report.")

        # If static 30k export exists and format is xlsx, return it immediately for instant download
        backup_xlsx = os.path.abspath(os.path.join(os.getcwd(), "vapt_30k_tagged_findings.xlsx"))
        if format == "xlsx" and os.path.exists(backup_xlsx) and str(thread_id).strip().lower() in ("latest", "30k", "vapt_mvhs1xuv_h76i0"):
            return FileResponse(
                path=backup_xlsx,
                filename="vapt_classified_findings.xlsx",
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        # Load items
        items = []
        if file_path and os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                items = json.load(f)
        elif isinstance(findings_items, list):
            items = findings_items

        if not items:
            raise HTTPException(status_code=404, detail=f"No findings found to export for thread '{thread_id}'.")

        # Standardize items with clean category labels
        normalized_items = []
        for idx, it in enumerate(items):
            if isinstance(it, dict):
                rec, norm_sev, category = format_single_record(it, idx)
                row = dict(it)
                # Deduplicate case variants so DataFrame export does not contain duplicate headers
                for k in list(row.keys()):
                    if k.lower() in ("severity", "category") and k not in ("severity", "category"):
                        row.pop(k, None)
                row["category"] = category
                row["severity"] = norm_sev
                if "confidence" not in row or row["confidence"] is None:
                    row["confidence"] = rec["confidence"]
                if "solution" not in row or not row["solution"]:
                    row["solution"] = rec["solution"]
                normalized_items.append(row)

        df = pd.DataFrame(normalized_items)

        # Temp directory for export
        temp_dir = os.path.abspath(os.path.join(os.getcwd(), "mnt", "temp"))
        os.makedirs(temp_dir, exist_ok=True)

        if format == "xlsx":
            export_path = os.path.join(temp_dir, f"vapt_findings_{resolved_thread_id}.xlsx")
            df.to_excel(export_path, index=False, engine="openpyxl")
            return FileResponse(
                path=export_path,
                filename=f"vapt_findings_{resolved_thread_id}.xlsx",
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        elif format == "csv":
            export_path = os.path.join(temp_dir, f"vapt_findings_{resolved_thread_id}.csv")
            df.to_csv(export_path, index=False, encoding="utf-8")
            return FileResponse(
                path=export_path,
                filename=f"vapt_findings_{resolved_thread_id}.csv",
                media_type="text/csv"
            )
        else: # json
            export_path = os.path.join(temp_dir, f"vapt_findings_{resolved_thread_id}.json")
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(normalized_items, f, indent=2, default=str)
            return FileResponse(
                path=export_path,
                filename=f"vapt_findings_{resolved_thread_id}.json",
                media_type="application/json"
            )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error exporting VAPT findings for thread '{thread_id}': {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to export findings: {exc}")


@router.post("/findings")
async def process_vapt_findings_direct(request: Request):
    """
    Accepts direct JSON payload containing items or thread_id to return dynamic dashboard metrics.
    Useful when frontend already holds the context from workflow execute response.
    """
    body = await request.json()
    thread_id = body.get("thread_id", "")
    if body.get("is_vapt") is False:
        reason = body.get("reason", "")
        scanner = body.get("scanner_type", "Unknown")
        return make_non_vapt_response(thread_id=thread_id, reason=reason, scanner_type=scanner)

    items = body.get("items") or body.get("excel_json_data") or []
    return build_findings_payload(items, thread_id=thread_id)


@router.get("/progress/{thread_id}")
async def get_vapt_progress(thread_id: str):
    """
    Returns real-time Jev/Laya evaluation progress for a given workflow session thread.
    Used by the Tantor UI VAPT Dashboard as fallback query.
    """
    prog = get_jev_progress(thread_id)
    if not prog:
        return {"status": "idle", "completed": 0, "total": 0, "percent": 0.0}

    res = {
        "status": prog.get("status", "evaluating"),
        "completed": prog.get("completed", 0),
        "total": prog.get("total", 0),
        "percent": prog.get("percent", 0.0),
        "node_id": prog.get("node_id"),
    }
    if "dashboard_data" in prog and prog["dashboard_data"]:
        res["dashboard_data"] = prog["dashboard_data"]
    else:
        records = prog.get("records")
        if records and isinstance(records, list) and len(records) > 0:
            res["dashboard_data"] = build_findings_payload(records, thread_id=thread_id)
        elif prog.get("checkpoint_path") and os.path.exists(prog["checkpoint_path"]):
            try:
                with open(prog["checkpoint_path"], "r", encoding="utf-8") as f:
                    chk_records = json.load(f)
                if isinstance(chk_records, list) and len(chk_records) > 0:
                    res["dashboard_data"] = build_findings_payload(chk_records, thread_id=thread_id)
            except Exception:
                pass
    return res


@router.get("/progress/{thread_id}/stream")
async def stream_vapt_progress(thread_id: str):
    """
    Server-Sent Events (SSE) streaming endpoint for live VAPT evaluation progress & findings.
    Pushes real-time progress and classified findings directly to the Tantor UI dashboard.
    """
    queue = subscribe_jev_progress(thread_id)

    async def event_generator():
        try:
            # 1. Send immediate snapshot if current progress exists
            current = get_jev_progress(thread_id)
            if current:
                yield f"data: {json.dumps(current, default=str)}\n\n"
                if current.get("status") in ("completed", "error"):
                    return

            # 2. Stream subsequent progress events pushed by node
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=20.0)
                    yield f"data: {json.dumps(event, default=str)}\n\n"
                    if event.get("status") in ("completed", "error"):
                        break
                except asyncio.TimeoutError:
                    # Keep-alive comment for long-running evaluations
                    yield ": keep-alive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            unsubscribe_jev_progress(thread_id, queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
    )


class JevStreamEvaluateRequest(BaseModel):
    file_path: str = Field(..., description="Absolute disk path to the JSON file containing records to evaluate.")
    questions: Dict[str, Dict[str, Any]] = Field(..., description="Canvas questions config (choice, noul, score).")
    model: str = Field("jev-latest", description="TypeSafe AI model name.")
    concurrency: int = Field(25, description="Max concurrent HTTP requests to Jev API.")
    batch_progress_size: int = Field(10, description="Rows to evaluate before emitting one SSE progress event.")
    api_key: Optional[str] = Field(None, description="Optional TypeSafe API key override.")
    api_url: Optional[str] = Field(None, description="Optional TypeSafe API URL override.")


@router.post("/jev/stream-evaluate")
async def jev_stream_evaluate(request: JevStreamEvaluateRequest):
    """
    Evaluates all rows in a JSON file against configured Jev questions.
    Streams progress and evaluated items as SSE events in chunks of batch_progress_size.
    """
    file_path = request.file_path
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            items = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read JSON file: {e}")

    if not isinstance(items, list):
        items = [items]

    total_items = len(items)
    api_key = request.api_key or os.environ.get("TYPESAFE_API_KEY", "")
    api_url = request.api_url or DEFAULT_TYPESAFE_API_URL
    model = request.model
    questions = request.questions
    concurrency = max(1, request.concurrency)
    chunk_size = max(1, request.batch_progress_size)

    async def event_generator():
        semaphore = asyncio.Semaphore(concurrency)
        limits = httpx.Limits(max_connections=concurrency * 2, max_keepalive_connections=concurrency)
        tagged_records = []
        completed_count = 0

        async def process_item(item: Any, http_client: httpx.AsyncClient) -> Any:
            state_value, state_meta = _build_item_state(item, None, None)
            row = dict(item) if isinstance(item, dict) else {"original": item}
            row["jev_fields_sent"] = state_meta["fields_sent"]

            # Refuse to spend a request the state cannot support.
            if not state_meta["usable"]:
                row["jev_status"] = "insufficient_state"
                row["jev_error"] = "No usable state fields resolved for this row."
                row["jev_evaluations"] = {}
                return row

            if not api_key:
                row["jev_status"] = "insufficient_state"
                row["jev_error"] = "TYPESAFE_API_KEY not configured"
                row["jev_evaluations"] = {}
                return row

            answers: Dict[str, Any] = {}
            async with semaphore:
                try:
                    answers, _usage = await evaluate_state_with_usage(
                        http_client, api_url, api_key, model, state_value, questions
                    )
                except Exception as exc:
                    row["jev_status"] = "error"
                    row["jev_error"] = str(exc)
                    row["jev_evaluations"] = {}
                    return row

            extracted, review_reason = _extract_answers_from_response(answers, questions)
            row.update(extracted)
            row["jev_status"] = "needs_review" if review_reason else "classified"
            if review_reason:
                row["jev_review_reason"] = review_reason
            row["jev_evaluations"] = answers
            return row

        async with httpx.AsyncClient(timeout=15.0, limits=limits) as http_client:
            for chunk_start in range(0, total_items, chunk_size):
                chunk = items[chunk_start : chunk_start + chunk_size]
                chunk_results = await asyncio.gather(
                    *[process_item(it, http_client) for it in chunk],
                    return_exceptions=True
                )

                clean_chunk = []
                for r in chunk_results:
                    if isinstance(r, Exception):
                        clean_chunk.append({"jev_status": "error", "jev_error": str(r)})
                    else:
                        clean_chunk.append(r)

                tagged_records.extend(clean_chunk)
                completed_count += len(clean_chunk)
                pct = round((completed_count / max(1, total_items)) * 100, 1)

                progress_data = {
                    "type": "jev_progress",
                    "completed": completed_count,
                    "total": total_items,
                    "percent": pct,
                    "rows": clean_chunk,
                }
                yield f"data: {json.dumps(progress_data)}\n\n"

        # Save tagged file
        tagged_path = file_path if file_path.endswith("_tagged.json") else file_path.replace(".json", "_tagged.json")
        try:
            with open(tagged_path, "w", encoding="utf-8") as f:
                json.dump(tagged_records, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to write tagged JSON file {tagged_path}: {e}")

        complete_data = {
            "type": "jev_complete",
            "total": total_items,
            "tagged_file_path": tagged_path,
        }
        yield f"data: {json.dumps(complete_data)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
    )


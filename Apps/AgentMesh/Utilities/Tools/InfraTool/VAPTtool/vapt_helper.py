import os
import json
import logging
from typing import Any, Optional, Tuple
import pandas as pd
from Apps.AgentMesh.Config import settings

logger = logging.getLogger(__name__)

NUMERIC_TO_LABEL = {
    "4": "critical",
    "4.0": "critical",
    "critical": "critical",
    "3": "high",
    "3.0": "high",
    "high": "high",
    "2": "medium",
    "2.0": "medium",
    "medium": "medium",
    "1": "low",
    "1.0": "low",
    "low": "low",
    "0": "info",
    "0.0": "info",
    "info": "info",
    "informational": "info",
    "none": "info"
}


def resolve_artifact_id_to_path(artifact_id: str) -> Optional[str]:
    """Resolves an artifact ID to its absolute path under ARTIFACT_MOUNT."""
    search_roots = []
    if getattr(settings, "ARTIFACT_MOUNT", None):
        search_roots.append(os.path.abspath(settings.ARTIFACT_MOUNT))
    search_roots.append(os.path.abspath(os.path.join(os.getcwd(), "mnt", "artifacts")))

    for root_dir in search_roots:
        if os.path.exists(root_dir):
            for dirpath, _, filenames in os.walk(root_dir):
                if os.path.basename(dirpath) == artifact_id:
                    for fname in filenames:
                        if not fname.endswith(".part"):
                            found_path = os.path.join(dirpath, fname)
                            logger.info(f"VAPT Resolver: Resolved artifact ID {artifact_id} -> {found_path}")
                            return found_path
    return None


def _is_vapt_report(file_path: str) -> bool:
    """Verifies whether a file contains recognizable VAPT vulnerability columns."""
    try:
        lower_name = os.path.basename(file_path).lower()
        if any(w in lower_name for w in ["scan", "vapt", "nessus", "vuln", "security", "nmap", "cve"]):
            return True
        df = load_vapt_dataframe(file_path)
        if df is not None and not df.empty:
            cols = [str(c).strip().lower() for c in df.columns]
            vapt_indicators = ["plugin id", "plugin name", "cve", "severity", "risk", "cvss", "vpr", "vulnerability", "synopsis", "solution"]
            matches = sum(1 for ind in vapt_indicators if any(ind in c for c in cols))
            if matches >= 2:
                return True
    except Exception:
        pass
    return False


def resolve_vapt_file(
    file_path: str = "",
    config: Optional[Any] = None,
    thread_id: Optional[str] = None,
    context: Optional[dict] = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolves the provided file path, artifact URI, or execution context into an absolute path on disk.
    Prioritizes:
    1. Explicit valid file_path or artifact:// URI.
    2. Context state from RunnableConfig (extracted_file_path, latest_file_path, excel_json_data_meta).
    3. Uploaded artifact ID from thread context.
    4. Safe fallback searching mnt/artifacts and mnt/temp, prioritizing verified VAPT reports.
    Returns (resolved_file_path, error_message).
    """
    clean_path = file_path.strip().strip('"').strip("'") if file_path else ""

    # 1. Handle explicit artifact:// URI
    if clean_path.startswith("artifact://"):
        art_id = clean_path.removeprefix("artifact://").strip()
        found = resolve_artifact_id_to_path(art_id)
        if found:
            return found, None

    # 2. Check if clean_path exists directly on disk
    if clean_path and os.path.exists(clean_path) and os.path.isfile(clean_path):
        return os.path.abspath(clean_path), None

    # 3. Extract context and thread_id from RunnableConfig if available
    if config:
        if isinstance(config, dict):
            cfg = config.get("configurable", {})
        else:
            cfg = getattr(config, "configurable", {}) or {}
        if not context:
            context = cfg.get("context") if isinstance(cfg, dict) else getattr(cfg, "context", None)
        if not thread_id:
            thread_id = cfg.get("thread_id") if isinstance(cfg, dict) else getattr(cfg, "thread_id", None)

    # 4. Check context keys populated by excel_to_json_node or ingestion

    if context and isinstance(context, dict):
        # 1. Primary: Canonical excel_json_data (points to pre-processed or JEV-tagged JSON)
        cand_path = context.get("excel_json_data")
        if cand_path and isinstance(cand_path, str) and os.path.exists(cand_path) and os.path.isfile(cand_path):
            logger.info(f"VAPT Resolver: Resolved from context.excel_json_data -> {cand_path}")
            return os.path.abspath(cand_path), None

        # 2. Uploaded file resolution if direct artifact URI or dict
        uploaded = context.get("uploaded_file")
        if uploaded:
            if isinstance(uploaded, dict) and uploaded.get("artifact_id"):
                found = resolve_artifact_id_to_path(str(uploaded["artifact_id"]))
                if found:
                    logger.info(f"VAPT Resolver: Resolved from context.uploaded_file.artifact_id -> {found}")
                    return found, None
            elif isinstance(uploaded, str):
                if uploaded.startswith("artifact://"):
                    found = resolve_artifact_id_to_path(uploaded.removeprefix("artifact://"))
                    if found:
                        return found, None
                elif os.path.exists(uploaded) and os.path.isfile(uploaded):
                    return os.path.abspath(uploaded), None

        # 3. Low-priority backward compatibility fallback for older workflow runs
        for legacy_k in ("excel_json_data_path", "extracted_file_path", "latest_file_path"):
            cand = context.get(legacy_k)
            if cand and isinstance(cand, str) and os.path.exists(cand) and os.path.isfile(cand):
                logger.info(f"VAPT Resolver: Resolved from legacy context.{legacy_k} -> {cand}")
                return os.path.abspath(cand), None

    # If thread_id is explicit, never fall back to an arbitrary report from another thread
    if thread_id and thread_id != "default_thread":
        err = f"Error: Could not locate uploaded VAPT report for thread '{thread_id}'."
        return None, err

    # 5. Fallback search in mnt/temp and mnt/artifacts (only when no thread_id was specified)
    search_dirs = []
    temp_dir = os.path.abspath(os.path.join(os.getcwd(), "mnt", "temp"))
    if os.path.exists(temp_dir):
        search_dirs.append(temp_dir)

    artifact_dir = os.path.abspath(os.path.join(os.getcwd(), "mnt", "artifacts"))
    if os.path.exists(artifact_dir):
        search_dirs.append(artifact_dir)

    if getattr(settings, "TEMP_MOUNT", None):
        cfg_temp = os.path.abspath(settings.TEMP_MOUNT)
        if os.path.exists(cfg_temp) and cfg_temp not in search_dirs:
            search_dirs.append(cfg_temp)

    if getattr(settings, "ARTIFACT_MOUNT", None):
        cfg_art = os.path.abspath(settings.ARTIFACT_MOUNT)
        if os.path.exists(cfg_art) and cfg_art not in search_dirs:
            search_dirs.append(cfg_art)

    candidates = []
    valid_exts = (".json", ".csv", ".xlsx", ".xls", ".xlsm", ".xlsb", ".ods", ".zip")

    for search_dir in search_dirs:
        for root, dirs, files in os.walk(search_dir):
            if "skills" in dirs:
                dirs.remove("skills")
            for f in files:
                if f.endswith(".part"):
                    continue
                if f.lower().endswith(valid_exts):
                    full_p = os.path.join(root, f)
                    try:
                        mtime = os.path.getmtime(full_p)
                        candidates.append((full_p, mtime))
                    except Exception:
                        pass

    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        # Prioritize candidates that pass VAPT verification
        for cand, _ in candidates:
            if _is_vapt_report(cand):
                logger.info(f"VAPT Resolver: Resolved verified VAPT report from storage: {cand}")
                return cand, None

        # Fallback to the newest candidate if none matched VAPT heuristics
        chosen_file = candidates[0][0]
        logger.info(f"VAPT Resolver: Falling back to most recent uploaded file: {chosen_file}")
        return chosen_file, None

    err = f"Error: File not found at '{file_path}' and no fallback reports exist in storage."
    return None, err


def load_vapt_dataframe(file_path: str, sheet_name: Any = 0) -> pd.DataFrame:
    """
    Robustly loads a VAPT report from JSON, Excel (.xlsx, .xls, .zip), or CSV into a pandas DataFrame.
    """
    ext = os.path.splitext(file_path)[1].lower()
    df: Optional[pd.DataFrame] = None

    # Handle structured JSON datasets (primary format for post-ingestion / JEV tagged findings)
    if ext == ".json":
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                records = None
                for k in ("findings", "items", "rows", "records", "data", "vulnerabilities"):
                    if k in data and isinstance(data[k], list):
                        records = data[k]
                        break
                df = pd.DataFrame(records if records is not None else [data])
            logger.info(f"VAPT Loader: Loaded {len(df)} rows from JSON report '{file_path}'")
        except Exception as e:
            logger.error(f"VAPT Loader: Error reading JSON file '{file_path}': {e}")

    # Handle Excel formats and artifact.zip (which are openpyxl-compatible xlsx packages)
    elif ext in (".xlsx", ".xls", ".xlsm", ".xlsb", ".ods") or ext == ".zip":
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name or 0, engine="openpyxl")
        except Exception:
            try:
                df = pd.read_excel(file_path, sheet_name=sheet_name or 0)
            except Exception:
                df = pd.read_csv(file_path)
    elif ext in (".csv", ".txt"):
        try:
            df = pd.read_csv(file_path)
        except Exception:
            try:
                df = pd.read_excel(file_path, sheet_name=sheet_name or 0, engine="openpyxl")
            except Exception:
                df = pd.read_excel(file_path, sheet_name=sheet_name or 0)
    else:
        # Unknown extension, try json, then excel, then csv
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            df = pd.DataFrame(data if isinstance(data, list) else [data])
        except Exception:
            try:
                df = pd.read_excel(file_path, sheet_name=sheet_name or 0, engine="openpyxl")
            except Exception:
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet_name or 0)
                except Exception:
                    df = pd.read_csv(file_path)

    if df is None:
        raise ValueError(f"Could not parse file '{file_path}' as JSON, CSV, or Excel.")

    # Deduplicate columns case-insensitively while preserving original order
    if not df.empty:
        seen_cols = set()
        keep_indices = []
        for i, col in enumerate(df.columns):
            norm_col = str(col).strip().lower()
            if norm_col not in seen_cols:
                seen_cols.add(norm_col)
                keep_indices.append(i)
        if len(keep_indices) < len(df.columns):
            df = df.iloc[:, keep_indices]

    return df


def normalize_severity(raw_severity: Any) -> str:
    """
    Normalizes severity from string or numeric values (4, 4.0, critical) into lowercase label.
    """
    s = str(raw_severity).strip().lower()
    if s.endswith(".0"):
        s = s[:-2]
    return NUMERIC_TO_LABEL.get(s, s)

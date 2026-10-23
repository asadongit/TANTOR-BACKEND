import os
import io
import json
import logging
from typing import Any, Optional
import pandas as pd
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from Apps.AgentMesh.Schema.ToolSchema import ToolMeta
from Apps.AgentMesh.Core.ToolRegistry import registry
from Apps.AgentMesh.Utilities.Tools.InfraTool.VAPTtool.vapt_helper import resolve_vapt_file, load_vapt_dataframe, normalize_severity

logger = logging.getLogger(__name__)

@tool
def analyze_vapt_csv(
    file_path: str = "",
    severity_filter: str = "critical,high",
    category_filter: str = "all",
    limit: int = 10,
    offset: int = 0,
    sheet_name: str = "",
    config: RunnableConfig = None,
    **kwargs: Any
) -> str:
    """
    Analyzes a VAPT (Vulnerability Assessment and Penetration Testing) report file (Excel .xlsx/.xls or CSV).
    - file_path: The file path, artifact URI (artifact://<id>), or leave empty to auto-detect the uploaded scan.
    - severity_filter: Comma-separated list of severities (e.g., 'critical,high', 'all'). Default 'critical,high'.
    - category_filter: Comma-separated list of categories (e.g., 'os,network,database,application,other/general', 'all'). Default 'all'.
    - limit: Max number of vulnerabilities to return in this batch (default 10).
    - offset: Number of vulnerabilities to skip (default 0). Use this to fetch the next batch.
    - sheet_name: Optional Excel sheet name or index (default first sheet).
    
    The tool filters by severity and category, and returns a STRICTLY paginated JSON summary.
    """
    resolved_path, err = resolve_vapt_file(file_path, config=config)
    if err or not resolved_path:
        return err or f"Error: Could not locate VAPT report file '{file_path}'."


    try:
        df = load_vapt_dataframe(resolved_path, sheet_name=sheet_name)

        if df is None or df.empty:
            return "Error: No data found in the report file."

        df.columns = [str(c).strip().lower() for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated(keep="first")]

        def _get_cell(r, col_name: str) -> str:
            v = r.get(col_name, "")
            if isinstance(v, pd.Series):
                v = v.dropna().iloc[0] if not v.dropna().empty else ""
            return str(v).strip()

        matched_vulns = []

        for _, row in df.iterrows():
            # Robust host resolution
            host_val = ""
            for h_col in ("host", "ip address", "ip_address", "target", "asset", "virtual_machine", "cpe", "plugin family"):
                if h_col in df.columns and _get_cell(row, h_col) not in ("", "nan", "None"):
                    host_val = _get_cell(row, h_col)
                    break
            host = host_val if host_val else "Target System"

            raw_risk = ""
            for s_col in ("risk", "severity", "risk_factor", "severity level", "score"):
                if s_col in df.columns and _get_cell(row, s_col) not in ("", "nan", "None"):
                    raw_risk = _get_cell(row, s_col)
                    break
            risk = normalize_severity(raw_risk) if raw_risk else "info"

            # Robust CVE / Plugin ID resolution
            cve = ""
            for c_col in ("cve", "cve id", "cve_id", "cves", "related_cve", "related_cve_alias", "cwe_id"):
                if c_col in df.columns and _get_cell(row, c_col) not in ("", "nan", "None"):
                    cve = _get_cell(row, c_col)
                    break
            if not cve and "plugin id" in df.columns and _get_cell(row, "plugin id") not in ("", "nan", "None"):
                cve = f"Plugin ID: {_get_cell(row, 'plugin id')}"

            # Name / Vulnerability resolution
            name = ""
            for n_col in ("name", "vulnerability name", "plugin name", "plugin_name", "title", "rule"):
                if n_col in df.columns and _get_cell(row, n_col) not in ("", "nan", "None"):
                    name = _get_cell(row, n_col)
                    break
            if not name:
                pkg = _get_cell(row, "affected_package")
                desc = _get_cell(row, "cwe_description") or _get_cell(row, "synopsis")
                if pkg and pkg not in ("nan", "None"):
                    name = f"{pkg}: {desc}" if desc and desc not in ("nan", "None") else pkg
                elif desc and desc not in ("nan", "None"):
                    name = desc
            if not name:
                name = "Unknown Vulnerability"

            # Solution resolution
            solution = ""
            for sol_col in ("solution", "remediation", "fix", "mitigation", "synopsis", "description"):
                if sol_col in df.columns and _get_cell(row, sol_col) not in ("", "nan", "None"):
                    solution = _get_cell(row, sol_col)
                    break
            if not solution and "patched_version" in df.columns and _get_cell(row, "patched_version") not in ("", "nan", "None"):
                solution = f"Upgrade package to version {_get_cell(row, 'patched_version')} or higher."
            if not solution:
                solution = "No solution provided."

            # Calibrated category stamped by Jev Engine
            raw_row_cat = _get_cell(row, "category")
            cat_map = {
                "os": "OS",
                "db": "Database",
                "database": "Database",
                "app": "Application",
                "application": "Application",
                "network": "Network",
                "other": "Other/General",
                "other/general": "Other/General",
                "insufficient_info": "Other/General",
            }
            category = cat_map.get(raw_row_cat.lower(), "Other/General" if not raw_row_cat or raw_row_cat.lower() in ("nan", "none", "unknown") else raw_row_cat.title())

            # Severity Filter
            if severity_filter.lower() != "all":
                allowed_sev = [s.strip().lower() for s in severity_filter.split(",")]
                if risk not in allowed_sev:
                    continue

            # Category Filter
            if category_filter.lower() != "all":
                allowed_cat = [c.strip().lower() for c in category_filter.split(",")]
                if category.lower() not in allowed_cat:
                    continue

            matched_vulns.append({
                "host": host,
                "category": category,
                "severity": risk.upper(),
                "cve": cve if cve != 'nan' else "",
                "name": name if name != 'nan' else "Unknown Vulnerability",
                "recommended_solution": (solution[:200] + "...") if solution != 'nan' and len(solution) > 200 else (solution if solution != 'nan' else "No solution provided.")
            })

        # Sort by severity (CRITICAL > HIGH > MEDIUM > LOW > INFO)
        severity_order = {"CRITICAL": 1, "HIGH": 2, "MEDIUM": 3, "LOW": 4, "INFO": 5}
        matched_vulns.sort(key=lambda x: severity_order.get(x["severity"], 99))

        total_matches = len(matched_vulns)
        
        # Apply pagination
        paginated_vulns = matched_vulns[offset:offset+limit]
        
        # Group by host for the paginated subset to keep JSON clean
        grouped_vulns = {}
        for v in paginated_vulns:
            h = v.pop("host")
            if h not in grouped_vulns:
                grouped_vulns[h] = []
            grouped_vulns[h].append(v)

        summary = {
            "status": "success",
            "file_analyzed": os.path.basename(resolved_path),
            "filters": {
                "severity": severity_filter,
                "category": category_filter
            },
            "pagination": {
                "total_matching_vulnerabilities": total_matches,
                "limit": limit,
                "offset": offset,
                "returned_in_this_batch": len(paginated_vulns),
                "has_more": (offset + limit) < total_matches
            },
            "vulnerabilities_by_host": grouped_vulns
        }

        if summary["pagination"]["has_more"]:
            summary["agent_instruction"] = f"CRITICAL: There are {total_matches - (offset + limit)} more vulnerabilities matching this criteria. Tell the user you can only fix {limit} at once, present this batch nicely in a table, and ask if they want to proceed with this batch or load the next batch."
            
        logger.info(f"VAPT Analyzer: Returned {len(paginated_vulns)} out of {total_matches} vulns from {resolved_path}.")
        
        return json.dumps(summary, indent=2)

    except Exception as e:
        logger.error(f"VAPT Analyzer: Error processing {resolved_path}: {e}")
        return f"Error analyzing VAPT file: {str(e)}"

# Register the tool
registry.register(
    analyze_vapt_csv,
    ToolMeta(
        name="analyze_vapt_csv",
        display_name="VAPT Report Analyzer",
        description="Analyzes a VAPT report Excel/CSV file. You MUST ask the user which category (e.g. OS, Network, Database, Application) and severity (e.g. critical, high) they want to focus on BEFORE calling this tool, unless they already specified it. Supports filtering by severity_filter and category_filter. Strictly paginates results (default limit 10) to prevent LLM overload. Use the 'offset' parameter to fetch subsequent batches.",
        category="SecurityTools"
    )
)


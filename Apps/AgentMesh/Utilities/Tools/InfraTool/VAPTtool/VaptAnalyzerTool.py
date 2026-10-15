import os
import io
import json
import logging
from langchain_core.tools import tool
from Apps.AgentMesh.Schema.ToolSchema import ToolMeta
from Apps.AgentMesh.Core.ToolRegistry import registry

import pandas as pd

logger = logging.getLogger(__name__)

@tool
def analyze_vapt_csv(file_path: str, severity_filter: str = "critical,high", category_filter: str = "all", limit: int = 10, offset: int = 0) -> str:
    """
    Analyzes a VAPT (Vulnerability Assessment and Penetration Testing) report file (CSV or XLSX).
    Provide the FULL absolute file path to the report.
    - severity_filter: Comma-separated list of severities (e.g., 'critical,high', 'all'). Default 'critical,high'.
    - category_filter: Comma-separated list of categories (e.g., 'os,network,database,application,other/general', 'all'). Default 'all'.
    - limit: Max number of vulnerabilities to return in this batch (default 10).
    - offset: Number of vulnerabilities to skip (default 0). Use this to fetch the next batch.
    
    The tool filters by severity and category, and returns a STRICTLY paginated JSON summary.
    """
    file_path = file_path.strip().strip('"').strip("'") if file_path else ""

    if not os.path.exists(file_path):
        temp_dir = os.path.join(os.getcwd(), "mnt", "temp")
        if os.path.exists(temp_dir):
            candidates = []
            for root, _, files in os.walk(temp_dir):
                for f in files:
                    if f.endswith(('.csv', '.xlsx')):
                        candidates.append(os.path.join(root, f))
            if candidates:
                candidates.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                file_path = candidates[0]
                logger.info(f"VAPT Analyzer: Falling back to most recent upload: {file_path}")
            else:
                return f"Error: File not found at '{file_path}' and no fallback uploads exist in {temp_dir}."
        else:
            return f"Error: File not found: '{file_path}'"

    def get_category(plugin_name: str) -> str:
        name = str(plugin_name).lower()
        os_kw = ['os identification', 'os fingerprints', 'os security patch', 'time zone', 'mounted devices', 'running processes', 'vim installed', 'xz utils', 'gnome-shell', 'local users information', 'ubuntu pro', 'unix software discovery', 'kernel', 'privilege escalation', 'local privilege', 'group policy', 'registry', 'dll hijacking', 'wmi', 'rhel', 'centos', 'debian', 'ubuntu', 'redhat', 'windows', 'oracle linux', 'elsa-', 'openssl']
        if any(kw in name for kw in os_kw): return 'OS'
        if 'usn-' in name: return 'OS'
        
        db_kw = ['database', 'oracle database', 'postgres', 'postgresql', 'sql', 'mongo', 'mysql', 'redis', 'memcached', 'couchdb', 'mariadb']
        if any(kw in name for kw in db_kw): return 'Database'
        
        net_kw = ['ssh', 'netstat', 'openvpn', 'network', 'ip assignment', 'mac address', 'ethernet', 'mdns', 'portscanner', 'routing', 'interfaces', 'dns', 'service detection', 'ftp', 'smb', 'snmp', 'tcp/', 'udp/', 'ping', 'firewall', 'telnet', 'icmp', 'cifs', 'ntp', 'rdp', 'vpn', 'router', 'switch', 'bgp', 'ospf', 'ike', 'ipsec', 'rsync', 'gnutls', ' tls ', ' ssl ']
        if any(kw in name for kw in net_kw): return 'Network'
        
        app_kw = ['http', 'nginx', 'imagemagick', 'python', 'curl', 'cipher', 'certificate', 'libndp', 'libgcrypt', 'libexiv2', 'libcurl', 'strict transport security', 'apache', 'php', 'tomcat', 'iis', 'application', 'cgi', 'browser', 'oracle java', 'oracle enterprise', 'oracle global', 'weblogic', 'java', 'node', 'npm', 'ruby', 'rails', 'wordpress', 'drupal', 'joomla', 'struts', 'spring', 'log4j', 'docker', 'kubernetes', 'jenkins', 'grafana']
        if any(kw in name for kw in app_kw): return 'Application'
        
        gen_kw = ['vmware', 'nessus', '(via dmi)', 'cpe', 'backported', 'time of last system startup', 'target credential', 'system restart', 'device hostname', 'device type', 'path variables', 'bios info', "shor's harvest", 'software enumeration']
        if any(kw in name for kw in gen_kw): return 'Other/General'
        if 'linux' in name or 'unix' in name: return 'OS'
        
        return 'Other/General'

    try:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ('.xlsx', '.xls'):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)

        if df is None or df.empty:
            return "Error: No data found in the file."

        df.columns = [str(c).strip().lower() for c in df.columns]

        # First, filter the entire dataframe
        matched_vulns = []
        NUMERIC_TO_LABEL = {"4": "critical", "3": "high", "2": "medium", "1": "low", "0": "info"}

        for _, row in df.iterrows():
            host = str(row.get("host", row.get("ip address", row.get("ip_address", "unknown_host")))).strip()
            raw_risk = str(row.get("risk", row.get("severity", row.get("risk_factor", "info")))).strip()
            risk = NUMERIC_TO_LABEL.get(raw_risk, raw_risk.lower())
            
            cve = str(row.get("cve", ""))
            name = str(row.get("name", row.get("vulnerability name", row.get("plugin name", row.get("plugin_name", "Unknown Vulnerability")))))
            solution = str(row.get("solution", "No solution provided."))
            
            category = get_category(name)

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
            "file_analyzed": os.path.basename(file_path),
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
            
        logger.info(f"VAPT Analyzer: Returned {len(paginated_vulns)} out of {total_matches} vulns.")
        
        return json.dumps(summary, indent=2)

    except Exception as e:
        logger.error(f"VAPT Analyzer: Error processing {file_path}: {e}")
        return f"Error analyzing VAPT file: {str(e)}"

# Register the tool
registry.register(
    analyze_vapt_csv,
    ToolMeta(
        name="analyze_vapt_csv",
        display_name="VAPT Report Analyzer",
        description="Analyzes a VAPT report CSV/XLSX file. You MUST ask the user which category (e.g. OS, Network, Database, Application) and severity (e.g. critical, high) they want to focus on BEFORE calling this tool, unless they already specified it. Supports filtering by severity_filter and category_filter. Strictly paginates results (default limit 10) to prevent LLM overload. Use the 'offset' parameter to fetch subsequent batches.",
        category="SecurityTools"
    )
)


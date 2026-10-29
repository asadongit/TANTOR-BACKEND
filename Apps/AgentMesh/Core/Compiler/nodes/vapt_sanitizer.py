"""
VAPT State Sanitizer & Normalization Module
Standardizes preprocessing, bias removal, port translation, and decision gating
for both Jev (TypeSafe System One) and Laya classifier nodes.
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple

# Well-known port translation dictionary to supply explicit semantic signal to models
PORT_MAP: Dict[int, str] = {
    1521: "1521 (Oracle Database Listener)",
    1526: "1526 (Oracle Database Listener)",
    3306: "3306 (MySQL Database)",
    5432: "5432 (PostgreSQL Database)",
    1433: "1433 (Microsoft SQL Server)",
    1434: "1434 (Microsoft SQL Server Monitor)",
    27017: "27017 (MongoDB Database)",
    6379: "6379 (Redis Cache/Key-Value Store)",
    9200: "9200 (Elasticsearch Engine)",
    445:  "445 (SMB Windows File Sharing)",
    139:  "139 (NetBIOS Session Service)",
    137:  "137 (NetBIOS Name Service)",
    138:  "138 (NetBIOS Datagram Service)",
    22:   "22 (SSH Remote Shell)",
    23:   "23 (Telnet Unencrypted Remote Shell)",
    21:   "21 (FTP File Transfer)",
    25:   "25 (SMTP Mail Protocol)",
    110:  "110 (POP3 Mail Protocol)",
    143:  "143 (IMAP Mail Protocol)",
    53:   "53 (DNS Domain Name System)",
    67:   "67 (DHCP Server)",
    68:   "68 (DHCP Client)",
    161:  "161 (SNMP Network Monitoring)",
    162:  "162 (SNMP Trap)",
    123:  "123 (NTP Network Time Protocol)",
    179:  "179 (BGP Routing Protocol)",
    500:  "500 (IPSec VPN Key Exchange)",
    4500: "4500 (IPSec NAT Traversal)",
    80:   "80 (HTTP Web Service)",
    443:  "443 (HTTPS Secure Web Service)",
    8080: "8080 (HTTP Web Application Server)",
    8443: "8443 (HTTPS Web Application Server)",
    8000: "8000 (HTTP Web Development Server)",
    8888: "8888 (HTTP Web Application Server)",
    9090: "9090 (HTTP Web Application Server)",
    0:    "0 (Local File Inspection / Host Patch)",
}

# Bias and anchor columns that must NEVER be fed to classification models
BIAS_COLUMNS_TO_DROP: Set[str] = {
    # Host and Network Topology Anchors (prevent model from anchoring on host rather than component)
    "ip", "ip_address", "ip address", "ipaddress",
    "dns", "dns_name", "dns name", "hostname", "host_name", "host",
    "mac", "mac_address", "mac address", "netbios_name", "fqdn",

    # Severity & Scoring (pure token noise for technical layer classification)
    "severity", "risk", "cvss", "cvss_score", "cvss_base", "cvss3", "cvss_v3", "cvss v3",
    "cve", "cve_id", "cve id", "cpe",

    # Internal pipeline / audit columns
    "sr._no.", "sr. no.", "sr_no", "sr no", "id", "#", "s.no",
    "time_frame", "time frame", "csb_/_sisa", "csb / sisa", "pca1/pca2/otherlinux/vxrail",
    "first_discovered", "last_observed", "scan_date", "audit_date",
    "owner", "assigned_to", "status", "remediation_status", "ticket_id",

    # Internal classification keys
    "jev_status", "jev_error", "jev_evaluations", "jev_fields_sent",
    "jev_review_reason", "jev_error_status_code", "confidence",
    "category_confidence", "unclassified", "jev_summary", "category",
    "vulnerability_category", "vulnerability_category_confidence",
    "vulnerability_category_probabilities", "vulnerability_category_margin",
    "vulnerability_category_runner_up",
}

# Mapping descriptive model options back to canonical React frontend tabs
CATEGORY_CANONICAL_MAP: Dict[str, str] = {
    "database_engine": "db",
    "operating_system": "os",
    "application_software": "app",
    "network_infrastructure": "network",
    "hardware_firmware_or_policy": "other",

    # Pass-through for existing/legacy codes
    "db": "db",
    "os": "os",
    "app": "app",
    "network": "network",
    "other": "other",
    "insufficient_info": "insufficient_info",
}

# Compact criteria for Laya (20-25 words each, key discriminating nouns first)
COMPACT_VAPT_QUESTIONS: Dict[str, Any] = {
    "vulnerability_category": {
        "type": "choice",
        "instructions": "Which technical layer does this vulnerable software component or finding belong to?",
        "criteria": {
            "operating_system": "OS kernel, system packages and daemons (Linux, Windows, glibc, OpenSSH, systemd, sudo), or OS-level security patches.",
            "network_infrastructure": "Network devices (routers, switches, firewalls), protocols (BGP, DNS, NTP, SNMP), TLS/SSL ciphers, VPNs, or open listening ports.",
            "database_engine": "Database engines, database servers, DB listeners, storage configuration, or DB patches (Oracle, MySQL, PostgreSQL, SQL Server, MongoDB).",
            "application_software": "Application software, web/app servers, middleware, runtimes, frameworks, and third-party code libraries (Java JARs, Python, Tomcat, Apache).",
            "hardware_firmware_or_policy": "Hardware BIOS/IPMI/BMC firmware, physical security, or organizational password/compliance governance policies.",
        },
        "options": [
            "operating_system",
            "network_infrastructure",
            "database_engine",
            "application_software",
            "hardware_firmware_or_policy",
        ],
    }
}

# Rich criteria for hosted Jev (explicit edge case disambiguation)
RICH_VAPT_QUESTIONS: Dict[str, Any] = {
    "vulnerability_category": {
        "type": "choice",
        "instructions": "Which technical layer does this vulnerable software component or finding belong to?",
        "criteria": {
            "operating_system": "Operating system kernel, system packages, base utilities, core daemons (Linux, Windows, glibc, OpenSSH, systemd, PAM, sudo), or host-level OS security patches. Excludes third-party application runtimes.",
            "network_infrastructure": "Network hardware (routers, switches, firewalls), network protocols (BGP, DNS, NTP, SNMP), TLS/SSL ciphers, VPN endpoints, or open port weaknesses. Note: SSL/TLS cipher weaknesses belong to network even if hosted by a web server.",
            "database_engine": "Database management systems, database servers, TNS listeners, DB configuration, or DB vendor patches (Oracle, MySQL, PostgreSQL, Microsoft SQL Server, MongoDB, Redis). Excludes client application code connecting to DB.",
            "application_software": "Application software, web servers, application middleware, runtimes, frameworks, and third-party code packages or libraries (Java JARs, Log4j, Spring, Python packages, Apache HTTPD, Tomcat, WebLogic, Nginx).",
            "hardware_firmware_or_policy": "Hardware BIOS, UEFI, IPMI, BMC management firmware, physical security findings, or organization-level compliance, auditing, and password governance policies.",
        },
        "options": [
            "operating_system",
            "network_infrastructure",
            "database_engine",
            "application_software",
            "hardware_firmware_or_policy",
        ],
    }
}

JUNK_TEXT_PATTERNS = [
    re.compile(r"^\s*scan\s+summary", re.IGNORECASE),
    re.compile(r"^\s*table\s+of\s+contents", re.IGNORECASE),
    re.compile(r"^\s*disclaimer\b", re.IGNORECASE),
    re.compile(r"^\s*target\s+credential\s+issues", re.IGNORECASE),
    re.compile(r"^\s*nessus\s+scan\s+information", re.IGNORECASE),
    re.compile(r"^\s*compliance\s+check\s+report", re.IGNORECASE),
]


class VaptSanitizer:
    """Centralized sanitation, feature isolation, and decision gating for VAPT records."""

    @staticmethod
    def is_unusable_row(row: Any) -> bool:
        """
        Deterministic code-level check for empty, header, summary, or disclaimer rows.
        Catches junk rows before model invocation (0ms latency, 0 tokens spent).
        """
        if not isinstance(row, dict) or not row:
            return True

        # Check vulnerability/plugin title
        name_keys = ["vulnerability name", "vulnerability_name", "plugin name", "plugin_name", "finding", "name", "title"]
        vuln_name = ""
        for k in name_keys:
            for actual_k, val in row.items():
                if actual_k.lower().strip() == k and val:
                    vuln_name = str(val).strip()
                    break
            if vuln_name:
                break

        if not vuln_name:
            # If no primary title, check if total non-empty text across all columns is trivial
            total_text = " ".join(str(v).strip() for v in row.values() if v and not str(v).strip().isdigit())
            if len(total_text) < 15:
                return True

        # Check against known scanner disclaimer / summary patterns
        for pattern in JUNK_TEXT_PATTERNS:
            if pattern.search(vuln_name):
                return True

        return False

    @staticmethod
    def prepare_model_state(row: Dict[str, Any], max_output_chars: int = 400) -> Dict[str, Any]:
        """
        Constructs a sanitized, bias-free, ordered state dictionary for model input.
        Drops IPs, hostnames, and severities; translates ports to semantic labels.
        """
        clean_state: Dict[str, Any] = {}

        # 1. Normalize and identify key finding fields
        name_val = None
        family_val = None
        package_val = None
        port_val = None
        output_val = None

        for k, v in row.items():
            if v is None:
                continue
            k_norm = k.lower().strip()
            v_str = str(v).strip()
            if not v_str:
                continue

            # Skip bias columns
            if k_norm in BIAS_COLUMNS_TO_DROP:
                continue

            if k_norm in ("vulnerability name", "vulnerability_name", "plugin name", "plugin_name", "name", "title"):
                name_val = v_str
            elif k_norm in ("plugin family", "plugin_family", "family", "category_name"):
                family_val = v_str
            elif k_norm in ("package", "package_name", "installed_package", "component"):
                package_val = v_str
            elif k_norm in ("port", "port_number", "service_port"):
                port_val = v
            elif k_norm in ("plugin output", "plugin_output", "description", "synopsis", "solution"):
                if output_val is None:
                    output_val = v_str
                else:
                    output_val += "\n" + v_str

        # 2. Assemble in strict semantic priority order
        if name_val:
            clean_state["Vulnerability Name"] = name_val
        if family_val:
            clean_state["Plugin Family"] = family_val
        if package_val:
            clean_state["Package"] = package_val

        # 3. Port translation
        if port_val is not None:
            try:
                p_int = int(str(port_val).strip())
                clean_state["Port"] = PORT_MAP.get(p_int, f"{p_int} (Port {p_int})")
            except (ValueError, TypeError):
                clean_state["Port"] = str(port_val)

        # 4. Truncate plugin output / description to prevent prompt bloat
        if output_val:
            if len(output_val) > max_output_chars:
                clean_state["Plugin Output"] = output_val[:max_output_chars] + "..."
            else:
                clean_state["Plugin Output"] = output_val

        # Fallback: if none of the standard keys matched, keep non-bias keys
        if not clean_state:
            for k, v in row.items():
                k_norm = k.lower().strip()
                if k_norm not in BIAS_COLUMNS_TO_DROP and v is not None and str(v).strip():
                    clean_state[k] = str(v).strip()[:max_output_chars]

        return clean_state

    @staticmethod
    def apply_decision_gates(
        row: Dict[str, Any],
        answers: Dict[str, Any],
        confidence_threshold: float = 0.60,
        top_prob_threshold: float = 0.60,
        margin_threshold: Optional[float] = None,
        flat_noise_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Applies model confidence and top probability thresholds in code to determine classification vs review.
        Preserves winning technical category tab without demoting to 'other'.
        Merges results back onto the original row so all raw columns (IP, Host, Severity) remain intact.
        """
        result_row = dict(row)

        # Extract category answer - supports vulnerability_category, layer, category, or first choice question
        ans = (
            answers.get("vulnerability_category")
            or answers.get("layer")
            or answers.get("category")
        )
        if not ans and isinstance(answers, dict):
            for k, v in answers.items():
                if isinstance(v, dict) and "choice" in v:
                    ans = v
                    break

        if not isinstance(ans, dict) or not ans:
            # Fallback if answer is missing
            result_row["category"] = "other"
            result_row["jev_status"] = "needs_review"
            result_row["jev_review_reason"] = "Missing category response from model"
            return result_row

        # Store individual question answers directly on the row
        for q_id, q_eval in answers.items():
            if isinstance(q_eval, dict) and "choice" in q_eval:
                result_row[q_id] = q_eval.get("choice")

        raw_choice = str(ans.get("choice") or "").strip().lower()
        probs = ans.get("probabilities", {}) or {}

        # Rank probabilities
        ranked: List[Tuple[str, float]] = []
        if isinstance(probs, dict) and probs:
            for k, v in probs.items():
                try:
                    ranked.append((str(k).lower(), float(v)))
                except (ValueError, TypeError):
                    pass
            ranked.sort(key=lambda kv: kv[1], reverse=True)

        top_choice = raw_choice
        top_prob = 0.0
        runner_up = None
        runner_prob = 0.0
        margin = 0.0

        if ranked:
            top_choice, top_prob = ranked[0]
            if len(ranked) > 1:
                runner_up, runner_prob = ranked[1]
                margin = round(top_prob - runner_prob, 4)
            else:
                margin = top_prob

        # Map to canonical frontend key (e.g. database_engine -> db)
        canonical_category = CATEGORY_CANONICAL_MAP.get(top_choice, "other")
        calibrated_conf = float(ans.get("confidence") or top_prob)

        # Store metadata - raw AI confidence number is preserved exactly as reported by the model
        result_row["category"] = canonical_category
        result_row["confidence"] = calibrated_conf
        result_row["category_confidence"] = calibrated_conf
        result_row["vulnerability_category"] = canonical_category
        result_row["vulnerability_category_confidence"] = calibrated_conf
        result_row["top_probability"] = top_prob
        result_row["vulnerability_category_top_prob"] = top_prob
        result_row["vulnerability_category_margin"] = margin
        result_row["vulnerability_category_runner_up"] = CATEGORY_CANONICAL_MAP.get(runner_up, runner_up) if runner_up else None
        result_row["vulnerability_category_probabilities"] = probs

        # Decision Gating:
        # Tag finding as 'needs_review' under dual OR condition:
        # (model_confidence < confidence_threshold) OR (top_prob < top_prob_threshold)
        # Findings stay in their winning category tab (Preserve Top Category rule).
        is_below_conf = calibrated_conf < confidence_threshold
        is_below_top_prob = top_prob < top_prob_threshold

        if is_below_conf or is_below_top_prob:
            result_row["jev_status"] = "needs_review"
            reasons = []
            if is_below_conf:
                reasons.append(f"Model confidence {int(calibrated_conf * 100)}% < {int(confidence_threshold * 100)}%")
            if is_below_top_prob:
                reasons.append(f"Top probability {int(top_prob * 100)}% < {int(top_prob_threshold * 100)}%")
            result_row["jev_review_reason"] = "; ".join(reasons)
        else:
            result_row["jev_status"] = "classified"
            result_row["jev_review_reason"] = None

        return result_row

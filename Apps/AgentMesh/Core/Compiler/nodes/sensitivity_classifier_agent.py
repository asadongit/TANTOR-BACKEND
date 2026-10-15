"""
sensitivity_classifier_agent.py
---------------------------------
Detects and labels sensitive data entities (PII, PHI, PCI) within document text.
Groups outputs by category and calculates criticality.
Supports dynamic engine selection and custom entity injection.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================================================
# DATA CLASSIFICATION (PRESIDIO & REGEX)
# ==============================================================================

try:
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    _nlp_config = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
    }
    _provider = NlpEngineProvider(nlp_configuration=_nlp_config)
    _nlp_engine = _provider.create_engine()
    _registry = RecognizerRegistry()
    _registry.load_predefined_recognizers(nlp_engine=_nlp_engine)
    _presidio_engine = AnalyzerEngine(
        nlp_engine=_nlp_engine,
        registry=_registry,
    )
    PRESIDIO_AVAILABLE = True
except Exception as e:
    PRESIDIO_AVAILABLE = False
    logger.warning(f"Presidio not available ({e}). Falling back to regex-only mode.")


import os

RULES_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "..", "Schema", "sensitivity_rules.json")

_CRITICALITY_MAP: Dict[str, str] = {}
_COMPILED_PATTERNS = []
_RULES_LAST_MTIME = 0

def _load_rules():
    global _CRITICALITY_MAP, _COMPILED_PATTERNS, _RULES_LAST_MTIME
    try:
        mtime = os.path.getmtime(RULES_FILE)
        if mtime > _RULES_LAST_MTIME:
            with open(RULES_FILE, "r") as f:
                data = json.load(f)
            _CRITICALITY_MAP.update(data.get("CRITICALITY_MAP", {}))
            
            new_compiled = []
            for item in data.get("REGEX_PATTERNS", []):
                pat = item.get("pattern")
                cat = item.get("category")
                name = item.get("entity_type")
                flags = re.IGNORECASE if "(?i)" in pat else 0
                new_compiled.append((re.compile(pat, flags), cat, name))
            _COMPILED_PATTERNS = new_compiled
            _RULES_LAST_MTIME = mtime
    except Exception as e:
        logger.error(f"Error loading sensitivity rules: {e}")

_load_rules()

def _luhn_valid(number: str) -> bool:
    digits = [int(d) for d in re.sub(r"[\s\-]", "", number)]
    digits.reverse()
    total = 0
    for i, d in enumerate(digits):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0

_PRESIDIO_CATEGORY_MAP: Dict[str, str] = {
    "PERSON": "PII", "EMAIL_ADDRESS": "PII", "PHONE_NUMBER": "PII", "IP_ADDRESS": "PII",
    "LOCATION": "PII", "DATE_TIME": "PII", "IN_AADHAAR": "PII", "IN_PAN": "PII",
    "IN_PASSPORT": "PII", "IN_VOTER_NUMBER": "PII", "CREDIT_CARD": "PCI", "IBAN_CODE": "PCI",
    "MEDICAL_LICENSE": "PHI", "NRP": "PII", "URL": "PII",
}
_PRESIDIO_ENTITIES_OF_INTEREST = list(_PRESIDIO_CATEGORY_MAP.keys())

def _get_criticality(entity_type: str, category: str, custom_crit: str = None) -> str:
    if custom_crit:
        return custom_crit.upper()
    if entity_type in _CRITICALITY_MAP:
        return _CRITICALITY_MAP[entity_type]
    if category in ("PCI", "PHI"):
        return "HIGH"
    return "LOW"

def _detect_with_presidio(text: str) -> List[Dict]:
    results = _presidio_engine.analyze(
        text=text,
        language="en",
        entities=_PRESIDIO_ENTITIES_OF_INTEREST,
        score_threshold=0.6,
    )
    findings = []
    for r in results:
        category = _PRESIDIO_CATEGORY_MAP.get(r.entity_type, "PII")
        findings.append({
            "entity_type": r.entity_type,
            "category": category,
            "start": r.start,
            "end": r.end,
            "score": round(r.score, 3),
            "value": text[r.start: r.end],
            "source": "presidio",
        })
    return findings

def _detect_with_regex(text: str, custom_compiled: List[Tuple] = None) -> List[Dict]:
    _load_rules()
    findings = []
    patterns_to_run = _COMPILED_PATTERNS
    
    # Add custom patterns if provided
    if custom_compiled:
        patterns_to_run = patterns_to_run + custom_compiled

    for compiled_pat, category, entity_type in patterns_to_run:
        for m in compiled_pat.finditer(text):
            value = m.group()
            if entity_type in ("CREDIT_CARD_NUMBER", "AMEX_CARD_NUMBER"):
                if not _luhn_valid(value):
                    continue
            findings.append({
                "entity_type": entity_type,
                "category": category,
                "start": m.start(),
                "end": m.end(),
                "score": 0.92,
                "value": value,
                "source": "regex",
            })
    return findings

def _spans_overlap(a: Dict, b: Dict) -> bool:
    return a["start"] < b["end"] and b["start"] < a["end"]

def _merge_findings(presidio: List[Dict], regex: List[Dict]) -> List[Dict]:
    combined = presidio + regex
    if not combined:
        return []
    combined.sort(key=lambda x: (x["start"], -x["score"]))
    merged: List[Dict] = []
    for item in combined:
        overlaps = [m for m in merged if _spans_overlap(m, item)]
        if not overlaps:
            merged.append(item)
        else:
            for o in overlaps:
                if item["score"] > o["score"]:
                    merged.remove(o)
                    merged.append(item)
                    break
    return merged

def _parse_custom_entities(raw_entities: str) -> List[Tuple]:
    """
    Parses custom entities from the frontend.
    Accepts either JSON array of regexes OR comma-separated keywords.
    Returns list of compiled patterns: (regex_obj, category, entity_type)
    """
    custom_compiled = []
    if not raw_entities or not raw_entities.strip():
        return custom_compiled
    
    try:
        # Try parsing as JSON first (Advanced Mode)
        custom_list = json.loads(raw_entities)
        if isinstance(custom_list, list):
            for item in custom_list:
                pat_str = item.get("pattern", "")
                cat = item.get("category", "CUSTOM")
                ent_type = item.get("entity_type", "CUSTOM_RULE")
                crit = item.get("criticality", "HIGH")
                
                if pat_str:
                    cpat = re.compile(pat_str, re.IGNORECASE)
                    # We store custom criticality mapping dynamically
                    _CRITICALITY_MAP[ent_type] = crit.upper()
                    custom_compiled.append((cpat, cat, ent_type))
            return custom_compiled
    except Exception:
        pass
    
    # Fallback to simple comma-separated keywords (Simple Mode)
    keywords = [k.strip() for k in raw_entities.split(",") if k.strip()]
    for kw in keywords:
        pat_str = r"(?i)\b" + re.escape(kw) + r"\b"
        cpat = re.compile(pat_str)
        ent_type = f"CUSTOM_KEYWORD_{kw.upper()[:10]}"
        _CRITICALITY_MAP[ent_type] = "HIGH"
        custom_compiled.append((cpat, "CUSTOM", ent_type))
        
    return custom_compiled

def detect_sensitive_data(text: str, engine_mode: str = "both", custom_entities_str: str = "") -> Dict[str, Any]:
    presidio_findings: List[Dict] = []
    
    # 1. Parse custom entities
    custom_compiled = _parse_custom_entities(custom_entities_str)
    
    # 2. Run Engines based on mode
    regex_findings = []
    if engine_mode in ["regex", "both"]:
        regex_findings = _detect_with_regex(text, custom_compiled)
        
    if engine_mode in ["presidio", "both"] and PRESIDIO_AVAILABLE:
        try:
            presidio_findings = _detect_with_presidio(text)
        except Exception as e:
            logger.warning(f"Presidio detection error: {e}")
            
    all_findings = _merge_findings(presidio_findings, regex_findings)
    categories_found = list({f["category"] for f in all_findings})

    entity_spans = [
        {
            "entity_type": f["entity_type"],
            "category": f["category"],
            "criticality": _get_criticality(f["entity_type"], f["category"]),
            "start": f["start"],
            "end": f["end"],
            "score": f["score"],
            "source": f["source"],
            "value": f["value"]
        }
        for f in all_findings
    ]
    
    return {
        "sensitivity_categories": categories_found,
        "masking_needed": len(all_findings) > 0,
        "entity_count": len(all_findings),
        "entity_spans": entity_spans,
        "detection_method": engine_mode,
    }


# ==============================================================================
# RICH OUTPUT FORMATTER
# ==============================================================================

# Maps snake_case entity_type to a human-readable field name
_ENTITY_LABEL_MAP: Dict[str, str] = {
    "PAN_NUMBER": "PAN Number",
    "AADHAAR_NUMBER": "Aadhaar Number",
    "IN_AADHAAR": "Aadhaar Number",
    "IN_PAN": "PAN Number",
    "IN_PASSPORT": "Passport Number",
    "IN_VOTER_NUMBER": "Voter ID",
    "PASSPORT_NUMBER": "Passport Number",
    "VOTER_ID": "Voter ID",
    "DRIVING_LICENCE": "Driving Licence",
    "EMAIL_ADDRESS": "Email Address",
    "IN_PHONE_NUMBER": "Mobile Number",
    "PHONE_NUMBER": "Mobile Number",
    "DATE_OF_BIRTH": "Date of Birth",
    "FATHER_NAME": "Father's Name",
    "IP_ADDRESS": "IP Address",
    "MAC_ADDRESS": "MAC Address",
    "US_SSN": "US Social Security Number",
    "UK_NIN": "UK National Insurance Number",
    "CREDIT_CARD_NUMBER": "Credit Card Number",
    "AMEX_CARD_NUMBER": "Amex Card Number",
    "CVV_CODE": "CVV Code",
    "CARD_EXPIRY": "Card Expiry Date",
    "BANK_ACCOUNT_NUMBER": "Bank Account Number",
    "IFSC_CODE": "IFSC Code",
    "IBAN_CODE_GENERAL": "IBAN Code",
    "IBAN_CODE": "IBAN Code",
    "CREDIT_CARD": "Credit Card",
    "ICD10_CODE": "ICD-10 Medical Code",
    "MEDICAL_RECORD_NUMBER": "Medical Record Number",
    "MEDICAL_DOSAGE": "Medical Dosage",
    "MEDICAL_CONDITION": "Medical Condition",
    "BLOOD_GROUP": "Blood Group",
    "PRESCRIPTION": "Prescription",
    "MEDICAL_LICENSE": "Medical License",
    "FINANCIAL_FIGURE": "Financial Figure",
    "EMPLOYER_NAME": "Employer Name",
    "PERSON": "Person Name",
    "LOCATION": "Location / Address",
    "DATE_TIME": "Date / Time",
    "URL": "URL",
    "NRP": "Nationality / Religion / Political Group",
}

def _entity_label(entity_type: str) -> str:
    """Returns human-readable label for an entity type."""
    if entity_type in _ENTITY_LABEL_MAP:
        return _ENTITY_LABEL_MAP[entity_type]
    # Fallback: convert SNAKE_CASE to Title Case
    return entity_type.replace("_", " ").title()

def _score_to_confidence(score: float) -> str:
    """Converts a float confidence score to HIGH / MEDIUM / LOW."""
    if score >= 0.85:
        return "HIGH"
    elif score >= 0.65:
        return "MEDIUM"
    return "LOW"

def _derive_overall_level(risk_levels: List[str], categories: List[str]) -> str:
    """Derives a human-readable overall sensitivity level string."""
    if "HIGH" in risk_levels:
        base = "HIGH"
        if any(c in categories for c in ["PCI", "PHI"]):
            return "RESTRICTED / HIGH"
        return "SENSITIVE / HIGH"
    elif "MEDIUM" in risk_levels:
        return "INTERNAL / MEDIUM"
    elif risk_levels:
        return "LOW"
    return "SAFE"

def _build_rich_output(data_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reshapes the flat detect_sensitive_data() output into the structured
    document-level JSON format with sensitivity_breakdown, tags, and metadata.
    """
    spans = data_result.get("entity_spans", [])
    categories = data_result.get("sensitivity_categories", [])

    # Derive overall risk
    risk_levels = [s["criticality"] for s in spans]
    if "HIGH" in risk_levels:
        overall_risk = "HIGH"
    elif "MEDIUM" in risk_levels:
        overall_risk = "MEDIUM"
    elif risk_levels:
        overall_risk = "LOW"
    else:
        overall_risk = None

    overall_sensitivity_level = _derive_overall_level(risk_levels, categories)

    # Build reason string
    if spans:
        top_entities = list({s["entity_type"] for s in spans if s["criticality"] == (overall_risk or "LOW")})
        reason = f"Escalated to {overall_risk} due to the presence of: " + ", ".join(top_entities[:3])
        if len(top_entities) > 3:
            reason += " and others."
    else:
        reason = "No sensitive entities detected."

    # Auto-derive tags for downstream catalog taggers
    tags = []
    if "PII" in categories:
        tags.append("contains-pii")
    if "PCI" in categories:
        tags.append("contains-pci")
    if "PHI" in categories:
        tags.append("contains-phi")
    if "CONFIDENTIAL" in categories:
        tags.append("contains-confidential")
    if categories:
        tags.append(f"security-classification:{overall_sensitivity_level.lower().replace(' ', '-')}")
    if not tags:
        tags.append("security-classification:safe")

    # Group entities into sensitivity_breakdown by category
    # Use all standard categories as keys so empty ones are explicit
    standard_categories = ["PII", "PCI", "PHI", "CONFIDENTIAL"]
    breakdown: Dict[str, List] = {cat: [] for cat in standard_categories}

    seen_values: set = set()
    for span in spans:
        cat = span["category"]
        if cat not in breakdown:
            breakdown[cat] = []
        # Deduplicate by entity_type + value
        dedup_key = (span["entity_type"], span["value"])
        if dedup_key in seen_values:
            continue
        seen_values.add(dedup_key)
        breakdown[cat].append({
            "field": _entity_label(span["entity_type"]),
            "entity_type": span["entity_type"],
            "value": span["value"],
            "confidence": _score_to_confidence(span["score"]),
            "criticality": span["criticality"],
            "source": span["source"],
        })

    return {
        "document_type": "GENERAL_DOCUMENT",
        "overall_sensitivity_level": overall_sensitivity_level,
        "reason": reason,
        "tags": tags,
        "masking_needed": data_result["masking_needed"],
        "entity_count": data_result["entity_count"],
        "detection_method": data_result["detection_method"],
        "sensitivity_breakdown": breakdown,
    }


# ==============================================================================
# AGENTMESH NODE BUILDER
# ==============================================================================

def build_sensitivity_classifier_node(node_config) -> Callable:
    def execute(state):
        context = state.get("context", {})
        # Input key is fully configurable from the UI — no hardcoded defaults
        input_key = getattr(node_config, "classification_input_key", None)
        text = context.get(input_key) if input_key else None

        if not text:
            messages = state.get("messages", [])
            if messages:
                last = messages[-1]
                text = getattr(last, "content", None) or (
                    last.get("content") if isinstance(last, dict) else None
                )

        if not text or not str(text).strip():
            error_msg = f"Sensitivity Classifier [{node_config.id}]: No input text found."
            return {
                "messages": [AIMessage(content=error_msg, response_metadata={"active_agent": node_config.id})],
                "active_agent": node_config.id,
                "error": {"message": error_msg, "type": "NoInput"},
            }

        text_str = str(text)

        engine_mode = getattr(node_config, "engine_mode", "both")
        custom_entities = getattr(node_config, "custom_entities", "")

        # 1. Data Classification
        data_result = detect_sensitive_data(text_str, engine_mode, custom_entities)

        output_format = getattr(node_config, "output_format", "markdown")

        # 2. Build Message
        if output_format == "json":
            rich_output = _build_rich_output(data_result)
            summary_msg = f"```json\n{json.dumps(rich_output, indent=2, ensure_ascii=False)}\n```"
        else:
            lines = []
            if data_result["masking_needed"]:
                # Check overall risk
                risk_levels = [s['criticality'] for s in data_result['entity_spans']]
                if "HIGH" in risk_levels:
                    overall = "HIGH"
                elif "MEDIUM" in risk_levels:
                    overall = "MEDIUM"
                else:
                    overall = "LOW"
                    
                # Generate reasoning
                highest_risk_entities = set()
                for span in data_result["entity_spans"]:
                    if span['criticality'] == overall:
                        highest_risk_entities.add(span['entity_type'])
                
                reason = f"Escalated to {overall} due to the presence of: " + ", ".join(list(highest_risk_entities)[:3])
                if len(highest_risk_entities) > 3:
                    reason += " and others."

                lines.append(f"🛡️ **Overall Document Risk:** **{overall}**")
                lines.append(f"📝 **Reason:** *{reason}*")
                lines.append("")
                
                # Group entities by category
                category_groups = {}
                for span in data_result["entity_spans"]:
                    cat = span['category']
                    if cat not in category_groups:
                        category_groups[cat] = []
                    category_groups[cat].append(span)

                for cat, spans in category_groups.items():
                    lines.append(f"**{cat} Entities Identified:**")
                    seen_types = set()
                    for span in spans:
                        if span['entity_type'] not in seen_types:
                            lines.append(f"- {span['entity_type']} ({span['criticality']} Risk)")
                            seen_types.add(span['entity_type'])
                    lines.append("")
                    
                lines.append("*Masking needed/recommended.*")
            else:
                lines.append("🛡️ **Overall Document Risk:** **Safe**")
                lines.append("📝 **Reason:** *No sensitive entities (PII/PHI/PCI) were detected.*")

            summary_msg = "\n".join(lines)

        # 3. Save to context
        output_key = getattr(node_config, "classification_output_key", None) or f"sensitivity_classifier_{node_config.id}"
        
        return {
            "messages": [AIMessage(content=summary_msg, response_metadata={"active_agent": node_config.id})],
            "active_agent": node_config.id,
            "context": {
                output_key: data_result,
                "sensitivity_categories": data_result["sensitivity_categories"],
                "entity_spans": data_result["entity_spans"],
                "masking_needed": data_result["masking_needed"]
            },
        }

    return RunnableLambda(execute)

"""
Data Cleaning Node
Parses JSON dataset, flattens it into a Pandas DataFrame, cleans data based on rules, 
computes fixes/issues metrics, unflattens it back to original nested JSON, 
and returns the cleaned dataset alongside metrics.
"""

import re
import json
import logging
import pandas as pd
from typing import Callable, Any, Dict, List
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from Apps.AgentMesh.Schema.StateSchema import AgentState

logger = logging.getLogger(__name__)

def unflatten_dict(flat_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unflattens a flat dictionary with dot-separated keys back into a nested dictionary.
    Example: {"sender.name": "John"} -> {"sender": {"name": "John"}}
    """
    nested_dict = {}
    for key, value in flat_dict.items():
        parts = key.split('.')
        d = nested_dict
        for part in parts[:-1]:
            if part not in d or not isinstance(d[part], dict):
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return nested_dict

# Cleaning Functions for Individual Cells
def clean_trim_whitespace(val):
    if isinstance(val, str):
        return val.strip()
    return val

def clean_proper_case(val):
    if not isinstance(val, str):
        return val
    words = val.split()
    title_words = []
    for w in words:
        # If the word is entirely uppercase and has length > 1, keep it uppercase
        clean_w = re.sub(r'[^a-zA-Z0-9]', '', w)
        if clean_w.isupper() and len(clean_w) > 1:
            title_words.append(w)
        else:
            title_words.append(w.capitalize())
    return " ".join(title_words)

def clean_strip_special_chars(val):
    if isinstance(val, str):
        return re.sub(r'[^a-zA-Z0-9\s]', '', val)
    return val

# Pattern to detect timestamp/date column names — these must NOT be stripped of special chars
# because ISO 8601 timestamps require '-', ':', '.', '+', 'T' to be valid in Postgres.
_DATETIME_COL_KEYWORDS = [
    'timestamp', 'datetime', 'date', 'time', 'created', 'updated',
    'dob', 'expiry', 'expiration', 'issued', 'valid'
]

# Columns that contain structured identifiers with intentional special chars (dots, colons)
# that must be preserved (e.g. IP addresses: '49.36.15.111')
_STRUCTURED_ID_KEYWORDS = [
    'ip_address', 'ip_addr', 'ipaddress', 'ifsc', 'url', 'uri', 'email',
    'host', 'subnet', 'cidr', 'mac_address'
]

def is_datetime_column(col_name: str) -> bool:
    """Returns True if the column name suggests it holds a date/timestamp value."""
    col_lower = col_name.lower()
    return any(k in col_lower for k in _DATETIME_COL_KEYWORDS)

def is_structured_id_column(col_name: str) -> bool:
    """Returns True if the column holds a structured identifier (IP, IFSC, URL etc.)
    whose special chars (dots, colons) must be preserved."""
    col_lower = col_name.lower()
    return any(k in col_lower for k in _STRUCTURED_ID_KEYWORDS)

def clean_strip_currency(val):
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        currency_symbols = ['$', '₹', '€', '£', '¥', 'INR', 'USD', 'EUR', 'GBP']
        has_currency = any(sym in val for sym in currency_symbols)
        has_number = re.search(r'\d', val)
        if has_number and (has_currency or ',' in val):
            cleaned_str = re.sub(r'[^0-9.-]', '', val)
            try:
                return float(cleaned_str)
            except ValueError:
                return val
    return val

def clean_normalize_category(val):
    if isinstance(val, str):
        s = val.strip().lower()
        s = re.sub(r'[^a-z0-9]+', '_', s)
        s = s.strip('_')
        return s
    return val

def is_categorical_column(col_name: str, series: pd.Series) -> bool:
    """
    Checks if a column should be treated as categorical based on key name suffixes or content.
    """
    col_lower = col_name.lower()
    
    # Suffixes/Keywords that guarantee a field is a categorical code
    cat_keywords = ['status', 'type', 'channel', 'category', 'gender', 'role', 'kyc', 'intent', 'currency']
    if any(col_lower.endswith(key) or f"_{key}" in col_lower or col_lower == key for key in cat_keywords):
        return True
        
    return False

def should_apply_proper_case(col_name: str) -> bool:
    """
    Determines if proper Title Case should be applied to a column based on its name.
    Identifiers, codes, VPAs, emails, phone numbers, IPs, and dates should not be proper cased.
    """
    col_lower = col_name.lower()
    non_proper_keywords = ['id', 'code', 'vpa', 'email', 'phone', 'ip', 'fingerprint', 'url', 'key', 'number', 'uuid', 'time', 'date', 'status', 'type', 'channel', 'currency']
    if any(k in col_lower for k in non_proper_keywords):
        return False
    return True

def is_numeric_column(series: pd.Series) -> bool:
    """
    Detects if a column is numeric-like (contains numbers, floats, or currency formatted strings).
    """
    if pd.api.types.is_numeric_dtype(series):
        return True
        
    non_null = series.dropna()
    if non_null.empty:
        return False
        
    numeric_count = 0
    for val in non_null:
        if isinstance(val, (int, float)):
            numeric_count += 1
        elif isinstance(val, str):
            s = val.strip()
            for sym in ['$', '₹', '€', '£', '¥', 'INR', 'USD', 'EUR', 'GBP']:
                s = s.replace(sym, '')
            s = s.replace(',', '').strip()
            try:
                float(s)
                numeric_count += 1
            except ValueError:
                pass
                
    return (numeric_count / len(non_null)) >= 0.5

RULE_MAP = {
    "GEN_TRIM_WHITESPACE": (clean_trim_whitespace, "Trimmed whitespace", "Leading/trailing whitespace found in '{}'"),
    "GEN_PROPER_CASE": (clean_proper_case, "Converted to proper Title Case", "Value not in Proper Title Case in '{}'"),
    "GEN_STRIP_SPECIAL_CHARS": (clean_strip_special_chars, "Removed special characters", "Special characters found in '{}'"),
    "GEN_STRIP_CURRENCY": (clean_strip_currency, "Stripped currency symbols and parsed as float", "Currency symbol or formatting found in '{}'"),
    "GEN_NORMALIZE_CATEGORY": (clean_normalize_category, "Normalized to lowercase snake_case", "Categorical string not normalized in '{}'")
}

def apply_rule(df: pd.DataFrame, rule_name: str, details: List[Dict[str, Any]]):
    """
    Applies a cleaning rule to the entire DataFrame and tracks changes for auditing.
    """
    if rule_name not in RULE_MAP:
        logger.warning(f"Data Cleaning Node: Rule '{rule_name}' is not supported. Skipping.")
        return

    rule_func, fix_desc, issue_template = RULE_MAP[rule_name]

    for col_name in df.columns:
        is_num = is_numeric_column(df[col_name])
        
        # If the column is numeric, only apply currency parsing, skip all text-only rules
        if is_num and rule_name != "GEN_STRIP_CURRENCY":
            continue
            
        # If the column is NOT numeric, skip currency parsing
        if not is_num and rule_name == "GEN_STRIP_CURRENCY":
            continue

        # Check if the series contains any string type values
        # If it doesn't contain any object/string type and the rule is text-only, skip.
        has_strings = df[col_name].apply(lambda x: isinstance(x, str)).any()
        if not has_strings and rule_name in ["GEN_TRIM_WHITESPACE", "GEN_PROPER_CASE", "GEN_STRIP_SPECIAL_CHARS", "GEN_NORMALIZE_CATEGORY"]:
            continue

        # Smart rule boundaries to prevent name/category conflicts
        is_cat = is_categorical_column(col_name, df[col_name])
        is_dt = is_datetime_column(col_name)
        is_struct_id = is_structured_id_column(col_name)

        if rule_name == "GEN_PROPER_CASE" and (is_cat or not should_apply_proper_case(col_name)):
            # Categories and identifiers/codes should not be proper title-cased
            continue
        if rule_name == "GEN_NORMALIZE_CATEGORY" and not is_cat:
            # Free-text like names/descriptions should not be snake-cased
            continue
        if rule_name == "GEN_STRIP_SPECIAL_CHARS" and is_cat:
            # Skip stripping special characters (like underscores/dashes) from category codes
            continue
        if rule_name == "GEN_STRIP_SPECIAL_CHARS" and is_dt:
            # ✅ CRITICAL FIX: Never strip special chars from timestamp/date columns.
            # ISO 8601 timestamps use '-', ':', '.', '+', 'T' which are essential for
            # Postgres to accept the value. Stripping them produces garbage like
            # '20260610T0652484418320000' which causes ML engine to return data_shape [0,0].
            continue
        if rule_name == "GEN_STRIP_SPECIAL_CHARS" and is_struct_id:
            # ✅ CRITICAL FIX: Never strip dots/colons from IP addresses, IFSC codes etc.
            # '49.36.15.111' must stay as-is — stripping dots gives '493615111' which is wrong.
            continue
        if rule_name == "GEN_NORMALIZE_CATEGORY" and is_dt:
            # Date columns should not be normalized to snake_case
            continue
        if rule_name == "GEN_PROPER_CASE" and is_dt:
            # Date/timestamp columns should never be title-cased
            continue

        # Save a copy of the column before modification
        orig_series = df[col_name].copy()

        # Apply rule element-wise, handling non-null values
        df[col_name] = df[col_name].apply(lambda x: rule_func(x) if pd.notna(x) else x)

        # Detect changes: values differ, and they are not both null
        changed_mask = df[col_name] != orig_series
        changed_mask = changed_mask & df[col_name].notna() & orig_series.notna()

        if changed_mask.any():
            for idx in df[changed_mask].index:
                original_value = orig_series.loc[idx]
                cleaned_value = df[col_name].loc[idx]

                # Prevent recording false matches from floating-point string representations
                if str(original_value) == str(cleaned_value):
                    continue

                details.append({
                    "column": col_name,
                    "row_index": int(idx),
                    "rule": rule_name,
                    "original_value": original_value,
                    "cleaned_value": cleaned_value,
                    "issue": issue_template.format(col_name),
                    "fix": fix_desc
                })

def build_data_cleaning_node(node_config: NodeConfig) -> Callable:
    """
    Builds a Data Cleaning node execution function.
    """
    async def execute_data_cleaning(state: AgentState):
        rules = node_config.data_cleaning_rules or ["GEN_TRIM_WHITESPACE"]
        target = node_config.data_cleaning_target or "latest_message"
        input_key = "canonical_transaction" if getattr(node_config, "data_cleaning_input_key", "") == "raw_transaction" else (node_config.data_cleaning_input_key or "")
        if not input_key: input_key = "enriched_transaction"
        output_key = node_config.data_cleaning_output_key or "cleaned_data"
        if not output_key: output_key = "cleaned_transaction"

        logger.info(f"Data Cleaning Node {node_config.id}: Running on target '{target}' using rules: {rules}")

        raw_input = None
        if target == "latest_message":
            if state.get("messages"):
                raw_input = state["messages"][-1].content
        elif target == "context_data":
            raw_input = state.get("context", {}).get(input_key)

        if not raw_input:
            logger.warning(f"Data Cleaning Node {node_config.id}: No input data found.")
            return {
                "active_agent": node_config.id,
                "messages": [AIMessage(content=f"Data Cleaning Node: No input data found at target '{target}'", name="Data_Cleaning", response_metadata={"active_agent": node_config.id})]
            }

        # Parse JSON input
        if isinstance(raw_input, str):
            try:
                json_data = json.loads(raw_input)
            except json.JSONDecodeError as e:
                logger.error(f"Data Cleaning Node: Input string is not valid JSON: {e}")
                raise ValueError(f"Data Cleaning Node requires a valid JSON string or dictionary as input. Error: {str(e)}")
        else:
            json_data = raw_input

        # Handle single dictionary or list of dictionaries
        is_single = isinstance(json_data, dict)
        if is_single:
            records = [json_data]
        elif isinstance(json_data, list):
            records = json_data
        else:
            raise ValueError("Data Cleaning Node: Input JSON must be a dictionary or a list of dictionaries.")

        # Convert to Pandas DataFrame
        df = pd.json_normalize(records)

        if df.empty:
            return {
                "active_agent": node_config.id,
                "context": {
                    output_key: {
                        "cleaned_data": json_data,
                        "metrics": {
                            "total_issues_found": 0,
                            "total_fixes_applied": 0,
                            "rules_applied": rules,
                            "details": []
                        }
                    }
                },
                "messages": [AIMessage(content="Data Cleaning Node: Input dataset was empty. No cleaning performed.", name="Data_Cleaning", response_metadata={"active_agent": node_config.id})]
            }

        # Apply each rule and track changes
        details = []
        for rule in rules:
            apply_rule(df, rule, details)

        # Replace NaN/NaT values with None for JSON compatibility
        df = df.where(pd.notnull(df), None)
        cleaned_records = df.to_dict(orient="records")

        # Unflatten nested structures
        unflattened_records = [unflatten_dict(rec) for rec in cleaned_records]
        cleaned_data = unflattened_records[0] if is_single else unflattened_records

        # Compile metrics
        total_fixes = len(details)
        metrics = {
            "total_issues_found": total_fixes,
            "total_fixes_applied": total_fixes,
            "rules_applied": rules,
            "details": details
        }

        logger.info(f"Data Cleaning Node {node_config.id} completed. Found & fixed {total_fixes} issues.")

        # Return state updates
        summary_msg = f"Data Cleaning Node: Processed {len(records)} record(s). Found and fixed {total_fixes} issue(s) across rules: {', '.join(rules)}."
        return {
            "active_agent": node_config.id,
            "context": {
                output_key: {
                    "cleaned_data": cleaned_data,
                    "metrics": metrics
                }
            },
            "messages": [AIMessage(content=summary_msg, name="Data_Cleaning", response_metadata={"active_agent": node_config.id})]
        }

    executor = RunnableLambda(execute_data_cleaning).with_retry(stop_after_attempt=3)
    setattr(executor, "name", node_config.id)
    return executor

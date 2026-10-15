"""
JSON Transform Node
Inspired by n8n's Set Node (visual field mapping), Code Node (custom expressions),
Limit / Sort / Remove Duplicates nodes, and JMESPath helpers.

Supports three modes:
  - jmespath    : Query language for JSON (like XPath for JSON). Most powerful & safe.
                  e.g. job_status.tasks[*].task_id
                       job_status.tasks[?status=='SUCCESS'].task_id
                       sort_by(job_status.tasks, &result.metrics.test_auc_roc)[-1]
                  `{{ var }}` interpolates a value from the workflow context into the
                  query (strings are quoted, numbers stay bare, objects become JSON
                  literals). Field paths are written plainly, e.g.
                  tasks[?status=='{{ target_status }}'].task_id
  - operations  : Visual no-code operations — extract_field, filter, map, rename, set_value,
                  delete_field, flatten, flatten_dict, pick, omit, rename_map, limit, sort,
                  unique, pattern_extract_identifiers, pattern_extract_field_mappings,
                  text_renderer
  - python_expr : Sandboxed Python list comprehension / expression. Uses 'data' as the input
                  variable. Restricted to safe builtins — no imports, no dunder access,
                  attribute access only on the `data` variable.
                  e.g. [t['task_id'] for t in data['job_status']['tasks']]
                  Also available: `today` ('YYYY-MM-DD') and `now`
                  ('YYYY-MM-DDTHH:MM:SS') as plain strings, for expressions that
                  must know when they are running — document age, expiry, or
                  which financial year has actually finished.
                  e.g. int(today[:4]) - (1 if int(today[5:7]) >= 4 else 2)

Production Features:
  - Zero LLM dependency — pure data transformation
  - Safe sandboxed python_expr (AST-validated: no imports, no dunders, no exec)
  - n8n Set-node parity: set_value supports value_type coercion (string/number/boolean/
    json/null/auto) and `{{ ... }}` value expressions
  - Dot-notation nested path resolution for operations mode
  - Comprehensive error handling with fallback
  - Full context passthrough option
"""

import json
import logging
import ast
import re
from datetime import datetime
from typing import Callable, Any, Dict, List, Optional

import jmespath
from langchain_core.messages import AIMessage

from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from Apps.AgentMesh.Schema.StateSchema import AgentState
from Apps.AgentMesh.Core.Compiler.nodes.utils import interpolate

logger = logging.getLogger(__name__)


# ── Safe builtins for python_expr sandbox ────────────────────────────────────
_SAFE_BUILTINS = {
    "__builtins__": {},
    "list": list, "dict": dict, "set": set, "tuple": tuple,
    "str": str, "int": int, "float": float, "bool": bool,
    "len": len, "range": range, "enumerate": enumerate, "zip": zip,
    "map": map, "filter": filter, "sorted": sorted, "reversed": reversed,
    "sum": sum, "min": min, "max": max, "abs": abs, "round": round,
    "any": any, "all": all, "isinstance": isinstance,
    "True": True, "False": False, "None": None,
}

# Identifiers that must never be *referenced as code* inside python_expr.
# Checked against the parsed syntax tree (see _safe_eval), never against the
# raw expression text: a regex over raw text cannot tell code from data and
# would reject ordinary string literals such as "-From banks" or
# "Loans and advances from related parties".
_FORBIDDEN_NAMES = frozenset({
    "exec", "eval", "compile", "open", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "__import__", "input", "breakpoint",
    "memoryview", "object", "type", "super", "classmethod", "staticmethod",
})


def _get_nested(obj: Any, path: str) -> Any:
    """
    Resolve a dot-notation path against a dict/list structure.
    e.g. 'job_status.tasks' on a nested dict.
    Supports numeric indices: 'tasks.0.task_id'
    Includes case-insensitive fallback for dict keys.
    """
    parts = path.split(".")
    current = obj
    for part in parts:
        if current is None:
            return None
        if isinstance(current, dict):
            val = current.get(part)
            if val is None and part not in current:
                part_lower = part.lower()
                for k in current:
                    if isinstance(k, str) and k.lower() == part_lower:
                        val = current[k]
                        break
            current = val
        elif isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return current


def _set_nested(obj: Dict, path: str, value: Any) -> Dict:
    """Set a value at a dot-notation path, creating intermediate dicts as needed."""
    parts = path.split(".")
    current = obj
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value
    return obj


def _matches_condition(item_val: Any, operator: str, target_val: Any) -> bool:
    """Helper to evaluate conditions safely with type-coercion fallbacks."""
    if item_val is None or target_val is None:
        if operator == "==":
            return item_val == target_val
        elif operator == "!=":
            return item_val != target_val
        return False

    if operator in ("==", "!="):
        # Handle boolean conversion checks
        is_item_bool = isinstance(item_val, bool)
        is_target_bool = isinstance(target_val, bool) or (isinstance(target_val, str) and target_val.lower() in ("true", "false"))
        
        if is_item_bool or is_target_bool:
            def to_bool(v):
                if isinstance(v, bool):
                    return v
                return str(v).lower() in ("true", "1", "yes")
            match = to_bool(item_val) == to_bool(target_val)
            return match if operator == "==" else not match

        # Standard string check
        match = str(item_val) == str(target_val)
        return match if operator == "==" else not match

    elif operator == "contains":
        return str(target_val).lower() in str(item_val).lower()

    elif operator in (">", "<", ">=", "<="):
        try:
            f_item = float(str(item_val))
            f_target = float(str(target_val))
            if operator == ">": return f_item > f_target
            if operator == "<": return f_item < f_target
            if operator == ">=": return f_item >= f_target
            if operator == "<=": return f_item <= f_target
        except (ValueError, TypeError):
            logger.warning(f"[JSON Transform] filter: cannot compare {item_val} and {target_val} as numbers")
            return False
    return False


def _flatten_dict(data: Any, separator: str = "/", parent_key: str = "", max_depth: int = 20) -> Any:
    """Recursively flatten a nested dictionary."""
    if not isinstance(data, dict):
        return data
    items = {}
    if max_depth <= 0:
        items[parent_key or "data"] = str(data)
        return items

    for key, value in data.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key
        if isinstance(value, dict):
            items.update(_flatten_dict(value, separator, new_key, max_depth - 1))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    items.update(_flatten_dict(item, separator, f"{new_key}{separator}{i}", max_depth - 1))
                elif isinstance(item, list):
                    items.update(_flatten_dict({"_": item}, separator, f"{new_key}{separator}{i}", max_depth - 1))
                else:
                    items[f"{new_key}{separator}{i}"] = item
        else:
            items[new_key] = value
    return items


def _render_as_text(
    data: Any,
    separator: str = ": ",
    indent: str = "  ",
    max_depth: int = 20,
    _level: int = 0,
) -> str:
    """Render structured data as human-readable indented text.

    Used by the `text_renderer` operation so a JSON branch can produce the same
    kind of plain text a document branch produces after OCR, letting both
    converge on one downstream shape.

    Keys are rendered in their original order — for financial data the order is
    meaningful (a statement reads top to bottom), so sorting would destroy
    information a reader relies on.
    """
    pad = indent * _level

    if _level >= max_depth:
        return f"{pad}{data}"

    if isinstance(data, dict):
        lines = []
        for key, value in data.items():
            label = str(key).replace("_", " ").strip()
            if isinstance(value, (dict, list)) and value:
                lines.append(f"{pad}{label}{separator.rstrip()}")
                lines.append(_render_as_text(value, separator, indent, max_depth, _level + 1))
            else:
                lines.append(f"{pad}{label}{separator}{_scalar_to_text(value)}")
        return "\n".join(lines)

    if isinstance(data, list):
        # A list of uniform flat records is a TABLE. Rendering it as repeated
        # key: value blocks restates every field name on every row, which on
        # real data produces output LARGER than the JSON it came from — measured
        # at 1.2x on a 24-month GST series. A table states each name once.
        table = _try_render_table(data, pad)
        if table is not None:
            return table

        lines = []
        for i, item in enumerate(data, start=1):
            if isinstance(item, (dict, list)) and item:
                lines.append(f"{pad}[{i}]")
                lines.append(_render_as_text(item, separator, indent, max_depth, _level + 1))
            else:
                lines.append(f"{pad}[{i}] {_scalar_to_text(item)}")
        return "\n".join(lines)

    return f"{pad}{_scalar_to_text(data)}"


def _try_render_table(rows: Any, pad: str = "", min_rows: int = 2):
    """Render a list of flat records as an aligned table, or None if unsuitable.

    Returns None — leaving the caller to fall back — for anything that is not a
    genuine table, rather than forcing a shape onto data that does not have one:

      - fewer than `min_rows` records (a header costs more than it saves)
      - any element that is not a dict (a list of scalars is not a table)
      - any value that is itself a dict or list (a cell cannot hold a subtree)

    Columns are the union of keys across rows, in first-seen order, so records
    that omit a field still line up instead of shifting later columns.
    """
    if not isinstance(rows, list) or len(rows) < min_rows:
        return None
    if not all(isinstance(r, dict) and r for r in rows):
        return None

    columns = []
    for r in rows:
        for k in r:
            if k not in columns:
                columns.append(k)
        for v in r.values():
            if isinstance(v, (dict, list)):
                return None  # nested structure — a table would lose it

    header = [str(c).replace("_", " ").strip() for c in columns]
    body = [[_scalar_to_text(r.get(c)) for c in columns] for r in rows]

    widths = [len(h) for h in header]
    for row in body:
        for i, cell in enumerate(row):
            if len(cell) > widths[i]:
                widths[i] = len(cell)

    def line(cells):
        return pad + "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells)).rstrip()

    return "\n".join([line(header)] + [line(r) for r in body])


def _scalar_to_text(value: Any) -> str:
    """Render a leaf value. None becomes an empty string rather than 'None',
    which would otherwise be read as document content by anything downstream."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _key_matches(key: str, keywords: List[str], strategy: str) -> bool:
    """
    Check if a dictionary key matches any of the keywords using the given strategy.

    Strategies:
        - contains:    Any keyword found as substring of key (case-insensitive)
        - exact:       Key exactly equals any keyword (case-insensitive)
        - starts_with: Key starts with any keyword (case-insensitive)
        - ends_with:   Key ends with any keyword (case-insensitive)
        - regex:       Any keyword treated as regex pattern

    Args:
        key: The dictionary key to test (e.g., "basic_info/pan_number")
        keywords: List of search terms (e.g., ["pan", "pan_number"])
        strategy: Matching strategy name

    Returns:
        True if key matches any keyword with the given strategy
    """
    key_lower = key.lower()

    for keyword in keywords:
        kw_lower = keyword.lower().strip()
        if not kw_lower:
            continue

        if strategy == "exact":
            # For exact match on flattened keys, also check the last segment
            # e.g., "basic_info/pan_number" should match keyword "pan_number"
            if key_lower == kw_lower:
                return True
            segments = key_lower.split("/")
            if segments and segments[-1] == kw_lower:
                return True
            # Also try dot-separated segments
            segments = key_lower.split(".")
            if segments and segments[-1] == kw_lower:
                return True

        elif strategy == "contains":
            if kw_lower in key_lower:
                return True

        elif strategy == "starts_with":
            if key_lower.startswith(kw_lower):
                return True
            # Check last segment too
            segments = key_lower.rsplit("/", 1)
            if len(segments) > 1 and segments[-1].startswith(kw_lower):
                return True

        elif strategy == "ends_with":
            if key_lower.endswith(kw_lower):
                return True

        elif strategy == "regex":
            try:
                if re.search(kw_lower, key_lower, re.IGNORECASE):
                    return True
            except re.error:
                logger.warning(f"[JSON_TRANSFORM] Invalid regex pattern: '{keyword}'")

    return False


def _extract_identifiers(
    flat_data: Dict[str, Any],
    patterns: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Scan a flat dictionary for values matching identifier patterns.

    Each pattern config:
    {
        "concept": "PAN",
        "display_label": "PAN Number",
        "search_keywords": ["pan", "pan_number", "permanent_account"],
        "match_strategy": "contains",
        "is_required": true
    }

    Returns list of {"label": "PAN Number", "value": "ABCDE1234F", "concept": "PAN", "source_key": "..."}
    Deduplicates by value to avoid reporting the same PAN from multiple keys.
    """
    results = []

    for pattern_idx, pattern in enumerate(patterns):
        concept = pattern.get("concept", "Unknown")
        display_label = pattern.get("display_label", concept)
        keywords = pattern.get("search_keywords", [])
        strategy = pattern.get("match_strategy", "contains")
        is_required = pattern.get("is_required", False)

        if not keywords:
            if is_required:
                results.append({
                    "label": display_label,
                    "value": None,
                    "concept": concept,
                    "source_key": None,
                    "status": "missing_no_keywords"
                })
            continue

        found = False
        # Per-pattern dedup: two patterns may intentionally search for the same
        # concept (e.g. "PAN" via different keyword sets), so a value seen by
        # pattern A must not suppress the same value from pattern B. Within a
        # single pattern we still dedup so one value isn't reported twice when
        # it matches several keys.
        seen_values = set()
        for key, value in flat_data.items():
            if _key_matches(key, keywords, strategy):
                # Normalize value to string for display
                str_value = str(value).strip() if value is not None else ""
                if not str_value:
                    continue

                # Dedup by value within this pattern
                dedup_key = f"{concept}:{str_value}"
                if dedup_key in seen_values:
                    continue
                seen_values.add(dedup_key)

                results.append({
                    "label": display_label,
                    "value": str_value,
                    "concept": concept,
                    "source_key": key,
                    "status": "found"
                })
                found = True

        if not found and is_required:
            results.append({
                "label": display_label,
                "value": None,
                "concept": concept,
                "source_key": None,
                "status": "missing"
            })

    return results


def _interpolate_op_classes(op: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve {{ }} references in each pattern's target_class.

    Class names have to match across the classification node, the agent routes
    and the scorecard; letting them come from one workflow variable removes the
    hand-copying that makes a typo silently drop data.

    Deliberately narrow: only target_class is interpolated. Other pattern fields
    hold regexes and JMESPath fragments where braces are syntax, not templates,
    so rendering them could change a working expression.
    """
    patterns = op.get("patterns")
    if not isinstance(patterns, list):
        return op

    resolved_patterns = []
    changed = False
    for pat in patterns:
        if not isinstance(pat, dict):
            resolved_patterns.append(pat)
            continue
        raw = pat.get("target_class")
        if isinstance(raw, str) and "{{" in raw:
            rendered = interpolate(raw, context)
            if not str(rendered).strip():
                # Jinja renders an undefined variable as empty. An empty
                # target_class would silently group fields under "" — invisible
                # to every downstream consumer.
                logger.warning(
                    f"[JSON Transform] target_class '{raw}' resolved to empty — "
                    f"the variable it references is not defined in this run's "
                    f"context. Fields matched by this pattern would be ungrouped."
                )
            pat = {**pat, "target_class": rendered}
            changed = True
        resolved_patterns.append(pat)

    if not changed:
        return op
    return {**op, "patterns": resolved_patterns}


def _extract_field_mappings(
    flat_data: Dict[str, Any],
    patterns: List[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Scan a flat dictionary and group matched fields by target_class.

    Enforces 1-to-1 mapping: each pattern extracts at most ONE field
    (the first match). This ensures the LLM receives unambiguous,
    clearly labeled data — identical to the standalone CD approach.

    Each pattern config (1 keyword → 1 field):
    {
        "concept": "annual_income",
        "display_label": "Annual Income",
        "search_keywords": ["annual_income"],
        "match_strategy": "contains",
        "target_class": "Financial_Analysis"
    }

    Returns grouped dict:
    {
        "Financial_Analysis": [
            {"display_label": "Annual Income", "value": "500000", "concept": "annual_income", "source_key": "..."},
            {"display_label": "Credit Score", "value": "750", ...}
        ],
        "Tax_GST": [...]
    }
    """
    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for pattern in patterns:
        concept = pattern.get("concept", "Unknown")
        display_label = pattern.get("display_label", concept)
        keywords = pattern.get("search_keywords", [])
        strategy = pattern.get("match_strategy", "contains")
        target_class = pattern.get("target_class", "DEFAULT")

        if not keywords:
            continue

        for key, value in flat_data.items():
            if _key_matches(key, keywords, strategy):
                str_value = str(value).strip() if value is not None else ""
                if not str_value:
                    continue

                entry = {
                    "display_label": display_label,
                    "value": str_value,
                    "concept": concept,
                    "source_key": key,
                }

                if target_class not in grouped:
                    grouped[target_class] = []
                grouped[target_class].append(entry)

                # Enforce 1-to-1 mapping: stop searching after the first match for this pattern
                break

    return grouped


def _pick_fields(data: Any, fields: List[str]) -> Dict[str, Any]:
    """Extract only specific fields from a dict using dot-notation paths."""
    if not isinstance(data, dict):
        return {}
    result = {}
    for field_path in fields:
        val = _get_nested(data, field_path)
        if val is not None:
            result[field_path] = val
    return result


def _rename_keys(data: Any, rename_map: Dict[str, str]) -> Dict[str, Any]:
    """Rename keys in a flat dictionary."""
    if not isinstance(data, dict):
        return {}
    result = {}
    for key, value in data.items():
        new_key = rename_map.get(key, key)
        result[new_key] = value
    return result


def _omit_keys(data: Any, keys_to_omit: List[str]) -> Dict[str, Any]:
    """Remove specific keys from a dictionary."""
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if k not in keys_to_omit}


def _coerce_value(value: Any, value_type: Optional[str] = None) -> Any:
    """
    Coerce a raw value to a declared type (n8n Set-node field-type parity).

    value_type may be:
      - 'string'  : force str
      - 'number'  : parse int/float
      - 'boolean' : parse true/false/1/0/yes/no
      - 'json'    : parse JSON (object/array/number/bool/null)
      - 'null'    : explicit None
      - None/'auto': best-effort inference (string → bool/int/float/json if parseable)

    Unparseable coercions fall back to the raw value and log a warning rather
    than raising, keeping a partially-bad operation from killing the workflow.
    """
    if value is None:
        return None
    t = (value_type or "auto").lower()

    if t == "string":
        return str(value)

    if t == "number":
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value
        try:
            return int(value)
        except (ValueError, TypeError):
            try:
                return float(value)
            except (ValueError, TypeError):
                logger.warning(f"[JSON Transform] set_value: cannot coerce {value!r} to number; keeping as-is")
                return value

    if t == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        s = str(value).strip().lower()
        if s in ("true", "1", "yes", "y", "on"):
            return True
        if s in ("false", "0", "no", "n", "off"):
            return False
        logger.warning(f"[JSON Transform] set_value: cannot coerce {value!r} to boolean; keeping as-is")
        return value

    if t == "json":
        if isinstance(value, (dict, list, int, float, bool)):
            return value
        try:
            return json.loads(str(value))
        except (ValueError, TypeError):
            logger.warning(f"[JSON Transform] set_value: cannot parse {value!r} as JSON; keeping as-is")
            return value

    if t == "null":
        return None

    # auto: infer from the raw string
    if isinstance(value, str):
        s = value.strip()
        if s.lower() in ("true", "yes"):
            return True
        if s.lower() in ("false", "no"):
            return False
        if s.lower() in ("null", "none"):
            return None
        if re.fullmatch(r"[-+]?\d+", s):
            return int(s)
        if re.fullmatch(r"[-+]?\d+\.\d+([eE][-+]?\d+)?", s) or re.fullmatch(r"[-+]?\d+[eE][-+]?\d+", s):
            return float(s)
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                return json.loads(s)
            except (ValueError, TypeError):
                pass
    return value


def _interpolate_value(value: Any, scope: Dict[str, Any]) -> Any:
    """Interpolate `{{ ... }}` expressions inside a string value against a scope.

    Only strings containing placeholders are touched; non-string values pass
    through untouched (so a typed value set by _coerce_value is never re-quoted).
    """
    if isinstance(value, str) and "{{" in value:
        rendered = interpolate(value, scope)
        if rendered != value:
            return rendered
    return value


def _apply_operation(data: Any, op: Dict[str, Any]) -> Any:
    """
    Apply a single visual operation to the data.

    Supported operations:
      extract_field  : Pull a nested field out of a dict/list into a new key
      filter         : Filter a list by a simple condition (field == value or field != value)
      map            : Extract a specific field from every item in a list
      rename         : Rename a key at the root level
      set_value      : Set a value at a path. Supports value_type coercion and {{ }} interpolation
      delete_field   : Remove a key from the root level
      flatten        : Flatten a list of lists into a single list
      flatten_dict   : Recursively flattens nested dictionaries
      pick           : Keep only specific fields
      omit           : Remove specific keys
      rename_map     : Rename multiple keys using a dictionary mapping
      limit          : Keep only the first N items of a list (n8n Limit parity)
      sort           : Sort a list by a field (asc/desc, string/number; n8n Sort parity)
      unique         : Remove duplicate items from a list by field or full value (n8n Remove Duplicates parity)
      pattern_extract_identifiers      : Scan flattened data for values matching identifier patterns
      pattern_extract_field_mappings   : Scan flattened data and group matched fields by target_class
      text_renderer                    : Render structured data as readable text, defaulting to
                                         latest_ocr_output so JSON and document branches converge
    """
    op_type = op.get("type", "")
    source_path = op.get("source_path", "")
    field = op.get("field", "")
    destination = op.get("destination", "")
    filter_field = op.get("filter_field", "")
    filter_value = op.get("filter_value", "")
    filter_operator = op.get("filter_operator", "==")

    result = data if isinstance(data, dict) else {}

    if op_type == "extract_field":
        # Pull a nested value to the top level or a new key
        source = _get_nested(data, source_path) if source_path else data
        value = _get_nested(source, field) if field else source
        dest_key = destination or field
        result = dict(result)
        result[dest_key] = value

    elif op_type == "filter":
        # Filter a list at source_path by (filter_field operator filter_value)
        source = _get_nested(data, source_path) if source_path else data
        if isinstance(source, list):
            filtered = []
            for item in source:
                item_val = _get_nested(item, filter_field) if filter_field else item
                if _matches_condition(item_val, filter_operator, filter_value):
                    filtered.append(item)
            dest_key = destination or source_path or "filtered"
            result = dict(result)
            result[dest_key] = filtered
        else:
            logger.warning(f"[JSON Transform] filter: source at '{source_path}' is not a list")

    elif op_type == "map":
        # Extract a field or perform object projection from every item in a list
        source = _get_nested(data, source_path) if source_path else data
        if isinstance(source, list):
            projections = op.get("projections", [])
            if projections:
                mapped = []
                for item in source:
                    new_item = {}
                    for proj in projections:
                        target = proj.get("target_path", "")
                        src_field = proj.get("source_field", "")
                        if target and src_field:
                            val = _get_nested(item, src_field)
                            _set_nested(new_item, target, val)
                    mapped.append(new_item)
            else:
                mapped = [_get_nested(item, field) for item in source if _get_nested(item, field) is not None]
            dest_key = destination or field or "mapped"
            result = dict(result)
            result[dest_key] = mapped
        else:
            logger.warning(f"[JSON Transform] map: source at '{source_path}' is not a list")

    elif op_type == "rename":
        # Rename a top-level key
        if source_path in result:
            result = dict(result)
            result[destination or field] = result.pop(source_path)

    elif op_type == "set_value":
        # Set a typed/interpolated value (n8n Set-node parity).
        # static_value supports {{ context_key }} interpolation, then value_type
        # coerces the result: string | number | boolean | json | null | auto.
        raw = op.get("static_value")
        interp_scope = data if isinstance(data, dict) else {}
        raw = _interpolate_value(raw, interp_scope)
        typed = _coerce_value(raw, op.get("value_type"))
        result = dict(result)
        _set_nested(result, destination or field, typed)

    elif op_type == "delete_field":
        result = dict(result)
        result.pop(field, None)

    elif op_type == "flatten":
        source = _get_nested(data, source_path) if source_path else data
        if isinstance(source, list):
            flat = [item for sublist in source for item in (sublist if isinstance(sublist, list) else [sublist])]
            dest_key = destination or source_path or "flattened"
            result = dict(result)
            result[dest_key] = flat

    elif op_type == "flatten_dict":
        sep = op.get("separator", "/") or "/"
        result = _flatten_dict(data, sep)

    elif op_type == "pick":
        raw_fields = op.get("fields", [])
        if isinstance(raw_fields, str):
            fields_list = [f.strip() for f in raw_fields.split(",") if f.strip()]
        else:
            fields_list = raw_fields
        result = _pick_fields(data, fields_list)

    elif op_type == "omit":
        raw_keys = op.get("keys", [])
        if isinstance(raw_keys, str):
            keys_list = [k.strip() for k in raw_keys.split(",") if k.strip()]
        else:
            keys_list = raw_keys
        result = _omit_keys(data, keys_list)

    elif op_type == "rename_map":
        remap = op.get("rename_map", {})
        result = _rename_keys(data, remap)

    elif op_type == "limit":
        # Keep only the first N items of a list (n8n Limit parity)
        source = _get_nested(data, source_path) if source_path else data
        if isinstance(source, list):
            try:
                count = int(op.get("limit", 1) or 1)
            except (ValueError, TypeError):
                count = 1
            count = max(0, count)
            dest_key = destination or source_path or "limited"
            result = dict(result)
            result[dest_key] = source[:count]
        else:
            logger.warning(f"[JSON Transform] limit: source at '{source_path}' is not a list")

    elif op_type == "sort":
        # Sort a list by field (n8n Sort parity)
        source = _get_nested(data, source_path) if source_path else data
        if isinstance(source, list):
            sort_field = op.get("sort_field", "")
            sort_order = str(op.get("sort_order", "asc") or "asc").lower()
            sort_type = str(op.get("sort_type", "string") or "string").lower()
            reverse = sort_order in ("desc", "descending", "-1")

            def _sort_key(item):
                val = _get_nested(item, sort_field) if sort_field else item
                if sort_type == "number":
                    try:
                        return float(str(val))
                    except (ValueError, TypeError):
                        return float("inf")
                return str(val) if val is not None else ""

            try:
                sorted_list = sorted(source, key=_sort_key, reverse=reverse)
            except TypeError:
                logger.warning(f"[JSON Transform] sort: could not sort list at '{source_path}'")
                sorted_list = list(source)
            dest_key = destination or source_path or "sorted"
            result = dict(result)
            result[dest_key] = sorted_list
        else:
            logger.warning(f"[JSON Transform] sort: source at '{source_path}' is not a list")

    elif op_type == "unique":
        # Remove duplicates by a field, or by the full item (n8n Remove Duplicates parity)
        source = _get_nested(data, source_path) if source_path else data
        if isinstance(source, list):
            key_field = op.get("key_field", "")
            seen = set()
            unique_list = []
            for item in source:
                if key_field:
                    key = json.dumps(_get_nested(item, key_field), default=str, sort_keys=True)
                else:
                    try:
                        key = json.dumps(item, default=str, sort_keys=True)
                    except (TypeError, ValueError):
                        key = repr(item)
                if key not in seen:
                    seen.add(key)
                    unique_list.append(item)
            dest_key = destination or source_path or "unique"
            result = dict(result)
            result[dest_key] = unique_list
        else:
            logger.warning(f"[JSON Transform] unique: source at '{source_path}' is not a list")

    elif op_type == "pattern_extract_identifiers":
        # Scan flattened input for values matching identifier patterns
        patterns = op.get("patterns", []) or []
        flat_data = _flatten_dict(data) if isinstance(data, dict) else {}
        extracted = _extract_identifiers(flat_data, patterns)
        dest_key = destination or "identifiers"
        result = dict(result)
        result[dest_key] = extracted

    elif op_type == "pattern_extract_field_mappings":
        # Scan flattened input and group matched fields by target_class
        patterns = op.get("patterns", []) or []
        flat_data = _flatten_dict(data) if isinstance(data, dict) else {}
        mappings = _extract_field_mappings(flat_data, patterns)
        dest_key = destination or "field_mappings"
        result = dict(result)
        result[dest_key] = mappings

        # ── Source-match check ────────────────────────────────────────────
        # A payload that matches nothing is almost always an unconfigured
        # source shape: a new provider whose keys nobody has mapped yet. The
        # extraction then yields an empty bucket and the pipeline carries on
        # as though the document held no data — a silent loss, and the exact
        # failure mode this system already has elsewhere.
        #
        # Report per class, because partial failure matters: GST patterns
        # matching while balance-sheet patterns find nothing is a different
        # problem from the whole payload being unrecognised.
        # Only count patterns that can actually match: _extract_field_mappings
        # skips any pattern with no search_keywords, so counting those would
        # warn about a class nobody ever asked this node to find.
        expected = set()
        for pat in patterns:
            if isinstance(pat, dict) and pat.get("search_keywords"):
                expected.add(str(pat.get("target_class") or "DEFAULT"))
        found = {k for k, v in (mappings or {}).items() if v}
        for cls in sorted(expected - found):
            logger.warning(
                f"[JSON Transform] class '{cls}': no field matched. "
                f"Either this payload does not contain that data, or its keys "
                f"are not mapped for this source."
            )
        if expected and not found:
            logger.warning(
                f"[JSON Transform] NO patterns matched anything across "
                f"{len(expected)} class(es) — this source's shape is very likely "
                f"unconfigured. Flattened input had {len(flat_data)} key(s)."
            )

    elif op_type == "text_renderer":
        # Render structured data as readable text.
        #
        # Restores the capability of the removed pattern_extract node's
        # "text_renderer" mode. Its purpose is convergence: a pipeline that
        # accepts both documents and structured data needs both to arrive at
        # downstream nodes in the SAME shape, so nothing after this point has to
        # know which one it was given.
        #
        # The default destination is "latest_ocr_output" because that is where
        # ocr_node writes (ocr_node.py, destination_key default) and what
        # classification reads (classification_input_key default). Writing there
        # makes a JSON branch indistinguishable from a document branch.
        source = _get_nested(data, source_path) if source_path else data
        text = _render_as_text(
            source,
            separator=op.get("separator", ": "),
            indent=op.get("indent", "  "),
            max_depth=int(op.get("max_depth", 20) or 20),
        )
        dest_key = destination or "latest_ocr_output"
        result = dict(result)
        result[dest_key] = text
        # Optional: name the source so downstream can report where text came from,
        # mirroring ocr_node's "latest_ocr_file_name".
        source_name = op.get("source_name")
        if source_name:
            result["latest_ocr_file_name"] = source_name

    # destination "$" — return the value itself instead of a dict wrapping it.
    #
    # Every operation above files its output under `destination`, so an operation
    # that produces one clean value still hands back the whole payload with that
    # value attached. Downstream keys then have to reach through a wrapper, and
    # the untouched payload rides along in state into every checkpoint and trace.
    #
    # "$" is generic dataflow, not domain logic: give me the value, not a box
    # around it. Because each operation already wrote to `destination`, the value
    # is sitting at result["$"] and one unwrap here covers all of them — no
    # per-operation changes, so nothing that omits "$" can behave differently.
    #
    # Guarded on the operation having actually declared "$" so a payload that
    # genuinely contains a "$" key is never mistaken for an unwrap request.
    if str(op.get("destination", "")).strip() == "$":
        if isinstance(result, dict) and "$" in result:
            return result["$"]
        logger.warning(
            f"[JSON Transform] operation '{op_type}' requested destination '$' "
            f"but produced no value to unwrap; returning the payload unchanged."
        )

    return result


def _safe_eval(expression: str, data: Any) -> Any:
    """
    Execute a sandboxed Python expression.
    'data' is available as the input variable.
    Only safe builtins are allowed. No imports, no exec, no file access,
    no dunder attribute access, no arbitrary attribute access.
    """
    # Validate it's a valid expression (not a statement)
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid Python expression syntax: {e}")

    # Everything dangerous is rejected from the parsed syntax tree, not from
    # the raw text. This is both stricter (it sees what the code actually
    # does) and correct for data: a string literal such as "-From banks" or
    # a label containing "__" is content, not an instruction, and must pass.
    #
    # Reachable escape vectors from `_SAFE_BUILTINS` + `data` are:
    #   * imports
    #   * dunder attribute traversal  (().__class__.__mro__ ...)
    #   * referencing a dangerous builtin by name
    # Subscripts like data["__class__"] are plain key lookups on JSON data and
    # cannot yield a Python class, so they are intentionally allowed.
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("imports are not allowed in python_expr")
        if isinstance(node, ast.Attribute):
            attr = node.attr
            if attr.startswith("__") or attr.endswith("__"):
                raise ValueError(f"Dunder attribute '{attr}' is not allowed in python_expr")
            if attr in _FORBIDDEN_NAMES:
                raise ValueError(f"Forbidden attribute '{attr}' in python_expr")
        if isinstance(node, ast.Name):
            name = node.id
            if name.startswith("__") or name.endswith("__"):
                raise ValueError(f"Dunder name '{name}' is not allowed in python_expr")
            if name in _FORBIDDEN_NAMES:
                raise ValueError(f"Forbidden name '{name}' in python_expr")

    # The clock, as plain text. Expressions regularly need to know when they are
    # running - whether a document is stale, which financial year has actually
    # finished, how old a balance is - and with imports blocked there is no way
    # to find out. These are strings, not date objects, so they add no attribute
    # surface to traverse and nothing new to escape through; slicing them
    # (today[:4]) is all an expression needs.
    _clock = datetime.now()
    local_env = {
        **_SAFE_BUILTINS,
        "data": data,
        "today": _clock.strftime("%Y-%m-%d"),
        "now": _clock.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return eval(expression, {"__builtins__": {}}, local_env)  # noqa: S307


def _resolve_jmespath_vars(query: str, context: Dict[str, Any]) -> str:
    """
    Substitute `{{ ... }}` placeholders in a JMESPath query.

    n8n-style semantics:
      - If the expression resolves to a value in `context`, its VALUE is embedded:
          * string  -> single-quoted literal  (tasks[?status=={{status}}])
          * number  -> bare literal            (tasks[?age > {{min_age}}])
          * bool    -> true/false
          * dict/list -> JSON literal
      - If the placeholder is already wrapped in quotes by the author
        (tasks[?status=='{{status}}']), those quotes are kept and the value is
        embedded verbatim — no double quoting.
      - If it does NOT resolve in context, the placeholder is treated as a
        field path into the input data (backward-compatible with the old
        unwrap behaviour): {{ job_status.tasks }} -> job_status.tasks
    """
    def _repl(match):
        expr = match.group(2).strip()
        if not expr:
            return match.group(0)
        try:
            val = _get_nested(context, expr)
            if val is not None:
                # Author already wrapped the placeholder in quotes -> embed verbatim.
                if match.group(1) and match.group(3) and match.group(1) == match.group(3):
                    return match.group(1) + str(val) + match.group(3)
                if isinstance(val, bool):
                    return "true" if val else "false"
                if isinstance(val, (int, float)):
                    return str(val)
                if isinstance(val, str):
                    return "'" + val.replace("'", "''") + "'"
                # dict / list -> JSON literal
                return json.dumps(val, default=str)
        except Exception:
            pass
        # Not in context -> keep as a field path into input_data
        return expr

    return re.sub(r"(['\"]?)\{\{\s*(.*?)\s*\}\}(['\"]?)", _repl, query)


# Keys that together identify a record produced by multi_file_ingestion.
# All three are required deliberately: matching on file_path alone would let a
# crafted webhook payload make this node read an arbitrary .json off the server
# and emit it into the workflow output. Only multi_file_ingestion emits this
# exact shape.
_INGESTION_ITEM_KEYS = ("file_name", "file_path", "extension")


def _load_ingested_json_file(value: Any, node_id: str) -> Any:
    """
    Replace an ingested .json file record with the file's parsed contents.

    This node's three historical inputs — a webhook body, an API response and
    an ingestion_router child payload — all arrive already in memory. An
    uploaded file does not: ingestion records only where it sits on disk, so
    without this the node would transform the file's METADATA (file_name,
    sha256_hash, …) and correctly report finding no data in it.

    Anything that is not an ingestion .json record is returned untouched, so
    the existing inputs are unaffected.
    """
    if not isinstance(value, dict):
        return value
    if not all(k in value for k in _INGESTION_ITEM_KEYS):
        return value
    if str(value.get("extension") or "").strip().lower() != ".json":
        return value

    path = value.get("file_path")
    if not isinstance(path, str) or not path.strip():
        return value

    try:
        # utf-8-sig, not utf-8: JSON exported by Windows tooling often carries a
        # BOM, which plain utf-8 leaves in place and json.load then rejects with
        # "Expecting value: line 1 column 1". utf-8-sig strips it when present
        # and behaves identically when it is not.
        with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
            loaded = json.load(fh)
    except FileNotFoundError:
        logger.warning(
            f"[{node_id}] JSON Transform: ingested file '{value.get('file_name')}' "
            f"not found at '{path}'. Leaving the file record unchanged."
        )
        return value
    except json.JSONDecodeError as e:
        logger.warning(
            f"[{node_id}] JSON Transform: ingested file '{value.get('file_name')}' "
            f"is not valid JSON ({e}). Leaving the file record unchanged."
        )
        return value
    except Exception as e:
        logger.warning(
            f"[{node_id}] JSON Transform: could not read ingested file "
            f"'{value.get('file_name')}' ({type(e).__name__}: {e}). "
            f"Leaving the file record unchanged."
        )
        return value

    logger.info(
        f"[{node_id}] JSON Transform: loaded ingested file "
        f"'{value.get('file_name')}' → {type(loaded).__name__}"
    )
    return loaded


def build_json_transform_node(node_config: NodeConfig) -> Callable:
    """
    Build a JSON Transform node that filters, extracts, maps, and reshapes
    JSON data from the workflow context.
    """
    # Standard json_transform config keys
    jt_input_key = getattr(node_config, "jt_input_key", "")
    if jt_input_key is None:
        jt_input_key = ""
    jt_output_key = getattr(node_config, "jt_output_key", None) or "transformed_data"
    jt_mode = getattr(node_config, "jt_mode", None) or "jmespath"
    jt_query = getattr(node_config, "jt_query", None) or ""
    jt_operations = getattr(node_config, "jt_operations", None) or []
    jt_expression = getattr(node_config, "jt_expression", None) or ""
    jt_pass_through = getattr(node_config, "jt_pass_through", False)

    async def json_transform(state: AgentState):
        context: Dict[str, Any] = state.get("context", {}) or {}

        # ── Resolve input data with smart fallbacks ───────────────────────────
        input_key_str = (jt_input_key or "").strip()
        if not input_key_str or input_key_str.lower() in ("context", "root"):
            input_data = context
        else:
            input_data = _get_nested(context, input_key_str)
            if input_data is None:
                # Try fallback locations
                if input_key_str != "payload":
                    input_data = _get_nested(context, "payload")
                if input_data is None and input_key_str != "latest_api_result":
                    input_data = _get_nested(context, "latest_api_result")
            if input_data is None:
                logger.warning(
                    f"[{node_config.id}] JSON Transform: input key '{input_key_str}' not found in context. Falling back to full context."
                )
                input_data = context

        # --- Ingestion Router Support ---
        if isinstance(input_data, dict) and "json_data" in input_data:
            input_data = input_data["json_data"]

        # --- Uploaded .json file support (mixed batches) ---
        # Runs after the router unwrap above, so a child payload that has
        # already been parsed is never re-read from disk.
        input_data = _load_ingested_json_file(input_data, node_config.id)

        # --- Parse raw JSON string or inner JSON strings if passed ---
        def _try_parse_json(val):
            if isinstance(val, str):
                s = val.strip()
                if s.startswith("```"):
                    import re
                    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
                    s = re.sub(r"\s*```$", "", s).strip()
                if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
                    try:
                        return json.loads(s)
                    except Exception:
                        pass
            return val

        input_data = _try_parse_json(input_data)
        if isinstance(input_data, list):
            input_data = [_try_parse_json(x) for x in input_data]

        logger.info(
            f"[{node_config.id}] JSON Transform | mode={jt_mode} | "
            f"input_key='{input_key_str}' | output_key={jt_output_key}"
        )

        # ── Execute transformation ────────────────────────────────────────────
        result = None
        try:
            if jt_mode == "jmespath":
                raw_query = jt_query.strip()
                if not raw_query:
                    raise ValueError("JMESPath query is empty. Please configure jt_query.")

                # n8n-style value interpolation: `{{ var }}` becomes the context
                # VALUE (quoted string / bare number / JSON literal), so value
                # comparisons like tasks[?age > {{min_age}}] work. Unresolvable
                # vars fall back to plain field paths (job_status.tasks).
                clean_query = _resolve_jmespath_vars(raw_query, context)

                result = jmespath.search(clean_query, input_data)
                logger.info(f"[{node_config.id}] JMESPath query '{clean_query}' → {type(result).__name__}")

            elif jt_mode == "operations":
                if not jt_operations:
                    raise ValueError("Operations list is empty. Please configure jt_operations.")
                # An operation using destination "$" replaces the payload with a
                # single value, so any operation after it receives that value
                # instead of the data it expects. Warn once, up front, rather
                # than letting it surface as a confusing failure further down.
                enabled = [j for j, o in enumerate(jt_operations) if o.get("disabled") is not True]
                if enabled:
                    last_enabled = enabled[-1]
                    for j in enabled[:-1]:
                        if str(jt_operations[j].get("destination", "")).strip() == "$":
                            logger.warning(
                                f"[{node_config.id}] operation [{j}] "
                                f"('{jt_operations[j].get('type', 'unknown')}') uses destination "
                                f"'$', which replaces the payload, but it is not the last enabled "
                                f"operation (that is [{last_enabled}]). Operations after it will "
                                f"receive a single value instead of the payload."
                            )

                current = input_data
                for i, op in enumerate(jt_operations):
                    op_type = op.get("type", "unknown")
                    if op.get("disabled") is True:
                        logger.info(f"[{node_config.id}] Skipping disabled operation [{i}]: {op_type}")
                        continue
                    logger.debug(f"[{node_config.id}] Applying operation [{i}]: {op_type}")
                    current = _apply_operation(current, _interpolate_op_classes(op, context))
                result = current
                logger.info(f"[{node_config.id}] Applied {len(jt_operations)} operations → {type(result).__name__}")

            elif jt_mode == "python_expr":
                if not jt_expression.strip():
                    raise ValueError("Python expression is empty. Please configure jt_expression.")
                result = _safe_eval(jt_expression, input_data)
                logger.info(f"[{node_config.id}] Python expression evaluated → {type(result).__name__}")

            elif jt_mode in ("pattern_extract", "pattern_extract_identifiers", "pattern_extract_field_mappings"):
                # Deprecated legacy mode: kept for backward-compat with workflows
                # saved before Visual Operations shipped. New configs should use
                # jt_mode="operations" with pattern_extract_* operations instead.
                logger.warning(
                    f"[{node_config.id}] jt_mode='{jt_mode}' is deprecated; "
                    "prefer jt_mode='operations' with pattern_extract_* operations."
                )
                raw_patterns = getattr(node_config, "extract_patterns", None)
                if not raw_patterns and hasattr(node_config, "model_extra") and node_config.model_extra:
                    raw_patterns = node_config.model_extra.get("extract_patterns")
                ops = jt_operations or [
                    {"type": "pattern_extract_identifiers", "patterns": raw_patterns or []}
                ]
                current = input_data
                for i, op in enumerate(ops):
                    current = _apply_operation(current, _interpolate_op_classes(op, context))
                result = current
                logger.info(f"[{node_config.id}] Pattern Extract executed via mode '{jt_mode}' → {type(result).__name__}")

            else:
                raise ValueError(f"Unknown jt_mode: '{jt_mode}'. Valid: jmespath, operations, python_expr, pattern_extract")

        except Exception as e:
            logger.error(f"[{node_config.id}] JSON Transform failed: {e}")
            error_msg = f"JSON Transform error ({jt_mode}): {str(e)}"
            return {
                "messages": [AIMessage(content=error_msg, name=node_config.id, response_metadata={"active_agent": node_config.id})],
                "context": {
                    **context,
                    jt_output_key: None,
                    f"{jt_output_key}_error": str(e)
                },
                "active_agent": node_config.id
            }

        # ── Build output context ──────────────────────────────────────────────
        new_context = dict(context)

        if jt_pass_through and isinstance(input_data, dict) and isinstance(result, dict):
            # Merge result into input data
            new_context[jt_output_key] = {**input_data, **result}
        else:
            new_context[jt_output_key] = result

        summary_lines = [
            f"✅ JSON Transform complete",
            f"Mode: `{jt_mode}`",
            f"Input: `{jt_input_key}`",
            f"Output: `{jt_output_key}` → {type(result).__name__}"
        ]
        if isinstance(result, list):
            summary_lines.append(f"Items: {len(result)}")

        # get_production_data() (utils.py) scans every node's chat message
        # for a JSON object and merges whatever it finds back into context —
        # meant for LLM agents that reply with JSON in plain text. It uses a
        # REPAIRING parser, so it isn't enough to just skip a ```json fence:
        # any {...}/[...] shaped text in the message gets found regardless.
        # This node's result is already written to context[jt_output_key];
        # a second, parseable copy here would only be re-extracted as
        # duplicate top-level keys — which is what was crashing response
        # serialization on a large result. So for dict/list results, the
        # chat message describes the SHAPE only, no braces/brackets, nothing
        # for that scanner to latch onto. Scalar results (str/int/float/
        # bool/None) are safe to show directly: get_production_data() only
        # merges a parsed dict or list, never a bare scalar.
        if isinstance(result, dict):
            keys_preview = ", ".join(str(k) for k in list(result.keys())[:10])
            more = f" (+{len(result) - 10} more)" if len(result) > 10 else ""
            summary_lines.append(f"\n{len(result)} top-level key(s): {keys_preview}{more}")
        elif isinstance(result, list):
            summary_lines.append(f"\nFirst item type: {type(result[0]).__name__}" if result else "\n(empty list)")
        else:
            result_str = str(result)
            summary_lines.append(f"\nValue: {result_str[:500]}{'...' if len(result_str) > 500 else ''}")

        chat_message = "\n".join(summary_lines)

        return {
            "messages": [AIMessage(content=chat_message, name=node_config.id, response_metadata={"active_agent": node_config.id})],
            "context": new_context,
            "active_agent": node_config.id
        }

    return json_transform

# Aliases: these node types are registered in NodeFactory but share the
# same implementation as build_json_transform_node.
build_json_transform_node_va = build_json_transform_node
build_pattern_extract_node   = build_json_transform_node
build_rule_engine_node       = build_json_transform_node

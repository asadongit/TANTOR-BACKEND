r"""
Unit tests for the pattern extraction operations added to the JSON Transform node.

Run from the AgentMesh backend root (so `Apps` imports resolve):
    ..\.venv\Scripts\python.exe Apps\AgentMesh\Core\Compiler\nodes\test_json_transform_pattern_extract.py

Also compatible with pytest discovery:
    ..\.venv\Scripts\pytest.exe Apps\AgentMesh\Core\Compiler\nodes\test_json_transform_pattern_extract.py
"""

import asyncio
import os
import sys

# Running the script by path sets sys.path[0] to this file's directory, so the
# backend root (parent of `Apps`) must be added explicitly for imports.
_BACKEND_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
)
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from Apps.AgentMesh.Core.Compiler.nodes.json_transform_node import (
    _apply_operation,
    _extract_field_mappings,
    _extract_identifiers,
    _key_matches,
    build_json_transform_node,
)
from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig


# ── Shared fixtures ──────────────────────────────────────────────────────────

NESTED_CUSTOMER = {
    "basic_info": {
        "pan_number": "ABCDE1234F",
        "name": "Rajesh Kumar",
        "account_number": "1234567890",
    },
    "loan": {
        "annual_income": "500000",
        "credit_score": "750",
    },
    "gst": {"gst_number": "GSTIN123456"},
    "itr": {"itr_status": "FILED"},
}

IDENTIFIER_PATTERNS = [
    {
        "concept": "PAN",
        "display_label": "PAN Number",
        "search_keywords": ["pan"],
        "match_strategy": "contains",
        "is_required": True,
    },
    {
        "concept": "Name",
        "display_label": "Customer Name",
        "search_keywords": ["name"],
        "match_strategy": "exact",
        "is_required": True,
    },
    {
        "concept": "AccountNumber",
        "display_label": "Account Number",
        "search_keywords": ["account_number"],
        "match_strategy": "exact",
        "is_required": True,
    },
    {
        "concept": "MissingField",
        "display_label": "Missing Field",
        "search_keywords": ["nonexistent_field"],
        "match_strategy": "contains",
        "is_required": True,
    },
]

FIELD_MAPPING_PATTERNS = [
    {
        "concept": "annual_income",
        "display_label": "Annual Income",
        "search_keywords": ["annual_income"],
        "match_strategy": "contains",
        "target_class": "Financial_Analysis",
    },
    {
        "concept": "credit_score",
        "display_label": "Credit Score",
        "search_keywords": ["credit_score"],
        "match_strategy": "contains",
        "target_class": "Financial_Analysis",
    },
    {
        "concept": "gst_number",
        "display_label": "GST Number",
        "search_keywords": ["gst_number"],
        "match_strategy": "contains",
        "target_class": "Tax_GST",
    },
]


# ── Core matching function tests ─────────────────────────────────────────────

def test_key_matches_strategies():
    assert _key_matches("basic_info/pan_number", ["pan"], "contains") is True
    assert _key_matches("basic_info/pan_number", ["gst"], "contains") is False
    assert _key_matches("basic_info/name", ["name"], "exact") is True
    assert _key_matches("basic_info/full_name", ["name"], "exact") is False
    assert _key_matches("loan/loan_amount", ["loan"], "starts_with") is True
    assert _key_matches("basic_info/customer_name", ["loan"], "starts_with") is False
    assert _key_matches("gst/gst_number", ["number"], "ends_with") is True
    assert _key_matches("itr/itr_status", ["itr_[a-z]+"], "regex") is True
    assert _key_matches("loan/annual_income", ["bad"], "contains") is False


def test_extract_identifiers_core():
    flat = {
        "basic_info/pan_number": "ABCDE1234F",
        "basic_info/name": "Rajesh Kumar",
    }
    result = _extract_identifiers(flat, IDENTIFIER_PATTERNS)
    by_concept = {r["concept"]: r for r in result}
    assert by_concept["PAN"]["value"] == "ABCDE1234F"
    assert by_concept["PAN"]["status"] == "found"
    assert by_concept["PAN"]["source_key"] == "basic_info/pan_number"
    assert by_concept["Name"]["value"] == "Rajesh Kumar"
    assert by_concept["Name"]["status"] == "found"
    assert by_concept["MissingField"]["status"] == "missing"
    assert by_concept["MissingField"]["value"] is None


def test_extract_field_mappings_core():
    flat = {
        "loan/annual_income": "500000",
        "loan/credit_score": "750",
        "gst/gst_number": "GSTIN123456",
    }
    result = _extract_field_mappings(flat, FIELD_MAPPING_PATTERNS)
    assert set(result.keys()) == {"Financial_Analysis", "Tax_GST"}
    assert result["Financial_Analysis"][0]["concept"] == "annual_income"
    assert result["Financial_Analysis"][0]["source_key"] == "loan/annual_income"
    assert result["Tax_GST"][0]["value"] == "GSTIN123456"


# ── New visual operations via _apply_operation ──────────────────────────────

def test_pattern_extract_identifiers_op():
    op = {"type": "pattern_extract_identifiers", "patterns": IDENTIFIER_PATTERNS, "destination": "ids"}
    result = _apply_operation(NESTED_CUSTOMER, op)
    ids = result["ids"]
    by_concept = {r["concept"]: r for r in ids}
    assert by_concept["PAN"]["value"] == "ABCDE1234F"
    assert by_concept["PAN"]["source_key"] == "basic_info/pan_number"
    assert by_concept["PAN"]["status"] == "found"
    assert by_concept["MissingField"]["status"] == "missing"
    # Nested input is flattened, original data preserved alongside the result key
    assert result["basic_info"]["name"] == "Rajesh Kumar"


def test_pattern_extract_identifiers_default_key():
    op = {"type": "pattern_extract_identifiers", "patterns": IDENTIFIER_PATTERNS}
    result = _apply_operation(NESTED_CUSTOMER, op)
    assert isinstance(result["identifiers"], list)
    assert len(result["identifiers"]) >= 3


def test_pattern_extract_field_mappings_op():
    op = {"type": "pattern_extract_field_mappings", "patterns": FIELD_MAPPING_PATTERNS, "destination": "mappings"}
    result = _apply_operation(NESTED_CUSTOMER, op)
    mappings = result["mappings"]
    assert set(mappings.keys()) == {"Financial_Analysis", "Tax_GST"}
    fa = {e["concept"]: e for e in mappings["Financial_Analysis"]}
    assert fa["annual_income"]["value"] == "500000"
    assert fa["credit_score"]["source_key"] == "loan/credit_score"
    assert mappings["Tax_GST"][0]["value"] == "GSTIN123456"


def test_pattern_extract_field_mappings_default_key():
    op = {"type": "pattern_extract_field_mappings", "patterns": FIELD_MAPPING_PATTERNS}
    result = _apply_operation(NESTED_CUSTOMER, op)
    assert set(result["field_mappings"].keys()) == {"Financial_Analysis", "Tax_GST"}


def test_pattern_extract_ops_non_dict_input():
    # Non-dict input cannot be flattened, so every required pattern is reported missing
    result = _apply_operation([1, 2, 3], {"type": "pattern_extract_identifiers", "patterns": IDENTIFIER_PATTERNS})
    assert isinstance(result["identifiers"], list)
    assert all(r["status"] == "missing" for r in result["identifiers"])
    result = _apply_operation("plain text", {"type": "pattern_extract_field_mappings", "patterns": FIELD_MAPPING_PATTERNS})
    assert result == {"field_mappings": {}}


# ── End-to-end node execution ────────────────────────────────────────────────

def _run_node(config_kwargs, state):
    node = build_json_transform_node(NodeConfig(id="test_jt", type="json_transform", **config_kwargs))
    return asyncio.run(node(state))


def test_node_pattern_extract_chaining():
    # Pattern Extract → Filter → Map in a single operations-mode execution
    config = {
        "jt_mode": "operations",
        "jt_input_key": "customer",
        "jt_output_key": "result",
        "jt_operations": [
            {"type": "pattern_extract_identifiers", "patterns": IDENTIFIER_PATTERNS, "destination": "ids"},
            {"type": "filter", "source_path": "ids", "filter_field": "status", "filter_operator": "==", "filter_value": "found"},
            {"type": "map", "source_path": "ids", "field": "label", "destination": "labels"},
        ],
    }
    state = {"context": {"customer": NESTED_CUSTOMER}}
    out = _run_node(config, state)
    result = out["context"]["result"]
    assert result["labels"] == ["PAN Number", "Customer Name", "Account Number"]
    # Only "found" items survive the filter
    assert all(r["status"] == "found" for r in result["ids"])
    assert out["context"].get("result_error") is None


def test_node_pattern_extract_field_mappings_node_run():
    config = {
        "jt_mode": "operations",
        "jt_input_key": "customer",
        "jt_output_key": "result",
        "jt_operations": [
            {"type": "pattern_extract_field_mappings", "patterns": FIELD_MAPPING_PATTERNS, "destination": "mappings"},
        ],
    }
    state = {"context": {"customer": NESTED_CUSTOMER}}
    out = _run_node(config, state)
    mappings = out["context"]["result"]["mappings"]
    assert mappings["Financial_Analysis"][0]["display_label"] == "Annual Income"
    assert mappings["Tax_GST"][0]["concept"] == "gst_number"


# ── Regression: existing modes must keep working ─────────────────────────────

def test_regression_jmespath_mode():
    config = {
        "jt_mode": "jmespath",
        "jt_input_key": "customer",
        "jt_output_key": "result",
        "jt_query": "basic_info.pan_number",
    }
    state = {"context": {"customer": NESTED_CUSTOMER}}
    out = _run_node(config, state)
    assert out["context"]["result"] == "ABCDE1234F"


def test_regression_python_expr_mode():
    config = {
        "jt_mode": "python_expr",
        "jt_input_key": "customer",
        "jt_output_key": "result",
        "jt_expression": "[r['label'] for r in data['ids'] if r['status'] == 'found']",
    }
    state = {"context": {"customer": {"ids": [
        {"label": "PAN", "status": "found"},
        {"label": "Missing", "status": "missing"},
    ]}}}
    out = _run_node(config, state)
    assert out["context"]["result"] == ["PAN"]


def test_regression_operations_mode_existing_ops():
    config = {
        "jt_mode": "operations",
        "jt_input_key": "customer",
        "jt_output_key": "result",
        "jt_operations": [
            {"type": "extract_field", "source_path": "", "field": "basic_info.pan_number", "destination": "pan"},
            {"type": "set_value", "field": "status", "static_value": "approved"},
            {"type": "rename", "source_path": "pan", "destination": "pan_number"},
        ],
    }
    state = {"context": {"customer": NESTED_CUSTOMER}}
    out = _run_node(config, state)
    result = out["context"]["result"]
    assert result["pan_number"] == "ABCDE1234F"
    assert result["status"] == "approved"


def test_top_level_pattern_extract_mode():
    config = {
        "jt_mode": "pattern_extract",
        "jt_input_key": "customer",
        "jt_output_key": "result",
        "extract_patterns": IDENTIFIER_PATTERNS,
    }
    state = {"context": {"customer": NESTED_CUSTOMER}}
    out = _run_node(config, state)
    ids = out["context"]["result"]["identifiers"]
    by_concept = {r["concept"]: r for r in ids}
    assert by_concept["PAN"]["value"] == "ABCDE1234F"


# ── n8n parity: set_value type coercion & interpolation ──────────────────────

def test_set_value_auto_coercion():
    op = {"type": "set_value", "destination": "n", "static_value": "42"}
    out = _apply_operation({}, op)
    assert out["n"] == 42 and isinstance(out["n"], int)

    op = {"type": "set_value", "destination": "b", "static_value": "true"}
    out = _apply_operation({}, op)
    assert out["b"] is True

    op = {"type": "set_value", "destination": "f", "static_value": "3.14"}
    out = _apply_operation({}, op)
    assert out["f"] == 3.14

    op = {"type": "set_value", "destination": "s", "static_value": "hello"}
    out = _apply_operation({}, op)
    assert out["s"] == "hello"


def test_set_value_explicit_type():
    op = {"type": "set_value", "destination": "n", "static_value": "42", "value_type": "string"}
    out = _apply_operation({}, op)
    assert out["n"] == "42" and isinstance(out["n"], str)

    op = {"type": "set_value", "destination": "j", "static_value": '{"a": 1}', "value_type": "json"}
    out = _apply_operation({}, op)
    assert out["j"] == {"a": 1}

    op = {"type": "set_value", "destination": "x", "static_value": "9.5", "value_type": "number"}
    out = _apply_operation({}, op)
    assert out["x"] == 9.5

    op = {"type": "set_value", "destination": "b", "static_value": "no", "value_type": "boolean"}
    out = _apply_operation({}, op)
    assert out["b"] is False


def test_set_value_interpolation():
    data = {"status": "SUCCESS", "score": 95}
    op = {"type": "set_value", "destination": "msg", "static_value": "status is {{ status }}"}
    out = _apply_operation(data, op)
    assert out["msg"] == "status is SUCCESS"

    op = {"type": "set_value", "destination": "score_copy", "static_value": "{{ score }}"}
    out = _apply_operation(data, op)
    assert out["score_copy"] == 95


# ── n8n parity: limit / sort / unique ────────────────────────────────────────

def test_limit_op():
    data = {"items": [1, 2, 3, 4, 5]}
    op = {"type": "limit", "source_path": "items", "limit": 2, "destination": "top"}
    out = _apply_operation(data, op)
    assert out["top"] == [1, 2]


def test_sort_op():
    data = {"items": [{"n": 3}, {"n": 1}, {"n": 2}]}
    op = {"type": "sort", "source_path": "items", "sort_field": "n", "sort_order": "desc", "sort_type": "number", "destination": "sorted"}
    out = _apply_operation(data, op)
    assert out["sorted"] == [{"n": 3}, {"n": 2}, {"n": 1}]


def test_unique_op():
    data = {"items": [{"id": 1, "x": "a"}, {"id": 2, "x": "b"}, {"id": 1, "x": "c"}]}
    op = {"type": "unique", "source_path": "items", "key_field": "id", "destination": "uniq"}
    out = _apply_operation(data, op)
    assert len(out["uniq"]) == 2
    assert out["uniq"][0]["id"] == 1


# ── Sandbox hardening ────────────────────────────────────────────────────────

def test_safe_eval_rejects_dangerous():
    from Apps.AgentMesh.Core.Compiler.nodes.json_transform_node import _safe_eval
    for bad_expr in [
        "import os",
        "__import__('os')",
        "().__class__",
        "open('/etc/passwd')",
        "getattr(data, 'x')",
        "eval('1')",
    ]:
        try:
            _safe_eval(bad_expr, {})
            raise AssertionError(f"expected rejection for: {bad_expr}")
        except ValueError:
            pass


def test_safe_eval_allows_legit():
    from Apps.AgentMesh.Core.Compiler.nodes.json_transform_node import _safe_eval
    assert _safe_eval("[x['id'] for x in data['items']]", {"items": [{"id": 1}]}) == [1]
    assert _safe_eval("sum(data)", [1, 2, 3]) == 6
    assert _safe_eval("[x.upper() for x in data['words']]", {"words": ["a", "b"]}) == ["A", "B"]
    # word-boundary blacklist: "important" is fine, "import" alone is not
    assert _safe_eval("data['important_data']", {"important_data": "ok"}) == "ok"


# ── JMESPath value interpolation ─────────────────────────────────────────────

def test_jmespath_context_value_interpolation():
    from Apps.AgentMesh.Core.Compiler.nodes.json_transform_node import _resolve_jmespath_vars
    ctx = {"target_status": "SUCCESS", "min_age": 18, "flag": True, "cfg": {"a": 1}}
    # bare placeholder: strings auto-quoted, numbers bare, bools bare
    assert _resolve_jmespath_vars("tasks[?status=={{ target_status }}].id", ctx) == "tasks[?status=='SUCCESS'].id"
    assert _resolve_jmespath_vars("tasks[?age > {{ min_age }}].id", ctx) == "tasks[?age > 18].id"
    assert _resolve_jmespath_vars("tasks[?ok == {{ flag }}].id", ctx) == "tasks[?ok == true].id"
    # author-quoted placeholder: no double quoting
    assert _resolve_jmespath_vars("tasks[?status=='{{ target_status }}'].id", ctx) == "tasks[?status=='SUCCESS'].id"
    # unresolvable var falls back to a field path (back-compat)
    assert _resolve_jmespath_vars("{{ job_status.tasks }}", {"other": 1}) == "job_status.tasks"


# ── Runner ───────────────────────────────────────────────────────────────────

_ALL_TESTS = [
    test_key_matches_strategies,
    test_extract_identifiers_core,
    test_extract_field_mappings_core,
    test_pattern_extract_identifiers_op,
    test_pattern_extract_identifiers_default_key,
    test_pattern_extract_field_mappings_op,
    test_pattern_extract_field_mappings_default_key,
    test_pattern_extract_ops_non_dict_input,
    test_node_pattern_extract_chaining,
    test_node_pattern_extract_field_mappings_node_run,
    test_regression_jmespath_mode,
    test_regression_python_expr_mode,
    test_regression_operations_mode_existing_ops,
    test_top_level_pattern_extract_mode,
    test_set_value_auto_coercion,
    test_set_value_explicit_type,
    test_set_value_interpolation,
    test_limit_op,
    test_sort_op,
    test_unique_op,
    test_safe_eval_rejects_dangerous,
    test_safe_eval_allows_legit,
    test_jmespath_context_value_interpolation,
]


def main():
    passed = 0
    failed = 0
    for fn in _ALL_TESTS:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL  {fn.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {passed + failed} total")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

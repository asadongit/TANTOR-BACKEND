import logging
import re
from datetime import datetime, date
from typing import Callable, Any, Dict, List, Optional, Union
from simpleeval import SimpleEval

from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from Apps.AgentMesh.Schema.StateSchema import AgentState
from langchain_core.messages import AIMessage
from Apps.AgentMesh.Core.Compiler.nodes.utils import get_production_data, interpolate, SmartInput, get_nested_val
from Apps.AgentMesh.Core.Compiler.nodes.logic import StringMatchers

logger = logging.getLogger(__name__)

class SafeNames(dict):
    """
    Safe dictionary resolver that returns None instead of raising NameError
    when a key is missing from SimpleEval evaluation scope.
    """
    def __getitem__(self, key):
        if key not in self:
            return None
        return super().__getitem__(key)


class CategoryEvalObj:
    """
    Helper object to allow cross-category rules to reference category scores in SimpleEval.
    e.g., aml_scoring.normalized_score > 50
    """
    def __init__(self, score: float, max_possible: float, normalized_score: float):
        self.score = score
        self.max_possible = max_possible
        self.normalized_score = normalized_score


# Date helper parsing function
def parse_date(val: Any) -> Optional[datetime]:
    if not val:
        return None
    if isinstance(val, (int, float)):
        try:
            return datetime.fromtimestamp(val)
        except Exception:
            return None
    if isinstance(val, (datetime, date)):
        if isinstance(val, date) and not isinstance(val, datetime):
            return datetime.combine(val, datetime.min.time())
        return val
    
    date_str = str(val).strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except Exception:
        pass
    return None


# Helper functions to register on SimpleEval
def days_since(date_str: Any) -> float:
    dt = parse_date(date_str)
    if not dt:
        return 999999.0
    dt = dt.replace(tzinfo=None)
    now = datetime.now()
    delta = now - dt
    return float(delta.days)


def to_num(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    s_val = str(val).strip()
    s_val = re.sub(r"[^\d\.\-]", "", s_val)
    try:
        return float(s_val)
    except ValueError:
        return default


def safe_sum(lst: Any) -> float:
    if not isinstance(lst, list):
        return 0.0
    return sum(to_num(x) for x in lst)


def safe_avg(lst: Any) -> float:
    if not isinstance(lst, list) or not lst:
        return 0.0
    nums = [to_num(x) for x in lst]
    return sum(nums) / len(nums)


def safe_count(lst: Any) -> int:
    if not isinstance(lst, list):
        return 0
    return len(lst)


def safe_min(lst: Any) -> float:
    if not isinstance(lst, list) or not lst:
        return 0.0
    return min(to_num(x) for x in lst)


def safe_max(lst: Any) -> float:
    if not isinstance(lst, list) or not lst:
        return 0.0
    return max(to_num(x) for x in lst)


def proportional_range(min_val: Any, max_val: Any, actual_val: Any) -> float:
    min_f = to_num(min_val)
    max_f = to_num(max_val)
    actual_f = to_num(actual_val)
    if min_f == max_f:
        return 100.0 if actual_f >= max_f else 0.0
    score = ((actual_f - min_f) / (max_f - min_f)) * 100.0
    return max(0.0, min(100.0, score))


def build_weightage_node(node_config: NodeConfig) -> Callable:
    """
    Builds an Enhanced Weightage System Node.

    Calculations and Flow:
    1.  **Scope Setup**: Loads context variable variables and registers specialized risk engine functions in SimpleEval:
        *   **Date/Time Helpers**: 
            - `DAYS_SINCE(date_str)` / `AGE_IN_DAYS(date_str)`: Parses date strings robustly and returns float days elapsed.
            - `NOW()`: Returns standard ISO date string.
        *   **Aggregation Functions**: 
            - `SUM(list)`, `AVG(list)`, `COUNT(list)`, `MIN(list)`, `MAX(list)`: Safe math operations over lists with string coercion.
        *   **Helpers**:
            - `RANGE(min, max, actual)`: Returns a proportional linear score scaled from 0 to 100.
            - `TO_NUM(val, default)`: Safely parses formatting characters (like $, commas) and returns a float.
    2.  **Category Evaluation**:
        *   Loops through all active categories. For each category:
            - Evaluates all fields and their conditions in a single pass.
            - Applies `cap_score` to each field's actual score and its maximum possible score limit.
            - Evaluates category fail threshold percentage comparison.
            - Detects kill switch activations (category score > 0 on a kill switch category).
            - Exposes category result namespaces (e.g. `aml_scoring.normalized_score`) for cross-category rules.
            - Scales both category score and category max possible score by category `weight` multiplier.
    3.  **Cross-Category Rules**:
        *   Executes a second pass applying penalties/bonuses based on category normalized scores.
        *   Binds the final adjusted score between 0.0 and the total max possible score.
    4.  **Decision & Grade Matching**:
        *   Decides on routing outputs: "approve", "review", or "reject" based on approve/review thresholds and kill switches.
        *   Maps normalized final scores to custom or default Risk Bands/Grades.
        *   Injects flat boolean flag outputs (like `is_approved`, `is_high_risk`, `kill_switch_triggered`) to the top-level context.

    Returns:
        execute_weightage (Callable): The LangGraph/compiler executable node function.
    """
    async def execute_weightage(state: AgentState):
        logger.info(f"Weightage Node {node_config.id}: Starting weightage calculation.")
        
        # 1. Resolve existing context & prepare SafeNames scope
        full_scope = get_production_data(state)
        
        # 2. Initialize SimpleEval with custom safe scope, string matchers, & functions
        s = SimpleEval()
        s.names = SafeNames({
            "True": True,
            "False": False,
            "None": None,
            "state": state,
            **full_scope
        })
        s.functions = {
            "endsWith": StringMatchers.endsWith,
            "startsWith": StringMatchers.startsWith,
            "contains": StringMatchers.contains,
            "matches": StringMatchers.matches,
            "equalsIgnoreCase": StringMatchers.equalsIgnoreCase,
            # Date/Time functions
            "DAYS_SINCE": days_since,
            "NOW": lambda: datetime.now().isoformat(),
            "AGE_IN_DAYS": days_since,
            # List aggregations
            "SUM": safe_sum,
            "AVG": safe_avg,
            "COUNT": safe_count,
            "MIN": safe_min,
            "MAX": safe_max,
            # Range & Coercion helpers
            "RANGE": proportional_range,
            "TO_NUM": to_num
        }

        # Retrieve Node configuration
        categories = node_config.weightage_categories or []
        output_key = node_config.weightage_output_key or "weightage_result"
        approve_threshold = getattr(node_config, "weightage_approve_threshold", 80.0)
        review_threshold = getattr(node_config, "weightage_review_threshold", 50.0)
        risk_bands = getattr(node_config, "weightage_risk_bands", []) or []
        output_flags = getattr(node_config, "weightage_output_flags", []) or []
        cross_rules = getattr(node_config, "weightage_cross_rules", []) or []

        total_score = 0.0
        max_possible_score = 0.0
        
        categories_breakdown = {}
        triggered_rules = []
        failed_categories = []
        kill_switch_triggered = False

        # 3. Process each category
        for cat in categories:
            if not getattr(cat, "enabled", True):
                continue
            cat_name = cat.name
            cat_weight = getattr(cat, "weight", 1.0)
            
            cat_score = 0.0
            cat_max_possible = 0.0
            fields_breakdown = {}

            # Process fields inside category
            for field in cat.fields:
                if not getattr(field, "enabled", True):
                    continue
                field_name = field.name
                match_strategy = field.match_strategy or "first_match"
                default_weight = field.default_weight or 0.0
                conditions = [c for c in (field.conditions or []) if getattr(c, "enabled", True)]
                field_cap = getattr(field, "cap_score", None)

                field_score = 0.0
                matched_conditions = []
                has_matched = False

                # Evaluate conditions and compile weight details in one pass
                resolved_weights = []
                for cond in conditions:
                    condition_expr = cond.condition
                    # quote_strings=True: evaluated by SimpleEval below (s.eval),
                    # so string values must render as quoted literals.
                    interpolated_expr = interpolate(condition_expr, full_scope, quote_strings=True)
                    clean_expr = interpolated_expr.replace("true", "True").replace("false", "False")

                    # Evaluate dynamic weight if it's a string expression
                    w = cond.weight
                    weight_value = 0.0
                    
                    # Evaluate condition match state
                    matched = False
                    try:
                        result = s.eval(clean_expr)
                        matched = bool(result)
                    except Exception as e:
                        logger.debug(f"Weightage Node {node_config.id} - Field '{field_name}' condition '{clean_expr}' eval error: {e}")
                        matched = False

                    # Get condition's weight value
                    if isinstance(w, str):
                        try:
                            # quote_strings=True: also evaluated by SimpleEval.
                            interpolated_w = interpolate(w, full_scope, quote_strings=True)
                            clean_w = interpolated_w.replace("true", "True").replace("false", "False")
                            weight_value = float(s.eval(clean_w))
                        except Exception:
                            weight_value = 0.0
                    else:
                        weight_value = float(w)
                    
                    resolved_weights.append(weight_value)

                    if matched and (not has_matched or match_strategy == "sum_all"):
                        has_matched = True
                        field_score += weight_value
                        rule_id = getattr(cond, "rule_id", None) or condition_expr
                        
                        matched_conditions.append({
                            "condition": condition_expr,
                            "evaluated_condition": clean_expr,
                            "weight": weight_value,
                            "rule_id": rule_id
                        })
                        if rule_id not in triggered_rules:
                            triggered_rules.append(rule_id)

                        if match_strategy == "first_match":
                            break

                # Apply default fallback weight if no conditions matched
                if not has_matched:
                    field_score = default_weight
                    if default_weight != 0.0:
                        matched_conditions.append({
                            "condition": "default_fallback",
                            "evaluated_condition": "No conditions matched",
                            "weight": default_weight
                        })

                # Calculate field_max_possible
                if match_strategy == "first_match":
                    field_max_possible = max(resolved_weights) if resolved_weights else 0.0
                else:  # sum_all
                    field_max_possible = sum(w for w in resolved_weights if w > 0.0)
                field_max_possible = max(field_max_possible, default_weight)

                # Apply cap_score to field score and max possible
                if field_cap is not None:
                    field_score = min(field_score, field_cap)
                    field_max_possible = min(field_max_possible, field_cap)

                field_value = get_nested_val(full_scope, field_name)
                cat_score += field_score
                cat_max_possible += field_max_possible

                fields_breakdown[field_name] = {
                    "value": field_value,
                    "score": field_score,
                    "max_possible_score": field_max_possible,
                    "match_strategy": match_strategy,
                    "matched_conditions": matched_conditions
                }

            cat_normalized = (cat_score / cat_max_possible * 100.0) if cat_max_possible > 0 else 0.0
            
            # Check per-category fail threshold
            cat_fail_threshold = getattr(cat, "fail_threshold", None)
            if cat_fail_threshold is not None and cat_normalized < cat_fail_threshold:
                failed_categories.append({
                    "category": cat_name,
                    "score": cat_normalized,
                    "threshold": cat_fail_threshold
                })

            # Check kill switch activation
            if getattr(cat, "is_kill_switch", False) and cat_score > 0:
                kill_switch_triggered = True

            categories_breakdown[cat_name] = {
                "score": cat_score,
                "max_possible_score": cat_max_possible,
                "normalized_score": round(cat_normalized, 2),
                "weight": cat_weight,
                "fields": fields_breakdown
            }

            # Add to total running scores scaled by category weight multiplier
            total_score += cat_score * cat_weight
            max_possible_score += cat_max_possible * cat_weight

            # Expose category for cross-category SimpleEval references
            s.names[cat_name] = CategoryEvalObj(cat_score, cat_max_possible, cat_normalized)

        # 4. Cross-Category rules processing (Pass 2)
        cross_penalties = 0.0
        triggered_cross_rules = []
        for rule in cross_rules:
            if not getattr(rule, "enabled", True):
                continue
            cond_expr = rule.condition
            try:
                interpolated_expr = interpolate(cond_expr, full_scope)
                clean_expr = interpolated_expr.replace("true", "True").replace("false", "False")
                if bool(s.eval(clean_expr)):
                    cross_penalties += rule.penalty_weight
                    triggered_cross_rules.append({
                        "rule_id": getattr(rule, "rule_id", None) or cond_expr,
                        "condition": cond_expr,
                        "penalty_weight": rule.penalty_weight
                    })
            except Exception as e:
                logger.error(f"Failed to evaluate cross-category rule '{cond_expr}': {e}")

        # Compute adjusted final total score
        adjusted_total_score = max(0.0, min(max_possible_score, total_score + cross_penalties))
        normalized_score = (adjusted_total_score / max_possible_score * 100.0) if max_possible_score > 0 else 0.0

        # 5. Routing decision matching
        if kill_switch_triggered:
            decision = "reject"
        elif normalized_score >= approve_threshold:
            decision = "approve"
        elif normalized_score >= review_threshold:
            decision = "review"
        else:
            decision = "reject"

        # 6. Risk bands / Letter grades resolving
        risk_label = "High Risk"
        risk_grade = "C"
        band_found = False
        for band in risk_bands:
            min_sc = getattr(band, "min_score", 0.0)
            max_sc = getattr(band, "max_score", 100.0)
            if min_sc <= normalized_score <= max_sc:
                risk_label = getattr(band, "label", risk_label)
                risk_grade = getattr(band, "grade", risk_grade)
                band_found = True
                break
        
        if not band_found:
            # Defaults
            if normalized_score >= 80:
                risk_label = "Low Risk"
                risk_grade = "A"
            elif normalized_score >= 50:
                risk_label = "Medium Risk"
                risk_grade = "B"
            else:
                risk_label = "High Risk"
                risk_grade = "C"

        weightage_result = {
            "total_score": round(adjusted_total_score, 2),
            "raw_total_score": round(total_score, 2),
            "max_possible_score": round(max_possible_score, 2),
            "normalized_score": round(normalized_score, 2),
            "decision": decision,
            "risk_label": risk_label,
            "risk_grade": risk_grade,
            "kill_switch_triggered": kill_switch_triggered,
            "failed_categories": failed_categories,
            "triggered_rules": triggered_rules,
            "triggered_cross_rules": triggered_cross_rules,
            "categories": categories_breakdown,
            "rule_version": "v4.0"
        }

        # Update context
        context = state.get("context", {}).copy()
        context[output_key] = weightage_result
        context[f"{output_key}_decision"] = decision

        # Resolve output flags
        for flag in output_flags:
            if flag == "is_approved":
                context[flag] = (decision == "approve")
            elif flag == "requires_review":
                context[flag] = (decision == "review")
            elif flag == "is_rejected":
                context[flag] = (decision == "reject")
            elif flag == "is_high_risk":
                context[flag] = (risk_label.lower() in ("high risk", "critical risk", "high") or risk_grade in ("D", "F"))
            elif flag == "kill_switch_triggered":
                context[flag] = kill_switch_triggered

        # Build response message with explainability block
        triggered_block = ", ".join(triggered_rules) if triggered_rules else "None"
        msg_content = (
            f"### Weightage System Calculation Completed\n"
            f"- **Decision**: `{decision.upper()}`\n"
            f"- **Risk Band**: {risk_label} ({risk_grade})\n"
            f"- **Total Score**: {round(adjusted_total_score, 2)} / {round(max_possible_score, 2)}\n"
            f"- **Normalized Score**: {round(normalized_score, 2)}%\n"
            f"- **Triggered Rules**: `{triggered_block}`\n"
            f"- **Kill Switch Triggered**: `{kill_switch_triggered}`\n"
            f"- **Audit Version**: `v4.0`\n"
        )

        return {
            "messages": [AIMessage(content=msg_content, response_metadata={"active_agent": node_config.id})],
            "active_agent": node_config.id,
            "context": context,
            output_key: weightage_result
        }

    # Bind name for GraphCompiler logging
    setattr(execute_weightage, "name", node_config.id)
    return execute_weightage

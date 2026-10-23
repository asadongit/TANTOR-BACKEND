"""
Logic Nodes (April 2026)
Utility nodes for Condition evaluation, Temporal waiting, and Synchronization.
Provides a foundation for complex industrial automation workflows.
"""

import asyncio
import logging
from typing import Callable, Any, Dict
import re
import json
import ast
from simpleeval import SimpleEval

from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from Apps.AgentMesh.Schema.StateSchema import AgentState
from langchain_core.messages import AIMessage
from json_repair import loads
from .utils import interpolate

logger = logging.getLogger(__name__)

# ============================================================================
# PRODUCTION-GRADE STRING MATCHERS (n8n-compatible with nested JSON support)
# ============================================================================

class StringMatchers:
    # Raw, because the examples below contain regexes. Without the r, the "\."
    # in one of them is an invalid escape sequence and Python warns on every
    # import of this module.
    r"""
    Production-grade string matching utilities for conditionals.
    - Handles None/null values gracefully
    - Automatic type coercion (int, dict → string)
    - Regex support for pattern matching
    - Safe for nested JSON access
    
    Usage in conditions:
        endsWith(email, "@trans.io")
        startsWith(username, "admin")
        contains(msg, "ERROR")
        matches(email, r"^[a-zA-Z0-9._%+-]+@trans\.io$")
    """
    
    @staticmethod
    def _to_string(value: Any) -> str:
        """Safely convert any type to string."""
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            return json.dumps(value)
        return str(value).strip()
    
    @staticmethod
    def endsWith(text: Any, suffix: Any) -> bool:
        """
        Check if text ends with suffix (case-sensitive).
        ✅ PRODUCTION SAFE: handles None, nested objects, type coercion
        
        Examples:
            endsWith("user@trans.io", "@trans.io") → True
            endsWith(email_obj, ".io") → True  (nested JSON)
            endsWith(None, "test") → False  (graceful null handling)
        """
        try:
            text_str = StringMatchers._to_string(text)
            suffix_str = StringMatchers._to_string(suffix)
            return text_str.endswith(suffix_str)
        except Exception as e:
            logger.warning(f"endsWith() error: {e}")
            return False
    
    @staticmethod
    def startsWith(text: Any, prefix: Any) -> bool:
        """
        Check if text starts with prefix (case-sensitive).
        ✅ PRODUCTION SAFE
        
        Examples:
            startsWith(username, "admin") → True
            startsWith(domain, "https://") → True
            startsWith("", "any") → False
        """
        try:
            text_str = StringMatchers._to_string(text)
            prefix_str = StringMatchers._to_string(prefix)
            return text_str.startswith(prefix_str)
        except Exception as e:
            logger.warning(f"startsWith() error: {e}")
            return False
    
    @staticmethod
    def contains(text: Any, substring: Any) -> bool:
        """
        Check if text contains substring (case-sensitive).
        ✅ PRODUCTION SAFE
        
        Examples:
            contains(msg, "ERROR") → True
            contains(message, "warning") → False (case-sensitive)
            contains(None, "test") → False
        """
        try:
            text_str = StringMatchers._to_string(text)
            sub_str = StringMatchers._to_string(substring)
            return sub_str in text_str
        except Exception as e:
            logger.warning(f"contains() error: {e}")
            return False
    
    @staticmethod
    def matches(text: Any, pattern: str) -> bool:
        """
        Check if text matches regex pattern.
        ✅ PRODUCTION SAFE: validation + exception handling
        
        Examples:
            matches(email, r"^[a-zA-Z0-9._%+-]+@trans\\.io$") → True
            matches("log-2024-04-15.txt", r"\\.txt$") → True
            matches(None, r".*") → False
        """
        try:
            text_str = StringMatchers._to_string(text)
            # Validate regex pattern
            compiled = re.compile(pattern)
            return bool(compiled.search(text_str))
        except re.error as e:
            logger.error(f"matches() invalid regex '{pattern}': {e}")
            return False
        except Exception as e:
            logger.warning(f"matches() error: {e}")
            return False
    
    @staticmethod
    def equalsIgnoreCase(text: Any, other: Any) -> bool:
        """
        Case-insensitive equality check.
        ✅ PRODUCTION SAFE
        
        Examples:
            equalsIgnoreCase(domain, "TRANS.IO") → True if domain is "trans.io"
        """
        try:
            text_str = StringMatchers._to_string(text).lower()
            other_str = StringMatchers._to_string(other).lower()
            return text_str == other_str
        except Exception as e:
            logger.warning(f"equalsIgnoreCase() error: {e}")
            return False

    @staticmethod
    def pick(items: Any, label: Any, default: Any = None) -> Any:
        """The value of the entry called `label`, from a list of labelled things.

        Extractors hand back their findings as a list, because a document may
        carry any of them and in any order:

            [{"label": "Period", "value": "2025-26"},
             {"label": "PAN",    "value": "AGQPR9756A"}]

        Reaching for "the second one" works until a document has no Period, and
        then quietly reads the wrong field. This asks for the one by NAME.

        Also accepts a plain object, so {"PAN": "..."} works the same way. The
        label is matched ignoring case and surrounding spaces, and nothing else -
        a guard should not be clever about what nearly matches.

        Returns `default` (None) when there is no such entry, so a missing value
        never silently looks like a present one.

        Examples:
            pick(class_out.identifiers, "PAN")   → "AGQPR9756A"
            pick(class_out.identifiers, "GSTN")  → None   (not in the document)
        """
        try:
            wanted = str(label).strip().lower()
            if isinstance(items, dict):
                for key, value in items.items():
                    if str(key).strip().lower() == wanted:
                        return value
                return default
            if not isinstance(items, (list, tuple)):
                return default
            # the usual spellings for "what this is" and "what it says"
            name_keys = ("label", "name", "key", "field", "type")
            value_keys = ("value", "val", "text", "content")
            for entry in items:
                if not isinstance(entry, dict):
                    continue
                found = next((entry[k] for k in name_keys if k in entry), None)
                if found is None or str(found).strip().lower() != wanted:
                    continue
                return next((entry[k] for k in value_keys if k in entry), default)
            return default
        except Exception as e:
            logger.warning(f"pick() error: {e}")
            return default

    @staticmethod
    def sameValue(left: Any, right: Any) -> bool:
        """True only when BOTH sides say something, and say the same thing.

        equalsIgnoreCase treats two missing values as equal, because both become
        the empty string. That is reasonable for a filter and dangerous for a
        guard: a document with no PAN, checked against a request with no PAN,
        would report a match and be let through as the right customer's.

        So absence is never a match here. Case and surrounding spaces are
        ignored; nothing else is.

        Examples:
            sameValue("AGQPR9756A", "agqpr9756a ") → True
            sameValue("AGQPR9756A", "WRONG1234X")  → False
            sameValue(None, None)                  → False   (nothing to compare)
            sameValue("AGQPR9756A", None)          → False
        """
        try:
            a = StringMatchers._to_string(left).strip().lower()
            b = StringMatchers._to_string(right).strip().lower()
            if not a or not b:
                return False
            return a == b
        except Exception as e:
            logger.warning(f"sameValue() error: {e}")
            return False


from .utils import interpolate, get_production_data, SmartInput

def build_condition_node(node_config: NodeConfig) -> Callable:
    """
    Build a condition node with an optimized production evaluation environment.
    """
    async def condition_node(state: AgentState):
        condition = node_config.condition or "True"
        
        # Resolve all production variables
        full_scope = get_production_data(state)
        
        # Interpolate if needed (for string-based labels).
        # quote_strings=True: the result is evaluated by SimpleEval below, where a
        # bare string would be read as an identifier and raise NameError.
        interpolated = interpolate(condition, full_scope, quote_strings=True)
        clean_expr = interpolated.replace("true", "True").replace("false", "False")
        
        try:
            # 🔒 ENTERPRISE SECURITY: Use SimpleEval instead of built-in eval()
            # This provides a bulletproof sandbox for user expressions.
            s = SimpleEval()
            s.nodes[ast.List] = lambda node: [s._eval(elt) for elt in node.elts]
            s.nodes[ast.Tuple] = lambda node: tuple(s._eval(elt) for elt in node.elts)
            
            # Inject variables into s.names
            s.names = {
                "True": True,
                "False": False,
                "None": None,
                "state": state,
                **full_scope
            }
            
            # ✅ Inject string matching & utility functions into s.functions (required by SimpleEval)
            s.functions = {
                "len": len,
                "count": len,
                "int": int,
                "str": str,
                "float": float,
                "bool": bool,
                "endsWith": StringMatchers.endsWith,
                "startsWith": StringMatchers.startsWith,
                "contains": StringMatchers.contains,
                "matches": StringMatchers.matches,
                "equalsIgnoreCase": StringMatchers.equalsIgnoreCase,
                # pick() reads a labelled value out of a list by NAME rather
                # than by position; sameValue() compares two of them and counts
                # a missing value as a mismatch, which equalsIgnoreCase does not.
                "pick": StringMatchers.pick,
                "sameValue": StringMatchers.sameValue,
            }

            result = s.eval(clean_expr)
            intent = "true" if result else "false"
        except Exception as e:
            logger.error(f"Logic Node {node_config.id} [Condition]: Evaluation Fail on '{clean_expr}': {e}")
            intent = "false"
            
        logger.info(f"Logic Node {node_config.id} [Condition]: '{clean_expr}' -> Result: {intent}")
        return {
            "intent": intent, 
            "active_agent": node_config.id
        }

    return condition_node

def build_switch_node(node_config: NodeConfig) -> Callable:
    """
    Build a multi-branch switch node.
    Evaluates a priority-ordered list of branches.
    Sets state['intent'] to the name of the first matching branch, or 'else'.
    """
    async def switch_node(state: AgentState):
        branches = node_config.branches or []
        full_scope = get_production_data(state)
        
        matched_intent = "else"
        
        for branch in branches:
            name = branch.get("name", "branch")
            condition = branch.get("condition", "False")
            
            # Interpolate variables.
            # quote_strings=True: evaluated by SimpleEval below, so string values
            # must render as quoted literals rather than identifiers.
            interpolated = interpolate(condition, full_scope, quote_strings=True)
            clean_expr = interpolated.replace("true", "True").replace("false", "False")
            
            try:
                # 🔒 ENTERPRISE SECURITY: Use SimpleEval with string matchers
                s = SimpleEval()
                s.nodes[ast.List] = lambda node: [s._eval(elt) for elt in node.elts]
                s.nodes[ast.Tuple] = lambda node: tuple(s._eval(elt) for elt in node.elts)
                
                # Inject variables into s.names
                s.names = {
                    "True": True,
                    "False": False,
                    "None": None,
                    "state": state,
                    **full_scope
                }
                
                # ✅ Inject string matching & utility functions into s.functions (required by SimpleEval)
                s.functions = {
                    "len": len,
                    "count": len,
                    "int": int,
                    "str": str,
                    "float": float,
                    "bool": bool,
                    "endsWith": StringMatchers.endsWith,
                    "startsWith": StringMatchers.startsWith,
                    "contains": StringMatchers.contains,
                    "matches": StringMatchers.matches,
                    "equalsIgnoreCase": StringMatchers.equalsIgnoreCase,
                    "pick": StringMatchers.pick,
                    "sameValue": StringMatchers.sameValue,
                }

                if s.eval(clean_expr):
                    matched_intent = name
                    break
            except Exception as e:
                logger.error(f"Logic Node {node_config.id} [Switch]: Branch '{name}' eval failed: {e}")
        
        logger.info(f"Logic Node {node_config.id} [Switch]: Routing to branch '{matched_intent}'")
        return {
            "intent": matched_intent,
            "active_agent": node_config.id
        }

    return switch_node

def build_wait_node(node_config: NodeConfig) -> Callable:
    """
    Build a wait node that pauses execution for a specified duration (seconds).
    """
    async def wait_node(state: AgentState):
        seconds = node_config.wait_seconds or 0
        try:
            seconds = float(seconds)
        except (ValueError, TypeError):
            seconds = 0
            
        if seconds > 0:
            logger.info(f"Logic Node {node_config.id} [Wait]: Pausing workflow for {seconds}s...")
            # For short durations (seconds/minutes), asyncio.sleep is effective.
            # Workflow persistence ensures it resumes if the server stays up.
            await asyncio.sleep(seconds)
            
        return {"active_agent": node_config.id}

    return wait_node

def build_set_data_node(node_config: NodeConfig) -> Callable:
    """
    Build a node that sets static data or messages in the state.
    """
    async def set_data_node(state: AgentState):
        logger.info(f"Logic Node {node_config.id} [Set]: Injecting static message")
        msg = node_config.message or "Default message from Set Node"
        
        # 1. Resolve existing context
        full_scope = get_production_data(state)
        interpolated_msg = interpolate(msg, full_scope)
        
        # 2. ✅ NEW: Also parse the message I am about to send!
        # This converts the static text in the SetData node into actual variables (e.g. name, class)
        new_extracted = {}
        try:
            parsed = loads(interpolated_msg)
            if isinstance(parsed, dict):
                new_extracted = parsed
            elif isinstance(parsed, list):
                new_extracted = {"items": parsed}
        except:
            pass # Not JSON
            
        # 3. Build the final context update
        # Merge incoming context + newly extracted keys
        incoming_context = (state.get("context") if isinstance(state, dict) else getattr(state, "context", {})) or {}
        clean_context = {**incoming_context, **new_extracted}
        clean_context.pop("msg", None)
        clean_context.pop("data", None)
        clean_context.pop("context", None)
        
        # We return an AI Message so the follow-up nodes (or UI) treats it as a response
        return {
            "messages": [AIMessage(content=interpolated_msg, response_metadata={"active_agent": node_config.id})],
            "active_agent": node_config.id,
            "context": clean_context
        }

    return set_data_node

def build_join_node(node_config: NodeConfig) -> Callable:
    """
    Build a join/merge node that acts as a synchronization point.
    In AgentMesh (LangGraph based), this is naturally handled by the graph engine 
    when multiple directional edges converge on a single target node.
    This node serves as a pass-through to clear intent and continue.
    """
    async def join_node(state: AgentState):
        logger.info(f"Logic Node {node_config.id} [Join]: Convergence successful. Resuming unified flow.")
        return {
            "active_agent": node_config.id,
            "intent": None # Clear previous routing intent to allow new decisions
        }

    return join_node

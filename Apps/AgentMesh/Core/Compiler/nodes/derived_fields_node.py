"""
Derived Fields Node Builder

Computes new values from existing ones using a table of named expressions.

Each row is a name, an expression and an optional group. The expressions are
plain arithmetic over the fields already in the data — `revenue - cost`,
`total / count * 100` — evaluated with the same sandboxed evaluator the Switch
and Weightage nodes use. No code is written or executed.

The point of a table rather than a code block: whoever maintains these values is
usually the person who understands the business rule, not a programmer, and
anyone reviewing the workflow can read what each value means without reading
code. Every row is evaluated independently, so one bad expression reports itself
by name instead of taking the whole node down.
"""

import ast
import logging
import math
import statistics
from typing import Any, Callable, Dict, List

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from simpleeval import SimpleEval

from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from Apps.AgentMesh.Schema.StateSchema import AgentState
from Apps.AgentMesh.Core.Compiler.nodes.utils import get_production_data, get_nested_val

logger = logging.getLogger(__name__)


def _safe_div(a, b):
    """Divide, returning None instead of raising when the divisor is zero.

    Dividing by zero is ordinary in real data — a ratio against a value that
    happens to be nil. Failing the whole row for it would be unhelpful; an
    empty result says "not applicable here", which is what it means.
    """
    try:
        if b in (0, None):
            return None
        return a / b
    except (TypeError, ZeroDivisionError):
        return None


def _to_number(value):
    """Coerce a value to a number where that is unambiguous, else leave it.

    Extracted data frequently arrives as text — "1,250,000" from a document, or
    "45.5" from a form. Expressions that would otherwise fail on a string then
    behave as the author expects.
    """
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            pass
        try:
            return float(text)
        except ValueError:
            pass
    return value


def _flatten_for_expressions(data: Any, prefix: str = "") -> Dict[str, Any]:
    """
    Make nested values reachable by a flat, expression-safe name.

    {"balance": {"assets": 10}}  ->  {"balance_assets": 10, ...}

    Expressions cannot use dots, because a dot means attribute access. Flattening
    with underscores keeps names readable while staying valid in an expression.
    The original top-level keys are preserved too, so a shallow payload needs no
    prefixes at all.
    """
    flat: Dict[str, Any] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            name = f"{prefix}_{key}" if prefix else str(key)
            if isinstance(value, dict):
                flat.update(_flatten_for_expressions(value, name))
            else:
                flat[name] = _to_number(value)
    return flat


_VERDICT_KEYS = ("stable", "certain", "trusted", "ok", "doubtful")


def _flatten_confidence(data: Any, prefix: str = "") -> Dict[str, Any]:
    """Reach a confidence map by the same flat names the expressions use.

    Flattening stops at a verdict rather than descending into it: a record such
    as {"stable": false, "reason": ...} is the answer about one value, not a
    further nesting of values, and taking it apart would invent names like
    'inventories_stable' that no expression ever refers to.
    """
    flat: Dict[str, Any] = {}
    if not isinstance(data, dict):
        return flat
    for key, value in data.items():
        name = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(value, dict) and not any(k in value for k in _VERDICT_KEYS):
            flat.update(_flatten_confidence(value, name))
        else:
            flat[name] = value
    return flat


def _referenced_names(expression: str) -> set:
    """The value names an expression reads.

    Taken from the parsed expression rather than by searching the text, so a
    name that merely appears inside another word is not mistaken for a
    reference. The names being called as functions - safe_div, min, sum - are
    removed: they are arithmetic, not data, and nothing is known about their
    certainty.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return set()
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    return {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    } - called - {"True", "False", "None"}


def _is_doubtful(entry: Any) -> bool:
    """Read one entry of a confidence map, whatever shape it arrived in.

    Upstream nodes describe certainty differently - a plain false, or a record
    carrying a 'stable' flag among other detail - and this node should not care
    which. Anything it cannot interpret is treated as certain, because a value
    nobody has raised a doubt about is not evidence of a problem.
    """
    if entry is None:
        return False
    if isinstance(entry, bool):
        return not entry
    if isinstance(entry, dict):
        for key in ("stable", "certain", "trusted", "ok"):
            if key in entry:
                return not bool(entry[key])
        if "doubtful" in entry:
            return bool(entry["doubtful"])
    return False


def _summarise(deviations: List[Dict[str, Any]],
               doubtful: Dict[str, Dict[str, Any]]) -> str:
    """The findings written out for somebody to read.

    A workbook fills one cell per marker and cannot grow rows to fit a list, so
    a table of fixed height would either truncate the findings or leave empty
    markers showing. One block of text holds however many there are.

    Written for an underwriter, not for a developer: what did not add up, by how
    much, and which figures to go and check on the statement.
    """
    if not deviations:
        return ("No integrity check failed. Every total on the statement agrees "
                "with the figures beneath it.")
    lines = [f"{len(deviations)} integrity check(s) did not hold. "
             f"{len(doubtful)} figure(s) are affected and should be verified "
             f"against the source document before this report is relied on.", ""]
    for i, d in enumerate(deviations, 1):
        value = d.get("value")
        try:
            shown = f"out by {float(value):,.2f}"
        except (TypeError, ValueError):
            shown = "could not be calculated - a figure it needs is missing"
        lines.append(f"{i}. {d.get('check')}")
        if d.get("description"):
            lines.append(f"   What it means: {d['description']}")
        lines.append(f"   Result: {shown}")
        reads = d.get("fields_in_question") or []
        if reads:
            lines.append(f"   Figures to check ({len(reads)}):")
            for r in reads[:8]:
                lines.append(f"      - {r}")
            if len(reads) > 8:
                lines.append(f"      ... and {len(reads) - 8} more")
        lines.append("")
    return "\n".join(lines).strip()


def _build_evaluator(scope: Dict[str, Any]) -> SimpleEval:
    """A sandboxed evaluator with arithmetic helpers and no access to anything else."""
    evaluator = SimpleEval()
    evaluator.nodes[ast.List] = lambda node: [evaluator._eval(e) for e in node.elts]
    evaluator.nodes[ast.Tuple] = lambda node: tuple(evaluator._eval(e) for e in node.elts)
    evaluator.names = {"True": True, "False": False, "None": None, **scope}
    evaluator.functions = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": lambda seq: sum(x for x in seq if isinstance(x, (int, float))),
        "len": len,
        "avg": lambda seq: statistics.mean([x for x in seq if isinstance(x, (int, float))]) or 0,
        "floor": math.floor,
        "ceil": math.ceil,
        "sqrt": math.sqrt,
        "safe_div": _safe_div,
    }
    return evaluator


def build_derived_fields_node(node_config: NodeConfig) -> Callable:
    """
    Build a node that computes named values from a table of expressions.
    """
    fields: List[Dict[str, Any]] = getattr(node_config, "df_fields", None) or []
    input_key = (getattr(node_config, "df_input_key", None) or "").strip()
    output_key = (getattr(node_config, "df_output_key", None) or "derived_fields").strip()
    on_error = (getattr(node_config, "df_on_error", None) or "null").strip().lower()
    confidence_key = (getattr(node_config, "df_confidence_key", None) or "").strip()
    node_id = node_config.id

    enabled = [
        f for f in fields
        if isinstance(f, dict) and f.get("enabled", True) and str(f.get("name", "")).strip()
    ]

    logger.info(
        f"[DERIVED_FIELDS] Node {node_id}: {len(enabled)} field(s) configured, "
        f"input='{input_key or 'whole context'}' output='{output_key}' on_error='{on_error}'"
    )

    async def execute_derived_fields(state: AgentState):
        scope_source = get_production_data(state)
        data = get_nested_val(scope_source, input_key) if input_key else scope_source
        if data is None:
            logger.warning(
                f"[DERIVED_FIELDS] Node {node_id}: input key '{input_key}' not found; "
                f"falling back to the full context."
            )
            data = scope_source

        scope = _flatten_for_expressions(data)
        results: Dict[str, Any] = {}
        failures: List[Dict[str, str]] = []

        # Certainty travels with the arithmetic. A value computed from something
        # doubtful is itself doubtful, however sound the sum - so a ratio built
        # on a figure nobody could read cannot present itself as fact. Nothing
        # here knows what made a value doubtful, only that something did, which
        # is what keeps it useful to any workflow rather than this one.
        raw_conf = get_nested_val(scope_source, confidence_key) if confidence_key else None
        incoming = _flatten_confidence(raw_conf)
        doubtful: Dict[str, Dict[str, Any]] = {}
        if confidence_key and raw_conf is None:
            logger.warning(
                f"[DERIVED_FIELDS] Node {node_id}: confidence key '{confidence_key}' "
                f"not found; every value will be treated as certain."
            )

        # Each row is evaluated on its own, and every result is added to the
        # scope as it is produced. That lets one row build on another —
        # "margin = profit / revenue" can be followed by "margin_pct = margin *
        # 100" — in the order the table is written.
        for field in enabled:
            name = str(field["name"]).strip()
            expression = str(field.get("expression", "")).strip()
            if not expression:
                continue
            try:
                value = _build_evaluator(scope).eval(expression)
                results[name] = value
                scope[name] = value
            except Exception as exc:
                failures.append({"name": name, "expression": expression, "error": str(exc)})
                if on_error == "fail":
                    logger.error(
                        f"[DERIVED_FIELDS] Node {node_id}: '{name}' failed and "
                        f"on_error='fail' — stopping. Expression: {expression}"
                    )
                    raise ValueError(
                        f"Derived field '{name}' failed: {exc}. Expression: {expression}"
                    )
                if on_error == "null":
                    results[name] = None
                    scope[name] = None
                # 'skip' leaves the name out of the result entirely.

        if failures:
            logger.warning(
                f"[DERIVED_FIELDS] Node {node_id}: {len(failures)} of {len(enabled)} "
                f"field(s) could not be computed (on_error='{on_error}'): "
                + ", ".join(f"{f['name']} ({f['error'][:60]})" for f in failures[:5])
            )
        logger.info(
            f"[DERIVED_FIELDS] Node {node_id}: computed {len(results)} field(s), "
            f"{len(failures)} failed"
        )

        # ── Where doubt comes from ──────────────────────────────────────────
        # A row marked as a check states something that must be true of the
        # figures - assets equal liabilities, a total equals the lines beneath
        # it - and is written to come out zero. When it does not, one of the
        # figures it referenced is wrong, and though nothing here can say WHICH,
        # every one of them is now in question. So a failed check hands its
        # doubt BACKWARDS to the fields it read.
        #
        # This is the dependable source of doubt. It is arithmetic, so it gives
        # the same answer every run - unlike asking a model to read the page
        # twice, which was measured wandering between 86% and 97% agreement on
        # one page and cannot be gated on.
        deviations: List[Dict[str, Any]] = []
        for field in enabled:
            if not field.get("is_check"):
                continue
            name = str(field["name"]).strip()
            if name not in results:
                continue
            value = results.get(name)
            try:
                tol = abs(float(field.get("tolerance", 1) or 0))
            except (TypeError, ValueError):
                tol = 1.0
            broke = value is None or abs(float(value)) > tol
            if not broke:
                continue
            reads = sorted(_referenced_names(str(field.get("expression", ""))))
            deviations.append({
                "check": name,
                "value": value,
                "tolerance": tol,
                "fields_in_question": reads,
                "description": field.get("description", ""),
            })
            for ref in reads:
                doubtful.setdefault(ref, {
                    "stable": False,
                    "depends_on": [],
                    "reason": f"'{name}' did not come out to zero, so this figure "
                              f"and the others it was checked against are in question",
                })

        # ── Carrying it forward ─────────────────────────────────────────────
        # Worked out AFTER every row is evaluated, so a check written at the
        # bottom of the table still reaches a ratio written at the top. Rows are
        # walked in order so that one row building on another inherits its doubt.
        for field in enabled:
            name = str(field["name"]).strip()
            expression = str(field.get("expression", "")).strip()
            if not expression or name in doubtful:
                continue
            tainted = sorted(
                ref for ref in _referenced_names(expression)
                if _is_doubtful(incoming.get(ref)) or ref in doubtful
            )
            if tainted:
                doubtful[name] = {
                    "stable": False,
                    "depends_on": tainted,
                    "reason": "computed from " + ", ".join(tainted[:4])
                    + (" and others" if len(tainted) > 4 else ""),
                }

        # A row that could not be calculated is never sound. It holds an empty
        # value rather than a computed one, and calling that trustworthy is the
        # precise false assurance this is here to prevent - the more so because
        # an expression that cannot even be parsed yields no names to inspect,
        # so nothing else would ever mark it.
        for failure in failures:
            if failure["name"] in results:
                doubtful.setdefault(failure["name"], {
                    "stable": False,
                    "depends_on": [],
                    "reason": "could not be calculated",
                })

        if deviations:
            logger.warning(
                f"[DERIVED_FIELDS] Node {node_id}: {len(deviations)} integrity check(s) "
                f"did not hold: "
                + ", ".join(f"{d['check']} ({d['value']})" for d in deviations[:4])
            )
        if doubtful:
            logger.warning(
                f"[DERIVED_FIELDS] Node {node_id}: {len(doubtful)} value(s) are in "
                f"question: " + ", ".join(sorted(doubtful)[:6])
                + (" ..." if len(doubtful) > 6 else "")
            )

        # Every computed name is reported, certain ones included, so that
        # whatever reads this next never has to tell "absent" from "fine". The
        # source figures that a failed check put in question are reported too,
        # even though this node did not compute them - they are the ones an
        # underwriter has to look at.
        confidence = {
            name: doubtful.get(name, {"stable": True, "depends_on": []})
            for name in results
        }
        for name, verdict in doubtful.items():
            confidence.setdefault(name, verdict)

        context = state.get("context", {}) or {}
        return {
            "messages": [
                AIMessage(
                    content=(
                        f"Computed {len(results)} derived field(s)"
                        + (f", {len(failures)} could not be computed" if failures else "")
                        + (f". {len(deviations)} integrity check(s) DID NOT HOLD — "
                           f"{len(doubtful)} figure(s) in question"
                           if deviations else "")
                    ),
                    name=node_id,
                    response_metadata={"active_agent": node_id},
                )
            ],
            "active_agent": node_id,
            "context": {
                **context,
                output_key: results,
                f"{output_key}_errors": failures,
                f"{output_key}_confidence": confidence,
                # The checks that did not hold, named and with the figures they
                # implicate, ready to be reported without anything downstream
                # having to understand how they were decided.
                f"{output_key}_deviations": deviations,
                f"{output_key}_deviation_count": len(deviations),
                f"{output_key}_in_question": sorted(doubtful),
                f"{output_key}_in_question_count": len(doubtful),
                # The same thing written out to be read. A workbook fills one
                # cell per marker and cannot grow rows to fit a list, so a
                # report that must survive any number of findings needs them in
                # a single readable block rather than a table of fixed height.
                f"{output_key}_deviations_summary": _summarise(deviations, doubtful),
            },
        }

    executor = RunnableLambda(execute_derived_fields)
    setattr(executor, "name", node_id)
    return executor

"""
Node Testing

Run a single node against sample data, without compiling or running a workflow.

Configuring a node is otherwise guesswork: the only way to find out whether a
setting is right is to run the whole pipeline and read the result. For anything
involving document reading or a model call that is minutes per attempt, which
makes iterating on configuration impractical and debugging worse.

This runs the real node, built from the config exactly as the panel has it, and
returns what came out. Nothing is simulated.
"""

import copy
import logging
import re
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from Apps.AgentMesh.Core.Compiler.NodeFactory import NodeFactory
from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Node Testing"])

# Keys the platform puts into every node's context regardless of what the node
# does — execution bookkeeping, not results. Reporting them as "produced" would
# bury the one or two values the person is actually looking for.
_PLUMBING_KEYS = {
    "context",
    "nodes",
    "msg",
    "data",
    "hitl_action",
    "hitl_feedback",
    "active_agent",
    "thread_id",
    "workflow_id",
    "workflow_name",
    "workflow_variables",
}


class NodeTestRequest(BaseModel):
    node_config: Dict[str, Any] = Field(
        ...,
        description="The node's configuration, exactly as the editor holds it.",
    )
    sample_input: Dict[str, Any] = Field(
        default_factory=dict,
        description="Values the node can read, as though produced by earlier steps.",
    )
    timeout_seconds: Optional[int] = Field(
        60, ge=1, le=300, description="Give up if the node takes longer than this."
    )


class NodeTestResponse(BaseModel):
    ok: bool
    node_id: Optional[str] = None
    node_type: Optional[str] = None
    duration_ms: Optional[int] = None
    produced: Dict[str, Any] = Field(default_factory=dict)
    messages: list = Field(default_factory=list)
    error: Optional[str] = None
    error_type: Optional[str] = None
    hint: Optional[str] = None


def _hint_for(exc: Exception, node_type: str) -> Optional[str]:
    """Turn a raw error into something the person configuring the node can act on."""
    text = str(exc)
    lowered = text.lower()

    if "unknown node type" in lowered:
        return (
            "This node type is not recognised by the server. It usually means the "
            "editor and the server are on different versions."
        )
    if "is not defined" in lowered:
        # Take the name immediately before "is not defined" — an error often
        # quotes the field being computed first, and naming that instead would
        # send the reader to the wrong place entirely.
        match = re.search(r"'([^']+)'\s+is not defined", text)
        name = match.group(1) if match else None
        if name:
            return (
                f"The calculation uses {name!r}, which is not in the sample input. "
                f"Either add {name!r} to the sample, or correct the name."
            )
        return (
            "The configuration refers to a value that is not present in the "
            "sample input."
        )
    if "validation error" in lowered or "field required" in lowered:
        return "A required setting is missing or has the wrong kind of value."
    if "template" in lowered and ("not found" in lowered or "requires" in lowered):
        return "The template could not be found. Check the template setting."
    if "timed out" in lowered:
        return (
            "The node did not finish in time. Nodes that call a model or read a "
            "large document can be slow; raise the time limit or use smaller input."
        )
    return None


class TemplateSuggestRequest(BaseModel):
    template_path: str = Field(
        ..., description="A filled-in spreadsheet to learn the layout from."
    )
    sample_data: Dict[str, Any] = Field(
        ..., description="Data containing the same values, under the names to use."
    )
    max_suggestions: Optional[int] = Field(500, ge=1, le=5000)


@router.post("/node/suggest-template-markers")
async def suggest_template_markers(request: TemplateSuggestRequest) -> Dict[str, Any]:
    """
    Propose where each value belongs in a spreadsheet, by matching what is
    already printed in it.

    A filled-in spreadsheet contains real values. Sample data contains those
    same values under the names a workflow uses. Wherever the two agree, the
    cell's purpose is known — so the placeholder can be proposed rather than
    hunted for by hand across thousands of cells.

    It proposes; it never decides. Where a value appears in more than one place,
    or a cell matches more than one name, every candidate is returned and marked
    ambiguous so a person can choose.
    """
    import os

    from Apps.AgentMesh.Core.Compiler.nodes.report_generation_node import (
        _resolve_placeholder_path,
    )

    try:
        import openpyxl
    except ImportError:
        raise HTTPException(status_code=500, detail="openpyxl is not installed.")

    if not os.path.exists(request.template_path):
        raise HTTPException(
            status_code=404, detail=f"No file at {request.template_path!r}."
        )

    # Flatten the sample into dotted paths so a suggestion names the exact route
    # to the value, which is what a placeholder needs.
    def flatten(node: Any, prefix: str = "") -> Dict[str, Any]:
        flat: Dict[str, Any] = {}
        if isinstance(node, dict):
            for key, value in node.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                flat.update(flatten(value, path))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                flat.update(flatten(value, f"{prefix}.{index}"))
        elif node is not None:
            flat[prefix] = node
        return flat

    flat_sample = flatten(request.sample_data)

    # Index by value. Text and numbers are matched separately, and very short or
    # obviously generic values are skipped: matching on 0, 1 or "N/A" would
    # produce noise rather than suggestions.
    by_value: Dict[Any, list] = {}
    for path, value in flat_sample.items():
        if value is None:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            if abs(value) < 100:
                continue
            key = round(float(value), 4)
        else:
            text = str(value).strip()
            if len(text) < 4:
                continue
            key = text
        by_value.setdefault(key, []).append(path)

    workbook = openpyxl.load_workbook(request.template_path)
    suggestions, ambiguous = [], 0

    for sheet_name in workbook.sheetnames:
        sheet = workbook[sheet_name]
        for row in sheet.iter_rows():
            for cell in row:
                value = cell.value
                if value is None or isinstance(value, bool):
                    continue
                if isinstance(value, str) and value.startswith("="):
                    # A formula computes itself; filling it would destroy the
                    # very behaviour the template relies on.
                    continue
                if isinstance(value, (int, float)):
                    if abs(value) < 100:
                        continue
                    key = round(float(value), 4)
                else:
                    text = str(value).strip()
                    if len(text) < 4:
                        continue
                    key = text

                paths = by_value.get(key)
                if not paths:
                    continue
                if len(paths) > 1:
                    ambiguous += 1
                suggestions.append(
                    {
                        "sheet": sheet_name,
                        "cell": cell.coordinate,
                        "current_value": value if not isinstance(value, str) else value[:80],
                        "suggested_path": paths[0],
                        "alternatives": paths[1:5],
                        "ambiguous": len(paths) > 1,
                        "placeholder": "{{ %s }}" % paths[0],
                    }
                )
                if len(suggestions) >= (request.max_suggestions or 500):
                    break
            if len(suggestions) >= (request.max_suggestions or 500):
                break
        if len(suggestions) >= (request.max_suggestions or 500):
            break

    sheets_total = len(workbook.sheetnames)
    workbook.close()

    logger.info(
        f"[TemplateSuggest] {len(suggestions)} suggestion(s) across "
        f"{sheets_total} sheet(s), {ambiguous} ambiguous"
    )
    return {
        "ok": True,
        "sheets_scanned": sheets_total,
        "values_available": len(flat_sample),
        "suggestions": suggestions,
        "ambiguous_count": ambiguous,
        "note": (
            "Each suggestion is a proposal. Review them, then write the "
            "placeholders into the spreadsheet and save it as your template."
        ),
    }


class ExpressionCheckRequest(BaseModel):
    expressions: list = Field(
        ...,
        description="Expressions to check: a list of strings, or of {name, expression}.",
    )
    known_names: Optional[list] = Field(
        None,
        description=(
            "Names available to the expressions. When given, a reference to "
            "anything else is reported. Omit to check only the syntax."
        ),
    )


@router.post("/node/check-expressions")
async def check_expressions(request: ExpressionCheckRequest) -> Dict[str, Any]:
    """
    Check expressions without running anything.

    Catches the two mistakes that are otherwise only discovered at run time: an
    expression that is not valid at all, and one that refers to a name which
    does not exist. Deliberately does not evaluate, so it is instant and safe to
    call as someone types.

    Names defined by earlier expressions count as known for the ones after them,
    matching how they are actually evaluated.
    """
    import ast as _ast

    from Apps.AgentMesh.Core.Compiler.nodes.derived_fields_node import _build_evaluator

    available = set(request.known_names or [])
    check_names = request.known_names is not None
    functions = set(_build_evaluator({}).functions.keys())
    results = []

    for index, raw in enumerate(request.expressions or []):
        if isinstance(raw, dict):
            name = str(raw.get("name") or f"#{index + 1}")
            expression = str(raw.get("expression") or "")
        else:
            name, expression = f"#{index + 1}", str(raw or "")

        entry: Dict[str, Any] = {"name": name, "expression": expression, "ok": True}

        if not expression.strip():
            entry.update(ok=False, error="The calculation is empty.")
            results.append(entry)
            continue

        try:
            tree = _ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            entry.update(
                ok=False,
                error=f"This is not a valid calculation: {exc.msg}",
                position=getattr(exc, "offset", None),
            )
            results.append(entry)
            available.add(name)
            continue

        referenced = {
            node.id for node in _ast.walk(tree) if isinstance(node, _ast.Name)
        }
        called = {
            node.func.id
            for node in _ast.walk(tree)
            if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Name)
        }

        unknown_functions = sorted(called - functions)
        if unknown_functions:
            entry.update(
                ok=False,
                error=(
                    f"Unknown function {unknown_functions[0]!r}. "
                    f"Available: {', '.join(sorted(functions))}"
                ),
            )

        if check_names and entry["ok"]:
            unknown = sorted(referenced - called - available - {"True", "False", "None"})
            if unknown:
                entry.update(
                    ok=False,
                    error=(
                        f"{unknown[0]!r} is not one of the values available here."
                        + (f" Also: {', '.join(unknown[1:4])}" if len(unknown) > 1 else "")
                    ),
                    unknown_names=unknown,
                )

        # Whatever this row produces is available to the rows below it, even if
        # this one has a problem — otherwise one mistake cascades into false
        # complaints about every row that follows.
        available.add(name)
        results.append(entry)

    return {
        "ok": all(r["ok"] for r in results),
        "checked": len(results),
        "problems": sum(1 for r in results if not r["ok"]),
        "results": results,
        "available_functions": sorted(functions),
    }


@router.post("/node/test", response_model=NodeTestResponse)
async def test_node(request: NodeTestRequest) -> NodeTestResponse:
    """
    Build one node from the given configuration and run it against sample data.

    Only the keys the node ADDS or CHANGES are returned, not the whole input
    echoed back — the useful answer is "what did this node produce", and a node
    that passes its input through would otherwise bury that in noise.
    """
    import asyncio

    started = time.perf_counter()
    raw = request.node_config or {}
    node_id = str(raw.get("id") or "test_node")
    node_type = str(raw.get("type") or "")

    if not node_type:
        raise HTTPException(
            status_code=400,
            detail="node_config must include a 'type' so the server knows what to build.",
        )

    # ── Build ────────────────────────────────────────────────────────────────
    try:
        config = NodeConfig(**{**raw, "id": node_id, "type": node_type})
        node = NodeFactory.build(config)
    except Exception as exc:
        logger.info(f"[NodeTest] build failed for '{node_type}': {exc}")
        return NodeTestResponse(
            ok=False,
            node_id=node_id,
            node_type=node_type,
            error=str(exc),
            error_type=type(exc).__name__,
            hint=_hint_for(exc, node_type),
        )

    # ── Run ──────────────────────────────────────────────────────────────────
    # A deep copy so a node that mutates its input cannot alter what we compare
    # against afterwards.
    sample = copy.deepcopy(request.sample_input or {})
    state = {"messages": [], "context": sample}

    try:
        # Some builders return a plain coroutine function, others a Runnable.
        call = node.ainvoke(state) if hasattr(node, "ainvoke") else node(state)
        result = await asyncio.wait_for(call, timeout=request.timeout_seconds or 60)
    except asyncio.TimeoutError as exc:
        return NodeTestResponse(
            ok=False,
            node_id=node_id,
            node_type=node_type,
            error=f"Timed out after {request.timeout_seconds}s",
            error_type="Timeout",
            hint=_hint_for(exc, node_type) or "The node did not finish in time.",
        )
    except Exception as exc:
        logger.info(f"[NodeTest] run failed for '{node_type}': {exc}")
        return NodeTestResponse(
            ok=False,
            node_id=node_id,
            node_type=node_type,
            duration_ms=int((time.perf_counter() - started) * 1000),
            error=str(exc),
            error_type=type(exc).__name__,
            hint=_hint_for(exc, node_type),
        )

    # ── Did the node fail even though it returned normally? ──────────────────
    # Nodes are wrapped so a failure becomes an error state rather than an
    # exception — that keeps a running graph alive for error routing. When
    # testing, though, a failure is exactly what the person needs to be told
    # about, so it is reported as one.
    if isinstance(result, dict) and result.get("error"):
        node_error = result["error"]
        message = (
            node_error.get("error") or node_error.get("message") or str(node_error)
            if isinstance(node_error, dict)
            else str(node_error)
        )
        return NodeTestResponse(
            ok=False,
            node_id=node_id,
            node_type=node_type,
            duration_ms=int((time.perf_counter() - started) * 1000),
            error=message,
            error_type=(
                node_error.get("type", "NodeError")
                if isinstance(node_error, dict)
                else "NodeError"
            ),
            hint=_hint_for(Exception(message), node_type),
        )

    # ── Report only what changed ─────────────────────────────────────────────
    produced: Dict[str, Any] = {}
    if isinstance(result, dict):
        out_context = result.get("context") or {}
        for key, value in out_context.items():
            if key in _PLUMBING_KEYS:
                continue
            if key not in sample or sample[key] != value:
                produced[key] = value

    messages = []
    for message in (result.get("messages") or []) if isinstance(result, dict) else []:
        content = getattr(message, "content", None)
        if content:
            messages.append(str(content)[:2000])

    duration = int((time.perf_counter() - started) * 1000)
    logger.info(
        f"[NodeTest] '{node_type}' ran in {duration}ms, produced {len(produced)} key(s)"
    )
    return NodeTestResponse(
        ok=True,
        node_id=node_id,
        node_type=node_type,
        duration_ms=duration,
        produced=produced,
        messages=messages,
    )

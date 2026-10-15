"""
Classification Node Builder
Classifies input data using an LLM or Regex based on user-defined classes and descriptions/patterns.
"""

import json
import logging
import re
from typing import Any, Callable, Dict, Iterator, List, Tuple

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate

from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from Apps.AgentMesh.Schema.StateSchema import AgentState
from Apps.AgentMesh.Utilities.LLM.llm_factory import init_chat_model
from Apps.AgentMesh.Core.Compiler.nodes.utils import (
    get_production_data,
    get_nested_val,
    interpolate,
)

logger = logging.getLogger(__name__)

def _extract_input(state: AgentState, input_field: str | None = None, input_key: str | None = None) -> str:
    messages = state.get("messages", [])
    context = state.get("context", {})
    
    target_fields = [f for f in [input_field, input_key] if f and isinstance(f, str) and f.strip()]
    if target_fields:
        scope = get_production_data(state)
        for field in target_fields:
            val = get_nested_val(scope, field.strip())
            if val is not None and val != "":
                return val if isinstance(val, str) else json.dumps(val, default=str)
    
    # Fallback to standard OCR keys if explicit fields were not found
    for key in ["vision_result", "latest_ocr_output", "ocr", "raw_text"]:
        val = context.get(key)
        if val is not None and val != "":
            return val if isinstance(val, str) else json.dumps(val, default=str)

    if messages:
        last_msg = messages[-1]
        if hasattr(last_msg, "content"):
            return last_msg.content
        if isinstance(last_msg, dict) and "content" in last_msg:
            return last_msg["content"]
        return str(last_msg)
        
    if context:
        payload = context.get("payload")
        if payload:
            return payload if isinstance(payload, str) else json.dumps(payload, default=str)
        return json.dumps(context, default=str)
        
    return ""

# A read leaves a marker at the top of every page it transcribes. Splitting on
# it is what lets a file be judged page by page instead of as one lump.
_PAGE_MARKER = re.compile(r"^-{2,}\s*Page\s+(\d+)\s*-{2,}\s*$", re.MULTILINE)


def _split_pages(text: str) -> List[Tuple[int, str]]:
    """The text broken into its pages, or empty if it carries no page markers.

    Pages with nothing on them are dropped: a blank sheet has no category, and
    asking for one wastes a call and invites a made-up answer.
    """
    marks = list(_PAGE_MARKER.finditer(text or ""))
    if len(marks) < 2:
        return []
    pages: List[Tuple[int, str]] = []
    for index, mark in enumerate(marks):
        end = marks[index + 1].start() if index + 1 < len(marks) else len(text)
        body = text[mark.end():end].strip()
        if body:
            try:
                pages.append((int(mark.group(1)), body))
            except ValueError:
                continue
    return pages


def _summarise_pages(page_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """One verdict for a file from the verdicts on its pages.

    The file is called whatever most of its pages are, and the confidence is
    the share of pages that agree - so a file of two minds reports itself as
    one, instead of the first page speaking for the rest. Every category found
    is listed with the pages holding it, because a file that contains more than
    one kind of document has to be routed by page, not by file.
    """
    counts: Dict[str, List[int]] = {}
    for result in page_results:
        counts.setdefault(str(result.get("class", "unknown")), []).append(
            result.get("page"))
    if not counts:
        return {"class": "unknown", "confidence": 0.0, "classes_present": []}
    winner = max(counts.items(), key=lambda kv: len(kv[1]))
    identifiers: List[Dict[str, Any]] = []
    seen = set()
    for result in page_results:
        for item in result.get("identifiers") or []:
            if not isinstance(item, dict):
                continue
            key = (str(item.get("label")), str(item.get("value")))
            if key not in seen:
                seen.add(key)
                identifiers.append(item)
    return {
        "class": winner[0],
        "confidence": round(len(winner[1]) / len(page_results), 3),
        "classes_present": [
            {"class": name, "pages": sorted(p for p in pages if p is not None)}
            for name, pages in sorted(counts.items(), key=lambda kv: -len(kv[1]))
        ],
        "identifiers": identifiers,
    }


def _match_precompiled_regex(text: str, precompiled_regexes: List[Tuple[str, List[re.Pattern]]]) -> str:
    """
    Perform highly optimized pattern searching using compile-time precompiled regex matchers.
    """
    for c_name, compiled_list in precompiled_regexes:
        for compiled in compiled_list:
            if compiled.search(text):
                return c_name
    return "unknown"

def _create_result(parsed_dict: dict, node_id: str, state: AgentState,
                   output_key: str | None = None) -> dict:
    context = state.get("context", {})
    item = context.get("item")
    
    # Inject classification directly into the loop item so the Aggregator collects it
    if isinstance(item, dict):
        # We copy to avoid mutating the original dict in unpredictable ways
        item_copy = item.copy()
        item_copy["class"] = parsed_dict.get("class", "unknown")
        item_copy["confidence"] = parsed_dict.get("confidence", 1.0)
        if "ocr" in parsed_dict:
            item_copy["ocr"] = parsed_dict["ocr"]
        if "summary" in parsed_dict:
            item_copy["summary"] = parsed_dict["summary"]
            item_copy["classification_summary"] = parsed_dict["summary"]
        if "identifiers" in parsed_dict:
            item_copy["identifiers"] = parsed_dict["identifiers"]
        context["item"] = item_copy

    return {
        "messages": [AIMessage(content=json.dumps(parsed_dict, indent=2), name=f"Classification_{node_id}", response_metadata={"active_agent": node_id})],
        "active_agent": node_id,
        "context": {
            f"classification_{node_id}": parsed_dict,
            "latest_classification": parsed_dict,
            # A workflow with two classifiers needs each result under a name it
            # chose; latest_classification keeps working for those that do not.
            **({output_key.strip(): parsed_dict} if output_key and output_key.strip() else {}),
            "item": context.get("item")
        }
    }

def _create_error_result(input_data: str, summary_option: bool, node_id: str,
                         e: Exception, state: AgentState,
                         output_key: str | None = None) -> dict:
    error_res = {
        "class": "Error",
        "confidence": 0.0,
        "ocr": input_data,
        "error": str(e)
    }
    if summary_option:
        error_res["summary"] = input_data

    context = state.get("context", {})
    item = context.get("item")
    if isinstance(item, dict):
        item_copy = item.copy()
        item_copy["class"] = "Error"
        item_copy["confidence"] = 0.0
        item_copy["ocr"] = input_data
        item_copy["error"] = str(e)
        context["item"] = item_copy

    return {
        "messages": [AIMessage(content=json.dumps(error_res), name=f"Classification_{node_id}", response_metadata={"active_agent": node_id})],
        "active_agent": node_id,
        "error": {"message": str(e), "type": type(e).__name__},
        "context": {
            f"classification_{node_id}": error_res,
            "latest_classification": error_res,
            **({output_key.strip(): error_res} if output_key and output_key.strip() else {}),
            "item": context.get("item")
        }
    }

def _iter_json_objects(text: str) -> Iterator[str]:
    """
    Yield every complete top-level {...} span in the text, in order.

    Brace counting, skipping braces inside strings, so a value like
    "amount: {net}" cannot end an object early. Unterminated trailing objects
    are dropped rather than guessed at.
    """
    i, n = 0, len(text)
    while i < n:
        start = text.find("{", i)
        if start == -1:
            return
        depth, in_str, esc, closed = 0, False, False, False
        for j in range(start, n):
            ch = text[j]
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = not in_str
            elif not in_str:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        yield text[start:j + 1]
                        i, closed = j + 1, True
                        break
        if not closed:
            return


def _parse_json_lenient(text: str) -> Tuple[Any, bool]:
    """
    Parse JSON, falling back to json_repair. Returns (value, used_repair).

    json_repair handles single quotes, trailing commas and unquoted keys, which
    models produce often enough that a strict parser throws away otherwise
    usable answers. Returns (None, True) when nothing can be salvaged.
    """
    try:
        return json.loads(text), False
    except json.JSONDecodeError:
        try:
            from json_repair import loads as _repair_loads
            return _repair_loads(text), True
        except Exception:
            return None, True


def _literal(text: Any) -> str:
    """
    Escape braces so user text survives ChatPromptTemplate's f-string parsing.

    Class names and descriptions are user-supplied and land inside a
    ChatPromptTemplate, which parses the result as an f-string. A single brace
    anywhere in them — a JSON example, a regex, "{amount}" — makes LangChain
    raise "Single '}' encountered in format string" at COMPILE time, with a
    traceback pointing into langchain_core and nothing naming the class that
    caused it. Escaping here keeps user text as literal text, which is what a
    description always is.
    """
    return str(text or "").replace("{", "{{").replace("}", "}}")


def _classes_are_templated(classes: List[Dict[str, Any]]) -> bool:
    """True if any class field references a workflow variable."""
    try:
        return "{{" in json.dumps(classes, default=str)
    except Exception:
        return False


def _interpolate_classes(
    classes: List[Dict[str, Any]],
    context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Resolve {{ }} references in class names, descriptions and regex patterns.

    Runs BEFORE _literal(), never after: interpolation must see the raw
    template, and brace-escaping must see the rendered value. Reversing them
    would escape the placeholder into `{{{{ ... }}}}` and leave it unresolved.
    """
    resolved: List[Dict[str, Any]] = []
    for c in classes:
        if not isinstance(c, dict):
            resolved.append(c)
            continue
        item = dict(c)
        for key in ("class", "description"):
            val = item.get(key)
            if isinstance(val, str):
                item[key] = interpolate(val, context)
        # Jinja renders an undefined variable as empty, so a mistyped variable
        # name turns into a class named "" — which the LLM can never return and
        # which quietly matches nothing. Say so; this is a configuration error,
        # not data.
        if isinstance(c.get("class"), str) and c["class"].strip() and not str(item.get("class", "")).strip():
            logger.warning(
                f"[CLASSIFICATION] class name '{c['class']}' resolved to empty — "
                f"the variable it references is not defined in this run's context. "
                f"This class can never be selected."
            )
        patterns = item.get("regex_patterns")
        if isinstance(patterns, str):
            item["regex_patterns"] = interpolate(patterns, context)
        elif isinstance(patterns, list):
            item["regex_patterns"] = [
                interpolate(p, context) if isinstance(p, str) else p for p in patterns
            ]
        resolved.append(item)
    return resolved


def _compile_class_regexes(
    classes: List[Dict[str, Any]],
    node_id: str,
) -> List[Tuple[str, List[re.Pattern]]]:
    """Precompile each class's regex patterns, skipping ones that do not compile."""
    precompiled: List[Tuple[str, List[re.Pattern]]] = []
    for c in classes:
        c_name = c.get("class", "Unknown")
        patterns_raw = c.get("regex_patterns")
        if not patterns_raw:
            continue

        patterns = []
        if isinstance(patterns_raw, list):
            patterns = [str(p).strip() for p in patterns_raw if p]
        elif isinstance(patterns_raw, str):
            patterns = [p.strip() for p in patterns_raw.split(",") if p.strip()]

        compiled_list = []
        for pattern in patterns:
            try:
                compiled_list.append(re.compile(pattern, re.IGNORECASE))
            except re.error as e:
                logger.error(f"[CLASSIFICATION] Node {node_id}: invalid regex pattern '{pattern}' in class '{c_name}': {e}")
        if compiled_list:
            precompiled.append((c_name, compiled_list))
    return precompiled


def _build_class_prompt(
    classes: List[Dict[str, Any]],
    summary_option: bool,
    extract_identifiers: bool,
) -> ChatPromptTemplate:
    """Build the classification prompt from a resolved class list."""
    system_prompt = (
        "You are a highly accurate data classification system. Your task is to analyze the input data and classify it into EXACTLY ONE of the allowed categories based on their descriptions.\n\n"
        "Allowed Categories:\n"
    )

    for c in classes:
        c_name = _literal(c.get("class", "Unknown"))
        c_desc = _literal(c.get("description", ""))
        system_prompt += f"- Category Name: {c_name}\n  Description: {c_desc}\n\n"

    system_prompt += (
        "CRITICAL: You MUST respond ONLY with a valid JSON object. Do NOT wrap your output in markdown code blocks like ```json ... ```, and do not include any other explanatory text.\n\n"
    )

    if extract_identifiers:
        system_prompt += (
            "Additionally, extract any prominent unique identifiers found in the document (such as government IDs, account numbers, organization names, or reference/invoice numbers) into a JSON array.\n"
            # Asked for explicitly because it was not being returned otherwise.
            # A date is what tells one edition of a recurring document from the
            # next, so anything downstream that files or orders documents needs
            # it, and it is almost always printed in the heading. Named here as
            # a plain date so it applies to whatever the document happens to be.
            "Always include one identifier labelled \"Period\" giving the date or "
            "period the document relates to, copied exactly as printed (for example "
            "a document date, a period ending date, or a year). If no date appears "
            "anywhere in the text, omit this identifier rather than inventing one.\n"
        )

    system_prompt += "The JSON object MUST contain exactly these keys:\n"
    system_prompt += "1. \"class\": The exact name of the selected category matching one of the names above.\n"
    system_prompt += "2. \"confidence\": A float between 0.0 and 1.0 indicating how confident you are in the classification. Use 0.9+ for clear matches, 0.5-0.8 for uncertain, and below 0.5 if the document barely fits any category.\n"

    key_idx = 3
    if summary_option:
        system_prompt += f"{key_idx}. \"summary\": A concise summary of the input data.\n"
        key_idx += 1

    if extract_identifiers:
        system_prompt += f"{key_idx}. \"identifiers\": A JSON array of objects, each containing a 'label' and a 'value'.\n"

    system_prompt += "\nFormat example:\n{{\n"
    system_prompt += "  \"class\": \"CategoryName\""
    system_prompt += ",\n  \"confidence\": 0.92"

    if summary_option:
        system_prompt += ",\n  \"summary\": \"Summarized input content.\""

    if extract_identifiers:
        system_prompt += ",\n  \"identifiers\": [{{\"label\": \"PAN\", \"value\": \"ABCDE1234F\"}}]"

    system_prompt += "\n}}\n"

    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "INPUT DATA TO CLASSIFY:\n\n{input_data}"),
    ])


def build_classification_node(node_config: NodeConfig) -> Callable:
    """
    Build a classification node that uses an LLM or Regex without memory
    to classify input text into a set of classes with optional summarization.
    """
    engine = getattr(node_config, "classification_engine", "llm") or "llm"
    classes = getattr(node_config, "classification_classes", None) or []
    summary_option = getattr(node_config, "classification_summary", False)
    extract_identifiers = getattr(node_config, "classification_extract_identifiers", False)
    # Off by default, and deliberately so: it costs one call per page instead of
    # one per file. Worth turning on when a single file can hold more than one
    # kind of document, which is the case a single verdict per file gets wrong.
    per_page = bool(getattr(node_config, "classification_per_page", False))
    input_field = getattr(node_config, "classification_input_field", None)
    input_key = getattr(node_config, "classification_input_key", None)
    output_key = getattr(node_config, "classification_output_key", None)
    try:
        max_input_chars = int(getattr(node_config, "classification_max_input_chars", 0) or 0)
    except (TypeError, ValueError):
        max_input_chars = 0
    node_id = node_config.id

    # Class names and descriptions may reference workflow variables, so one
    # canonical list of document classes can drive several nodes instead of
    # being hand-typed into each — the same names are also needed by the JSON
    # branch, the agent routes and the scorecard, and a typo in any copy
    # silently drops documents.
    #
    # Interpolation needs run context, which does not exist while the graph is
    # being compiled. So templated classes are resolved per run, and everything
    # that derives from them is rebuilt then. When no template is present — the
    # common case — all of this stays at build time exactly as before.
    templated = _classes_are_templated(classes)

    # 1. Compile-time optimization: Precompile regex rules if engine is 'regex'
    precompiled_regexes = []
    if engine == "regex" and not templated:
        precompiled_regexes = _compile_class_regexes(classes, node_id)

    # 2. Initialize LLM prompt chain if using the LLM engine
    chain = None
    llm = None
    if engine == "llm":
        llm = init_chat_model(node_config.llm)
        if not templated:
            chain = _build_class_prompt(classes, summary_option, extract_identifiers) | llm

    # Stated at build time so a cap that never arrived is visible without
    # having to reproduce the failure it was meant to prevent.
    if max_input_chars > 0:
        logger.info(
            f"[CLASSIFICATION] Node {node_id}: prompt input capped at "
            f"{max_input_chars} chars."
        )
    else:
        logger.info(
            f"[CLASSIFICATION] Node {node_id}: no prompt input cap configured "
            f"(classification_max_input_chars is unset)."
        )

    if templated:
        logger.info(
            f"[CLASSIFICATION] Node {node_id}: classes reference workflow variables — "
            f"resolving them per run"
        )

    async def execute_classification(state: AgentState):
        context = state.get("context", {})

        # Resolve templated classes against this run's context, then rebuild
        # whatever depends on them. Bound to locals so a run never mutates the
        # build-time values shared by every other run of this graph.
        active_classes = classes
        active_chain = chain
        active_regexes = precompiled_regexes
        if templated:
            active_classes = _interpolate_classes(classes, context)
            if engine == "llm":
                active_chain = _build_class_prompt(
                    active_classes, summary_option, extract_identifiers
                ) | llm
            elif engine == "regex":
                active_regexes = _compile_class_regexes(active_classes, node_id)

        # ── PRE-CLASSIFICATION PASSTHROUGH ──
        # If the current loop item already carries a classification tag from an
        # upstream node (e.g., a field_mapper or API connector), honour it instead
        # of wasting an LLM call.  This is fully generic: any item dict that has
        # 'doc_type' or 'class' already set is treated as pre-classified.
        item = context.get("item")
        if isinstance(item, dict):
            pre_class = item.get("doc_type") or item.get("class")
            if pre_class and isinstance(pre_class, str) and pre_class.strip():
                pre_class = pre_class.strip()
                # Validate against configured classes (if any) to avoid garbage passthrough
                valid_class_names = {c.get("class", "").strip() for c in active_classes if c.get("class")}
                if not valid_class_names or pre_class in valid_class_names:
                    logger.info(
                        f"[CLASSIFICATION] Node {node_id}: Pre-classified as '{pre_class}' "
                        f"from item field — skipping LLM/regex call"
                    )
                    pre_result = {
                        "class": pre_class,
                        "confidence": 1.0,
                        "ocr": item.get("ocr") or item.get("raw_text") or item.get("original_data") or str(item),
                    }
                    if summary_option:
                        pre_result["summary"] = item.get("ai_summary") or str(item)
                    return _create_result(pre_result, node_id, state, output_key)
        
        input_data = _extract_input(state, input_field, input_key)
        
        logger.info(f"[CLASSIFICATION] Node {node_id} starting classification of input data (length={len(input_data)}). Engine={engine}, Summary={summary_option}")
        
        if not input_data.strip():
            logger.warning(f"[CLASSIFICATION] Node {node_id} received empty input data.")
            fallback_res = {
                "class": "unknown",
                "confidence": 0.0,
                "ocr": "",
            }
            if summary_option:
                fallback_res["summary"] = ""
            return _create_result(fallback_res, node_id, state, output_key)


        if engine == "regex":
            try:
                matched_class = _match_precompiled_regex(input_data, active_regexes)
                parsed = {
                    "class": matched_class,
                    "confidence": 1.0 if matched_class != "unknown" else 0.0,
                    "ocr": input_data,
                }
                if summary_option:
                    parsed["summary"] = input_data
                logger.info(f"[CLASSIFICATION] Node {node_id} (Regex) classification result: {parsed['class']}")
                return _create_result(parsed, node_id, state, output_key)
            except Exception as e:
                logger.exception(f"[CLASSIFICATION] Node {node_id} (Regex) failed")
                return _create_error_result(input_data, summary_option, node_id, e, state, output_key)

        # LLM Engine Execution Path
        async def _classify_once(text: str) -> dict:
            """Decide the category of one piece of text.

            Written as one step so it can be applied either to a whole file or
            to each of its pages in turn, without the two paths drifting apart.
            """
            # Cap ONLY what the prompt sees.
            #
            # `input_data` is also stored as parsed["ocr"] and travels onward to
            # the agents and to the review screen, so truncating it in place
            # would silently shorten every document's text — far worse than the
            # problem being solved. Two variables, deliberately.
            #
            # Why cap at all: choosing a class needs a sample, not the whole
            # document — the answer is in the header. A reasoning model handed
            # 28k characters can spend its entire output budget thinking and
            # never emit the answer, which is exactly what happened here.
            input_data = text
            prompt_input = input_data
            if max_input_chars > 0 and len(input_data) > max_input_chars:
                prompt_input = input_data[:max_input_chars]
                logger.info(
                    f"[CLASSIFICATION] Node {node_id}: prompt input capped to "
                    f"{max_input_chars} of {len(input_data)} chars "
                    f"(classification_max_input_chars). Stored text is unaffected."
                )

            response = await active_chain.ainvoke({"input_data": prompt_input})
            content = getattr(response, 'content', str(response)).strip()
            # Kept unmodified for diagnostics: the cleanup and brace scan below
            # rewrite `content`, so by the time a parse fails the original reply
            # is gone. Without it "response was not valid JSON" says nothing
            # about WHY — a reasoning model that ran out of output budget, a
            # refusal, and an empty reply all look identical.
            raw_response = content

            # Clean up potential markdown formatting from LLM response
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\n", "", content)
                content = re.sub(r"\n```$", "", content)
            content = content.strip()
            
            def _diagnostic(reason: str) -> str:
                """Describe an unusable reply, with enough of it to act on."""
                excerpt = raw_response[:500].replace("\n", "\\n")
                tail = "" if len(raw_response) <= 500 else f" ...[+{len(raw_response) - 500} more chars]"
                return (
                    f"[CLASSIFICATION] Node {node_id}: {reason}. "
                    f"Input was {len(input_data)} chars; reply was {len(raw_response)} chars"
                    f"{' and contained no {' if '{' not in raw_response else ''}. "
                    f"Raw reply: {excerpt}{tail}"
                )

            # Select the object that IS the classification.
            #
            # Taking the first complete object was right until the prompt began
            # asking for identifiers too: the model then emits objects such as
            # {"label": "PAN", "value": "..."} and, whenever it writes one before
            # the answer, "first" picks an identifier entry. The classification
            # is the object carrying a "class" key, so select on that, keeping
            # the first object as a fallback when none qualifies. A reasoning
            # block before the answer is skipped for the same reason.
            parsed = None
            used_repair = False
            fallback = None
            for span in _iter_json_objects(content):
                obj, repaired = _parse_json_lenient(span)
                if not isinstance(obj, dict):
                    continue
                if fallback is None:
                    fallback, used_repair = obj, repaired
                if "class" in obj:
                    parsed, used_repair = obj, repaired
                    break

            if parsed is None:
                parsed = fallback
            if parsed is None:
                # No complete object anywhere — parse the whole reply so the
                # not-a-dict path below reports it with the raw text.
                parsed, used_repair = _parse_json_lenient(content)

            if used_repair:
                logger.warning(
                    f"[CLASSIFICATION] Node {node_id}: response was not valid JSON; "
                    f"recovered via json_repair."
                )
            if not isinstance(parsed, dict):
                # A non-dict here means no JSON object was ever produced, so
                # json_repair fell back to reading the whole reply as a scalar.
                # Log the reply before raising: the exception alone cannot tell
                # you whether the model reasoned past its output limit, refused,
                # or answered in prose.
                logger.error(_diagnostic(f"expected a JSON object, got {type(parsed).__name__}"))
                raise ValueError(f"Expected a JSON object, got {type(parsed).__name__}")
            
            # Ensure keys exist
            class_missing = "class" not in parsed
            if class_missing:
                # Say so. A missing class defaulting silently to "unknown" is
                # how a wrong-object parse reads as a normal successful result
                # in the logs, which is exactly how this went unnoticed.
                logger.warning(
                    f"[CLASSIFICATION] Node {node_id}: reply had no 'class' key "
                    f"(keys present: {sorted(str(k) for k in parsed.keys())}). "
                    f"Falling back to 'unknown' at confidence 0.0."
                )
                parsed["class"] = "unknown"

            # Ensure confidence is a valid float.
            #
            # When the class is missing, confidence is forced to 0.0 rather than
            # defaulting to 1.0: the old default asserted total certainty about a
            # value the model never produced, and any confidence present belongs
            # to whatever object was actually parsed, not to "unknown".
            if class_missing:
                parsed["confidence"] = 0.0
            else:
                try:
                    parsed["confidence"] = float(parsed.get("confidence", 1.0))
                except (ValueError, TypeError):
                    parsed["confidence"] = 1.0
            return parsed

        try:
            # One file can hold more than one kind of document - a return with
            # the accounts it was filed with, for instance. Judged as a lump,
            # such a file takes the name of whatever happens to be on the front,
            # and everything behind it is routed as though it were that too.
            pages = _split_pages(input_data) if per_page else []
            if len(pages) > 1:
                logger.info(f"[CLASSIFICATION] Node {node_id}: reading {len(pages)} "
                            f"page(s) separately")
                page_results: List[Dict[str, Any]] = []
                for number, body in pages:
                    one = await _classify_once(body)
                    page_results.append({
                        "page": number,
                        "class": one.get("class"),
                        "confidence": one.get("confidence"),
                        **({"identifiers": one["identifiers"]}
                           if "identifiers" in one else {}),
                        **({"summary": one["summary"]} if "summary" in one else {}),
                    })
                parsed = _summarise_pages(page_results)
                parsed["pages"] = page_results
                logger.info(
                    f"[CLASSIFICATION] Node {node_id}: "
                    + "; ".join(f"{c['class']} on page(s) {c['pages']}"
                                for c in parsed.get("classes_present", []))
                )
            else:
                if per_page:
                    logger.info(f"[CLASSIFICATION] Node {node_id}: no page markers "
                                f"in the text, judging it as one document")
                parsed = await _classify_once(input_data)

            # Attach OCR text for standard schema
            parsed["ocr"] = input_data

            if summary_option and "summary" not in parsed:
                parsed["summary"] = "No summary generated."
                
            logger.info(f"[CLASSIFICATION] Node {node_id} (LLM) classification result: {parsed['class']} (confidence: {parsed.get('confidence', 'N/A')})")
            return _create_result(parsed, node_id, state, output_key)
        except Exception as e:
            logger.exception(f"[CLASSIFICATION] Node {node_id} (LLM) failed")
            return _create_error_result(input_data, summary_option, node_id, e, state, output_key)

    executor = RunnableLambda(execute_classification)
    setattr(executor, "name", node_id)
    return executor

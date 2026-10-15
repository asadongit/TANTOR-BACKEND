import logging
import re
from typing import Optional, Literal, Annotated, Dict, Any, List
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from datetime import datetime

from Apps.AgentMesh.Utilities.State.ToolStateManager import ToolStateManager
from Apps.AgentMesh.Schema.BlackboardSchema import BlackboardItem, BlackboardState
from Apps.AgentMesh.Schema.ToolSchema import ToolMeta
from Apps.AgentMesh.Core.ToolRegistry import registry

logger = logging.getLogger(__name__)

# --- 1. Standardized Input Schema ---

class BlackboardInput(BaseModel):
    action: str = Field(..., description="The operation: 'create', 'read', 'edit', 'done', 'delete', 'clear'.")
    data: Optional[Any] = Field(None, description="Note content. Supports strings or structured JSON objects.")
    category: Optional[str] = Field(None, description="Category or tag (e.g., 'plan', 'feedback', 'results'). Defaults to 'general' on create, and all categories on read.")
    item_id: Optional[int] = Field(None, description="Unique integer ID of the item.")
    pad_id: Optional[str] = Field(None, description="Optional manual Blackboard ID.")
    start: Optional[int] = Field(None, description="Optional start range ID for reading.")
    end: Optional[int] = Field(None, description="Optional end range ID for reading.")
    agent_name: Optional[str] = Field(None, description="Optional name of the agent posting or updating this item.")
    pattern: Optional[str] = Field(None, description="Regex pattern to filter items by content. Uses Python re syntax, case-insensitive. E.g. 'price|model', '\\d{4}', 'error.'.")
    patterns: Optional[List[str]] = Field(None, description="List of multiple regex patterns/search terms. Matches if ANY pattern matches (OR logic).")
    # Resilience Aliases (Legacy support)
    task_text: Optional[Any] = Field(None, description="Alias for 'data'.")
    content: Optional[Any] = Field(None, description="Alias for 'data'.")
    task: Optional[Dict[str, Any]] = Field(None, description="Structured object support.")


# --- 2. Action Sub-Handlers ---

def _handle_create(state: BlackboardState, data: Optional[str], category: Optional[str], thread_id: str, agent_name: str) -> str:
    if not data: return "❌ Error: 'data' is required."
    category = category or "general"
    
    # 🧠 Detect JSON structure to prevent splitting brackets/elements
    import json
    is_json = False
    try:
        parsed = json.loads(data)
        if isinstance(parsed, (dict, list)):
            is_json = True
    except:
        pass
        
    if is_json:
        clean_items = [data.strip()]
    else:
        # Split ONLY by newline to avoid splitting sentences on commas
        items = [line.strip() for line in data.split('\n') if line.strip()]
        clean_items = []
        for item in items:
            clean = re.sub(r'^(\d+\.|-|\*)\s*', '', item).strip()
            if clean: clean_items.append(clean)

    added_count = 0
    for clean_text in clean_items:
        new_item = BlackboardItem(
            id=state.next_id, 
            content=clean_text, 
            status="todo",
            category=category or "general",
            agent_name=agent_name
        )
        state.tasks.append(new_item)
        state.next_id += 1
        added_count += 1
    return f"✅ Posted {added_count} notes by agent '{agent_name}' back to category '{category}' on blackboard '{thread_id}'."

def _handle_read(state: BlackboardState, category: Optional[str], thread_id: str, start: Optional[int], end: Optional[int], pattern: Optional[str] = None, patterns: Optional[List[str]] = None) -> str:
    if not state.tasks: return f"📋 Blackboard '{thread_id}' is empty."
    
    total_items = len(state.tasks)
    all_content = "\n".join([f"{t.id}. {t.content}" for t in state.tasks])
    approx_tokens = int(len(all_content) / 4)
    
    # 1. Collect all search terms
    search_patterns = []
    if pattern:
        search_patterns.append(pattern)
    if patterns:
        search_patterns.extend(patterns)
        
    compiled_patterns = []
    if search_patterns:
        for p in search_patterns:
            try:
                compiled_patterns.append(re.compile(p, re.IGNORECASE))
            except re.error:
                return f"❌ Invalid regex pattern: '{p}'"

    # 2. If patterns exist, perform a global search across ALL items on the blackboard (ignoring categories and range)
    if compiled_patterns:
        items = [t for t in state.tasks if any(cp.search(t.content) for cp in compiled_patterns)]
        if items:
            pattern_msg = f" (Patterns: {', '.join(f'/{p}/' for p in search_patterns)})"
            output = [f"📋 Agent Blackboard: {thread_id}{pattern_msg}"]
            for t in items:
                icon = "✅" if t.status == "done" else "⬜"
                output.append(f"{t.id}. {icon} {t.content} #{t.category} (by {t.agent_name})")
            return "\n".join(output)
        else:
            return f"📋 No notes matching patterns {', '.join(f'/{p}/' for p in search_patterns)} found on the blackboard."

    # 3. If no patterns, fetch latest data (default to last 10 items if range is not specified)
    if start is None and end is None:
        items = state.tasks[-10:]
        range_msg = " (Showing last 10 recent notes)"
    else:
        items = []
        for t in state.tasks:
            if start is not None and t.id < start:
                continue
            if end is not None and t.id > end:
                continue
            items.append(t)
        range_msg = f" (Range: {start or '*'}-{end or '*'})"

    output = [f"📋 Agent Blackboard: {thread_id}{range_msg}"]
    for t in items:
        icon = "✅" if t.status == "done" else "⬜"
        output.append(f"{t.id}. {icon} {t.content} #{t.category} (by {t.agent_name})")
        
    output.append(
        f"\n[Blackboard Metadata]\n"
        f"Total items on blackboard: {total_items}\n"
        f"Approximate total blackboard size: {approx_tokens} tokens\n"
        f"Tip: If you did not find what you were looking for, perform a search using the 'pattern' or 'patterns' argument."
    )
    return "\n".join(output)

def _handle_edit(state: BlackboardState, item_id: Optional[int], data: Optional[str], thread_id: str) -> str:
    if item_id is None or not data: return "❌ Error: ID and Data required."
    target = next((t for t in state.tasks if t.id == item_id), None)
    if not target: return "❌ Error: Not found."
    target.content = data.strip()
    target.updated_at = datetime.now().isoformat()
    return f"✏️ Updated Note [{target.id}]"

def _handle_done(state: BlackboardState, item_id: Optional[int], thread_id: str) -> str:
    if item_id is None: return "❌ Error: ID required."
    target = next((t for t in state.tasks if t.id == item_id), None)
    if not target: return "❌ Error: Not found."
    target.status = "done"
    target.updated_at = datetime.now().isoformat()
    return f"✅ Resolved [{target.id}]"

def _handle_delete(state: BlackboardState, item_id: Optional[int], thread_id: str) -> str:
    if item_id is None: return "❌ Error: ID required."
    orig = len(state.tasks)
    state.tasks = [t for t in state.tasks if t.id != item_id]
    return f"🗑️ Deleted" if len(state.tasks) < orig else "❌ Error: Not found."


# --- 3. Main Tool ---

@tool("agent_blackboard", args_schema=BlackboardInput)
async def agent_blackboard(
    action: str,
    data: Optional[Any] = None,
    category: Optional[str] = None,
    item_id: Optional[int] = None,
    pad_id: Optional[str] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
    agent_name: Optional[str] = None,
    pattern: Optional[str] = None,
    task_text: Optional[Any] = None,
    content: Optional[Any] = None,
    task: Optional[Dict[str, Any]] = None,
    config: RunnableConfig = None
) -> str:
    """A shared workspace for multi-agent collaboration, execution plans, and human handoffs.
    CRITICAL RULES:
    1. The data stored on this blackboard must be plain English, summarized Markdown bullet or numbered points. Do NOT store raw JSON objects or strings (e.g. do not write '{"key": "value"}'). Convert all details into clean text summaries.
    2. When using the 'edit' action to update a point, the new content completely replaces the old content. You MUST retain and carry forward all other unrelated details currently inside that point so they are not lost.
    """
    TOOL_KEY = "agent_blackboard"
    configurable = config.get("configurable", {}) if config else {}
    thread_id = pad_id or configurable.get("thread_id") or "default_thread"
    
    # Resolve agent name
    if not agent_name:
        agent_name = configurable.get("active_agent") or "system"
    
    # 🧠 Resilience Aliases
    if action in ["add", "new", "insert"]: action = "create"
    if action in ["remove", "rm", "delete"]: action = "delete"
    if action in ["finish", "complete", "done"]: action = "done"
    if isinstance(task, dict):
        data = data or task.get("text") or task.get("content") or task.get("description")
        category = category or task.get("category")
        if "id" in task:
            try:
                item_id = int(task["id"])
            except:
                pass
    data = data or task_text or content

    # Ensure category is normalized to lowercase with underscores for consistency
    if category and str(category).lower().strip() != "all":
        category = str(category).lower().strip().replace(" ", "_")

    # If data is a string digit, try to map to item_id for delete/done
    if action in ["delete", "done"] and item_id is None and data is not None:
        try:
            item_id = int(str(data).strip().replace(".", ""))
        except ValueError:
            pass

    # 🔄 JSON Serialization for structured data
    import json
    if data and not isinstance(data, str):
        try:
            data = json.dumps(data, indent=2)
        except:
            data = str(data)

    # State Load
    raw = await ToolStateManager.get_state(thread_id, TOOL_KEY)
    state = BlackboardState(**raw) if raw else BlackboardState(pad_id=thread_id, tasks=[], next_id=1)

    try:
        if action == "create": result = _handle_create(state, data, category, thread_id, agent_name)
        elif action == "read": result = _handle_read(state, category, thread_id, start, end, pattern)
        elif action == "edit": result = _handle_edit(state, item_id, data, thread_id)
        elif action == "done": result = _handle_done(state, item_id, thread_id)
        elif action == "delete": result = _handle_delete(state, item_id, thread_id)
        elif action == "clear": 
            state.tasks = []
            state.next_id = 1
            result = "🧹 Blackboard Cleared"
        else: return f"❌ Error: Unsupported action '{action}'."

        await ToolStateManager.save_state(thread_id, TOOL_KEY, state.model_dump())
        return result
    except Exception as e:
        logger.error(f"Blackboard Error: {str(e)}")
        return f"❌ System Error: {str(e)}"

# --- 4. Registry ---

def register_blackboard_tools():
    registry.register(
        agent_blackboard,
        ToolMeta(
            name="agent_blackboard",
            display_name="Agent Execution Blackboard",
            description="A shared collaborative canvas for multi-agent coordination and human feedback. Stored information should be summarized and contain only important information.",
            category="Utilities",
            package="built-in",
            version="2.1.0",
            ai_visible_fields=["action", "data", "category", "item_id", "start", "end", "agent_name", "pattern"]
        )
    )

register_blackboard_tools()

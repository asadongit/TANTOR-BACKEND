from typing import TypedDict, Annotated, Dict, Any, Optional
from langgraph.graph.message import add_messages


def pick_last_reducer(old_val: Any, new_val: Any) -> Any:
    """Reducer that simply takes the latest value. Useful for parallel nodes updating shared keys."""
    return new_val

def merge_context(old_val: Dict[str, Any], new_val: Dict[str, Any]) -> Dict[str, Any]:
    """Reducer that merges new context updates into the existing context dictionary."""
    if old_val is None: return new_val or {}
    if new_val is None: return old_val
    
    res = {**old_val, **new_val}
    
    # CRITICAL FIX: Deep-merge 'nodes' to prevent parallel execution branches from overwriting each other
    if "nodes" in old_val and "nodes" in new_val:
        res["nodes"] = {**old_val["nodes"], **new_val["nodes"]}
        
    return res

def merge_vfs(old_val: Dict[str, Any], new_val: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhanced Reducer (Apr 2026) for the Virtual Filesystem.
    Recursively merges file and directory structures between agents.
    """
    if old_val is None: return new_val or {"files": {}, "dirs": []}
    if new_val is None: return old_val
    
    res = old_val.copy()
    
    # Deep merge files
    if "files" in new_val:
        res.setdefault("files", {}).update(new_val["files"])
    
    # Merge directories (unique list)
    if "dirs" in new_val:
        res["dirs"] = list(set(res.get("dirs", []) + new_val["dirs"]))
        
    return res

class AgentState(TypedDict):
    """
    Represents the state of the graph during execution within the AgentMesh Engine.
    Follows LangGraph 1.1+ standards for robust state handling.
    """
    # The messages key uses the standard `add_messages` reducer to safely append 
    # new messages to the conversation history.
    messages: Annotated[list, add_messages]
    
    # Shared artifacts, intermediate variables, or structured data processed by nodes
    context: Annotated[Dict[str, Any], merge_context]
    
    # Dynamic intent (used for routing)
    intent: Annotated[Optional[str], pick_last_reducer]
    
    # Used by router nodes/supervisors to dynamically dictate the next node.
    next: Annotated[Optional[str], pick_last_reducer]
    
    # Tracks who sent the last message (useful in swarm/handoff mode)
    sender: Annotated[Optional[str], pick_last_reducer]
    
    # Tracks the currently active agent or node running
    active_agent: Annotated[Optional[str], pick_last_reducer]

    # 🔄 Error Handling State
    error: Annotated[Optional[Dict[str, Any]], pick_last_reducer]
    error_source: Annotated[Optional[str], pick_last_reducer]

    # 🔄 Iterative Loop State (Self-Correction)
    iter_next: Annotated[Optional[str], pick_last_reducer]
    iter_feedback: Annotated[Optional[str], pick_last_reducer]

    # 📁 Persistent Virtual Filesystem (VFS)
    vfs: Annotated[Dict[str, Any], merge_vfs]

    # 📥 User input payload (used by Form Nodes to capture human input)
    user_input: Annotated[Optional[Dict[str, Any]], pick_last_reducer]

    # 🧠 Persistent running conversation summaries (per-agent), merged across nodes.
    # Each value is a dict for a node_id:
    #   {"summary": str, "summarized_ids": list[str], "turn_count": int}
    # Backed by the checkpointer so summary_buffer_memory accumulates facts across
    # turns and survives restarts (mirrors LangGraph RunningSummary).
    memory_summary: Annotated[Dict[str, Any], merge_context]

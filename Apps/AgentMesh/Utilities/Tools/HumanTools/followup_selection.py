import json
from typing import List, Optional
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from Apps.AgentMesh.Schema.ToolSchema import ToolMeta
from Apps.AgentMesh.Core.ToolRegistry import registry

@tool
def followup_selection(question: str, options: Optional[List[str]] = None, allow_custom: bool = False, config: RunnableConfig = None) -> dict:
    """
    Ask the user a follow-up question with optional multiple-choice options.
    Use this tool when you need the user's input/choice to proceed.
    
    Args:
        question: The text of the question to ask the user.
        options: Optional list of options the user can select.
        allow_custom: If True, allows the user to type a custom text response.
    """
    if not options or len(options) == 0:
        options = []
        allow_custom = True
        
    return {
        "__human_interaction__": True,
        "pause_config": {
            "type": "followup_selection",
            "question": question,
            "options": options,
            "allow_custom": allow_custom
        }
    }

# Register the tool immediately upon import
registry.register(
    followup_selection,
    ToolMeta(
        name="followup_selection",
        display_name="Follow-up Questions Selection",
        description="Presents a question and choices/custom input to the user inline and waits for selection.",
        category="Human",
        ai_visible_fields=["question", "options", "allow_custom"]
    )
)

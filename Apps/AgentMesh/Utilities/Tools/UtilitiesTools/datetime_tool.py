from langchain_core.tools import tool
from Apps.AgentMesh.Schema.ToolSchema import ToolMeta
from Apps.AgentMesh.Core.ToolRegistry import registry

@tool
def current_datetime() -> str:
    """Get the current date and time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Register the tool immediately upon import
registry.register(
    current_datetime, 
    ToolMeta(
        name="current_datetime",
        display_name="Current Date & Time",
        description="Returns the current local date and time.",
        category="Time"
    )
)

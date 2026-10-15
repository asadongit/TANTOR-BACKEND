from langchain_community.tools import DuckDuckGoSearchRun
from Apps.AgentMesh.Schema.ToolSchema import ToolMeta
from Apps.AgentMesh.Core.ToolRegistry import registry

# Initialize the DuckDuckGo search tool
ddg_search = DuckDuckGoSearchRun()

# Register the tool immediately upon import
registry.register(
    ddg_search, 
    ToolMeta(
        name="duckduckgo_search",
        display_name="DuckDuckGo Search",
        description="Search the web using DuckDuckGo to get real-time information, news, and facts.",
        category="Search",
        package="built-in",
        version="1.1.0",
    )
)

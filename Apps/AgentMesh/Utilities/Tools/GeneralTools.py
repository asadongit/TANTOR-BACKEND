import wikipedia as wiki_lib
from datetime import datetime
from langchain_core.tools import StructuredTool
from langchain_community.utilities import WikipediaAPIWrapper
from Apps.AgentMesh.Schema.ToolSchema import ToolMeta
from Apps.AgentMesh.Core.ToolRegistry import registry

def wikipedia_search(query: str, user_agent: str = "AgentMesh/2.0 (admin@translabtechnologies.com)") -> str:
    """Search Wikipedia for general knowledge, biographies, and historical facts."""
    # Set the user agent dynamically for this request
    wiki_lib.set_user_agent(user_agent)
    api_wrapper = WikipediaAPIWrapper()
    return api_wrapper.run(query)

# Initialize Wikipedia tool as a StructuredTool to expose user_agent dynamically
wikipedia = StructuredTool.from_function(
    func=wikipedia_search,
    name="wikipedia_search",
    description="Search Wikipedia for general knowledge, biographies, and historical facts."
)

# Register the tool
registry.register(
    wikipedia,
    ToolMeta(
        name="wikipedia_search",
        display_name="Wikipedia Search",
        description="Search Wikipedia for general knowledge, biographies, and historical facts.",
        category="Search"
    )
)

def get_current_date_time() -> str:
    """Retrieve the current local date and time in YYYY-MM-DD HH:MM:SS format."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

get_date_time_tool = StructuredTool.from_function(
    func=get_current_date_time,
    name="get_date_time",
    description="Retrieve the current local date and time."
)

# Register the tool
registry.register(
    get_date_time_tool,
    ToolMeta(
        name="get_date_time",
        display_name="Get Date Time",
        description="Retrieve the current local date and time.",
        category="Utility"
    )
)

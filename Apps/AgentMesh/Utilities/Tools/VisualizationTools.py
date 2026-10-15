from typing import Any, Optional
from langchain_core.tools import tool
from Apps.AgentMesh.Schema.ToolSchema import ToolMeta
from Apps.AgentMesh.Core.ToolRegistry import registry
from Apps.AgentMesh.Utilities.LLM.llm_factory import init_chat_model
import json

@tool
async def chart_generator(
    data: Any, 
    instruction: str = "Generate a suitable chart",
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_temperature: Optional[float] = 0.1,
    llm_api_key_override: Optional[str] = None,
    llm_base_url: Optional[str] = None
):
    """
    Transforms data into a visual chart JSON.
    'data': The raw data or summary to be visualized.
    'instruction': specific instructions like 'make it a bar chart for monthly sales'.
    Returns a JSON string wrapped in a special block for the frontend.
    """
    if llm_provider or llm_model:
        from Apps.AgentMesh.Schema.WorkflowSchema import LLMConfig
        llm_config = LLMConfig(
            provider=llm_provider or "openai",
            model=llm_model or "gpt-4o",
            temperature=llm_temperature if llm_temperature is not None else 0.1,
            api_key_override=llm_api_key_override,
            base_url=llm_base_url
        )
        llm = init_chat_model(llm_config)
    else:
        llm = init_chat_model() # Default LLM
    
    prompt = f"""
    You are a data visualization expert.
    Transform the following data into a structured JSON format for Recharts.
    
    Data: {data}
    Instruction: {instruction}
    
    The output must ONLY be a JSON object with:
    - type: "bar" | "line" | "pie" | "area"
    - title: string
    - data: Array of objects with keys corresponding to the axes.
    - xKey: The key for the X axis (or name in pie).
    - yKeys: Array of keys for the Y axis (values).
    
    Example:
    {{
      "type": "bar",
      "title": "Monthly Sales",
      "data": [ {{"name": "Jan", "sales": 400}}, {{"name": "Feb", "sales": 300}} ],
      "xKey": "name",
      "yKeys": ["sales"]
    }}
    
    IMPORTANT: Respond ONLY with valid JSON. No other text.
    
    JSON Output:
    """
    
    response = await llm.ainvoke(prompt)
    content = response.content.strip()
    
    # Clean up markdown code blocks if the LLM adds them
    if content.startswith("```json"):
        content = content[7:-3].strip()
    elif content.startswith("```"):
        content = content[3:-3].strip()
    
    # Return wrapped in a special format for the frontend to detect
    return f"```chart\n{content}\n```"

# Register the tool
registry.register(
    chart_generator,
    ToolMeta(
        name="chart_generator",
        display_name="Chart Generator",
        description="Converts data into interactive visual charts (Bar, Line, Pie, etc.) using AI.",
        category="Visualization",
        configs=[
            {"name": "llm_provider", "type": "string", "description": "LLM Provider (default: openai)", "required": False, "default": "openai"},
            {"name": "llm_model", "type": "string", "description": "LLM Model Name (default: gpt-4o)", "required": False, "default": "gpt-4o"},
            {"name": "llm_temperature", "type": "number", "description": "LLM Temperature (default: 0.1)", "required": False, "default": 0.1},
            {"name": "llm_api_key_override", "type": "string", "description": "API Key override", "required": False, "default": ""},
            {"name": "llm_base_url", "type": "string", "description": "Base URL override", "required": False, "default": ""}
        ]
    )
)

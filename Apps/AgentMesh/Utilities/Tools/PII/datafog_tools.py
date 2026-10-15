from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from Apps.AgentMesh.Schema.ToolSchema import ToolMeta
from Apps.AgentMesh.Core.ToolRegistry import registry
import datafog

@tool
def redact_pii(text: str, engine: str = "regex", config: RunnableConfig = None) -> str:
    """
    Scans and redacts Personally Identifiable Information (PII) from text 
    to ensure data privacy and security.
    
    Args:
        text (str): The input text to be sanitized.
    """
    try:
        # DataFog sanitize logic
        result = datafog.sanitize(text, engine=engine)
        return result
    except Exception as e:
        return f"Error redacting PII: {str(e)}"

@tool
def scan_pii_entities(text: str, engine: str = "regex", config: RunnableConfig = None) -> str:
    """
    Scans text for Personally Identifiable Information (PII) entities and 
    lists what it found. This does NOT mask the data.
    
    Args:
        text (str): The input text to scan.
    """
    try:
        result = datafog.scan_prompt(text, engine=engine)
        # Scan result usually contains an entities list or object
        if hasattr(result, 'entities') and result.entities:
            # Format entities for readability
            entities_info = []
            for ent in result.entities:
                label = getattr(ent, 'label', 'Unknown')
                text_val = getattr(ent, 'text', 'Unknown')
                entities_info.append(f"[{label}: {text_val}]")
            
            return f"Detected {len(result.entities)} PII entities: " + ", ".join(entities_info)
        
        return "No PII entities detected."
    except Exception as e:
        return f"Error scanning for PII: {str(e)}"

def register_pii_tools():
    meta_redact = ToolMeta(
        name="redact_pii",
        display_name="PII: Redact Data",
        description="Masks sensitive information like emails, phone numbers, and SSNs from text.",
        category="Security",
        package="core",
        version="1.0.0",
        ai_visible_fields=["text"], # Hide 'engine' from LLM
        configs=[
            {
                "name": "engine", 
                "type": "select", 
                "options": ["regex", "gliner"], 
                "default": "regex",
                "description": "The detection engine. 'regex' is fast; 'gliner' is smarter."
            }
        ]
    )
    
    meta_scan = ToolMeta(
        name="scan_pii_entities",
        display_name="PII: Scan Entities",
        description="Identifies and lists PII entities without redacting them.",
        category="Security",
        package="core",
        version="1.0.0",
        ai_visible_fields=["text"], # Hide 'engine' from LLM
        configs=[
            {
                "name": "engine", 
                "type": "select", 
                "options": ["regex", "gliner"], 
                "default": "regex",
                "description": "The detection engine to use for scanning."
            }
        ]
    )
    
    registry.register(redact_pii, meta_redact)
    registry.register(scan_pii_entities, meta_scan)

register_pii_tools()

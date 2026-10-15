from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ToolConfigMeta(BaseModel):
    """Configuration specific to the tool that might be collected from the frontend"""
    name: str = Field(..., description="Configuration parameter name")
    type: str = Field(..., description="Data type (string, int, boolean, password)")
    description: str = Field(..., description="Description for the UI")
    required: bool = Field(True, description="Is this config required?")
    
    # NEW: Support for Dropdowns and Defaults
    options: Optional[List[str]] = Field(None, description="Optional list of values for a Dropdown/Select input")
    default: Optional[Any] = Field(None, description="Default value to pre-fill the field")
    visible_if: Optional[Dict[str, Any]] = Field(None, description="Conditional visibility rules, e.g. {'field': 'auth_method', 'value': 'smtp'}")

class ToolMeta(BaseModel):
    """Metadata describing a single tool for the frontend Visual Editor"""
    name: str = Field(..., description="Unique code-level name of the tool")
    display_name: str = Field(..., description="Human readable name for the frontend")
    description: str = Field(..., description="Detailed description of what the tool does")
    category: str = Field("General", description="Category for grouping in the tool palette (e.g. Database, Search, File)")
    package: Optional[str] = Field("built-in", description="Package name for grouping in the tool page (e.g. Azure, AWS, Google)")
    version: Optional[str] = Field("1.0.0", description="Version of the tool package")
    configs: Optional[List[ToolConfigMeta]] = Field(default_factory=list, description="Any expected configurations from the UI for this tool")
    allow_duplicate: bool = Field(True, description="Can this tool have multiple instances inside a single agent?")
    ai_visible_fields: Optional[List[str]] = Field(default=None, description="Explicit list of fields the LLM is allowed to see. If None, all non-overridden fields are visible.")
    args: Optional[Dict[str, Any]] = Field(None, description="JSON Schema representing the expected input arguments for this tool")

class ToolListResponse(BaseModel):
    """Response model for the /tools API endpoint"""
    tools: List[ToolMeta] = Field(..., description="List of all available tools")

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from .base import EdgeConfig
from .node import NodeConfig
from .triggers import TriggerConfig


class WorkflowConfig(BaseModel):
    """Root configuration for an entire executable AgentMesh Pipeline"""

    id: str = Field(..., description="Global unique ID of the workflow")
    name: str = Field(..., description="Human readable name")
    description: Optional[str] = None
    group: Optional[str] = Field(
        "Default", description="Group/Folder name for organization"
    )
    category: Optional[str] = Field("General", description="Workflow category")
    triggers: Optional[TriggerConfig] = Field(
        None, description="Triggers that can automatically start this workflow"
    )
    nodes: List[NodeConfig] = Field(
        ..., description="List of all nodes in this workflow"
    )
    edges: List[EdgeConfig] = Field(..., description="List of all connections (edges)")
    recursion_limit: int = Field(
        100,
        description="Maximum number of node transitions before hitting a stop condition.",
    )
    dynamic_resumption: bool = Field(
        False,
        description="If true, the workflow will resume from the last active agent instead of starting from the beginning.",
    )
    warmup: bool = Field(
        False,
        description="If true, this workflow is pre-compiled at server startup for zero cold-start latency.",
    )
    auto_new_session: bool = Field(
        False,
        description="If true, every run/trigger of the workflow automatically creates a new session/thread instead of retaining the active chat thread.",
    )

    # Global Error Handling
    global_error_node: Optional[str] = Field(
        None,
        description="Global node ID for handling failures if on_error is not set per-node.",
    )

    # Context Schema for Global Validation
    context_schema: Optional[Dict[str, Any]] = Field(
        None, description="Global schema for context data validation"
    )

    # Frontend-only visual metadata: stores canvas-only nodes (chat_model, memory visual helpers)
    # and their connection edges so the visual layout can be fully restored after save/load.
    # This field is stored in the DB but ignored by the compiler.
    frontend_meta: Optional[Dict[str, Any]] = Field(
        None,
        description="Frontend-only visual state: param nodes (chat_model, memory), their edges, and canvas positions. Not used by the compiler.",
    )

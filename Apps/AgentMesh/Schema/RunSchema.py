from pydantic import BaseModel, Field, field_validator
import re
from typing import Dict, Any, List, Optional, Literal
from datetime import datetime

class RunWorkflowRequest(BaseModel):
    workflow_id: str
    input: str
    thread_id: Optional[str] = "default_thread"
    context: Optional[Dict[str, Any]] = {}
    target_node_id: Optional[str] = None
    trim_context: Optional[bool] = None

class RunResponse(BaseModel):
    status: str
    output: Optional[str] = None
    history_count: int
    active_agent: Optional[str] = None
    session_id: str # thread_id

class StreamEvent(BaseModel):
    """Schema for SSE streaming events"""
    type: str # token, node_start, node_end, tool_call, tool_result, interrupt, final, error
    content: Optional[str] = None
    node: Optional[str] = None
    tool: Optional[str] = None
    timestamp: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class RunWorkflowResponse(BaseModel):
    status: str
    output: Optional[str] = None
    thread_id: str
    is_paused: bool = False
    next_node: Optional[str] = None
    next_nodes: List[str] = []
    messages: List[Dict[str, Any]] = []
    active_agent: Optional[str] = None
    history_count: int = 0
    rework_applied: bool = False
    context: Dict[str, Any] = {}

class StreamWorkflowRequest(BaseModel):
    input: str
    thread_id: Optional[str] = "default_thread"
    context: Optional[Dict[str, Any]] = {}

class ResumeWorkflowRequest(BaseModel):
    thread_id: str
    action: Literal["approve", "rework", "reject", "submit"]
    payload: Optional[Dict[str, Any]] = None
    trim_context: Optional[bool] = None

class ExecuteWorkflowRequest(BaseModel):
    mode: Literal["run", "stream", "resume"] = "stream"
    input: Optional[str] = None
    thread_id: Optional[str] = "default_thread"
    context: Optional[Dict[str, Any]] = {}
    target_node_id: Optional[str] = None
    action: Optional[Literal["approve", "rework", "reject", "submit"]] = None
    payload: Optional[Dict[str, Any]] = None
    trim_context: Optional[bool] = None

class IngestTextRequest(BaseModel):
    text: str
    collection_name: str = Field("default", min_length=1, max_length=100)
    vector_store: str = "chroma"
    metadata: Optional[Dict[str, Any]] = None
    chunk_size: int = Field(1000, ge=100, le=10000)
    chunk_overlap: int = Field(100, ge=0)
    chunking_strategy: str = "recursive"

    @field_validator("collection_name")
    @classmethod
    def validate_collection_name(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9_\-]+$", v):
            raise ValueError("collection_name must contain only alphanumeric characters, underscores, or hyphens")
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, v: int, info) -> int:
        if "chunk_size" in info.data and v >= info.data["chunk_size"]:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v

class ThreadSession(BaseModel):
    thread_id: str
    workflow_id: str
    first_message: str
    created_at: datetime
    updated_at: datetime

class ThreadListResponse(BaseModel):
    status: str
    threads: List[ThreadSession]

class ThreadHistoryResponse(BaseModel):
    status: str
    thread_id: str
    messages: List[Dict[str, Any]]
    is_paused: bool = False
    next_node: Optional[str] = None
    active_agent: Optional[str] = None
    context: Dict[str, Any] = {}
    pause_type: Optional[str] = None
    pause_config: Optional[Dict[str, Any]] = None

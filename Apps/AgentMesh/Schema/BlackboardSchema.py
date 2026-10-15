from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime

class BlackboardItem(BaseModel):
    id: int = Field(..., description="Sequential integer ID of the item")
    content: str = Field(..., description="The note content")
    status: Literal["todo", "in_progress", "done"] = Field("todo", description="Current status")
    category: Optional[str] = Field("general", description="The category or tag for this item")
    agent_name: Optional[str] = Field("system", description="Name of the agent that posted this item")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())

class BlackboardState(BaseModel):
    pad_id: str
    tasks: List[BlackboardItem] = []
    next_id: int = 1

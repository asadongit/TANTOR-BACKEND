from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class WebhookTrigger(BaseModel):
    """Configuration for a webhook trigger"""

    enabled: bool = Field(True, description="Whether the webhook trigger is active")
    path: Optional[str] = Field(
        None, description="Optional custom webhook path, defaults to workflow ID"
    )
    secret: Optional[str] = Field(
        None, description="Optional secret key for validation", repr=False
    )
    auth_method: str = Field(
        "header",
        description="Auth method: 'header' (X-AgentMesh-Secret), 'hmac' (SHA256 signature), or 'none'",
    )
    response_mode: str = Field(
        "immediate",
        description="Response mode: 'immediate' (202 Accepted) or 'on_completion' (Wait for result)",
    )
    response_node: Optional[str] = Field(
        None,
        description="Specific node ID to return data from. If None, returns final state.",
    )


class CronTrigger(BaseModel):
    """Configuration for a scheduled cron trigger"""

    enabled: bool = Field(True, description="Whether the cron trigger is active")
    cron_expression: str = Field(
        ..., description="Cron expression (e.g., '0 9 * * *' for 9 AM daily)"
    )
    timezone: str = Field("UTC", description="Timezone for the schedule")


class PollingTrigger(BaseModel):
    """Configuration for an active programmatic background polling trigger"""

    enabled: bool = Field(True, description="Whether the polling trigger is active")
    poller_tool: str = Field(
        ..., description="The registered tool name to execute (e.g. 'fetch_emails')"
    )
    tool_arguments: Dict[str, Any] = Field(
        default_factory=dict, description="Parameters passed directly to the tool"
    )
    destination_key: str = Field(
        "payload", description="Context key where new items are placed for the workflow"
    )
    cron_expression: str = Field(
        "*/5 * * * *",
        description="Cron schedule to run the poller (e.g., '*/5 * * * *')",
    )
    timezone: str = Field("UTC", description="Timezone for the cron expression")


class TriggerConfig(BaseModel):
    """Container for various triggers that can start a workflow"""

    webhook: Optional[WebhookTrigger] = None
    cron: Optional[CronTrigger] = None
    polling: Optional[PollingTrigger] = None

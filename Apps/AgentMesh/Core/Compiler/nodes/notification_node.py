"""
Notification Node Builder
Stores a notification in the database when hit during a workflow.
"""

import logging
from typing import Callable, Any, Dict
from datetime import datetime

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from Apps.AgentMesh.Schema.StateSchema import AgentState
from Database import AgentMeshCrud
from Database.Database import AsyncSessionLocal
from Apps.AgentMesh.Core.WebSocketManager import ws_manager
from .utils import interpolate

logger = logging.getLogger(__name__)

def build_notification_node(node_config: NodeConfig) -> Callable:
    """
    Build a Notification execution node.
    Stores a notification in the system database for the user to see.
    """
    async def execute_notification(state: AgentState):
        context = state.get("context", {})
        # FIX: Check context for original thread_id (session id) first, otherwise fallback to root state
        thread_id = context.get("thread_id") or state.get("thread_id") or "local_run"
        workflow_id = context.get("workflow_id")
        
        logger.info(f"Notification Node {node_config.id}: Storing notification for thread {thread_id}")

        # 1. Prepare Content (with interpolation)
        raw_title = node_config.notification_title or f"Workflow Alert: {node_config.id}"
        raw_message = node_config.notification_message or f"The workflow hit notification node '{node_config.id}'."
        
        title = interpolate(raw_title, context)
        message = interpolate(raw_message, context)
        ntype = node_config.notification_type or "info"

        # 2. Store in DB
        db_notification_id = None
        try:
            async with AsyncSessionLocal() as db:
                db_notification = await AgentMeshCrud.create_notification(
                    db=db,
                    workflow_id=workflow_id,
                    thread_id=thread_id,
                    title=title,
                    message=message,
                    type=ntype,
                    metadata_json={
                        "node_id": node_config.id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "show_chat_button": node_config.show_chat_button,
                        # CLEANUP: Exclude huge 'nodes' and 'messages' data from context snapshot to avoid "useless data"
                        "context_snapshot": {k: v for k, v in context.items() if k not in ["nodes", "messages"] and isinstance(v, (str, int, float, bool, list, dict))}
                    }
                )
                db_notification_id = db_notification.id
                title = db_notification.title
                message = db_notification.message
            
            # 3. Broadcast real-time update via WebSocket
            await ws_manager.broadcast({
                "type": "notification",
                "data": {
                    "id": db_notification_id,
                    "title": title,
                    "message": message,
                    "type": ntype,
                    "created_at": datetime.utcnow().isoformat()
                }
            })

            status_msg = f"Notification stored: {title}"
            logger.info(f"Notification Node {node_config.id}: {status_msg}")
            
        except Exception as e:
            status_msg = f"⚠️ Failed to store notification: {str(e)}"
            logger.error(f"Notification Node {node_config.id} Error: {str(e)}")

        # 3. Return updates
        return {
            "messages": [] if node_config.mute_notification_message else [AIMessage(content=status_msg, response_metadata={"active_agent": node_config.id})],
            "active_agent": node_config.id,
            "context": {
                **context,
                "latest_notification_id": db_notification_id
            }
        }

    return RunnableLambda(execute_notification).with_retry(stop_after_attempt=3)

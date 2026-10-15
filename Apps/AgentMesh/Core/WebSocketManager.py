import logging
from typing import List, Dict, Any, Optional
from fastapi import WebSocket
from Apps.AgentMesh.Utilities.Security import redact_sensitive_data, sanitize_for_logging

logger = logging.getLogger(__name__)

class ConnectionManager:
    """
    Manages active WebSocket connections for real-time notifications and
    live workflow run status.
    """
    def __init__(self):
        # active_connections tracks all connected clients
        self.active_connections: List[WebSocket] = []

        # user_connections allows targeted notifications (if you add auth later)
        # For now, we'll use a simple broadcast approach
        self.user_connections: Dict[str, List[WebSocket]] = {}

        # thread_connections routes run-status events to clients subscribed
        # to a specific execution thread (keyed by thread_id)
        self.thread_connections: Dict[str, List[WebSocket]] = {}

        # workflow_connections routes run-status events to clients subscribed
        # to a specific workflow (keyed by workflow_id). Background/cron runs
        # spawn their own threads per execution, so editors subscribe by
        # workflow_id to see all of them on the canvas.
        self.workflow_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, client_id: str = "anonymous",
                      thread_id: Optional[str] = None,
                      workflow_id: Optional[str] = None,
                      subprotocol: Optional[str] = None):
        await websocket.accept(subprotocol=subprotocol)
        self.active_connections.append(websocket)

        if client_id not in self.user_connections:
            self.user_connections[client_id] = []
        self.user_connections[client_id].append(websocket)

        if thread_id:
            self.thread_connections.setdefault(thread_id, []).append(websocket)

        if workflow_id:
            self.workflow_connections.setdefault(workflow_id, []).append(websocket)

        logger.info(
            f"WebSocket connected: {client_id} (thread={thread_id or '-'}, workflow={workflow_id or '-'}). "
            f"Total connections: {len(self.active_connections)}"
        )

    def disconnect(self, websocket: WebSocket, client_id: str = "anonymous"):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

        if client_id in self.user_connections and websocket in self.user_connections[client_id]:
            self.user_connections[client_id].remove(websocket)
            if not self.user_connections[client_id]:
                del self.user_connections[client_id]

        for channel_map in (self.thread_connections, self.workflow_connections):
            for channel in list(channel_map.keys()):
                conns = channel_map[channel]
                if websocket in conns:
                    conns.remove(websocket)
                    if not conns:
                        del channel_map[channel]

        logger.info(f"WebSocket disconnected: {client_id}. Remaining: {len(self.active_connections)}")

    async def _send_to_connections(self, connections: List[WebSocket], message: Dict[str, Any]):
        safe_message = sanitize_for_logging(redact_sensitive_data(message))
        dead_connections = []
        for connection in list(connections):
            try:
                await connection.send_json(safe_message)
            except Exception as e:
                logger.error(f"Error sending message: {e}")
                dead_connections.append(connection)

        for dead in dead_connections:
            if dead in self.active_connections:
                self.disconnect(dead)

    async def broadcast(self, message: Dict[str, Any]):
        """Sends a message to all connected clients."""
        await self._send_to_connections(self.active_connections, message)

    async def send_to_client(self, client_id: str, message: Dict[str, Any]):
        """Sends a message to a specific client/user."""
        if client_id in self.user_connections:
            await self._send_to_connections(self.user_connections[client_id], message)

    async def send_to_thread(self, thread_id: str, message: Dict[str, Any]):
        """Sends a message to clients subscribed to a specific execution thread."""
        if thread_id in self.thread_connections:
            await self._send_to_connections(self.thread_connections[thread_id], message)

    async def send_to_workflow(self, workflow_id: str, message: Dict[str, Any]):
        """Sends a message to clients subscribed to a specific workflow."""
        if workflow_id in self.workflow_connections:
            await self._send_to_connections(self.workflow_connections[workflow_id], message)

# Global instance for app-wide access
ws_manager = ConnectionManager()

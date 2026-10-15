import httpx
import logging
from langchain_core.messages import AIMessage
from Apps.AgentMesh.Schema.StateSchema import AgentState
from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from Apps.AgentMesh.Core.Compiler.nodes.utils import interpolate

logger = logging.getLogger(__name__)

def build_event_emitter_node(node_config: NodeConfig):
    """Post a status event to an external system.

    This node is deliberately free of any single application's vocabulary. It
    previously carried five pieces of Credit Decisioning specifics — a default
    URL pointing at a CD endpoint, CD's status words as fallbacks, CD's field
    names hardcoded into the payload, and `cd_customer_id` among the forwarded
    keys — while its own comment claimed it was "100% generic". Anything an
    application needs is now supplied through configuration or the
    `emit_data_*` mechanism, so the node works the same for any consumer.
    """
    async def emit_event(state: AgentState):
        context = state.get("context", {})

        # ── Destination ──
        # No default. A default URL means an unconfigured node silently posts
        # somewhere — which is how this node came to point at a CD endpoint that
        # does not exist, failing quietly on every run. Unset now means skip.
        webhook_url = getattr(node_config, "webhook_url", None)
        if not webhook_url:
            logger.info(
                f"[Event Emitter {node_config.id}] Skipped: no webhook_url configured."
            )
            return {"messages": [AIMessage(content="Event Emitter skipped: no webhook_url configured")]}

        # Interpolate so the URL can come from the run, e.g.
        # {{ payload.callback_base }}/status — letting one workflow serve every
        # environment instead of hardcoding a host on the canvas.
        webhook_url = interpolate(webhook_url, context)

        # ── Status ──
        # Config or context only. There is no fallback to a particular
        # application's status words: "staged" and "awaiting_verification" mean
        # nothing outside the system that defines them.
        status = (
            context.get("emit_status")
            or getattr(node_config, "emit_status", None)
        )

        # ── Payload ──
        # Only run-identifying facts, which are true for every workflow.
        event_payload = {
            "thread_id": context.get("thread_id"),
            "workflow_id": context.get("workflow_id"),
        }
        if status:
            event_payload["status"] = status

        # Anything else is opt-in: a node writes context["emit_data_<name>"] and
        # it arrives as payload["<name>"]. This is the extension point, and it
        # is the reason no application's field names need to live in this file.
        for key, value in context.items():
            if key.startswith("emit_data_"):
                event_payload[key[len("emit_data_"):]] = value

        timeout = float(getattr(node_config, "emit_timeout_sec", None) or 30.0)

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(webhook_url, json=event_payload)
                response.raise_for_status()
                logger.info(
                    f"[Event Emitter {node_config.id}] Posted to {webhook_url}"
                    + (f" (status={status})" if status else "")
                )
        except Exception as e:
            # Reporting a status must never fail the pipeline that produced it.
            logger.error(f"[Event Emitter {node_config.id}] Failed to post to {webhook_url}: {e}")

        return {"messages": [AIMessage(content=f"Event emitted{f': status={status}' if status else ''}")]}

    return emit_event

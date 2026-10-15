"""
Circuit Breaker Node
Wraps an HTTP request with a state machine (Closed, Open, Half-Open).
"""
import json
import logging
import time
from typing import Callable, Dict, Any

import httpx
from langchain_core.messages import AIMessage

from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from Apps.AgentMesh.Schema.StateSchema import AgentState
from Apps.AgentMesh.Utilities.Security import contains_sensitive_data, validate_credential_destination
from .utils import interpolate

logger = logging.getLogger(__name__)

# Global in-memory state for circuit breakers (in production, use Redis/Memcached)
# Format: { "url": { "failures": int, "state": "CLOSED" | "OPEN" | "HALF_OPEN", "last_failure_time": float } }
CIRCUIT_STATES: Dict[str, Dict[str, Any]] = {}

def build_circuit_breaker_node(node_config: NodeConfig) -> Callable:
    async def execute_circuit_breaker(state: AgentState):
        context = state.get("context", {})
        url = interpolate(node_config.api_url or "", context)
        
        # 1. State Machine Logic
        circuit = CIRCUIT_STATES.get(url, {"failures": 0, "state": "CLOSED", "last_failure_time": 0})
        
        # Configuration parameters from Schema
        threshold = getattr(node_config, "circuit_threshold", 3)
        raw_timeout = getattr(node_config, "circuit_timeout", 30)
        timeout_unit = getattr(node_config, "circuit_timeout_unit", "s")
        
        # Convert to seconds
        multiplier = {"ms": 0.001, "s": 1, "m": 60, "h": 3600}
        timeout_window = raw_timeout * multiplier.get(timeout_unit, 1)
        
        if circuit["state"] == "OPEN":
            if time.time() - circuit["last_failure_time"] > timeout_window:
                circuit["state"] = "HALF_OPEN"
                logger.warning(f"Circuit {url} transitioned to HALF_OPEN")
            else:
                logger.error(f"Circuit {url} is OPEN. Failing fast to prevent cascading timeouts.")
                return {
                    "messages": [AIMessage(content="Circuit Breaker is OPEN. Failing fast.", name=node_config.id)],
                    "context": {"api_response": {"error": "Circuit OPEN", "status": "CIRCUIT_BREAKER_TRIPPED"}}
                }
                
        # 2. Execute Request
        headers = {k: interpolate(v, context) for k, v in (node_config.api_headers or {}).items()}
        method = getattr(node_config, "api_method", "GET").upper()
        
        body_str = node_config.api_body
        if body_str and isinstance(body_str, str):
            body_str = interpolate(body_str, context)

        if contains_sensitive_data({"headers": headers, "body": body_str}):
            validate_credential_destination(url)
            
        timeout = httpx.Timeout(15.0) # Tighter timeout for breaker
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                request_kwargs = {"method": method, "url": url, "headers": headers}
                if body_str:
                    if headers.get("Content-Type") == "application/json":
                        request_kwargs["json"] = json.loads(body_str)
                    else:
                        request_kwargs["content"] = body_str
                
                logger.info(f"Circuit Breaker Node executing {method} to {url}")
                resp = await client.request(**request_kwargs)
                resp.raise_for_status()
                
                # Success: Reset circuit
                if circuit["state"] == "HALF_OPEN":
                    logger.info(f"Circuit {url} recovered. Transitioning to CLOSED.")
                circuit["failures"] = 0
                circuit["state"] = "CLOSED"
                CIRCUIT_STATES[url] = circuit
                
                try:
                    resp_data = resp.json()
                except:
                    resp_data = {"text": resp.text}
                    
                return {
                    "messages": [AIMessage(content=f"API Request to {url} succeeded.", name=node_config.id)],
                    "context": {"api_response": resp_data}
                }
                
        except Exception as e:
            # Failure: Trip circuit
            circuit["failures"] += 1
            circuit["last_failure_time"] = time.time()
            if circuit["failures"] >= threshold or circuit["state"] == "HALF_OPEN":
                circuit["state"] = "OPEN"
                logger.error(f"Circuit {url} tripped to OPEN state after {circuit['failures']} failures.")
            CIRCUIT_STATES[url] = circuit
            
            return {
                "messages": [AIMessage(content=f"API Request failed: {str(e)}", name=node_config.id)],
                "context": {"api_response": {"error": str(e), "status": "FAILED"}}
            }

    return execute_circuit_breaker

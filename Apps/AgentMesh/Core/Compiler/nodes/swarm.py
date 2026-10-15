"""
Swarm Node Builder
Handles agent swarms with handoff capabilities using langgraph-swarm.
"""

import logging
from typing import Callable, List

from langchain_core.runnables import RunnableLambda
from langchain.agents import create_agent
from langgraph_swarm import create_swarm, create_handoff_tool

from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from Apps.AgentMesh.Schema.StateSchema import AgentState
from Apps.AgentMesh.Core.ToolRegistry import registry
from Apps.AgentMesh.Utilities.LLM.llm_factory import init_chat_model
from .utils import process_tools_with_overrides

logger = logging.getLogger(__name__)


def build_swarm_node(node_config: NodeConfig, all_nodes: List[NodeConfig] = None) -> Callable:
    """
    Build a swarm node where agents can hand off control to each other.
    Agents have access to handoff tools for dynamic routing.
    """
    # Include the swarm node itself as the default entry point
    members = list(node_config.member_ids or [])
    if node_config.id not in members:
        members.insert(0, node_config.id)
    
    # Build worker agents
    worker_agents = []
    for m_id in members:
        member_config = next(
            (n for n in (all_nodes or []) if n.id == m_id),
            node_config if m_id == node_config.id else None
        )
        
        if not member_config:
            logger.warning(f"Swarm {node_config.id}: Member {m_id} config not found")
            continue
        
        # Gather tools for this agent - ALWAYS COPY to avoid mutation
        raw_agent_tools = registry.get_tools(member_config.tools)
        agent_tools = process_tools_with_overrides(raw_agent_tools, member_config.tool_configs)
        
        # Add handoff tools to all OTHER members (or filtered by allowed_transitions)
        allowed_transitions = getattr(member_config, 'allowed_transitions', None)
        for handoff_id in members:
            if handoff_id == m_id:
                continue
            
            # If allowed_transitions is defined, respect it to avoid handoff explosion
            if allowed_transitions is not None and handoff_id not in allowed_transitions:
                continue
            
            # Get handoff target config for better description
            h_config = next(
                (n for n in (all_nodes or []) if n.id == handoff_id),
                node_config if handoff_id == node_config.id else None
            )
            h_label = h_config.label if h_config and h_config.label else handoff_id
            
            agent_tools.append(create_handoff_tool(
                agent_name=handoff_id,
                description=f"Transfer control to {h_label}. Use this tool to hand over context."
            ))
        
        # Create the agent with swarm protocol instructions
        worker_model = init_chat_model(member_config.llm or node_config.llm)
        
        logger.debug(f"Swarm {node_config.id}: Agent {m_id} has {len(agent_tools)} tool(s)")
        
        worker = create_agent(
            model=worker_model,
            tools=agent_tools,
            system_prompt=(
                f"{member_config.task or 'You are a helpful part of a swarm.'}\n\n"
                "SWARM PROTOCOL:\n"
                "1. You are a specialized worker. You DO NOT have authority to end the entire conversation.\n"
                "2. If the user is finished with your specialty or asks something outside your scope, "
                "you MUST use a 'transfer_to_...' tool immediately.\n"
                "3. EVEN IF YOU HAVE THE ANSWER, if the user requested involvement from another team member, "
                "you MUST provide your part and then IMMEDIATELY call the transfer tool to the next person.\n"
                "4. NEVER just say you are handing over. YOU MUST PHYSICALLY MOVE THE SESSION WITH A TOOL CALL."
            ),
            name=m_id
        )
        worker_agents.append(worker)
    
    # Create the swarm
    swarm_app = create_swarm(
        agents=worker_agents,
        default_active_agent=members[0] if members else None
    ).compile()
    
    async def execute_swarm(state: AgentState):
        # Track original message count to identify what's new
        original_messages = state.get("messages", [])
        original_msg_count = len(original_messages)
        
        # 🛡️ Safety: Create a clean copy of the state for the sub-graph
        # This prevents any accidental mutation of parent state objects
        swarm_state = {
            **state,
            "messages": list(original_messages)
        }
        
        # Preserve valid active_agent or remove invalid ones
        if "active_agent" in swarm_state and swarm_state["active_agent"] not in members:
            swarm_state.pop("active_agent", None)
        
        # Invoke the swarm
        logger.debug(f"Swarm '{node_config.id}' - Invoking nested swarm app...")
        result = await swarm_app.ainvoke(swarm_state)
        
        # 1. Identify only NEW messages generated inside the swarm
        all_messages = result.get("messages", [])
        new_messages = all_messages[original_msg_count:]
        
        # 2. Extract state overrides (intent, next, etc.)
        # This allows agents IN the swarm to influence the MAIN graph's routing
        intent = result.get("intent")
        next_override = result.get("next")
        active_agent = result.get("active_agent", members[0] if members else node_config.id)
        
        logger.info(
            f"Swarm '{node_config.id}' finished. New messages: {len(new_messages)}, "
            f"Active: {active_agent}, Intent: {intent}, Next: {next_override}"
        )
        
        res_dict = {
            "messages": new_messages,
            "active_agent": active_agent,
            "next": next_override or "FINISH"
        }
        
        if intent:
            res_dict["intent"] = intent
            
        return res_dict
    
    return RunnableLambda(execute_swarm)

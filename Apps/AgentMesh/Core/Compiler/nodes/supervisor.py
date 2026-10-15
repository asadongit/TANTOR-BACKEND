"""
Supervisor Node Builder
Handles multi-agent coordination using langgraph-supervisor.
"""

import logging
from typing import Callable, List

from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from Apps.AgentMesh.Schema.StateSchema import AgentState
from Apps.AgentMesh.Core.ToolRegistry import registry
from Apps.AgentMesh.Utilities.LLM.llm_factory import init_chat_model
from .utils import process_tools_with_overrides

logger = logging.getLogger(__name__)


def build_supervisor_node(node_config: NodeConfig, all_nodes: List[NodeConfig] = None) -> Callable:
    """
    Build a supervisor node that orchestrates multiple worker agents.
    The supervisor manages task coordination and handoffs.
    
    CRITICAL: create_supervisor requires all members to be create_react_agent
    instances with proper .name attributes. RunnableLambda won't work.
    """
    llm = init_chat_model(node_config.llm)
    members = node_config.member_ids or []
    
    # Build worker agents from member configs
    worker_agents = []
    for m_id in members:
        member_config = next((n for n in (all_nodes or []) if n.id == m_id), None)
        if not member_config:
            logger.warning(f"Supervisor {node_config.id}: Member {m_id} config not found")
            continue
        
        llm_member = init_chat_model(member_config.llm)
        raw_tools_member = registry.get_tools(member_config.tools) if member_config.tools else []
        tools_member = process_tools_with_overrides(raw_tools_member, member_config.tool_configs)
        
        worker = create_react_agent(
            llm_member,
            tools=tools_member,
            prompt=member_config.task or "You are a helpful assistant.",
            name=member_config.id
        )
        
        logger.debug(
            f"Supervisor {node_config.id}: Built worker '{member_config.id}' "
            f"with {len(tools_member)} tool(s)"
        )
        worker_agents.append(worker)
    
    # Build supervisor prompt with clear termination instructions
    base_prompt = node_config.task or "Coordinate the workers to solve the task."
    final_prompt = (
        f"{base_prompt}\n\n"
        "IMPORTANT: When the workers have completed their tasks and you have a final answer, "
        "provide your final response to the user and DO NOT call any more tools. "
        "This will signal the end of your turn and move the workflow forward."
    )
    
    # Create and compile the supervisor
    supervisor_app = create_supervisor(
        agents=worker_agents,
        model=llm,
        prompt=final_prompt,
        add_messages_to_output=True
    ).compile()
    
    async def execute_supervisor(state: AgentState):
        # Track original message count to avoid duplication
        original_msg_count = len(state.get("messages", []))
        
        # Invoke the supervisor subgraph
        result = await supervisor_app.ainvoke(state)
        
        # Only return NEW messages
        all_messages = result.get("messages", [])
        new_messages = all_messages[original_msg_count:]
        
        logger.debug(f"Supervisor {node_config.id}: Generated {len(new_messages)} new message(s)")
        
        return {
            "messages": new_messages,
            "active_agent": node_config.id,
            "next": "FINISH"
        }
    
    return RunnableLambda(execute_supervisor)
